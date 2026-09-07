//! First observations retain measured bytes without inventing prerequest identities.
use crate::{ProvenanceStore, artifact_paths::verified_file_bytes};
use anyhow::{Context, Result, bail};
use chrono::{DateTime, NaiveDate, Utc};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{io::Read, path::Path};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceStorageEncoding {
    Identity,
    Gzip,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceObservedFile {
    pub path: String,
    pub storage_sha256: String,
    pub encoding: SourceStorageEncoding,
    pub decoded_sha256: String,
    pub decoded_bytes: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceTimePrecision {
    Day,
    Timestamp,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceTransportOutcome {
    BodyRetained,
    RequestFailed,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceArtifactPrestate {
    pub artifact_id: String,
    pub artifact_key: String,
    pub canonical_url: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceObservationSpec {
    pub schema_version: u32,
    pub observation_key: String,
    pub source_key: String,
    pub actor: String,
    pub reason: String,
    pub requested_url: String,
    pub final_url: Option<String>,
    pub http_status: Option<u16>,
    pub outcome: SourceTransportOutcome,
    pub observed_at: String,
    pub time_precision: SourceTimePrecision,
    pub tool: String,
    pub request_evidence: SourceObservedFile,
    pub request_evidence_limit: String,
    pub body: Option<SourceObservedFile>,
    pub artifact_prestate: Option<SourceArtifactPrestate>,
    pub absent_prior_expectation_reason: String,
    pub document_identity_limit: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub corrects_observation_key: Option<String>,
}
fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn verify_url(value: &str) -> Result<()> {
    let url = url::Url::parse(value)?;
    if !matches!(url.scheme(), "https" | "http" | "ftp" | "ftps" | "rsync")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
    {
        bail!("source URL requires a supported transport, host and absent credentials");
    }
    Ok(())
}
fn verify_file(root: &Path, file: &SourceObservedFile) -> Result<Vec<u8>> {
    const MAX_BYTES: u64 = 64 * 1024 * 1024;
    if file.decoded_bytes == 0 || file.decoded_bytes > MAX_BYTES {
        bail!("source evidence size exceeds bounded nonempty admission range");
    }
    let stored = verified_file_bytes(root, &file.path, &file.storage_sha256)?;
    let bytes = match file.encoding {
        SourceStorageEncoding::Identity => stored,
        SourceStorageEncoding::Gzip => {
            let mut decoded = Vec::new();
            flate2::read::MultiGzDecoder::new(stored.as_slice())
                .take(file.decoded_bytes + 1)
                .read_to_end(&mut decoded)?;
            decoded
        }
    };
    if bytes.len() as u64 != file.decoded_bytes || digest(&bytes) != file.decoded_sha256 {
        bail!("source decoded digest or size mismatch");
    }
    if bytes.starts_with(b"version https://git-lfs.github.com/spec/v1") {
        bail!("source evidence is an LFS pointer");
    }
    Ok(bytes)
}
fn artifact_snapshot(connection: &Connection, expected: &SourceArtifactPrestate) -> Result<Value> {
    let row = connection
        .query_row(
            "SELECT key,canonical_functional_url FROM artifacts WHERE id=?1",
            [&expected.artifact_id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
        )
        .optional()?
        .context("source association artifact missing")?;
    if row.0 != expected.artifact_key || row.1 != expected.canonical_url {
        bail!("source association artifact prestate drift");
    }
    Ok(json!({"artifact_id":expected.artifact_id,"key":row.0,"canonical_url":row.1}))
}

/// Operator-recorded HTTP metadata binds declared fields to retained bytes;
/// metadata correspondence does not authenticate a network exchange.
#[derive(Deserialize)]
struct PythonHttpReceipt {
    url: String,
    #[serde(deserialize_with = "required_nullable")]
    final_url: Option<String>,
    #[serde(deserialize_with = "required_nullable")]
    status: Option<u16>,
    retrieved_utc: String,
    client: String,
    client_version: String,
    complete: bool,
    error: String,
    path: String,
    sha256: String,
    bytes: u64,
    storage_sha256: String,
}

fn required_nullable<'de, T: Deserialize<'de>, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<T>, D::Error> {
    Option::<T>::deserialize(deserializer)
}

fn verify_python_receipt(spec: &SourceObservationSpec, text: &str) -> Result<&'static str> {
    let receipt: PythonHttpReceipt = serde_json::from_str(text)
        .context("Python request evidence requires one structured HTTP receipt object")?;
    for value in std::iter::once(receipt.url.as_str()).chain(receipt.final_url.as_deref()) {
        if !matches!(url::Url::parse(value)?.scheme(), "http" | "https") {
            bail!("Python HTTP receipt requires HTTP or HTTPS URLs");
        }
    }
    if !matches!(receipt.client.as_str(), "requests" | "urllib")
        || receipt.client_version.trim().is_empty()
        || receipt.client_version.chars().any(char::is_whitespace)
        || spec.tool
            != format!(
                "Python {} {} (structured receipt)",
                receipt.client, receipt.client_version
            )
        || !matches!(spec.time_precision, SourceTimePrecision::Timestamp)
        || receipt.url != spec.requested_url
        || receipt.final_url != spec.final_url
        || receipt.status != spec.http_status
        || receipt.retrieved_utc != spec.observed_at
    {
        bail!("Python receipt URL, status, timestamp or client attribution mismatch");
    }
    match spec.outcome {
        SourceTransportOutcome::BodyRetained => {
            let body = spec
                .body
                .as_ref()
                .context("Python success requires retained body")?;
            if !receipt.complete
                || !receipt.error.is_empty()
                || !receipt
                    .status
                    .is_some_and(|status| (200..300).contains(&status))
                || receipt.final_url.is_none()
                || receipt.path != body.path
                || receipt.sha256 != body.decoded_sha256
                || receipt.bytes != body.decoded_bytes
                || receipt.storage_sha256 != body.storage_sha256
            {
                bail!("Python receipt lacks complete successful body correspondence");
            }
        }
        SourceTransportOutcome::RequestFailed => {
            if spec.body.is_some()
                || !match receipt.status {
                    Some(status) => !(200..300).contains(&status),
                    None => !receipt.error.trim().is_empty(),
                }
            {
                bail!(
                    "Python failure requires non-success status or an explicit transport error without status"
                );
            }
        }
    }
    Ok("operator_recorded_structured_receipt_correspondence")
}

fn verify_request(spec: &SourceObservationSpec, bytes: &[u8]) -> Result<&'static str> {
    let text = std::str::from_utf8(bytes)?;
    if text.trim_start().starts_with('{') || spec.tool.starts_with("Python ") {
        return verify_python_receipt(spec, text);
    }
    if let Ok(manifest) = toml::from_str::<toml::Value>(text) {
        let date = manifest.get("retrieval_date").and_then(toml::Value::as_str);
        let sources = manifest
            .get("source")
            .and_then(toml::Value::as_array)
            .context("request manifest requires sources")?;
        let matched: Vec<_> = sources
            .iter()
            .filter(|row| {
                row.get("id").and_then(toml::Value::as_str) == Some(&spec.source_key)
                    && row.get("url").and_then(toml::Value::as_str) == Some(&spec.requested_url)
            })
            .collect();
        if matched.len() != 1
            || date != Some(spec.observed_at.as_str())
            || !matches!(spec.time_precision, SourceTimePrecision::Day)
            || spec.final_url.is_some()
            || spec.http_status.is_some()
            || spec.tool != "curl (audit-manifest attribution)"
        {
            bail!(
                "request manifest only supports matching source URL, date and bounded tool attribution; final URL and status require raw logs"
            );
        }
        let body = spec
            .body
            .as_ref()
            .context("request manifest requires retained body")?;
        if matched[0].get("sha256").and_then(toml::Value::as_str) != Some(&body.decoded_sha256) {
            bail!("request manifest body digest mismatch");
        }
        return Ok("retained_manifest_digest_association");
    }
    if !matches!(spec.time_precision, SourceTimePrecision::Day)
        || spec.tool != "GNU wget (retained log)"
    {
        bail!("wget log admission requires day precision and explicit tool attribution");
    }
    let requested = url::Url::parse(&spec.requested_url)?;
    let mut logged_url = spec.requested_url.clone();
    if text.starts_with("URL transformed to HTTPS due to an HSTS policy\n") {
        if requested.scheme() != "http" {
            bail!("HSTS log requires HTTP input URL");
        }
        let mut transformed = requested;
        transformed
            .set_scheme("https")
            .map_err(|()| anyhow::anyhow!("HSTS scheme mutation failed"))?;
        logged_url = transformed.to_string();
    }
    let starts: Vec<_> = text.lines().filter(|line| line.starts_with("--")).collect();
    if starts.len() != 1
        || !starts[0].starts_with(&format!("--{} ", spec.observed_at))
        || !starts[0].ends_with(&format!("  {logged_url}"))
        || spec.final_url.as_deref() != Some(logged_url.as_str())
    {
        bail!("wget URL or observation date differs from retained request log");
    }
    let statuses: Vec<_> = text
        .lines()
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let protocol = fields.next()?;
            if !protocol.starts_with("HTTP/") {
                return None;
            }
            fields.next()?.parse::<u16>().ok()
        })
        .collect();
    if statuses.len() != 1 || spec.http_status != Some(statuses[0]) {
        bail!("wget status differs from retained request log");
    }
    match &spec.outcome {
        SourceTransportOutcome::BodyRetained => {
            let body = spec
                .body
                .as_ref()
                .context("successful wget requires body")?;
            if !text.contains(&format!("[{0}/{0}]", body.decoded_bytes))
                || !text.contains(" saved ")
            {
                bail!("wget log lacks completed byte-count witness");
            }
        }
        SourceTransportOutcome::RequestFailed => {
            if !text.contains(&format!("ERROR {}:", statuses[0])) {
                bail!("wget failure lacks terminal error witness");
            }
            return Ok("retained_failure_log_without_body");
        }
    }
    Ok("operator_association_with_matching_size")
}
pub(crate) fn refuse_source_observation_history_loss(connection: &Connection) -> Result<()> {
    let exists: bool = connection.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='source_observations')", [], |row| row.get(0))?;
    if exists
        && connection.query_row("SELECT count(*) FROM source_observations", [], |row| {
            row.get::<_, i64>(0)
        })? > 0
    {
        bail!("refusing to discard canonical first-observation source history");
    }
    Ok(())
}
impl ProvenanceStore {
    /// Admit measured source bytes and request evidence without changing scientific status.
    pub fn record_source_observation(
        &mut self,
        root: &Path,
        spec: &SourceObservationSpec,
    ) -> Result<Value> {
        if spec.schema_version != 1 {
            bail!("source observation requires schema_version 1");
        }
        for value in [
            &spec.observation_key,
            &spec.source_key,
            &spec.actor,
            &spec.reason,
            &spec.tool,
            &spec.request_evidence_limit,
            &spec.absent_prior_expectation_reason,
            &spec.document_identity_limit,
        ] {
            if value.trim().is_empty() {
                bail!("source observation identifiers and limitations must be nonempty");
            }
        }
        verify_url(&spec.requested_url)?;
        if let Some(url) = &spec.final_url {
            verify_url(url)?;
        }
        match spec.time_precision {
            SourceTimePrecision::Day => {
                NaiveDate::parse_from_str(&spec.observed_at, "%Y-%m-%d")?;
                if spec.observed_at.len() != 10 {
                    bail!("day precision requires YYYY-MM-DD");
                }
            }
            SourceTimePrecision::Timestamp => {
                DateTime::parse_from_rfc3339(&spec.observed_at)?;
            }
        }
        if let Some(status) = spec.http_status
            && !(100..600).contains(&status)
        {
            bail!("invalid HTTP status");
        }
        let correspondence_basis =
            verify_request(spec, &verify_file(root, &spec.request_evidence)?)?;
        match (&spec.outcome, &spec.body) {
            (SourceTransportOutcome::BodyRetained, Some(body)) => {
                if spec
                    .http_status
                    .is_some_and(|status| !(200..300).contains(&status))
                {
                    bail!("retained successful source requires successful status");
                }
                verify_file(root, body)?;
            }
            (SourceTransportOutcome::RequestFailed, None) => {
                if spec
                    .http_status
                    .is_some_and(|status| (200..300).contains(&status))
                {
                    bail!("failed request conflicts with successful HTTP status");
                }
            }
            _ => bail!("source outcome and body retention disagree"),
        }
        let spec_sha256 = digest(&serde_json::to_vec(spec)?);
        let transaction = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let association = spec
            .artifact_prestate
            .as_ref()
            .map(|expected| artifact_snapshot(&transaction, expected))
            .transpose()?;
        if let Some(predecessor_key) = &spec.corrects_observation_key {
            if predecessor_key.trim().is_empty() || predecessor_key == &spec.observation_key {
                bail!("metadata correction requires a distinct nonempty predecessor key");
            }
            let branched: bool = transaction.query_row(
                "SELECT EXISTS(SELECT 1 FROM source_observations WHERE json_extract(report_json,'$.corrects_observation_key')=?1 AND observation_key<>?2)",
                params![predecessor_key, spec.observation_key], |row| row.get(0),
            )?;
            if branched {
                bail!(
                    "metadata correction predecessor already has a successor; correct the latest record"
                );
            }
            let (source_key, predecessor_report): (String, String) = transaction.query_row(
                "SELECT source_key,report_json FROM source_observations WHERE observation_key=?1",
                [predecessor_key], |row| Ok((row.get(0)?, row.get(1)?)),
            ).optional()?.context("metadata correction predecessor is absent")?;
            let predecessor: Value = serde_json::from_str(&predecessor_report)?;
            let predecessor_spec: SourceObservationSpec =
                serde_json::from_value(predecessor["spec"].clone())?;
            if source_key != spec.source_key || predecessor_spec.observation_key != *predecessor_key
            {
                bail!("metadata correction predecessor source or key differs");
            }
            let mut old_acquisition = serde_json::to_value(predecessor_spec)?;
            let mut new_acquisition = serde_json::to_value(spec)?;
            for value in [&mut old_acquisition, &mut new_acquisition] {
                let object = value
                    .as_object_mut()
                    .context("source specification must be an object")?;
                for field in [
                    "observation_key",
                    "actor",
                    "reason",
                    "corrects_observation_key",
                ] {
                    object.remove(field);
                }
            }
            if old_acquisition != new_acquisition {
                bail!("metadata correction changes retained acquisition fields");
            }
        }
        let previous: Option<(String, String)> = transaction
            .query_row(
                "SELECT spec_sha256,report_json FROM source_observations WHERE observation_key=?1",
                [&spec.observation_key],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()?;
        if let Some((old_digest, report)) = previous {
            if old_digest != spec_sha256 {
                bail!("source observation key already records another specification");
            }
            return Ok(serde_json::from_str(&report)?);
        }
        let record_kind = if spec.corrects_observation_key.is_some() {
            "metadata_correction"
        } else {
            "observation"
        };
        let report = json!({"schema_version":1,"record_kind":record_kind,"corrects_observation_key":spec.corrects_observation_key,"observation_key":spec.observation_key,"spec_sha256":spec_sha256,"spec":spec,"artifact_prestate":association,"prior_body_expectation":null,"document_identity":"unresolved","scientific_promotion":false,"correspondence_basis":correspondence_basis,"cryptographic_request_binding":false});
        transaction.execute("INSERT INTO source_observations(observation_key,source_key,artifact_id,spec_sha256,admitted_at,report_json) VALUES (?1,?2,?3,?4,?5,?6)",params![spec.observation_key,spec.source_key,spec.artifact_prestate.as_ref().map(|row|&row.artifact_id),spec_sha256,Utc::now().to_rfc3339(),serde_json::to_string(&report)?])?;
        transaction.commit()?;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        io::Write,
        path::PathBuf,
        process::Command,
        sync::atomic::{AtomicUsize, Ordering},
    };
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    fn write_structured_receipt(fixture: &mut Fixture, receipt: &Value) {
        let bytes = serde_json::to_vec(receipt).unwrap();
        fs::write(fixture.root.join("request.txt"), &bytes).unwrap();
        fixture.spec.request_evidence.storage_sha256 = digest(&bytes);
        fixture.spec.request_evidence.decoded_sha256 = digest(&bytes);
        fixture.spec.request_evidence.decoded_bytes = bytes.len() as u64;
    }

    fn structured_fixture(client: &str) -> (Fixture, Value) {
        let mut fixture = fixture();
        fixture.spec.time_precision = SourceTimePrecision::Timestamp;
        fixture.spec.observed_at = "2026-09-07T00:24:07.924838+00:00".into();
        fixture.spec.tool = format!("Python {client} 3.14.7 (structured receipt)");
        fixture.spec.final_url = Some("https://example.org/redirected".into());
        let body = fixture.spec.body.as_ref().unwrap();
        let receipt = json!({
            "url": fixture.spec.requested_url, "final_url": fixture.spec.final_url,
            "status": 200, "retrieved_utc": fixture.spec.observed_at,
            "client": client, "client_version": "3.14.7", "complete": true,
            "error": "", "path": body.path, "sha256": body.decoded_sha256,
            "bytes": body.decoded_bytes, "storage_sha256": body.storage_sha256,
            "elapsed_seconds": 0.25,
        });
        write_structured_receipt(&mut fixture, &receipt);
        (fixture, receipt)
    }

    #[test]
    fn python_receipts_preserve_replay_and_append_only_history() {
        for client in ["requests", "urllib"] {
            let (mut fixture, _) = structured_fixture(client);
            let report = fixture
                .store
                .record_source_observation(&fixture.root, &fixture.spec)
                .unwrap();
            assert_eq!(
                report["correspondence_basis"],
                "operator_recorded_structured_receipt_correspondence"
            );
            assert_eq!(report["cryptographic_request_binding"], false);
            assert_eq!(
                fixture
                    .store
                    .record_source_observation(&fixture.root, &fixture.spec)
                    .unwrap(),
                report
            );
            assert!(refuse_source_observation_history_loss(&fixture.store.conn).is_err());
            for sql in [
                "UPDATE source_observations SET source_key='changed'",
                "DELETE FROM source_observations",
            ] {
                assert!(fixture.store.conn.execute(sql, []).is_err());
            }
            fs::write(fixture.root.join("request.txt"), b"changed").unwrap();
            assert!(
                fixture
                    .store
                    .record_source_observation(&fixture.root, &fixture.spec)
                    .is_err()
            );
            let retained: String = fixture
                .store
                .conn
                .query_row("SELECT report_json FROM source_observations", [], |row| {
                    row.get(0)
                })
                .unwrap();
            assert_eq!(serde_json::from_str::<Value>(&retained).unwrap(), report);
        }
    }

    #[test]
    fn python_receipt_tampering_is_rejected_before_mutation() {
        for (field, replacement) in [
            ("url", json!("https://different.example")),
            ("final_url", json!("https://different.example")),
            ("status", json!(201)),
            ("retrieved_utc", json!("2026-09-07T00:24:07.924839+00:00")),
            ("client", json!("curl")),
            ("client_version", json!("9.9")),
            ("complete", json!(false)),
            ("error", json!("truncated")),
            ("path", json!("different.gz")),
            ("sha256", json!("0".repeat(64))),
            ("bytes", json!(1)),
            ("storage_sha256", json!("0".repeat(64))),
        ] {
            let (mut fixture, mut receipt) = structured_fixture("requests");
            receipt[field] = replacement;
            write_structured_receipt(&mut fixture, &receipt);
            assert!(
                fixture
                    .store
                    .record_source_observation(&fixture.root, &fixture.spec)
                    .is_err(),
                "field {field}"
            );
            let count: i64 = fixture
                .store
                .conn
                .query_row("SELECT count(*) FROM source_observations", [], |row| {
                    row.get(0)
                })
                .unwrap();
            assert_eq!(count, 0);
        }
        for field in [
            "url",
            "final_url",
            "status",
            "retrieved_utc",
            "client",
            "client_version",
            "complete",
            "error",
            "path",
            "sha256",
            "bytes",
            "storage_sha256",
        ] {
            let (mut fixture, mut receipt) = structured_fixture("urllib");
            receipt.as_object_mut().unwrap().remove(field);
            write_structured_receipt(&mut fixture, &receipt);
            assert!(
                fixture
                    .store
                    .record_source_observation(&fixture.root, &fixture.spec)
                    .is_err(),
                "missing {field}"
            );
        }
    }

    #[test]
    fn python_failures_require_http_failure_or_transport_error() {
        for transport in [false, true] {
            let (mut fixture, mut receipt) = structured_fixture("urllib");
            fixture.spec.body = None;
            fixture.spec.outcome = SourceTransportOutcome::RequestFailed;
            fixture.spec.http_status = if transport { None } else { Some(404) };
            receipt["status"] = json!(fixture.spec.http_status);
            if transport {
                receipt["complete"] = json!(false);
                receipt["error"] = json!("connection timed out");
            }
            write_structured_receipt(&mut fixture, &receipt);
            assert!(
                fixture
                    .store
                    .record_source_observation(&fixture.root, &fixture.spec)
                    .is_ok()
            );
        }
        let (mut fixture, mut receipt) = structured_fixture("requests");
        fixture.spec.body = None;
        fixture.spec.outcome = SourceTransportOutcome::RequestFailed;
        receipt["complete"] = json!(false);
        receipt["error"] = json!("response exceeded bound");
        write_structured_receipt(&mut fixture, &receipt);
        assert!(
            fixture
                .store
                .record_source_observation(&fixture.root, &fixture.spec)
                .is_err()
        );
        fixture.spec.http_status = None;
        receipt["status"] = Value::Null;
        receipt["error"] = json!("");
        write_structured_receipt(&mut fixture, &receipt);
        assert!(
            fixture
                .store
                .record_source_observation(&fixture.root, &fixture.spec)
                .is_err()
        );
    }
    struct Fixture {
        root: PathBuf,
        store: ProvenanceStore,
        spec: SourceObservationSpec,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.root).unwrap();
        }
    }
    fn fixture() -> Fixture {
        let root = std::env::temp_dir().join(format!(
            "source-observation-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        let body = b"%PDF-1.7 retained document";
        let request = format!(
            "--2026-09-06 12:00:00--  https://example.org\n  HTTP/1.1 200 OK\nfile saved [{0}/{0}]\n",
            body.len()
        );
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(body).unwrap();
        let encoded = encoder.finish().unwrap();
        fs::write(root.join("body.gz"), &encoded).unwrap();
        fs::write(root.join("request.txt"), request.as_bytes()).unwrap();
        for arguments in [
            vec!["init", "--quiet"],
            vec!["add", "body.gz", "request.txt"],
        ] {
            let mut command = Command::new("git");
            for (name, _) in std::env::vars_os() {
                if name.as_encoded_bytes().starts_with(b"GIT_") {
                    command.env_remove(name);
                }
            }
            assert!(
                command
                    .arg("-C")
                    .arg(&root)
                    .args(arguments)
                    .status()
                    .unwrap()
                    .success()
            );
        }
        let store = ProvenanceStore::open(&root.join("test.sqlite3")).unwrap();
        let spec = SourceObservationSpec {
            schema_version: 1,
            observation_key: "test".into(),
            source_key: "source".into(),
            actor: "test".into(),
            reason: "retain first observation".into(),
            requested_url: "https://example.org".into(),
            final_url: Some("https://example.org".into()),
            http_status: Some(200),
            outcome: SourceTransportOutcome::BodyRetained,
            observed_at: "2026-09-06".into(),
            time_precision: SourceTimePrecision::Day,
            tool: "GNU wget (retained log)".into(),
            request_evidence: SourceObservedFile {
                path: "request.txt".into(),
                storage_sha256: digest(request.as_bytes()),
                encoding: SourceStorageEncoding::Identity,
                decoded_sha256: digest(request.as_bytes()),
                decoded_bytes: 32,
            },
            request_evidence_limit: "fixture receipt".into(),
            body: Some(SourceObservedFile {
                path: "body.gz".into(),
                storage_sha256: digest(&encoded),
                encoding: SourceStorageEncoding::Gzip,
                decoded_sha256: digest(body),
                decoded_bytes: body.len() as u64,
            }),
            artifact_prestate: None,
            absent_prior_expectation_reason: "new observation".into(),
            document_identity_limit: "attribution unresolved".into(),
            corrects_observation_key: None,
        };
        let mut fixture = Fixture { root, store, spec };
        fixture.spec.request_evidence.decoded_bytes = request.len() as u64;
        fixture
    }
    #[test]
    fn source_observation_retains_absence_and_replays_exactly() {
        let mut fixture = fixture();
        let report = fixture
            .store
            .record_source_observation(&fixture.root, &fixture.spec)
            .unwrap();
        assert!(report["prior_body_expectation"].is_null());
        assert_eq!(report["document_identity"], "unresolved");
        assert_eq!(
            fixture
                .store
                .record_source_observation(&fixture.root, &fixture.spec)
                .unwrap(),
            report
        );
        assert!(refuse_source_observation_history_loss(&fixture.store.conn).is_err());
        for sql in [
            "UPDATE source_observations SET source_key='changed'",
            "DELETE FROM source_observations",
        ] {
            assert!(fixture.store.conn.execute(sql, []).is_err());
        }
        fixture.spec.reason.push_str(" changed");
        assert!(
            fixture
                .store
                .record_source_observation(&fixture.root, &fixture.spec)
                .is_err()
        );
    }
    #[test]
    fn source_observation_rejects_invalid_evidence_without_writes() {
        for case in 0..14 {
            let mut fixture = fixture();
            match case {
                0 => fixture.spec.body.as_mut().unwrap().decoded_sha256 = "0".repeat(64),
                1 => fixture.spec.body.as_mut().unwrap().decoded_bytes = 2,
                2 => fixture.spec.request_evidence.path = "../escape".into(),
                3 => fixture.spec.http_status = Some(403),
                4 => fixture.spec.body = None,
                5 => fixture.spec.observed_at = "2026-09-06T00:00:00Z".into(),
                6 => fixture.spec.absent_prior_expectation_reason.clear(),
                7 => fixture.spec.requested_url = "https://secret@example.org".into(),
                8 => {
                    fixture.spec.artifact_prestate = Some(SourceArtifactPrestate {
                        artifact_id: "missing".into(),
                        artifact_key: "missing".into(),
                        canonical_url: None,
                    })
                }
                9 => {
                    fs::write(
                        fixture.root.join("body.gz"),
                        b"version https://git-lfs.github.com/spec/v1\n",
                    )
                    .unwrap();
                    let body = fixture.spec.body.as_mut().unwrap();
                    body.encoding = SourceStorageEncoding::Identity;
                    body.storage_sha256 = digest(b"version https://git-lfs.github.com/spec/v1\n");
                    body.decoded_sha256 = body.storage_sha256.clone();
                    body.decoded_bytes = 41;
                }
                10 => fixture.spec.requested_url = "https://different.example.org".into(),
                11 => fixture.spec.http_status = Some(201),
                12 => fixture.spec.observed_at = "2026-09-07".into(),
                13 => fixture.spec.final_url = Some("https://different.example.org".into()),
                _ => unreachable!(),
            }
            assert!(
                fixture
                    .store
                    .record_source_observation(&fixture.root, &fixture.spec)
                    .is_err(),
                "case {case}"
            );
            let count: i64 = fixture
                .store
                .conn
                .query_row("SELECT count(*) FROM source_observations", [], |row| {
                    row.get(0)
                })
                .unwrap();
            assert_eq!(count, 0);
        }
    }
    #[test]
    fn source_observation_preserves_nullable_artifact_and_records_failure() {
        let mut fixture = fixture();
        fixture.store.conn.execute("INSERT INTO artifacts VALUES ('A','key','title','citation','citation_only',0,NULL,NULL)",[]).unwrap();
        fixture.spec.artifact_prestate = Some(SourceArtifactPrestate {
            artifact_id: "A".into(),
            artifact_key: "key".into(),
            canonical_url: None,
        });
        fixture.spec.body = None;
        fixture.spec.outcome = SourceTransportOutcome::RequestFailed;
        fixture.spec.http_status = Some(403);
        let request=b"--2026-09-06 12:00:00--  https://example.org\n  HTTP/1.1 403 Forbidden\nERROR 403: Forbidden.\n";
        fs::write(fixture.root.join("request.txt"), request).unwrap();
        fixture.spec.request_evidence.storage_sha256 = digest(request);
        fixture.spec.request_evidence.decoded_sha256 = digest(request);
        fixture.spec.request_evidence.decoded_bytes = request.len() as u64;
        let report = fixture
            .store
            .record_source_observation(&fixture.root, &fixture.spec)
            .unwrap();
        assert_eq!(
            report["correspondence_basis"],
            "retained_failure_log_without_body"
        );
        let url: Option<String> = fixture
            .store
            .conn
            .query_row(
                "SELECT canonical_functional_url FROM artifacts WHERE id='A'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(url.is_none());
        fixture
            .store
            .conn
            .execute(
                "UPDATE artifacts SET canonical_functional_url='https://example.org' WHERE id='A'",
                [],
            )
            .unwrap();
        assert!(
            fixture
                .store
                .record_source_observation(&fixture.root, &fixture.spec)
                .is_err()
        );
    }
    #[test]
    fn source_observation_rechecks_files_on_replay() {
        let mut fixture = fixture();
        fixture
            .store
            .record_source_observation(&fixture.root, &fixture.spec)
            .unwrap();
        fs::write(fixture.root.join("request.txt"), b"changed").unwrap();
        assert!(
            fixture
                .store
                .record_source_observation(&fixture.root, &fixture.spec)
                .is_err()
        );
    }
    #[test]
    fn wget_same_length_body_substitution_retains_operator_association_boundary() {
        let mut fixture = fixture();
        let original = fixture.spec.body.as_ref().unwrap();
        let substituted = vec![b'x'; original.decoded_bytes as usize];
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(&substituted).unwrap();
        let encoded = encoder.finish().unwrap();
        fs::write(fixture.root.join("body.gz"), &encoded).unwrap();
        let body = fixture.spec.body.as_mut().unwrap();
        body.storage_sha256 = digest(&encoded);
        body.decoded_sha256 = digest(&substituted);
        let report = fixture
            .store
            .record_source_observation(&fixture.root, &fixture.spec)
            .unwrap();
        assert_eq!(
            report["correspondence_basis"],
            "operator_association_with_matching_size"
        );
        assert_eq!(report["cryptographic_request_binding"], false);
        assert_eq!(report["document_identity"], "unresolved");
        assert_eq!(report["scientific_promotion"], false);
    }
    #[test]
    fn source_manifest_admission_binds_url_date_and_body_without_raw_log_claims() {
        let mut fixture = fixture();
        let manifest = format!(
            "retrieval_date = \"2026-09-06\"\n[[source]]\nid = \"source\"\nurl = \"https://example.org\"\nsha256 = \"{}\"\n",
            fixture.spec.body.as_ref().unwrap().decoded_sha256
        );
        fixture.spec.request_evidence.storage_sha256 = digest(manifest.as_bytes());
        fixture.spec.request_evidence.decoded_sha256 = digest(manifest.as_bytes());
        fixture.spec.request_evidence.decoded_bytes = manifest.len() as u64;
        fs::write(fixture.root.join("request.txt"), manifest).unwrap();
        fixture.spec.tool = "curl (audit-manifest attribution)".into();
        fixture.spec.final_url = None;
        fixture.spec.http_status = None;
        let valid = fixture.spec.clone();
        for case in 0..5 {
            fixture.spec = valid.clone();
            match case {
                0 => fixture.spec.source_key = "other".into(),
                1 => fixture.spec.requested_url = "https://other.example.org".into(),
                2 => fixture.spec.http_status = Some(200),
                3 => fixture.spec.final_url = Some("https://example.org".into()),
                4 => fixture.spec.body.as_mut().unwrap().decoded_sha256 = "0".repeat(64),
                _ => unreachable!(),
            }
            assert!(
                fixture
                    .store
                    .record_source_observation(&fixture.root, &fixture.spec)
                    .is_err()
            );
        }
        fixture.spec = valid;
        let report = fixture
            .store
            .record_source_observation(&fixture.root, &fixture.spec)
            .unwrap();
        assert_eq!(
            report["correspondence_basis"],
            "retained_manifest_digest_association"
        );
        assert_eq!(report["cryptographic_request_binding"], false);
    }
    #[test]
    fn source_observation_rolls_back_rejected_insert() {
        let mut fixture = fixture();
        fixture.store.conn.execute_batch("CREATE TRIGGER reject_source_insert BEFORE INSERT ON source_observations BEGIN SELECT RAISE(ABORT, 'injected failure'); END;").unwrap();
        assert!(
            fixture
                .store
                .record_source_observation(&fixture.root, &fixture.spec)
                .is_err()
        );
        let count: i64 = fixture
            .store
            .conn
            .query_row("SELECT count(*) FROM source_observations", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(count, 0);
    }
    #[test]
    fn source_metadata_correction_preserves_predecessor_and_original_spec_encoding() {
        let mut fixture = fixture();
        assert!(
            serde_json::to_value(&fixture.spec)
                .unwrap()
                .get("corrects_observation_key")
                .is_none()
        );
        let original = fixture
            .store
            .record_source_observation(&fixture.root, &fixture.spec)
            .unwrap();
        assert_eq!(original["record_kind"], "observation");
        let original_text: String = fixture
            .store
            .conn
            .query_row(
                "SELECT report_json FROM source_observations WHERE observation_key='test'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        fixture.spec.observation_key = "correction".into();
        fixture.spec.corrects_observation_key = Some("test".into());
        fixture.spec.actor = "reviewer".into();
        fixture.spec.reason = "Clarify correspondence metadata without changing acquisition".into();
        let report = fixture
            .store
            .record_source_observation(&fixture.root, &fixture.spec)
            .unwrap();
        assert_eq!(report["record_kind"], "metadata_correction");
        assert_eq!(report["corrects_observation_key"], "test");
        assert_eq!(
            fixture
                .store
                .record_source_observation(&fixture.root, &fixture.spec)
                .unwrap(),
            report
        );
        let retained: String = fixture
            .store
            .conn
            .query_row(
                "SELECT report_json FROM source_observations WHERE observation_key='test'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(retained, original_text);
        let mut branched = fixture.spec.clone();
        branched.observation_key = "parallel-correction".into();
        assert!(
            fixture
                .store
                .record_source_observation(&fixture.root, &branched)
                .unwrap_err()
                .to_string()
                .contains("already has a successor")
        );
        fixture.spec.observation_key = "correction-chain".into();
        fixture.spec.corrects_observation_key = Some("correction".into());
        fixture
            .store
            .record_source_observation(&fixture.root, &fixture.spec)
            .unwrap();
    }
    #[test]
    fn source_metadata_correction_rejects_changed_acquisition_or_invalid_predecessor() {
        for case in 0..8 {
            let mut fixture = fixture();
            fixture
                .store
                .record_source_observation(&fixture.root, &fixture.spec)
                .unwrap();
            fixture.spec.observation_key = "correction".into();
            fixture.spec.corrects_observation_key = Some("test".into());
            match case {
                0 => fixture.spec.corrects_observation_key = Some("missing".into()),
                1 => fixture.spec.corrects_observation_key = Some("correction".into()),
                2 => fixture.spec.corrects_observation_key = Some(String::new()),
                3 => fixture.spec.source_key = "another-source".into(),
                4 => fixture.spec.tool = "another-tool".into(),
                5 => fixture.spec.document_identity_limit = "changed identity scope".into(),
                6 => fixture.spec.absent_prior_expectation_reason = "changed acquisition".into(),
                7 => fixture.spec.request_evidence_limit = "changed request scope".into(),
                _ => unreachable!(),
            }
            assert!(
                fixture
                    .store
                    .record_source_observation(&fixture.root, &fixture.spec)
                    .is_err(),
                "case {case}"
            );
            let count: i64 = fixture
                .store
                .conn
                .query_row("SELECT count(*) FROM source_observations", [], |row| {
                    row.get(0)
                })
                .unwrap();
            assert_eq!(count, 1);
        }
    }
}
