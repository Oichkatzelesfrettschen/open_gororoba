//! Exact retrieval associations preserve historical expectations and transport evidence.

use crate::{ProvenanceStore, artifact_paths::verified_file_bytes};
use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use rusqlite::{Connection, OptionalExtension, params};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalEvidenceFile {
    pub path: String,
    pub sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DocumentIdentityStatus {
    Verified,
    Unresolved,
}
impl DocumentIdentityStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::Unresolved => "unresolved",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetrievalRequestEvidence {
    pub requested_url: String,
    pub final_url: String,
    pub method: String,
    pub http_status: u16,
    pub completed: bool,
    pub observed_at: String,
    pub tool: String,
    pub body_sha256: Option<String>,
    pub body_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactRetrievalSpec {
    pub schema_version: u32,
    pub observation_key: String,
    pub actor: String,
    pub reason: String,
    pub artifact_id: String,
    pub artifact_key: String,
    pub expected_canonical_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub historical_expectation_url: Option<String>,
    pub expected_sha256: String,
    pub expected_bytes: u64,
    pub correct_canonical_url: bool,
    pub expectation_source: RetrievalEvidenceFile,
    pub request_evidence: RetrievalEvidenceFile,
    pub response: Option<RetrievalEvidenceFile>,
    pub document_identity: DocumentIdentityStatus,
    pub document_identity_evidence: Option<RetrievalEvidenceFile>,
}

fn digest(bytes: impl AsRef<[u8]>) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn verify_url(value: &str) -> Result<()> {
    let url = url::Url::parse(value)?;
    if !matches!(url.scheme(), "http" | "https") || url.host_str().is_none() {
        bail!("retrieval identity requires an HTTP(S) URL: {value}");
    }
    if !url.username().is_empty() || url.password().is_some() {
        bail!("retrieval evidence must exclude URL credentials");
    }
    Ok(())
}

fn evidence_bytes(root: &Path, evidence: &RetrievalEvidenceFile) -> Result<Vec<u8>> {
    verified_file_bytes(root, &evidence.path, &evidence.sha256)
}

fn validate_evidence(
    root: &Path,
    spec: &ArtifactRetrievalSpec,
) -> Result<RetrievalRequestEvidence> {
    if spec.schema_version != 1 {
        bail!("retrieval specification requires schema_version 1");
    }
    for value in [
        &spec.observation_key,
        &spec.actor,
        &spec.reason,
        &spec.artifact_id,
        &spec.artifact_key,
    ] {
        if value.trim().is_empty() {
            bail!("retrieval specification identifiers and rationale must be nonempty");
        }
    }
    verify_url(&spec.expected_canonical_url)?;
    let historical_url = spec
        .historical_expectation_url
        .as_ref()
        .unwrap_or(&spec.expected_canonical_url);
    verify_url(historical_url)?;
    let expectation: toml::Value = toml::from_str(std::str::from_utf8(&evidence_bytes(
        root,
        &spec.expectation_source,
    )?)?)?;
    let records = expectation
        .get("inventory_row")
        .and_then(toml::Value::as_array)
        .context("expectation source requires inventory_row records")?;
    let rows: Vec<_> = records
        .iter()
        .filter(|row| row.get("id").and_then(toml::Value::as_str) == Some(&spec.artifact_id))
        .collect();
    if rows.len() != 1 {
        bail!("expectation source requires exactly one matching artifact ID");
    }
    let row = rows[0];
    for (field, expected) in [
        ("key", &spec.artifact_key),
        ("url", historical_url),
        ("sha256", &spec.expected_sha256),
    ] {
        if row.get(field).and_then(toml::Value::as_str) != Some(expected.as_str()) {
            bail!("historical expectation {field} differs from specification");
        }
    }
    let expected_bytes =
        i64::try_from(spec.expected_bytes).context("expected size exceeds SQLite integer range")?;
    if row.get("byte_length").and_then(toml::Value::as_integer) != Some(expected_bytes) {
        bail!("historical expectation byte count differs from specification");
    }
    if spec.expected_sha256.len() != 64
        || !spec
            .expected_sha256
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("historical expectation requires a lowercase SHA256");
    }
    let request: RetrievalRequestEvidence =
        serde_json::from_slice(&evidence_bytes(root, &spec.request_evidence)?)?;
    verify_url(&request.requested_url)?;
    verify_url(&request.final_url)?;
    DateTime::parse_from_rfc3339(&request.observed_at)?;
    if request.method != "GET" || request.tool.trim().is_empty() {
        bail!("request evidence requires a GET method and named retrieval tool");
    }
    i64::try_from(request.body_bytes).context("response size exceeds SQLite integer range")?;
    if request.completed && !(200..300).contains(&request.http_status) {
        bail!("complete retrieval requires a successful HTTP response");
    }
    match &spec.response {
        Some(response) => {
            let bytes = evidence_bytes(root, response)?;
            if request.body_sha256.as_deref() != Some(response.sha256.as_str())
                || request.body_bytes != bytes.len() as u64
            {
                bail!("retained response differs from request evidence digest or size");
            }
        }
        None => {
            if request.completed || request.body_sha256.is_some() || request.body_bytes != 0 {
                bail!("response-free failure requires zero bytes and an absent body digest");
            }
        }
    }
    match (&spec.document_identity, &spec.document_identity_evidence) {
        (DocumentIdentityStatus::Verified, Some(evidence)) => {
            if evidence_bytes(root, evidence)?.is_empty() || !request.completed {
                bail!(
                    "verified document attribution requires retained evidence and a complete response"
                );
            }
        }
        (DocumentIdentityStatus::Verified, None) => {
            bail!("verified document attribution requires evidence")
        }
        (DocumentIdentityStatus::Unresolved, Some(evidence)) => {
            evidence_bytes(root, evidence)?;
        }
        (DocumentIdentityStatus::Unresolved, None) => {}
    }
    Ok(request)
}

fn snapshot(connection: &Connection, artifact_id: &str) -> Result<Value> {
    let artifact = connection
        .query_row(
            "SELECT key, canonical_functional_url FROM artifacts WHERE id=?1",
            [artifact_id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
        )
        .optional()?
        .context("retrieval artifact ID does not exist")?;
    let mut query = connection.prepare(
        "SELECT url,relation FROM artifact_links WHERE artifact_id=?1 ORDER BY url,relation",
    )?;
    let links = query
        .query_map([artifact_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    let metadata = connection.query_row(
        "SELECT title,citation,status,minimum_requirement_met,canonical_download_path FROM artifacts WHERE id=?1",
        [artifact_id],
        |row| Ok(json!({"title":row.get::<_,String>(0)?,"citation":row.get::<_,String>(1)?,
            "status":row.get::<_,String>(2)?,"minimum_requirement_met":row.get::<_,i64>(3)?,
            "canonical_download_path":row.get::<_,Option<String>>(4)?})),
    )?;
    Ok(json!({"key": artifact.0, "canonical_url": artifact.1, "links": links,"metadata":metadata}))
}

pub(crate) fn refuse_retrieval_history_loss(connection: &Connection) -> Result<()> {
    let exists: bool = connection.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='artifact_retrieval_observations')", [], |row| row.get(0))?;
    if exists
        && connection.query_row(
            "SELECT count(*) FROM artifact_retrieval_observations",
            [],
            |row| row.get::<_, i64>(0),
        )? > 0
    {
        bail!(
            "refusing to discard canonical artifact retrieval history; compatibility imports cannot restore exact response associations"
        );
    }
    Ok(())
}

impl ProvenanceStore {
    pub fn record_artifact_retrieval(
        &mut self,
        repo_root: &Path,
        spec: &ArtifactRetrievalSpec,
    ) -> Result<Value> {
        let request = validate_evidence(repo_root, spec)?;
        let matches = request.completed
            && request.body_sha256.as_deref() == Some(spec.expected_sha256.as_str())
            && request.body_bytes == spec.expected_bytes;
        if spec.correct_canonical_url && !matches {
            bail!(
                "canonical URL correction requires a complete GET matching the historical digest and size"
            );
        }
        let spec_sha256 = digest(serde_json::to_vec(spec)?);
        let transaction = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        let retained: Option<(String, String)> = transaction.query_row(
            "SELECT spec_sha256,report_json FROM artifact_retrieval_observations WHERE observation_key=?1", [&spec.observation_key],
            |row| Ok((row.get(0)?, row.get(1)?)),
        ).optional()?;
        if let Some((prior_sha256, report)) = retained {
            if prior_sha256 != spec_sha256 {
                bail!("observation key already records a different specification");
            }
            let report: Value = serde_json::from_str(&report)?;
            if snapshot(&transaction, &spec.artifact_id)? != report["after"] {
                bail!("retrieval post-state drift");
            }
            return Ok(report);
        }
        let before = snapshot(&transaction, &spec.artifact_id)?;
        if before["key"] != spec.artifact_key
            || before["canonical_url"] != spec.expected_canonical_url
        {
            bail!("stale artifact key or canonical URL pre-state");
        }
        if spec.correct_canonical_url {
            for url in [
                &spec.expected_canonical_url,
                &request.requested_url,
                &request.final_url,
            ] {
                transaction.execute(
                    "INSERT INTO links (url) VALUES (?1) ON CONFLICT DO NOTHING",
                    [url],
                )?;
            }
            transaction.execute("INSERT INTO artifact_links (artifact_id,url,relation) VALUES (?1,?2,'historical_canonical_url') ON CONFLICT DO NOTHING", params![spec.artifact_id,spec.expected_canonical_url])?;
            for url in [&request.requested_url, &request.final_url] {
                transaction.execute("INSERT INTO artifact_links (artifact_id,url,relation) VALUES (?1,?2,'all_links') ON CONFLICT DO NOTHING", params![spec.artifact_id,url])?;
            }
            transaction.execute(
                "UPDATE artifacts SET canonical_functional_url=?2 WHERE id=?1",
                params![spec.artifact_id, request.requested_url],
            )?;
        }
        let after = snapshot(&transaction, &spec.artifact_id)?;
        let report = json!({"schema_version": 1, "spec_sha256": spec_sha256, "spec": spec, "request": request,
            "digest_matches": matches, "before": before, "after": after,
            "outcome": if !request.completed {"retrieval_failed"} else if matches {"expected_bytes_matched"} else {"expected_bytes_mismatched"}});
        transaction.execute(
            "INSERT INTO artifact_retrieval_observations (observation_key,artifact_id,artifact_key,original_url,requested_url,final_url,expected_sha256,expected_bytes,response_path,observed_sha256,observed_bytes,completed,http_status,digest_matches,canonical_url_corrected,document_identity,recorded_at,spec_sha256,report_json) VALUES (?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19)",
            params![spec.observation_key,spec.artifact_id,spec.artifact_key,spec.expected_canonical_url,request.requested_url,request.final_url,spec.expected_sha256,spec.expected_bytes as i64,spec.response.as_ref().map(|response| response.path.as_str()),request.body_sha256,request.body_bytes as i64,request.completed,request.http_status,matches,spec.correct_canonical_url,spec.document_identity.as_str(),Utc::now().to_rfc3339(),spec_sha256,serde_json::to_string(&report)?],
        )?;
        transaction.commit()?;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        fs,
        path::PathBuf,
        process::Command,
        sync::atomic::{AtomicUsize, Ordering},
    };
    static NEXT: AtomicUsize = AtomicUsize::new(0);
    struct Fixture {
        root: PathBuf,
        store: ProvenanceStore,
        spec: ArtifactRetrievalSpec,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.root).expect("remove retrieval fixture");
        }
    }
    fn git(root: &Path, arguments: &[&str]) {
        let mut command = Command::new("git");
        for (name, _) in std::env::vars_os() {
            if name.as_encoded_bytes().starts_with(b"GIT_") {
                command.env_remove(name);
            }
        }
        assert!(
            command
                .arg("-C")
                .arg(root)
                .args(arguments)
                .status()
                .unwrap()
                .success()
        );
    }
    fn evidence(root: &Path, path: &str, bytes: &[u8]) -> RetrievalEvidenceFile {
        fs::write(root.join(path), bytes).unwrap();
        git(root, &["add", "--", path]);
        RetrievalEvidenceFile {
            path: path.into(),
            sha256: digest(bytes),
        }
    }
    fn fixture() -> Fixture {
        let root = std::env::temp_dir().join(format!(
            "retrieval-identity-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        git(&root, &["init", "--quiet"]);
        let store = ProvenanceStore::open(&root.join("store.sqlite3")).unwrap();
        store.conn.execute("INSERT INTO artifacts VALUES ('A','artifact:a','title','citation','unverified',0,'https://example.org/abstract','old.pdf')", []).unwrap();
        let response = evidence(&root, "response.pdf", b"expected document bytes");
        let expected_bytes = 23;
        let expectation = format!(
            "[[inventory_row]]\nid='A'\nkey='artifact:a'\nurl='https://example.org/abstract'\nsha256='{}'\nbyte_length={expected_bytes}\n",
            response.sha256
        );
        let expectation_source = evidence(&root, "expectation.toml", expectation.as_bytes());
        let request = RetrievalRequestEvidence {
            requested_url: "https://example.org/document-v1.pdf".into(),
            final_url: "https://example.org/document-v1.pdf".into(),
            method: "GET".into(),
            http_status: 200,
            completed: true,
            observed_at: "2026-09-05T00:00:00Z".into(),
            tool: "fixture".into(),
            body_sha256: Some(response.sha256.clone()),
            body_bytes: expected_bytes,
        };
        let request_evidence = evidence(
            &root,
            "request.json",
            &serde_json::to_vec(&request).unwrap(),
        );
        let spec = ArtifactRetrievalSpec {
            schema_version: 1,
            observation_key: "retrieval:a".into(),
            actor: "test".into(),
            reason: "verify exact correspondence".into(),
            artifact_id: "A".into(),
            artifact_key: "artifact:a".into(),
            expected_canonical_url: "https://example.org/abstract".into(),
            historical_expectation_url: None,
            expected_sha256: response.sha256.clone(),
            expected_bytes,
            correct_canonical_url: true,
            expectation_source,
            request_evidence,
            response: Some(response),
            document_identity: DocumentIdentityStatus::Unresolved,
            document_identity_evidence: None,
        };
        Fixture { root, store, spec }
    }
    fn change_response(fixture: &mut Fixture, bytes: &[u8], completed: bool, status: u16) {
        let response = evidence(&fixture.root, "response.pdf", bytes);
        let mut request: RetrievalRequestEvidence =
            serde_json::from_slice(&fs::read(fixture.root.join("request.json")).unwrap()).unwrap();
        request.body_sha256 = Some(response.sha256.clone());
        request.body_bytes = bytes.len() as u64;
        request.completed = completed;
        request.http_status = status;
        fixture.spec.response = Some(response);
        fixture.spec.request_evidence = evidence(
            &fixture.root,
            "request.json",
            &serde_json::to_vec(&request).unwrap(),
        );
    }
    #[test]
    fn correction_preserves_expectation_and_replays_exactly() -> Result<()> {
        let mut fixture = fixture();
        let report = fixture
            .store
            .record_artifact_retrieval(&fixture.root, &fixture.spec)?;
        assert_eq!(report["digest_matches"], true);
        assert_eq!(report["spec"]["document_identity"], "unresolved");
        assert_eq!(report["before"]["metadata"], report["after"]["metadata"]);
        assert_eq!(
            report,
            fixture
                .store
                .record_artifact_retrieval(&fixture.root, &fixture.spec)?
        );
        let retained: (String,String,String)=fixture.store.conn.query_row("SELECT original_url,expected_sha256,document_identity FROM artifact_retrieval_observations",[],|row|Ok((row.get(0)?,row.get(1)?,row.get(2)?)))?;
        assert_eq!(
            retained,
            (
                fixture.spec.expected_canonical_url.clone(),
                fixture.spec.expected_sha256.clone(),
                "unresolved".into()
            )
        );
        assert!(
            fixture
                .store
                .conn
                .execute("DELETE FROM artifact_retrieval_observations", [])
                .is_err()
        );
        assert!(
            fixture
                .store
                .conn
                .execute(
                    "UPDATE artifact_retrieval_observations SET observed_bytes=1",
                    []
                )
                .is_err()
        );
        assert!(crate::table_ops::clear_tables(&fixture.store.conn).is_err());
        assert!(
            ProvenanceStore::ensure_artifact_reimport_safe(&fixture.root.join("store.sqlite3"))
                .is_err()
        );
        Ok(())
    }
    #[test]
    fn mismatch_and_failure_record_observations_without_correction() -> Result<()> {
        for (bytes, completed, status, outcome) in [
            (
                b"other bytes".as_slice(),
                true,
                200,
                "expected_bytes_mismatched",
            ),
            (b"".as_slice(), false, 403, "retrieval_failed"),
        ] {
            let mut fixture = fixture();
            change_response(&mut fixture, bytes, completed, status);
            let before = snapshot(&fixture.store.conn, "A")?;
            assert!(
                fixture
                    .store
                    .record_artifact_retrieval(&fixture.root, &fixture.spec)
                    .is_err()
            );
            fixture.spec.correct_canonical_url = false;
            let report = fixture
                .store
                .record_artifact_retrieval(&fixture.root, &fixture.spec)?;
            assert_eq!(report["outcome"], outcome);
            assert_eq!(snapshot(&fixture.store.conn, "A")?, before);
            assert_eq!(
                report["spec"]["expected_sha256"],
                fixture.spec.expected_sha256
            );
        }
        Ok(())
    }
    #[test]
    fn historical_expectation_and_canonical_prestate_are_independently_pinned() -> Result<()> {
        let mut fixture = fixture();
        let historical_url = fixture.spec.expected_canonical_url.clone();
        let canonical_url = "https://example.org/document-v1.pdf";
        fixture.store.conn.execute(
            "UPDATE artifacts SET canonical_functional_url=?1 WHERE id='A'",
            [canonical_url],
        )?;
        fixture.spec.historical_expectation_url = Some(historical_url.clone());
        fixture.spec.correct_canonical_url = false;
        assert!(
            fixture
                .store
                .record_artifact_retrieval(&fixture.root, &fixture.spec)
                .is_err()
        );
        fixture.spec.expected_canonical_url = canonical_url.into();
        for (field, value) in [
            ("url", "https://example.org/wrong"),
            ("key", "wrong"),
            ("hash", "wrong"),
            ("id", "wrong"),
        ] {
            let mut invalid = fixture.spec.clone();
            match field {
                "url" => invalid.historical_expectation_url = Some(value.into()),
                "key" => invalid.artifact_key = value.into(),
                "hash" => invalid.expected_sha256 = value.into(),
                _ => invalid.artifact_id = value.into(),
            }
            assert!(
                fixture
                    .store
                    .record_artifact_retrieval(&fixture.root, &invalid)
                    .is_err()
            );
        }
        let before = snapshot(&fixture.store.conn, "A")?;
        let report = fixture
            .store
            .record_artifact_retrieval(&fixture.root, &fixture.spec)?;
        assert_eq!(snapshot(&fixture.store.conn, "A")?, before);
        assert_eq!(report["spec"]["historical_expectation_url"], historical_url);
        assert_eq!(report["spec"]["expected_canonical_url"], canonical_url);
        assert_eq!(report["digest_matches"], true);
        Ok(())
    }
    #[test]
    fn rejects_false_evidence_stale_identity_and_unattributed_document() -> Result<()> {
        let mut fixture = fixture();
        let mut invalid = fixture.spec.clone();
        invalid.expected_sha256 = "0".repeat(64);
        assert!(
            fixture
                .store
                .record_artifact_retrieval(&fixture.root, &invalid)
                .is_err()
        );
        invalid = fixture.spec.clone();
        invalid.artifact_key = "wrong-key".into();
        assert!(
            fixture
                .store
                .record_artifact_retrieval(&fixture.root, &invalid)
                .is_err()
        );
        invalid = fixture.spec.clone();
        invalid.document_identity = DocumentIdentityStatus::Verified;
        assert!(
            fixture
                .store
                .record_artifact_retrieval(&fixture.root, &invalid)
                .is_err()
        );
        fs::write(fixture.root.join("response.pdf"), b"changed bytes")?;
        assert!(
            fixture
                .store
                .record_artifact_retrieval(&fixture.root, &fixture.spec)
                .is_err()
        );
        assert_eq!(
            fixture.store.conn.query_row(
                "SELECT count(*) FROM artifact_retrieval_observations",
                [],
                |row| row.get::<_, i64>(0)
            )?,
            0
        );
        Ok(())
    }
    #[test]
    fn late_insert_failure_rolls_back_url_and_history() -> Result<()> {
        let mut fixture = fixture();
        let before = snapshot(&fixture.store.conn, "A")?;
        fixture.store.conn.execute_batch("CREATE TRIGGER reject_retrieval BEFORE INSERT ON artifact_retrieval_observations BEGIN SELECT RAISE(ABORT,'injected late failure'); END;")?;
        assert!(
            fixture
                .store
                .record_artifact_retrieval(&fixture.root, &fixture.spec)
                .is_err()
        );
        assert_eq!(snapshot(&fixture.store.conn, "A")?, before);
        assert_eq!(
            fixture.store.conn.query_row(
                "SELECT count(*) FROM artifact_retrieval_observations",
                [],
                |row| row.get::<_, i64>(0)
            )?,
            0
        );
        Ok(())
    }
    #[test]
    fn replay_refuses_changed_specification_and_post_state() -> Result<()> {
        let mut fixture = fixture();
        fixture
            .store
            .record_artifact_retrieval(&fixture.root, &fixture.spec)?;
        let mut changed = fixture.spec.clone();
        changed.reason = "changed rationale".into();
        assert!(
            fixture
                .store
                .record_artifact_retrieval(&fixture.root, &changed)
                .is_err()
        );
        fixture.store.conn.execute("UPDATE artifacts SET canonical_functional_url='https://example.org/drift' WHERE id='A'",[])?;
        assert!(
            fixture
                .store
                .record_artifact_retrieval(&fixture.root, &fixture.spec)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn head_response_and_stale_canonical_url_fail_admission() -> Result<()> {
        let mut fixture = fixture();
        let mut request: RetrievalRequestEvidence =
            serde_json::from_slice(&fs::read(fixture.root.join("request.json"))?)?;
        request.method = "HEAD".into();
        let mut specification = fixture.spec.clone();
        specification.request_evidence =
            evidence(&fixture.root, "head.json", &serde_json::to_vec(&request)?);
        assert!(
            fixture
                .store
                .record_artifact_retrieval(&fixture.root, &specification)
                .is_err()
        );
        fixture.store.conn.execute(
            "UPDATE artifacts SET canonical_functional_url='https://example.org/changed' WHERE id='A'", [],
        )?;
        assert!(
            fixture
                .store
                .record_artifact_retrieval(&fixture.root, &fixture.spec)
                .is_err()
        );
        assert_eq!(
            fixture.store.conn.query_row(
                "SELECT count(*) FROM artifact_retrieval_observations",
                [],
                |row| row.get::<_, i64>(0),
            )?,
            0
        );
        Ok(())
    }
}
