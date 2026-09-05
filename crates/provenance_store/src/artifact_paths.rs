//! Transactional artifact path reconciliation with retained historical identities.

use crate::ProvenanceStore;
use anyhow::{Context, Result, bail};
use chrono::Utc;
use rusqlite::{Connection, params, types::ValueRef};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    path::{Component, Path},
    process::Command,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactPathRepairSpec {
    pub schema_version: u32,
    pub repair_key: String,
    pub actor: String,
    pub reason: String,
    pub repair: Vec<ArtifactPathRepair>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArtifactPathRepair {
    pub id: String,
    pub expected_canonical_path: String,
    pub expected_downloaded_paths: Vec<String>,
    pub replacement_path: String,
    pub replacement_sha256: String,
    pub old_path_relation: HistoricalPathRelation,
    #[serde(default)]
    pub related_path: Vec<RelatedArtifactPath>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistoricalPathRelation {
    Referenced,
    HistoricalDownload,
}
impl HistoricalPathRelation {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Referenced => "referenced",
            Self::HistoricalDownload => "historical_download",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelatedArtifactPath {
    pub path: String,
    pub sha256: String,
    pub relation: TransformedPathRelation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TransformedPathRelation {
    TransformedCopy,
}

fn sha256(bytes: impl AsRef<[u8]>) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn verify_file(root: &Path, relative: &str, expected: &str) -> Result<()> {
    verified_file_bytes(root, relative, expected).map(|_| ())
}

pub(crate) fn verified_file_bytes(root: &Path, relative: &str, expected: &str) -> Result<Vec<u8>> {
    let path = Path::new(relative);
    if relative.is_empty()
        || !path
            .components()
            .all(|part| matches!(part, Component::Normal(_)))
    {
        bail!("artifact path must contain only repository-relative normal components: {relative}");
    }
    if expected.len() != 64
        || !expected
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("invalid lowercase SHA256 for {relative}");
    }
    let mut full = root.canonicalize()?;
    for part in path.components() {
        full.push(part);
        if fs::symlink_metadata(&full)
            .with_context(|| format!("inspect {relative}"))?
            .file_type()
            .is_symlink()
        {
            bail!("symlink component in artifact path {relative}");
        }
    }
    if !fs::metadata(&full)?.is_file() {
        bail!("artifact path is not a regular file: {relative}");
    }
    let bytes = fs::read(&full)?;
    let digest = sha256(&bytes);
    if digest != expected {
        bail!("SHA256 mismatch for {relative}: expected {expected}, observed {digest}");
    }
    verify_tracked_file(root, relative)?;
    Ok(bytes)
}

fn repository_git(root: &Path) -> Command {
    let mut command = Command::new("git");
    // Hook environments can redirect Git to the caller's index and common dir.
    // Repository admission and test fixtures must resolve the explicit root.
    for (name, _) in std::env::vars_os() {
        if name.as_encoded_bytes().starts_with(b"GIT_") {
            command.env_remove(name);
        }
    }
    command.arg("-C").arg(root);
    command
}

fn verify_tracked_file(root: &Path, relative: &str) -> Result<()> {
    let repository = repository_git(root)
        .args(["rev-parse", "--show-toplevel"])
        .output()
        .context("resolve artifact repository Git root")?;
    if !repository.status.success() {
        bail!(
            "artifact repository Git root lookup failed: {}",
            String::from_utf8_lossy(&repository.stderr).trim()
        );
    }
    let observed_root = std::str::from_utf8(&repository.stdout)?.trim_end_matches('\n');
    if Path::new(observed_root).canonicalize()? != root.canonicalize()? {
        bail!("artifact repository root must match the Git worktree root");
    }
    let indexed = repository_git(root)
        .arg("--literal-pathspecs")
        .args([
            "ls-files",
            "--error-unmatch",
            "--stage",
            "-z",
            "--",
            relative,
        ])
        .output()
        .context("inspect artifact Git index membership")?;
    if !indexed.status.success() {
        bail!(
            "artifact path must be Git-tracked: {relative}: {}",
            String::from_utf8_lossy(&indexed.stderr).trim()
        );
    }
    let records: Vec<_> = indexed
        .stdout
        .split(|byte| *byte == 0)
        .filter(|record| !record.is_empty())
        .collect();
    if records.len() != 1 {
        bail!("artifact path requires one resolved Git index entry: {relative}");
    }
    let separator = records[0]
        .iter()
        .position(|byte| *byte == b'\t')
        .context("malformed artifact Git index entry")?;
    let metadata = &records[0][..separator];
    let indexed_path = &records[0][separator + 1..];
    let fields: Vec<_> = std::str::from_utf8(metadata)?.split_whitespace().collect();
    if fields.len() != 3
        || !matches!(fields[0], "100644" | "100755")
        || fields[2] != "0"
        || indexed_path != relative.as_bytes()
    {
        bail!("artifact path requires a regular resolved Git index entry: {relative}");
    }
    Ok(())
}

fn rows(connection: &Connection, query: &str, id: &str) -> Result<Vec<Value>> {
    let mut statement = connection.prepare(query)?;
    let columns: Vec<String> = statement
        .column_names()
        .iter()
        .map(|name| (*name).to_owned())
        .collect();
    let mut cursor = statement.query([id])?;
    let mut result = Vec::new();
    while let Some(row) = cursor.next()? {
        let mut object = serde_json::Map::new();
        for (index, name) in columns.iter().enumerate() {
            let value = match row.get_ref(index)? {
                ValueRef::Null => Value::Null,
                ValueRef::Integer(value) => json!(value),
                ValueRef::Real(value) => json!(value),
                ValueRef::Text(value) => json!(std::str::from_utf8(value)?),
                ValueRef::Blob(_) => bail!("unexpected blob in artifact snapshot"),
            };
            object.insert(name.clone(), value);
        }
        result.push(Value::Object(object));
    }
    Ok(result)
}

fn snapshot(connection: &Connection, id: &str) -> Result<Value> {
    Ok(json!({
        "artifact": rows(connection, "SELECT * FROM artifacts WHERE id = ?1", id)?,
        "artifact_paths": rows(connection, "SELECT * FROM artifact_paths WHERE artifact_id = ?1 ORDER BY path, relation", id)?,
    }))
}

impl ProvenanceStore {
    /// Repair an explicitly bounded artifact set and retain complete row evidence atomically.
    pub fn repair_artifact_paths(
        &mut self,
        repo_root: &Path,
        spec: &ArtifactPathRepairSpec,
    ) -> Result<Value> {
        if spec.schema_version != 1 || spec.repair.is_empty() {
            bail!("repair requires schema_version 1 and a nonempty repair set");
        }
        for (name, value) in [
            ("repair_key", &spec.repair_key),
            ("actor", &spec.actor),
            ("reason", &spec.reason),
        ] {
            if value.trim().is_empty() {
                bail!("{name} must be nonempty");
            }
        }
        let mut ids = BTreeSet::new();
        for repair in &spec.repair {
            if repair.id.is_empty() || !ids.insert(&repair.id) {
                bail!("empty or duplicate artifact ID {}", repair.id);
            }
            let old: BTreeSet<_> = repair.expected_downloaded_paths.iter().collect();
            if old.is_empty()
                || old.len() != repair.expected_downloaded_paths.len()
                || old.iter().any(|path| path.is_empty())
            {
                bail!(
                    "expected downloaded paths must be distinct and nonempty for {}",
                    repair.id
                );
            }
            if !old.contains(&repair.expected_canonical_path) {
                bail!(
                    "expected canonical path must belong to the historical downloaded set for {}",
                    repair.id
                );
            }
            if old.contains(&repair.replacement_path) {
                bail!(
                    "replacement must differ from historical downloaded paths for {}",
                    repair.id
                );
            }
            verify_file(
                repo_root,
                &repair.replacement_path,
                &repair.replacement_sha256,
            )?;
            let mut targets = BTreeSet::from([&repair.replacement_path]);
            for related in &repair.related_path {
                if !targets.insert(&related.path) || old.contains(&related.path) {
                    bail!("duplicate or historical related path for {}", repair.id);
                }
                verify_file(repo_root, &related.path, &related.sha256)?;
            }
        }
        let spec_sha256 = sha256(serde_json::to_vec(spec)?);
        let transaction = self
            .conn
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
        {
            let mut statement = transaction.prepare("SELECT details_json FROM export_runs WHERE action = 'repair-artifact-paths' ORDER BY id")?;
            let reports = statement.query_map([], |row| row.get::<_, String>(0))?;
            for report in reports {
                let report: Value = serde_json::from_str(&report?)?;
                if report["repair_key"] == spec.repair_key {
                    if report["spec_sha256"] != spec_sha256 {
                        bail!("repair key already records a different specification");
                    }
                    let retained = report["repairs"]
                        .as_array()
                        .context("repair report lacks row evidence")?;
                    if retained.len() != spec.repair.len() {
                        bail!("repair report row count mismatch");
                    }
                    for (repair, evidence) in spec.repair.iter().zip(retained) {
                        if evidence["id"] != repair.id
                            || snapshot(&transaction, &repair.id)? != evidence["after"]
                        {
                            bail!("post-repair state drift for {}", repair.id);
                        }
                    }
                    return Ok(report);
                }
            }
        }
        let mut evidence = Vec::new();
        for repair in &spec.repair {
            let before = snapshot(&transaction, &repair.id)?;
            let artifacts = before["artifact"]
                .as_array()
                .context("artifact snapshot is not an array")?;
            if artifacts.len() != 1
                || artifacts[0]["canonical_download_path"] != repair.expected_canonical_path
            {
                bail!(
                    "stale expected canonical path or missing artifact {}",
                    repair.id
                );
            }
            let downloaded: BTreeSet<&str> = before["artifact_paths"]
                .as_array()
                .context("path snapshot is not an array")?
                .iter()
                .filter(|row| row["relation"] == "downloaded")
                .filter_map(|row| row["path"].as_str())
                .collect();
            let expected: BTreeSet<&str> = repair
                .expected_downloaded_paths
                .iter()
                .map(String::as_str)
                .collect();
            if downloaded != expected {
                bail!("stale expected downloaded path set for {}", repair.id);
            }
            for path in &repair.expected_downloaded_paths {
                transaction.execute("INSERT INTO artifact_paths (artifact_id, path, relation) VALUES (?1, ?2, ?3) ON CONFLICT DO NOTHING", params![repair.id, path, repair.old_path_relation.as_str()])?;
            }
            transaction.execute(
                "DELETE FROM artifact_paths WHERE artifact_id = ?1 AND relation = 'downloaded'",
                [&repair.id],
            )?;
            transaction.execute("INSERT INTO artifact_paths (artifact_id, path, relation) VALUES (?1, ?2, 'downloaded')", params![repair.id, repair.replacement_path])?;
            for related in &repair.related_path {
                transaction.execute("INSERT INTO artifact_paths (artifact_id, path, relation) VALUES (?1, ?2, 'transformed_copy') ON CONFLICT DO NOTHING", params![repair.id, related.path])?;
            }
            transaction.execute(
                "UPDATE artifacts SET canonical_download_path = ?2 WHERE id = ?1",
                params![repair.id, repair.replacement_path],
            )?;
            let after = snapshot(&transaction, &repair.id)?;
            evidence.push(json!({"id": repair.id, "before": before, "after": after}));
        }
        let report = json!({"schema_version": 1, "repair_key": spec.repair_key, "actor": spec.actor, "reason": spec.reason,
            "spec_sha256": spec_sha256, "spec_hash_encoding": "serde_json serialization of typed specification", "spec": spec, "repairs": evidence});
        transaction.execute("INSERT INTO export_runs (action, created_at, artifact_count, document_count, details_json) VALUES ('repair-artifact-paths', ?1, ?2, 0, ?3)", params![Utc::now().to_rfc3339(), spec.repair.len() as i64, serde_json::to_string(&report)?])?;
        transaction.commit()?;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        path::PathBuf,
        sync::atomic::{AtomicUsize, Ordering},
    };

    static NEXT: AtomicUsize = AtomicUsize::new(0);
    struct Fixture {
        root: PathBuf,
        store: ProvenanceStore,
        spec: ArtifactPathRepairSpec,
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.root).expect("remove test fixture");
        }
    }
    fn fixture() -> Fixture {
        let root = std::env::temp_dir().join(format!(
            "artifact-path-test-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&root).unwrap();
        fs::write(root.join("replacement.txt"), b"retained bytes").unwrap();
        fs::write(root.join("transformed.txt"), b"transformed bytes").unwrap();
        git(&root, &["init", "--quiet"]);
        git(&root, &["add", "--", "replacement.txt", "transformed.txt"]);
        let store = ProvenanceStore::open(&root.join("test.sqlite3")).unwrap();
        for id in ["A", "B"] {
            store.conn.execute("INSERT INTO artifacts VALUES (?1, ?1, 'title', 'citation', 'local', 1, 'https://example.org', 'old.txt')", [id]).unwrap();
            store
                .conn
                .execute(
                    "INSERT INTO artifact_paths VALUES (?1, 'old.txt', 'downloaded')",
                    [id],
                )
                .unwrap();
            store
                .conn
                .execute(
                    "INSERT INTO artifact_paths VALUES (?1, 'source.txt', 'referenced')",
                    [id],
                )
                .unwrap();
            store
                .conn
                .execute(
                    "INSERT INTO record_sources VALUES ('artifact', ?1, 'primary-source')",
                    [id],
                )
                .unwrap();
            store
                .conn
                .execute(
                    "INSERT INTO lane_assignments VALUES (?1, 'papers_pdf')",
                    [id],
                )
                .unwrap();
            store.conn.execute("INSERT INTO citations (artifact_id, citation_text) VALUES (?1, 'original citation')", [id]).unwrap();
        }
        let spec = ArtifactPathRepairSpec {
            schema_version: 1,
            repair_key: "repair-test".into(),
            actor: "test".into(),
            reason: "retain migrated bytes".into(),
            repair: vec![ArtifactPathRepair {
                id: "A".into(),
                expected_canonical_path: "old.txt".into(),
                expected_downloaded_paths: vec!["old.txt".into()],
                replacement_path: "replacement.txt".into(),
                replacement_sha256: sha256(b"retained bytes"),
                old_path_relation: HistoricalPathRelation::HistoricalDownload,
                related_path: vec![RelatedArtifactPath {
                    path: "transformed.txt".into(),
                    sha256: sha256(b"transformed bytes"),
                    relation: TransformedPathRelation::TransformedCopy,
                }],
            }],
        };
        Fixture { root, store, spec }
    }

    fn git(root: &Path, arguments: &[&str]) {
        let output = repository_git(root).args(arguments).output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    #[test]
    fn artifact_path_repair_isolates_inherited_git_environment() {
        let fixture = fixture();
        let caller = fixture.root.join("caller");
        fs::create_dir(&caller).unwrap();
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "artifact_paths::tests::artifact_path_repair_preserves_metadata_and_replays_exactly",
                "--nocapture",
            ])
            .env("GIT_DIR", caller.join("redirected.git"))
            .env("GIT_COMMON_DIR", caller.join("common.git"))
            .env("GIT_WORK_TREE", &caller)
            .env("GIT_INDEX_FILE", caller.join("index"))
            .env("GIT_CONFIG_COUNT", "1")
            .env("GIT_CONFIG_KEY_0", "core.bare")
            .env("GIT_CONFIG_VALUE_0", "true")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "stdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
        assert_eq!(fs::read_dir(&caller).unwrap().count(), 0);
    }

    #[test]
    fn artifact_path_repair_rejects_untracked_and_ignored_witnesses() {
        for related in [false, true] {
            for ignored in [false, true] {
                let mut fixture = fixture();
                let target = if related {
                    "transformed.txt"
                } else {
                    "replacement.txt"
                };
                git(&fixture.root, &["rm", "--cached", "--", target]);
                if ignored {
                    fs::write(fixture.root.join(".gitignore"), format!("{target}\n")).unwrap();
                }
                let before = snapshot(&fixture.store.conn, "A").unwrap();
                let error = fixture
                    .store
                    .repair_artifact_paths(&fixture.root, &fixture.spec)
                    .unwrap_err();
                assert!(
                    error.to_string().contains("must be Git-tracked"),
                    "{error:#}"
                );
                assert_eq!(snapshot(&fixture.store.conn, "A").unwrap(), before);
            }
        }
    }

    #[test]
    fn artifact_path_repair_rejects_missing_git_repository() {
        let mut fixture = fixture();
        fs::rename(fixture.root.join(".git"), fixture.root.join("retained-git")).unwrap();
        let error = fixture
            .store
            .repair_artifact_paths(&fixture.root, &fixture.spec)
            .unwrap_err();
        assert!(
            error.to_string().contains("Git root lookup failed"),
            "{error:#}"
        );
    }

    #[test]
    fn artifact_path_repair_preserves_metadata_and_replays_exactly() {
        let mut fixture = fixture();
        let untouched = snapshot(&fixture.store.conn, "B").unwrap();
        let preserved_queries = [
            "SELECT * FROM record_sources WHERE entity_id = ?1 ORDER BY source_ref",
            "SELECT * FROM lane_assignments WHERE artifact_id = ?1 ORDER BY lane_name",
            "SELECT * FROM citations WHERE artifact_id = ?1 ORDER BY id",
        ];
        let preserved: Vec<_> = preserved_queries
            .iter()
            .map(|query| rows(&fixture.store.conn, query, "A").unwrap())
            .collect();
        let report = fixture
            .store
            .repair_artifact_paths(&fixture.root, &fixture.spec)
            .unwrap();
        let before = &report["repairs"][0]["before"]["artifact"][0];
        let after = &report["repairs"][0]["after"]["artifact"][0];
        for (name, value) in before.as_object().unwrap() {
            if name != "canonical_download_path" {
                assert_eq!(value, &after[name]);
            }
        }
        assert_eq!(snapshot(&fixture.store.conn, "B").unwrap(), untouched);
        for (query, original) in preserved_queries.iter().zip(preserved) {
            assert_eq!(rows(&fixture.store.conn, query, "A").unwrap(), original);
        }
        assert_eq!(
            fixture
                .store
                .repair_artifact_paths(&fixture.root, &fixture.spec)
                .unwrap(),
            report
        );
        let count: i64 = fixture
            .store
            .conn
            .query_row(
                "SELECT count(*) FROM export_runs WHERE action = 'repair-artifact-paths'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 1);
        let paths = report["repairs"][0]["after"]["artifact_paths"]
            .as_array()
            .unwrap();
        assert!(
            paths
                .iter()
                .any(|row| row["path"] == "old.txt" && row["relation"] == "historical_download")
        );
        assert!(
            paths.iter().any(
                |row| row["path"] == "transformed.txt" && row["relation"] == "transformed_copy"
            )
        );
        fixture.spec.reason.push_str(" changed");
        assert!(
            fixture
                .store
                .repair_artifact_paths(&fixture.root, &fixture.spec)
                .is_err()
        );
        fixture.spec.reason = "retain migrated bytes".into();
        fixture
            .store
            .conn
            .execute("UPDATE artifacts SET title = 'drift' WHERE id = 'A'", [])
            .unwrap();
        assert!(
            fixture
                .store
                .repair_artifact_paths(&fixture.root, &fixture.spec)
                .is_err()
        );
    }

    #[test]
    fn artifact_path_repair_rejects_invalid_admission_atomically() {
        for case in 0..9 {
            let mut fixture = fixture();
            let before = snapshot(&fixture.store.conn, "A").unwrap();
            match case {
                0 => fixture.spec.repair[0].expected_canonical_path = "wrong".into(),
                1 => fixture.spec.repair[0]
                    .expected_downloaded_paths
                    .push("extra".into()),
                2 => fixture.spec.repair[0].replacement_path = "absent".into(),
                3 => fixture.spec.repair[0].replacement_sha256 = "0".repeat(64),
                4 => fixture.spec.repair.push(fixture.spec.repair[0].clone()),
                5 => fixture.spec.repair[0].replacement_path = "../escape".into(),
                6 => fixture.spec.repair[0].replacement_path = "/absolute".into(),
                7 => fixture.spec.repair[0].id = "missing".into(),
                8 => fixture.spec.repair[0].related_path[0].sha256 = "bad".into(),
                _ => unreachable!(),
            }
            assert!(
                fixture
                    .store
                    .repair_artifact_paths(&fixture.root, &fixture.spec)
                    .is_err(),
                "case {case}"
            );
            assert_eq!(snapshot(&fixture.store.conn, "A").unwrap(), before);
            let count: i64 = fixture
                .store
                .conn
                .query_row("SELECT count(*) FROM export_runs", [], |row| row.get(0))
                .unwrap();
            assert_eq!(count, 0);
        }
    }

    #[test]
    fn artifact_path_repair_rolls_back_late_trigger_failure() {
        let mut fixture = fixture();
        let before = snapshot(&fixture.store.conn, "A").unwrap();
        let mut second = fixture.spec.repair[0].clone();
        second.id = "B".into();
        fixture.spec.repair.push(second);
        fixture.store.conn.execute_batch("CREATE TRIGGER reject_second BEFORE UPDATE ON artifacts WHEN OLD.id = 'B' BEGIN SELECT RAISE(ABORT, 'injected late failure'); END;").unwrap();
        assert!(
            fixture
                .store
                .repair_artifact_paths(&fixture.root, &fixture.spec)
                .is_err()
        );
        assert_eq!(snapshot(&fixture.store.conn, "A").unwrap(), before);
        let count: i64 = fixture
            .store
            .conn
            .query_row("SELECT count(*) FROM export_runs", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[cfg(unix)]
    #[test]
    fn artifact_path_repair_rejects_symlink_components() {
        let mut fixture = fixture();
        std::os::unix::fs::symlink(&fixture.root, fixture.root.join("linked")).unwrap();
        fixture.spec.repair[0].replacement_path = "linked/replacement.txt".into();
        assert!(
            fixture
                .store
                .repair_artifact_paths(&fixture.root, &fixture.spec)
                .is_err()
        );
        std::os::unix::fs::symlink("replacement.txt", fixture.root.join("linked-file")).unwrap();
        fixture.spec.repair[0].replacement_path = "linked-file".into();
        assert!(
            fixture
                .store
                .repair_artifact_paths(&fixture.root, &fixture.spec)
                .is_err()
        );
    }

    #[test]
    fn artifact_path_repair_rechecks_files_on_replay() {
        let mut fixture = fixture();
        fixture
            .store
            .repair_artifact_paths(&fixture.root, &fixture.spec)
            .unwrap();
        fs::write(fixture.root.join("replacement.txt"), b"changed").unwrap();
        assert!(
            fixture
                .store
                .repair_artifact_paths(&fixture.root, &fixture.spec)
                .is_err()
        );
    }
}
