//! Canonical exact-response identities take precedence over inferred URL aliases.

use super::{
    Result, RetentionSet, UnifiedArtifact, artifact_retention, dedupe, materialization_target,
    retrieval_command,
};
use anyhow::{Context, bail};
use rusqlite::{Connection, OpenFlags};
use std::{
    collections::HashMap,
    io::ErrorKind,
    path::{Component, Path},
};

pub(super) struct VerifiedRetrieval {
    pub url: String,
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

pub(super) fn load_verified_retrievals(root: &Path) -> Result<HashMap<String, VerifiedRetrieval>> {
    let database = root.join("registry/canonical/control_plane.sqlite3");
    if !database.is_file() {
        return Ok(HashMap::new());
    }
    let connection = Connection::open_with_flags(database, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    let exists: bool = connection.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name='artifact_retrieval_observations')", [], |row| row.get(0))?;
    if !exists {
        return Ok(HashMap::new());
    }
    let mut query = connection.prepare("SELECT o.artifact_key,o.requested_url,o.response_path,o.expected_sha256,o.expected_bytes,o.observed_sha256,o.observed_bytes,o.http_status FROM artifact_retrieval_observations o JOIN artifacts a ON a.id=o.artifact_id AND a.key=o.artifact_key AND a.canonical_functional_url=o.requested_url WHERE o.digest_matches=1 AND o.completed=1 ORDER BY o.rowid DESC")?;
    let rows = query.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            VerifiedRetrieval {
                url: row.get(1)?,
                path: row.get(2)?,
                sha256: row.get(3)?,
                bytes: u64::try_from(row.get::<_, i64>(4)?)
                    .map_err(|_| rusqlite::Error::InvalidQuery)?,
            },
            row.get::<_, Option<String>>(5)?,
            row.get::<_, i64>(6)?,
            row.get::<_, u16>(7)?,
        ))
    })?;
    let mut identities = HashMap::new();
    for row in rows {
        let (key, identity, observed_sha256, observed_bytes, http_status) = row?;
        if identities.contains_key(&key) {
            continue;
        }
        if observed_sha256.as_deref() != Some(identity.sha256.as_str())
            || u64::try_from(observed_bytes).ok() != Some(identity.bytes)
            || !(200..300).contains(&http_status)
        {
            bail!("inconsistent canonical retrieval receipt for {key}");
        }
        if identity.path.is_empty()
            || !Path::new(&identity.path)
                .components()
                .all(|part| matches!(part, Component::Normal(_)))
        {
            bail!("invalid canonical retrieval response path");
        }
        let mut path = root.to_path_buf();
        let mut absent = false;
        for component in Path::new(&identity.path).components() {
            path.push(component);
            let metadata = match std::fs::symlink_metadata(&path) {
                Ok(metadata) => metadata,
                Err(error) if error.kind() == ErrorKind::NotFound => {
                    absent = true;
                    break;
                }
                Err(error) => return Err(error.into()),
            };
            if metadata.file_type().is_symlink() {
                bail!("symlink in canonical retrieval response path");
            }
        }
        if !absent {
            if !std::fs::metadata(&path)?.is_file() {
                bail!("canonical retrieval response path is not a regular file");
            }
            let bytes = std::fs::read(&path).context("read canonical retrieval response")?;
            let observed = match artifact_retention::lfs_pointer_identity(&bytes)? {
                Some(pointer) => pointer,
                None => artifact_retention::identity_from_bytes(&bytes),
            };
            if observed.sha256 != identity.sha256 || observed.byte_length != identity.bytes {
                bail!("canonical retrieval response object identity changed for {key}");
            }
        }
        identities.insert(key, identity);
    }
    Ok(identities)
}

pub(super) fn apply_verified_retrievals(
    artifacts: &mut [UnifiedArtifact],
    identities: &HashMap<String, VerifiedRetrieval>,
    retention: &RetentionSet,
) {
    for artifact in artifacts {
        let Some(identity) = identities.get(&artifact.key) else {
            continue;
        };
        artifact.links.push(identity.url.clone());
        artifact.links = dedupe(std::mem::take(&mut artifact.links));
        artifact.canonical_functional_url = identity.url.clone();
        artifact.sha256 = identity.sha256.clone();
        artifact.byte_length = identity.bytes;
        if retention.contains(&identity.path) {
            artifact.downloaded_paths.push(identity.path.clone());
            artifact.downloaded_paths = dedupe(std::mem::take(&mut artifact.downloaded_paths));
            artifact.canonical_download_path = identity.path.clone();
            artifact.status = "downloaded".to_string();
            artifact.retrieval_command.clear();
        } else {
            artifact.host_only_paths.push(identity.path.clone());
            artifact.host_only_paths = dedupe(std::mem::take(&mut artifact.host_only_paths));
            artifact.canonical_download_path.clear();
            artifact.status = "remotely_materializable".to_string();
            artifact.retrieval_command = retrieval_command(
                &identity.url,
                &materialization_target(&artifact.key, &artifact.lane),
            );
        }
        artifact.minimum_requirement_met = true;
        artifact.manual_intervention_required = false;
        artifact.manual_intervention_reason.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::params;

    fn fixture() -> Result<(tempfile::TempDir, artifact_retention::FileIdentity)> {
        let fixture = tempfile::tempdir()?;
        let root = fixture.path();
        std::fs::create_dir_all(root.join("registry/canonical"))?;
        std::fs::write(root.join("response.pdf"), b"exact response")?;
        let identity = artifact_retention::file_identity(&root.join("response.pdf")).unwrap();
        let connection = Connection::open(root.join("registry/canonical/control_plane.sqlite3"))?;
        connection.execute_batch(
            "CREATE TABLE artifacts (id TEXT PRIMARY KEY,key TEXT,canonical_functional_url TEXT);",
        )?;
        connection.execute_batch(include_str!(
            "../../../../db/migrations/0021_artifact_retrieval_identity.sql"
        ))?;
        connection.execute(
            "INSERT INTO artifacts VALUES ('A','artifact:a','https://example.org/v1.pdf')",
            [],
        )?;
        connection.execute("INSERT INTO artifact_retrieval_observations VALUES ('record','A','artifact:a','https://example.org/abstract','https://example.org/v1.pdf','https://example.org/v1.pdf',?1,?2,'response.pdf',?1,?2,1,200,1,1,'unresolved','time','spec','{}')",params![identity.sha256,identity.byte_length as i64])?;
        Ok((fixture, identity))
    }

    #[test]
    fn exact_response_overrides_unrelated_local_hash_and_alias() -> Result<()> {
        let (fixture, identity) = fixture()?;
        let root = fixture.path();
        let connection = Connection::open(root.join("registry/canonical/control_plane.sqlite3"))?;
        let identities = load_verified_retrievals(root)?;
        assert_eq!(identities.len(), 1);
        let mut artifacts = vec![UnifiedArtifact {
            key: "artifact:a".into(),
            sha256: "unrelated CSV hash".into(),
            canonical_functional_url: "https://example.org/abstract".into(),
            lane: "papers_pdf".into(),
            ..Default::default()
        }];
        apply_verified_retrievals(
            &mut artifacts,
            &identities,
            &RetentionSet::from_paths(["response.pdf"]),
        );
        assert_eq!(artifacts[0].sha256, identity.sha256);
        assert_eq!(
            artifacts[0].canonical_functional_url,
            "https://example.org/v1.pdf"
        );
        assert_eq!(artifacts[0].canonical_download_path, "response.pdf");
        assert_eq!(artifacts[0].status, "downloaded");
        std::fs::write(root.join("response.pdf"), b"drift")?;
        assert!(load_verified_retrievals(root).is_err());
        connection.execute(
            "UPDATE artifacts SET canonical_functional_url='https://example.org/another-version'",
            [],
        )?;
        assert!(load_verified_retrievals(root)?.is_empty());
        Ok(())
    }

    fn pointer(identity: &artifact_retention::FileIdentity) -> String {
        format!(
            "version https://git-lfs.github.com/spec/v1\noid sha256:{}\nsize {}\n",
            identity.sha256, identity.byte_length
        )
    }

    fn assert_unavailable_host_payload(root: &Path) {
        let host = artifact_retention::observe_host_materialization(
            root,
            &RetentionSet::from_paths(["response.pdf"]),
            "A",
            "artifact:a",
            "downloaded",
            "response.pdf",
        );
        assert!(!host.present);
        assert!(host.sha256.is_empty());
        assert_eq!(host.byte_length, 0);
        assert!(host.git_tracked);
    }

    #[test]
    fn absent_response_preserves_canonical_receipt_without_host_payload() -> Result<()> {
        let (fixture, identity) = fixture()?;
        let root = fixture.path();
        std::fs::remove_file(root.join("response.pdf"))?;
        let identities = load_verified_retrievals(root)?;
        assert_eq!(identities["artifact:a"].sha256, identity.sha256);
        assert_unavailable_host_payload(root);
        Ok(())
    }

    #[test]
    fn matching_lfs_pointer_preserves_object_identity_without_host_payload() -> Result<()> {
        let (fixture, identity) = fixture()?;
        let root = fixture.path();
        for encoded in [pointer(&identity), pointer(&identity).replace('\n', "\r\n")] {
            std::fs::write(root.join("response.pdf"), encoded)?;
            let identities = load_verified_retrievals(root)?;
            assert_eq!(identities["artifact:a"].sha256, identity.sha256);
            assert_eq!(identities["artifact:a"].bytes, identity.byte_length);
            assert_unavailable_host_payload(root);
            let artifact = UnifiedArtifact {
                downloaded_paths: vec!["response.pdf".into()],
                ..Default::default()
            };
            assert!(
                super::super::host_observations::first_local_identity(&artifact, root).is_none()
            );
        }
        Ok(())
    }

    #[test]
    fn mismatching_and_malformed_lfs_pointers_fail_export() -> Result<()> {
        let (fixture, identity) = fixture()?;
        let root = fixture.path();
        for encoded in [
            pointer(&identity).replace(&identity.sha256, &"0".repeat(64)),
            pointer(&identity).replace(
                &format!("size {}", identity.byte_length),
                &format!("size {}", identity.byte_length + 1),
            ),
            pointer(&identity).replace("oid sha256:", "oid sha1:"),
            pointer(&identity).replace("size ", "size +"),
            format!("{}extra field\n", pointer(&identity)),
        ] {
            std::fs::write(root.join("response.pdf"), encoded)?;
            assert!(load_verified_retrievals(root).is_err());
            assert_unavailable_host_payload(root);
        }
        Ok(())
    }

    #[test]
    fn corrupted_materialized_response_fails_export() -> Result<()> {
        let (fixture, _) = fixture()?;
        std::fs::write(fixture.path().join("response.pdf"), b"wrong response")?;
        assert!(load_verified_retrievals(fixture.path()).is_err());
        Ok(())
    }

    #[test]
    fn absent_payload_does_not_excuse_inconsistent_canonical_receipt() -> Result<()> {
        let (fixture, identity) = fixture()?;
        let root = fixture.path();
        std::fs::remove_file(root.join("response.pdf"))?;
        let connection = Connection::open(root.join("registry/canonical/control_plane.sqlite3"))?;
        connection.execute_batch("PRAGMA ignore_check_constraints=ON;")?;
        connection.execute("INSERT INTO artifact_retrieval_observations VALUES ('inconsistent','A','artifact:a','https://example.org/abstract','https://example.org/v1.pdf','https://example.org/v1.pdf',?1,?2,'response.pdf','wrong digest',?2,1,200,1,1,'unresolved','time','spec','{}')",params![identity.sha256,identity.byte_length as i64])?;
        assert!(load_verified_retrievals(root).is_err());
        Ok(())
    }
}
