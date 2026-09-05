// SPDX-License-Identifier: GPL-2.0-or-later

//! Metadata admission tests use synthetic URLs and local bytes. Remote content
//! equality and retrieval availability require independent observations.

use super::*;

const REMOTE_URL: &str = "https://example.org/evidence.pdf";
const CONTENTS: &[u8] = b"retained scientific evidence\n";

fn classify_one(
    root: &Path,
    paths: &[&str],
    links: &[&str],
    retention: &RetentionSet,
    carried: HashMap<String, DurableFacts>,
) -> UnifiedArtifact {
    let mut artifacts = vec![UnifiedArtifact {
        key: "artifact:materializability-fixture".to_string(),
        title: "Materializability fixture".to_string(),
        local_paths: paths.iter().map(|path| (*path).to_string()).collect(),
        links: links.iter().map(|url| (*url).to_string()).collect(),
        ..Default::default()
    }];
    classify_artifacts(
        &mut artifacts,
        &HashMap::new(),
        &HashMap::new(),
        root,
        retention,
        &carried,
    );
    artifacts.remove(0)
}

fn rendered_row(artifact: &UnifiedArtifact) -> Result<Value> {
    let text = render_artifact_registry(
        std::slice::from_ref(artifact),
        &["ASOT-0001".to_string()],
        &[],
        &[],
        "2026-01-01",
    );
    let registry: Value = toml::from_str(&text)?;
    Ok(registry["artifact"][0].clone())
}

fn assert_valid(root: &Path, artifact: &UnifiedArtifact) -> Result<()> {
    let row = rendered_row(artifact)?;
    let mut state = ValidationState::default();
    let missing = if artifact.minimum_requirement_met {
        Vec::new()
    } else {
        vec![artifact.key.clone()]
    };
    validate_artifact_entry(0, &row, root, &missing, &mut state);
    assert!(state.failures.is_empty(), "{:?}", state.failures);
    Ok(())
}

#[test]
fn linkless_host_file_keeps_identity_without_claiming_remote_retrieval() -> Result<()> {
    let root = tempfile::tempdir()?;
    fs::write(root.path().join("evidence.pdf"), CONTENTS)?;
    let artifact = classify_one(
        root.path(),
        &["evidence.pdf"],
        &[],
        &RetentionSet::default(),
        HashMap::new(),
    );
    assert_eq!(artifact.status, "citation_only_no_link");
    assert_eq!(artifact.host_only_paths, ["evidence.pdf"]);
    assert_eq!(artifact.sha256.len(), 64);
    assert!(!artifact.minimum_requirement_met);
    assert!(artifact.retrieval_command.is_empty());
    assert_valid(root.path(), &artifact)
}

#[test]
fn host_directory_without_content_identity_is_unverified() -> Result<()> {
    let root = tempfile::tempdir()?;
    fs::create_dir(root.path().join("evidence"))?;
    let artifact = classify_one(
        root.path(),
        &["evidence"],
        &[REMOTE_URL],
        &RetentionSet::default(),
        HashMap::new(),
    );
    assert_eq!(artifact.status, "unverified");
    assert!(artifact.sha256.is_empty());
    assert!(!artifact.minimum_requirement_met);
    assert_valid(root.path(), &artifact)
}

#[test]
fn remote_url_and_measured_host_file_admit_materialization_metadata() -> Result<()> {
    let root = tempfile::tempdir()?;
    fs::write(root.path().join("evidence.pdf"), CONTENTS)?;
    let artifact = classify_one(
        root.path(),
        &["evidence.pdf"],
        &[REMOTE_URL],
        &RetentionSet::default(),
        HashMap::new(),
    );
    assert_eq!(artifact.status, "remotely_materializable");
    assert!(artifact.minimum_requirement_met);
    assert!(artifact.downloaded_paths.is_empty());
    assert_eq!(artifact.byte_length, CONTENTS.len() as u64);
    assert!(artifact.retrieval_command.contains(REMOTE_URL));
    assert_valid(root.path(), &artifact)
}

#[test]
fn doi_key_with_measured_direct_content_admits_remote_metadata() -> Result<()> {
    let root = tempfile::tempdir()?;
    fs::write(root.path().join("evidence.pdf"), CONTENTS)?;
    let mut artifacts = [UnifiedArtifact {
        key: "doi:10.1234/evidence".to_string(),
        title: "Direct content for a DOI identity".to_string(),
        local_paths: vec!["evidence.pdf".to_string()],
        links: vec![REMOTE_URL.to_string()],
        ..Default::default()
    }];
    classify_artifacts(
        &mut artifacts,
        &HashMap::new(),
        &HashMap::new(),
        root.path(),
        &RetentionSet::default(),
        &HashMap::new(),
    );
    let artifact = &artifacts[0];
    assert_eq!(artifact.status, "remotely_materializable");
    assert!(artifact.minimum_requirement_met);
    assert_eq!(artifact.canonical_functional_url, REMOTE_URL);
    assert_eq!(artifact.sha256.len(), 64);
    assert_valid(root.path(), artifact)
}

#[test]
fn carried_identity_admits_remote_metadata_without_local_bytes() -> Result<()> {
    let root = tempfile::tempdir()?;
    let carried = HashMap::from([(
        "artifact:materializability-fixture".to_string(),
        DurableFacts {
            sha256: "ab".repeat(32),
            byte_length: 27,
            canonical_url: REMOTE_URL.to_string(),
            ..Default::default()
        },
    )]);
    let artifact = classify_one(
        root.path(),
        &[],
        &[REMOTE_URL],
        &RetentionSet::default(),
        carried,
    );
    assert_eq!(artifact.status, "remotely_materializable");
    assert_eq!(artifact.sha256, "ab".repeat(32));
    assert!(artifact.host_only_paths.is_empty());
    assert!(artifact.minimum_requirement_met);
    assert_valid(root.path(), &artifact)
}

#[test]
fn tracked_file_admits_retention_without_remote_identity() -> Result<()> {
    let root = tempfile::tempdir()?;
    fs::write(root.path().join("evidence.pdf"), CONTENTS)?;
    let artifact = classify_one(
        root.path(),
        &["evidence.pdf"],
        &[],
        &RetentionSet::from_paths(["evidence.pdf"]),
        HashMap::new(),
    );
    assert_eq!(artifact.status, "downloaded");
    assert_eq!(artifact.downloaded_paths, ["evidence.pdf"]);
    assert!(artifact.host_only_paths.is_empty());
    assert!(artifact.minimum_requirement_met);
    assert_valid(root.path(), &artifact)
}

#[test]
fn malformed_remote_identity_fails_classification_and_validation() -> Result<()> {
    let root = tempfile::tempdir()?;
    for (url, hash) in [
        (REMOTE_URL, "ab".repeat(31)),
        (REMOTE_URL, "zz".repeat(32)),
        ("file:///tmp/evidence.pdf", "ab".repeat(32)),
        ("https://", "ab".repeat(32)),
        ("invalid URL", "ab".repeat(32)),
    ] {
        let carried = HashMap::from([(
            "artifact:materializability-fixture".to_string(),
            DurableFacts {
                sha256: hash.clone(),
                canonical_url: url.to_string(),
                ..Default::default()
            },
        )]);
        let artifact = classify_one(root.path(), &[], &[url], &RetentionSet::default(), carried);
        assert_ne!(artifact.status, "remotely_materializable", "{url} {hash}");
        let mut row = rendered_row(&artifact)?;
        row["status"] = Value::String("remotely_materializable".to_string());
        row["minimum_requirement_met"] = Value::Boolean(true);
        let mut state = ValidationState::default();
        validate_artifact_entry(0, &row, root.path(), &[], &mut state);
        assert!(
            state
                .failures
                .iter()
                .any(|failure| failure.contains("HTTP(S) URL and a SHA256")),
            "{:?}",
            state.failures
        );
    }
    Ok(())
}

#[test]
fn staging_keeps_linkless_host_file_in_materialization_manifest() -> Result<()> {
    let root = tempfile::tempdir()?;
    fs::create_dir(root.path().join("registry"))?;
    fs::create_dir(root.path().join("data"))?;
    fs::write(root.path().join("data/evidence.pdf"), CONTENTS)?;
    fs::write(
        root.path().join("registry/fixture.toml"),
        "[[source]]\nid = 'FIXTURE'\ntitle = 'Host evidence'\npath = 'data/evidence.pdf'\n",
    )?;
    let mut set = StagedWriteSet::default();
    let (summary, _) = stage_artifact_source_of_truth(
        root.path(),
        &root.path().join("registry/artifact_source_of_truth.toml"),
        &root.path().join("report.toml"),
        &RetentionSet::default(),
        &mut set,
    )?;
    let observed = summary
        .host_materialization
        .iter()
        .find(|row| row.path == "data/evidence.pdf")
        .expect("host-only artifact remains observable");
    assert!(observed.present);
    assert!(!observed.git_tracked);
    assert_eq!(observed.status, "citation_only_no_link");
    Ok(())
}
