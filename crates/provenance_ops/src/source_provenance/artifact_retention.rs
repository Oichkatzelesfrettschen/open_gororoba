// SPDX-License-Identifier: GPL-2.0-or-later
//
// Separates repository truth from per-host materialization for cited artifacts.

use anyhow::{Context, Result, ensure};
use provenance_store::retained_archive::RetainedArchive;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
};

/// User agent the repository's fetch binaries already present, so a recorded
/// retrieval_command reproduces the request that produced the bytes.
pub const RETRIEVAL_USER_AGENT: &str = "gororoba-provenance-fetch/0.1 (research)";

/// Git index membership and hash-bound archive retention are separate facts.
/// Managed retention does not establish host materialization or provider availability.
#[derive(Clone, Debug, Default)]
pub struct RetentionSet {
    paths: BTreeSet<String>,
    archive_identities: BTreeMap<String, FileIdentity>,
}

impl RetentionSet {
    pub fn from_paths<I, S>(paths: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            paths: paths.into_iter().map(Into::into).collect(),
            archive_identities: BTreeMap::new(),
        }
    }

    /// Reads the git index. A checkout without git, or a path outside a work
    /// tree, yields an empty set, which classifies every artifact as
    /// per-host rather than inventing retention that git cannot confirm.
    pub fn from_git_index(repo_root: &Path) -> Self {
        let mut command = Command::new("git");
        for (name, _) in std::env::vars_os() {
            if name.as_encoded_bytes().starts_with(b"GIT_") {
                command.env_remove(name);
            }
        }
        let output = command
            .arg("-C")
            .arg(repo_root)
            .args(["ls-files", "-z"])
            .output();
        let Ok(output) = output else {
            return Self::default();
        };
        if !output.status.success() {
            return Self::default();
        }
        let text = String::from_utf8_lossy(&output.stdout);
        Self::from_paths(
            text.split('\0')
                .filter(|entry| !entry.is_empty())
                .map(str::to_owned),
        )
    }

    pub fn contains(&self, repo_relative: &str) -> bool {
        self.paths.contains(&repo_relative.replace('\\', "/"))
    }

    pub fn from_repository(repo_root: &Path) -> Result<Self> {
        let archive = RetainedArchive::load_optional(repo_root)?;
        let mut retention = Self::from_git_index(repo_root);
        if let Some(archive) = archive {
            retention.archive_identities = archive
                .members()
                .map(|member| {
                    (
                        member.path.clone(),
                        FileIdentity {
                            sha256: member.sha256.clone(),
                            byte_length: member.byte_length,
                        },
                    )
                })
                .collect();
        }
        Ok(retention)
    }

    pub fn is_retained(&self, repo_relative: &str) -> bool {
        self.contains(repo_relative) || self.archive_identities.contains_key(repo_relative)
    }

    pub(super) fn verify_archive_identity(
        &self,
        path: &str,
        sha256: &str,
        byte_length: u64,
    ) -> Result<()> {
        if let Some(identity) = self.archive_identities.get(path) {
            ensure!(
                identity.sha256 == sha256 && identity.byte_length == byte_length,
                "canonical retrieval identity conflicts with retained archive member: {path}"
            );
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.paths.len()
    }

    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }
}

/// Content identity of one file as observed on the host running the export.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileIdentity {
    pub sha256: String,
    pub byte_length: u64,
}

/// Decode the three-field Git LFS v1 representation without treating it as payload bytes.
pub(super) fn lfs_pointer_identity(bytes: &[u8]) -> Result<Option<FileIdentity>> {
    if !bytes.starts_with(b"version https://git-lfs.github.com/spec/") {
        return Ok(None);
    }
    let text = std::str::from_utf8(bytes).context("LFS pointer is not UTF-8")?;
    let mut lines = text.lines();
    ensure!(
        lines.next() == Some("version https://git-lfs.github.com/spec/v1"),
        "unsupported LFS pointer version"
    );
    let sha256 = lines
        .next()
        .and_then(|line| line.strip_prefix("oid sha256:"))
        .context("LFS pointer requires SHA256 object identity")?;
    ensure!(
        sha256.len() == 64
            && sha256
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "invalid LFS pointer SHA256"
    );
    let size = lines
        .next()
        .and_then(|line| line.strip_prefix("size "))
        .context("LFS pointer requires object size")?;
    ensure!(
        !size.is_empty() && size.bytes().all(|byte| byte.is_ascii_digit()),
        "invalid LFS pointer size"
    );
    let byte_length = size.parse().context("LFS pointer size exceeds u64")?;
    ensure!(lines.next().is_none(), "unexpected LFS pointer fields");
    Ok(Some(FileIdentity {
        sha256: sha256.to_owned(),
        byte_length,
    }))
}

pub fn file_identity(path: &Path) -> Option<FileIdentity> {
    let bytes = fs::read(path).ok()?;
    if lfs_pointer_identity(&bytes).ok()?.is_some() {
        return None;
    }
    Some(identity_from_bytes(&bytes))
}

pub(super) fn identity_from_bytes(bytes: &[u8]) -> FileIdentity {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut sha256 = String::with_capacity(digest.len() * 2);
    for byte in digest {
        sha256.push_str(&format!("{byte:02x}"));
    }
    FileIdentity {
        sha256,
        byte_length: bytes.len() as u64,
    }
}

/// The command that re-fetches the artifact on a host that lacks it. An empty
/// URL yields an empty command rather than a plausible-looking line, because a
/// fabricated retrieval path is worse than a recorded gap.
pub fn retrieval_command(canonical_url: &str, target_path: &str) -> String {
    let url = canonical_url.trim();
    if url.is_empty() {
        return String::new();
    }
    let target = target_path.trim();
    if target.is_empty() {
        format!("curl -fSL -A '{RETRIEVAL_USER_AGENT}' '{url}'")
    } else {
        format!("curl -fSL -A '{RETRIEVAL_USER_AGENT}' -o '{target}' '{url}'")
    }
}

/// Rows of the gitignored per-host manifest.
#[derive(Clone, Debug)]
pub struct HostMaterializationRow {
    pub artifact_id: String,
    pub key: String,
    pub status: String,
    pub path: String,
    pub present: bool,
    pub sha256: String,
    pub byte_length: u64,
    pub git_tracked: bool,
}

pub fn observe_host_materialization(
    repo_root: &Path,
    retention: &RetentionSet,
    artifact_id: &str,
    key: &str,
    status: &str,
    path: &str,
) -> HostMaterializationRow {
    let absolute: PathBuf = repo_root.join(path);
    let identity = file_identity(&absolute);
    HostMaterializationRow {
        artifact_id: artifact_id.to_string(),
        key: key.to_string(),
        status: status.to_string(),
        path: path.to_string(),
        present: identity.is_some(),
        sha256: identity
            .as_ref()
            .map(|value| value.sha256.clone())
            .unwrap_or_default(),
        byte_length: identity.map(|value| value.byte_length).unwrap_or_default(),
        git_tracked: retention.contains(path),
    }
}

pub fn render_host_materialization(rows: &[HostMaterializationRow], generated_at: &str) -> String {
    let present = rows.iter().filter(|row| row.present).count();
    let tracked = rows.iter().filter(|row| row.git_tracked).count();
    let mut lines = vec![
        "# Per-host materialization manifest. Host state, not repository truth;".to_string(),
        "# gitignored on purpose. Regenerate with: provenance materialize-status".to_string(),
        String::new(),
        "[host_materialization]".to_string(),
        format!("generated_at = \"{generated_at}\""),
        format!("row_count = {}", rows.len()),
        format!("present_count = {present}"),
        format!("absent_count = {}", rows.len() - present),
        format!("git_tracked_count = {tracked}"),
        String::new(),
    ];
    for row in rows {
        lines.push("[[materialized]]".to_string());
        lines.push(format!("artifact_id = \"{}\"", row.artifact_id));
        lines.push(format!("key = \"{}\"", row.key));
        lines.push(format!("status = \"{}\"", row.status));
        lines.push(format!("path = \"{}\"", row.path));
        lines.push(format!("present = {}", row.present));
        lines.push(format!("git_tracked = {}", row.git_tracked));
        lines.push(format!("sha256 = \"{}\"", row.sha256));
        lines.push(format!("byte_length = {}", row.byte_length));
        lines.push(String::new());
    }
    lines.join("\n")
}

pub fn write_host_materialization(path: &Path, text: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    fs::write(path, text).with_context(|| format!("write {}", path.display()))
}

#[cfg(test)]
pub(super) fn archive_fixture(
    root: &Path,
    path: &str,
    identity: &FileIdentity,
) -> Result<RetentionSet> {
    use provenance_store::retained_archive::MANIFEST_PATH;
    let member = format!(
        "{}/{}/{}",
        &identity.sha256[..2],
        &identity.sha256[2..4],
        identity.sha256
    );
    let manifest = serde_json::json!({"schema_version":1,
        "archive":{"url":"https://example.org/archive.zst","sha256":"a".repeat(64),"byte_length":100,"format":"tar-zstd"},
        "objects":[{"sha256":identity.sha256,"byte_length":identity.byte_length,"archive_member":member}],
        "files":[{"path":path,"sha256":identity.sha256,"byte_length":identity.byte_length,"archive_member":member}]});
    fs::create_dir_all(root.join("data/retention"))?;
    fs::write(root.join(MANIFEST_PATH), serde_json::to_vec(&manifest)?)?;
    for args in [vec!["init", "--quiet"], vec!["add", "--", MANIFEST_PATH]] {
        let mut command = Command::new("git");
        for (name, _) in std::env::vars_os() {
            if name.as_encoded_bytes().starts_with(b"GIT_") {
                command.env_remove(name);
            }
        }
        ensure!(
            command.arg("-C").arg(root).args(args).status()?.success(),
            "fixture Git command failed"
        );
    }
    RetentionSet::from_repository(root)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archived_retention_preserves_missing_host_and_git_observations() -> Result<()> {
        let fixture = tempfile::tempdir()?;
        let identity = identity_from_bytes(b"payload");
        let retention = archive_fixture(fixture.path(), "payload.bin", &identity)?;
        assert!(retention.is_retained("payload.bin"));
        assert!(!retention.contains("payload.bin"));
        let mut artifact = super::super::UnifiedArtifact {
            local_paths: vec!["payload.bin".into(), "host-only.bin".into()],
            ..Default::default()
        };
        super::super::host_observations::classify_paths_and_mirrors(
            &mut artifact,
            &Default::default(),
            &Default::default(),
            &retention,
        );
        assert_eq!(artifact.downloaded_paths, ["payload.bin"]);
        assert_eq!(artifact.host_only_paths, ["host-only.bin"]);
        let observed = observe_host_materialization(
            fixture.path(),
            &retention,
            "A",
            "key",
            "downloaded",
            "payload.bin",
        );
        assert!(!observed.present);
        assert!(!observed.git_tracked);
        assert!(observed.sha256.is_empty());
        fs::write(fixture.path().join("payload.bin"), b"payload")?;
        let observed = observe_host_materialization(
            fixture.path(),
            &retention,
            "A",
            "key",
            "downloaded",
            "payload.bin",
        );
        assert!(observed.present);
        assert!(!observed.git_tracked);
        fs::write(
            fixture
                .path()
                .join(provenance_store::retained_archive::MANIFEST_PATH),
            b"{}",
        )?;
        assert!(RetentionSet::from_repository(fixture.path()).is_err());
        Ok(())
    }

    #[test]
    fn materializable_state_separates_a_tracked_pdf_from_an_absent_one() {
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        fs::create_dir_all(root.join("papers/pdf")).expect("mkdir");
        fs::write(root.join("papers/pdf/tracked.pdf"), b"%PDF-1.4 tracked").expect("write");
        let retention = RetentionSet::from_paths(["papers/pdf/tracked.pdf"]);

        let tracked = observe_host_materialization(
            root,
            &retention,
            "ART-0001",
            "tracked",
            "downloaded",
            "papers/pdf/tracked.pdf",
        );
        assert!(tracked.present);
        assert!(tracked.git_tracked);
        assert_eq!(tracked.byte_length, 16);
        assert_eq!(tracked.sha256.len(), 64);

        let absent = observe_host_materialization(
            root,
            &retention,
            "ART-0002",
            "absent",
            "remotely_materializable",
            "data/papers/intake/absent.pdf",
        );
        assert!(!absent.present);
        assert!(!absent.git_tracked);
        assert!(absent.sha256.is_empty());

        let text = render_host_materialization(&[tracked, absent], "2026-09-01");
        let parsed: toml::Value = toml::from_str(&text).expect("manifest parses");
        assert_eq!(
            parsed["host_materialization"]["present_count"].as_integer(),
            Some(1)
        );
        assert_eq!(
            parsed["host_materialization"]["git_tracked_count"].as_integer(),
            Some(1)
        );
    }

    #[test]
    fn retrieval_command_records_a_gap_rather_than_a_plausible_line() {
        assert!(retrieval_command("", "papers/pdf/x.pdf").is_empty());
        let command = retrieval_command("https://arxiv.org/pdf/1306.1646", "papers/pdf/x.pdf");
        assert!(command.contains(RETRIEVAL_USER_AGENT));
        assert!(command.contains("-o 'papers/pdf/x.pdf'"));
    }
}
