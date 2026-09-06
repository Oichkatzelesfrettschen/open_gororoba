//! Hash-bound release retention, independent of host Git tracking observations.
use anyhow::{Context, Result, bail};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
};

pub const MANIFEST_PATH: &str = "data/retention/scientific-payloads.json";
pub const MAX_MANIFEST_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchiveMetadata {
    pub url: String,
    pub sha256: String,
    pub byte_length: u64,
    pub format: String,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchiveMember {
    pub path: String,
    pub sha256: String,
    pub byte_length: u64,
    pub archive_member: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    schema_version: u32,
    archive: ArchiveMetadata,
    files: Vec<ArchiveMember>,
    objects: Vec<ArchiveObject>,
}
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArchiveObject {
    pub sha256: String,
    pub byte_length: u64,
    pub archive_member: String,
}
#[derive(Debug, PartialEq, Eq)]
pub enum Materialization {
    Missing,
    Verified,
}
pub struct RetainedArchive {
    root: PathBuf,
    pub archive: ArchiveMetadata,
    members: BTreeMap<String, ArchiveMember>,
    pub objects: BTreeMap<String, ArchiveObject>,
}
fn hash_valid(hash: &str) -> bool {
    hash.len() == 64
        && hash
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
fn validate_path(path: &str) -> Result<()> {
    if path.is_empty()
        || path.contains('\\')
        || path.contains(':')
        || path.chars().any(char::is_control)
        || path.split('/').any(|part| {
            part.is_empty()
                || matches!(part, "." | "..")
                || part.eq_ignore_ascii_case(".git")
                || part.ends_with(['.', ' '])
        })
        || Path::new(path).is_absolute()
    {
        bail!("retention path must be normalized repository-relative components: {path}");
    }
    Ok(())
}
fn materialized_path(root: &Path, relative: &str) -> Result<Option<PathBuf>> {
    validate_path(relative)?;
    let mut full = root.to_path_buf();
    let parts: Vec<_> = relative.split('/').collect();
    for (index, part) in parts.iter().enumerate() {
        full.push(part);
        let metadata = match fs::symlink_metadata(&full) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        if metadata.file_type().is_symlink() {
            bail!("symlink component in retention path: {relative}");
        }
        if index + 1 == parts.len() {
            if !metadata.is_file() {
                bail!("retention member must be a regular file: {relative}");
            }
        } else if !metadata.is_dir() {
            bail!("retention parent must be a directory: {relative}");
        }
    }
    Ok(Some(full))
}
fn git(root: &Path) -> Command {
    let mut command = Command::new("git");
    for (name, _) in std::env::vars_os() {
        if name.as_encoded_bytes().starts_with(b"GIT_") {
            command.env_remove(name);
        }
    }
    command.arg("-C").arg(root);
    command
}
fn git_output(root: &Path, arguments: &[&str]) -> Result<Vec<u8>> {
    let output = git(root).args(arguments).output()?;
    if !output.status.success() {
        bail!(
            "retention Git inspection failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(output.stdout)
}
fn verify_index(root: &Path, bytes: &[u8]) -> Result<()> {
    let top = git_output(root, &["rev-parse", "--show-toplevel"])?;
    if Path::new(std::str::from_utf8(&top)?.trim_end_matches('\n')).canonicalize()? != root {
        bail!("retention root must match Git worktree root");
    }
    let records = git_output(
        root,
        &[
            "--literal-pathspecs",
            "ls-files",
            "--error-unmatch",
            "--stage",
            "-z",
            "--",
            MANIFEST_PATH,
        ],
    )?;
    let records: Vec<_> = records
        .split(|byte| *byte == 0)
        .filter(|record| !record.is_empty())
        .collect();
    if records.len() != 1 {
        bail!("retention manifest requires one stage-0 index entry");
    }
    let record = std::str::from_utf8(records[0])?;
    let (metadata, path) = record
        .split_once('\t')
        .context("malformed retention index entry")?;
    let fields: Vec<_> = metadata.split_whitespace().collect();
    if fields.len() != 3
        || !matches!(fields[0], "100644" | "100755")
        || fields[2] != "0"
        || path != MANIFEST_PATH
    {
        bail!("retention manifest requires a regular stage-0 index entry");
    }
    // Object lookup bypasses working-tree filters and binds the exact indexed JSON.
    let size = git_output(root, &["cat-file", "-s", fields[1]])?;
    if std::str::from_utf8(&size)?.trim().parse::<u64>()? > MAX_MANIFEST_BYTES {
        bail!("indexed retention manifest exceeds size bound");
    }
    let indexed = git_output(root, &["cat-file", "blob", fields[1]])?;
    if indexed != bytes {
        bail!("retention manifest bytes differ from index blob");
    }
    Ok(())
}
impl RetainedArchive {
    pub fn load_optional(root: &Path) -> Result<Option<Self>> {
        let root = root.canonicalize()?;
        if materialized_path(&root, MANIFEST_PATH)?.is_none() {
            let indexed = git_output(
                &root,
                &[
                    "--literal-pathspecs",
                    "ls-files",
                    "--stage",
                    "-z",
                    "--",
                    MANIFEST_PATH,
                ],
            )?;
            if !indexed.is_empty() {
                bail!("indexed retention manifest is missing from worktree");
            }
            return Ok(None);
        }
        Self::load(&root).map(Some)
    }
    pub fn members(&self) -> impl Iterator<Item = &ArchiveMember> {
        self.members.values()
    }
    pub fn load(root: &Path) -> Result<Self> {
        let root = root.canonicalize()?;
        let path =
            materialized_path(&root, MANIFEST_PATH)?.context("retention manifest is missing")?;
        let mut bytes = Vec::new();
        fs::File::open(path)?
            .take(MAX_MANIFEST_BYTES + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() as u64 > MAX_MANIFEST_BYTES {
            bail!("retention manifest exceeds size bound");
        }
        verify_index(&root, &bytes)?;
        Self::parse(root, &bytes)
    }
    fn parse(root: PathBuf, bytes: &[u8]) -> Result<Self> {
        let manifest: Manifest = serde_json::from_slice(bytes)?;
        if manifest.schema_version != 1 {
            bail!("unsupported retention schema version");
        }
        let url = url::Url::parse(&manifest.archive.url)?;
        if url.scheme() != "https"
            || url.host_str().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
            || url.fragment().is_some()
        {
            bail!("retention archive requires an HTTPS URL without credentials or fragment");
        }
        if !hash_valid(&manifest.archive.sha256)
            || manifest.archive.byte_length == 0
            || manifest.archive.format != "tar-zstd"
        {
            bail!("invalid retention archive identity or format");
        }
        let mut members = BTreeMap::new();
        let mut identities = BTreeMap::new();
        for object in manifest.objects {
            if !hash_valid(&object.sha256) {
                bail!("invalid retained object hash");
            }
            let expected = format!(
                "{}/{}/{}",
                &object.sha256[..2],
                &object.sha256[2..4],
                object.sha256
            );
            if object.archive_member != expected {
                bail!("retained object member must match hash layout");
            }
            if identities.insert(object.sha256.clone(), object).is_some() {
                bail!("duplicate retained object identity");
            }
        }
        for member in manifest.files {
            validate_path(&member.path)?;
            if member.path == MANIFEST_PATH || !hash_valid(&member.sha256) {
                bail!("invalid retention member identity");
            }
            let expected = format!(
                "{}/{}/{}",
                &member.sha256[..2],
                &member.sha256[2..4],
                member.sha256
            );
            if member.archive_member != expected {
                bail!("retention archive member must match hash layout");
            }
            materialized_path(&root, &member.path)?;
            let object = identities
                .get(&member.sha256)
                .context("retention file references undeclared object")?;
            if object.byte_length != member.byte_length
                || object.archive_member != member.archive_member
            {
                bail!("retention file conflicts with object identity");
            }
            if members.insert(member.path.clone(), member).is_some() {
                bail!("duplicate retention path");
            }
        }
        // A file path cannot also serve as another retained file's parent.
        for path in members.keys() {
            for (separator, _) in path.match_indices('/') {
                if members.contains_key(&path[..separator]) {
                    bail!("retention paths conflict as file and directory");
                }
            }
        }
        Ok(Self {
            root,
            archive: manifest.archive,
            members,
            objects: identities,
        })
    }
    pub fn resolve(&self, path: &str, expected_sha256: &str) -> Result<Option<&ArchiveMember>> {
        validate_path(path)?;
        if !hash_valid(expected_sha256) {
            bail!("invalid expected retention SHA256");
        }
        let Some(member) = self.members.get(path) else {
            return Ok(None);
        };
        if member.sha256 != expected_sha256 {
            bail!("expected SHA256 conflicts with retention manifest");
        }
        Ok(Some(member))
    }
    pub fn materialization(&self, path: &str, expected_sha256: &str) -> Result<Materialization> {
        let member = self
            .resolve(path, expected_sha256)?
            .context("path is absent from retention manifest")?;
        let Some(full) = materialized_path(&self.root, path)? else {
            return Ok(Materialization::Missing);
        };
        let mut file = fs::File::open(full)?;
        if file.metadata()?.len() != member.byte_length {
            bail!("retention member length mismatch");
        }
        let mut hasher = Sha256::new();
        let mut reader = Read::by_ref(&mut file).take(member.byte_length.saturating_add(1));
        let mut copied = 0_u64;
        let mut buffer = [0_u8; 65536];
        loop {
            let count = reader.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            copied += count as u64;
            hasher.update(&buffer[..count]);
        }
        let digest: String = hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        if copied != member.byte_length || digest != member.sha256 {
            bail!("retention member SHA256 mismatch");
        }
        Ok(Materialization::Verified)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(0);
    struct Fixture(PathBuf);
    impl Fixture {
        fn new() -> Self {
            let root = std::env::temp_dir().join(format!(
                "retained-archive-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&root).unwrap();
            git_output(&root, &["init", "--quiet"]).unwrap();
            Self(root)
        }
        fn stage(&self, value: &Value) {
            fs::create_dir_all(self.0.join("data/retention")).unwrap();
            fs::write(
                self.0.join(MANIFEST_PATH),
                serde_json::to_vec(value).unwrap(),
            )
            .unwrap();
            git_output(&self.0, &["add", "--", MANIFEST_PATH]).unwrap();
        }
    }
    impl Drop for Fixture {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }
    fn spec() -> Value {
        let hash: String = Sha256::digest(b"payload")
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let member = format!("{}/{}/{}", &hash[..2], &hash[2..4], hash);
        json!({"schema_version":1,"archive":{"url":"https://example.org/archive.zst","sha256":"a".repeat(64),"byte_length":100,"format":"tar-zstd"},
            "objects":[{"sha256":hash,"byte_length":7,"archive_member":member}],
            "files":[{"path":"data/payload.bin","sha256":hash,"byte_length":7,"archive_member":member}]})
    }
    #[test]
    fn indexed_manifest_distinguishes_missing_verified_and_corrupt() {
        let fixture = Fixture::new();
        let value = spec();
        fixture.stage(&value);
        let retained = RetainedArchive::load(&fixture.0).unwrap();
        let hash = value["files"][0]["sha256"].as_str().unwrap();
        assert_eq!(
            retained.materialization("data/payload.bin", hash).unwrap(),
            Materialization::Missing
        );
        fs::write(fixture.0.join("data/payload.bin"), b"payload").unwrap();
        assert_eq!(
            retained.materialization("data/payload.bin", hash).unwrap(),
            Materialization::Verified
        );
        fs::write(fixture.0.join("data/payload.bin"), b"changed").unwrap();
        assert!(retained.materialization("data/payload.bin", hash).is_err());
        assert!(
            retained
                .resolve("data/payload.bin", &"b".repeat(64))
                .is_err()
        );
        assert!(retained.resolve("data/unknown", hash).unwrap().is_none());
    }
    #[test]
    fn rejects_untracked_and_modified_manifest() {
        let fixture = Fixture::new();
        fixture.stage(&spec());
        fs::write(fixture.0.join(MANIFEST_PATH), b"{}").unwrap();
        assert!(RetainedArchive::load(&fixture.0).is_err());
        git_output(&fixture.0, &["rm", "-f", "--cached", "--", MANIFEST_PATH]).unwrap();
        assert!(RetainedArchive::load(&fixture.0).is_err());
    }
    #[test]
    fn rejects_schema_identity_and_path_mutations() {
        let fixture = Fixture::new();
        let mut cases = Vec::new();
        for path in [
            "/absolute",
            "a/../b",
            "a//b",
            "a/./b",
            "a/",
            ".git/config",
            ".GiT/config",
            "C:/alias",
            "a./file",
            r"a\b",
        ] {
            let mut value = spec();
            value["files"][0]["path"] = json!(path);
            cases.push(value);
        }
        let mut value = spec();
        let duplicate = value["files"][0].clone();
        value["files"].as_array_mut().unwrap().push(duplicate);
        cases.push(value);
        let mut value = spec();
        value["files"][0]["byte_length"] = json!(8);
        cases.push(value);
        let mut value = spec();
        value["objects"][0]["archive_member"] = json!("arbitrary");
        cases.push(value);
        let mut value = spec();
        value["objects"][0]["sha256"] = json!("A".repeat(64));
        cases.push(value);
        let mut value = spec();
        value["archive"]["url"] = json!("http://example.org/archive");
        cases.push(value);
        let mut value = spec();
        value["schema_version"] = json!(2);
        cases.push(value);
        let mut value = spec();
        value["unknown"] = json!(true);
        cases.push(value);
        for value in cases {
            fixture.stage(&value);
            assert!(
                RetainedArchive::load(&fixture.0).is_err(),
                "accepted {value}"
            );
        }
    }
    #[cfg(unix)]
    #[test]
    fn rejects_symlink_member_and_manifest_parents() {
        let fixture = Fixture::new();
        fixture.stage(&spec());
        std::os::unix::fs::symlink("absent", fixture.0.join("data/payload.bin")).unwrap();
        assert!(RetainedArchive::load(&fixture.0).is_err());
        fs::remove_file(fixture.0.join("data/payload.bin")).unwrap();
        fs::rename(fixture.0.join("data/retention"), fixture.0.join("stored")).unwrap();
        std::os::unix::fs::symlink("../stored", fixture.0.join("data/retention")).unwrap();
        assert!(RetainedArchive::load(&fixture.0).is_err());
    }
    #[test]
    fn optional_loading_rejects_missing_indexed_and_oversized_manifest() {
        let fixture = Fixture::new();
        assert!(
            RetainedArchive::load_optional(&fixture.0)
                .unwrap()
                .is_none()
        );
        fixture.stage(&spec());
        fs::remove_file(fixture.0.join(MANIFEST_PATH)).unwrap();
        assert!(RetainedArchive::load_optional(&fixture.0).is_err());
        let file = fs::File::create(fixture.0.join(MANIFEST_PATH)).unwrap();
        file.set_len(MAX_MANIFEST_BYTES + 1).unwrap();
        assert!(RetainedArchive::load(&fixture.0).is_err());
    }
    #[test]
    fn rejects_duplicate_objects_and_file_parent_conflicts() {
        let fixture = Fixture::new();
        let mut value = spec();
        let duplicate = value["objects"][0].clone();
        value["objects"].as_array_mut().unwrap().push(duplicate);
        fixture.stage(&value);
        assert!(RetainedArchive::load(&fixture.0).is_err());
        let mut value = spec();
        let mut child = value["files"][0].clone();
        child["path"] = json!("data/payload.bin/child");
        value["files"].as_array_mut().unwrap().push(child);
        fixture.stage(&value);
        assert!(RetainedArchive::load(&fixture.0).is_err());
    }
}
