//! Verify an archived scientific corpus before explicit local materialization.

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use provenance_store::retained_archive::{ArchiveObject, MANIFEST_PATH, RetainedArchive};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Component, Path, PathBuf},
};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    manifest: PathBuf,
    /// Local tar-zstd archive whose complete digest matches the manifest.
    #[arg(long)]
    archive: PathBuf,
    #[arg(long)]
    repo_root: PathBuf,
}

#[derive(Default, Serialize)]
struct Report {
    verified_archive_sha256: String,
    verified_objects: usize,
    installed_files: usize,
    reused_files: usize,
    globally_atomic: bool,
    installation_scope: &'static str,
}

#[cfg(test)]
fn hash(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn safe_relative(value: &str) -> Result<()> {
    ensure!(
        !value.is_empty() && !value.contains('\\') && !value.contains('\0') && !value.contains(':'),
        "invalid relative path"
    );
    ensure!(
        value.split('/').all(|part| !part.is_empty()
            && part != "."
            && part != ".."
            && !part.eq_ignore_ascii_case(".git")),
        "unsafe relative path {value}"
    );
    ensure!(
        Path::new(value)
            .components()
            .all(|part| matches!(part, Component::Normal(_))),
        "absolute or non-normal path {value}"
    );
    Ok(())
}

fn copy_hashed(
    reader: &mut impl Read,
    writer: &mut impl Write,
    expected_size: u64,
) -> Result<String> {
    let mut hasher = Sha256::new();
    let mut total = 0_u64;
    let mut buffer = [0_u8; 65536];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        total = total
            .checked_add(count as u64)
            .context("byte count overflow")?;
        ensure!(
            total <= expected_size,
            "payload exceeds declared byte length"
        );
        writer.write_all(&buffer[..count])?;
        hasher.update(&buffer[..count]);
    }
    ensure!(
        total == expected_size,
        "payload byte length mismatch: expected {expected_size}, observed {total}"
    );
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn check_existing(root: &Path, relative: &str, expected_hash: &str, size: u64) -> Result<bool> {
    safe_relative(relative)?;
    let mut path = root.to_owned();
    let parts: Vec<_> = Path::new(relative).components().collect();
    for (index, component) in parts.iter().enumerate() {
        path.push(component.as_os_str());
        let metadata = match fs::symlink_metadata(&path) {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
            Err(error) => return Err(error.into()),
        };
        ensure!(
            !metadata.file_type().is_symlink(),
            "symlink in output path {}",
            path.display()
        );
        if index + 1 == parts.len() {
            ensure!(metadata.is_file(), "output path is not a regular file");
            let mut file = File::open(&path)?;
            ensure!(
                copy_hashed(&mut file, &mut io::sink(), size)? == expected_hash,
                "existing payload conflicts at {}",
                path.display()
            );
            return Ok(true);
        }
        ensure!(metadata.is_dir(), "output parent is not a directory");
    }
    bail!("empty output path")
}

struct ExpansionBound<R> {
    inner: R,
    remaining: u64,
}

impl<R: Read> Read for ExpansionBound<R> {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        if self.remaining == 0 {
            let mut probe = [0_u8; 1];
            return if self.inner.read(&mut probe)? == 0 {
                Ok(0)
            } else {
                Err(io::Error::other(
                    "archive expansion exceeds declared tar bound",
                ))
            };
        }
        let capacity = output
            .len()
            .min(usize::try_from(self.remaining).unwrap_or(usize::MAX));
        let count = self.inner.read(&mut output[..capacity])?;
        self.remaining -= count as u64;
        Ok(count)
    }
}

fn stage_objects(
    archive_file: File,
    objects: &BTreeMap<String, ArchiveObject>,
    staging: &Path,
) -> Result<()> {
    let expansion_bound = objects.values().try_fold(10240_u64, |bound, object| {
        object
            .byte_length
            .checked_add(511)
            .map(|rounded| rounded / 512 * 512)
            .and_then(|padded| padded.checked_add(512))
            .and_then(|entry_size| bound.checked_add(entry_size))
            .context("declared archive expansion overflow")
    })?;
    let mut decoder = zstd::stream::read::Decoder::new(archive_file)?;
    decoder.window_log_max(27)?;
    let mut archive = tar::Archive::new(ExpansionBound {
        inner: decoder,
        remaining: expansion_bound,
    });
    archive.set_ignore_zeros(true);
    let mut seen = BTreeSet::new();
    for entry in archive.entries()?.raw(true) {
        let mut entry = entry?;
        ensure!(
            entry.header().entry_type().is_file(),
            "archive contains a non-regular entry"
        );
        ensure!(
            entry.header().link_name_bytes().is_none(),
            "archive contains a link target"
        );
        let member = std::str::from_utf8(&entry.path_bytes())?.to_owned();
        safe_relative(&member)?;
        let object = objects
            .values()
            .find(|object| object.archive_member == member)
            .context("archive contains undeclared member")?;
        ensure!(seen.insert(member), "duplicate archive member");
        ensure!(
            entry.size() == object.byte_length,
            "archive header size differs from declared object"
        );
        let mut output = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(staging.join(&object.sha256))?;
        ensure!(
            copy_hashed(&mut entry, &mut output, object.byte_length)? == object.sha256,
            "archive object digest mismatch"
        );
        output.sync_all()?;
    }
    ensure!(
        seen.len() == objects.len(),
        "archive is missing declared objects"
    );
    io::copy(&mut archive.into_inner(), &mut io::sink())?;
    Ok(())
}

fn create_parents(root: &Path, relative: &str) -> Result<()> {
    let mut path = root.to_owned();
    for component in Path::new(relative)
        .parent()
        .context("output parent missing")?
        .components()
    {
        path.push(component.as_os_str());
        match fs::create_dir(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
        let metadata = fs::symlink_metadata(&path)?;
        ensure!(
            metadata.is_dir() && !metadata.file_type().is_symlink(),
            "output parent is a symlink or non-directory"
        );
    }
    Ok(())
}

fn hydrate(args: &Args) -> Result<Report> {
    let root_metadata = fs::symlink_metadata(&args.repo_root)?;
    ensure!(
        root_metadata.is_dir() && !root_metadata.file_type().is_symlink(),
        "repository root must be a directory, not a symlink"
    );
    let root = args.repo_root.canonicalize()?;
    ensure!(
        args.manifest.canonicalize()? == root.join(MANIFEST_PATH),
        "manifest must be the fixed repository retention manifest"
    );
    let manifest = RetainedArchive::load(&root)?;
    let objects = &manifest.objects;
    ensure!(
        !objects.is_empty() && manifest.members().next().is_some(),
        "empty retention inventory"
    );
    let mut archive_file = File::open(&args.archive)?;
    ensure!(
        archive_file.metadata()?.is_file(),
        "archive must be a regular file"
    );
    ensure!(
        copy_hashed(
            &mut archive_file,
            &mut io::sink(),
            manifest.archive.byte_length
        )? == manifest.archive.sha256,
        "archive SHA256 mismatch"
    );
    archive_file.seek(SeekFrom::Start(0))?;
    for payload in manifest.members() {
        check_existing(&root, &payload.path, &payload.sha256, payload.byte_length)?;
    }
    let staging = tempfile::Builder::new()
        .prefix(".scientific-payload-hydration-")
        .tempdir_in(&root)?;
    stage_objects(archive_file, objects, staging.path())?;
    let mut report = Report {
        verified_archive_sha256: manifest.archive.sha256.clone(),
        verified_objects: objects.len(),
        installation_scope: "Each new file is atomically persisted without replacement; a later failure preserves earlier installed files and created directories. The complete operation is not globally atomic.",
        ..Report::default()
    };
    let install = (|| -> Result<()> {
        for payload in manifest.members() {
            if check_existing(&root, &payload.path, &payload.sha256, payload.byte_length)? {
                report.reused_files += 1;
                continue;
            }
            create_parents(&root, &payload.path)?;
            let mut output = tempfile::NamedTempFile::new_in(staging.path())?;
            let mut source = File::open(staging.path().join(&payload.sha256))?;
            ensure!(
                copy_hashed(&mut source, output.as_file_mut(), payload.byte_length)?
                    == payload.sha256,
                "staged object changed before installation"
            );
            output.as_file().sync_all()?;
            check_existing(&root, &payload.path, &payload.sha256, payload.byte_length)?;
            output
                .persist_noclobber(root.join(&payload.path))
                .map_err(|error| {
                    anyhow::anyhow!("atomic no-clobber installation failed: {}", error.error)
                })?;
            report.installed_files += 1;
        }
        Ok(())
    })();
    install.with_context(|| format!("hydration stopped after {} installed and {} reused files; earlier installations and created directories remain", report.installed_files, report.reused_files))?;
    Ok(report)
}

fn main() -> Result<()> {
    let report = hydrate(&Args::parse())?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};
    use std::process::Command;

    fn member(bytes: &[u8]) -> String {
        let identity = hash(bytes);
        format!("{}/{}/{identity}", &identity[..2], &identity[2..4])
    }

    fn packed(entries: &[(String, Vec<u8>, u8)]) -> Result<Vec<u8>> {
        let mut builder = tar::Builder::new(Vec::new());
        for (path, bytes, kind) in entries {
            let mut header = tar::Header::new_ustar();
            header.set_mode(0o644);
            header.set_uid(0);
            header.set_gid(0);
            header.set_mtime(0);
            header.set_size(bytes.len() as u64);
            header.set_entry_type(tar::EntryType::new(*kind));
            ensure!(path.len() < 100, "fixture path too long");
            header.as_mut_bytes()[..100].fill(0);
            header.as_mut_bytes()[..path.len()].copy_from_slice(path.as_bytes());
            if matches!(kind, b'1' | b'2') {
                header.set_link_name("outside")?;
            }
            header.set_cksum();
            builder.append(&header, bytes.as_slice())?;
        }
        builder.finish()?;
        Ok(zstd::stream::encode_all(
            builder.into_inner()?.as_slice(),
            1,
        )?)
    }

    struct Fixture {
        _directory: tempfile::TempDir,
        args: Args,
        manifest: Value,
    }

    impl Fixture {
        fn new(objects: &[&[u8]], entries: &[(String, Vec<u8>, u8)]) -> Result<Self> {
            let directory = tempfile::tempdir()?;
            let root = directory.path().join("repo");
            fs::create_dir(&root)?;
            ensure!(
                Command::new("git")
                    .args(["init", "--quiet"])
                    .arg(&root)
                    .status()?
                    .success(),
                "fixture Git initialization"
            );
            fs::create_dir_all(root.join("data/retention"))?;
            let archive_bytes = packed(entries)?;
            let archive_path = directory.path().join("objects.tar.zst");
            fs::write(&archive_path, &archive_bytes)?;
            let manifest = json!({
                "schema_version": 1,
                "archive": {"url": "https://example.org/scientific-payloads.tar.zst", "sha256": hash(&archive_bytes), "byte_length": archive_bytes.len(), "format": "tar-zstd"},
                "objects": objects.iter().map(|bytes| json!({"sha256":hash(bytes), "byte_length": bytes.len(), "archive_member": member(bytes)})).collect::<Vec<_>>(),
                "files": objects.iter().enumerate().map(|(index,bytes)| json!({"path":format!("payloads/{index}.dat"), "sha256":hash(bytes), "byte_length":bytes.len(), "archive_member":member(bytes)})).collect::<Vec<_>>()
            });
            let fixture = Self {
                args: Args {
                    manifest: root.join(MANIFEST_PATH),
                    archive: archive_path,
                    repo_root: root,
                },
                _directory: directory,
                manifest,
            };
            fixture.stage_manifest()?;
            Ok(fixture)
        }

        fn stage_manifest(&self) -> Result<()> {
            fs::write(&self.args.manifest, serde_json::to_vec(&self.manifest)?)?;
            ensure!(
                Command::new("git")
                    .arg("-C")
                    .arg(&self.args.repo_root)
                    .args(["add", "--", MANIFEST_PATH])
                    .status()?
                    .success(),
                "fixture manifest index binding"
            );
            Ok(())
        }

        fn assert_clean_failure(&self) -> Result<()> {
            assert!(hydrate(&self.args).is_err());
            assert!(!self.args.repo_root.join("payloads").exists());
            assert!(fs::read_dir(&self.args.repo_root)?.all(|entry| {
                !entry
                    .unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".scientific-payload-hydration-")
            }));
            Ok(())
        }
    }

    fn regular(bytes: &[u8]) -> (String, Vec<u8>, u8) {
        (member(bytes), bytes.to_vec(), b'0')
    }

    #[test]
    fn verifies_historical_superset_installs_and_reuses_exact_current_bytes() -> Result<()> {
        let mut fixture = Fixture::new(
            &[b"current", b"historical"],
            &[regular(b"historical"), regular(b"current")],
        )?;
        fixture.manifest["files"]
            .as_array_mut()
            .unwrap()
            .truncate(1);
        fixture.stage_manifest()?;
        let report = hydrate(&fixture.args)?;
        assert_eq!(report.verified_objects, 2);
        assert_eq!(report.installed_files, 1);
        assert!(!report.globally_atomic);
        assert_eq!(
            fs::read(fixture.args.repo_root.join("payloads/0.dat"))?,
            b"current"
        );
        let report = hydrate(&fixture.args)?;
        assert_eq!((report.installed_files, report.reused_files), (0, 1));
        Ok(())
    }

    #[test]
    fn rejects_archive_hash_size_and_object_mismatch_before_installation() -> Result<()> {
        for mutation in 0..3 {
            let entries = if mutation == 2 {
                vec![(member(b"good"), b"evil".to_vec(), b'0')]
            } else {
                vec![regular(b"good")]
            };
            let mut fixture = Fixture::new(&[b"good"], &entries)?;
            match mutation {
                0 => fixture.manifest["archive"]["sha256"] = json!("0".repeat(64)),
                1 => fixture.manifest["archive"]["byte_length"] = json!(1),
                _ => {}
            }
            fixture.stage_manifest()?;
            fixture.assert_clean_failure()?;
        }
        Ok(())
    }

    #[test]
    fn rejects_missing_duplicate_undeclared_and_link_entries() -> Result<()> {
        let cases = [
            vec![],
            vec![regular(b"good"), regular(b"good")],
            vec![regular(b"undeclared")],
            vec![(member(b"good"), Vec::new(), b'1')],
            vec![(member(b"good"), Vec::new(), b'2')],
            vec![("directory/".to_owned(), Vec::new(), b'5')],
        ];
        for entries in cases {
            Fixture::new(&[b"good"], &entries)?.assert_clean_failure()?;
        }
        Ok(())
    }

    #[test]
    fn rejects_archive_traversal_and_metadata_entries() -> Result<()> {
        for path in [
            "../outside",
            "/absolute",
            ".git/config",
            "aa/../outside",
            "C:\\outside",
            "payload:stream",
        ] {
            Fixture::new(&[b"good"], &[(path.to_owned(), b"good".to_vec(), b'0')])?
                .assert_clean_failure()?;
        }
        Fixture::new(
            &[b"good"],
            &[("pax".to_owned(), b"metadata".to_vec(), b'x')],
        )?
        .assert_clean_failure()?;
        Ok(())
    }

    #[test]
    fn rejects_unsafe_destination_paths_and_manifest_conflicts() -> Result<()> {
        for path in [
            "../outside",
            "/absolute",
            ".git/config",
            "data/.GiT/index",
            "C:/outside",
            "payload:stream",
            "a\\b",
            "a./b",
            "a /b",
        ] {
            let mut fixture = Fixture::new(&[b"good"], &[regular(b"good")])?;
            fixture.manifest["files"][0]["path"] = json!(path);
            fixture.stage_manifest()?;
            fixture.assert_clean_failure()?;
        }
        for mutation in 0..4 {
            let mut fixture = Fixture::new(&[b"good"], &[regular(b"good")])?;
            match mutation {
                0 => {
                    fixture.manifest.as_object_mut().unwrap().remove("objects");
                }
                1 => {
                    let duplicate = fixture.manifest["objects"][0].clone();
                    fixture.manifest["objects"]
                        .as_array_mut()
                        .unwrap()
                        .push(duplicate);
                }
                2 => fixture.manifest["files"][0]["byte_length"] = json!(99),
                _ => {
                    let duplicate = fixture.manifest["files"][0].clone();
                    fixture.manifest["files"]
                        .as_array_mut()
                        .unwrap()
                        .push(duplicate);
                }
            }
            fixture.stage_manifest()?;
            fixture.assert_clean_failure()?;
        }
        Ok(())
    }

    #[test]
    fn preflights_every_conflict_before_installing_any_file() -> Result<()> {
        let fixture = Fixture::new(
            &[b"first", b"second"],
            &[regular(b"first"), regular(b"second")],
        )?;
        fs::create_dir(fixture.args.repo_root.join("payloads"))?;
        fs::write(fixture.args.repo_root.join("payloads/1.dat"), b"conflict")?;
        assert!(hydrate(&fixture.args).is_err());
        assert!(!fixture.args.repo_root.join("payloads/0.dat").exists());
        assert_eq!(
            fs::read(fixture.args.repo_root.join("payloads/1.dat"))?,
            b"conflict"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_destinations_and_parents() -> Result<()> {
        for leaf in [false, true] {
            let fixture = Fixture::new(&[b"good"], &[regular(b"good")])?;
            let outside = fixture._directory.path().join("outside");
            fs::create_dir(&outside)?;
            let path = if leaf {
                fs::create_dir(fixture.args.repo_root.join("payloads"))?;
                fixture.args.repo_root.join("payloads/0.dat")
            } else {
                fixture.args.repo_root.join("payloads")
            };
            std::os::unix::fs::symlink(&outside, &path)?;
            assert!(hydrate(&fixture.args).is_err());
            assert_eq!(fs::read_dir(outside)?.count(), 0);
        }
        Ok(())
    }

    #[test]
    fn expansion_bound_rejects_oversized_zero_padding() -> Result<()> {
        let fixture = Fixture::new(&[b"good"], &[regular(b"good")])?;
        let mut objects = BTreeMap::new();
        objects.insert(
            hash(b"good"),
            ArchiveObject {
                sha256: hash(b"good"),
                byte_length: 4,
                archive_member: member(b"good"),
            },
        );
        let padding = zstd::stream::encode_all(vec![0_u8; 32768].as_slice(), 1)?;
        let path = fixture._directory.path().join("padding.zst");
        fs::write(&path, padding)?;
        let staging = tempfile::tempdir()?;
        assert!(stage_objects(File::open(path)?, &objects, staging.path()).is_err());
        Ok(())
    }
}
