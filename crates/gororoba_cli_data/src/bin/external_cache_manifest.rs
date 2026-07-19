//! Hash-addressed manifest for external data caches excluded from version
//! control.
//!
//! A cache covered by this tool lives outside git and Git LFS; the tracked
//! manifest pins expected cache content as `sha256  size_bytes  path` lines,
//! the source contract's retrieval_method attempts reconstruction, and
//! `verify` detects upstream drift by recomputing every hash and failing on
//! missing, drifted, or extra files. The manifest provides content identity
//! and integrity; it cannot force an upstream archive to retain historical
//! bytes.
//!
//! Payload validation excludes fetch-attempt markers from both generation
//! and verification: an HTTP-era cache accumulates JSON ("no data for time
//! range") and HTML error bodies alongside real data files, and those
//! markers stay on disk so re-fetch skips the day, but they never enter the
//! manifest and never count as extra files.

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::Read,
    path::{Path, PathBuf},
};
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(
    name = "external-cache-manifest",
    about = "Generate or verify a sha256 manifest for an external data cache kept out of version control"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Walk the cache root and write the sorted manifest.
    Generate {
        /// Cache directory the manifest describes.
        #[arg(long)]
        root: PathBuf,
        /// Manifest file to write (tracked ASCII text).
        #[arg(long)]
        out: PathBuf,
        /// Only include files whose name matches this prefix (e.g. tha_fgm_).
        #[arg(long)]
        prefix: Option<String>,
    },
    /// Recompute hashes under the cache root and compare against the manifest.
    Verify {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        manifest: PathBuf,
        /// Report files present in the cache but absent from the manifest.
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        fail_on_extra: bool,
        /// Only consider cache files whose name matches this prefix.
        #[arg(long)]
        prefix: Option<String>,
    },
}

const MANIFEST_SEPARATOR: &str = "  ";

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

/// A data payload starts with neither a JSON brace nor an HTML/XML angle
/// bracket; empty files carry no data. Error bodies cached under a data
/// filename (HAPI status JSON, HTML error pages) are fetch-attempt markers,
/// excluded from manifest scope on both sides of the comparison.
fn is_data_payload(path: &Path) -> Result<bool> {
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut first = [0u8; 1];
    let read = file.read(&mut first)?;
    Ok(read == 1 && first[0] != b'{' && first[0] != b'<')
}

fn validate_manifest_path(rel: &str) -> Result<()> {
    if rel.is_empty() {
        bail!("empty relative path");
    }
    if rel.chars().any(|c| c == '\r' || c == '\n') {
        bail!("path contains a line break: {rel:?}");
    }
    if !rel.is_ascii() {
        bail!("path is not ASCII: {rel:?}");
    }
    Ok(())
}

fn validate_digest(digest: &str) -> Result<()> {
    if digest.len() != 64 || !digest.bytes().all(|b| b.is_ascii_hexdigit()) {
        bail!("malformed sha256 digest: {digest:?}");
    }
    Ok(())
}

/// Relative path -> (sha256, size). BTreeMap keeps manifest order
/// deterministic. `exclude` drops one absolute path from the scan so a
/// manifest written beneath its own root never pins itself.
fn scan_cache(
    root: &Path,
    prefix: Option<&str>,
    exclude: Option<&Path>,
) -> Result<BTreeMap<String, (String, u64)>> {
    let mut entries = BTreeMap::new();
    for item in WalkDir::new(root).sort_by_file_name() {
        let item = item?;
        if !item.file_type().is_file() {
            continue;
        }
        if let Some(excluded) = exclude
            && item.path() == excluded
        {
            continue;
        }
        if let Some(pfx) = prefix {
            let name = item.file_name().to_string_lossy();
            if !name.starts_with(pfx) {
                continue;
            }
        }
        if !is_data_payload(item.path())? {
            continue;
        }
        let rel = item
            .path()
            .strip_prefix(root)
            .expect("walkdir yields paths under root")
            .to_string_lossy()
            .replace('\\', "/");
        validate_manifest_path(&rel).context("unrepresentable path in cache")?;
        let size = item.metadata()?.len();
        let digest = sha256_file(item.path())?;
        entries.insert(rel, (digest, size));
    }
    Ok(entries)
}

fn render_manifest(root: &Path, entries: &BTreeMap<String, (String, u64)>) -> String {
    let mut text = String::new();
    text.push_str("# external-cache-manifest v1\n");
    text.push_str(&format!("# root: {}\n", root.display()));
    text.push_str("# columns: sha256  size_bytes  relative_path (two-space separated)\n");
    for (rel, (digest, size)) in entries {
        text.push_str(&format!(
            "{digest}{MANIFEST_SEPARATOR}{size}{MANIFEST_SEPARATOR}{rel}\n"
        ));
    }
    text
}

fn parse_manifest_text(text: &str) -> Result<BTreeMap<String, (String, u64)>> {
    let mut entries = BTreeMap::new();
    for (lineno, line) in text.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(3, MANIFEST_SEPARATOR);
        let (Some(digest), Some(size), Some(rel)) = (parts.next(), parts.next(), parts.next())
        else {
            bail!("malformed manifest line {}: {line:?}", lineno + 1);
        };
        validate_digest(digest).with_context(|| format!("manifest line {}", lineno + 1))?;
        let size: u64 = size
            .parse()
            .with_context(|| format!("bad size on manifest line {}", lineno + 1))?;
        validate_manifest_path(rel).with_context(|| format!("manifest line {}", lineno + 1))?;
        if entries
            .insert(rel.to_string(), (digest.to_string(), size))
            .is_some()
        {
            bail!("duplicate manifest path: {rel:?}");
        }
    }
    Ok(entries)
}

struct VerifyOutcome {
    missing: Vec<String>,
    drifted: Vec<String>,
    extra: Vec<String>,
}

fn compare(
    expected: &BTreeMap<String, (String, u64)>,
    actual: &BTreeMap<String, (String, u64)>,
) -> VerifyOutcome {
    let mut outcome = VerifyOutcome {
        missing: Vec::new(),
        drifted: Vec::new(),
        extra: Vec::new(),
    };
    for (rel, (digest, size)) in expected {
        match actual.get(rel) {
            None => outcome.missing.push(rel.clone()),
            Some((actual_digest, actual_size)) => {
                if actual_digest != digest || actual_size != size {
                    outcome.drifted.push(rel.clone());
                }
            }
        }
    }
    for rel in actual.keys() {
        if !expected.contains_key(rel) {
            outcome.extra.push(rel.clone());
        }
    }
    outcome
}

fn verification_failed(outcome: &VerifyOutcome, fail_on_extra: bool) -> bool {
    !outcome.missing.is_empty()
        || !outcome.drifted.is_empty()
        || (fail_on_extra && !outcome.extra.is_empty())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Generate { root, out, prefix } => {
            let exclude = out.canonicalize().ok();
            let entries = scan_cache(&root, prefix.as_deref(), exclude.as_deref())?;
            if let Some(parent) = out.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&out, render_manifest(&root, &entries))
                .with_context(|| format!("write {}", out.display()))?;
            let total_bytes: u64 = entries.values().map(|(_, size)| size).sum();
            println!(
                "wrote {} entries ({} bytes cached) to {}",
                entries.len(),
                total_bytes,
                out.display()
            );
        }
        Command::Verify {
            root,
            manifest,
            fail_on_extra,
            prefix,
        } => {
            let text = fs::read_to_string(&manifest)
                .with_context(|| format!("read {}", manifest.display()))?;
            let expected = parse_manifest_text(&text)?;
            let exclude = manifest.canonicalize().ok();
            let actual = scan_cache(&root, prefix.as_deref(), exclude.as_deref())?;
            let outcome = compare(&expected, &actual);
            for rel in &outcome.missing {
                println!("MISSING  {rel}");
            }
            for rel in &outcome.drifted {
                println!("DRIFT    {rel}");
            }
            for rel in &outcome.extra {
                println!("EXTRA    {rel}");
            }
            println!(
                "verified {} manifest entries: {} missing, {} drifted, {} extra",
                expected.len(),
                outcome.missing.len(),
                outcome.drifted.len(),
                outcome.extra.len()
            );
            if verification_failed(&outcome, fail_on_extra) {
                bail!("cache does not match manifest");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "external_cache_manifest_test_{tag}_{}",
            std::process::id()
        ));
        if dir.exists() {
            fs::remove_dir_all(&dir).expect("clean stale test dir");
        }
        fs::create_dir_all(&dir).expect("create test dir");
        dir
    }

    fn roundtrip(entries: &BTreeMap<String, (String, u64)>) -> BTreeMap<String, (String, u64)> {
        parse_manifest_text(&render_manifest(Path::new("root"), entries)).expect("roundtrip")
    }

    #[test]
    fn paths_with_spaces_roundtrip() {
        let mut entries = BTreeMap::new();
        entries.insert("my file.csv".to_string(), ("0".repeat(64), 3));
        entries.insert("double  space.csv".to_string(), ("a".repeat(64), 9));
        assert_eq!(roundtrip(&entries), entries);
    }

    #[test]
    fn empty_manifest_roundtrips() {
        assert!(roundtrip(&BTreeMap::new()).is_empty());
    }

    #[test]
    fn duplicate_manifest_paths_rejected() {
        let digest = "0".repeat(64);
        let text = format!("{digest}  1  same.csv\n{digest}  2  same.csv\n");
        assert!(parse_manifest_text(&text).is_err());
    }

    #[test]
    fn malformed_digest_rejected() {
        assert!(parse_manifest_text("nothex  1  f.csv\n").is_err());
    }

    #[test]
    fn generate_verify_detects_missing_drifted_extra() {
        let dir = temp_dir("mde");
        fs::write(dir.join("keep.csv"), "1,2,3\n").unwrap();
        fs::write(dir.join("drift.csv"), "4,5,6\n").unwrap();
        fs::write(dir.join("gone.csv"), "7,8,9\n").unwrap();
        let expected = scan_cache(&dir, None, None).unwrap();

        fs::write(dir.join("drift.csv"), "changed\n").unwrap();
        fs::remove_file(dir.join("gone.csv")).unwrap();
        fs::write(dir.join("extra.csv"), "9,9,9\n").unwrap();
        let actual = scan_cache(&dir, None, None).unwrap();

        let outcome = compare(&expected, &actual);
        assert_eq!(outcome.missing, vec!["gone.csv"]);
        assert_eq!(outcome.drifted, vec!["drift.csv"]);
        assert_eq!(outcome.extra, vec!["extra.csv"]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn json_and_html_payloads_excluded_both_sides() {
        let dir = temp_dir("payload");
        fs::write(dir.join("data.csv"), "1,2\n").unwrap();
        fs::write(dir.join("hapi_error.csv"), "{\"status\": 1201}\n").unwrap();
        fs::write(dir.join("html_error.csv"), "<html></html>\n").unwrap();
        fs::write(dir.join("empty.csv"), "").unwrap();
        let entries = scan_cache(&dir, None, None).unwrap();
        assert_eq!(entries.keys().collect::<Vec<_>>(), vec!["data.csv"]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn manifest_beneath_root_excluded_from_scan() {
        let dir = temp_dir("selfref");
        fs::write(dir.join("data.csv"), "1,2\n").unwrap();
        let manifest = dir.join("manifest.txt");
        let entries = scan_cache(&dir, None, None).unwrap();
        fs::write(&manifest, render_manifest(&dir, &entries)).unwrap();
        let rescanned = scan_cache(&dir, None, Some(&manifest.canonicalize().unwrap())).unwrap();
        assert_eq!(rescanned, entries);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn prefix_filter_limits_scope() {
        let dir = temp_dir("prefix");
        fs::write(dir.join("tha_fgm_x.csv"), "1\n").unwrap();
        fs::write(dir.join("thb_fgm_x.csv"), "2\n").unwrap();
        let entries = scan_cache(&dir, Some("tha_fgm_"), None).unwrap();
        assert_eq!(entries.keys().collect::<Vec<_>>(), vec!["tha_fgm_x.csv"]);
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn extra_only_failure_gated_by_flag() {
        let outcome = VerifyOutcome {
            missing: Vec::new(),
            drifted: Vec::new(),
            extra: vec!["extra.csv".to_string()],
        };
        assert!(verification_failed(&outcome, true));
        assert!(!verification_failed(&outcome, false));
    }

    #[test]
    fn path_with_line_break_rejected() {
        assert!(validate_manifest_path("bad\npath.csv").is_err());
        assert!(validate_manifest_path("").is_err());
    }
}
