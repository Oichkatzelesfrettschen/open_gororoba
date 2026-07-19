//! Hash-addressed manifest for gitignored external data caches.
//!
//! An external cache under `data/external/` lives outside git and Git LFS;
//! its reproducibility contract is a tracked ASCII manifest listing every
//! cached file as `sha256  size_bytes  relative_path`, sorted by path.
//! `generate` walks the cache root and writes that manifest; `verify`
//! recomputes the hashes and exits nonzero on any missing, extra, resized,
//! or content-drifted file, so the manifest pins the cache exactly the way
//! an LFS pointer would without consuming LFS storage. Retrieval of the
//! files themselves stays with the source contract's retrieval_method
//! (for THEMIS FGM, the CDAWeb HAPI fetch in ThemisFgmProvider).

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
    about = "Generate or verify a sha256 manifest for a gitignored external data cache"
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

/// Relative path -> (sha256, size). BTreeMap keeps manifest order deterministic.
fn scan_cache(root: &Path, prefix: Option<&str>) -> Result<BTreeMap<String, (String, u64)>> {
    let mut entries = BTreeMap::new();
    for item in WalkDir::new(root).sort_by_file_name() {
        let item = item?;
        if !item.file_type().is_file() {
            continue;
        }
        if let Some(pfx) = prefix {
            let name = item.file_name().to_string_lossy();
            if !name.starts_with(pfx) {
                continue;
            }
        }
        let rel = item
            .path()
            .strip_prefix(root)
            .expect("walkdir yields paths under root")
            .to_string_lossy()
            .replace('\\', "/");
        let size = item.metadata()?.len();
        let digest = sha256_file(item.path())?;
        entries.insert(rel, (digest, size));
    }
    Ok(entries)
}

fn write_manifest(out: &Path, root: &Path, entries: &BTreeMap<String, (String, u64)>) -> Result<()> {
    if let Some(parent) = out.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut text = String::new();
    text.push_str("# external-cache-manifest v1\n");
    text.push_str(&format!("# root: {}\n", root.display()));
    text.push_str("# columns: sha256  size_bytes  relative_path\n");
    for (rel, (digest, size)) in entries {
        text.push_str(&format!("{digest}  {size}  {rel}\n"));
    }
    fs::write(out, text).with_context(|| format!("write {}", out.display()))?;
    Ok(())
}

fn parse_manifest(path: &Path) -> Result<BTreeMap<String, (String, u64)>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut entries = BTreeMap::new();
    for (lineno, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split_whitespace();
        let (Some(digest), Some(size), Some(rel)) = (parts.next(), parts.next(), parts.next())
        else {
            bail!("malformed manifest line {}: {line}", lineno + 1);
        };
        let size: u64 = size
            .parse()
            .with_context(|| format!("bad size on manifest line {}", lineno + 1))?;
        entries.insert(rel.to_string(), (digest.to_string(), size));
    }
    Ok(entries)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::Generate { root, out, prefix } => {
            let entries = scan_cache(&root, prefix.as_deref())?;
            write_manifest(&out, &root, &entries)?;
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
            let expected = parse_manifest(&manifest)?;
            let actual = scan_cache(&root, prefix.as_deref())?;
            let mut missing = 0usize;
            let mut drifted = 0usize;
            for (rel, (digest, size)) in &expected {
                match actual.get(rel) {
                    None => {
                        println!("MISSING  {rel}");
                        missing += 1;
                    }
                    Some((actual_digest, actual_size)) => {
                        if actual_digest != digest || actual_size != size {
                            println!("DRIFT    {rel}");
                            drifted += 1;
                        }
                    }
                }
            }
            let mut extra = 0usize;
            for rel in actual.keys() {
                if !expected.contains_key(rel) {
                    println!("EXTRA    {rel}");
                    extra += 1;
                }
            }
            println!(
                "verified {} manifest entries: {} missing, {} drifted, {} extra",
                expected.len(),
                missing,
                drifted,
                extra
            );
            if missing > 0 || drifted > 0 || (fail_on_extra && extra > 0) {
                bail!("cache does not match manifest");
            }
        }
    }
    Ok(())
}
