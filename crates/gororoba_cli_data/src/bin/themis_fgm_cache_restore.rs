//! Fetch-only reconstruction of a THEMIS FGM daily-file cache from its
//! tracked manifest.
//!
//! The manifest is the authority on which days belong to the cache; this
//! tool iterates its entries independently of crossing catalogs or analysis
//! state, fetches each absent day through ThemisFgmProvider (CDAWeb HAPI
//! daily CSV), leaves existing files alone unless `--refresh true`, and
//! preserves HAPI no-data markers by never deleting a file it did not
//! fetch. After retrieval it recomputes every manifest hash and fails on
//! any mismatch, so a zero exit means the cache byte-matches the pinned
//! content and a nonzero exit surfaces upstream drift or retrieval gaps.

use anyhow::{Context, Result, bail};
use clap::Parser;
use data_core::{
    catalogs::themis_fetch::ThemisFgmProvider,
    fetcher::{DatasetProvider, FetchConfig},
};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(
    name = "themis-fgm-cache-restore",
    about = "Reconstruct a THEMIS FGM daily-file cache from its tracked sha256 manifest"
)]
struct Cli {
    /// Probe letter: a, b, c, d, or e.
    #[arg(long)]
    probe: String,
    /// Tracked manifest listing sha256, size, and relative path per day file.
    #[arg(long)]
    manifest: PathBuf,
    /// Root directory holding the themis/ cache subdirectory.
    #[arg(long, default_value = "data/external")]
    data_dir: PathBuf,
    /// Re-fetch manifest entries even when a file already exists.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    refresh: bool,
}

struct ManifestEntry {
    rel: String,
    digest: String,
    size: u64,
    year: u16,
    doy: u16,
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

/// Accepts `th{probe}_fgm_{year}_{doy}.csv`; the year/doy pair drives the
/// per-day provider fetch.
fn parse_day_filename(rel: &str, probe: &str) -> Option<(u16, u16)> {
    let prefix = format!("th{probe}_fgm_");
    let stem = rel.strip_prefix(&prefix)?.strip_suffix(".csv")?;
    let (year, doy) = stem.split_once('_')?;
    Some((year.parse().ok()?, doy.parse().ok()?))
}

fn parse_manifest(path: &Path, probe: &str) -> Result<Vec<ManifestEntry>> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut entries = Vec::new();
    for (lineno, line) in text.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.splitn(3, "  ");
        let (Some(digest), Some(size), Some(rel)) = (parts.next(), parts.next(), parts.next())
        else {
            bail!("malformed manifest line {}: {line:?}", lineno + 1);
        };
        let size: u64 = size
            .parse()
            .with_context(|| format!("bad size on manifest line {}", lineno + 1))?;
        let Some((year, doy)) = parse_day_filename(rel, probe) else {
            bail!(
                "manifest line {} is not a th{probe}_fgm daily file: {rel:?}",
                lineno + 1
            );
        };
        entries.push(ManifestEntry {
            rel: rel.to_string(),
            digest: digest.to_string(),
            size,
            year,
            doy,
        });
    }
    Ok(entries)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let probe = cli.probe.to_lowercase();
    if !matches!(probe.as_str(), "a" | "b" | "c" | "d" | "e") {
        bail!("probe must be one of a, b, c, d, e");
    }
    let entries = parse_manifest(&cli.manifest, &probe)?;
    let cache_dir = cli.data_dir.join("themis");
    fs::create_dir_all(&cache_dir)?;

    let config = FetchConfig {
        output_dir: cli.data_dir.clone(),
        skip_existing: !cli.refresh,
        verify_checksums: false,
    };

    let mut fetched = 0usize;
    let mut skipped = 0usize;
    let mut fetch_errors = 0usize;
    for entry in &entries {
        let target = cache_dir.join(&entry.rel);
        if target.exists() && !cli.refresh {
            skipped += 1;
            continue;
        }
        let provider = ThemisFgmProvider {
            probe: format!("TH{}", probe.to_uppercase()),
            year: entry.year,
            doy_start: entry.doy,
            doy_end: entry.doy,
        };
        match provider.fetch(&config) {
            Ok(_) => fetched += 1,
            Err(err) => {
                eprintln!("FETCH-ERROR  {}: {err}", entry.rel);
                fetch_errors += 1;
            }
        }
    }
    println!(
        "restore pass: {} manifest entries, {} fetched, {} already present, {} fetch errors",
        entries.len(),
        fetched,
        skipped,
        fetch_errors
    );

    let mut missing = 0usize;
    let mut drifted = 0usize;
    for entry in &entries {
        let target = cache_dir.join(&entry.rel);
        if !target.exists() {
            println!("MISSING  {}", entry.rel);
            missing += 1;
            continue;
        }
        let size = fs::metadata(&target)?.len();
        let digest = sha256_file(&target)?;
        if size != entry.size || digest != entry.digest {
            println!("DRIFT    {}", entry.rel);
            drifted += 1;
        }
    }
    println!(
        "manifest verification: {} entries, {} missing, {} drifted",
        entries.len(),
        missing,
        drifted
    );
    if missing > 0 || drifted > 0 || fetch_errors > 0 {
        bail!("cache does not match manifest after restore");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn day_filename_parses_year_and_doy() {
        assert_eq!(parse_day_filename("tha_fgm_2008_153.csv", "a"), Some((2008, 153)));
        assert_eq!(parse_day_filename("thb_fgm_2008_153.csv", "a"), None);
        assert_eq!(parse_day_filename("tha_fgm_2008_153.txt", "a"), None);
        assert_eq!(parse_day_filename("tha_fgm_bad.csv", "a"), None);
    }

    #[test]
    fn manifest_paths_with_directories_rejected_as_day_files() {
        assert_eq!(parse_day_filename("sub/tha_fgm_2008_153.csv", "a"), None);
    }
}
