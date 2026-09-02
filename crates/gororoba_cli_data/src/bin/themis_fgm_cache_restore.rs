//! Fetch-only reconstruction of a THEMIS daily-file cache (FGM field or
//! ESA ion moments) from its tracked manifest.
//!
//! The manifest is the authority on which days belong to the cache; this
//! tool iterates its entries independently of crossing catalogs or analysis
//! state, fetches each absent day through ThemisFgmProvider or
//! ThemisEsaProvider (CDAWeb HAPI daily CSV), leaves existing files alone
//! unless `--refresh true`, and keeps provider retrieval failures as errors
//! even when a stale manifest file already exists. After retrieval it
//! recomputes every manifest hash and fails on any mismatch, so a zero exit
//! means the cache byte-matches the pinned content and every requested
//! retrieval completed.

use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use data_core::{
    catalogs::{themis_esa_fetch::ThemisEsaProvider, themis_fetch::ThemisFgmProvider},
    fetcher::{FetchConfig, is_hapi_no_data_response},
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
    about = "Reconstruct a THEMIS FGM or ESA daily-file cache from its tracked sha256 manifest"
)]
struct Cli {
    /// Probe letter: a, b, c, d, or e.
    #[arg(long)]
    probe: String,
    /// Which daily-file cache the manifest describes.
    #[arg(long, value_enum, default_value_t = Instrument::Fgm)]
    instrument: Instrument,
    /// Tracked manifest listing sha256, size, and relative path per day file.
    #[arg(long)]
    manifest: PathBuf,
    /// Root directory holding the themis/ or themis_esa/ cache subdirectory.
    #[arg(long, default_value = "data/external")]
    data_dir: PathBuf,
    /// Re-fetch manifest entries even when a file already exists.
    #[arg(long, default_value_t = false, action = clap::ArgAction::Set)]
    refresh: bool,
}

/// The two THEMIS daily-file caches share one manifest grammar and differ
/// in HAPI dataset, cache subdirectory and filename tag.
#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Instrument {
    /// `th<p>_fgm_<year>_<doy>.csv` under `themis/`, dataset TH<P>_L2_FGM@0.
    Fgm,
    /// `th<p>_esa_<year>_<doy>.csv` under `themis_esa/`, dataset TH<P>_L2_ESA@0.
    Esa,
}

impl Instrument {
    fn subdir(self) -> &'static str {
        match self {
            Instrument::Fgm => "themis",
            Instrument::Esa => "themis_esa",
        }
    }

    fn tag(self) -> &'static str {
        match self {
            Instrument::Fgm => "fgm",
            Instrument::Esa => "esa",
        }
    }
}

struct ManifestEntry {
    rel: String,
    digest: String,
    size: u64,
    year: u16,
    doy: u16,
}

/// A cache file is an unaudited extra when it matches this probe's daily-file
/// rule (`th<p>_<tag>_*.csv`) yet carries no manifest entry. Such a file still
/// satisfies the THEMIS source glob and, lacking a provenance hash, would pass
/// every downstream origin check on trust alone.
fn is_unlisted_cache_extra(
    name: &str,
    probe_prefix: &str,
    listed: &std::collections::BTreeSet<&str>,
) -> bool {
    name.starts_with(probe_prefix) && name.ends_with(".csv") && !listed.contains(name)
}

fn cache_probe_prefix(probe: &str, instrument: Instrument) -> String {
    format!("th{}_{}_", probe.to_ascii_lowercase(), instrument.tag())
}

fn is_hapi_no_data_marker(path: &Path) -> bool {
    let Ok(metadata) = fs::metadata(path) else {
        return false;
    };
    if metadata.len() > 65536 {
        return false;
    }
    fs::read_to_string(path)
        .map(|body| is_hapi_no_data_response(&body))
        .unwrap_or(false)
}

fn refresh_backup_path(target: &Path) -> PathBuf {
    target.with_extension("csv.refresh-backup")
}

fn begin_refresh_backup(target: &Path) -> Result<Option<PathBuf>> {
    if !target.exists() {
        return Ok(None);
    }
    if !target.is_file() {
        bail!("refresh target is not a regular file: {}", target.display());
    }
    let backup = refresh_backup_path(target);
    if backup.exists() {
        bail!("refresh backup already exists: {}", backup.display());
    }
    fs::rename(target, &backup)
        .with_context(|| format!("move {} aside before refresh", target.display()))?;
    Ok(Some(backup))
}

fn rollback_refresh(target: &Path, backup: Option<&Path>) -> Result<()> {
    if target.exists() {
        fs::remove_file(target)
            .with_context(|| format!("remove failed refresh output {}", target.display()))?;
    }
    if let Some(backup) = backup {
        fs::rename(backup, target).with_context(|| {
            format!(
                "restore refresh backup {} to {}",
                backup.display(),
                target.display()
            )
        })?;
    }
    Ok(())
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

/// Accepts `th{probe}_{tag}_{year}_{doy}.csv`; the year/doy pair drives the
/// per-day provider fetch.
fn parse_day_filename(rel: &str, probe: &str, instrument: Instrument) -> Option<(u16, u16)> {
    let prefix = cache_probe_prefix(probe, instrument);
    let stem = rel.strip_prefix(&prefix)?.strip_suffix(".csv")?;
    let (year, doy) = stem.split_once('_')?;
    Some((year.parse().ok()?, doy.parse().ok()?))
}

fn parse_manifest(path: &Path, probe: &str, instrument: Instrument) -> Result<Vec<ManifestEntry>> {
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
        let Some((year, doy)) = parse_day_filename(rel, probe, instrument) else {
            bail!(
                "manifest line {} is not a th{probe}_{} daily file: {rel:?}",
                lineno + 1,
                instrument.tag()
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
    let entries = parse_manifest(&cli.manifest, &probe, cli.instrument)?;
    let cache_dir = cli.data_dir.join(cli.instrument.subdir());
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
        let probe_id = format!("TH{}", probe.to_uppercase());
        let mut backup = if cli.refresh {
            begin_refresh_backup(&target)?
        } else {
            None
        };
        let fetched_dir = match cli.instrument {
            Instrument::Fgm => ThemisFgmProvider {
                probe: probe_id,
                year: entry.year,
                doy_start: entry.doy,
                doy_end: entry.doy,
            }
            .fetch_raw(&config),
            Instrument::Esa => ThemisEsaProvider {
                probe: probe_id,
                year: entry.year,
                doy_start: entry.doy,
                doy_end: entry.doy,
            }
            .fetch_raw(&config),
        };
        match fetched_dir {
            Ok(_) if target.is_file() => fetched += 1,
            Ok(_) => {
                rollback_refresh(&target, backup.as_deref())?;
                backup = None;
                eprintln!(
                    "FETCH-ERROR  {}: provider returned success without materializing {}",
                    entry.rel,
                    target.display()
                );
                fetch_errors += 1;
            }
            Err(err) => {
                rollback_refresh(&target, backup.as_deref())?;
                backup = None;
                eprintln!("FETCH-ERROR  {}: {err}", entry.rel);
                fetch_errors += 1;
            }
        }
        if let Some(backup) = backup {
            fs::remove_file(&backup)
                .with_context(|| format!("remove refresh backup {}", backup.display()))?;
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
    let listed: std::collections::BTreeSet<&str> =
        entries.iter().map(|entry| entry.rel.as_str()).collect();
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
    // Scan for cache files that match this probe's daily-file rule yet carry no
    // manifest entry. Payloads without a provenance hash remain extras, while
    // an explicit HAPI no-data marker stays as an out-of-scope fetch attempt.
    let mut extra = 0usize;
    let probe_prefix = cache_probe_prefix(&probe, cli.instrument);
    if let Ok(read_dir) = fs::read_dir(&cache_dir) {
        for dir_entry in read_dir.filter_map(|e| e.ok()) {
            let name = dir_entry.file_name().to_string_lossy().to_string();
            if is_unlisted_cache_extra(&name, &probe_prefix, &listed)
                && !is_hapi_no_data_marker(&dir_entry.path())
            {
                println!("EXTRA    {name}");
                extra += 1;
            }
        }
    }
    println!(
        "manifest verification: {} entries, {} missing, {} drifted, {} extra",
        entries.len(),
        missing,
        drifted,
        extra
    );
    if missing > 0 || drifted > 0 || extra > 0 || fetch_errors > 0 {
        bail!("cache does not match manifest after restore");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn day_filename_parses_year_and_doy() {
        assert_eq!(
            parse_day_filename("tha_fgm_2008_153.csv", "a", Instrument::Fgm),
            Some((2008, 153))
        );
        assert_eq!(
            parse_day_filename("thb_fgm_2008_153.csv", "a", Instrument::Fgm),
            None
        );
        assert_eq!(
            parse_day_filename("tha_fgm_2008_153.txt", "a", Instrument::Fgm),
            None
        );
        assert_eq!(
            parse_day_filename("tha_fgm_bad.csv", "a", Instrument::Fgm),
            None
        );
    }

    #[test]
    fn manifest_paths_with_directories_rejected_as_day_files() {
        assert_eq!(
            parse_day_filename("sub/tha_fgm_2008_153.csv", "a", Instrument::Fgm),
            None
        );
    }

    #[test]
    fn unlisted_probe_csv_flagged_as_extra() {
        let listed: std::collections::BTreeSet<&str> =
            ["tha_fgm_2008_153.csv"].into_iter().collect();
        // Listed manifest day: audited, not an extra.
        assert!(!is_unlisted_cache_extra(
            "tha_fgm_2008_153.csv",
            "tha_fgm_",
            &listed
        ));
        // Same probe rule, no manifest entry, no provenance hash: an extra.
        assert!(is_unlisted_cache_extra(
            "tha_fgm_2008_200.csv",
            "tha_fgm_",
            &listed
        ));
        // Another probe's file does not match this probe's rule.
        assert!(!is_unlisted_cache_extra(
            "thb_fgm_2008_200.csv",
            "tha_fgm_",
            &listed
        ));
        // Non-payload extension is out of scope.
        assert!(!is_unlisted_cache_extra(
            "tha_fgm_2008_200.txt",
            "tha_fgm_",
            &listed
        ));
    }

    #[test]
    fn cache_prefix_normalizes_uppercase_probe() {
        assert_eq!(cache_probe_prefix("A", Instrument::Fgm), "tha_fgm_");
        assert_eq!(cache_probe_prefix("a", Instrument::Esa), "tha_esa_");
        assert_eq!(
            parse_day_filename("tha_esa_2008_301.csv", "a", Instrument::Esa),
            Some((2008, 301))
        );
        assert_eq!(
            parse_day_filename("tha_esa_2008_301.csv", "a", Instrument::Fgm),
            None
        );
        assert_eq!(cache_probe_prefix("c", Instrument::Fgm), "thc_fgm_");
    }

    #[test]
    fn hapi_no_data_marker_is_not_an_extra() {
        let marker = "{\n  \"HAPI\": \"2.0\",\n  \"status\": {\"code\": 1201, \"message\": \"OK - no data for time range\"}\n}\n";
        assert!(is_hapi_no_data_response(marker));
        assert!(!is_hapi_no_data_response("{\"pages\": 12}\n"));
    }

    #[test]
    fn refresh_rollback_restores_existing_payload() {
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("tha_fgm_2008_153.csv");
        fs::write(&target, "old payload").unwrap();

        let backup = begin_refresh_backup(&target).unwrap();
        assert!(!target.exists());
        fs::write(&target, "partial replacement").unwrap();
        rollback_refresh(&target, backup.as_deref()).unwrap();

        assert_eq!(fs::read_to_string(&target).unwrap(), "old payload");
        assert!(!refresh_backup_path(&target).exists());
    }

    #[test]
    fn refresh_rollback_removes_partial_payload_without_backup() {
        let temp = tempfile::tempdir().unwrap();
        let target = temp.path().join("tha_fgm_2008_153.csv");
        fs::write(&target, "partial replacement").unwrap();

        rollback_refresh(&target, None).unwrap();

        assert!(!target.exists());
    }
}
