//! Event Horizon Telescope public data products.
//!
//! This module provides access to all publicly released EHT calibrated
//! visibility datasets. Each source has UVFITS (full complex visibilities),
//! CSV (tabulated amplitudes/phases), and TXT (human-readable) formats.
//!
//! Sources (6 releases, 2017-2018 campaigns):
//! - M87* 2017 (first black hole image, Paper I-VI, 2019-D01-01)
//! - M87* 2018 (multi-epoch follow-up, 2024-D01-01)
//! - Sgr A* 2017 (Milky Way SMBH, Paper I-VI, 2022-D02-01)
//! - 3C 279 2017 (quasar jet, 2020-D01-01)
//! - Centaurus A 2017 (radio galaxy jet, 2021-D03-01)
//! - M87* 2011-2013 legacy monitoring (pre-EHT array, 2020-D03-01)
//!
//! Reference: https://eventhorizontelescope.org/for-astronomers/data

use crate::fetcher::{download_with_fallbacks, DatasetProvider, FetchConfig, FetchError};
use flate2::read::GzDecoder;
use std::path::{Path, PathBuf};
use tar::Archive;

// ---------------------------------------------------------------------------
// URL constants (verified 2026-02-07, all HTTP 200)
// ---------------------------------------------------------------------------

// M87 2018 (2024-D01-01, branch: main)
const EHT_M87_2018_CSV: &[&str] =
    &["https://github.com/eventhorizontelescope/2024-D01-01/raw/main/EHTC_M872018_csv.tgz"];
const EHT_M87_2018_UVFITS: &[&str] =
    &["https://github.com/eventhorizontelescope/2024-D01-01/raw/main/EHTC_M872018_uvfits.tgz"];
const EHT_M87_2018_TXT: &[&str] =
    &["https://github.com/eventhorizontelescope/2024-D01-01/raw/main/EHTC_M872018_txt.tgz"];

// M87 2017 -- first image (2019-D01-01, branch: master)
const EHT_M87_2017_CSV: &[&str] = &[
    "https://github.com/eventhorizontelescope/2019-D01-01/raw/master/EHTC_FirstM87Results_Apr2019_csv.tgz",
];
const EHT_M87_2017_UVFITS: &[&str] = &[
    "https://github.com/eventhorizontelescope/2019-D01-01/raw/master/EHTC_FirstM87Results_Apr2019_uvfits.tgz",
];
const EHT_M87_2017_TXT: &[&str] = &[
    "https://github.com/eventhorizontelescope/2019-D01-01/raw/master/EHTC_FirstM87Results_Apr2019_txt.tgz",
];

// Sgr A* 2017 (2022-D02-01, branch: main)
const EHT_SGRA_CSV: &[&str] = &[
    "https://github.com/eventhorizontelescope/2022-D02-01/raw/main/EHTC_FirstSgrAResults_May2022_csv.tgz",
];
const EHT_SGRA_UVFITS: &[&str] = &[
    "https://github.com/eventhorizontelescope/2022-D02-01/raw/main/EHTC_FirstSgrAResults_May2022_uvfits.tgz",
];
const EHT_SGRA_TXT: &[&str] = &[
    "https://github.com/eventhorizontelescope/2022-D02-01/raw/main/EHTC_FirstSgrAResults_May2022_txt.tgz",
];

// 3C 279 2017 (2020-D01-01, branch: master)
const EHT_3C279_CSV: &[&str] = &[
    "https://github.com/eventhorizontelescope/2020-D01-01/raw/master/EHTC_First3C279Results_May2020_csv.tgz",
];
const EHT_3C279_UVFITS: &[&str] = &[
    "https://github.com/eventhorizontelescope/2020-D01-01/raw/master/EHTC_First3C279Results_May2020_uvfits.tgz",
];
const EHT_3C279_TXT: &[&str] = &[
    "https://github.com/eventhorizontelescope/2020-D01-01/raw/master/EHTC_First3C279Results_May2020_txt.tgz",
];

// Centaurus A 2017 (2021-D03-01, branch: main)
const EHT_CENA_CSV: &[&str] = &[
    "https://github.com/eventhorizontelescope/2021-D03-01/raw/main/EHTC_CenA_data_July2021_csv.tgz",
];
const EHT_CENA_UVFITS: &[&str] = &[
    "https://github.com/eventhorizontelescope/2021-D03-01/raw/main/EHTC_CenA_data_July2021_uvfits.tgz",
];
const EHT_CENA_TXT: &[&str] = &[
    "https://github.com/eventhorizontelescope/2021-D03-01/raw/main/EHTC_CenA_data_July2021_txt.tgz",
];

// M87 2011-2013 legacy monitoring (2020-D03-01, branch: master)
const EHT_M87_LEGACY: &[&str] = &[
    "https://github.com/eventhorizontelescope/2020-D03-01/raw/master/EHTC_MonitoringM87_Sep2020.tgz",
];

// ---------------------------------------------------------------------------
// Archive utilities
// ---------------------------------------------------------------------------

/// Count members in a .tgz archive.
pub fn tgz_member_count(path: &Path) -> Result<usize, FetchError> {
    let file = std::fs::File::open(path)?;
    let gz = GzDecoder::new(file);
    let mut tar = Archive::new(gz);
    let mut count = 0usize;
    for entry in tar.entries()? {
        let _ = entry?;
        count += 1;
    }
    Ok(count)
}

/// List filenames in a .tgz archive.
pub fn list_tgz_members(path: &Path) -> Result<Vec<String>, FetchError> {
    let file = std::fs::File::open(path)?;
    let gz = GzDecoder::new(file);
    let mut tar = Archive::new(gz);
    let mut names = Vec::new();
    for entry in tar.entries()? {
        let entry = entry?;
        if let Ok(p) = entry.path() {
            names.push(p.display().to_string());
        }
    }
    Ok(names)
}

/// Check that an EHT archive contains at least one file matching a pattern
/// with an expected extension (csv, uvfits, or txt).
pub fn validate_eht_archive(path: &Path, pattern: &str) -> Result<(), FetchError> {
    let members = list_tgz_members(path)?;
    let pat_lower = pattern.to_lowercase();
    let has_match = members.iter().any(|name| {
        let lower = name.to_lowercase();
        lower.contains(&pat_lower)
            && (lower.ends_with(".csv")
                || lower.ends_with(".uvfits")
                || lower.ends_with(".txt")
                || lower.ends_with(".uvf"))
    });
    if !has_match {
        return Err(FetchError::Validation(format!(
            "EHT archive at {} contains no data files matching pattern '{}'",
            path.display(),
            pattern
        )));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helper: multi-format download
// ---------------------------------------------------------------------------

/// Download all format bundles (CSV, UVFITS, TXT) for an EHT source.
/// Returns the path to the UVFITS bundle (primary data product).
fn fetch_eht_multi(
    name: &str,
    dir: &Path,
    prefix: &str,
    csv_urls: &[&str],
    uvfits_urls: &[&str],
    txt_urls: &[&str],
    skip_existing: bool,
) -> Result<PathBuf, FetchError> {
    let csv_out = dir.join(format!("{prefix}_csv.tgz"));
    let uvfits_out = dir.join(format!("{prefix}_uvfits.tgz"));
    let txt_out = dir.join(format!("{prefix}_txt.tgz"));

    // UVFITS is the primary product; download it first
    download_with_fallbacks(name, uvfits_urls, &uvfits_out, skip_existing)?;

    // CSV and TXT are supplementary; log but do not fail on error
    if let Err(e) = download_with_fallbacks(name, csv_urls, &csv_out, skip_existing) {
        eprintln!("  Warning: CSV bundle for {name} failed: {e}");
    }
    if let Err(e) = download_with_fallbacks(name, txt_urls, &txt_out, skip_existing) {
        eprintln!("  Warning: TXT bundle for {name} failed: {e}");
    }

    Ok(uvfits_out)
}

// ---------------------------------------------------------------------------
// Providers
// ---------------------------------------------------------------------------

/// EHT M87* 2017 -- first black hole image (Papers I-VI).
/// Bands: low + high. Pipeline: EHT-HOPS. Stokes I only.
pub struct EhtM87_2017Provider;

impl DatasetProvider for EhtM87_2017Provider {
    fn name(&self) -> &str {
        "EHT M87 2017 (First Image)"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        fetch_eht_multi(
            self.name(),
            &config.output_dir,
            "eht_m87_2017",
            EHT_M87_2017_CSV,
            EHT_M87_2017_UVFITS,
            EHT_M87_2017_TXT,
            config.skip_existing,
        )
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("eht_m87_2017_uvfits.tgz").exists()
    }
}

/// EHT M87* 2018 multi-epoch follow-up.
/// Bands: b1-b4. Pipelines: EHT-HOPS + CASA rPICARD. Stokes I only.
pub struct EhtM87Provider;

impl DatasetProvider for EhtM87Provider {
    fn name(&self) -> &str {
        "EHT M87 2018 Data Bundle"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        fetch_eht_multi(
            self.name(),
            &config.output_dir,
            "eht_m87_2018",
            EHT_M87_2018_CSV,
            EHT_M87_2018_UVFITS,
            EHT_M87_2018_TXT,
            config.skip_existing,
        )
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("eht_m87_2018_uvfits.tgz").exists()
    }
}

/// EHT Sgr A* 2017 -- first Milky Way black hole image (Papers I-VI).
/// Bands: low + high. Pipelines: EHT-HOPS + CASA rPICARD. Stokes I only.
/// Includes standard, lightcurve-normalized, and 100-min optimal subarray variants.
pub struct EhtSgrAProvider;

impl DatasetProvider for EhtSgrAProvider {
    fn name(&self) -> &str {
        "EHT Sgr A* 2017 Data Bundle"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        fetch_eht_multi(
            self.name(),
            &config.output_dir,
            "eht_sgra_2017",
            EHT_SGRA_CSV,
            EHT_SGRA_UVFITS,
            EHT_SGRA_TXT,
            config.skip_existing,
        )
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("eht_sgra_2017_uvfits.tgz").exists()
    }
}

/// EHT 3C 279 2017 -- quasar jet morphology.
/// Bands: low + high. Pipeline: EHT-HOPS.
pub struct Eht3c279Provider;

impl DatasetProvider for Eht3c279Provider {
    fn name(&self) -> &str {
        "EHT 3C279 2017 Data Bundle"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        fetch_eht_multi(
            self.name(),
            &config.output_dir,
            "eht_3c279_2017",
            EHT_3C279_CSV,
            EHT_3C279_UVFITS,
            EHT_3C279_TXT,
            config.skip_existing,
        )
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("eht_3c279_2017_uvfits.tgz").exists()
    }
}

/// EHT Centaurus A 2017 -- nearby radio galaxy jet.
/// Single date (April 10, 2017). Bands: low + high. Pipeline: EHT-HOPS.
pub struct EhtCenAProvider;

impl DatasetProvider for EhtCenAProvider {
    fn name(&self) -> &str {
        "EHT Centaurus A 2017 Data Bundle"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        fetch_eht_multi(
            self.name(),
            &config.output_dir,
            "eht_cena_2017",
            EHT_CENA_CSV,
            EHT_CENA_UVFITS,
            EHT_CENA_TXT,
            config.skip_existing,
        )
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("eht_cena_2017_uvfits.tgz").exists()
    }
}

/// EHT M87* 2011-2013 legacy monitoring data.
/// Pre-full-array observations, amplitude-only visibility data.
/// Single .tgz containing all epochs (24 KB).
pub struct EhtM87LegacyProvider;

impl DatasetProvider for EhtM87LegacyProvider {
    fn name(&self) -> &str {
        "EHT M87 2011-2013 Legacy"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("eht_m87_legacy_2011_2013.tgz");
        download_with_fallbacks(self.name(), EHT_M87_LEGACY, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config
            .output_dir
            .join("eht_m87_legacy_2011_2013.tgz")
            .exists()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Create a synthetic .tgz containing files with given names.
    fn create_test_tgz(path: &Path, filenames: &[&str]) {
        let file = std::fs::File::create(path).unwrap();
        let gz = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        let mut tar = tar::Builder::new(gz);
        for name in filenames {
            let data = b"col1,col2\n1,2\n";
            let mut header = tar::Header::new_gnu();
            header.set_size(data.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            tar.append_data(&mut header, name, &data[..]).unwrap();
        }
        tar.finish().unwrap();
    }

    #[test]
    fn test_list_tgz_members() {
        let dir = std::env::temp_dir().join("eht_list_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.tgz");
        create_test_tgz(&path, &["data/ehtc_m87_results.csv", "data/readme.txt"]);

        let members = list_tgz_members(&path).unwrap();
        assert_eq!(members.len(), 2);
        assert!(members.iter().any(|m| m.contains("m87")));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_validate_eht_archive_accepts_csv() {
        let dir = std::env::temp_dir().join("eht_validate_csv_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("good.tgz");
        create_test_tgz(&path, &["EHTC_M87_uv_data.csv", "EHTC_M87_image.csv"]);

        assert!(validate_eht_archive(&path, "m87").is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_validate_eht_archive_accepts_uvfits() {
        let dir = std::env::temp_dir().join("eht_validate_uvfits_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("uv.tgz");
        create_test_tgz(&path, &["hops_lo/M87_b1.uvfits", "hops_hi/M87_b2.uvfits"]);

        assert!(validate_eht_archive(&path, "m87").is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_validate_eht_archive_rejects_missing() {
        let dir = std::env::temp_dir().join("eht_validate_bad_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.tgz");
        create_test_tgz(&path, &["unrelated_data.csv"]);

        assert!(validate_eht_archive(&path, "m87").is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_tgz_member_count_synthetic() {
        let dir = std::env::temp_dir().join("eht_count_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("count.tgz");
        create_test_tgz(&path, &["a.csv", "b.csv", "c.csv"]);

        let n = tgz_member_count(&path).unwrap();
        assert_eq!(n, 3);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_eht_m87_2018_archive_if_available() {
        // Check either the old CSV-only path or the new UVFITS path
        let paths = [
            Path::new("data/external/eht_m87_2018_uvfits.tgz"),
            Path::new("data/external/eht_m87_2018_csv.tgz"),
        ];
        let path = match paths.iter().find(|p| p.exists()) {
            Some(p) => p,
            None => {
                eprintln!("Skipping: no EHT M87 2018 archive available");
                return;
            }
        };
        let n = tgz_member_count(path).expect("failed to read EHT M87 archive");
        assert!(n > 0, "EHT M87 archive should contain members");
    }

    #[test]
    fn test_eht_sgra_archive_if_available() {
        let paths = [
            Path::new("data/external/eht_sgra_2017_uvfits.tgz"),
            Path::new("data/external/eht_sgra_2022_csv.tgz"),
        ];
        let path = match paths.iter().find(|p| p.exists()) {
            Some(p) => p,
            None => {
                eprintln!("Skipping: no EHT SgrA archive available");
                return;
            }
        };
        let n = tgz_member_count(path).expect("failed to read EHT SgrA archive");
        assert!(n > 0, "EHT SgrA archive should contain members");
    }

    #[test]
    fn test_provider_names_are_unique() {
        let names: Vec<&str> = vec![
            EhtM87_2017Provider.name(),
            EhtM87Provider.name(),
            EhtSgrAProvider.name(),
            Eht3c279Provider.name(),
            EhtCenAProvider.name(),
            EhtM87LegacyProvider.name(),
        ];
        let mut sorted = names.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            names.len(),
            sorted.len(),
            "EHT provider names must be unique"
        );
    }
}
