//! Event Horizon Telescope public data products.
//!
//! This module caches public release bundles for M87* and Sgr A* from the
//! official EHT collaboration repositories.
//!
//! Source: https://github.com/eventhorizontelescope

use crate::fetcher::{download_with_fallbacks, DatasetProvider, FetchConfig, FetchError};
use flate2::read::GzDecoder;
use std::path::{Path, PathBuf};
use tar::Archive;

const EHT_M87_URLS: &[&str] = &[
    "https://raw.githubusercontent.com/eventhorizontelescope/2024-D01-01/main/EHTC_M872018_csv.tgz",
];

const EHT_SGRA_URLS: &[&str] = &[
    "https://raw.githubusercontent.com/eventhorizontelescope/2022-D02-01/main/EHTC_FirstSgrAResults_May2022_csv.tgz",
];

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

/// EHT M87* data bundle provider.
pub struct EhtM87Provider;

impl DatasetProvider for EhtM87Provider {
    fn name(&self) -> &str {
        "EHT M87 2018 Data Bundle"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("eht_m87_2018_csv.tgz");
        download_with_fallbacks(self.name(), EHT_M87_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("eht_m87_2018_csv.tgz").exists()
    }
}

/// EHT Sgr A* data bundle provider.
pub struct EhtSgrAProvider;

impl DatasetProvider for EhtSgrAProvider {
    fn name(&self) -> &str {
        "EHT SgrA 2022 Data Bundle"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("eht_sgra_2022_csv.tgz");
        download_with_fallbacks(self.name(), EHT_SGRA_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("eht_sgra_2022_csv.tgz").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_eht_m87_archive_if_available() {
        let path = Path::new("data/external/eht_m87_2018_csv.tgz");
        if !path.exists() {
            eprintln!("Skipping: EHT M87 archive not available");
            return;
        }
        let n = tgz_member_count(path).expect("failed to read EHT M87 archive");
        assert!(n > 0, "EHT M87 archive should contain members");
    }

    #[test]
    fn test_eht_sgra_archive_if_available() {
        let path = Path::new("data/external/eht_sgra_2022_csv.tgz");
        if !path.exists() {
            eprintln!("Skipping: EHT SgrA archive not available");
            return;
        }
        let n = tgz_member_count(path).expect("failed to read EHT SgrA archive");
        assert!(n > 0, "EHT SgrA archive should contain members");
    }
}
