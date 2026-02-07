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

/// Check that an EHT archive contains at least one CSV file matching a pattern.
pub fn validate_eht_archive(path: &Path, pattern: &str) -> Result<(), FetchError> {
    let members = list_tgz_members(path)?;
    let has_match = members.iter().any(|name| {
        let lower = name.to_lowercase();
        lower.contains(&pattern.to_lowercase()) && lower.ends_with(".csv")
    });
    if !has_match {
        return Err(FetchError::Validation(format!(
            "EHT archive at {} contains no CSV files matching pattern '{}'",
            path.display(), pattern
        )));
    }
    Ok(())
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
    fn test_validate_eht_archive_accepts_matching() {
        let dir = std::env::temp_dir().join("eht_validate_ok_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("good.tgz");
        create_test_tgz(&path, &["EHTC_M87_uv_data.csv", "EHTC_M87_image.csv"]);

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
