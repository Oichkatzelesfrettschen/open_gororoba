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

use crate::fetcher::FetchError;
use flate2::read::GzDecoder;
use std::path::Path;
use tar::Archive;

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

}
