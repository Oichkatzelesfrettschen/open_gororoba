//! JPL DE440 and DE441 planetary ephemeris kernels.
//!
//! These SPK kernels provide high-precision N-body ephemerides used for
//! spacecraft navigation and high-accuracy solar-system dynamics.
//!
//! Source: NAIF/JPL
//! https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/

use crate::fetcher::{download_with_fallbacks, DatasetProvider, FetchConfig, FetchError};
use std::path::{Path, PathBuf};

/// DAF/SPK magic bytes at the start of a binary SPK kernel.
const DAF_SPK_MAGIC: &[u8; 7] = b"DAF/SPK";

/// Validate that a file starts with the DAF/SPK magic header.
///
/// SPK kernels use NASA's Double-precision Array File format. The first
/// 7 bytes identify the file type.
pub fn validate_spk_magic(path: &Path) -> Result<(), FetchError> {
    let data = std::fs::read(path)?;
    if data.len() < 7 {
        return Err(FetchError::Validation(format!(
            "SPK file too small ({} bytes): {}",
            data.len(),
            path.display()
        )));
    }
    if &data[..7] != DAF_SPK_MAGIC {
        return Err(FetchError::Validation(format!(
            "SPK file missing DAF/SPK magic header: {}",
            path.display()
        )));
    }
    Ok(())
}

/// Minimum expected file size for DE440 (~120 MB).
pub const DE440_MIN_SIZE: u64 = 100_000_000;

/// Minimum expected file size for DE441 part-1 (~2 GB).
pub const DE441_MIN_SIZE: u64 = 1_000_000_000;

const DE440_URLS: &[&str] =
    &["https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp"];

const DE441_URLS: &[&str] =
    &["https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de441_part-1.bsp"];

/// DE440 ephemeris kernel provider.
pub struct De440Provider;

impl DatasetProvider for De440Provider {
    fn name(&self) -> &str {
        "JPL DE440 Ephemeris Kernel"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("de440.bsp");
        download_with_fallbacks(self.name(), DE440_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("de440.bsp").exists()
    }
}

/// DE441 ephemeris kernel provider.
pub struct De441Provider;

impl DatasetProvider for De441Provider {
    fn name(&self) -> &str {
        "JPL DE441 Ephemeris Kernel"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("de441_part1.bsp");
        download_with_fallbacks(self.name(), DE441_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("de441_part1.bsp").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_validate_spk_magic_accepts_valid() {
        let dir = std::env::temp_dir().join("spk_magic_ok_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("valid.bsp");

        let mut f = std::fs::File::create(&path).unwrap();
        // Write DAF/SPK header followed by filler
        f.write_all(b"DAF/SPK ").unwrap();
        f.write_all(&[0u8; 100]).unwrap();

        assert!(validate_spk_magic(&path).is_ok());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_validate_spk_magic_rejects_wrong_header() {
        let dir = std::env::temp_dir().join("spk_magic_bad_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("invalid.bsp");

        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"NOT_SPK_DATA").unwrap();

        assert!(validate_spk_magic(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_validate_spk_magic_rejects_too_small() {
        let dir = std::env::temp_dir().join("spk_magic_small_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("tiny.bsp");

        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"DA").unwrap();

        assert!(validate_spk_magic(&path).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn test_de440_if_available() {
        let path = std::path::Path::new("data/external/de440.bsp");
        if !path.exists() {
            eprintln!("Skipping: DE440 kernel not available");
            return;
        }
        validate_spk_magic(path).expect("DE440 should have valid DAF/SPK header");
        let size = std::fs::metadata(path).unwrap().len();
        assert!(
            size > DE440_MIN_SIZE,
            "DE440 should be > 100 MB, got {} bytes",
            size
        );
    }
}
