//! Hipparcos legacy stellar astrometry catalog.
//!
//! Hipparcos is the predecessor to Gaia and remains important for legacy
//! cross-calibration and long-baseline astrometry comparisons.
//!
//! Source: CDS catalog I/239
//! https://cdsarc.cds.unistra.fr/ftp/cats/I/239/

use crate::fetcher::{download_with_fallbacks, DatasetProvider, FetchConfig, FetchError};
use flate2::read::GzDecoder;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

const HIPPARCOS_URLS: &[&str] = &[
    "https://cdsarc.cds.unistra.fr/ftp/cats/I/239/hip_main.dat.gz",
    "https://cdsarc.u-strasbg.fr/ftp/cats/I/239/hip_main.dat.gz",
];

/// Count rows in a gzipped Hipparcos main catalog file.
pub fn hipparcos_row_count_gzip(path: &Path) -> Result<usize, FetchError> {
    let file = std::fs::File::open(path)?;
    let decoder = GzDecoder::new(file);
    let reader = BufReader::new(decoder);
    let mut count = 0usize;
    for line in reader.lines() {
        let line = line?;
        if !line.trim().is_empty() {
            count += 1;
        }
    }
    Ok(count)
}

/// Hipparcos catalog provider.
pub struct HipparcosProvider;

impl DatasetProvider for HipparcosProvider {
    fn name(&self) -> &str {
        "Hipparcos Legacy Catalog"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("hip_main.dat.gz");
        download_with_fallbacks(self.name(), HIPPARCOS_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("hip_main.dat.gz").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn test_hipparcos_if_available() {
        let path = Path::new("data/external/hip_main.dat.gz");
        if !path.exists() {
            eprintln!("Skipping: Hipparcos data not available");
            return;
        }
        let rows = hipparcos_row_count_gzip(path).expect("failed to count Hipparcos rows");
        assert!(
            rows > 1000,
            "Hipparcos file should have many rows, got {}",
            rows
        );
    }
}
