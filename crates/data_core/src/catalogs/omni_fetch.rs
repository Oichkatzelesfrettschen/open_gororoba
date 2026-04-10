//! Fetch/provider support for NASA OMNI2 hourly solar wind + IMF data.

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_string};
use std::path::PathBuf;

const OMNI2_BASE: &str = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_";

/// NASA OMNI2 dataset provider.
///
/// Fetches hourly solar wind + IMF data for a specified year range.
/// Each yearly file is ~2.7 MB of ASCII data (~8760 rows).
pub struct OmniProvider {
    /// Start year (inclusive).
    pub year_start: u16,
    /// End year (inclusive).
    pub year_end: u16,
}

impl Default for OmniProvider {
    fn default() -> Self {
        Self {
            year_start: 2024,
            year_end: 2024,
        }
    }
}

impl DatasetProvider for OmniProvider {
    fn name(&self) -> &str {
        "NASA OMNI2 Solar Wind + IMF"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("omni2");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let fname = format!("omni2_{}.dat", year);
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            let url = format!("{}{}.dat", OMNI2_BASE, year);
            match download_to_string(&url) {
                Ok(data) => {
                    std::fs::write(&output, data)?;
                    log::info!("Saved {}", fname);
                }
                Err(e) => {
                    log::warn!("Failed to download OMNI2 {}: {}", year, e);
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("omni2").exists()
    }
}
