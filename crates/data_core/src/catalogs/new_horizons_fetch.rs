//! Fetch/provider support for New Horizons SWAP data.

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv};
use std::path::PathBuf;

const NH_POSITION_HAPI_DATASET: &str = "NEW_HORIZONS_HELIO1HR_POSITION";

/// NASA SPDF New Horizons SWAP dataset provider.
pub struct NhSwapProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for NhSwapProvider {
    fn default() -> Self {
        Self {
            year_start: 2015,
            year_end: 2023,
        }
    }
}

impl DatasetProvider for NhSwapProvider {
    fn name(&self) -> &str {
        "New Horizons SWAP Hourly"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("new_horizons");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let fname = format!("new_horizons_helio1hr_position_{year}.csv");
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            match download_hapi_csv(
                NH_POSITION_HAPI_DATASET,
                &format!("{year}-01-01T00:00:00Z"),
                &format!("{}-01-01T00:00:00Z", year + 1),
                Some(&["Time", "RAD_AU", "HG_LAT", "HG_LON"]),
            ) {
                Ok(data) => {
                    std::fs::write(&output, data)?;
                    log::info!("saved {}", fname);
                }
                Err(e) => {
                    log::warn!("failed to download NH SWAP {}: {}", year, e);
                }
            }
        }
        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("new_horizons").exists()
    }
}
