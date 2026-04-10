//! Fetch/provider support for BepiColombo position data.

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv};
use std::path::PathBuf;

const BEPICOLOMBO_POSITION_HAPI_DATASET: &str = "BEPICOLOMBO_HELIO1HR_POSITION";

/// BepiColombo hourly position support dataset provider.
pub struct BepicolomboProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for BepicolomboProvider {
    fn default() -> Self {
        Self {
            year_start: 2020,
            year_end: 2020,
        }
    }
}

impl DatasetProvider for BepicolomboProvider {
    fn name(&self) -> &str {
        "BepiColombo Position Hourly"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("bepicolombo");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let fname = format!("bepicolombo_helio1hr_position_{year}.csv");
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            match download_hapi_csv(
                BEPICOLOMBO_POSITION_HAPI_DATASET,
                &format!("{year}-01-01T00:00:00Z"),
                &format!("{}-01-01T00:00:00Z", year + 1),
                Some(&["Time", "RAD_AU", "HG_LAT", "HG_LON"]),
            ) {
                Ok(data) => {
                    std::fs::write(&output, data)?;
                    log::info!("saved {}", fname);
                }
                Err(e) => {
                    log::warn!("failed to download BepiColombo {}: {}", year, e);
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let dir = config.output_dir.join("bepicolombo");
        std::fs::read_dir(&dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.starts_with("bepicolombo_helio1hr_position_") && name.ends_with(".csv")
            })
    }
}
