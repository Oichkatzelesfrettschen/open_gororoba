//! Fetch/provider support for Cassini cruise merged hourly data.

use crate::{
    catalogs::cassini::CASSINI_CRUISE_BASE,
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_string},
};
use std::path::PathBuf;

/// NASA SPDF Cassini cruise dataset provider.
pub struct CassiniCruiseProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for CassiniCruiseProvider {
    fn default() -> Self {
        Self {
            year_start: 1997,
            year_end: 2004,
        }
    }
}

impl DatasetProvider for CassiniCruiseProvider {
    fn name(&self) -> &str {
        "Cassini Cruise Merged Hourly"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("cassini");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let fname = format!("cassini_{}_merged_hourly.asc", year);
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            let url = format!("{}{}", CASSINI_CRUISE_BASE, fname);
            match download_to_string(&url) {
                Ok(data) => {
                    std::fs::write(&output, data)?;
                    log::info!("saved {}", fname);
                }
                Err(e) => {
                    log::warn!("failed to download Cassini {}: {}", year, e);
                }
            }
        }
        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("cassini").exists()
    }
}
