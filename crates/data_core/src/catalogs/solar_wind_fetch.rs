//! Fetch/provider support for ACE SWEPAM solar wind data.

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv};
use std::path::PathBuf;

const ACE_SWEPAM_HAPI_DATASET: &str = "AC_H2_SWE";

/// ACE SWEPAM dataset provider.
///
/// Fetches hourly solar wind data for a specified year range.
pub struct AceSwepamProvider {
    /// Start year (inclusive).
    pub year_start: u16,
    /// End year (inclusive).
    pub year_end: u16,
}

impl Default for AceSwepamProvider {
    fn default() -> Self {
        Self {
            year_start: 2024,
            year_end: 2024,
        }
    }
}

impl DatasetProvider for AceSwepamProvider {
    fn name(&self) -> &str {
        "ACE SWEPAM Solar Wind"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("ace_swepam");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let fname = format!("ac_h2_swe_{year}.csv");
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            match download_hapi_csv(
                ACE_SWEPAM_HAPI_DATASET,
                &format!("{year}-01-01T00:00:00Z"),
                &format!("{}-01-01T00:00:00Z", year + 1),
                Some(&["Time", "Np", "Vp", "Tpr"]),
            ) {
                Ok(data) => {
                    std::fs::write(&output, data)?;
                    log::info!("Saved {}", fname);
                }
                Err(e) => {
                    log::warn!("Failed to download ACE SWEPAM {}: {}", year, e);
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("ace_swepam").exists()
    }
}
