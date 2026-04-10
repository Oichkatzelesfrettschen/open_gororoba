//! Fetch/provider support for ACE MAG data.

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv};
use std::{fs, path::PathBuf};

const ACE_MAG_HAPI_DATASET: &str = "AC_H2_MFI";

/// ACE MAG dataset provider.
///
/// Fetches hourly ACE magnetic-field samples through CDAWeb HAPI.
pub struct AceMagProvider {
    /// Start year (inclusive).
    pub year_start: u16,
    /// End year (inclusive).
    pub year_end: u16,
    /// Day of year range (inclusive). If None, fetches all 365/366 days.
    pub doy_range: Option<(u16, u16)>,
}

impl Default for AceMagProvider {
    fn default() -> Self {
        Self {
            year_start: 2024,
            year_end: 2024,
            doy_range: None,
        }
    }
}

impl DatasetProvider for AceMagProvider {
    fn name(&self) -> &str {
        "ACE MAG L2 Browse 16-sec"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("ace_mag");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let fname = format!("ac_h2_mfi_{year}.csv");
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            let body = download_hapi_csv(
                ACE_MAG_HAPI_DATASET,
                &format!("{year}-01-01T00:00:00Z"),
                &format!("{}-01-01T00:00:00Z", year + 1),
                Some(&["Time", "Magnitude", "BGSEc"]),
            )?;
            fs::write(&output, body)?;
            log::info!("Saved {}", fname);
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("ace_mag").exists()
    }
}
