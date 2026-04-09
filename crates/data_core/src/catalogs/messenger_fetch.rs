//! Fetch/provider support for MESSENGER MAG data.

use crate::fetcher::{DailyHapiFetchRequest, DatasetProvider, FetchConfig, FetchError, fetch_daily_hapi_csv_range};
use std::path::PathBuf;

const MESSENGER_MAG_HAPI_DATASET: &str = "MESSENGER_MAG_RTN@0";

/// MESSENGER MAG provider. Daily HAPI chunks.
pub struct MessengerMagProvider {
    pub year: u16,
    pub doy_start: u16,
    pub doy_end: u16,
}

impl DatasetProvider for MessengerMagProvider {
    fn name(&self) -> &str {
        "MESSENGER MAG RTN"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        fetch_daily_hapi_csv_range(
            config,
            &DailyHapiFetchRequest {
                subdir: "messenger",
                file_prefix: "messenger_mag",
                log_label: "MESSENGER MAG",
                dataset_id: MESSENGER_MAG_HAPI_DATASET,
                year: self.year,
                doy_start: self.doy_start,
                doy_end: self.doy_end,
                parameters: None,
            },
        )
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("messenger").exists()
    }
}
