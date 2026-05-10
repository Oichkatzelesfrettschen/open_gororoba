//! Fetch/provider support for MAVEN MAG data.

use crate::fetcher::{
    DailyHapiFetchRequest, DatasetProvider, FetchConfig, FetchError, fetch_daily_hapi_csv_range,
};
use std::path::PathBuf;

const MAVEN_MAG_HAPI_DATASET: &str = "MVN_MAG_L2-SUNSTATE-1SEC";

/// MAVEN MAG provider. Fetches in daily chunks (1-sec data is large).
pub struct MavenMagProvider {
    pub year: u16,
    pub doy_start: u16,
    pub doy_end: u16,
}

impl DatasetProvider for MavenMagProvider {
    fn name(&self) -> &str {
        "MAVEN MAG L2 SunState 1-sec"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        fetch_daily_hapi_csv_range(
            config,
            &DailyHapiFetchRequest {
                subdir: "maven",
                file_prefix: "maven_mag",
                log_label: "MAVEN MAG",
                dataset_id: MAVEN_MAG_HAPI_DATASET,
                year: self.year,
                doy_start: self.doy_start,
                doy_end: self.doy_end,
                parameters: Some(&["Time", "OB_B"]),
            },
        )
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("maven").exists()
    }
}
