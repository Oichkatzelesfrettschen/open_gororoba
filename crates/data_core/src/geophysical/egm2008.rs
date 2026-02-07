//! EGM2008 static Earth gravity model provider.
//!
//! Source: NGA Earth Gravity Model 2008 distribution.

use crate::fetcher::{download_with_fallbacks, DatasetProvider, FetchConfig, FetchError};
use std::path::PathBuf;

const EGM2008_URLS: &[&str] = &[
    "https://earth-info.nga.mil/GandG/wgs84/gravitymod/egm2008/EGM2008_to2190_TideFree.gz",
    "http://earth-info.nga.mil/GandG/wgs84/gravitymod/egm2008/EGM2008_to2190_TideFree.gz",
];

/// EGM2008 provider.
pub struct Egm2008Provider;

impl DatasetProvider for Egm2008Provider {
    fn name(&self) -> &str {
        "EGM2008 Static Geoid"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("EGM2008_to2190_TideFree.gz");
        download_with_fallbacks(self.name(), EGM2008_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config
            .output_dir
            .join("EGM2008_to2190_TideFree.gz")
            .exists()
    }
}
