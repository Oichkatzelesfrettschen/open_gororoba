//! GRACE GGM05S static gravity field model provider.
//!
//! GGM05S is a satellite-only gravity field model derived from GRACE
//! (Gravity Recovery and Climate Experiment) data. Degree/order 180.
//! Available in ICGEM .gfc format (~350 KB).
//!
//! Source: ICGEM, http://icgem.gfz-potsdam.de/
//! Reference: Ries et al. (2016), GFZ Data Services

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_with_fallbacks};
use std::path::PathBuf;

/// ICGEM .gfc download URLs for GGM05S.
const GGM05S_URLS: &[&str] = &[
    "https://icgem.gfz-potsdam.de/getmodel/gfc/06a6faa24892df587d29c8a345e09e7031428cf97d4fcc9435b31ae8e4ccc021/GGM05S.gfc",
];

/// GRACE GGM05S gravity field model provider.
pub struct GraceGgm05sProvider;

impl DatasetProvider for GraceGgm05sProvider {
    fn name(&self) -> &str { "GRACE GGM05S Gravity Field" }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("GGM05S.gfc");
        download_with_fallbacks(self.name(), GGM05S_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("GGM05S.gfc").exists()
    }
}
