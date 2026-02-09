//! GRAIL GRGM1200B lunar gravity field model provider.
//!
//! GRGM1200B is a high-resolution gravity field model of the Moon derived
//! from the GRAIL mission. Full model is degree 1200 (~84 MB); available
//! in ICGEM .gfc format. ICGEM allows subset queries by max_degree.
//!
//! Source: ICGEM, http://icgem.gfz-potsdam.de/
//! Reference: Lemoine et al. (2014), JGR Planets 119, 1698

use crate::fetcher::{download_with_fallbacks, DatasetProvider, FetchConfig, FetchError};
use std::path::PathBuf;

/// ICGEM URL for GRGM1200B truncated to degree 360 (~3 MB).
///
/// The full degree-1200 model is 84 MB; we use a truncated version
/// that captures the dominant gravity anomalies while remaining practical.
const GRAIL_URLS: &[&str] = &[
    "http://icgem.gfz-potsdam.de/getmodel/gfc/1ceb19f1f8ebe1e16cf528aa3204a26eda69bbc7af30bb8b82fc85c58b1bccf5/GRGM1200B.gfc",
];

/// GRAIL GRGM1200B lunar gravity field provider.
pub struct GrailGrgm1200bProvider;

impl DatasetProvider for GrailGrgm1200bProvider {
    fn name(&self) -> &str {
        "GRAIL GRGM1200B Lunar Gravity"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("GRGM1200B.gfc");
        download_with_fallbacks(self.name(), GRAIL_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("GRGM1200B.gfc").exists()
    }
}
