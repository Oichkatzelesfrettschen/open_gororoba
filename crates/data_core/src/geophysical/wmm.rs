//! WMM 2025 (World Magnetic Model) provider.
//!
//! WMM is the standard geomagnetic model for navigation, produced by
//! NOAA/NCEI and the British Geological Survey. Updated every 5 years.
//! Degree/order up to 12, valid 2025-2030.
//!
//! Source: NOAA/NCEI, https://www.ncei.noaa.gov/products/world-magnetic-model
//! Reference: Chulliat et al. (2024), NOAA Technical Report

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_with_fallbacks};
use std::path::PathBuf;

/// WMM 2025 coefficient file URLs.
///
/// The ZIP contains WMM.COF (the coefficient file) and supporting documents.
const WMM_URLS: &[&str] = &[
    "https://www.ncei.noaa.gov/products/world-magnetic-model/wmm2025_Linux.zip",
    "https://www.ngdc.noaa.gov/geomag/WMM/data/WMM2025/WMM2025COF.zip",
];

/// WMM 2025 geomagnetic model provider.
pub struct Wmm2025Provider;

impl DatasetProvider for Wmm2025Provider {
    fn name(&self) -> &str { "WMM 2025 Coefficients" }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("wmm2025.zip");
        download_with_fallbacks(self.name(), WMM_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("wmm2025.zip").exists()
    }
}
