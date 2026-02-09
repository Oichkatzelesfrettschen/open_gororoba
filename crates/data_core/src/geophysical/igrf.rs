//! IGRF-13 (International Geomagnetic Reference Field) provider.
//!
//! IGRF-13 provides the main geomagnetic field as spherical harmonic
//! coefficients from 1900 to 2025, with predictive secular variation
//! to 2025. Degree/order up to 13.
//!
//! Source: NOAA/NCEI, https://www.ngdc.noaa.gov/IAGA/vmod/igrf.html
//! Reference: Alken et al. (2021), Earth, Planets and Space 73, 49

use crate::fetcher::{download_with_fallbacks, DatasetProvider, FetchConfig, FetchError};
use std::path::PathBuf;

/// IGRF-13 coefficient file URLs.
const IGRF_URLS: &[&str] = &[
    "https://www.ngdc.noaa.gov/IAGA/vmod/coeffs/igrf13coeffs.txt",
    "https://www.ngdc.noaa.gov/IAGA/vmod/igrf13coeffs.txt",
];

/// IGRF-13 geomagnetic field coefficient provider.
pub struct Igrf13Provider;

impl DatasetProvider for Igrf13Provider {
    fn name(&self) -> &str {
        "IGRF-13 Coefficients"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("igrf13coeffs.txt");
        download_with_fallbacks(self.name(), IGRF_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("igrf13coeffs.txt").exists()
    }
}
