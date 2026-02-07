//! JPL DE440 and DE441 planetary ephemeris kernels.
//!
//! These SPK kernels provide high-precision N-body ephemerides used for
//! spacecraft navigation and high-accuracy solar-system dynamics.
//!
//! Source: NAIF/JPL
//! https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/

use crate::fetcher::{download_with_fallbacks, DatasetProvider, FetchConfig, FetchError};
use std::path::PathBuf;

const DE440_URLS: &[&str] =
    &["https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp"];

const DE441_URLS: &[&str] =
    &["https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de441_part-1.bsp"];

/// DE440 ephemeris kernel provider.
pub struct De440Provider;

impl DatasetProvider for De440Provider {
    fn name(&self) -> &str {
        "JPL DE440 Ephemeris Kernel"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("de440.bsp");
        download_with_fallbacks(self.name(), DE440_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("de440.bsp").exists()
    }
}

/// DE441 ephemeris kernel provider.
pub struct De441Provider;

impl DatasetProvider for De441Provider {
    fn name(&self) -> &str {
        "JPL DE441 Ephemeris Kernel"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("de441_part1.bsp");
        download_with_fallbacks(self.name(), DE441_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("de441_part1.bsp").exists()
    }
}
