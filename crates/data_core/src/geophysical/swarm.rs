//! ESA Swarm magnetic field sample provider.
//!
//! Fetches a short, reproducible sample via the VirES HAPI endpoint for
//! magnetic field vectors and total intensity.
//!
//! Source: https://vires.services/

use crate::fetcher::{download_with_fallbacks, DatasetProvider, FetchConfig, FetchError};
use std::path::PathBuf;

const SWARM_URLS: &[&str] = &[
    "https://vires.services/hapi/data?dataset=SW_OPER_MAGA_LR_1B&parameters=Latitude,Longitude,Radius,F,B_NEC&start=2014-01-01T00:00:00Z&stop=2014-01-01T00:30:00Z&format=csv&include=header",
    "https://vires.services/hapi/data?dataset=SW_OPER_MAGB_LR_1B&parameters=Latitude,Longitude,Radius,F,B_NEC&start=2014-01-01T00:00:00Z&stop=2014-01-01T00:30:00Z&format=csv&include=header",
];

/// Swarm provider.
pub struct SwarmMagAProvider;

impl DatasetProvider for SwarmMagAProvider {
    fn name(&self) -> &str {
        "Swarm L1B Magnetic Sample"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("swarm_magnetic_sample.csv");
        download_with_fallbacks(self.name(), SWARM_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("swarm_magnetic_sample.csv").exists()
    }
}
