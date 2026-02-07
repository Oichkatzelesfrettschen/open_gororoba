//! GRACE-FO gravity field model provider.
//!
//! GRACE Follow-On extends time-variable gravity monitoring after GRACE.
//! This provider fetches a representative monthly model file.
//!
//! Source: ICGEM GFZ model services

use crate::fetcher::{download_with_fallbacks, DatasetProvider, FetchConfig, FetchError};
use std::path::PathBuf;

const GRACE_FO_URLS: &[&str] = &[
    "https://icgem.gfz-potsdam.de/getseries/01_GRACE/GFZ/GFZ%20Release%2006.3%20%28GFO%29/60x60/unfiltered/GSM-2_2018152-2018181_GRFO_GFZOP_BA01_0603.gfc",
    "https://icgem.gfz-potsdam.de/getseries/01_GRACE/JPL/JPL%20Release%2006.3%20%28GFO%29/60x60/unfiltered/GSM-2_2018152-2018181_GRFO_JPLEM_BA01_0603.gfc",
];

/// GRACE-FO provider.
pub struct GraceFoProvider;

impl DatasetProvider for GraceFoProvider {
    fn name(&self) -> &str {
        "GRACE-FO Gravity Field"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("GRACEFO_monthly_sample.gfc");
        download_with_fallbacks(self.name(), GRACE_FO_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config
            .output_dir
            .join("GRACEFO_monthly_sample.gfc")
            .exists()
    }
}
