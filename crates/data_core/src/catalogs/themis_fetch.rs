//! Fetch/provider support for THEMIS/ARTEMIS FGM data.

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv};
use chrono::NaiveDate;
use std::{fs, path::PathBuf};

fn hapi_dataset_for_probe(probe: &str) -> String {
    format!("{}_L2_FGM@0", probe.to_uppercase())
}

fn gse_param_for_probe(probe: &str) -> String {
    format!("{}_fgs_gse", probe.to_lowercase())
}

/// THEMIS/ARTEMIS FGM provider.
pub struct ThemisFgmProvider {
    /// Probe identifier: "THA", "THB", "THC", "THD", "THE".
    pub probe: String,
    pub year: u16,
    pub doy_start: u16,
    pub doy_end: u16,
}

impl DatasetProvider for ThemisFgmProvider {
    fn name(&self) -> &str {
        "THEMIS FGM L2"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("themis");
        fs::create_dir_all(&dir)?;

        let dataset = hapi_dataset_for_probe(&self.probe);
        let gse_param = gse_param_for_probe(&self.probe);

        for doy in self.doy_start..=self.doy_end {
            let date = NaiveDate::from_yo_opt(self.year as i32, doy as u32)
                .ok_or_else(|| FetchError::Validation(format!("invalid DOY {doy}")))?;
            let fname = format!(
                "{}_fgm_{:04}_{:03}.csv",
                self.probe.to_lowercase(),
                self.year,
                doy
            );
            let output = dir.join(&fname);

            if config.skip_existing && output.exists() {
                continue;
            }

            let t_min = format!("{}T00:00:00Z", date);
            let t_max = format!("{}T23:59:59Z", date);

            println!("Fetching {} FGM {} DOY {}...", self.probe, self.year, doy);

            match download_hapi_csv(&dataset, &t_min, &t_max, Some(&["Time", &gse_param])) {
                Ok(body) => {
                    fs::write(&output, body)?;
                }
                Err(e) => {
                    eprintln!("  Warning: {} DOY {}: {}", self.probe, doy, e);
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("themis").exists()
    }
}
