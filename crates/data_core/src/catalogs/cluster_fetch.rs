//! Fetch/provider support for Cluster FGM SPIN data.

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv};
use chrono::NaiveDate;
use std::{fs, path::PathBuf};

fn hapi_dataset_for_probe(probe_id: u8) -> String {
    format!("C{probe_id}_CP_FGM_SPIN")
}

/// Cluster FGM provider for a single probe.
pub struct ClusterFgmProvider {
    pub probe_id: u8, // 1-4
    pub year: u16,
    pub doy_start: u16,
    pub doy_end: u16,
}

impl DatasetProvider for ClusterFgmProvider {
    fn name(&self) -> &str {
        "Cluster FGM SPIN"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("cluster");
        fs::create_dir_all(&dir)?;

        let dataset = hapi_dataset_for_probe(self.probe_id);

        for doy in self.doy_start..=self.doy_end {
            let date = NaiveDate::from_yo_opt(self.year as i32, doy as u32)
                .ok_or_else(|| FetchError::Validation(format!("invalid DOY {doy}")))?;
            let fname = format!(
                "c{}_fgm_spin_{:04}_{:03}.csv",
                self.probe_id, self.year, doy
            );
            let output = dir.join(&fname);

            if config.skip_existing && output.exists() {
                continue;
            }

            let t_min = format!("{}T00:00:00Z", date);
            let t_max = format!("{}T23:59:59Z", date);

            println!(
                "Fetching C{} FGM SPIN {} DOY {}...",
                self.probe_id, self.year, doy
            );

            match download_hapi_csv(&dataset, &t_min, &t_max, None) {
                Ok(body) => {
                    fs::write(&output, body)?;
                }
                Err(e) => {
                    eprintln!("  Warning: C{} DOY {}: {}", self.probe_id, doy, e);
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("cluster").exists()
    }
}
