//! Fetch/provider support for Solar Orbiter SWA-PAS data.

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv};
use std::path::PathBuf;

const SOLO_SWA_PAS_HAPI_DATASET: &str = "SOLO_L2_SWA-PAS-GRND-MOM";

pub struct SolarOrbiterSwaProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for SolarOrbiterSwaProvider {
    fn default() -> Self {
        Self {
            year_start: 2020,
            year_end: 2020,
        }
    }
}

impl DatasetProvider for SolarOrbiterSwaProvider {
    fn name(&self) -> &str {
        "Solar Orbiter SWA-PAS Ground Moments"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("solo").join("soar_swa_pas");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let output = dir.join(format!("solo_l2_swa_pas_grnd_mom_{year}.csv"));
            if config.skip_existing && output.exists() {
                continue;
            }
            match download_hapi_csv(
                SOLO_SWA_PAS_HAPI_DATASET,
                &format!("{year}-01-01T00:00:00Z"),
                &format!("{}-01-01T00:00:00Z", year + 1),
                Some(&["Time", "N", "V_RTN", "T"]),
            ) {
                Ok(body) => {
                    std::fs::write(&output, body)?;
                    log::info!("saved {}", output.display());
                }
                Err(err) => {
                    log::warn!(
                        "failed to download Solar Orbiter SWA {} via HAPI: {}",
                        year,
                        err
                    );
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let dir = config.output_dir.join("solo").join("soar_swa_pas");
        std::fs::read_dir(dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| entry.path().extension().and_then(|value| value.to_str()) == Some("csv"))
    }
}
