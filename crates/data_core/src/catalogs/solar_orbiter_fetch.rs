//! Fetch/provider support for Solar Orbiter merged hourly data.

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv};
use std::path::PathBuf;

const SOLO_HAPI_DATASET: &str = "SOLO_COHO1HR_MERGED_MAG_PLASMA";

/// Solar Orbiter merged-hourly dataset provider.
pub struct SolarOrbiterProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for SolarOrbiterProvider {
    fn default() -> Self {
        Self {
            year_start: 2020,
            year_end: 2020,
        }
    }
}

impl DatasetProvider for SolarOrbiterProvider {
    fn name(&self) -> &str {
        "Solar Orbiter Merged Hourly"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("solar_orbiter");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let fname = format!("solo_coho1hr_merged_mag_plasma_{year}.csv");
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            match download_hapi_csv(
                SOLO_HAPI_DATASET,
                &format!("{year}-01-01T00:00:00Z"),
                &format!("{}-01-01T00:00:00Z", year + 1),
                Some(&[
                    "Time",
                    "radialDistance",
                    "heliographicLatitude",
                    "heliographicLongitude",
                    "BR",
                    "BT",
                    "BN",
                    "B",
                    "VR",
                    "VT",
                    "VN",
                    "ProtonSpeed",
                    "flow_theta",
                    "flow_lon",
                    "protonDensity",
                    "protonTemp",
                ]),
            ) {
                Ok(data) => {
                    std::fs::write(&output, data)?;
                    log::info!("saved {}", fname);
                }
                Err(e) => {
                    log::warn!("failed to download Solar Orbiter {} via HAPI: {}", year, e);
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let dir = config.output_dir.join("solar_orbiter");
        std::fs::read_dir(&dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| {
                let name = entry.file_name();
                let name = name.to_string_lossy();
                name.starts_with("solo_coho1hr_merged_mag_plasma_") && name.ends_with(".csv")
            })
    }
}
