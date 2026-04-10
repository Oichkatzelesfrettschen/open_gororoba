//! Fetch implementation for solar_orbiter_rpw_density. See solar_orbiter_rpw_density.rs for record types and parsers.

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv};
use chrono::{Datelike, NaiveDate};
use std::path::PathBuf;

const SOLO_RPW_BIA_DENSITY_HAPI_DATASET: &str = "SOLO_L3_RPW-BIA-DENSITY";

pub struct SolarOrbiterRpwDensityProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for SolarOrbiterRpwDensityProvider {
    fn default() -> Self {
        Self {
            year_start: 2020,
            year_end: 2020,
        }
    }
}

impl DatasetProvider for SolarOrbiterRpwDensityProvider {
    fn name(&self) -> &str {
        "Solar Orbiter RPW BIA Density"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config
            .output_dir
            .join("solar_orbiter")
            .join("rpw_bia_density");
        std::fs::create_dir_all(&dir)?;
        for year in self.year_start..=self.year_end {
            let mut day = NaiveDate::from_ymd_opt(year as i32, 6, 15).ok_or_else(|| {
                FetchError::Validation(format!(
                    "invalid Solar Orbiter RPW density start date for {year}"
                ))
            })?;
            let end = NaiveDate::from_ymd_opt(year as i32 + 1, 1, 1).expect("valid next-year date");
            while day < end {
                let next_day = day
                    .succ_opt()
                    .ok_or_else(|| FetchError::Validation(format!("advance {day}")))?;
                let output = dir.join(format!(
                    "solo_l3_rpw-bia-density_{}{:02}{:02}.csv",
                    day.year(),
                    day.month(),
                    day.day()
                ));
                if config.skip_existing && output.exists() {
                    day = next_day;
                    continue;
                }
                match download_hapi_csv(
                    SOLO_RPW_BIA_DENSITY_HAPI_DATASET,
                    &format!("{}T00:00:00Z", day.format("%Y-%m-%d")),
                    &format!("{}T00:00:00Z", next_day.format("%Y-%m-%d")),
                    Some(&["Time", "DENSITY"]),
                ) {
                    Ok(body) => {
                        std::fs::write(&output, body)?;
                        log::info!("saved {}", output.display());
                    }
                    Err(err) => {
                        log::warn!(
                            "failed to download Solar Orbiter RPW density for {}-{:02}-{:02}: {}",
                            day.year(),
                            day.month(),
                            day.day(),
                            err
                        );
                    }
                }
                day = next_day;
            }
        }
        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let dir = config
            .output_dir
            .join("solar_orbiter")
            .join("rpw_bia_density");
        std::fs::read_dir(dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| entry.path().extension().and_then(|value| value.to_str()) == Some("csv"))
    }
}
