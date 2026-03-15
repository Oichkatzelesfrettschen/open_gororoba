//! Solar Orbiter RPW BIA density parser via CDAWeb HAPI.
//!
//! Public official source:
//!   <https://cdaweb.gsfc.nasa.gov/hapi/info?id=SOLO_L3_RPW-BIA-DENSITY>
//!
//! This higher-cadence plasma-density lane complements MAG, merged hourly, and
//! SCPOT products for inner-heliosphere feature cubes.

use crate::{
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv},
    parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh},
};
use chrono::{Datelike, NaiveDate};
use csv::ReaderBuilder;
use std::{collections::BTreeMap, path::PathBuf};

const SOLO_RPW_BIA_DENSITY_HAPI_DATASET: &str = "SOLO_L3_RPW-BIA-DENSITY";

#[derive(Debug, Clone)]
pub struct SolarOrbiterRpwDensityRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub density_cm3: f64,
}

pub fn parse_solar_orbiter_rpw_density_csv(content: &str) -> Vec<SolarOrbiterRpwDensityRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let headers = match reader.headers() {
        Ok(headers) => headers.clone(),
        Err(_) => return Vec::new(),
    };
    let Some(density_col) = headers.iter().position(|value| value == "DENSITY") else {
        return Vec::new();
    };
    let mut hourly: BTreeMap<(u16, u16, u8), (f64, usize)> = BTreeMap::new();
    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else {
            continue;
        };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            continue;
        };
        let density = parse_hapi_spacephysics_f64_or_nan(record.get(density_col).unwrap_or(""));
        if !density.is_finite() {
            continue;
        }
        let entry = hourly.entry((year, doy, hour)).or_default();
        entry.0 += density;
        entry.1 += 1;
    }
    hourly
        .into_iter()
        .map(
            |((year, doy, hour), (sum, count))| SolarOrbiterRpwDensityRecord {
                year,
                doy,
                hour,
                density_cm3: sum / count.max(1) as f64,
            },
        )
        .collect()
}

pub fn parse_solar_orbiter_rpw_density_file(
    path: &std::path::Path,
) -> Result<Vec<SolarOrbiterRpwDensityRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|err| FetchError::Validation(format!("read error: {err}")))?;
    let rows = parse_solar_orbiter_rpw_density_csv(&content);
    if rows.is_empty() {
        return Err(FetchError::Validation(format!(
            "Solar Orbiter RPW density CSV {} had no finite hourly rows",
            path.display()
        )));
    }
    Ok(rows)
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_solar_orbiter_rpw_density_csv() {
        let csv = "Time,DENSITY\n\
2020-07-01T00:00:00Z,18.0\n\
2020-07-01T00:30:00Z,22.0\n";
        let rows = parse_solar_orbiter_rpw_density_csv(csv);
        assert_eq!(rows.len(), 1);
        assert!((rows[0].density_cm3 - 20.0).abs() < 1.0e-9);
    }
}
