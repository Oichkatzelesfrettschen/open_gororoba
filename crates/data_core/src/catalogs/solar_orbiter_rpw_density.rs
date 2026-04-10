//! Solar Orbiter RPW BIA density parser via CDAWeb HAPI.
//!
//! Public official source:
//!   <https://cdaweb.gsfc.nasa.gov/hapi/info?id=SOLO_L3_RPW-BIA-DENSITY>
//!
//! This higher-cadence plasma-density lane complements MAG, merged hourly, and
//! SCPOT products for inner-heliosphere feature cubes.

use crate::{
    fetcher::FetchError,
    parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh},
};
use csv::ReaderBuilder;
use std::collections::BTreeMap;

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
