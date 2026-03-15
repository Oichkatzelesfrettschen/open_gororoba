//! Solar Orbiter SWA-PAS ground-moment parser via CDAWeb HAPI.
//!
//! The merged hourly Solar Orbiter lane already exists, but this module brings
//! the higher-cadence proton moments into the Rust execution path so the
//! feature-cube lane can preserve a second, mission-native plasma product.

use crate::{
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv},
    parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh},
};
use csv::ReaderBuilder;
use std::{collections::BTreeMap, path::PathBuf};

const SOLO_SWA_PAS_HAPI_DATASET: &str = "SOLO_L2_SWA-PAS-GRND-MOM";

#[derive(Debug, Clone)]
pub struct SolarOrbiterSwaRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub proton_density: f64,
    pub vr: f64,
    pub vt: f64,
    pub vn: f64,
    pub bulk_speed: f64,
    pub proton_temperature: f64,
}

#[derive(Default)]
struct SwaAccumulator {
    density_sum: f64,
    vr_sum: f64,
    vt_sum: f64,
    vn_sum: f64,
    temp_sum: f64,
    count: usize,
}

pub fn parse_solar_orbiter_swa_hapi_csv(content: &str) -> Vec<SolarOrbiterSwaRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let mut hourly: BTreeMap<(u16, u16, u8), SwaAccumulator> = BTreeMap::new();
    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else {
            continue;
        };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            continue;
        };
        let density = parse_hapi_spacephysics_f64_or_nan(record.get(1).unwrap_or(""));
        let vr = parse_hapi_spacephysics_f64_or_nan(record.get(2).unwrap_or(""));
        let vt = parse_hapi_spacephysics_f64_or_nan(record.get(3).unwrap_or(""));
        let vn = parse_hapi_spacephysics_f64_or_nan(record.get(4).unwrap_or(""));
        let temp = parse_hapi_spacephysics_f64_or_nan(record.get(5).unwrap_or(""));
        if !density.is_finite() || !vr.is_finite() || !vt.is_finite() || !vn.is_finite() {
            continue;
        }
        let entry = hourly.entry((year, doy, hour)).or_default();
        entry.density_sum += density;
        entry.vr_sum += vr;
        entry.vt_sum += vt;
        entry.vn_sum += vn;
        if temp.is_finite() {
            entry.temp_sum += temp;
        }
        entry.count += 1;
    }
    hourly
        .into_iter()
        .map(|((year, doy, hour), acc)| {
            let count = acc.count.max(1) as f64;
            let vr = acc.vr_sum / count;
            let vt = acc.vt_sum / count;
            let vn = acc.vn_sum / count;
            SolarOrbiterSwaRecord {
                year,
                doy,
                hour,
                proton_density: acc.density_sum / count,
                vr,
                vt,
                vn,
                bulk_speed: (vr * vr + vt * vt + vn * vn).sqrt(),
                proton_temperature: if acc.temp_sum > 0.0 {
                    acc.temp_sum / count
                } else {
                    f64::NAN
                },
            }
        })
        .collect()
}

pub fn parse_solar_orbiter_swa_file(
    path: &std::path::Path,
) -> Result<Vec<SolarOrbiterSwaRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|err| FetchError::Validation(format!("read error: {err}")))?;
    Ok(parse_solar_orbiter_swa_hapi_csv(&content))
}

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
                    log::warn!("failed to download Solar Orbiter SWA {} via HAPI: {}", year, err);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_solar_orbiter_swa_hapi_csv() {
        let csv = "Time,N,V_RTN_0,V_RTN_1,V_RTN_2,T\n\
2020-07-08T00:00:02.517631000Z,1.05e+01,3.12e+02,2.49e+00,-1.96e+01,5.77e+00\n\
2020-07-08T00:00:06.517637000Z,1.08e+01,3.13e+02,1.80e+00,-1.96e+01,5.78e+00\n";
        let rows = parse_solar_orbiter_swa_hapi_csv(csv);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.year, 2020);
        assert_eq!(row.hour, 0);
        assert!((row.proton_density - 10.65).abs() < 0.05);
        assert!(row.bulk_speed > 300.0);
    }
}
