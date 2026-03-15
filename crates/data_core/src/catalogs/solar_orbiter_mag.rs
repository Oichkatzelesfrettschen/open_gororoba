//! Solar Orbiter MAG RTN 1-minute parser via CDAWeb HAPI.
//!
//! This promotes a mission-native magnetic-field follow-on beyond the merged
//! hourly support lane so the feature-cube path can compare Solar Orbiter's
//! native MAG product against the merged feed.

use crate::{
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv},
    parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh},
};
use chrono::{Datelike, NaiveDate};
use csv::ReaderBuilder;
use std::{collections::BTreeMap, path::PathBuf};

const SOLO_MAG_1MIN_HAPI_DATASET: &str = "SOLO_L2_MAG-RTN-NORMAL-1-MINUTE";

#[derive(Debug, Clone)]
pub struct SolarOrbiterMagRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub br: f64,
    pub bt: f64,
    pub bn: f64,
    pub b_magnitude: f64,
}

#[derive(Default)]
struct MagAccumulator {
    br_sum: f64,
    bt_sum: f64,
    bn_sum: f64,
    count: usize,
}

pub fn parse_solar_orbiter_mag_hapi_csv(content: &str) -> Vec<SolarOrbiterMagRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let headers = match reader.headers() {
        Ok(headers) => headers.clone(),
        Err(_) => return Vec::new(),
    };
    let br_col = headers.iter().position(|value| value == "B_RTN_0");
    let bt_col = headers.iter().position(|value| value == "B_RTN_1");
    let bn_col = headers.iter().position(|value| value == "B_RTN_2");
    let (Some(br_col), Some(bt_col), Some(bn_col)) = (br_col, bt_col, bn_col) else {
        return Vec::new();
    };

    let mut hourly: BTreeMap<(u16, u16, u8), MagAccumulator> = BTreeMap::new();
    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else {
            continue;
        };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            continue;
        };
        let br = parse_hapi_spacephysics_f64_or_nan(record.get(br_col).unwrap_or(""));
        let bt = parse_hapi_spacephysics_f64_or_nan(record.get(bt_col).unwrap_or(""));
        let bn = parse_hapi_spacephysics_f64_or_nan(record.get(bn_col).unwrap_or(""));
        if !br.is_finite() || !bt.is_finite() || !bn.is_finite() {
            continue;
        }
        let entry = hourly.entry((year, doy, hour)).or_default();
        entry.br_sum += br;
        entry.bt_sum += bt;
        entry.bn_sum += bn;
        entry.count += 1;
    }

    hourly
        .into_iter()
        .map(|((year, doy, hour), acc)| {
            let count = acc.count.max(1) as f64;
            let br = acc.br_sum / count;
            let bt = acc.bt_sum / count;
            let bn = acc.bn_sum / count;
            SolarOrbiterMagRecord {
                year,
                doy,
                hour,
                br,
                bt,
                bn,
                b_magnitude: (br * br + bt * bt + bn * bn).sqrt(),
            }
        })
        .collect()
}

pub fn parse_solar_orbiter_mag_file(
    path: &std::path::Path,
) -> Result<Vec<SolarOrbiterMagRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|err| FetchError::Validation(format!("read error: {err}")))?;
    Ok(parse_solar_orbiter_mag_hapi_csv(&content))
}

pub struct SolarOrbiterMagProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for SolarOrbiterMagProvider {
    fn default() -> Self {
        Self {
            year_start: 2020,
            year_end: 2020,
        }
    }
}

impl DatasetProvider for SolarOrbiterMagProvider {
    fn name(&self) -> &str {
        "Solar Orbiter MAG RTN 1-minute"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config
            .output_dir
            .join("solar_orbiter")
            .join("mag_rtn_normal_1min");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let mut day = NaiveDate::from_ymd_opt(year as i32, 1, 1).ok_or_else(|| {
                FetchError::Validation(format!("invalid Solar Orbiter MAG year {year}"))
            })?;
            let end = NaiveDate::from_ymd_opt(year as i32 + 1, 1, 1).expect("valid next-year date");
            while day < end {
                let next_day = day.succ_opt().ok_or_else(|| {
                    FetchError::Validation(format!(
                        "failed to advance Solar Orbiter MAG day {}",
                        day
                    ))
                })?;
                let output = dir.join(format!(
                    "solo_l2_mag_rtn_normal_1minute_{}_{:02}_{:02}.csv",
                    day.year(),
                    day.month(),
                    day.day()
                ));
                if config.skip_existing && output.exists() {
                    day = next_day;
                    continue;
                }
                match download_hapi_csv(
                    SOLO_MAG_1MIN_HAPI_DATASET,
                    &format!("{}T00:00:00Z", day.format("%Y-%m-%d")),
                    &format!("{}T00:00:00Z", next_day.format("%Y-%m-%d")),
                    Some(&["Time", "B_RTN"]),
                ) {
                    Ok(body) => {
                        std::fs::write(&output, body)?;
                        log::info!("saved {}", output.display());
                    }
                    Err(err) => {
                        log::warn!(
                            "failed to download Solar Orbiter MAG {}-{:02}-{:02}: {}",
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
            .join("mag_rtn_normal_1min");
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
    fn test_parse_solar_orbiter_mag_hapi_csv() {
        let csv = "Time,B_RTN_0,B_RTN_1,B_RTN_2\n\
2020-06-16T00:00:29Z,1.0,-2.0,3.0\n\
2020-06-16T00:01:29Z,3.0,-4.0,5.0\n";
        let rows = parse_solar_orbiter_mag_hapi_csv(csv);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.year, 2020);
        assert_eq!(row.hour, 0);
        assert!((row.br - 2.0).abs() < 1.0e-9);
        assert!(row.b_magnitude > 5.0);
    }
}
