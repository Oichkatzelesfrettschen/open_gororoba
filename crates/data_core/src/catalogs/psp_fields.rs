//! Parker Solar Probe FIELDS MAG RTN high-cadence parser.
//!
//! The operational fleet lane already uses the merged hourly PSP dataset.
//! This module promotes the higher-cadence FIELDS magnetometer feed into the
//! Rust fetch/parse path so it can enrich feature cubes and cross-check the
//! merged hourly magnetic-field lane.

use crate::{
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv},
    parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh},
};
use chrono::{Datelike, NaiveDate};
use csv::ReaderBuilder;
use std::{collections::BTreeMap, path::PathBuf};

const PSP_FIELDS_HAPI_DATASET: &str = "PSP_FLD_L2_MAG_RTN";

#[derive(Debug, Clone)]
pub struct PspFieldsMagRecord {
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

pub fn parse_psp_fields_hapi_csv(content: &str) -> Vec<PspFieldsMagRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let mut hourly: BTreeMap<(u16, u16, u8), MagAccumulator> = BTreeMap::new();
    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else {
            continue;
        };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            continue;
        };
        let br = parse_hapi_spacephysics_f64_or_nan(record.get(1).unwrap_or(""));
        let bt = parse_hapi_spacephysics_f64_or_nan(record.get(2).unwrap_or(""));
        let bn = parse_hapi_spacephysics_f64_or_nan(record.get(3).unwrap_or(""));
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
            PspFieldsMagRecord {
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

pub fn parse_psp_fields_file(
    path: &std::path::Path,
) -> Result<Vec<PspFieldsMagRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|err| FetchError::Validation(format!("read error: {err}")))?;
    Ok(parse_psp_fields_hapi_csv(&content))
}

pub struct PspFieldsProvider {
    pub year_start: u16,
    pub year_end: u16,
    pub month_start: u8,
    pub month_end: u8,
}

impl Default for PspFieldsProvider {
    fn default() -> Self {
        Self {
            year_start: 2020,
            year_end: 2020,
            month_start: 1,
            month_end: 1,
        }
    }
}

impl DatasetProvider for PspFieldsProvider {
    fn name(&self) -> &str {
        "Parker Solar Probe FIELDS MAG RTN"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("psp").join("berkeley_fields");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            for month in self.month_start..=self.month_end {
                let month_start = NaiveDate::from_ymd_opt(year as i32, month as u32, 1)
                    .ok_or_else(|| FetchError::Validation(format!("invalid PSP month {year}-{month:02}")))?;
                let month_end = if month == 12 {
                    NaiveDate::from_ymd_opt(year as i32 + 1, 1, 1).expect("valid January date")
                } else {
                    NaiveDate::from_ymd_opt(year as i32, month as u32 + 1, 1)
                        .expect("valid next-month date")
                };
                let mut day = month_start;
                while day < month_end {
                    let next_day = day.succ_opt().ok_or_else(|| {
                        FetchError::Validation(format!("failed to advance PSP day {}", day))
                    })?;
                    let output = dir.join(format!(
                        "psp_fld_l2_mag_rtn_{}_{:02}_{:02}.csv",
                        day.year(),
                        day.month(),
                        day.day()
                    ));
                    if config.skip_existing && output.exists() {
                        day = next_day;
                        continue;
                    }
                    match download_hapi_csv(
                        PSP_FIELDS_HAPI_DATASET,
                        &format!("{}T00:00:00Z", day.format("%Y-%m-%d")),
                        &format!("{}T00:00:00Z", next_day.format("%Y-%m-%d")),
                        Some(&["Time", "psp_fld_l2_mag_RTN"]),
                    ) {
                        Ok(body) => {
                            std::fs::write(&output, body)?;
                            log::info!("saved {}", output.display());
                        }
                        Err(err) => {
                            log::warn!(
                                "failed to download PSP FIELDS {}-{:02}-{:02} via HAPI: {}",
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
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let dir = config.output_dir.join("psp").join("berkeley_fields");
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
    fn test_parse_psp_fields_hapi_csv() {
        let csv = "Time,psp_fld_l2_mag_RTN_0,psp_fld_l2_mag_RTN_1,psp_fld_l2_mag_RTN_2\n\
2020-01-01T00:00:00.009929600Z,-5.07e+00,7.14e+00,5.47e-01\n\
2020-01-01T00:00:00.119156352Z,-5.01e+00,7.02e+00,5.42e-01\n";
        let rows = parse_psp_fields_hapi_csv(csv);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.year, 2020);
        assert_eq!(row.hour, 0);
        assert!((row.br + 5.04).abs() < 0.05);
        assert!(row.b_magnitude > 8.5);
    }
}
