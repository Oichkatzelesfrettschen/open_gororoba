//! Parker Solar Probe FIELDS SQTN density/temperature parser via CDAWeb HAPI.
//!
//! Public official source:
//!   <https://cdaweb.gsfc.nasa.gov/hapi/info?id=PSP_FLD_L3_SQTN_RFS_V1V2>
//!
//! This product provides electron-density and core-temperature estimates and
//! complements the merged hourly and FIELDS MAG lanes for later mission windows.

use crate::{
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv},
    parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh},
};
use chrono::{Datelike, NaiveDate};
use csv::ReaderBuilder;
use std::{collections::BTreeMap, path::PathBuf};

const PSP_SQTN_DATASET: &str = "PSP_FLD_L3_SQTN_RFS_V1V2";
const EV_TO_K: f64 = 11_604.518_121_550_08;

#[derive(Debug, Clone)]
pub struct PspSqtnRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub electron_density_cm3: f64,
    pub electron_core_temperature_k: f64,
}

#[derive(Default)]
struct SqtnHourAccumulator {
    density_sum: f64,
    density_count: usize,
    temperature_sum: f64,
    temperature_count: usize,
}

pub fn parse_psp_sqtn_csv(content: &str) -> Vec<PspSqtnRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let headers = match reader.headers() {
        Ok(headers) => headers.clone(),
        Err(_) => return Vec::new(),
    };
    let density_col = headers.iter().position(|value| value == "electron_density");
    let quality_col = headers
        .iter()
        .position(|value| value == "density_quality_flag");
    let temp_col = headers
        .iter()
        .position(|value| value == "electron_core_temperature");
    let (Some(density_col), Some(quality_col), Some(temp_col)) =
        (density_col, quality_col, temp_col)
    else {
        return Vec::new();
    };
    let mut hourly: BTreeMap<(u16, u16, u8), SqtnHourAccumulator> = BTreeMap::new();
    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else {
            continue;
        };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            continue;
        };
        let quality = record
            .get(quality_col)
            .and_then(|value| value.parse::<i64>().ok())
            .unwrap_or(0);
        if quality <= 0 {
            continue;
        }
        let density = parse_hapi_spacephysics_f64_or_nan(record.get(density_col).unwrap_or(""));
        let temp_ev = parse_hapi_spacephysics_f64_or_nan(record.get(temp_col).unwrap_or(""));
        let entry = hourly.entry((year, doy, hour)).or_default();
        if density.is_finite() {
            entry.density_sum += density;
            entry.density_count += 1;
        }
        if temp_ev.is_finite() {
            entry.temperature_sum += temp_ev * EV_TO_K;
            entry.temperature_count += 1;
        }
    }
    hourly
        .into_iter()
        .filter_map(|((year, doy, hour), acc)| {
            if acc.density_count == 0 && acc.temperature_count == 0 {
                return None;
            }
            Some(PspSqtnRecord {
                year,
                doy,
                hour,
                electron_density_cm3: if acc.density_count > 0 {
                    acc.density_sum / acc.density_count as f64
                } else {
                    f64::NAN
                },
                electron_core_temperature_k: if acc.temperature_count > 0 {
                    acc.temperature_sum / acc.temperature_count as f64
                } else {
                    f64::NAN
                },
            })
        })
        .collect()
}

pub fn parse_psp_sqtn_file(path: &std::path::Path) -> Result<Vec<PspSqtnRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|err| FetchError::Validation(format!("read error: {err}")))?;
    let rows = parse_psp_sqtn_csv(&content);
    if rows.is_empty() {
        return Err(FetchError::Validation(format!(
            "PSP SQTN CSV {} had no finite hourly rows",
            path.display()
        )));
    }
    Ok(rows)
}

pub struct PspSqtnProvider {
    pub year_start: u16,
    pub year_end: u16,
    pub month_start: u8,
    pub month_end: u8,
}

impl Default for PspSqtnProvider {
    fn default() -> Self {
        Self {
            year_start: 2025,
            year_end: 2025,
            month_start: 9,
            month_end: 9,
        }
    }
}

impl DatasetProvider for PspSqtnProvider {
    fn name(&self) -> &str {
        "Parker Solar Probe SQTN RFS V1V2"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let root = config.output_dir.join("psp").join("sqtn_rfs_v1v2");
        std::fs::create_dir_all(&root)?;
        for year in self.year_start..=self.year_end {
            let year_dir = root.join(year.to_string());
            std::fs::create_dir_all(&year_dir)?;
            for month in self.month_start..=self.month_end {
                let mut day =
                    NaiveDate::from_ymd_opt(year as i32, month as u32, 1).ok_or_else(|| {
                        FetchError::Validation(format!("invalid PSP SQTN month {year}-{month:02}"))
                    })?;
                let end = if month == 12 {
                    NaiveDate::from_ymd_opt(year as i32 + 1, 1, 1).expect("valid January date")
                } else {
                    NaiveDate::from_ymd_opt(year as i32, month as u32 + 1, 1)
                        .expect("valid next-month date")
                };
                while day < end {
                    let next_day = day
                        .succ_opt()
                        .ok_or_else(|| FetchError::Validation(format!("advance {day}")))?;
                    let output = year_dir.join(format!(
                        "psp_fld_l3_sqtn_rfs_v1v2_{}{:02}{:02}.csv",
                        day.year(),
                        day.month(),
                        day.day()
                    ));
                    if config.skip_existing && output.exists() {
                        day = next_day;
                        continue;
                    }
                    match download_hapi_csv(
                        PSP_SQTN_DATASET,
                        &format!("{}T00:00:00Z", day.format("%Y-%m-%d")),
                        &format!("{}T00:00:00Z", next_day.format("%Y-%m-%d")),
                        Some(&[
                            "Time",
                            "electron_density",
                            "density_quality_flag",
                            "electron_core_temperature",
                        ]),
                    ) {
                        Ok(body) => {
                            std::fs::write(&output, body)?;
                            log::info!("saved {}", output.display());
                        }
                        Err(err) => {
                            log::warn!(
                                "failed to download PSP SQTN for {}-{:02}-{:02}: {}",
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
        Ok(root)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let root = config.output_dir.join("psp").join("sqtn_rfs_v1v2");
        std::fs::read_dir(root)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| entry.path().is_dir())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_psp_sqtn_csv() {
        let csv = "Time,electron_density,density_quality_flag,electron_core_temperature\n\
2025-09-23T22:00:00Z,120.0,2,20.0\n\
2025-09-23T22:10:00Z,180.0,2,10.0\n\
2025-09-23T22:20:00Z,-1.0E31,0,-1.0E31\n";
        let rows = parse_psp_sqtn_csv(csv);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert!((row.electron_density_cm3 - 150.0).abs() < 1.0e-9);
        assert!(row.electron_core_temperature_k > 100_000.0);
    }
}
