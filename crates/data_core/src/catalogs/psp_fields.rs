//! Parker Solar Probe FIELDS MAG RTN high-cadence parser.
//!
//! The operational fleet lane already uses the merged hourly PSP dataset.
//! This module promotes the higher-cadence FIELDS magnetometer feed into the
//! Rust fetch/parse path so it can enrich feature cubes and cross-check the
//! merged hourly magnetic-field lane.

use crate::{
    cdf_support::{cdf_type_to_ydh, cdf_variable_rows, cdf_vector_f64_rows, read_cdf_file},
    fetcher::{
        DatasetProvider, FetchConfig, FetchError, download_hapi_csv, download_to_file,
        download_to_string,
    },
    parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh},
};
use chrono::{Datelike, NaiveDate};
use csv::ReaderBuilder;
use regex::Regex;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

const PSP_FIELDS_CDF_ROOT: &str =
    "https://cdaweb.gsfc.nasa.gov/pub/data/psp/fields/l2/mag_rtn_1min/";
const PSP_FIELDS_1MIN_HAPI_DATASET: &str = "PSP_FLD_L2_MAG_RTN_1MIN";

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
    let headers = match reader.headers() {
        Ok(headers) => headers.clone(),
        Err(_) => return Vec::new(),
    };
    let br_col = headers
        .iter()
        .position(|value| matches!(value, "psp_fld_l2_mag_RTN_0" | "psp_fld_l2_mag_RTN_1min_0"));
    let bt_col = headers
        .iter()
        .position(|value| matches!(value, "psp_fld_l2_mag_RTN_1" | "psp_fld_l2_mag_RTN_1min_1"));
    let bn_col = headers
        .iter()
        .position(|value| matches!(value, "psp_fld_l2_mag_RTN_2" | "psp_fld_l2_mag_RTN_1min_2"));
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

pub fn parse_psp_fields_cdf_file(path: &Path) -> Result<Vec<PspFieldsMagRecord>, FetchError> {
    let cdf = read_cdf_file(path)?;
    let epoch_rows = cdf_variable_rows(&cdf, "Epoch")
        .or_else(|_| cdf_variable_rows(&cdf, "epoch_mag_RTN_1min"))?;
    let b_rows = cdf_vector_f64_rows(&cdf, "B_RTN")
        .or_else(|_| cdf_vector_f64_rows(&cdf, "psp_fld_l2_mag_RTN_1min"))?;
    let row_count = epoch_rows.len().min(b_rows.len());
    if row_count == 0 {
        return Err(FetchError::Validation(format!(
            "PSP FIELDS CDF {} yielded zero rows",
            path.display()
        )));
    }
    let mut hourly: BTreeMap<(u16, u16, u8), MagAccumulator> = BTreeMap::new();
    for idx in 0..row_count {
        let Some((year, doy, hour)) = epoch_rows[idx].first().and_then(cdf_type_to_ydh) else {
            continue;
        };
        let vector = &b_rows[idx];
        let br = vector.first().copied().unwrap_or(f64::NAN);
        let bt = vector.get(1).copied().unwrap_or(f64::NAN);
        let bn = vector.get(2).copied().unwrap_or(f64::NAN);
        if !br.is_finite() || !bt.is_finite() || !bn.is_finite() {
            continue;
        }
        let entry = hourly.entry((year, doy, hour)).or_default();
        entry.br_sum += br;
        entry.bt_sum += bt;
        entry.bn_sum += bn;
        entry.count += 1;
    }
    let rows = hourly
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
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return Err(FetchError::Validation(format!(
            "PSP FIELDS CDF {} had no finite magnetic rows",
            path.display()
        )));
    }
    Ok(rows)
}

pub fn parse_psp_fields_file(
    path: &std::path::Path,
) -> Result<Vec<PspFieldsMagRecord>, FetchError> {
    if path.extension().and_then(|value| value.to_str()) == Some("cdf") {
        return parse_psp_fields_cdf_file(path);
    }
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
            month_end: 2,
        }
    }
}

impl DatasetProvider for PspFieldsProvider {
    fn name(&self) -> &str {
        "Parker Solar Probe FIELDS MAG RTN"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let root = config.output_dir.join("psp").join("fields_l2_mag_rtn_1min");
        std::fs::create_dir_all(&root)?;

        for year in self.year_start..=self.year_end {
            let year_url = format!("{PSP_FIELDS_CDF_ROOT}{year}/");
            let year_dir = root.join(year.to_string());
            std::fs::create_dir_all(&year_dir)?;
            let entries = directory_entries(&year_url)?;
            for month in self.month_start..=self.month_end {
                let month_start = NaiveDate::from_ymd_opt(year as i32, month as u32, 1)
                    .ok_or_else(|| {
                        FetchError::Validation(format!("invalid PSP month {year}-{month:02}"))
                    })?;
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
                    let output = year_dir.join(format!(
                        "psp_fld_l2_mag_rtn_1min_{}{:02}{:02}.cdf",
                        day.year(),
                        day.month(),
                        day.day()
                    ));
                    let entry = format!(
                        "psp_fld_l2_mag_rtn_1min_{}{:02}{:02}",
                        day.year(),
                        day.month(),
                        day.day()
                    );
                    let csv_output = year_dir.join(format!(
                        "psp_fld_l2_mag_rtn_1min_{}{:02}{:02}.csv",
                        day.year(),
                        day.month(),
                        day.day()
                    ));
                    let have_cdf = output.exists();
                    let have_csv = csv_output.exists();
                    if config.skip_existing && have_cdf && have_csv {
                        day = next_day;
                        continue;
                    }
                    if let Some(file_name) = entries
                        .iter()
                        .find(|value| value.starts_with(&entry) && value.ends_with(".cdf"))
                        .filter(|_| !(config.skip_existing && have_cdf))
                    {
                        let url = format!("{year_url}{file_name}");
                        download_to_file(&url, &output)?;
                        log::info!("saved {}", output.display());
                    }
                    if !(config.skip_existing && have_csv) {
                        match download_hapi_csv(
                            PSP_FIELDS_1MIN_HAPI_DATASET,
                            &format!("{}T00:00:00Z", day.format("%Y-%m-%d")),
                            &format!("{}T00:00:00Z", next_day.format("%Y-%m-%d")),
                            Some(&["Time", "psp_fld_l2_mag_RTN_1min"]),
                        ) {
                            Ok(body) => {
                                std::fs::write(&csv_output, body)?;
                                log::info!("saved {}", csv_output.display());
                            }
                            Err(err) => {
                                log::warn!(
                                    "failed to download PSP FIELDS HAPI CSV for {}-{:02}-{:02}: {}",
                                    day.year(),
                                    day.month(),
                                    day.day(),
                                    err
                                );
                            }
                        }
                    }
                    if !output.exists() && !csv_output.exists() {
                        log::warn!(
                            "no PSP FIELDS direct CDF or HAPI CSV staged for {}-{:02}-{:02}",
                            day.year(),
                            day.month(),
                            day.day()
                        );
                    } else {
                        log::debug!(
                            "PSP FIELDS {}-{:02}-{:02} has at least one staged representation",
                            day.year(),
                            day.month(),
                            day.day()
                        );
                    }
                    day = next_day;
                }
            }
        }

        Ok(root)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let cdf_dir = config.output_dir.join("psp").join("fields_l2_mag_rtn_1min");
        if has_matching_files(&cdf_dir, ".cdf") {
            return true;
        }
        let dir = config.output_dir.join("psp").join("berkeley_fields");
        std::fs::read_dir(dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| entry.path().extension().and_then(|value| value.to_str()) == Some("csv"))
    }
}

fn directory_entries(url: &str) -> Result<Vec<String>, FetchError> {
    let html = download_to_string(url)?;
    let regex = Regex::new(r#"href="([^"]+)""#)
        .map_err(|err| FetchError::Validation(format!("invalid PSP directory regex: {err}")))?;
    let mut entries = Vec::new();
    for capture in regex.captures_iter(&html) {
        let Some(href) = capture.get(1).map(|value| value.as_str()) else {
            continue;
        };
        if href.starts_with('/') || href.starts_with('?') || href == "Parent Directory" {
            continue;
        }
        entries.push(href.to_string());
    }
    entries.sort();
    entries.dedup();
    Ok(entries)
}

fn has_matching_files(dir: &Path, suffix: &str) -> bool {
    if std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .any(|entry| {
            entry.path().extension().and_then(|value| value.to_str())
                == Some(suffix.trim_start_matches('.'))
        })
    {
        return true;
    }
    std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .any(|entry| match entry.file_type() {
            Ok(file_type) if file_type.is_dir() => has_matching_files(&entry.path(), suffix),
            _ => false,
        })
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

    #[test]
    fn test_parse_psp_fields_1min_csv() {
        let csv = "Time,psp_fld_l2_mag_RTN_1min_0,psp_fld_l2_mag_RTN_1min_1,psp_fld_l2_mag_RTN_1min_2\n\
2020-02-01T00:00:30Z,-4.67e+01,5.02e+01,-3.30e+00\n\
2020-02-01T00:01:30Z,-4.56e+01,4.81e+01,2.27e+00\n";
        let rows = parse_psp_fields_hapi_csv(csv);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.year, 2020);
        assert_eq!(row.doy, 32);
        assert_eq!(row.hour, 0);
        assert!(row.b_magnitude > 60.0);
    }
}
