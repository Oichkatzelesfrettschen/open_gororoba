//! Solar Orbiter RPW BIA spacecraft-potential parser via direct CDAWeb CDFs.
//!
//! This follow-on product complements the merged hourly and SWA/MAG lanes with
//! higher-cadence electrostatic/plasma-environment support from the official
//! public RPW BIA spacecraft-potential product family.

use crate::{
    cdf_support::{cdf_scalar_f64_rows, cdf_type_to_ydh, cdf_variable_rows, read_cdf_file},
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

const SOLO_RPW_BIA_SCPOT_ROOT: &str =
    "https://cdaweb.gsfc.nasa.gov/pub/data/solar-orbiter/rpw/science/l3/bia-scpot-10-seconds/";
const SOLO_RPW_BIA_SCPOT_HAPI_DATASET: &str = "SOLO_L3_RPW-BIA-SCPOT-10-SECONDS";

#[derive(Debug, Clone)]
pub struct SolarOrbiterRpwRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub scpot: f64,
    pub psp: f64,
}

#[derive(Default)]
struct ScalarAccumulator {
    scpot_sum: f64,
    scpot_count: usize,
    psp_sum: f64,
    psp_count: usize,
}

fn parse_solar_orbiter_rpw_hapi_csv(content: &str) -> Vec<SolarOrbiterRpwRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let headers = match reader.headers() {
        Ok(headers) => headers.clone(),
        Err(_) => return Vec::new(),
    };
    let scpot_col = headers.iter().position(|value| value == "SCPOT");
    let psp_col = headers.iter().position(|value| value == "PSP");
    let (Some(scpot_col), Some(psp_col)) = (scpot_col, psp_col) else {
        return Vec::new();
    };
    let mut hourly: BTreeMap<(u16, u16, u8), ScalarAccumulator> = BTreeMap::new();
    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else {
            continue;
        };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            continue;
        };
        let scpot = parse_hapi_spacephysics_f64_or_nan(record.get(scpot_col).unwrap_or(""));
        let psp = parse_hapi_spacephysics_f64_or_nan(record.get(psp_col).unwrap_or(""));
        let entry = hourly.entry((year, doy, hour)).or_default();
        if scpot.is_finite() {
            entry.scpot_sum += scpot;
            entry.scpot_count += 1;
        }
        if psp.is_finite() {
            entry.psp_sum += psp;
            entry.psp_count += 1;
        }
    }
    hourly
        .into_iter()
        .filter_map(|((year, doy, hour), acc)| {
            let scpot = if acc.scpot_count > 0 {
                acc.scpot_sum / acc.scpot_count as f64
            } else {
                f64::NAN
            };
            let psp = if acc.psp_count > 0 {
                acc.psp_sum / acc.psp_count as f64
            } else {
                f64::NAN
            };
            if !scpot.is_finite() && !psp.is_finite() {
                return None;
            }
            Some(SolarOrbiterRpwRecord {
                year,
                doy,
                hour,
                scpot,
                psp,
            })
        })
        .collect()
}

pub fn parse_solar_orbiter_rpw_file(path: &Path) -> Result<Vec<SolarOrbiterRpwRecord>, FetchError> {
    if path.extension().and_then(|value| value.to_str()) == Some("csv") {
        let content = std::fs::read_to_string(path)
            .map_err(|err| FetchError::Validation(format!("read error: {err}")))?;
        let rows = parse_solar_orbiter_rpw_hapi_csv(&content);
        if rows.is_empty() {
            return Err(FetchError::Validation(format!(
                "Solar Orbiter RPW HAPI CSV {} had no finite hourly rows",
                path.display()
            )));
        }
        return Ok(rows);
    }
    let cdf = read_cdf_file(path)?;
    let epoch_rows = cdf_variable_rows(&cdf, "Epoch")?;
    let scpot_rows = cdf_scalar_f64_rows(&cdf, "SCPOT")?;
    let psp_rows =
        cdf_scalar_f64_rows(&cdf, "PSP").or_else(|_| cdf_scalar_f64_rows(&cdf, "PSPSTD"))?;
    let row_count = epoch_rows.len().min(scpot_rows.len()).min(psp_rows.len());
    if row_count == 0 {
        return Err(FetchError::Validation(format!(
            "Solar Orbiter RPW {} yielded zero rows",
            path.display()
        )));
    }
    let mut hourly: BTreeMap<(u16, u16, u8), ScalarAccumulator> = BTreeMap::new();
    for idx in 0..row_count {
        let Some((year, doy, hour)) = epoch_rows[idx].first().and_then(cdf_type_to_ydh) else {
            continue;
        };
        let scpot = scpot_rows[idx];
        let psp = psp_rows[idx];
        let entry = hourly.entry((year, doy, hour)).or_default();
        if scpot.is_finite() {
            entry.scpot_sum += scpot;
            entry.scpot_count += 1;
        }
        if psp.is_finite() {
            entry.psp_sum += psp;
            entry.psp_count += 1;
        }
    }
    let rows = hourly
        .into_iter()
        .map(|((year, doy, hour), acc)| SolarOrbiterRpwRecord {
            year,
            doy,
            hour,
            scpot: if acc.scpot_count > 0 {
                acc.scpot_sum / acc.scpot_count as f64
            } else {
                f64::NAN
            },
            psp: if acc.psp_count > 0 {
                acc.psp_sum / acc.psp_count as f64
            } else {
                f64::NAN
            },
        })
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return Err(FetchError::Validation(format!(
            "Solar Orbiter RPW {} had no finite hourly rows",
            path.display()
        )));
    }
    Ok(rows)
}

pub struct SolarOrbiterRpwProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for SolarOrbiterRpwProvider {
    fn default() -> Self {
        Self {
            year_start: 2020,
            year_end: 2020,
        }
    }
}

impl DatasetProvider for SolarOrbiterRpwProvider {
    fn name(&self) -> &str {
        "Solar Orbiter RPW BIA SCPOT 10-second"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let root = config
            .output_dir
            .join("solar_orbiter")
            .join("rpw_bia_scpot_10s");
        std::fs::create_dir_all(&root)?;

        for year in self.year_start..=self.year_end {
            let year_url = format!("{SOLO_RPW_BIA_SCPOT_ROOT}{year}/");
            let year_dir = root.join(year.to_string());
            std::fs::create_dir_all(&year_dir)?;
            for entry in directory_entries(&year_url)? {
                if !entry.ends_with(".cdf") {
                    continue;
                }
                let output = year_dir.join(&entry);
                if config.skip_existing && output.exists() {
                    continue;
                }
                let url = format!("{year_url}{entry}");
                download_to_file(&url, &output)?;
                log::info!("saved {}", output.display());
            }
            let Some(mut day) = NaiveDate::from_ymd_opt(year as i32, 6, 15) else {
                return Err(FetchError::Validation(format!(
                    "invalid Solar Orbiter RPW start date for {year}"
                )));
            };
            let Some(year_end) = NaiveDate::from_ymd_opt(year as i32 + 1, 1, 1) else {
                return Err(FetchError::Validation(format!(
                    "invalid Solar Orbiter RPW end date for {year}"
                )));
            };
            while day < year_end {
                let next_day = day.succ_opt().ok_or_else(|| {
                    FetchError::Validation(format!("failed to advance Solar Orbiter RPW day {day}"))
                })?;
                let csv_output = year_dir.join(format!(
                    "solo_l3_rpw-bia-scpot-10-seconds_{}{:02}{:02}.csv",
                    day.year(),
                    day.month(),
                    day.day()
                ));
                if !(config.skip_existing && csv_output.exists()) {
                    match download_hapi_csv(
                        SOLO_RPW_BIA_SCPOT_HAPI_DATASET,
                        &format!("{}T00:00:00Z", day.format("%Y-%m-%d")),
                        &format!("{}T00:00:00Z", next_day.format("%Y-%m-%d")),
                        Some(&["Time", "SCPOT", "PSP"]),
                    ) {
                        Ok(body) => {
                            std::fs::write(&csv_output, body)?;
                            log::info!("saved {}", csv_output.display());
                        }
                        Err(err) => {
                            log::warn!(
                                "failed to download Solar Orbiter RPW HAPI CSV for {}-{:02}-{:02}: {}",
                                day.year(),
                                day.month(),
                                day.day(),
                                err
                            );
                        }
                    }
                }
                day = next_day;
            }
        }

        Ok(root)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        has_matching_files(
            &config
                .output_dir
                .join("solar_orbiter")
                .join("rpw_bia_scpot_10s"),
            ".csv",
        )
    }
}

fn directory_entries(url: &str) -> Result<Vec<String>, FetchError> {
    let html = download_to_string(url)?;
    let regex = Regex::new(r#"href="([^"]+)""#).map_err(|err| {
        FetchError::Validation(format!("invalid Solar Orbiter RPW directory regex: {err}"))
    })?;
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
