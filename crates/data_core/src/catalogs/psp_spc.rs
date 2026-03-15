//! Parker Solar Probe SWEAP SPC L3 ion-moment parser.
//!
//! Official public surfaces:
//!   - Daily CDF archive:
//!     <https://cdaweb.gsfc.nasa.gov/pub/data/psp/sweap/spc/l3/l3i/>
//!   - Compact HAPI subset:
//!     <https://cdaweb.gsfc.nasa.gov/hapi/info?id=PSP_SWP_SPC_L3I>
//!
//! The executed Rust lane stages daily CDFs for provenance and uses compact
//! HAPI CSV mirrors for the feature-cube path.

use crate::{
    cdf_support::filename_date_yyyymmdd,
    fetcher::{
        DatasetProvider, FetchConfig, FetchError, download_hapi_csv, download_to_file,
        download_to_string,
    },
    parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh},
};
use chrono::NaiveDate;
use csv::ReaderBuilder;
use regex::Regex;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

const PSP_SPC_L3I_ROOT: &str = "https://cdaweb.gsfc.nasa.gov/pub/data/psp/sweap/spc/l3/l3i/";
const PSP_SPC_L3I_HAPI_DATASET: &str = "PSP_SWP_SPC_L3I";
const PROTON_MASS_KG: f64 = 1.672_621_923_69e-27;
const BOLTZMANN_J_PER_K: f64 = 1.380_649e-23;

#[derive(Debug, Clone)]
pub struct PspSpcL3iRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub density_cm3: f64,
    pub speed_kms: f64,
    pub temperature_k: f64,
}

#[derive(Default)]
struct SpcAccumulator {
    density_sum: f64,
    density_count: usize,
    speed_sum: f64,
    speed_count: usize,
    temp_sum: f64,
    temp_count: usize,
}

fn thermal_speed_to_temperature_k(speed_kms: f64) -> f64 {
    if !speed_kms.is_finite() {
        return f64::NAN;
    }
    let speed_ms = speed_kms * 1_000.0;
    PROTON_MASS_KG * speed_ms * speed_ms / (2.0 * BOLTZMANN_J_PER_K)
}

pub fn parse_psp_spc_l3i_csv(content: &str) -> Vec<PspSpcL3iRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let headers = match reader.headers() {
        Ok(headers) => headers.clone(),
        Err(_) => return Vec::new(),
    };
    let quality_col = headers.iter().position(|value| value == "general_flag");
    let density_col = headers.iter().position(|value| value == "np_moment_gd");
    let thermal_col = headers.iter().position(|value| value == "wp_moment_gd");
    let vector_cols = headers
        .iter()
        .enumerate()
        .filter_map(|(idx, value)| value.starts_with("vp_moment_RTN_gd_").then_some(idx))
        .take(3)
        .collect::<Vec<_>>();
    let (Some(quality_col), Some(density_col), Some(thermal_col)) =
        (quality_col, density_col, thermal_col)
    else {
        return Vec::new();
    };
    if vector_cols.len() != 3 {
        return Vec::new();
    }

    let mut hourly: BTreeMap<(u16, u16, u8), SpcAccumulator> = BTreeMap::new();
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
            .unwrap_or(-1);
        if quality != 0 {
            continue;
        }
        let density = parse_hapi_spacephysics_f64_or_nan(record.get(density_col).unwrap_or(""));
        let thermal_speed =
            parse_hapi_spacephysics_f64_or_nan(record.get(thermal_col).unwrap_or(""));
        let velocity = vector_cols
            .iter()
            .filter_map(|idx| record.get(*idx))
            .map(parse_hapi_spacephysics_f64_or_nan)
            .collect::<Vec<_>>();
        let speed = if velocity.len() == 3 && velocity.iter().all(|value| value.is_finite()) {
            let vr = velocity[0];
            let vt = velocity[1];
            let vn = velocity[2];
            (vr * vr + vt * vt + vn * vn).sqrt()
        } else {
            f64::NAN
        };
        let temperature = thermal_speed_to_temperature_k(thermal_speed);
        let entry = hourly.entry((year, doy, hour)).or_default();
        if density.is_finite() {
            entry.density_sum += density;
            entry.density_count += 1;
        }
        if speed.is_finite() {
            entry.speed_sum += speed;
            entry.speed_count += 1;
        }
        if temperature.is_finite() {
            entry.temp_sum += temperature;
            entry.temp_count += 1;
        }
    }

    hourly
        .into_iter()
        .filter_map(|((year, doy, hour), acc)| {
            if acc.density_count == 0 && acc.speed_count == 0 && acc.temp_count == 0 {
                return None;
            }
            Some(PspSpcL3iRecord {
                year,
                doy,
                hour,
                density_cm3: if acc.density_count > 0 {
                    acc.density_sum / acc.density_count as f64
                } else {
                    f64::NAN
                },
                speed_kms: if acc.speed_count > 0 {
                    acc.speed_sum / acc.speed_count as f64
                } else {
                    f64::NAN
                },
                temperature_k: if acc.temp_count > 0 {
                    acc.temp_sum / acc.temp_count as f64
                } else {
                    f64::NAN
                },
            })
        })
        .collect()
}

pub fn parse_psp_spc_l3i_file(path: &Path) -> Result<Vec<PspSpcL3iRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|err| FetchError::Validation(format!("read error: {err}")))?;
    let rows = parse_psp_spc_l3i_csv(&content);
    if rows.is_empty() {
        return Err(FetchError::Validation(format!(
            "PSP SPC L3I CSV {} had no finite hourly rows",
            path.display()
        )));
    }
    Ok(rows)
}

pub struct PspSpcL3iProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for PspSpcL3iProvider {
    fn default() -> Self {
        Self {
            year_start: 2025,
            year_end: 2025,
        }
    }
}

impl DatasetProvider for PspSpcL3iProvider {
    fn name(&self) -> &str {
        "Parker Solar Probe SWEAP SPC L3 ion moments"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let root = config.output_dir.join("psp").join("sweap_spc_l3i");
        std::fs::create_dir_all(&root)?;
        for year in self.year_start..=self.year_end {
            let year_url = format!("{PSP_SPC_L3I_ROOT}{year}/");
            let year_dir = root.join(year.to_string());
            std::fs::create_dir_all(&year_dir)?;
            for entry in directory_entries(&year_url)? {
                if !entry.ends_with(".cdf") {
                    continue;
                }
                let output = year_dir.join(&entry);
                if !(config.skip_existing && output.exists()) {
                    let url = format!("{year_url}{entry}");
                    download_to_file(&url, &output)?;
                    log::info!("saved {}", output.display());
                }
                let Some((entry_year, month, day)) = filename_date_yyyymmdd(Path::new(&entry))
                else {
                    continue;
                };
                let Some(day_start) = NaiveDate::from_ymd_opt(
                    i32::from(entry_year),
                    u32::from(month),
                    u32::from(day),
                ) else {
                    continue;
                };
                let next_day = day_start
                    .succ_opt()
                    .ok_or_else(|| FetchError::Validation(format!("advance {day_start}")))?;
                let csv_output = year_dir.join(format!(
                    "psp_swp_spc_l3i_{entry_year:04}{month:02}{day:02}.csv"
                ));
                if config.skip_existing && csv_output.exists() {
                    continue;
                }
                match download_hapi_csv(
                    PSP_SPC_L3I_HAPI_DATASET,
                    &format!("{}T00:00:00Z", day_start.format("%Y-%m-%d")),
                    &format!("{}T00:00:00Z", next_day.format("%Y-%m-%d")),
                    Some(&[
                        "Time",
                        "general_flag",
                        "vp_moment_RTN_gd",
                        "np_moment_gd",
                        "wp_moment_gd",
                    ]),
                ) {
                    Ok(body) => {
                        std::fs::write(&csv_output, body)?;
                        log::info!("saved {}", csv_output.display());
                    }
                    Err(err) => {
                        log::warn!(
                            "failed to download PSP SPC L3I for {}-{:02}-{:02}: {}",
                            entry_year,
                            month,
                            day,
                            err
                        );
                    }
                }
            }
        }
        Ok(root)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let root = config.output_dir.join("psp").join("sweap_spc_l3i");
        std::fs::read_dir(root)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| entry.path().is_dir())
    }
}

fn directory_entries(url: &str) -> Result<Vec<String>, FetchError> {
    let html = download_to_string(url)?;
    let regex = Regex::new(r#"href="([^"]+)""#).expect("valid directory regex");
    let mut entries = Vec::new();
    for capture in regex.captures_iter(&html) {
        let href = capture
            .get(1)
            .map(|value| value.as_str())
            .unwrap_or_default();
        if href == "../" || href == "./" || href.starts_with('?') || href.starts_with('/') {
            continue;
        }
        entries.push(href.to_string());
    }
    entries.sort();
    entries.dedup();
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_psp_spc_l3i_csv() {
        let csv = "Time,general_flag,vp_moment_RTN_gd_0,vp_moment_RTN_gd_1,vp_moment_RTN_gd_2,np_moment_gd,wp_moment_gd\n\
2025-01-01T00:00:00Z,0,100.0,20.0,-10.0,45.0,35.0\n\
2025-01-01T00:10:00Z,0,110.0,10.0,0.0,55.0,25.0\n\
2025-01-01T01:00:00Z,1,999.0,999.0,999.0,999.0,999.0\n";
        let rows = parse_psp_spc_l3i_csv(csv);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.year, 2025);
        assert_eq!(row.hour, 0);
        assert!((row.density_cm3 - 50.0).abs() < 1.0e-9);
        assert!(row.speed_kms > 100.0);
        assert!(row.temperature_k > 10_000.0);
    }
}
