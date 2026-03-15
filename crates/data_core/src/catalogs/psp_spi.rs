//! Parker Solar Probe SWEAP/SPI SF00 level-3 moment parser via CDAWeb HAPI.
//!
//! Official public sources:
//!   <https://cdaweb.gsfc.nasa.gov/hapi/info?id=PSP_SWP_SPI_SF00_L3_MOM>
//!   <https://cdaweb.gsfc.nasa.gov/pub/data/psp/sweap/spi/l3/spi_sf00_l3_mom/>
//!
//! The direct CDAWeb daily CDF directory is used as the authoritative day
//! manifest, while the executed Rust science path stages parser-friendly daily
//! HAPI CSV slices for the available public days.

use crate::{
    cdf_support::filename_date_yyyymmdd,
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv, download_to_string},
    parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh},
};
use chrono::NaiveDate;
use csv::ReaderBuilder;
use regex::Regex;
use std::{collections::BTreeMap, path::PathBuf};

const PSP_SPI_SF00_L3_MOM_ROOT: &str =
    "https://cdaweb.gsfc.nasa.gov/pub/data/psp/sweap/spi/l3/spi_sf00_l3_mom/";
const PSP_SPI_SF00_L3_MOM_HAPI_DATASET: &str = "PSP_SWP_SPI_SF00_L3_MOM";
const EV_TO_K: f64 = 11_604.518_121_550_08;

#[derive(Debug, Clone)]
pub struct PspSpiMomRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub density_cm3: f64,
    pub speed_kms: f64,
    pub temperature_k: f64,
}

#[derive(Default)]
struct SpiHourAccumulator {
    density_sum: f64,
    density_count: usize,
    speed_sum: f64,
    speed_count: usize,
    temperature_sum: f64,
    temperature_count: usize,
}

pub fn parse_psp_spi_mom_csv(content: &str) -> Vec<PspSpiMomRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let headers = match reader.headers() {
        Ok(headers) => headers.clone(),
        Err(_) => return Vec::new(),
    };
    let density_col = headers.iter().position(|value| value == "DENS");
    let temp_col = headers.iter().position(|value| value == "TEMP");
    let quality_col = headers.iter().position(|value| value == "QUALITY_FLAG");
    let vr_col = headers.iter().position(|value| value == "VEL_RTN_SUN_0");
    let vt_col = headers.iter().position(|value| value == "VEL_RTN_SUN_1");
    let vn_col = headers.iter().position(|value| value == "VEL_RTN_SUN_2");
    let (
        Some(density_col),
        Some(temp_col),
        Some(quality_col),
        Some(vr_col),
        Some(vt_col),
        Some(vn_col),
    ) = (density_col, temp_col, quality_col, vr_col, vt_col, vn_col)
    else {
        return Vec::new();
    };
    let mut hourly: BTreeMap<(u16, u16, u8), SpiHourAccumulator> = BTreeMap::new();
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
        if quality == 65_535 {
            continue;
        }
        let density = parse_hapi_spacephysics_f64_or_nan(record.get(density_col).unwrap_or(""));
        let temp_ev = parse_hapi_spacephysics_f64_or_nan(record.get(temp_col).unwrap_or(""));
        let vr = parse_hapi_spacephysics_f64_or_nan(record.get(vr_col).unwrap_or(""));
        let vt = parse_hapi_spacephysics_f64_or_nan(record.get(vt_col).unwrap_or(""));
        let vn = parse_hapi_spacephysics_f64_or_nan(record.get(vn_col).unwrap_or(""));
        let speed = if vr.is_finite() || vt.is_finite() || vn.is_finite() {
            let vr = if vr.is_finite() { vr } else { 0.0 };
            let vt = if vt.is_finite() { vt } else { 0.0 };
            let vn = if vn.is_finite() { vn } else { 0.0 };
            (vr * vr + vt * vt + vn * vn).sqrt()
        } else {
            f64::NAN
        };
        let entry = hourly.entry((year, doy, hour)).or_default();
        if density.is_finite() {
            entry.density_sum += density;
            entry.density_count += 1;
        }
        if speed.is_finite() {
            entry.speed_sum += speed;
            entry.speed_count += 1;
        }
        if temp_ev.is_finite() {
            entry.temperature_sum += temp_ev * EV_TO_K;
            entry.temperature_count += 1;
        }
    }
    hourly
        .into_iter()
        .filter_map(|((year, doy, hour), acc)| {
            if acc.density_count == 0 && acc.speed_count == 0 && acc.temperature_count == 0 {
                return None;
            }
            Some(PspSpiMomRecord {
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
                temperature_k: if acc.temperature_count > 0 {
                    acc.temperature_sum / acc.temperature_count as f64
                } else {
                    f64::NAN
                },
            })
        })
        .collect()
}

pub fn parse_psp_spi_mom_file(path: &std::path::Path) -> Result<Vec<PspSpiMomRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|err| FetchError::Validation(format!("read error: {err}")))?;
    let rows = parse_psp_spi_mom_csv(&content);
    if rows.is_empty() {
        return Err(FetchError::Validation(format!(
            "PSP SPI moment CSV {} had no finite hourly rows",
            path.display()
        )));
    }
    Ok(rows)
}

pub struct PspSpiMomProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for PspSpiMomProvider {
    fn default() -> Self {
        Self {
            year_start: 2025,
            year_end: 2025,
        }
    }
}

impl DatasetProvider for PspSpiMomProvider {
    fn name(&self) -> &str {
        "Parker Solar Probe SWEAP SPI SF00 L3 moments"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let root = config.output_dir.join("psp").join("sweap_spi_sf00_l3_mom");
        std::fs::create_dir_all(&root)?;
        for year in self.year_start..=self.year_end {
            let year_url = format!("{PSP_SPI_SF00_L3_MOM_ROOT}{year}/");
            let year_dir = root.join(year.to_string());
            std::fs::create_dir_all(&year_dir)?;
            for entry in directory_entries(&year_url)? {
                if !entry.ends_with(".cdf") {
                    continue;
                }
                let Some((entry_year, month, day)) =
                    filename_date_yyyymmdd(std::path::Path::new(&entry))
                else {
                    continue;
                };
                if entry_year != year {
                    continue;
                }
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
                    "psp_swp_spi_sf00_l3_mom_{}{:02}{:02}.csv",
                    entry_year, month, day
                ));
                if config.skip_existing && csv_output.exists() {
                    continue;
                }
                match download_hapi_csv(
                    PSP_SPI_SF00_L3_MOM_HAPI_DATASET,
                    &format!("{}T00:00:00Z", day_start.format("%Y-%m-%d")),
                    &format!("{}T00:00:00Z", next_day.format("%Y-%m-%d")),
                    Some(&[
                        "Time",
                        "CNTS",
                        "QUALITY_FLAG",
                        "DENS",
                        "VEL_RTN_SUN",
                        "TEMP",
                        "SUN_DIST",
                    ]),
                ) {
                    Ok(body) => {
                        std::fs::write(&csv_output, body)?;
                        log::info!("saved {}", csv_output.display());
                    }
                    Err(err) => {
                        log::warn!(
                            "failed to download PSP SPI moments for {}-{:02}-{:02}: {}",
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
        let root = config.output_dir.join("psp").join("sweap_spi_sf00_l3_mom");
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
    let regex = Regex::new(r#"href="([^"]+)""#)
        .map_err(|err| FetchError::Validation(format!("invalid PSP SPI directory regex: {err}")))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_psp_spi_mom_csv() {
        let csv = "Time,CNTS,QUALITY_FLAG,DENS,VEL_RTN_SUN_0,VEL_RTN_SUN_1,VEL_RTN_SUN_2,TEMP,SUN_DIST\n\
2025-07-01T00:03:42.373199744Z,152.0,4288,0.045442,292.95,88.471,30.321,64.683,64683000\n\
2025-07-01T00:07:26.069820160Z,145.0,4288,0.040847,310.63,87.622,35.143,64.729,64692000\n";
        let rows = parse_psp_spi_mom_csv(csv);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert!(row.density_cm3.is_finite());
        assert!(row.speed_kms > 300.0);
        assert!(row.temperature_k > 700_000.0);
    }
}
