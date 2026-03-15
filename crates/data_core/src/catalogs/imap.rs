//! IMAP public-product staging providers.
//!
//! Verified public official surfaces:
//!   - IMAP helio1hr position support:
//!     <https://cdaweb.gsfc.nasa.gov/pub/data/imap/helio1hr/>
//!   - IMAP-Hi L2 ENA h90 product family:
//!     <https://cdaweb.gsfc.nasa.gov/pub/data/imap/hi/l2/h90-ena-h-sf-nsp-full-4deg-3mo/>
//!   - IMAP I-ALiRT archive CDF lane:
//!     <https://cdaweb.gsfc.nasa.gov/pub/data/imap/ialirt/l1/realtime/>
//!   - IMAP I-ALiRT public live JSON API:
//!     <https://ialirt.imap-mission.com/space-weather>
//!     <https://ialirt.imap-mission.com/ialirt-archive-query>
//!
//! The executed Rust lane prefers the public JSON API for live I-ALiRT science
//! rows while still staging the official archive CDFs for provenance.

use crate::{
    cdf_support::{
        cdf_scalar_f64_rows, cdf_type_to_unix_ms, cdf_type_to_ydh, cdf_variable_rows,
        cdf_vector_f64_rows, filename_date_yyyymmdd, read_cdf_file, ymd_to_doy,
    },
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_file, download_to_string},
};
use chrono::{DateTime, Datelike, Duration, NaiveDateTime, TimeZone, Timelike, Utc};
use regex::Regex;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

const IMAP_HELIO1HR_ROOT: &str = "https://cdaweb.gsfc.nasa.gov/pub/data/imap/helio1hr/";
const IMAP_HI_L2_H90_ROOT: &str =
    "https://cdaweb.gsfc.nasa.gov/pub/data/imap/hi/l2/h90-ena-h-sf-nsp-full-4deg-3mo/";
const IMAP_IALIRT_L1_REALTIME_ROOT: &str =
    "https://cdaweb.gsfc.nasa.gov/pub/data/imap/ialirt/l1/realtime/";
const IMAP_IALIRT_API_ROOT: &str = "https://ialirt.imap-mission.com/space-weather";
const AU_KM: f64 = 149_597_870.7;

#[derive(Debug, Clone)]
pub struct ImapHelio1hrRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub r_au: f64,
    pub lat_deg: f64,
    pub lon_deg: f64,
}

#[derive(Debug, Clone)]
pub struct ImapHiH90Summary {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub map_flux_mean: f64,
    pub map_flux_std: f64,
    pub pixel_count: usize,
    pub energy_bin_count: usize,
}

#[derive(Debug, Clone)]
pub struct ImapIalirtRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub r_au: f64,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub pseudo_density: f64,
    pub pseudo_speed: f64,
    pub pseudo_temperature: f64,
    pub br: f64,
    pub bt: f64,
    pub bn: f64,
    pub b_magnitude: f64,
    pub spectral_mean: f64,
    pub spectral_peak: f64,
}

#[derive(Default)]
struct IalirtAccumulator {
    r_sum: f64,
    lat_sum: f64,
    lon_sum: f64,
    pos_count: usize,
    density_sum: f64,
    density_count: usize,
    speed_sum: f64,
    speed_count: usize,
    temp_sum: f64,
    temp_count: usize,
    br_sum: f64,
    bt_sum: f64,
    bn_sum: f64,
    bmag_sum: f64,
    mag_count: usize,
    spectral_mean_sum: f64,
    spectral_mean_count: usize,
    spectral_peak_sum: f64,
    spectral_peak_count: usize,
}

fn sanitize_numeric(value: f64) -> f64 {
    if !value.is_finite() || value.abs() >= 1.0e30 {
        f64::NAN
    } else {
        value
    }
}

fn mean(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.iter().sum::<f64>() / finite.len() as f64
}

fn stddev(values: &[f64], mean: f64) -> f64 {
    let finite: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    if finite.len() < 2 || !mean.is_finite() {
        return 0.0;
    }
    let var = finite
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / finite.len() as f64;
    var.sqrt()
}

fn json_f64(value: &Value, key: &str) -> f64 {
    value
        .get(key)
        .and_then(Value::as_f64)
        .map_or(f64::NAN, sanitize_numeric)
}

fn json_vec_f64(value: &Value, key: &str) -> Vec<f64> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|values| {
            values
                .iter()
                .filter_map(Value::as_f64)
                .map(sanitize_numeric)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn iso_time_to_ydh(value: &str) -> Option<(u16, u16, u8)> {
    let dt = if let Ok(dt) = DateTime::parse_from_rfc3339(value) {
        dt.with_timezone(&Utc)
    } else {
        let naive = NaiveDateTime::parse_from_str(value, "%Y-%m-%dT%H:%M:%S").ok()?;
        Utc.from_utc_datetime(&naive)
    };
    Some((dt.year() as u16, dt.ordinal() as u16, dt.hour() as u8))
}

fn km_xyz_to_spherical_au(x_km: f64, y_km: f64, z_km: f64) -> (f64, f64, f64) {
    if !x_km.is_finite() || !y_km.is_finite() || !z_km.is_finite() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let r_km = (x_km * x_km + y_km * y_km + z_km * z_km).sqrt();
    if r_km == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let r_au = r_km / AU_KM;
    let lat_deg = (z_km / r_km).asin().to_degrees();
    let lon_deg = y_km.atan2(x_km).to_degrees();
    (r_au, lat_deg, lon_deg)
}

fn rows_from_accumulator(
    hourly: BTreeMap<(u16, u16, u8), IalirtAccumulator>,
) -> Vec<ImapIalirtRecord> {
    hourly
        .into_iter()
        .map(|((year, doy, hour), acc)| ImapIalirtRecord {
            year,
            doy,
            hour,
            r_au: if acc.pos_count > 0 {
                acc.r_sum / acc.pos_count as f64
            } else {
                f64::NAN
            },
            lat_deg: if acc.pos_count > 0 {
                acc.lat_sum / acc.pos_count as f64
            } else {
                f64::NAN
            },
            lon_deg: if acc.pos_count > 0 {
                acc.lon_sum / acc.pos_count as f64
            } else {
                f64::NAN
            },
            pseudo_density: if acc.density_count > 0 {
                acc.density_sum / acc.density_count as f64
            } else {
                f64::NAN
            },
            pseudo_speed: if acc.speed_count > 0 {
                acc.speed_sum / acc.speed_count as f64
            } else {
                f64::NAN
            },
            pseudo_temperature: if acc.temp_count > 0 {
                acc.temp_sum / acc.temp_count as f64
            } else {
                f64::NAN
            },
            br: if acc.mag_count > 0 {
                acc.br_sum / acc.mag_count as f64
            } else {
                f64::NAN
            },
            bt: if acc.mag_count > 0 {
                acc.bt_sum / acc.mag_count as f64
            } else {
                f64::NAN
            },
            bn: if acc.mag_count > 0 {
                acc.bn_sum / acc.mag_count as f64
            } else {
                f64::NAN
            },
            b_magnitude: if acc.mag_count > 0 {
                acc.bmag_sum / acc.mag_count as f64
            } else {
                f64::NAN
            },
            spectral_mean: if acc.spectral_mean_count > 0 {
                acc.spectral_mean_sum / acc.spectral_mean_count as f64
            } else {
                f64::NAN
            },
            spectral_peak: if acc.spectral_peak_count > 0 {
                acc.spectral_peak_sum / acc.spectral_peak_count as f64
            } else {
                f64::NAN
            },
        })
        .collect()
}

pub fn parse_imap_helio1hr_file(path: &Path) -> Result<Vec<ImapHelio1hrRecord>, FetchError> {
    let cdf = read_cdf_file(path)?;
    let (file_year, file_month, file_day) = filename_date_yyyymmdd(path).ok_or_else(|| {
        FetchError::Validation(format!(
            "could not infer IMAP helio1hr date from filename {}",
            path.display()
        ))
    })?;
    let file_doy = ymd_to_doy(file_year, file_month, file_day).ok_or_else(|| {
        FetchError::Validation(format!(
            "invalid IMAP helio1hr file date {}-{:02}-{:02}",
            file_year, file_month, file_day
        ))
    })?;
    let radii = cdf_scalar_f64_rows(&cdf, "RAD_AU")?;
    let latitudes = cdf_scalar_f64_rows(&cdf, "HG_LAT")?;
    let longitudes = cdf_scalar_f64_rows(&cdf, "HG_LON")?;
    let epoch_rows = cdf_variable_rows(&cdf, "Epoch").ok();

    let row_count = *[radii.len(), latitudes.len(), longitudes.len()]
        .iter()
        .max()
        .unwrap_or(&0);
    if row_count == 0 {
        return Err(FetchError::Validation(format!(
            "IMAP helio1hr {} yielded zero rows",
            path.display()
        )));
    }

    let mut rows = Vec::with_capacity(row_count);
    for idx in 0..row_count {
        let (year, doy, hour) = if let Some(epoch_rows) = epoch_rows.as_deref() {
            if let Some(value) = epoch_rows.get(idx).and_then(|row| row.first()) {
                if let Some(timestamp_ms) = cdf_type_to_unix_ms(value) {
                    if let Some(dt) = Utc.timestamp_millis_opt(timestamp_ms).single() {
                        (dt.year() as u16, dt.ordinal() as u16, dt.hour() as u8)
                    } else {
                        (file_year, file_doy, idx.min(23) as u8)
                    }
                } else {
                    (file_year, file_doy, idx.min(23) as u8)
                }
            } else {
                (file_year, file_doy, idx.min(23) as u8)
            }
        } else {
            (file_year, file_doy, idx.min(23) as u8)
        };
        rows.push(ImapHelio1hrRecord {
            year,
            doy,
            hour,
            r_au: radii
                .get(idx)
                .copied()
                .map(sanitize_numeric)
                .unwrap_or(f64::NAN),
            lat_deg: latitudes
                .get(idx)
                .copied()
                .map(sanitize_numeric)
                .unwrap_or(f64::NAN),
            lon_deg: longitudes
                .get(idx)
                .copied()
                .map(sanitize_numeric)
                .unwrap_or(f64::NAN),
        });
    }
    Ok(rows)
}

pub fn parse_imap_hi_h90_file(path: &Path) -> Result<ImapHiH90Summary, FetchError> {
    let cdf = read_cdf_file(path)?;
    let (year, month, day) = filename_date_yyyymmdd(path).ok_or_else(|| {
        FetchError::Validation(format!(
            "could not infer IMAP-Hi date from filename {}",
            path.display()
        ))
    })?;
    let doy = ymd_to_doy(year, month, day).ok_or_else(|| {
        FetchError::Validation(format!(
            "invalid IMAP-Hi file date {}-{:02}-{:02}",
            year, month, day
        ))
    })?;
    let intensity_rows = cdf_vector_f64_rows(&cdf, "ena_intensity")?;
    let energy_rows = cdf_vector_f64_rows(&cdf, "energy").ok();

    let values: Vec<f64> = intensity_rows
        .iter()
        .flat_map(|row| row.iter().copied())
        .map(sanitize_numeric)
        .filter(|value| value.is_finite())
        .collect();
    if values.is_empty() {
        return Err(FetchError::Validation(format!(
            "IMAP-Hi {} contained no finite ENA intensities",
            path.display()
        )));
    }
    let flux_mean = mean(&values);
    Ok(ImapHiH90Summary {
        year,
        doy,
        hour: 0,
        map_flux_mean: flux_mean,
        map_flux_std: stddev(&values, flux_mean),
        pixel_count: values.len(),
        energy_bin_count: energy_rows
            .as_ref()
            .and_then(|rows| rows.first())
            .map(|row| row.len())
            .unwrap_or(0),
    })
}

pub fn parse_imap_ialirt_file(path: &Path) -> Result<Vec<ImapIalirtRecord>, FetchError> {
    let cdf = read_cdf_file(path)?;
    let mut hourly: BTreeMap<(u16, u16, u8), IalirtAccumulator> = BTreeMap::new();

    let mag_epoch_rows = cdf_variable_rows(&cdf, "mag_epoch").ok();
    let mag_b_rtn_rows = cdf_vector_f64_rows(&cdf, "mag_B_RTN").ok();
    let mag_b_mag_rows = cdf_scalar_f64_rows(&cdf, "mag_B_magnitude").ok();
    if let (Some(epoch_rows), Some(b_rtn_rows), Some(b_mag_rows)) = (
        mag_epoch_rows.as_deref(),
        mag_b_rtn_rows.as_deref(),
        mag_b_mag_rows.as_deref(),
    ) {
        let row_count = epoch_rows.len().min(b_rtn_rows.len()).min(b_mag_rows.len());
        for idx in 0..row_count {
            let Some((year, doy, hour)) = epoch_rows
                .get(idx)
                .and_then(|row| row.first())
                .and_then(cdf_type_to_ydh)
            else {
                continue;
            };
            let vector = &b_rtn_rows[idx];
            let br = vector
                .first()
                .copied()
                .map(sanitize_numeric)
                .unwrap_or(f64::NAN);
            let bt = vector
                .get(1)
                .copied()
                .map(sanitize_numeric)
                .unwrap_or(f64::NAN);
            let bn = vector
                .get(2)
                .copied()
                .map(sanitize_numeric)
                .unwrap_or(f64::NAN);
            let b_mag = sanitize_numeric(b_mag_rows[idx]);
            if !br.is_finite() || !bt.is_finite() || !bn.is_finite() {
                continue;
            }
            let entry = hourly.entry((year, doy, hour)).or_default();
            entry.br_sum += br;
            entry.bt_sum += bt;
            entry.bn_sum += bn;
            if b_mag.is_finite() {
                entry.bmag_sum += b_mag;
            }
            entry.mag_count += 1;
        }
    }

    let swapi_epoch_rows = cdf_variable_rows(&cdf, "swapi_epoch").ok();
    let swapi_density_rows = cdf_scalar_f64_rows(&cdf, "swapi_pseudo_proton_density").ok();
    let swapi_speed_rows = cdf_scalar_f64_rows(&cdf, "swapi_pseudo_proton_speed").ok();
    let swapi_temp_rows = cdf_scalar_f64_rows(&cdf, "swapi_pseudo_proton_temperature").ok();
    if let (Some(epoch_rows), Some(density_rows), Some(speed_rows), Some(temp_rows)) = (
        swapi_epoch_rows.as_deref(),
        swapi_density_rows.as_deref(),
        swapi_speed_rows.as_deref(),
        swapi_temp_rows.as_deref(),
    ) {
        let row_count = epoch_rows
            .len()
            .min(density_rows.len())
            .min(speed_rows.len())
            .min(temp_rows.len());
        for idx in 0..row_count {
            let Some((year, doy, hour)) = epoch_rows
                .get(idx)
                .and_then(|row| row.first())
                .and_then(cdf_type_to_ydh)
            else {
                continue;
            };
            let density = sanitize_numeric(density_rows[idx]);
            let speed = sanitize_numeric(speed_rows[idx]);
            let temperature = sanitize_numeric(temp_rows[idx]);
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
    }

    let rows = rows_from_accumulator(hourly);
    if rows.is_empty() {
        return Err(FetchError::Validation(format!(
            "IMAP I-ALiRT {} yielded zero rows",
            path.display()
        )));
    }
    Ok(rows)
}

pub fn parse_imap_ialirt_live_day(
    science_path: &Path,
    spacecraft_path: Option<&Path>,
) -> Result<Vec<ImapIalirtRecord>, FetchError> {
    let science_raw = std::fs::read_to_string(science_path)
        .map_err(|err| FetchError::Validation(format!("read error: {err}")))?;
    let science_value: Value = serde_json::from_str(&science_raw).map_err(|err| {
        FetchError::Validation(format!(
            "IMAP I-ALiRT science JSON parse error for {}: {}",
            science_path.display(),
            err
        ))
    })?;
    let mut hourly: BTreeMap<(u16, u16, u8), IalirtAccumulator> = BTreeMap::new();

    for row in science_value
        .get("data")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let instrument = row
            .get("instrument")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let time = row
            .get("time_utc")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let Some((year, doy, hour)) = iso_time_to_ydh(time) else {
            continue;
        };
        let entry = hourly.entry((year, doy, hour)).or_default();
        match instrument {
            "mag" => {
                let rtn = json_vec_f64(row, "mag_B_RTN");
                let br = rtn.first().copied().unwrap_or(f64::NAN);
                let bt = rtn.get(1).copied().unwrap_or(f64::NAN);
                let bn = rtn.get(2).copied().unwrap_or(f64::NAN);
                let b_mag = json_f64(row, "mag_B_magnitude");
                if br.is_finite() && bt.is_finite() && bn.is_finite() {
                    entry.br_sum += br;
                    entry.bt_sum += bt;
                    entry.bn_sum += bn;
                    if b_mag.is_finite() {
                        entry.bmag_sum += b_mag;
                    }
                    entry.mag_count += 1;
                }
            }
            "swapi" => {
                let density = json_f64(row, "swapi_pseudo_proton_density");
                let speed = json_f64(row, "swapi_pseudo_proton_speed");
                let temp = json_f64(row, "swapi_pseudo_proton_temperature");
                if density.is_finite() {
                    entry.density_sum += density;
                    entry.density_count += 1;
                }
                if speed.is_finite() {
                    entry.speed_sum += speed;
                    entry.speed_count += 1;
                }
                if temp.is_finite() {
                    entry.temp_sum += temp;
                    entry.temp_count += 1;
                }
            }
            "swe" => {
                let counts = json_vec_f64(row, "swe_normalized_counts");
                if !counts.is_empty() {
                    let filtered = counts
                        .into_iter()
                        .filter(|value| value.is_finite())
                        .collect::<Vec<_>>();
                    if !filtered.is_empty() {
                        entry.spectral_mean_sum += mean(&filtered);
                        entry.spectral_mean_count += 1;
                        let peak = filtered.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                        if peak.is_finite() {
                            entry.spectral_peak_sum += peak;
                            entry.spectral_peak_count += 1;
                        }
                    }
                }
            }
            _ => {}
        }
    }

    if let Some(spacecraft_path) = spacecraft_path {
        let spacecraft_raw = std::fs::read_to_string(spacecraft_path)
            .map_err(|err| FetchError::Validation(format!("read error: {err}")))?;
        let spacecraft_value: Value = serde_json::from_str(&spacecraft_raw).map_err(|err| {
            FetchError::Validation(format!(
                "IMAP I-ALiRT spacecraft JSON parse error for {}: {}",
                spacecraft_path.display(),
                err
            ))
        })?;
        for row in spacecraft_value
            .get("data")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let time = row
                .get("time_utc")
                .and_then(Value::as_str)
                .unwrap_or_default();
            let Some((year, doy, hour)) = iso_time_to_ydh(time) else {
                continue;
            };
            let pos = json_vec_f64(row, "sc_position_GSE");
            if pos.len() < 3 {
                continue;
            }
            let (r_au, lat_deg, lon_deg) = km_xyz_to_spherical_au(pos[0], pos[1], pos[2]);
            if !r_au.is_finite() || !lat_deg.is_finite() || !lon_deg.is_finite() {
                continue;
            }
            let entry = hourly.entry((year, doy, hour)).or_default();
            entry.r_sum += r_au;
            entry.lat_sum += lat_deg;
            entry.lon_sum += lon_deg;
            entry.pos_count += 1;
        }
    }

    let rows = rows_from_accumulator(hourly);
    if rows.is_empty() {
        return Err(FetchError::Validation(format!(
            "IMAP I-ALiRT live JSON {} yielded zero rows",
            science_path.display()
        )));
    }
    Ok(rows)
}

fn directory_entries(url: &str) -> Result<Vec<String>, FetchError> {
    let html = download_to_string(url)?;
    let regex = Regex::new(r#"href="([^"]+)""#)
        .map_err(|err| FetchError::Validation(format!("invalid IMAP directory regex: {err}")))?;
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
    std::fs::read_dir(dir)
        .ok()
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.ok())
        .any(|entry| {
            let file_type = match entry.file_type() {
                Ok(file_type) => file_type,
                Err(_) => return false,
            };
            if !file_type.is_file() {
                return false;
            }
            entry.file_name().to_string_lossy().ends_with(suffix)
        })
}

fn ialirt_live_url(instrument: Option<&str>, start: &str, end: &str) -> String {
    let mut url = format!("{IMAP_IALIRT_API_ROOT}?time_utc_start={start}&time_utc_end={end}");
    if let Some(instrument) = instrument {
        url.push_str("&instrument=");
        url.push_str(instrument);
    }
    url
}

fn ialirt_json_rows(body: &str) -> Result<Vec<Value>, FetchError> {
    let value: Value = serde_json::from_str(body)
        .map_err(|err| FetchError::Validation(format!("IMAP I-ALiRT JSON parse error: {err}")))?;
    Ok(value
        .get("data")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default())
}

fn ialirt_json_payload(
    kind: &str,
    instrument: Option<&str>,
    rows: Vec<Value>,
) -> Result<String, FetchError> {
    serde_json::to_string_pretty(&serde_json::json!({
        "meta": {
            "count": rows.len(),
            "type": kind,
            "instrument": instrument.unwrap_or("combined"),
        },
        "data": rows,
    }))
    .map_err(|err| FetchError::Validation(format!("IMAP I-ALiRT JSON encode error: {err}")))
}

fn fetch_ialirt_live_rows(
    instrument: &str,
    start: DateTime<Utc>,
    end: DateTime<Utc>,
    chunk_hours: i64,
) -> Result<Vec<Value>, FetchError> {
    let mut rows = Vec::new();
    let mut cursor = start;
    while cursor < end {
        let chunk_end = std::cmp::min(cursor + Duration::hours(chunk_hours), end);
        let url = ialirt_live_url(
            Some(instrument),
            &cursor.format("%Y-%m-%dT%H:%M:%S").to_string(),
            &chunk_end.format("%Y-%m-%dT%H:%M:%S").to_string(),
        );
        let body = download_to_string(&url)?;
        rows.extend(ialirt_json_rows(&body)?);
        cursor = chunk_end;
    }
    Ok(rows)
}

/// Public IMAP heliocentric hourly position support from the official archive.
pub struct ImapHelio1hrProvider;

impl DatasetProvider for ImapHelio1hrProvider {
    fn name(&self) -> &str {
        "IMAP Helio1hr Position"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("imap").join("helio1hr");
        std::fs::create_dir_all(&dir)?;
        for entry in directory_entries(IMAP_HELIO1HR_ROOT)? {
            if !entry.ends_with(".cdf") {
                continue;
            }
            let output = dir.join(&entry);
            if config.skip_existing && output.exists() {
                continue;
            }
            let url = format!("{IMAP_HELIO1HR_ROOT}{entry}");
            download_to_file(&url, &output)?;
            log::info!("saved {}", output.display());
        }
        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        has_matching_files(&config.output_dir.join("imap").join("helio1hr"), ".cdf")
    }
}

/// Public IMAP-Hi L2 ENA h90 CDF staging provider.
pub struct ImapHiL2H90Provider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for ImapHiL2H90Provider {
    fn default() -> Self {
        Self {
            year_start: 2026,
            year_end: 2026,
        }
    }
}

impl DatasetProvider for ImapHiL2H90Provider {
    fn name(&self) -> &str {
        "IMAP-Hi L2 ENA h90"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let root = config
            .output_dir
            .join("imap")
            .join("hi")
            .join("l2")
            .join("h90");
        std::fs::create_dir_all(&root)?;
        for year in self.year_start..=self.year_end {
            let year_url = format!("{IMAP_HI_L2_H90_ROOT}{year}/");
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
        }
        Ok(root)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let root = config
            .output_dir
            .join("imap")
            .join("hi")
            .join("l2")
            .join("h90");
        if has_matching_files(&root, ".cdf") {
            return true;
        }
        std::fs::read_dir(&root)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| {
                let file_type = match entry.file_type() {
                    Ok(file_type) => file_type,
                    Err(_) => return false,
                };
                if !file_type.is_dir() {
                    return false;
                }
                has_matching_files(&entry.path(), ".cdf")
            })
    }
}

/// Public IMAP I-ALiRT archive + live JSON staging provider.
pub struct ImapIalirtRealtimeProvider {
    pub year_start: u16,
    pub year_end: u16,
    pub live_lookback_days: u16,
}

impl Default for ImapIalirtRealtimeProvider {
    fn default() -> Self {
        Self {
            year_start: 2026,
            year_end: 2026,
            live_lookback_days: 7,
        }
    }
}

impl DatasetProvider for ImapIalirtRealtimeProvider {
    fn name(&self) -> &str {
        "IMAP I-ALiRT L1 Realtime"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let root = config
            .output_dir
            .join("imap")
            .join("ialirt")
            .join("l1")
            .join("realtime");
        std::fs::create_dir_all(&root)?;
        for year in self.year_start..=self.year_end {
            let year_url = format!("{IMAP_IALIRT_L1_REALTIME_ROOT}{year}/");
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
        }

        let live_root = config
            .output_dir
            .join("imap")
            .join("ialirt")
            .join("space_weather");
        std::fs::create_dir_all(&live_root)?;
        let today = Utc::now().date_naive();
        for delta in 0..=self.live_lookback_days {
            let day = today - Duration::days(i64::from(delta));
            let next_day = day
                .succ_opt()
                .ok_or_else(|| FetchError::Validation(format!("advance live day {day}")))?;
            let start = Utc.from_utc_datetime(
                &day.and_hms_opt(0, 0, 0)
                    .ok_or_else(|| FetchError::Validation(format!("start of day {day}")))?,
            );
            let end =
                Utc.from_utc_datetime(&next_day.and_hms_opt(0, 0, 0).ok_or_else(|| {
                    FetchError::Validation(format!("start of next day {next_day}"))
                })?);
            let year_dir = live_root.join(day.year().to_string());
            std::fs::create_dir_all(&year_dir)?;

            let science_output = year_dir.join(format!(
                "imap_ialirt_space_weather_science_{}.json",
                day.format("%Y%m%d")
            ));
            if !(config.skip_existing && science_output.exists()) {
                let mut science_rows = Vec::new();
                science_rows.extend(fetch_ialirt_live_rows("mag", start, end, 1)?);
                science_rows.extend(fetch_ialirt_live_rows("swapi", start, end, 24)?);
                science_rows.extend(fetch_ialirt_live_rows("swe", start, end, 24)?);
                if !science_rows.is_empty() {
                    let body = ialirt_json_payload("science", None, science_rows)?;
                    std::fs::write(&science_output, body)?;
                    log::info!("saved {}", science_output.display());
                }
            }

            let spacecraft_output = year_dir.join(format!(
                "imap_ialirt_space_weather_spacecraft_{}.json",
                day.format("%Y%m%d")
            ));
            if !(config.skip_existing && spacecraft_output.exists()) {
                let spacecraft_rows = fetch_ialirt_live_rows("spacecraft", start, end, 24)?;
                if !spacecraft_rows.is_empty() {
                    let body =
                        ialirt_json_payload("spacecraft", Some("spacecraft"), spacecraft_rows)?;
                    std::fs::write(&spacecraft_output, body)?;
                    log::info!("saved {}", spacecraft_output.display());
                }
            }
        }
        Ok(root)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let archive_root = config
            .output_dir
            .join("imap")
            .join("ialirt")
            .join("l1")
            .join("realtime");
        let live_root = config
            .output_dir
            .join("imap")
            .join("ialirt")
            .join("space_weather");
        has_matching_files(&archive_root, ".cdf")
            || has_matching_files(&live_root, ".json")
            || std::fs::read_dir(&archive_root)
                .ok()
                .into_iter()
                .flatten()
                .filter_map(|entry| entry.ok())
                .any(|entry| entry.path().is_dir() && has_matching_files(&entry.path(), ".cdf"))
            || std::fs::read_dir(&live_root)
                .ok()
                .into_iter()
                .flatten()
                .filter_map(|entry| entry.ok())
                .any(|entry| entry.path().is_dir() && has_matching_files(&entry.path(), ".json"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imap_summary_stats() {
        let values = vec![1.0, 2.0, 3.0];
        let mean_value = mean(&values);
        assert!((mean_value - 2.0).abs() < 1.0e-12);
        assert!(stddev(&values, mean_value) > 0.8);
    }

    #[test]
    fn test_sanitize_numeric() {
        assert!(sanitize_numeric(-1.0e31).is_nan());
        assert!((sanitize_numeric(3.5) - 3.5).abs() < 1.0e-12);
    }

    #[test]
    fn test_parse_imap_ialirt_live_day() {
        let dir = tempfile::tempdir().expect("tempdir");
        let science = dir.path().join("science.json");
        let spacecraft = dir.path().join("spacecraft.json");
        std::fs::write(
            &science,
            r#"{"meta":{"count":3},"data":[
{"instrument":"mag","time_utc":"2026-03-15T08:00:01","mag_B_RTN":[1.0,2.0,3.0],"mag_B_magnitude":3.741},
{"instrument":"swapi","time_utc":"2026-03-15T08:00:07","swapi_pseudo_proton_density":4.0,"swapi_pseudo_proton_speed":500.0,"swapi_pseudo_proton_temperature":100000.0},
{"instrument":"swe","time_utc":"2026-03-15T08:00:03","swe_normalized_counts":[1.0,3.0,5.0],"swe_counterstreaming_electrons":0}]}"#,
        )
        .expect("write science");
        std::fs::write(
            &spacecraft,
            r#"{"meta":{"count":1},"data":[
{"instrument":"spacecraft","time_utc":"2026-03-15T08:00:03","sc_position_GSE":[1495978.707,0.0,0.0]}]}"#,
        )
        .expect("write spacecraft");
        let rows = parse_imap_ialirt_live_day(&science, Some(&spacecraft)).expect("parse");
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.hour, 8);
        assert!((row.pseudo_density - 4.0).abs() < 1.0e-9);
        assert!((row.br - 1.0).abs() < 1.0e-9);
        assert!((row.spectral_mean - 3.0).abs() < 1.0e-9);
        assert!((row.spectral_peak - 5.0).abs() < 1.0e-9);
        assert!((row.r_au - 0.01).abs() < 1.0e-6);
    }
}
