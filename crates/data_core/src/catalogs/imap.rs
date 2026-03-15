//! IMAP public-product staging providers.
//!
//! Current public official surfaces verified from the CDAWeb/SPDF archive:
//!   - IMAP helio1hr position support:
//!     <https://cdaweb.gsfc.nasa.gov/pub/data/imap/helio1hr/>
//!   - IMAP-Hi L2 ENA h90 product family:
//!     <https://cdaweb.gsfc.nasa.gov/pub/data/imap/hi/l2/h90-ena-h-sf-nsp-full-4deg-3mo/>
//!
//! These providers stage the official CDF products into the Rust fetch lane.

use crate::{
    cdf_support::{
        cdf_scalar_f64_rows, cdf_type_to_unix_ms, cdf_variable_rows, cdf_vector_f64_rows,
        filename_date_yyyymmdd, read_cdf_file, ymd_to_doy,
    },
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_file, download_to_string},
};
use chrono::{Datelike, TimeZone, Timelike, Utc};
use regex::Regex;
use std::path::PathBuf;

const IMAP_HELIO1HR_ROOT: &str = "https://cdaweb.gsfc.nasa.gov/pub/data/imap/helio1hr/";
const IMAP_HI_L2_H90_ROOT: &str =
    "https://cdaweb.gsfc.nasa.gov/pub/data/imap/hi/l2/h90-ena-h-sf-nsp-full-4deg-3mo/";

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

fn sanitize_numeric(value: f64) -> f64 {
    if !value.is_finite() || value.abs() >= 1.0e30 {
        f64::NAN
    } else {
        value
    }
}

fn mean(values: &[f64]) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|value| value.is_finite()).collect();
    if finite.is_empty() {
        return f64::NAN;
    }
    finite.iter().sum::<f64>() / finite.len() as f64
}

fn stddev(values: &[f64], mean: f64) -> f64 {
    let finite: Vec<f64> = values.iter().copied().filter(|value| value.is_finite()).collect();
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

pub fn parse_imap_helio1hr_file(
    path: &std::path::Path,
) -> Result<Vec<ImapHelio1hrRecord>, FetchError> {
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
            r_au: radii.get(idx).copied().map(sanitize_numeric).unwrap_or(f64::NAN),
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

pub fn parse_imap_hi_h90_file(
    path: &std::path::Path,
) -> Result<ImapHiH90Summary, FetchError> {
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

fn has_matching_files(dir: &std::path::Path, suffix: &str) -> bool {
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
        let root = config.output_dir.join("imap").join("hi").join("l2").join("h90");
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
        let root = config.output_dir.join("imap").join("hi").join("l2").join("h90");
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
}
