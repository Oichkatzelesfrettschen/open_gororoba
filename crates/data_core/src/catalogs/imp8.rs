//! IMP 8 field-plasma merged 1-minute reference lane.
//!
//! Official source:
//!   <https://spdf.gsfc.nasa.gov/pub/data/imp/imp8/merged/>
//!   Monthly ASCII files named `imp_min_mergeYYYYMM.asc`.
//!
//! The readme documents minute-resolution near-Earth reference records with
//! GSE magnetic-field vectors, plasma moments, and spacecraft position in Re.
//! This module aggregates those minute records to hourly means for historical
//! Helios-era comparisons.

use crate::{
    catalogs::omni::OmniRecord,
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_file},
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

const IMP8_MERGED_ROOT: &str = "https://spdf.gsfc.nasa.gov/pub/data/imp/imp8/merged/";
const EARTH_RADIUS_KM: f64 = 6378.137;
const AU_KM: f64 = 149_597_870.7;

#[derive(Debug, Clone)]
pub struct Imp8HourlyRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub r_au: f64,
    pub lat_deg: f64,
    pub lon_deg: f64,
    pub bx_gse: f64,
    pub by_gse: f64,
    pub bz_gse: f64,
    pub b_mag: f64,
    pub density_cm3: f64,
    pub speed_kms: f64,
    pub temperature_k: f64,
}

#[derive(Default)]
struct HourAccumulator {
    x_re_sum: f64,
    y_re_sum: f64,
    z_re_sum: f64,
    pos_count: usize,
    bx_sum: f64,
    by_sum: f64,
    bz_sum: f64,
    bmag_sum: f64,
    b_count: usize,
    density_sum: f64,
    density_count: usize,
    speed_sum: f64,
    speed_count: usize,
    temp_sum: f64,
    temp_count: usize,
}

fn parse_or_nan(token: &str, fill: f64) -> f64 {
    let Ok(value) = token.parse::<f64>() else {
        return f64::NAN;
    };
    if !value.is_finite() || (value - fill).abs() < 1.0e-6 || value.abs() >= 1.0e30 {
        f64::NAN
    } else {
        value
    }
}

fn re_xyz_to_spherical(x_re: f64, y_re: f64, z_re: f64) -> (f64, f64, f64) {
    if !x_re.is_finite() || !y_re.is_finite() || !z_re.is_finite() {
        return (f64::NAN, f64::NAN, f64::NAN);
    }
    let r_re = (x_re * x_re + y_re * y_re + z_re * z_re).sqrt();
    if r_re == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let r_au = (r_re * EARTH_RADIUS_KM) / AU_KM;
    let lat_deg = (z_re / r_re).asin().to_degrees();
    let lon_deg = y_re.atan2(x_re).to_degrees();
    (r_au, lat_deg, lon_deg)
}

pub fn parse_imp8_file(path: &Path) -> Result<Vec<Imp8HourlyRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|err| FetchError::Validation(format!("read error: {err}")))?;
    let mut hourly: BTreeMap<(u16, u16, u8), HourAccumulator> = BTreeMap::new();
    for line in content.lines() {
        let cols = line.split_whitespace().collect::<Vec<_>>();
        if cols.len() < 47 {
            continue;
        }
        let Ok(year) = cols[0].parse::<u16>() else {
            continue;
        };
        let Ok(doy) = cols[1].parse::<u16>() else {
            continue;
        };
        let Ok(hour) = cols[2].parse::<u8>() else {
            continue;
        };
        let solar_wind_flag = cols[4].parse::<u8>().unwrap_or(9);
        if solar_wind_flag != 1 {
            continue;
        }

        let x_re = parse_or_nan(cols[5], 9999.99);
        let y_re = parse_or_nan(cols[6], 9999.99);
        let z_re = parse_or_nan(cols[7], 9999.99);
        let bmag = parse_or_nan(cols[14], 9999.99);
        let bx = parse_or_nan(cols[17], 9999.99);
        let by = parse_or_nan(cols[18], 9999.99);
        let bz = parse_or_nan(cols[19], 9999.99);
        let speed_fit = parse_or_nan(cols[31], 9999.9);
        let density_fit = parse_or_nan(cols[37], 9999.9);
        let temp_fit = parse_or_nan(cols[38], 9_999_999.0);
        let speed_mom = parse_or_nan(cols[39], 9999.9);
        let density_mom = parse_or_nan(cols[45], 9999.9);
        let temp_mom = parse_or_nan(cols[46], 9_999_999.0);

        let entry = hourly.entry((year, doy, hour)).or_default();
        if x_re.is_finite() && y_re.is_finite() && z_re.is_finite() {
            entry.x_re_sum += x_re;
            entry.y_re_sum += y_re;
            entry.z_re_sum += z_re;
            entry.pos_count += 1;
        }
        if bx.is_finite() && by.is_finite() && bz.is_finite() {
            entry.bx_sum += bx;
            entry.by_sum += by;
            entry.bz_sum += bz;
            entry.b_count += 1;
        }
        if bmag.is_finite() {
            entry.bmag_sum += bmag;
            entry.b_count += usize::from(!(bx.is_finite() && by.is_finite() && bz.is_finite()));
        }
        let speed = if speed_fit.is_finite() {
            speed_fit
        } else {
            speed_mom
        };
        let density = if density_fit.is_finite() {
            density_fit
        } else {
            density_mom
        };
        let temp = if temp_fit.is_finite() {
            temp_fit
        } else {
            temp_mom
        };
        if speed.is_finite() {
            entry.speed_sum += speed;
            entry.speed_count += 1;
        }
        if density.is_finite() {
            entry.density_sum += density;
            entry.density_count += 1;
        }
        if temp.is_finite() {
            entry.temp_sum += temp;
            entry.temp_count += 1;
        }
    }

    let rows = hourly
        .into_iter()
        .map(|((year, doy, hour), acc)| {
            let pos_div = acc.pos_count.max(1) as f64;
            let x_re = if acc.pos_count > 0 {
                acc.x_re_sum / pos_div
            } else {
                f64::NAN
            };
            let y_re = if acc.pos_count > 0 {
                acc.y_re_sum / pos_div
            } else {
                f64::NAN
            };
            let z_re = if acc.pos_count > 0 {
                acc.z_re_sum / pos_div
            } else {
                f64::NAN
            };
            let (r_au, lat_deg, lon_deg) = re_xyz_to_spherical(x_re, y_re, z_re);
            let b_div = acc.b_count.max(1) as f64;
            let bx_gse = if acc.b_count > 0 {
                acc.bx_sum / b_div
            } else {
                f64::NAN
            };
            let by_gse = if acc.b_count > 0 {
                acc.by_sum / b_div
            } else {
                f64::NAN
            };
            let bz_gse = if acc.b_count > 0 {
                acc.bz_sum / b_div
            } else {
                f64::NAN
            };
            let b_mag = if acc.b_count > 0 {
                acc.bmag_sum / b_div
            } else if bx_gse.is_finite() || by_gse.is_finite() || bz_gse.is_finite() {
                let bx = if bx_gse.is_finite() { bx_gse } else { 0.0 };
                let by = if by_gse.is_finite() { by_gse } else { 0.0 };
                let bz = if bz_gse.is_finite() { bz_gse } else { 0.0 };
                (bx * bx + by * by + bz * bz).sqrt()
            } else {
                f64::NAN
            };
            Imp8HourlyRecord {
                year,
                doy,
                hour,
                r_au,
                lat_deg,
                lon_deg,
                bx_gse,
                by_gse,
                bz_gse,
                b_mag,
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
            }
        })
        .collect::<Vec<_>>();
    if rows.is_empty() {
        return Err(FetchError::Validation(format!(
            "IMP 8 file {} yielded zero solar-wind hourly rows",
            path.display()
        )));
    }
    Ok(rows)
}

pub fn imp8_to_omni(records: &[Imp8HourlyRecord]) -> Vec<OmniRecord> {
    records
        .iter()
        .map(|record| OmniRecord {
            year: record.year,
            doy: record.doy,
            hour: record.hour,
            b_magnitude: record.b_mag,
            bx_gse: record.bx_gse,
            by_gse: record.by_gse,
            bz_gse: record.bz_gse,
            proton_density: record.density_cm3,
            bulk_speed: record.speed_kms,
            proton_temperature: record.temperature_k,
            flow_pressure: f64::NAN,
            plasma_beta: f64::NAN,
            alfven_mach: f64::NAN,
            dst_index: f64::NAN,
            ae_index: f64::NAN,
            kp_times_10: 0,
            r_au: record.r_au,
            lat_deg: record.lat_deg,
            lon_deg: record.lon_deg,
        })
        .collect()
}

pub struct Imp8Provider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for Imp8Provider {
    fn default() -> Self {
        Self {
            year_start: 1976,
            year_end: 1980,
        }
    }
}

impl DatasetProvider for Imp8Provider {
    fn name(&self) -> &str {
        "IMP 8 Merged 1-minute"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("imp8");
        std::fs::create_dir_all(&dir)?;
        for year in self.year_start..=self.year_end {
            for month in 1..=12 {
                let file_name = format!("imp_min_merge{year}{month:02}.asc");
                let output = dir.join(&file_name);
                if config.skip_existing && output.exists() {
                    continue;
                }
                let url = format!("{IMP8_MERGED_ROOT}{file_name}");
                match download_to_file(&url, &output) {
                    Ok(_) => log::info!("saved {}", output.display()),
                    Err(err) => log::warn!("failed to download {}: {}", url, err),
                }
            }
        }
        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        let dir = config.output_dir.join("imp8");
        std::fs::read_dir(&dir)
            .ok()
            .into_iter()
            .flatten()
            .filter_map(|entry| entry.ok())
            .any(|entry| {
                entry
                    .file_name()
                    .to_string_lossy()
                    .starts_with("imp_min_merge")
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_imp8_file_hourly() {
        let sample = "\
1976   1  0  0 1  -25.96    7.73  -23.40   13.92  -20.33 9 99 9.99   12.0   10.0   20.0   30.0    1.0    2.0    3.0    4.0    5.0    0.1    0.2    0.3    0.4    0.5 3 1 22 1.00  398.5  -390.7   -34.2   -70.6   -5.0  -10.2    3.6 1052555.  427.5  -419.8    25.7   -76.4    3.5  -10.3    2.8  532306.\n\
1976   1  0  1 1  -25.97    7.72  -23.40   13.90  -20.34 9 99 9.99   14.0   12.0   22.0   32.0    2.0    4.0    6.0    8.0   10.0    0.1    0.2    0.3    0.4    0.5 3 1 22 1.00  418.5  -400.7   -14.2   -50.6   -2.0  -11.2    4.6 2052555.  417.5  -409.8    15.7   -66.4    2.5  -10.3    3.8  632306.\n";
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("imp_min_merge197601.asc");
        std::fs::write(&path, sample).expect("write sample");
        let rows = parse_imp8_file(&path).expect("parse");
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.year, 1976);
        assert_eq!(row.doy, 1);
        assert_eq!(row.hour, 0);
        assert!((row.bx_gse - 1.5).abs() < 1.0e-9);
        assert!((row.speed_kms - 408.5).abs() < 1.0e-9);
        assert!((row.density_cm3 - 4.1).abs() < 1.0e-9);
        assert!(row.r_au.is_finite());
    }
}
