//! STEREO-A PLASTIC and IMPACT/MAG data providers.
//!
//! STEREO-A orbits the Sun ahead of Earth at a different heliocentric
//! longitude, enabling spatial triangulation that breaks the 1D Taylor
//! hypothesis limitation of single-point L1 measurements.
//!
//! **PLASTIC** (PLasma and Supra-Thermal Ion Composition):
//!   Source: <https://stereo-ssc.nascom.nasa.gov/data/ins_data/plastic/level2/>
//!   Reference: Galvin et al. (2008), Space Sci. Rev. 136, 437
//!   Format: Tab-delimited ASCII, 2-line header, 38 columns.
//!   Coordinate system: RTN (Radial-Tangential-Normal).
//!
//! **IMPACT/MAG** (In-situ Measurements of Particles and CME Transients):
//!   Combined MAGPLASMA product via CDAWeb text export.
//!   Columns: B_total, Br, Bt, Bn (RTN), Np, Vp, Tp, Beta, Entropy.
//!   Fill: -1.0E+31 for all parameters.
//!
//! RTN -> GSE coordinate transform requires Earth-STEREO separation angle.

use crate::catalogs::omni::OmniRecord;
use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_file};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// STEREO PLASTIC (plasma)
// ---------------------------------------------------------------------------

/// A single hourly STEREO-A PLASTIC measurement.
///
/// Parsed from STEREO SSC tab-delimited ASCII files.
/// Key columns (0-based): 0=YEAR, 1=DOY, 2=hour, 4=Np, 5=Bulk_Speed,
/// 6=Tkin, 14/15/16=Vr/Vt/Vn (RTN). RTN velocity vectors are fill
/// from 2020-03 onward.
#[derive(Debug, Clone)]
pub struct StereoPlasticRecord {
    /// Calendar year.
    pub year: u16,
    /// Day of year (1-366).
    pub doy: u16,
    /// Hour of day (0-23).
    pub hour: u8,
    /// Proton number density (cm^-3).
    pub proton_density: f64,
    /// Bulk solar wind speed, scalar (km/s).
    pub bulk_speed: f64,
    /// Kinetic temperature (K).
    pub temperature: f64,
    /// Radial velocity (km/s, RTN). NaN if fill.
    pub vr: f64,
    /// Tangential velocity (km/s, RTN). NaN if fill.
    pub vt: f64,
    /// Normal velocity (km/s, RTN). NaN if fill.
    pub vn: f64,
}

// ---------------------------------------------------------------------------
// STEREO IMPACT/MAG (magnetic field, from MAGPLASMA CDAWeb export)
// ---------------------------------------------------------------------------

/// A single STEREO-A IMPACT/MAG measurement.
///
/// From CDAWeb text export of STA_L2_MAGPLASMA_1M (1-minute).
/// Columns: Epoch, B_total, Br, Bt, Bn, Np, Vp, Tp, Beta, Entropy.
/// All B-field in RTN coordinates.
#[derive(Debug, Clone)]
pub struct StereoMagRecord {
    /// Calendar year.
    pub year: u16,
    /// Day of year (1-366).
    pub doy: u16,
    /// Hour of day (0-23).
    pub hour: u8,
    /// Minute (0-59).
    pub minute: u8,
    /// Radial B-field component (nT, RTN).
    pub br: f64,
    /// Tangential B-field component (nT, RTN).
    pub bt: f64,
    /// Normal B-field component (nT, RTN).
    pub bn: f64,
    /// Scalar B-field magnitude (nT).
    pub b_magnitude: f64,
}

/// Hourly-averaged STEREO magnetic field in RTN.
#[derive(Debug, Clone)]
pub struct StereoMagHourly {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub br: f64,
    pub bt: f64,
    pub bn: f64,
    pub b_magnitude: f64,
    pub sample_count: usize,
}

/// SPDF fill value for STEREO data.
const FILL_STEREO: f64 = -1.0e31;

/// Threshold for detecting STEREO/SPDF fill.
const FILL_STEREO_THRESHOLD: f64 = -1.0e30;

/// Hard physical ceiling for B-field at STEREO-A orbit (~1 AU).
const CEIL_B: f64 = 200.0;

/// Hard physical ceiling for speed.
const CEIL_SPEED: f64 = 3000.0;

/// Hard physical ceiling for density.
const CEIL_DENSITY: f64 = 500.0;

/// Hard physical ceiling for temperature.
const CEIL_TEMP: f64 = 1.0e8;

/// Two-stage filter for STEREO/SPDF fill values.
fn parse_or_nan_stereo(v: f64, ceiling: f64) -> f64 {
    if v < FILL_STEREO_THRESHOLD || v.abs() > ceiling {
        f64::NAN
    } else {
        v
    }
}

/// Parse STEREO-A PLASTIC 1-hour tab-delimited ASCII.
///
/// Format: 2-line header (title + column names), then 24 rows/day.
/// Tab-delimited. Key columns are Np (density), Bulk_Speed, Tkin,
/// Vr_RTN, Vt_RTN, Vn_RTN.
pub fn parse_stereo_plastic(content: &str) -> Vec<StereoPlasticRecord> {
    let mut records = Vec::new();
    let mut header_lines = 0;

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // Skip first 2 header lines (title + column names).
        if header_lines < 2 {
            header_lines += 1;
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 17 {
            // Try whitespace splitting as fallback.
            let fields: Vec<&str> = line.split_whitespace().collect();
            if fields.len() < 7 {
                continue;
            }
            // Minimal whitespace format: YEAR DOY hour datetime Np Speed Tkin ...
            let year: u16 = match fields[0].parse() {
                Ok(v) if v > 1900 => v,
                _ => continue,
            };
            let doy: u16 = match fields[1].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let hour: u8 = match fields[2].parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            let np: f64 = fields.get(4).and_then(|s| s.parse().ok()).unwrap_or(FILL_STEREO);
            let speed: f64 = fields.get(5).and_then(|s| s.parse().ok()).unwrap_or(FILL_STEREO);
            let tkin: f64 = fields.get(6).and_then(|s| s.parse().ok()).unwrap_or(FILL_STEREO);

            records.push(StereoPlasticRecord {
                year,
                doy,
                hour,
                proton_density: parse_or_nan_stereo(np, CEIL_DENSITY),
                bulk_speed: parse_or_nan_stereo(speed, CEIL_SPEED),
                temperature: parse_or_nan_stereo(tkin, CEIL_TEMP),
                vr: f64::NAN,
                vt: f64::NAN,
                vn: f64::NAN,
            });
            continue;
        }

        // Tab-delimited: cols 0=YEAR, 1=DOY, 2=hour
        let year: u16 = match fields[0].trim().parse() {
            Ok(v) if v > 1900 => v,
            _ => continue,
        };
        let doy: u16 = match fields[1].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let hour: u8 = match fields[2].trim().parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        // STEREO SSC 1-hr format: col 4=Np, col 5=Bulk_Speed, col 6=Tkin
        let np: f64 = fields[4].trim().parse().unwrap_or(FILL_STEREO);
        let speed: f64 = fields[5].trim().parse().unwrap_or(FILL_STEREO);
        let tkin: f64 = fields[6].trim().parse().unwrap_or(FILL_STEREO);

        // RTN velocity: cols 14-16 (may be fill from 2020-03 onward)
        let vr: f64 = fields.get(14).and_then(|s| s.trim().parse().ok()).unwrap_or(FILL_STEREO);
        let vt: f64 = fields.get(15).and_then(|s| s.trim().parse().ok()).unwrap_or(FILL_STEREO);
        let vn: f64 = fields.get(16).and_then(|s| s.trim().parse().ok()).unwrap_or(FILL_STEREO);

        records.push(StereoPlasticRecord {
            year,
            doy,
            hour,
            proton_density: parse_or_nan_stereo(np, CEIL_DENSITY),
            bulk_speed: parse_or_nan_stereo(speed, CEIL_SPEED),
            temperature: parse_or_nan_stereo(tkin, CEIL_TEMP),
            vr: parse_or_nan_stereo(vr, CEIL_SPEED),
            vt: parse_or_nan_stereo(vt, CEIL_SPEED),
            vn: parse_or_nan_stereo(vn, CEIL_SPEED),
        });
    }
    records
}

/// Parse a STEREO PLASTIC file from disk.
pub fn parse_stereo_plastic_file(path: &Path) -> Result<Vec<StereoPlasticRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {}", e)))?;
    Ok(parse_stereo_plastic(&content))
}

/// Parse STEREO-A MAGPLASMA CDAWeb text export.
///
/// CDAWeb text format: comment header (lines starting with anything
/// non-numeric), then data rows with epoch + variables.
/// Expected columns after header: dd-mm-yyyy hh:mm:ss.ms, B_total,
/// Br, Bt, Bn, Np, Vp, Tp, ...
pub fn parse_stereo_magplasma(content: &str) -> Vec<StereoMagRecord> {
    let mut records = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        // CDAWeb text: data rows start with dd-mm-yyyy.
        // Skip lines that don't start with a digit.
        let first_char = line.chars().next().unwrap_or('#');
        if !first_char.is_ascii_digit() {
            continue;
        }

        let fields: Vec<&str> = line.split_whitespace().collect();
        // Expect: "dd-mm-yyyy" "hh:mm:ss.ms" B_total Br Bt Bn ...
        if fields.len() < 6 {
            continue;
        }

        // Parse date: dd-mm-yyyy
        let date_parts: Vec<&str> = fields[0].split('-').collect();
        if date_parts.len() < 3 {
            continue;
        }
        let day: u8 = match date_parts[0].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let month: u8 = match date_parts[1].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let year: u16 = match date_parts[2].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Parse time: hh:mm:ss.ms
        let time_parts: Vec<&str> = fields[1].split(':').collect();
        if time_parts.len() < 2 {
            continue;
        }
        let hour: u8 = time_parts[0].parse().unwrap_or(0);
        let minute: u8 = time_parts[1].parse().unwrap_or(0);

        // B-field: cols 2-5 (B_total, Br, Bt, Bn)
        let b_total: f64 = fields[2].parse().unwrap_or(FILL_STEREO);
        let br: f64 = fields[3].parse().unwrap_or(FILL_STEREO);
        let bt: f64 = fields[4].parse().unwrap_or(FILL_STEREO);
        let bn: f64 = fields[5].parse().unwrap_or(FILL_STEREO);

        let doy = month_day_to_doy(year, month, day);

        records.push(StereoMagRecord {
            year,
            doy,
            hour,
            minute,
            br: parse_or_nan_stereo(br, CEIL_B),
            bt: parse_or_nan_stereo(bt, CEIL_B),
            bn: parse_or_nan_stereo(bn, CEIL_B),
            b_magnitude: parse_or_nan_stereo(b_total, CEIL_B),
        });
    }
    records
}

/// Parse a STEREO MAGPLASMA file from disk.
pub fn parse_stereo_magplasma_file(path: &Path) -> Result<Vec<StereoMagRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {}", e)))?;
    Ok(parse_stereo_magplasma(&content))
}

// ---------------------------------------------------------------------------
// RTN -> GSE coordinate transform
// ---------------------------------------------------------------------------

/// Transform RTN (Radial-Tangential-Normal) to GSE using Earth-STEREO
/// separation angle.
///
/// RTN at STEREO-A:
///   R = radially outward from Sun through spacecraft
///   T = tangential (in ecliptic plane, prograde)
///   N = northward (ecliptic north)
///
/// GSE:
///   X = Sun-Earth line (toward Sun)
///   Y = ecliptic plane, dawn side
///   Z = ecliptic north
///
/// The rotation by separation angle `sep` (positive = STEREO ahead of Earth):
///   Bx_GSE = Br * cos(sep) - Bt * sin(sep)
///   By_GSE = Br * sin(sep) + Bt * cos(sep)
///   Bz_GSE = Bn
pub fn rtn_to_gse(br: f64, bt: f64, bn: f64, separation_angle_deg: f64) -> (f64, f64, f64) {
    let sep = separation_angle_deg.to_radians();
    let cos_sep = sep.cos();
    let sin_sep = sep.sin();

    let bx_gse = br * cos_sep - bt * sin_sep;
    let by_gse = br * sin_sep + bt * cos_sep;
    let bz_gse = bn;

    (bx_gse, by_gse, bz_gse)
}

// ---------------------------------------------------------------------------
// Merge + adapt -> OmniRecord
// ---------------------------------------------------------------------------

/// Average STEREO MAG 1-minute records into hourly bins.
pub fn average_stereo_mag_hourly(records: &[StereoMagRecord]) -> Vec<StereoMagHourly> {
    let mut bins: HashMap<(u16, u16, u8), Vec<&StereoMagRecord>> = HashMap::new();
    for r in records {
        bins.entry((r.year, r.doy, r.hour)).or_default().push(r);
    }

    let mut hourly: Vec<StereoMagHourly> = bins
        .into_iter()
        .filter_map(|((year, doy, hour), samples)| {
            let valid: Vec<&&StereoMagRecord> = samples
                .iter()
                .filter(|s| !s.br.is_nan() && !s.bt.is_nan() && !s.bn.is_nan())
                .collect();

            if valid.is_empty() {
                return None;
            }

            let n = valid.len() as f64;
            Some(StereoMagHourly {
                year,
                doy,
                hour,
                br: valid.iter().map(|s| s.br).sum::<f64>() / n,
                bt: valid.iter().map(|s| s.bt).sum::<f64>() / n,
                bn: valid.iter().map(|s| s.bn).sum::<f64>() / n,
                b_magnitude: valid.iter().map(|s| s.b_magnitude).sum::<f64>() / n,
                sample_count: valid.len(),
            })
        })
        .collect();

    hourly.sort_by(|a, b| {
        (a.year, a.doy, a.hour).cmp(&(b.year, b.doy, b.hour))
    });
    hourly
}

/// Convert merged STEREO PLASTIC+MAG data to OmniRecords.
///
/// Applies RTN->GSE transform to B-field using the provided separation angle.
/// Plasma parameters map directly (scalar speed, density, temperature).
pub fn stereo_to_omni(
    plastic: &[StereoPlasticRecord],
    mag_hourly: &[StereoMagHourly],
    separation_angle_deg: f64,
) -> Vec<OmniRecord> {
    // Build mag lookup: (year, doy, hour) -> StereoMagHourly
    let mut mag_map: HashMap<(u16, u16, u8), &StereoMagHourly> = HashMap::new();
    for m in mag_hourly {
        mag_map.insert((m.year, m.doy, m.hour), m);
    }

    // Collect all unique time keys.
    let mut all_keys: Vec<(u16, u16, u8)> = plastic
        .iter()
        .map(|p| (p.year, p.doy, p.hour))
        .chain(mag_hourly.iter().map(|m| (m.year, m.doy, m.hour)))
        .collect();
    all_keys.sort();
    all_keys.dedup();

    // Build plasma lookup.
    let mut plasma_map: HashMap<(u16, u16, u8), &StereoPlasticRecord> = HashMap::new();
    for p in plastic {
        plasma_map.insert((p.year, p.doy, p.hour), p);
    }

    all_keys
        .iter()
        .map(|&(year, doy, hour)| {
            let (density, speed, temp) = plasma_map
                .get(&(year, doy, hour))
                .map(|p| (p.proton_density, p.bulk_speed, p.temperature))
                .unwrap_or((f64::NAN, f64::NAN, f64::NAN));

            let (bx, by, bz, bmag) = mag_map
                .get(&(year, doy, hour))
                .map(|m| {
                    let (bx_gse, by_gse, bz_gse) =
                        rtn_to_gse(m.br, m.bt, m.bn, separation_angle_deg);
                    (bx_gse, by_gse, bz_gse, m.b_magnitude)
                })
                .unwrap_or((f64::NAN, f64::NAN, f64::NAN, f64::NAN));

            OmniRecord {
                year,
                doy,
                hour,
                b_magnitude: bmag,
                bx_gse: bx,
                by_gse: by,
                bz_gse: bz,
                proton_temperature: temp,
                proton_density: density,
                bulk_speed: speed,
                flow_pressure: f64::NAN,
                plasma_beta: f64::NAN,
                alfven_mach: f64::NAN,
                dst_index: f64::NAN,
                ae_index: f64::NAN,
                kp_times_10: 0,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Convert (month, day) to day-of-year (1-based).
fn month_day_to_doy(year: u16, month: u8, day: u8) -> u16 {
    let leap = year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400));
    let days_before: [u16; 12] = if leap {
        [0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    } else {
        [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    };
    let m = (month as usize).saturating_sub(1).min(11);
    days_before[m] + day as u16
}

// ---------------------------------------------------------------------------
// DatasetProviders
// ---------------------------------------------------------------------------

/// STEREO-A PLASTIC dataset provider (1-hour plasma).
pub struct StereoPlasticProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for StereoPlasticProvider {
    fn default() -> Self {
        Self {
            year_start: 2024,
            year_end: 2024,
        }
    }
}

impl DatasetProvider for StereoPlasticProvider {
    fn name(&self) -> &str {
        "STEREO-A PLASTIC 1-hour Plasma"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("stereo_plastic");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let n_days = if year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400)) {
                366
            } else {
                365
            };
            for doy in 1..=n_days {
                let fname = format!(
                    "STA_L2_PLA_1DMax_1hr_{}{:03}_{:03}_V10.txt",
                    year,
                    // Convert DOY to MMDD for filename
                    doy, doy,
                );
                let output = dir.join(&fname);
                if config.skip_existing && output.exists() {
                    continue;
                }
                // STEREO SSC URL pattern
                let url = format!(
                    "https://stereo-ssc.nascom.nasa.gov/data/ins_data/plastic/level2/Protons/Derived_from_1D_Maxwellian/ASCII/1hr/A/{}/{}",
                    year, fname,
                );
                match download_to_file(&url, &output) {
                    Ok(bytes) if bytes > 0 => {
                        log::info!("Saved {} ({} bytes)", fname, bytes);
                    }
                    _ => {
                        log::debug!("STEREO PLASTIC {} not found", fname);
                    }
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("stereo_plastic").exists()
    }
}

/// STEREO-A IMPACT/MAG dataset provider (MAGPLASMA via CDAWeb).
pub struct StereoMagProvider {
    pub year_start: u16,
    pub year_end: u16,
}

impl Default for StereoMagProvider {
    fn default() -> Self {
        Self {
            year_start: 2024,
            year_end: 2024,
        }
    }
}

impl DatasetProvider for StereoMagProvider {
    fn name(&self) -> &str {
        "STEREO-A IMPACT/MAG MAGPLASMA"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("stereo_impact");
        std::fs::create_dir_all(&dir)?;
        // MAGPLASMA CDF files at SPDF are yearly (~75 MB).
        // For now, create directory and rely on CDAWeb REST API for text export.
        log::info!(
            "STEREO MAGPLASMA: use CDAWeb REST API or bin/fetch_stereo.py for text export"
        );
        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("stereo_impact").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Synthetic STEREO PLASTIC 1-hr tab-delimited data.
    const SAMPLE_PLASTIC: &str = "\
STEREO-A PLASTIC L2 1DMax 1hr proton moments\n\
YEAR\tDOY\thour\tdatetime\tNp\tBulk_Speed\tTkin\tv_th\tflow_ang_inst\tflow_ang_hertn\tflow_ang_rtn\tflow_ang_chi2\tVx_hertn\tVy_hertn\tVr_rtn\tVt_rtn\tVn_rtn\n\
2024\t120\t0\t2024-04-29T00:30:00\t5.20\t410.5\t85000.0\t40.2\t0.5\t1.2\t0.8\t1.5\t-408.0\t25.0\t-410.0\t15.0\t-3.0\n\
2024\t120\t1\t2024-04-29T01:30:00\t4.80\t425.3\t92000.0\t42.1\t0.3\t1.0\t0.6\t1.2\t-422.0\t28.0\t-425.0\t18.0\t-2.0\n\
2024\t120\t2\t2024-04-29T02:30:00\t-1.0E+31\t-1.0E+31\t-1.0E+31\t-1.0E+31\t-1.0E+31\t-1.0E+31\t-1.0E+31\t-1.0E+31\t-1.0E+31\t-1.0E+31\t-1.0E+31\t-1.0E+31\t-1.0E+31\n\
";

    // Synthetic STEREO MAGPLASMA CDAWeb text export.
    const SAMPLE_MAGPLASMA: &str = "\
# CDAWeb text export
# Dataset: STA_L2_MAGPLASMA_1M
# Variables: B_total, Br_RTN, Bt_RTN, Bn_RTN
29-04-2024 00:00:30.000       6.500          5.200         -3.100          2.400
29-04-2024 00:01:30.000       6.800          5.400         -3.300          2.600
29-04-2024 00:02:30.000   -1.00E+31      -1.00E+31      -1.00E+31      -1.00E+31
29-04-2024 01:00:30.000       7.100          5.800         -3.500          2.200
";

    #[test]
    fn test_parse_stereo_plastic_basic() {
        let records = parse_stereo_plastic(SAMPLE_PLASTIC);
        assert_eq!(records.len(), 3);

        let r = &records[0];
        assert_eq!(r.year, 2024);
        assert_eq!(r.doy, 120);
        assert_eq!(r.hour, 0);
        assert!((r.proton_density - 5.20).abs() < 0.01);
        assert!((r.bulk_speed - 410.5).abs() < 0.1);
        assert!((r.temperature - 85000.0).abs() < 1.0);
        assert!((r.vr - (-410.0)).abs() < 0.1);
        assert!((r.vt - 15.0).abs() < 0.1);
    }

    #[test]
    fn test_stereo_fill_values() {
        let records = parse_stereo_plastic(SAMPLE_PLASTIC);
        let r = &records[2]; // fill row
        assert!(r.proton_density.is_nan(), "density should be NaN");
        assert!(r.bulk_speed.is_nan(), "speed should be NaN");
        assert!(r.temperature.is_nan(), "temp should be NaN");
        assert!(r.vr.is_nan(), "vr should be NaN");
    }

    #[test]
    fn test_parse_stereo_impact_mag_basic() {
        let records = parse_stereo_magplasma(SAMPLE_MAGPLASMA);
        assert_eq!(records.len(), 4);

        let r = &records[0];
        assert_eq!(r.year, 2024);
        assert_eq!(r.doy, 120); // Apr 29 in leap year = DOY 120
        assert_eq!(r.hour, 0);
        assert!((r.b_magnitude - 6.500).abs() < 0.001);
        assert!((r.br - 5.200).abs() < 0.001);
        assert!((r.bt - (-3.100)).abs() < 0.001);
        assert!((r.bn - 2.400).abs() < 0.001);
    }

    #[test]
    fn test_stereo_magplasma_fill() {
        let records = parse_stereo_magplasma(SAMPLE_MAGPLASMA);
        let r = &records[2]; // fill row
        assert!(r.br.is_nan(), "Br should be NaN");
        assert!(r.bt.is_nan(), "Bt should be NaN");
        assert!(r.bn.is_nan(), "Bn should be NaN");
    }

    #[test]
    fn test_rtn_to_gse_zero_separation() {
        // At 0 degree separation, RTN = GSE (STEREO at Earth position)
        let (bx, by, bz) = rtn_to_gse(5.0, -3.0, 2.0, 0.0);
        assert!((bx - 5.0).abs() < 1e-10, "Bx should equal Br at sep=0");
        assert!((by - (-3.0)).abs() < 1e-10, "By should equal Bt at sep=0");
        assert!((bz - 2.0).abs() < 1e-10, "Bz should equal Bn at sep=0");
    }

    #[test]
    fn test_rtn_to_gse_90_degrees() {
        // At 90 degrees: Bx = -Bt, By = Br
        let (bx, by, bz) = rtn_to_gse(5.0, -3.0, 2.0, 90.0);
        assert!((bx - 3.0).abs() < 1e-10, "Bx = -Bt at 90 deg: got {}", bx);
        assert!((by - 5.0).abs() < 1e-10, "By = Br at 90 deg: got {}", by);
        assert!((bz - 2.0).abs() < 1e-10, "Bz = Bn always");
    }

    #[test]
    fn test_stereo_to_omni_adapter() {
        let plastic = parse_stereo_plastic(SAMPLE_PLASTIC);
        let mag = parse_stereo_magplasma(SAMPLE_MAGPLASMA);
        let mag_hourly = average_stereo_mag_hourly(&mag);
        let omni = stereo_to_omni(&plastic, &mag_hourly, 15.0);

        assert!(!omni.is_empty());

        // Find DOY 120, hour 0 (should have both plasma and B-field).
        let r = omni
            .iter()
            .find(|r| r.doy == 120 && r.hour == 0)
            .expect("should have DOY 120 hour 0");

        // Plasma from PLASTIC
        assert!(!r.proton_density.is_nan(), "density from PLASTIC");
        assert!(!r.bulk_speed.is_nan(), "speed from PLASTIC");

        // B-field from MAG (RTN->GSE transformed)
        assert!(!r.bx_gse.is_nan(), "Bx from MAG");
        assert!(!r.by_gse.is_nan(), "By from MAG");
        assert!(!r.bz_gse.is_nan(), "Bz from MAG");

        // B-field should differ from raw RTN due to 15-degree rotation
        // (At 15 deg, cos=0.966, sin=0.259 -- noticeable mixing)
    }

    #[test]
    fn test_stereo_velocity_fill_post_2020() {
        // Simulate post-2020 data where RTN velocity vectors are fill.
        let data = "\
STEREO title\n\
YEAR\tDOY\thour\tdt\tNp\tBulk\tTkin\tvth\tfa1\tfa2\tfa3\tfa4\tvx\tvy\tVr\tVt\tVn\n\
2021\t100\t0\t2021-04-10T00:30:00\t6.00\t380.0\t70000.0\t38.0\t0.5\t1.0\t0.5\t1.0\t-378.0\t20.0\t-1.0E+31\t-1.0E+31\t-1.0E+31\n\
";
        let records = parse_stereo_plastic(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        // Scalar parameters should be valid
        assert!(!r.proton_density.is_nan());
        assert!(!r.bulk_speed.is_nan());
        // RTN velocity should be NaN (fill)
        assert!(r.vr.is_nan(), "Vr should be NaN post-2020");
        assert!(r.vt.is_nan(), "Vt should be NaN post-2020");
        assert!(r.vn.is_nan(), "Vn should be NaN post-2020");
    }
}
