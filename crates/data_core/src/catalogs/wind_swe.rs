//! WIND spacecraft SWE + MFI data providers.
//!
//! WIND orbits L1 independently of ACE, providing cross-validation
//! of solar wind plasma and magnetic field measurements.
//!
//! **MFI** (Magnetic Field Investigation): 1-hour ASCII averages.
//!   Source: <https://spdf.gsfc.nasa.gov/pub/data/wind/mfi/ascii/1hour_ascii/>
//!   Reference: Lepping et al. (1995), Space Sci. Rev. 71, 207
//!   Format: 21 whitespace-delimited columns, NO header lines.
//!   Fill: -1.00E+31 for all magnetic field parameters.
//!
//! **SWE** (Solar Wind Experiment): KP unspiked ~92-sec cadence.
//!   Source: <https://spdf.gsfc.nasa.gov/pub/data/wind/swe/ascii/swe_kp_unspike/>
//!   Reference: Ogilvie et al. (1995), Space Sci. Rev. 71, 55
//!   Format: 12 whitespace-delimited columns, NO header lines.
//!   Fill: varies per column (see constants below).

use crate::{catalogs::omni::OmniRecord, fetcher::FetchError};
use std::{collections::HashMap, path::Path};

// ---------------------------------------------------------------------------
// WIND MFI (magnetic field)
// ---------------------------------------------------------------------------

/// A single hourly WIND MFI magnetic field measurement.
///
/// Columns (all whitespace-delimited, no header):
///   0: Year (I4)
///   1: Month (I3)
///   2: Day (I3)
///   3: Hour (I3)
///   4: Min (I3, midpoint of average -- typically 30)
///   5: Bx GSE (F10.3, nT)
///   6: By GSE (F10.3, nT)
///   7: Bz GSE (F10.3, nT)
///   8: By GSM (F10.3, nT) -- skipped
///   9: Bz GSM (F10.3, nT) -- skipped
///  10: |B| (F10.3, nT)
///  11-14: RMS components (nT) -- skipped
///  15: N fine points (I6)
///  16-20: S/C position (Re) -- skipped
#[derive(Debug, Clone)]
pub struct WindMfiRecord {
    /// Calendar year.
    pub year: u16,
    /// Month (1-12).
    pub month: u8,
    /// Day of month (1-31).
    pub day: u8,
    /// Hour of day (0-23).
    pub hour: u8,
    /// IMF Bx in GSE coordinates (nT).
    pub bx_gse: f64,
    /// IMF By in GSE coordinates (nT).
    pub by_gse: f64,
    /// IMF Bz in GSE coordinates (nT).
    pub bz_gse: f64,
    /// Scalar magnetic field magnitude (nT).
    pub b_magnitude: f64,
    /// Number of fine-scale samples in average.
    pub n_points: u32,
}

#[cfg(feature = "fetch")]
pub use super::wind_swe_fetch::{
    WindAmdaMagRecord, WindAmdaPlasmaRecord, WindAmdaProvider, WindMfiProvider, WindSweProvider,
    merge_wind_amda, parse_wind_amda_mfi, parse_wind_amda_swe,
};

/// SPDF fill value for all WIND MFI fields.
const FILL_SPDF: f64 = -1.0e31;

/// Threshold for detecting SPDF fill (generous to catch -1.00E+31 variants).
const FILL_SPDF_THRESHOLD: f64 = -1.0e30;

/// Hard physical ceiling for B-field at 1 AU (nT).
const CEIL_B: f64 = 200.0;

/// Two-stage filter for SPDF data: reject fill and unphysical values.
fn parse_or_nan_spdf(v: f64, ceiling: f64) -> f64 {
    if v < FILL_SPDF_THRESHOLD || v.abs() > ceiling {
        f64::NAN
    } else {
        v
    }
}

/// Parse WIND MFI 1-hour ASCII data.
///
/// No header lines -- data starts on line 1. Monthly files named
/// `YYYYMM_wind_mag_1hour.asc`.
pub fn parse_wind_mfi(content: &str) -> Vec<WindMfiRecord> {
    let mut records = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 16 {
            continue;
        }

        let year: u16 = match fields[0].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let month: u8 = match fields[1].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let day: u8 = match fields[2].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let hour: u8 = match fields[3].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        let bx_raw: f64 = fields[5].parse().unwrap_or(FILL_SPDF);
        let by_raw: f64 = fields[6].parse().unwrap_or(FILL_SPDF);
        let bz_raw: f64 = fields[7].parse().unwrap_or(FILL_SPDF);
        let bmag_raw: f64 = fields[10].parse().unwrap_or(FILL_SPDF);
        let n_points: u32 = fields[15].parse().unwrap_or(0);

        records.push(WindMfiRecord {
            year,
            month,
            day,
            hour,
            bx_gse: parse_or_nan_spdf(bx_raw, CEIL_B),
            by_gse: parse_or_nan_spdf(by_raw, CEIL_B),
            bz_gse: parse_or_nan_spdf(bz_raw, CEIL_B),
            b_magnitude: parse_or_nan_spdf(bmag_raw, CEIL_B),
            n_points,
        });
    }
    records
}

/// Parse a WIND MFI file from disk.
pub fn parse_wind_mfi_file(path: &Path) -> Result<Vec<WindMfiRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {}", e)))?;
    Ok(parse_wind_mfi(&content))
}

// ---------------------------------------------------------------------------
// WIND SWE (plasma)
// ---------------------------------------------------------------------------

/// A single ~92-second WIND SWE KP unspiked measurement.
///
/// Columns (all whitespace-delimited, no header):
///   0: Year (I5)
///   1: Decimal DOY (F12.6, noon Jan 1 = 1.5)
///   2: Flow speed (F8.1, km/s)
///   3: Vx GSE (F8.1, km/s)
///   4: Vy GSE (F8.1, km/s)
///   5: Vz GSE (F8.1, km/s)
///   6: Temperature (F9.0, K)
///   7: Proton density (F7.2, cm^-3)
///   8: Alpha/proton ratio (F6.3) -- all fill (9.999)
///   9: X GSE (F8.2, Re)
///  10: Y GSE (F8.2, Re)
///  11: Z GSE (F8.2, Re)
#[derive(Debug, Clone)]
pub struct WindSweRecord {
    /// Calendar year.
    pub year: u16,
    /// Decimal day of year (noon Jan 1 = 1.5).
    pub decimal_doy: f64,
    /// Bulk solar wind speed, scalar (km/s).
    pub flow_speed: f64,
    /// Vx in GSE (km/s) -- predominantly negative (anti-sunward).
    pub vx_gse: f64,
    /// Proton temperature (K).
    pub temperature: f64,
    /// Proton number density (cm^-3).
    pub proton_density: f64,
}

// SWE fill values per column.
const FILL_SWE_SPEED: f64 = 99999.9;
const FILL_SWE_TEMP: f64 = 9999999.0;
const FILL_SWE_DENSITY: f64 = 999.99;

// SWE physical ceilings.
const CEIL_SPEED: f64 = 3000.0;
const CEIL_TEMP: f64 = 1.0e8;
const CEIL_DENSITY: f64 = 500.0;

/// Two-stage filter for SWE data.
fn parse_or_nan_swe(v: f64, fill: f64, ceiling: f64) -> f64 {
    if (v - fill).abs() < 1.0 || v.abs() > ceiling || v < 0.0 {
        f64::NAN
    } else {
        v
    }
}

/// Parse WIND SWE KP unspiked ASCII data.
///
/// No header lines -- data starts on line 1. Annual files named
/// `wind_kp_unspike{YYYY}.txt`.
pub fn parse_wind_swe(content: &str) -> Vec<WindSweRecord> {
    let mut records = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 8 {
            continue;
        }

        let year: u16 = match fields[0].parse() {
            Ok(v) if v > 1900 && v < 2100 => v,
            _ => continue,
        };
        let decimal_doy: f64 = match fields[1].parse() {
            Ok(v) if v > 0.0 && v < 367.0 => v,
            _ => continue,
        };

        let flow_speed: f64 = fields[2].parse().unwrap_or(FILL_SWE_SPEED);
        let vx_gse: f64 = fields[3].parse().unwrap_or(FILL_SWE_SPEED);
        let temperature: f64 = fields[6].parse().unwrap_or(FILL_SWE_TEMP);
        let density: f64 = fields[7].parse().unwrap_or(FILL_SWE_DENSITY);

        records.push(WindSweRecord {
            year,
            decimal_doy,
            flow_speed: parse_or_nan_swe(flow_speed, FILL_SWE_SPEED, CEIL_SPEED),
            vx_gse: if (vx_gse - FILL_SWE_SPEED).abs() < 1.0 {
                f64::NAN
            } else {
                vx_gse
            },
            temperature: parse_or_nan_swe(temperature, FILL_SWE_TEMP, CEIL_TEMP),
            proton_density: parse_or_nan_swe(density, FILL_SWE_DENSITY, CEIL_DENSITY),
        });
    }
    records
}

/// Parse a WIND SWE file from disk.
pub fn parse_wind_swe_file(path: &Path) -> Result<Vec<WindSweRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {}", e)))?;
    Ok(parse_wind_swe(&content))
}

// ---------------------------------------------------------------------------
// Merge SWE + MFI -> OmniRecord
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

/// Average SWE records into hourly bins keyed by (year, doy, hour).
fn average_swe_hourly(records: &[WindSweRecord]) -> HashMap<(u16, u16, u8), (f64, f64, f64)> {
    // Key: (year, doy, hour) -> (sum_density, sum_speed, sum_temp, count)
    #[allow(clippy::type_complexity)]
    let mut bins: HashMap<(u16, u16, u8), (f64, f64, f64, f64)> = HashMap::new();

    for r in records {
        let doy = r.decimal_doy.floor() as u16;
        let frac = r.decimal_doy - doy as f64;
        let hour = (frac * 24.0).floor() as u8;

        let entry = bins
            .entry((r.year, doy, hour))
            .or_insert((0.0, 0.0, 0.0, 0.0));
        let mut count_inc = 0.0;
        if !r.proton_density.is_nan() {
            entry.0 += r.proton_density;
            count_inc = 1.0;
        }
        if !r.flow_speed.is_nan() {
            entry.1 += r.flow_speed;
            count_inc = 1.0;
        }
        if !r.temperature.is_nan() {
            entry.2 += r.temperature;
            count_inc = 1.0;
        }
        if count_inc > 0.0 {
            entry.3 += 1.0;
        }
    }

    bins.into_iter()
        .filter_map(|(key, (d, s, t, n))| {
            if n > 0.0 {
                Some((key, (d / n, s / n, t / n)))
            } else {
                None
            }
        })
        .collect()
}

/// Merge WIND SWE (plasma) and MFI (magnetic field) into OmniRecords.
///
/// Time-aligns by (year, doy, hour). MFI provides B-field, SWE provides
/// plasma. If only one source has data for a given hour, the other fields
/// are NaN (triggering fallback defaults in UnitConversion).
pub fn merge_wind_swe_mfi(swe: &[WindSweRecord], mfi: &[WindMfiRecord]) -> Vec<OmniRecord> {
    let swe_hourly = average_swe_hourly(swe);

    // Build MFI lookup: (year, doy, hour) -> WindMfiRecord
    let mut mfi_map: HashMap<(u16, u16, u8), &WindMfiRecord> = HashMap::new();
    for r in mfi {
        let doy = month_day_to_doy(r.year, r.month, r.day);
        mfi_map.insert((r.year, doy, r.hour), r);
    }

    // Collect all unique time keys from both sources.
    let mut all_keys: Vec<(u16, u16, u8)> =
        swe_hourly.keys().chain(mfi_map.keys()).copied().collect();
    all_keys.sort();
    all_keys.dedup();

    all_keys
        .iter()
        .map(|&(year, doy, hour)| {
            let (density, speed, temp) = swe_hourly.get(&(year, doy, hour)).copied().unwrap_or((
                f64::NAN,
                f64::NAN,
                f64::NAN,
            ));

            let (bx, by, bz, bmag) = mfi_map
                .get(&(year, doy, hour))
                .map(|m| (m.bx_gse, m.by_gse, m.bz_gse, m.b_magnitude))
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
                r_au: 1.0,
                lat_deg: f64::NAN,
                lon_deg: f64::NAN,
            }
        })
        .collect()
}

/// Convert WIND MFI-only records to OmniRecords (B-field, NaN plasma).
pub fn wind_mfi_to_omni(records: &[WindMfiRecord]) -> Vec<OmniRecord> {
    records
        .iter()
        .map(|r| {
            let doy = month_day_to_doy(r.year, r.month, r.day);
            OmniRecord {
                year: r.year,
                doy,
                hour: r.hour,
                b_magnitude: r.b_magnitude,
                bx_gse: r.bx_gse,
                by_gse: r.by_gse,
                bz_gse: r.bz_gse,
                proton_temperature: f64::NAN,
                proton_density: f64::NAN,
                bulk_speed: f64::NAN,
                flow_pressure: f64::NAN,
                plasma_beta: f64::NAN,
                alfven_mach: f64::NAN,
                dst_index: f64::NAN,
                ae_index: f64::NAN,
                kp_times_10: 0,
                r_au: 1.0,
                lat_deg: f64::NAN,
                lon_deg: f64::NAN,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Fetch/provider support moved to wind_swe_fetch (feature-gated on `fetch`).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Knudsen number diagnostics
// ---------------------------------------------------------------------------

/// Collisionality regime classification based on Knudsen number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KnudsenRegime {
    /// Kn < 10: fluid approximation valid (LBM safe).
    Fluid,
    /// 10 <= Kn < 100: transitional (LBM results should be interpreted with caution).
    Transitional,
    /// Kn >= 100: kinetic regime (LBM not valid, PIC/Vlasov required).
    Kinetic,
}

/// Compute the Knudsen number for solar wind protons at 1 AU.
///
/// Kn = lambda_mfp / L_gradient where:
///   lambda_mfp = proton Coulomb mean free path
///   L_gradient = ion inertial length d_i = c / omega_pi
///
/// The proton mean free path in the solar wind is dominated by Coulomb
/// collisions. At 1 AU with T ~ 10^5 K and n ~ 5 cm^-3:
///   lambda_mfp ~ 2.4e11 * T^2 / (n * ln(Lambda)) meters
///   where ln(Lambda) ~ 20 is the Coulomb logarithm.
///
/// The ion inertial length (gradient scale):
///   d_i = c / omega_pi = 228 / sqrt(n_p) km
///   where n_p is in cm^-3.
///
/// Returns NaN if any input is NaN or non-physical (n <= 0, T <= 0).
pub fn knudsen_number(n_p_cm3: f64, t_p_k: f64) -> f64 {
    if n_p_cm3.is_nan() || t_p_k.is_nan() || n_p_cm3 <= 0.0 || t_p_k <= 0.0 {
        return f64::NAN;
    }

    // Coulomb logarithm (weakly dependent on n, T; ~20 for solar wind)
    let ln_lambda = 20.0;

    // Proton mean free path (m): lambda = 2.4e11 * T^2 / (n * ln_lambda)
    // From Spitzer (1962), scaled for proton-proton collisions.
    // n in m^-3 = n_cm3 * 1e6
    let n_m3 = n_p_cm3 * 1.0e6;
    let lambda_mfp = 2.4e11 * t_p_k * t_p_k / (n_m3 * ln_lambda);

    // Ion inertial length (m): d_i = c / omega_pi = 228 / sqrt(n_cm3) * 1e3
    // where omega_pi = sqrt(n * e^2 / (m_p * epsilon_0))
    let d_i = 228.0e3 / n_p_cm3.sqrt();

    lambda_mfp / d_i
}

/// Classify the Knudsen number into a collisionality regime.
pub fn classify_knudsen(kn: f64) -> KnudsenRegime {
    if kn.is_nan() || kn < 10.0 {
        KnudsenRegime::Fluid
    } else if kn < 100.0 {
        KnudsenRegime::Transitional
    } else {
        KnudsenRegime::Kinetic
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Based on real WIND MFI 1-hour data (Jan 2026, from SPDF).
    // Format: no header, 21 whitespace-delimited columns.
    const SAMPLE_MFI: &str = "\
2026  1  1  0 30     1.024    -4.711     4.441    -5.661     3.172     7.897    1.475    3.006    2.887    0.585 38836  252.620  -42.981   12.106  -44.634    1.291
2026  1  1  1 30     2.150    -3.800     3.200    -4.500     2.100     6.500    1.200    2.500    2.300    0.400 35000  252.600  -42.900   12.100  -44.600    1.280
2026  1  1  2 30  -1.00E+31  -1.00E+31  -1.00E+31  -1.00E+31  -1.00E+31  -1.00E+31  -1.00E+31  -1.00E+31  -1.00E+31  -1.00E+31     0  252.580  -42.850   12.090  -44.560    1.270
";

    // Based on real WIND SWE KP unspiked data (Jan 2026, from SPDF).
    // Format: no header, 12 whitespace-delimited columns.
    const SAMPLE_SWE: &str = "\
 2026    1.002691   482.9  -480.0    51.1    12.6  188510.   5.79 9.999  252.63  -42.89   12.11
 2026    1.044560   475.2  -472.5    48.3    11.2  175000.   6.10 9.999  252.60  -42.85   12.10
 2026    1.085000   490.0  -487.0    52.0    13.0  195000.   5.50 9.999  252.58  -42.80   12.09
 2026    1.502000   460.0  -457.0    45.0    10.0  160000.   7.20 9.999  252.55  -42.75   12.08
 2026    2.010000 99999.9 99999.9 99999.9 99999.9 9999999. 999.99 9.999 9999.99 9999.99 9999.99
";

    #[test]
    fn test_parse_wind_mfi_basic() {
        let records = parse_wind_mfi(SAMPLE_MFI);
        assert_eq!(records.len(), 3);

        let r = &records[0];
        assert_eq!(r.year, 2026);
        assert_eq!(r.month, 1);
        assert_eq!(r.day, 1);
        assert_eq!(r.hour, 0);
        assert!((r.bx_gse - 1.024).abs() < 0.001);
        assert!((r.by_gse - (-4.711)).abs() < 0.001);
        assert!((r.bz_gse - 4.441).abs() < 0.001);
        assert!((r.b_magnitude - 7.897).abs() < 0.001);
        assert_eq!(r.n_points, 38836);
    }

    #[test]
    fn test_parse_wind_mfi_fill() {
        let records = parse_wind_mfi(SAMPLE_MFI);
        let r = &records[2]; // fill row
        assert!(r.bx_gse.is_nan(), "Bx should be NaN for SPDF fill");
        assert!(r.by_gse.is_nan(), "By should be NaN for SPDF fill");
        assert!(r.bz_gse.is_nan(), "Bz should be NaN for SPDF fill");
        assert!(r.b_magnitude.is_nan(), "|B| should be NaN for SPDF fill");
        assert_eq!(r.n_points, 0);
    }

    #[test]
    fn test_parse_wind_mfi_no_header() {
        // Verify data is parsed from line 1 (no header to skip).
        let data = "2024  6 15 12 30     5.000    -3.000     2.000    -3.500     1.500     6.500    0.800    1.200    1.000    0.300 40000  200.000  -30.000   10.000  -32.000    5.000\n";
        let records = parse_wind_mfi(data);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].year, 2024);
        assert_eq!(records[0].month, 6);
        assert_eq!(records[0].day, 15);
    }

    #[test]
    fn test_parse_wind_swe_basic() {
        let records = parse_wind_swe(SAMPLE_SWE);
        assert_eq!(records.len(), 5);

        let r = &records[0];
        assert_eq!(r.year, 2026);
        assert!((r.decimal_doy - 1.002691).abs() < 1e-6);
        assert!((r.flow_speed - 482.9).abs() < 0.1);
        assert!((r.vx_gse - (-480.0)).abs() < 0.1);
        assert!((r.temperature - 188510.0).abs() < 1.0);
        assert!((r.proton_density - 5.79).abs() < 0.01);
    }

    #[test]
    fn test_parse_wind_swe_fill() {
        let records = parse_wind_swe(SAMPLE_SWE);
        let r = &records[4]; // fill row
        assert!(r.flow_speed.is_nan(), "speed should be NaN for fill");
        assert!(r.temperature.is_nan(), "temp should be NaN for fill");
        assert!(r.proton_density.is_nan(), "density should be NaN for fill");
    }

    #[test]
    fn test_merge_wind_swe_mfi_hourly() {
        let mfi = parse_wind_mfi(SAMPLE_MFI);
        let swe = parse_wind_swe(SAMPLE_SWE);
        let merged = merge_wind_swe_mfi(&swe, &mfi);

        // Should have at least 2 merged records (DOY 1 hours 0 and 1 have both).
        assert!(merged.len() >= 2, "merged len: {}", merged.len());

        // Find DOY 1, hour 0 record.
        let r = merged
            .iter()
            .find(|r| r.doy == 1 && r.hour == 0)
            .expect("should have DOY 1 hour 0");

        // B-field from MFI
        assert!(!r.bx_gse.is_nan(), "Bx should be from MFI");
        assert!((r.bx_gse - 1.024).abs() < 0.001);

        // Plasma from SWE (averaged over ~92s samples in hour 0)
        assert!(!r.proton_density.is_nan(), "density should be from SWE");
        assert!(!r.bulk_speed.is_nan(), "speed should be from SWE");
    }

    #[test]
    fn test_wind_to_omni_adapter() {
        let mfi = parse_wind_mfi(SAMPLE_MFI);
        let omni = wind_mfi_to_omni(&mfi);

        assert_eq!(omni.len(), 3);
        let r = &omni[0];
        // B-field should be populated (GSE direct mapping)
        assert!(!r.bx_gse.is_nan());
        // Plasma should be NaN (MFI has no plasma data)
        assert!(r.proton_density.is_nan());
        assert!(r.bulk_speed.is_nan());
        // DOY computed from month/day: Jan 1 = DOY 1
        assert_eq!(r.doy, 1);
    }

    #[test]
    fn test_wind_physical_ranges() {
        let mfi = parse_wind_mfi(SAMPLE_MFI);
        for r in &mfi {
            if !r.b_magnitude.is_nan() {
                assert!(
                    r.b_magnitude > 0.1 && r.b_magnitude < 100.0,
                    "|B| out of range: {}",
                    r.b_magnitude,
                );
            }
        }
        let swe = parse_wind_swe(SAMPLE_SWE);
        for r in &swe {
            if !r.flow_speed.is_nan() {
                assert!(
                    r.flow_speed > 100.0 && r.flow_speed < 1500.0,
                    "speed out of range: {}",
                    r.flow_speed,
                );
            }
            if !r.proton_density.is_nan() {
                assert!(
                    r.proton_density > 0.1 && r.proton_density < 200.0,
                    "density out of range: {}",
                    r.proton_density,
                );
            }
        }
    }

    #[test]
    fn test_month_day_to_doy() {
        // Non-leap year
        assert_eq!(month_day_to_doy(2023, 1, 1), 1);
        assert_eq!(month_day_to_doy(2023, 2, 1), 32);
        assert_eq!(month_day_to_doy(2023, 3, 1), 60);
        assert_eq!(month_day_to_doy(2023, 12, 31), 365);

        // Leap year
        assert_eq!(month_day_to_doy(2024, 3, 1), 61);
        assert_eq!(month_day_to_doy(2024, 12, 31), 366);
    }

    // ---- Knudsen number tests ----

    #[test]
    fn test_knudsen_quiet_solar_wind() {
        // Quiet solar wind: n ~ 5 cm^-3, T ~ 1e5 K
        // Expected Kn >> 100 (collisionless plasma)
        let kn = knudsen_number(5.0, 1.0e5);
        assert!(kn.is_finite(), "Kn should be finite");
        assert!(kn > 100.0, "quiet SW should be kinetic regime: Kn={kn:.1}");
        assert_eq!(classify_knudsen(kn), KnudsenRegime::Kinetic);
    }

    #[test]
    fn test_knudsen_dense_cool_plasma() {
        // Dense cool plasma: n ~ 100 cm^-3, T ~ 1e4 K
        // Higher density + lower T -> shorter mean free path, possibly transitional
        let kn = knudsen_number(100.0, 1.0e4);
        assert!(kn.is_finite(), "Kn should be finite");
        // Even dense solar wind is kinetic at 1 AU
        assert!(kn > 1.0, "dense plasma still has Kn > 1: Kn={kn:.1}");
    }

    #[test]
    fn test_knudsen_nan_inputs() {
        assert!(knudsen_number(f64::NAN, 1e5).is_nan());
        assert!(knudsen_number(5.0, f64::NAN).is_nan());
        assert!(knudsen_number(-1.0, 1e5).is_nan());
        assert!(knudsen_number(5.0, 0.0).is_nan());
    }

    #[test]
    fn test_knudsen_regime_classification() {
        assert_eq!(classify_knudsen(5.0), KnudsenRegime::Fluid);
        assert_eq!(classify_knudsen(50.0), KnudsenRegime::Transitional);
        assert_eq!(classify_knudsen(500.0), KnudsenRegime::Kinetic);
        assert_eq!(classify_knudsen(f64::NAN), KnudsenRegime::Fluid); // NaN -> Fluid (safe default)
    }

    #[test]
    fn test_knudsen_monotonic_with_temperature() {
        // Higher T -> longer mean free path -> higher Kn
        let kn_low = knudsen_number(5.0, 5.0e4);
        let kn_high = knudsen_number(5.0, 2.0e5);
        assert!(
            kn_high > kn_low,
            "Kn should increase with T: low={kn_low:.1}, high={kn_high:.1}"
        );
    }
}
