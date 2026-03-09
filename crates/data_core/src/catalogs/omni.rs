//! NASA OMNI2 hourly solar wind and IMF data provider.
//!
//! The OMNI dataset merges solar wind and interplanetary magnetic field
//! measurements from multiple spacecraft (ACE, WIND, IMP-8) into a single,
//! time-aligned hourly dataset, propagated to the Earth's bow shock nose.
//!
//! Source: <https://omniweb.gsfc.nasa.gov/>
//! FTP: <https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/>
//! Format: OMNI2 fixed-width ASCII, 57 columns per row.
//! Reference: King & Papitashvili (2005), J. Geophys. Res., 110, A02104.

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_string};
use std::path::{Path, PathBuf};

/// A single hourly OMNI2 solar wind + IMF record.
///
/// Contains both plasma parameters (from SWEPAM-type instruments) and
/// magnetic field measurements (from MAG-type instruments), pre-merged
/// and time-shifted to the bow shock nose by NASA/GSFC.
///
/// Coordinate system: GSE (Geocentric Solar Ecliptic) for B-field and flow.
/// Fill values are converted to NaN.
#[derive(Debug, Clone)]
pub struct OmniRecord {
    /// Year.
    pub year: u16,
    /// Day of year (1-366).
    pub doy: u16,
    /// Hour of day (0-23).
    pub hour: u8,

    // -- IMF parameters (GSE coordinates) --

    /// Scalar magnetic field magnitude (nT). Fill: 999.9
    pub b_magnitude: f64,
    /// IMF Bx in GSE coordinates (nT). Fill: 999.9
    pub bx_gse: f64,
    /// IMF By in GSE coordinates (nT). Fill: 999.9
    pub by_gse: f64,
    /// IMF Bz in GSE coordinates (nT). Fill: 999.9
    pub bz_gse: f64,

    // -- Plasma parameters --

    /// Proton temperature (K). Fill: 9999999.
    pub proton_temperature: f64,
    /// Proton number density (cm^-3). Fill: 999.9
    pub proton_density: f64,
    /// Bulk solar wind speed, scalar (km/s). Fill: 9999.
    pub bulk_speed: f64,
    /// Flow pressure (nPa). Fill: 99.99
    pub flow_pressure: f64,
    /// Plasma beta (dimensionless). Fill: 999.99
    pub plasma_beta: f64,
    /// Alfven Mach number (dimensionless). Fill: 999.9
    pub alfven_mach: f64,

    // -- Geomagnetic indices --

    /// Dst index (nT). Fill: 99999
    pub dst_index: f64,
    /// AE index (nT). Fill: 9999
    pub ae_index: f64,
    /// Kp index (times 10, e.g. 30 = Kp 3.0). Fill: 99
    pub kp_times_10: u8,

    // -- Heliocentric position --

    /// Heliocentric distance (AU). 1.0 for L1 spacecraft.
    pub r_au: f64,
    /// Heliographic latitude (deg). NaN for ecliptic-only spacecraft.
    pub lat_deg: f64,
    /// Heliographic longitude (deg). NaN when unknown.
    pub lon_deg: f64,
}

// Fill values from the OMNI2 format specification.
// These are the exact sentinel values NASA uses for missing data.
const FILL_B: f64 = 999.9;
const FILL_TEMP: f64 = 9999999.0;
const FILL_DENSITY: f64 = 999.9;
const FILL_SPEED: f64 = 9999.0;
const FILL_PRESSURE: f64 = 99.99;
const FILL_BETA: f64 = 999.99;
const FILL_MACH: f64 = 999.9;
const FILL_DST: f64 = 99999.0;
const FILL_AE: f64 = 9999.0;

// Hard physical ceilings for solar wind parameters at 1 AU.
// Any value above these is rejected as unphysical or a partial fill leak.
//
// Rationale:
//   B: largest CME sheath fields at 1 AU peak ~100 nT (rare). 200 nT is
//       beyond any credible measurement. Typical quiet: 3-8 nT.
//   T: coronal holes push ~1e6 K; CME sheaths ~1e7 K. 1e8 K is unphysical
//       at 1 AU (that's stellar interior territory).
//   n: densest CME ejecta ~100 cm^-3; 500 is generous ceiling.
//   V: fastest CMEs reach ~2500 km/s (rare). 3000 is ceiling.
//   P: flow pressure = 0.5 * m_p * n * V^2; maxes ~200 nPa in extreme CMEs.
//   beta: plasma beta >100 is suspicious at 1 AU (typical: 0.1-5).
//   Ma: Alfven Mach >100 is unphysical (typical: 5-20).
const CEIL_B: f64 = 200.0;
const CEIL_TEMP: f64 = 1.0e8;
const CEIL_DENSITY: f64 = 500.0;
const CEIL_SPEED: f64 = 3000.0;
const CEIL_PRESSURE: f64 = 500.0;
const CEIL_BETA: f64 = 500.0;
const CEIL_MACH: f64 = 200.0;

/// Two-stage filter: (1) reject exact fill values, (2) reject values above
/// hard physical ceilings. Returns NaN for anything suspicious.
fn parse_or_nan(s: &str, fill: f64, ceiling: f64) -> f64 {
    match s.trim().parse::<f64>() {
        Ok(v) if (v - fill).abs() < 0.5 => f64::NAN,
        Ok(v) if v.abs() > ceiling => f64::NAN,
        Ok(v) => v,
        Err(_) => f64::NAN,
    }
}

/// Parse without physical ceiling (for indices where ceilings don't apply).
fn parse_or_nan_fill_only(s: &str, fill: f64) -> f64 {
    match s.trim().parse::<f64>() {
        Ok(v) if (v - fill).abs() < 0.5 => f64::NAN,
        Ok(v) => v,
        Err(_) => f64::NAN,
    }
}

/// Distance-scaled physical ceilings for outer heliosphere spacecraft.
///
/// At heliocentric distance `r_au`, the solar wind parameters scale as:
/// - Density: n ~ r^-2 (continuity equation for spherical expansion)
/// - B-field: B_r ~ r^-2, B_phi ~ r^-1 (flux conservation + Parker spiral)
/// - Temperature: T ~ r^(-2/3) (adiabatic expansion, gamma=5/3)
/// - Speed: approximately constant (super-Alfvenic flow)
///
/// Returns (ceil_b, ceil_temp, ceil_density, ceil_speed).
/// At r=1.0 AU, returns the same values as the 1 AU constants.
pub fn distance_scaled_ceilings(r_au: f64) -> (f64, f64, f64, f64) {
    let r = if r_au > 0.0 { r_au } else { 1.0 };
    let ceil_b = CEIL_B / (r * r);
    let ceil_temp = CEIL_TEMP / r.powf(2.0 / 3.0);
    let ceil_density = CEIL_DENSITY / (r * r);
    let ceil_speed = CEIL_SPEED; // constant (super-Alfvenic)
    (ceil_b, ceil_temp, ceil_density, ceil_speed)
}

/// Two-stage filter using distance-scaled ceilings.
///
/// Same logic as `parse_or_nan` but uses ceilings appropriate for
/// heliocentric distance `r_au`. For L1 spacecraft (r_au=1.0), behavior
/// is identical to `parse_or_nan` with the 1 AU constants.
pub fn parse_or_nan_at_distance(s: &str, fill: f64, ceiling_at_r: f64) -> f64 {
    parse_or_nan(s, fill, ceiling_at_r)
}

/// Parse NASA OMNI2 hourly ASCII data.
///
/// Format: 57 fixed-width columns per row, space-separated.
/// First 3 fields: Year (I4), DOY (I4), Hour (I3).
///
/// Key column indices (0-based after split_whitespace):
///   8  = |B| (nT)
///  12  = Bx GSE (nT)
///  13  = By GSE (nT)
///  14  = Bz GSE (nT)
///  22  = Proton temperature (K)
///  23  = Proton density (cm^-3)
///  24  = Bulk speed (km/s)
///  28  = Flow pressure (nPa)
///  36  = Plasma beta
///  37  = Alfven Mach number
///  38  = Kp * 10
///  40  = Dst index
///  41  = AE index
pub fn parse_omni_hourly(content: &str) -> Vec<OmniRecord> {
    let mut records = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 42 {
            continue;
        }

        let year: u16 = match fields[0].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let doy: u16 = match fields[1].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let hour: u8 = match fields[2].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Two-stage parsing: fill detection + physical ceiling rejection.
        // This prevents partial fills (e.g. 998.0 for B) from leaking
        // into the LBM normalizer and blowing up the simulation.
        let b_magnitude = parse_or_nan(fields[8], FILL_B, CEIL_B);
        let bx_gse = parse_or_nan(fields[12], FILL_B, CEIL_B);
        let by_gse = parse_or_nan(fields[13], FILL_B, CEIL_B);
        let bz_gse = parse_or_nan(fields[14], FILL_B, CEIL_B);

        let proton_temperature = parse_or_nan(fields[22], FILL_TEMP, CEIL_TEMP);
        let proton_density = parse_or_nan(fields[23], FILL_DENSITY, CEIL_DENSITY);
        let bulk_speed = parse_or_nan(fields[24], FILL_SPEED, CEIL_SPEED);
        let flow_pressure = parse_or_nan(fields[28], FILL_PRESSURE, CEIL_PRESSURE);

        let plasma_beta = parse_or_nan(fields[36], FILL_BETA, CEIL_BETA);
        let alfven_mach = parse_or_nan(fields[37], FILL_MACH, CEIL_MACH);

        let kp_times_10: u8 = fields[38].trim().parse().unwrap_or(99);
        // Geomagnetic indices: fill-only (no physical ceiling -- Dst can
        // be large negative during extreme storms, AE can reach thousands).
        let dst_index = parse_or_nan_fill_only(fields[40], FILL_DST);
        let ae_index = parse_or_nan_fill_only(fields[41], FILL_AE);

        records.push(OmniRecord {
            year,
            doy,
            hour,
            b_magnitude,
            bx_gse,
            by_gse,
            bz_gse,
            proton_temperature,
            proton_density,
            bulk_speed,
            flow_pressure,
            plasma_beta,
            alfven_mach,
            dst_index,
            ae_index,
            kp_times_10,
            r_au: 1.0,
            lat_deg: f64::NAN,
            lon_deg: f64::NAN,
        });
    }
    records
}

/// Parse an OMNI2 yearly file from disk.
pub fn parse_omni_file(path: &Path) -> Result<Vec<OmniRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {}", e)))?;
    Ok(parse_omni_hourly(&content))
}

/// Base URL for OMNI2 low-resolution hourly data.
const OMNI2_BASE: &str = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_";

/// NASA OMNI2 dataset provider.
///
/// Fetches hourly solar wind + IMF data for a specified year range.
/// Each yearly file is ~2.7 MB of ASCII data (~8760 rows).
pub struct OmniProvider {
    /// Start year (inclusive).
    pub year_start: u16,
    /// End year (inclusive).
    pub year_end: u16,
}

impl Default for OmniProvider {
    fn default() -> Self {
        Self {
            year_start: 2024,
            year_end: 2024,
        }
    }
}

impl DatasetProvider for OmniProvider {
    fn name(&self) -> &str {
        "NASA OMNI2 Solar Wind + IMF"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("omni2");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let fname = format!("omni2_{}.dat", year);
            let output = dir.join(&fname);
            if config.skip_existing && output.exists() {
                continue;
            }
            let url = format!("{}{}.dat", OMNI2_BASE, year);
            match download_to_string(&url) {
                Ok(data) => {
                    std::fs::write(&output, data)?;
                    log::info!("Saved {}", fname);
                }
                Err(e) => {
                    log::warn!("Failed to download OMNI2 {}: {}", year, e);
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("omni2").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Real OMNI2 data (2024 DOY 1, hours 0-4) from
    // https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_2024.dat
    const SAMPLE_OMNI: &str = "\
2024   1  0 2596 51 52  60  36   5.3   5.1  -7.3 322.9   4.0  -3.0  -0.6  -2.8  -1.3   0.1   1.5   0.3   0.4   1.5   31114.   7.4  306.  -2.3   2.8 0.042  1.35    4691.   0.4    2.   0.3   1.3 0.007   0.40   1.75   7.9  7  55     0   20 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   3 131.2   0.3    -9    11  5.0
2024   1  1 2596 51 52  53  33   5.4   5.2 -21.3 333.4   4.4  -2.2  -1.9  -1.7  -2.3   0.1   1.1   0.5   0.4   1.0   28455.   6.5  301.  -2.1   1.9 0.040  1.14    4102.   0.4    4.   0.4   1.2 0.008   0.69   1.45   7.1  7  55     2   25 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   3 131.2   0.3   -11    14  4.7
2024   1  2 2596 51 52  63  33   4.5   4.1 -42.3 324.4   2.5  -1.8  -2.8  -1.3  -3.1   0.7   2.0   0.6   1.2   1.4   36413.   7.9  309.  -0.8   1.7 0.034  1.43    2767.   1.0    2.   0.6   1.0 0.008   0.96   2.67   9.7  7  55     4   32 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   3 131.2   0.3   -11    21  5.3
2024   1  3 2596 51 52  58  34   4.2   3.9 -12.2 329.6   3.3  -1.9  -0.8  -1.8  -1.1   0.1   1.4   0.3   0.4   1.3   32023.   8.9  306.  -0.8   1.7 0.037  1.60    1565.   0.3    2.   0.2   0.3 0.003   0.34   3.37  10.9  3  55     5   29 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.5   -10    19  5.5
2024   1  4 2596 51 52  62  34   4.2   4.1 -23.0 341.0   3.6  -1.2  -1.6  -1.0  -1.7   0.1   1.0   0.3   0.6   0.8   33006.   9.1  307.  -0.4   1.1 0.035  1.63    1953.   0.3    2.   0.4   0.4 0.003   0.52   3.46  11.0  3  55     3   33 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3   -13    21  5.5
";

    #[test]
    fn test_parse_omni_basic() {
        let records = parse_omni_hourly(SAMPLE_OMNI);
        assert_eq!(records.len(), 5);

        let r = &records[0];
        assert_eq!(r.year, 2024);
        assert_eq!(r.doy, 1);
        assert_eq!(r.hour, 0);

        // B-field: |B|=5.3, Bx=4.0, By=-3.0, Bz=-0.6
        assert!((r.b_magnitude - 5.3).abs() < 0.01);
        assert!((r.bx_gse - 4.0).abs() < 0.01);
        assert!((r.by_gse - (-3.0)).abs() < 0.01);
        assert!((r.bz_gse - (-0.6)).abs() < 0.01);

        // Plasma: T=31114K, n=7.4 cm^-3, V=306 km/s
        assert!((r.proton_temperature - 31114.0).abs() < 1.0);
        assert!((r.proton_density - 7.4).abs() < 0.01);
        assert!((r.bulk_speed - 306.0).abs() < 0.1);

        // Derived: pressure=1.35 nPa, beta=1.75, Ma=7.9
        assert!((r.flow_pressure - 1.35).abs() < 0.01);
        assert!((r.plasma_beta - 1.75).abs() < 0.01);
        assert!((r.alfven_mach - 7.9).abs() < 0.1);
    }

    #[test]
    fn test_omni_fill_values_to_nan() {
        // Construct a line with fill values for proton flux (words 42-47)
        // The sample data already has 999999.99 for proton flux columns,
        // but let's test B-field fill values directly.
        let data = "2024   1  0 2596 99 99 999 999 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 999.9 9999999.  999.9 9999.   0.0   0.0 0.000 99.99 9999999.   0.0    0.   0.0   0.0 0.000 999.99 999.99 999.9 99 999 99999 9999 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0 999 999.9 999.9 99999 99999 99.9\n";
        let records = parse_omni_hourly(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!(r.b_magnitude.is_nan(), "B mag should be NaN");
        assert!(r.bx_gse.is_nan(), "Bx should be NaN");
        assert!(r.by_gse.is_nan(), "By should be NaN");
        assert!(r.bz_gse.is_nan(), "Bz should be NaN");
        assert!(r.proton_temperature.is_nan(), "T should be NaN");
        assert!(r.proton_density.is_nan(), "n should be NaN");
        assert!(r.bulk_speed.is_nan(), "V should be NaN");
        assert!(r.flow_pressure.is_nan(), "P should be NaN");
        assert!(r.plasma_beta.is_nan(), "beta should be NaN");
        assert!(r.alfven_mach.is_nan(), "Ma should be NaN");
    }

    #[test]
    fn test_omni_physical_ranges() {
        let records = parse_omni_hourly(SAMPLE_OMNI);
        for r in &records {
            // B-field: typically 1-30 nT at 1 AU
            if !r.b_magnitude.is_nan() {
                assert!(
                    r.b_magnitude > 0.1 && r.b_magnitude < 100.0,
                    "|B| out of range: {}",
                    r.b_magnitude,
                );
            }
            // Density: typically 1-50 cm^-3
            if !r.proton_density.is_nan() {
                assert!(
                    r.proton_density > 0.1 && r.proton_density < 200.0,
                    "n out of range: {}",
                    r.proton_density,
                );
            }
            // Speed: typically 250-800 km/s
            if !r.bulk_speed.is_nan() {
                assert!(
                    r.bulk_speed > 100.0 && r.bulk_speed < 1500.0,
                    "V out of range: {}",
                    r.bulk_speed,
                );
            }
            // Temperature: typically 1e3-1e7 K
            if !r.proton_temperature.is_nan() {
                assert!(
                    r.proton_temperature > 1.0e2 && r.proton_temperature < 1.0e8,
                    "T out of range: {}",
                    r.proton_temperature,
                );
            }
        }
    }

    #[test]
    fn test_omni_empty() {
        let records = parse_omni_hourly("# comment only\n");
        assert!(records.is_empty());
    }

    #[test]
    fn test_omni_dst_and_kp() {
        let records = parse_omni_hourly(SAMPLE_OMNI);
        let r = &records[0];
        // Dst = 0 (quiet), AE = 20
        assert!((r.dst_index - 0.0).abs() < 0.5);
        assert!((r.ae_index - 20.0).abs() < 0.5);
        assert_eq!(r.kp_times_10, 7); // Kp = 0.7
    }

    #[test]
    fn test_omni_ceiling_rejects_near_fill_values() {
        // A near-fill value of 998.0 for B-field would pass the fill check
        // (999.9 - 998.0 = 1.9 > 0.5 tolerance) but is unphysical at 1 AU
        // (typical B ~ 5 nT, max CME ~ 100 nT). The ceiling at 200 nT
        // must catch this.
        //
        // Similarly, T=9999998 would pass fill check but is 10^7 K -- the
        // ceiling at 1e8 K passes it (it IS physically possible in extreme
        // CME sheaths). But V=9998 km/s passes fill (9999-9998=1) and the
        // ceiling at 3000 km/s must reject it.
        assert!(parse_or_nan("998.0", FILL_B, CEIL_B).is_nan());
        assert!(parse_or_nan("500.0", FILL_B, CEIL_B).is_nan());
        assert!(parse_or_nan("201.0", FILL_B, CEIL_B).is_nan());
        // 100 nT is a huge but real CME sheath field -- should survive
        assert!(!parse_or_nan("100.0", FILL_B, CEIL_B).is_nan());
        assert!((parse_or_nan("100.0", FILL_B, CEIL_B) - 100.0).abs() < 0.01);

        // Speed: 9998 km/s is a near-fill leak, must be rejected
        assert!(parse_or_nan("9998.0", FILL_SPEED, CEIL_SPEED).is_nan());
        assert!(parse_or_nan("5000.0", FILL_SPEED, CEIL_SPEED).is_nan());
        // 2000 km/s is a fast CME -- should survive
        assert!(!parse_or_nan("2000.0", FILL_SPEED, CEIL_SPEED).is_nan());

        // Density: 998 is near-fill, must be rejected
        assert!(parse_or_nan("998.0", FILL_DENSITY, CEIL_DENSITY).is_nan());
        // 50 cm^-3 is dense CME ejecta -- should survive
        assert!(!parse_or_nan("50.0", FILL_DENSITY, CEIL_DENSITY).is_nan());

        // Temperature: 9999998 is near-fill but below ceiling
        // (1e8 ceiling is generous for extreme CME sheath T)
        assert!(
            !parse_or_nan("9999998.0", FILL_TEMP, CEIL_TEMP).is_nan(),
            "9999998 is near-fill but below 1e8 ceiling -- ambiguous case"
        );
        // But the fill check (within 0.5) should catch 9999999.0 exactly
        assert!(parse_or_nan("9999999.0", FILL_TEMP, CEIL_TEMP).is_nan());
        // And anything above 1e8 is clearly unphysical
        assert!(parse_or_nan("200000000.0", FILL_TEMP, CEIL_TEMP).is_nan());
    }

    #[test]
    fn test_omni_legitimate_extreme_values_survive() {
        // Carrington-class event values that are extreme but real:
        // B = 50 nT (large CME sheath), V = 2500 km/s (fast CME),
        // n = 100 cm^-3 (dense sheath), T = 5e6 K (hot sheath)
        let r = parse_or_nan("50.0", FILL_B, CEIL_B);
        assert!(!r.is_nan() && (r - 50.0).abs() < 0.01);

        let r = parse_or_nan("2500.0", FILL_SPEED, CEIL_SPEED);
        assert!(!r.is_nan() && (r - 2500.0).abs() < 0.1);

        let r = parse_or_nan("100.0", FILL_DENSITY, CEIL_DENSITY);
        assert!(!r.is_nan() && (r - 100.0).abs() < 0.01);

        let r = parse_or_nan("5000000.0", FILL_TEMP, CEIL_TEMP);
        assert!(!r.is_nan() && (r - 5e6).abs() < 1.0);

        // Negative Bx (sunward B in GSE) is physical and common
        let r = parse_or_nan("-30.0", FILL_B, CEIL_B);
        assert!(!r.is_nan() && (r - (-30.0)).abs() < 0.01);
    }
}
