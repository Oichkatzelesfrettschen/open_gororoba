//! ACE MAG Level 2 magnetic field data provider.
//!
//! The Advanced Composition Explorer (ACE) MAG instrument measures
//! the interplanetary magnetic field at 16-second cadence. Browse-quality
//! ASCII data (daily files) contain both RTN and GSE B-field components.
//!
//! Source: <https://izw1.caltech.edu/ACE/ASC/DATA/mag16_browse/>
//! Reference: Smith et al. (1998), Space Sci. Rev. 86, 613
//!
//! Data format: 12 whitespace-delimited columns per row.
//! Fill value: -999.9 for all magnetic field parameters.
//! Additional bad-data indicator: B = (0, 0, 0) with |B| = 0.

use crate::{
    catalogs::omni::OmniRecord,
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_file},
};
use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

/// A single 16-second ACE MAG measurement.
#[derive(Debug, Clone)]
pub struct AceMagRecord {
    /// Spacecraft clock at midpoint (seconds).
    pub sc_clock: f64,
    /// ACE epoch start (seconds since 1996-01-01 UT).
    pub epoch_start: f64,
    /// ACE epoch end (seconds since 1996-01-01 UT).
    pub epoch_end: f64,
    /// IMF Bx in GSE coordinates (nT).
    pub bx_gse: f64,
    /// IMF By in GSE coordinates (nT).
    pub by_gse: f64,
    /// IMF Bz in GSE coordinates (nT).
    pub bz_gse: f64,
    /// Scalar magnetic field magnitude (nT).
    pub b_magnitude: f64,
    /// RMS variation over 16-second window (nT).
    pub db_rms: f64,
    /// Number of sub-samples in average.
    pub weight: u32,
}

/// Hourly-averaged ACE MAG record.
#[derive(Debug, Clone)]
pub struct AceMagHourly {
    /// Hour index (sequential from start of data).
    pub hour_index: usize,
    /// Mean Bx GSE (nT).
    pub bx_gse: f64,
    /// Mean By GSE (nT).
    pub by_gse: f64,
    /// Mean Bz GSE (nT).
    pub bz_gse: f64,
    /// Mean |B| (nT).
    pub b_magnitude: f64,
    /// Number of valid 16-sec samples in this hour.
    pub sample_count: usize,
}

/// Fill value for ACE MAG data.
const FILL_MAG: f64 = -999.9;

/// Hard physical ceiling for B-field at 1 AU (nT).
/// Same as omni.rs: largest CME sheath fields peak ~100 nT.
const CEIL_B: f64 = 200.0;

/// Seconds per hour (for grouping 16-sec samples).
const SECS_PER_HOUR: f64 = 3600.0;

/// Two-stage filter: reject fill values and unphysical magnitudes.
fn parse_or_nan_mag(v: f64) -> f64 {
    if (v - FILL_MAG).abs() < 0.5 || v.abs() > CEIL_B {
        f64::NAN
    } else {
        v
    }
}

/// Check if a record has a zero B-field vector (bad data indicator).
fn is_zero_b(bx: f64, by: f64, bz: f64) -> bool {
    bx.abs() < 1e-10 && by.abs() < 1e-10 && bz.abs() < 1e-10
}

/// Parse ACE MAG browse 16-second ASCII data.
///
/// Format: 12 whitespace-delimited columns. Header lines start with `#`.
///
/// Columns:
///   0: SCclock (midpoint, seconds)
///   1: ACE epoch start (seconds since 1996-01-01)
///   2: ACE epoch end
///   3: Br (RTN, nT) -- skipped, we use GSE
///   4: Bt (RTN, nT) -- skipped
///   5: Bn (RTN, nT) -- skipped
///   6: Bx (GSE, nT)
///   7: By (GSE, nT)
///   8: Bz (GSE, nT)
///   9: |B| (nT)
///  10: dBrms (nT)
///  11: weight (count of sub-samples)
pub fn parse_ace_mag_16sec(content: &str) -> Vec<AceMagRecord> {
    let mut records = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 12 {
            continue;
        }

        let sc_clock: f64 = match fields[0].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let epoch_start: f64 = match fields[1].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let epoch_end: f64 = match fields[2].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        let bx_raw: f64 = fields[6].parse().unwrap_or(FILL_MAG);
        let by_raw: f64 = fields[7].parse().unwrap_or(FILL_MAG);
        let bz_raw: f64 = fields[8].parse().unwrap_or(FILL_MAG);
        let bmag_raw: f64 = fields[9].parse().unwrap_or(FILL_MAG);
        let db_rms: f64 = fields[10].parse().unwrap_or(f64::NAN);
        let weight: u32 = fields[11].parse().unwrap_or(0);

        // Reject zero-vector B-field (bad data indicator per ACE release notes).
        let bx = if is_zero_b(bx_raw, by_raw, bz_raw) {
            f64::NAN
        } else {
            parse_or_nan_mag(bx_raw)
        };
        let by = if is_zero_b(bx_raw, by_raw, bz_raw) {
            f64::NAN
        } else {
            parse_or_nan_mag(by_raw)
        };
        let bz = if is_zero_b(bx_raw, by_raw, bz_raw) {
            f64::NAN
        } else {
            parse_or_nan_mag(bz_raw)
        };
        let bmag = if is_zero_b(bx_raw, by_raw, bz_raw) {
            f64::NAN
        } else {
            parse_or_nan_mag(bmag_raw)
        };

        records.push(AceMagRecord {
            sc_clock,
            epoch_start,
            epoch_end,
            bx_gse: bx,
            by_gse: by,
            bz_gse: bz,
            b_magnitude: bmag,
            db_rms,
            weight,
        });
    }
    records
}

/// Parse an ACE MAG browse file from disk.
pub fn parse_ace_mag_file(path: &Path) -> Result<Vec<AceMagRecord>, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read error: {}", e)))?;
    Ok(parse_ace_mag_16sec(&content))
}

/// Average 16-second ACE MAG records into hourly bins.
///
/// Groups by floor(epoch_start / 3600). Only includes hours with
/// at least one valid (non-NaN) B-field sample.
pub fn average_to_hourly(records: &[AceMagRecord]) -> Vec<AceMagHourly> {
    if records.is_empty() {
        return Vec::new();
    }

    // Group by hour index (epoch_start / 3600, floored).
    let mut bins: HashMap<usize, Vec<&AceMagRecord>> = HashMap::new();
    for r in records {
        let hour = (r.epoch_start / SECS_PER_HOUR).floor() as usize;
        bins.entry(hour).or_default().push(r);
    }

    let mut hourly: Vec<AceMagHourly> = bins
        .into_iter()
        .filter_map(|(hour_index, samples)| {
            let valid: Vec<&AceMagRecord> = samples
                .iter()
                .filter(|s| !s.bx_gse.is_nan() && !s.by_gse.is_nan() && !s.bz_gse.is_nan())
                .copied()
                .collect();

            if valid.is_empty() {
                return None;
            }

            let n = valid.len() as f64;
            let bx = valid.iter().map(|s| s.bx_gse).sum::<f64>() / n;
            let by = valid.iter().map(|s| s.by_gse).sum::<f64>() / n;
            let bz = valid.iter().map(|s| s.bz_gse).sum::<f64>() / n;
            let bmag = valid.iter().map(|s| s.b_magnitude).sum::<f64>() / n;

            Some(AceMagHourly {
                hour_index,
                bx_gse: bx,
                by_gse: by,
                bz_gse: bz,
                b_magnitude: bmag,
                sample_count: valid.len(),
            })
        })
        .collect();

    hourly.sort_by_key(|h| h.hour_index);
    hourly
}

/// Convert hourly ACE MAG records to OmniRecord format.
///
/// B-field maps directly (GSE -> GSE). Plasma fields (density, speed,
/// temperature) are set to NaN since ACE MAG measures only magnetic field.
/// NaN plasma triggers fallback defaults in `UnitConversion::from_omni()`.
pub fn ace_mag_to_omni(records: &[AceMagHourly]) -> Vec<OmniRecord> {
    records
        .iter()
        .enumerate()
        .map(|(i, r)| OmniRecord {
            year: 0,
            doy: 0,
            hour: (i % 24) as u8,
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
        })
        .collect()
}

/// Base URL for ACE MAG browse data (16-sec, daily ASCII).
const ACE_MAG_BROWSE_BASE: &str = "https://izw1.caltech.edu/ACE/ASC/DATA/mag16_browse/";

/// ACE MAG dataset provider.
///
/// Fetches 16-second magnetic field browse data for a specified year range.
/// Each daily file is ~300 KB (5400 rows x 12 columns).
pub struct AceMagProvider {
    /// Start year (inclusive).
    pub year_start: u16,
    /// End year (inclusive).
    pub year_end: u16,
    /// Day of year range (inclusive). If None, fetches all 365/366 days.
    pub doy_range: Option<(u16, u16)>,
}

impl Default for AceMagProvider {
    fn default() -> Self {
        Self {
            year_start: 2024,
            year_end: 2024,
            doy_range: None,
        }
    }
}

impl DatasetProvider for AceMagProvider {
    fn name(&self) -> &str {
        "ACE MAG L2 Browse 16-sec"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("ace_mag");
        std::fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            let days_in_year = if year.is_multiple_of(4)
                && (!year.is_multiple_of(100) || year.is_multiple_of(400))
            {
                366
            } else {
                365
            };
            let (doy_start, doy_end) = self.doy_range.unwrap_or((1, days_in_year));

            for doy in doy_start..=doy_end.min(days_in_year) {
                let fname = format!("ACE_MAG16_{}-{:03}_V4-0.txt", year, doy);
                let output = dir.join(&fname);
                if config.skip_existing && output.exists() {
                    continue;
                }
                // Browse data: YYYY/ACE_MAG16_YYYY-DOY_V4-0.zip
                // Try unzipped .txt first (some mirrors serve raw ASCII).
                let url = format!("{}{}/{}", ACE_MAG_BROWSE_BASE, year, fname);
                match download_to_file(&url, &output) {
                    Ok(bytes) if bytes > 0 => {
                        log::info!("Saved {} ({} bytes)", fname, bytes);
                    }
                    _ => {
                        log::debug!("ACE MAG browse {} not found at {}", fname, url);
                    }
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("ace_mag").exists()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Synthetic ACE MAG 16-sec browse data (based on real format).
    const SAMPLE_ACE_MAG: &str = "\
# SCclock    -------ACE Epoch Time-----  ------RTN Coordinates----  ------GSE Coordinates----
#   Mid         Start        End            Br       Bt       Bn       Bx       By       Bz     <|B|>    dBrms   wt
 768472442.0 820540804.95 820540820.95    6.201   -2.474   -4.630   -6.201    1.932   -4.881    8.143    0.588   48
 768472458.0 820540820.95 820540836.95    5.800   -2.100   -4.200   -5.800    1.700   -4.500    7.600    0.450   48
 768472474.0 820540836.95 820540852.95    6.500   -2.800   -5.000   -6.500    2.200   -5.200    8.700    0.620   48
 768472490.0 820540852.95 820540868.95 -999.9   -999.9   -999.9   -999.9   -999.9   -999.9   -999.9   -999.9    0
";

    #[test]
    fn test_parse_ace_mag_basic() {
        let records = parse_ace_mag_16sec(SAMPLE_ACE_MAG);
        assert_eq!(records.len(), 4);

        let r = &records[0];
        assert!((r.bx_gse - (-6.201)).abs() < 0.001);
        assert!((r.by_gse - 1.932).abs() < 0.001);
        assert!((r.bz_gse - (-4.881)).abs() < 0.001);
        assert!((r.b_magnitude - 8.143).abs() < 0.001);
        assert_eq!(r.weight, 48);
    }

    #[test]
    fn test_parse_ace_mag_fill_values() {
        let records = parse_ace_mag_16sec(SAMPLE_ACE_MAG);
        let r = &records[3]; // fill row
        assert!(r.bx_gse.is_nan(), "Bx should be NaN for fill");
        assert!(r.by_gse.is_nan(), "By should be NaN for fill");
        assert!(r.bz_gse.is_nan(), "Bz should be NaN for fill");
        assert!(r.b_magnitude.is_nan(), "|B| should be NaN for fill");
    }

    #[test]
    fn test_parse_ace_mag_ceiling_rejection() {
        let data = "\
 768472442.0 820540804.95 820540820.95    6.0    -2.0    -4.0  250.0   200.1  -300.0  450.0    0.5   48
";
        let records = parse_ace_mag_16sec(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!(r.bx_gse.is_nan(), "250 nT should be rejected by ceiling");
        assert!(r.by_gse.is_nan(), "200.1 nT should be rejected by ceiling");
        assert!(r.bz_gse.is_nan(), "-300 nT should be rejected by ceiling");
    }

    #[test]
    fn test_parse_ace_mag_zero_b_rejection() {
        let data = "\
 768472442.0 820540804.95 820540820.95    0.0    0.0    0.0    0.0    0.0    0.0    0.0    0.0   48
";
        let records = parse_ace_mag_16sec(data);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!(r.bx_gse.is_nan(), "zero B should be rejected");
        assert!(r.by_gse.is_nan(), "zero B should be rejected");
        assert!(r.bz_gse.is_nan(), "zero B should be rejected");
    }

    #[test]
    fn test_ace_mag_hourly_averaging() {
        // 4 samples spanning 48 seconds -- all same hour bin.
        let records = parse_ace_mag_16sec(SAMPLE_ACE_MAG);
        let hourly = average_to_hourly(&records);
        assert_eq!(hourly.len(), 1, "all samples should fall in one hour");
        let h = &hourly[0];
        assert_eq!(h.sample_count, 3, "fill row excluded, 3 valid");

        // Mean of 3 valid Bx values: (-6.201 + -5.800 + -6.500) / 3
        let expected_bx = (-6.201 + -5.800 + -6.500) / 3.0;
        assert!(
            (h.bx_gse - expected_bx).abs() < 0.001,
            "bx mean: got {} expected {}",
            h.bx_gse,
            expected_bx,
        );
    }

    #[test]
    fn test_ace_mag_hourly_partial_nan() {
        // Mix of valid and NaN samples in one hour.
        let data = "\
 768472442.0 820540804.95 820540820.95    6.0   -2.0   -4.0   -6.0    2.0   -4.0    8.0    0.5   48
 768472458.0 820540820.95 820540836.95 -999.9 -999.9 -999.9 -999.9 -999.9 -999.9 -999.9 -999.9    0
 768472474.0 820540836.95 820540852.95    4.0   -1.0   -3.0   -4.0    1.0   -3.0    6.0    0.3   48
";
        let records = parse_ace_mag_16sec(data);
        let hourly = average_to_hourly(&records);
        assert_eq!(hourly.len(), 1);
        assert_eq!(hourly[0].sample_count, 2, "NaN row excluded from average");
        let expected_bx = (-6.0 + -4.0) / 2.0;
        assert!((hourly[0].bx_gse - expected_bx).abs() < 0.001);
    }

    #[test]
    fn test_ace_mag_to_omni_adapter() {
        let records = parse_ace_mag_16sec(SAMPLE_ACE_MAG);
        let hourly = average_to_hourly(&records);
        let omni = ace_mag_to_omni(&hourly);

        assert_eq!(omni.len(), 1);
        let r = &omni[0];
        // B-field should be populated
        assert!(!r.bx_gse.is_nan(), "Bx should be valid");
        assert!(!r.by_gse.is_nan(), "By should be valid");
        assert!(!r.bz_gse.is_nan(), "Bz should be valid");
        // Plasma should be NaN (ACE MAG has no plasma data)
        assert!(r.proton_density.is_nan(), "density should be NaN");
        assert!(r.bulk_speed.is_nan(), "speed should be NaN");
        assert!(r.proton_temperature.is_nan(), "temperature should be NaN");
    }

    #[test]
    fn test_ace_mag_physical_ranges() {
        let records = parse_ace_mag_16sec(SAMPLE_ACE_MAG);
        for r in &records {
            if !r.b_magnitude.is_nan() {
                assert!(
                    r.b_magnitude > 0.1 && r.b_magnitude < 100.0,
                    "|B| out of physical range: {}",
                    r.b_magnitude,
                );
            }
            if !r.bx_gse.is_nan() {
                assert!(r.bx_gse.abs() < 100.0, "Bx out of range: {}", r.bx_gse,);
            }
        }
    }
}
