//! Shared SPDF merged hourly ASCII parser infrastructure.
//!
//! NASA SPDF (Space Physics Data Facility) provides merged hourly datasets
//! for many deep-space missions (Voyager, Pioneer, Ulysses, etc.) in a
//! common ASCII format: space-separated columns with year, DOY, hour,
//! position, B-field, and plasma parameters.
//!
//! This module provides:
//! - `SpdfColumnLayout`: maps column indices per spacecraft
//! - `SpdfMergedRecord`: generic intermediate record
//! - `parse_spdf_merged()`: parametric parser
//! - `spdf_to_omni()`: generic adapter to OmniRecord

use crate::catalogs::omni::{OmniRecord, distance_scaled_ceilings};

/// Column layout for SPDF merged hourly ASCII files.
///
/// Different spacecraft have different column orderings. This struct
/// maps the semantic meaning of each field to its 0-based column index
/// in the ASCII file. Set any index to `None` if the spacecraft does
/// not provide that measurement.
#[derive(Debug, Clone)]
pub struct SpdfColumnLayout {
    /// Minimum number of columns required per row.
    pub min_columns: usize,

    // Time columns (always present)
    /// Column index for year (4-digit).
    pub col_year: usize,
    /// Column index for day of year (1-366).
    pub col_doy: usize,
    /// Column index for hour (0-23).
    pub col_hour: usize,

    // Position columns
    /// Column index for heliocentric distance (AU).
    pub col_distance_au: Option<usize>,
    /// Column index for heliographic latitude (deg).
    pub col_lat_deg: Option<usize>,
    /// Column index for heliographic longitude (deg).
    pub col_lon_deg: Option<usize>,

    // B-field columns (RTN or SE coordinates)
    /// Column index for B magnitude (nT).
    pub col_b_mag: Option<usize>,
    /// Column index for B radial or Bx (nT).
    pub col_br: Option<usize>,
    /// Column index for B tangential or By (nT).
    pub col_bt: Option<usize>,
    /// Column index for B normal or Bz (nT).
    pub col_bn: Option<usize>,

    // Plasma columns
    /// Column index for proton density (cm^-3).
    pub col_density: Option<usize>,
    /// Column index for bulk speed (km/s).
    pub col_speed: Option<usize>,
    /// Column index for proton temperature (K).
    pub col_temperature: Option<usize>,

    // Fill values (spacecraft-specific)
    /// Fill value for B-field components (nT). Common: 9999.99 or 999.99.
    pub fill_b: f64,
    /// Fill value for density (cm^-3). Common: 999.99 or 999.9.
    pub fill_density: f64,
    /// Fill value for speed (km/s). Common: 9999.9 or 9999.0.
    pub fill_speed: f64,
    /// Fill value for temperature (K). Common: 999999.0 or 9999999.0.
    pub fill_temperature: f64,
    /// Fill value for distance (AU). Common: 999.999.
    pub fill_distance: f64,

    /// Whether B-field is in SE (Solar Ecliptic) coordinates (true) or
    /// RTN (Radial-Tangential-Normal) coordinates (false).
    pub b_is_se: bool,
}

/// Raw parsed record from SPDF merged hourly ASCII file.
#[derive(Debug, Clone)]
pub struct SpdfMergedRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,

    /// Heliocentric distance (AU). NaN if not available.
    pub distance_au: f64,
    /// Heliographic latitude (deg). NaN if not available.
    pub lat_deg: f64,
    /// Heliographic longitude (deg). NaN if not available.
    pub lon_deg: f64,

    /// B-field magnitude (nT). NaN if fill.
    pub b_magnitude: f64,
    /// Br or Bx (nT). NaN if fill.
    pub br: f64,
    /// Bt or By (nT). NaN if fill.
    pub bt: f64,
    /// Bn or Bz (nT). NaN if fill.
    pub bn: f64,

    /// Proton density (cm^-3). NaN if fill.
    pub proton_density: f64,
    /// Bulk speed (km/s). NaN if fill.
    pub bulk_speed: f64,
    /// Proton temperature (K). NaN if fill.
    pub proton_temperature: f64,
}

/// Parse a value from a string field, returning NaN if it matches the
/// fill value (within tolerance 0.5) or exceeds the ceiling.
fn parse_field(s: &str, fill: f64, ceiling: f64) -> f64 {
    match s.trim().parse::<f64>() {
        Ok(v) if (v - fill).abs() < 0.5 => f64::NAN,
        Ok(v) if v.abs() > ceiling => f64::NAN,
        Ok(v) => v,
        Err(_) => f64::NAN,
    }
}

/// Parse a value without ceiling (for position fields).
fn parse_field_fill_only(s: &str, fill: f64) -> f64 {
    match s.trim().parse::<f64>() {
        Ok(v) if (v - fill).abs() < 0.5 => f64::NAN,
        Ok(v) => v,
        Err(_) => f64::NAN,
    }
}

/// Extract a column value, returning None if the column index is None
/// or out of bounds.
fn col_val<'a>(fields: &[&'a str], col: Option<usize>) -> Option<&'a str> {
    col.and_then(|c| fields.get(c).copied())
}

/// Parse SPDF merged hourly ASCII data using the given column layout.
///
/// Handles SPDF fill values (spacecraft-specific), distance-scaled
/// physical ceilings, and missing columns gracefully.
pub fn parse_spdf_merged(content: &str, layout: &SpdfColumnLayout) -> Vec<SpdfMergedRecord> {
    let mut records = Vec::new();

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < layout.min_columns {
            continue;
        }

        let year: u16 = match fields[layout.col_year].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let doy: u16 = match fields[layout.col_doy].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let hour: u8 = match fields[layout.col_hour].parse() {
            Ok(v) => v,
            Err(_) => continue,
        };

        // Position
        let distance_au = col_val(&fields, layout.col_distance_au)
            .map(|s| parse_field_fill_only(s, layout.fill_distance))
            .unwrap_or(f64::NAN);
        let lat_deg = col_val(&fields, layout.col_lat_deg)
            .map(|s| parse_field_fill_only(s, 999.99))
            .unwrap_or(f64::NAN);
        let lon_deg = col_val(&fields, layout.col_lon_deg)
            .map(|s| parse_field_fill_only(s, 999.99))
            .unwrap_or(f64::NAN);

        // Distance-scaled ceilings for physics validation
        let r = if distance_au.is_finite() && distance_au > 0.0 {
            distance_au
        } else {
            1.0
        };
        let (ceil_b, ceil_temp, ceil_density, ceil_speed) = distance_scaled_ceilings(r);

        // B-field
        let b_magnitude = col_val(&fields, layout.col_b_mag)
            .map(|s| parse_field(s, layout.fill_b, ceil_b))
            .unwrap_or(f64::NAN);
        let br = col_val(&fields, layout.col_br)
            .map(|s| parse_field(s, layout.fill_b, ceil_b))
            .unwrap_or(f64::NAN);
        let bt = col_val(&fields, layout.col_bt)
            .map(|s| parse_field(s, layout.fill_b, ceil_b))
            .unwrap_or(f64::NAN);
        let bn = col_val(&fields, layout.col_bn)
            .map(|s| parse_field(s, layout.fill_b, ceil_b))
            .unwrap_or(f64::NAN);

        // Plasma
        let proton_density = col_val(&fields, layout.col_density)
            .map(|s| parse_field(s, layout.fill_density, ceil_density))
            .unwrap_or(f64::NAN);
        let bulk_speed = col_val(&fields, layout.col_speed)
            .map(|s| parse_field(s, layout.fill_speed, ceil_speed))
            .unwrap_or(f64::NAN);
        let proton_temperature = col_val(&fields, layout.col_temperature)
            .map(|s| parse_field(s, layout.fill_temperature, ceil_temp))
            .unwrap_or(f64::NAN);

        records.push(SpdfMergedRecord {
            year,
            doy,
            hour,
            distance_au,
            lat_deg,
            lon_deg,
            b_magnitude,
            br,
            bt,
            bn,
            proton_density,
            bulk_speed,
            proton_temperature,
        });
    }

    records
}

/// Convert SPDF merged records to OmniRecord format.
///
/// For SE-coordinate spacecraft (Voyager, Pioneer), Bx/By/Bz map
/// directly to GSE (Solar Ecliptic is close to GSE for spacecraft
/// near the ecliptic plane).
///
/// For RTN-coordinate spacecraft at radial positions, RTN->GSE with
/// separation_angle=0 is used (radial spacecraft have no azimuthal
/// offset from the Sun-Earth line in the corotating frame).
pub fn spdf_to_omni(records: &[SpdfMergedRecord], b_is_se: bool) -> Vec<OmniRecord> {
    records
        .iter()
        .map(|r| {
            // B-field coordinate handling
            let (bx_gse, by_gse, bz_gse) = if b_is_se {
                // Solar Ecliptic: Bx ~ radial, By ~ ecliptic transverse,
                // Bz ~ ecliptic north. Close to GSE for near-ecliptic S/C.
                (r.br, r.bt, r.bn)
            } else {
                // RTN with separation_angle=0 (radially outward spacecraft):
                // Br -> Bx (radial), Bt -> -By (tangential), Bn -> Bz (north)
                // The sign flip on Bt accounts for the right-handed RTN
                // convention vs left-handed GSE-Y.
                (r.br, -r.bt, r.bn)
            };

            OmniRecord {
                year: r.year,
                doy: r.doy,
                hour: r.hour,
                b_magnitude: r.b_magnitude,
                bx_gse,
                by_gse,
                bz_gse,
                proton_temperature: r.proton_temperature,
                proton_density: r.proton_density,
                bulk_speed: r.bulk_speed,
                flow_pressure: f64::NAN,
                plasma_beta: f64::NAN,
                alfven_mach: f64::NAN,
                dst_index: f64::NAN,
                ae_index: f64::NAN,
                kp_times_10: 0,
                r_au: if r.distance_au.is_finite() {
                    r.distance_au
                } else {
                    1.0
                },
                lat_deg: r.lat_deg,
                lon_deg: r.lon_deg,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_layout() -> SpdfColumnLayout {
        // Minimal layout: year, doy, hour, distance, density, speed, temp, |B|
        SpdfColumnLayout {
            min_columns: 8,
            col_year: 0,
            col_doy: 1,
            col_hour: 2,
            col_distance_au: Some(3),
            col_lat_deg: None,
            col_lon_deg: None,
            col_b_mag: Some(4),
            col_br: None,
            col_bt: None,
            col_bn: None,
            col_density: Some(5),
            col_speed: Some(6),
            col_temperature: Some(7),
            fill_b: 9999.99,
            fill_density: 999.99,
            fill_speed: 9999.9,
            fill_temperature: 999999.0,
            fill_distance: 999.999,
            b_is_se: true,
        }
    }

    #[test]
    fn test_parse_spdf_basic() {
        // At 93.5 AU: B ceiling = 200/93.5^2 = 0.023 nT, density ceiling = 0.057
        // Use values below the distance-scaled ceilings.
        let data = "2004 120 12 93.5 0.02 0.001 400.0 50000.0\n";
        let layout = test_layout();
        let records = parse_spdf_merged(data, &layout);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert_eq!(r.year, 2004);
        assert_eq!(r.doy, 120);
        assert_eq!(r.hour, 12);
        assert!((r.distance_au - 93.5).abs() < 0.01);
        assert!((r.b_magnitude - 0.02).abs() < 0.001);
        assert!((r.proton_density - 0.001).abs() < 0.0001);
        assert!((r.bulk_speed - 400.0).abs() < 0.1);
        assert!((r.proton_temperature - 50000.0).abs() < 1.0);
    }

    #[test]
    fn test_parse_spdf_fill_detection() {
        // All measurements are fill values
        let data = "2004 120 12 999.999 9999.99 999.99 9999.9 999999.0\n";
        let layout = test_layout();
        let records = parse_spdf_merged(data, &layout);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!(r.distance_au.is_nan(), "distance fill not detected");
        assert!(r.b_magnitude.is_nan(), "B fill not detected");
        assert!(r.proton_density.is_nan(), "density fill not detected");
        assert!(r.bulk_speed.is_nan(), "speed fill not detected");
        assert!(r.proton_temperature.is_nan(), "temp fill not detected");
    }

    #[test]
    fn test_parse_spdf_distance_scaled_ceilings() {
        // At r=100 AU, density ceiling = 500/(100^2) = 0.05 cm^-3.
        // A value of 0.1 cm^-3 at 100 AU exceeds the ceiling.
        let data = "2004 120 12 100.0 0.01 0.1 400.0 50000.0\n";
        let layout = test_layout();
        let records = parse_spdf_merged(data, &layout);
        assert_eq!(records.len(), 1);
        let r = &records[0];
        assert!(
            r.proton_density.is_nan(),
            "density 0.1 at 100 AU should exceed ceiling (0.05)"
        );
        // B=0.01 at 100 AU: ceiling = 200/(100^2) = 0.02. 0.01 < 0.02 -> valid
        assert!(
            !r.b_magnitude.is_nan(),
            "B=0.01 at 100 AU should be below ceiling (0.02)"
        );
    }

    #[test]
    fn test_spdf_to_omni_se_coords() {
        let rec = SpdfMergedRecord {
            year: 2020,
            doy: 100,
            hour: 6,
            distance_au: 140.0,
            lat_deg: f64::NAN,
            lon_deg: f64::NAN,
            b_magnitude: 0.01,
            br: 0.005,
            bt: -0.003,
            bn: 0.001,
            proton_density: 0.001,
            bulk_speed: 400.0,
            proton_temperature: 5000.0,
        };

        let omni = spdf_to_omni(&[rec], true);
        assert_eq!(omni.len(), 1);
        let o = &omni[0];
        assert!((o.r_au - 140.0).abs() < 0.01);
        // SE coords: br -> bx, bt -> by, bn -> bz
        assert!((o.bx_gse - 0.005).abs() < 1e-6);
        assert!((o.by_gse - (-0.003)).abs() < 1e-6);
        assert!((o.bz_gse - 0.001).abs() < 1e-6);
    }

    #[test]
    fn test_spdf_to_omni_rtn_coords() {
        let rec = SpdfMergedRecord {
            year: 2020,
            doy: 100,
            hour: 6,
            distance_au: 5.0,
            lat_deg: 30.0,
            lon_deg: 45.0,
            b_magnitude: 1.0,
            br: 0.5,
            bt: 0.3,
            bn: 0.1,
            proton_density: 0.5,
            bulk_speed: 700.0,
            proton_temperature: 200000.0,
        };

        let omni = spdf_to_omni(&[rec], false);
        assert_eq!(omni.len(), 1);
        let o = &omni[0];
        // RTN: br -> bx, -bt -> by, bn -> bz
        assert!((o.bx_gse - 0.5).abs() < 1e-6);
        assert!((o.by_gse - (-0.3)).abs() < 1e-6);
        assert!((o.bz_gse - 0.1).abs() < 1e-6);
        assert!((o.lat_deg - 30.0).abs() < 0.01);
        assert!((o.lon_deg - 45.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_spdf_skips_comments_and_short_lines() {
        let data = "# header comment\n\
                     2004 120\n\
                     2004 120 12 93.5 0.05 0.002 400.0 50000.0\n";
        let layout = test_layout();
        let records = parse_spdf_merged(data, &layout);
        assert_eq!(records.len(), 1, "should skip comment and short line");
    }
}
