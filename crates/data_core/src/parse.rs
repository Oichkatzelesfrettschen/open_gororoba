//! Canonical parsing helpers for catalog and geophysical data fields.
//!
//! Every catalog parser used to carry its own `parse_f64()` with a slightly
//! different sentinel set. This module consolidates them into two functions
//! whose sentinel lists are the union of all per-catalog variants.

use chrono::{DateTime, Datelike, Timelike, Utc};

/// Sentinels that map to NAN across all catalogs.
///
/// Union of: empty, "nan", "NaN", "null", "NULL", "--", "...",
///           "-99", "-999", "inf", "-inf".
const NAN_SENTINELS: &[&str] = &[
    "", "nan", "NaN", "null", "NULL", "--", "...", "-99", "-999", "inf", "-inf",
];

/// Parse a trimmed string to `f64`, returning NAN for any known sentinel.
///
/// Covers every sentinel variant seen in catalog parsers (CHIME, Pantheon+,
/// Fermi GBM, SDSS, Gaia, ATNF, McGill, SORCE, TSI, Swarm, Union3).
pub fn parse_f64_or_nan(s: &str) -> f64 {
    let s = s.trim();
    if NAN_SENTINELS.contains(&s) {
        return f64::NAN;
    }
    s.parse::<f64>().unwrap_or(f64::NAN)
}

/// Parse a HAPI space-physics numeric field to `f64`, mapping transport fill
/// sentinels such as `-1.0E31` to NAN.
///
/// CDAWeb/SPDF HAPI feeds commonly use huge-magnitude values near `1e31` to
/// represent missing data. Those values are valid transport sentinels but are
/// never physically meaningful for the heliosphere mission lanes in this repo.
pub fn parse_hapi_spacephysics_f64_or_nan(s: &str) -> f64 {
    let parsed = parse_f64_or_nan(s);
    if parsed.is_nan() || parsed.abs() >= 1.0e30 {
        f64::NAN
    } else {
        parsed
    }
}

/// Parse a trimmed string to `f64`, returning 0.0 for unparseable values.
///
/// Used by GWTC parser where missing numeric fields default to zero
/// (mass, spin, distance columns in GWOSC CSV).
pub fn parse_f64_or_zero(s: &str) -> f64 {
    s.trim().parse::<f64>().unwrap_or(0.0)
}

fn sexagesimal_tokens(s: &str) -> Vec<f64> {
    let normalized = s
        .trim()
        .replace(['h', 'H', 'm', 'M', 's', 'S', 'd', 'D', ':'], " ");
    normalized
        .split_whitespace()
        .filter_map(|token| token.parse::<f64>().ok())
        .collect()
}

/// Parse a sexagesimal right ascension string into decimal degrees.
///
/// Accepts common forms such as `01 46 22.407`, `01:46:22.407`, or `1.77289`.
/// Single-token inputs are treated as decimal degrees when their absolute value
/// exceeds 24, otherwise as decimal hours.
pub fn parse_sexagesimal_ra_to_deg(s: &str) -> f64 {
    let tokens = sexagesimal_tokens(s);
    match tokens.as_slice() {
        [] => f64::NAN,
        [value] if value.abs() > 24.0 => *value,
        [hours] => hours * 15.0,
        [hours, minutes] => (hours + minutes / 60.0) * 15.0,
        [hours, minutes, seconds, ..] => (hours + minutes / 60.0 + seconds / 3600.0) * 15.0,
    }
}

/// Parse a sexagesimal declination string into decimal degrees.
///
/// Accepts common forms such as `+61 45 03.19`, `-72:11:33.8`, or `-28.94`.
pub fn parse_sexagesimal_dec_to_deg(s: &str) -> f64 {
    let trimmed = s.trim();
    let sign = if trimmed.starts_with('-') { -1.0 } else { 1.0 };
    let tokens = sexagesimal_tokens(trimmed);
    match tokens.as_slice() {
        [] => f64::NAN,
        [degrees] => *degrees,
        [degrees, minutes] => sign * (degrees.abs() + minutes.abs() / 60.0),
        [degrees, minutes, seconds, ..] => {
            sign * (degrees.abs() + minutes.abs() / 60.0 + seconds.abs() / 3600.0)
        }
    }
}

/// Convert Galactic coordinates (l, b) in degrees to J2000 equatorial (RA, Dec).
///
/// Uses the standard IAU 1958 Galactic system expressed in the J2000/ICRS
/// rotation matrix adopted by Hipparcos/astrometry toolchains.
pub fn galactic_to_equatorial_j2000(l_deg: f64, b_deg: f64) -> (f64, f64) {
    let l = l_deg.to_radians();
    let b = b_deg.to_radians();

    let x_gal = b.cos() * l.cos();
    let y_gal = b.cos() * l.sin();
    let z_gal = b.sin();

    // Equatorial <- Galactic rotation = transpose of the standard
    // J2000 equatorial->Galactic rotation matrix.
    let x_eq = -0.054_875_560_4 * x_gal + 0.494_109_427_9 * y_gal - 0.867_666_149 * z_gal;
    let y_eq = -0.873_437_090_2 * x_gal - 0.444_829_63 * y_gal - 0.198_076_373_4 * z_gal;
    let z_eq = -0.483_835_015_5 * x_gal + 0.746_982_244_5 * y_gal + 0.455_983_776_2 * z_gal;

    let mut ra_deg = y_eq.atan2(x_eq).to_degrees();
    if ra_deg < 0.0 {
        ra_deg += 360.0;
    }
    let dec_deg = z_eq.clamp(-1.0, 1.0).asin().to_degrees();
    (ra_deg, dec_deg)
}

/// Parse an RFC 3339 / ISO-8601 timestamp into (year, day-of-year, hour).
pub fn parse_hapi_time_to_ydh(s: &str) -> Option<(u16, u16, u8)> {
    let dt = DateTime::parse_from_rfc3339(s).ok()?;
    let utc = dt.with_timezone(&Utc);
    Some((utc.year() as u16, utc.ordinal() as u16, utc.hour() as u8))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_f64_or_nan_numbers() {
        assert!((parse_f64_or_nan("1.234") - 1.234).abs() < 1e-12);
        assert!((parse_f64_or_nan("  42  ") - 42.0).abs() < 1e-12);
        assert!((parse_f64_or_nan("-1.5e3") - -1500.0).abs() < 1e-12);
    }

    #[test]
    fn test_parse_f64_or_nan_sentinels() {
        for sentinel in NAN_SENTINELS {
            assert!(
                parse_f64_or_nan(sentinel).is_nan(),
                "Expected NAN for sentinel {:?}",
                sentinel
            );
        }
        // Padded sentinels should also work via trim
        assert!(parse_f64_or_nan("  nan  ").is_nan());
        assert!(parse_f64_or_nan("  --  ").is_nan());
    }

    #[test]
    fn test_parse_f64_or_nan_garbage() {
        assert!(parse_f64_or_nan("abc").is_nan());
        assert!(parse_f64_or_nan("N/A").is_nan());
    }

    #[test]
    fn test_parse_hapi_spacephysics_f64_or_nan_fill_values() {
        assert!(parse_hapi_spacephysics_f64_or_nan("-1.0E31").is_nan());
        assert!(parse_hapi_spacephysics_f64_or_nan("1.0E31").is_nan());
        assert!(parse_hapi_spacephysics_f64_or_nan("-9.9e30").is_nan());
        assert!((parse_hapi_spacephysics_f64_or_nan("12.5") - 12.5).abs() < 1e-12);
    }

    #[test]
    fn test_parse_f64_or_zero() {
        assert!((parse_f64_or_zero("1.234") - 1.234).abs() < 1e-12);
        assert!((parse_f64_or_zero("") - 0.0).abs() < 1e-12);
        assert!((parse_f64_or_zero("abc") - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_parse_sexagesimal_ra_to_deg() {
        let ra_deg = parse_sexagesimal_ra_to_deg("01 46 22.407");
        assert!((ra_deg - 26.5933625).abs() < 1e-6);
        assert!((parse_sexagesimal_ra_to_deg("180.0") - 180.0).abs() < 1e-12);
    }

    #[test]
    fn test_parse_sexagesimal_dec_to_deg() {
        let dec_deg = parse_sexagesimal_dec_to_deg("+61 45 03.19");
        assert!((dec_deg - 61.75088611111111).abs() < 1e-6);
        let neg_dec = parse_sexagesimal_dec_to_deg("-72 11 33.8");
        assert!((neg_dec + 72.19272222222222).abs() < 1e-6);
    }

    #[test]
    fn test_galactic_to_equatorial_j2000_galactic_center() {
        let (ra_deg, dec_deg) = galactic_to_equatorial_j2000(0.0, 0.0);
        assert!((ra_deg - 266.40499).abs() < 0.02, "ra_deg={}", ra_deg);
        assert!((dec_deg + 28.93617).abs() < 0.02, "dec_deg={}", dec_deg);
    }

    #[test]
    fn test_parse_hapi_time_to_ydh() {
        let parsed = parse_hapi_time_to_ydh("2016-01-02T03:04:05.000Z").expect("time");
        assert_eq!(parsed, (2016, 2, 3));
    }
}
