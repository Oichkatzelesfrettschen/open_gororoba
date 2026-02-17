//! Canonical parsing helpers for catalog and geophysical data fields.
//!
//! Every catalog parser used to carry its own `parse_f64()` with a slightly
//! different sentinel set. This module consolidates them into two functions
//! whose sentinel lists are the union of all per-catalog variants.

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

/// Parse a trimmed string to `f64`, returning 0.0 for unparseable values.
///
/// Used by GWTC parser where missing numeric fields default to zero
/// (mass, spin, distance columns in GWOSC CSV).
pub fn parse_f64_or_zero(s: &str) -> f64 {
    s.trim().parse::<f64>().unwrap_or(0.0)
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
    fn test_parse_f64_or_zero() {
        assert!((parse_f64_or_zero("1.234") - 1.234).abs() < 1e-12);
        assert!((parse_f64_or_zero("") - 0.0).abs() < 1e-12);
        assert!((parse_f64_or_zero("abc") - 0.0).abs() < 1e-12);
    }
}
