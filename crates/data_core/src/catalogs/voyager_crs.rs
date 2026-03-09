//! Voyager CRS (Cosmic Ray Subsystem) particle count rate parser.
//!
//! The Voyager CRS instrument measures energetic particle count rates across
//! 8 proton energy channels (C-D detectors: ~3-500 MeV/nuc), 4 helium channels,
//! and 4 electron channels. These data constrain the Parker modulation potential
//! phi(r) as a function of heliocentric distance (V1: up to 157 AU, V2: up to 134 AU).
//!
//! SPDF CRS data path (daily ASCII files):
//!   https://spdf.gsfc.nasa.gov/pub/data/voyager/voyager{1|2}/particle/crs/
//!
//! File naming: vy{N}crs_{YEAR}.asc
//!
//! Format: space-delimited ASCII, columns:
//!   0: Year (decimal fraction, e.g., 1980.003)
//!   1: Heliocentric distance (AU)
//!   2-9: Proton count rates by energy channel (counts/s)
//!   10-13: Helium count rates (counts/s)
//!   14-17: Electron count rates (counts/s)
//!
//! Fill values: 999.9 (data gap), -1.0 (out of range)
//!
//! Energy channels (approximate midpoint energies in MeV/nuc):
//!   Proton: CH1=3.4, CH2=6.2, CH3=10.5, CH4=18.4, CH5=30.2, CH6=53.0, CH7=84.0, CH8=500.0
//!   Helium: CH1=4.0, CH2=7.2, CH3=24.0, CH4=95.0
//!   Electron: CH1=4.0, CH2=7.0, CH3=15.0, CH4=70.0

/// Fill value threshold for CRS count rates.
const FILL_RATE: f64 = 900.0;
/// Negative fill sentinel.
const FILL_NEGATIVE: f64 = -0.5;
/// Minimum expected columns in a valid CRS row.
const MIN_COLUMNS: usize = 18;

/// Proton energy channel midpoints in MeV/nuc (CRS D1-D8 detectors).
pub const PROTON_CHANNEL_MEV: [f64; 8] = [3.4, 6.2, 10.5, 18.4, 30.2, 53.0, 84.0, 500.0];

/// Helium energy channel midpoints in MeV/nuc.
pub const HELIUM_CHANNEL_MEV: [f64; 4] = [4.0, 7.2, 24.0, 95.0];

/// Electron energy channel midpoints in MeV.
pub const ELECTRON_CHANNEL_MEV: [f64; 4] = [4.0, 7.0, 15.0, 70.0];

/// A single CRS measurement record (daily or sub-daily cadence).
#[derive(Clone, Debug)]
pub struct VoyagerCrsRecord {
    /// Decimal year (e.g., 1980.003).
    pub decimal_year: f64,
    /// Spacecraft identifier: 1 or 2.
    pub spacecraft: u8,
    /// Heliocentric distance (AU).
    pub distance_au: f64,
    /// Proton count rates by energy channel (counts/s). NaN = fill/gap.
    pub proton_rates: [f64; 8],
    /// Helium count rates by energy channel (counts/s). NaN = fill/gap.
    pub helium_rates: [f64; 4],
    /// Electron count rates by energy channel (counts/s). NaN = fill/gap.
    pub electron_rates: [f64; 4],
    /// True if any channel has fill values.
    pub fill_flag: bool,
}

/// Parse a fill-aware count rate from a string token.
fn parse_rate(s: &str) -> f64 {
    match s.trim().parse::<f64>() {
        Ok(v) if !(FILL_NEGATIVE..=FILL_RATE).contains(&v) => f64::NAN,
        Ok(v) => v,
        Err(_) => f64::NAN,
    }
}

/// Parse Voyager CRS ASCII data from a string.
///
/// Returns a Vec of records and the number of skipped lines.
pub fn parse_voyager_crs(
    data: &str,
    spacecraft: u8,
) -> (Vec<VoyagerCrsRecord>, usize) {
    let mut records = Vec::new();
    let mut skipped = 0_usize;

    for line in data.lines() {
        let line = line.trim();
        // Skip comment lines (start with # or ;)
        if line.starts_with('#') || line.starts_with(';') || line.is_empty() {
            continue;
        }

        let cols: Vec<&str> = line.split_whitespace().collect();
        if cols.len() < MIN_COLUMNS {
            skipped += 1;
            continue;
        }

        let decimal_year = match cols[0].parse::<f64>() {
            Ok(v) if v > 1900.0 && v < 2100.0 => v,
            _ => {
                skipped += 1;
                continue;
            }
        };

        let distance_au = match cols[1].parse::<f64>() {
            Ok(v) if v > 0.0 && v < 1000.0 => v,
            _ => f64::NAN,
        };

        let mut proton_rates = [f64::NAN; 8];
        for (i, r) in proton_rates.iter_mut().enumerate() {
            *r = parse_rate(cols[2 + i]);
        }
        let mut helium_rates = [f64::NAN; 4];
        for (i, r) in helium_rates.iter_mut().enumerate() {
            *r = parse_rate(cols[10 + i]);
        }
        let mut electron_rates = [f64::NAN; 4];
        for (i, r) in electron_rates.iter_mut().enumerate() {
            *r = parse_rate(cols[14 + i]);
        }

        let fill_flag = proton_rates
            .iter()
            .chain(helium_rates.iter())
            .chain(electron_rates.iter())
            .any(|v| v.is_nan());

        records.push(VoyagerCrsRecord {
            decimal_year,
            spacecraft,
            distance_au,
            proton_rates,
            helium_rates,
            electron_rates,
            fill_flag,
        });
    }

    (records, skipped)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_CRS: &str = "\
# Voyager 1 CRS data
1980.050  5.2  2.3  4.1  6.7  8.2  3.1  1.2  0.8  0.4  1.1  0.9  0.6  0.3  2.1  1.8  1.0  0.5
1980.053  5.3  999.9  4.2  6.8  8.3  3.2  1.3  0.9  0.5  1.2  1.0  0.7  0.4  2.2  1.9  1.1  0.6
1980.056  5.4  2.5  4.3  6.9  8.4  3.3  1.4  1.0  0.6  1.3  1.1  0.8  0.5  2.3  2.0  1.2  0.7
";

    #[test]
    fn test_parse_sample_crs() {
        let (records, skipped) = parse_voyager_crs(SAMPLE_CRS, 1);
        assert_eq!(records.len(), 3);
        assert_eq!(skipped, 0);
        assert_eq!(records[0].spacecraft, 1);
        assert!((records[0].distance_au - 5.2).abs() < 0.01);
    }

    #[test]
    fn test_fill_flag_propagates() {
        let (records, _) = parse_voyager_crs(SAMPLE_CRS, 1);
        // Second row has 999.9 in first proton channel
        assert!(records[1].fill_flag, "fill_flag should be set for row 2");
        assert!(records[1].proton_rates[0].is_nan(), "999.9 should parse as NaN");
    }

    #[test]
    fn test_valid_row_no_fill_flag() {
        let (records, _) = parse_voyager_crs(SAMPLE_CRS, 1);
        assert!(!records[0].fill_flag, "row 0 has no fill values");
        assert!(!records[2].fill_flag, "row 2 has no fill values");
    }

    #[test]
    fn test_comment_lines_skipped() {
        let data = "# header\n; another comment\n1980.100  10.0  1.0  2.0  3.0  4.0  5.0  6.0  7.0  8.0  1.0  2.0  3.0  4.0  1.0  2.0  3.0  4.0\n";
        let (records, _) = parse_voyager_crs(data, 2);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].spacecraft, 2);
    }
}
