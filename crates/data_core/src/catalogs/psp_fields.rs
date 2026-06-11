//! Parker Solar Probe FIELDS MAG RTN high-cadence parser.
//!
//! The operational fleet lane already uses the merged hourly PSP dataset.
//! This module promotes the higher-cadence FIELDS magnetometer feed into the
//! Rust parse path so it can enrich feature cubes and cross-check the merged
//! hourly magnetic-field lane.
//!
//! Fetch logic lives in `psp_fields_fetch`.

use crate::parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh};
use chrono::{DateTime, Datelike, Timelike, Utc};
use csv::ReaderBuilder;
use std::collections::BTreeMap;

#[derive(Debug, Clone)]
pub struct PspFieldsMagRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub br: f64,
    pub bt: f64,
    pub bn: f64,
    pub b_magnitude: f64,
}

#[cfg(feature = "fetch")]
pub use super::psp_fields_fetch::{
    PspFieldsProvider, parse_psp_fields_cdf_file, parse_psp_fields_file,
};

#[derive(Default)]
pub(crate) struct MagAccumulator {
    pub br_sum: f64,
    pub bt_sum: f64,
    pub bn_sum: f64,
    pub count: usize,
}

pub fn parse_psp_fields_hapi_csv(content: &str) -> Vec<PspFieldsMagRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let headers = match reader.headers() {
        Ok(headers) => headers.clone(),
        Err(_) => return Vec::new(),
    };
    let br_col = headers
        .iter()
        .position(|value| matches!(value, "psp_fld_l2_mag_RTN_0" | "psp_fld_l2_mag_RTN_1min_0"));
    let bt_col = headers
        .iter()
        .position(|value| matches!(value, "psp_fld_l2_mag_RTN_1" | "psp_fld_l2_mag_RTN_1min_1"));
    let bn_col = headers
        .iter()
        .position(|value| matches!(value, "psp_fld_l2_mag_RTN_2" | "psp_fld_l2_mag_RTN_1min_2"));
    let (Some(br_col), Some(bt_col), Some(bn_col)) = (br_col, bt_col, bn_col) else {
        return Vec::new();
    };
    let mut hourly: BTreeMap<(u16, u16, u8), MagAccumulator> = BTreeMap::new();
    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else {
            continue;
        };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            continue;
        };
        let br = parse_hapi_spacephysics_f64_or_nan(record.get(br_col).unwrap_or(""));
        let bt = parse_hapi_spacephysics_f64_or_nan(record.get(bt_col).unwrap_or(""));
        let bn = parse_hapi_spacephysics_f64_or_nan(record.get(bn_col).unwrap_or(""));
        if !br.is_finite() || !bt.is_finite() || !bn.is_finite() {
            continue;
        }
        let entry = hourly.entry((year, doy, hour)).or_default();
        entry.br_sum += br;
        entry.bt_sum += bt;
        entry.bn_sum += bn;
        entry.count += 1;
    }
    hourly
        .into_iter()
        .map(|((year, doy, hour), acc)| {
            let count = acc.count.max(1) as f64;
            let br = acc.br_sum / count;
            let bt = acc.bt_sum / count;
            let bn = acc.bn_sum / count;
            PspFieldsMagRecord {
                year,
                doy,
                hour,
                br,
                bt,
                bn,
                b_magnitude: (br * br + bt * bt + bn * bn).sqrt(),
            }
        })
        .collect()
}

// ============================================================================
// Minute-resolution record and parser
// ============================================================================

/// One minute of PSP FIELDS L2 MAG RTN data (PSP_FLD_L2_MAG_RTN_1MIN).
///
/// Unlike `PspFieldsMagRecord` (which averages sub-minute samples to hourly),
/// this preserves each 1-minute row without accumulation.  Used by the
/// Alfvenic-control experiment (E-239) which needs minute-level timestamps to
/// classify windows and run Takens-delay embeddings.
#[derive(Debug, Clone)]
pub struct PspFieldsMagMinuteRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub minute: u8,
    /// Elapsed hours from first record; set to 0.0 by the parser, caller fills.
    pub elapsed_hours: f64,
    pub br: f64,
    pub bt: f64,
    pub bn: f64,
    pub b_magnitude: f64,
}

/// Parse a `PSP_FLD_L2_MAG_RTN_1MIN` HAPI CSV preserving minute resolution.
///
/// Each valid row becomes exactly one `PspFieldsMagMinuteRecord`.  No
/// additional averaging is performed: `PSP_FLD_L2_MAG_RTN_1MIN` is a
/// designated SPDF Level-2 product whose samples already represent the
/// arithmetic mean of all valid high-rate FIELDS magnetometer samples
/// (native ~73 Hz or higher) within each calendar minute, computed by
/// the FIELDS pipeline at SPDF.  This is the decimation protocol for
/// all Takens delay embeddings built from PSP data.
/// `elapsed_hours` is set to 0.0 and must be recomputed by the caller
/// relative to a reference time.
pub fn parse_psp_fields_hapi_csv_minutes(content: &str) -> Vec<PspFieldsMagMinuteRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let headers = match reader.headers() {
        Ok(h) => h.clone(),
        Err(_) => return Vec::new(),
    };
    let br_col = headers
        .iter()
        .position(|v| matches!(v, "psp_fld_l2_mag_RTN_0" | "psp_fld_l2_mag_RTN_1min_0"));
    let bt_col = headers
        .iter()
        .position(|v| matches!(v, "psp_fld_l2_mag_RTN_1" | "psp_fld_l2_mag_RTN_1min_1"));
    let bn_col = headers
        .iter()
        .position(|v| matches!(v, "psp_fld_l2_mag_RTN_2" | "psp_fld_l2_mag_RTN_1min_2"));
    let (Some(br_col), Some(bt_col), Some(bn_col)) = (br_col, bt_col, bn_col) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for record in reader.records().flatten() {
        let Some(time_str) = record.get(0) else {
            continue;
        };
        // Parse RFC3339 timestamp; extract year/doy/hour/minute.
        let Ok(dt) = DateTime::parse_from_rfc3339(time_str.trim()) else {
            continue;
        };
        let utc: DateTime<Utc> = dt.with_timezone(&Utc);
        let year = utc.year() as u16;
        let doy = utc.ordinal() as u16;
        let hour = utc.hour() as u8;
        let minute = utc.minute() as u8;
        let br = parse_hapi_spacephysics_f64_or_nan(record.get(br_col).unwrap_or(""));
        let bt = parse_hapi_spacephysics_f64_or_nan(record.get(bt_col).unwrap_or(""));
        let bn = parse_hapi_spacephysics_f64_or_nan(record.get(bn_col).unwrap_or(""));
        if !br.is_finite() || !bt.is_finite() || !bn.is_finite() {
            continue;
        }
        let b_magnitude = (br * br + bt * bt + bn * bn).sqrt();
        out.push(PspFieldsMagMinuteRecord {
            year,
            doy,
            hour,
            minute,
            elapsed_hours: 0.0,
            br,
            bt,
            bn,
            b_magnitude,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_psp_fields_hapi_csv() {
        let csv = "Time,psp_fld_l2_mag_RTN_0,psp_fld_l2_mag_RTN_1,psp_fld_l2_mag_RTN_2\n\
2020-01-01T00:00:00.009929600Z,-5.07e+00,7.14e+00,5.47e-01\n\
2020-01-01T00:00:00.119156352Z,-5.01e+00,7.02e+00,5.42e-01\n";
        let rows = parse_psp_fields_hapi_csv(csv);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.year, 2020);
        assert_eq!(row.hour, 0);
        assert!((row.br + 5.04).abs() < 0.05);
        assert!(row.b_magnitude > 8.5);
    }

    #[test]
    fn test_parse_psp_fields_1min_csv() {
        let csv = "Time,psp_fld_l2_mag_RTN_1min_0,psp_fld_l2_mag_RTN_1min_1,psp_fld_l2_mag_RTN_1min_2\n\
2020-02-01T00:00:30Z,-4.67e+01,5.02e+01,-3.30e+00\n\
2020-02-01T00:01:30Z,-4.56e+01,4.81e+01,2.27e+00\n";
        let rows = parse_psp_fields_hapi_csv(csv);
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.year, 2020);
        assert_eq!(row.doy, 32);
        assert_eq!(row.hour, 0);
        assert!(row.b_magnitude > 60.0);
    }
}
