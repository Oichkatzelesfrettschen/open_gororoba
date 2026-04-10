//! Rosetta RPC-MAG data provider for cometary draping analysis.
//!
//! The Rosetta Plasma Consortium Magnetometer (RPC-MAG) measured the magnetic
//! field around comet 67P/Churyumov-Gerasimenko from 2014-2016. The draping
//! pattern (solar wind B-field wrapping around the coma) produces distinct
//! boundary transitions at the bow shock and diamagnetic cavity boundary.
//!
//! Data source: AMDA HAPI (IRAP Toulouse) -- `ros-mag-cesam`
//! Reference: Glassmeier et al. (2007), Space Sci. Rev. 128, 649

use crate::parse::parse_hapi_time_to_ydh;
use chrono::{DateTime, Datelike, Timelike, Utc};
use csv::ReaderBuilder;
use std::collections::BTreeMap;

/// Rosetta RPC-MAG hourly-averaged record.
#[derive(Debug, Clone)]
pub struct RosettaMagRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub bx_cseq: f64,
    pub by_cseq: f64,
    pub bz_cseq: f64,
    pub b_magnitude: f64,
}

/// Rosetta RPC-MAG minute-resolution record for boundary detection.
#[derive(Debug, Clone)]
pub struct RosettaMagMinuteRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub minute: u8,
    pub elapsed_hours: f64,
    pub bx_cseq: f64,
    pub by_cseq: f64,
    pub bz_cseq: f64,
    pub b_magnitude: f64,
}

#[derive(Default)]
struct MagHourAccumulator {
    bx_sum: f64,
    by_sum: f64,
    bz_sum: f64,
    count: usize,
}

/// Parse AMDA HAPI CSV for Rosetta RPC-MAG into hourly records.
///
/// AMDA CSV includes a header row; column positions are determined from headers.
pub fn parse_rosetta_amda_mag_csv(content: &str) -> Vec<RosettaMagRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(content.as_bytes());

    let mut hourly: BTreeMap<(u16, u16, u8), MagHourAccumulator> = BTreeMap::new();

    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else { continue };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            // AMDA sometimes uses space-separated ISO format
            if let Some((y, d, h)) = parse_amda_timestamp(time) {
                let entry = hourly.entry((y, d, h)).or_default();
                let bx = parse_amda_f64(record.get(1));
                let by = parse_amda_f64(record.get(2));
                let bz = parse_amda_f64(record.get(3));
                if bx.is_finite() && by.is_finite() && bz.is_finite() {
                    entry.bx_sum += bx;
                    entry.by_sum += by;
                    entry.bz_sum += bz;
                    entry.count += 1;
                }
            }
            continue;
        };

        let bx = parse_amda_f64(record.get(1));
        let by = parse_amda_f64(record.get(2));
        let bz = parse_amda_f64(record.get(3));
        if !bx.is_finite() || !by.is_finite() || !bz.is_finite() {
            continue;
        }

        let entry = hourly.entry((year, doy, hour)).or_default();
        entry.bx_sum += bx;
        entry.by_sum += by;
        entry.bz_sum += bz;
        entry.count += 1;
    }

    hourly
        .into_iter()
        .filter_map(|((year, doy, hour), acc)| {
            if acc.count == 0 {
                return None;
            }
            let n = acc.count as f64;
            let bx = acc.bx_sum / n;
            let by = acc.by_sum / n;
            let bz = acc.bz_sum / n;
            Some(RosettaMagRecord {
                year,
                doy,
                hour,
                bx_cseq: bx,
                by_cseq: by,
                bz_cseq: bz,
                b_magnitude: (bx * bx + by * by + bz * bz).sqrt(),
            })
        })
        .collect()
}

/// Parse AMDA HAPI CSV into minute-averaged records for draping boundary detection.
pub fn parse_rosetta_amda_mag_csv_minutes(content: &str) -> Vec<RosettaMagMinuteRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(content.as_bytes());

    #[derive(Default)]
    struct MinuteAcc {
        bx_sum: f64,
        by_sum: f64,
        bz_sum: f64,
        count: usize,
    }

    let mut buckets: BTreeMap<(u16, u16, u8, u8), MinuteAcc> = BTreeMap::new();

    for record in reader.records().flatten() {
        let Some(time_str) = record.get(0) else {
            continue;
        };

        let (year, doy, hour, minute) = if let Ok(dt) = DateTime::parse_from_rfc3339(time_str) {
            let utc = dt.with_timezone(&Utc);
            (
                utc.year() as u16,
                utc.ordinal() as u16,
                utc.hour() as u8,
                utc.minute() as u8,
            )
        } else if let Some((y, d, h, m)) = parse_amda_timestamp_full(time_str) {
            (y, d, h, m)
        } else {
            continue;
        };

        let bx = parse_amda_f64(record.get(1));
        let by = parse_amda_f64(record.get(2));
        let bz = parse_amda_f64(record.get(3));
        if !bx.is_finite() || !by.is_finite() || !bz.is_finite() {
            continue;
        }

        let acc = buckets.entry((year, doy, hour, minute)).or_default();
        acc.bx_sum += bx;
        acc.by_sum += by;
        acc.bz_sum += bz;
        acc.count += 1;
    }

    let keys: Vec<_> = buckets.keys().copied().collect();
    let first = keys.first().copied();

    keys.into_iter()
        .filter_map(|key| {
            let acc = &buckets[&key];
            if acc.count == 0 {
                return None;
            }
            let n = acc.count as f64;
            let (year, doy, hour, minute) = key;
            let bx = acc.bx_sum / n;
            let by = acc.by_sum / n;
            let bz = acc.bz_sum / n;

            let elapsed = match first {
                Some((fy, fd, fh, fm)) => {
                    let day_diff = (year as f64 - fy as f64) * 365.25 + (doy as f64 - fd as f64);
                    day_diff * 24.0
                        + (hour as f64 - fh as f64)
                        + (minute as f64 - fm as f64) / 60.0
                }
                None => 0.0,
            };

            Some(RosettaMagMinuteRecord {
                year,
                doy,
                hour,
                minute,
                elapsed_hours: elapsed,
                bx_cseq: bx,
                by_cseq: by,
                bz_cseq: bz,
                b_magnitude: (bx * bx + by * by + bz * bz).sqrt(),
            })
        })
        .collect()
}

/// Parse AMDA-style float (empty, NaN, or huge sentinel -> NaN).
pub(crate) fn parse_amda_f64(s: Option<&str>) -> f64 {
    let s = s.unwrap_or("").trim();
    if s.is_empty() || s == "NaN" || s == "nan" {
        return f64::NAN;
    }
    match s.parse::<f64>() {
        Ok(v) if v.abs() < 1.0e30 => v,
        _ => f64::NAN,
    }
}

/// Parse AMDA timestamp formats (space-separated ISO or RFC3339).
pub(crate) fn parse_amda_timestamp(s: &str) -> Option<(u16, u16, u8)> {
    // AMDA often uses "YYYY-MM-DD HH:MM:SS.sss" (space instead of T)
    let normalized = s.trim().replace(' ', "T");
    let with_tz = if normalized.ends_with('Z') || normalized.contains('+') {
        normalized
    } else {
        format!("{normalized}Z")
    };
    let dt = DateTime::parse_from_rfc3339(&with_tz).ok()?;
    let utc = dt.with_timezone(&Utc);
    Some((utc.year() as u16, utc.ordinal() as u16, utc.hour() as u8))
}

/// Parse AMDA timestamp with minute resolution.
pub(crate) fn parse_amda_timestamp_full(s: &str) -> Option<(u16, u16, u8, u8)> {
    let normalized = s.trim().replace(' ', "T");
    let with_tz = if normalized.ends_with('Z') || normalized.contains('+') {
        normalized
    } else {
        format!("{normalized}Z")
    };
    let dt = DateTime::parse_from_rfc3339(&with_tz).ok()?;
    let utc = dt.with_timezone(&Utc);
    Some((
        utc.year() as u16,
        utc.ordinal() as u16,
        utc.hour() as u8,
        utc.minute() as u8,
    ))
}
