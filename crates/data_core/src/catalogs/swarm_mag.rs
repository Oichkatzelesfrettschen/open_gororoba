//! Swarm MAG data provider for ionospheric FAC sheet detection.
//!
//! ESA Swarm (3 LEO spacecraft: A, B, C) measures the geomagnetic field
//! at 450-530 km altitude. High-latitude passes cross field-aligned current
//! (FAC) sheets. This module tests whether the CD associator responds to
//! ionospheric current sheets -- a much subtler boundary than magnetopause
//! or bow shock crossings.
//!
//! Data: AMDA HAPI -- `swarma-mag-all` (NEC frame, 1 Hz).
//! Reference: Friis-Christensen et al. (2006), Earth Planets Space 58, 351

use chrono::{DateTime, Datelike, Timelike, Utc};
use csv::ReaderBuilder;
use std::collections::BTreeMap;

/// Minute-resolution Swarm MAG record in NEC (North-East-Center) frame.
#[derive(Debug, Clone)]
pub struct SwarmMagMinuteRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub minute: u8,
    pub elapsed_hours: f64,
    pub b_north: f64,
    pub b_east: f64,
    pub b_center: f64,
    pub b_magnitude: f64,
}

#[cfg(feature = "fetch")]
pub use super::swarm_mag_fetch::SwarmMagProvider;

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

/// Parse Swarm AMDA MAG CSV into minute-averaged records.
pub fn parse_swarm_amda_mag_csv_minutes(content: &str) -> Vec<SwarmMagMinuteRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(content.as_bytes());

    #[derive(Default)]
    struct MinuteAcc {
        bn: f64,
        be: f64,
        bc: f64,
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
        } else if let Some(t) = parse_amda_timestamp_full(time_str) {
            t
        } else {
            continue;
        };

        let bn = parse_amda_f64(record.get(1));
        let be = parse_amda_f64(record.get(2));
        let bc = parse_amda_f64(record.get(3));
        if !bn.is_finite() || !be.is_finite() || !bc.is_finite() {
            continue;
        }

        let acc = buckets.entry((year, doy, hour, minute)).or_default();
        acc.bn += bn;
        acc.be += be;
        acc.bc += bc;
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
            let bn = acc.bn / n;
            let be = acc.be / n;
            let bc = acc.bc / n;

            let elapsed = match first {
                Some((fy, fd, fh, fm)) => {
                    (year as f64 - fy as f64) * 365.25 * 24.0
                        + (doy as f64 - fd as f64) * 24.0
                        + (hour as f64 - fh as f64)
                        + (minute as f64 - fm as f64) / 60.0
                }
                None => 0.0,
            };

            Some(SwarmMagMinuteRecord {
                year,
                doy,
                hour,
                minute,
                elapsed_hours: elapsed,
                b_north: bn,
                b_east: be,
                b_center: bc,
                b_magnitude: (bn * bn + be * be + bc * bc).sqrt(),
            })
        })
        .collect()
}
