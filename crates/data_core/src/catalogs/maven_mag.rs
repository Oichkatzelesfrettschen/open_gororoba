//! MAVEN MAG parse types for Mars induced magnetosphere boundary analysis.
//!
//! MAVEN (Mars Atmosphere and Volatile EvolutioN) orbits Mars measuring the
//! magnetic field through the induced magnetosphere boundary (IMB) and
//! magnetic pileup boundary (MPB). Mars has no global dipole but has crustal
//! magnetic anomalies creating patchy mini-magnetospheres.
//!
//! Data: 1-second outboard MAG in Sun-State coordinates.
//! Source: <https://cdaweb.gsfc.nasa.gov/hapi/info?id=MVN_MAG_L2-SUNSTATE-1SEC>
//! Reference: Connerney et al. (2015), Space Sci. Rev. 195, 257
//!
//! Fetch/provider support is in `maven_mag_fetch` (feature-gated on `fetch`).

use crate::parse::parse_hapi_spacephysics_f64_or_nan;
use chrono::{DateTime, Datelike, Timelike, Utc};
use csv::ReaderBuilder;
use std::collections::BTreeMap;

/// Minute-resolution MAVEN MAG record in Sun-State coordinates.
#[derive(Debug, Clone)]
pub struct MavenMagMinuteRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub minute: u8,
    pub elapsed_hours: f64,
    pub bx_ss: f64,
    pub by_ss: f64,
    pub bz_ss: f64,
    pub b_magnitude: f64,
}

#[cfg(feature = "fetch")]
pub use super::maven_mag_fetch::MavenMagProvider;

/// Parse MAVEN MAG HAPI CSV into minute-averaged records.
pub fn parse_maven_mag_hapi_csv_minutes(content: &str) -> Vec<MavenMagMinuteRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());

    #[derive(Default)]
    struct MinuteAcc {
        bx: f64,
        by: f64,
        bz: f64,
        count: usize,
    }

    let mut buckets: BTreeMap<(u16, u16, u8, u8), MinuteAcc> = BTreeMap::new();

    for record in reader.records().flatten() {
        let Some(time_str) = record.get(0) else {
            continue;
        };
        let Ok(dt) = DateTime::parse_from_rfc3339(time_str) else {
            continue;
        };
        let utc = dt.with_timezone(&Utc);

        // OB_B columns: indices 1, 2, 3 (3-vector)
        let bx = parse_hapi_spacephysics_f64_or_nan(record.get(1).unwrap_or(""));
        let by = parse_hapi_spacephysics_f64_or_nan(record.get(2).unwrap_or(""));
        let bz = parse_hapi_spacephysics_f64_or_nan(record.get(3).unwrap_or(""));
        if !bx.is_finite() || !by.is_finite() || !bz.is_finite() {
            continue;
        }

        let key = (
            utc.year() as u16,
            utc.ordinal() as u16,
            utc.hour() as u8,
            utc.minute() as u8,
        );
        let acc = buckets.entry(key).or_default();
        acc.bx += bx;
        acc.by += by;
        acc.bz += bz;
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
            let bx = acc.bx / n;
            let by = acc.by / n;
            let bz = acc.bz / n;

            let elapsed = match first {
                Some((fy, fd, fh, fm)) => {
                    (year as f64 - fy as f64) * 365.25 * 24.0
                        + (doy as f64 - fd as f64) * 24.0
                        + (hour as f64 - fh as f64)
                        + (minute as f64 - fm as f64) / 60.0
                }
                None => 0.0,
            };

            Some(MavenMagMinuteRecord {
                year,
                doy,
                hour,
                minute,
                elapsed_hours: elapsed,
                bx_ss: bx,
                by_ss: by,
                bz_ss: bz,
                b_magnitude: (bx * bx + by * by + bz * bz).sqrt(),
            })
        })
        .collect()
}
