//! THEMIS/ARTEMIS FGM data provider.
//!
//! THEMIS (Time History of Events and Macroscale Interactions during Substorms)
//! has 5 probes (THA-THE). After 2011, THB and THC were redirected to lunar
//! orbit as ARTEMIS. This module handles all 5 probes via CDAWeb HAPI.
//!
//! THEMIS FGM provides spin-resolution (~3s) magnetic field in GSE coordinates.
//! ARTEMIS (THB/THC post-2011) crosses the lunar wake.
//!
//! Source: <https://cdaweb.gsfc.nasa.gov/hapi/info?id=THA_L2_FGM@0>
//! Reference: Auster et al. (2008), Space Sci. Rev. 141, 235

use crate::parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh};
use chrono::{DateTime, Datelike, Timelike, Utc};
use csv::ReaderBuilder;
use std::collections::BTreeMap;

/// Minute-resolution THEMIS FGM record.
#[derive(Debug, Clone)]
pub struct ThemisFgmMinuteRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub minute: u8,
    pub elapsed_hours: f64,
    pub bx_gse: f64,
    pub by_gse: f64,
    pub bz_gse: f64,
    pub b_magnitude: f64,
}

/// Hourly THEMIS FGM record.
#[derive(Debug, Clone)]
pub struct ThemisFgmRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub bx_gse: f64,
    pub by_gse: f64,
    pub bz_gse: f64,
    pub b_magnitude: f64,
}

/// Parse THEMIS FGM HAPI CSV into minute-averaged records.
pub fn parse_themis_fgm_hapi_csv_minutes(content: &str, probe: &str) -> Vec<ThemisFgmMinuteRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());

    let headers = match reader.headers() {
        Ok(h) => h.clone(),
        Err(_) => return Vec::new(),
    };

    // Find GSE columns: {probe}_fgs_gse_0, _1, _2
    let prefix = format!("{}_fgs_gse", probe.to_lowercase());
    let bx_col = headers.iter().position(|h| h == format!("{prefix}_0"));
    let by_col = headers.iter().position(|h| h == format!("{prefix}_1"));
    let bz_col = headers.iter().position(|h| h == format!("{prefix}_2"));

    let (Some(bx_col), Some(by_col), Some(bz_col)) = (bx_col, by_col, bz_col) else {
        return Vec::new();
    };

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
        let Ok(dt) = DateTime::parse_from_rfc3339(time_str) else {
            continue;
        };
        let utc = dt.with_timezone(&Utc);

        let bx = parse_hapi_spacephysics_f64_or_nan(record.get(bx_col).unwrap_or(""));
        let by = parse_hapi_spacephysics_f64_or_nan(record.get(by_col).unwrap_or(""));
        let bz = parse_hapi_spacephysics_f64_or_nan(record.get(bz_col).unwrap_or(""));
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
                    (year as f64 - fy as f64) * 365.25 * 24.0
                        + (doy as f64 - fd as f64) * 24.0
                        + (hour as f64 - fh as f64)
                        + (minute as f64 - fm as f64) / 60.0
                }
                None => 0.0,
            };

            Some(ThemisFgmMinuteRecord {
                year,
                doy,
                hour,
                minute,
                elapsed_hours: elapsed,
                bx_gse: bx,
                by_gse: by,
                bz_gse: bz,
                b_magnitude: (bx * bx + by * by + bz * bz).sqrt(),
            })
        })
        .collect()
}

/// Parse THEMIS FGM HAPI CSV into hourly records.
pub fn parse_themis_fgm_hapi_csv(content: &str) -> Vec<ThemisFgmRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());

    #[derive(Default)]
    struct HourAcc {
        bx: f64,
        by: f64,
        bz: f64,
        count: usize,
    }

    let mut hourly: BTreeMap<(u16, u16, u8), HourAcc> = BTreeMap::new();

    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else { continue };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            continue;
        };

        // Columns 1,2,3 are the GSE vector for @0 dataset
        let bx = parse_hapi_spacephysics_f64_or_nan(record.get(1).unwrap_or(""));
        let by = parse_hapi_spacephysics_f64_or_nan(record.get(2).unwrap_or(""));
        let bz = parse_hapi_spacephysics_f64_or_nan(record.get(3).unwrap_or(""));
        if !bx.is_finite() || !by.is_finite() || !bz.is_finite() {
            continue;
        }

        let acc = hourly.entry((year, doy, hour)).or_default();
        acc.bx += bx;
        acc.by += by;
        acc.bz += bz;
        acc.count += 1;
    }

    hourly
        .into_iter()
        .filter_map(|((year, doy, hour), acc)| {
            if acc.count == 0 {
                return None;
            }
            let n = acc.count as f64;
            let bx = acc.bx / n;
            let by = acc.by / n;
            let bz = acc.bz / n;
            Some(ThemisFgmRecord {
                year,
                doy,
                hour,
                bx_gse: bx,
                by_gse: by,
                bz_gse: bz,
                b_magnitude: (bx * bx + by * by + bz * bz).sqrt(),
            })
        })
        .collect()
}

