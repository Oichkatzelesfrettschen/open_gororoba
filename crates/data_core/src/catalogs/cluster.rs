//! Cluster FGM data provider for multi-spacecraft boundary analysis.
//!
//! The Cluster mission (4 identical spacecraft: C1-C4) provides simultaneous
//! magnetic field measurements at different spatial separations (100 km to 2 RE).
//! This enables testing whether CD associator transitions are spatially coherent.
//!
//! FGM SPIN data provides spin-resolution (~4s) B-field in GSE coordinates
//! plus spacecraft position for timing analysis.
//!
//! Source: <https://cdaweb.gsfc.nasa.gov/hapi/info?id=C1_CP_FGM_SPIN>
//! Reference: Balogh et al. (2001), Ann. Geophys. 19, 1207

use crate::parse::parse_hapi_spacephysics_f64_or_nan;
use chrono::{DateTime, Datelike, Timelike, Utc};
use csv::ReaderBuilder;
use std::collections::BTreeMap;

/// Minute-resolution Cluster FGM record with spacecraft position.
#[derive(Debug, Clone)]
pub struct ClusterFgmMinuteRecord {
    pub probe_id: u8, // 1-4
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub minute: u8,
    pub elapsed_hours: f64,
    pub bx_gse: f64,
    pub by_gse: f64,
    pub bz_gse: f64,
    pub b_magnitude: f64,
    /// Spacecraft position in GSE (km), for timing analysis.
    pub x_gse_km: f64,
    pub y_gse_km: f64,
    pub z_gse_km: f64,
}

#[cfg(feature = "fetch")]
pub use super::cluster_fetch::ClusterFgmProvider;

/// Parse Cluster FGM SPIN HAPI CSV into minute-averaged records.
pub fn parse_cluster_fgm_hapi_csv_minutes(
    content: &str,
    probe_id: u8,
) -> Vec<ClusterFgmMinuteRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());

    let headers = match reader.headers() {
        Ok(h) => h.clone(),
        Err(_) => return Vec::new(),
    };

    // Columns: Time, B_vec_xyz_gse__C{n}_CP_FGM_SPIN_0/1/2, B_mag__C{n}_CP_FGM_SPIN,
    //          sc_pos_xyz_gse__C{n}_CP_FGM_SPIN_0/1/2
    let b_prefix = format!("B_vec_xyz_gse__C{probe_id}_CP_FGM_SPIN");
    let bmag_name = format!("B_mag__C{probe_id}_CP_FGM_SPIN");
    let pos_prefix = format!("sc_pos_xyz_gse__C{probe_id}_CP_FGM_SPIN");

    let bx_col = headers.iter().position(|h| h == format!("{b_prefix}_0"));
    let by_col = headers.iter().position(|h| h == format!("{b_prefix}_1"));
    let bz_col = headers.iter().position(|h| h == format!("{b_prefix}_2"));
    let bmag_col = headers.iter().position(|h| h == bmag_name);
    let px_col = headers.iter().position(|h| h == format!("{pos_prefix}_0"));
    let py_col = headers.iter().position(|h| h == format!("{pos_prefix}_1"));
    let pz_col = headers.iter().position(|h| h == format!("{pos_prefix}_2"));

    let (Some(bx_col), Some(by_col), Some(bz_col)) = (bx_col, by_col, bz_col) else {
        return Vec::new();
    };

    #[derive(Default)]
    struct MinuteAcc {
        bx: f64,
        by: f64,
        bz: f64,
        bmag: f64,
        px: f64,
        py: f64,
        pz: f64,
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

        let bmag = bmag_col
            .and_then(|c| record.get(c))
            .map(parse_hapi_spacephysics_f64_or_nan)
            .filter(|v| v.is_finite())
            .unwrap_or_else(|| (bx * bx + by * by + bz * bz).sqrt());

        let px = px_col
            .and_then(|c| record.get(c))
            .map(parse_hapi_spacephysics_f64_or_nan)
            .unwrap_or(f64::NAN);
        let py = py_col
            .and_then(|c| record.get(c))
            .map(parse_hapi_spacephysics_f64_or_nan)
            .unwrap_or(f64::NAN);
        let pz = pz_col
            .and_then(|c| record.get(c))
            .map(parse_hapi_spacephysics_f64_or_nan)
            .unwrap_or(f64::NAN);

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
        acc.bmag += bmag;
        acc.px += px;
        acc.py += py;
        acc.pz += pz;
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

            let elapsed = match first {
                Some((fy, fd, fh, fm)) => {
                    (year as f64 - fy as f64) * 365.25 * 24.0
                        + (doy as f64 - fd as f64) * 24.0
                        + (hour as f64 - fh as f64)
                        + (minute as f64 - fm as f64) / 60.0
                }
                None => 0.0,
            };

            Some(ClusterFgmMinuteRecord {
                probe_id,
                year,
                doy,
                hour,
                minute,
                elapsed_hours: elapsed,
                bx_gse: acc.bx / n,
                by_gse: acc.by / n,
                bz_gse: acc.bz / n,
                b_magnitude: acc.bmag / n,
                x_gse_km: acc.px / n,
                y_gse_km: acc.py / n,
                z_gse_km: acc.pz / n,
            })
        })
        .collect()
}
