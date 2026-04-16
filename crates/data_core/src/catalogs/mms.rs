//! MMS Fluxgate Magnetometer (FGM) data provider.
//!
//! The Magnetospheric Multiscale (MMS) mission provides high-cadence
//! magnetic field measurements. This module fetches Survey (SRVY) mode
//! Level 2 data via CDAWeb HAPI.
//!
//! Source: <https://cdaweb.gsfc.nasa.gov/hapi/info?id=MMS1_FGM_SRVY_L2>
//! Reference: Russell et al. (2016), Space Sci. Rev. 199, 189

use crate::parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh};
use chrono::{DateTime, Datelike, Timelike, Utc};
use csv::ReaderBuilder;
use std::collections::BTreeMap;

/// MMS FGM Survey Level 2 record.
#[derive(Debug, Clone)]
pub struct MmsFgmRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub bx_gse: f64,
    pub by_gse: f64,
    pub bz_gse: f64,
    pub b_magnitude: f64,
}

#[derive(Default)]
struct FgmHourAccumulator {
    bx_sum: f64,
    by_sum: f64,
    bz_sum: f64,
    bmag_sum: f64,
    count: usize,
}


pub fn parse_mms_fgm_hapi_csv(content: &str) -> Vec<MmsFgmRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let mut rows = Vec::new();

    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else {
            continue;
        };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else {
            continue;
        };

        rows.push(MmsFgmRecord {
            year,
            doy,
            hour,
            bx_gse: parse_hapi_spacephysics_f64_or_nan(record.get(1).unwrap_or("")),
            by_gse: parse_hapi_spacephysics_f64_or_nan(record.get(2).unwrap_or("")),
            bz_gse: parse_hapi_spacephysics_f64_or_nan(record.get(3).unwrap_or("")),
            b_magnitude: parse_hapi_spacephysics_f64_or_nan(record.get(4).unwrap_or("")),
        });
    }
    rows
}

/// MMS FGM record with minute-level resolution for boundary detection.
#[derive(Debug, Clone)]
pub struct MmsFgmMinuteRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub minute: u8,
    /// Fractional hours from epoch start (monotonically increasing).
    pub elapsed_hours: f64,
    pub bx_gse: f64,
    pub by_gse: f64,
    pub bz_gse: f64,
    pub b_magnitude: f64,
}

/// Parse HAPI CSV into minute-averaged records for crossing detection.
///
/// Groups raw high-cadence samples (~16 Hz SRVY) into 1-minute bins,
/// preserving sub-hour resolution needed for magnetopause identification.
pub fn parse_mms_fgm_hapi_csv_minutes(content: &str) -> Vec<MmsFgmMinuteRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());

    #[derive(Default)]
    struct MinuteAcc {
        bx_sum: f64,
        by_sum: f64,
        bz_sum: f64,
        bmag_sum: f64,
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
        let year = utc.year() as u16;
        let doy = utc.ordinal() as u16;
        let hour = utc.hour() as u8;
        let minute = utc.minute() as u8;

        let bx = parse_hapi_spacephysics_f64_or_nan(record.get(1).unwrap_or(""));
        let by = parse_hapi_spacephysics_f64_or_nan(record.get(2).unwrap_or(""));
        let bz = parse_hapi_spacephysics_f64_or_nan(record.get(3).unwrap_or(""));
        let bmag = parse_hapi_spacephysics_f64_or_nan(record.get(4).unwrap_or(""));

        if !bx.is_finite() || !by.is_finite() || !bz.is_finite() {
            continue;
        }

        let acc = buckets.entry((year, doy, hour, minute)).or_default();
        acc.bx_sum += bx;
        acc.by_sum += by;
        acc.bz_sum += bz;
        acc.bmag_sum += if bmag.is_finite() {
            bmag
        } else {
            (bx * bx + by * by + bz * bz).sqrt()
        };
        acc.count += 1;
    }

    // Compute elapsed hours from the first record for monotonic time axis.
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
                    let day_diff = (year as f64 - fy as f64) * 365.25 + (doy as f64 - fd as f64);
                    day_diff * 24.0 + (hour as f64 - fh as f64) + (minute as f64 - fm as f64) / 60.0
                }
                None => 0.0,
            };

            Some(MmsFgmMinuteRecord {
                year,
                doy,
                hour,
                minute,
                elapsed_hours: elapsed,
                bx_gse: acc.bx_sum / n,
                by_gse: acc.by_sum / n,
                bz_gse: acc.bz_sum / n,
                b_magnitude: acc.bmag_sum / n,
            })
        })
        .collect()
}

/// Detect magnetopause crossings from minute-resolution FGM data.
///
/// Uses the standard |B| gradient + rotation method:
/// 1. Compute sliding-window |B| variance (high variance = boundary layer)
/// 2. Detect |dB/dt| exceeding threshold (rapid field change)
/// 3. Look for B_z sign changes (field rotation across current sheet)
///
/// Returns indices into the minute record array where crossings are detected.
pub fn detect_magnetopause_crossings(
    records: &[MmsFgmMinuteRecord],
    window_minutes: usize,
    bmag_gradient_threshold: f64,
) -> Vec<usize> {
    detect_magnetopause_crossings_filtered(records, window_minutes, bmag_gradient_threshold, None)
}

/// Detect magnetopause crossings with an optional rotation filter.
///
/// If `rotation_threshold_deg` is `Some(thresh)`, a candidate must satisfy BOTH
/// the |B| gradient criterion AND a vector rotation of at least `thresh` degrees
/// between the pre- and post-window mean B vectors.  This filters out compressive
/// events (which change |B| without rotating it) that are not boundary crossings.
///
/// WHY: Pure |B| gradient labels include solar wind pressure pulses and flares.
/// Adding a rotation criterion (>=30 deg) selects events that are directional
/// discontinuities -- the hallmark of a current-sheet or boundary crossing.
pub fn detect_magnetopause_crossings_filtered(
    records: &[MmsFgmMinuteRecord],
    window_minutes: usize,
    bmag_gradient_threshold: f64,
    rotation_threshold_deg: Option<f64>,
) -> Vec<usize> {
    if records.len() < window_minutes * 2 + 1 {
        return vec![];
    }

    let half = window_minutes;
    let mut crossings = Vec::new();

    for i in half..records.len().saturating_sub(half) {
        let pre_start = i.saturating_sub(half);
        let post_end = (i + half).min(records.len());
        let n_pre = (i - pre_start) as f64;
        let n_post = (post_end - i) as f64;

        let pre_mean_b: f64 = records[pre_start..i]
            .iter()
            .map(|r| r.b_magnitude)
            .sum::<f64>()
            / n_pre;
        let post_mean_b: f64 = records[i..post_end]
            .iter()
            .map(|r| r.b_magnitude)
            .sum::<f64>()
            / n_post;
        let b_jump = (post_mean_b - pre_mean_b).abs();

        let pre_bz_mean: f64 =
            records[pre_start..i].iter().map(|r| r.bz_gse).sum::<f64>() / n_pre;
        let post_bz_mean: f64 =
            records[i..post_end].iter().map(|r| r.bz_gse).sum::<f64>() / n_post;
        let bz_sign_change = pre_bz_mean * post_bz_mean < 0.0;

        // Gradient criterion (unchanged from original).
        let gradient_ok = b_jump > bmag_gradient_threshold
            || (b_jump > bmag_gradient_threshold * 0.5 && bz_sign_change);
        if !gradient_ok {
            continue;
        }

        // Optional rotation criterion.
        if let Some(rot_thresh) = rotation_threshold_deg {
            let pre_bx = records[pre_start..i].iter().map(|r| r.bx_gse).sum::<f64>() / n_pre;
            let pre_by = records[pre_start..i].iter().map(|r| r.by_gse).sum::<f64>() / n_pre;
            let pre_bz = pre_bz_mean;
            let post_bx = records[i..post_end].iter().map(|r| r.bx_gse).sum::<f64>() / n_post;
            let post_by = records[i..post_end].iter().map(|r| r.by_gse).sum::<f64>() / n_post;
            let post_bz = post_bz_mean;
            let pre_mag = (pre_bx.powi(2) + pre_by.powi(2) + pre_bz.powi(2)).sqrt();
            let post_mag = (post_bx.powi(2) + post_by.powi(2) + post_bz.powi(2)).sqrt();
            if pre_mag > 0.1 && post_mag > 0.1 {
                let cos_a = ((pre_bx * post_bx + pre_by * post_by + pre_bz * post_bz)
                    / (pre_mag * post_mag))
                    .clamp(-1.0, 1.0);
                let rotation_deg = cos_a.acos().to_degrees();
                if rotation_deg < rot_thresh {
                    continue; // Insufficient rotation -- compressive, not boundary crossing.
                }
            }
        }

        let dominated = crossings
            .last()
            .is_some_and(|&prev: &usize| i.saturating_sub(prev) < window_minutes);
        if !dominated {
            crossings.push(i);
        }
    }

    crossings
}

/// Average high-cadence MMS records to hourly bins.
pub fn average_to_hourly(records: &[MmsFgmRecord]) -> Vec<MmsFgmRecord> {
    let mut hourly: BTreeMap<(u16, u16, u8), FgmHourAccumulator> = BTreeMap::new();
    for r in records {
        let entry = hourly.entry((r.year, r.doy, r.hour)).or_default();
        if r.bx_gse.is_finite() && r.by_gse.is_finite() && r.bz_gse.is_finite() {
            entry.bx_sum += r.bx_gse;
            entry.by_sum += r.by_gse;
            entry.bz_sum += r.bz_gse;
            entry.bmag_sum += r.b_magnitude;
            entry.count += 1;
        }
    }

    hourly
        .into_iter()
        .filter_map(|((year, doy, hour), acc)| {
            if acc.count == 0 {
                return None;
            }
            let n = acc.count as f64;
            Some(MmsFgmRecord {
                year,
                doy,
                hour,
                bx_gse: acc.bx_sum / n,
                by_gse: acc.by_sum / n,
                bz_gse: acc.bz_sum / n,
                b_magnitude: acc.bmag_sum / n,
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// MMS FPI DIS-MOMS: ion density + bulk velocity for composite crossing labels
// ---------------------------------------------------------------------------

/// HAPI dataset ID for MMS1 FPI Fast Survey DIS moments (ion density + velocity).
pub const MMS_FPI_DIS_HAPI_DATASET: &str = "MMS1_FPI_FAST_L2_DIS-MOMS";

/// Minute-resolution MMS FPI ion moments (density + velocity).
#[derive(Debug, Clone)]
pub struct MmsFpiMinuteRecord {
    pub year: u16,
    pub doy: u16,
    pub hour: u8,
    pub minute: u8,
    pub elapsed_hours: f64,
    pub ion_density: f64, // cm^-3
    pub vx_gse: f64,      // km/s
    pub vy_gse: f64,
    pub vz_gse: f64,
}

/// Parse MMS FPI DIS-MOMS HAPI CSV into minute-averaged ion moments.
pub fn parse_mms_fpi_hapi_csv_minutes(content: &str) -> Vec<MmsFpiMinuteRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());

    let headers = match reader.headers() {
        Ok(h) => h.clone(),
        Err(_) => return Vec::new(),
    };

    // Find density and velocity columns
    let den_col = headers
        .iter()
        .position(|h| h == "mms1_dis_numberdensity_fast");
    let vx_col = headers
        .iter()
        .position(|h| h == "mms1_dis_bulkv_gse_fast_0");
    let vy_col = headers
        .iter()
        .position(|h| h == "mms1_dis_bulkv_gse_fast_1");
    let vz_col = headers
        .iter()
        .position(|h| h == "mms1_dis_bulkv_gse_fast_2");

    let Some(den_col) = den_col else {
        return Vec::new();
    };

    #[derive(Default)]
    struct Acc {
        den: f64,
        vx: f64,
        vy: f64,
        vz: f64,
        count: usize,
    }

    let mut buckets: BTreeMap<(u16, u16, u8, u8), Acc> = BTreeMap::new();

    for record in reader.records().flatten() {
        let Some(time_str) = record.get(0) else {
            continue;
        };
        let Ok(dt) = DateTime::parse_from_rfc3339(time_str) else {
            continue;
        };
        let utc = dt.with_timezone(&Utc);

        let den = parse_hapi_spacephysics_f64_or_nan(record.get(den_col).unwrap_or(""));
        if !den.is_finite() || den <= 0.0 {
            continue;
        }

        let vx = vx_col
            .and_then(|c| record.get(c))
            .map(parse_hapi_spacephysics_f64_or_nan)
            .unwrap_or(0.0);
        let vy = vy_col
            .and_then(|c| record.get(c))
            .map(parse_hapi_spacephysics_f64_or_nan)
            .unwrap_or(0.0);
        let vz = vz_col
            .and_then(|c| record.get(c))
            .map(parse_hapi_spacephysics_f64_or_nan)
            .unwrap_or(0.0);

        let key = (
            utc.year() as u16,
            utc.ordinal() as u16,
            utc.hour() as u8,
            utc.minute() as u8,
        );
        let acc = buckets.entry(key).or_default();
        acc.den += den;
        acc.vx += vx;
        acc.vy += vy;
        acc.vz += vz;
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

            Some(MmsFpiMinuteRecord {
                year,
                doy,
                hour,
                minute,
                elapsed_hours: elapsed,
                ion_density: acc.den / n,
                vx_gse: acc.vx / n,
                vy_gse: acc.vy / n,
                vz_gse: acc.vz / n,
            })
        })
        .collect()
}

/// Composite magnetopause crossing detector using FGM + FPI data.
///
/// A crossing is identified when BOTH conditions hold across adjacent windows:
/// 1. Ion density ratio > `density_ratio_threshold` (magnetosheath is denser)
/// 2. Magnetic field rotation > `rotation_threshold_deg`
///
/// This is strictly stronger than |B|-gradient alone and matches standard
/// MMS magnetopause identification practice (Trattner et al., Paschmann et al.).
pub fn detect_composite_crossings(
    mag: &[MmsFgmMinuteRecord],
    fpi: &[MmsFpiMinuteRecord],
    window_minutes: usize,
    density_ratio_threshold: f64,
    rotation_threshold_deg: f64,
) -> Vec<usize> {
    if mag.len() < window_minutes * 2 + 1 {
        return vec![];
    }

    // Build density lookup by (year, doy, hour, minute)
    let mut density_map: BTreeMap<(u16, u16, u8, u8), f64> = BTreeMap::new();
    for r in fpi {
        density_map.insert((r.year, r.doy, r.hour, r.minute), r.ion_density);
    }

    let half = window_minutes;
    let mut crossings = Vec::new();

    for i in half..mag.len().saturating_sub(half) {
        // |B| rotation angle between pre and post windows
        let pre_bx: f64 = mag[i.saturating_sub(half)..i]
            .iter()
            .map(|r| r.bx_gse)
            .sum::<f64>()
            / half as f64;
        let pre_by: f64 = mag[i.saturating_sub(half)..i]
            .iter()
            .map(|r| r.by_gse)
            .sum::<f64>()
            / half as f64;
        let pre_bz: f64 = mag[i.saturating_sub(half)..i]
            .iter()
            .map(|r| r.bz_gse)
            .sum::<f64>()
            / half as f64;

        let post_end = (i + half).min(mag.len());
        let post_n = (post_end - i) as f64;
        let post_bx: f64 = mag[i..post_end].iter().map(|r| r.bx_gse).sum::<f64>() / post_n;
        let post_by: f64 = mag[i..post_end].iter().map(|r| r.by_gse).sum::<f64>() / post_n;
        let post_bz: f64 = mag[i..post_end].iter().map(|r| r.bz_gse).sum::<f64>() / post_n;

        let pre_mag = (pre_bx * pre_bx + pre_by * pre_by + pre_bz * pre_bz).sqrt();
        let post_mag = (post_bx * post_bx + post_by * post_by + post_bz * post_bz).sqrt();

        let cos_angle = if pre_mag > 1e-6 && post_mag > 1e-6 {
            ((pre_bx * post_bx + pre_by * post_by + pre_bz * post_bz) / (pre_mag * post_mag))
                .clamp(-1.0, 1.0)
        } else {
            1.0
        };
        let rotation_deg = cos_angle.acos().to_degrees();

        // Density ratio: compare pre/post window mean densities
        let pre_densities: Vec<f64> = mag[i.saturating_sub(half)..i]
            .iter()
            .filter_map(|r| density_map.get(&(r.year, r.doy, r.hour, r.minute)).copied())
            .collect();
        let post_densities: Vec<f64> = mag[i..post_end]
            .iter()
            .filter_map(|r| density_map.get(&(r.year, r.doy, r.hour, r.minute)).copied())
            .collect();

        if pre_densities.is_empty() || post_densities.is_empty() {
            continue;
        }

        let pre_den = pre_densities.iter().sum::<f64>() / pre_densities.len() as f64;
        let post_den = post_densities.iter().sum::<f64>() / post_densities.len() as f64;
        let den_ratio = if pre_den > 0.01 && post_den > 0.01 {
            (pre_den / post_den).max(post_den / pre_den)
        } else {
            1.0
        };

        // Composite criterion: BOTH density jump AND field rotation
        if den_ratio >= density_ratio_threshold && rotation_deg >= rotation_threshold_deg {
            let dominated = crossings
                .last()
                .is_some_and(|&prev: &usize| i.saturating_sub(prev) < half);
            if !dominated {
                crossings.push(i);
            }
        }
    }

    crossings
}
