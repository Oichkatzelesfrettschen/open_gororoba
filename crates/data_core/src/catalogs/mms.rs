//! MMS Fluxgate Magnetometer (FGM) data provider.
//!
//! The Magnetospheric Multiscale (MMS) mission provides high-cadence
//! magnetic field measurements. This module fetches Survey (SRVY) mode
//! Level 2 data via CDAWeb HAPI.
//!
//! Source: <https://cdaweb.gsfc.nasa.gov/hapi/info?id=MMS1_FGM_SRVY_L2>
//! Reference: Russell et al. (2016), Space Sci. Rev. 199, 189

use crate::parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh};
use chrono::{DateTime, Datelike, NaiveDateTime, Timelike, Utc};
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

#[cfg(feature = "fetch")]
pub use super::mms_fetch::MmsFgmProvider;

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
/// Groups raw high-cadence samples (~16 Hz SRVY) into 1-minute bins by
/// client-side arithmetic mean over all finite samples within each UTC
/// minute boundary.  `|B|` is averaged directly when provided; otherwise
/// it is recomputed from the mean component vector.  This is the decimation
/// protocol for all Takens delay embeddings built from MMS data.
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

        let pre_bz_mean: f64 = records[pre_start..i].iter().map(|r| r.bz_gse).sum::<f64>() / n_pre;
        let post_bz_mean: f64 = records[i..post_end].iter().map(|r| r.bz_gse).sum::<f64>() / n_post;
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

// ============================================================================
// MMS SITL / GLS event catalog
// ============================================================================

/// One SITL-selected or GLS-automated event interval from the MMS SDC catalog.
///
/// The MMS Science Data Center publishes scientist-in-the-loop (SITL) and
/// ground-loop-segment (GLS) selections as CSV files.  Each row captures an
/// interval a scientist (or automated algorithm) flagged for burst-mode
/// downlink, typically because it contains a scientifically interesting
/// boundary.  `fom` (figure of merit) encodes scientist confidence; 100 is
/// maximum priority.  `discussion` is a free-text note that often names the
/// boundary type ("magnetopause", "reconnection", "current sheet", etc.).
///
/// WHY: Using SITL intervals as ground truth replaces |B|-gradient
/// pseudo-labels with expert-curated boundary annotations, stripping
/// compressive false positives (pressure pulses, dipolarizations) that change
/// |B| without rotating the field.
#[derive(Debug, Clone)]
pub struct MmsEventInterval {
    pub start: NaiveDateTime,
    pub end: NaiveDateTime,
    /// Figure of merit (0-100).  Higher means higher scientist priority.
    pub fom: f64,
    /// Free-text annotation from the scientist or GLS algorithm.
    pub discussion: String,
}

/// Parse the MMS SDC SITL/GLS CSV format into event intervals.
///
/// Expected header line (present or absent):
/// ```text
/// tstart,tstop,fom,sourceid,createtime,discussion
/// ```
/// Times are ISO 8601 with optional trailing `Z`.  Lines starting with `#`
/// or the literal header are skipped.  Rows with unparseable times are
/// silently skipped -- a missing or corrupt row does not abort the parse.
///
/// Also accepts a simplified two-column format `start_time,stop_time` (no
/// header required) used by some published MMS crossing catalogs.
pub fn parse_mms_sitl_csv(content: &str) -> Vec<MmsEventInterval> {
    let mut events = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        // Skip header rows.
        if trimmed.starts_with("tstart")
            || trimmed.starts_with("start_time")
            || trimmed.starts_with("StartTime")
        {
            continue;
        }

        let cols: Vec<&str> = trimmed.splitn(6, ',').collect();
        if cols.len() < 2 {
            continue;
        }

        let start = parse_sitl_time(cols[0].trim());
        let end = parse_sitl_time(cols[1].trim());
        let (start, end) = match (start, end) {
            (Some(s), Some(e)) => (s, e),
            _ => continue,
        };

        let fom = if cols.len() >= 3 {
            cols[2].trim().parse::<f64>().unwrap_or(0.0)
        } else {
            0.0
        };

        // Column 5 (index 4 = createtime) may be absent; discussion is last.
        let discussion = if cols.len() >= 6 {
            cols[5].trim().trim_matches('"').to_string()
        } else if cols.len() == 5 {
            cols[4].trim().trim_matches('"').to_string()
        } else {
            String::new()
        };

        events.push(MmsEventInterval {
            start,
            end,
            fom,
            discussion,
        });
    }

    events.sort_by_key(|e| e.start);
    events
}

/// Parse ISO 8601 datetime strings emitted by the MMS SDC API.
///
/// Tries several common variants:
/// - `2024-01-01T00:00:00.000Z`  (milliseconds + Z)
/// - `2024-01-01T00:00:00Z`      (no subseconds)
/// - `2024-01-01T00:00:00.000`   (milliseconds, no Z)
/// - `2024-01-01T00:00:00`       (bare)
fn parse_sitl_time(s: &str) -> Option<NaiveDateTime> {
    let s = s.trim_end_matches('Z');
    // Try with milliseconds first.
    if let Ok(dt) = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f") {
        return Some(dt);
    }
    // Fallback to whole-second.
    NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S").ok()
}

/// Filter a SITL event list to intervals likely corresponding to magnetopause
/// crossings.
///
/// Retains events whose `discussion` field contains at least one of:
/// "magnetopause", "mag pause", " mp ", " mp,", " mp.", "current sheet",
/// "reconnect", "dayside".  The match is case-insensitive.
///
/// WHY: SITL selections cover all burst-priority events (flux ropes, jets,
/// reconnection, etc.).  Restricting to magnetopause keywords gives a clean
/// ground-truth set comparable to the THEMIS/Cluster/MAVEN curated crossing
/// lists used for the cross-mission table.
///
/// If the filtered list is empty (e.g., the catalog covers no magnetopause
/// events in the window) the full unfiltered list is returned so the binary
/// can still produce output rather than silently failing.
pub fn filter_magnetopause_events(events: &[MmsEventInterval]) -> Vec<MmsEventInterval> {
    const KEYWORDS: &[&str] = &[
        "magnetopause",
        "mag pause",
        " mp ",
        " mp,",
        " mp.",
        "(mp)",
        "current sheet",
        "reconnect",
        "dayside",
        "boundary layer",
    ];

    let filtered: Vec<MmsEventInterval> = events
        .iter()
        .filter(|e| {
            let lower = e.discussion.to_lowercase();
            KEYWORDS.iter().any(|kw| lower.contains(kw))
        })
        .cloned()
        .collect();

    if filtered.is_empty() {
        events.to_vec()
    } else {
        filtered
    }
}

/// Return true if `elapsed_hours` from `reference_midnight` falls within any
/// SITL event interval.
///
/// An overlap is defined as `event.start <= t < event.end`.
pub fn timestamp_in_sitl_event(
    elapsed_hours: f64,
    reference_midnight: &NaiveDateTime,
    events: &[MmsEventInterval],
) -> bool {
    use chrono::Duration;
    let t = *reference_midnight + Duration::seconds((elapsed_hours * 3600.0) as i64);
    events.iter().any(|e| e.start <= t && t < e.end)
}

// ============================================================================
// FPI-based crossing event derivation (locally cached headerless CSV)
// ============================================================================

/// Parse a headerless MMS FPI DIS CSV file cached in DOY format.
///
/// The locally-cached files lack a HAPI header row and use Windows CRLF line
/// endings.  Column layout per line:
///   `<ISO-8601-timestamp>,<density_cm3>,<Vx_km_s>,<Vy_km_s>,<Vz_km_s>`
///
/// Returns `(NaiveDateTime, density_cm3)` pairs, filtering fill values
/// (density <= 0 or > 5 000 cm^-3).
pub fn parse_mms_fpi_csv_headerless(content: &str) -> Vec<(NaiveDateTime, f64)> {
    content
        .lines()
        .filter_map(|raw| {
            let line = raw.trim();
            if line.is_empty() {
                return None;
            }
            let mut cols = line.splitn(5, ',');
            let ts = cols.next()?;
            let den_str = cols.next()?;
            let dt = parse_sitl_time(ts)?;
            let den: f64 = den_str.trim().parse().ok()?;
            if !den.is_finite() || den <= 0.0 || den > 5_000.0 {
                return None;
            }
            Some((dt, den))
        })
        .collect()
}

/// Derive magnetopause crossing event intervals from an FPI ion density series.
///
/// # Algorithm
/// 1. Minute-average the raw density series.
/// 2. Classify each minute: MSP (n < `lo_threshold`), MSH (n > `hi_threshold`),
///    or ambiguous (in between).
/// 3. Debounced state machine: confirmed state changes only when
///    `min_regime_minutes` consecutive non-ambiguous samples agree on the
///    new state.  Ambiguous samples neither reset nor count toward the run.
/// 4. Each confirmed transition yields one `MmsEventInterval` spanning
///    `[last_minute_of_old_state - pad_minutes, first_confirmed_new_state + pad_minutes]`.
/// 5. Crossings whose window exceeds `max_crossing_minutes` are skipped.
///    Long windows arise when FPI data is sparse or MMS lingers in an ambiguous
///    flux-transfer / boundary layer region for many hours.  Such intervals
///    would match any detector by sheer duration and inflate recall estimates.
/// 6. Crossings within `min_gap_minutes` of each other are suppressed
///    (grazing / oscillating boundary -- counts as one event).
///
/// # WHY
/// FPI ion density is the most physically rigorous magnetopause marker:
/// * Magnetosheath: n_i > 10 cm^-3 (compressed solar wind)
/// * Magnetosphere: n_i < 3 cm^-3 (hot, tenuous plasma sheet)
/// * Compressive events (pressure pulses, dipolarizations) do NOT change the
///   density ratio -- they cannot trigger this detector, eliminating the FAR
///   paradox that inflates false-alarm counts in |B|-gradient pseudo-labels.
pub fn derive_fpi_crossing_events(
    density_series: &[(NaiveDateTime, f64)],
    lo_threshold: f64,
    hi_threshold: f64,
    min_regime_minutes: usize,
    pad_minutes: i64,
    min_gap_minutes: i64,
    max_crossing_minutes: i64,
) -> Vec<MmsEventInterval> {
    if density_series.is_empty() {
        return Vec::new();
    }

    // -- Minute average (truncate to minute boundary) --
    let mut minute_map: BTreeMap<NaiveDateTime, (f64, usize)> = BTreeMap::new();
    for (dt, den) in density_series {
        let trunc = dt
            .with_second(0)
            .and_then(|d| d.with_nanosecond(0))
            .unwrap_or(*dt);
        let e = minute_map.entry(trunc).or_insert((0.0, 0));
        e.0 += den;
        e.1 += 1;
    }

    let series: Vec<(NaiveDateTime, f64)> = minute_map
        .into_iter()
        .map(|(dt, (s, n))| (dt, s / n as f64))
        .collect();

    let n = series.len();
    if n < min_regime_minutes + 1 {
        return Vec::new();
    }

    // -- Classify: -1=MSP, +1=MSH, 0=ambiguous --
    let class: Vec<i8> = series
        .iter()
        .map(|(_, d)| {
            if *d < lo_threshold {
                -1_i8
            } else if *d > hi_threshold {
                1_i8
            } else {
                0_i8
            }
        })
        .collect();

    // -- Bootstrap: find initial confirmed state --
    let mut confirmed_state: i8 = 0;
    'bootstrap: for i in 0..n.saturating_sub(min_regime_minutes) {
        let window = &class[i..i + min_regime_minutes];
        if window.iter().all(|&c| c == 1) {
            confirmed_state = 1;
            break 'bootstrap;
        }
        if window.iter().all(|&c| c == -1) {
            confirmed_state = -1;
            break 'bootstrap;
        }
    }
    if confirmed_state == 0 {
        return Vec::new();
    }

    // -- Debounced transition detection --
    let pad = chrono::Duration::minutes(pad_minutes);
    let min_gap = chrono::Duration::minutes(min_gap_minutes);

    let mut crossings: Vec<MmsEventInterval> = Vec::new();
    let mut last_old_idx: usize = 0;
    let mut candidate_val: i8 = 0;
    let mut candidate_count: usize = 0;

    for i in 0..n {
        let c = class[i];
        if c == confirmed_state {
            last_old_idx = i;
            candidate_val = 0;
            candidate_count = 0;
            continue;
        }
        if c == 0 {
            // Ambiguous: neutral -- neither resets nor counts.
            continue;
        }
        // c is the opposite non-ambiguous state.
        if c != candidate_val {
            candidate_val = c;
            candidate_count = 1;
        } else {
            candidate_count += 1;
        }
        if candidate_count >= min_regime_minutes {
            let transition_start = series[last_old_idx].0;
            let transition_end = series[i].0;
            let duration = transition_end.signed_duration_since(transition_start);
            let max_dur = chrono::Duration::minutes(max_crossing_minutes);
            let too_long = duration > max_dur;
            let too_close = crossings.last().is_some_and(|prev: &MmsEventInterval| {
                transition_start.signed_duration_since(prev.start) < min_gap
            });
            if !too_close && !too_long {
                crossings.push(MmsEventInterval {
                    start: transition_start - pad,
                    end: transition_end + pad,
                    fom: 100.0,
                    discussion: format!(
                        "FPI: {}->{} near {}",
                        if confirmed_state == -1 { "MSP" } else { "MSH" },
                        if c == -1 { "MSP" } else { "MSH" },
                        transition_start.format("%Y-%m-%dT%H:%M")
                    ),
                });
            }
            confirmed_state = c;
            last_old_idx = i;
            candidate_val = 0;
            candidate_count = 0;
        }
    }

    crossings.sort_by_key(|e| e.start);
    crossings
}

#[cfg(test)]
mod sitl_tests {
    use super::*;

    #[test]
    fn test_parse_sitl_csv_full_format() {
        let csv = "tstart,tstop,fom,sourceid,createtime,discussion\n\
                   2024-01-01T06:00:00.000Z,2024-01-01T06:05:00.000Z,100,1,\
                   2024-01-01T12:00:00Z,\"Magnetopause crossing\"\n\
                   2024-01-01T08:00:00Z,2024-01-01T08:02:00Z,50,2,\
                   2024-01-01T12:00:00Z,\"flux rope in magnetosheath\"\n";
        let events = parse_mms_sitl_csv(csv);
        assert_eq!(events.len(), 2);
        assert!((events[0].fom - 100.0).abs() < 1e-9);
        assert!(events[0].discussion.contains("Magnetopause"));
    }

    #[test]
    fn test_parse_sitl_csv_simple_format() {
        let csv = "2024-01-02T10:00:00,2024-01-02T10:10:00\n\
                   2024-01-02T14:00:00Z,2024-01-02T14:05:00Z\n";
        let events = parse_mms_sitl_csv(csv);
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_filter_magnetopause_events_retains_mp_keywords() {
        let make = |disc: &str| MmsEventInterval {
            start: NaiveDateTime::parse_from_str("2024-01-01T00:00:00", "%Y-%m-%dT%H:%M:%S")
                .unwrap(),
            end: NaiveDateTime::parse_from_str("2024-01-01T00:05:00", "%Y-%m-%dT%H:%M:%S").unwrap(),
            fom: 50.0,
            discussion: disc.to_string(),
        };
        let events = vec![
            make("Magnetopause current sheet"),
            make("Flux rope"),
            make("reconnection site"),
            make("plasma jet"),
        ];
        let filtered = filter_magnetopause_events(&events);
        // "Magnetopause" and "reconnect" match; "Flux rope" and "plasma jet" do not.
        assert_eq!(filtered.len(), 2);
    }

    #[test]
    fn test_filter_falls_back_to_full_when_empty() {
        let make = |disc: &str| MmsEventInterval {
            start: NaiveDateTime::parse_from_str("2024-01-01T00:00:00", "%Y-%m-%dT%H:%M:%S")
                .unwrap(),
            end: NaiveDateTime::parse_from_str("2024-01-01T00:05:00", "%Y-%m-%dT%H:%M:%S").unwrap(),
            fom: 50.0,
            discussion: disc.to_string(),
        };
        let events = vec![make("plasma jet"), make("flux rope")];
        let filtered = filter_magnetopause_events(&events);
        assert_eq!(filtered.len(), 2); // fallback: return all
    }
}
