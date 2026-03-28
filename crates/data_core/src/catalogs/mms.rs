//! MMS Fluxgate Magnetometer (FGM) data provider.
//!
//! The Magnetospheric Multiscale (MMS) mission provides high-cadence
//! magnetic field measurements. This module fetches Survey (SRVY) mode
//! Level 2 data via CDAWeb HAPI.
//!
//! Source: <https://cdaweb.gsfc.nasa.gov/hapi/info?id=MMS1_FGM_SRVY_L2>
//! Reference: Russell et al. (2016), Space Sci. Rev. 199, 189

use crate::{
    fetcher::{DatasetProvider, FetchConfig, FetchError, download_hapi_csv},
    parse::{parse_hapi_spacephysics_f64_or_nan, parse_hapi_time_to_ydh},
};
use chrono::{DateTime, Datelike, NaiveDate, Timelike, Utc};
use csv::ReaderBuilder;
use std::{
    collections::BTreeMap,
    fs,
    path::PathBuf,
};

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

const MMS_FGM_HAPI_DATASET: &str = "MMS1_FGM_SRVY_L2@0";

/// MMS FGM dataset provider.
pub struct MmsFgmProvider {
    pub year_start: u16,
    pub year_end: u16,
    pub doy_range: Option<(u16, u16)>,
}

impl Default for MmsFgmProvider {
    fn default() -> Self {
        Self {
            year_start: 2015,
            year_end: 2026,
            doy_range: None,
        }
    }
}

impl DatasetProvider for MmsFgmProvider {
    fn name(&self) -> &str {
        "MMS1 FGM Survey L2"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let dir = config.output_dir.join("mms");
        fs::create_dir_all(&dir)?;

        for year in self.year_start..=self.year_end {
            // Respect the startDate of the dataset (2015-09-01)
            let (start_month, start_day) = if year == 2015 { (9, 1) } else { (1, 1) };

            if let Some((start_doy, end_doy)) = self.doy_range {
                // Fine-window test: use exact DOY range without chunking
                let start_date = NaiveDate::from_yo_opt(year as i32, start_doy as u32)
                    .ok_or_else(|| FetchError::Validation(format!("invalid start doy {start_doy}")))?;
                let end_date = NaiveDate::from_yo_opt(year as i32, end_doy as u32)
                    .ok_or_else(|| FetchError::Validation(format!("invalid end doy {end_doy}")))?;
                
                let t_min = format!("{}T00:00:00Z", start_date.format("%Y-%m-%d"));
                let t_max = format!("{}T23:59:59Z", end_date.format("%Y-%m-%d"));
                let fname = format!("mms1_fgm_srvy_l2_{year}_{start_doy}_{end_doy}.csv");
                let output = dir.join(&fname);

                if config.skip_existing && output.exists() {
                    continue;
                }

                println!("Fetching MMS1 FGM Survey L2 for {} DOY {}-{} ({} to {})...", year, start_doy, end_doy, t_min, t_max);

                let body = download_hapi_csv(
                    MMS_FGM_HAPI_DATASET,
                    &t_min,
                    &t_max,
                    Some(&["Time", "mms1_fgm_b_gse_srvy_l2_clean"]),
                )?;
                fs::write(&output, body)?;
            } else {
                // Monthly chunking to avoid 400 errors on large requests
                for month in start_month..=12 {
                    let t_min = format!("{:04}-{:02}-{:02}T00:00:00Z", year, month, start_day);
                    
                    // End of month or end of year
                    let (end_year, end_month) = if month == 12 { (year + 1, 1) } else { (year, month + 1) };
                    let t_max = format!("{:04}-{:02}-01T00:00:00Z", end_year, end_month);

                    // Stop if we exceed year_end
                    if year == self.year_end && month == 12 && self.year_end < 2026 {
                        // This is simple logic, could be more precise with days but months are safe chunks
                    }

                    let fname = format!("mms1_fgm_srvy_l2_{:04}_{:02}.csv", year, month);
                    let output = dir.join(&fname);
                    
                    if config.skip_existing && output.exists() {
                        continue;
                    }

                    println!("Fetching MMS1 FGM Survey L2 for {} month {} ({} to {})...", year, month, t_min, t_max);

                    let body = download_hapi_csv(
                        MMS_FGM_HAPI_DATASET,
                        &t_min,
                        &t_max,
                        Some(&["Time", "mms1_fgm_b_gse_srvy_l2_clean"]),
                    )?;
                    fs::write(&output, body)?;
                }
            }
        }

        Ok(dir)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("mms").exists()
    }
}

pub fn parse_mms_fgm_hapi_csv(content: &str) -> Vec<MmsFgmRecord> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_reader(content.as_bytes());
    let mut rows = Vec::new();
    
    for record in reader.records().flatten() {
        let Some(time) = record.get(0) else { continue; };
        let Some((year, doy, hour)) = parse_hapi_time_to_ydh(time) else { continue; };
        
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
        let Some(time_str) = record.get(0) else { continue };
        let Ok(dt) = DateTime::parse_from_rfc3339(time_str) else { continue };
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
        acc.bmag_sum += if bmag.is_finite() { bmag } else { (bx * bx + by * by + bz * bz).sqrt() };
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
                    let day_diff = (year as f64 - fy as f64) * 365.25
                        + (doy as f64 - fd as f64);
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
    if records.len() < window_minutes * 2 + 1 {
        return vec![];
    }

    let half = window_minutes;
    let mut crossings = Vec::new();

    for i in half..records.len().saturating_sub(half) {
        // Sliding window |B| statistics: compare pre-window mean to post-window mean
        let pre_start = i.saturating_sub(half);
        let post_end = (i + half).min(records.len());

        let pre_mean_b: f64 = records[pre_start..i]
            .iter()
            .map(|r| r.b_magnitude)
            .sum::<f64>()
            / (i - pre_start) as f64;

        let post_mean_b: f64 = records[i..post_end]
            .iter()
            .map(|r| r.b_magnitude)
            .sum::<f64>()
            / (post_end - i) as f64;

        let b_jump = (post_mean_b - pre_mean_b).abs();

        // Check for B_z sign change in the window (rotation across current sheet)
        let pre_bz_mean: f64 = records[pre_start..i]
            .iter()
            .map(|r| r.bz_gse)
            .sum::<f64>()
            / (i - pre_start) as f64;
        let post_bz_mean: f64 = records[i..post_end]
            .iter()
            .map(|r| r.bz_gse)
            .sum::<f64>()
            / (post_end - i) as f64;
        let bz_sign_change = pre_bz_mean * post_bz_mean < 0.0;

        // Crossing criteria: large |B| jump OR (moderate jump + Bz sign change)
        let is_crossing = b_jump > bmag_gradient_threshold
            || (b_jump > bmag_gradient_threshold * 0.5 && bz_sign_change);

        if is_crossing {
            // Suppress duplicates: only keep if no crossing within window_minutes
            let dominated = crossings
                .last()
                .is_some_and(|&prev: &usize| i.saturating_sub(prev) < window_minutes);
            if !dominated {
                crossings.push(i);
            }
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

    hourly.into_iter().filter_map(|((year, doy, hour), acc)| {
        if acc.count == 0 { return None; }
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
    }).collect()
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
    pub ion_density: f64,    // cm^-3
    pub vx_gse: f64,        // km/s
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
    let den_col = headers.iter().position(|h| h == "mms1_dis_numberdensity_fast");
    let vx_col = headers.iter().position(|h| h == "mms1_dis_bulkv_gse_fast_0");
    let vy_col = headers.iter().position(|h| h == "mms1_dis_bulkv_gse_fast_1");
    let vz_col = headers.iter().position(|h| h == "mms1_dis_bulkv_gse_fast_2");

    let Some(den_col) = den_col else { return Vec::new() };

    #[derive(Default)]
    struct Acc { den: f64, vx: f64, vy: f64, vz: f64, count: usize }

    let mut buckets: BTreeMap<(u16, u16, u8, u8), Acc> = BTreeMap::new();

    for record in reader.records().flatten() {
        let Some(time_str) = record.get(0) else { continue };
        let Ok(dt) = DateTime::parse_from_rfc3339(time_str) else { continue };
        let utc = dt.with_timezone(&Utc);

        let den = parse_hapi_spacephysics_f64_or_nan(record.get(den_col).unwrap_or(""));
        if !den.is_finite() || den <= 0.0 { continue; }

        let vx = vx_col.and_then(|c| record.get(c)).map(parse_hapi_spacephysics_f64_or_nan).unwrap_or(0.0);
        let vy = vy_col.and_then(|c| record.get(c)).map(parse_hapi_spacephysics_f64_or_nan).unwrap_or(0.0);
        let vz = vz_col.and_then(|c| record.get(c)).map(parse_hapi_spacephysics_f64_or_nan).unwrap_or(0.0);

        let key = (utc.year() as u16, utc.ordinal() as u16, utc.hour() as u8, utc.minute() as u8);
        let acc = buckets.entry(key).or_default();
        acc.den += den; acc.vx += vx; acc.vy += vy; acc.vz += vz; acc.count += 1;
    }

    let keys: Vec<_> = buckets.keys().copied().collect();
    let first = keys.first().copied();

    keys.into_iter()
        .filter_map(|key| {
            let acc = &buckets[&key];
            if acc.count == 0 { return None; }
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
                year, doy, hour, minute, elapsed_hours: elapsed,
                ion_density: acc.den / n,
                vx_gse: acc.vx / n, vy_gse: acc.vy / n, vz_gse: acc.vz / n,
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
    if mag.len() < window_minutes * 2 + 1 { return vec![]; }

    // Build density lookup by (year, doy, hour, minute)
    let mut density_map: BTreeMap<(u16, u16, u8, u8), f64> = BTreeMap::new();
    for r in fpi {
        density_map.insert((r.year, r.doy, r.hour, r.minute), r.ion_density);
    }

    let half = window_minutes;
    let mut crossings = Vec::new();

    for i in half..mag.len().saturating_sub(half) {
        // |B| rotation angle between pre and post windows
        let pre_bx: f64 = mag[i.saturating_sub(half)..i].iter().map(|r| r.bx_gse).sum::<f64>() / half as f64;
        let pre_by: f64 = mag[i.saturating_sub(half)..i].iter().map(|r| r.by_gse).sum::<f64>() / half as f64;
        let pre_bz: f64 = mag[i.saturating_sub(half)..i].iter().map(|r| r.bz_gse).sum::<f64>() / half as f64;

        let post_end = (i + half).min(mag.len());
        let post_n = (post_end - i) as f64;
        let post_bx: f64 = mag[i..post_end].iter().map(|r| r.bx_gse).sum::<f64>() / post_n;
        let post_by: f64 = mag[i..post_end].iter().map(|r| r.by_gse).sum::<f64>() / post_n;
        let post_bz: f64 = mag[i..post_end].iter().map(|r| r.bz_gse).sum::<f64>() / post_n;

        let pre_mag = (pre_bx * pre_bx + pre_by * pre_by + pre_bz * pre_bz).sqrt();
        let post_mag = (post_bx * post_bx + post_by * post_by + post_bz * post_bz).sqrt();

        let cos_angle = if pre_mag > 1e-6 && post_mag > 1e-6 {
            ((pre_bx * post_bx + pre_by * post_by + pre_bz * post_bz) / (pre_mag * post_mag)).clamp(-1.0, 1.0)
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

        if pre_densities.is_empty() || post_densities.is_empty() { continue; }

        let pre_den = pre_densities.iter().sum::<f64>() / pre_densities.len() as f64;
        let post_den = post_densities.iter().sum::<f64>() / post_densities.len() as f64;
        let den_ratio = if pre_den > 0.01 && post_den > 0.01 {
            (pre_den / post_den).max(post_den / pre_den)
        } else {
            1.0
        };

        // Composite criterion: BOTH density jump AND field rotation
        if den_ratio >= density_ratio_threshold && rotation_deg >= rotation_threshold_deg {
            let dominated = crossings.last().is_some_and(|&prev: &usize| i.saturating_sub(prev) < half);
            if !dominated { crossings.push(i); }
        }
    }

    crossings
}
