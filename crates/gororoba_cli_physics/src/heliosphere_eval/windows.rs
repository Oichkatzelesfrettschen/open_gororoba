//! Time- and window-related pure helpers used across heliosphere
//! evaluation paths.
//!
//! Includes:
//!   * `parse_timestamp`     -- parse RFC3339 into chrono UTC
//!   * `contains_time`       -- true if `timestamp` falls inside a
//!     HeliosphereEventWindow's `[start, end]`
//!   * `cube_date_bounds`    -- min/max date over a cube's rows
//!   * `positive_row_count`  -- count rows whose datetime hits any window
//!   * `normalize_text`      -- ASCII-lowercase/dehyphenate for join keys

use std::collections::BTreeMap;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};

use data_core::{HeliosphereEventWindow, HeliosphereFeatureRow, heliosphere_row_datetime};

use super::{public_types::RowKey, row_key, stats::finite_median_opt};

pub(super) fn parse_timestamp(value: &str) -> Result<DateTime<Utc>> {
    Ok(DateTime::parse_from_rfc3339(value)
        .with_context(|| format!("parse timestamp {value}"))?
        .with_timezone(&Utc))
}

pub(super) fn contains_time(window: &HeliosphereEventWindow, timestamp: DateTime<Utc>) -> bool {
    let start = DateTime::parse_from_rfc3339(&window.window_start_utc)
        .map(|value| value.with_timezone(&Utc));
    let end =
        DateTime::parse_from_rfc3339(&window.window_end_utc).map(|value| value.with_timezone(&Utc));
    match (start, end) {
        (Ok(start), Ok(end)) => timestamp >= start && timestamp <= end,
        _ => false,
    }
}

pub(super) fn cube_date_bounds(
    rows: &[HeliosphereFeatureRow],
) -> Result<(chrono::NaiveDate, chrono::NaiveDate)> {
    let start_date = rows
        .iter()
        .filter_map(heliosphere_row_datetime)
        .map(|value| value.date_naive())
        .min()
        .ok_or_else(|| anyhow::anyhow!("cube contains no timestamped rows"))?;
    let end_date = rows
        .iter()
        .filter_map(heliosphere_row_datetime)
        .map(|value| value.date_naive())
        .max()
        .ok_or_else(|| anyhow::anyhow!("cube contains no timestamped rows"))?;
    Ok((start_date, end_date))
}

pub(super) fn positive_row_count(
    rows: &[&HeliosphereFeatureRow],
    windows: &[HeliosphereEventWindow],
) -> usize {
    rows.iter()
        .filter_map(|row| heliosphere_row_datetime(row))
        .filter(|timestamp| {
            windows
                .iter()
                .any(|window| contains_time(window, *timestamp))
        })
        .count()
}

pub(super) fn normalize_text(value: &str) -> String {
    value
        .trim()
        .to_ascii_lowercase()
        .replace([' ', '_', '/'], "-")
        .replace("--", "-")
}

/// Map each row's `RowKey` to its RFC3339 timestamp, skipping rows
/// whose datetime cannot be parsed.
pub(super) fn raw_time_index(rows: &[HeliosphereFeatureRow]) -> BTreeMap<RowKey, String> {
    rows.iter()
        .filter_map(|row| {
            heliosphere_row_datetime(row).map(|timestamp| (row_key(row), timestamp.to_rfc3339()))
        })
        .collect()
}

/// For each mission, scan chronologically; for every positive-label
/// row, walk backwards while the active mask is contiguously on and
/// take the earliest such timestamp as the "prediction start." Return
/// the median lead time (hours) over all such (event, prediction)
/// pairs, or None if there were no usable pairs.
pub(super) fn median_mask_lead_time_hours(
    time_index: &BTreeMap<RowKey, String>,
    label_index: &BTreeMap<RowKey, bool>,
    active_index: &BTreeMap<RowKey, bool>,
) -> Option<f64> {
    let mut grouped: BTreeMap<String, Vec<(String, bool, bool)>> = BTreeMap::new();
    for (key, timestamp) in time_index {
        let mission = key.1.clone();
        grouped.entry(mission).or_default().push((
            timestamp.clone(),
            *label_index.get(key).unwrap_or(&false),
            *active_index.get(key).unwrap_or(&false),
        ));
    }
    let mut leads = Vec::new();
    for rows in grouped.values_mut() {
        rows.sort_by(|a, b| a.0.cmp(&b.0));
        for positive_idx in rows
            .iter()
            .enumerate()
            .filter_map(|(idx, (_, positive, _))| (*positive).then_some(idx))
        {
            let event_time = parse_timestamp(&rows[positive_idx].0).ok()?;
            let mut earliest_prediction = None;
            for (timestamp, _positive, active) in rows[..=positive_idx].iter().rev() {
                if *active {
                    earliest_prediction = parse_timestamp(timestamp).ok();
                } else if earliest_prediction.is_some() {
                    break;
                }
            }
            if let Some(start) = earliest_prediction {
                let hours = (event_time - start).num_minutes() as f64 / 60.0;
                if hours.is_finite() && hours >= 0.0 {
                    leads.push(hours);
                }
            }
        }
    }
    finite_median_opt(&leads)
}
