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

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};

use data_core::{HeliosphereEventWindow, HeliosphereFeatureRow, heliosphere_row_datetime};

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
