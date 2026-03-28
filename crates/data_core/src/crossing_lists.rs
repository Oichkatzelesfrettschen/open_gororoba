//! Curated boundary crossing list parser.
//!
//! Parses expert-curated crossing times from published databases into
//! `NaiveDateTime` arrays. Supports three formats:
//!
//! - **THEMIS V2**: Tab-separated `TIMESTAMP X Y Z ... PROBE` (Staples et al. 2020)
//! - **MESSENGER**: Space-separated `start_time end_time` per line (Zenodo)
//! - **OMNI**: Space-separated `SC YEAR DOY HR MIN SS ...` (SPDF)
//!
//! # Usage
//! ```no_run
//! use data_core::crossing_lists::parse_crossing_list;
//! use chrono::NaiveDate;
//!
//! let content = std::fs::read_to_string("crossings.txt").unwrap();
//! let start = NaiveDate::from_ymd_opt(2008, 10, 27).unwrap();
//! let end = NaiveDate::from_ymd_opt(2008, 11, 2).unwrap();
//! let crossings = parse_crossing_list(&content, None, &start, &end);
//! ```

use chrono::{NaiveDate, NaiveDateTime};

/// A parsed crossing event with timestamp and optional metadata.
#[derive(Debug, Clone)]
pub struct CrossingEvent {
    pub datetime: NaiveDateTime,
    /// Elapsed hours from `window_start`.
    pub elapsed_hours: f64,
}

/// Parse a crossing list file, filtering to the `[start, end]` date window.
///
/// Auto-detects format from line structure. If `probe_filter` is `Some`,
/// only THEMIS-format lines matching that probe ID are included.
///
/// Returns events sorted by time, with `elapsed_hours` relative to `start`
/// midnight.
pub fn parse_crossing_list(
    content: &str,
    probe_filter: Option<&str>,
    start: &NaiveDate,
    end: &NaiveDate,
) -> Vec<CrossingEvent> {
    let start_dt = start.and_hms_opt(0, 0, 0).unwrap();
    let end_dt = end.and_hms_opt(23, 59, 59).unwrap();

    let mut events: Vec<CrossingEvent> = Vec::new();

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("TIMESTAMP") {
            continue;
        }

        // Try THEMIS format: TIMESTAMP\tX\tY\tZ\t...\tPROBE
        let fields: Vec<&str> = trimmed.split('\t').collect();
        if fields.len() >= 2 {
            if let Some(filter) = probe_filter
                && fields
                    .last()
                    .is_some_and(|pf| pf.trim().to_lowercase() != filter.to_lowercase())
            {
                continue;
            }
            if let Ok(dt) = NaiveDateTime::parse_from_str(
                fields[0].trim().trim_end_matches('Z'),
                "%Y-%m-%dT%H:%M:%S",
            ) && dt >= start_dt
                && dt <= end_dt
            {
                let elapsed = (dt - start_dt).num_minutes() as f64 / 60.0;
                events.push(CrossingEvent {
                    datetime: dt,
                    elapsed_hours: elapsed,
                });
            }
            continue;
        }

        // Try MESSENGER format: start_time end_time (space-separated)
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if !parts.is_empty() {
            let ts = parts[0].trim_end_matches('Z');
            if let Ok(dt) = NaiveDateTime::parse_from_str(ts, "%Y-%m-%d/%H:%M:%S")
                && dt >= start_dt
                && dt <= end_dt
            {
                let elapsed = (dt - start_dt).num_minutes() as f64 / 60.0;
                events.push(CrossingEvent {
                    datetime: dt,
                    elapsed_hours: elapsed,
                });
                continue;
            }
        }

        // Try OMNI format: SC YEAR DOY HR MIN SS ... (space-delimited)
        if parts.len() >= 5
            && let (Ok(y), Ok(d), Ok(h), Ok(m)) = (
                parts[1].parse::<i32>(),
                parts[2].parse::<u32>(),
                parts[3].parse::<u32>(),
                parts[4].parse::<u32>(),
            )
            && let Some(date) = NaiveDate::from_yo_opt(y, d)
            && let Some(dt) = date.and_hms_opt(h, m, 0)
            && dt >= start_dt
            && dt <= end_dt
        {
            let elapsed = (dt - start_dt).num_minutes() as f64 / 60.0;
            events.push(CrossingEvent {
                datetime: dt,
                elapsed_hours: elapsed,
            });
        }
    }

    events.sort_by_key(|a| a.datetime);
    events
}

/// Match data timestamps against curated crossing times within tolerance.
///
/// Returns `(matched_count, offsets_minutes)` where each offset is the
/// absolute time difference in minutes between a curated crossing and its
/// nearest matched data transition.
pub fn match_crossings(
    curated_hours: &[f64],
    transition_hours: &[f64],
    tolerance_hours: f64,
) -> (usize, Vec<f64>) {
    let mut matched = 0usize;
    let mut offsets = Vec::new();

    for &ch in curated_hours {
        let best = transition_hours
            .iter()
            .filter(|&&th| (th - ch).abs() < tolerance_hours)
            .map(|&th| (th - ch).abs() * 60.0)
            .fold(f64::MAX, f64::min);

        if best < f64::MAX {
            matched += 1;
            offsets.push(best);
        }
    }

    offsets.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (matched, offsets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_themis_format() {
        let content = "# header\nTIMESTAMP\tX\tPROBE\n\
                        2008-10-29T12:00:00Z\t5.0\ta\n\
                        2008-10-29T14:00:00Z\t6.0\tb\n";
        let start = NaiveDate::from_ymd_opt(2008, 10, 29).unwrap();
        let end = NaiveDate::from_ymd_opt(2008, 10, 29).unwrap();

        let all = parse_crossing_list(content, None, &start, &end);
        assert_eq!(all.len(), 2);

        let probe_a = parse_crossing_list(content, Some("a"), &start, &end);
        assert_eq!(probe_a.len(), 1);
        assert!((probe_a[0].elapsed_hours - 12.0).abs() < 0.01);
    }

    #[test]
    fn test_parse_messenger_format() {
        let content = "2011-03-24/01:10:01 2011-03-24/01:13:35\n\
                        2011-03-24/13:21:51 2011-03-24/13:23:22\n";
        let start = NaiveDate::from_ymd_opt(2011, 3, 24).unwrap();
        let end = NaiveDate::from_ymd_opt(2011, 3, 24).unwrap();
        let events = parse_crossing_list(content, None, &start, &end);
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_parse_omni_format() {
        let content = "CL 2002  2  8 15  0  11.7  9.3  8.0  5.0\n\
                        CL 2002  2  9 30  0  12.0  9.4  7.9  5.0\n";
        let start = NaiveDate::from_ymd_opt(2002, 1, 1).unwrap();
        let end = NaiveDate::from_ymd_opt(2002, 1, 3).unwrap();
        let events = parse_crossing_list(content, None, &start, &end);
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_match_crossings() {
        let curated = vec![1.0, 5.0, 10.0];
        let transitions = vec![1.1, 4.8, 7.0];
        let (matched, offsets) = match_crossings(&curated, &transitions, 0.5);
        assert_eq!(matched, 2); // 1.0 matches 1.1, 5.0 matches 4.8, 10.0 has no match
        assert_eq!(offsets.len(), 2);
    }
}
