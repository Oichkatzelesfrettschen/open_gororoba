//! INTERMAGNET observatory geomagnetic data provider.
//!
//! Parses IAGA-2002 format 1-minute definitive vector magnetograms from
//! global geomagnetic observatories. Used for geomagnetic jerk detection
//! and South Atlantic Anomaly cavity analysis.
//!
//! Data source: BGS GIN V1 API
//! Format: IAGA-2002 (space-delimited text with header)

use std::path::Path;

/// 1-minute INTERMAGNET geomagnetic record.
#[derive(Debug, Clone)]
pub struct GeomagRecord {
    pub year: u16,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    /// Elapsed hours from epoch start.
    pub elapsed_hours: f64,
    /// North component (nT).
    pub x: f64,
    /// East component (nT).
    pub y: f64,
    /// Vertical (downward) component (nT).
    pub z: f64,
    /// Total field (nT).
    pub f: f64,
}

/// Parse an IAGA-2002 format file into GeomagRecords.
///
/// The IAGA-2002 format has a header section (lines starting with space)
/// followed by data lines: DATE TIME DOY X Y Z F
pub fn parse_iaga2002(content: &str) -> Vec<GeomagRecord> {
    let mut records = Vec::new();
    let mut in_data = false;

    for line in content.lines() {
        // Header ends at the line containing "DATE"
        if line.contains("DATE") && line.contains("TIME") {
            in_data = true;
            continue;
        }
        if !in_data { continue; }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 7 { continue; }

        // Format: YYYY-MM-DD HH:MM:SS.SSS DOY X Y Z F
        let date_parts: Vec<&str> = parts[0].split('-').collect();
        if date_parts.len() != 3 { continue; }

        let year: u16 = date_parts[0].parse().unwrap_or(0);
        let month: u8 = date_parts[1].parse().unwrap_or(0);
        let day: u8 = date_parts[2].parse().unwrap_or(0);

        let time_parts: Vec<&str> = parts[1].split(':').collect();
        if time_parts.len() < 2 { continue; }
        let hour: u8 = time_parts[0].parse().unwrap_or(0);
        let minute: u8 = time_parts[1].parse().unwrap_or(0);

        // DOY is parts[2], skip it
        let x: f64 = parts[3].parse().unwrap_or(f64::NAN);
        let y: f64 = parts[4].parse().unwrap_or(f64::NAN);
        let z: f64 = parts[5].parse().unwrap_or(f64::NAN);
        let f: f64 = parts[6].parse().unwrap_or(f64::NAN);

        // IAGA-2002 uses 99999.00 as fill value
        if x.abs() > 90000.0 || y.abs() > 90000.0 || z.abs() > 90000.0 {
            continue;
        }

        records.push(GeomagRecord {
            year, month, day, hour, minute,
            elapsed_hours: 0.0, // computed after sorting
            x, y, z, f,
        });
    }

    // Compute elapsed hours from first record
    if let Some(first) = records.first() {
        let fy = first.year;
        let fm = first.month;
        let fd = first.day;
        let fh = first.hour;
        let fmin = first.minute;

        for rec in &mut records {
            let day_diff = (rec.year as f64 - fy as f64) * 365.25
                + (rec.month as f64 - fm as f64) * 30.44
                + (rec.day as f64 - fd as f64);
            rec.elapsed_hours = day_diff * 24.0
                + (rec.hour as f64 - fh as f64)
                + (rec.minute as f64 - fmin as f64) / 60.0;
        }
    }

    records
}

/// Load all IAGA-2002 files for an observatory from a directory.
pub fn load_observatory_year(dir: &Path, observatory: &str, year: u16) -> Vec<GeomagRecord> {
    let mut all = Vec::new();

    for month in 1..=12 {
        let fname = format!("{}_{:04}_{:02}.iaga", observatory, year, month);
        let path = dir.join(&fname);
        if let Ok(content) = std::fs::read_to_string(&path) {
            let mut records = parse_iaga2002(&content);
            all.append(&mut records);
        }
    }

    // Recompute elapsed hours across all months
    if let Some(first) = all.first() {
        let fy = first.year;
        let fm = first.month;
        let fd = first.day;
        let fh = first.hour;
        let fmin = first.minute;

        for rec in &mut all {
            let day_diff = (rec.year as f64 - fy as f64) * 365.25
                + (rec.month as f64 - fm as f64) * 30.44
                + (rec.day as f64 - fd as f64);
            rec.elapsed_hours = day_diff * 24.0
                + (rec.hour as f64 - fh as f64)
                + (rec.minute as f64 - fmin as f64) / 60.0;
        }
    }

    all
}
