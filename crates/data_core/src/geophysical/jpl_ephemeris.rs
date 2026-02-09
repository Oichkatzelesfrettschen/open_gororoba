//! JPL Horizons ephemeris provider.
//!
//! Fetches planetary positions via the JPL Horizons REST API in CSV format.
//! Much simpler than downloading and parsing SPK binary files (31+ MB).
//!
//! Source: JPL Horizons, https://ssd.jpl.nasa.gov/horizons/
//! Reference: Giorgini et al. (1996), Bull. AAS 28, 1158

use crate::fetcher::{download_to_string, DatasetProvider, FetchConfig, FetchError};
use std::fs;
use std::path::{Path, PathBuf};

/// A single ephemeris point from JPL Horizons.
#[derive(Debug, Clone)]
pub struct EphemerisPoint {
    /// Julian date.
    pub jd: f64,
    /// Calendar date string.
    pub date: String,
    /// Right ascension (degrees).
    pub ra: f64,
    /// Declination (degrees).
    pub dec: f64,
    /// Distance from observer (AU).
    pub delta: f64,
    /// Target body name/ID.
    pub body: String,
}

/// Build a Horizons REST API URL for a single body.
///
/// Queries observer table (OBS_TABLE) for geocentric apparent RA/Dec.
pub fn horizons_query_url(body_id: &str, start: &str, stop: &str, step: &str) -> String {
    format!(
        "https://ssd.jpl.nasa.gov/api/horizons.api?\
         format=text&\
         COMMAND='{}'&\
         OBJ_DATA='NO'&\
         MAKE_EPHEM='YES'&\
         EPHEM_TYPE='OBSERVER'&\
         CENTER='500@399'&\
         START_TIME='{}'&\
         STOP_TIME='{}'&\
         STEP_SIZE='{}'&\
         QUANTITIES='1,20'&\
         CSV_FORMAT='YES'",
        body_id, start, stop, step
    )
}

/// Parse Horizons CSV output for ephemeris points.
///
/// Horizons text output contains header/footer markers ($$SOE / $$EOE)
/// around the actual data.
pub fn parse_horizons_csv(content: &str, body_name: &str) -> Vec<EphemerisPoint> {
    let mut points = Vec::new();
    let mut in_data = false;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("$$SOE") {
            in_data = true;
            continue;
        }
        if trimmed.starts_with("$$EOE") {
            break;
        }
        if !in_data {
            continue;
        }

        let fields: Vec<&str> = trimmed.split(',').collect();
        if fields.len() < 5 {
            continue;
        }

        let jd = fields[0].trim().parse::<f64>().unwrap_or(f64::NAN);
        let date = fields[1].trim().to_string();
        let ra = fields[2].trim().parse::<f64>().unwrap_or(f64::NAN);
        let dec = fields[3].trim().parse::<f64>().unwrap_or(f64::NAN);
        let delta = fields[4].trim().parse::<f64>().unwrap_or(f64::NAN);

        points.push(EphemerisPoint {
            jd,
            date,
            ra,
            dec,
            delta,
            body: body_name.to_string(),
        });
    }

    points
}

/// Fetch ephemeris for a single body and return parsed points.
pub fn fetch_horizons(
    body_id: &str,
    body_name: &str,
    start: &str,
    stop: &str,
    step: &str,
) -> Result<Vec<EphemerisPoint>, FetchError> {
    let url = horizons_query_url(body_id, start, stop, step);
    let body = download_to_string(&url)?;
    Ok(parse_horizons_csv(&body, body_name))
}

/// Planet IDs for the 8 major planets in JPL Horizons.
const PLANET_IDS: &[(&str, &str)] = &[
    ("199", "Mercury"),
    ("299", "Venus"),
    ("399", "Earth"),
    ("499", "Mars"),
    ("599", "Jupiter"),
    ("699", "Saturn"),
    ("799", "Uranus"),
    ("899", "Neptune"),
];

/// JPL Horizons planetary ephemeris provider (8 planets, 2020-2030).
pub struct JplEphemerisProvider;

impl DatasetProvider for JplEphemerisProvider {
    fn name(&self) -> &str {
        "JPL Horizons Planetary Ephemeris"
    }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("jpl_planets_2020_2030.csv");
        if config.skip_existing && output.exists() {
            eprintln!("  {} already cached at {}", self.name(), output.display());
            return Ok(output);
        }

        eprintln!(
            "  Fetching {} from JPL Horizons (8 planets)...",
            self.name()
        );
        let mut all_lines = vec!["body,jd,date,ra,dec,delta".to_string()];

        for (body_id, body_name) in PLANET_IDS {
            // Skip Earth when observing from Earth
            if *body_id == "399" {
                continue;
            }

            eprintln!("    Querying {}...", body_name);
            let points = fetch_horizons(body_id, body_name, "2020-01-01", "2030-01-01", "30d")?;

            for pt in &points {
                all_lines.push(format!(
                    "{},{},{},{},{},{}",
                    pt.body, pt.jd, pt.date, pt.ra, pt.dec, pt.delta
                ));
            }
        }

        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }
        let content = all_lines.join("\n");
        fs::write(&output, &content)?;
        eprintln!("  Saved {} bytes to {}", content.len(), output.display());
        Ok(output)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("jpl_planets_2020_2030.csv").exists()
    }
}

/// Parse the combined planetary ephemeris CSV.
pub fn parse_jpl_ephemeris_csv(path: &Path) -> Result<Vec<EphemerisPoint>, FetchError> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .has_headers(true)
        .from_path(path)
        .map_err(|e| FetchError::Validation(format!("CSV read error: {}", e)))?;

    let headers = reader
        .headers()
        .map_err(|e| FetchError::Validation(format!("Header read error: {}", e)))?
        .clone();

    let col =
        |name: &str| -> Option<usize> { headers.iter().position(|h| h.eq_ignore_ascii_case(name)) };

    let idx_body = col("body");
    let idx_jd = col("jd");
    let idx_date = col("date");
    let idx_ra = col("ra");
    let idx_dec = col("dec");
    let idx_delta = col("delta");

    let get_str = |record: &csv::StringRecord, idx: Option<usize>| -> String {
        idx.and_then(|i| record.get(i))
            .unwrap_or("")
            .trim()
            .to_string()
    };

    let get_f64 = |record: &csv::StringRecord, idx: Option<usize>| -> f64 {
        idx.and_then(|i| record.get(i))
            .and_then(|s| s.trim().parse::<f64>().ok())
            .unwrap_or(f64::NAN)
    };

    let mut points = Vec::new();
    for result in reader.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        points.push(EphemerisPoint {
            body: get_str(&record, idx_body),
            jd: get_f64(&record, idx_jd),
            date: get_str(&record, idx_date),
            ra: get_f64(&record, idx_ra),
            dec: get_f64(&record, idx_dec),
            delta: get_f64(&record, idx_delta),
        });
    }

    Ok(points)
}
