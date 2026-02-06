//! McGill Online Magnetar Catalog parser and fetcher.
//!
//! The McGill catalog is the canonical reference for known magnetars
//! (SGRs and AXPs). Small dataset (~30 objects) but critical for
//! compact object population studies.
//!
//! Source: http://www.physics.mcgill.ca/~pulsar/magnetar/main.html
//! Reference: Olausen & Kaspi (2014), ApJS 212, 6

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_with_fallbacks};
use std::path::{Path, PathBuf};

/// A magnetar from the McGill catalog.
#[derive(Debug, Clone)]
pub struct Magnetar {
    /// Source name (e.g., SGR 0418+5729).
    pub name: String,
    /// Right ascension (degrees).
    pub ra: f64,
    /// Declination (degrees).
    pub dec: f64,
    /// Galactic longitude (degrees).
    pub gl: f64,
    /// Galactic latitude (degrees).
    pub gb: f64,
    /// Spin period (seconds).
    pub period: f64,
    /// Period derivative (s/s).
    pub pdot: f64,
    /// Inferred dipole field (10^14 G).
    pub b_dipole: f64,
    /// Characteristic age (kyr).
    pub age: f64,
    /// Spin-down luminosity (erg/s).
    pub edot: f64,
    /// Distance (kpc).
    pub distance: f64,
    /// Dispersion measure (pc/cm^3).
    pub dm: f64,
    /// 2-10 keV luminosity (erg/s).
    pub lx: f64,
}

fn parse_f64(s: &str) -> f64 {
    let s = s.trim();
    if s.is_empty() || s == "--" || s == "..." || s == "nan" {
        return f64::NAN;
    }
    // Handle ranges like "3.1-5.5" by taking midpoint
    if s.contains('-') && !s.starts_with('-') {
        let parts: Vec<&str> = s.split('-').collect();
        if parts.len() == 2 {
            if let (Ok(lo), Ok(hi)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) {
                return (lo + hi) / 2.0;
            }
        }
    }
    s.parse::<f64>().unwrap_or(f64::NAN)
}

/// Parse McGill magnetar catalog CSV.
pub fn parse_mcgill_csv(path: &Path) -> Result<Vec<Magnetar>, FetchError> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .has_headers(true)
        .from_path(path)
        .map_err(|e| FetchError::Validation(format!("CSV read error: {}", e)))?;

    let headers = reader
        .headers()
        .map_err(|e| FetchError::Validation(format!("Header read error: {}", e)))?
        .clone();

    let col = |name: &str| -> Option<usize> {
        headers
            .iter()
            .position(|h| h.to_lowercase().contains(&name.to_lowercase()))
    };

    let idx_name = col("name").or_else(|| col("source"));
    let idx_ra = col("ra");
    let idx_dec = col("dec");
    let idx_gl = col("gl").or_else(|| col("l"));
    let idx_gb = col("gb").or_else(|| col("b"));
    let idx_p = col("period").or_else(|| col("p0"));
    let idx_pdot = col("pdot").or_else(|| col("p1"));
    let idx_b = col("dipole").or_else(|| col("bfield")).or_else(|| col("b_"));
    let idx_age = col("age");
    let idx_edot = col("edot").or_else(|| col("lsd"));
    let idx_dist = col("dist");
    let idx_dm = col("dm");
    let idx_lx = col("lx").or_else(|| col("lumin"));

    let get_str = |record: &csv::StringRecord, idx: Option<usize>| -> String {
        idx.and_then(|i| record.get(i))
            .unwrap_or("")
            .trim()
            .to_string()
    };

    let get_f64 = |record: &csv::StringRecord, idx: Option<usize>| -> f64 {
        idx.and_then(|i| record.get(i))
            .map(parse_f64)
            .unwrap_or(f64::NAN)
    };

    let mut magnetars = Vec::new();
    for result in reader.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        let name = get_str(&record, idx_name);
        if name.is_empty() {
            continue;
        }

        magnetars.push(Magnetar {
            name,
            ra: get_f64(&record, idx_ra),
            dec: get_f64(&record, idx_dec),
            gl: get_f64(&record, idx_gl),
            gb: get_f64(&record, idx_gb),
            period: get_f64(&record, idx_p),
            pdot: get_f64(&record, idx_pdot),
            b_dipole: get_f64(&record, idx_b),
            age: get_f64(&record, idx_age),
            edot: get_f64(&record, idx_edot),
            distance: get_f64(&record, idx_dist),
            dm: get_f64(&record, idx_dm),
            lx: get_f64(&record, idx_lx),
        });
    }

    Ok(magnetars)
}

const MCGILL_URLS: &[&str] = &[
    "http://www.physics.mcgill.ca/~pulsar/magnetar/TabO1.csv",
];

/// McGill magnetar catalog dataset provider.
pub struct McgillProvider;

impl DatasetProvider for McgillProvider {
    fn name(&self) -> &str { "McGill Magnetar Catalog" }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("mcgill_magnetars.csv");
        download_with_fallbacks(self.name(), MCGILL_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("mcgill_magnetars.csv").exists()
    }
}
