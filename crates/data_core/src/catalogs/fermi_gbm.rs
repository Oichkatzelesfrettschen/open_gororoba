//! Fermi GBM Burst Catalog parser and fetcher.
//!
//! The Fermi Gamma-ray Burst Monitor (GBM) catalog contains GRBs detected
//! by the Fermi satellite. Fetched via the HEASARC Xamin API.
//!
//! Source: https://heasarc.gsfc.nasa.gov/W3Browse/fermi/fermigbrst.html

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_heasarc_csv};
use std::path::{Path, PathBuf};

/// A gamma-ray burst from the Fermi GBM catalog.
#[derive(Debug, Clone)]
pub struct GrbEvent {
    /// GRB name (e.g., GRB080714086).
    pub name: String,
    /// Trigger time (MET seconds).
    pub trigger_time: f64,
    /// Right ascension (degrees).
    pub ra: f64,
    /// Declination (degrees).
    pub dec: f64,
    /// Error radius (degrees).
    pub error_radius: f64,
    /// T90 duration (seconds).
    pub t90: f64,
    /// T50 duration (seconds).
    pub t50: f64,
    /// Fluence in 10-1000 keV band (erg/cm^2).
    pub fluence: f64,
    /// Peak flux in 64ms timescale (ph/cm^2/s).
    pub pflx_best: f64,
    /// Spectral model best fit.
    pub flnc_spectrum_type: String,
}

fn parse_f64(s: &str) -> f64 {
    let s = s.trim();
    if s.is_empty() || s == "null" || s == "NULL" || s == "nan" {
        return f64::NAN;
    }
    s.parse::<f64>().unwrap_or(f64::NAN)
}

/// Parse Fermi GBM burst catalog CSV.
pub fn parse_fermi_gbm_csv(path: &Path) -> Result<Vec<GrbEvent>, FetchError> {
    let mut reader = csv::ReaderBuilder::new()
        .flexible(true)
        .has_headers(true)
        .comment(Some(b'#'))
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

    let idx_name = col("name").or_else(|| col("trigger_name"));
    let idx_time = col("trigger_time");
    let idx_ra = col("ra");
    let idx_dec = col("dec");
    let idx_err = col("error_radius");
    let idx_t90 = col("t90");
    let idx_t50 = col("t50");
    let idx_fluence = col("fluence");
    let idx_pflx = col("pflx_best").or_else(|| col("flux"));
    let idx_spec = col("flnc_spectrum_type").or_else(|| col("spectrum"));

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

    let mut events = Vec::new();
    for result in reader.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        let name = get_str(&record, idx_name);
        if name.is_empty() {
            continue;
        }

        events.push(GrbEvent {
            name,
            trigger_time: get_f64(&record, idx_time),
            ra: get_f64(&record, idx_ra),
            dec: get_f64(&record, idx_dec),
            error_radius: get_f64(&record, idx_err),
            t90: get_f64(&record, idx_t90),
            t50: get_f64(&record, idx_t50),
            fluence: get_f64(&record, idx_fluence),
            pflx_best: get_f64(&record, idx_pflx),
            flnc_spectrum_type: get_str(&record, idx_spec),
        });
    }

    Ok(events)
}

/// HEASARC QueryServlet URL for Fermi GBM burst catalog (pipe-delimited).
const FERMI_GBM_URL: &str = "https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3query.pl?\
tablehead=name%3Dfermigbrst&Action=Query&ResultMax=0&displaymode=FlatDisplay&\
Fields=name,trigger_time,ra,dec,error_radius,t90,t50,fluence,pflx_best,flnc_spectrum_type";

/// Fermi GBM burst catalog dataset provider.
pub struct FermiGbmProvider;

impl DatasetProvider for FermiGbmProvider {
    fn name(&self) -> &str { "Fermi GBM Burst Catalog" }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("fermi_gbm_grbs.csv");
        if config.skip_existing && output.exists() {
            eprintln!("  {} already cached at {}", self.name(), output.display());
            return Ok(output);
        }
        eprintln!("  Downloading {} from HEASARC...", self.name());
        let bytes = download_heasarc_csv(FERMI_GBM_URL, &output)?;
        eprintln!("  Saved {} bytes to {}", bytes, output.display());
        Ok(output)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("fermi_gbm_grbs.csv").exists()
    }
}
