//! ATNF Pulsar Catalogue (Manchester et al. 2005) parser and fetcher.
//!
//! The ATNF pulsar catalogue is the canonical reference for known pulsars.
//! We fetch via the HEASARC Xamin query API which provides CSV output.
//!
//! Source: https://www.atnf.csiro.au/research/pulsar/psrcat/
//! HEASARC mirror: https://heasarc.gsfc.nasa.gov/xamin/

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_with_fallbacks};
use std::path::{Path, PathBuf};

/// A pulsar from the ATNF catalogue.
#[derive(Debug, Clone)]
pub struct Pulsar {
    /// Pulsar J-name (e.g., J0534+2200 for Crab).
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
    pub p0: f64,
    /// Period derivative (s/s).
    pub p1: f64,
    /// Dispersion measure (pc/cm^3).
    pub dm: f64,
    /// Rotation measure (rad/m^2).
    pub rm: f64,
    /// Characteristic age (years).
    pub age: f64,
    /// Surface magnetic field (Gauss).
    pub bsurf: f64,
    /// Spin-down luminosity (erg/s).
    pub edot: f64,
    /// Distance (kpc).
    pub dist: f64,
    /// S1400: flux density at 1400 MHz (mJy).
    pub s1400: f64,
    /// Binary period (days).
    pub pb: f64,
    /// Pulsar type.
    pub ptype: String,
}

fn parse_f64(s: &str) -> f64 {
    let s = s.trim();
    if s.is_empty() || s == "null" || s == "NULL" || s == "nan" {
        return f64::NAN;
    }
    s.parse::<f64>().unwrap_or(f64::NAN)
}

/// Parse ATNF pulsar catalog CSV.
pub fn parse_atnf_csv(path: &Path) -> Result<Vec<Pulsar>, FetchError> {
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
        headers.iter().position(|h| {
            let h_lower = h.to_lowercase();
            h_lower == name.to_lowercase() || h_lower.contains(&name.to_lowercase())
        })
    };

    let idx_name = col("name").or_else(|| col("jname")).or_else(|| col("psrj"));
    let idx_ra = col("rajd").or_else(|| col("ra"));
    let idx_dec = col("decjd").or_else(|| col("dec"));
    let idx_gl = col("gl");
    let idx_gb = col("gb");
    let idx_p0 = col("p0");
    let idx_p1 = col("p1");
    let idx_dm = col("dm");
    let idx_rm = col("rm");
    let idx_age = col("age");
    let idx_bsurf = col("bsurf");
    let idx_edot = col("edot");
    let idx_dist = col("dist");
    let idx_s1400 = col("s1400");
    let idx_pb = col("pb");
    let idx_type = col("type");

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

    let mut pulsars = Vec::new();
    for result in reader.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        let name = get_str(&record, idx_name);
        if name.is_empty() {
            continue;
        }

        pulsars.push(Pulsar {
            name,
            ra: get_f64(&record, idx_ra),
            dec: get_f64(&record, idx_dec),
            gl: get_f64(&record, idx_gl),
            gb: get_f64(&record, idx_gb),
            p0: get_f64(&record, idx_p0),
            p1: get_f64(&record, idx_p1),
            dm: get_f64(&record, idx_dm),
            rm: get_f64(&record, idx_rm),
            age: get_f64(&record, idx_age),
            bsurf: get_f64(&record, idx_bsurf),
            edot: get_f64(&record, idx_edot),
            dist: get_f64(&record, idx_dist),
            s1400: get_f64(&record, idx_s1400),
            pb: get_f64(&record, idx_pb),
            ptype: get_str(&record, idx_type),
        });
    }

    Ok(pulsars)
}

const ATNF_URLS: &[&str] = &[
    // HEASARC Xamin Query API -- returns pipe-delimited CSV (needs post-processing)
    // Note: the VO cone search endpoint returns VOTable XML, not CSV.
    // The QueryServlet with format=csv returns pipe-delimited text that needs
    // conversion to proper CSV. For pre-processed data, use the cached file.
    "https://heasarc.gsfc.nasa.gov/xamin/QueryServlet?table=atnfpulsar&format=csv&resultmax=0&fields=name,lii,bii,period,period_dot,dm,s1400,dist",
];

/// ATNF pulsar catalog dataset provider.
pub struct AtnfProvider;

impl DatasetProvider for AtnfProvider {
    fn name(&self) -> &str { "ATNF Pulsar Catalogue" }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("atnf_pulsars.csv");
        download_with_fallbacks(self.name(), ATNF_URLS, &output, config.skip_existing)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("atnf_pulsars.csv").exists()
    }
}
