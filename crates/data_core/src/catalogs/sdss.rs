//! SDSS DR18 quasar catalog via SkyServer REST API.
//!
//! Queries the SDSS SkyServer for spectroscopically confirmed quasars,
//! returning RA, Dec, redshift, and photometric magnitudes.
//!
//! Source: https://skyserver.sdss.org/
//! Reference: Almeida et al. (2023), ApJS 267, 44

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, download_to_string, validate_not_html};
use std::fs;
use std::path::{Path, PathBuf};

/// A quasar from SDSS DR18.
#[derive(Debug, Clone)]
pub struct SdssQuasar {
    /// SDSS object ID.
    pub objid: String,
    /// Right ascension (degrees).
    pub ra: f64,
    /// Declination (degrees).
    pub dec: f64,
    /// Spectroscopic redshift.
    pub z: f64,
    /// Redshift error.
    pub z_err: f64,
    /// PSF magnitude in u band.
    pub mag_u: f64,
    /// PSF magnitude in g band.
    pub mag_g: f64,
    /// PSF magnitude in r band.
    pub mag_r: f64,
    /// PSF magnitude in i band.
    pub mag_i: f64,
    /// PSF magnitude in z band.
    pub mag_z: f64,
}

fn parse_f64(s: &str) -> f64 {
    let s = s.trim();
    if s.is_empty() || s == "null" || s == "NULL" {
        return f64::NAN;
    }
    s.parse::<f64>().unwrap_or(f64::NAN)
}

/// Parse SDSS quasar CSV data.
pub fn parse_sdss_quasar_csv(path: &Path) -> Result<Vec<SdssQuasar>, FetchError> {
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
        headers.iter().position(|h| h.eq_ignore_ascii_case(name))
    };

    let idx_objid = col("objid").or_else(|| col("objID"));
    let idx_ra = col("ra");
    let idx_dec = col("dec");
    let idx_z = col("z");
    let idx_zerr = col("zerr").or_else(|| col("zErr"));
    let idx_u = col("u");
    let idx_g = col("g");
    let idx_r = col("r");
    let idx_i = col("i");
    let idx_zband = col("zmag").or_else(|| {
        // "z" is already taken for redshift; use column position
        headers.iter().position(|h| h == "z_mag")
    });

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

    let mut quasars = Vec::new();
    for result in reader.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        let objid = get_str(&record, idx_objid);
        if objid.is_empty() {
            continue;
        }

        quasars.push(SdssQuasar {
            objid,
            ra: get_f64(&record, idx_ra),
            dec: get_f64(&record, idx_dec),
            z: get_f64(&record, idx_z),
            z_err: get_f64(&record, idx_zerr),
            mag_u: get_f64(&record, idx_u),
            mag_g: get_f64(&record, idx_g),
            mag_r: get_f64(&record, idx_r),
            mag_i: get_f64(&record, idx_i),
            mag_z: get_f64(&record, idx_zband),
        });
    }

    Ok(quasars)
}

/// SDSS SkyServer SQL query for TOP 50000 quasars.
const SDSS_QUERY: &str = "\
SELECT TOP 50000 \
  s.objID as objid, s.ra, s.dec, s.z, s.zErr as zerr, \
  p.psfMag_u as u, p.psfMag_g as g, p.psfMag_r as r, \
  p.psfMag_i as i \
FROM SpecObj s \
JOIN PhotoObj p ON s.bestObjID = p.objID \
WHERE s.class = 'QSO' AND s.zWarning = 0 AND s.z > 0.1 \
ORDER BY s.z";

/// Build the SkyServer CSV download URL.
fn skyserver_csv_url(query: &str) -> String {
    let encoded = query.replace(' ', "+");
    format!(
        "https://skyserver.sdss.org/dr18/SkyServerWS/SearchTools/SqlSearch?cmd={}&format=csv",
        encoded
    )
}

/// SDSS DR18 quasar catalog dataset provider.
pub struct SdssQsoProvider;

impl DatasetProvider for SdssQsoProvider {
    fn name(&self) -> &str { "SDSS DR18 Quasars" }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("sdss_dr18_quasars.csv");
        if config.skip_existing && output.exists() {
            eprintln!("  {} already cached at {}", self.name(), output.display());
            return Ok(output);
        }

        let url = skyserver_csv_url(SDSS_QUERY);
        eprintln!("  Downloading {} from SkyServer...", self.name());
        let body = download_to_string(&url)?;
        validate_not_html(body.as_bytes())?;

        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&output, &body)?;
        eprintln!("  Saved {} bytes to {}", body.len(), output.display());
        Ok(output)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("sdss_dr18_quasars.csv").exists()
    }
}
