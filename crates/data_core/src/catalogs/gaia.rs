//! Gaia DR3 stellar catalog via ESA TAP.
//!
//! Queries the Gaia archive TAP endpoint for nearby stars with radial
//! velocities, providing 6D phase-space information for Galactic dynamics.
//!
//! Source: https://gea.esac.esa.int/archive/
//! Reference: Gaia Collaboration, Vallenari et al. (2023), A&A 674, A1

use crate::fetcher::{DatasetProvider, FetchConfig, FetchError, validate_not_html};
use crate::formats::tap;
use std::fs;
use std::path::{Path, PathBuf};

/// A stellar source from Gaia DR3.
#[derive(Debug, Clone)]
pub struct GaiaSource {
    /// Gaia source ID.
    pub source_id: String,
    /// Right ascension (degrees, ICRS, epoch 2016.0).
    pub ra: f64,
    /// Declination (degrees, ICRS, epoch 2016.0).
    pub dec: f64,
    /// Parallax (mas).
    pub parallax: f64,
    /// Parallax error (mas).
    pub parallax_error: f64,
    /// Proper motion in RA * cos(dec) (mas/yr).
    pub pmra: f64,
    /// Proper motion in Dec (mas/yr).
    pub pmdec: f64,
    /// Radial velocity (km/s).
    pub radial_velocity: f64,
    /// Radial velocity error (km/s).
    pub radial_velocity_error: f64,
    /// G-band mean magnitude.
    pub phot_g_mean_mag: f64,
    /// BP-RP color index.
    pub bp_rp: f64,
}

fn parse_f64(s: &str) -> f64 {
    let s = s.trim();
    if s.is_empty() || s == "null" || s == "NULL" {
        return f64::NAN;
    }
    s.parse::<f64>().unwrap_or(f64::NAN)
}

/// Parse Gaia DR3 CSV data.
pub fn parse_gaia_csv(path: &Path) -> Result<Vec<GaiaSource>, FetchError> {
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
        headers.iter().position(|h| h.eq_ignore_ascii_case(name))
    };

    let idx_sid = col("source_id");
    let idx_ra = col("ra");
    let idx_dec = col("dec");
    let idx_plx = col("parallax");
    let idx_plxe = col("parallax_error");
    let idx_pmra = col("pmra");
    let idx_pmdec = col("pmdec");
    let idx_rv = col("radial_velocity");
    let idx_rve = col("radial_velocity_error");
    let idx_gmag = col("phot_g_mean_mag");
    let idx_bprp = col("bp_rp");

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

    let mut sources = Vec::new();
    for result in reader.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        let source_id = get_str(&record, idx_sid);
        if source_id.is_empty() {
            continue;
        }

        sources.push(GaiaSource {
            source_id,
            ra: get_f64(&record, idx_ra),
            dec: get_f64(&record, idx_dec),
            parallax: get_f64(&record, idx_plx),
            parallax_error: get_f64(&record, idx_plxe),
            pmra: get_f64(&record, idx_pmra),
            pmdec: get_f64(&record, idx_pmdec),
            radial_velocity: get_f64(&record, idx_rv),
            radial_velocity_error: get_f64(&record, idx_rve),
            phot_g_mean_mag: get_f64(&record, idx_gmag),
            bp_rp: get_f64(&record, idx_bprp),
        });
    }

    Ok(sources)
}

/// ADQL query for nearby stars with radial velocities from Gaia DR3.
const GAIA_ADQL: &str = "\
SELECT TOP 50000 \
  source_id, ra, dec, parallax, parallax_error, \
  pmra, pmdec, radial_velocity, radial_velocity_error, \
  phot_g_mean_mag, bp_rp \
FROM gaiadr3.gaia_source \
WHERE radial_velocity IS NOT NULL \
  AND parallax > 5 \
  AND parallax_error/parallax < 0.1 \
ORDER BY parallax DESC";

const GAIA_TAP_BASE: &str = "https://gea.esac.esa.int/tap-server/tap";

/// Gaia DR3 nearby stars with radial velocities.
pub struct GaiaDr3Provider;

impl DatasetProvider for GaiaDr3Provider {
    fn name(&self) -> &str { "Gaia DR3 Nearby Stars" }

    fn fetch(&self, config: &FetchConfig) -> Result<PathBuf, FetchError> {
        let output = config.output_dir.join("gaia_dr3_nearby.csv");
        if config.skip_existing && output.exists() {
            eprintln!("  {} already cached at {}", self.name(), output.display());
            return Ok(output);
        }

        eprintln!("  Querying {} via TAP...", self.name());
        let body = tap::tap_query(GAIA_TAP_BASE, GAIA_ADQL, "csv")?;
        validate_not_html(body.as_bytes())?;

        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&output, &body)?;
        eprintln!("  Saved {} bytes to {}", body.len(), output.display());
        Ok(output)
    }

    fn is_cached(&self, config: &FetchConfig) -> bool {
        config.output_dir.join("gaia_dr3_nearby.csv").exists()
    }
}
