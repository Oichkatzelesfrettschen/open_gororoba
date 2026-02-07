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

/// Number of fields in the GaiaSource struct.
pub const GAIA_SOURCE_FIELD_COUNT: usize = 11;

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_csv(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_parse_synthetic_gaia() {
        let csv = "\
source_id,ra,dec,parallax,parallax_error,pmra,pmdec,radial_velocity,radial_velocity_error,phot_g_mean_mag,bp_rp
4472832130942575872,269.448,-1.942,546.976,0.029,-3775.4,10362.5,-110.51,0.16,9.53,2.88
5853498713190525696,217.429,-62.681,768.067,0.049,3853.2,-693.8,21.7,0.28,0.01,-0.02
";
        let f = write_temp_csv(csv);
        let sources = parse_gaia_csv(f.path()).unwrap();
        assert_eq!(sources.len(), 2);

        // Barnard's Star (roughly)
        let s1 = &sources[0];
        assert_eq!(s1.source_id, "4472832130942575872");
        assert!((s1.parallax - 546.976).abs() < 0.001);
        assert!((s1.pmra - (-3775.4)).abs() < 0.1);
        assert!((s1.radial_velocity - (-110.51)).abs() < 0.01);
        assert!((s1.phot_g_mean_mag - 9.53).abs() < 0.01);

        // Alpha Centauri (roughly)
        let s2 = &sources[1];
        assert_eq!(s2.source_id, "5853498713190525696");
        assert!((s2.parallax - 768.067).abs() < 0.001);
        assert!((s2.bp_rp - (-0.02)).abs() < 0.01);
    }

    #[test]
    fn test_gaia_empty_source_id_skipped() {
        let csv = "\
source_id,ra,dec,parallax,parallax_error,pmra,pmdec,radial_velocity,radial_velocity_error,phot_g_mean_mag,bp_rp
,0,0,100,1,0,0,0,0,5.0,0.5
";
        let f = write_temp_csv(csv);
        let sources = parse_gaia_csv(f.path()).unwrap();
        assert_eq!(sources.len(), 0, "empty source_id should be skipped");
    }

    #[test]
    fn test_gaia_null_fields() {
        let csv = "\
source_id,ra,dec,parallax,parallax_error,pmra,pmdec,radial_velocity,radial_velocity_error,phot_g_mean_mag,bp_rp
123456789,null,NULL,500,null,null,null,null,null,null,null
";
        let f = write_temp_csv(csv);
        let sources = parse_gaia_csv(f.path()).unwrap();
        assert_eq!(sources.len(), 1);
        assert!(sources[0].ra.is_nan());
        assert!(sources[0].pmra.is_nan());
    }

    #[test]
    fn test_gaia_source_field_count() {
        assert_eq!(GAIA_SOURCE_FIELD_COUNT, 11);
    }
}
