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

/// Number of fields in the Magnetar struct.
pub const MAGNETAR_FIELD_COUNT: usize = 13;

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
    fn test_parse_synthetic_mcgill() {
        let csv = "\
name,ra,dec,gl,gb,period,pdot,dipole,age,edot,dist,dm,lx
SGR 0418+5729,64.68,57.54,152.35,0.84,9.078,4.0e-15,0.061,36000,1.7e29,2.0,...,2.5e31
1E 2259+586,345.28,58.88,109.08,-0.99,6.979,4.8e-13,5.9,230,3.6e31,3.2,...,1.8e35
";
        let f = write_temp_csv(csv);
        let magnetars = parse_mcgill_csv(f.path()).unwrap();
        assert_eq!(magnetars.len(), 2);

        let sgr = &magnetars[0];
        assert_eq!(sgr.name, "SGR 0418+5729");
        assert!((sgr.ra - 64.68).abs() < 0.01);
        assert!((sgr.period - 9.078).abs() < 0.001);
        assert!((sgr.b_dipole - 0.061).abs() < 0.001);

        let axp = &magnetars[1];
        assert_eq!(axp.name, "1E 2259+586");
        assert!((axp.distance - 3.2).abs() < 0.01);
    }

    #[test]
    fn test_mcgill_range_midpoint() {
        // McGill uses dashes for ranges like "3.1-5.5"
        let csv = "\
name,ra,dec,gl,gb,period,pdot,dipole,age,edot,dist,dm,lx
SGR TEST,0,0,0,0,5.0,0,0,0,0,3.0-5.0,0,0
";
        let f = write_temp_csv(csv);
        let magnetars = parse_mcgill_csv(f.path()).unwrap();
        assert_eq!(magnetars.len(), 1);
        // Distance should be midpoint of 3.0 and 5.0
        assert!((magnetars[0].distance - 4.0).abs() < 0.01,
            "range midpoint should be 4.0, got {}", magnetars[0].distance);
    }

    #[test]
    fn test_mcgill_empty_name_skipped() {
        let csv = "\
name,ra,dec,gl,gb,period,pdot,dipole,age,edot,dist,dm,lx
,0,0,0,0,5.0,0,0,0,0,0,0,0
";
        let f = write_temp_csv(csv);
        let magnetars = parse_mcgill_csv(f.path()).unwrap();
        assert_eq!(magnetars.len(), 0, "empty name should be skipped");
    }

    #[test]
    fn test_mcgill_dash_sentinel() {
        let csv = "\
name,ra,dec,gl,gb,period,pdot,dipole,age,edot,dist,dm,lx
SGR TEST,0,0,0,0,5.0,--,...,0,0,0,...,0
";
        let f = write_temp_csv(csv);
        let magnetars = parse_mcgill_csv(f.path()).unwrap();
        assert_eq!(magnetars.len(), 1);
        assert!(magnetars[0].pdot.is_nan(), "-- should parse as NaN");
        assert!(magnetars[0].dm.is_nan(), "... should parse as NaN");
    }

    #[test]
    fn test_magnetar_field_count() {
        assert_eq!(MAGNETAR_FIELD_COUNT, 13);
    }
}
