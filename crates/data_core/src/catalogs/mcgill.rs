//! McGill Online Magnetar Catalog parser and fetcher.
//!
//! The McGill catalog is the canonical reference for known magnetars
//! (SGRs and AXPs). Small dataset (~30 objects) but critical for
//! compact object population studies.
//!
//! Source: http://www.physics.mcgill.ca/~pulsar/magnetar/main.html
//! Reference: Olausen & Kaspi (2014), ApJS 212, 6

use crate::{
    fetcher::FetchError,
    parse::{parse_sexagesimal_dec_to_deg, parse_sexagesimal_ra_to_deg},
};
use std::path::Path;

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
        if parts.len() == 2
            && let (Ok(lo), Ok(hi)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>())
        {
            return (lo + hi) / 2.0;
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

    let normalize = |value: &str| -> String {
        value
            .trim()
            .to_ascii_lowercase()
            .chars()
            .filter(|ch| ch.is_ascii_alphanumeric())
            .collect()
    };
    let col = |names: &[&str]| -> Option<usize> {
        let normalized_names = names.iter().map(|name| normalize(name)).collect::<Vec<_>>();
        headers.iter().position(|header| {
            let normalized_header = normalize(header);
            normalized_names.contains(&normalized_header)
        })
    };

    let idx_name = col(&["name", "source"]);
    let idx_ra = col(&["ra"]);
    let idx_dec = col(&["decl", "dec"]);
    let idx_gl = col(&["gl", "l"]);
    let idx_gb = col(&["gb"]);
    let idx_p = col(&["period", "p0"]);
    let idx_pdot = col(&["pdot", "p1"]);
    let idx_b = col(&["b", "bdipole", "dipole", "bfield"]);
    let idx_age = col(&["age"]);
    let idx_edot = col(&["edot", "lsd"]);
    let idx_dist = col(&["dist", "distance"]);
    let idx_dm = col(&["dm"]);
    let idx_lx = col(&["lx", "lumin", "luminosity"]);

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

        let ra_str = get_str(&record, idx_ra);
        let dec_str = get_str(&record, idx_dec);
        let mut ra = parse_f64(&ra_str);
        let mut dec = parse_f64(&dec_str);
        if !ra.is_finite() && !ra_str.is_empty() {
            ra = parse_sexagesimal_ra_to_deg(&ra_str);
        }
        if !dec.is_finite() && !dec_str.is_empty() {
            dec = parse_sexagesimal_dec_to_deg(&dec_str);
        }

        magnetars.push(Magnetar {
            name,
            ra,
            dec,
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

const MCGILL_URLS: &[&str] = &["http://www.physics.mcgill.ca/~pulsar/magnetar/TabO1.csv"];

simple_provider! {
    /// McGill magnetar catalog dataset provider.
    pub struct McgillProvider;
    name = "McGill Magnetar Catalog";
    output = "mcgill_magnetars.csv";
    urls = MCGILL_URLS;
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
        assert!(
            (magnetars[0].distance - 4.0).abs() < 0.01,
            "range midpoint should be 4.0, got {}",
            magnetars[0].distance
        );
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

    #[test]
    fn test_parse_mcgill_sexagesimal_coordinates() {
        let csv = "\
Name,Period,RA,Decl,Dist
4U 0142+61,8.68,01 46 22.407,+61 45 03.19,3.6
";
        let f = write_temp_csv(csv);
        let magnetars = parse_mcgill_csv(f.path()).unwrap();
        assert_eq!(magnetars.len(), 1);
        assert!((magnetars[0].ra - 26.5933625).abs() < 1e-6);
        assert!((magnetars[0].dec - 61.75088611111111).abs() < 1e-6);
    }

    #[test]
    fn test_parse_mcgill_prefers_true_ra_column_over_ref_xray() {
        let csv = "\
Name,Ref_Xray,RA,Decl,Period,B
SGR TEST,tem08,01 46 22.407,+61 45 03.19,8.68869249,1.34E+14
";
        let f = write_temp_csv(csv);
        let magnetars = parse_mcgill_csv(f.path()).unwrap();
        assert_eq!(magnetars.len(), 1);
        assert!((magnetars[0].ra - 26.5933625).abs() < 1e-6);
        assert!((magnetars[0].dec - 61.75088611111111).abs() < 1e-6);
    }
}
