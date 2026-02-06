//! ICGEM .gfc (Gravity Field Coefficients) format parser.
//!
//! The ICGEM format stores spherical harmonic gravity field models with
//! header metadata and coefficient data lines. Used by GRACE, GOCE, GRAIL,
//! and other gravity missions.
//!
//! Format specification: http://icgem.gfz-potsdam.de/ICGEM-Format-2023.pdf

use crate::fetcher::FetchError;
use std::path::Path;

/// A single spherical harmonic gravity coefficient.
#[derive(Debug, Clone)]
pub struct GravityCoefficient {
    /// Degree n.
    pub n: u32,
    /// Order m.
    pub m: u32,
    /// Cosine coefficient Cnm (fully normalized).
    pub cnm: f64,
    /// Sine coefficient Snm (fully normalized).
    pub snm: f64,
    /// Formal error of Cnm (if available).
    pub sigma_cnm: f64,
    /// Formal error of Snm (if available).
    pub sigma_snm: f64,
}

/// A complete gravity field model parsed from .gfc format.
#[derive(Debug, Clone)]
pub struct GravityField {
    /// Model name (from header).
    pub modelname: String,
    /// Earth gravity constant GM (m^3/s^2).
    pub earth_gravity_constant: f64,
    /// Reference radius (m).
    pub radius: f64,
    /// Maximum degree of the model.
    pub max_degree: u32,
    /// Spherical harmonic coefficients.
    pub coefficients: Vec<GravityCoefficient>,
}

/// Parse a Fortran-style float that may use 'D' as exponent delimiter.
fn parse_fortran_f64(s: &str) -> f64 {
    let s = s.trim();
    if s.is_empty() {
        return f64::NAN;
    }
    // Replace Fortran 'D' exponent notation with 'E'
    let normalized = s.replace(['D', 'd'], "E");
    normalized.parse::<f64>().unwrap_or(f64::NAN)
}

/// Parse an ICGEM .gfc file into a GravityField.
pub fn parse_gfc(path: &Path) -> Result<GravityField, FetchError> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| FetchError::Validation(format!("Read .gfc: {}", e)))?;

    let mut modelname = String::new();
    let mut gm = f64::NAN;
    let mut radius = f64::NAN;
    let mut max_degree: u32 = 0;
    let mut coefficients = Vec::new();
    let mut in_header = true;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Header parsing
        if in_header {
            if trimmed.starts_with("end_of_head") {
                in_header = false;
                continue;
            }
            if let Some(rest) = trimmed.strip_prefix("modelname") {
                modelname = rest.trim().to_string();
            } else if let Some(rest) = trimmed.strip_prefix("earth_gravity_constant") {
                gm = parse_fortran_f64(rest);
            } else if let Some(rest) = trimmed.strip_prefix("radius") {
                radius = parse_fortran_f64(rest);
            } else if let Some(rest) = trimmed.strip_prefix("max_degree") {
                max_degree = rest.trim().parse::<u32>().unwrap_or(0);
            }
            continue;
        }

        // Data lines: gfc n m Cnm Snm [sigma_Cnm sigma_Snm]
        if !trimmed.starts_with("gfc") {
            continue;
        }

        let fields: Vec<&str> = trimmed.split_whitespace().collect();
        if fields.len() < 5 {
            continue;
        }

        let n = fields[1].parse::<u32>().unwrap_or(0);
        let m = fields[2].parse::<u32>().unwrap_or(0);
        let cnm = parse_fortran_f64(fields[3]);
        let snm = parse_fortran_f64(fields[4]);
        let sigma_cnm = fields.get(5).map(|s| parse_fortran_f64(s)).unwrap_or(f64::NAN);
        let sigma_snm = fields.get(6).map(|s| parse_fortran_f64(s)).unwrap_or(f64::NAN);

        coefficients.push(GravityCoefficient {
            n, m, cnm, snm, sigma_cnm, sigma_snm,
        });
    }

    Ok(GravityField {
        modelname,
        earth_gravity_constant: gm,
        radius,
        max_degree,
        coefficients,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn test_parse_fortran_f64() {
        assert!((parse_fortran_f64("1.234D-05") - 1.234e-5).abs() < 1e-15);
        assert!((parse_fortran_f64("3.986004415D+14") - 3.986004415e14).abs() < 1.0);
        assert!((parse_fortran_f64("6.3781363D+06") - 6.3781363e6).abs() < 0.1);
        assert!((parse_fortran_f64("1.0E-10") - 1.0e-10).abs() < 1e-25);
    }

    #[test]
    fn test_parse_gfc_synthetic() {
        let dir = std::env::temp_dir().join("gfc_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.gfc");

        let mut f = std::fs::File::create(&path).unwrap();
        writeln!(f, "modelname       TestModel").unwrap();
        writeln!(f, "earth_gravity_constant  3.986004415D+14").unwrap();
        writeln!(f, "radius          6.3781363D+06").unwrap();
        writeln!(f, "max_degree      4").unwrap();
        writeln!(f, "end_of_head ===").unwrap();
        writeln!(f, "gfc  0  0  1.000000D+00  0.000000D+00  0.0D+00  0.0D+00").unwrap();
        writeln!(f, "gfc  2  0 -4.841653D-04  0.000000D+00  3.5D-11  0.0D+00").unwrap();
        writeln!(f, "gfc  2  1 -2.066155D-10  1.384413D-09  3.4D-11  3.4D-11").unwrap();

        let gf = parse_gfc(&path).unwrap();
        assert_eq!(gf.modelname, "TestModel");
        assert!((gf.earth_gravity_constant - 3.986004415e14).abs() < 1.0);
        assert!((gf.radius - 6.3781363e6).abs() < 0.1);
        assert_eq!(gf.max_degree, 4);
        assert_eq!(gf.coefficients.len(), 3);
        assert_eq!(gf.coefficients[1].n, 2);
        assert_eq!(gf.coefficients[1].m, 0);
        assert!((gf.coefficients[1].cnm - (-4.841653e-4)).abs() < 1e-10);

        std::fs::remove_dir_all(&dir).ok();
    }
}
