//! Gaia DR3 Milky Way rotation curve parser (Jiao et al. 2023).
//!
//! Parses the Milky Way circular velocity profile derived from Gaia DR3
//! astrometry. This is the most precise single-galaxy rotation curve
//! available, with sub-km/s errors from 5-25 kpc.
//!
//! The VizieR table (J/ApJ/946/73/table1) has columns:
//!   R(kpc)  V_c(km/s)  eV_lo(km/s)  eV_hi(km/s)
//!
//! where eV_lo and eV_hi are asymmetric 1-sigma uncertainties.
//!
//! Reference: Jiao, Hammer, Wang et al., ApJ 946 (2023) 73.

use std::path::Path;

/// A single radial bin in the Milky Way rotation curve.
#[derive(Debug, Clone)]
pub struct MwRotationPoint {
    /// Galactocentric radius in kpc.
    pub r_kpc: f64,
    /// Circular velocity in km/s.
    pub v_c_km_s: f64,
    /// Lower 1-sigma uncertainty in km/s (asymmetric).
    pub v_err_lo_km_s: f64,
    /// Upper 1-sigma uncertainty in km/s (asymmetric).
    pub v_err_hi_km_s: f64,
}

impl MwRotationPoint {
    /// Symmetrized uncertainty: average of lower and upper errors.
    pub fn v_err_sym_km_s(&self) -> f64 {
        (self.v_err_lo_km_s + self.v_err_hi_km_s) / 2.0
    }
}

/// The complete Milky Way rotation curve.
#[derive(Debug, Clone)]
pub struct MwRotationCurve {
    /// Source reference (e.g. "Jiao2023" or "Ou2024").
    pub source: String,
    /// Data points sorted by radius.
    pub points: Vec<MwRotationPoint>,
}

/// Parse a Gaia MW rotation curve table (VizieR format).
///
/// Handles both whitespace-separated and fixed-width formats.
/// Expected columns: R V_c eV_lo eV_hi (minimum 4 columns).
///
/// If only 3 columns are present, treats column 3 as symmetric error.
pub fn parse_mw_rotation_curve(path: &Path, source: &str) -> Result<MwRotationCurve, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("read error: {e}"))?;

    let mut points = Vec::new();
    for line in content.lines() {
        let line = line.trim();
        if line.is_empty()
            || line.starts_with('#')
            || line.starts_with('-')
            || line.starts_with('|')
        {
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        if fields.len() < 3 {
            continue;
        }
        let pf = |i: usize| -> f64 { fields[i].parse().unwrap_or(f64::NAN) };

        // Auto-detect: skip leading non-numeric columns (e.g. Cepheid name).
        let offset = if pf(0).is_finite() { 0 } else { 1 };
        if fields.len() < offset + 3 {
            continue;
        }
        let pfo = |i: usize| -> f64 { pf(i + offset) };

        let (r, v_c, err_lo, err_hi);
        let ncols = fields.len() - offset;
        if offset > 0 && ncols >= 4 {
            // Name R R_err V V_err (Mroz 2019 Cepheids)
            r = pfo(0);
            v_c = pfo(2);
            let verr = pfo(3).abs();
            err_lo = verr;
            err_hi = verr;
        } else if ncols >= 4 {
            // R V_c sigma+ sigma- [N_star] (Ou 2024)
            r = pfo(0);
            v_c = pfo(1);
            err_lo = pfo(2).abs();
            err_hi = pfo(3).abs();
        } else {
            // R V_c sigma (Jiao 2023)
            r = pfo(0);
            v_c = pfo(1);
            let err = pfo(2).abs();
            err_lo = err;
            err_hi = err;
        }

        if !r.is_finite() || !v_c.is_finite() {
            continue;
        }

        points.push(MwRotationPoint {
            r_kpc: r,
            v_c_km_s: v_c,
            v_err_lo_km_s: err_lo,
            v_err_hi_km_s: err_hi,
        });
    }

    if points.is_empty() {
        return Err(format!("no valid data points in {}", path.display()));
    }

    // Sort by radius
    points.sort_by(|a, b| a.r_kpc.partial_cmp(&b.r_kpc).unwrap());

    Ok(MwRotationCurve {
        source: source.to_string(),
        points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_temp(content: &str) -> tempfile::TempPath {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f.into_temp_path()
    }

    #[test]
    fn test_parse_four_column() {
        let content = "\
# R(kpc) V_c(km/s) eV_lo eV_hi
 5.0  234.5  2.1  2.3
 8.0  229.0  1.5  1.8
12.0  220.3  3.0  3.2
20.0  195.1  5.5  6.0
25.0  180.2  8.1  8.5
";
        let path = write_temp(content);
        let curve = parse_mw_rotation_curve(Path::new(&path), "Jiao2023").unwrap();
        assert_eq!(curve.points.len(), 5);
        assert_eq!(curve.source, "Jiao2023");
        assert!((curve.points[0].r_kpc - 5.0).abs() < 1e-10);
        assert!((curve.points[0].v_c_km_s - 234.5).abs() < 1e-10);
        assert!((curve.points[0].v_err_lo_km_s - 2.1).abs() < 1e-10);
        assert!((curve.points[0].v_err_hi_km_s - 2.3).abs() < 1e-10);
    }

    #[test]
    fn test_parse_three_column_symmetric() {
        let content = "\
 5.0  234.5  2.2
 8.0  229.0  1.6
";
        let path = write_temp(content);
        let curve = parse_mw_rotation_curve(Path::new(&path), "test").unwrap();
        assert_eq!(curve.points.len(), 2);
        // Symmetric: lo == hi
        assert!((curve.points[0].v_err_lo_km_s - 2.2).abs() < 1e-10);
        assert!((curve.points[0].v_err_hi_km_s - 2.2).abs() < 1e-10);
    }

    #[test]
    fn test_sorted_by_radius() {
        let content = "\
20.0  195.1  5.5  6.0
 5.0  234.5  2.1  2.3
12.0  220.3  3.0  3.2
";
        let path = write_temp(content);
        let curve = parse_mw_rotation_curve(Path::new(&path), "test").unwrap();
        for w in curve.points.windows(2) {
            assert!(w[0].r_kpc < w[1].r_kpc);
        }
    }

    #[test]
    fn test_symmetrized_error() {
        let pt = MwRotationPoint {
            r_kpc: 8.0,
            v_c_km_s: 229.0,
            v_err_lo_km_s: 1.5,
            v_err_hi_km_s: 1.9,
        };
        assert!((pt.v_err_sym_km_s() - 1.7).abs() < 1e-10);
    }

    #[test]
    fn test_empty_file() {
        let content = "# header only\n";
        let path = write_temp(content);
        let result = parse_mw_rotation_curve(Path::new(&path), "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_mw_velocity_physical_range() {
        // The MW rotation curve at 8 kpc (solar radius) should be ~220-240 km/s
        let content = "\
 8.0  229.0  1.5  1.8
";
        let path = write_temp(content);
        let curve = parse_mw_rotation_curve(Path::new(&path), "Jiao2023").unwrap();
        let v = curve.points[0].v_c_km_s;
        assert!(
            (200.0..260.0).contains(&v),
            "MW v_c at 8 kpc should be ~220-240 km/s, got {v}"
        );
    }
}
