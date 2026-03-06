//! Minimal CIF parser for unit cell parameters.
//!
//! Extracts only:
//! - `_cell_length_a/b/c` (Angstrom)
//! - `_cell_angle_alpha/beta/gamma` (degrees)
//! - `_symmetry_space_group_name_H-M` or `_space_group_IT_number`
//!
//! This is NOT a full CIF parser. It handles the subset of CIF needed to
//! extract unit cell geometry for sedenion field grid seeding.
//!
//! Reference: Hall, Allen, Brown (1991), Acta Cryst. A47, 655-685.

/// Unit cell parameters parsed from a CIF file.
#[derive(Debug, Clone)]
pub struct CellParams {
    /// Lattice parameter a (Angstrom).
    pub a: f64,
    /// Lattice parameter b (Angstrom).
    pub b: f64,
    /// Lattice parameter c (Angstrom).
    pub c: f64,
    /// Angle alpha between b and c axes (degrees).
    pub alpha_deg: f64,
    /// Angle beta between a and c axes (degrees).
    pub beta_deg: f64,
    /// Angle gamma between a and b axes (degrees).
    pub gamma_deg: f64,
    /// Space group H-M symbol, if found.
    pub space_group_hm: Option<String>,
    /// Space group IT number, if found.
    pub space_group_number: Option<u32>,
}

impl CellParams {
    /// Whether this cell is cubic (a=b=c, alpha=beta=gamma=90).
    pub fn is_cubic(&self, tol: f64) -> bool {
        (self.a - self.b).abs() < tol
            && (self.b - self.c).abs() < tol
            && (self.alpha_deg - 90.0).abs() < tol
            && (self.beta_deg - 90.0).abs() < tol
            && (self.gamma_deg - 90.0).abs() < tol
    }

    /// Aspect ratios (b/a, c/a) for anisotropic grid seeding.
    pub fn aspect_ratios(&self) -> (f64, f64) {
        (self.b / self.a, self.c / self.a)
    }

    /// Cell volume (Angstrom^3) from general triclinic formula.
    pub fn volume(&self) -> f64 {
        let alpha = self.alpha_deg.to_radians();
        let beta = self.beta_deg.to_radians();
        let gamma = self.gamma_deg.to_radians();
        let ca = alpha.cos();
        let cb = beta.cos();
        let cg = gamma.cos();
        self.a * self.b * self.c * (1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg).sqrt()
    }
}

/// Parse unit cell parameters from CIF text content.
///
/// Returns `None` if any of the three cell lengths are missing.
pub fn parse_cif_cell_params(content: &str) -> Option<CellParams> {
    let mut a = None;
    let mut b = None;
    let mut c = None;
    let mut alpha = Some(90.0);
    let mut beta = Some(90.0);
    let mut gamma = Some(90.0);
    let mut sg_hm = None;
    let mut sg_num = None;

    for line in content.lines() {
        let line = line.trim();

        if let Some(val) = extract_cif_value(line, "_cell_length_a") {
            a = parse_cif_float(val);
        } else if let Some(val) = extract_cif_value(line, "_cell_length_b") {
            b = parse_cif_float(val);
        } else if let Some(val) = extract_cif_value(line, "_cell_length_c") {
            c = parse_cif_float(val);
        } else if let Some(val) = extract_cif_value(line, "_cell_angle_alpha") {
            alpha = parse_cif_float(val);
        } else if let Some(val) = extract_cif_value(line, "_cell_angle_beta") {
            beta = parse_cif_float(val);
        } else if let Some(val) = extract_cif_value(line, "_cell_angle_gamma") {
            gamma = parse_cif_float(val);
        } else if let Some(val) = extract_cif_value(line, "_symmetry_space_group_name_H-M") {
            sg_hm = Some(val.trim_matches('\'').trim_matches('"').to_string());
        } else if let Some(val) = extract_cif_value(line, "_space_group_IT_number") {
            sg_num = val.parse::<u32>().ok();
        }
    }

    Some(CellParams {
        a: a?,
        b: b?,
        c: c?,
        alpha_deg: alpha?,
        beta_deg: beta?,
        gamma_deg: gamma?,
        space_group_hm: sg_hm,
        space_group_number: sg_num,
    })
}

/// Extract the value after a CIF tag on the same line.
fn extract_cif_value<'a>(line: &'a str, tag: &str) -> Option<&'a str> {
    if let Some(rest) = line.strip_prefix(tag) {
        let rest = rest.trim();
        if rest.is_empty() { None } else { Some(rest) }
    } else {
        None
    }
}

/// Parse a CIF numeric value, stripping uncertainty parentheses: "4.0862(3)" -> 4.0862.
fn parse_cif_float(s: &str) -> Option<f64> {
    let s = s.trim();
    // Strip trailing uncertainty in parentheses: "4.0862(3)" -> "4.0862"
    let s = if let Some(idx) = s.find('(') {
        &s[..idx]
    } else {
        s
    };
    s.parse::<f64>().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SILVER_CIF: &str = r#"
data_9008459
_chemical_formula_sum            Ag
_chemical_name_mineral           Silver
_space_group_IT_number           225
_symmetry_space_group_name_H-M   'F m -3 m'
_cell_angle_alpha                90
_cell_angle_beta                 90
_cell_angle_gamma                90
_cell_length_a                   4.0862
_cell_length_b                   4.0862
_cell_length_c                   4.0862
_cell_volume                     68.227
"#;

    #[test]
    fn test_parse_silver_cif() {
        let params = parse_cif_cell_params(SILVER_CIF).unwrap();
        assert!((params.a - 4.0862).abs() < 1e-4, "a = {}", params.a);
        assert!((params.b - 4.0862).abs() < 1e-4);
        assert!((params.c - 4.0862).abs() < 1e-4);
        assert!(params.is_cubic(0.01));
        assert_eq!(params.space_group_number, Some(225));
        assert_eq!(params.space_group_hm.as_deref(), Some("F m -3 m"));
    }

    #[test]
    fn test_parse_uncertainty_stripping() {
        let val = parse_cif_float("4.0862(3)");
        assert!((val.unwrap() - 4.0862).abs() < 1e-4);
    }

    #[test]
    fn test_volume_cubic() {
        let params = parse_cif_cell_params(SILVER_CIF).unwrap();
        let vol = params.volume();
        // a^3 = 4.0862^3 = 68.227
        assert!((vol - 68.227).abs() < 0.1, "Volume = {}", vol);
    }

    #[test]
    fn test_aspect_ratios_cubic() {
        let params = parse_cif_cell_params(SILVER_CIF).unwrap();
        let (r1, r2) = params.aspect_ratios();
        assert!((r1 - 1.0).abs() < 1e-6);
        assert!((r2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_orthorhombic() {
        let cif = r#"
_cell_length_a                   20.07
_cell_length_b                   19.92
_cell_length_c                   13.42
_cell_angle_alpha                90
_cell_angle_beta                 90
_cell_angle_gamma                90
_space_group_IT_number           62
"#;
        let params = parse_cif_cell_params(cif).unwrap();
        assert!(!params.is_cubic(0.1));
        let (r1, r2) = params.aspect_ratios();
        assert!((r1 - 19.92 / 20.07).abs() < 0.01);
        assert!((r2 - 13.42 / 20.07).abs() < 0.01);
    }

    #[test]
    fn test_missing_lengths_returns_none() {
        let cif = "_cell_length_a  4.0\n";
        assert!(parse_cif_cell_params(cif).is_none());
    }

    #[test]
    fn test_parse_real_silver_file() {
        let path = "/usr/share/avogadro2/crystals/elements/Ag-Silver.cif";
        if let Ok(content) = std::fs::read_to_string(path) {
            let params = parse_cif_cell_params(&content).unwrap();
            assert!((params.a - 4.0862).abs() < 0.001, "Ag a = {}", params.a);
            assert_eq!(params.space_group_number, Some(225));
        }
    }
}
