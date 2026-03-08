//! DES Year 6 cosmological constraints from galaxy clustering and weak lensing.
//!
//! DES Y6 provides the final 3x2pt analysis combining galaxy clustering,
//! galaxy-galaxy lensing, and cosmic shear over ~5000 deg^2.
//!
//! Reference: DES Collaboration (2026), arXiv:2601.14559

/// Summary cosmological constraint from DES Y6 3x2pt analysis.
#[derive(Debug, Clone)]
pub struct DesY6Constraint {
    /// Parameter name (e.g., "S8", "Omega_m", "sigma_8").
    pub parameter: String,
    /// Best-fit (posterior mean) value.
    pub value: f64,
    /// 68% CL lower uncertainty (negative direction).
    pub err_lo: f64,
    /// 68% CL upper uncertainty (positive direction).
    pub err_hi: f64,
    /// Analysis label (e.g., "3x2pt", "cosmic-shear-only").
    pub analysis: String,
}

/// Hardcoded DES Y6 3x2pt cosmological constraints.
///
/// These are the headline results from the Y6 combined 3x2pt analysis
/// using the fiducial LCDM model.
///
/// Source: arXiv:2601.14559, Table 2 / Abstract
pub fn des_y6_3x2pt_constraints() -> Vec<DesY6Constraint> {
    vec![
        DesY6Constraint {
            parameter: "S8".to_string(),
            value: 0.782,
            err_lo: 0.015,
            err_hi: 0.015,
            analysis: "3x2pt-LCDM".to_string(),
        },
        DesY6Constraint {
            parameter: "Omega_m".to_string(),
            value: 0.292,
            err_lo: 0.020,
            err_hi: 0.020,
            analysis: "3x2pt-LCDM".to_string(),
        },
        DesY6Constraint {
            parameter: "sigma_8".to_string(),
            value: 0.812,
            err_lo: 0.031,
            err_hi: 0.031,
            analysis: "3x2pt-LCDM".to_string(),
        },
    ]
}

/// DES Y6 cosmic shear-only constraints for cross-validation.
///
/// Source: arXiv:2601.14559, Table 2
pub fn des_y6_cosmic_shear_constraints() -> Vec<DesY6Constraint> {
    vec![DesY6Constraint {
        parameter: "S8".to_string(),
        value: 0.784,
        err_lo: 0.020,
        err_hi: 0.020,
        analysis: "cosmic-shear-LCDM".to_string(),
    }]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_des_y6_3x2pt_constraints() {
        let c = des_y6_3x2pt_constraints();
        assert_eq!(c.len(), 3);

        let s8 = &c[0];
        assert_eq!(s8.parameter, "S8");
        // S8 should be in the range [0.7, 0.9] for LCDM
        assert!(s8.value > 0.7 && s8.value < 0.9, "S8 = {}", s8.value);
        assert!(s8.err_lo > 0.0 && s8.err_lo < 0.1);

        let omega_m = &c[1];
        assert_eq!(omega_m.parameter, "Omega_m");
        assert!(
            omega_m.value > 0.2 && omega_m.value < 0.4,
            "Omega_m = {}",
            omega_m.value
        );
    }

    #[test]
    fn test_des_y6_s8_consistency() {
        // 3x2pt and cosmic-shear S8 values should be consistent within 2-sigma
        let full = des_y6_3x2pt_constraints();
        let shear = des_y6_cosmic_shear_constraints();

        let s8_full = full.iter().find(|c| c.parameter == "S8").unwrap();
        let s8_shear = &shear[0];

        let diff = (s8_full.value - s8_shear.value).abs();
        let combined_err = (s8_full.err_hi.powi(2) + s8_shear.err_hi.powi(2)).sqrt();
        assert!(
            diff < 2.0 * combined_err,
            "S8 3x2pt vs shear differ by {:.3}, 2-sigma = {:.3}",
            diff,
            2.0 * combined_err
        );
    }

    #[test]
    fn test_des_y6_s8_lower_than_planck() {
        // DES Y6 S8 is systematically below Planck 2018 S8 ~ 0.832
        let c = des_y6_3x2pt_constraints();
        let s8 = c.iter().find(|c| c.parameter == "S8").unwrap();
        assert!(
            s8.value < 0.832,
            "DES Y6 S8 should be below Planck: {}",
            s8.value
        );
    }
}
