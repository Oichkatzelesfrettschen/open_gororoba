//! PMNS angle extraction from unitary matrices.
//!
//! # PDG parameterization
//!
//! The standard PMNS parameterization (PDG 2024) uses three mixing angles
//! and one CP-violating phase:
//!
//! ```text
//! U_PMNS = R_23(theta_23) * diag(1, 1, e^{-i*delta}) * R_13(theta_13)
//!        * diag(1, e^{i*delta}, 1) * R_12(theta_12)
//! ```
//!
//! From a real PMNS matrix (no CP phase), the angles are extracted via:
//!
//! ```text
//! sin(theta_13) = |U_e3|          -- reactor angle
//! sin(theta_12) = |U_e2| / cos(theta_13)  -- solar angle
//! sin(theta_23) = |U_mu3| / cos(theta_13) -- atmospheric angle
//! ```
//!
//! # Callers
//!
//! - `neutrino_sector::test_pmns_gauss_newton_regression` (primary regression)
//! - `neutrino_sector::test_cp_violation_phase_only` (CP pipeline)
//! - All PMNS construction functions in `neutrino_sector.rs`

/// Extract PMNS mixing angles (in degrees) from a 3x3 unitary matrix.
///
/// Uses the standard PDG parameterization:
///   sin(theta_13) = |U_e3|
///   sin(theta_12) = |U_e2| / cos(theta_13)
///   sin(theta_23) = |U_mu3| / cos(theta_13)
pub fn extract_pmns_angles(u: &faer::Mat<f64>) -> (f64, f64, f64) {
    let u_e3 = u.read(0, 2).abs();
    let theta_13 = u_e3.min(1.0).asin();
    let cos_13 = theta_13.cos();

    let theta_12 = if cos_13 > 1e-15 {
        (u.read(0, 1).abs() / cos_13).min(1.0).asin()
    } else {
        0.0
    };
    let theta_23 = if cos_13 > 1e-15 {
        (u.read(1, 2).abs() / cos_13).min(1.0).asin()
    } else {
        0.0
    };

    (theta_12.to_degrees(), theta_13.to_degrees(), theta_23.to_degrees())
}

/// PDG 2024 central values and 1-sigma uncertainties (normal ordering).
#[derive(Clone, Copy)]
pub struct Pdg2024 {
    pub theta_12_deg: f64,
    pub theta_12_err: f64,
    pub theta_13_deg: f64,
    pub theta_13_err: f64,
    pub theta_23_deg: f64,
    pub theta_23_err: f64,
    pub delta_cp_deg: f64,
    pub delta_cp_err: f64,
    pub dm21_sq_ev2: f64,
    pub dm21_sq_err: f64,
    pub dm31_sq_ev2: f64,
    pub dm31_sq_err: f64,
}

impl Default for Pdg2024 {
    fn default() -> Self {
        Self {
            theta_12_deg: 33.41, theta_12_err: 0.75,
            theta_13_deg: 8.54,  theta_13_err: 0.12,
            theta_23_deg: 49.0,  theta_23_err: 1.1,
            delta_cp_deg: 195.0, delta_cp_err: 25.0,
            dm21_sq_ev2: 7.53e-5, dm21_sq_err: 0.18e-5,
            dm31_sq_ev2: 2.453e-3, dm31_sq_err: 0.033e-3,
        }
    }
}
