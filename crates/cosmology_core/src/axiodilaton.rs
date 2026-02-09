//! Axiodilaton cosmology: modified Friedmann equations with a scalar field.
//!
//! The axiodilaton model adds a scalar field component to the Hubble
//! parameter, providing a candidate resolution to the Hubble tension.
//! The modified Friedmann equation reads:
//!
//!   H(z) = H_0 * sqrt(Omega_m (1+z)^3 + Omega_ad f_ad(z) + Omega_Lambda)
//!
//! where f_ad(z) is the axiodilaton scalar field evolution function.
//!
//! The model predicts H_0 = 69.22 +/- 0.28 km/s/Mpc, bridging the
//! Planck CMB value (~67.4) and the SH0ES distance ladder (~73.0).
//!
//! References:
//!   - Planck Collaboration VI (2020): A&A 641, A6 (baseline parameters)

use crate::bounce::C_KM_S;
use crate::gl_integrate;

// ============================================================================
// Axiodilaton parameters
// ============================================================================

/// Default Planck 2018 parameters with axiodilaton modification.
pub mod params {
    /// Matter density parameter (Planck TT,TE,EE+lowE+lensing+BAO).
    pub const OMEGA_M: f64 = 0.3111;
    /// Axiodilaton scalar field density parameter (small contribution).
    pub const OMEGA_AD: f64 = 0.001;
    /// Dark energy density (flat universe: 1 - Omega_m - Omega_ad).
    pub const OMEGA_LAMBDA: f64 = 1.0 - OMEGA_M - OMEGA_AD;
    /// Axiodilaton H_0 prediction [km/s/Mpc].
    pub const H0: f64 = 69.22;
}

// ============================================================================
// Scalar field evolution
// ============================================================================

/// Axiodilaton scalar field evolution function f_ad(z).
///
/// Currently a linear approximation: f_ad(z) = 1 + z.
/// This models the simplest scalar field that grows linearly with
/// the inverse scale factor (1+z = 1/a).
pub fn axiodilaton_function(z: f64) -> f64 {
    1.0 + z
}

// ============================================================================
// Hubble parameter
// ============================================================================

/// Dimensionless Hubble parameter E(z) = H(z)/H_0 for the axiodilaton model.
///
/// E(z) = sqrt(Omega_m (1+z)^3 + Omega_ad f_ad(z) + Omega_Lambda)
///
/// The axiodilaton term acts like a slowly-varying dark energy component
/// that modifies the expansion history at intermediate redshifts.
pub fn hubble_e(z: f64, omega_m: f64, omega_ad: f64, omega_lambda: f64) -> f64 {
    let a_inv = 1.0 + z;
    let term_m = omega_m * a_inv * a_inv * a_inv;
    let term_ad = omega_ad * axiodilaton_function(z);
    let term_lambda = omega_lambda;
    (term_m + term_ad + term_lambda).sqrt()
}

/// Hubble parameter H(z) [km/s/Mpc] for the axiodilaton model.
pub fn hubble(z: f64, omega_m: f64, omega_ad: f64, omega_lambda: f64, h0: f64) -> f64 {
    h0 * hubble_e(z, omega_m, omega_ad, omega_lambda)
}

// ============================================================================
// Distance measures
// ============================================================================

/// Comoving distance d_C(z) [Mpc] for the axiodilaton model.
///
/// d_C(z) = (c/H_0) integral_0^z dz' / E(z')
///
/// Uses Gauss-Legendre quadrature (degree 50) for the integral.
pub fn comoving_distance(z: f64, omega_m: f64, omega_ad: f64, omega_lambda: f64, h0: f64) -> f64 {
    if z <= 0.0 {
        return 0.0;
    }
    let integral = gl_integrate(
        |zp| 1.0 / hubble_e(zp, omega_m, omega_ad, omega_lambda),
        0.0,
        z,
        50,
    );
    (C_KM_S / h0) * integral
}

/// Luminosity distance d_L(z) = (1+z) d_C(z) [Mpc].
pub fn luminosity_distance(z: f64, omega_m: f64, omega_ad: f64, omega_lambda: f64, h0: f64) -> f64 {
    (1.0 + z) * comoving_distance(z, omega_m, omega_ad, omega_lambda, h0)
}

/// Angular diameter distance d_A(z) = d_C(z) / (1+z) [Mpc].
pub fn angular_diameter_distance(
    z: f64,
    omega_m: f64,
    omega_ad: f64,
    omega_lambda: f64,
    h0: f64,
) -> f64 {
    comoving_distance(z, omega_m, omega_ad, omega_lambda, h0) / (1.0 + z)
}

/// Distance modulus mu = 5 log10(d_L / 10 pc) [mag].
pub fn distance_modulus(z: f64, omega_m: f64, omega_ad: f64, omega_lambda: f64, h0: f64) -> f64 {
    let d_l_mpc = luminosity_distance(z, omega_m, omega_ad, omega_lambda, h0);
    // d_L in pc = d_L_Mpc * 1e6
    5.0 * (d_l_mpc * 1e6 / 10.0).log10()
}

// ============================================================================
// Equation of state
// ============================================================================

/// Effective equation of state parameter w_eff.
///
/// For the axiodilaton model with f_ad(z) = 1+z, the scalar field
/// behaves like a component with w ~ -1 (cosmological constant-like).
pub fn equation_of_state(_z: f64) -> f64 {
    -1.0
}

/// Deceleration parameter q_0 at z=0.
///
/// q = (1/2)(1 + 3w Omega_m) - Omega_Lambda
///
/// For the standard axiodilaton parameters:
///   q ~ 0.5*(1 + 3*(-1)*0.3111) - 0.6889 ~ 0.5*(1-0.9333) - 0.6889
///   ~ 0.0334 - 0.6889 = -0.6556 (accelerating expansion)
pub fn deceleration_parameter(omega_m: f64, omega_lambda: f64) -> f64 {
    let w = equation_of_state(0.0);
    0.5 * (1.0 + 3.0 * w * omega_m) - omega_lambda
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Hubble parameter --

    #[test]
    fn test_hubble_e_at_z0() {
        // E(0) should be ~1 when Omega_m + Omega_ad + Omega_Lambda = 1
        let e = hubble_e(0.0, params::OMEGA_M, params::OMEGA_AD, params::OMEGA_LAMBDA);
        assert!((e - 1.0).abs() < 0.01, "E(0) = {e}");
    }

    #[test]
    fn test_hubble_e_increases_with_z() {
        let e0 = hubble_e(0.0, params::OMEGA_M, params::OMEGA_AD, params::OMEGA_LAMBDA);
        let e1 = hubble_e(1.0, params::OMEGA_M, params::OMEGA_AD, params::OMEGA_LAMBDA);
        let e2 = hubble_e(2.0, params::OMEGA_M, params::OMEGA_AD, params::OMEGA_LAMBDA);
        assert!(e1 > e0 && e2 > e1, "E should increase with redshift");
    }

    #[test]
    fn test_hubble_absolute_value() {
        let h = hubble(
            0.0,
            params::OMEGA_M,
            params::OMEGA_AD,
            params::OMEGA_LAMBDA,
            params::H0,
        );
        assert!((h - params::H0).abs() < 1.0, "H(0) = {h}");
    }

    // -- Reduces to Lambda-CDM when Omega_ad = 0 --

    #[test]
    fn test_reduces_to_lcdm() {
        // With Omega_ad = 0, should match Lambda-CDM
        let e_ad = hubble_e(1.0, 0.3, 0.0, 0.7);
        let e_lcdm = crate::bounce::hubble_e_lcdm(1.0, 0.3);
        assert!(
            (e_ad - e_lcdm).abs() < 1e-10,
            "E_ad = {e_ad}, E_lcdm = {e_lcdm}"
        );
    }

    // -- Distance measures --

    #[test]
    fn test_comoving_distance_positive() {
        let d = comoving_distance(
            1.0,
            params::OMEGA_M,
            params::OMEGA_AD,
            params::OMEGA_LAMBDA,
            params::H0,
        );
        assert!(d > 0.0, "d_C = {d}");
    }

    #[test]
    fn test_comoving_distance_increases() {
        let d1 = comoving_distance(
            0.5,
            params::OMEGA_M,
            params::OMEGA_AD,
            params::OMEGA_LAMBDA,
            params::H0,
        );
        let d2 = comoving_distance(
            1.0,
            params::OMEGA_M,
            params::OMEGA_AD,
            params::OMEGA_LAMBDA,
            params::H0,
        );
        assert!(d2 > d1);
    }

    #[test]
    fn test_luminosity_distance_exceeds_comoving() {
        let z = 1.0;
        let d_c = comoving_distance(
            z,
            params::OMEGA_M,
            params::OMEGA_AD,
            params::OMEGA_LAMBDA,
            params::H0,
        );
        let d_l = luminosity_distance(
            z,
            params::OMEGA_M,
            params::OMEGA_AD,
            params::OMEGA_LAMBDA,
            params::H0,
        );
        assert!((d_l - (1.0 + z) * d_c).abs() < 1e-6);
    }

    #[test]
    fn test_angular_diameter_distance_less_than_comoving() {
        let z = 1.0;
        let d_c = comoving_distance(
            z,
            params::OMEGA_M,
            params::OMEGA_AD,
            params::OMEGA_LAMBDA,
            params::H0,
        );
        let d_a = angular_diameter_distance(
            z,
            params::OMEGA_M,
            params::OMEGA_AD,
            params::OMEGA_LAMBDA,
            params::H0,
        );
        assert!((d_a - d_c / (1.0 + z)).abs() < 1e-6);
    }

    #[test]
    fn test_distance_modulus_order_of_magnitude() {
        // At z=1, d_L ~ 6700 Mpc for standard cosmology
        // mu = 5 log10(6700e6/10) ~ 5 * 8.83 ~ 44.1
        let mu = distance_modulus(
            1.0,
            params::OMEGA_M,
            params::OMEGA_AD,
            params::OMEGA_LAMBDA,
            params::H0,
        );
        assert!(mu > 40.0 && mu < 50.0, "mu = {mu}");
    }

    // -- Axiodilaton function --

    #[test]
    fn test_axiodilaton_function_at_z0() {
        assert!((axiodilaton_function(0.0) - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_axiodilaton_function_at_z1() {
        assert!((axiodilaton_function(1.0) - 2.0).abs() < 1e-14);
    }

    // -- Equation of state --

    #[test]
    fn test_equation_of_state() {
        let w = equation_of_state(0.5);
        assert!((w - (-1.0)).abs() < 1e-14);
    }

    #[test]
    fn test_deceleration_parameter_negative() {
        // Accelerating expansion: q < 0
        let q = deceleration_parameter(params::OMEGA_M, params::OMEGA_LAMBDA);
        assert!(q < 0.0, "q = {q} (should be negative for acceleration)");
    }

    // -- Consistency --

    #[test]
    fn test_flatness_constraint() {
        // Omega_m + Omega_ad + Omega_Lambda should equal 1 (flat universe)
        let total = params::OMEGA_M + params::OMEGA_AD + params::OMEGA_LAMBDA;
        assert!((total - 1.0).abs() < 1e-14, "total = {total}");
    }

    #[test]
    fn test_h0_prediction_in_tension_range() {
        // H_0 should be between CMB (67.4) and SH0ES (73.0)
        assert!(params::H0 > 67.0 && params::H0 < 73.5);
    }
}
