//! Photon-graviton mixing scope limits (Ahmadiniaz et al. 2026).
//!
//! One-loop photon-graviton conversion in a constant EM field produces three
//! diagrams (irreducible, tadpole, vacuum-polarization subdiagram). The mixing
//! amplitude scales as kappa * (alpha/pi) * (B/B_cr)^2, which is unmeasurably
//! small for laboratory fields.
//!
//! This module uses SI units (Tesla, kg, m/s) following the QED literature
//! convention. See `constants.rs` for CGS astrophysical constants.
//!
//! # References
//! - Ahmadiniaz, N. et al. (2026), arXiv:2601.23279
//! - Rodal, J. (2025), "Metamaterial Gravitational Coupling"

use std::f64::consts::PI;

// -- SI fundamental constants --

/// Electron mass [kg]
const M_E_SI: f64 = 9.109_383_701_5e-31;

/// Speed of light [m/s]
const C_SI: f64 = 2.997_924_58e8;

/// Elementary charge [C]
const E_SI: f64 = 1.602_176_634e-19;

/// Reduced Planck constant [J*s]
const HBAR_SI: f64 = 1.054_571_817e-34;

/// Gravitational constant [m^3 kg^-1 s^-2]
const G_SI: f64 = 6.674_30e-11;

/// Fine-structure constant (dimensionless)
const ALPHA_EM: f64 = 1.0 / 137.036;

/// Schwinger critical magnetic field B_cr = m_e^2 c^2 / (e hbar) [T].
///
/// Above this field strength, vacuum pair production becomes significant.
/// B_cr ~ 4.41e9 T.
pub fn schwinger_critical_field() -> f64 {
    M_E_SI * M_E_SI * C_SI * C_SI / (E_SI * HBAR_SI)
}

/// Gravitational coupling constant kappa = sqrt(16 pi G / c^4) [SI].
///
/// This sets the scale of gravitational interactions in linearized GR.
/// kappa ~ 2.04e-22 in SI units.
pub fn gravitational_coupling() -> f64 {
    (16.0 * PI * G_SI / (C_SI * C_SI * C_SI * C_SI)).sqrt()
}

/// One-loop photon-graviton mixing amplitude estimate.
///
/// The dimensionless amplitude ratio is:
///   A ~ (alpha/pi) * (B_lab / B_cr)^2
///
/// This is the suppression factor for photon-graviton conversion in
/// a laboratory magnetic field. For B_lab = 10 T, the ratio is ~ 1.2e-20,
/// making the effect unmeasurable.
///
/// Returns `(b_cr, kappa, amplitude_ratio)`.
pub fn mixing_amplitude_estimate(b_lab: f64) -> (f64, f64, f64) {
    let b_cr = schwinger_critical_field();
    let kappa = gravitational_coupling();
    let ratio = (ALPHA_EM / PI) * (b_lab / b_cr).powi(2);
    (b_cr, kappa, ratio)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schwinger_critical_field() {
        let b_cr = schwinger_critical_field();
        // B_cr ~ 4.41e9 T (within 1%)
        assert!(
            b_cr > 4.4e9 && b_cr < 4.5e9,
            "B_cr = {b_cr:.3e}, expected ~4.41e9 T"
        );
    }

    #[test]
    fn test_gravitational_coupling_tiny() {
        let kappa = gravitational_coupling();
        assert!(kappa > 0.0, "kappa must be positive");
        assert!(kappa < 1e-20, "kappa = {kappa:.3e}, expected << 1");
    }

    #[test]
    fn test_mixing_amplitude_negligible() {
        let (_b_cr, _kappa, ratio) = mixing_amplitude_estimate(10.0);
        // (alpha/pi) * (10/4.4e9)^2 ~ 2.3e-3 * 5.2e-18 ~ 1.2e-20
        assert!(
            ratio < 1e-15,
            "Amplitude ratio {ratio:.3e} unexpectedly large"
        );
    }

    #[test]
    fn test_mixing_amplitude_zero_field() {
        let (_b_cr, _kappa, ratio) = mixing_amplitude_estimate(0.0);
        assert!(ratio == 0.0, "Zero field should give zero mixing amplitude");
    }

    #[test]
    fn test_mixing_amplitude_scales_quadratically() {
        let (_, _, r1) = mixing_amplitude_estimate(1.0);
        let (_, _, r10) = mixing_amplitude_estimate(10.0);
        let scale = r10 / r1;
        assert!(
            (scale - 100.0).abs() < 1e-10,
            "Amplitude should scale as B^2: ratio = {scale}"
        );
    }
}
