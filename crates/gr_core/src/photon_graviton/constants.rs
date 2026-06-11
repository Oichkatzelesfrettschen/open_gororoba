//! QED and gravitational constants in natural units.
//!
//! All computations in this module use natural units where hbar = c = 1 and
//! m_e = 1 (electron mass sets the scale). The key consequence:
//!
//! - Energies, momenta, masses are dimensionless multiples of m_e
//! - Lengths and times are in units of 1/m_e (Compton wavelength)
//! - The fine-structure constant alpha = e^2/(4 pi) is the expansion parameter
//! - B_critical = m_e^2/e = 1/(e_natural) in these units
//!
//! SI conversion factors are provided for final output.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Fundamental dimensionless constants
// ---------------------------------------------------------------------------

/// Fine-structure constant alpha = e^2 / (4 pi) ~ 1/137.036.
pub const ALPHA_EM: f64 = 1.0 / 137.035_999_084;

/// Elementary charge in natural units: e = sqrt(4 pi alpha).
pub const E_NATURAL: f64 = 0.302_822_120_872; // sqrt(4*pi/137.036)

/// Schwinger critical field in natural units: B_cr = m^2/e = 1/e_natural.
///
/// In SI: B_cr ~ 4.414e9 T. In natural units this is simply 1/e.
pub const B_CR_NATURAL: f64 = 3.302_389_4; // 1/E_NATURAL

/// Gravitational coupling kappa = sqrt(16 pi G) in natural units.
///
/// kappa = sqrt(16 pi) * m_e / M_Planck.
/// M_Planck / m_e ~ 2.389e22, so kappa ~ sqrt(16*pi) / 2.389e22 ~ 2.96e-22.
pub const KAPPA_NATURAL: f64 = 2.963e-22;

/// Prefactor for one-loop amplitude: kappa * alpha / (8 pi^2).
///
/// This appears in the overall normalization of the irreducible diagram.
pub const ONE_LOOP_PREFACTOR: f64 = KAPPA_NATURAL * ALPHA_EM / (8.0 * PI * PI);

// ---------------------------------------------------------------------------
// SI fundamental constants (for conversion to/from natural units)
// ---------------------------------------------------------------------------

/// Electron mass `[kg]`.
pub const M_E_SI: f64 = 9.109_383_701_5e-31;

/// Speed of light `[m/s]`.
pub const C_SI: f64 = 2.997_924_58e8;

/// Elementary charge `[C]`.
pub const E_SI: f64 = 1.602_176_634e-19;

/// Reduced Planck constant `[J*s]`.
pub const HBAR_SI: f64 = 1.054_571_817e-34;

/// Gravitational constant [m^3 kg^-1 s^-2].
pub const G_SI: f64 = 6.674_30e-11;

/// Schwinger critical field in SI `[T]`.
pub const B_CR_SI: f64 = 4.414e9;

/// Planck mass `[kg]`.
pub const M_PLANCK_SI: f64 = 2.176_434e-8;

// ---------------------------------------------------------------------------
// Conversion utilities
// ---------------------------------------------------------------------------

/// Convert a magnetic field from Tesla to natural units (B / B_cr).
pub fn tesla_to_natural(b_tesla: f64) -> f64 {
    b_tesla / B_CR_SI
}

/// Convert a photon energy from eV to natural units (omega / m_e c^2).
pub fn ev_to_natural(energy_ev: f64) -> f64 {
    let m_e_ev = 0.510_998_950e6; // m_e c^2 in eV
    energy_ev / m_e_ev
}

/// Convert a cross section from natural units (1/m_e^2) to cm^2.
pub fn cross_section_to_cm2(sigma_natural: f64) -> f64 {
    let compton_cm = HBAR_SI / (M_E_SI * C_SI) * 100.0; // ~3.86e-11 cm
    sigma_natural * compton_cm * compton_cm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alpha_em() {
        assert!((ALPHA_EM - 1.0 / 137.036).abs() < 1e-6);
    }

    #[test]
    fn test_e_natural_squared() {
        // e^2 = 4*pi*alpha
        let e_sq = E_NATURAL * E_NATURAL;
        let expected = 4.0 * PI * ALPHA_EM;
        assert!(
            (e_sq - expected).abs() / expected < 1e-6,
            "e^2 = {e_sq}, expected {expected}"
        );
    }

    #[test]
    fn test_b_cr_natural() {
        // B_cr = 1/e in natural units
        let expected = 1.0 / E_NATURAL;
        assert!(
            (B_CR_NATURAL - expected).abs() / expected < 1e-4,
            "B_cr = {B_CR_NATURAL}, expected {expected}"
        );
    }

    #[test]
    fn test_kappa_tiny() {
        let kappa = KAPPA_NATURAL;
        assert!(kappa > 0.0);
        assert!(kappa < 1e-20, "kappa = {kappa}");
    }

    #[test]
    fn test_tesla_to_natural() {
        let b_nat = tesla_to_natural(4.414e9);
        assert!(
            (b_nat - 1.0).abs() < 0.01,
            "B_cr should map to ~1 in natural units, got {b_nat}"
        );
    }

    #[test]
    fn test_ev_to_natural() {
        // 0.511 MeV photon should be omega/m ~ 1
        let omega = ev_to_natural(0.511e6);
        assert!(
            (omega - 1.0).abs() < 0.01,
            "m_e c^2 photon should give omega/m ~ 1, got {omega}"
        );
    }

    #[test]
    fn test_cross_section_conversion() {
        // sigma = 1 (natural) ~ lambda_C^2 ~ 1.49e-21 cm^2
        let sigma_cm2 = cross_section_to_cm2(1.0);
        assert!(
            sigma_cm2 > 1e-22 && sigma_cm2 < 1e-20,
            "sigma = {sigma_cm2:.3e} cm^2"
        );
    }
}
