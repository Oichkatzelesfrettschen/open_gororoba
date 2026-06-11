//! Fano lineshape and asymmetry parameter calculations for photon-graviton TCMT
//!
//! Implements the Fano lineshape formalism from Ruan & Fan (2009) Eqs. 14-23,
//! extended to include gravitational coupling effects.

use crate::photon_graviton_tcmt::{AsymmetryParameter, CrossSections, GravitationalCoupling};
use num_complex::Complex64 as C64;

/// Compute the Fano lineshape for normalized detuning parameter
///
/// Based on Ruan-Fan Eq. 23:
/// ```text
/// sigma = |S|^2 = |(x + iq) / (x + i)|^2
///       = (x^2 + q^2) / (x^2 + 1) * correction_factor
/// ```
///
/// where:
/// - x = (\omega - \omega_0) / \gamma (normalized detuning)
/// - q = asymmetry parameter (cot(\phi/2))
/// - correction_factor accounts for gravitational coupling
///
/// # Arguments
/// * `x` - Normalized detuning (omega - omega_0) / gamma
/// * `asymmetry` - Asymmetry parameter q_grav
/// * `_weak_field_param` - Weak-field parameter epsilon (dimensionless)
///
/// # Returns
/// Normalized lineshape (dimensionless), range `[0, 1]` typical for lossless case
pub fn fano_lineshape(x: f64, asymmetry: &AsymmetryParameter, _weak_field_param: f64) -> f64 {
    let x_squared = x * x;
    let _q_squared = asymmetry.q * asymmetry.q;

    // Base Fano formula (Ruan-Fan Eq. 23)
    let numerator = x_squared + asymmetry.q * asymmetry.q;
    let denominator = (x_squared + 1.0).max(1e-15); // Avoid division by zero
    let base_lineshape = numerator / denominator;

    // Gravitational correction factor
    // In weak-field limit, gravitational contribution is O(epsilon)
    let grav_correction =
        1.0 + _weak_field_param * asymmetry.gravitational_factor * (x / (x_squared + 1.0));

    (base_lineshape * grav_correction).max(0.0) // Ensure non-negative
}

/// Compute scattering cross-section from Fano lineshape
///
/// Based on Ruan-Fan Eq. 14:
/// ```text
/// \sigma_sct = \sigma_0 * |S_l|^2 = \sigma_0 * fano_lineshape(x, q, epsilon)
/// ```
///
/// # Arguments
/// * `x` - Normalized detuning
/// * `asymmetry` - Asymmetry parameter
/// * `weak_field_param` - Weak-field parameter
/// * `cross_section_amplitude` - \sigma_0 normalization factor
pub fn scattering_cross_section(
    x: f64,
    asymmetry: &AsymmetryParameter,
    weak_field_param: f64,
    cross_section_amplitude: f64,
) -> f64 {
    let lineshape = fano_lineshape(x, asymmetry, weak_field_param);
    cross_section_amplitude * lineshape
}

/// Compute absorption cross-section
///
/// In lossless scattering (gamma_0 = 0), absorption vanishes.
/// With loss (gamma_0 > 0), absorption follows:
/// ```text
/// \sigma_abs = \sigma_0 * (gamma_0 / gamma) * lineshape_asymmetric(x, q)
/// ```
///
/// # Arguments
/// * `x` - Normalized detuning
/// * `asymmetry` - Asymmetry parameter
/// * `weak_field_param` - Weak-field parameter
/// * `cross_section_amplitude` - \sigma_0
/// * `loss_ratio` - \gamma_0 / \gamma (loss / decay ratio)
pub fn absorption_cross_section(
    x: f64,
    asymmetry: &AsymmetryParameter,
    _weak_field_param: f64,
    cross_section_amplitude: f64,
    loss_ratio: f64,
) -> f64 {
    let x_squared = x * x;
    let _q_squared = asymmetry.q * asymmetry.q;

    // Absorption lineshape: symmetric Lorentzian with asymmetric tail modulation
    let base_absorption = 1.0 / ((x_squared + 1.0).max(1e-15));
    let asymmetric_tail = 1.0 + (2.0 * asymmetry.q * x / (x_squared + 1.0));

    let absorption_factor = loss_ratio * base_absorption * asymmetric_tail;
    cross_section_amplitude * absorption_factor.max(0.0)
}

/// Compute extinction cross-section (= scattering + absorption)
pub fn extinction_cross_section(
    x: f64,
    asymmetry: &AsymmetryParameter,
    weak_field_param: f64,
    cross_section_amplitude: f64,
    loss_ratio: f64,
) -> f64 {
    let c_sct = scattering_cross_section(x, asymmetry, weak_field_param, cross_section_amplitude);
    let c_abs = absorption_cross_section(
        x,
        asymmetry,
        weak_field_param,
        cross_section_amplitude,
        loss_ratio,
    );
    c_sct + c_abs
}

/// Compute all three cross-sections as CrossSections struct
pub fn compute_cross_sections(
    x: f64,
    asymmetry: &AsymmetryParameter,
    weak_field_param: f64,
    cross_section_amplitude: f64,
    loss_ratio: f64,
) -> CrossSections {
    let c_sct = scattering_cross_section(x, asymmetry, weak_field_param, cross_section_amplitude);
    let c_abs = absorption_cross_section(
        x,
        asymmetry,
        weak_field_param,
        cross_section_amplitude,
        loss_ratio,
    );
    CrossSections::new(c_sct, c_abs)
}

/// Compute asymmetry parameter from coupling parameters
///
/// Maps worldline amplitude parameters to TCMT asymmetry parameter.
/// Following Ruan-Fan Eq. 21 with gravitational correction.
pub fn compute_asymmetry_from_coupling(coupling: &GravitationalCoupling) -> AsymmetryParameter {
    AsymmetryParameter::from_coupling(coupling)
}

/// Compute reflected wave amplitude from incident wave
///
/// Ruan-Fan Eq. 21:
/// ```text
/// R = e^{i*phi} * [i*(\omega_0 - \omega) + \gamma_0 - \gamma] / [i*(\omega_0 - \omega) + \gamma_0 + \gamma]
/// ```
pub fn reflection_coefficient(frequency: f64, coupling: &GravitationalCoupling) -> C64 {
    let detuning = C64::new(0.0, coupling.resonance_frequency - frequency);
    let numerator =
        detuning + C64::new(coupling.gamma_nonradiative - coupling.gamma_radiative, 0.0);
    let denominator =
        detuning + C64::new(coupling.gamma_nonradiative + coupling.gamma_radiative, 0.0);

    let phase = C64::from_polar(1.0, coupling.coupling_phase);
    // Avoid division by zero with explicit check
    let safe_denominator = if denominator.norm() > 1e-15 {
        denominator
    } else {
        C64::new(1e-15, 1e-15)
    };
    phase * numerator / safe_denominator
}

/// Compute scattering coefficient from reflection coefficient
///
/// Ruan-Fan Eq. 8:
/// ```text
/// S = (R - 1) / 2
/// ```
pub fn scattering_coefficient(reflection: C64) -> C64 {
    (reflection - 1.0) / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fano_lineshape_at_resonance() {
        let asym = AsymmetryParameter::from_q(1.0);
        let lineshape_at_resonance = fano_lineshape(0.0, &asym, 0.01);
        // At resonance (x=0) with q=1: lineshape = 1 / 1 = 1
        assert!((lineshape_at_resonance - 1.0).abs() < 0.05); // Allow gravitational correction
    }

    #[test]
    fn fano_lineshape_symmetry_limit() {
        // As q -> 0, Fano becomes symmetric Lorentzian
        let asym = AsymmetryParameter::from_q(0.1);
        let shape_pos = fano_lineshape(1.0, &asym, 0.0);
        let shape_neg = fano_lineshape(-1.0, &asym, 0.0);
        // Symmetric part should be roughly equal
        assert!(((shape_pos - shape_neg).abs() / shape_pos.max(shape_neg)) < 0.15);
    }

    #[test]
    fn absorption_zero_loss_ratio() {
        let asym = AsymmetryParameter::from_q(1.0);
        let absorption = absorption_cross_section(0.0, &asym, 0.01, 1.0, 0.0);
        assert!(absorption.abs() < 1e-10); // Zero with zero loss
    }

    #[test]
    fn optical_theorem_with_loss() {
        let coupling = GravitationalCoupling::new(1e-3, 1e-6, 1e-7, 1e-8, 1e15, 0.5);
        let asym = compute_asymmetry_from_coupling(&coupling);
        let weak_field = coupling.weak_field_parameter();
        let loss_ratio = coupling.gamma_nonradiative / coupling.gamma_radiative;

        let cs = compute_cross_sections(0.0, &asym, weak_field, 1.0, loss_ratio);
        assert!(cs.verify_optical_theorem(1e-8));
    }

    #[test]
    fn scattering_coefficient_from_reflection() {
        let r = C64::new(1.0, 0.0);
        let s = scattering_coefficient(r);
        // S = (1 - 1) / 2 = 0
        assert!(s.norm_sqr() < 1e-10);
    }
}
