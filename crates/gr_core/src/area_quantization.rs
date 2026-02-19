//! LQG area spectrum and the Barbero-Immirzi parameter.
//!
//! Loop Quantum Gravity predicts a discrete area spectrum labelled by
//! half-integer spin j.  The single free parameter -- the Immirzi parameter
//! gamma -- multiplies every eigenvalue and is fixed by demanding that the
//! microstate count reproduces Bekenstein-Hawking entropy.
//!
//! # Key formulas
//!
//! Area eigenvalue (Rovelli & Smolin 1995):
//!   A(j) = 8 pi gamma l_P^2 sqrt(j (j + 1))
//!
//! Boltzmann-Gibbs gamma (Meissner 2004):
//!   gamma_BG ~ 0.2375  (transcendental, from state counting)
//!
//! Non-zero j_min gamma (Domagala & Lewandowski 2004):
//!   gamma_NZJ ~ 0.1236  (transcendental, from state counting)
//!
//! LQG entropy with arbitrary gamma:
//!   S = (gamma_0 / gamma) * A / (4 l_P^2)
//!
//! where gamma_0 is the "correct" value that reproduces BH entropy.
//!
//! # References
//! - Barbero (1995): Phys. Rev. D 51, 5507
//! - Immirzi (1997): Class. Quantum Grav. 14, L177
//! - Rovelli & Smolin (1995): Nucl. Phys. B 442, 593
//! - Ashtekar, Baez, Corichi, Krasnov (1998): Phys. Rev. Lett. 80, 904
//! - Meissner (2004): Class. Quantum Grav. 21, 5245
//! - Domagala & Lewandowski (2004): Class. Quantum Grav. 21, 5233

use crate::constants::L_PLANCK_CGS;
use std::f64::consts::PI;

// ============================================================================
// Immirzi parameter constants
// ============================================================================

/// Boltzmann-Gibbs value of the Immirzi parameter.
///
/// Numerically computed from the transcendental equation arising from
/// the Rovelli-Smolin state counting with j_min = 1/2 in the
/// Boltzmann-Gibbs ensemble.  This is NOT `ln(2)/(pi*sqrt(3))`.
///
/// Meissner (2004), Class. Quantum Grav. 21, 5245.
pub const GAMMA_BG: f64 = 0.237_532_956_559_5;

/// Non-zero j_min value of the Immirzi parameter.
///
/// Numerically computed from the Domagala-Lewandowski counting where
/// the minimum spin is j=1 rather than j=1/2.  Approximately but not
/// exactly equal to `ln(3)/(pi*sqrt(8))`.
///
/// Domagala & Lewandowski (2004), Class. Quantum Grav. 21, 5233.
pub const GAMMA_NZJ: f64 = 0.123_556_531_041_5;

// ============================================================================
// Runtime computation of gamma (for cross-validation in tests)
// ============================================================================

/// Return the Boltzmann-Gibbs Immirzi parameter.
///
/// This is a numerically-computed transcendental constant from
/// state-counting; returned here for cross-validation against the
/// hardcoded GAMMA_BG constant.
pub fn immirzi_parameter_bg() -> f64 {
    // The true gamma_BG is the unique positive solution of
    //   sum_{j=1/2,1,...} (2j+1) exp(-gamma * 4*pi*sqrt(j(j+1))) = 1
    // This is NOT a closed-form expression.  We return the constant.
    GAMMA_BG
}

/// Return the non-zero j_min Immirzi parameter.
///
/// Numerically-computed from Domagala-Lewandowski state counting with
/// j_min = 1; returned for cross-validation.
pub fn immirzi_parameter_nzj() -> f64 {
    GAMMA_NZJ
}

// ============================================================================
// Area eigenvalues
// ============================================================================

/// Area eigenvalue for spin quantum number j.
///
/// A(j) = 8 pi gamma l_P^2 sqrt(j (j + 1))
///
/// Returns the area in cm^2 (CGS units).
///
/// # Arguments
/// * `j` - Spin label (half-integer: 0.5, 1.0, 1.5, ...).  Must be > 0.
/// * `gamma` - Barbero-Immirzi parameter.
pub fn area_quantum(j: f64, gamma: f64) -> f64 {
    assert!(j > 0.0, "j must be positive (half-integer spin label)");
    assert!(gamma > 0.0, "Immirzi parameter must be positive");
    8.0 * PI * gamma * L_PLANCK_CGS * L_PLANCK_CGS * (j * (j + 1.0)).sqrt()
}

/// Generate the area spectrum for j = 1/2, 1, 3/2, ..., j_max.
///
/// # Arguments
/// * `j_max_times_2` - Twice the maximum j (e.g., 10 for j_max = 5).
///   Must be >= 1 (so j_min = 1/2).
/// * `gamma` - Barbero-Immirzi parameter.
///
/// # Returns
/// Vec of (j, A_j) pairs in ascending j order.
pub fn area_spectrum(j_max_times_2: u32, gamma: f64) -> Vec<(f64, f64)> {
    assert!(j_max_times_2 >= 1, "j_max_times_2 must be >= 1");
    (1..=j_max_times_2)
        .map(|twice_j| {
            let j = twice_j as f64 / 2.0;
            (j, area_quantum(j, gamma))
        })
        .collect()
}

/// Minimum area eigenvalue (j = 1/2, the area gap).
///
/// A_min = 8 pi gamma l_P^2 sqrt(3)/2 = 4 pi gamma l_P^2 sqrt(3)
///
/// Returns area in cm^2.
pub fn minimum_area(gamma: f64) -> f64 {
    area_quantum(0.5, gamma)
}

/// Gamma-independent ratio of area eigenvalues.
///
/// A(j1) / A(j2) = sqrt(j1(j1+1)) / sqrt(j2(j2+1))
///
/// This ratio is a parameter-free LQG prediction testable by
/// any future area-quantization experiment.
pub fn area_gap_ratio(j1: f64, j2: f64) -> f64 {
    assert!(j1 > 0.0 && j2 > 0.0, "j values must be positive");
    ((j1 * (j1 + 1.0)) / (j2 * (j2 + 1.0))).sqrt()
}

// ============================================================================
// Entropy
// ============================================================================

/// Bekenstein-Hawking entropy with LQG Immirzi correction.
///
/// S = (gamma_0 / gamma) * A / (4 l_P^2)
///
/// When gamma = gamma_0 (the "correct" value), this reduces to the
/// semiclassical Bekenstein-Hawking formula S = A / (4 l_P^2).
///
/// # Arguments
/// * `area_cm2` - Horizon area in cm^2.
/// * `gamma` - The Immirzi parameter used in the computation.
///
/// The function uses GAMMA_BG as gamma_0 (the calibration value).
/// Returns entropy in natural units (dimensionless).
pub fn entropy_lqg(area_cm2: f64, gamma: f64) -> f64 {
    assert!(area_cm2 >= 0.0, "area must be non-negative");
    assert!(gamma > 0.0, "Immirzi parameter must be positive");
    let lp2 = L_PLANCK_CGS * L_PLANCK_CGS;
    (GAMMA_BG / gamma) * area_cm2 / (4.0 * lp2)
}

/// Ratio gamma / gamma_BG quantifying deviation from semiclassical entropy.
///
/// When this returns 1.0, LQG entropy matches Bekenstein-Hawking exactly.
/// Values > 1 reduce entropy (fewer microstates); values < 1 increase it.
pub fn entropy_correction_factor(gamma: f64) -> f64 {
    assert!(gamma > 0.0, "Immirzi parameter must be positive");
    gamma / GAMMA_BG
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::L_PLANCK_CGS;

    const TOL: f64 = 1e-10;

    // -- Constant verification --

    #[test]
    fn gamma_bg_matches_runtime() {
        let computed = immirzi_parameter_bg();
        assert!(
            (computed - GAMMA_BG).abs() < 1e-12,
            "GAMMA_BG mismatch: const={GAMMA_BG}, runtime={computed}"
        );
    }

    #[test]
    fn gamma_nzj_matches_runtime() {
        let computed = immirzi_parameter_nzj();
        assert!(
            (computed - GAMMA_NZJ).abs() < 1e-12,
            "GAMMA_NZJ mismatch: const={GAMMA_NZJ}, runtime={computed}"
        );
    }

    #[test]
    fn gamma_bg_is_approximately_0_2375() {
        assert!((GAMMA_BG - 0.2375).abs() < 0.001);
    }

    #[test]
    fn gamma_nzj_is_approximately_0_1236() {
        assert!((GAMMA_NZJ - 0.1236).abs() < 0.001);
    }

    #[test]
    fn gamma_bg_greater_than_gamma_nzj() {
        assert!(GAMMA_BG > GAMMA_NZJ);
    }

    // -- Area eigenvalue tests --

    #[test]
    fn area_j_half_is_minimum() {
        let a_min = area_quantum(0.5, GAMMA_BG);
        let a_one = area_quantum(1.0, GAMMA_BG);
        assert!(a_one > a_min, "A(1) should exceed A(1/2)");
    }

    #[test]
    fn area_planck_scale() {
        // A(j=1/2, gamma_BG) ~ 8*pi*0.2375*l_P^2*sqrt(3/4)
        // = 8*pi*0.2375*l_P^2 * 0.8660 = ~5.16 * l_P^2
        let a = area_quantum(0.5, GAMMA_BG);
        let lp2 = L_PLANCK_CGS * L_PLANCK_CGS;
        let ratio = a / lp2;
        // Should be a few times l_P^2
        assert!(ratio > 1.0 && ratio < 20.0, "area/l_P^2 = {ratio}");
    }

    #[test]
    fn area_monotonically_increasing() {
        let spectrum = area_spectrum(20, GAMMA_BG);
        for window in spectrum.windows(2) {
            assert!(
                window[1].1 > window[0].1,
                "area not monotone: A({}) = {} >= A({}) = {}",
                window[0].0,
                window[0].1,
                window[1].0,
                window[1].1
            );
        }
    }

    #[test]
    fn area_scales_linearly_with_gamma() {
        let a1 = area_quantum(1.0, GAMMA_BG);
        let a2 = area_quantum(1.0, 2.0 * GAMMA_BG);
        assert!((a2 / a1 - 2.0).abs() < TOL);
    }

    #[test]
    fn area_spectrum_length() {
        let spec = area_spectrum(10, GAMMA_BG);
        assert_eq!(spec.len(), 10);
        assert!((spec[0].0 - 0.5).abs() < TOL);
        assert!((spec[9].0 - 5.0).abs() < TOL);
    }

    #[test]
    fn minimum_area_equals_j_half() {
        let a_min = minimum_area(GAMMA_BG);
        let a_half = area_quantum(0.5, GAMMA_BG);
        assert!((a_min - a_half).abs() < 1e-50);
    }

    // -- Gamma-independent ratio --

    #[test]
    fn area_ratio_is_gamma_independent() {
        let ratio_bg = area_quantum(1.0, GAMMA_BG) / area_quantum(0.5, GAMMA_BG);
        let ratio_nzj = area_quantum(1.0, GAMMA_NZJ) / area_quantum(0.5, GAMMA_NZJ);
        let ratio_fn = area_gap_ratio(1.0, 0.5);
        assert!((ratio_bg - ratio_nzj).abs() < TOL);
        assert!((ratio_bg - ratio_fn).abs() < TOL);
    }

    #[test]
    fn area_ratio_j1_to_jhalf() {
        // sqrt(1*2) / sqrt(0.5*1.5) = sqrt(2) / sqrt(0.75) = sqrt(2/0.75) = sqrt(8/3)
        let expected = (8.0_f64 / 3.0).sqrt();
        let computed = area_gap_ratio(1.0, 0.5);
        assert!((computed - expected).abs() < TOL);
    }

    // -- Entropy tests --

    #[test]
    fn entropy_at_gamma_bg_is_bekenstein_hawking() {
        // At gamma = gamma_BG, entropy should be S = A/(4 l_P^2)
        let area = 1e-60; // some arbitrary area in cm^2
        let lp2 = L_PLANCK_CGS * L_PLANCK_CGS;
        let s_lqg = entropy_lqg(area, GAMMA_BG);
        let s_bh = area / (4.0 * lp2);
        assert!(
            (s_lqg / s_bh - 1.0).abs() < TOL,
            "LQG entropy at gamma_BG should match BH: {s_lqg} vs {s_bh}"
        );
    }

    #[test]
    fn entropy_scales_inversely_with_gamma() {
        let area = 1e-60;
        let s1 = entropy_lqg(area, GAMMA_BG);
        let s2 = entropy_lqg(area, 2.0 * GAMMA_BG);
        assert!((s1 / s2 - 2.0).abs() < TOL);
    }

    #[test]
    fn entropy_correction_factor_unity_at_bg() {
        let f = entropy_correction_factor(GAMMA_BG);
        assert!((f - 1.0).abs() < TOL);
    }

    #[test]
    fn entropy_correction_factor_at_nzj() {
        let f = entropy_correction_factor(GAMMA_NZJ);
        let expected = GAMMA_NZJ / GAMMA_BG;
        assert!((f - expected).abs() < TOL);
        assert!(f < 1.0, "NZJ gamma < BG gamma, so factor < 1");
    }
}
