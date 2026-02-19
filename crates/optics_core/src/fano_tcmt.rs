//! Temporal Coupled-Mode Theory for Fano resonance in scattering.
//!
//! Implements the TCMT formalism from Ruan & Fan (2009), "Temporal coupled-mode
//! theory for Fano resonance in light scattering by a single obstacle"
//! (arXiv:0909.3323v2). Covers:
//! - Single-channel and multi-channel Fano lineshapes
//! - Lossless and lossy cross-sections (scattering, absorption, extinction)
//! - Drude permittivity model
//!
//! # Key Equations
//!
//! Reflection coefficient (Eq. 21):
//! ```text
//!   R_l = e^{i*phi} * [i*(w0-w) + g0 - g] / [i*(w0-w) + g0 + g]
//! ```
//!
//! Cross-sections (Eq. 23):
//! ```text
//!   C_sct = (2*lam/pi) * sum_l |S_l|^2
//!   C_abs = -(2*lam/pi) * sum_l Re(S_l)    (where S_l = (R_l - 1)/2)
//! ```
//!
//! # References
//! - Ruan & Fan, J. Phys. Chem. C 114, 7324 (2010) / arXiv:0909.3323v2

use num_complex::Complex64;

/// Parameters for a single scattering channel in the TCMT formalism.
#[derive(Debug, Clone, Copy)]
pub struct FanoChannel {
    /// Resonance angular frequency omega_0.
    pub omega_0: f64,
    /// Radiative decay rate gamma (coupling to radiation).
    pub gamma: f64,
    /// Non-radiative (absorptive) decay rate gamma_0.
    pub gamma_0: f64,
    /// Background scattering phase for this channel.
    pub phi: f64,
    /// Angular momentum channel index l.
    pub l: i32,
}

/// Drude permittivity model parameters.
#[derive(Debug, Clone, Copy)]
pub struct FanoDrudeParams {
    /// Plasma frequency omega_p.
    pub omega_p: f64,
    /// Damping rate gamma_d.
    pub gamma_d: f64,
}

/// Scattering cross-section results.
#[derive(Debug, Clone, Copy)]
pub struct CrossSections {
    /// Scattering cross-section.
    pub c_sct: f64,
    /// Absorption cross-section.
    pub c_abs: f64,
    /// Extinction cross-section (= c_sct + c_abs).
    pub c_ext: f64,
}

/// Reflection coefficient R_l for a single channel (Eq. 21).
///
/// R_l = e^{i*phi} * [i*(omega_0 - omega) + gamma_0 - gamma]
///                   / [i*(omega_0 - omega) + gamma_0 + gamma]
pub fn fano_reflection(ch: &FanoChannel, omega: f64) -> Complex64 {
    let delta = ch.omega_0 - omega;
    let numerator = Complex64::new(ch.gamma_0 - ch.gamma, delta);
    let denominator = Complex64::new(ch.gamma_0 + ch.gamma, delta);
    let phase = Complex64::new(0.0, ch.phi).exp();
    phase * numerator / denominator
}

/// Scattering coefficient S_l = (R_l - 1) / 2 (Eq. 8).
pub fn scattering_coefficient(ch: &FanoChannel, omega: f64) -> Complex64 {
    (fano_reflection(ch, omega) - 1.0) / 2.0
}

/// Single-channel cross-sections in units of (2*lambda/pi).
///
/// Returns (C_sct, C_abs, C_ext) each divided by (2*lambda/pi).
pub fn fano_cross_sections_normalized(ch: &FanoChannel, omega: f64) -> CrossSections {
    let s = scattering_coefficient(ch, omega);
    let c_sct = s.norm_sqr();
    let c_abs = -s.re;
    CrossSections {
        c_sct,
        c_abs,
        c_ext: c_sct + c_abs,
    }
}

/// Multi-channel cross-sections (sum over channels), normalized by (2*lambda/pi).
pub fn multi_channel_cross_sections(channels: &[FanoChannel], omega: f64) -> CrossSections {
    let mut c_sct = 0.0;
    let mut c_abs = 0.0;
    for ch in channels {
        let s = scattering_coefficient(ch, omega);
        c_sct += s.norm_sqr();
        c_abs += -s.re;
    }
    CrossSections {
        c_sct,
        c_abs,
        c_ext: c_sct + c_abs,
    }
}

/// Normalized scattering cross-section for pure analytical Fano lineshape.
///
/// This is for Figs 1-2 of Ruan & Fan where x = (omega - omega_0)/gamma
/// and the result is C_sct in units of (2*lambda/pi).
///
/// For lossless case (gamma_0 = 0):
///   C_sct = |S|^2 = |(R-1)/2|^2
///
/// For lossy case (gamma_0 > 0):
///   Same formula with absorption included.
pub fn normalized_fano_c_sct(x: f64, phi: f64, gamma_0_over_gamma: f64) -> f64 {
    let ch = FanoChannel {
        omega_0: 0.0,
        gamma: 1.0,
        gamma_0: gamma_0_over_gamma,
        phi,
        l: 0,
    };
    // omega = omega_0 + x*gamma = 0 + x*1 = x (since gamma=1)
    // But omega_0 - omega = -x, so delta = -x
    fano_cross_sections_normalized(&ch, x).c_sct
}

/// Normalized absorption cross-section for pure analytical Fano lineshape.
pub fn normalized_fano_c_abs(x: f64, phi: f64, gamma_0_over_gamma: f64) -> f64 {
    let ch = FanoChannel {
        omega_0: 0.0,
        gamma: 1.0,
        gamma_0: gamma_0_over_gamma,
        phi,
        l: 0,
    };
    fano_cross_sections_normalized(&ch, x).c_abs
}

/// Normalized extinction cross-section for pure analytical Fano lineshape.
pub fn normalized_fano_c_ext(x: f64, phi: f64, gamma_0_over_gamma: f64) -> f64 {
    let ch = FanoChannel {
        omega_0: 0.0,
        gamma: 1.0,
        gamma_0: gamma_0_over_gamma,
        phi,
        l: 0,
    };
    fano_cross_sections_normalized(&ch, x).c_ext
}

/// Drude permittivity: eps(omega) = 1 - omega_p^2 / (omega^2 + i*gamma_d*omega).
pub fn drude_epsilon(params: &FanoDrudeParams, omega: f64) -> Complex64 {
    let omega_c = Complex64::new(omega, 0.0);
    let denom = omega_c * omega_c + Complex64::new(0.0, params.gamma_d * omega);
    Complex64::new(1.0, 0.0) - params.omega_p * params.omega_p / denom
}

/// Fano asymmetry parameter q = -cot(phi/2) (Eq. 15).
pub fn fano_q(phi: f64) -> f64 {
    -(phi / 2.0).cos() / (phi / 2.0).sin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn lossless_phi_zero_lorentzian() {
        // phi=0, gamma_0=0: symmetric Lorentzian peak
        // At x=0: R = e^0 * (0-1)/(0+1) = -1 => S = (-1-1)/2 = -1 => |S|^2 = 1
        let c_at_resonance = normalized_fano_c_sct(0.0, 0.0, 0.0);
        assert!(
            (c_at_resonance - 1.0).abs() < 1e-12,
            "C_sct at resonance should be 1.0 for lossless phi=0, got {}",
            c_at_resonance
        );

        // Far from resonance: should decay to 0
        let c_far = normalized_fano_c_sct(100.0, 0.0, 0.0);
        assert!(
            c_far < 1e-4,
            "C_sct far from resonance should be near 0, got {}",
            c_far
        );
    }

    #[test]
    fn lossless_phi_pi_antiresonance() {
        // phi=pi, gamma_0=0: C_sct = 0 at omega_0
        // At x=0: R = e^{i*pi} * (-1)/(1) = (-1)(-1) = 1 => S = (1-1)/2 = 0
        let c_at_resonance = normalized_fano_c_sct(0.0, PI, 0.0);
        assert!(
            c_at_resonance < 1e-12,
            "C_sct at resonance should be 0 for phi=pi, got {}",
            c_at_resonance
        );
    }

    #[test]
    fn absorption_lorentzian() {
        // C_abs = -Re(S_l) is symmetric in (omega - omega_0) ONLY for phi = 0.
        // For phi = 0: S = -gamma / (i*delta + gamma_0 + gamma), so
        //   -Re(S) = gamma*(gamma_0+gamma) / [delta^2 + (gamma_0+gamma)^2]
        // which is a symmetric Lorentzian.
        // For phi != 0, the exp(i*phi) factor breaks symmetry (Fano interference).
        let c_pos = normalized_fano_c_abs(3.0, 0.0, 1.0);
        let c_neg = normalized_fano_c_abs(-3.0, 0.0, 1.0);
        assert!(
            (c_pos - c_neg).abs() < 1e-12,
            "C_abs should be symmetric at phi=0: pos={}, neg={}",
            c_pos,
            c_neg,
        );
        // At resonance (x=0, phi=0, gamma_0=gamma=1):
        // C_abs = 1*2/(0+4) = 0.5
        let c_res = normalized_fano_c_abs(0.0, 0.0, 1.0);
        assert!(
            (c_res - 0.5).abs() < 1e-12,
            "C_abs at resonance (equal damping) should be 0.5, got {}",
            c_res,
        );
        // For phi != 0, C_abs is asymmetric (Fano effect on absorption)
        let c_pos_pi2 = normalized_fano_c_abs(3.0, PI / 2.0, 1.0);
        let c_neg_pi2 = normalized_fano_c_abs(-3.0, PI / 2.0, 1.0);
        assert!(
            (c_pos_pi2 - c_neg_pi2).abs() > 0.01,
            "C_abs should be asymmetric for phi=pi/2"
        );
    }

    #[test]
    fn optical_theorem() {
        // C_ext = C_sct + C_abs at every frequency
        for x in [-5.0, -1.0, 0.0, 0.5, 3.0] {
            for phi in [0.0, PI / 4.0, PI / 2.0, PI] {
                let cs = fano_cross_sections_normalized(
                    &FanoChannel {
                        omega_0: 0.0,
                        gamma: 1.0,
                        gamma_0: 0.5,
                        phi,
                        l: 0,
                    },
                    x,
                );
                assert!(
                    (cs.c_ext - cs.c_sct - cs.c_abs).abs() < 1e-14,
                    "Optical theorem violated at x={}, phi={}",
                    x,
                    phi
                );
            }
        }
    }

    #[test]
    fn background_limit() {
        // |omega - omega_0| >> gamma => R -> e^{i*phi}
        let ch = FanoChannel {
            omega_0: 10.0,
            gamma: 0.01,
            gamma_0: 0.005,
            phi: PI / 3.0,
            l: 0,
        };
        // At 10000*gamma away from resonance, R -> e^{i*phi} to O(gamma/delta)
        let r = fano_reflection(&ch, 10.0 + 10000.0 * ch.gamma);
        let expected = Complex64::new(0.0, ch.phi).exp();
        let diff = (r - expected).norm();
        assert!(
            diff < 1e-3,
            "R should approach e^{{i*phi}} far from resonance, diff={}",
            diff
        );
    }

    #[test]
    fn fano_parameter_values() {
        // q = -cot(phi/2); for phi=pi/2: q = -cos(pi/4)/sin(pi/4) = -1
        let q = fano_q(PI / 2.0);
        assert!(
            (q - (-1.0)).abs() < 1e-12,
            "q for phi=pi/2 should be -1, got {}",
            q
        );

        // phi=pi: q = -cos(pi/2)/sin(pi/2) = 0
        let q_pi = fano_q(PI);
        assert!(
            q_pi.abs() < 1e-12,
            "q for phi=pi should be 0, got {}",
            q_pi
        );
    }
}
