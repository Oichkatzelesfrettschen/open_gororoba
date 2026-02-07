//! Synchrotron radiation from relativistic electrons in magnetic fields.
//!
//! Synchrotron emission occurs when relativistic electrons spiral in
//! magnetic fields. Key astrophysical applications:
//! - Accretion disk coronae
//! - Relativistic jets (AGN, GRBs, X-ray binaries)
//! - Radio lobes and supernova remnants
//!
//! Key formulas:
//!   Critical frequency: nu_c = (3/4pi) (eB)/(m_e c) gamma^2
//!   Power per electron: P = (4/3) sigma_T c gamma^2 beta^2 U_B
//!   Cooling time:       t_cool = gamma m_e c^2 / P
//!
//! References:
//!   - Rybicki & Lightman (1979): Radiative Processes in Astrophysics, Ch. 6
//!   - Longair (2011): High Energy Astrophysics, Ch. 8
//!   - Fouka & Ouichaoui (2013): ApJ 774, 94 (synchrotron function fits)

use crate::constants::*;
use std::f64::consts::PI;

/// Fine structure constant alpha ~ 1/137.
pub const ALPHA_FINE: f64 = 7.297_352_569_3e-3;

/// Classical electron radius r_e = e^2/(m_e c^2) [cm].
pub const R_ELECTRON: f64 = 2.817_940_326_2e-13;

// ============================================================================
// Electron gyration
// ============================================================================

/// Electron gyrofrequency (cyclotron frequency) [Hz].
///
/// nu_B = eB / (2 pi m_e c)
pub fn gyrofrequency(b_gauss: f64) -> f64 {
    E_CHARGE_CGS * b_gauss.abs() / (2.0 * PI * M_ELECTRON_CGS * C_CGS)
}

/// Electron gyroradius (Larmor radius) [cm].
///
/// r_L = gamma m_e c^2 / (eB) for ultrarelativistic electrons (v ~ c).
///
/// In CGS Gaussian units, the Lorentz force has an extra 1/c factor:
/// F = (q/c) v x B, so r = gamma m v c / (eB). Setting v = c gives
/// r = gamma m c^2 / (eB).
pub fn gyroradius(gamma: f64, b_gauss: f64) -> f64 {
    if b_gauss.abs() < 1e-30 {
        return f64::INFINITY;
    }
    gamma * M_ELECTRON_CGS * C_CGS * C_CGS / (E_CHARGE_CGS * b_gauss.abs())
}

// ============================================================================
// Synchrotron emission
// ============================================================================

/// Synchrotron critical frequency [Hz].
///
/// nu_c = (3/4pi) (eB)/(m_e c) gamma^2 sin(alpha)
///
/// For pitch angle alpha = pi/2 (perpendicular to field).
pub fn critical_frequency(gamma: f64, b_gauss: f64, pitch_angle: f64) -> f64 {
    let sin_a = pitch_angle.sin();
    (3.0 / (4.0 * PI)) * (E_CHARGE_CGS * b_gauss.abs()) / (M_ELECTRON_CGS * C_CGS)
        * gamma * gamma * sin_a
}

/// Peak synchrotron frequency [Hz].
///
/// nu_peak ~ 0.29 * nu_c (where the synchrotron function F(x) peaks).
pub fn peak_frequency(gamma: f64, b_gauss: f64) -> f64 {
    0.29 * critical_frequency(gamma, b_gauss, PI / 2.0)
}

/// Synchrotron power radiated by a single electron [erg/s].
///
/// P = (4/3) sigma_T c gamma^2 beta^2 U_B
///
/// where U_B = B^2/(8 pi) is the magnetic energy density.
pub fn power_single_electron(gamma: f64, b_gauss: f64) -> f64 {
    let u_b = b_gauss * b_gauss / (8.0 * PI);
    let beta_sq = 1.0 - 1.0 / (gamma * gamma);
    (4.0 / 3.0) * SIGMA_THOMSON * C_CGS * gamma * gamma * beta_sq * u_b
}

/// Synchrotron cooling time [s].
///
/// t_cool = gamma m_e c^2 / P
///
/// Returns infinity if the power is negligible.
pub fn cooling_time(gamma: f64, b_gauss: f64) -> f64 {
    let p = power_single_electron(gamma, b_gauss);
    if p < 1e-50 {
        return f64::INFINITY;
    }
    gamma * M_ELECTRON_CGS * C_CGS * C_CGS / p
}

/// Cooling Lorentz factor at a given time.
///
/// gamma_cool = 6 pi m_e c / (sigma_T B^2 t)
///
/// Electrons above gamma_cool have radiated away most of their energy.
pub fn cooling_lorentz_factor(b_gauss: f64, t_s: f64) -> f64 {
    if t_s < 1e-50 || b_gauss.abs() < 1e-30 {
        return f64::INFINITY;
    }
    6.0 * PI * M_ELECTRON_CGS * C_CGS
        / (SIGMA_THOMSON * b_gauss * b_gauss * t_s)
}

// ============================================================================
// Synchrotron spectrum functions
// ============================================================================

/// Synchrotron function F(x) for a single electron.
///
/// F(x) = x integral_x^inf K_{5/3}(xi) d xi
///
/// where x = nu / nu_c. Uses asymptotic approximations:
///   Low-x:  F(x) ~ 1.8084 x^{1/3}
///   High-x: F(x) ~ sqrt(pi/2) sqrt(x) exp(-x)
///   Intermediate: Fouka & Ouichaoui (2013) polynomial fit
pub fn synchrotron_f(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < 0.01 {
        return 1.8084 * x.powf(1.0 / 3.0);
    }
    if x > 10.0 {
        return (PI / 2.0).sqrt() * x.sqrt() * (-x).exp();
    }
    // Intermediate (Fouka & Ouichaoui 2013 fit)
    1.8084
        * x.powf(1.0 / 3.0)
        * (-x).exp()
        * (1.0 + 0.884 * x.powf(2.0 / 3.0) + 0.471 * x.powf(4.0 / 3.0))
}

/// Synchrotron function G(x) for polarized emission.
///
/// G(x) = x K_{2/3}(x)
///
/// Used for computing the polarization degree Pi = G(x)/F(x).
pub fn synchrotron_g(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x < 0.01 {
        return 1.3541 * x.powf(1.0 / 3.0);
    }
    if x > 10.0 {
        return (PI / 2.0).sqrt() * x.sqrt() * (-x).exp();
    }
    1.3541
        * x.powf(1.0 / 3.0)
        * (-x).exp()
        * (1.0 + 0.6 * x.powf(2.0 / 3.0))
}

/// Synchrotron polarization degree Pi = G(x)/F(x).
///
/// For a power-law electron distribution with index p:
///   Pi_max = (p + 1) / (p + 7/3)
pub fn polarization_degree(x: f64) -> f64 {
    let f_val = synchrotron_f(x);
    if f_val < 1e-50 {
        return 0.0;
    }
    synchrotron_g(x) / f_val
}

// ============================================================================
// Power-law electron distribution
// ============================================================================

/// Spectral index for power-law electrons.
///
/// alpha = -(p - 1)/2 where F_nu ~ nu^alpha.
pub fn spectral_index(p: f64) -> f64 {
    -(p - 1.0) / 2.0
}

/// Electron power-law index from spectral index.
///
/// p = 1 - 2 alpha.
pub fn electron_index_from_spectral(alpha: f64) -> f64 {
    1.0 - 2.0 * alpha
}

/// Synchrotron spectrum from power-law electrons (normalized).
///
/// For N(gamma) ~ gamma^{-p} between gamma_min and gamma_max:
///   nu < nu_min: self-absorbed, F ~ nu^{5/2}
///   nu_min < nu < nu_max: power-law, F ~ nu^{-(p-1)/2}
///   nu > nu_max: exponential cutoff
pub fn spectrum_power_law(
    nu: f64,
    b_gauss: f64,
    gamma_min: f64,
    gamma_max: f64,
    p: f64,
) -> f64 {
    let nu_min = critical_frequency(gamma_min, b_gauss, PI / 2.0);
    let nu_max = critical_frequency(gamma_max, b_gauss, PI / 2.0);
    if nu <= 0.0 {
        return 0.0;
    }
    let alpha = spectral_index(p);

    if nu < nu_min {
        // Self-absorbed regime
        (nu / nu_min).powf(2.5)
    } else if nu < nu_max {
        // Power-law regime
        (nu / nu_min).powf(alpha)
    } else {
        // Exponential cutoff
        (nu_max / nu_min).powf(alpha) * (-(nu - nu_max) / nu_max).exp()
    }
}

// ============================================================================
// Self-absorption
// ============================================================================

/// Synchrotron self-absorption coefficient [cm^-1].
///
/// alpha_nu ~ (order of magnitude) for power-law electrons with index p.
/// Scales as nu^{-(p+4)/2}.
pub fn absorption_coefficient(nu: f64, b_gauss: f64, n_e: f64, p: f64) -> f64 {
    let nu_b = gyrofrequency(b_gauss);
    let prefactor = 0.02 * E_CHARGE_CGS * n_e / (M_ELECTRON_CGS * C_CGS);
    let exponent = -(p + 4.0) / 2.0;
    prefactor * nu_b.powf((p + 2.0) / 2.0) * nu.powf(exponent)
}

/// Self-absorption frequency where optical depth = 1 [Hz].
///
/// nu_a ~ nu_B * (n_e R / 10^20)^{2/(p+4)} (approximate).
pub fn self_absorption_frequency(b_gauss: f64, n_e: f64, source_r_cm: f64, p: f64) -> f64 {
    let nu_b = gyrofrequency(b_gauss);
    let exponent = 2.0 / (p + 4.0);
    nu_b * (n_e * source_r_cm / 1e20).powf(exponent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gyrofrequency() {
        // nu_B ~ 2.8 MHz/Gauss
        let nu = gyrofrequency(1.0);
        assert!(nu > 2.7e6 && nu < 2.9e6, "nu_B = {nu}");
    }

    #[test]
    fn test_gyrofrequency_scales_linearly() {
        let nu1 = gyrofrequency(1.0);
        let nu2 = gyrofrequency(2.0);
        assert!((nu2 / nu1 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_gyroradius() {
        // For gamma=1000, B=1G: r_L = gamma * m_e * c^2 / (eB) ~ 1.705e6 cm
        let r = gyroradius(1000.0, 1.0);
        assert!(r > 1.5e6 && r < 2.0e6, "r_L = {r}");
    }

    #[test]
    fn test_gyroradius_zero_field() {
        assert!(gyroradius(1.0, 0.0).is_infinite());
    }

    #[test]
    fn test_critical_frequency_scales_gamma_squared() {
        let nu1 = critical_frequency(10.0, 1.0, PI / 2.0);
        let nu2 = critical_frequency(100.0, 1.0, PI / 2.0);
        assert!((nu2 / nu1 - 100.0).abs() < 0.1, "should scale as gamma^2");
    }

    #[test]
    fn test_peak_frequency() {
        let nu_c = critical_frequency(100.0, 1.0, PI / 2.0);
        let nu_p = peak_frequency(100.0, 1.0);
        assert!((nu_p / nu_c - 0.29).abs() < 1e-10);
    }

    #[test]
    fn test_power_positive() {
        let p = power_single_electron(100.0, 1.0);
        assert!(p > 0.0);
    }

    #[test]
    fn test_power_scales_gamma_squared() {
        // For ultrarelativistic (beta~1): P ~ gamma^2
        let p1 = power_single_electron(100.0, 1.0);
        let p2 = power_single_electron(200.0, 1.0);
        assert!((p2 / p1 - 4.0).abs() < 0.01, "P should scale as gamma^2");
    }

    #[test]
    fn test_power_scales_b_squared() {
        let p1 = power_single_electron(100.0, 1.0);
        let p2 = power_single_electron(100.0, 2.0);
        assert!((p2 / p1 - 4.0).abs() < 0.01, "P should scale as B^2");
    }

    #[test]
    fn test_cooling_time_positive() {
        let t = cooling_time(100.0, 1.0);
        assert!(t > 0.0 && t.is_finite());
    }

    #[test]
    fn test_cooling_time_shorter_for_higher_gamma() {
        let t1 = cooling_time(10.0, 1.0);
        let t2 = cooling_time(100.0, 1.0);
        assert!(t2 < t1, "higher gamma cools faster");
    }

    #[test]
    fn test_cooling_lorentz_factor() {
        // gamma_cool should decrease with time (more electrons cooled)
        let g1 = cooling_lorentz_factor(1.0, 1e6);
        let g2 = cooling_lorentz_factor(1.0, 1e7);
        assert!(g2 < g1);
    }

    #[test]
    fn test_synchrotron_f_low_x() {
        let f = synchrotron_f(0.001);
        // F(x) ~ 1.808 * x^{1/3} for small x
        let expected = 1.8084 * 0.001_f64.powf(1.0 / 3.0);
        assert!((f - expected).abs() / expected < 0.01);
    }

    #[test]
    fn test_synchrotron_f_high_x() {
        let f = synchrotron_f(50.0);
        // Should be very small (exponential cutoff)
        assert!(f < 1e-10, "F(50) = {f}");
    }

    #[test]
    fn test_synchrotron_f_positive() {
        for x in [0.01, 0.1, 1.0, 5.0, 10.0] {
            assert!(synchrotron_f(x) > 0.0, "F({x}) should be positive");
        }
    }

    #[test]
    fn test_polarization_degree_bounded() {
        for x in [0.01, 0.1, 1.0, 5.0] {
            let pi = polarization_degree(x);
            assert!(pi >= 0.0 && pi <= 1.0, "Pi({x}) = {pi}");
        }
    }

    #[test]
    fn test_spectral_index() {
        // p=2: alpha = -0.5
        assert!((spectral_index(2.0) - (-0.5)).abs() < 1e-10);
        // p=3: alpha = -1.0
        assert!((spectral_index(3.0) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_electron_index_round_trip() {
        let p = 2.5;
        let alpha = spectral_index(p);
        let p_back = electron_index_from_spectral(alpha);
        assert!((p_back - p).abs() < 1e-10);
    }

    #[test]
    fn test_spectrum_power_law_self_absorbed() {
        let f = spectrum_power_law(1e6, 1.0, 10.0, 1e6, 2.5);
        // Below gamma_min frequency: should be positive (self-absorbed)
        assert!(f > 0.0);
    }

    #[test]
    fn test_absorption_coefficient_positive() {
        let alpha = absorption_coefficient(1e9, 1.0, 1e3, 2.5);
        assert!(alpha > 0.0);
    }
}
