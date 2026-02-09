//! Scattering models for radiative transfer in astrophysical environments.
//!
//! Three scattering mechanisms are implemented:
//!
//! 1. Thomson scattering: sigma_T ~ 6.65e-25 cm^2 (energy-independent electron scattering)
//!    Dominates for non-relativistic electrons at E << m_e c^2.
//!
//! 2. Rayleigh scattering: sigma_R ~ (a/lambda)^4 (particles much smaller than wavelength)
//!    Applies to dust grains in the ISM and circumstellar environments.
//!
//! 3. Mie scattering: Q_sca transitions from Rayleigh (x<<1) to geometric (x>>1)
//!    For particles with size comparable to wavelength.
//!
//! These complement the absorption models in [`crate::absorption`], completing the
//! radiative transfer toolkit: absorption removes photons, scattering redirects them.
//!
//! # References
//!
//! - Rybicki & Lightman (1979): Radiative Processes in Astrophysics, Ch. 3, 7
//! - Bohren & Huffman (1983): Absorption and Scattering of Light by Small Particles
//! - Klein & Nishina (1929): Z. Phys. 52, 853

use crate::constants::*;
use std::f64::consts::PI;

/// Planck constant [erg s].
const H_PLANCK: f64 = 6.626_070_15e-27;

// ============================================================================
// Thomson scattering
// ============================================================================

/// Thomson scattering cross-section [cm^2].
///
/// The classical electron scattering cross-section, independent of photon
/// energy in the non-relativistic limit (h*nu << m_e*c^2).
///
/// sigma_T = (8*pi/3) * r_e^2 = 6.6525e-25 cm^2
pub fn thomson_cross_section() -> f64 {
    SIGMA_THOMSON
}

/// Klein-Nishina corrected Thomson cross-section [cm^2].
///
/// At high photon energies (h*nu ~ m_e*c^2), electron recoil reduces the
/// effective scattering cross-section below the Thomson value. The full
/// Klein-Nishina formula is complex; we use the compact approximation
/// sigma_KN ~ sigma_T / (1 + 2.7 * x) where x = h*nu / (m_e*c^2).
///
/// A thermal correction (1 + theta) accounts for hot plasma where
/// electrons have significant thermal velocity (theta = k*T / (m_e*c^2)).
///
/// # Arguments
/// * `nu` - Photon frequency [Hz]
/// * `theta` - Dimensionless electron temperature k*T / (m_e*c^2)
pub fn thomson_corrected(nu: f64, theta: f64) -> f64 {
    // Photon energy in units of electron rest mass energy
    let x = (H_PLANCK * nu) / (M_ELECTRON_CGS * C_CGS * C_CGS);

    // Klein-Nishina reduction
    let sigma = SIGMA_THOMSON / (1.0 + 2.7 * x);

    // Hot plasma correction
    sigma * (1.0 + theta)
}

/// Thomson scattering opacity [cm^-1].
///
/// kappa_T = n_e * sigma_T
///
/// # Arguments
/// * `n_e` - Electron number density [cm^-3]
pub fn thomson_opacity(n_e: f64) -> f64 {
    n_e * SIGMA_THOMSON
}

// ============================================================================
// Rayleigh scattering
// ============================================================================

/// Rayleigh scattering cross-section for a dielectric sphere [cm^2].
///
/// sigma_R = (128 * pi^5 / 3) * (a / lambda)^4 * pi*a^2 * F(n)
///
/// where F(n) = ((n^2 - 1) / (n^2 + 2))^2 is the polarizability factor
/// and a is the grain radius.
///
/// The characteristic nu^4 dependence means blue light scatters ~16x more
/// than red light, explaining both blue skies and red sunsets.
///
/// # Arguments
/// * `nu` - Observing frequency [Hz]
/// * `grain_radius` - Dust grain radius [cm] (typically 0.01 to 1 micron)
/// * `refractive_index` - Real part of refractive index (typically 1.3-2.0)
pub fn rayleigh_cross_section(nu: f64, grain_radius: f64, refractive_index: f64) -> f64 {
    let lambda = C_CGS / nu;

    // Size parameter x = 2*pi*a / lambda
    let x = 2.0 * PI * grain_radius / lambda;
    let area = PI * grain_radius * grain_radius;

    // Clausius-Mossotti polarizability factor
    let n_sq = refractive_index * refractive_index;
    let polarizability = ((n_sq - 1.0) / (n_sq + 2.0)).powi(2);

    // Rayleigh formula: (128*pi^5/3) * x^4 * area * F(n)
    let sigma = (128.0 * PI.powi(5) / 3.0) * x.powi(4) * area * polarizability;

    sigma.max(0.0)
}

/// Rayleigh scattering opacity [cm^-1].
///
/// kappa_R = n_grain * sigma_R
///
/// # Arguments
/// * `nu` - Observing frequency [Hz]
/// * `grain_radius` - Dust grain radius [cm]
/// * `grain_density` - Grain number density [cm^-3]
/// * `refractive_index` - Real part of refractive index
pub fn rayleigh_opacity(
    nu: f64,
    grain_radius: f64,
    grain_density: f64,
    refractive_index: f64,
) -> f64 {
    grain_density * rayleigh_cross_section(nu, grain_radius, refractive_index)
}

// ============================================================================
// Mie scattering
// ============================================================================

/// Mie scattering efficiency Q_sca [dimensionless].
///
/// Transitions smoothly between three regimes:
/// - x << 1 (Rayleigh limit): Q_sca ~ x^4
/// - x ~ 1 (resonance): Q_sca ~ x
/// - x >> 1 (geometric limit): Q_sca -> 2 + oscillations (extinction paradox)
///
/// The size parameter is x = 2*pi*a / lambda.
///
/// # Arguments
/// * `nu` - Observing frequency [Hz]
/// * `grain_radius` - Grain radius [cm]
pub fn mie_efficiency(nu: f64, grain_radius: f64) -> f64 {
    let lambda = C_CGS / nu;
    let x = 2.0 * PI * grain_radius / lambda;

    if x < 0.05 {
        // Rayleigh limit
        x.powi(4)
    } else if x < 1.0 {
        // Transition regime
        x
    } else {
        // Large particles: Q_sca oscillates around 2 (extinction paradox)
        // Approximation: 2 + 4/x * sin(x) - 8/x^2 * (1 - cos(x))
        let q = 2.0 + 4.0 / x * x.sin() - (8.0 / (x * x)) * (1.0 - x.cos());
        q.clamp(0.0, 4.0)
    }
}

/// Mie scattering cross-section [cm^2].
///
/// sigma_Mie = Q_sca * pi * a^2
///
/// # Arguments
/// * `nu` - Observing frequency [Hz]
/// * `grain_radius` - Grain radius [cm]
pub fn mie_cross_section(nu: f64, grain_radius: f64) -> f64 {
    let area = PI * grain_radius * grain_radius;
    mie_efficiency(nu, grain_radius) * area
}

/// Mie scattering opacity [cm^-1].
///
/// kappa_Mie = n_grain * sigma_Mie
///
/// # Arguments
/// * `nu` - Observing frequency [Hz]
/// * `grain_radius` - Grain radius [cm]
/// * `grain_density` - Grain number density [cm^-3]
pub fn mie_opacity(nu: f64, grain_radius: f64, grain_density: f64) -> f64 {
    grain_density * mie_cross_section(nu, grain_radius)
}

// ============================================================================
// Scattering albedo and asymmetry
// ============================================================================

/// Single-scattering albedo [dimensionless, 0 to 1].
///
/// The probability that an interaction event is scattering rather than
/// absorption: omega = kappa_sca / (kappa_sca + kappa_abs).
///
/// - omega = 0: pure absorption (no scattering)
/// - omega = 1: pure scattering (no absorption)
///
/// # Arguments
/// * `kappa_sca` - Scattering opacity [cm^-1]
/// * `kappa_abs` - Absorption opacity [cm^-1]
pub fn single_scattering_albedo(kappa_sca: f64, kappa_abs: f64) -> f64 {
    let total = kappa_sca + kappa_abs;
    if total < 1e-30 {
        return 0.5;
    }
    kappa_sca / total
}

/// Scattering type for asymmetry parameter computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatteringType {
    /// Thomson: slightly forward-peaked (g ~ 0.2)
    Thomson,
    /// Rayleigh: nearly isotropic (g ~ 0)
    Rayleigh,
    /// Mie: strongly forward-peaked for large particles (g up to ~0.95)
    Mie,
}

/// Asymmetry parameter g = <cos(theta)> [dimensionless, -1 to 1].
///
/// Characterizes the angular distribution of scattered photons:
/// - g = 0: isotropic scattering
/// - g > 0: preferentially forward scattering
/// - g < 0: preferentially backward scattering
///
/// Thomson: g ~ 0.2 (slightly forward-peaked from dipole radiation pattern)
/// Rayleigh: g ~ 0 (symmetric for small particles)
/// Mie: g depends on size parameter x = 2*pi*a/lambda
///
/// # Arguments
/// * `scat_type` - Type of scattering mechanism
/// * `nu` - Observing frequency [Hz] (only needed for Mie)
/// * `grain_radius` - Grain radius [cm] (only needed for Mie)
pub fn asymmetry_parameter(scat_type: ScatteringType, nu: f64, grain_radius: f64) -> f64 {
    match scat_type {
        ScatteringType::Thomson => 0.2,
        ScatteringType::Rayleigh => 0.0,
        ScatteringType::Mie => {
            let lambda = C_CGS / nu;
            let x = 2.0 * PI * grain_radius / lambda;

            if x < 0.1 {
                0.0 // Rayleigh-like regime
            } else if x < 1.0 {
                0.3 * x // Transition
            } else {
                // Large particles: strongly forward scattering
                (0.5 + 0.3 * (x + 1.0).log10()).clamp(0.0, 0.95)
            }
        }
    }
}

// ============================================================================
// Scattering regime classification
// ============================================================================

/// Physical scattering regime based on the Mie size parameter x = 2*pi*a/lambda.
///
/// Determines which approximation is valid for scattering cross-section
/// calculations. This complements [`ScatteringType`] which is used for
/// asymmetry parameter dispatch -- [`ScatteringRegime`] classifies the
/// physical regime itself.
///
/// Boundaries (Bohren & Huffman 1983):
/// - x < 0.05: Rayleigh (dipole approximation valid, sigma ~ x^4)
/// - 0.05 <= x < 1.0: Transition (neither Rayleigh nor geometric works)
/// - x >= 1.0: Geometric (ray optics + diffraction, Q_sca -> 2)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScatteringRegime {
    /// Rayleigh regime: particle much smaller than wavelength (x < 0.05)
    Rayleigh,
    /// Transition regime: full Mie theory needed (0.05 <= x < 1.0)
    Transition,
    /// Geometric regime: ray optics valid (x >= 1.0)
    Geometric,
}

/// Classify the scattering regime from the Mie size parameter.
///
/// # Arguments
/// * `x` - Mie size parameter: x = 2*pi*a / lambda
pub fn classify_scattering_regime(x: f64) -> ScatteringRegime {
    if x < 0.05 {
        ScatteringRegime::Rayleigh
    } else if x < 1.0 {
        ScatteringRegime::Transition
    } else {
        ScatteringRegime::Geometric
    }
}

/// Classify the scattering regime from frequency and grain radius.
///
/// Convenience wrapper that computes x = 2*pi*a/lambda internally.
///
/// # Arguments
/// * `nu` - Observing frequency [Hz]
/// * `grain_radius` - Grain radius [cm]
pub fn classify_scattering_regime_from_params(nu: f64, grain_radius: f64) -> ScatteringRegime {
    let lambda = C_CGS / nu;
    let x = 2.0 * PI * grain_radius / lambda;
    classify_scattering_regime(x)
}

// ============================================================================
// Combined opacity
// ============================================================================

/// Total scattering opacity from all mechanisms [cm^-1].
///
/// kappa_sca = kappa_Thomson + kappa_Rayleigh + kappa_Mie
///
/// # Arguments
/// * `nu` - Observing frequency [Hz]
/// * `n_e` - Electron number density [cm^-3]
/// * `grain_radius` - Dust grain radius [cm]
/// * `grain_density` - Dust grain number density [cm^-3]
/// * `refractive_index` - Real part of grain refractive index
pub fn total_scattering_opacity(
    nu: f64,
    n_e: f64,
    grain_radius: f64,
    grain_density: f64,
    refractive_index: f64,
) -> f64 {
    let k_thomson = thomson_opacity(n_e);
    let k_rayleigh = rayleigh_opacity(nu, grain_radius, grain_density, refractive_index);
    let k_mie = mie_opacity(nu, grain_radius, grain_density);

    k_thomson + k_rayleigh + k_mie
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Thomson scattering --

    #[test]
    fn test_thomson_constant() {
        let sigma = thomson_cross_section();
        assert!((sigma - 6.6525e-25).abs() / 6.6525e-25 < 1e-3);
    }

    #[test]
    fn test_thomson_kn_reduces_at_high_energy() {
        // At x-ray energies, Klein-Nishina reduces cross-section
        let nu_low: f64 = 1e10; // 10 GHz (radio)
        let nu_high: f64 = 1e20; // 100 keV (hard X-ray)
        let theta = 0.0; // cold electrons

        let sigma_low = thomson_corrected(nu_low, theta);
        let sigma_high = thomson_corrected(nu_high, theta);

        assert!(
            sigma_low > sigma_high,
            "KN should reduce cross-section at high energy"
        );
        assert!(
            sigma_low / SIGMA_THOMSON > 0.99,
            "low-energy should be close to Thomson"
        );
    }

    #[test]
    fn test_thomson_hot_plasma_enhancement() {
        let nu: f64 = 1e10;
        let sigma_cold = thomson_corrected(nu, 0.0);
        let sigma_hot = thomson_corrected(nu, 1.0);

        assert!(
            sigma_hot > sigma_cold,
            "hot plasma should enhance scattering"
        );
        // theta = 1: factor should be ~2
        assert!((sigma_hot / sigma_cold - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_thomson_opacity_linear_in_density() {
        let k1 = thomson_opacity(1e8);
        let k2 = thomson_opacity(2e8);
        assert!((k2 / k1 - 2.0).abs() < 1e-10);
    }

    // -- Rayleigh scattering --

    #[test]
    fn test_rayleigh_nu4_dependence() {
        let a: f64 = 1e-5; // 0.1 micron grain
        let n: f64 = 1.5;

        let sigma1 = rayleigh_cross_section(1e14, a, n);
        let sigma2 = rayleigh_cross_section(2e14, a, n);

        // sigma ~ nu^4, so doubling frequency should increase by 16x
        let ratio = sigma2 / sigma1;
        assert!(
            (ratio - 16.0).abs() / 16.0 < 0.01,
            "ratio = {ratio}, expected 16"
        );
    }

    #[test]
    fn test_rayleigh_nonnegative() {
        let sigma = rayleigh_cross_section(1e15, 1e-5, 1.5);
        assert!(sigma >= 0.0);
    }

    #[test]
    fn test_rayleigh_zero_for_n_equals_1() {
        // Refractive index 1.0: no polarizability, no scattering
        let sigma = rayleigh_cross_section(1e14, 1e-5, 1.0);
        assert!(sigma.abs() < 1e-40, "sigma = {sigma}");
    }

    // -- Mie scattering --

    #[test]
    fn test_mie_rayleigh_limit() {
        // Very small particles (x << 1): Q_sca ~ x^4
        let nu: f64 = 1e10; // 10 GHz, lambda ~ 3 cm
        let a: f64 = 1e-4; // 1 micron, x ~ 2e-4
        let q = mie_efficiency(nu, a);
        assert!(q < 1e-10, "Q = {q}, should be tiny in Rayleigh limit");
    }

    #[test]
    fn test_mie_geometric_limit() {
        // Very large particles (x >> 1): Q_sca -> 2 (extinction paradox)
        let nu: f64 = 1e15; // UV, lambda ~ 3000 A
        let a: f64 = 0.01; // 100 micron, x ~ 200
        let q = mie_efficiency(nu, a);
        assert!(
            q > 0.5 && q <= 4.0,
            "Q = {q}, should be near 2 in geometric limit"
        );
    }

    #[test]
    fn test_mie_efficiency_bounded() {
        // Q_sca should be clamped to [0, 4]
        for &nu in &[1e10_f64, 1e12, 1e14, 1e16] {
            let q = mie_efficiency(nu, 1e-4);
            assert!(q >= 0.0 && q <= 4.0, "Q({nu}) = {q}");
        }
    }

    #[test]
    fn test_mie_cross_section_positive() {
        let sigma = mie_cross_section(1e14, 1e-4);
        assert!(sigma > 0.0);
    }

    // -- Albedo and asymmetry --

    #[test]
    fn test_albedo_bounds() {
        let omega = single_scattering_albedo(1.0, 1.0);
        assert!(omega >= 0.0 && omega <= 1.0, "omega = {omega}");
    }

    #[test]
    fn test_albedo_pure_scattering() {
        let omega = single_scattering_albedo(1.0, 0.0);
        assert!((omega - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_albedo_pure_absorption() {
        let omega = single_scattering_albedo(0.0, 1.0);
        assert!(omega.abs() < 1e-10);
    }

    #[test]
    fn test_asymmetry_thomson() {
        let g = asymmetry_parameter(ScatteringType::Thomson, 1e10, 0.0);
        assert!((g - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_asymmetry_rayleigh_isotropic() {
        let g = asymmetry_parameter(ScatteringType::Rayleigh, 1e10, 0.0);
        assert!(g.abs() < 1e-10);
    }

    #[test]
    fn test_asymmetry_mie_forward() {
        // Large particles: strongly forward scattering
        let nu: f64 = 1e15;
        let a: f64 = 0.01;
        let g = asymmetry_parameter(ScatteringType::Mie, nu, a);
        assert!(g > 0.5, "g = {g}, should be forward-peaked for large x");
    }

    #[test]
    fn test_asymmetry_mie_bounded() {
        for &nu in &[1e10_f64, 1e12, 1e14, 1e16] {
            let g = asymmetry_parameter(ScatteringType::Mie, nu, 1e-4);
            assert!(g >= 0.0 && g <= 0.95, "g({nu}) = {g}");
        }
    }

    // -- Total opacity --

    #[test]
    fn test_total_opacity_nonnegative() {
        let k = total_scattering_opacity(1e14, 1e8, 1e-5, 1e3, 1.5);
        assert!(k > 0.0, "total opacity = {k}");
    }

    #[test]
    fn test_total_opacity_dominated_by_thomson_at_radio() {
        // At radio frequencies, Rayleigh x^4 suppression makes dust negligible.
        // Thomson dominates for ionized plasma in radio/microwave band.
        let n_e: f64 = 1e12;
        let n_grain: f64 = 1.0;
        let nu: f64 = 1e9; // 1 GHz radio
        let k_total = total_scattering_opacity(nu, n_e, 1e-5, n_grain, 1.5);
        let k_thom = thomson_opacity(n_e);

        assert!(
            k_thom / k_total > 0.99,
            "Thomson should dominate at radio frequencies"
        );
    }

    // -- Scattering regime classification --

    #[test]
    fn test_regime_rayleigh() {
        assert_eq!(classify_scattering_regime(0.01), ScatteringRegime::Rayleigh);
        assert_eq!(
            classify_scattering_regime(0.049),
            ScatteringRegime::Rayleigh
        );
    }

    #[test]
    fn test_regime_transition() {
        assert_eq!(
            classify_scattering_regime(0.05),
            ScatteringRegime::Transition
        );
        assert_eq!(
            classify_scattering_regime(0.5),
            ScatteringRegime::Transition
        );
        assert_eq!(
            classify_scattering_regime(0.999),
            ScatteringRegime::Transition
        );
    }

    #[test]
    fn test_regime_geometric() {
        assert_eq!(classify_scattering_regime(1.0), ScatteringRegime::Geometric);
        assert_eq!(
            classify_scattering_regime(100.0),
            ScatteringRegime::Geometric
        );
    }

    #[test]
    fn test_regime_consistent_with_mie_efficiency() {
        // Rayleigh regime: mie_efficiency should return x^4
        let nu: f64 = 1e10;
        let a: f64 = 1e-4; // x ~ 2e-4
        let lambda = C_CGS / nu;
        let x = 2.0 * PI * a / lambda;
        assert_eq!(classify_scattering_regime(x), ScatteringRegime::Rayleigh);
        let q = mie_efficiency(nu, a);
        let expected = x.powi(4);
        assert!((q - expected).abs() / expected.max(1e-30) < 0.01);
    }

    #[test]
    fn test_regime_from_params() {
        // Large grain at optical frequency -> Geometric
        let regime = classify_scattering_regime_from_params(1e15, 0.01);
        assert_eq!(regime, ScatteringRegime::Geometric);
        // Small grain at radio frequency -> Rayleigh
        let regime = classify_scattering_regime_from_params(1e9, 1e-5);
        assert_eq!(regime, ScatteringRegime::Rayleigh);
    }
}
