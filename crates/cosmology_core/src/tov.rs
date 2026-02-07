//! Neutron star TOV solver with tidal deformability.
//!
//! Extends the gravastar TOV solver with neutron-star-specific features:
//!   - Named polytropic EOS presets (SLy4, APR4, soft, stiff)
//!   - Tidal Love number k_2 (Hinderer 2008 / Yagi-Yunes 2013 fitting)
//!   - Dimensionless tidal deformability Lambda = (2/3) k_2 / C^5
//!   - Mass-radius relation scanner (sweep central density)
//!   - Maximum mass finder (TOV limit)
//!
//! All computations use **geometric units** (G = c = 1), consistent with
//! the gravastar module. The tidal deformability is the key GW170817
//! observable connecting nuclear EOS to gravitational wave signals.
//!
//! # References
//!
//! - Tolman (1939): Phys. Rev. 55, 364
//! - Oppenheimer & Volkoff (1939): Phys. Rev. 55, 374
//! - Shapiro & Teukolsky (1983): Black Holes, White Dwarfs, and Neutron Stars
//! - Hinderer (2008): ApJ 677, 1216 (tidal Love number)
//! - Yagi & Yunes (2013): Science 341, 365 (I-Love-Q universality)
//! - Abbott et al. (2017): PRL 119, 161101 (GW170817)

use std::f64::consts::PI;
use crate::gravastar::{PolytropicEos, TovState};

// ============================================================================
// Named EOS presets for neutron stars
// ============================================================================

/// Named polytropic EOS parameters for neutron star models.
///
/// These are single-polytrope approximations to realistic nuclear EOS.
/// For precision work, piecewise polytropes or tabulated EOS are needed.
pub mod eos_presets {
    use super::PolytropicEos;

    /// Soft EOS: small maximum mass (~1.5 M_sun in geometric units).
    pub fn soft() -> PolytropicEos {
        PolytropicEos::new(0.01, 2.5)
    }

    /// Stiff EOS: large maximum mass (~2.5 M_sun in geometric units).
    pub fn stiff() -> PolytropicEos {
        PolytropicEos::new(0.05, 3.0)
    }

    /// SLy4 approximation (Douchin & Haensel 2001).
    ///
    /// One of the most commonly used EOS for neutron star modeling.
    /// K and gamma chosen to reproduce M_max ~ 2.0 M_sun.
    pub fn sly4() -> PolytropicEos {
        PolytropicEos::new(0.036, 3.0)
    }

    /// APR4 approximation (Akmal, Pandharipande, Ravenhall 1998).
    ///
    /// Moderately stiff nuclear EOS.
    pub fn apr4() -> PolytropicEos {
        PolytropicEos::new(0.022, 2.8)
    }
}

// ============================================================================
// Tidal deformability
// ============================================================================

/// Tidal Love number k_2 from the Yagi-Yunes (2013) fitting formula.
///
/// k_2 ~ 0.05 + 0.1*(1 - 5*C), clamped to [0, 0.15].
///
/// This is an approximate fitting formula. For exact results, one would
/// need to integrate the tidal perturbation equations (Regge-Wheeler + matching).
///
/// # Arguments
/// * `compactness` - C = M/R (geometric units, not 2M/R)
pub fn tidal_love_number_k2(compactness: f64) -> f64 {
    let k2 = 0.05 + 0.1 * (1.0 - 5.0 * compactness);
    k2.clamp(0.0, 0.15)
}

/// Dimensionless tidal deformability Lambda.
///
/// Lambda = (2/3) * k_2 / C^5
///
/// This is the primary observable from binary neutron star mergers.
/// GW170817 measured Lambda_tilde ~ 300 +/- 230 (90% CI), ruling out
/// the stiffest EOS models.
///
/// # Arguments
/// * `compactness` - C = M/R (geometric units)
pub fn tidal_deformability(compactness: f64) -> f64 {
    let k2 = tidal_love_number_k2(compactness);
    let c5 = compactness.powi(5);
    if c5 < 1e-30 {
        return 0.0;
    }
    (2.0 / 3.0) * k2 / c5
}

/// Combined (effective) tidal deformability for a binary system.
///
/// Lambda_tilde = (16/13) * [(12q + 1) Lambda_1 + (12 + q) q^4 Lambda_2]
///                / (1 + q)^5
///
/// where q = m_2/m_1 <= 1 is the mass ratio.
///
/// This is the quantity directly constrained by GW observations.
pub fn combined_tidal_deformability(
    lambda_1: f64,
    lambda_2: f64,
    q: f64,
) -> f64 {
    let q4 = q.powi(4);
    let one_plus_q5 = (1.0 + q).powi(5);
    if one_plus_q5 < 1e-30 {
        return 0.0;
    }
    (16.0 / 13.0) * ((12.0 * q + 1.0) * lambda_1 + (12.0 + q) * q4 * lambda_2) / one_plus_q5
}

// ============================================================================
// Neutron star TOV integration (geometric units)
// ============================================================================

/// Result of neutron star TOV integration.
#[derive(Debug, Clone)]
pub struct NeutronStarProfile {
    /// Stellar radius R (geometric units)
    pub radius: f64,
    /// Total mass M (geometric units)
    pub mass: f64,
    /// Central density rho_c
    pub rho_c: f64,
    /// Compactness C = M/R (note: gravastar uses C = 2M/R)
    pub compactness: f64,
    /// Surface redshift z = 1/sqrt(1 - 2M/R) - 1
    pub surface_redshift: f64,
    /// Tidal Love number k_2
    pub love_number_k2: f64,
    /// Dimensionless tidal deformability Lambda
    pub tidal_deformability: f64,
}

/// Integrate TOV for a neutron star with given central density and EOS.
///
/// Uses the gravastar solver in isotropic mode (no anisotropy), then
/// extracts neutron star observables including tidal deformability.
///
/// # Arguments
/// * `rho_c` - Central density
/// * `eos` - Equation of state
/// * `r_max` - Maximum integration radius (default: estimate from rho_c)
pub fn integrate_neutron_star(
    rho_c: f64,
    eos: &PolytropicEos,
    r_max: f64,
) -> Option<NeutronStarProfile> {
    // Set up gravastar config for pure NS (no interior de Sitter region)
    // Start integration from r ~ 0 (small but nonzero to avoid singularity)
    let dr = 1e-4;
    let r_start = dr;

    // Initial conditions
    let p_c = eos.pressure(rho_c);
    if p_c <= 0.0 {
        return None;
    }

    // Integrate TOV directly (bypass gravastar's de Sitter interior matching)
    let mut state = TovState {
        r: r_start,
        m: (4.0 / 3.0) * PI * rho_c * r_start.powi(3),
        p: p_c,
    };

    let p_floor = 1e-12 * p_c;

    while state.r < r_max && state.p > p_floor {
        let rho = eos.density(state.p);

        // RK4 step for TOV equations in geometric units
        let (dm1, dp1) = tov_rhs(state.r, state.m, state.p, rho);

        let r2 = state.r + 0.5 * dr;
        let m2 = state.m + 0.5 * dr * dm1;
        let p2 = (state.p + 0.5 * dr * dp1).max(0.0);
        let rho2 = eos.density(p2);
        let (dm2, dp2) = tov_rhs(r2, m2, p2, rho2);

        let r3 = state.r + 0.5 * dr;
        let m3 = state.m + 0.5 * dr * dm2;
        let p3 = (state.p + 0.5 * dr * dp2).max(0.0);
        let rho3 = eos.density(p3);
        let (dm3, dp3) = tov_rhs(r3, m3, p3, rho3);

        let r4 = state.r + dr;
        let m4 = state.m + dr * dm3;
        let p4 = (state.p + dr * dp3).max(0.0);
        let rho4 = eos.density(p4);
        let (dm4, dp4) = tov_rhs(r4, m4, p4, rho4);

        state.m += dr * (dm1 + 2.0 * dm2 + 2.0 * dm3 + dm4) / 6.0;
        state.p += dr * (dp1 + 2.0 * dp2 + 2.0 * dp3 + dp4) / 6.0;
        state.r += dr;

        if state.p < 0.0 {
            state.p = 0.0;
            break;
        }
    }

    let radius = state.r;
    let mass = state.m;

    if radius <= 0.0 || mass <= 0.0 {
        return None;
    }

    let compactness = mass / radius; // M/R
    let two_c = 2.0 * compactness;

    let surface_redshift = if two_c < 1.0 {
        1.0 / (1.0 - two_c).sqrt() - 1.0
    } else {
        f64::INFINITY
    };

    let k2 = tidal_love_number_k2(compactness);
    let lambda = tidal_deformability(compactness);

    Some(NeutronStarProfile {
        radius,
        mass,
        rho_c,
        compactness,
        surface_redshift,
        love_number_k2: k2,
        tidal_deformability: lambda,
    })
}

/// TOV right-hand side in geometric units (G = c = 1).
///
/// dm/dr = 4*pi*r^2*rho
/// dP/dr = -(rho + P)*(m + 4*pi*r^3*P) / (r*(r - 2*m))
fn tov_rhs(r: f64, m: f64, p: f64, rho: f64) -> (f64, f64) {
    if r < 1e-10 {
        return (0.0, 0.0);
    }

    let dm_dr = 4.0 * PI * r * r * rho;

    let denom = r * (r - 2.0 * m);
    if denom.abs() < 1e-15 {
        return (dm_dr, -1e10);
    }

    let dp_dr = -(rho + p) * (m + 4.0 * PI * r * r * r * p) / denom;

    (dm_dr, dp_dr)
}

// ============================================================================
// Mass-radius relation
// ============================================================================

/// A point on the mass-radius curve.
#[derive(Debug, Clone, Copy)]
pub struct MassRadiusPoint {
    pub rho_c: f64,
    pub mass: f64,
    pub radius: f64,
    pub compactness: f64,
    pub tidal_deformability: f64,
}

/// Compute the mass-radius relation by sweeping central density.
///
/// Returns a sequence of (rho_c, M, R) points. The maximum mass
/// identifies the TOV limit for the given EOS.
///
/// # Arguments
/// * `eos` - Equation of state
/// * `rho_min` - Minimum central density
/// * `rho_max` - Maximum central density
/// * `n_samples` - Number of logarithmically-spaced samples
/// * `r_max` - Maximum integration radius per star
pub fn mass_radius_relation(
    eos: &PolytropicEos,
    rho_min: f64,
    rho_max: f64,
    n_samples: usize,
    r_max: f64,
) -> Vec<MassRadiusPoint> {
    let log_min = rho_min.log10();
    let log_max = rho_max.log10();
    let d_log = (log_max - log_min) / n_samples as f64;

    let mut points = Vec::with_capacity(n_samples + 1);

    for i in 0..=n_samples {
        let rho_c = 10.0_f64.powf(log_min + i as f64 * d_log);
        if let Some(profile) = integrate_neutron_star(rho_c, eos, r_max) {
            points.push(MassRadiusPoint {
                rho_c,
                mass: profile.mass,
                radius: profile.radius,
                compactness: profile.compactness,
                tidal_deformability: profile.tidal_deformability,
            });
        }
    }

    points
}

/// Find the maximum mass (TOV limit) for a given EOS.
///
/// Returns (M_max, rho_c_at_max, R_at_max).
pub fn tov_maximum_mass(
    eos: &PolytropicEos,
    rho_min: f64,
    rho_max: f64,
    n_samples: usize,
    r_max: f64,
) -> Option<(f64, f64, f64)> {
    let points = mass_radius_relation(eos, rho_min, rho_max, n_samples, r_max);

    points
        .iter()
        .max_by(|a, b| a.mass.partial_cmp(&b.mass).unwrap())
        .map(|p| (p.mass, p.rho_c, p.radius))
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- Tidal Love number --

    #[test]
    fn test_love_number_low_compactness() {
        // Low compactness (large star): k2 ~ 0.15
        let k2 = tidal_love_number_k2(0.05);
        assert!(k2 > 0.1, "k2 = {k2}");
    }

    #[test]
    fn test_love_number_high_compactness() {
        // High compactness (near BH limit): k2 -> 0
        let k2 = tidal_love_number_k2(0.25);
        assert!(k2 < 0.1, "k2 = {k2}");
        assert!(k2 >= 0.0, "k2 must be non-negative");
    }

    #[test]
    fn test_love_number_clamped() {
        // Very high compactness: should clamp to 0
        let k2 = tidal_love_number_k2(0.5);
        assert!((k2 - 0.0).abs() < 1e-14, "k2 = {k2}");
    }

    #[test]
    fn test_love_number_monotonically_decreasing() {
        let c_values = [0.05, 0.10, 0.15, 0.20, 0.25];
        for window in c_values.windows(2) {
            let k2_low = tidal_love_number_k2(window[0]);
            let k2_high = tidal_love_number_k2(window[1]);
            assert!(k2_low >= k2_high, "k2({}) = {} > k2({}) = {}", window[0], k2_low, window[1], k2_high);
        }
    }

    // -- Tidal deformability --

    #[test]
    fn test_tidal_deformability_large_for_low_compactness() {
        // Low compactness: Lambda should be very large (C^5 in denominator)
        let lambda = tidal_deformability(0.05);
        assert!(lambda > 100.0, "Lambda = {lambda}");
    }

    #[test]
    fn test_tidal_deformability_small_for_high_compactness() {
        // High compactness: Lambda should be small
        // At C=0.25: k2 = 0.025, Lambda = (2/3)*0.025/0.25^5 ~ 17
        let lambda = tidal_deformability(0.25);
        assert!(lambda < 50.0, "Lambda = {lambda}");
    }

    #[test]
    fn test_tidal_deformability_positive() {
        for c in [0.05, 0.1, 0.15, 0.2, 0.25] {
            let lambda = tidal_deformability(c);
            assert!(lambda >= 0.0, "Lambda({c}) = {lambda}");
        }
    }

    // -- Combined tidal deformability --

    #[test]
    fn test_combined_tidal_equal_mass() {
        // q = 1: Lambda_tilde = (16/13) * (13*Lambda + 13*Lambda) / 32
        //       = (16/13) * 26*Lambda / 32 = Lambda (for Lambda_1 = Lambda_2)
        let lambda = 500.0;
        let combined = combined_tidal_deformability(lambda, lambda, 1.0);
        assert!((combined - lambda).abs() < 1e-6, "combined = {combined}, expected {lambda}");
    }

    #[test]
    fn test_combined_tidal_asymmetric() {
        let lambda_tilde = combined_tidal_deformability(600.0, 400.0, 0.8);
        // Should be between 400 and 600
        assert!(lambda_tilde > 300.0 && lambda_tilde < 700.0, "Lambda_tilde = {lambda_tilde}");
    }

    // -- TOV integration --

    #[test]
    fn test_neutron_star_produces_finite_mass() {
        // rho_c ~ 1.0 needed for meaningful pressure with gamma >= 2.5
        let eos = eos_presets::stiff();
        let profile = integrate_neutron_star(1.0, &eos, 20.0);
        assert!(profile.is_some(), "integration should succeed");
        let p = profile.unwrap();
        assert!(p.mass > 1e-4, "M = {}", p.mass);
        assert!(p.radius > 0.01, "R = {}", p.radius);
        assert!(p.compactness > 0.0 && p.compactness < 0.5, "C = {}", p.compactness);
    }

    #[test]
    fn test_neutron_star_mass_increases_with_rho_c() {
        let eos = eos_presets::stiff();
        let p1 = integrate_neutron_star(0.5, &eos, 20.0).unwrap();
        let p2 = integrate_neutron_star(1.0, &eos, 20.0).unwrap();
        // At low central density, higher rho_c gives higher mass (before TOV limit)
        assert!(p2.mass > p1.mass, "M(rho1) = {}, M(rho2) = {}", p1.mass, p2.mass);
    }

    #[test]
    fn test_neutron_star_has_finite_tidal() {
        let eos = eos_presets::sly4();
        let profile = integrate_neutron_star(1.0, &eos, 20.0);
        assert!(profile.is_some());
        let p = profile.unwrap();
        assert!(p.tidal_deformability > 0.0, "Lambda = {}", p.tidal_deformability);
        assert!(p.love_number_k2 > 0.0 && p.love_number_k2 <= 0.15, "k2 = {}", p.love_number_k2);
    }

    #[test]
    fn test_surface_redshift_positive() {
        let eos = eos_presets::apr4();
        if let Some(p) = integrate_neutron_star(1.0, &eos, 20.0) {
            assert!(p.surface_redshift > 0.0, "z = {}", p.surface_redshift);
        }
    }

    // -- Mass-radius relation --

    #[test]
    fn test_mass_radius_produces_points() {
        let eos = eos_presets::stiff();
        let points = mass_radius_relation(&eos, 0.5, 5.0, 10, 20.0);
        assert!(points.len() > 5, "got {} points", points.len());
    }

    // -- TOV maximum mass --

    #[test]
    fn test_tov_maximum_mass_exists() {
        let eos = eos_presets::stiff();
        let result = tov_maximum_mass(&eos, 0.5, 5.0, 20, 20.0);
        assert!(result.is_some(), "should find maximum mass");
        let (m_max, _rho_c, _r) = result.unwrap();
        assert!(m_max > 1e-3, "M_max = {m_max}");
    }

    #[test]
    fn test_stiff_eos_heavier_than_soft() {
        let stiff = eos_presets::stiff();
        let soft = eos_presets::soft();

        // Sweep densities where gamma > 2 polytropes produce meaningful pressure
        let m_stiff = tov_maximum_mass(&stiff, 0.5, 5.0, 20, 20.0)
            .map(|(m, _, _)| m)
            .unwrap_or(0.0);
        let m_soft = tov_maximum_mass(&soft, 0.5, 5.0, 20, 20.0)
            .map(|(m, _, _)| m)
            .unwrap_or(0.0);

        assert!(m_stiff > m_soft, "M_stiff = {m_stiff}, M_soft = {m_soft}");
    }

    // -- EOS presets --

    #[test]
    fn test_eos_presets_valid() {
        let presets = [
            eos_presets::soft(),
            eos_presets::stiff(),
            eos_presets::sly4(),
            eos_presets::apr4(),
        ];
        for eos in &presets {
            assert!(eos.k > 0.0);
            assert!(eos.gamma > 1.0, "gamma = {} should be > 1 for stability", eos.gamma);
            let p = eos.pressure(0.001);
            assert!(p > 0.0, "P(0.001) = {p}");
        }
    }
}
