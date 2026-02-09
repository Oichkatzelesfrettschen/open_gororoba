//! Penrose process and energy extraction from rotating black holes.
//!
//! The Penrose process (1969) extracts rotational energy from a Kerr black
//! hole's ergosphere. A particle entering the ergosphere can split so that
//! one fragment falls in with negative energy (as measured at infinity),
//! while the other escapes with more energy than the original.
//!
//! Also includes the Blandford-Znajek (1977) electromagnetic extraction
//! mechanism and superradiant scattering amplification.
//!
//! References:
//!   - Penrose (1969): "Gravitational Collapse: The Role of GR"
//!   - Blandford & Znajek (1977): MNRAS 179, 433
//!   - Tchekhovskoy, Narayan, McKinney (2010): ApJ 711, 50 (BZ numerical factor)
//!   - Chandrasekhar (1983): Mathematical Theory of Black Holes, Ch. 7

use crate::constants::*;
use std::f64::consts::PI;

// ============================================================================
// Penrose process efficiency
// ============================================================================

/// Maximum Penrose process efficiency.
///
/// eta_max = 1 - sqrt((1 + sqrt(1 - a*^2)) / 2)
///
/// For a* -> 1 (maximal spin): eta_max = 1 - 1/sqrt(2) ~ 29.3%.
/// For a* = 0 (Schwarzschild): eta_max = 0 (no ergosphere).
pub fn penrose_maximum_efficiency(a_star: f64) -> f64 {
    let a_star = a_star.abs().clamp(0.0, 1.0);
    let sqrt_f = (1.0 - a_star * a_star).sqrt();
    1.0 - ((1.0 + sqrt_f) / 2.0).sqrt()
}

/// Irreducible mass in CGS units.
///
/// M_irr = M * sqrt((1 + sqrt(1 - a*^2)) / 2) [g]
///
/// This mass cannot be reduced by any classical process; it only increases
/// (Hawking area theorem).
pub fn irreducible_mass_cgs(mass_g: f64, a_star: f64) -> f64 {
    let a_star = a_star.abs().clamp(0.0, 1.0);
    let sqrt_f = (1.0 - a_star * a_star).sqrt();
    mass_g * ((1.0 + sqrt_f) / 2.0).sqrt()
}

/// Rotational energy: E_rot = (M - M_irr) c^2 [erg].
///
/// This is the maximum energy extractable via the Penrose process.
/// For a maximally spinning BH, E_rot = M c^2 (1 - 1/sqrt(2)) ~ 29.3% Mc^2.
pub fn rotational_energy(mass_g: f64, a_star: f64) -> f64 {
    let m_irr = irreducible_mass_cgs(mass_g, a_star);
    (mass_g - m_irr) * C_CGS * C_CGS
}

/// Horizon angular velocity in CGS [rad/s].
///
/// Omega_H = a c / (r+^2 + a^2) where r+ and a are in geometric units [cm].
pub fn horizon_angular_velocity_cgs(mass_g: f64, a_star: f64) -> f64 {
    let a_star = a_star.abs().clamp(0.0, 0.9999);
    let m_geo = G_CGS * mass_g / (C_CGS * C_CGS);
    let r_plus = m_geo * (1.0 + (1.0 - a_star * a_star).sqrt());
    let a = a_star * m_geo;
    a * C_CGS / (r_plus * r_plus + a * a)
}

// ============================================================================
// Blandford-Znajek mechanism
// ============================================================================

/// Blandford-Znajek power output [erg/s].
///
/// P_BZ ~ kappa * Phi^2 * Omega_H^2 / c
///
/// where Phi^2 ~ pi B^2 r+^2 is the magnetic flux squared and
/// kappa ~ 0.05 is a numerical factor from GRMHD simulations
/// (Tchekhovskoy, Narayan, McKinney 2010).
///
/// Arguments:
///   mass_g: black hole mass [g]
///   a_star: dimensionless spin |a*| <= 1
///   b_gauss: magnetic field strength at the horizon [Gauss]
pub fn blandford_znajek_power(mass_g: f64, a_star: f64, b_gauss: f64) -> f64 {
    let a_star = a_star.abs().clamp(0.0, 0.998);
    let m_geo = G_CGS * mass_g / (C_CGS * C_CGS);
    let r_plus = m_geo * (1.0 + (1.0 - a_star * a_star).sqrt());

    let phi_sq = b_gauss * b_gauss * PI * r_plus * r_plus;
    let omega_h = a_star * C_CGS / (2.0 * r_plus);

    let kappa = 0.05;
    kappa * phi_sq * omega_h * omega_h / C_CGS
}

/// Magnetic field required for Eddington-limited BZ power [Gauss].
///
/// Inverts the BZ formula: for P_BZ = L_Edd, find B.
pub fn bz_eddington_field(mass_g: f64, a_star: f64) -> f64 {
    let a_star = a_star.abs().clamp(0.0, 0.998);
    let l_edd = 4.0 * PI * G_CGS * mass_g * M_PROTON_CGS * C_CGS / SIGMA_THOMSON;

    let m_geo = G_CGS * mass_g / (C_CGS * C_CGS);
    let r_plus = m_geo * (1.0 + (1.0 - a_star * a_star).sqrt());
    let omega_h = a_star * C_CGS / (2.0 * r_plus);

    if omega_h.abs() < 1e-30 {
        return f64::INFINITY;
    }

    let kappa = 0.05;
    let b_sq = l_edd * C_CGS / (kappa * PI * r_plus * r_plus * omega_h * omega_h);
    b_sq.sqrt()
}

// ============================================================================
// Superradiant scattering
// ============================================================================

/// Check if a wave frequency allows superradiant scattering.
///
/// Waves with omega < m * Omega_H can extract energy from a rotating BH.
///
/// Arguments:
///   omega: wave frequency [rad/s]
///   m_mode: azimuthal mode number
///   mass_g: black hole mass [g]
///   a_star: dimensionless spin
pub fn is_superradiant(omega: f64, m_mode: i32, mass_g: f64, a_star: f64) -> bool {
    let omega_h = horizon_angular_velocity_cgs(mass_g, a_star);
    omega < f64::from(m_mode) * omega_h
}

/// Superradiant amplification factor (approximate).
///
/// For scalar waves, maximum amplification is ~0.4% per scattering.
/// For gravitational waves, up to ~138% for extremal Kerr.
///
/// The exact factor requires solving the Teukolsky equation; this is
/// a simplified estimate proportional to (m*Omega_H - omega) / (m*Omega_H).
///
/// Returns 1.0 if no superradiance (omega >= m * Omega_H).
pub fn superradiant_amplification(omega: f64, m_mode: i32, mass_g: f64, a_star: f64) -> f64 {
    let a_star = a_star.abs().clamp(0.0, 0.998);
    let omega_h = horizon_angular_velocity_cgs(mass_g, a_star);
    let m_omega_h = f64::from(m_mode) * omega_h;

    if omega >= m_omega_h {
        return 1.0; // no superradiance
    }

    let sr_factor = (m_omega_h - omega) / m_omega_h;
    1.0 + 0.05 * sr_factor * a_star * a_star
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_penrose_efficiency_schwarzschild() {
        // No ergosphere -> zero efficiency
        let eta = penrose_maximum_efficiency(0.0);
        assert!(eta.abs() < TOL, "eta = {eta}");
    }

    #[test]
    fn test_penrose_efficiency_extremal() {
        // a* = 1: eta = 1 - 1/sqrt(2) ~ 0.2929
        let eta = penrose_maximum_efficiency(1.0);
        let expected = 1.0 - 1.0 / 2.0_f64.sqrt();
        assert!((eta - expected).abs() < TOL, "eta = {eta}");
    }

    #[test]
    fn test_penrose_efficiency_moderate_spin() {
        // a* = 0.5: should be between 0 and 0.293
        let eta = penrose_maximum_efficiency(0.5);
        assert!(eta > 0.0 && eta < 0.293, "eta = {eta}");
    }

    #[test]
    fn test_penrose_efficiency_monotonic() {
        // Efficiency increases with spin
        let eta_low = penrose_maximum_efficiency(0.3);
        let eta_high = penrose_maximum_efficiency(0.9);
        assert!(eta_high > eta_low);
    }

    #[test]
    fn test_irreducible_mass_schwarzschild() {
        // For a*=0: M_irr = M
        let m_irr = irreducible_mass_cgs(M_SUN_CGS, 0.0);
        assert!((m_irr - M_SUN_CGS).abs() / M_SUN_CGS < TOL);
    }

    #[test]
    fn test_irreducible_mass_extremal() {
        // For a*=1: M_irr = M/sqrt(2) ~ 0.707 M
        let m_irr = irreducible_mass_cgs(M_SUN_CGS, 1.0);
        let expected = M_SUN_CGS / 2.0_f64.sqrt();
        assert!((m_irr - expected).abs() / expected < TOL, "M_irr = {m_irr}");
    }

    #[test]
    fn test_rotational_energy_schwarzschild() {
        // No rotation -> no rotational energy
        let e_rot = rotational_energy(M_SUN_CGS, 0.0);
        assert!(e_rot.abs() < 1e20, "E_rot = {e_rot}"); // tiny numerical residual
    }

    #[test]
    fn test_rotational_energy_extremal() {
        // E_rot / Mc^2 ~ 0.293 for a*=1
        let e_rot = rotational_energy(M_SUN_CGS, 1.0);
        let mc2 = M_SUN_CGS * C_CGS * C_CGS;
        let frac = e_rot / mc2;
        let expected = 1.0 - 1.0 / 2.0_f64.sqrt();
        assert!((frac - expected).abs() < 1e-6, "frac = {frac}");
    }

    #[test]
    fn test_bz_power_positive() {
        let p = blandford_znajek_power(M_SUN_CGS, 0.9, 1e4);
        assert!(p > 0.0, "P_BZ should be positive");
    }

    #[test]
    fn test_bz_power_zero_spin() {
        // No spin -> no BZ power
        let p = blandford_znajek_power(M_SUN_CGS, 0.0, 1e4);
        assert!(p.abs() < 1e-20, "P_BZ should vanish for a*=0");
    }

    #[test]
    fn test_bz_power_scales_with_b_squared() {
        let p1 = blandford_znajek_power(M_SUN_CGS, 0.9, 1e4);
        let p2 = blandford_znajek_power(M_SUN_CGS, 0.9, 2e4);
        assert!((p2 / p1 - 4.0).abs() < 0.01, "BZ should scale as B^2");
    }

    #[test]
    fn test_bz_eddington_field_scales_with_mass() {
        // P_BZ = kappa*pi*B^2*a*^2*c/4 is mass-independent at fixed a* and B.
        // L_Edd ~ M. So B_Edd ~ sqrt(M): bigger BHs need MORE field to
        // reach Eddington because L_Edd grows while P_BZ does not.
        let b10 = bz_eddington_field(10.0 * M_SUN_CGS, 0.9);
        let b100 = bz_eddington_field(100.0 * M_SUN_CGS, 0.9);
        assert!(b100 > b10, "B_Edd should increase with mass");
        // Check sqrt(M) scaling: b100/b10 ~ sqrt(10) ~ 3.16
        let ratio = b100 / b10;
        assert!(
            (ratio - 10.0_f64.sqrt()).abs() < 0.1,
            "ratio = {ratio}, expected {}",
            10.0_f64.sqrt()
        );
        assert!(b10 > 0.0 && b10.is_finite());
    }

    #[test]
    fn test_superradiance_condition() {
        let omega_h = horizon_angular_velocity_cgs(M_SUN_CGS, 0.9);
        // omega < m * Omega_H -> superradiant
        assert!(is_superradiant(0.5 * omega_h, 1, M_SUN_CGS, 0.9));
        // omega > m * Omega_H -> not superradiant
        assert!(!is_superradiant(2.0 * omega_h, 1, M_SUN_CGS, 0.9));
    }

    #[test]
    fn test_superradiant_amplification_no_superradiance() {
        let omega_h = horizon_angular_velocity_cgs(M_SUN_CGS, 0.9);
        let amp = superradiant_amplification(2.0 * omega_h, 1, M_SUN_CGS, 0.9);
        assert!((amp - 1.0).abs() < TOL, "should be 1.0 (no amplification)");
    }

    #[test]
    fn test_superradiant_amplification_boost() {
        let omega_h = horizon_angular_velocity_cgs(M_SUN_CGS, 0.9);
        let amp = superradiant_amplification(0.1 * omega_h, 1, M_SUN_CGS, 0.9);
        assert!(amp > 1.0, "should amplify: {amp}");
    }

    #[test]
    fn test_horizon_angular_velocity_schwarzschild() {
        // a*=0: Omega_H = 0
        let omega = horizon_angular_velocity_cgs(M_SUN_CGS, 0.0);
        assert!(omega.abs() < 1e-20);
    }
}
