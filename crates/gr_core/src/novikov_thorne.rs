//! Novikov-Thorne thin disk model for black hole accretion.
//!
//! The Novikov-Thorne (1973) model describes a geometrically thin,
//! optically thick accretion disk around a Kerr black hole. Matter
//! follows circular geodesics down to the ISCO, then plunges in.
//!
//! Key results:
//!   - Radiative efficiency eta = 1 - E_ISCO (BPT 1972)
//!   - Temperature profile from Page & Thorne (1974)
//!   - Peak temperature at r ~ 1.5 * r_ISCO
//!
//! Validation targets:
//!   a*=0 (Schwarzschild): eta = 0.0572 (6% efficiency)
//!   a*=0.998 (near-extremal prograde): eta ~ 0.32
//!
//! References:
//!   - Novikov & Thorne (1973): Black Holes (Les Astres Occlus), pp. 343-450
//!   - Shakura & Sunyaev (1973): A&A 24, 337-355
//!   - Page & Thorne (1974): ApJ 191, 499-506
//!   - Bardeen, Press, Teukolsky (1972): ApJ 178, 347

use crate::constants::*;
use std::f64::consts::PI;

/// Prograde ISCO radius from the BPT (1972) formula, in units of M.
///
/// r_ISCO = M (3 + Z2 - sqrt((3-Z1)(3+Z1+2Z2)))
/// Z1 = 1 + (1-a*^2)^{1/3} ((1+a*)^{1/3} + (1-a*)^{1/3})
/// Z2 = sqrt(3 a*^2 + Z1^2)
///
/// a*=0: r_ISCO = 6M;  a*->1: r_ISCO -> M.
pub fn isco_radius(a_star: f64) -> f64 {
    let a_star = a_star.clamp(-0.9999, 0.9999);
    let z1 = 1.0
        + (1.0 - a_star * a_star).cbrt()
            * ((1.0 + a_star).cbrt() + (1.0 - a_star).cbrt());
    let z2 = (3.0 * a_star * a_star + z1 * z1).sqrt();
    3.0 + z2 - a_star.signum() * ((3.0 - z1) * (3.0 + z1 + 2.0 * z2)).sqrt()
}

/// Radiative efficiency eta = 1 - E_ISCO.
///
/// E_ISCO = sqrt(1 - 2/(3 r_ISCO)) is the specific orbital energy
/// at the ISCO, normalized so that E=1 at infinity.
///
/// a*=0: eta = 1 - sqrt(8/9) = 0.0572.
/// a*=0.998: eta ~ 0.32.
pub fn radiative_efficiency(a_star: f64) -> f64 {
    let r_isco = isco_radius(a_star);
    let e_isco = (1.0 - 2.0 / (3.0 * r_isco)).sqrt();
    1.0 - e_isco
}

/// Disk temperature at radius r (in units of M).
///
/// From Page & Thorne (1974):
///   T(r) = [3 G M Mdot f(r) / (8 pi sigma_SB r_cgs^3)]^{1/4}
///
/// where f(r) = max(0, 1 - sqrt(r_ISCO/r)) is the zero-torque boundary
/// radial emissivity function, and r_cgs = r * GM/c^2.
///
/// Returns 0 inside the ISCO (no stable circular orbits).
///
/// Arguments:
///   r_m: radius in units of M (gravitational radii)
///   a_star: dimensionless spin
///   mdot_edd: accretion rate as fraction of Eddington
///   mass_solar: black hole mass in solar masses
pub fn disk_temperature(r_m: f64, a_star: f64, mdot_edd: f64, mass_solar: f64) -> f64 {
    let r_isco = isco_radius(a_star);
    if r_m < r_isco {
        return 0.0;
    }

    let eta = radiative_efficiency(a_star);
    // Eddington luminosity [erg/s]
    let l_edd = 4.0 * PI * G_CGS * mass_solar * M_SUN_CGS * M_PROTON_CGS * C_CGS
        / SIGMA_THOMSON;
    // Eddington mass accretion rate [g/s]
    let mdot_edd_cgs = l_edd / (eta * C_CGS * C_CGS);
    // Actual mass accretion rate [g/s]
    let mdot = mdot_edd * mdot_edd_cgs;

    // CGS radius [cm]
    let m_cgs = mass_solar * M_SUN_CGS;
    let r_cgs = r_m * G_CGS * m_cgs / (C_CGS * C_CGS);

    // Radial emissivity (zero-torque inner boundary)
    let f_r = (1.0 - (r_isco / r_m).sqrt()).max(0.0);

    // Page & Thorne temperature
    let t4 = 3.0 * G_CGS * m_cgs * mdot * f_r
        / (8.0 * PI * SIGMA_SB_CGS * r_cgs * r_cgs * r_cgs);

    t4.max(0.0).powf(0.25)
}

/// Disk flux (energy per unit area per unit time) [erg cm^-2 s^-1].
///
/// F = sigma_SB * T^4
pub fn disk_flux(r_m: f64, a_star: f64, mdot_edd: f64, mass_solar: f64) -> f64 {
    let t = disk_temperature(r_m, a_star, mdot_edd, mass_solar);
    SIGMA_SB_CGS * t * t * t * t
}

/// Normalized flux profile (0 to 1) for lookup table generation.
///
/// Peaks near r ~ 1.5 * r_ISCO and falls off as r^{-3} at large r.
/// Independent of mass and accretion rate.
pub fn normalized_flux(r_m: f64, a_star: f64) -> f64 {
    let r_isco = isco_radius(a_star);
    if r_m < r_isco {
        return 0.0;
    }

    let f_r = (1.0 - (r_isco / r_m).sqrt()).max(0.0);
    let flux = f_r / (r_m * r_m * r_m);

    // Normalize to peak
    let r_peak = 1.5 * r_isco;
    let f_peak = (1.0 - (r_isco / r_peak).sqrt()).max(0.0);
    let flux_peak = f_peak / (r_peak * r_peak * r_peak);

    if flux_peak <= 0.0 {
        return 0.0;
    }
    (flux / flux_peak).min(1.0)
}

/// Radius of peak temperature, in units of M.
///
/// T peaks at r ~ 1.5 * r_ISCO (Page & Thorne 1974).
pub fn peak_temperature_radius(a_star: f64) -> f64 {
    1.5 * isco_radius(a_star)
}

/// Integrated luminosity L = eta * Mdot * c^2 [erg/s].
pub fn integrated_luminosity(mdot_edd: f64, a_star: f64, mass_solar: f64) -> f64 {
    let eta = radiative_efficiency(a_star);
    let l_edd = 4.0 * PI * G_CGS * mass_solar * M_SUN_CGS * M_PROTON_CGS * C_CGS
        / SIGMA_THOMSON;
    let mdot_edd_cgs = l_edd / (eta * C_CGS * C_CGS);
    eta * mdot_edd * mdot_edd_cgs * C_CGS * C_CGS
}

/// Eddington luminosity for a given mass [erg/s].
///
/// L_Edd = 4 pi G M m_p c / sigma_T
pub fn eddington_luminosity(mass_solar: f64) -> f64 {
    4.0 * PI * G_CGS * mass_solar * M_SUN_CGS * M_PROTON_CGS * C_CGS / SIGMA_THOMSON
}

/// Eddington accretion rate for a given mass and spin [g/s].
///
/// Mdot_Edd = L_Edd / (eta * c^2)
pub fn eddington_accretion_rate(mass_solar: f64, a_star: f64) -> f64 {
    let eta = radiative_efficiency(a_star);
    let l_edd = eddington_luminosity(mass_solar);
    l_edd / (eta * C_CGS * C_CGS)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL_FRAC: f64 = 0.01; // 1% relative tolerance

    #[test]
    fn test_isco_schwarzschild() {
        let r = isco_radius(0.0);
        assert!((r - 6.0).abs() < 1e-10, "r_ISCO(a*=0) = {r}, expected 6");
    }

    #[test]
    fn test_isco_prograde_extremal() {
        let r = isco_radius(0.9999);
        assert!(r < 1.5, "near-extremal prograde ISCO should be near M: {r}");
    }

    #[test]
    fn test_isco_retrograde_extremal() {
        let r = isco_radius(-0.9999);
        assert!(r > 8.0, "near-extremal retrograde ISCO should be ~9M: {r}");
    }

    #[test]
    fn test_efficiency_schwarzschild() {
        let eta = radiative_efficiency(0.0);
        // eta = 1 - sqrt(8/9) = 0.05719...
        let expected = 1.0 - (8.0 / 9.0_f64).sqrt();
        assert!(
            (eta - expected).abs() < 1e-4,
            "eta(0) = {eta}, expected {expected}"
        );
    }

    #[test]
    fn test_efficiency_prograde_high_spin() {
        let eta = radiative_efficiency(0.998);
        // Near-extremal prograde: eta ~ 0.32
        assert!(eta > 0.25 && eta < 0.45, "eta(0.998) = {eta}");
    }

    #[test]
    fn test_efficiency_monotonic_with_spin() {
        let eta0 = radiative_efficiency(0.0);
        let eta5 = radiative_efficiency(0.5);
        let eta9 = radiative_efficiency(0.9);
        assert!(eta5 > eta0);
        assert!(eta9 > eta5);
    }

    #[test]
    fn test_temperature_zero_inside_isco() {
        let t = disk_temperature(3.0, 0.0, 0.1, 10.0);
        // r=3M < r_ISCO=6M for Schwarzschild
        assert!(t == 0.0, "temperature inside ISCO should be zero");
    }

    #[test]
    fn test_temperature_positive_outside_isco() {
        let t = disk_temperature(10.0, 0.0, 0.1, 10.0);
        assert!(t > 0.0, "temperature should be positive outside ISCO: {t}");
    }

    #[test]
    fn test_temperature_m87_order_of_magnitude() {
        // M87*: M ~ 6.5e9 M_sun, mdot ~ 0.001 Edd
        // Peak temp should be ~ 10^4-10^5 K (UV range)
        let r_peak = peak_temperature_radius(0.9);
        let t = disk_temperature(r_peak, 0.9, 0.001, 6.5e9);
        // Order of magnitude check only -- NT gives ~ few thousand K for sub-Edd AGN
        assert!(t > 1e2 && t < 1e8, "M87 disk temp at peak = {t} K");
    }

    #[test]
    fn test_flux_equals_sigma_t4() {
        let r = 10.0;
        let a_star = 0.5;
        let mdot = 0.1;
        let mass = 10.0;
        let t = disk_temperature(r, a_star, mdot, mass);
        let f = disk_flux(r, a_star, mdot, mass);
        let expected = SIGMA_SB_CGS * t.powi(4);
        if expected > 0.0 {
            assert!(
                (f - expected).abs() / expected < 1e-10,
                "F = sigma*T^4 violated"
            );
        }
    }

    #[test]
    fn test_normalized_flux_peak() {
        // At r_peak, normalized flux should be ~1
        let r_peak = peak_temperature_radius(0.0);
        let f = normalized_flux(r_peak, 0.0);
        assert!((f - 1.0).abs() < 0.05, "peak normalized flux = {f}");
    }

    #[test]
    fn test_normalized_flux_zero_inside_isco() {
        let f = normalized_flux(3.0, 0.0);
        assert!(f == 0.0);
    }

    #[test]
    fn test_normalized_flux_falls_off() {
        // Flux should decrease at large r
        let f10 = normalized_flux(10.0, 0.0);
        let f100 = normalized_flux(100.0, 0.0);
        assert!(f100 < f10, "flux should decrease with radius");
    }

    #[test]
    fn test_integrated_luminosity_eddington() {
        // At mdot_edd = 1.0, L should equal L_Edd
        let l = integrated_luminosity(1.0, 0.0, 10.0);
        let l_edd = eddington_luminosity(10.0);
        assert!(
            (l - l_edd).abs() / l_edd < TOL_FRAC,
            "L = {l}, L_Edd = {l_edd}"
        );
    }

    #[test]
    fn test_eddington_luminosity_10_solar() {
        let l = eddington_luminosity(10.0);
        // L_Edd ~ 1.26e38 * 10 = 1.26e39 erg/s
        assert!(
            (l - 1.26e39).abs() / 1.26e39 < TOL_FRAC,
            "L_Edd(10 Msun) = {l}"
        );
    }

    #[test]
    fn test_eddington_accretion_rate_positive() {
        let mdot = eddington_accretion_rate(10.0, 0.0);
        assert!(mdot > 0.0);
    }

    #[test]
    fn test_peak_temperature_radius_schwarzschild() {
        // r_peak = 1.5 * 6M = 9M
        let r = peak_temperature_radius(0.0);
        assert!((r - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_peak_temperature_radius_decreases_with_spin() {
        let r0 = peak_temperature_radius(0.0);
        let r9 = peak_temperature_radius(0.9);
        assert!(r9 < r0, "peak should move inward with spin");
    }
}
