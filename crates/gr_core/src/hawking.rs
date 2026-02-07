//! Hawking radiation and black hole thermodynamics.
//!
//! Hawking (1974) showed that black holes emit thermal radiation with
//! temperature inversely proportional to their mass. Combined with
//! Bekenstein's entropy formula, this establishes black hole thermodynamics.
//!
//! All functions take mass in grams (CGS) unless otherwise noted.
//!
//! Key formulas:
//!   Temperature:      T_H  = hbar c^3 / (8 pi G M k_B)
//!   Luminosity:       L_H  = hbar c^6 / (15360 pi G^2 M^2)
//!   Entropy:          S_BH = k_B c^3 A / (4 G hbar)
//!   Evaporation time: t_evap = 5120 pi G^2 M^3 / (hbar c^4)
//!
//! References:
//!   - Hawking (1974): "Black hole explosions?" Nature 248, 30
//!   - Bekenstein (1973): "Black Holes and Entropy" PRD 7, 2333
//!   - Page (1976): "Particle emission rates from a black hole"

use crate::constants::*;
use std::f64::consts::PI;

// ============================================================================
// Planck scale constants
// ============================================================================

/// Planck mass: m_P = sqrt(hbar c / G) [g]
pub const M_PLANCK: f64 = 2.176_434e-5; // sqrt(1.0546e-27 * 2.998e10 / 6.674e-8)

/// Planck length: l_P = sqrt(hbar G / c^3) [cm]
pub const L_PLANCK: f64 = 1.616_255e-33;

/// Planck time: t_P = sqrt(hbar G / c^5) [s]
pub const T_PLANCK_TIME: f64 = 5.391_247e-44;

/// Planck temperature: T_P = sqrt(hbar c^5 / (G k_B^2)) [K]
pub const T_PLANCK_TEMP: f64 = 1.416_784e32;

/// Wien displacement constant [cm K]
const B_WIEN: f64 = 0.289_8;

/// Hubble time (age of the universe) [s]
const HUBBLE_TIME: f64 = 4.35e17;

// ============================================================================
// Hawking temperature
// ============================================================================

/// Hawking temperature for a Schwarzschild black hole.
///
/// T_H = hbar c^3 / (8 pi G M k_B)
///
/// For 1 solar mass: T_H ~ 6.2e-8 K (extremely cold).
/// Becomes significant only for M < ~1e15 g (asteroid mass).
pub fn hawking_temperature(mass_g: f64) -> f64 {
    if mass_g <= 0.0 {
        return f64::INFINITY;
    }
    HBAR_CGS * C_CGS * C_CGS * C_CGS / (8.0 * PI * G_CGS * mass_g * K_B_CGS)
}

/// Hawking temperature for a Kerr black hole.
///
/// T_H = hbar kappa / (2 pi k_B c)
///
/// where kappa = (r+ - r-) / (2(r+^2 + a^2)) is the surface gravity
/// in geometric units, converted to CGS via c^2.
///
/// For extremal Kerr (a* = 1): T_H = 0 (third law of BH thermodynamics).
pub fn hawking_temperature_kerr(mass_g: f64, a_star: f64) -> f64 {
    if mass_g <= 0.0 {
        return f64::INFINITY;
    }
    let a_star = a_star.abs().clamp(0.0, 0.9999);
    let m_geo = G_CGS * mass_g / (C_CGS * C_CGS); // geometric mass [cm]
    let sqrt_f = (1.0 - a_star * a_star).sqrt();
    let r_plus = m_geo * (1.0 + sqrt_f);
    let r_minus = m_geo * (1.0 - sqrt_f);
    let a = a_star * m_geo;

    // Surface gravity in geometric units [1/cm]
    let kappa_geo = (r_plus - r_minus) / (2.0 * (r_plus * r_plus + a * a));
    // Convert to CGS [cm/s^2]
    let kappa = kappa_geo * C_CGS * C_CGS;

    HBAR_CGS * kappa / (2.0 * PI * K_B_CGS * C_CGS)
}

// ============================================================================
// Luminosity and evaporation
// ============================================================================

/// Hawking luminosity (power radiated) for Schwarzschild.
///
/// L_H = hbar c^6 / (15360 pi G^2 M^2) [erg/s]
///
/// This is the Stefan-Boltzmann radiation from a sphere of radius r_s.
pub fn hawking_luminosity(mass_g: f64) -> f64 {
    if mass_g <= 0.0 {
        return f64::INFINITY;
    }
    let c6 = C_CGS.powi(6);
    HBAR_CGS * c6 / (15360.0 * PI * G_CGS * G_CGS * mass_g * mass_g)
}

/// Black hole evaporation time.
///
/// t_evap = 5120 pi G^2 M^3 / (hbar c^4) [s]
///
/// For 1 solar mass: t_evap ~ 10^67 years.
/// For M ~ 1e15 g: t_evap ~ age of universe.
pub fn hawking_evaporation_time(mass_g: f64) -> f64 {
    if mass_g <= 0.0 {
        return 0.0;
    }
    let c4 = C_CGS.powi(4);
    5120.0 * PI * G_CGS * G_CGS * mass_g * mass_g * mass_g / (HBAR_CGS * c4)
}

/// Mass at which evaporation time equals given time.
///
/// M = (hbar c^4 t / (5120 pi G^2))^{1/3} [g]
pub fn evaporating_mass(t_s: f64) -> f64 {
    if t_s <= 0.0 {
        return 0.0;
    }
    let c4 = C_CGS.powi(4);
    (HBAR_CGS * c4 * t_s / (5120.0 * PI * G_CGS * G_CGS)).cbrt()
}

/// Mass loss rate dM/dt = -L_H / c^2 [g/s] (negative).
pub fn mass_loss_rate(mass_g: f64) -> f64 {
    -hawking_luminosity(mass_g) / (C_CGS * C_CGS)
}

// ============================================================================
// Hawking spectrum
// ============================================================================

/// Peak wavelength of Hawking radiation (Wien displacement).
///
/// lambda_peak = b / T_H [cm]
pub fn hawking_peak_wavelength(mass_g: f64) -> f64 {
    let t = hawking_temperature(mass_g);
    if t <= 0.0 || t.is_infinite() {
        return f64::INFINITY;
    }
    B_WIEN / t
}

/// Peak frequency of Hawking radiation.
///
/// nu_peak = 2.821 k_B T / h [Hz]
pub fn hawking_peak_frequency(mass_g: f64) -> f64 {
    let t = hawking_temperature(mass_g);
    2.821 * K_B_CGS * t / (2.0 * PI * HBAR_CGS)
}

/// Hawking radiation power spectrum (simplified blackbody).
///
/// dE/(dt dnu) = area * pi * B_nu [erg/s/Hz]
///
/// where B_nu is the Planck function evaluated at T_H.
/// Greybody factors are omitted (would require solving the
/// Teukolsky equation for each angular momentum mode).
pub fn hawking_spectrum(nu: f64, mass_g: f64) -> f64 {
    let t = hawking_temperature(mass_g);
    if t <= 0.0 || nu <= 0.0 {
        return 0.0;
    }
    let x = 2.0 * PI * HBAR_CGS * nu / (K_B_CGS * t);
    if x > 700.0 {
        return 0.0; // avoid overflow in exp
    }
    let r_s = 2.0 * G_CGS * mass_g / (C_CGS * C_CGS);
    let area = 4.0 * PI * r_s * r_s;
    // Planck spectral radiance B_nu = 2 h nu^3 / c^2 / (exp(h nu / kT) - 1)
    let b_nu = 2.0 * 2.0 * PI * HBAR_CGS * nu * nu * nu / (C_CGS * C_CGS)
        / (x.exp() - 1.0);
    area * PI * b_nu
}

// ============================================================================
// Bekenstein-Hawking entropy
// ============================================================================

/// Bekenstein-Hawking entropy for Schwarzschild.
///
/// S = k_B c^3 A / (4 G hbar) [erg/K]
///
/// where A = 4 pi r_s^2 is the horizon area.
pub fn bekenstein_hawking_entropy(mass_g: f64) -> f64 {
    let r_s = 2.0 * G_CGS * mass_g / (C_CGS * C_CGS);
    let area = 4.0 * PI * r_s * r_s;
    K_B_CGS * C_CGS * C_CGS * C_CGS * area / (4.0 * G_CGS * HBAR_CGS)
}

/// Dimensionless entropy S / k_B = A / (4 l_P^2).
pub fn entropy_dimensionless(mass_g: f64) -> f64 {
    let r_s = 2.0 * G_CGS * mass_g / (C_CGS * C_CGS);
    let area = 4.0 * PI * r_s * r_s;
    area / (4.0 * L_PLANCK * L_PLANCK)
}

/// Bekenstein-Hawking entropy for Kerr.
///
/// S = k_B c^3 A / (4 G hbar) where A = 4 pi (r+^2 + a^2) for Kerr.
pub fn bekenstein_hawking_entropy_kerr(mass_g: f64, a_star: f64) -> f64 {
    let a_star = a_star.abs().clamp(0.0, 0.9999);
    let m_geo = G_CGS * mass_g / (C_CGS * C_CGS);
    let sqrt_f = (1.0 - a_star * a_star).sqrt();
    let r_plus = m_geo * (1.0 + sqrt_f);
    let a = a_star * m_geo;
    let area = 4.0 * PI * (r_plus * r_plus + a * a);
    K_B_CGS * C_CGS * C_CGS * C_CGS * area / (4.0 * G_CGS * HBAR_CGS)
}

// ============================================================================
// Information paradox timescales
// ============================================================================

/// Page time: when entanglement entropy of radiation peaks.
///
/// t_Page ~ t_evap / 2^{2/3} ~ 0.63 * t_evap [s]
///
/// After Page time, information should start being recovered.
pub fn page_time(mass_g: f64) -> f64 {
    hawking_evaporation_time(mass_g) / 2.0_f64.powf(2.0 / 3.0)
}

/// Scrambling time: how long for thrown-in information to be scrambled.
///
/// t_scr = (r_s / c) * ln(S / k_B) [s]
///
/// Fast compared to Page time, slow compared to light-crossing time.
pub fn scrambling_time(mass_g: f64) -> f64 {
    let r_s = 2.0 * G_CGS * mass_g / (C_CGS * C_CGS);
    let s = entropy_dimensionless(mass_g).max(1.0);
    (r_s / C_CGS) * s.ln()
}

/// Check if a black hole is "primordial" (evaporating within ~Hubble time).
///
/// A primordial black hole with t_evap ~ age of universe has
/// mass ~ 5e14 g ~ 1e-19 M_sun.
pub fn is_primordial_evaporating(mass_g: f64) -> bool {
    let t_evap = hawking_evaporation_time(mass_g);
    t_evap < 10.0 * HUBBLE_TIME && t_evap > 0.1 * HUBBLE_TIME
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planck_constants() {
        // Planck mass ~ 2.18e-5 g
        assert!((M_PLANCK - 2.18e-5).abs() / 2.18e-5 < 0.01);
        // Planck length ~ 1.62e-33 cm
        assert!((L_PLANCK - 1.62e-33).abs() / 1.62e-33 < 0.01);
        // Planck time ~ 5.39e-44 s
        assert!((T_PLANCK_TIME - 5.39e-44).abs() / 5.39e-44 < 0.01);
    }

    #[test]
    fn test_hawking_temp_solar_mass() {
        let t = hawking_temperature(M_SUN_CGS);
        // T_H ~ 6.2e-8 K for 1 solar mass
        assert!(t > 5e-8 && t < 7e-8, "T_H = {t}");
    }

    #[test]
    fn test_hawking_temp_inversely_proportional() {
        let t1 = hawking_temperature(M_SUN_CGS);
        let t10 = hawking_temperature(10.0 * M_SUN_CGS);
        assert!((t1 / t10 - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_hawking_temp_kerr_schwarzschild_limit() {
        // At a*=0, Kerr temperature should match Schwarzschild
        let t_schw = hawking_temperature(M_SUN_CGS);
        let t_kerr = hawking_temperature_kerr(M_SUN_CGS, 0.0);
        assert!(
            (t_schw - t_kerr).abs() / t_schw < 0.01,
            "Schw={t_schw}, Kerr={t_kerr}"
        );
    }

    #[test]
    fn test_hawking_temp_kerr_extremal_cold() {
        // Near-extremal Kerr should be much colder than Schwarzschild
        let t_schw = hawking_temperature(M_SUN_CGS);
        let t_kerr = hawking_temperature_kerr(M_SUN_CGS, 0.999);
        assert!(t_kerr < 0.1 * t_schw, "extremal not cold enough: {t_kerr}");
    }

    #[test]
    fn test_hawking_luminosity_solar() {
        let l = hawking_luminosity(M_SUN_CGS);
        // L_H ~ 9e-22 erg/s for 1 solar mass (Stefan-Boltzmann from horizon area)
        assert!(l > 1e-23 && l < 1e-20, "L_H = {l}");
    }

    #[test]
    fn test_evaporation_time_solar() {
        let t = hawking_evaporation_time(M_SUN_CGS);
        // t_evap ~ 2e67 years ~ 6e74 s for 1 solar mass
        assert!(t > 1e73 && t < 1e76, "t_evap = {t}");
    }

    #[test]
    fn test_evaporation_time_scales_cubic() {
        let t1 = hawking_evaporation_time(M_SUN_CGS);
        let t2 = hawking_evaporation_time(2.0 * M_SUN_CGS);
        // M^3 scaling: t(2M)/t(M) = 8
        assert!((t2 / t1 - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_evaporating_mass_round_trip() {
        let m = 1e15; // grams (asteroid mass)
        let t = hawking_evaporation_time(m);
        let m_back = evaporating_mass(t);
        assert!((m_back - m).abs() / m < 1e-10);
    }

    #[test]
    fn test_mass_loss_rate_negative() {
        let dm = mass_loss_rate(M_SUN_CGS);
        assert!(dm < 0.0);
    }

    #[test]
    fn test_entropy_solar_mass() {
        let s = entropy_dimensionless(M_SUN_CGS);
        // S/k_B ~ 1e77 for 1 solar mass
        assert!(s > 1e76 && s < 1e79, "S/k_B = {s}");
    }

    #[test]
    fn test_entropy_kerr_schwarzschild_limit() {
        let s_schw = bekenstein_hawking_entropy(M_SUN_CGS);
        let s_kerr = bekenstein_hawking_entropy_kerr(M_SUN_CGS, 0.0);
        assert!(
            (s_schw - s_kerr).abs() / s_schw < 0.01,
            "Schw={s_schw}, Kerr={s_kerr}"
        );
    }

    #[test]
    fn test_entropy_kerr_decreases_with_spin() {
        // Higher spin = smaller horizon area = less entropy
        let s0 = bekenstein_hawking_entropy_kerr(M_SUN_CGS, 0.0);
        let s9 = bekenstein_hawking_entropy_kerr(M_SUN_CGS, 0.9);
        assert!(s9 < s0, "spinning BH should have less entropy");
    }

    #[test]
    fn test_page_time_fraction() {
        let t_evap = hawking_evaporation_time(M_SUN_CGS);
        let t_page = page_time(M_SUN_CGS);
        let ratio = t_page / t_evap;
        // 1/2^{2/3} ~ 0.63
        assert!((ratio - 0.63).abs() < 0.01, "ratio = {ratio}");
    }

    #[test]
    fn test_scrambling_time_shorter_than_page() {
        let t_scr = scrambling_time(M_SUN_CGS);
        let t_page = page_time(M_SUN_CGS);
        assert!(
            t_scr < t_page,
            "scrambling should be much shorter than Page time"
        );
    }

    #[test]
    fn test_primordial_evaporating() {
        // Mass ~ 2e14 g evaporates in ~ Hubble time
        // (critical mass ~ 1.7e14 g where t_evap = t_Hubble)
        assert!(is_primordial_evaporating(2e14));
        // Solar mass does NOT evaporate in Hubble time
        assert!(!is_primordial_evaporating(M_SUN_CGS));
    }

    #[test]
    fn test_peak_wavelength_inversely_proportional() {
        let lam1 = hawking_peak_wavelength(M_SUN_CGS);
        let lam2 = hawking_peak_wavelength(2.0 * M_SUN_CGS);
        // lambda_peak ~ T^{-1} ~ M, so doubling mass doubles wavelength
        assert!((lam2 / lam1 - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_spectrum_positive() {
        let nu = 1e10; // 10 GHz
        let power = hawking_spectrum(nu, 1e12); // small BH (hot)
        assert!(power > 0.0, "spectrum should be positive: {power}");
    }

    #[test]
    fn test_spectrum_zero_outside_range() {
        // Very high frequency for solar mass BH should be zero (exponential cutoff)
        let power = hawking_spectrum(1e30, M_SUN_CGS);
        assert!(power == 0.0 || power < 1e-300);
    }
}
