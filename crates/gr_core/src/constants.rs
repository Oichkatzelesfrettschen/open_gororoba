//! Astrophysical constants and unit conversions for GR computations.
//!
//! Provides constants in both CGS and natural (G=c=1) units, plus
//! astrophysical conversion factors commonly needed in black hole
//! and neutron star physics.
//!
//! Sources:
//!   - CODATA 2018 / NIST 2024 recommended values
//!   - IAU 2015 nominal solar parameters (Prsa et al. 2016)
//!   - Particle Data Group 2024

// -- Fundamental constants (CGS) --

/// Speed of light [cm/s]
pub const C_CGS: f64 = 2.997_924_58e10;

/// Gravitational constant [cm^3 g^-1 s^-2]
pub const G_CGS: f64 = 6.674_30e-8;

/// Reduced Planck constant [erg s]
pub const HBAR_CGS: f64 = 1.054_571_817e-27;

/// Boltzmann constant [erg K^-1]
pub const K_B_CGS: f64 = 1.380_649e-16;

/// Stefan-Boltzmann constant [erg cm^-2 s^-1 K^-4]
pub const SIGMA_SB_CGS: f64 = 5.670_374_419e-5;

/// Proton mass [g]
pub const M_PROTON_CGS: f64 = 1.672_621_924e-24;

/// Electron mass [g]
pub const M_ELECTRON_CGS: f64 = 9.109_383_702e-28;

/// Electron charge [esu = statcoulomb]
pub const E_CHARGE_CGS: f64 = 4.803_204_7e-10;

/// Thomson cross section [cm^2]
pub const SIGMA_THOMSON: f64 = 6.652_458_732e-25;

// -- Solar parameters (IAU 2015 nominal) --

/// Solar mass [g]
pub const M_SUN_CGS: f64 = 1.988_41e33;

/// Solar radius [cm]
pub const R_SUN_CGS: f64 = 6.957e10;

/// Solar luminosity [erg/s]
pub const L_SUN_CGS: f64 = 3.828e33;

// -- Distance units --

/// Parsec [cm]
pub const PARSEC_CGS: f64 = 3.085_677_581e18;

/// Megaparsec [cm]
pub const MPC_CGS: f64 = 3.085_677_581e24;

/// Astronomical unit [cm]
pub const AU_CGS: f64 = 1.495_978_707e13;

// -- Derived GR quantities --

/// Schwarzschild radius per solar mass: r_s = 2GM/c^2 [cm]
pub const R_SCHW_PER_MSUN: f64 = 2.0 * G_CGS * M_SUN_CGS / (C_CGS * C_CGS);

/// Gravitational radius per solar mass: r_g = GM/c^2 [cm]
pub const R_GRAV_PER_MSUN: f64 = G_CGS * M_SUN_CGS / (C_CGS * C_CGS);

/// ISCO radius for Schwarzschild: 6 * r_g = 6 GM/c^2 [cm per solar mass]
pub const R_ISCO_SCHW_PER_MSUN: f64 = 6.0 * R_GRAV_PER_MSUN;

/// Time unit per solar mass: t_g = GM/c^3 [s]
pub const T_GRAV_PER_MSUN: f64 = G_CGS * M_SUN_CGS / (C_CGS * C_CGS * C_CGS);

/// Hawking temperature per solar mass: T_H = hbar c^3 / (8 pi G M k_B) [K]
pub const T_HAWKING_PER_MSUN: f64 =
    HBAR_CGS * C_CGS * C_CGS * C_CGS
    / (8.0 * std::f64::consts::PI * G_CGS * M_SUN_CGS * K_B_CGS);

/// Eddington luminosity per solar mass: L_Edd = 4 pi G M m_p c / sigma_T [erg/s]
pub const L_EDDINGTON_PER_MSUN: f64 =
    4.0 * std::f64::consts::PI * G_CGS * M_SUN_CGS * M_PROTON_CGS * C_CGS / SIGMA_THOMSON;

// -- Conversion utilities --

/// Convert mass in solar masses to gravitational radius [cm].
pub fn mass_to_r_grav(m_solar: f64) -> f64 {
    m_solar * R_GRAV_PER_MSUN
}

/// Convert mass in solar masses to Schwarzschild radius [cm].
pub fn mass_to_r_schw(m_solar: f64) -> f64 {
    m_solar * R_SCHW_PER_MSUN
}

/// Convert mass in solar masses to Hawking temperature [K].
pub fn mass_to_hawking_temp(m_solar: f64) -> f64 {
    T_HAWKING_PER_MSUN / m_solar
}

/// Convert mass in solar masses to Eddington luminosity [erg/s].
pub fn mass_to_eddington_luminosity(m_solar: f64) -> f64 {
    m_solar * L_EDDINGTON_PER_MSUN
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schwarzschild_radius() {
        // r_s for 1 solar mass ~ 2.95 km
        let r_s = mass_to_r_schw(1.0);
        assert!((r_s - 2.953e5).abs() / 2.953e5 < 0.01);
    }

    #[test]
    fn test_gravitational_radius() {
        // r_g = r_s / 2
        let r_g = mass_to_r_grav(1.0);
        let r_s = mass_to_r_schw(1.0);
        assert!((r_g - r_s / 2.0).abs() < 1.0);
    }

    #[test]
    fn test_hawking_temperature() {
        // T_H for 1 solar mass ~ 6.2e-8 K (extremely cold)
        let t_h = mass_to_hawking_temp(1.0);
        assert!(t_h > 5e-8 && t_h < 7e-8);
    }

    #[test]
    fn test_eddington_luminosity() {
        // L_Edd for 1 solar mass ~ 1.26e38 erg/s
        let l_edd = mass_to_eddington_luminosity(1.0);
        assert!((l_edd - 1.26e38).abs() / 1.26e38 < 0.02);
    }

    #[test]
    fn test_eddington_scales_linearly() {
        let l1 = mass_to_eddington_luminosity(1.0);
        let l10 = mass_to_eddington_luminosity(10.0);
        assert!((l10 / l1 - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_hawking_inversely_proportional() {
        let t1 = mass_to_hawking_temp(1.0);
        let t10 = mass_to_hawking_temp(10.0);
        assert!((t1 / t10 - 10.0).abs() < 1e-10);
    }
}
