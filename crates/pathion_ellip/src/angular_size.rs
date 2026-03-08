//! EHT angular size computation for black hole shadows and photon sub-rings.
//!
//! Provides Schwarzschild-radius, shadow-diameter, and sub-ring separation
//! formulae calibrated against Event Horizon Telescope observations (M87*).

/// Solar mass in kilograms.
pub const M_SUN_KG: f64 = 1.98892e30;

/// Gravitational constant in SI (m^3 kg^-1 s^-2).
pub const G_SI: f64 = 6.67430e-11;

/// Speed of light in SI (m/s).
pub const C_SI: f64 = 2.99792458e8;

/// One parsec in meters.
pub const PC_M: f64 = 3.08567758e16;

/// One megaparsec in meters.
pub const MPC_M: f64 = PC_M * 1e6;

/// Microarcseconds per radian.
pub const UAS_PER_RAD: f64 = 206_264_806_247.0;

/// Schwarzschild radius r_s = 2GM/c^2 in meters.
pub fn schwarzschild_radius_m(mass_kg: f64) -> f64 {
    2.0 * G_SI * mass_kg / (C_SI * C_SI)
}

/// Angular size of the photon sphere shadow in microarcseconds.
///
/// Shadow radius = 3*sqrt(3)/2 * r_s (exact for Schwarzschild).
/// For Kerr, this is the spin-averaged approximation.
///
/// The full angular diameter is twice the shadow radius divided by the
/// luminosity distance, converted to microarcseconds.
pub fn shadow_angular_size_uas(mass_solar: f64, distance_mpc: f64) -> f64 {
    let mass_kg = mass_solar * M_SUN_KG;
    let rs = schwarzschild_radius_m(mass_kg);
    // Shadow radius in the image plane (linear size at the source).
    let shadow_radius = 3.0 * 3.0_f64.sqrt() / 2.0 * rs;
    // Angular diameter = 2 * shadow_radius / distance.
    let distance_m = distance_mpc * MPC_M;
    let angular_diameter_rad = 2.0 * shadow_radius / distance_m;
    angular_diameter_rad * UAS_PER_RAD
}

/// Sub-ring angular separation for the n-th photon ring.
///
/// Successive photon rings are separated by exp(-pi) in angular size.
/// Returns the separation between ring `n` and ring `n+1` in
/// microarcseconds.
pub fn sub_ring_separation_uas(mass_solar: f64, distance_mpc: f64, n: u32) -> f64 {
    let shadow = shadow_angular_size_uas(mass_solar, distance_mpc);
    // Each successive ring is demagnified by exp(-pi).
    shadow * (-std::f64::consts::PI * f64::from(n)).exp()
}

/// EHT M87* parameters: mass 6.5e9 Msun, distance 16.8 Mpc.
pub fn m87_star_shadow_uas() -> f64 {
    shadow_angular_size_uas(6.5e9, 16.8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schwarzschild_radius_sun() {
        let rs = schwarzschild_radius_m(M_SUN_KG);
        // ~2953 meters
        assert!((rs - 2953.25).abs() < 1.0);
    }

    #[test]
    fn test_m87_shadow_size() {
        let uas = m87_star_shadow_uas();
        // EHT measured ~42 uas; Schwarzschild prediction ~39.8 uas
        assert!(uas > 35.0 && uas < 45.0, "M87* shadow = {:.1} uas", uas);
    }

    #[test]
    fn test_sub_ring_exponential_decay() {
        let s1 = sub_ring_separation_uas(6.5e9, 16.8, 1);
        let s2 = sub_ring_separation_uas(6.5e9, 16.8, 2);
        let ratio = s1 / s2;
        // Should be approximately exp(pi) ~ 23.14
        assert!(
            (ratio - std::f64::consts::PI.exp()).abs() < 1.0,
            "Ring separation ratio = {:.2}, expected ~23.14",
            ratio
        );
    }
}
