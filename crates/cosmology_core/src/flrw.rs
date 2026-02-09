//! Flat Lambda-CDM (FLRW) cosmology with struct-based interface.
//!
//! Provides the standard Friedmann-Lemaitre-Robertson-Walker cosmological
//! model in a flat (k=0) universe with matter and a cosmological constant.
//! The `FlatLCDM` struct encapsulates cosmological parameters and provides
//! methods for computing distances, volumes, and derived quantities.
//!
//! Derived from Rocq formalization:
//!   - rocq/theories/Cosmology/FLRW.v
//!   - rocq/theories/Cosmology/Distances.v
//!
//! # Key equations
//!
//!   E(z) = H(z)/H_0 = sqrt(Omega_m (1+z)^3 + Omega_Lambda)
//!   d_C(z) = (c/H_0) integral_0^z dz'/E(z')
//!   d_L(z) = (1+z) d_C(z)
//!   d_A(z) = d_C(z)/(1+z)
//!
//! # References
//!
//! - Planck Collaboration VI (2020), A&A 641, A6
//! - Hogg (1999), arXiv:astro-ph/9905116 (distance measures)
//! - Etherington (1933), Phil. Mag. 15, 761 (distance duality)

use crate::bounce::{hubble_e_lcdm, C_KM_S};
use crate::gl_integrate;

/// Conversion factor: 1 Mpc/(km/s) in Gyr.
///
/// 1 Mpc = 3.08567758e19 km, 1 Gyr = 3.15576e16 s.
/// So (1 Mpc/(km/s)) = 3.08567758e19 / 3.15576e16 = 977.792 Gyr.
///
/// Used to convert Hubble time 1/H_0 [Mpc s/km] to Gyr.
const MPC_PER_KM_S_TO_GYR: f64 = 977.792;

// ============================================================================
// FlatLCDM cosmology struct
// ============================================================================

/// Flat Lambda-CDM cosmological model.
///
/// Encapsulates the four core Planck parameters for a spatially flat universe:
/// matter density, baryon density, Hubble constant, and CMB temperature.
/// Dark energy density is derived from flatness: Omega_Lambda = 1 - Omega_m.
#[derive(Clone, Copy, Debug)]
pub struct FlatLCDM {
    /// Hubble constant [km/s/Mpc]
    pub h0: f64,
    /// Total matter density parameter (CDM + baryons)
    pub omega_m: f64,
    /// Baryon density parameter
    pub omega_b: f64,
    /// CMB temperature [K]
    pub t_cmb: f64,
}

impl FlatLCDM {
    /// Planck 2018 TT,TE,EE+lowE+lensing+BAO best-fit cosmology.
    pub fn planck2018() -> Self {
        Self {
            h0: PLANCK18_H0,
            omega_m: PLANCK18_OMEGA_M,
            omega_b: PLANCK18_OMEGA_B,
            t_cmb: PLANCK18_T_CMB,
        }
    }

    /// Dark energy density parameter (flat universe: Omega_Lambda = 1 - Omega_m).
    pub fn omega_lambda(&self) -> f64 {
        1.0 - self.omega_m
    }

    /// Dimensionless Hubble parameter E(z) = H(z)/H_0.
    ///
    /// E(z) = sqrt(Omega_m (1+z)^3 + Omega_Lambda)
    pub fn e_z(&self, z: f64) -> f64 {
        hubble_e_lcdm(z, self.omega_m)
    }

    /// Hubble parameter H(z) [km/s/Mpc].
    pub fn hubble(&self, z: f64) -> f64 {
        self.h0 * self.e_z(z)
    }

    /// Hubble time t_H = 1/H_0 [Mpc/(km/s)].
    ///
    /// Converts to ~14.5 Gyr for H_0 = 67.36 km/s/Mpc when multiplied
    /// by the Mpc-to-km / s-to-Gyr conversion factor.
    pub fn hubble_time(&self) -> f64 {
        1.0 / self.h0
    }

    /// Hubble length D_H = c/H_0 [Mpc].
    ///
    /// The characteristic distance scale of the universe, ~4450 Mpc for Planck 2018.
    pub fn hubble_length(&self) -> f64 {
        C_KM_S / self.h0
    }

    /// Comoving distance d_C(z) [Mpc].
    ///
    /// d_C(z) = (c/H_0) integral_0^z dz'/E(z')
    ///
    /// Uses Gauss-Legendre quadrature (degree 50), which converges exponentially
    /// for the smooth integrand 1/E(z').
    pub fn comoving_distance(&self, z: f64) -> f64 {
        if z <= 0.0 {
            return 0.0;
        }
        let integral = gl_integrate(|zp| 1.0 / hubble_e_lcdm(zp, self.omega_m), 0.0, z, 50);
        self.hubble_length() * integral
    }

    /// Linear approximation for comoving distance (valid for z << 1).
    ///
    /// d_C(z) ~ (c/H_0) z
    ///
    /// Accurate to ~5% for z < 0.1, degrades rapidly beyond that.
    pub fn comoving_distance_linear(&self, z: f64) -> f64 {
        self.hubble_length() * z
    }

    /// Luminosity distance d_L(z) = (1+z) d_C(z) [Mpc].
    pub fn luminosity_distance(&self, z: f64) -> f64 {
        (1.0 + z) * self.comoving_distance(z)
    }

    /// Angular diameter distance d_A(z) = d_C(z)/(1+z) [Mpc].
    ///
    /// Has a maximum at z ~ 1.6 for Planck 2018 parameters: objects at higher
    /// redshift appear _larger_ on the sky (the angular diameter distance turnover).
    pub fn angular_diameter_distance(&self, z: f64) -> f64 {
        if z <= 0.0 {
            return 0.0;
        }
        self.comoving_distance(z) / (1.0 + z)
    }

    /// Distance modulus mu = 5 log10(d_L [Mpc]) + 25 [mag].
    ///
    /// The +25 comes from expressing d_L in Mpc: 5 log10(d_L_Mpc * 1e6 / 10) =
    /// 5 log10(d_L_Mpc) + 5 log10(1e5) = 5 log10(d_L_Mpc) + 25.
    pub fn distance_modulus(&self, z: f64) -> f64 {
        let d_l = self.luminosity_distance(z);
        if d_l <= 0.0 {
            return f64::NEG_INFINITY;
        }
        5.0 * d_l.log10() + 25.0
    }

    /// Comoving volume V_C(z) = (4 pi/3) d_C(z)^3 [Mpc^3].
    ///
    /// Total comoving volume enclosed within a sphere of comoving radius d_C(z).
    pub fn comoving_volume(&self, z: f64) -> f64 {
        let d_c = self.comoving_distance(z);
        (4.0 * std::f64::consts::PI / 3.0) * d_c * d_c * d_c
    }

    /// Deceleration parameter q(z).
    ///
    /// q(z) = -1 + (1+z) (dE/dz) / E(z)
    ///
    /// where dE/dz = (3/2) Omega_m (1+z)^2 / (2 E(z)).
    ///
    /// q < 0 indicates accelerating expansion. The transition redshift
    /// z_acc (where q = 0) is ~0.67 for Planck 2018 parameters.
    pub fn deceleration(&self, z: f64) -> f64 {
        deceleration_parameter(self.omega_m, z)
    }

    /// Lookback time t_L(z) [Gyr].
    ///
    /// t_L(z) = (1/H_0) integral_0^z dz' / ((1+z') E(z'))
    ///
    /// The lookback time is how far back in time we observe an object at
    /// redshift z. For Planck 2018 at z=1, t_L ~ 7.9 Gyr; at z=1100 (CMB),
    /// t_L ~ 13.8 Gyr (essentially the age of the universe).
    pub fn lookback_time(&self, z: f64) -> f64 {
        lookback_time(z, self.h0, self.omega_m)
    }

    /// Age of the universe at redshift z [Gyr].
    ///
    /// Computes the integral directly from z to z_max to avoid catastrophic
    /// cancellation from subtracting two nearly-equal large numbers.
    ///
    /// t(z) = (1/H_0) integral_z^inf dz' / ((1+z') E(z'))
    ///
    /// At z=0, returns the current age (~13.8 Gyr for Planck 2018).
    ///
    /// Note: this model includes only matter + Lambda (no radiation), so
    /// ages at very high z (e.g., z=1100) will differ from the full-physics
    /// answer that includes radiation.
    pub fn age_at_redshift(&self, z: f64) -> f64 {
        age_at_z(z, self.h0, self.omega_m)
    }
}

impl Default for FlatLCDM {
    fn default() -> Self {
        Self::planck2018()
    }
}

// ============================================================================
// Planck 2018 constants
// ============================================================================

/// Planck 2018 Hubble constant [km/s/Mpc].
pub const PLANCK18_H0: f64 = 67.36;

/// Planck 2018 matter density parameter.
pub const PLANCK18_OMEGA_M: f64 = 0.3153;

/// Planck 2018 baryon density parameter.
pub const PLANCK18_OMEGA_B: f64 = 0.0493;

/// Planck 2018 CMB temperature [K].
pub const PLANCK18_T_CMB: f64 = 2.7255;

/// Planck 2018 sound horizon at recombination [Mpc].
pub const PLANCK18_SOUND_HORIZON: f64 = 147.09;

// ============================================================================
// Standalone functions (from Rocq extraction interface)
// ============================================================================

/// Deceleration parameter q(z) for flat Lambda-CDM.
///
/// q(z) = -1 + (1+z) (dE/dz) / E(z)
///
/// From E^2 = Omega_m (1+z)^3 + Omega_Lambda, differentiating:
///   2 E dE/dz = 3 Omega_m (1+z)^2
///   dE/dz = 3 Omega_m (1+z)^2 / (2 E)
///
/// Substituting: q(z) = (3/2) Omega_m (1+z)^3 / E(z)^2 - 1.
///
/// The universe accelerates (q < 0) when Omega_Lambda dominates over
/// the decelerating matter term. For Einstein-de Sitter (Omega_m=1),
/// q = 0.5 at all z.
pub fn deceleration_parameter(omega_m: f64, z: f64) -> f64 {
    let e = hubble_e_lcdm(z, omega_m);
    let one_plus_z = 1.0 + z;
    // dE/dz = 3 * Omega_m * (1+z)^2 / (2 * E)
    // Note: the C++ source (cosmology.hpp) has a factor-of-2 error here,
    // writing 1.5/(2E) instead of 3/(2E). We fix it.
    let de_dz = 3.0 * omega_m * one_plus_z * one_plus_z / (2.0 * e);
    -1.0 + one_plus_z * de_dz / e
}

/// Redshift of matter-radiation equality.
///
/// z_eq = Omega_m / Omega_r - 1
///
/// For Planck 2018 (Omega_m = 0.3153, Omega_r ~ 9.15e-5), z_eq ~ 3402.
/// Note: Omega_r includes photons + 3 massless neutrino species.
pub fn z_equality(omega_m: f64, omega_r: f64) -> f64 {
    omega_m / omega_r - 1.0
}

/// Verify the Etherington distance duality (reciprocity theorem).
///
/// D_L = (1+z)^2 D_A
///
/// This relation holds for any metric theory of gravity with photon
/// conservation. Violations would indicate exotic photon physics
/// (e.g., photon-axion oscillation) or spacetime torsion.
///
/// Returns the fractional deviation |D_L / ((1+z)^2 D_A) - 1|.
pub fn distance_duality_deviation(d_l: f64, d_a: f64, z: f64) -> f64 {
    if z <= 0.0 || d_a <= 0.0 {
        return 0.0;
    }
    let one_plus_z = 1.0 + z;
    let ratio = d_l / (one_plus_z * one_plus_z * d_a);
    (ratio - 1.0).abs()
}

/// Verify distance duality within a given tolerance.
///
/// Returns true if |D_L / ((1+z)^2 D_A) - 1| < tol.
pub fn verify_distance_duality(d_l: f64, d_a: f64, z: f64, tol: f64) -> bool {
    distance_duality_deviation(d_l, d_a, z) < tol
}

// ============================================================================
// Lookback time
// ============================================================================

/// Lookback time to redshift z [Gyr].
///
/// t_L(z) = (1/H_0) integral_0^z dz' / ((1+z') E(z'))
///
/// The integrand 1/((1+z)E(z)) is smooth and well-behaved for flat LCDM,
/// so Gauss-Legendre quadrature converges exponentially.
///
/// # Units
///
/// H_0 is in km/s/Mpc. The result is converted to Gyr using the factor
/// 1 Mpc/(km/s) = 977.792 Gyr.
pub fn lookback_time(z: f64, h0: f64, omega_m: f64) -> f64 {
    if z <= 0.0 {
        return 0.0;
    }
    let integral = gl_integrate(
        |zp| 1.0 / ((1.0 + zp) * hubble_e_lcdm(zp, omega_m)),
        0.0,
        z,
        50,
    );
    (MPC_PER_KM_S_TO_GYR / h0) * integral
}

/// Age of the universe [Gyr] for given cosmological parameters.
///
/// Equivalent to `age_at_z(0.0, h0, omega_m)`.
///
/// For Planck 2018: t_0 ~ 13.8 Gyr.
pub fn universe_age(h0: f64, omega_m: f64) -> f64 {
    age_at_z(0.0, h0, omega_m)
}

/// Age of the universe at redshift z [Gyr].
///
/// Uses scale-factor integration for numerical stability:
///
///   t(z) = (1/H_0) integral_0^{a(z)} da / sqrt(Omega_m/a + Omega_Lambda a^2)
///
/// where a(z) = 1/(1+z). This change of variables from dz to da concentrates
/// the quadrature nodes where the integrand has support (a in [0, 1]) rather
/// than spreading them over an infinite redshift range.
///
/// The integrand goes as sqrt(a / Omega_m) near a=0, which is integrable
/// and smooth -- ideal for Gauss-Legendre quadrature.
pub fn age_at_z(z: f64, h0: f64, omega_m: f64) -> f64 {
    let omega_lambda = 1.0 - omega_m;
    let a_z = 1.0 / (1.0 + z);
    if a_z <= 0.0 {
        return 0.0;
    }
    // t(z) = (1/H_0) * integral_0^{a(z)} da / sqrt(Omega_m/a + Omega_Lambda*a^2)
    // Derived from dt = da / (a H(a)) and H(a) = H_0 sqrt(Omega_m a^{-3} + Omega_Lambda).
    let integral = gl_integrate(
        |a| {
            if a <= 0.0 {
                return 0.0;
            }
            1.0 / (omega_m / a + omega_lambda * a * a).sqrt()
        },
        0.0,
        a_z,
        50,
    );
    (MPC_PER_KM_S_TO_GYR / h0) * integral
}

// ============================================================================
// Redshift relations
// ============================================================================

/// Convert observed and emitted wavelengths to redshift.
///
/// z = lambda_obs / lambda_emit - 1
///
/// Works for any wavelength units as long as both are the same.
/// Returns negative values for blueshifted sources.
pub fn wavelength_to_redshift(lambda_observed: f64, lambda_emitted: f64) -> f64 {
    lambda_observed / lambda_emitted - 1.0
}

/// Apply cosmological redshift to an emitted wavelength.
///
/// lambda_obs = lambda_emit * (1 + z)
///
/// Useful for computing where a known emission line (e.g., Lyman-alpha
/// at 121.6 nm) appears in the observer's frame.
pub fn apply_redshift_to_wavelength(lambda_emitted: f64, z: f64) -> f64 {
    lambda_emitted * (1.0 + z)
}

/// Flux dimming factor due to cosmological redshift.
///
/// F_obs / F_emit = 1 / (1+z)^4
///
/// The four powers of (1+z) arise from:
/// - (1+z)^{-1}: photon energy decrease (E_obs = E_emit / (1+z))
/// - (1+z)^{-1}: photon arrival rate decrease (dt_obs = dt_emit * (1+z))
/// - (1+z)^{-2}: solid angle dilution (d_L = (1+z) d_C, flux ~ 1/d_L^2)
///
/// This is the bolometric flux ratio. Spectral (per-frequency) flux has
/// an additional k-correction factor.
pub fn redshift_flux_dimming(z: f64) -> f64 {
    let one_plus_z = 1.0 + z;
    let f = one_plus_z * one_plus_z;
    1.0 / (f * f)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planck() -> FlatLCDM {
        FlatLCDM::planck2018()
    }

    // -- FlatLCDM struct --

    #[test]
    fn test_planck2018_flatness() {
        let cosmo = planck();
        let total = cosmo.omega_m + cosmo.omega_lambda();
        assert!((total - 1.0).abs() < 1e-14, "Omega_total = {total}");
    }

    #[test]
    fn test_default_is_planck2018() {
        let a = FlatLCDM::default();
        let b = FlatLCDM::planck2018();
        assert!((a.h0 - b.h0).abs() < 1e-14);
        assert!((a.omega_m - b.omega_m).abs() < 1e-14);
    }

    // -- E(z) and Hubble --

    #[test]
    fn test_e_z_at_z0() {
        let cosmo = planck();
        let e = cosmo.e_z(0.0);
        assert!((e - 1.0).abs() < 1e-10, "E(0) = {e}");
    }

    #[test]
    fn test_e_z_increases_with_redshift() {
        let cosmo = planck();
        let e0 = cosmo.e_z(0.0);
        let e1 = cosmo.e_z(1.0);
        let e2 = cosmo.e_z(2.0);
        assert!(e1 > e0 && e2 > e1);
    }

    #[test]
    fn test_hubble_at_z0() {
        let cosmo = planck();
        let h = cosmo.hubble(0.0);
        assert!((h - PLANCK18_H0).abs() < 1e-10, "H(0) = {h}");
    }

    // -- Hubble time and length --

    #[test]
    fn test_hubble_time() {
        let cosmo = planck();
        let t_h = cosmo.hubble_time();
        // t_H = 1/67.36 ~ 0.01485 Mpc/(km/s)
        // In Gyr: 0.01485 * 3.0857e19 / 3.1557e16 ~ 14.5 Gyr
        assert!((t_h - 1.0 / PLANCK18_H0).abs() < 1e-14);
    }

    #[test]
    fn test_hubble_length() {
        let cosmo = planck();
        let d_h = cosmo.hubble_length();
        // c/H0 = 299792.458/67.36 ~ 4451 Mpc
        let expected = C_KM_S / PLANCK18_H0;
        assert!((d_h - expected).abs() < 1e-6, "D_H = {d_h} Mpc");
        assert!(d_h > 4400.0 && d_h < 4500.0);
    }

    // -- Distances --

    #[test]
    fn test_comoving_distance_at_z0() {
        let cosmo = planck();
        assert!((cosmo.comoving_distance(0.0)).abs() < 1e-10);
    }

    #[test]
    fn test_comoving_distance_at_z1() {
        let cosmo = planck();
        let d = cosmo.comoving_distance(1.0);
        // Planck18: d_C(z=1) ~ 3300 Mpc
        assert!(d > 3200.0 && d < 3500.0, "d_C(z=1) = {d} Mpc");
    }

    #[test]
    fn test_comoving_distance_increases() {
        let cosmo = planck();
        let d1 = cosmo.comoving_distance(0.5);
        let d2 = cosmo.comoving_distance(1.0);
        let d3 = cosmo.comoving_distance(2.0);
        assert!(d2 > d1 && d3 > d2);
    }

    #[test]
    fn test_linear_approximation_valid_at_low_z() {
        let cosmo = planck();
        let z = 0.01;
        let d_exact = cosmo.comoving_distance(z);
        let d_linear = cosmo.comoving_distance_linear(z);
        let rel_err = (d_exact - d_linear).abs() / d_exact;
        assert!(
            rel_err < 0.01,
            "Linear approx error at z=0.01: {rel_err:.4}"
        );
    }

    #[test]
    fn test_linear_approximation_degrades_at_high_z() {
        let cosmo = planck();
        let z = 1.0;
        let d_exact = cosmo.comoving_distance(z);
        let d_linear = cosmo.comoving_distance_linear(z);
        // At z=1, linear overestimates significantly
        assert!(d_linear > d_exact, "Linear should overestimate at z=1");
    }

    #[test]
    fn test_luminosity_exceeds_comoving() {
        let cosmo = planck();
        let z = 1.0;
        let d_c = cosmo.comoving_distance(z);
        let d_l = cosmo.luminosity_distance(z);
        assert!((d_l - (1.0 + z) * d_c).abs() < 1e-6);
    }

    #[test]
    fn test_angular_diameter_less_than_comoving() {
        let cosmo = planck();
        let z = 1.0;
        let d_c = cosmo.comoving_distance(z);
        let d_a = cosmo.angular_diameter_distance(z);
        assert!((d_a - d_c / (1.0 + z)).abs() < 1e-6);
    }

    #[test]
    fn test_distance_modulus_order_of_magnitude() {
        let cosmo = planck();
        // At z=1: d_L ~ 6600 Mpc, mu ~ 5*log10(6600) + 25 ~ 44.1
        let mu = cosmo.distance_modulus(1.0);
        assert!(mu > 43.0 && mu < 46.0, "mu(z=1) = {mu}");
    }

    // -- Comoving volume --

    #[test]
    fn test_comoving_volume_positive() {
        let cosmo = planck();
        let v = cosmo.comoving_volume(1.0);
        assert!(v > 0.0);
    }

    #[test]
    fn test_comoving_volume_scales_as_d3() {
        let cosmo = planck();
        let d = cosmo.comoving_distance(1.0);
        let v = cosmo.comoving_volume(1.0);
        let expected = (4.0 * std::f64::consts::PI / 3.0) * d * d * d;
        assert!((v - expected).abs() / expected < 1e-10);
    }

    #[test]
    fn test_comoving_volume_increases_with_z() {
        let cosmo = planck();
        let v1 = cosmo.comoving_volume(0.5);
        let v2 = cosmo.comoving_volume(1.0);
        let v3 = cosmo.comoving_volume(2.0);
        assert!(v2 > v1 && v3 > v2);
    }

    // -- Deceleration parameter --

    #[test]
    fn test_deceleration_at_z0_negative() {
        // Today: accelerating expansion, q < 0
        let q = deceleration_parameter(PLANCK18_OMEGA_M, 0.0);
        assert!(q < 0.0, "q(0) = {q} (should be negative)");
    }

    #[test]
    fn test_deceleration_at_high_z_positive() {
        // At z=5: matter-dominated, q > 0
        let q = deceleration_parameter(PLANCK18_OMEGA_M, 5.0);
        assert!(q > 0.0, "q(5) = {q} (should be positive)");
    }

    #[test]
    fn test_deceleration_transition_redshift() {
        // Transition z_acc where q=0: should be around z ~ 0.6-0.7 for Planck18
        // Bisect to find it
        let mut z_lo = 0.0;
        let mut z_hi = 2.0;
        for _ in 0..100 {
            let z_mid = 0.5 * (z_lo + z_hi);
            if deceleration_parameter(PLANCK18_OMEGA_M, z_mid) < 0.0 {
                z_lo = z_mid;
            } else {
                z_hi = z_mid;
            }
        }
        let z_acc = 0.5 * (z_lo + z_hi);
        assert!(
            z_acc > 0.55 && z_acc < 0.80,
            "z_acc = {z_acc} (expected ~0.67)"
        );
    }

    #[test]
    fn test_deceleration_einstein_desitter() {
        // Einstein-de Sitter (Omega_m=1, Omega_Lambda=0): q = 0.5 at all z
        let q = deceleration_parameter(1.0, 0.0);
        assert!((q - 0.5).abs() < 1e-10, "q_EdS(0) = {q}");

        let q_z5 = deceleration_parameter(1.0, 5.0);
        assert!((q_z5 - 0.5).abs() < 1e-10, "q_EdS(5) = {q_z5}");
    }

    // -- Matter-radiation equality --

    #[test]
    fn test_z_equality_order_of_magnitude() {
        // Planck18: Omega_r ~ 9.15e-5 (photons + 3 neutrino species)
        // z_eq ~ 0.3153/9.15e-5 - 1 ~ 3445
        // (Planck best-fit from chains is 3402; the discrepancy is because
        // the simple formula ignores neutrino mass effects)
        let z_eq = z_equality(PLANCK18_OMEGA_M, 9.15e-5);
        assert!(
            z_eq > 3000.0 && z_eq < 4000.0,
            "z_eq = {z_eq} (expected ~3400)"
        );
    }

    #[test]
    fn test_z_equality_scales_with_omega_m() {
        let z1 = z_equality(0.2, 9.15e-5);
        let z2 = z_equality(0.3, 9.15e-5);
        assert!(z2 > z1, "Higher Omega_m -> later equality");
    }

    // -- Distance duality --

    #[test]
    fn test_distance_duality_holds() {
        let cosmo = planck();
        let z = 1.0;
        let d_l = cosmo.luminosity_distance(z);
        let d_a = cosmo.angular_diameter_distance(z);
        assert!(
            verify_distance_duality(d_l, d_a, z, 1e-10),
            "Etherington reciprocity violated"
        );
    }

    #[test]
    fn test_distance_duality_deviation_zero() {
        let cosmo = planck();
        let z = 2.0;
        let d_l = cosmo.luminosity_distance(z);
        let d_a = cosmo.angular_diameter_distance(z);
        let dev = distance_duality_deviation(d_l, d_a, z);
        assert!(dev < 1e-10, "Duality deviation = {dev}");
    }

    #[test]
    fn test_distance_duality_detects_violation() {
        // Artificially break duality
        let d_l = 100.0;
        let d_a = 100.0; // Should be d_l/(1+z)^2 = 25 at z=1
        assert!(!verify_distance_duality(d_l, d_a, 1.0, 0.01));
    }

    // -- Cross-validation with existing modules --

    #[test]
    fn test_matches_distances_module() {
        // FlatLCDM.comoving_distance should match distances::comoving_distance
        let cosmo = planck();
        let z = 1.0;
        let d_flrw = cosmo.comoving_distance(z);
        let d_dist = crate::distances::comoving_distance(z, PLANCK18_OMEGA_M, PLANCK18_H0);
        assert!(
            (d_flrw - d_dist).abs() < 1e-6,
            "FLRW: {d_flrw}, distances: {d_dist}"
        );
    }

    #[test]
    fn test_matches_bounce_module() {
        // FlatLCDM.e_z should match bounce::hubble_e_lcdm
        let cosmo = planck();
        let z = 1.5;
        let e_flrw = cosmo.e_z(z);
        let e_bounce = crate::bounce::hubble_e_lcdm(z, PLANCK18_OMEGA_M);
        assert!(
            (e_flrw - e_bounce).abs() < 1e-14,
            "FLRW: {e_flrw}, bounce: {e_bounce}"
        );
    }

    // -- Lookback time --

    #[test]
    fn test_lookback_time_at_z0() {
        let t = lookback_time(0.0, PLANCK18_H0, PLANCK18_OMEGA_M);
        assert!(t.abs() < 1e-15, "t_L(0) = {t}");
    }

    #[test]
    fn test_lookback_time_at_z1() {
        // Planck 2018: t_L(z=1) ~ 7.9 Gyr
        let t = lookback_time(1.0, PLANCK18_H0, PLANCK18_OMEGA_M);
        assert!(t > 7.5 && t < 8.5, "t_L(z=1) = {t} Gyr");
    }

    #[test]
    fn test_lookback_time_increases_with_z() {
        let t1 = lookback_time(0.5, PLANCK18_H0, PLANCK18_OMEGA_M);
        let t2 = lookback_time(1.0, PLANCK18_H0, PLANCK18_OMEGA_M);
        let t3 = lookback_time(3.0, PLANCK18_H0, PLANCK18_OMEGA_M);
        assert!(t2 > t1 && t3 > t2);
    }

    #[test]
    fn test_lookback_time_method_matches_standalone() {
        let cosmo = planck();
        let z = 2.0;
        let t_method = cosmo.lookback_time(z);
        let t_standalone = lookback_time(z, PLANCK18_H0, PLANCK18_OMEGA_M);
        assert!((t_method - t_standalone).abs() < 1e-14);
    }

    // -- Universe age --

    #[test]
    fn test_universe_age_planck() {
        // Planck 2018 best-fit: 13.797 +/- 0.023 Gyr
        let age = universe_age(PLANCK18_H0, PLANCK18_OMEGA_M);
        assert!(age > 13.5 && age < 14.1, "t_0 = {age} Gyr (expected ~13.8)");
    }

    #[test]
    fn test_age_at_z0_equals_universe_age() {
        let cosmo = planck();
        let age_z0 = cosmo.age_at_redshift(0.0);
        let age_total = universe_age(PLANCK18_H0, PLANCK18_OMEGA_M);
        assert!(
            (age_z0 - age_total).abs() < 1e-10,
            "age(z=0) = {age_z0}, t_0 = {age_total}"
        );
    }

    #[test]
    fn test_age_at_redshift_decreases() {
        let cosmo = planck();
        let age0 = cosmo.age_at_redshift(0.0);
        let age1 = cosmo.age_at_redshift(1.0);
        let age3 = cosmo.age_at_redshift(3.0);
        assert!(age0 > age1 && age1 > age3);
    }

    #[test]
    fn test_age_at_cmb_very_small() {
        // At z=1100 in matter-only flat LCDM (no radiation), the age is
        // ~0.5 Myr. With radiation included (full Planck model), it would
        // be ~380 kyr. Our model has no radiation component, so the age
        // is somewhat larger since E(z) is smaller at high z.
        let cosmo = planck();
        let age_cmb = cosmo.age_at_redshift(1100.0);
        assert!(
            age_cmb > 0.0 && age_cmb < 0.01,
            "age(z=1100) = {age_cmb} Gyr (expected < 10 Myr)"
        );
    }

    // -- Wavelength-redshift relations --

    #[test]
    fn test_wavelength_to_redshift_identity() {
        // Same wavelength -> z=0
        let z = wavelength_to_redshift(500.0, 500.0);
        assert!(z.abs() < 1e-15, "z = {z}");
    }

    #[test]
    fn test_wavelength_to_redshift_double() {
        // Observed at 2x emitted -> z=1
        let z = wavelength_to_redshift(1000.0, 500.0);
        assert!((z - 1.0).abs() < 1e-15, "z = {z}");
    }

    #[test]
    fn test_wavelength_roundtrip() {
        // Apply redshift then recover z
        let lambda_emit = 121.6; // Lyman-alpha [nm]
        let z = 2.5;
        let lambda_obs = apply_redshift_to_wavelength(lambda_emit, z);
        let z_recovered = wavelength_to_redshift(lambda_obs, lambda_emit);
        assert!(
            (z_recovered - z).abs() < 1e-14,
            "z_recovered = {z_recovered}"
        );
    }

    #[test]
    fn test_apply_redshift_increases_wavelength() {
        let lambda_obs = apply_redshift_to_wavelength(500.0, 1.0);
        assert!(
            (lambda_obs - 1000.0).abs() < 1e-10,
            "lambda_obs = {lambda_obs}"
        );
    }

    #[test]
    fn test_blueshift() {
        // Approaching source: lambda_obs < lambda_emit -> z < 0
        let z = wavelength_to_redshift(400.0, 500.0);
        assert!(z < 0.0, "Blueshift should give z < 0, got {z}");
        assert!((z - (-0.2)).abs() < 1e-15);
    }

    // -- Flux dimming --

    #[test]
    fn test_flux_dimming_at_z0() {
        let f = redshift_flux_dimming(0.0);
        assert!((f - 1.0).abs() < 1e-15, "f(z=0) = {f}");
    }

    #[test]
    fn test_flux_dimming_at_z1() {
        // (1+1)^4 = 16, so dimming = 1/16 = 0.0625
        let f = redshift_flux_dimming(1.0);
        assert!((f - 0.0625).abs() < 1e-15, "f(z=1) = {f}");
    }

    #[test]
    fn test_flux_dimming_decreases_with_z() {
        let f1 = redshift_flux_dimming(0.5);
        let f2 = redshift_flux_dimming(1.0);
        let f3 = redshift_flux_dimming(3.0);
        assert!(f1 > f2 && f2 > f3);
    }

    #[test]
    fn test_flux_dimming_fourth_power() {
        // Verify the (1+z)^{-4} scaling explicitly
        let z = 2.0;
        let expected = 1.0 / (3.0_f64.powi(4)); // (1+2)^{-4} = 1/81
        let f = redshift_flux_dimming(z);
        assert!(
            (f - expected).abs() < 1e-15,
            "f(z=2) = {f}, expected {expected}"
        );
    }
}
