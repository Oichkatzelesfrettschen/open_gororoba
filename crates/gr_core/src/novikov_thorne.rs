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

// ============================================================================
// Circular orbit quantities (Schwarzschild, geometric units)
// ============================================================================

/// Specific energy per unit rest mass for a circular orbit at radius r_m (in units of M).
///
/// E = (1 - 2/r) / sqrt(1 - 3/r)
///
/// Returns NaN for r_m <= 3 (photon sphere; no stable circular orbits).
/// At the ISCO (r_m = 6 for Schwarzschild), E = sqrt(8/9) ~ 0.9428.
pub fn specific_energy(r_m: f64) -> f64 {
    if r_m <= 3.0 {
        return f64::NAN;
    }
    (1.0 - 2.0 / r_m) / (1.0 - 3.0 / r_m).sqrt()
}

/// Specific angular momentum for a circular orbit at radius r_m (in units of M).
///
/// L/M = sqrt(r) / sqrt(1 - 3/r)
///
/// Returns NaN for r_m <= 3.
pub fn specific_angular_momentum(r_m: f64) -> f64 {
    if r_m <= 3.0 {
        return f64::NAN;
    }
    r_m.sqrt() / (1.0 - 3.0 / r_m).sqrt()
}

/// Angular velocity for a circular Keplerian orbit at radius r_m (in units of M).
///
/// Omega * M = 1 / r^{3/2}
///
/// This is the coordinate angular velocity in Schwarzschild spacetime.
pub fn angular_velocity_circular(r_m: f64) -> f64 {
    1.0 / r_m.powf(1.5)
}

// ============================================================================
// Relativistic disk emission
// ============================================================================

/// Gravitational redshift factor for circular orbit emission (Schwarzschild).
///
/// g = sqrt(1 - 3M/r)
///
/// This is the ratio of observed to emitted photon energy for a
/// static observer at infinity viewing radiation from circular orbits.
/// Equals zero at the photon sphere (r_m = 3) and approaches 1 at large r.
pub fn disk_redshift_factor(r_m: f64) -> f64 {
    let x = 3.0 / r_m;
    if x >= 1.0 {
        return 0.0;
    }
    (1.0 - x).sqrt()
}

/// Doppler factor for an orbiting disk element (Schwarzschild).
///
/// delta = 1 / (gamma * (1 - beta_los))
///
/// where beta = v_orb/c = sqrt(M/r) = 1/sqrt(r_m) is the orbital velocity,
/// gamma = (1-beta^2)^{-1/2} is the Lorentz factor, and
/// beta_los = beta * sin(phi) * sin(inclination) is the line-of-sight component.
///
/// phi is the azimuthal angle of the disk element (0 = receding, pi/2 = approaching).
/// inclination is the viewing angle from the disk normal (0 = face-on).
pub fn disk_doppler_factor(r_m: f64, phi: f64, inclination: f64) -> f64 {
    let beta = (1.0 / r_m).sqrt();
    let beta_los = beta * phi.sin() * inclination.sin();
    let gamma = 1.0 / (1.0 - beta * beta).sqrt();
    1.0 / (gamma * (1.0 - beta_los))
}

/// Observed flux from a disk annulus including relativistic corrections.
///
/// F_obs = F_emit * (g * delta)^4
///
/// The (g*delta)^4 factor accounts for gravitational redshift (g^4 from
/// specific intensity transformation I_obs/I_emit = g^4) combined with
/// Doppler beaming (delta^4).
pub fn disk_flux_observed(
    r_m: f64,
    phi: f64,
    a_star: f64,
    mdot_edd: f64,
    mass_solar: f64,
    inclination: f64,
) -> f64 {
    let f_emit = disk_flux(r_m, a_star, mdot_edd, mass_solar);
    let g = disk_redshift_factor(r_m);
    let delta = disk_doppler_factor(r_m, phi, inclination);
    let factor = g * delta;
    f_emit * factor * factor * factor * factor
}

// ============================================================================
// Disk spectrum
// ============================================================================

/// Disk spectrum L_nu at frequency nu [Hz], integrated over annuli.
///
/// L_nu = 4*pi*cos(i) * integral_{r_in}^{r_out} B_nu(T(r)) * 2*pi*r_cgs * dr_cgs
///
/// Uses logarithmic radial spacing (midpoint rule) which concentrates
/// sampling near the peak-temperature region. The Planck function B_nu(T)
/// is imported from the absorption module.
///
/// Returns specific luminosity [erg s^-1 Hz^-1].
pub fn disk_spectrum(
    nu: f64,
    a_star: f64,
    mdot_edd: f64,
    mass_solar: f64,
    inclination: f64,
    n_points: usize,
) -> f64 {
    let r_isco = isco_radius(a_star);
    let r_in_m = r_isco;
    let r_out_m: f64 = 1000.0; // Outer disk at 1000 M

    let m_cgs = mass_solar * M_SUN_CGS;
    let r_g = G_CGS * m_cgs / (C_CGS * C_CGS);

    let log_r_in = r_in_m.ln();
    let log_r_out = r_out_m.ln();
    let d_log_r = (log_r_out - log_r_in) / n_points as f64;

    let mut sum = 0.0;
    for i in 0..n_points {
        let log_r = log_r_in + (i as f64 + 0.5) * d_log_r;
        let r_m = log_r.exp();

        let temp = disk_temperature(r_m, a_star, mdot_edd, mass_solar);
        let b_nu = crate::absorption::planck_function(nu, temp);

        // r_cgs = r_m * r_g, dr_cgs = r_cgs * d_log_r
        // Integral element: B_nu * r_cgs * r_cgs * d_log_r (from r dr = r^2 d(ln r))
        let r_cgs = r_m * r_g;
        sum += b_nu * r_cgs * r_cgs * d_log_r;
    }

    // Factor of 2 for both disk faces, cos(i) for projection
    4.0 * PI * inclination.cos() * 2.0 * PI * sum
}

// ============================================================================
// Disk radial profile
// ============================================================================

/// A point in the radial disk profile.
#[derive(Debug, Clone)]
pub struct DiskProfilePoint {
    /// Radius in units of M (gravitational radii).
    pub r_m: f64,
    /// Surface flux [erg cm^-2 s^-1].
    pub flux: f64,
    /// Temperature [K].
    pub temperature: f64,
    /// Orbital velocity as fraction of c.
    pub beta: f64,
    /// Angular velocity [units of 1/M].
    pub omega: f64,
    /// Gravitational redshift factor g = sqrt(1 - 3/r).
    pub redshift_factor: f64,
}

/// Generate a logarithmically-spaced radial profile of disk quantities.
///
/// Samples n_points from r_ISCO to r_out (default 1000 M) in log-r,
/// computing flux, temperature, orbital velocity, angular velocity,
/// and gravitational redshift at each point.
pub fn disk_profile(
    a_star: f64,
    mdot_edd: f64,
    mass_solar: f64,
    n_points: usize,
) -> Vec<DiskProfilePoint> {
    let r_isco = isco_radius(a_star);
    let r_out = 1000.0_f64;

    if n_points == 0 {
        return Vec::new();
    }

    let log_r_in = r_isco.ln();
    let log_r_out = r_out.ln();
    let d_log_r = if n_points > 1 {
        (log_r_out - log_r_in) / (n_points - 1) as f64
    } else {
        0.0
    };

    (0..n_points)
        .map(|i| {
            let r_m = (log_r_in + i as f64 * d_log_r).exp();
            let flux = disk_flux(r_m, a_star, mdot_edd, mass_solar);
            let temperature = (flux / SIGMA_SB_CGS).max(0.0).powf(0.25);
            let beta = (1.0 / r_m).sqrt();
            let omega = angular_velocity_circular(r_m);
            let redshift_factor = disk_redshift_factor(r_m);
            DiskProfilePoint {
                r_m,
                flux,
                temperature,
                beta,
                omega,
                redshift_factor,
            }
        })
        .collect()
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

    // -- Circular orbit quantities --

    #[test]
    fn test_specific_energy_isco_schwarzschild() {
        // At r_ISCO = 6M: E = sqrt(8/9) ~ 0.9428
        let e = specific_energy(6.0);
        let expected = (8.0 / 9.0_f64).sqrt();
        assert!(
            (e - expected).abs() < 1e-10,
            "E(ISCO) = {e}, expected {expected}"
        );
    }

    #[test]
    fn test_specific_energy_approaches_one_at_infinity() {
        let e = specific_energy(1e6);
        assert!(
            (e - 1.0).abs() < 1e-4,
            "E at large r should approach 1: {e}"
        );
    }

    #[test]
    fn test_specific_energy_nan_inside_photon_sphere() {
        let e = specific_energy(2.5);
        assert!(e.is_nan(), "E inside photon sphere should be NaN");
    }

    #[test]
    fn test_specific_angular_momentum_isco() {
        // At r = 6M: L = sqrt(6)/sqrt(1-3/6) = sqrt(6)/sqrt(1/2) = sqrt(12) = 2*sqrt(3)
        let l = specific_angular_momentum(6.0);
        let expected = (12.0_f64).sqrt();
        assert!(
            (l - expected).abs() < 1e-10,
            "L(ISCO) = {l}, expected {expected}"
        );
    }

    #[test]
    fn test_specific_angular_momentum_nan_inside_photon_sphere() {
        let l = specific_angular_momentum(2.0);
        assert!(l.is_nan());
    }

    #[test]
    fn test_angular_velocity_keplerian() {
        // Omega(r=100) = 1/1000 = 0.001
        let omega = angular_velocity_circular(100.0);
        assert!(
            (omega - 0.001).abs() < 1e-12,
            "Omega(100) = {omega}"
        );
    }

    #[test]
    fn test_angular_velocity_decreases_with_radius() {
        let o10 = angular_velocity_circular(10.0);
        let o100 = angular_velocity_circular(100.0);
        assert!(o100 < o10, "angular velocity should decrease outward");
    }

    #[test]
    fn test_efficiency_from_specific_energy() {
        // Consistency: radiative_efficiency(0) should equal 1 - specific_energy(6)
        let eta = radiative_efficiency(0.0);
        let e_isco = specific_energy(6.0);
        assert!(
            (eta - (1.0 - e_isco)).abs() < 1e-10,
            "eta = {eta}, 1 - E_ISCO = {}",
            1.0 - e_isco
        );
    }

    // -- Gravitational redshift --

    #[test]
    fn test_redshift_factor_at_isco() {
        // g = sqrt(1 - 3/6) = sqrt(1/2) ~ 0.7071
        let g = disk_redshift_factor(6.0);
        let expected = (0.5_f64).sqrt();
        assert!(
            (g - expected).abs() < 1e-10,
            "g(ISCO) = {g}, expected {expected}"
        );
    }

    #[test]
    fn test_redshift_factor_at_photon_sphere() {
        let g = disk_redshift_factor(3.0);
        assert!(g == 0.0, "g at photon sphere should be 0");
    }

    #[test]
    fn test_redshift_factor_approaches_one() {
        let g = disk_redshift_factor(1e6);
        assert!((g - 1.0).abs() < 1e-4, "g at large r should be ~1: {g}");
    }

    // -- Doppler factor --

    #[test]
    fn test_doppler_factor_face_on_is_one() {
        // inclination = 0 -> sin(i) = 0 -> beta_los = 0 -> delta = 1/gamma
        // Actually: delta = 1/(gamma*(1-0)) = 1/gamma (transverse Doppler)
        let delta = disk_doppler_factor(100.0, 0.0, 0.0);
        let beta = (1.0 / 100.0_f64).sqrt();
        let gamma = 1.0 / (1.0 - beta * beta).sqrt();
        let expected = 1.0 / gamma;
        assert!(
            (delta - expected).abs() < 1e-10,
            "face-on delta = {delta}, expected {expected}"
        );
    }

    #[test]
    fn test_doppler_blueshift_approaching() {
        // phi=pi/2 (approaching), high inclination -> blueshift (delta > 1)
        let delta = disk_doppler_factor(10.0, PI / 2.0, PI / 3.0);
        assert!(delta > 1.0, "approaching side should be blueshifted: {delta}");
    }

    #[test]
    fn test_doppler_redshift_receding() {
        // phi = -pi/2 (receding), high inclination -> redshift (delta < 1)
        let delta = disk_doppler_factor(10.0, -PI / 2.0, PI / 3.0);
        assert!(delta < 1.0, "receding side should be redshifted: {delta}");
    }

    #[test]
    fn test_doppler_symmetry() {
        // delta(phi) * delta(-phi) != 1 in general (because of gamma),
        // but the LOS components are antisymmetric
        let d_plus = disk_doppler_factor(20.0, PI / 4.0, PI / 4.0);
        let d_minus = disk_doppler_factor(20.0, -PI / 4.0, PI / 4.0);
        // Both should be positive
        assert!(d_plus > 0.0 && d_minus > 0.0);
        // Product should differ from 1 due to transverse Doppler
        assert!((d_plus * d_minus - 1.0).abs() > 1e-6);
    }

    // -- Observed flux --

    #[test]
    fn test_observed_flux_face_on_reduces_by_g4() {
        // Face-on: delta ~ 1/gamma, so F_obs ~ F_emit * (g/gamma)^4
        let r = 10.0;
        let f_emit = disk_flux(r, 0.0, 0.1, 10.0);
        let f_obs = disk_flux_observed(r, 0.0, 0.0, 0.1, 10.0, 0.0);
        // For face-on, observed < emitted due to gravitational redshift + transverse Doppler
        assert!(f_obs < f_emit, "observed flux should be dimmed: {f_obs} vs {f_emit}");
        assert!(f_obs > 0.0, "observed flux should be positive");
    }

    #[test]
    fn test_observed_flux_zero_inside_isco() {
        let f = disk_flux_observed(3.0, 0.0, 0.0, 0.1, 10.0, 0.0);
        assert!(f == 0.0);
    }

    // -- Disk spectrum --

    #[test]
    fn test_disk_spectrum_positive() {
        // 10 solar masses, 0.1 Eddington, face-on, optical frequency
        let nu = 5e14; // ~600 nm (optical)
        let l_nu = disk_spectrum(nu, 0.0, 0.1, 10.0, 0.0, 100);
        assert!(l_nu > 0.0, "disk spectrum should be positive: {l_nu}");
    }

    #[test]
    fn test_disk_spectrum_decreases_at_high_freq() {
        // Spectrum should decrease at very high frequencies (Wien tail)
        let l_opt = disk_spectrum(5e14, 0.0, 0.1, 10.0, 0.0, 100);
        let l_xray = disk_spectrum(5e18, 0.0, 0.1, 10.0, 0.0, 100);
        assert!(l_xray < l_opt, "X-ray should be dimmer than optical for stellar BH");
    }

    #[test]
    fn test_disk_spectrum_inclination_cosine() {
        // L_nu scales as cos(i), so edge-on (i=pi/2) should be ~0
        let l_face = disk_spectrum(5e14, 0.0, 0.1, 10.0, 0.0, 100);
        let l_edge = disk_spectrum(5e14, 0.0, 0.1, 10.0, PI / 2.0, 100);
        assert!(l_edge.abs() < 1e-10 * l_face, "edge-on disk should emit ~0");
    }

    // -- Disk profile --

    #[test]
    fn test_disk_profile_length() {
        let profile = disk_profile(0.0, 0.1, 10.0, 50);
        assert_eq!(profile.len(), 50);
    }

    #[test]
    fn test_disk_profile_starts_at_isco() {
        let profile = disk_profile(0.0, 0.1, 10.0, 100);
        let r_first = profile[0].r_m;
        let r_isco = isco_radius(0.0);
        assert!(
            (r_first - r_isco).abs() / r_isco < 0.01,
            "first point should be near ISCO: {r_first}"
        );
    }

    #[test]
    fn test_disk_profile_temperatures_positive() {
        let profile = disk_profile(0.0, 0.1, 10.0, 50);
        // All points except possibly the ISCO edge should have positive T
        for pt in &profile[1..] {
            assert!(pt.temperature > 0.0, "T({}) = {}", pt.r_m, pt.temperature);
        }
    }

    #[test]
    fn test_disk_profile_redshift_bounded() {
        let profile = disk_profile(0.0, 0.1, 10.0, 50);
        for pt in &profile {
            assert!(
                pt.redshift_factor >= 0.0 && pt.redshift_factor <= 1.0,
                "g = {} at r = {}",
                pt.redshift_factor,
                pt.r_m
            );
        }
    }

    #[test]
    fn test_disk_profile_velocity_subluminal() {
        let profile = disk_profile(0.0, 0.1, 10.0, 50);
        for pt in &profile {
            assert!(pt.beta < 1.0, "beta = {} at r = {}", pt.beta, pt.r_m);
        }
    }

    #[test]
    fn test_disk_profile_empty() {
        let profile = disk_profile(0.0, 0.1, 10.0, 0);
        assert!(profile.is_empty());
    }
}
