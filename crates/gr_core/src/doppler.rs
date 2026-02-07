//! Relativistic Doppler effect and beaming calculations.
//!
//! Implements special-relativistic Doppler effects for:
//! - Moving sources (jets, orbiting disk matter)
//! - Cosmological redshift
//! - Relativistic beaming/boosting
//! - Aberration and headlight effect
//!
//! Key formulas:
//!   Lorentz factor: gamma = 1/sqrt(1 - beta^2)
//!   Doppler factor: delta = 1/(gamma (1 - beta cos theta))
//!   Beamed flux:    F_obs = delta^{3+alpha} F_emit   (power-law spectrum F ~ nu^alpha)
//!
//! References:
//!   - Rybicki & Lightman (1979): Radiative Processes in Astrophysics
//!   - Begelman, Blandford, Rees (1984): Rev. Mod. Phys. 56, 255

use std::f64::consts::PI;

// ============================================================================
// Lorentz factor
// ============================================================================

/// Lorentz factor gamma from velocity beta = v/c.
///
/// gamma = 1/sqrt(1 - beta^2)
///
/// Returns infinity for beta >= 1.
pub fn lorentz_factor(beta: f64) -> f64 {
    let b = beta.abs();
    if b >= 1.0 {
        return f64::INFINITY;
    }
    1.0 / (1.0 - b * b).sqrt()
}

/// Velocity beta = v/c from Lorentz factor gamma.
///
/// beta = sqrt(1 - 1/gamma^2)
///
/// Clamps gamma to >= 1.
pub fn beta_from_gamma(gamma: f64) -> f64 {
    let g = gamma.max(1.0);
    (1.0 - 1.0 / (g * g)).sqrt()
}

// ============================================================================
// Doppler factor
// ============================================================================

/// Relativistic Doppler factor.
///
/// delta = 1/(gamma (1 - beta cos theta))
///
/// theta = 0 means source approaching (blueshift).
/// theta = pi means source receding (redshift).
pub fn doppler_factor(beta: f64, theta: f64) -> f64 {
    let gamma = lorentz_factor(beta);
    let cos_theta = theta.cos();
    let denom = gamma * (1.0 - beta * cos_theta);

    if denom.abs() < 1e-30 {
        return f64::INFINITY;
    }

    1.0 / denom
}

/// Doppler factor for head-on approach (theta = 0).
///
/// delta_max = sqrt((1 + beta)/(1 - beta))
pub fn doppler_factor_approaching(beta: f64) -> f64 {
    let b = beta.abs();
    if b >= 1.0 {
        return f64::INFINITY;
    }
    ((1.0 + b) / (1.0 - b)).sqrt()
}

/// Doppler factor for direct recession (theta = pi).
///
/// delta_min = sqrt((1 - beta)/(1 + beta))
pub fn doppler_factor_receding(beta: f64) -> f64 {
    let b = beta.abs();
    if b >= 1.0 {
        return 0.0;
    }
    ((1.0 - b) / (1.0 + b)).sqrt()
}

// ============================================================================
// Frequency shifts
// ============================================================================

/// Relativistic Doppler shift: observed frequency.
///
/// nu_obs = delta * nu_emit
pub fn doppler_shift_relativistic(nu_emit: f64, beta: f64, theta: f64) -> f64 {
    doppler_factor(beta, theta) * nu_emit
}

/// Transverse Doppler shift (theta = pi/2).
///
/// nu_obs = nu_emit / gamma (always redshift due to time dilation).
pub fn doppler_shift_transverse(nu_emit: f64, beta: f64) -> f64 {
    nu_emit / lorentz_factor(beta)
}

/// Cosmological redshift: observed frequency.
///
/// nu_obs = nu_emit / (1 + z)
pub fn observed_frequency(nu_emit: f64, z: f64) -> f64 {
    nu_emit / (1.0 + z)
}

/// Cosmological redshift: rest-frame frequency from observed.
///
/// nu_emit = nu_obs * (1 + z)
pub fn rest_frame_frequency(nu_obs: f64, z: f64) -> f64 {
    nu_obs * (1.0 + z)
}

// ============================================================================
// Relativistic beaming
// ============================================================================

/// Intensity boost from relativistic beaming.
///
/// I_obs = delta^(3+alpha) * I_emit
///
/// The delta^3 comes from solid angle aberration (delta^2) and photon
/// arrival rate (delta^1). The extra delta^alpha comes from the spectral
/// slope for a power-law spectrum F_nu ~ nu^alpha.
///
/// Use alpha = 0 for blackbody (optically thick) emission.
pub fn relativistic_beaming_intensity(
    i_emit: f64,
    beta: f64,
    theta: f64,
    alpha: f64,
) -> f64 {
    let delta = doppler_factor(beta, theta);
    i_emit * delta.powf(3.0 + alpha)
}

/// Flux density boost from relativistic beaming.
///
/// F_obs = delta^(3+alpha) * F_emit for isotropic emitter.
pub fn relativistic_beaming_flux(
    f_emit: f64,
    beta: f64,
    theta: f64,
    alpha: f64,
) -> f64 {
    let delta = doppler_factor(beta, theta);
    f_emit * delta.powf(3.0 + alpha)
}

/// Apparent superluminal velocity [units of c].
///
/// beta_app = beta sin(theta) / (1 - beta cos(theta))
///
/// Can exceed 1 for relativistic sources at small viewing angles --
/// an illusion caused by the source nearly keeping pace with its own
/// photons.
pub fn apparent_superluminal_velocity(beta: f64, theta: f64) -> f64 {
    let sin_theta = theta.sin();
    let cos_theta = theta.cos();
    let denom = 1.0 - beta * cos_theta;

    if denom.abs() < 1e-30 {
        return f64::INFINITY;
    }

    (beta * sin_theta) / denom
}

/// Viewing angle that maximizes apparent superluminal velocity.
///
/// theta_opt = arccos(beta)
///
/// At this angle, beta_app = gamma * beta.
pub fn superluminal_optimal_angle(beta: f64) -> f64 {
    beta.clamp(-1.0, 1.0).acos()
}

/// Maximum apparent superluminal velocity [units of c].
///
/// beta_app_max = gamma * beta
pub fn superluminal_velocity_max(beta: f64) -> f64 {
    lorentz_factor(beta) * beta
}

// ============================================================================
// Aberration
// ============================================================================

/// Relativistic aberration of angle.
///
/// cos theta' = (cos theta - beta) / (1 - beta cos theta)
///
/// Light emitted at angle theta in the source rest frame appears at
/// theta' in the observer frame (headlight effect).
pub fn relativistic_aberration(theta: f64, beta: f64) -> f64 {
    let cos_theta = theta.cos();
    let cos_theta_prime = (cos_theta - beta) / (1.0 - beta * cos_theta);
    cos_theta_prime.clamp(-1.0, 1.0).acos()
}

/// Beaming half-angle (headlight effect).
///
/// theta_beam ~ 1/gamma
///
/// For highly relativistic motion, most radiation is concentrated
/// within a forward cone of this half-angle.
pub fn beaming_half_angle(gamma: f64) -> f64 {
    1.0 / gamma.max(1.0)
}

// ============================================================================
// K-corrections
// ============================================================================

/// K-correction for a power-law spectrum.
///
/// F_rest = F_obs * (1 + z)^(1 + alpha)
///
/// Converts observed flux density to rest-frame flux density at the
/// corresponding rest frequency, for F_nu ~ nu^alpha.
pub fn k_correction_power_law(f_obs: f64, z: f64, alpha: f64) -> f64 {
    f_obs * (1.0 + z).powf(1.0 + alpha)
}

/// K-correction factor alone.
///
/// K(z, alpha) = (1 + z)^(1 + alpha)
pub fn k_correction_factor(z: f64, alpha: f64) -> f64 {
    (1.0 + z).powf(1.0 + alpha)
}

// ============================================================================
// Accretion disk Doppler beaming
// ============================================================================

/// Doppler boost for orbiting accretion disk material.
///
/// Computes the relativistic Doppler boost factor delta^(3+alpha) for
/// matter in a circular Keplerian orbit at radius r around a Kerr BH.
///
/// The orbital velocity is:
///   v_phi/c = sqrt(M / (r - 2M + a*sqrt(M/r)))   (units where M=1)
///
/// Line-of-sight velocity component:
///   cos(angle) = sin(i) cos(phi)
///
/// where i is the disk inclination and phi is the azimuthal angle
/// (phi=0 = approaching limb, phi=pi = receding limb).
///
/// Returns 1.0 if the orbit is unphysical (inside ISCO or singular).
///
/// Arguments:
///   r: radius in units of M
///   a_star: dimensionless spin (-1 < a* < 1)
///   phi: azimuthal angle [rad] (0 = approaching)
///   inclination: disk inclination [rad] (0 = face-on, pi/2 = edge-on)
///   alpha: spectral index (F_nu ~ nu^alpha), use 0 for blackbody
///
/// References:
///   - Cunningham (1975): ApJ 202, 788
///   - Begelman, Blandford, Rees (1984): Rev. Mod. Phys. 56, 255
pub fn disk_doppler_boost(
    r: f64,
    a_star: f64,
    phi: f64,
    inclination: f64,
    alpha: f64,
) -> f64 {
    let a = a_star.clamp(-0.9999, 0.9999);
    let r = r.max(1.1); // avoid singularities

    // Keplerian discriminant (r - 2M + a*sqrt(M/r)) with M=1
    let discriminant = r - 2.0 + a * (1.0 / r).sqrt();
    if discriminant <= 0.0 {
        return 1.0; // inside ISCO or unphysical
    }

    let v_orbital = (1.0 / discriminant).sqrt();
    let beta = v_orbital.min(0.99); // cap at 0.99c

    // Line-of-sight velocity: v_los = v_phi * sin(i) * cos(phi)
    // Angle between velocity and line of sight:
    // cos(theta) = sin(inclination) * cos(phi)
    let cos_theta = inclination.sin() * phi.cos();
    let theta = cos_theta.clamp(-1.0, 1.0).acos();

    let delta = doppler_factor(beta, theta);
    let boost = delta.powf(3.0 + alpha);

    boost.clamp(0.01, 1000.0)
}

/// Maximum Doppler boost for disk (edge-on, approaching limb).
pub fn disk_doppler_boost_max(r: f64, a_star: f64, alpha: f64) -> f64 {
    disk_doppler_boost(r, a_star, 0.0, PI / 2.0, alpha)
}

/// Minimum Doppler boost for disk (edge-on, receding limb).
pub fn disk_doppler_boost_min(r: f64, a_star: f64, alpha: f64) -> f64 {
    disk_doppler_boost(r, a_star, PI, PI / 2.0, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    // -- Lorentz factor --

    #[test]
    fn test_lorentz_factor_stationary() {
        let g = lorentz_factor(0.0);
        assert!((g - 1.0).abs() < TOL);
    }

    #[test]
    fn test_lorentz_factor_relativistic() {
        // beta=0.6: gamma = 1/sqrt(1-0.36) = 1/sqrt(0.64) = 1.25
        let g = lorentz_factor(0.6);
        assert!((g - 1.25).abs() < TOL, "gamma = {g}");
    }

    #[test]
    fn test_lorentz_factor_ultra_relativistic() {
        // beta=0.99: gamma ~ 7.089
        let g = lorentz_factor(0.99);
        assert!((g - 7.0888).abs() < 0.001, "gamma = {g}");
    }

    #[test]
    fn test_lorentz_factor_luminal() {
        assert!(lorentz_factor(1.0).is_infinite());
    }

    #[test]
    fn test_beta_from_gamma_round_trip() {
        for beta in [0.1, 0.5, 0.9, 0.99] {
            let g = lorentz_factor(beta);
            let b = beta_from_gamma(g);
            assert!((b - beta).abs() < TOL, "beta={beta} -> gamma={g} -> beta={b}");
        }
    }

    // -- Doppler factor --

    #[test]
    fn test_doppler_approaching() {
        // theta=0 (approaching): delta = sqrt((1+b)/(1-b))
        let d = doppler_factor(0.9, 0.0);
        let expected = ((1.0 + 0.9) / (1.0 - 0.9_f64)).sqrt();
        assert!((d - expected).abs() < 0.001, "delta = {d}, expected {expected}");
    }

    #[test]
    fn test_doppler_receding() {
        // theta=pi (receding): delta = sqrt((1-b)/(1+b))
        let d = doppler_factor(0.9, PI);
        let expected = ((1.0 - 0.9) / (1.0 + 0.9_f64)).sqrt();
        assert!((d - expected).abs() < 0.001, "delta = {d}, expected {expected}");
    }

    #[test]
    fn test_doppler_transverse() {
        // theta=pi/2: delta = 1/gamma (transverse Doppler)
        let d = doppler_factor(0.9, PI / 2.0);
        let expected = 1.0 / lorentz_factor(0.9);
        assert!((d - expected).abs() < 0.001, "delta = {d}, expected {expected}");
    }

    #[test]
    fn test_doppler_approaching_convenience() {
        let d1 = doppler_factor_approaching(0.9);
        let d2 = doppler_factor(0.9, 0.0);
        assert!((d1 - d2).abs() < 0.001);
    }

    #[test]
    fn test_doppler_receding_convenience() {
        let d1 = doppler_factor_receding(0.9);
        let d2 = doppler_factor(0.9, PI);
        assert!((d1 - d2).abs() < 0.001);
    }

    #[test]
    fn test_doppler_product() {
        // delta_approaching * delta_receding = 1 (for same beta)
        let da = doppler_factor_approaching(0.9);
        let dr = doppler_factor_receding(0.9);
        assert!((da * dr - 1.0).abs() < TOL, "product = {}", da * dr);
    }

    // -- Frequency shifts --

    #[test]
    fn test_doppler_shift_blueshift() {
        // Approaching: observed > emitted
        let nu_obs = doppler_shift_relativistic(1e9, 0.5, 0.0);
        assert!(nu_obs > 1e9);
    }

    #[test]
    fn test_doppler_shift_redshift() {
        // Receding: observed < emitted
        let nu_obs = doppler_shift_relativistic(1e9, 0.5, PI);
        assert!(nu_obs < 1e9);
    }

    #[test]
    fn test_transverse_always_redshift() {
        // Time dilation: nu_obs < nu_emit
        let nu_obs = doppler_shift_transverse(1e9, 0.9);
        assert!(nu_obs < 1e9, "nu_obs = {nu_obs}");
    }

    #[test]
    fn test_cosmological_redshift_round_trip() {
        let z = 1.0;
        let nu_emit = 1e15;
        let nu_obs = observed_frequency(nu_emit, z);
        let nu_back = rest_frame_frequency(nu_obs, z);
        assert!((nu_back - nu_emit).abs() < 1.0);
    }

    // -- Relativistic beaming --

    #[test]
    fn test_beaming_boost_approaching() {
        // Approaching jet: large boost
        let f_obs = relativistic_beaming_flux(1.0, 0.99, 0.1, 0.0);
        assert!(f_obs > 100.0, "should be highly boosted: {f_obs}");
    }

    #[test]
    fn test_beaming_deboosted_receding() {
        // Receding counterjet: strongly suppressed
        let f_obs = relativistic_beaming_flux(1.0, 0.99, PI - 0.1, 0.0);
        assert!(f_obs < 0.01, "should be strongly deboosted: {f_obs}");
    }

    #[test]
    fn test_beaming_steeper_spectrum_more_boost() {
        // Steeper spectrum (larger alpha) -> more beaming
        let f_flat = relativistic_beaming_flux(1.0, 0.9, 0.3, 0.0);
        let f_steep = relativistic_beaming_flux(1.0, 0.9, 0.3, 1.0);
        assert!(f_steep > f_flat, "steeper spectrum should boost more");
    }

    // -- Superluminal motion --

    #[test]
    fn test_superluminal_exceeds_c() {
        // Ultrarelativistic jet at optimal angle: beta_app > 1
        let theta_opt = superluminal_optimal_angle(0.99);
        let beta_app = apparent_superluminal_velocity(0.99, theta_opt);
        assert!(beta_app > 1.0, "beta_app = {beta_app}");
    }

    #[test]
    fn test_superluminal_max_equals_gamma_beta() {
        let beta = 0.99;
        let v_max = superluminal_velocity_max(beta);
        let expected = lorentz_factor(beta) * beta;
        assert!((v_max - expected).abs() < TOL);
    }

    #[test]
    fn test_superluminal_at_optimal_angle() {
        let beta = 0.95;
        let theta = superluminal_optimal_angle(beta);
        let v = apparent_superluminal_velocity(beta, theta);
        let v_max = superluminal_velocity_max(beta);
        // At optimal angle, should get maximum
        assert!((v - v_max).abs() / v_max < 0.01, "v = {v}, v_max = {v_max}");
    }

    // -- Aberration --

    #[test]
    fn test_aberration_moving_observer() {
        // For observer moving at beta=0.9 past a source at rest:
        // Light emitted perpendicular (theta=pi/2) in source frame appears
        // to come from behind (theta' > pi/2) because the observer outruns it.
        // cos(theta') = (0 - 0.9)/(1 - 0) = -0.9 -> theta' ~ 2.69 rad
        let theta_prime = relativistic_aberration(PI / 2.0, 0.9);
        assert!(theta_prime > PI / 2.0, "should appear from behind: {theta_prime}");
        let expected = (-0.9_f64).acos();
        assert!((theta_prime - expected).abs() < 0.01);
    }

    #[test]
    fn test_aberration_stationary() {
        // beta=0: no aberration
        let theta_prime = relativistic_aberration(1.0, 0.0);
        assert!((theta_prime - 1.0).abs() < TOL);
    }

    #[test]
    fn test_beaming_half_angle() {
        let gamma = 10.0;
        let angle = beaming_half_angle(gamma);
        assert!((angle - 0.1).abs() < TOL);
    }

    #[test]
    fn test_beaming_half_angle_nonrelativistic() {
        let angle = beaming_half_angle(1.0);
        assert!((angle - 1.0).abs() < TOL); // 1 radian ~ isotropic
    }

    // -- K-corrections --

    #[test]
    fn test_k_correction_flat_spectrum() {
        // alpha = 0: K = (1+z)^1
        let k = k_correction_factor(1.0, 0.0);
        assert!((k - 2.0).abs() < TOL);
    }

    #[test]
    fn test_k_correction_steep_spectrum() {
        // alpha = -1: K = (1+z)^0 = 1
        let k = k_correction_factor(1.0, -1.0);
        assert!((k - 1.0).abs() < TOL);
    }

    #[test]
    fn test_k_correction_power_law_applied() {
        let f_rest = k_correction_power_law(1.0, 1.0, 0.5);
        let expected = (2.0_f64).powf(1.5);
        assert!((f_rest - expected).abs() < TOL);
    }

    // -- Disk Doppler boost --

    #[test]
    fn test_disk_boost_face_on_near_unity() {
        // Face-on (i=0): no line-of-sight velocity, but transverse Doppler
        // (time dilation) causes delta = 1/gamma < 1 -> boost = delta^3 < 1.
        // At large r, v_orbital is small so boost approaches 1.
        let boost = disk_doppler_boost(100.0, 0.0, 0.0, 0.0, 0.0);
        assert!((boost - 1.0).abs() < 0.05, "face-on boost at r=100M = {boost}");
    }

    #[test]
    fn test_disk_boost_asymmetry() {
        // Edge-on: approaching side brighter than receding
        let boost_app = disk_doppler_boost_max(6.0, 0.0, 0.0);
        let boost_rec = disk_doppler_boost_min(6.0, 0.0, 0.0);
        assert!(
            boost_app > boost_rec,
            "approaching {boost_app} should exceed receding {boost_rec}"
        );
    }

    #[test]
    fn test_disk_boost_product_near_unity() {
        // For edge-on, boost_max * boost_min should be close to 1
        // (this is exact for the simple model without GR corrections)
        let bmax = disk_doppler_boost_max(20.0, 0.0, 0.0);
        let bmin = disk_doppler_boost_min(20.0, 0.0, 0.0);
        let product = bmax * bmin;
        // Product won't be exactly 1 because delta^3 rather than delta^1
        // but should be O(1) for large radii
        assert!(product > 0.1 && product < 10.0, "product = {product}");
    }

    #[test]
    fn test_disk_boost_stronger_at_small_radius() {
        // Faster orbit -> more beaming
        let b6 = disk_doppler_boost_max(6.0, 0.0, 0.0);
        let b20 = disk_doppler_boost_max(20.0, 0.0, 0.0);
        assert!(b6 > b20, "inner orbit should beam more: {b6} vs {b20}");
    }
}
