//! Acoustic metric and analog gravity from LBM velocity fields.
//!
//! Unruh (1981) showed that sound waves in a moving fluid obey the same equation
//! as a scalar field in a curved spacetime, with effective metric:
//!
//!   g_mu_nu = (rho / c_s) * [-((c_s^2 - v^2)), -v_j; -v_i, delta_ij]
//!
//! An acoustic horizon forms where |v| = c_s (local flow speed equals sound speed).
//! The acoustic surface gravity kappa determines an analog Hawking temperature:
//!
//!   T_H = hbar * kappa / (2 * pi * k_B)
//!
//! References:
//!   - Unruh (1981): "Experimental Black-Hole Evaporation?" PRL 46, 1351
//!   - Visser (1998): "Acoustic black holes" CQG 15, 1767
//!   - Barcelo, Liberati, Visser (2011): "Analogue Gravity" LRR 14, 3

use std::f64::consts::PI;

/// Speed of sound in a D3Q19 LBM lattice: c_s = 1/sqrt(3) in lattice units.
pub const LBM_CS: f64 = 0.577_350_269_189_626; // 1/sqrt(3)

/// Find acoustic horizon locations in a 1D velocity profile.
///
/// Returns indices where |v| crosses c_s (the sound speed).
/// Uses linear interpolation to find crossing points.
pub fn acoustic_horizon_1d(velocity_magnitude: &[f64], c_s: f64) -> Vec<f64> {
    let n = velocity_magnitude.len();
    let mut crossings = Vec::new();

    for i in 0..n.saturating_sub(1) {
        let v0 = velocity_magnitude[i];
        let v1 = velocity_magnitude[i + 1];

        // Check if c_s is crossed between i and i+1
        if (v0 - c_s) * (v1 - c_s) < 0.0 {
            // Linear interpolation for crossing position
            let frac = (c_s - v0) / (v1 - v0);
            crossings.push(i as f64 + frac);
        }
    }

    crossings
}

/// Compute acoustic surface gravity from velocity gradient at the horizon.
///
/// kappa = (1/2) * |d|v|/dr| evaluated at the horizon (where |v| = c_s).
/// Uses finite differences for the gradient.
///
/// # Arguments
/// * `velocity_magnitude` - 1D array of |v| at each grid point
/// * `dx` - Grid spacing
/// * `horizon_pos` - Position of the horizon (fractional index)
pub fn acoustic_surface_gravity(velocity_magnitude: &[f64], dx: f64, horizon_pos: f64) -> f64 {
    let n = velocity_magnitude.len();
    let i = horizon_pos.floor() as usize;

    if i + 1 >= n {
        return 0.0;
    }

    // Gradient of |v| at the horizon via finite difference
    let dv_dr = (velocity_magnitude[i + 1] - velocity_magnitude[i]) / dx;

    0.5 * dv_dr.abs()
}

/// Compute analog Hawking temperature from acoustic surface gravity.
///
/// T_H = hbar * kappa / (2 * pi * k_B)
///
/// In lattice units where hbar and k_B are 1:
/// T_H = kappa / (2 * pi)
///
/// In physical units, multiply by hbar/k_B = 7.638e-12 K*s.
pub fn acoustic_hawking_temperature(kappa: f64) -> f64 {
    kappa / (2.0 * PI)
}

/// Acoustic Hawking temperature in physical units (Kelvin).
///
/// # Arguments
/// * `kappa` - Surface gravity [s^-1]
/// * `hbar` - Reduced Planck constant [J*s]
/// * `k_b` - Boltzmann constant [J/K]
pub fn acoustic_hawking_temperature_physical(kappa: f64, hbar: f64, k_b: f64) -> f64 {
    hbar * kappa / (2.0 * PI * k_b)
}

/// Generate a radial inflow velocity profile for an acoustic black hole analog.
///
/// Models a sink flow: v(r) = -v_0 * (r_0 / r)^alpha
/// The acoustic horizon is at r_h where |v(r_h)| = c_s.
///
/// # Arguments
/// * `n` - Number of grid points
/// * `v0` - Characteristic velocity at reference radius
/// * `r0` - Reference radius (in grid units)
/// * `alpha` - Power-law exponent (1 for 2D, 2 for 3D point sink)
/// * `c_s` - Speed of sound
///
/// Returns (velocity_magnitudes, dx) where dx = 1.0 for unit grid spacing.
pub fn radial_inflow_profile(n: usize, v0: f64, r0: f64, alpha: f64, c_s: f64) -> (Vec<f64>, f64) {
    let dx = 1.0;
    let center = n as f64 / 2.0;

    let velocities: Vec<f64> = (0..n)
        .map(|i| {
            let r = (i as f64 - center).abs().max(0.5); // avoid singularity at center
            let v = v0 * (r0 / r).powf(alpha);
            v.min(c_s * 5.0) // cap at 5*c_s to avoid unphysical values
        })
        .collect();

    // Scale so that the profile actually crosses c_s somewhere
    let v_max = velocities.iter().cloned().fold(0.0_f64, f64::max);
    let v_min = velocities.iter().cloned().fold(f64::INFINITY, f64::min);

    if v_max < c_s || v_min > c_s {
        // Rescale to ensure horizon exists
        let scale = c_s * 1.5 / v_max;
        let scaled: Vec<f64> = velocities.iter().map(|&v| v * scale).collect();
        return (scaled, dx);
    }

    (velocities, dx)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lbm_cs_value() {
        let expected = 1.0 / 3.0_f64.sqrt();
        assert!((LBM_CS - expected).abs() < 1e-12);
    }

    #[test]
    fn test_acoustic_horizon_crossing() {
        // Linear velocity profile crossing c_s between grid points
        let n = 100;
        let c_s = 0.505; // avoids exact grid alignment at i=50
        let velocities: Vec<f64> = (0..n)
            .map(|i| i as f64 / n as f64) // 0.0 to ~0.99
            .collect();
        let crossings = acoustic_horizon_1d(&velocities, c_s);
        assert_eq!(crossings.len(), 1, "should have exactly one crossing");
        assert!(
            (crossings[0] - 50.5).abs() < 1.0,
            "crossing near index 50.5"
        );
    }

    #[test]
    fn test_surface_gravity_positive() {
        let velocities: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
        let kappa = acoustic_surface_gravity(&velocities, 1.0, 50.0);
        assert!(kappa > 0.0, "kappa should be positive for inflow");
    }

    #[test]
    fn test_hawking_temperature_proportional_to_kappa() {
        let t1 = acoustic_hawking_temperature(1.0);
        let t2 = acoustic_hawking_temperature(2.0);
        assert!(
            (t2 / t1 - 2.0).abs() < 1e-10,
            "T should be proportional to kappa"
        );
    }

    #[test]
    fn test_radial_inflow_has_horizon() {
        let (v, _dx) = radial_inflow_profile(128, 0.3, 20.0, 1.0, LBM_CS);
        let crossings = acoustic_horizon_1d(&v, LBM_CS);
        assert!(
            !crossings.is_empty(),
            "radial inflow should have at least one horizon"
        );
    }
}
