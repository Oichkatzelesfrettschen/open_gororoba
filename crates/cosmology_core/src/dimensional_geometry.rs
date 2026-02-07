//! Analytic continuation of sphere surface areas and ball volumes.
//!
//! The unit sphere surface area S_{d-1} and ball volume V_d extend to
//! non-integer dimensions d via the Gamma function:
//!   S_{d-1} = 2 * pi^(d/2) / Gamma(d/2)
//!   V_d(r)  = pi^(d/2) / Gamma(d/2 + 1) * r^d
//!
//! These satisfy S_{d-1} = d * V_d for all d (away from Gamma poles).
//! Poles occur at d = 0, -2, -4, ... (non-positive even integers).
//!
//! # Literature
//! - Stillwell (1992): "The word problem and repetitions in groups"
//! - Krasnov (2020): "A dimensional approach to Casimir effects"

use std::f64::consts::PI;
use statrs::function::gamma::ln_gamma;

/// Surface area of the unit (d-1)-sphere embedded in R^d.
///
/// Analytically continued to non-integer d via:
///   S_{d-1} = 2 * pi^(d/2) / Gamma(d/2)
///
/// Returns f64::INFINITY at Gamma poles (d/2 = 0, -1, -2, ...).
pub fn unit_sphere_surface_area(d: f64) -> f64 {
    let z = d / 2.0;
    if z <= 0.0 && z == z.floor() {
        // Gamma pole at non-positive integer
        return f64::INFINITY;
    }
    2.0 * PI.powf(z) / gamma_real(z)
}

/// Volume of a d-dimensional ball of radius r.
///
/// Analytically continued to non-integer d via:
///   V_d(r) = pi^(d/2) / Gamma(d/2 + 1) * r^d
pub fn ball_volume(d: f64, r: f64) -> f64 {
    let z = d / 2.0;
    let z1 = z + 1.0;
    if z1 <= 0.0 && z1 == z1.floor() {
        return f64::INFINITY;
    }
    PI.powf(z) / gamma_real(z1) * r.powf(d)
}

/// Sample S_{d-1} and V_d over a uniform grid of d values.
///
/// Returns (dimensions, volumes, surface_areas) as parallel arrays.
pub fn sample_dimensional_range(
    d_min: f64,
    d_max: f64,
    n: usize,
    r: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let ds: Vec<f64> = (0..n)
        .map(|i| d_min + (d_max - d_min) * i as f64 / (n - 1).max(1) as f64)
        .collect();
    let vols: Vec<f64> = ds.iter().map(|&d| ball_volume(d, r)).collect();
    let areas: Vec<f64> = ds.iter().map(|&d| unit_sphere_surface_area(d)).collect();
    (ds, vols, areas)
}

/// Compute Gamma(z) for real z using statrs ln_gamma.
///
/// For z > 0, Gamma(z) = exp(ln_gamma(z)).
/// For z < 0 (not a non-positive integer), uses the reflection formula:
///   Gamma(z) = pi / (sin(pi*z) * Gamma(1-z))
fn gamma_real(z: f64) -> f64 {
    if z > 0.0 {
        ln_gamma(z).exp()
    } else if z == z.floor() {
        // Pole at non-positive integer
        f64::INFINITY
    } else {
        // Reflection formula: Gamma(z) * Gamma(1-z) = pi / sin(pi*z)
        let sin_pz = (PI * z).sin();
        PI / (sin_pz * ln_gamma(1.0 - z).exp())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ball_volume_known_integers() {
        let eps = 1e-12;
        assert!((ball_volume(0.0, 1.0) - 1.0).abs() < eps);
        assert!((ball_volume(1.0, 1.0) - 2.0).abs() < eps);
        assert!((ball_volume(2.0, 1.0) - PI).abs() < eps);
        assert!((ball_volume(3.0, 1.0) - 4.0 * PI / 3.0).abs() < eps);
    }

    #[test]
    fn test_unit_sphere_surface_area_known_integers() {
        let eps = 1e-12;
        // S^0 = 2 points (d=1)
        assert!((unit_sphere_surface_area(1.0) - 2.0).abs() < eps);
        // S^1 = circle circumference 2*pi (d=2)
        assert!((unit_sphere_surface_area(2.0) - 2.0 * PI).abs() < eps);
        // S^2 = sphere surface 4*pi (d=3)
        assert!((unit_sphere_surface_area(3.0) - 4.0 * PI).abs() < eps);
    }

    #[test]
    fn test_area_volume_relation() {
        // S_{d-1} = d * V_d for all d (away from poles)
        let d = 2.5;
        let s = unit_sphere_surface_area(d);
        let v = ball_volume(d, 1.0);
        assert!(
            (s - d * v).abs() < 1e-10,
            "S_{{d-1}} = {s}, d * V_d = {}",
            d * v
        );
    }

    #[test]
    fn test_ball_volume_scales_with_radius() {
        let d = 3.0;
        let r = 2.0;
        let v1 = ball_volume(d, 1.0);
        let vr = ball_volume(d, r);
        assert!((vr - v1 * r.powi(3)).abs() < 1e-10);
    }

    #[test]
    fn test_fractional_dimension() {
        // d=2.5 should give a finite positive value
        let v = ball_volume(2.5, 1.0);
        assert!(v > 0.0 && v.is_finite(), "V_2.5 = {v}");
        let s = unit_sphere_surface_area(2.5);
        assert!(s > 0.0 && s.is_finite(), "S_1.5 = {s}");
    }
}
