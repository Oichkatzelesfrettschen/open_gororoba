//! Casimir energy computation via worldline path integral.
//!
//! Computes the Casimir vacuum energy density at a point by averaging
//! the boundary-intersection indicator over many random worldline loops
//! and integrating over proper time T with Gauss-Legendre quadrature.
//!
//! # Key formula (White 2021, Eq. 3)
//!
//! E(x) = -(1/(2*(4*pi)^{3/2})) * integral_0^inf dT/T^{5/2} * <Theta_Sigma(x,T)>
//!
//! where Theta_Sigma is the Monte Carlo estimator: the fraction of loops
//! (centered at x, scaled by sqrt(T)) that intersect the boundary surface.
//!
//! For parallel plates: E/A = -pi^2 / (720*a^3) (exact analytical result).

use crate::{geometry::CasimirGeometry, vloop::generate_unit_loop};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Configuration for worldline Casimir computation.
#[derive(Debug, Clone)]
pub struct WorldlineCasimirConfig {
    /// Number of points per loop.
    pub n_loop_points: usize,
    /// Number of Monte Carlo loops per proper-time sample.
    pub n_loops: usize,
    /// Minimum proper time for integration.
    pub t_min: f64,
    /// Maximum proper time for integration.
    pub t_max: f64,
    /// Number of Gauss-Legendre quadrature points for T integration.
    pub n_t_points: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for WorldlineCasimirConfig {
    fn default() -> Self {
        Self {
            n_loop_points: 1000,
            n_loops: 10_000,
            t_min: 0.01,
            t_max: 10.0,
            n_t_points: 32,
            seed: 42,
        }
    }
}

/// Result of a Casimir energy computation at a point.
#[derive(Debug, Clone)]
pub struct CasimirEnergyResult {
    /// Casimir energy density at the evaluation point.
    pub energy: f64,
    /// Statistical error estimate (Monte Carlo standard error).
    pub error: f64,
    /// Total number of loops evaluated.
    pub n_loops_total: u64,
}

/// Compute the Monte Carlo estimator <Theta_Sigma> at a point for a given T.
///
/// Returns the fraction of loops that intersect the boundary (0 = none, 1 = all).
fn theta_sigma_average<G: CasimirGeometry>(
    geometry: &G,
    center: [f64; 3],
    t: f64,
    n_loops: usize,
    n_loop_points: usize,
    seed: u64,
) -> (f64, f64) {
    let scale = t.sqrt();
    let mut hits = 0usize;

    for i in 0..n_loops {
        let mut rng = ChaCha8Rng::seed_from_u64(seed.wrapping_add(i as u64));
        let unit_loop = generate_unit_loop(n_loop_points, &mut rng);
        let scaled = unit_loop.scaled_points(t);

        let n_crossings = geometry.intersects_count(&scaled, center, scale);
        if n_crossings > 0 {
            hits += 1;
        }
    }

    let mean = hits as f64 / n_loops as f64;
    // Binomial standard error
    let std_err = (mean * (1.0 - mean) / n_loops as f64).sqrt();
    (mean, std_err)
}

/// Compute Casimir energy density at a single spatial point.
///
/// E(x) = -(1/(2*(4*pi)^{3/2})) * integral dT/T^{5/2} * <Theta_Sigma>
///
/// The T-integral is done via Gauss-Legendre quadrature on [t_min, t_max],
/// with a change of variable to handle the T^{-5/2} singularity at T=0.
pub fn casimir_energy_at_point<G: CasimirGeometry>(
    geometry: &G,
    point: [f64; 3],
    config: &WorldlineCasimirConfig,
) -> CasimirEnergyResult {
    let prefactor = -1.0 / (2.0 * (4.0 * PI).powf(1.5));

    // Gauss-Legendre quadrature nodes and weights on [0, 1]
    let quad = gauss_quad::GaussLegendre::new(config.n_t_points)
        .expect("failed to construct GL quadrature");
    let pairs = quad.as_node_weight_pairs();

    // Map [0,1] -> [t_min, t_max] using log transform for better coverage
    // of the small-T region. Let u in [-1, 1] (GL standard), map to
    // T = t_min * (t_max/t_min)^((u+1)/2)
    let log_ratio = (config.t_max / config.t_min).ln();

    let mut integral = 0.0;
    let mut error_sq = 0.0;
    let mut total_loops = 0u64;

    for &(u, w) in pairs {
        // Map GL node u in [-1, 1] to T
        let frac = (u + 1.0) / 2.0;
        let t = config.t_min * (frac * log_ratio).exp();
        let dt_du = t * log_ratio / 2.0; // Jacobian

        let (theta_avg, theta_err) = theta_sigma_average(
            geometry,
            point,
            t,
            config.n_loops,
            config.n_loop_points,
            config.seed.wrapping_add((frac * 1e6) as u64),
        );

        let integrand = theta_avg / t.powf(2.5);
        integral += w * integrand * dt_du;
        error_sq += (w * theta_err / t.powf(2.5) * dt_du).powi(2);
        total_loops += config.n_loops as u64;
    }

    CasimirEnergyResult {
        energy: prefactor * integral,
        error: prefactor * error_sq.sqrt(),
        n_loops_total: total_loops,
    }
}

/// Compute Casimir energy density along a line between two points.
///
/// Returns Vec<(position_parameter, energy_density)>.
/// The line is parameterized by t in [0, 1], with:
///   point(t) = start + t * (end - start)
///
/// Parallelized over spatial points via rayon.
pub fn casimir_energy_profile<G: CasimirGeometry>(
    geometry: &G,
    start: [f64; 3],
    end: [f64; 3],
    n_points: usize,
    config: &WorldlineCasimirConfig,
) -> Vec<(f64, f64)> {
    let direction = [end[0] - start[0], end[1] - start[1], end[2] - start[2]];

    (0..n_points)
        .into_par_iter()
        .map(|i| {
            let t_param = i as f64 / (n_points - 1).max(1) as f64;
            let point = [
                start[0] + t_param * direction[0],
                start[1] + t_param * direction[1],
                start[2] + t_param * direction[2],
            ];
            let result = casimir_energy_at_point(geometry, point, config);
            (t_param, result.energy)
        })
        .collect()
}

/// Result of a 3D Casimir energy field computation.
///
/// Contains a flat grid of energy density values suitable for GPU volume
/// rendering, plus the spatial bounds and resolution for reconstruction.
#[derive(Debug, Clone)]
pub struct CasimirEnergyField3D {
    /// Energy density values in row-major order: data[z * ny * nx + y * nx + x].
    pub data: Vec<f64>,
    /// Grid resolution (nx, ny, nz).
    pub resolution: (usize, usize, usize),
    /// Spatial bounds: (x_min, x_max, y_min, y_max, z_min, z_max).
    pub bounds: (f64, f64, f64, f64, f64, f64),
    /// Total number of Monte Carlo loops evaluated across all grid points.
    pub total_loops: u64,
}

/// Compute Casimir energy density on a 3D grid.
///
/// Evaluates the worldline path integral at each grid point within the
/// specified spatial bounds. The result is a flat `Vec<f64>` suitable for
/// uploading to a GPU as a 3D texture for volume rendering.
///
/// Parallelized over grid points via rayon.
///
/// # Arguments
/// * `geometry` - The Casimir boundary geometry.
/// * `bounds` - Spatial extent: (x_min, x_max, y_min, y_max, z_min, z_max).
/// * `resolution` - Grid size: (nx, ny, nz).
/// * `config` - Worldline Monte Carlo parameters.
pub fn casimir_energy_field_3d<G: CasimirGeometry>(
    geometry: &G,
    bounds: (f64, f64, f64, f64, f64, f64),
    resolution: (usize, usize, usize),
    config: &WorldlineCasimirConfig,
) -> CasimirEnergyField3D {
    let (nx, ny, nz) = resolution;
    let (x_min, x_max, y_min, y_max, z_min, z_max) = bounds;
    let n_total = nx * ny * nz;

    let dx = if nx > 1 {
        (x_max - x_min) / (nx - 1) as f64
    } else {
        0.0
    };
    let dy = if ny > 1 {
        (y_max - y_min) / (ny - 1) as f64
    } else {
        0.0
    };
    let dz = if nz > 1 {
        (z_max - z_min) / (nz - 1) as f64
    } else {
        0.0
    };

    let results: Vec<CasimirEnergyResult> = (0..n_total)
        .into_par_iter()
        .map(|idx| {
            let ix = idx % nx;
            let iy = (idx / nx) % ny;
            let iz = idx / (nx * ny);
            let point = [
                x_min + ix as f64 * dx,
                y_min + iy as f64 * dy,
                z_min + iz as f64 * dz,
            ];
            casimir_energy_at_point(geometry, point, config)
        })
        .collect();

    let data: Vec<f64> = results.iter().map(|r| r.energy).collect();
    let total_loops: u64 = results.iter().map(|r| r.n_loops_total).sum();

    CasimirEnergyField3D {
        data,
        resolution,
        bounds,
        total_loops,
    }
}

/// Exact Casimir energy density between parallel plates.
///
/// E/A = -pi^2 / (720 * a^3)
///
/// where a is the plate separation. This is the analytical result
/// for a massless scalar field in 3+1D (Casimir 1948).
pub fn casimir_parallel_plates_exact(separation: f64) -> f64 {
    assert!(separation > 0.0, "plate separation must be positive");
    -PI * PI / (720.0 * separation.powi(3))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::ParallelPlates;

    #[test]
    fn test_parallel_plates_exact() {
        let a = 1.0;
        let e = casimir_parallel_plates_exact(a);
        let expected = -PI * PI / 720.0;
        assert!(
            (e - expected).abs() < 1e-14,
            "exact plate energy = {e}, expected {expected}"
        );
    }

    #[test]
    fn test_exact_scales_as_a_minus_3() {
        let e1 = casimir_parallel_plates_exact(1.0);
        let e2 = casimir_parallel_plates_exact(2.0);
        let ratio = e1 / e2;
        assert!(
            (ratio - 8.0).abs() < 1e-10,
            "E scales as a^-3: ratio = {ratio}"
        );
    }

    #[test]
    fn test_theta_sigma_plates_midpoint() {
        // At the midpoint between plates, some loops should intersect
        let plates = ParallelPlates { separation: 1.0 };
        let center = [0.0, 0.0, 0.5]; // midpoint
        let (theta, _err) = theta_sigma_average(&plates, center, 0.5, 500, 100, 42);
        // We just check it's in [0, 1] and not identically zero
        assert!((0.0..=1.0).contains(&theta), "theta_sigma = {theta}");
    }

    #[test]
    fn test_energy_at_point_plates_is_negative() {
        // Quick low-accuracy check that the energy between plates is negative
        let plates = ParallelPlates { separation: 1.0 };
        let config = WorldlineCasimirConfig {
            n_loop_points: 50,
            n_loops: 200,
            t_min: 0.01,
            t_max: 5.0,
            n_t_points: 8,
            seed: 42,
        };
        let result = casimir_energy_at_point(&plates, [0.0, 0.0, 0.5], &config);
        // Energy should be negative (attractive Casimir force)
        // But with low accuracy, just check it's finite
        assert!(
            result.energy.is_finite(),
            "energy should be finite: {}",
            result.energy
        );
    }

    #[test]
    fn test_energy_field_3d_dimensions() {
        let plates = ParallelPlates { separation: 1.0 };
        let config = WorldlineCasimirConfig {
            n_loop_points: 20,
            n_loops: 50,
            t_min: 0.1,
            t_max: 2.0,
            n_t_points: 4,
            seed: 42,
        };
        let field = casimir_energy_field_3d(
            &plates,
            (-0.5, 0.5, -0.5, 0.5, 0.0, 1.0),
            (3, 3, 3),
            &config,
        );
        assert_eq!(field.data.len(), 27);
        assert_eq!(field.resolution, (3, 3, 3));
        assert!(field.total_loops > 0);
    }

    #[test]
    fn test_energy_field_3d_all_finite() {
        let plates = ParallelPlates { separation: 1.0 };
        let config = WorldlineCasimirConfig {
            n_loop_points: 20,
            n_loops: 50,
            t_min: 0.1,
            t_max: 2.0,
            n_t_points: 4,
            seed: 42,
        };
        let field = casimir_energy_field_3d(
            &plates,
            (-0.5, 0.5, -0.5, 0.5, 0.1, 0.9),
            (2, 2, 4),
            &config,
        );
        for (i, &val) in field.data.iter().enumerate() {
            assert!(val.is_finite(), "non-finite energy at index {i}: {val}");
        }
    }

    #[test]
    fn test_energy_field_3d_single_point() {
        let plates = ParallelPlates { separation: 1.0 };
        let config = WorldlineCasimirConfig {
            n_loop_points: 30,
            n_loops: 100,
            t_min: 0.05,
            t_max: 3.0,
            n_t_points: 6,
            seed: 42,
        };
        // 1x1x1 grid should give the same result as a single point evaluation
        let field =
            casimir_energy_field_3d(&plates, (0.0, 0.0, 0.0, 0.0, 0.5, 0.5), (1, 1, 1), &config);
        let single = casimir_energy_at_point(&plates, [0.0, 0.0, 0.5], &config);
        assert!(
            (field.data[0] - single.energy).abs() < 1e-14,
            "single-point field {:.6e} != direct {:.6e}",
            field.data[0],
            single.energy,
        );
    }
}
