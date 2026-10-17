//! Real-scalar Dirichlet interaction energy in three spatial dimensions.
//!
//! Gies, Langfeld, Moyaerts, Eqs. 8, 11, 13 and 44 give
//! e(x) = -1/(32*pi^2) integral dT T^-3 <hit(body1) hit(body2)>.
//! Subtracting the separate body energies removes single-body hits. The
//! density is assigned to loop centers of mass; spatial integration supplies
//! interaction energy. Natural units hbar=c=1 give inverse-length^4 density;
//! multiply by hbar*c for SI energy per volume. Electromagnetic polarization
//! and material-response effects require separate models.
//!
//! Finite proper-time bounds, polygon resolution and quadrature introduce
//! systematic errors outside the reported Monte Carlo standard error.

use crate::{
    geometry::{CasimirGeometry, InteractionGeometryError},
    vloop::generate_unit_loop,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use std::{f64::consts::PI, num::NonZeroUsize};

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
    /// Interaction energy density, or energy per area for the integrated plate estimator.
    pub energy: f64,
    /// Nonnegative Monte Carlo standard error, excluding discretization and truncation.
    /// A zero empirical variance does not establish a confidence interval.
    pub error: f64,
    /// Total number of loops evaluated.
    pub n_loops_total: u64,
}

/// Errors in a checked scalar interaction calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CasimirComputationError {
    /// The geometry fails the explicit two-body contract.
    Geometry(InteractionGeometryError),
    /// Sampling counts, coordinates or integration bounds are invalid.
    InvalidConfiguration,
    /// Floating-point arithmetic cannot represent the requested calculation.
    NumericalRange,
}

impl std::fmt::Display for CasimirComputationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Geometry(error) => error.fmt(formatter),
            Self::InvalidConfiguration => {
                formatter.write_str("invalid scalar worldline sampling configuration")
            }
            Self::NumericalRange => {
                formatter.write_str("scalar worldline result exceeds floating-point range")
            }
        }
    }
}
impl std::error::Error for CasimirComputationError {}
impl From<InteractionGeometryError> for CasimirComputationError {
    fn from(error: InteractionGeometryError) -> Self {
        Self::Geometry(error)
    }
}

/// Sample distinct-body interaction hits using a disjoint ChaCha stream.
fn theta_sigma_average<G: CasimirGeometry>(
    geometry: &G,
    center: [f64; 3],
    t: f64,
    n_loops: usize,
    n_loop_points: usize,
    seed: u64,
    stream: u64,
) -> Result<(f64, f64), CasimirComputationError> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    rng.set_stream(stream);
    let mut hits = 0usize;
    for _ in 0..n_loops {
        let scaled = generate_unit_loop(n_loop_points, &mut rng).scaled_points(t);
        if scaled.iter().any(|point| {
            point.iter().zip(center).any(|(coordinate, translation)| {
                !coordinate.is_finite() || !(coordinate + translation).is_finite()
            })
        }) {
            return Err(CasimirComputationError::NumericalRange);
        }
        let mask = geometry.interaction_hit_mask(&scaled, center)?;
        hits += usize::from(mask[0] && mask[1]);
    }
    let mean = hits as f64 / n_loops as f64;
    let standard_error = (mean * (1.0 - mean) / (n_loops - 1) as f64).sqrt();
    Ok((mean, standard_error))
}

/// Compute a checked loop-center interaction energy density in hbar=c=1 units.
///
/// Each quadrature node uses its own ChaCha stream. Equal seeds across spatial
/// points give common random numbers, so spatial estimates are correlated.
/// The standard error accounts for independent quadrature streams at one point.
pub fn try_casimir_energy_at_point<G: CasimirGeometry>(
    geometry: &G,
    point: [f64; 3],
    config: &WorldlineCasimirConfig,
) -> Result<CasimirEnergyResult, CasimirComputationError> {
    geometry.validate_interaction()?;
    if config.n_loop_points < 3
        || config.n_loops < 2
        || config.n_t_points == 0
        || !config.t_min.is_finite()
        || !config.t_max.is_finite()
        || config.t_min <= 0.0
        || config.t_max <= config.t_min
        || point.iter().any(|coordinate| !coordinate.is_finite())
    {
        return Err(CasimirComputationError::InvalidConfiguration);
    }
    let total_loops = u64::try_from(config.n_loops)
        .ok()
        .and_then(|count| {
            u64::try_from(config.n_t_points)
                .ok()
                .and_then(|nodes| count.checked_mul(nodes))
        })
        .ok_or(CasimirComputationError::InvalidConfiguration)?;
    let prefactor = -1.0 / (32.0 * PI * PI);
    let quad = gauss_quad::GaussLegendre::new(
        NonZeroUsize::new(config.n_t_points).expect("validated quadrature degree"),
    );
    let log_ratio = config.t_max.ln() - config.t_min.ln();
    let mut integral = 0.0;
    let mut error_norm = 0.0_f64;
    for (node, &(abscissa, weight)) in quad.as_node_weight_pairs().iter().enumerate() {
        let fraction = (abscissa + 1.0) / 2.0;
        let proper_time = (config.t_min.ln() + fraction * log_ratio).exp();
        // dT/T^3 = log(t_max/t_min)/(2*T^2) du under logarithmic mapping.
        let quadrature_factor = (weight * (log_ratio / 2.0) / proper_time) / proper_time;
        if !quadrature_factor.is_finite() || quadrature_factor <= 0.0 {
            return Err(CasimirComputationError::NumericalRange);
        }
        let (mean, error) = theta_sigma_average(
            geometry,
            point,
            proper_time,
            config.n_loops,
            config.n_loop_points,
            config.seed,
            node as u64,
        )?;
        let weighted_mean = quadrature_factor * mean;
        if weighted_mean == 0.0 && mean != 0.0 {
            return Err(CasimirComputationError::NumericalRange);
        }
        integral += weighted_mean;
        let weighted_error = quadrature_factor * error;
        if weighted_error == 0.0 && error != 0.0 {
            return Err(CasimirComputationError::NumericalRange);
        }
        error_norm = error_norm.hypot(weighted_error);
    }
    let result = CasimirEnergyResult {
        energy: prefactor * integral,
        error: prefactor.abs() * error_norm,
        n_loops_total: total_loops,
    };
    if result.energy.is_finite()
        && result.error.is_finite()
        && (error_norm == 0.0 || result.error > 0.0)
        && (integral == 0.0 || result.energy < 0.0)
    {
        Ok(result)
    } else {
        Err(CasimirComputationError::NumericalRange)
    }
}

/// Compute a scalar interaction density; use the checked API for input errors.
///
/// # Panics
/// Panics for unsupported geometries, invalid sampling, or unrepresentable
/// arithmetic. Single-body self energies require a separate renormalization.
pub fn casimir_energy_at_point<G: CasimirGeometry>(
    geometry: &G,
    point: [f64; 3],
    config: &WorldlineCasimirConfig,
) -> CasimirEnergyResult {
    try_casimir_energy_at_point(geometry, point, config)
        .expect("valid disconnected scalar interaction")
}

/// Integrate plate center position and proper time analytically for each loop.
///
/// A unit polygon with vertical range R permits a center interval of length
/// max(sqrt(T)*R-a,0). Its exact proper-time integral is R^4/(6*a^3).
/// The estimator therefore has no quadrature, spatial-grid or proper-time
/// cutoff error. Polygon discretization remains a downward magnitude bias.
/// Results have inverse-length^3 units; multiplying by hbar*c gives J/m^2.
/// Independent loops are processed through Rayon's native worker pool; ordered
/// collection and serial moment reduction preserve seeded reproducibility.
pub fn scalar_parallel_plates_worldline(
    separation: f64,
    n_loop_points: usize,
    n_loops: usize,
    seed: u64,
) -> Result<CasimirEnergyResult, CasimirComputationError> {
    if !separation.is_finite() || separation <= 0.0 || n_loop_points < 3 || n_loops < 2 {
        return Err(CasimirComputationError::InvalidConfiguration);
    }
    let samples: Vec<f64> = (0..n_loops)
        .into_par_iter()
        .map(|index| {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            rng.set_stream(index as u64);
            let worldline = generate_unit_loop(n_loop_points, &mut rng);
            let minimum = worldline
                .points
                .iter()
                .map(|point| point[2])
                .fold(f64::INFINITY, f64::min);
            let maximum = worldline
                .points
                .iter()
                .map(|point| point[2])
                .fold(f64::NEG_INFINITY, f64::max);
            (maximum - minimum).powi(4)
        })
        .collect();
    let mut mean = 0.0;
    let mut squared_deviations = 0.0;
    for (index, sample) in samples.into_iter().enumerate() {
        let delta = sample - mean;
        mean += delta / (index + 1) as f64;
        squared_deviations += delta * (sample - mean);
    }
    let error = (squared_deviations.max(0.0) / ((n_loops - 1) as f64 * n_loops as f64)).sqrt();
    if !mean.is_finite() || !error.is_finite() {
        return Err(CasimirComputationError::NumericalRange);
    }
    let scale_moment = |moment: f64| -> Result<f64, CasimirComputationError> {
        let mut scaled = moment / (192.0 * PI * PI);
        for _ in 0..3 {
            scaled /= separation;
        }
        if !scaled.is_finite() || (moment != 0.0 && scaled == 0.0) {
            Err(CasimirComputationError::NumericalRange)
        } else {
            Ok(scaled)
        }
    };
    Ok(CasimirEnergyResult {
        energy: -scale_moment(mean)?,
        error: scale_moment(error)?,
        n_loops_total: n_loops as u64,
    })
}

/// Compute Casimir energy density along a line between two points.
///
/// Returns Vec<(position_parameter, energy_density)>.
/// The line is parameterized by t in [0, 1], with:
///   point(t) = start + t * (end - start)
///
/// Parallelized over spatial points via rayon. Values have inverse-length^4
/// units and share random samples across points.
///
/// # Panics
/// Panics under the same invalid-input conditions as casimir_energy_at_point.
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
/// Parallelized over grid points via rayon. Values have inverse-length^4
/// units and share random samples across points.
///
/// # Panics
/// Panics under the same invalid-input conditions as casimir_energy_at_point
/// or when the requested grid size overflows usize.
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

/// Exact electromagnetic ideal-conductor interaction energy per area.
///
/// E/A = -pi^2 / (720 * a^3)
///
/// where a is the plate separation. This is the analytical result
/// for two electromagnetic polarizations in 3+1 dimensions.
/// The historical helper name retains its electromagnetic normalization.
pub fn casimir_parallel_plates_exact(separation: f64) -> f64 {
    assert!(
        separation.is_finite() && separation > 0.0,
        "plate separation must be finite and positive"
    );
    -PI * PI / (720.0 * separation.powi(3))
}

/// Exact real-scalar Dirichlet plate interaction energy per area, hbar=c=1.
pub fn scalar_parallel_plates_exact(separation: f64) -> f64 {
    casimir_parallel_plates_exact(separation) / 2.0
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
        let (theta, _err) = theta_sigma_average(&plates, center, 0.5, 500, 100, 42, 0).unwrap();
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
        // A two-body hit sample gives negative interaction and positive uncertainty.
        assert!(
            result.energy < 0.0 && result.error > 0.0,
            "interaction and uncertainty have invalid signs: {}",
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

    #[test]
    fn checked_interaction_rejects_single_body_and_invalid_sampling() {
        let sphere = crate::geometry::SolidSphere {
            center: [0.0; 3],
            radius: 1.0,
        };
        assert_eq!(
            try_casimir_energy_at_point(&sphere, [0.0; 3], &WorldlineCasimirConfig::default())
                .unwrap_err(),
            CasimirComputationError::Geometry(InteractionGeometryError::UnsupportedGeometry)
        );
        let config = WorldlineCasimirConfig {
            n_loops: 1,
            ..Default::default()
        };
        assert!(
            try_casimir_energy_at_point(&ParallelPlates { separation: 1.0 }, [0.0; 3], &config)
                .is_err()
        );
    }

    #[test]
    fn scalar_density_dilation_and_transverse_translation() {
        let config = WorldlineCasimirConfig {
            n_loop_points: 64,
            n_loops: 400,
            t_min: 0.02,
            t_max: 20.0,
            n_t_points: 16,
            seed: 981,
        };
        let original = casimir_energy_at_point(
            &ParallelPlates { separation: 1.0 },
            [0.0, 0.0, 0.5],
            &config,
        );
        let translated = casimir_energy_at_point(
            &ParallelPlates { separation: 1.0 },
            [7.0, -3.0, 0.5],
            &config,
        );
        assert_eq!(original.energy, translated.energy);
        let dilated_config = WorldlineCasimirConfig {
            t_min: 4.0 * config.t_min,
            t_max: 4.0 * config.t_max,
            ..config
        };
        let dilated = casimir_energy_at_point(
            &ParallelPlates { separation: 2.0 },
            [0.0, 0.0, 1.0],
            &dilated_config,
        );
        assert!((original.energy - 16.0 * dilated.energy).abs() < 1e-12);
        assert!((original.error - 16.0 * dilated.error).abs() < 1e-12);
    }

    #[test]
    fn scalar_plate_polygon_refinement_matches_known_energy() {
        let coarse = scalar_parallel_plates_worldline(1.0, 32, 5000, 8127).unwrap();
        let fine = scalar_parallel_plates_worldline(1.0, 512, 5000, 8127).unwrap();
        let exact = scalar_parallel_plates_exact(1.0);
        eprintln!("scalar_plate_refinement coarse={coarse:?} fine={fine:?} exact={exact:.16e}");
        assert!(
            fine.energy < coarse.energy,
            "coarse={coarse:?}, fine={fine:?}"
        );
        // Polygon extrema miss continuous-path excursions. Five MC errors plus
        // a 15% finite-polygon allowance bound the declared diagnostic surface.
        assert!(
            (fine.energy - exact).abs() < 5.0 * fine.error + 0.15 * exact.abs(),
            "fine={fine:?}, exact={exact}"
        );
        assert!(fine.error > 0.0);
        let dilated = scalar_parallel_plates_worldline(2.0, 512, 5000, 8127).unwrap();
        assert_eq!(fine.energy, 8.0 * dilated.energy);
        assert_eq!(fine.error, 8.0 * dilated.error);
        assert_eq!(casimir_parallel_plates_exact(1.0), 2.0 * exact);
    }

    #[test]
    fn integrated_plate_samples_are_worker_schedule_invariant() {
        let serial = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let parallel = rayon::ThreadPoolBuilder::new()
            .num_threads(2)
            .build()
            .unwrap();
        let evaluate = || scalar_parallel_plates_worldline(1.0, 64, 400, 1298).unwrap();
        let serial_result = serial.install(evaluate);
        let parallel_result = parallel.install(evaluate);
        assert_eq!(
            serial_result.energy.to_bits(),
            parallel_result.energy.to_bits()
        );
        assert_eq!(
            serial_result.error.to_bits(),
            parallel_result.error.to_bits()
        );
    }

    #[test]
    fn local_density_spatial_integral_and_quadrature_agree_with_plate_range_estimator() {
        let plates = ParallelPlates { separation: 1.0 };
        let n_spatial = 33;
        let spacing = 9.0 / (n_spatial - 1) as f64;
        let integrate = |n_t_points| {
            let config = WorldlineCasimirConfig {
                n_loop_points: 64,
                n_loops: 400,
                t_min: 0.01,
                t_max: 200.0,
                n_t_points,
                seed: 1297,
            };
            (0..n_spatial)
                .into_par_iter()
                .map(|index| {
                    let result = casimir_energy_at_point(
                        &plates,
                        [0.0, 0.0, -4.0 + index as f64 * spacing],
                        &config,
                    );
                    let weight = spacing
                        * if index == 0 || index == n_spatial - 1 {
                            0.5
                        } else {
                            1.0
                        };
                    (weight * result.energy, weight * result.error)
                })
                .collect::<Vec<_>>()
                .into_iter()
                .fold((0.0, 0.0), |accumulator, contribution| {
                    (
                        accumulator.0 + contribution.0,
                        accumulator.1 + contribution.1,
                    )
                })
        };
        let coarse = integrate(16);
        let fine = integrate(40);
        let reference = scalar_parallel_plates_worldline(1.0, 64, 10000, 1296).unwrap();
        eprintln!(
            "scalar_density_spatial_integral coarse={coarse:?} fine={fine:?} reference={reference:?}"
        );
        // Spatial points share random numbers, so summing weighted point errors
        // bounds the standard deviation without assuming independent points.
        // The 8% allowance covers the declared finite grid and T quadrature.
        assert!(
            (fine.0 - reference.energy).abs()
                < 5.0 * (fine.1 + reference.error) + 0.08 * reference.energy.abs(),
            "fine={fine:?}, reference={reference:?}"
        );
        assert!(
            (fine.0 - coarse.0).abs() < 5.0 * (fine.1 + coarse.1) + 0.08 * reference.energy.abs(),
            "fine={fine:?}, coarse={coarse:?}"
        );
    }

    #[test]
    fn translating_geometry_and_evaluation_point_preserves_interaction() {
        struct ShiftedPlates {
            displacement: [f64; 3],
        }
        impl CasimirGeometry for ShiftedPlates {
            fn intersects_count(&self, points: &[[f64; 3]], center: [f64; 3], scale: f64) -> usize {
                ParallelPlates { separation: 1.0 }.intersects_count(
                    points,
                    std::array::from_fn(|dimension| {
                        center[dimension] - self.displacement[dimension]
                    }),
                    scale,
                )
            }
            fn validate_interaction(&self) -> Result<(), InteractionGeometryError> {
                Ok(())
            }
            fn interaction_hit_mask(
                &self,
                points: &[[f64; 3]],
                center: [f64; 3],
            ) -> Result<[bool; 2], InteractionGeometryError> {
                ParallelPlates { separation: 1.0 }.interaction_hit_mask(
                    points,
                    std::array::from_fn(|dimension| {
                        center[dimension] - self.displacement[dimension]
                    }),
                )
            }
            fn name(&self) -> &str {
                "translated_plates"
            }
        }
        let config = WorldlineCasimirConfig {
            n_loop_points: 32,
            n_loops: 200,
            n_t_points: 12,
            ..Default::default()
        };
        let original = casimir_energy_at_point(
            &ParallelPlates { separation: 1.0 },
            [0.0, 0.0, 0.5],
            &config,
        );
        let translated = casimir_energy_at_point(
            &ShiftedPlates {
                displacement: [2.0, -4.0, 8.0],
            },
            [2.0, -4.0, 8.5],
            &config,
        );
        assert_eq!(original.energy.to_bits(), translated.energy.to_bits());
        assert_eq!(original.error.to_bits(), translated.error.to_bits());
    }

    #[test]
    fn plate_standard_error_preserves_large_length_dilation() {
        let ordinary = scalar_parallel_plates_worldline(1.0, 32, 64, 8127).unwrap();
        let dilated = scalar_parallel_plates_worldline(1e70, 32, 64, 8127).unwrap();
        assert!(dilated.error > 0.0);
        assert!((dilated.energy / 1e-210 / ordinary.energy - 1.0).abs() < 1e-12);
        assert!((dilated.error / 1e-210 / ordinary.error - 1.0).abs() < 1e-12);
    }

    #[test]
    fn density_standard_error_preserves_large_length_dilation() {
        let config = WorldlineCasimirConfig {
            n_loop_points: 32,
            n_loops: 200,
            n_t_points: 12,
            ..Default::default()
        };
        let ordinary = casimir_energy_at_point(
            &ParallelPlates { separation: 1.0 },
            [0.0, 0.0, 0.5],
            &config,
        );
        let dilated_config = WorldlineCasimirConfig {
            t_min: config.t_min * 1e140,
            t_max: config.t_max * 1e140,
            ..config
        };
        let dilated = casimir_energy_at_point(
            &ParallelPlates { separation: 1e70 },
            [0.0, 0.0, 0.5e70],
            &dilated_config,
        );
        assert!(dilated.error > 0.0);
        assert!((dilated.energy / 1e-280 / ordinary.energy - 1.0).abs() < 1e-10);
        assert!((dilated.error / 1e-280 / ordinary.error - 1.0).abs() < 1e-10);
        assert_eq!(
            scalar_parallel_plates_worldline(1e-200, 32, 64, 8127).unwrap_err(),
            CasimirComputationError::NumericalRange
        );
    }
}
