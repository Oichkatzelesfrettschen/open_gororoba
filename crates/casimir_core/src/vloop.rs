//! Gaussian Brownian bridges for a real-scalar worldline path integral.
//!
//! The polygon measure is exp(-N * sum |delta y|^2 / 4). Independent
//! Gaussian increments of variance 2/N are conditioned to sum to zero,
//! integrated, and translated to zero center of mass. The closing edge joins
//! the last stored vertex to the first; storing a duplicate endpoint would
//! bias the center-of-mass measure. Finite polygons approach continuous loops
//! as their vertex count grows (Gies, Langfeld, Moyaerts, Sec. 3).

use rand::RngExt;
use rand_distr::StandardNormal;

/// A closed worldline loop in 3D.
#[derive(Debug, Clone)]
pub struct WorldlineLoop {
    /// Points on the loop (each is [x, y, z]).
    pub points: Vec<[f64; 3]>,
}

impl WorldlineLoop {
    /// Number of points on this loop.
    pub fn n_points(&self) -> usize {
        self.points.len()
    }

    /// Centering residual: norm of the sum of vertices.
    ///
    /// The historical method name refers to centering, not endpoint closure.
    /// Polygon closure uses the implicit last-to-first edge.
    pub fn closure_error(&self) -> f64 {
        let mut sum = [0.0; 3];
        for p in &self.points {
            for d in 0..3 {
                sum[d] += p[d];
            }
        }
        (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt()
    }

    /// Center of mass (should be ~ 0 for v-loops).
    pub fn center_of_mass(&self) -> [f64; 3] {
        let n = self.points.len() as f64;
        let mut com = [0.0; 3];
        for p in &self.points {
            for d in 0..3 {
                com[d] += p[d];
            }
        }
        for c in &mut com {
            *c /= n;
        }
        com
    }

    /// Scale the loop by sqrt(T) for proper-time evaluation.
    pub fn scaled_points(&self, t: f64) -> Vec<[f64; 3]> {
        let scale = t.sqrt();
        self.points
            .iter()
            .map(|p| [p[0] * scale, p[1] * scale, p[2] * scale])
            .collect()
    }
}

/// Sample the centered N-vertex Brownian bridge with variance 2/N increments.
///
/// Conditioning Gaussian increments by subtracting their sample mean gives
/// the exact discrete bridge measure. Recentring changes neither increments
/// nor their covariance. The polygon includes its closing last-to-first edge.
///
/// # Panics
/// Panics when fewer than three vertices are requested.
pub fn generate_unit_loop<R: RngExt>(n_points: usize, rng: &mut R) -> WorldlineLoop {
    assert!(
        n_points >= 3,
        "a Brownian polygon requires at least three vertices"
    );
    let increment_scale = (2.0 / n_points as f64).sqrt();
    let mut increments = Vec::with_capacity(n_points);
    let mut increment_mean = [0.0; 3];
    for _ in 0..n_points {
        let increment: [f64; 3] =
            std::array::from_fn(|_| rng.sample::<f64, _>(StandardNormal) * increment_scale);
        for (mean, value) in increment_mean.iter_mut().zip(increment) {
            *mean += value / n_points as f64;
        }
        increments.push(increment);
    }
    let mut position = [0.0; 3];
    let mut points = Vec::with_capacity(n_points);
    for increment in increments {
        points.push(position);
        for ((coordinate, displacement), mean) in
            position.iter_mut().zip(increment).zip(increment_mean)
        {
            *coordinate += displacement - mean;
        }
    }
    let mut worldline = WorldlineLoop { points };
    let center = worldline.center_of_mass();
    for point in &mut worldline.points {
        for (coordinate, mean) in point.iter_mut().zip(center) {
            *coordinate -= mean;
        }
    }
    worldline
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn centered_polygon_closes_through_implicit_edge() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let worldline = generate_unit_loop(200, &mut rng);
        assert!(worldline.closure_error() < 1e-10);
        let mut displacement_sum = [0.0; 3];
        for vertex in 0..worldline.n_points() {
            let next = (vertex + 1) % worldline.n_points();
            for (dimension, sum) in displacement_sum.iter_mut().enumerate() {
                *sum += worldline.points[next][dimension] - worldline.points[vertex][dimension];
            }
        }
        assert!(displacement_sum.iter().all(|value| value.abs() < 1e-12));
        assert_ne!(worldline.points.first(), worldline.points.last());
    }

    #[test]
    fn test_seed_reproducibility() {
        let wl1 = {
            let mut rng = ChaCha8Rng::seed_from_u64(123);
            generate_unit_loop(50, &mut rng)
        };
        let wl2 = {
            let mut rng = ChaCha8Rng::seed_from_u64(123);
            generate_unit_loop(50, &mut rng)
        };
        for i in 0..50 {
            for d in 0..3 {
                assert!(
                    (wl1.points[i][d] - wl2.points[i][d]).abs() < 1e-14,
                    "seed reproducibility failed at point {i} dim {d}"
                );
            }
        }
    }

    #[test]
    fn test_scaling() {
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let wl = generate_unit_loop(50, &mut rng);
        let t = 4.0;
        let scaled = wl.scaled_points(t);
        let scale = t.sqrt();
        for (sp, op) in scaled.iter().zip(wl.points.iter()) {
            for (&s, &o) in sp.iter().zip(op.iter()) {
                assert!((s - o * scale).abs() < 1e-14, "scaling failed");
            }
        }
    }

    #[test]
    fn test_minimum_points() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let wl = generate_unit_loop(3, &mut rng);
        assert_eq!(wl.n_points(), 3);
    }

    #[test]
    fn centered_bridge_has_correct_covariance_and_scale() {
        let n_points = 64;
        let n_samples = 6000;
        let mut rng = ChaCha8Rng::seed_from_u64(9237);
        let mut radius_squared = 0.0;
        let mut half_increment_squared = 0.0;
        for _ in 0..n_samples {
            let worldline = generate_unit_loop(n_points, &mut rng);
            assert!(
                worldline
                    .center_of_mass()
                    .iter()
                    .all(|coordinate| coordinate.abs() < 1e-12)
            );
            for point in &worldline.points {
                radius_squared += point[0] * point[0] / n_points as f64;
            }
            half_increment_squared +=
                (worldline.points[n_points / 2][0] - worldline.points[0][0]).powi(2);
        }
        let expected_variance = (1.0 - 1.0 / (n_points * n_points) as f64) / 6.0;
        assert!((radius_squared / n_samples as f64 - expected_variance).abs() < 0.009);
        assert!((half_increment_squared / n_samples as f64 - 0.5).abs() < 0.035);
    }
}
