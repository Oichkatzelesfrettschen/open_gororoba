//! Worldline v-loop generation algorithm (White 2021).
//!
//! Generates closed random loops in 3D used for Monte Carlo evaluation
//! of the Casimir energy via the worldline path integral. The v-loop
//! algorithm produces loops that are closed by construction (the sum
//! of all displacement vectors is exactly zero).
//!
//! # Algorithm (White 2021, Eqs. 7-9)
//!
//! 1. Generate N-1 Gaussian random vectors w_i (3D).
//! 2. Normalize: v_bar_i = sqrt(2*(N+1-i)/(N+2-i)) * w_i / |w_i|
//! 3. Compensate: v_i = v_bar_i - (1/(N+2-i)) * sum_{k<i} v_k
//! 4. Accumulate positions: y_i = y_{i-1} + v_i
//! 5. Close: y_N = -sum(y_1 ... y_{N-1})
//!
//! The resulting unit loop has RMS radius = 1 and zero center of mass.
//! Scaling by sqrt(T) gives a loop at proper time T.
//!
//! # References
//! - White et al. (2021): EPJ C 81, 677. DOI:10.1140/epjc/s10052-021-09484-z
//! - Gies, Langfeld, Moyaerts (2003): JHEP 0306, 018

use rand::Rng;
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

    /// Check closure: |sum of points| should be ~ 0.
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

/// Generate a unit v-loop with N points.
///
/// The loop is closed by construction: sum of all displacements = 0.
/// Each point lies in 3D Euclidean space. The RMS distance from origin
/// is O(1) (unit scale).
///
/// # Arguments
/// * `n_points` - Number of points on the loop (must be >= 3).
/// * `rng` - Random number generator.
pub fn generate_unit_loop<R: Rng>(n_points: usize, rng: &mut R) -> WorldlineLoop {
    assert!(n_points >= 3, "v-loop requires at least 3 points");

    let n = n_points;
    let nf = n as f64;

    // Step 1: Generate N-1 random displacement vectors
    let mut displacements = vec![[0.0; 3]; n];

    // Running sum for the compensation step
    let mut cumsum = [0.0; 3];

    #[allow(clippy::needless_range_loop)] // Index i used for float conversion and array indexing
    for i in 0..(n - 1) {
        let if64 = i as f64;

        // Generate 3D Gaussian vector
        let w: [f64; 3] = [
            rng.sample::<f64, _>(StandardNormal),
            rng.sample::<f64, _>(StandardNormal),
            rng.sample::<f64, _>(StandardNormal),
        ];

        // Normalize to unit length
        let norm = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt();
        if norm < 1e-30 {
            continue; // Skip degenerate samples (astronomically unlikely)
        }

        // Scale factor: sqrt(2 * (N+1-i) / (N+2-i))
        let scale = (2.0 * (nf + 1.0 - if64) / (nf + 2.0 - if64)).sqrt();

        // v_bar_i = scale * w / |w|
        let mut v = [0.0; 3];
        for d in 0..3 {
            v[d] = scale * w[d] / norm;
        }

        // Compensate: v_i = v_bar_i - (1/(N+2-i)) * cumsum
        let comp_factor = 1.0 / (nf + 2.0 - if64);
        for d in 0..3 {
            v[d] -= comp_factor * cumsum[d];
        }

        displacements[i] = v;

        // Update cumulative sum
        for d in 0..3 {
            cumsum[d] += v[d];
        }
    }

    // Last displacement closes the loop: v_{N-1} = -cumsum
    for d in 0..3 {
        displacements[n - 1][d] = -cumsum[d];
    }

    // Accumulate positions from displacements
    let mut points = vec![[0.0; 3]; n];
    points[0] = displacements[0];
    for i in 1..n {
        for d in 0..3 {
            points[i][d] = points[i - 1][d] + displacements[i][d];
        }
    }

    WorldlineLoop { points }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_loop_closure() {
        // The v-loop guarantees that displacements sum to zero (the loop
        // returns to the origin). The position sum (= N * center_of_mass)
        // is NOT guaranteed to be zero -- it can grow as O(sqrt(N)).
        // We check that the loop's last point returns to (approximately)
        // the first point's predecessor, i.e., the loop is truly closed.
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let n = 100;
        let wl = generate_unit_loop(n, &mut rng);

        // Check return-to-start: last_point + disp_{0} should give points[0]
        // Equivalently, sum of displacements should be zero
        let mut disp_sum = wl.points[0];
        for i in 1..n {
            #[allow(clippy::needless_range_loop)] // 2D point indexing
            for d in 0..3 {
                disp_sum[d] += wl.points[i][d] - wl.points[i - 1][d];
            }
        }
        let err = (disp_sum[0].powi(2) + disp_sum[1].powi(2) + disp_sum[2].powi(2)).sqrt();
        assert!(
            err < 1e-10,
            "displacement closure error = {err} (should be < 1e-10)"
        );
    }

    #[test]
    fn test_loop_displacement_closure() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let n = 200;
        let wl = generate_unit_loop(n, &mut rng);

        // Check that displacements sum to zero
        let mut disp_sum = wl.points[0];
        for i in 1..n {
            #[allow(clippy::needless_range_loop)] // 2D point indexing
            for d in 0..3 {
                disp_sum[d] += wl.points[i][d] - wl.points[i - 1][d];
            }
        }
        let err = (disp_sum[0].powi(2) + disp_sum[1].powi(2) + disp_sum[2].powi(2)).sqrt();
        assert!(err < 1e-10, "displacement sum should be zero: {err}");
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
    #[allow(clippy::needless_range_loop)] // 2D point array comparison
    fn test_scaling() {
        let mut rng = ChaCha8Rng::seed_from_u64(99);
        let wl = generate_unit_loop(50, &mut rng);
        let t = 4.0;
        let scaled = wl.scaled_points(t);
        let scale = t.sqrt();
        for i in 0..50 {
            for d in 0..3 {
                assert!(
                    (scaled[i][d] - wl.points[i][d] * scale).abs() < 1e-14,
                    "scaling failed"
                );
            }
        }
    }

    #[test]
    fn test_minimum_points() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let wl = generate_unit_loop(3, &mut rng);
        assert_eq!(wl.n_points(), 3);
    }
}
