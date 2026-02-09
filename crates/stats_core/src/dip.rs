//! Hartigan dip test for unimodality.
//!
//! Implements the dip statistic (Hartigan & Hartigan, 1985) which measures
//! departure from unimodality in a sample.  The dip is the maximum difference
//! between the empirical CDF and the unimodal distribution function that
//! minimizes that maximum difference.
//!
//! # Algorithm
//!
//! 1. Sort the data.
//! 2. Compute the empirical CDF F_n.
//! 3. Find the Greatest Convex Minorant (GCM) from below and the Least
//!    Concave Majorant (LCM) from above.
//! 4. The dip = max(GCM deviation, LCM deviation) / 2.
//!
//! The p-value is estimated via permutation: repeatedly draw from U(0,1),
//! compute the dip, and count how often the uniform dip exceeds the observed.
//!
//! # Literature
//!
//! - Hartigan, J. A. & Hartigan, P. M. (1985). The dip test of unimodality.
//!   Annals of Statistics, 13(1), 70-84.
//! - Hartigan, P. M. (1985). Algorithm AS 217: Computation of the dip statistic.
//!   Journal of the Royal Statistical Society C, 34(3), 320-325.

use rand::Rng;
use rand_chacha::ChaCha8Rng;

/// Result of the Hartigan dip test.
#[derive(Debug, Clone)]
pub struct DipTestResult {
    /// The dip statistic D_n.  Larger values indicate stronger evidence
    /// against unimodality.
    pub dip_statistic: f64,
    /// Permutation-based p-value: fraction of uniform samples with dip >= D_n.
    pub p_value: f64,
    /// Number of permutations used for p-value estimation.
    pub n_permutations: usize,
    /// Sample size.
    pub n_samples: usize,
}

/// Compute the dip statistic for sorted data.
///
/// Implements Algorithm AS 217 (Hartigan 1985): iteratively narrows a
/// "modal interval" [low, high] from which a unimodal distribution rises
/// to the left and falls to the right, computing the GCM on the left and
/// LCM on the right to find the maximum deviation.
///
/// # Arguments
/// * `sorted` - Data sorted in non-decreasing order.  Must have length >= 2.
///
/// # Returns
/// The dip statistic D_n (non-negative).
pub fn dip_statistic(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n < 2 {
        return 0.0;
    }

    // Normalized uniform CDF: F_n(x_i) = (i+1)/(n+1) for i in 0..n
    // This is the Hazen plotting position, smoothed to avoid 0 and 1.
    let nf = n as f64;

    // We iterate to find the modal interval.
    // Start with the full range.
    let mut low = 0_usize;
    let mut high = n - 1;

    let max_iter = n; // Convergence guaranteed in at most n iterations

    let mut dip = 0.0_f64;

    for _ in 0..max_iter {
        if high <= low + 1 {
            break;
        }

        // On [low..=high], compute GCM from the left and LCM from the right.
        let _len = high - low + 1;

        // GCM: greatest convex minorant of the empirical CDF on [low..high]
        // Using (x, F) pairs where x = sorted[i], F = (i - low) / (high - low)
        // Normalized so F(low) = 0, F(high) = 1.

        // GCM from left
        let mut gcm_indices = vec![low];
        for i in (low + 1)..=high {
            while gcm_indices.len() >= 2 {
                let j = gcm_indices[gcm_indices.len() - 1];
                let k = gcm_indices[gcm_indices.len() - 2];
                // Slopes in data space: (F(i) - F(k)) / (x(i) - x(k))
                // F(i) = (i - low) / (high - low), but for comparison
                // we can just use index differences (monotone transform)
                let dx_ki = sorted[i] - sorted[k];
                let dx_kj = sorted[j] - sorted[k];
                let di_ki = (i - k) as f64;
                let di_kj = (j - k) as f64;
                // slope = dF/dx = (delta_index / delta_x) * (1/len)
                // Compare slopes: (di_ki / dx_ki) vs (di_kj / dx_kj)
                // Cross multiply to avoid division by zero:
                // di_ki * dx_kj vs di_kj * dx_ki
                if dx_ki.abs() < 1e-30 || dx_kj.abs() < 1e-30 {
                    // Coincident data points
                    hull_pop_tie(&mut gcm_indices, i, j, true);
                } else if di_kj * dx_ki >= di_ki * dx_kj {
                    // slope(k->j) >= slope(k->i): j is above the line k->i
                    gcm_indices.pop();
                } else {
                    break;
                }
            }
            gcm_indices.push(i);
        }

        // LCM from left (concave majorant)
        let mut lcm_indices = vec![low];
        for i in (low + 1)..=high {
            while lcm_indices.len() >= 2 {
                let j = lcm_indices[lcm_indices.len() - 1];
                let k = lcm_indices[lcm_indices.len() - 2];
                let dx_ki = sorted[i] - sorted[k];
                let dx_kj = sorted[j] - sorted[k];
                let di_ki = (i - k) as f64;
                let di_kj = (j - k) as f64;
                if dx_ki.abs() < 1e-30 || dx_kj.abs() < 1e-30 {
                    hull_pop_tie(&mut lcm_indices, i, j, false);
                } else if di_kj * dx_ki <= di_ki * dx_kj {
                    lcm_indices.pop();
                } else {
                    break;
                }
            }
            lcm_indices.push(i);
        }

        // Compute maximum deviation between empirical CDF and GCM/LCM
        let range = (high - low) as f64;
        let mut max_dev = 0.0_f64;
        let mut mn = low;
        let mut mx = high;

        // GCM deviations
        for w in 0..(gcm_indices.len() - 1) {
            let il = gcm_indices[w];
            let ir = gcm_indices[w + 1];
            let dx = sorted[ir] - sorted[il];
            for i in il..=ir {
                let f_empirical = (i - low) as f64 / range;
                let f_gcm = if dx.abs() < 1e-30 {
                    (il - low) as f64 / range
                } else {
                    let t = (sorted[i] - sorted[il]) / dx;
                    ((il - low) as f64 + t * (ir - il) as f64) / range
                };
                let dev = (f_empirical - f_gcm).abs();
                if dev > max_dev {
                    max_dev = dev;
                    mn = i;
                }
            }
        }

        // LCM deviations
        for w in 0..(lcm_indices.len() - 1) {
            let il = lcm_indices[w];
            let ir = lcm_indices[w + 1];
            let dx = sorted[ir] - sorted[il];
            for i in il..=ir {
                let f_empirical = (i - low) as f64 / range;
                let f_lcm = if dx.abs() < 1e-30 {
                    (ir - low) as f64 / range
                } else {
                    let t = (sorted[i] - sorted[il]) / dx;
                    ((il - low) as f64 + t * (ir - il) as f64) / range
                };
                let dev = (f_empirical - f_lcm).abs();
                if dev > max_dev {
                    max_dev = dev;
                    mx = i;
                }
            }
        }

        dip = dip.max(max_dev / (2.0 * nf / range));

        // Narrow the modal interval
        let new_low = mn;
        let new_high = mx;
        if new_low >= new_high {
            break;
        }
        if new_low == low && new_high == high {
            break; // Converged
        }
        low = new_low;
        high = new_high;
    }

    dip
}

/// Helper for hull construction with tied x-values.
fn hull_pop_tie(hull: &mut Vec<usize>, _new: usize, _existing: usize, is_gcm: bool) {
    if is_gcm {
        hull.pop(); // For GCM, keep lower y (the new point)
    } else {
        hull.pop(); // For LCM, keep higher y (the new point)
    }
}

/// Run the Hartigan dip test with permutation-based p-value.
///
/// # Arguments
/// * `data` - Sample data (any order; will be sorted internally).
/// * `n_perm` - Number of permutations for p-value estimation.
/// * `rng` - Deterministic RNG for reproducibility.
///
/// # Returns
/// `DipTestResult` with the dip statistic, p-value, and metadata.
///
/// # Panics
/// Panics if `data` is empty.
pub fn hartigan_dip_test(data: &[f64], n_perm: usize, rng: &mut ChaCha8Rng) -> DipTestResult {
    assert!(!data.is_empty(), "data must not be empty");

    let n = data.len();

    // Sort the data
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let observed_dip = dip_statistic(&sorted);

    if n < 2 || n_perm == 0 {
        return DipTestResult {
            dip_statistic: observed_dip,
            p_value: 1.0,
            n_permutations: 0,
            n_samples: n,
        };
    }

    // Permutation test: draw uniform samples and compute dip
    let mut count_ge = 0_usize;

    for _ in 0..n_perm {
        let mut uniform_sample: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
        uniform_sample.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let perm_dip = dip_statistic(&uniform_sample);
        if perm_dip >= observed_dip {
            count_ge += 1;
        }
    }

    let p_value = (count_ge as f64 + 1.0) / (n_perm as f64 + 1.0);

    DipTestResult {
        dip_statistic: observed_dip,
        p_value,
        n_permutations: n_perm,
        n_samples: n,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_uniform_small_dip() {
        // Uniform data is unimodal -> small dip statistic
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let n = 200;
        let mut data: Vec<f64> = (0..n).map(|_| rng.gen::<f64>()).collect();
        data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let dip = dip_statistic(&data);
        // For uniform(0,1) with n=200, expected dip is ~0.01-0.03
        assert!(
            dip < 0.10,
            "Uniform data should have small dip, got {}",
            dip
        );
    }

    #[test]
    fn test_bimodal_large_dip() {
        // Bimodal: N(0,1) + N(5,1) mixture -> large dip, p < 0.01
        let mut rng = ChaCha8Rng::seed_from_u64(123);
        let n = 200;
        let mut data = Vec::with_capacity(n);

        for _ in 0..(n / 2) {
            // N(0, 0.5)
            let u1: f64 = rng.gen::<f64>();
            let u2: f64 = rng.gen::<f64>();
            let z = (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();
            data.push(z * 0.5);
        }
        for _ in 0..(n / 2) {
            // N(5, 0.5)
            let u1: f64 = rng.gen::<f64>();
            let u2: f64 = rng.gen::<f64>();
            let z = (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();
            data.push(5.0 + z * 0.5);
        }

        let mut test_rng = ChaCha8Rng::seed_from_u64(456);
        let result = hartigan_dip_test(&data, 1000, &mut test_rng);

        assert!(
            result.dip_statistic > 0.05,
            "Bimodal data should have large dip, got {}",
            result.dip_statistic
        );
        assert!(
            result.p_value < 0.01,
            "Bimodal data should have p < 0.01, got {}",
            result.p_value
        );
    }

    #[test]
    fn test_single_sample() {
        // N=1 edge case
        let data = vec![42.0];
        let mut rng = ChaCha8Rng::seed_from_u64(0);
        let result = hartigan_dip_test(&data, 100, &mut rng);

        assert_eq!(result.dip_statistic, 0.0);
        assert_eq!(result.n_samples, 1);
    }

    #[test]
    fn test_deterministic_seeding() {
        // Same seed -> same result
        let data: Vec<f64> = (0..50).map(|i| i as f64 / 50.0).collect();

        let mut rng1 = ChaCha8Rng::seed_from_u64(999);
        let result1 = hartigan_dip_test(&data, 500, &mut rng1);

        let mut rng2 = ChaCha8Rng::seed_from_u64(999);
        let result2 = hartigan_dip_test(&data, 500, &mut rng2);

        assert_eq!(result1.dip_statistic, result2.dip_statistic);
        assert_eq!(result1.p_value, result2.p_value);
    }

    #[test]
    fn test_two_point_sample() {
        // N=2: two distinct points form a unimodal distribution
        // The dip should be 0 (any two-point sample is trivially unimodal)
        let data = vec![0.0, 1.0];
        let dip = dip_statistic(&data);
        assert!(
            dip < 0.01,
            "N=2 dip should be near zero (trivially unimodal), got {}",
            dip
        );
    }
}
