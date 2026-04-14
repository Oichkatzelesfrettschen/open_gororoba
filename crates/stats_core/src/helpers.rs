//! Canonical statistical helper functions.
//!
//! These are simple, commonly needed functions that were duplicated across
//! multiple binaries. They live in stats_core as the Tier 1 home for
//! statistical operations.
//!
//! ## Hypothesis-test p-value surface
//!
//! Three functions cover the p-values needed across the workspace.  All wrap
//! statrs distributions so no CLI crate needs a direct statrs dependency:
//!
//! - [`chi_squared_p_value`] -- upper-tail p-value (Fisher combined test, etc.)
//! - [`normal_two_tailed_p_value`] -- symmetric two-tailed p-value from z-score
//! - [`students_t_two_tailed_p_value`] -- symmetric two-tailed p-value from t-score

/// Arithmetic mean of a slice. Returns 0.0 for empty input.
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Sample standard deviation given pre-computed mean.
///
/// Uses Bessel correction (N-1 denominator). Returns 0.0 for fewer than
/// 2 elements.
pub fn std_dev(values: &[f64], mu: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let var = values
        .iter()
        .map(|v| {
            let d = v - mu;
            d * d
        })
        .sum::<f64>()
        / (values.len() - 1) as f64;
    var.sqrt()
}

/// Median of a mutable slice. Sorts in place. Returns 0.0 for empty input.
///
/// For even-length slices, returns the average of the two middle elements.
pub fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}

/// Chi-squared upper-tail p-value.
///
/// WHY: Chi-squared test statistics are non-negative and large values indicate
/// departure from the null, so the p-value is always the upper tail P(X > x).
/// Fisher's combined test, goodness-of-fit tests, and independence tests all
/// use this convention. Wrapping statrs here removes the direct dep from every
/// CLI crate that runs such a test.
///
/// Panics if `df <= 0`. Returns 1.0 for `statistic <= 0` (entire mass is above
/// a non-positive value for a non-negative random variable).
pub fn chi_squared_p_value(df: f64, statistic: f64) -> f64 {
    use statrs::distribution::{ChiSquared, ContinuousCDF};
    let chi = ChiSquared::new(df).expect("chi_squared_p_value: df must be positive");
    1.0 - chi.cdf(statistic)
}

/// Standard-normal two-tailed p-value from a z-score.
///
/// WHY: Mann-Whitney U and other rank-based tests produce a z-score under the
/// normal approximation and report `2 * P(Z > |z|)`. Callers that directly
/// imported `statrs::Normal` for this single operation no longer need the dep.
///
/// Returns 1.0 for z = 0, approaches 0.0 for large |z|. Always in [0, 1].
pub fn normal_two_tailed_p_value(z: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, Normal};
    let n = Normal::new(0.0, 1.0).expect("normal_two_tailed_p_value: constant params are valid");
    2.0 * (1.0 - n.cdf(z.abs()))
}

/// Student's t two-tailed p-value from a t-score and degrees of freedom.
///
/// WHY: Pearson correlation significance and Welch/Student t-tests report
/// `2 * P(T > |t|)` under the t distribution. Centralising this in stats_core
/// removes the direct statrs dep from every CLI bin that performs such a test.
///
/// Panics if `dof <= 0`. Returns 1.0 for t = 0, approaches 0.0 for large |t|.
pub fn students_t_two_tailed_p_value(t: f64, dof: f64) -> f64 {
    use statrs::distribution::{ContinuousCDF, StudentsT};
    let dist = StudentsT::new(0.0, 1.0, dof)
        .expect("students_t_two_tailed_p_value: dof must be positive");
    2.0 * (1.0 - dist.cdf(t.abs()))
}

/// Compute a histogram of values.
///
/// Returns (bin_centers, counts).
pub fn histogram(values: &[f64], n_bins: usize, min: f64, max: f64) -> (Vec<f64>, Vec<usize>) {
    if values.is_empty() || n_bins == 0 || max <= min {
        return (vec![], vec![]);
    }

    let mut counts = vec![0; n_bins];
    let bin_width = (max - min) / n_bins as f64;
    let bin_centers: Vec<f64> = (0..n_bins)
        .map(|i| min + (i as f64 + 0.5) * bin_width)
        .collect();

    for &v in values {
        if v >= min && v <= max {
            let mut bin = ((v - min) / bin_width).floor() as usize;
            if bin >= n_bins {
                bin = n_bins - 1;
            }
            counts[bin] += 1;
        }
    }

    (bin_centers, counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_basic() {
        assert!((mean(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_mean_empty() {
        assert!((mean(&[]) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_mean_single() {
        assert!((mean(&[42.0]) - 42.0).abs() < 1e-12);
    }

    #[test]
    fn test_std_dev_basic() {
        let vals = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let mu = mean(&vals);
        let sd = std_dev(&vals, mu);
        // Known population std dev is 2.0, sample std dev ~2.138
        assert!((sd - 2.138).abs() < 0.01, "std_dev={}", sd);
    }

    #[test]
    fn test_std_dev_empty() {
        assert!((std_dev(&[], 0.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_std_dev_single() {
        assert!((std_dev(&[5.0], 5.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_median_odd() {
        let mut vals = vec![3.0, 1.0, 2.0];
        assert!((median(&mut vals) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_median_even() {
        let mut vals = vec![4.0, 1.0, 3.0, 2.0];
        assert!((median(&mut vals) - 2.5).abs() < 1e-12);
    }

    #[test]
    fn test_median_empty() {
        assert!((median(&mut []) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_median_single() {
        let mut vals = vec![42.0];
        assert!((median(&mut vals) - 42.0).abs() < 1e-12);
    }

    // --- chi_squared_p_value ---

    #[test]
    fn test_chi_squared_p_value_at_zero() {
        // Entire probability mass lies above 0 for a non-negative r.v.
        let p = chi_squared_p_value(2.0, 0.0);
        assert!((p - 1.0).abs() < 1e-12, "p={}", p);
    }

    #[test]
    fn test_chi_squared_p_value_large_statistic() {
        // Very large statistic -> negligible tail probability.
        let p = chi_squared_p_value(2.0, 100.0);
        assert!(p < 1e-20, "p={}", p);
    }

    #[test]
    fn test_chi_squared_p_value_known_analytic() {
        // chi2(df=2) is Exponential(1/2): CDF = 1 - exp(-x/2).
        // At x=2.0: p-value = exp(-1.0) ~= 0.36787944117.
        let p = chi_squared_p_value(2.0, 2.0);
        assert!((p - (-1.0_f64).exp()).abs() < 1e-9, "p={}", p);
    }

    // --- normal_two_tailed_p_value ---

    #[test]
    fn test_normal_two_tailed_at_zero() {
        // z=0: 2*(1 - Phi(0)) = 2*0.5 = 1.0 exactly.
        let p = normal_two_tailed_p_value(0.0);
        assert!((p - 1.0).abs() < 1e-12, "p={}", p);
    }

    #[test]
    fn test_normal_two_tailed_large_z() {
        // Large |z| -> nearly zero p-value.
        let p = normal_two_tailed_p_value(10.0);
        assert!(p < 1e-20, "p={}", p);
    }

    #[test]
    fn test_normal_two_tailed_symmetric() {
        // Two-tailed p-value is symmetric: p(z) == p(-z).
        let p_pos = normal_two_tailed_p_value(1.5);
        let p_neg = normal_two_tailed_p_value(-1.5);
        assert!((p_pos - p_neg).abs() < 1e-15, "asymmetry: {} vs {}", p_pos, p_neg);
    }

    #[test]
    fn test_normal_two_tailed_known_z196() {
        // z ~= 1.96 gives the canonical alpha=0.05 threshold.
        let p = normal_two_tailed_p_value(1.959_963_985);
        assert!((p - 0.05).abs() < 1e-6, "p={}", p);
    }

    // --- students_t_two_tailed_p_value ---

    #[test]
    fn test_students_t_at_zero() {
        // t=0: 2*(1 - T_cdf(0)) = 1.0 for any dof.
        let p = students_t_two_tailed_p_value(0.0, 10.0);
        assert!((p - 1.0).abs() < 1e-12, "p={}", p);
    }

    #[test]
    fn test_students_t_large_t() {
        // Large |t| -> negligible p-value.
        // NOTE: dof=5 has heavy tails (Cauchy-like), so the tail at t=100 is ~2.45e-08,
        // not as extreme as a normal tail. Threshold reflects actual t-distribution behavior.
        let p = students_t_two_tailed_p_value(100.0, 5.0);
        assert!(p < 1e-4, "p={}", p);
    }

    #[test]
    fn test_students_t_symmetric() {
        // Two-tailed p-value is symmetric.
        let p_pos = students_t_two_tailed_p_value(2.0, 10.0);
        let p_neg = students_t_two_tailed_p_value(-2.0, 10.0);
        assert!((p_pos - p_neg).abs() < 1e-15, "asymmetry: {} vs {}", p_pos, p_neg);
    }

    #[test]
    fn test_students_t_approaches_normal_at_large_dof() {
        // t(dof=1000) at z=1.96 should be very close to normal_two_tailed_p_value(1.96).
        // Tolerance 1e-3: statrs t vs normal CDF implementations diverge slightly at dof=1000.
        let p_t = students_t_two_tailed_p_value(1.959_963_985, 1000.0);
        let p_n = normal_two_tailed_p_value(1.959_963_985);
        assert!((p_t - p_n).abs() < 1e-3, "t={} normal={}", p_t, p_n);
    }
}
