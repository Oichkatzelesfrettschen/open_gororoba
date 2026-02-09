//! Adaptive/sequential permutation testing (Besag & Clifford 1991).
//!
//! Instead of always running a fixed number of permutations, adaptively
//! decides when enough evidence has been gathered to either reject or
//! accept the null hypothesis. This can save orders of magnitude of
//! computation for clearly significant or clearly non-significant tests,
//! while still running the full budget for borderline cases.
//!
//! # Algorithm
//!
//! The test processes permutations in batches. After each batch, it
//! computes a binomial confidence interval for the p-value. If the
//! interval lies entirely above or below alpha, the test stops early.
//!
//! The Phipson-Smyth (2010) correction `(r+1)/(k+1)` is always used
//! for the final p-value, ensuring it is never exactly zero.
//!
//! # References
//!
//! - Besag & Clifford (1991): Sequential Monte Carlo p-values
//! - Phipson & Smyth (2010): Permutation p-values should never be zero

use statrs::distribution::{ContinuousCDF, Normal};

/// Configuration for adaptive permutation testing.
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Number of permutations per batch before checking stopping rule.
    pub batch_size: usize,
    /// Maximum total permutations (hard cap).
    pub max_permutations: usize,
    /// Significance threshold.
    pub alpha: f64,
    /// Confidence level for the binomial CI stopping rule (e.g. 0.99).
    pub confidence: f64,
    /// Minimum permutations before early stopping is allowed.
    pub min_permutations: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            batch_size: 20,
            max_permutations: 10_000,
            alpha: 0.05,
            confidence: 0.99,
            min_permutations: 100,
        }
    }
}

/// Reason the adaptive test stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// P-value CI is entirely below alpha (significant).
    SignificantEarly,
    /// P-value CI is entirely above alpha (non-significant).
    NonSignificantEarly,
    /// Reached the maximum number of permutations.
    MaxReached,
}

/// Result of an adaptive permutation test.
#[derive(Debug, Clone)]
pub struct AdaptiveResult {
    /// Estimated p-value with Phipson-Smyth correction.
    pub p_value: f64,
    /// Total permutations actually used.
    pub n_permutations_used: usize,
    /// Whether the test stopped before max_permutations.
    pub stopped_early: bool,
    /// Why the test stopped.
    pub stop_reason: StopReason,
    /// P-value trajectory: (n_perms, p_value) after each batch.
    pub p_trajectory: Vec<(usize, f64)>,
}

/// Check whether to stop early based on binomial CI for the p-value.
///
/// Given `r` extreme values out of `k` permutations, the point estimate
/// of the p-value is `(r+1)/(k+1)`. The binomial proportion CI using
/// the normal approximation determines if we can confidently declare
/// the result significant or non-significant.
///
/// Returns `Some(StopReason)` if stopping is warranted, `None` to continue.
fn should_stop(r: usize, k: usize, alpha: f64, confidence: f64) -> Option<StopReason> {
    if k == 0 {
        return None;
    }
    let p_hat = (r as f64 + 1.0) / (k as f64 + 1.0);
    let se = (p_hat * (1.0 - p_hat) / k as f64).sqrt();

    let normal = Normal::new(0.0, 1.0).unwrap();
    let z = normal.inverse_cdf(0.5 + confidence / 2.0);

    let lower = p_hat - z * se;
    let upper = p_hat + z * se;

    if upper < alpha {
        Some(StopReason::SignificantEarly)
    } else if lower > alpha {
        Some(StopReason::NonSignificantEarly)
    } else {
        None
    }
}

/// Run an adaptive permutation test.
///
/// The caller provides a closure `run_batch(batch_size) -> n_extreme`
/// that runs `batch_size` permutations and returns how many had a test
/// statistic at least as extreme as the observed value. This decouples
/// the stopping logic from the specific test being run.
///
/// # Sidedness
///
/// The stopping rule assumes a one-sided or two-sided interpretation
/// depending on how the caller defines "extreme" in `run_batch`. The
/// caller must ensure consistency: if the underlying test is one-sided
/// (e.g., `null_frac >= obs_frac`), `run_batch` should count one-sided
/// extremes. If two-sided, count two-sided extremes. The adaptive
/// engine is agnostic -- it only sees the count of extreme permutations.
///
/// # Arguments
///
/// * `config` - Adaptive testing configuration
/// * `run_batch` - Closure that takes batch_size, returns count of extreme permutations
pub fn adaptive_permutation_test<F>(config: &AdaptiveConfig, mut run_batch: F) -> AdaptiveResult
where
    F: FnMut(usize) -> usize,
{
    let mut total_k = 0usize; // total permutations
    let mut total_r = 0usize; // total extreme
    let mut trajectory = Vec::new();

    while total_k < config.max_permutations {
        let remaining = config.max_permutations - total_k;
        let batch = config.batch_size.min(remaining);
        if batch == 0 {
            break;
        }

        let r_batch = run_batch(batch);
        total_k += batch;
        total_r += r_batch;

        let p_val = (total_r as f64 + 1.0) / (total_k as f64 + 1.0);
        trajectory.push((total_k, p_val));

        // Only consider early stopping after minimum permutations
        if total_k >= config.min_permutations {
            if let Some(reason) = should_stop(total_r, total_k, config.alpha, config.confidence) {
                return AdaptiveResult {
                    p_value: p_val,
                    n_permutations_used: total_k,
                    stopped_early: true,
                    stop_reason: reason,
                    p_trajectory: trajectory,
                };
            }
        }
    }

    let p_val = (total_r as f64 + 1.0) / (total_k as f64 + 1.0);
    AdaptiveResult {
        p_value: p_val,
        n_permutations_used: total_k,
        stopped_early: false,
        stop_reason: StopReason::MaxReached,
        p_trajectory: trajectory,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_early_stop_clearly_nonsignificant() {
        // Simulate a test where observed is never extreme (p ~ 1.0).
        // Should stop early as non-significant.
        let config = AdaptiveConfig {
            batch_size: 20,
            max_permutations: 10_000,
            alpha: 0.05,
            confidence: 0.99,
            min_permutations: 100,
        };

        let result = adaptive_permutation_test(&config, |batch_size| {
            // Every permutation is "extreme" -> p ~ 1.0
            batch_size
        });

        assert!(result.stopped_early, "Should stop early for p ~ 1.0");
        assert_eq!(result.stop_reason, StopReason::NonSignificantEarly);
        assert!(result.p_value > 0.5);
        assert!(result.n_permutations_used <= 200);
    }

    #[test]
    fn test_early_stop_clearly_significant() {
        // Simulate a test where observed is never exceeded (p ~ 0).
        let config = AdaptiveConfig {
            batch_size: 20,
            max_permutations: 10_000,
            alpha: 0.05,
            confidence: 0.99,
            min_permutations: 100,
        };

        let result = adaptive_permutation_test(&config, |_batch_size| {
            // No permutation is extreme -> p = 1/(k+1)
            0
        });

        assert!(result.stopped_early, "Should stop early for p ~ 0");
        assert_eq!(result.stop_reason, StopReason::SignificantEarly);
        assert!(result.p_value < 0.05);
        assert!(result.n_permutations_used <= 200);
    }

    #[test]
    fn test_max_reached_for_borderline() {
        // Simulate a borderline case where p ~ alpha.
        // The test should run to the max.
        let config = AdaptiveConfig {
            batch_size: 20,
            max_permutations: 200,
            alpha: 0.05,
            confidence: 0.99,
            min_permutations: 40,
        };

        // About 5% of permutations are extreme -> p ~ 0.05
        let mut call_count = 0usize;
        let result = adaptive_permutation_test(&config, |batch_size| {
            call_count += 1;
            // Return ~5% extreme: 1 out of 20
            let extreme = batch_size / 20;
            extreme.max(if call_count.is_multiple_of(2) { 1 } else { 0 })
        });

        assert_eq!(
            result.n_permutations_used, 200,
            "Borderline should reach max"
        );
        assert_eq!(result.stop_reason, StopReason::MaxReached);
    }

    #[test]
    fn test_p_value_matches_fixed_permutation() {
        // With all-extreme permutations and small max, the p-value from
        // adaptive should match the Phipson-Smyth formula exactly.
        let config = AdaptiveConfig {
            batch_size: 100,
            max_permutations: 100,
            alpha: 0.01,
            confidence: 0.99,
            min_permutations: 100,
        };

        let result = adaptive_permutation_test(&config, |batch_size| {
            // Half are extreme
            batch_size / 2
        });

        // r=50 out of k=100: p = (50+1)/(100+1) = 51/101
        let expected_p = 51.0 / 101.0;
        assert!(
            (result.p_value - expected_p).abs() < 1e-10,
            "p_value should be {expected_p}, got {}",
            result.p_value
        );
    }

    #[test]
    fn test_trajectory_recorded() {
        let config = AdaptiveConfig {
            batch_size: 50,
            max_permutations: 200,
            alpha: 0.01, // tight alpha so it runs to max
            confidence: 0.99,
            min_permutations: 200,
        };

        let result = adaptive_permutation_test(&config, |batch_size| batch_size / 4);

        // 200 / 50 = 4 batches
        assert_eq!(result.p_trajectory.len(), 4);
        // Trajectory should be monotonically indexed
        for (i, &(k, _p)) in result.p_trajectory.iter().enumerate() {
            assert_eq!(k, (i + 1) * 50);
        }
    }

    #[test]
    fn test_p_value_never_zero() {
        // Even with zero extreme permutations, Phipson-Smyth ensures p > 0.
        let config = AdaptiveConfig {
            batch_size: 100,
            max_permutations: 100,
            alpha: 0.05,
            confidence: 0.50, // low confidence to avoid early stop
            min_permutations: 100,
        };

        let result = adaptive_permutation_test(&config, |_| 0);
        assert!(result.p_value > 0.0, "p-value should never be zero");
        // p = (0+1)/(100+1) = 1/101 ~ 0.0099
        assert!((result.p_value - 1.0 / 101.0).abs() < 1e-10);
    }

    #[test]
    fn test_respects_min_permutations() {
        // Even for clearly significant results, must do min_permutations first.
        let config = AdaptiveConfig {
            batch_size: 10,
            max_permutations: 10_000,
            alpha: 0.05,
            confidence: 0.99,
            min_permutations: 200,
        };

        let result = adaptive_permutation_test(&config, |_| 0);
        assert!(result.n_permutations_used >= 200);
    }

    /// Monte Carlo test: verify that the adaptive test preserves the
    /// type-I error rate at the nominal alpha level.
    ///
    /// Under the null hypothesis, each permutation is extreme with some
    /// true probability p_true. For a valid test, the fraction of trials
    /// where the adaptive p-value < alpha should not exceed alpha + margin.
    #[test]
    fn test_type_i_error_rate_preserved() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let n_trials = 2000;
        let alpha = 0.05;
        let config = AdaptiveConfig {
            batch_size: 50,
            max_permutations: 1000,
            alpha,
            confidence: 0.99,
            min_permutations: 100,
        };

        let mut rng = StdRng::seed_from_u64(42);
        let mut rejections = 0usize;

        for _ in 0..n_trials {
            // Under the null: draw a true p-value uniformly from [0,1].
            // Each permutation is extreme with probability p_true.
            let p_true: f64 = rng.gen();
            let mut trial_rng = StdRng::seed_from_u64(rng.gen());

            let result = adaptive_permutation_test(&config, |batch_size| {
                let mut extreme = 0usize;
                for _ in 0..batch_size {
                    if trial_rng.gen::<f64>() < p_true {
                        extreme += 1;
                    }
                }
                extreme
            });

            if result.p_value < alpha {
                rejections += 1;
            }
        }

        let observed_rate = rejections as f64 / n_trials as f64;
        // Standard error of the binomial proportion
        let se = (alpha * (1.0 - alpha) / n_trials as f64).sqrt();

        // The rejection rate should be within alpha +/- 3*SE (conservative).
        // With 2000 trials, SE ~ 0.0049, so 3*SE ~ 0.015.
        let margin = 3.0 * se;
        assert!(
            observed_rate <= alpha + margin,
            "Type-I error rate {observed_rate:.4} exceeds alpha + 3*SE = {:.4} \
             ({rejections}/{n_trials} rejections)",
            alpha + margin,
        );
    }

    /// Verify that adaptive testing achieves substantial efficiency gains:
    /// clearly non-significant tests should use far fewer than max_permutations.
    #[test]
    fn test_efficiency_clearly_nonsignificant() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let n_trials = 500;
        let config = AdaptiveConfig {
            batch_size: 50,
            max_permutations: 10_000,
            alpha: 0.05,
            confidence: 0.99,
            min_permutations: 100,
        };

        let mut rng = StdRng::seed_from_u64(123);
        let mut total_perms_used = 0usize;
        let mut n_early_stopped = 0usize;

        for _ in 0..n_trials {
            // Clearly non-significant: true p ~ 0.5 (half of perms are extreme)
            let p_true = 0.4 + rng.gen::<f64>() * 0.2; // p in [0.4, 0.6]
            let mut trial_rng = StdRng::seed_from_u64(rng.gen());

            let result = adaptive_permutation_test(&config, |batch_size| {
                let mut extreme = 0usize;
                for _ in 0..batch_size {
                    if trial_rng.gen::<f64>() < p_true {
                        extreme += 1;
                    }
                }
                extreme
            });

            total_perms_used += result.n_permutations_used;
            if result.stopped_early {
                n_early_stopped += 1;
            }
        }

        let mean_perms = total_perms_used as f64 / n_trials as f64;
        let fraction_early = n_early_stopped as f64 / n_trials as f64;

        // For clearly non-significant tests (p ~ 0.5), the adaptive test
        // should stop early most of the time and use far less than max.
        assert!(
            mean_perms < config.max_permutations as f64 * 0.5,
            "Mean perms {mean_perms:.0} should be < 50% of max {}",
            config.max_permutations,
        );
        assert!(
            fraction_early > 0.90,
            "At least 90% should stop early, got {fraction_early:.2}",
        );
    }
}
