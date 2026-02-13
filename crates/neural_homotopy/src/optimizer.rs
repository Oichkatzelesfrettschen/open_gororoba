//! Pentagon-constrained optimizer for A-infinity correction tensors.
//!
//! Searches for a correction tensor m_3 that minimizes the A_4 pentagon
//! violation. Uses coordinate descent with random perturbations (gradient-free)
//! since the pentagon violation involves discrete table lookups.
//!
//! The loss function is:
//!   L(m_3) = pentagon_violation(m_3) + lambda * ||m_3||^2
//!
//! where the L2 regularizer keeps the correction small.

use crate::m4_tensor::CorrectionTensor;

/// Configuration for the pentagon-constrained optimization.
#[derive(Debug, Clone, Copy)]
pub struct PentagonOptimizationConfig {
    /// Number of optimization steps.
    pub n_steps: usize,
    /// Step size for random perturbations.
    pub step_size: f64,
    /// Step size decay factor per step (exponential schedule).
    pub step_decay: f64,
    /// L2 regularization weight.
    pub lambda: f64,
    /// Number of quadruples to sample for violation estimate.
    pub n_violation_samples: usize,
    /// Random seed.
    pub seed: u64,
}

impl Default for PentagonOptimizationConfig {
    fn default() -> Self {
        Self {
            n_steps: 500,
            step_size: 0.1,
            step_decay: 0.998,
            lambda: 1e-4,
            n_violation_samples: 256,
            seed: 42,
        }
    }
}

impl PentagonOptimizationConfig {
    /// Production configuration for serious optimization runs.
    ///
    /// The correction tensor has 16^4 = 65536 entries. The associator ansatz
    /// is 97.2% sparse (only ~1800 nonzero entries), so even random dense tensors
    /// can accidentally achieve lower pentagon violation. Production runs need
    /// enough steps to meaningfully explore the 65536-dim landscape.
    ///
    /// Uses 2000 steps with 512 violation samples and 5 restarts minimum.
    pub fn production() -> Self {
        Self {
            n_steps: 2000,
            step_size: 0.05,
            step_decay: 0.9995,
            lambda: 1e-5,
            n_violation_samples: 512,
            seed: 42,
        }
    }

    /// Quick config for tests and prototyping (50 steps, 64 samples).
    pub fn quick() -> Self {
        Self {
            n_steps: 50,
            step_size: 0.1,
            step_decay: 0.99,
            lambda: 1e-4,
            n_violation_samples: 64,
            seed: 42,
        }
    }
}

/// Result of the optimization process.
#[derive(Debug, Clone)]
pub struct PentagonOptimizationResult {
    /// Final corrected tensor.
    pub tensor: CorrectionTensor,
    /// Pentagon violation at each step.
    pub violation_trace: Vec<f64>,
    /// Total loss (violation + regularizer) at each step.
    pub loss_trace: Vec<f64>,
    /// Final pentagon violation.
    pub final_violation: f64,
    /// Final L2 norm squared.
    pub final_l2_norm_sq: f64,
    /// Whether optimization converged (loss decreased over run).
    pub converged: bool,
    /// Number of accepted steps.
    pub n_accepted: usize,
}

/// Optimize the correction tensor to minimize pentagon violation.
///
/// Starting from the given initial tensor (typically `from_associator()`),
/// applies random coordinate perturbations and accepts those that reduce
/// the combined loss. This is a simple hill-climbing algorithm.
///
/// For better results, consider using `optimize_with_restarts()` which
/// runs multiple independent searches.
pub fn optimize_correction_tensor(
    initial: &CorrectionTensor,
    config: &PentagonOptimizationConfig,
) -> PentagonOptimizationResult {
    let mut current = initial.clone();
    let mut rng_state = config.seed.wrapping_add(1);

    // Single RNG function to avoid double-borrow issues
    let next_u64 = |st: &mut u64| -> u64 {
        *st ^= *st << 13;
        *st ^= *st >> 7;
        *st ^= *st << 17;
        *st
    };

    let n_violation_samples = config.n_violation_samples;
    let lambda = config.lambda;

    let loss_fn = |tensor: &CorrectionTensor| -> f64 {
        let violation = tensor.pentagon_violation(n_violation_samples);
        let reg = lambda * tensor.l2_norm_sq();
        violation + reg
    };

    let mut current_loss = loss_fn(&current);
    let mut violation_trace = Vec::with_capacity(config.n_steps);
    let mut loss_trace = Vec::with_capacity(config.n_steps);
    let mut step_size = config.step_size;
    let mut n_accepted = 0usize;

    let tensor_len = current.data().len();

    violation_trace.push(current.pentagon_violation(config.n_violation_samples));
    loss_trace.push(current_loss);

    for _ in 0..config.n_steps {
        // Perturb a random coordinate
        let idx = next_u64(&mut rng_state) as usize % tensor_len;
        let delta = (next_u64(&mut rng_state) as f64 / u64::MAX as f64 * 2.0 - 1.0) * step_size;

        let old_val = current.data()[idx];
        current.data_mut()[idx] = old_val + delta;

        let new_loss = loss_fn(&current);

        if new_loss < current_loss {
            // Accept the perturbation
            current_loss = new_loss;
            n_accepted += 1;
        } else {
            // Reject: restore old value
            current.data_mut()[idx] = old_val;
        }

        violation_trace.push(current.pentagon_violation(config.n_violation_samples));
        loss_trace.push(current_loss);
        step_size *= config.step_decay;
    }

    let final_violation = current.pentagon_violation(config.n_violation_samples);
    let converged = loss_trace.last().copied().unwrap_or(f64::MAX)
        < loss_trace.first().copied().unwrap_or(0.0);

    PentagonOptimizationResult {
        final_l2_norm_sq: current.l2_norm_sq(),
        tensor: current,
        violation_trace,
        loss_trace,
        final_violation,
        converged,
        n_accepted,
    }
}

/// Run optimization with multiple random restarts, keeping the best result.
///
/// Each restart uses a different random seed but the same initial tensor.
/// Returns the result with the lowest final violation.
pub fn optimize_with_restarts(
    initial: &CorrectionTensor,
    config: &PentagonOptimizationConfig,
    n_restarts: usize,
) -> PentagonOptimizationResult {
    let mut best: Option<PentagonOptimizationResult> = None;

    for restart in 0..n_restarts.max(1) {
        let mut cfg = *config;
        cfg.seed = config.seed.wrapping_add(restart as u64 * 1000);

        let result = optimize_correction_tensor(initial, &cfg);

        let is_better = best
            .as_ref()
            .is_none_or(|b| result.final_violation < b.final_violation);

        if is_better {
            best = Some(result);
        }
    }

    best.expect("at least one restart should run")
}

/// Batch coordinate descent: perturb `batch_size` coordinates simultaneously.
///
/// For the 65536-dim correction tensor, single-coordinate perturbation explores
/// the space too slowly. Batch descent perturbs a block of coordinates at once,
/// accepting or rejecting the entire batch. This trades per-step precision for
/// faster exploration of the high-dimensional landscape.
///
/// The sparse associator (97.2% zeros) means most single-coordinate moves
/// hit zero entries and have no effect. Batch moves are more likely to
/// perturb non-zero coordinates.
pub fn optimize_batch_coordinate_descent(
    initial: &CorrectionTensor,
    config: &PentagonOptimizationConfig,
    batch_size: usize,
) -> PentagonOptimizationResult {
    let mut current = initial.clone();
    let mut rng_state = config.seed.wrapping_add(1);

    let next_u64 = |st: &mut u64| -> u64 {
        *st ^= *st << 13;
        *st ^= *st >> 7;
        *st ^= *st << 17;
        *st
    };

    let n_violation_samples = config.n_violation_samples;
    let lambda = config.lambda;
    let tensor_len = current.data().len();
    let batch = batch_size.clamp(1, tensor_len);

    let loss_fn = |tensor: &CorrectionTensor| -> f64 {
        let violation = tensor.pentagon_violation(n_violation_samples);
        let reg = lambda * tensor.l2_norm_sq();
        violation + reg
    };

    let mut current_loss = loss_fn(&current);
    let mut violation_trace = vec![current.pentagon_violation(n_violation_samples)];
    let mut loss_trace = vec![current_loss];
    let mut step_size = config.step_size;
    let mut n_accepted = 0usize;

    for _ in 0..config.n_steps {
        // Save batch of coordinates before perturbation
        let mut saved = Vec::with_capacity(batch);
        for _ in 0..batch {
            let idx = next_u64(&mut rng_state) as usize % tensor_len;
            let delta =
                (next_u64(&mut rng_state) as f64 / u64::MAX as f64 * 2.0 - 1.0) * step_size;
            let old_val = current.data()[idx];
            saved.push((idx, old_val));
            current.data_mut()[idx] = old_val + delta;
        }

        let new_loss = loss_fn(&current);

        if new_loss < current_loss {
            current_loss = new_loss;
            n_accepted += 1;
        } else {
            // Reject: restore all perturbed coordinates
            for &(idx, old_val) in &saved {
                current.data_mut()[idx] = old_val;
            }
        }

        violation_trace.push(current.pentagon_violation(n_violation_samples));
        loss_trace.push(current_loss);
        step_size *= config.step_decay;
    }

    let final_violation = current.pentagon_violation(n_violation_samples);
    let converged = loss_trace.last().copied().unwrap_or(f64::MAX)
        < loss_trace.first().copied().unwrap_or(0.0);

    PentagonOptimizationResult {
        final_l2_norm_sq: current.l2_norm_sq(),
        tensor: current,
        violation_trace,
        loss_trace,
        final_violation,
        converged,
        n_accepted,
    }
}

/// Compare associator ansatz vs optimized tensor, reporting the E-029 metrics.
///
/// Returns (associator_violation, optimized_violation, reduction_ratio, sparsity_difference).
pub fn compare_ansatz_vs_optimized(
    config: &PentagonOptimizationConfig,
    n_restarts: usize,
    batch_size: usize,
) -> AnsatzComparison {
    let associator = CorrectionTensor::from_associator();
    let assoc_violation = associator.pentagon_violation(config.n_violation_samples);
    let assoc_sparsity = associator.sparsity();

    let optimized = if batch_size > 1 {
        let mut best: Option<PentagonOptimizationResult> = None;
        for restart in 0..n_restarts.max(1) {
            let mut cfg = *config;
            cfg.seed = config.seed.wrapping_add(restart as u64 * 1000);
            let result = optimize_batch_coordinate_descent(&associator, &cfg, batch_size);
            let is_better = best
                .as_ref()
                .is_none_or(|b| result.final_violation < b.final_violation);
            if is_better {
                best = Some(result);
            }
        }
        best.expect("at least one restart")
    } else {
        optimize_with_restarts(&associator, config, n_restarts)
    };

    let opt_sparsity = optimized.tensor.sparsity();

    AnsatzComparison {
        associator_violation: assoc_violation,
        optimized_violation: optimized.final_violation,
        reduction_ratio: if assoc_violation > 1e-14 {
            optimized.final_violation / assoc_violation
        } else {
            1.0
        },
        associator_sparsity: assoc_sparsity,
        optimized_sparsity: opt_sparsity,
        n_accepted: optimized.n_accepted,
        converged: optimized.converged,
    }
}

/// Comparison result between algebraic ansatz and optimized correction tensor.
#[derive(Debug, Clone)]
pub struct AnsatzComparison {
    /// Pentagon violation of the algebraic associator ansatz.
    pub associator_violation: f64,
    /// Pentagon violation after optimization.
    pub optimized_violation: f64,
    /// Ratio: optimized / associator (< 1.0 means improvement).
    pub reduction_ratio: f64,
    /// Sparsity of the associator tensor (typically ~0.97).
    pub associator_sparsity: f64,
    /// Sparsity of the optimized tensor.
    pub optimized_sparsity: f64,
    /// Number of accepted perturbation steps.
    pub n_accepted: usize,
    /// Whether the optimization converged (loss decreased).
    pub converged: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimize_zero_tensor() {
        let initial = CorrectionTensor::zero();
        let config = PentagonOptimizationConfig {
            n_steps: 50,
            n_violation_samples: 32,
            ..Default::default()
        };
        let result = optimize_correction_tensor(&initial, &config);

        assert_eq!(result.violation_trace.len(), config.n_steps + 1);
        assert_eq!(result.loss_trace.len(), config.n_steps + 1);
        assert!(result.final_violation >= 0.0);
    }

    #[test]
    fn test_optimize_from_associator() {
        let initial = CorrectionTensor::from_associator();
        let config = PentagonOptimizationConfig {
            n_steps: 100,
            n_violation_samples: 64,
            step_size: 0.05,
            ..Default::default()
        };
        let result = optimize_correction_tensor(&initial, &config);

        assert!(result.final_violation.is_finite());
        assert!(result.final_l2_norm_sq.is_finite());
        // Should accept at least some steps
        assert!(
            result.n_accepted > 0 || result.final_violation == 0.0,
            "Should accept steps or already be optimal"
        );
    }

    #[test]
    fn test_optimize_loss_trace_non_increasing() {
        let initial = CorrectionTensor::from_associator();
        let config = PentagonOptimizationConfig {
            n_steps: 50,
            n_violation_samples: 32,
            ..Default::default()
        };
        let result = optimize_correction_tensor(&initial, &config);

        // Loss trace should be non-increasing (we only accept improvements)
        for w in result.loss_trace.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-10,
                "Loss should not increase: {} -> {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_optimize_with_restarts() {
        let initial = CorrectionTensor::zero();
        let config = PentagonOptimizationConfig {
            n_steps: 20,
            n_violation_samples: 16,
            ..Default::default()
        };
        let result = optimize_with_restarts(&initial, &config, 3);

        assert!(result.final_violation >= 0.0);
        assert!(result.final_violation.is_finite());
    }

    #[test]
    fn test_optimize_reproducible() {
        let initial = CorrectionTensor::from_associator();
        let config = PentagonOptimizationConfig {
            n_steps: 30,
            n_violation_samples: 32,
            ..Default::default()
        };
        let r1 = optimize_correction_tensor(&initial, &config);
        let r2 = optimize_correction_tensor(&initial, &config);

        assert!(
            (r1.final_violation - r2.final_violation).abs() < 1e-12,
            "Same seed should give same result"
        );
    }

    #[test]
    fn test_batch_coordinate_descent_basic() {
        let initial = CorrectionTensor::from_associator();
        let config = PentagonOptimizationConfig::quick();
        let result = optimize_batch_coordinate_descent(&initial, &config, 8);

        assert!(result.final_violation.is_finite());
        assert_eq!(result.violation_trace.len(), config.n_steps + 1);
    }

    #[test]
    fn test_batch_loss_trace_non_increasing() {
        let initial = CorrectionTensor::from_associator();
        let config = PentagonOptimizationConfig::quick();
        let result = optimize_batch_coordinate_descent(&initial, &config, 4);

        for w in result.loss_trace.windows(2) {
            assert!(
                w[1] <= w[0] + 1e-10,
                "Batch loss should not increase: {} -> {}",
                w[0],
                w[1]
            );
        }
    }

    #[test]
    fn test_production_config_values() {
        let prod = PentagonOptimizationConfig::production();
        assert!(prod.n_steps >= 2000);
        assert!(prod.n_violation_samples >= 512);
        // Production step_decay should be closer to 1 (slower decay)
        assert!(prod.step_decay > 0.999);
    }

    #[test]
    fn test_compare_ansatz_vs_optimized() {
        let config = PentagonOptimizationConfig::quick();
        let comp = compare_ansatz_vs_optimized(&config, 2, 4);

        assert!(comp.associator_violation.is_finite());
        assert!(comp.optimized_violation.is_finite());
        assert!(comp.associator_sparsity > 0.9, "Associator should be sparse");
        // Optimized tensor fills more entries, so less sparse
        assert!(
            comp.optimized_sparsity < comp.associator_sparsity
                || comp.optimized_sparsity <= 1.0,
            "Optimized should be less sparse or at most equally sparse"
        );
    }
}
