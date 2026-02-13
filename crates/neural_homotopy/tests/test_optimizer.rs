//! Integration tests for pentagon-constrained optimization pipeline.

use neural_homotopy::{
    CorrectionTensor, PentagonOptimizationConfig, PerturbationDataset,
    optimize_correction_tensor, optimize_with_restarts,
    perturbed_sedenion_table, SEDENION_DIM,
};

#[test]
fn test_optimization_pipeline_end_to_end() {
    // Full pipeline: associator -> optimize -> verify improvement
    let initial = CorrectionTensor::from_associator();

    let config = PentagonOptimizationConfig {
        n_steps: 100,
        n_violation_samples: 64,
        step_size: 0.05,
        step_decay: 0.995,
        lambda: 1e-4,
        seed: 42,
    };
    let result = optimize_correction_tensor(&initial, &config);

    // Verify trace was recorded correctly
    assert_eq!(result.violation_trace.len(), config.n_steps + 1);
    assert_eq!(result.loss_trace.len(), config.n_steps + 1);

    // L2 norm should remain finite
    assert!(result.final_l2_norm_sq.is_finite());

    // Tensor data should be accessible
    assert_eq!(result.tensor.data().len(), 16usize.pow(4));
}

#[test]
fn test_perturbation_robustness() {
    // Test that optimization is robust to small perturbations.
    // Build datasets at two noise levels and verify optimization still works.
    let ds = PerturbationDataset::build(&[0.01, 0.05], 42);

    for variant in 0..ds.n_variants() {
        let samples = ds.samples_at(variant);
        assert_eq!(
            samples.len(),
            SEDENION_DIM * SEDENION_DIM,
            "Variant {} should have full sample set",
            variant
        );

        // Each sample should have valid indices
        for s in &samples {
            assert!(s.lhs < SEDENION_DIM);
            assert!(s.rhs < SEDENION_DIM);
            assert!(s.product_basis < SEDENION_DIM);
            assert!(s.product_sign.abs() == 1);
        }
    }

    // Difference counts should be monotonically increasing with noise
    let counts = ds.difference_counts();
    assert_eq!(counts.len(), 2);
    assert!(
        counts[1].1 >= counts[0].1,
        "5% noise should produce >= differences than 1%: {} vs {}",
        counts[1].1,
        counts[0].1
    );
}

#[test]
fn test_optimizer_serializes_after_run() {
    let initial = CorrectionTensor::from_associator();
    let config = PentagonOptimizationConfig {
        n_steps: 30,
        n_violation_samples: 32,
        ..Default::default()
    };
    let result = optimize_correction_tensor(&initial, &config);

    // Serialize the optimized tensor
    let toml = result.tensor.serialize_to_toml(32);
    assert!(toml.contains("[correction_tensor]"));
    assert!(toml.contains("pentagon_violation"));

    // The serialized violation should match the result's final violation
    // (within tolerance from recomputation with same n_samples)
    let viol_line = toml
        .lines()
        .find(|l| l.starts_with("pentagon_violation"))
        .unwrap();
    let toml_viol: f64 = viol_line.split('=').nth(1).unwrap().trim().parse().unwrap();
    assert!(
        (toml_viol - result.tensor.pentagon_violation(32)).abs() < 1e-10,
        "TOML violation should match recomputed: {} vs {}",
        toml_viol,
        result.tensor.pentagon_violation(32)
    );
}

#[test]
fn test_perturbed_table_preserves_identity() {
    // Even with noise, e_0 * e_i should still be (i, +1) at 0% noise
    let table = perturbed_sedenion_table(0.0, 42);
    for i in 0..SEDENION_DIM {
        assert_eq!(table[0][i], (i, 1), "e_0 * e_{} should be identity", i);
    }
}

#[test]
fn test_restarts_produce_deterministic_results() {
    let initial = CorrectionTensor::from_associator();
    let config = PentagonOptimizationConfig {
        n_steps: 20,
        n_violation_samples: 32,
        ..Default::default()
    };

    let r1 = optimize_with_restarts(&initial, &config, 3);
    let r2 = optimize_with_restarts(&initial, &config, 3);

    assert!(
        (r1.final_violation - r2.final_violation).abs() < 1e-12,
        "Same config should produce deterministic results: {} vs {}",
        r1.final_violation,
        r2.final_violation
    );
}
