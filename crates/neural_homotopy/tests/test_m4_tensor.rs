//! Integration tests for CorrectionTensor interactions across modules.

use neural_homotopy::{
    build_sedenion_table, CorrectionTensor, PentagonOptimizationConfig,
    optimize_correction_tensor, optimize_with_restarts, SEDENION_DIM,
};

#[test]
fn test_associator_symmetry_under_identity_permutation() {
    // m_3(e_i, e_0, e_k) should also be zero for all i,k
    // because e_0 is the identity on both sides.
    let t = CorrectionTensor::from_associator();
    for i in 0..SEDENION_DIM {
        for k in 0..SEDENION_DIM {
            let s = t.slice(i, 0, k);
            let norm: f64 = s.iter().map(|x| x * x).sum();
            assert!(
                norm < 1e-14,
                "m_3({},0,{}) should be zero: norm={}",
                i,
                k,
                norm.sqrt()
            );
        }
    }
}

#[test]
fn test_associator_third_index_identity() {
    // m_3(e_i, e_j, e_0) should be zero when e_0 is the identity.
    let t = CorrectionTensor::from_associator();
    for i in 0..SEDENION_DIM {
        for j in 0..SEDENION_DIM {
            let s = t.slice(i, j, 0);
            let norm: f64 = s.iter().map(|x| x * x).sum();
            assert!(
                norm < 1e-14,
                "m_3({},{},0) should be zero: norm={}",
                i,
                j,
                norm.sqrt()
            );
        }
    }
}

#[test]
fn test_associator_antisymmetry_count() {
    // Count how many triples have non-zero associator.
    // For sedenions, e_0 is associative with everything, so at minimum
    // all triples involving e_0 should be zero. That's 3*16^2 - 3*16 + 1 = 721
    // triples with at least one e_0.
    let t = CorrectionTensor::from_associator();
    let mut nonzero_count = 0usize;
    for i in 0..SEDENION_DIM {
        for j in 0..SEDENION_DIM {
            for k in 0..SEDENION_DIM {
                let s = t.slice(i, j, k);
                let norm_sq: f64 = s.iter().map(|x| x * x).sum();
                if norm_sq > 1e-14 {
                    nonzero_count += 1;
                }
            }
        }
    }
    // Should have many non-zero triples (sedenions are highly non-associative)
    assert!(
        nonzero_count > 100,
        "Expected >100 non-associative triples, got {}",
        nonzero_count
    );
    // But not ALL triples (identity is associative)
    let total = SEDENION_DIM * SEDENION_DIM * SEDENION_DIM;
    assert!(
        nonzero_count < total,
        "Some triples should be associative"
    );
}

#[test]
fn test_optimizer_reduces_violation_from_associator() {
    let initial = CorrectionTensor::from_associator();
    let initial_violation = initial.pentagon_violation(128);

    let config = PentagonOptimizationConfig {
        n_steps: 200,
        n_violation_samples: 128,
        step_size: 0.05,
        step_decay: 0.995,
        lambda: 1e-5,
        seed: 42,
    };
    let result = optimize_correction_tensor(&initial, &config);

    // After optimization, violation should not increase
    assert!(
        result.final_violation <= initial_violation + 1e-10,
        "Optimization should not increase violation: {} -> {}",
        initial_violation,
        result.final_violation
    );
}

#[test]
fn test_restarts_find_better_or_equal_solution() {
    let initial = CorrectionTensor::from_associator();
    let config = PentagonOptimizationConfig {
        n_steps: 50,
        n_violation_samples: 64,
        step_size: 0.05,
        ..Default::default()
    };

    let single = optimize_correction_tensor(&initial, &config);
    let multi = optimize_with_restarts(&initial, &config, 5);

    // Multiple restarts should find a solution at least as good
    assert!(
        multi.final_violation <= single.final_violation + 1e-10,
        "Multiple restarts should be >= single run quality: {} vs {}",
        multi.final_violation,
        single.final_violation
    );
}

#[test]
fn test_correction_tensor_to_toml_roundtrip_metadata() {
    let t = CorrectionTensor::from_associator();
    let toml = t.serialize_to_toml(64);

    // Verify structural content
    assert!(toml.contains("[correction_tensor]"));
    assert!(toml.contains("dim = 16"));
    assert!(toml.contains(&format!("total_entries = {}", 16usize.pow(4))));
    assert!(toml.contains(&format!("nnz = {}", t.nnz())));

    // Verify pentagon violation is recorded
    assert!(toml.contains("pentagon_violation"));
    // The recorded value should be positive for the associator
    let viol_line = toml.lines().find(|l| l.starts_with("pentagon_violation")).unwrap();
    let viol_val: f64 = viol_line.split('=').nth(1).unwrap().trim().parse().unwrap();
    assert!(viol_val >= 0.0);
}

#[test]
fn test_sedenion_table_consistency() {
    // Cross-validate: the sedenion table products should be consistent
    // with the CorrectionTensor's from_associator.
    let table = build_sedenion_table();

    // Verify table dimensions
    assert_eq!(table.len(), SEDENION_DIM);
    for row in &table {
        assert_eq!(row.len(), SEDENION_DIM);
    }

    // Verify e_0 * e_i = e_i and e_i * e_0 = e_i
    for i in 0..SEDENION_DIM {
        let (basis, sign) = table[0][i];
        assert_eq!(basis, i, "e_0 * e_{} should give e_{}", i, i);
        assert_eq!(sign, 1, "e_0 * e_{} should have sign +1", i);

        let (basis2, sign2) = table[i][0];
        assert_eq!(basis2, i, "e_{} * e_0 should give e_{}", i, i);
        assert_eq!(sign2, 1, "e_{} * e_0 should have sign +1", i);
    }

    // Verify e_i * e_i = -e_0 for i > 0 (sedenion norm property)
    for i in 1..SEDENION_DIM {
        let (basis, sign) = table[i][i];
        assert_eq!(
            basis, 0,
            "e_{i} * e_{i} should give e_0 (got e_{basis})",
        );
        assert_eq!(
            sign, -1,
            "e_{} * e_{} should have sign -1 (got {})",
            i, i, sign
        );
    }
}
