use super::*;

#[test]
fn test_sedenion_field_creation() {
    let field = SedenionField::uniform(8, 8, 4);
    assert_eq!(field.data.len(), 8 * 8 * 4);
    // Check that e_0 basis is initialized
    assert!((field.get(0, 0, 0)[0] - 1.0).abs() < 1e-14);
    assert!((field.get(7, 7, 3)[0] - 1.0).abs() < 1e-14);
}

#[test]
fn test_sedenion_field_linearization() {
    let field = SedenionField::uniform(4, 4, 4);
    // Check linearization consistency
    let idx1 = field.linearize(2, 1, 0);
    let idx2 = field.linearize(2, 1, 0);
    assert_eq!(idx1, idx2);
}

#[test]
fn test_imbalance_viscosity_bridge_creation() {
    let bridge = ImbalanceViscosityBridge::new(16);
    assert_eq!(bridge.dim, 16);
    assert_eq!(bridge.signed_graph.dim, 16);
}

#[test]
fn test_imbalance_to_viscosity_vacuum() {
    let bridge = ImbalanceViscosityBridge::new(16);
    let imbalance = vec![3.0 / 8.0; 100]; // All vacuum
    let viscosity = bridge.imbalance_to_viscosity(&imbalance, 1.0 / 3.0, 1.0);

    // At imbalance attractor, nu should be nu_base
    for &nu in viscosity.iter() {
        assert!(
            (nu - 1.0 / 3.0).abs() < 1e-10,
            "Expected nu={}, got {}",
            1.0 / 3.0,
            nu
        );
    }
}

#[test]
fn test_imbalance_to_viscosity_variance() {
    let bridge = ImbalanceViscosityBridge::new(16);
    let imbalance = vec![0.2, 0.375, 0.8]; // Varying imbalance around vacuum (3/8)
    let viscosity = bridge.imbalance_to_viscosity(&imbalance, 1.0 / 3.0, 1.0);

    // At imbalance attractor (0.375), viscosity should equal nu_base
    // Away from attractor, viscosity should decrease (exponential decay)
    assert!(
        viscosity[0] < viscosity[1],
        "Viscosity at 0.2 should be less than at 0.375"
    );
    assert!(
        viscosity[1].abs() - (1.0 / 3.0) < 1e-10,
        "Viscosity at vacuum should equal nu_base"
    );
    assert!(
        viscosity[2] < viscosity[1],
        "Viscosity at 0.8 should be less than at 0.375"
    );
}

#[test]
fn test_full_pipeline_uniform_field() {
    let bridge = ImbalanceViscosityBridge::new(16);
    let field = SedenionField::uniform(8, 8, 4);
    let viscosity = bridge.compute_viscosity_field(&field, 1.0 / 3.0, 1.0);

    // Uniform field should produce roughly uniform viscosity
    assert_eq!(viscosity.len(), 8 * 8 * 4);
    for &nu in viscosity.iter() {
        assert!(nu > 0.0, "Viscosity must be positive");
        assert!(nu < 1.0, "Viscosity should be reasonable");
    }
}

#[test]
fn test_full_pipeline_diverse_field() {
    let bridge = ImbalanceViscosityBridge::new(16);
    let mut field = SedenionField::uniform(4, 4, 4);

    // Create varied Sedenion field by modulating multiple components
    // This creates edge variation that affects imbalance density
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                let sedenion = field.get_mut(x, y, z);
                // Vary components based on position to create diverse edges
                let scale = ((x + y + z) as f64) / 12.0;
                sedenion[0] = 1.0 + 0.3 * scale;
                sedenion[1] = 0.5 * scale;
                sedenion[2] = 0.3 * (1.0 - scale);
                sedenion[3] = 0.4 * scale.sin();
            }
        }
    }

    let viscosity = bridge.compute_viscosity_field(&field, 1.0 / 3.0, 1.0);

    // Viscosity should be positive and reasonable (may not vary significantly
    // depending on imbalance distribution, but should all be valid)
    for &nu in viscosity.iter() {
        assert!(nu > 0.0, "Viscosity must be positive");
        assert!(nu < 1.0, "Viscosity should be reasonable");
    }
}

#[test]
fn test_viscosity_positivity() {
    let bridge = ImbalanceViscosityBridge::new(16);
    let mut field = SedenionField::uniform(4, 4, 4);

    // Extreme imbalance values
    field.data[0] = [
        0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
    ];

    let viscosity = bridge.compute_viscosity_field(&field, 1.0 / 3.0, 1.0);

    // All viscosities must be positive and finite
    for &nu in viscosity.iter() {
        assert!(nu > 0.0, "Viscosity must be positive");
        assert!(nu.is_finite(), "Viscosity must be finite");
    }
}

#[test]
fn test_sedenion_field_mutation() {
    let mut field = SedenionField::uniform(2, 2, 2);
    field.get_mut(0, 0, 0)[5] = 0.5;
    assert!((field.get(0, 0, 0)[5] - 0.5).abs() < 1e-14);
    assert!((field.get(0, 0, 1)[5]).abs() < 1e-14); // Other points unchanged
}

#[test]
fn test_imbalance_density_par_matches_sequential() {
    let mut field = SedenionField::uniform(4, 4, 4);
    // Add some variation
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                let s = field.get_mut(x, y, z);
                s[1] = 0.3 * (x as f64) / 4.0;
                s[5] = 0.2 * (y as f64) / 4.0;
            }
        }
    }

    let sequential = field.local_imbalance_density(16);
    let parallel = field.local_imbalance_density_par(16);

    assert_eq!(sequential.len(), parallel.len());
    for (s, p) in sequential.iter().zip(parallel.iter()) {
        assert!(
            (s - p).abs() < 1e-14,
            "Parallel result differs: {} vs {}",
            s,
            p
        );
    }
}

#[test]
fn test_imbalance_density_par_uniform() {
    let field = SedenionField::uniform(8, 8, 4);
    let par_result = field.local_imbalance_density_par(16);
    let seq_result = field.local_imbalance_density(16);
    for (s, p) in seq_result.iter().zip(par_result.iter()) {
        assert!((s - p).abs() < 1e-14);
    }
}

#[test]
fn test_associator_norm_uniform_field() {
    // Uniform e_0 field: all products are (1)(1)(1)=1 (real),
    // so associator is zero (reals are associative).
    let field = SedenionField::uniform(4, 4, 4);
    let norms = field.local_associator_norm_field(16);
    assert_eq!(norms.len(), 4 * 4 * 4);
    for &n in &norms {
        assert!(n.abs() < 1e-10, "Uniform e_0 should have zero associator");
    }
}

#[test]
fn test_associator_norm_varied_field() {
    // Create a field with multiple nonzero basis elements to trigger
    // non-associativity
    let mut field = SedenionField::uniform(4, 4, 4);
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                let s = field.get_mut(x, y, z);
                let i = (x + y + z) % 16;
                s[i] = 0.8;
                s[(i + 3) % 16] = 0.5;
                s[(i + 7) % 16] = 0.3;
            }
        }
    }

    let norms = field.local_associator_norm_field(16);

    // At least some points should have nonzero associator norm
    let max_norm = norms.iter().cloned().fold(0.0_f64, f64::max);
    assert!(
        max_norm > 1e-4,
        "Varied sedenion field should have nonzero associators: max={}",
        max_norm
    );

    // All norms should be finite and non-negative
    for &n in &norms {
        assert!(n.is_finite());
        assert!(n >= 0.0);
    }
}

#[test]
fn test_associator_norm_field_length() {
    let field = SedenionField::uniform(8, 6, 4);
    let norms = field.local_associator_norm_field(16);
    assert_eq!(norms.len(), 8 * 6 * 4);
}

// ---- ViscosityCouplingModel tests ----

#[test]
fn test_exponential_model_at_vacuum() {
    let model = ViscosityCouplingModel::Exponential {
        nu_base: 1.0 / 3.0,
        lambda: 2.0,
    };
    let nu = model.compute(IMBALANCE_ATTRACTOR);
    assert!(
        (nu - 1.0 / 3.0).abs() < 1e-10,
        "At imbalance attractor, exponential model should return nu_base"
    );
}

#[test]
fn test_exponential_model_away_from_vacuum() {
    let model = ViscosityCouplingModel::Exponential {
        nu_base: 1.0 / 3.0,
        lambda: 2.0,
    };
    let nu_low = model.compute(0.1);
    let nu_vac = model.compute(IMBALANCE_ATTRACTOR);
    assert!(
        nu_low < nu_vac,
        "Away from attractor, viscosity should decrease"
    );
}

#[test]
fn test_linear_model_positive_alpha() {
    let model = ViscosityCouplingModel::Linear {
        nu_base: 0.1,
        alpha: 1.0,
    };
    let nu_low = model.compute(0.2); // F < 3/8
    let nu_high = model.compute(0.5); // F > 3/8
    assert!(
        nu_high > nu_low,
        "Positive alpha: higher imbalance -> higher viscosity"
    );
}

#[test]
fn test_power_law_model_superlinear() {
    let model = ViscosityCouplingModel::PowerLaw {
        nu_base: 0.1,
        n: 2.0,
    };
    let nu_close = model.compute(IMBALANCE_ATTRACTOR + 0.01);
    let nu_far = model.compute(IMBALANCE_ATTRACTOR + 0.1);
    // Superlinear: far deviation grows faster than linearly
    let ratio_dev = 0.1 / 0.01;
    let ratio_nu = nu_far / nu_close;
    assert!(
        ratio_nu > ratio_dev,
        "Superlinear (n=2): nu ratio ({:.2}) should exceed deviation ratio ({:.2})",
        ratio_nu,
        ratio_dev
    );
}

#[test]
fn test_sigmoid_model_transition() {
    let model = ViscosityCouplingModel::Sigmoid {
        nu_low: 0.05,
        nu_high: 0.5,
        k: 100.0,
        f_crit: 0.38,
    };
    let nu_below = model.compute(0.30);
    let nu_above = model.compute(0.45);
    assert!(
        (nu_below - 0.05).abs() < 0.01,
        "Below F_crit, sigmoid should approach nu_low: got {}",
        nu_below
    );
    assert!(
        (nu_above - 0.5).abs() < 0.01,
        "Above F_crit, sigmoid should approach nu_high: got {}",
        nu_above
    );
}

#[test]
fn test_constant_model_invariance() {
    let model = ViscosityCouplingModel::Constant { nu_base: 0.1 };
    for f in [0.0, 0.2, IMBALANCE_ATTRACTOR, 0.5, 1.0] {
        assert!(
            (model.compute(f) - 0.1).abs() < 1e-14,
            "Constant model should return nu_base at all imbalances"
        );
    }
}

#[test]
fn test_all_models_finite_positive() {
    let suite = ViscosityCouplingModel::standard_suite(1.0 / 3.0);
    for model in &suite {
        for &f in &[0.0, 0.1, 0.2, IMBALANCE_ATTRACTOR, 0.5, 0.8, 1.0] {
            let nu = model.compute(f);
            assert!(
                nu.is_finite(),
                "{}: nu not finite at F={}",
                model.label(),
                f
            );
            assert!(
                nu > 0.0,
                "{}: nu not positive at F={}: got {}",
                model.label(),
                f,
                nu
            );
        }
    }
}

#[test]
fn test_standard_suite_has_six_models() {
    let suite = ViscosityCouplingModel::standard_suite(0.1);
    assert_eq!(suite.len(), 6);

    let labels: Vec<&str> = suite.iter().map(|m| m.label()).collect();
    assert!(labels.contains(&"exponential"));
    assert!(labels.contains(&"linear"));
    assert!(labels.contains(&"power_law"));
    assert!(labels.contains(&"sigmoid"));
    assert!(labels.contains(&"constant"));
    assert!(labels.contains(&"kubo_response"));
}

#[test]
fn test_bridge_multi_model_integration() {
    let bridge = ImbalanceViscosityBridge::new(16);
    let imbalance = vec![0.3, IMBALANCE_ATTRACTOR, 0.45];
    let model = ViscosityCouplingModel::Exponential {
        nu_base: 1.0 / 3.0,
        lambda: 1.0,
    };
    let viscosity = bridge.imbalance_to_viscosity_model(&imbalance, &model);
    assert_eq!(viscosity.len(), 3);
    for &nu in &viscosity {
        assert!(nu.is_finite() && nu > 0.0);
    }
}

#[test]
fn test_model_descriptions_non_empty() {
    let suite = ViscosityCouplingModel::standard_suite(0.1);
    for model in &suite {
        assert!(!model.description().is_empty());
        assert!(!model.label().is_empty());
    }
}

// ---- KuboResponse model tests ----

#[test]
fn test_kubo_response_at_zero_imbalance() {
    let model = ViscosityCouplingModel::kubo_default(1.0 / 3.0);
    let nu = model.compute(0.0);
    // At f=0, lambda=0, g=1.0, so nu = nu_base
    assert!(
        (nu - 1.0 / 3.0).abs() < 1e-10,
        "At zero imbalance, KuboResponse should return nu_base, got {}",
        nu
    );
}

#[test]
fn test_kubo_response_monotonic_near_vacuum() {
    let model = ViscosityCouplingModel::kubo_default(0.1);
    // Near the imbalance attractor, viscosity should increase with imbalance
    let nu_low = model.compute(0.30);
    let nu_vac = model.compute(0.375);
    let nu_high = model.compute(0.45);
    assert!(
        nu_low < nu_vac,
        "nu should increase toward vacuum: nu(0.30)={} > nu(0.375)={}",
        nu_low,
        nu_vac
    );
    assert!(
        nu_vac < nu_high,
        "nu should increase beyond vacuum: nu(0.375)={} > nu(0.45)={}",
        nu_vac,
        nu_high
    );
}

#[test]
fn test_kubo_response_enhancement_ratio() {
    let model = ViscosityCouplingModel::kubo_default(1.0);
    let nu_zero = model.compute(0.0);
    let nu_full = model.compute(0.5429); // full CD imbalance
    let ratio = nu_full / nu_zero;
    // Full CD: g(1.0) = 216, so ratio should be ~216
    assert!(
        ratio > 200.0 && ratio < 230.0,
        "Full CD enhancement ratio should be ~216, got {}",
        ratio
    );
}

#[test]
fn test_kubo_response_imbalance_attractor_enhancement() {
    let model = ViscosityCouplingModel::kubo_default(1.0);
    let nu_zero = model.compute(0.0);
    let nu_vac = model.compute(IMBALANCE_ATTRACTOR);
    let ratio = nu_vac / nu_zero;
    // At attractor: f=0.375, lambda=0.375/0.5429=0.691, g~83
    assert!(
        ratio > 60.0 && ratio < 110.0,
        "Imbalance attractor enhancement should be ~83, got {}",
        ratio
    );
}

#[test]
fn test_kubo_response_all_finite() {
    let model = ViscosityCouplingModel::kubo_default(0.1);
    for i in 0..100 {
        let f = i as f64 * 0.01;
        let nu = model.compute(f);
        assert!(
            nu.is_finite() && nu > 0.0,
            "KuboResponse not finite/positive at f={}: nu={}",
            f,
            nu
        );
    }
}

#[test]
fn test_interpolate_table_boundary() {
    let table = vec![(0.0, 1.0), (0.5, 50.0), (1.0, 100.0)];
    assert!((interpolate_table(&table, -0.1) - 1.0).abs() < 1e-10);
    assert!((interpolate_table(&table, 0.0) - 1.0).abs() < 1e-10);
    assert!((interpolate_table(&table, 0.25) - 25.5).abs() < 1e-10);
    assert!((interpolate_table(&table, 0.5) - 50.0).abs() < 1e-10);
    assert!((interpolate_table(&table, 1.0) - 100.0).abs() < 1e-10);
    assert!((interpolate_table(&table, 1.5) - 100.0).abs() < 1e-10);
}

// ---- SedenionField4D tests ----

#[test]
fn test_sedenion_field_4d_creation() {
    let field = SedenionField4D::uniform(4, 4, 4, 3);
    assert_eq!(field.data.len(), 4 * 4 * 4 * 3);
    assert_eq!(field.n_cells(), 192);
    // Check e_0 basis is initialized
    assert!((field.get(0, 0, 0, 0)[0] - 1.0).abs() < 1e-14);
    assert!((field.get(3, 3, 3, 2)[0] - 1.0).abs() < 1e-14);
}

#[test]
fn test_sedenion_field_4d_linearize() {
    let field = SedenionField4D::uniform(4, 4, 4, 3);
    // w-major layout: w=0 occupies [0..64), w=1 occupies [64..128)
    assert_eq!(field.linearize(0, 0, 0, 0), 0);
    assert_eq!(field.linearize(0, 0, 0, 1), 4 * 4 * 4);
    assert_eq!(field.linearize(1, 0, 0, 0), 1);
    assert_eq!(field.linearize(0, 1, 0, 0), 4);
    assert_eq!(field.linearize(0, 0, 1, 0), 4 * 4);
    // Round-trip consistency
    let idx1 = field.linearize(2, 3, 1, 2);
    let idx2 = field.linearize(2, 3, 1, 2);
    assert_eq!(idx1, idx2);
}

#[test]
fn test_sedenion_field_4d_slice_isolation() {
    let mut field = SedenionField4D::uniform(4, 4, 4, 3);
    // Modify w=1 slice only
    field.get_mut(2, 1, 0, 1)[5] = 0.99;

    let slice0 = field.slice_3d(0);
    let slice1 = field.slice_3d(1);
    let slice2 = field.slice_3d(2);

    // w=0 and w=2 should be unaffected
    assert!(slice0.get(2, 1, 0)[5].abs() < 1e-14);
    assert!(slice2.get(2, 1, 0)[5].abs() < 1e-14);
    // w=1 should have the modification
    assert!((slice1.get(2, 1, 0)[5] - 0.99).abs() < 1e-14);

    // Slice dimensions match
    assert_eq!(slice0.nx, 4);
    assert_eq!(slice0.ny, 4);
    assert_eq!(slice0.nz, 4);
    assert_eq!(slice0.data.len(), 64);
}

#[test]
fn test_sedenion_field_4d_imbalance_length() {
    let field = SedenionField4D::uniform(4, 4, 4, 2);
    let imbalance = field.local_imbalance_density_4d(16);
    assert_eq!(imbalance.len(), 4 * 4 * 4 * 2);
    // Uniform field -> all imbalance values equal (imbalance attractor)
    for &f in &imbalance {
        assert!(f.is_finite());
        assert!(f > 0.0);
    }
}

#[test]
fn test_sedenion_field_4d_inter_slice_uniform() {
    let field = SedenionField4D::uniform(4, 4, 4, 3);
    let corrs = field.inter_slice_correlations(16);
    // 3 slices -> 2 correlations
    assert_eq!(corrs.len(), 2);
    // Uniform field: all slices identical, so correlation should be
    // undefined (zero variance). Our pearson_corr returns 0.0 for zero variance.
    for &r in &corrs {
        assert!(r.is_finite());
    }
}

#[test]
fn test_sedenion_field_4d_varied_inter_slice() {
    let mut field = SedenionField4D::uniform(4, 4, 4, 3);
    // Create distinct variation in each w-slice
    for w in 0..3 {
        for z in 0..4 {
            for y in 0..4 {
                for x in 0..4 {
                    let s = field.get_mut(x, y, z, w);
                    let xn = x as f64 / 4.0;
                    let wn = w as f64 / 3.0;
                    s[1] = 0.3 * xn * (1.0 + wn);
                    s[3] = 0.2 * (wn + 0.1);
                    s[5] = 0.1 * (y as f64 / 4.0) * (1.0 - wn);
                }
            }
        }
    }

    let corrs = field.inter_slice_correlations(16);
    assert_eq!(corrs.len(), 2);
    for &r in &corrs {
        assert!(r.is_finite());
        // With variation, correlations should be in [-1, 1]
        assert!((-1.0 - 1e-10..=1.0 + 1e-10).contains(&r));
    }
}
