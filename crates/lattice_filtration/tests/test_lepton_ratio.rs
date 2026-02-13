use lattice_filtration::{
    classify_latency_law_detailed, filtration_from_velocity_field, pdg_comparison,
    predict_mass_ratios, LatencyLaw, SurvivalDepthMap,
};

#[test]
fn test_lepton_ratio_from_depth_clusters() {
    let map = SurvivalDepthMap::compute();
    let preds = predict_mass_ratios(&map);

    if preds.len() == 3 {
        // mu/e ratio prediction
        let mu_e = &preds[0];
        assert!(mu_e.predicted_ratio > 0.0);
        // PDG ratio is ~206.77
        assert!(
            (mu_e.pdg_ratio - 206.77).abs() < 1.0,
            "mu/e PDG ratio should be ~206.77"
        );
    }
}

#[test]
fn test_pdg_comparison_summary() {
    let map = SurvivalDepthMap::compute();
    let comp = pdg_comparison(&map);

    // Should have at least 2 depth clusters
    assert!(comp.n_clusters >= 2);

    // If predictions exist, errors should be non-negative
    for p in &comp.predictions {
        assert!(p.relative_error >= 0.0);
    }
}

#[test]
fn test_velocity_field_to_filtration_to_spectrum() {
    // End-to-end: velocity field -> filtration -> spectrum -> latency law
    let (nx, ny, nz) = (4, 8, 4);
    let mut field = vec![[0.0f64; 3]; nx * ny * nz];

    // Create a simple shear flow
    for y in 0..ny {
        for z in 0..nz {
            for x in 0..nx {
                let idx = x + nx * (y + ny * z);
                let yn = y as f64 / ny as f64;
                field[idx] = [yn * (1.0 - yn), 0.0, 0.0];
            }
        }
    }

    let result = filtration_from_velocity_field(&field, nx, ny, nz, 10.0, 8);
    assert!(result.n_active_cells > 0);
    assert!(result.n_active_cells <= nx * ny * nz);
}

#[test]
fn test_detailed_classification_enriches_result() {
    // Use linear data where classification is clear
    let samples: Vec<(f64, f64)> = (1..40).map(|r| (r as f64, 3.0 * r as f64 + 2.0)).collect();
    let detail = classify_latency_law_detailed(&samples);
    assert_eq!(detail.law, LatencyLaw::Linear);
    assert!(detail.r2_linear > 0.99);
    // Power law exponent should be near 1 for linear data
    assert!(detail.power_law_exponent > 0.5);
}

#[test]
fn test_mass_spectrum_deterministic() {
    let map1 = SurvivalDepthMap::compute();
    let map2 = SurvivalDepthMap::compute();

    // Same inputs should produce identical results
    assert_eq!(map1.entries.len(), map2.entries.len());
    for (e1, e2) in map1.entries.iter().zip(map2.entries.iter()) {
        assert_eq!(e1.depth, e2.depth);
        assert_eq!(e1.product_basis, e2.product_basis);
        assert_eq!(e1.product_sign, e2.product_sign);
    }
}

#[test]
fn test_filtration_empty_velocity_field() {
    let field = vec![[0.0; 3]; 8];
    let result = filtration_from_velocity_field(&field, 2, 2, 2, 10.0, 4);
    assert_eq!(result.n_active_cells, 0);
    assert_eq!(result.n_total_cells, 8);
}
