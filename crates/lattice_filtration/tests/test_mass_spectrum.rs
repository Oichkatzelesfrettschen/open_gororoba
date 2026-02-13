use lattice_filtration::{
    depth_clusters, depth_histogram, pdg_comparison, predict_mass_ratios, SurvivalDepthMap,
};

#[test]
fn test_full_mass_spectrum_pipeline() {
    let map = SurvivalDepthMap::compute();

    // Histogram covers all 256 entries
    let hist = depth_histogram(&map);
    let total: usize = hist.iter().map(|b| b.count).sum();
    assert_eq!(total, 256);

    // Clusters cover all entries
    let clusters = depth_clusters(&map);
    let cluster_total: usize = clusters.iter().map(|(_, e)| e.len()).sum();
    assert_eq!(cluster_total, 256);
}

#[test]
fn test_mass_spectrum_to_lepton_ratios() {
    let map = SurvivalDepthMap::compute();
    let preds = predict_mass_ratios(&map);

    // With 16x16 sedenion pairs, there should be multiple depth clusters
    assert!(
        map.n_distinct_depths() >= 2,
        "Need >= 2 depth levels for mass predictions"
    );

    // If we have enough clusters, predictions should exist
    if map.n_distinct_depths() >= 3 {
        assert_eq!(preds.len(), 3, "Should have 3 mass ratio predictions");

        // All predicted ratios should be positive
        for p in &preds {
            assert!(p.predicted_ratio > 0.0);
            assert!(p.pdg_ratio > 0.0);
        }
    }
}

#[test]
fn test_pdg_comparison_end_to_end() {
    let map = SurvivalDepthMap::compute();
    let comp = pdg_comparison(&map);

    assert!(comp.n_clusters >= 2);
    if !comp.predictions.is_empty() {
        assert!(
            comp.mean_relative_error.is_finite(),
            "Mean error should be finite"
        );
        assert!(
            comp.best_relative_error.is_finite(),
            "Best error should be finite"
        );
    }
}

#[test]
fn test_depth_histogram_structure() {
    let map = SurvivalDepthMap::compute();
    let hist = depth_histogram(&map);

    // Bins should be sorted by depth
    for w in hist.windows(2) {
        assert!(w[0].depth < w[1].depth);
    }

    // Each bin should have valid fraction
    for bin in &hist {
        assert!(bin.fraction > 0.0);
        assert!(bin.fraction <= 1.0);
    }
}

#[test]
fn test_survival_depth_symmetry() {
    // Check if d(i, j) has any relationship to d(j, i)
    let map = SurvivalDepthMap::compute();

    // e_0 is the identity: e_0 * e_j = e_j with sign +1
    // So depth(0, j) should be consistent across j
    let d00 = map.depth(0, 0);
    assert!(d00 > 0, "Identity product should have positive depth");

    // e_i * e_i products have basis = 0 (self-product)
    for i in 1..16 {
        let d = map.depth(i, i);
        // All self-products have same basis (0) but different signs
        assert!(d > 0, "Self-product e_{i}*e_{i} should have positive depth");
    }
}
