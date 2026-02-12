use vacuum_frustration::{evaluate_frustration_star, FrustrationStarConfig, ScalarFrustrationMap};

#[test]
fn test_scalar_frustration_bridge_integration() {
    let map = ScalarFrustrationMap::new(1.0);
    let cfg = FrustrationStarConfig {
        run_tov: false,
        ..FrustrationStarConfig::default()
    };
    let frustration = vec![0.3, 0.4, 0.5, 0.6];
    let result = evaluate_frustration_star(&frustration, map, &cfg);
    assert!(result.mean_phi > 0.0 && result.mean_phi <= 1.0);
    assert!(result.omega_eff > 0.0);
}
