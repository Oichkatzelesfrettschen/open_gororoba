use sign_imbalance::{ImbalanceStarConfig, ScalarImbalanceMap, evaluate_imbalance_star};

#[test]
fn test_scalar_imbalance_bridge_integration() {
    let map = ScalarImbalanceMap::new(1.0);
    let cfg = ImbalanceStarConfig {
        run_tov: false,
        ..ImbalanceStarConfig::default()
    };
    let imbalance = vec![0.3, 0.4, 0.5, 0.6];
    let result = evaluate_imbalance_star(&imbalance, map, &cfg);
    assert!(result.mean_phi > 0.0 && result.mean_phi <= 1.0);
    assert!(result.omega_eff > 0.0);
}
