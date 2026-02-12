use lattice_filtration::{simulate_fibonacci_collision_storm, LatencyLaw};

#[test]
fn test_collision_storm_pipeline_runs() {
    let (stats, observations) = simulate_fibonacci_collision_storm(256, 19);
    assert_eq!(observations.len(), 256);
    assert!(stats.peak_bucket_occupancy >= 1);
    assert!(stats.mean_latency > 0.0);
    assert!(matches!(
        stats.latency_law,
        LatencyLaw::InverseSquare
            | LatencyLaw::Linear
            | LatencyLaw::Uniform
            | LatencyLaw::Undetermined
    ));
}
