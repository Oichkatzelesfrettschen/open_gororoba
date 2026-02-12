use neural_homotopy::{
    alignment_score, reference_hubble_curve, train_homotopy_surrogate, HomotopyTrainingConfig,
};

#[test]
fn test_training_trace_and_hubble_alignment() {
    let cfg = HomotopyTrainingConfig {
        epochs: 48,
        learning_rate: 0.04,
        plateau_tolerance: 1e-5,
    };
    let trace = train_homotopy_surrogate(cfg);
    assert_eq!(trace.losses.len(), 48);
    assert!((0.0..=1.0).contains(&trace.hubble_alignment));

    let hubble = reference_hubble_curve(48);
    let score = alignment_score(&trace.losses, &hubble);
    assert!((0.0..=1.0).contains(&score));
}
