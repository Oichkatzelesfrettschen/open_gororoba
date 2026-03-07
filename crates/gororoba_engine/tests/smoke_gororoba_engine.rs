use gororoba_engine::GororobaEngine;

#[test]
fn smoke_engine_end_to_end_run_is_nonempty() {
    let engine = GororobaEngine::default();
    let (state, report) = engine.run(64);

    assert_eq!(state.words.len(), 64);
    assert_eq!(state.signs.len(), 64);
    assert_eq!(state.imbalance.len(), 64);
    assert_eq!(state.viscosity.len(), 64);
    assert!(!report.messages.is_empty());
}
