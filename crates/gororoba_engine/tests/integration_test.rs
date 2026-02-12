use gororoba_engine::GororobaEngine;

#[test]
fn test_engine_end_to_end_execution() {
    let engine = GororobaEngine::default();
    let (state, report) = engine.run(256);
    assert_eq!(state.words.len(), 256);
    assert_eq!(state.signs.len(), 256);
    assert_eq!(state.frustration.len(), 256);
    assert_eq!(state.viscosity.len(), 256);
    assert!(!report.messages.is_empty());
}
