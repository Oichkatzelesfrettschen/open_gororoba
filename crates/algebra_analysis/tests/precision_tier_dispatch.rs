//! Deterministic checks for the x87/Kahan crossover heuristic used in the
//! claim lane. These tests keep the arithmetic and repo-specific size bounds
//! explicit without depending on benchmark noise.

use algebra_analysis::precision_policy::{
    JacobiBackend, JacobiDispatchInput, OBSTRUCTION_SPECTRUM_REFERENCE_MAX_ORDER,
    choose_jacobi_backend,
};

#[test]
fn x87_kahan_crossover_is_2048_terms() {
    let f64_unit_roundoff = 2f64.powi(-53);
    let x87_unit_roundoff = 2f64.powi(-64);
    let crossover = (f64_unit_roundoff / x87_unit_roundoff).round() as usize;

    assert_eq!(crossover, 2048);
}

#[test]
fn sedenion_reduction_sizes_stay_below_crossover() {
    let crossover = 2048usize;
    let repo_dims = [16usize, 32, 64, 128, 256, 512, 1024];

    assert!(repo_dims.into_iter().all(|n| n < crossover));
}

#[test]
fn square_grid_sums_cross_the_threshold_quickly() {
    let crossover = 2048usize;
    let first_grid = (1usize..=128)
        .find(|&n_grid| n_grid * n_grid >= crossover)
        .expect("expected a modest square grid to exceed the crossover");

    assert_eq!(first_grid, 46);
    assert!(first_grid * first_grid > 1024);
}

#[test]
fn default_obstruction_policy_is_deterministic() {
    let decision = choose_jacobi_backend(JacobiDispatchInput::obstruction_spectrum(
        OBSTRUCTION_SPECTRUM_REFERENCE_MAX_ORDER,
    ));

    assert_eq!(decision.backend, JacobiBackend::ReferenceF64);
}

#[test]
fn obstruction_policy_escalates_past_offline_threshold() {
    let decision = choose_jacobi_backend(JacobiDispatchInput::obstruction_spectrum(
        OBSTRUCTION_SPECTRUM_REFERENCE_MAX_ORDER + 1,
    ));

    let expected = if cfg!(target_arch = "x86_64") {
        JacobiBackend::X87
    } else {
        JacobiBackend::DoubleDouble
    };

    assert_eq!(decision.backend, expected);
}

#[test]
fn portable_policy_forces_double_double_lane() {
    let decision = choose_jacobi_backend(JacobiDispatchInput::portable_high_precision(64));

    assert_eq!(decision.backend, JacobiBackend::DoubleDouble);
}

#[test]
fn reference_cross_check_policy_selects_f64_oracle() {
    let decision = choose_jacobi_backend(JacobiDispatchInput::reference_cross_check(16));

    assert_eq!(decision.backend, JacobiBackend::ReferenceF64);
}
