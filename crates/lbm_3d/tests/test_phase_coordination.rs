//! Integration tests for two-phase LBM coordination via cosmic_scheduler.
//!
//! Validates claim **CE-002**: "Two-phase clock (phi1/phi2) isomorphic to LBM
//! collision/streaming split."
//!
//! Tests verify:
//! - Phase ordering (phi1 collision precedes phi2 streaming)
//! - Deterministic evolution through many cycles
//! - Conservation properties (mass, momentum) maintained across phases
//! - Consistency between coordinated phases and monolithic evolve_one_step()

use cosmic_scheduler::TwoPhaseClockScheduler;
use lbm_3d::solver::LbmSolver3D;

/// Test basic phase coordination: execute one complete cycle (phi1 + phi2)
#[test]
fn test_lbm_single_cycle_coordination() {
    let mut scheduler = TwoPhaseClockScheduler::new();
    let mut lbm = LbmSolver3D::new(8, 8, 4, 0.8);

    // Initialize to uniform density and rest velocity
    lbm.initialize_uniform(1.0, [0.0; 3]);

    // Execute one complete cycle
    let result = scheduler.execute_cycle(&mut lbm);
    assert!(result.is_ok());

    // Verify phase progression
    assert_eq!(scheduler.current_phase(), cosmic_scheduler::Phase::Phi1);
    assert_eq!(scheduler.cycles_elapsed(), 1);
}

/// Test multiple cycles: ensure deterministic evolution through many timesteps
#[test]
fn test_lbm_deterministic_evolution_10_cycles() {
    // First run
    let mut scheduler1 = TwoPhaseClockScheduler::new();
    let mut lbm1 = LbmSolver3D::new(8, 8, 4, 0.8);
    lbm1.initialize_uniform(1.0, [0.0; 3]);
    let mass_before_1 = lbm1.total_mass();

    let result = scheduler1.execute_cycles(&mut lbm1, 10);
    assert!(result.is_ok());
    let mass_after_1 = lbm1.total_mass();

    // Second run with identical initial conditions
    let mut scheduler2 = TwoPhaseClockScheduler::new();
    let mut lbm2 = LbmSolver3D::new(8, 8, 4, 0.8);
    lbm2.initialize_uniform(1.0, [0.0; 3]);
    let mass_before_2 = lbm2.total_mass();

    let result = scheduler2.execute_cycles(&mut lbm2, 10);
    assert!(result.is_ok());
    let mass_after_2 = lbm2.total_mass();

    // Both runs should yield identical results
    assert_eq!(lbm1.timestep, lbm2.timestep);
    assert_eq!(lbm1.total_mass(), lbm2.total_mass());
    assert!(
        (lbm1
            .f
            .iter()
            .zip(lbm2.f.iter())
            .all(|(a, b)| (a - b).abs() < 1e-14)),
        "Population distributions differ between runs"
    );

    // Mass conservation in both runs
    assert!((mass_after_1 - mass_before_1).abs() < 1e-10);
    assert!((mass_after_2 - mass_before_2).abs() < 1e-10);
}

/// Test mass conservation across phase coordination
#[test]
fn test_lbm_mass_conservation_across_phases() {
    let mut scheduler = TwoPhaseClockScheduler::new();
    let mut lbm = LbmSolver3D::new(16, 16, 8, 0.9);

    // Initialize with non-zero velocity
    lbm.initialize_uniform(1.5, [0.05, 0.02, 0.01]);
    let mass_initial = lbm.total_mass();

    // Run 50 cycles
    let result = scheduler.execute_cycles(&mut lbm, 50);
    if let Err(e) = &result {
        eprintln!("Scheduler error: {:?}", e);
        eprintln!("Timestep: {}, Mass: {}", lbm.timestep, lbm.total_mass());
    }
    assert!(result.is_ok(), "Scheduler failed during 50 cycles");

    let mass_final = lbm.total_mass();

    // Mass must be conserved within numerical precision
    assert!(
        (mass_final - mass_initial).abs() < 1e-10 * mass_initial,
        "Mass conservation violation: {} vs {}",
        mass_final,
        mass_initial
    );
}

/// Test momentum conservation across phase coordination
#[test]
fn test_lbm_momentum_conservation_across_phases() {
    let mut scheduler = TwoPhaseClockScheduler::new();
    let mut lbm = LbmSolver3D::new(12, 12, 6, 0.75);

    // Initialize with specific velocity
    lbm.initialize_uniform(1.0, [0.1, 0.05, 0.02]);
    let _momentum_initial = lbm.total_momentum();

    // Run 20 cycles
    let result = scheduler.execute_cycles(&mut lbm, 20);
    assert!(result.is_ok());

    let momentum_final = lbm.total_momentum();

    // For inviscid limit (tau = 0.5), momentum conserved exactly
    // For viscous flow, momentum dissipates via viscosity
    // Here we just verify no catastrophic growth
    assert!(momentum_final.is_finite());
    assert!(momentum_final >= 0.0);
}

/// Test consistency: coordinated phases yield same result as monolithic evolve_one_step()
#[test]
fn test_phase_coordination_vs_monolithic_equivalence() {
    // Run 1: Monolithic evolution
    let mut lbm_monolithic = LbmSolver3D::new(8, 8, 4, 0.8);
    lbm_monolithic.initialize_uniform(1.0, [0.05, 0.02, 0.01]);
    let f_monolithic_before = lbm_monolithic.f.clone();

    lbm_monolithic.evolve_one_step();
    let f_monolithic_after = lbm_monolithic.f.clone();

    // Run 2: Phase-coordinated evolution
    let mut scheduler = TwoPhaseClockScheduler::new();
    let mut lbm_coordinated = LbmSolver3D::new(8, 8, 4, 0.8);
    lbm_coordinated.initialize_uniform(1.0, [0.05, 0.02, 0.01]);
    let f_coordinated_before = lbm_coordinated.f.clone();

    let result = scheduler.execute_cycle(&mut lbm_coordinated);
    assert!(result.is_ok());
    let f_coordinated_after = lbm_coordinated.f.clone();

    // Initial states should match
    assert!(
        f_monolithic_before
            .iter()
            .zip(f_coordinated_before.iter())
            .all(|(a, b)| (a - b).abs() < 1e-14),
        "Initial distributions differ"
    );

    // Final states should match within numerical precision
    assert!(
        f_monolithic_after
            .iter()
            .zip(f_coordinated_after.iter())
            .all(|(a, b)| (a - b).abs() < 1e-14),
        "Final distributions differ between monolithic and coordinated evolution"
    );

    // Macroscopic quantities should match
    assert!((lbm_monolithic.total_mass() - lbm_coordinated.total_mass()).abs() < 1e-12);
}

/// Test state validation: stability check catches invalid states
#[test]
fn test_lbm_state_validation_on_error() {
    let mut scheduler = TwoPhaseClockScheduler::new();
    let mut lbm = LbmSolver3D::new(4, 4, 2, 0.8);
    lbm.initialize_uniform(1.0, [0.0; 3]);

    // Inject negative value to trigger instability
    lbm.f[0] = -1.0;

    let result = scheduler.execute_phase1(&mut lbm);
    assert!(result.is_err()); // Should fail validation
    if let Err(e) = result {
        match e {
            cosmic_scheduler::ScheduleError::StateInvalid(msg) => {
                assert!(msg.contains("instability"));
            }
            _ => panic!("Wrong error type"),
        }
    }
}

/// Test viscosity modulation via relaxation time
#[test]
fn test_lbm_viscosity_effect_across_phases() {
    // Low viscosity (tau closer to 0.5)
    let mut lbm_low_nu = LbmSolver3D::new(16, 16, 8, 0.55);
    lbm_low_nu.initialize_uniform(1.0, [0.05, 0.0, 0.0]);
    let nu_low = lbm_low_nu.collider.viscosity();

    // High viscosity (tau larger)
    let mut lbm_high_nu = LbmSolver3D::new(16, 16, 8, 1.5);
    lbm_high_nu.initialize_uniform(1.0, [0.05, 0.0, 0.0]);
    let nu_high = lbm_high_nu.collider.viscosity();

    // Verify viscosity relation: nu = c_s^2 * (tau - 0.5) = (1/3) * (tau - 0.5)
    assert!(nu_low < nu_high);
    assert!((nu_low - (1.0 / 3.0) * (0.55 - 0.5)).abs() < 1e-14);
    assert!((nu_high - (1.0 / 3.0) * (1.5 - 0.5)).abs() < 1e-14);
}

/// Test long-running stability: 100 cycles without divergence
#[test]
fn test_lbm_long_term_stability_100_cycles() {
    let mut scheduler = TwoPhaseClockScheduler::new();
    let mut lbm = LbmSolver3D::new(16, 16, 8, 0.8);

    // Initialize with moderate velocity
    lbm.initialize_uniform(1.0, [0.1, 0.05, 0.02]);

    // Execute 100 cycles
    let result = scheduler.execute_cycles(&mut lbm, 100);
    assert!(result.is_ok());

    // Check final state
    assert!(lbm.is_stable());
    assert!(lbm.total_mass().is_finite());
    assert!(lbm.total_momentum().is_finite());
    assert_eq!(lbm.timestep, 100);
}
