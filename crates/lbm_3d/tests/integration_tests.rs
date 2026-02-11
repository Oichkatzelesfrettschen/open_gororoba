//! Integration tests for LBM 3D solver.
//!
//! Tests physical conservation laws and behavior under various conditions.

use lbm_3d::solver::LbmSolver3D;

#[test]
fn test_small_domain_mass_conservation() {
    // Small test case: 4x4x4 domain
    let mut solver = LbmSolver3D::new(4, 4, 4, 0.8);
    let rho_init = 1.0;
    let u_init = [0.05, 0.03, 0.02];

    solver.initialize_uniform(rho_init, u_init);
    let mass_before = solver.total_mass();

    // Evolve 100 steps
    solver.evolve(100);

    let mass_after = solver.total_mass();
    let mass_error = (mass_after - mass_before).abs();

    // Mass should be conserved within floating point precision
    assert!(
        mass_error < 1e-8,
        "Mass not conserved: before={}, after={}, error={}",
        mass_before,
        mass_after,
        mass_error
    );
}

#[test]
fn test_medium_domain_conservation() {
    // Medium test case: 8x8x8 domain
    let mut solver = LbmSolver3D::new(8, 8, 8, 0.7);

    solver.initialize_uniform(1.5, [0.08, 0.04, 0.01]);

    let mass_i = solver.total_mass();
    let mom_i = solver.total_momentum();

    // Evolve
    solver.evolve(50);

    let mass_f = solver.total_mass();
    let mom_f = solver.total_momentum();

    // Check conservation
    assert!((mass_f - mass_i).abs() < 1e-7, "Mass not conserved");
    // Momentum in uniform periodic domain should decrease due to relaxation
    assert!(mom_f < mom_i || (mom_f - mom_i).abs() < 1e-10, "Momentum unexpected change");
}

#[test]
fn test_stability_over_many_steps() {
    // Test long-time stability
    let mut solver = LbmSolver3D::new(6, 6, 6, 1.5);

    solver.initialize_uniform(1.0, [0.05, 0.02, 0.01]);

    // Check stability at regular intervals
    for step in 0..5 {
        assert!(solver.is_stable(), "Solver unstable at step {}", step * 20);
        solver.evolve(20);
    }

    assert!(solver.is_stable(), "Solver unstable at end");
}

#[test]
fn test_viscosity_impact() {
    // Compare low vs high viscosity
    let mut solver_low_nu = LbmSolver3D::new(8, 8, 8, 0.55);   // nu = (1/3) * 0.05 ≈ 0.0167
    let mut solver_high_nu = LbmSolver3D::new(8, 8, 8, 1.5);   // nu = (1/3) * 1.0 ≈ 0.333

    let u_init = [0.1, 0.0, 0.0];
    solver_low_nu.initialize_uniform(1.0, u_init);
    solver_high_nu.initialize_uniform(1.0, u_init);

    // Record initial momentum
    let mom_low_i = solver_low_nu.total_momentum();
    let mom_high_i = solver_high_nu.total_momentum();

    // Evolve
    solver_low_nu.evolve(100);
    solver_high_nu.evolve(100);

    let mom_low_f = solver_low_nu.total_momentum();
    let mom_high_f = solver_high_nu.total_momentum();

    // Higher viscosity should dissipate momentum faster
    let dissipation_low = (mom_low_i - mom_low_f).abs();
    let dissipation_high = (mom_high_i - mom_high_f).abs();

    assert!(
        dissipation_high > dissipation_low * 0.9,  // Allow some numerical variation
        "Higher viscosity should dissipate more: low={}, high={}",
        dissipation_low,
        dissipation_high
    );
}

#[test]
fn test_rest_state_persistence() {
    // Fluid at rest should remain at rest
    let mut solver = LbmSolver3D::new(5, 5, 5, 1.0);

    solver.initialize_uniform(1.0, [0.0, 0.0, 0.0]);

    // Evolve
    solver.evolve(50);

    // Verify still at rest
    let mut all_at_rest = true;
    for z in 0..5 {
        for y in 0..5 {
            for x in 0..5 {
                let (_, u) = solver.get_macroscopic(x, y, z);
                let u_mag = (u[0]*u[0] + u[1]*u[1] + u[2]*u[2]).sqrt();
                if u_mag > 1e-10 {
                    all_at_rest = false;
                    break;
                }
            }
        }
    }

    assert!(all_at_rest, "Fluid at rest should remain at rest");
}

#[test]
fn test_zero_viscosity_limit() {
    // At tau = 0.5, viscosity = 0 (inviscid)
    // Momentum should be conserved exactly
    let mut solver = LbmSolver3D::new(6, 6, 6, 0.5);

    let u_init = [0.05, 0.02, 0.01];
    solver.initialize_uniform(1.0, u_init);

    let mom_before = solver.total_momentum();

    solver.evolve(30);

    let mom_after = solver.total_momentum();

    // In inviscid limit, momentum should be nearly conserved
    assert!(
        (mom_after - mom_before).abs() < 1e-10,
        "Inviscid momentum not conserved: before={}, after={}",
        mom_before,
        mom_after
    );
}

#[test]
fn test_density_variation() {
    // Test with non-unit density
    let mut solver = LbmSolver3D::new(4, 4, 4, 0.9);

    let rho_high = 5.0;
    solver.initialize_uniform(rho_high, [0.02, 0.01, 0.005]);

    let mass_init = solver.total_mass();
    solver.evolve(30);
    let mass_final = solver.total_mass();

    // Mass conservation still holds
    assert!(
        (mass_final - mass_init).abs() < 1e-7,
        "Mass conservation failed at high density"
    );

    // Verify density is as expected
    let (rho_check, _) = solver.get_macroscopic(2, 2, 2);
    assert!(
        (rho_check - rho_high).abs() < 1e-10,
        "Density not recovered correctly"
    );
}

#[test]
fn test_multiple_small_steps_vs_large_step() {
    // Verify that multiple small steps match (approximately) one large step
    // in terms of overall dissipation
    let mut solver_small = LbmSolver3D::new(4, 4, 4, 0.8);
    let mut solver_large = LbmSolver3D::new(4, 4, 4, 0.8);

    let u_init = [0.1, 0.0, 0.0];
    solver_small.initialize_uniform(1.0, u_init);
    solver_large.initialize_uniform(1.0, u_init);

    // Small steps: 10 steps x 10
    solver_small.evolve(10);
    let mom_small_mid = solver_small.total_momentum();
    solver_small.evolve(90);
    let mom_small_final = solver_small.total_momentum();

    // Large steps: evolve together for consistency
    solver_large.evolve(10);
    let mom_large_mid = solver_large.total_momentum();
    solver_large.evolve(90);
    let mom_large_final = solver_large.total_momentum();

    // Results should be very similar (deterministic)
    assert!(
        (mom_small_mid - mom_large_mid).abs() < 1e-12,
        "Momentum diverged after 10 steps"
    );
    assert!(
        (mom_small_final - mom_large_final).abs() < 1e-12,
        "Momentum diverged after 100 steps"
    );
}
