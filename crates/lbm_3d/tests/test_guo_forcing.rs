//! Tests for Guo body force implementation in LBM solver.
//!
//! Validates the Guo et al. (2002) forcing scheme:
//! - API correctness (set/clear/has_forcing)
//! - Physical conservation laws (momentum injection)
//! - Numerical stability
//! - Equivalence to no-forcing case when F=0

use lbm_3d::solver::LbmSolver3D;

#[test]
fn test_force_field_api_set_and_clear() {
    let mut solver = LbmSolver3D::new(8, 8, 8, 0.6);

    // Initially no forcing
    assert!(!solver.has_forcing());

    // Set uniform force
    let force = vec![[0.0, 0.0, -0.001]; 8 * 8 * 8];
    solver.set_force_field(force.clone()).unwrap();
    assert!(solver.has_forcing());

    // Clear forcing
    solver.clear_force_field();
    assert!(!solver.has_forcing());

    // Re-set forcing
    solver.set_force_field(force).unwrap();
    assert!(solver.has_forcing());
}

#[test]
fn test_force_field_length_validation() {
    let mut solver = LbmSolver3D::new(8, 8, 8, 0.6);

    // Wrong length: too short
    let force_short = vec![[0.0, 0.0, -0.001]; 8 * 8 * 7];
    let result = solver.set_force_field(force_short);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("length mismatch"));

    // Wrong length: too long
    let force_long = vec![[0.0, 0.0, -0.001]; 8 * 8 * 9];
    let result = solver.set_force_field(force_long);
    assert!(result.is_err());

    // Correct length
    let force_correct = vec![[0.0, 0.0, -0.001]; 8 * 8 * 8];
    let result = solver.set_force_field(force_correct);
    assert!(result.is_ok());
}

#[test]
fn test_force_field_finite_validation() {
    let mut solver = LbmSolver3D::new(4, 4, 4, 0.6);

    // NaN force
    let mut force_nan = vec![[0.0, 0.0, -0.001]; 4 * 4 * 4];
    force_nan[10][0] = f64::NAN;
    let result = solver.set_force_field(force_nan);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Non-finite"));

    // Inf force
    let mut force_inf = vec![[0.0, 0.0, -0.001]; 4 * 4 * 4];
    force_inf[10][1] = f64::INFINITY;
    let result = solver.set_force_field(force_inf);
    assert!(result.is_err());
}

#[test]
fn test_uniform_gravity_momentum_injection() {
    let mut solver = LbmSolver3D::new(8, 8, 8, 0.6);

    // Initialize stationary fluid
    solver.initialize_uniform(1.0, [0.0, 0.0, 0.0]);

    // Apply uniform gravity in -z direction
    let g = -0.001;
    let force = vec![[0.0, 0.0, g]; 8 * 8 * 8];
    solver.set_force_field(force).unwrap();

    // Evolve for 100 steps
    solver.evolve(100);

    // Check that fluid has accelerated in -z direction
    let mut total_uz = 0.0;
    for z in 0..8 {
        for y in 0..8 {
            for x in 0..8 {
                let idx = z * 64 + y * 8 + x;
                total_uz += solver.u[idx][2];
            }
        }
    }
    let avg_uz = total_uz / (8.0 * 8.0 * 8.0);

    // After 100 steps with gravity g=-0.001, expect negative velocity
    assert!(avg_uz < -1e-4, "Expected negative z-velocity, got {}", avg_uz);

    // Check mass conservation
    let total_mass: f64 = solver.rho.iter().sum();
    let expected_mass = 8.0 * 8.0 * 8.0 * 1.0;
    assert!(
        (total_mass - expected_mass).abs() < 1e-10,
        "Mass not conserved: expected {}, got {}",
        expected_mass,
        total_mass
    );
}

#[test]
fn test_gradient_force_shear_flow() {
    let mut solver = LbmSolver3D::new(8, 8, 8, 0.6);

    // Initialize stationary fluid
    solver.initialize_uniform(1.0, [0.0, 0.0, 0.0]);

    // Apply force gradient: F_x increases with z
    let mut force = vec![[0.0, 0.0, 0.0]; 8 * 8 * 8];
    for z in 0..8 {
        for y in 0..8 {
            for x in 0..8 {
                let idx = z * 64 + y * 8 + x;
                let f_x = 0.0005 * (z as f64) / 8.0; // Linearly increasing force
                force[idx] = [f_x, 0.0, 0.0];
            }
        }
    }
    solver.set_force_field(force).unwrap();

    // Evolve
    solver.evolve(100);

    // Check that u_x increases with z
    let u_x_bottom = solver.u[4 * 8 + 4][0]; // z=0, center
    let u_x_top = solver.u[7 * 64 + 4 * 8 + 4][0];    // z=7, center

    assert!(
        u_x_top > u_x_bottom,
        "Expected shear: u_x(z=7) > u_x(z=0), got {} vs {}",
        u_x_top,
        u_x_bottom
    );
}

#[test]
fn test_zero_force_equivalence() {
    // Two solvers: one with F=0 forcing, one without forcing
    let mut solver_no_force = LbmSolver3D::new(8, 8, 8, 0.6);
    let mut solver_zero_force = LbmSolver3D::new(8, 8, 8, 0.6);

    // Initialize both identically
    solver_no_force.initialize_uniform(1.0, [0.01, 0.0, 0.0]);
    solver_zero_force.initialize_uniform(1.0, [0.01, 0.0, 0.0]);

    // Apply zero force to second solver
    let force_zero = vec![[0.0, 0.0, 0.0]; 8 * 8 * 8];
    solver_zero_force.set_force_field(force_zero).unwrap();

    // Evolve both
    solver_no_force.evolve(50);
    solver_zero_force.evolve(50);

    // Results should be identical (within floating-point tolerance)
    for idx in 0..(8 * 8 * 8) {
        let rho_diff = (solver_no_force.rho[idx] - solver_zero_force.rho[idx]).abs();
        assert!(
            rho_diff < 1e-12,
            "Density mismatch at idx {}: {} vs {}",
            idx,
            solver_no_force.rho[idx],
            solver_zero_force.rho[idx]
        );

        for k in 0..3 {
            let u_diff = (solver_no_force.u[idx][k] - solver_zero_force.u[idx][k]).abs();
            assert!(
                u_diff < 1e-12,
                "Velocity mismatch at idx {}, component {}: {} vs {}",
                idx,
                k,
                solver_no_force.u[idx][k],
                solver_zero_force.u[idx][k]
            );
        }
    }
}

#[test]
fn test_forcing_numerical_stability() {
    let mut solver = LbmSolver3D::new(16, 16, 16, 0.6);

    // Initialize with small velocity
    solver.initialize_uniform(1.0, [0.01, 0.0, 0.0]);

    // Apply moderate force
    let force = vec![[0.002, 0.001, -0.001]; 16 * 16 * 16];
    solver.set_force_field(force).unwrap();

    // Evolve for many steps
    solver.evolve(500);

    // Check stability: all f_i should be non-negative
    assert!(solver.is_stable(), "Solver became unstable with forcing");

    // Check all velocities are finite
    for idx in 0..(16 * 16 * 16) {
        for k in 0..3 {
            assert!(
                solver.u[idx][k].is_finite(),
                "Non-finite velocity at idx {}, component {}: {}",
                idx,
                k,
                solver.u[idx][k]
            );
        }
    }

    // Check densities remain positive
    for idx in 0..(16 * 16 * 16) {
        assert!(
            solver.rho[idx] > 0.0,
            "Non-positive density at idx {}: {}",
            idx,
            solver.rho[idx]
        );
    }
}

#[test]
fn test_momentum_conservation_with_periodic_bc() {
    // In fully periodic domain, external force should inject momentum
    // Total momentum should grow linearly with time
    let mut solver = LbmSolver3D::new(8, 8, 8, 0.6);

    // Initialize stationary
    solver.initialize_uniform(1.0, [0.0, 0.0, 0.0]);

    // Apply uniform force
    let force = vec![[0.001, 0.0, 0.0]; 8 * 8 * 8];
    solver.set_force_field(force).unwrap();

    // Measure initial momentum
    let p0 = solver.total_momentum();
    assert!(p0.abs() < 1e-10, "Initial momentum should be zero");

    // Evolve 50 steps
    solver.evolve(50);
    let p1 = solver.total_momentum();

    // Evolve another 50 steps
    solver.evolve(50);
    let p2 = solver.total_momentum();

    // Momentum should be increasing
    assert!(
        p2 > p1 && p1 > p0,
        "Momentum should grow with forcing: p0={}, p1={}, p2={}",
        p0,
        p1,
        p2
    );
}

#[test]
fn test_forcing_with_spatial_viscosity() {
    // Test that forcing works correctly with spatially-varying viscosity
    let mut solver = LbmSolver3D::new(8, 8, 8, 0.6);

    // Set spatially-varying viscosity: high viscosity at z=0, low at z=7
    let mut tau_field = vec![0.6; 8 * 8 * 8];
    for z in 0..8 {
        for y in 0..8 {
            for x in 0..8 {
                let idx = z * 64 + y * 8 + x;
                // tau varies from 0.6 (z=0) to 1.2 (z=7)
                tau_field[idx] = 0.6 + 0.6 * (z as f64) / 7.0;
            }
        }
    }
    solver.set_viscosity_field(tau_field).unwrap();

    // Initialize uniform
    solver.initialize_uniform(1.0, [0.0, 0.0, 0.0]);

    // Apply uniform force in x-direction
    let force = vec![[0.001, 0.0, 0.0]; 8 * 8 * 8];
    solver.set_force_field(force).unwrap();

    // Evolve
    solver.evolve(100);

    // Low-viscosity region (z=7) should develop higher velocity
    let u_x_low_visc = solver.u[7 * 64 + 4 * 8 + 4][0]; // z=7, center
    let u_x_high_visc = solver.u[4 * 8 + 4][0]; // z=0, center

    assert!(
        u_x_low_visc > u_x_high_visc,
        "Low-viscosity region should accelerate more: u_x(z=7)={} vs u_x(z=0)={}",
        u_x_low_visc,
        u_x_high_visc
    );
}
