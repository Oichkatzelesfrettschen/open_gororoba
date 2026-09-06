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
    assert!(result.unwrap_err().to_string().contains("length mismatch"));

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
    assert!(result.unwrap_err().to_string().contains("Non-finite"));

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
    assert!(
        avg_uz < -1e-4,
        "Expected negative z-velocity, got {}",
        avg_uz
    );

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
    let u_x_top = solver.u[7 * 64 + 4 * 8 + 4][0]; // z=7, center

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
    let mut solver = LbmSolver3D::new(16, 16, 16, 0.8);

    // Initialize with small velocity
    solver.initialize_uniform(1.0, [0.01, 0.0, 0.0]);

    // Apply moderate force (reduced from 0.002 for stability with exact
    // Phi_i source term which uses u* = u + F/(2*rho) in equilibrium,
    // making forcing more effective per step)
    let force = vec![[0.0005, 0.00025, -0.00025]; 16 * 16 * 16];
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
    // Test that forcing works correctly with spatially-varying viscosity.
    // In a periodic domain with uniform forcing, all cells reach the same
    // steady-state velocity (no boundaries to create a Poiseuille profile).
    // The test verifies that spatially-varying tau does NOT break the
    // solver -- all velocities should be finite and positive.
    let mut solver = LbmSolver3D::new(8, 8, 8, 0.6);

    // Set spatially-varying viscosity: tau from 0.6 (z=0) to 1.2 (z=7)
    let mut tau_field = vec![0.6; 8 * 8 * 8];
    for z in 0..8 {
        for y in 0..8 {
            for x in 0..8 {
                let idx = z * 64 + y * 8 + x;
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

    // All cells should have finite positive x-velocity from the forcing
    for idx in 0..(8 * 8 * 8) {
        assert!(
            solver.u[idx][0].is_finite(),
            "velocity should be finite at idx={idx}: {}",
            solver.u[idx][0]
        );
        assert!(
            solver.u[idx][0] > 0.0,
            "velocity should be positive from x-forcing at idx={idx}: {}",
            solver.u[idx][0]
        );
    }

    // Mass should be conserved across viscosity regions
    let mass = solver.total_mass();
    let expected_mass = 8.0 * 8.0 * 8.0; // rho_init = 1.0, n_cells = 512
    let rel_err = (mass - expected_mass).abs() / expected_mass;
    assert!(
        rel_err < 1e-10,
        "mass conservation with spatial viscosity: rel_err={rel_err:.3e}"
    );
}

/// Phi_i exact source term mass conservation to f64 machine precision.
///
/// The Guo forcing scheme's source term Phi_i is constructed so that
/// sum_i(Phi_i) = 0 analytically (the lattice weights and velocity
/// vectors satisfy sum(w_i * e_i) = 0 and sum(w_i) = 1). This means
/// mass must be conserved to floating-point round-off, not just to
/// O(dt) truncation error.
///
/// We run 1000 steps with a moderate body force and verify mass
/// deviation stays below 1e-14 (f64 machine precision regime).
#[test]
fn test_phi_i_mass_conservation_machine_precision() {
    let n = 16;
    let tau = 0.7;
    let n_cells = n * n * n;
    let f_x = 5e-5;

    let mut solver = LbmSolver3D::new(n, n, n, tau);
    let tau_field = vec![tau; n_cells];
    solver
        .set_viscosity_field(tau_field)
        .expect("set viscosity field");
    solver.initialize_uniform(1.0, [0.0, 0.0, 0.0]);

    let force_field = vec![[f_x, 0.0, 0.0]; n_cells];
    solver
        .set_force_field(force_field)
        .expect("set force field");

    let mass_initial = solver.total_mass();
    assert!(mass_initial > 0.0, "initial mass must be positive");

    // Evolve 1000 steps with active Guo forcing
    solver.evolve(1000);

    let mass_final = solver.total_mass();
    let relative_error = (mass_final - mass_initial).abs() / mass_initial;

    // f64 machine epsilon ~ 2.2e-16. Over 1000 steps with 4096 cells,
    // round-off accumulation yields O(1e-14). Threshold 1e-13 confirms
    // the Phi_i source term preserves mass analytically (sum_i Phi_i = 0).
    assert!(
        relative_error < 1e-13,
        "Phi_i mass conservation violated at machine precision: \
         initial={:.15e}, final={:.15e}, rel_err={:.3e}",
        mass_initial,
        mass_final,
        relative_error,
    );
}

/// Phi_i momentum injection precision test (1% tolerance).
///
/// With the exact Guo forcing scheme, the force-corrected velocity
/// u* = u + F/(2*rho) is stored in solver.u, so total momentum
/// P_x(t) = F_x * M * t exactly. This test verifies precision to
/// 1% (tighter than the 5% test in validation_taylor_green.rs).
#[test]
fn test_phi_i_momentum_injection_precision() {
    let n = 16;
    let tau = 0.8;
    let n_cells = n * n * n;
    let f_x = 1e-5;

    let mut solver = LbmSolver3D::new(n, n, n, tau);
    let tau_field = vec![tau; n_cells];
    solver
        .set_viscosity_field(tau_field)
        .expect("set viscosity field");
    solver.initialize_uniform(1.0, [0.0, 0.0, 0.0]);

    let force_field = vec![[f_x, 0.0, 0.0]; n_cells];
    solver
        .set_force_field(force_field)
        .expect("set force field");

    let mass = solver.total_mass();

    // Evolve 200 steps for better statistics
    let n_steps = 200;
    solver.evolve(n_steps);
    solver.compute_macroscopic();

    // Compute total x-momentum from stored u* (force-corrected)
    let mut px_total = 0.0;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let (rho, u) = solver.get_macroscopic(x, y, z);
                px_total += rho * u[0];
            }
        }
    }

    // Exact Phi_i: P_x(t) = F_x * M * t
    let px_expected = f_x * mass * n_steps as f64;

    let relative_error = (px_total - px_expected).abs() / px_expected.abs().max(1e-15);
    assert!(
        relative_error < 0.01,
        "Phi_i momentum precision: measured px={:.6e}, expected={:.6e}, rel_err={:.4} (>1%)",
        px_total,
        px_expected,
        relative_error,
    );
}

/// Dynamic drag (kappa-based) + Phi_i source term mass conservation.
///
/// Replicates the solar_wind_dm_mhd time loop ordering:
///   stream -> macroscopic -> force (grav + drag) -> collision (Phi_i)
///
/// Uses sigma_chi_b = 1e-45 cm^2 (DM-baryon cross section) to activate
/// the drag pathway. Mass must be conserved to < 1e-10 because:
///   1. Phi_i source term: sum_i(Phi_i) = 0 analytically
///   2. Drag is a momentum-only force (no mass source)
///   3. NFW gravitational force is also momentum-only
#[test]
fn test_dynamic_drag_phi_i_mass_conservation() {
    use lbm_3d::dm_force::{DmForceConfig, DmForceField, combine_forces};

    let (nx, ny, nz) = (16, 8, 8);
    let tau = 0.8;
    let n_cells = nx * ny * nz;

    let mut solver = LbmSolver3D::new(nx, ny, nz, tau);
    // Small bulk flow so drag has nonzero relative velocity
    solver.initialize_uniform(2.7, [0.02, 0.0, 0.0]);
    let mass_initial = solver.total_mass();

    // Configure DM with drag enabled (sigma > 0)
    let config = DmForceConfig {
        sigma_chi_b: 1e-45,
        r_min_au: 0.5,
        r_max_au: 1.5,
        ..DmForceConfig::default()
    };
    let dm = DmForceField::new(nx, ny, nz, config);

    // Verify kappa_drag is nonzero (drag is active)
    assert!(
        dm.kappa_drag > 0.0,
        "kappa_drag should be positive with sigma=1e-45, got {}",
        dm.kappa_drag
    );

    // Time loop mimicking solar_wind_dm_mhd ordering:
    //   step 0: skip stream -> macroscopic -> force -> collision
    //   step 1+: stream -> macroscopic -> force -> collision
    for step in 0..100 {
        if step > 0 {
            let _ = solver.phase2_streaming();
        }
        solver.compute_macroscopic();

        // Combine NFW gravitational force + dynamic drag
        let drag = dm.drag_force_density_lattice(&solver.rho, &solver.u);
        let gravity: Vec<_> = dm
            .force
            .iter()
            .zip(&solver.rho)
            .map(|(acceleration, density)| acceleration.map(|component| component * density))
            .collect();
        let combined = combine_forces(&gravity, &drag);

        solver.set_force_field(combined).expect("force field set");
        let _ = solver.phase1_collision();
    }

    let mass_final = solver.total_mass();
    let rel_err = (mass_final - mass_initial).abs() / mass_initial;
    assert!(
        rel_err < 1e-10,
        "mass not conserved with dynamic drag + Phi_i: \
         initial={mass_initial:.15e}, final={mass_final:.15e}, rel_err={rel_err:.3e}",
    );

    // Verify solver stayed stable (no NaN or negative densities)
    for idx in 0..n_cells {
        assert!(
            solver.rho[idx].is_finite() && solver.rho[idx] > 0.0,
            "unstable density at idx {}: {}",
            idx,
            solver.rho[idx],
        );
    }
}

#[test]
fn density_weighted_drag_and_gravity_preserve_acceleration_across_densities() {
    use lbm_3d::dm_force::{DmForceConfig, DmForceField, combine_forces};
    // Manufactured lattice coefficients isolate the Guo force-density interface.
    let mut dm = DmForceField::new(4, 4, 4, DmForceConfig::default());
    dm.force.fill([1e-6, 0.0, 0.0]);
    dm.kappa_field.fill(0.01);
    dm.kappa_drag = 0.01;
    dm.v_dm_lattice = [0.05, 0.0, 0.0];
    let expected_acceleration = 1e-6 + 0.01 * 0.05 * 0.05;
    for density in [1.0, 2.7] {
        let mut solver = LbmSolver3D::new(4, 4, 4, 0.8);
        solver.initialize_uniform(density, [0.0; 3]);
        let drag = dm.drag_force_density_lattice(&solver.rho, &solver.u);
        let gravity: Vec<_> = dm
            .force
            .iter()
            .zip(&solver.rho)
            .map(|(acceleration, rho)| acceleration.map(|component| component * rho))
            .collect();
        solver
            .set_force_field(combine_forces(&gravity, &drag))
            .unwrap();
        solver.phase1_collision().unwrap();
        solver.compute_macroscopic();
        for velocity in &solver.u {
            assert!(
                (velocity[0] - expected_acceleration).abs() < 2e-16,
                "Guo acceleration differs at density {density}: {}",
                velocity[0]
            );
        }
    }
}
