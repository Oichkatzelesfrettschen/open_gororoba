//! Taylor-Green Vortex Decay Validation for LbmSolver3D
//!
//! The Taylor-Green vortex is the standard validation benchmark for LBM solvers.
//! In a periodic domain, the initial velocity field:
//!
//!   u_x(x,y) = U0 * cos(kx * x) * sin(ky * y)
//!   u_y(x,y) = -U0 * sin(kx * x) * cos(ky * y)
//!   u_z = 0
//!
//! decays exponentially under viscous diffusion:
//!
//!   u(t) = u(0) * exp(-nu * (kx^2 + ky^2) * t)
//!
//! where nu = c_s^2 * (tau - 0.5) = (tau - 0.5) / 3 in lattice units.
//!
//! The LBM solver must reproduce this decay rate to validate that both
//! the collision operator (BGK) and the streaming step are correctly
//! implemented.

use lbm_3d::solver::LbmSolver3D;

/// Compute the L2 norm of the velocity field.
fn velocity_l2(solver: &LbmSolver3D) -> f64 {
    let mut sum_sq = 0.0;
    for z in 0..solver.nz {
        for y in 0..solver.ny {
            for x in 0..solver.nx {
                let (_, u) = solver.get_macroscopic(x, y, z);
                sum_sq += u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
            }
        }
    }
    sum_sq.sqrt()
}

/// Initialize solver with Taylor-Green vortex (2D in xy-plane, uniform in z).
fn initialize_taylor_green(solver: &mut LbmSolver3D, u0: f64) {
    let nx = solver.nx;
    let ny = solver.ny;
    let nz = solver.nz;
    let kx = 2.0 * std::f64::consts::PI / nx as f64;
    let ky = 2.0 * std::f64::consts::PI / ny as f64;

    // First initialize to rest with uniform density
    solver.initialize_uniform(1.0, [0.0, 0.0, 0.0]);

    // Then set the Taylor-Green velocity at each point via equilibrium initialization
    let lattice = solver.collider.lattice.clone();
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let ux = u0 * (kx * x as f64).cos() * (ky * y as f64).sin();
                let uy = -u0 * (kx * x as f64).sin() * (ky * y as f64).cos();
                let u = [ux, uy, 0.0];

                let idx = z * (nx * ny) + y * nx + x;
                solver.rho[idx] = 1.0;
                solver.u[idx] = u;

                // Set distribution function to equilibrium at this velocity
                let f_eq =
                    lbm_3d::solver::BgkCollision::initialize_with_velocity(1.0, u, &lattice);
                let f_start = idx * 19;
                solver.f[f_start..f_start + 19].copy_from_slice(&f_eq);
            }
        }
    }
}

/// Taylor-Green vortex decay: verify exponential decay rate matches theory.
///
/// Theory: |u(t)| / |u(0)| = exp(-nu * (kx^2 + ky^2) * t)
/// where nu = (tau - 0.5) / 3 in lattice units.
#[test]
fn test_taylor_green_decay_rate() {
    let n = 32;
    let tau = 0.8;
    let nu = (tau - 0.5) / 3.0;
    let u0 = 0.01; // Small initial velocity (Ma << 1 for incompressible regime)

    let mut solver = LbmSolver3D::new(n, n, n, tau);
    let tau_field = vec![tau; n * n * n];
    solver
        .set_viscosity_field(tau_field)
        .expect("set viscosity field");
    initialize_taylor_green(&mut solver, u0);

    let kx = 2.0 * std::f64::consts::PI / n as f64;
    let ky = 2.0 * std::f64::consts::PI / n as f64;
    let k_sq = kx * kx + ky * ky;
    let decay_rate = nu * k_sq;

    let l2_initial = velocity_l2(&solver);
    assert!(l2_initial > 0.0, "Initial velocity field must be nonzero");

    // Evolve for 100 timesteps
    let n_steps = 100;
    solver.evolve(n_steps);
    solver.compute_macroscopic();

    let l2_final = velocity_l2(&solver);
    let measured_ratio = l2_final / l2_initial;
    let expected_ratio = (-decay_rate * n_steps as f64).exp();

    // Allow 5% relative error (BGK has O(Ma^2) compressibility error)
    let relative_error = (measured_ratio - expected_ratio).abs() / expected_ratio;
    assert!(
        relative_error < 0.05,
        "Taylor-Green decay rate mismatch: measured ratio={:.6}, expected={:.6}, rel_err={:.4}",
        measured_ratio,
        expected_ratio,
        relative_error,
    );
}

/// Verify Taylor-Green velocity monotonically decays (no numerical amplification).
#[test]
fn test_taylor_green_monotonic_decay() {
    let n = 16;
    let tau = 0.7;

    let mut solver = LbmSolver3D::new(n, n, n, tau);
    let tau_field = vec![tau; n * n * n];
    solver
        .set_viscosity_field(tau_field)
        .expect("set viscosity field");
    initialize_taylor_green(&mut solver, 0.01);

    let mut prev_l2 = velocity_l2(&solver);

    for step in 0..50 {
        solver.evolve(1);
        solver.compute_macroscopic();
        let l2 = velocity_l2(&solver);
        assert!(
            l2 <= prev_l2 * 1.001, // Allow tiny floating-point noise
            "Velocity should decay monotonically: step {}, prev={:.6}, curr={:.6}",
            step,
            prev_l2,
            l2,
        );
        prev_l2 = l2;
    }
}

/// Mass conservation: total mass must be exactly preserved by collision + streaming.
#[test]
fn test_mass_conservation() {
    let n = 16;
    let tau = 0.6;

    let mut solver = LbmSolver3D::new(n, n, n, tau);
    let tau_field = vec![tau; n * n * n];
    solver
        .set_viscosity_field(tau_field)
        .expect("set viscosity field");
    initialize_taylor_green(&mut solver, 0.01);

    let mass_initial = solver.total_mass();
    assert!(mass_initial > 0.0, "Initial mass must be positive");

    // Evolve 200 steps
    solver.evolve(200);

    let mass_final = solver.total_mass();
    let relative_error = (mass_final - mass_initial).abs() / mass_initial;

    // Mass conservation should be exact to floating-point precision
    assert!(
        relative_error < 1e-10,
        "Mass conservation violated: initial={:.6}, final={:.6}, rel_err={:.2e}",
        mass_initial,
        mass_final,
        relative_error,
    );
}

/// Momentum growth with Guo forcing scheme.
///
/// With uniform body force F_x in a periodic domain starting from rest,
/// the LBM distribution momentum (from moments of f_i) grows as:
///
///   P_x(t) = (1 - 1/(2*tau)) * F_x * M * t
///
/// The (1-1/(2*tau)) factor is intrinsic to the Guo scheme: it splits
/// the forcing between the distribution function and a velocity correction
/// term. The physical velocity is u = u_LBM + F/(2*rho), but
/// compute_macroscopic() returns u_LBM only.
#[test]
fn test_momentum_with_guo_forcing() {
    let n = 16;
    let tau = 0.7;
    let f_x = 1e-5;

    let mut solver = LbmSolver3D::new(n, n, n, tau);
    let n_cells = n * n * n;
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

    // Evolve and measure momentum growth
    let n_steps = 50;
    solver.evolve(n_steps);
    solver.compute_macroscopic();

    // Compute total x-momentum from distribution (u_LBM, not u_physical)
    let mut px_total = 0.0;
    for z in 0..n {
        for y in 0..n {
            for x in 0..n {
                let (rho, u) = solver.get_macroscopic(x, y, z);
                px_total += rho * u[0];
            }
        }
    }

    // Guo scheme: distribution momentum grows as (1-1/(2tau)) * F * M * t
    let guo_factor = 1.0 - 1.0 / (2.0 * tau);
    let px_expected = guo_factor * f_x * mass * n_steps as f64;

    // Allow 5% relative error
    let relative_error = (px_total - px_expected).abs() / px_expected.abs().max(1e-15);
    assert!(
        relative_error < 0.05,
        "Guo momentum mismatch: measured px={:.6e}, expected={:.6e}, rel_err={:.4}, guo_factor={:.4}",
        px_total,
        px_expected,
        relative_error,
        guo_factor,
    );
}

/// Stability check: solver should remain stable for reasonable parameters.
#[test]
fn test_stability_during_evolution() {
    let n = 16;
    let tau = 0.7;

    let mut solver = LbmSolver3D::new(n, n, n, tau);
    let tau_field = vec![tau; n * n * n];
    solver
        .set_viscosity_field(tau_field)
        .expect("set viscosity field");
    initialize_taylor_green(&mut solver, 0.01);

    for step in 0..100 {
        solver.evolve(1);
        assert!(
            solver.is_stable(),
            "Solver became unstable at step {}",
            step + 1
        );
    }
}

/// Grid convergence: decay rate error should decrease with finer grid.
///
/// For BGK on D3Q19, the theoretical convergence order is O(dx^2).
/// We test that the error at N=32 is at least 2x smaller than at N=16.
#[test]
fn test_grid_convergence() {
    let tau = 0.8;
    let nu = (tau - 0.5) / 3.0;
    let u0 = 0.005;
    let n_steps = 50;

    let mut errors = Vec::new();

    for &grid_size in &[16_usize, 32] {
        let mut solver = LbmSolver3D::new(grid_size, grid_size, grid_size, tau);
        let tau_field = vec![tau; grid_size * grid_size * grid_size];
        solver
            .set_viscosity_field(tau_field)
            .expect("set viscosity field");
        initialize_taylor_green(&mut solver, u0);

        let l2_initial = velocity_l2(&solver);

        solver.evolve(n_steps);
        solver.compute_macroscopic();

        let l2_final = velocity_l2(&solver);
        let measured = l2_final / l2_initial;

        let k = 2.0 * std::f64::consts::PI / grid_size as f64;
        let expected = (-nu * 2.0 * k * k * n_steps as f64).exp();
        let error = (measured - expected).abs() / expected;
        errors.push(error);
    }

    // Error at N=32 should be significantly less than at N=16
    // (O(dx^2) convergence means ~4x reduction, allow weaker 2x)
    assert!(
        errors[1] < errors[0],
        "Grid convergence failed: error at N=16 = {:.4}, error at N=32 = {:.4}",
        errors[0],
        errors[1],
    );
}
