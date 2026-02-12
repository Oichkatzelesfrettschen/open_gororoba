//! GPU-CPU equivalence tests for D3Q19 LBM solver.
//!
//! Validates that GPU kernel outputs match CPU reference implementation
//! to within numerical precision (RMS error < 1e-6 for f64).
//!
//! Test scenarios:
//! - Uniform viscosity field (baseline)
//! - Gradient viscosity field (frustration-like)
//! - Mass conservation (total density invariant)
//! - Momentum conservation (zero-force equilibrium)
//! - Percolation-like channels (high-contrast viscosity)
//! - Determinism (multiple runs, same result)

use approx::assert_abs_diff_eq;
use lbm_3d::solver::LbmSolver3D;
use lbm_3d_cuda::LbmSolver3DCuda;

/// Compute RMS error between two f64 vectors
fn rms_error(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sum_sq: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum();
    (sum_sq / a.len() as f64).sqrt()
}

/// Compute RMS error for velocity fields (Vec<[f64; 3]>)
fn rms_error_velocity(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sum_sq: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(va, vb)| {
            (va[0] - vb[0]).powi(2) + (va[1] - vb[1]).powi(2) + (va[2] - vb[2]).powi(2)
        })
        .sum();
    (sum_sq / (3 * a.len()) as f64).sqrt()
}

/// Test 1: Uniform viscosity field, short evolution
#[test]
fn test_uniform_field_equivalence() {
    let (nx, ny, nz) = (16, 16, 16);
    let tau = 1.0;
    let steps = 100;

    // CPU solver
    let mut cpu_solver = LbmSolver3D::new(nx, ny, nz, tau);
    cpu_solver.initialize_uniform(1.0, [0.01, 0.0, 0.0]);
    cpu_solver.evolve(steps);

    // GPU solver
    let mut gpu_solver = match LbmSolver3DCuda::new(nx, ny, nz, tau) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
    };
    gpu_solver.initialize_uniform(1.0, [0.01, 0.0, 0.0]).unwrap();
    gpu_solver.evolve(steps).unwrap();

    // Compare results
    let rho_err = rms_error(&cpu_solver.rho, &gpu_solver.rho);
    let u_err = rms_error_velocity(&cpu_solver.u, &gpu_solver.u);

    println!("Uniform field: rho_err={:.3e}, u_err={:.3e}", rho_err, u_err);

    assert!(rho_err < 1e-6, "Density RMS error too large: {}", rho_err);
    assert!(u_err < 1e-6, "Velocity RMS error too large: {}", u_err);
}

/// Test 2: Gradient viscosity field (mimics frustration-viscosity coupling)
#[test]
fn test_gradient_viscosity_equivalence() {
    let (nx, ny, nz) = (16, 16, 16);
    let n_cells = nx * ny * nz;
    let steps = 100;

    // Create linear gradient: nu(x) = 0.05 + 0.1 * x / nx, then convert to tau
    let mut nu_field = vec![0.0; n_cells];
    let mut tau_field = vec![0.0; n_cells];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let idx = x + nx * (y + ny * z);
                nu_field[idx] = 0.05 + 0.1 * (x as f64 / nx as f64);
                tau_field[idx] = 3.0 * nu_field[idx] + 0.5; // Chapman-Enskog
            }
        }
    }

    // CPU solver (takes tau directly)
    let mut cpu_solver = LbmSolver3D::new(nx, ny, nz, 1.0); // Tau will be overridden
    cpu_solver.set_viscosity_field(tau_field).unwrap();
    cpu_solver.initialize_uniform(1.0, [0.01, 0.0, 0.0]);
    cpu_solver.evolve(steps);

    // GPU solver
    let mut gpu_solver = match LbmSolver3DCuda::new(nx, ny, nz, 1.0) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
    };
    gpu_solver.set_viscosity_field(&nu_field).unwrap();
    gpu_solver.initialize_uniform(1.0, [0.01, 0.0, 0.0]).unwrap();
    gpu_solver.evolve(steps).unwrap();

    // Compare results
    let rho_err = rms_error(&cpu_solver.rho, &gpu_solver.rho);
    let u_err = rms_error_velocity(&cpu_solver.u, &gpu_solver.u);

    println!("Gradient field: rho_err={:.3e}, u_err={:.3e}", rho_err, u_err);

    assert!(rho_err < 1e-6, "Density RMS error too large: {}", rho_err);
    assert!(u_err < 1e-6, "Velocity RMS error too large: {}", u_err);
}

/// Test 3: Mass conservation (total density should be invariant)
#[test]
fn test_mass_conservation_equivalence() {
    let (nx, ny, nz) = (16, 16, 16);
    let tau = 1.0;
    let steps = 200;
    let rho_init = 1.5;

    // CPU solver
    let mut cpu_solver = LbmSolver3D::new(nx, ny, nz, tau);
    cpu_solver.initialize_uniform(rho_init, [0.02, 0.01, 0.0]);
    let cpu_mass_init: f64 = cpu_solver.rho.iter().sum();
    cpu_solver.evolve(steps);
    let cpu_mass_final: f64 = cpu_solver.rho.iter().sum();

    // GPU solver
    let mut gpu_solver = match LbmSolver3DCuda::new(nx, ny, nz, tau) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
    };
    gpu_solver.initialize_uniform(rho_init, [0.02, 0.01, 0.0]).unwrap();
    let gpu_mass_init: f64 = gpu_solver.rho.iter().sum();
    gpu_solver.evolve(steps).unwrap();
    let gpu_mass_final: f64 = gpu_solver.rho.iter().sum();

    println!("CPU mass: init={:.6}, final={:.6}, delta={:.3e}",
             cpu_mass_init, cpu_mass_final, cpu_mass_final - cpu_mass_init);
    println!("GPU mass: init={:.6}, final={:.6}, delta={:.3e}",
             gpu_mass_init, gpu_mass_final, gpu_mass_final - gpu_mass_init);

    // Both should conserve mass
    assert_abs_diff_eq!(cpu_mass_init, cpu_mass_final, epsilon = 1e-6);
    assert_abs_diff_eq!(gpu_mass_init, gpu_mass_final, epsilon = 1e-6);

    // And agree with each other
    assert_abs_diff_eq!(cpu_mass_final, gpu_mass_final, epsilon = 1e-6);
}

/// Test 4: Equilibrium state (zero velocity should remain stable)
#[test]
fn test_equilibrium_stability_equivalence() {
    let (nx, ny, nz) = (16, 16, 16);
    let tau = 1.0;
    let steps = 100;

    // CPU solver (zero velocity = equilibrium)
    let mut cpu_solver = LbmSolver3D::new(nx, ny, nz, tau);
    cpu_solver.initialize_uniform(1.0, [0.0, 0.0, 0.0]);
    cpu_solver.evolve(steps);

    // GPU solver
    let mut gpu_solver = match LbmSolver3DCuda::new(nx, ny, nz, tau) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
    };
    gpu_solver.initialize_uniform(1.0, [0.0, 0.0, 0.0]).unwrap();
    gpu_solver.evolve(steps).unwrap();

    // Both should stay at rho=1.0, u=0.0
    let cpu_max_u: f64 = cpu_solver
        .u
        .iter()
        .map(|v| (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)).sqrt())
        .fold(0.0_f64, f64::max);

    let gpu_max_u: f64 = gpu_solver
        .u
        .iter()
        .map(|v| (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)).sqrt())
        .fold(0.0_f64, f64::max);

    println!("Equilibrium: CPU max_u={:.3e}, GPU max_u={:.3e}", cpu_max_u, gpu_max_u);

    assert!(cpu_max_u < 1e-10, "CPU equilibrium not stable");
    assert!(gpu_max_u < 1e-10, "GPU equilibrium not stable");

    // Compare final states
    let rho_err = rms_error(&cpu_solver.rho, &gpu_solver.rho);
    let u_err = rms_error_velocity(&cpu_solver.u, &gpu_solver.u);

    assert!(rho_err < 1e-12, "Equilibrium density error: {}", rho_err);
    assert!(u_err < 1e-12, "Equilibrium velocity error: {}", u_err);
}

/// Test 5: High-contrast viscosity (percolation-like channels)
#[test]
fn test_high_contrast_viscosity_equivalence() {
    let (nx, ny, nz) = (16, 16, 16);
    let n_cells = nx * ny * nz;
    let steps = 50; // Shorter evolution for stability

    // Create checkerboard pattern: nu = 0.05 (low) or 0.5 (high), then convert to tau
    let mut nu_field = vec![0.0; n_cells];
    let mut tau_field = vec![0.0; n_cells];
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let idx = x + nx * (y + ny * z);
                nu_field[idx] = if (x + y + z) % 2 == 0 { 0.05 } else { 0.5 };
                tau_field[idx] = 3.0 * nu_field[idx] + 0.5; // Chapman-Enskog
            }
        }
    }

    // CPU solver (takes tau directly)
    let mut cpu_solver = LbmSolver3D::new(nx, ny, nz, 1.0);
    cpu_solver.set_viscosity_field(tau_field).unwrap();
    cpu_solver.initialize_uniform(1.0, [0.01, 0.0, 0.0]);
    cpu_solver.evolve(steps);

    // GPU solver
    let mut gpu_solver = match LbmSolver3DCuda::new(nx, ny, nz, 1.0) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
    };
    gpu_solver.set_viscosity_field(&nu_field).unwrap();
    gpu_solver.initialize_uniform(1.0, [0.01, 0.0, 0.0]).unwrap();
    gpu_solver.evolve(steps).unwrap();

    // Compare results
    let rho_err = rms_error(&cpu_solver.rho, &gpu_solver.rho);
    let u_err = rms_error_velocity(&cpu_solver.u, &gpu_solver.u);

    println!("High contrast: rho_err={:.3e}, u_err={:.3e}", rho_err, u_err);

    assert!(rho_err < 1e-6, "Density RMS error too large: {}", rho_err);
    assert!(u_err < 1e-6, "Velocity RMS error too large: {}", u_err);
}

/// Test 6: Larger grid (32x32x32) for scaling validation
#[test]
fn test_large_grid_equivalence() {
    let (nx, ny, nz) = (32, 32, 32);
    let tau = 1.0;
    let steps = 50; // Keep short for test speed

    // CPU solver
    let mut cpu_solver = LbmSolver3D::new(nx, ny, nz, tau);
    cpu_solver.initialize_uniform(1.0, [0.01, 0.0, 0.0]);
    cpu_solver.evolve(steps);

    // GPU solver
    let mut gpu_solver = match LbmSolver3DCuda::new(nx, ny, nz, tau) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
    };
    gpu_solver.initialize_uniform(1.0, [0.01, 0.0, 0.0]).unwrap();
    gpu_solver.evolve(steps).unwrap();

    // Compare results
    let rho_err = rms_error(&cpu_solver.rho, &gpu_solver.rho);
    let u_err = rms_error_velocity(&cpu_solver.u, &gpu_solver.u);

    println!("Large grid (32^3): rho_err={:.3e}, u_err={:.3e}", rho_err, u_err);

    assert!(rho_err < 1e-6, "Density RMS error too large: {}", rho_err);
    assert!(u_err < 1e-6, "Velocity RMS error too large: {}", u_err);
}

/// Test 7: Determinism (multiple GPU runs with same seed)
#[test]
fn test_gpu_determinism() {
    let (nx, ny, nz) = (16, 16, 16);
    let tau = 1.0;
    let steps = 100;

    // First GPU run
    let mut gpu1 = match LbmSolver3DCuda::new(nx, ny, nz, tau) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
    };
    gpu1.initialize_uniform(1.0, [0.01, 0.0, 0.0]).unwrap();
    gpu1.evolve(steps).unwrap();

    // Second GPU run (identical parameters)
    let mut gpu2 = LbmSolver3DCuda::new(nx, ny, nz, tau).unwrap();
    gpu2.initialize_uniform(1.0, [0.01, 0.0, 0.0]).unwrap();
    gpu2.evolve(steps).unwrap();

    // Should be bit-identical
    let rho_err = rms_error(&gpu1.rho, &gpu2.rho);
    let u_err = rms_error_velocity(&gpu1.u, &gpu2.u);

    println!("Determinism: rho_err={:.3e}, u_err={:.3e}", rho_err, u_err);

    assert!(rho_err < 1e-14, "GPU not deterministic (density)");
    assert!(u_err < 1e-14, "GPU not deterministic (velocity)");
}

/// Test 8: Momentum magnitude evolution (non-zero initial velocity)
#[test]
fn test_momentum_evolution_equivalence() {
    let (nx, ny, nz) = (16, 16, 16);
    let tau = 1.0;
    let steps = 100;
    let u_init = [0.05, 0.03, 0.01]; // Non-trivial velocity

    // CPU solver
    let mut cpu_solver = LbmSolver3D::new(nx, ny, nz, tau);
    cpu_solver.initialize_uniform(1.0, u_init);
    cpu_solver.evolve(steps);

    let cpu_momentum: f64 = cpu_solver
        .u
        .iter()
        .map(|v| (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)).sqrt())
        .sum();

    // GPU solver
    let mut gpu_solver = match LbmSolver3DCuda::new(nx, ny, nz, tau) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
    };
    gpu_solver.initialize_uniform(1.0, u_init).unwrap();
    gpu_solver.evolve(steps).unwrap();

    let gpu_momentum: f64 = gpu_solver
        .u
        .iter()
        .map(|v| (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)).sqrt())
        .sum();

    println!("Momentum: CPU={:.6}, GPU={:.6}, err={:.3e}",
             cpu_momentum, gpu_momentum, (cpu_momentum - gpu_momentum).abs());

    assert_abs_diff_eq!(cpu_momentum, gpu_momentum, epsilon = 1e-6);
}

/// Test 9: Viscosity field roundtrip (set -> get -> compare)
#[test]
fn test_viscosity_field_roundtrip() {
    let (nx, ny, nz) = (16, 16, 16);
    let n_cells = nx * ny * nz;

    // Create arbitrary viscosity field
    let nu_original: Vec<f64> = (0..n_cells).map(|i| 0.1 + 0.01 * (i % 10) as f64).collect();

    // GPU solver
    let mut gpu_solver = match LbmSolver3DCuda::new(nx, ny, nz, 1.0) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
    };

    gpu_solver.set_viscosity_field(&nu_original).unwrap();
    let nu_retrieved = gpu_solver.get_viscosity_field().unwrap();

    // Compare
    let err = rms_error(&nu_original, &nu_retrieved);
    println!("Viscosity roundtrip: err={:.3e}", err);

    assert!(err < 1e-12, "Viscosity roundtrip error: {}", err);
}

/// Test 10: Small grid edge case (8x8x8)
#[test]
fn test_small_grid_equivalence() {
    let (nx, ny, nz) = (8, 8, 8);
    let tau = 1.0;
    let steps = 100;

    // CPU solver
    let mut cpu_solver = LbmSolver3D::new(nx, ny, nz, tau);
    cpu_solver.initialize_uniform(1.0, [0.01, 0.0, 0.0]);
    cpu_solver.evolve(steps);

    // GPU solver
    let mut gpu_solver = match LbmSolver3DCuda::new(nx, ny, nz, tau) {
        Ok(s) => s,
        Err(_) => {
            eprintln!("GPU not available, skipping test");
            return;
        }
    };
    gpu_solver.initialize_uniform(1.0, [0.01, 0.0, 0.0]).unwrap();
    gpu_solver.evolve(steps).unwrap();

    // Compare results
    let rho_err = rms_error(&cpu_solver.rho, &gpu_solver.rho);
    let u_err = rms_error_velocity(&cpu_solver.u, &gpu_solver.u);

    println!("Small grid (8^3): rho_err={:.3e}, u_err={:.3e}", rho_err, u_err);

    assert!(rho_err < 1e-6, "Density RMS error too large: {}", rho_err);
    assert!(u_err < 1e-6, "Velocity RMS error too large: {}", u_err);
}
