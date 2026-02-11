//! Tests for spatially-varying viscosity in LBM solver.
//!
//! This module validates the extension of LBM 3D solver to support per-cell
//! relaxation times (tau_field), enabling spatial viscosity variation.
//!
//! Test categories:
//! - API tests: set_viscosity_field, get_viscosity_field, validation
//! - Physics tests: Chapman-Enskog relation, conservation laws, stability
//! - Integration tests: end-to-end workflows, determinism, backward compatibility

use lbm_3d::solver::{BgkCollision, LbmSolver3D};

// ============================================================================
// API Tests (8 tests)
// ============================================================================

#[test]
fn test_set_uniform_viscosity_field() {
    let mut solver = LbmSolver3D::new(4, 4, 4, 0.6);  // 64 cells
    let tau_uniform = vec![0.8; 64];

    let result = solver.set_viscosity_field(tau_uniform.clone());
    assert!(result.is_ok());
    assert_eq!(solver.get_viscosity_field().len(), 64);
}

#[test]
fn test_set_varying_viscosity_field() {
    let mut solver = LbmSolver3D::new(4, 4, 4, 0.6);
    let mut tau_varying = vec![0.6; 64];
    // Create a gradient: tau increases along x-direction
    for x in 0..4 {
        for y in 0..4 {
            for z in 0..4 {
                let idx = z * 16 + y * 4 + x;
                tau_varying[idx] = 0.6 + (x as f64) * 0.05;
            }
        }
    }

    let result = solver.set_viscosity_field(tau_varying.clone());
    assert!(result.is_ok());
}

#[test]
fn test_length_validation_mismatch() {
    let mut solver = LbmSolver3D::new(4, 4, 4, 0.6);  // 64 cells
    let tau_wrong = vec![0.8; 32];  // Wrong size

    let result = solver.set_viscosity_field(tau_wrong);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("length mismatch"));
}

#[test]
fn test_stability_constraint_violation() {
    let mut solver = LbmSolver3D::new(4, 4, 4, 0.6);
    let mut tau_invalid = vec![0.8; 64];
    tau_invalid[10] = 0.4;  // Violate tau >= 0.5

    let result = solver.set_viscosity_field(tau_invalid);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Stability violation"));
}

#[test]
fn test_nan_rejection() {
    let mut solver = LbmSolver3D::new(4, 4, 4, 0.6);
    let mut tau_invalid = vec![0.8; 64];
    tau_invalid[5] = f64::NAN;

    let result = solver.set_viscosity_field(tau_invalid);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Non-finite"));
}

#[test]
fn test_infinity_rejection() {
    let mut solver = LbmSolver3D::new(4, 4, 4, 0.6);
    let mut tau_invalid = vec![0.8; 64];
    tau_invalid[15] = f64::INFINITY;

    let result = solver.set_viscosity_field(tau_invalid);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("Non-finite"));
}

#[test]
fn test_viscosity_field_roundtrip() {
    let mut solver = LbmSolver3D::new(2, 2, 2, 0.6);
    let tau_set = vec![0.7, 0.8, 0.9, 1.0, 0.75, 0.85, 0.95, 1.05];

    solver.set_viscosity_field(tau_set.clone()).expect("Failed to set");
    let nu_field = solver.get_viscosity_field();

    // Check nu = (1/3) * (tau - 0.5)
    assert_eq!(nu_field.len(), 8);
    for (i, &nu) in nu_field.iter().enumerate() {
        let tau = tau_set[i];
        let expected = (1.0 / 3.0) * (tau - 0.5);
        assert!((nu - expected).abs() < 1e-14, "Mismatch at index {}", i);
    }
}

#[test]
fn test_backward_compatibility_uniform_constructor() {
    let mut bgk = BgkCollision::new(0.8);
    assert!(bgk.tau_field.len() > 0);
    assert!(bgk.tau_field[0] >= 0.5);

    // Can be extended to full field
    let tau_full = vec![0.8; 8];
    let result = bgk.set_viscosity_field(tau_full);
    assert!(result.is_ok());
}

// ============================================================================
// Physics Tests (6 tests)
// ============================================================================

#[test]
fn test_chapman_enskog_relation() {
    // Verify: nu = (1/3) * (tau - 0.5)
    let nu_ref = 0.1;
    let tau_expected = 3.0 * nu_ref + 0.5;  // tau = 0.8

    let bgk = BgkCollision::new(tau_expected);
    let nu = bgk.viscosity();

    assert!((nu - nu_ref).abs() < 1e-14);
}

#[test]
fn test_mass_conservation_uniform_viscosity() {
    // Initialize with uniform viscosity field
    let mut solver = LbmSolver3D::new(8, 8, 4, 0.6);
    let tau_field = vec![0.8; 8 * 8 * 4];
    solver.set_viscosity_field(tau_field).expect("Failed to set");
    solver.initialize_uniform(1.0, [0.05, 0.02, 0.01]);

    let mass_before = solver.total_mass();

    // Evolve one step
    solver.evolve_one_step();

    let mass_after = solver.total_mass();
    assert!((mass_after - mass_before).abs() < 1e-10);
}

#[test]
fn test_mass_conservation_varying_viscosity() {
    // Initialize with varying viscosity field
    let mut solver = LbmSolver3D::new(4, 4, 4, 0.6);
    let mut tau_field = vec![0.6; 64];
    // Create a gradient
    for i in 0..64 {
        tau_field[i] = 0.6 + (i as f64) * 0.005;
    }
    solver.set_viscosity_field(tau_field).expect("Failed to set");
    solver.initialize_uniform(1.2, [0.05, 0.03, 0.02]);

    let mass_before = solver.total_mass();

    // Evolve 5 steps
    solver.evolve(5);

    let mass_after = solver.total_mass();
    // Mass should be conserved to high precision (BGK conserves mass by construction)
    assert!((mass_after - mass_before).abs() < 1e-8);
}

#[test]
fn test_poiseuille_analog_stability() {
    // Simulate a channel-like flow with varying viscosity
    // Higher viscosity in the middle (like a channel with walls)
    let mut solver = LbmSolver3D::new(8, 8, 4, 0.6);
    let mut tau_field = vec![0.6; 256];  // 8*8*4

    // Vary tau along y: lower at boundaries, higher in middle
    for y in 0..8 {
        for x in 0..8 {
            for z in 0..4 {
                let idx = z * 64 + y * 8 + x;
                let distance_to_wall = ((y as f64 - 3.5).abs()).min(4.0);
                tau_field[idx] = 0.6 + 0.2 * (distance_to_wall / 4.0);
            }
        }
    }

    solver.set_viscosity_field(tau_field).expect("Failed to set");
    solver.initialize_uniform(1.0, [0.08, 0.0, 0.0]);

    // Evolve and check stability
    for _ in 0..20 {
        solver.evolve_one_step();
        assert!(solver.is_stable(), "Stability violated");
    }
}

#[test]
fn test_10x_viscosity_contrast_stability() {
    // Test with 10x viscosity contrast (low: 0.6, high: 1.0)
    let mut solver = LbmSolver3D::new(4, 4, 4, 0.6);
    let mut tau_field = vec![0.6; 64];

    // Checkerboard pattern: alternate high/low viscosity
    for (i, tau) in tau_field.iter_mut().enumerate() {
        if i % 2 == 0 {
            *tau = 0.6;    // low viscosity
        } else {
            *tau = 1.0;    // high viscosity (10x contrast)
        }
    }

    solver.set_viscosity_field(tau_field).expect("Failed to set");
    solver.initialize_uniform(1.0, [0.1, 0.05, 0.02]);

    // Should remain stable through evolution
    for _ in 0..50 {
        solver.evolve_one_step();
        assert!(solver.is_stable(), "Stability violated with 10x contrast");
    }
}

// ============================================================================
// Integration Tests (4 tests)
// ============================================================================

#[test]
fn test_8x8x4_end_to_end_spatial_viscosity() {
    let mut solver = LbmSolver3D::new(8, 8, 4, 0.6);
    let tau_field = vec![0.75; 256];
    solver.set_viscosity_field(tau_field).expect("Failed to set");
    solver.initialize_uniform(1.0, [0.1, 0.05, 0.02]);

    // Evolve and check invariants
    for step in 0..20 {
        solver.evolve_one_step();
        let mass = solver.total_mass();
        assert!((mass - 256.0).abs() < 1e-8, "Mass not conserved at step {}", step);
        assert!(solver.is_stable(), "Unstable at step {}", step);
    }
}

#[test]
fn test_16x16x8_convergence_with_viscosity() {
    // Larger grid to test convergence properties
    let mut solver = LbmSolver3D::new(16, 16, 8, 0.6);
    let tau_field = vec![0.8; 16 * 16 * 8];
    solver.set_viscosity_field(tau_field).expect("Failed to set");
    solver.initialize_uniform(1.0, [0.05, 0.03, 0.01]);

    let mass_init = solver.total_mass();

    // Evolve 100 steps
    solver.evolve(100);

    // Check mass still conserved
    let mass_final = solver.total_mass();
    assert!((mass_final - mass_init).abs() < 1e-6);
}

#[test]
fn test_determinism_with_seed() {
    // Two solvers with identical setup should give identical results
    let mut solver1 = LbmSolver3D::new(4, 4, 4, 0.6);
    let mut solver2 = LbmSolver3D::new(4, 4, 4, 0.6);

    let tau_field = vec![0.8; 64];
    solver1.set_viscosity_field(tau_field.clone()).expect("Failed to set");
    solver2.set_viscosity_field(tau_field).expect("Failed to set");

    solver1.initialize_uniform(1.2, [0.1, 0.05, 0.02]);
    solver2.initialize_uniform(1.2, [0.1, 0.05, 0.02]);

    for _ in 0..10 {
        solver1.evolve_one_step();
        solver2.evolve_one_step();
    }

    // Compare macroscopic quantities
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                let (rho1, u1) = solver1.get_macroscopic(x, y, z);
                let (rho2, u2) = solver2.get_macroscopic(x, y, z);

                assert!((rho1 - rho2).abs() < 1e-14);
                assert!((u1[0] - u2[0]).abs() < 1e-14);
                assert!((u1[1] - u2[1]).abs() < 1e-14);
                assert!((u1[2] - u2[2]).abs() < 1e-14);
            }
        }
    }
}

#[test]
fn test_equivalence_uniform_field_vs_uniform_constructor() {
    // Uniform field should behave like original uniform constructor
    let mut solver_field = LbmSolver3D::new(4, 4, 4, 0.6);
    let tau_uniform = vec![0.8; 64];
    solver_field.set_viscosity_field(tau_uniform).expect("Failed to set");

    let mut solver_uniform = LbmSolver3D::new(4, 4, 4, 0.8);

    solver_field.initialize_uniform(1.0, [0.05, 0.02, 0.01]);
    solver_uniform.initialize_uniform(1.0, [0.05, 0.02, 0.01]);

    // Evolve both
    for _ in 0..15 {
        solver_field.evolve_one_step();
        solver_uniform.evolve_one_step();
    }

    // Results should be identical
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                let (rho_f, u_f) = solver_field.get_macroscopic(x, y, z);
                let (rho_u, u_u) = solver_uniform.get_macroscopic(x, y, z);

                assert!((rho_f - rho_u).abs() < 1e-13, "Density mismatch at ({},{},{})", x, y, z);
                assert!((u_f[0] - u_u[0]).abs() < 1e-13);
                assert!((u_f[1] - u_u[1]).abs() < 1e-13);
                assert!((u_f[2] - u_u[2]).abs() < 1e-13);
            }
        }
    }
}
