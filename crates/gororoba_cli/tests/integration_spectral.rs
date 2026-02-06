//! Integration tests for spectral analysis workflows.
//!
//! Tests cross-crate integration between spectral_core (fractional Laplacian,
//! negative dimension PDE) and cosmology_core (bounce cosmology).

use spectral_core::{
    fractional_laplacian_periodic_1d, fractional_laplacian_periodic_2d,
    fractional_laplacian_dirichlet_1d,
};
use spectral_core::neg_dim::{eigenvalues_imaginary_time, caffarelli_silvestre_eigenvalues};

/// Test 1D periodic fractional Laplacian on a Fourier mode.
#[test]
fn test_periodic_1d_fourier_eigenfunction() {
    let n = 128;
    let l = 1.0;
    let s = 0.5;
    let m = 5; // mode number

    // Construct pure Fourier mode: cos(2*pi*m*x/L)
    let u: Vec<f64> = (0..n)
        .map(|i| {
            let x = i as f64 * l / n as f64;
            (2.0 * std::f64::consts::PI * m as f64 * x / l).cos()
        })
        .collect();

    let result = fractional_laplacian_periodic_1d(&u, s, l);

    // Expected eigenvalue: (2*pi*m/L)^(2s)
    let k = 2.0 * std::f64::consts::PI * m as f64 / l;
    let eigenvalue = k.powf(2.0 * s);

    // Result should be eigenvalue * u
    let expected: Vec<f64> = u.iter().map(|&ui| eigenvalue * ui).collect();

    let rel_error: f64 = result
        .iter()
        .zip(expected.iter())
        .map(|(r, e)| (r - e).abs())
        .sum::<f64>()
        / expected.iter().map(|e| e.abs()).sum::<f64>();

    assert!(
        rel_error < 1e-10,
        "Periodic 1D fractional Laplacian error {} too large",
        rel_error
    );
}

/// Test Dirichlet fractional Laplacian with s=1 matches standard Laplacian.
///
/// dirichlet_laplacian_1d returns -Delta u (negative Laplacian).
/// fractional_laplacian_dirichlet_1d with s=1 returns (-Delta)^1 u = -Delta u.
/// They should match directly.
#[test]
fn test_dirichlet_s1_matches_standard() {
    let n = 15;
    let l = 1.0;

    // Use a parabola that vanishes at boundaries (satisfies Dirichlet BCs)
    let u: Vec<f64> = (1..=n)
        .map(|i| {
            let x = i as f64 * l / (n + 1) as f64;
            x * (l - x)
        })
        .collect();

    // Both should return -Delta u
    let frac_result = fractional_laplacian_dirichlet_1d(&u, 1.0, l);
    let direct_result = spectral_core::dirichlet_laplacian_1d(&u, l);

    // They should match (no negation needed!)
    for (i, (&f, &d)) in frac_result.iter().zip(direct_result.iter()).enumerate() {
        let rel_error = if d.abs() > 1e-10 {
            (f - d).abs() / d.abs()
        } else {
            (f - d).abs()
        };
        assert!(
            rel_error < 1e-4,
            "Dirichlet s=1 mismatch at index {}: frac={}, direct={}, rel_error={}",
            i, f, d, rel_error
        );
    }
}

/// Test negative dimension eigenvalues are distinct.
#[test]
fn test_negative_dimension_eigenvalues_distinct() {
    let alpha = -1.5; // Negative dimension
    let epsilon = 0.1;
    let n_grid = 64;
    let x_max = 10.0;
    let n_states = 4;

    let result = eigenvalues_imaginary_time(alpha, epsilon, n_grid, x_max, n_states, 0.005, 2000);

    // All eigenvalues should be distinct
    for i in 0..result.eigenvalues.len() {
        for j in (i + 1)..result.eigenvalues.len() {
            assert!(
                (result.eigenvalues[i] - result.eigenvalues[j]).abs() > 1e-6,
                "Eigenvalues {} and {} are not distinct: {} vs {}",
                i,
                j,
                result.eigenvalues[i],
                result.eigenvalues[j]
            );
        }
    }
}

/// Test Caffarelli-Silvestre extension eigenvalues are positive.
#[test]
fn test_caffarelli_silvestre_eigenvalues_positive() {
    let s = 0.5;
    let n_grid = 64;
    let x_max = 10.0;
    let n_states = 3;

    let eigenvalues = caffarelli_silvestre_eigenvalues(s, n_grid, x_max, n_states);

    for (i, &e) in eigenvalues.iter().enumerate() {
        assert!(
            e > 0.0,
            "Caffarelli-Silvestre eigenvalue {} should be positive, got {}",
            i,
            e
        );
    }
}

/// Test eigenvalue ordering for POSITIVE alpha (standard physics).
///
/// For positive alpha, kinetic energy increases with |k|, so eigenvalues
/// should be ordered: E_0 < E_1 < E_2.
#[test]
fn test_positive_alpha_eigenvalue_ordering() {
    // Use positive alpha = 1.0 (standard Laplacian physics)
    let result = eigenvalues_imaginary_time(1.0, 0.1, 64, 10.0, 3, 0.005, 3000);

    // Should produce requested number of eigenvalues
    assert_eq!(
        result.eigenvalues.len(), 3,
        "Should produce 3 eigenvalues"
    );

    // For positive alpha, eigenvalues should be ordered
    assert!(
        result.eigenvalues[0] < result.eigenvalues[1],
        "E_0={} should be < E_1={} for positive alpha",
        result.eigenvalues[0], result.eigenvalues[1]
    );
    assert!(
        result.eigenvalues[1] < result.eigenvalues[2],
        "E_1={} should be < E_2={} for positive alpha",
        result.eigenvalues[1], result.eigenvalues[2]
    );
}

/// Test negative alpha eigenvalues are distinct (ordering not guaranteed).
///
/// For negative alpha, kinetic energy DECREASES with |k| (inverted physics).
/// The eigenvalues should be positive and distinct, but ordering depends on
/// the interplay between inverted kinetic and harmonic potential.
#[test]
fn test_negative_alpha_eigenvalues_distinct() {
    let result = eigenvalues_imaginary_time(-1.0, 0.1, 64, 10.0, 3, 0.005, 2000);

    // All eigenvalues should be positive
    for (i, &e) in result.eigenvalues.iter().enumerate() {
        assert!(
            e > 0.0,
            "Eigenvalue {} should be positive, got {}",
            i, e
        );
    }

    // Eigenvalues should be distinct (but not necessarily ordered!)
    let tol = 1e-6;
    assert!(
        (result.eigenvalues[0] - result.eigenvalues[1]).abs() > tol,
        "E_0 and E_1 should be distinct"
    );
    assert!(
        (result.eigenvalues[1] - result.eigenvalues[2]).abs() > tol,
        "E_1 and E_2 should be distinct"
    );
}

/// Test eigenstates are normalized.
///
/// The implementation uses dx = l/n and normalizes so that sum(psi^2 * dx) = 1.
#[test]
fn test_eigenstates_normalized() {
    let n_grid = 64;
    let l = 10.0; // Full domain size (NOT x_max)
    let result = eigenvalues_imaginary_time(-1.5, 0.1, n_grid, l, 2, 0.005, 2000);

    // Should produce requested number of eigenstates
    assert_eq!(
        result.eigenstates.len(), 2,
        "Should produce 2 eigenstates"
    );

    // The grid spacing is l/n (domain from -l/2 to +l/2 with n points)
    let dx = l / n_grid as f64;

    for (i, state) in result.eigenstates.iter().enumerate() {
        assert_eq!(
            state.len(), n_grid,
            "Eigenstate {} should have {} points",
            i, n_grid
        );

        // Compute L2 norm squared: integral(|psi|^2 dx) = sum(psi^2 * dx)
        let norm_sq: f64 = state.iter().map(|&psi| psi * psi).sum::<f64>() * dx;

        // Should be normalized to 1
        assert!(
            (norm_sq - 1.0).abs() < 0.05,
            "Eigenstate {} not normalized: |psi|^2 = {} (expected 1.0)",
            i, norm_sq
        );
    }
}

/// Test 2D periodic fractional Laplacian on product mode.
#[test]
fn test_periodic_2d_product_mode() {
    let nx = 32;
    let ny = 32;
    let lx = 1.0;
    let ly = 1.0;
    let s = 0.5;
    let mx = 2;
    let my = 3;

    // Construct product mode: cos(2*pi*mx*x/Lx) * cos(2*pi*my*y/Ly)
    let mut u = ndarray::Array2::zeros((nx, ny));
    for i in 0..nx {
        for j in 0..ny {
            let x = i as f64 * lx / nx as f64;
            let y = j as f64 * ly / ny as f64;
            u[[i, j]] = (2.0 * std::f64::consts::PI * mx as f64 * x / lx).cos()
                * (2.0 * std::f64::consts::PI * my as f64 * y / ly).cos();
        }
    }

    let result = fractional_laplacian_periodic_2d(&u, s, lx, ly);

    // Expected eigenvalue: (kx^2 + ky^2)^s
    let kx = 2.0 * std::f64::consts::PI * mx as f64 / lx;
    let ky = 2.0 * std::f64::consts::PI * my as f64 / ly;
    let eigenvalue = (kx * kx + ky * ky).powf(s);

    // Compute relative error
    let mut error_sum = 0.0;
    let mut norm_sum = 0.0;
    for i in 0..nx {
        for j in 0..ny {
            let expected = eigenvalue * u[[i, j]];
            error_sum += (result[[i, j]] - expected).abs();
            norm_sum += expected.abs();
        }
    }
    let rel_error = error_sum / norm_sum;

    assert!(
        rel_error < 1e-9,
        "Periodic 2D fractional Laplacian error {} too large",
        rel_error
    );
}
