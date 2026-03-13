//! x87-backed Jacobi eigensolver wired through the shared flat-array scaffold.

#![cfg(target_arch = "x86_64")]

use crate::jacobi_shared::symmetric_eigenvalues_with_backend;
use cd_kernel::{
    AlgebraResult,
    x87_jacobi_kernels::{
        x87_givens_diagonal_update as diagonal_update_f64, x87_givens_sincos as givens_sincos_f64,
    },
};

// ── Flat-array Jacobi solver ────────────────────────────────────────────────

/// Compute eigenvalues of a symmetric matrix using Jacobi iteration with
/// x87 80-bit precision for Givens rotations.
pub fn symmetric_eigenvalues_x87(
    matrix: &[f64],
    n: usize,
    max_iterations: usize,
    tolerance: f64,
) -> AlgebraResult<Vec<f64>> {
    symmetric_eigenvalues_with_backend(
        matrix,
        n,
        max_iterations,
        tolerance,
        x87_rotation_f64,
        diagonal_update_f64,
        "x87",
    )
}

fn x87_rotation_f64(app: f64, aqq: f64, apq: f64) -> (f64, f64) {
    let y = 2.0 * apq;
    let x = app - aqq;
    givens_sincos_f64(y, x)
}

#[cfg(test)]
mod tests {
    use super::symmetric_eigenvalues_x87;

    #[test]
    fn test_x87_jacobi_diagonal_matrix() {
        let matrix = [5.0, 0.0, 0.0, 3.0];
        let eigs = symmetric_eigenvalues_x87(&matrix, 2, 64, 1.0e-14).unwrap();

        assert_eq!(eigs.len(), 2);
        assert!((eigs[0] - 5.0).abs() < 1.0e-14);
        assert!((eigs[1] - 3.0).abs() < 1.0e-14);
    }

    #[test]
    fn test_x87_jacobi_known_spectrum() {
        let matrix = [
            vec![4.0, 1.0, 0.5],
            vec![1.0, 3.0, 0.2],
            vec![0.5, 0.2, 2.0],
        ];
        let flat: Vec<f64> = matrix.iter().flat_map(|row| row.iter().copied()).collect();
        let eigs = symmetric_eigenvalues_x87(&flat, 3, 500, 1.0e-14).unwrap();

        assert_eq!(eigs.len(), 3);
        assert!(eigs[0].abs() >= eigs[1].abs());
        assert!(eigs[1].abs() >= eigs[2].abs());
    }
}
