//! Reference f64 Jacobi eigensolver wired through the shared flat-array scaffold.

use crate::jacobi_shared::{
    flatten_square_matrix, recommended_jacobi_iterations, symmetric_eigenvalues_with_backend,
};
use cd_kernel::error::AlgebraResult;

#[inline]
fn f64_rotation(app: f64, aqq: f64, apq: f64) -> (f64, f64) {
    let theta = 0.5 * (2.0 * apq).atan2(app - aqq);
    theta.sin_cos()
}

#[inline]
fn f64_diagonal_update(sin_t: f64, cos_t: f64, app: f64, apq: f64, aqq: f64) -> (f64, f64) {
    (
        cos_t * cos_t * app + 2.0 * sin_t * cos_t * apq + sin_t * sin_t * aqq,
        sin_t * sin_t * app - 2.0 * sin_t * cos_t * apq + cos_t * cos_t * aqq,
    )
}

/// Compute eigenvalues of a symmetric matrix using the plain f64 Jacobi lane.
///
/// This is the reference/oracle backend for cross-checking the higher-precision
/// x87 and double-double lanes through the same shared scaffold.
pub fn symmetric_eigenvalues_f64(matrix: &[Vec<f64>]) -> AlgebraResult<Vec<f64>> {
    let n = matrix.len();
    let flat = flatten_square_matrix(matrix)?;
    symmetric_eigenvalues_with_backend(
        &flat,
        n,
        recommended_jacobi_iterations(n),
        1.0e-12,
        f64_rotation,
        f64_diagonal_update,
        "f64",
    )
}

#[cfg(test)]
mod tests {
    use super::symmetric_eigenvalues_f64;

    #[test]
    fn test_reference_jacobi_diagonal_matrix() {
        let matrix = vec![vec![5.0, 0.0], vec![0.0, 3.0]];
        let eigs = symmetric_eigenvalues_f64(&matrix).unwrap();

        assert_eq!(eigs, vec![5.0, 3.0]);
    }

    #[test]
    fn test_reference_jacobi_known_spectrum() {
        let matrix = vec![
            vec![4.0, 1.0, 0.5],
            vec![1.0, 3.0, 0.2],
            vec![0.5, 0.2, 2.0],
        ];
        let eigs = symmetric_eigenvalues_f64(&matrix).unwrap();

        assert_eq!(eigs.len(), 3);
        assert!(eigs[0].abs() >= eigs[1].abs());
        assert!(eigs[1].abs() >= eigs[2].abs());
    }
}
