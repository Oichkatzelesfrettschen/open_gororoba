//! Block-Jacobi prototype for symmetric matrices.

use cd_kernel::error::{AlgebraError, AlgebraResult};
use nalgebra::{DMatrix, SymmetricEigen};

/// Compute the full spectrum of a symmetric matrix with a simple cyclic
/// block-Jacobi prototype.
pub fn symmetric_eigenvalues_block_jacobi(
    matrix: &[Vec<f64>],
    block_size: usize,
    max_sweeps: usize,
    tolerance: f64,
) -> AlgebraResult<Vec<f64>> {
    let n = matrix.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if block_size == 0 {
        return Err(AlgebraError::NumericalError(
            "block Jacobi requires block_size > 0".to_string(),
        ));
    }

    for row in matrix {
        if row.len() != n {
            return Err(AlgebraError::DimensionMismatch {
                expected: n,
                found: row.len(),
            });
        }
    }

    let mut a = DMatrix::from_fn(n, n, |i, j| matrix[i][j]);
    let mut converged = false;

    for _ in 0..max_sweeps.max(1) {
        if offdiag_fro_norm(&a) <= tolerance {
            converged = true;
            break;
        }

        for p_start in (0..n).step_by(block_size) {
            for q_start in ((p_start + block_size)..n).step_by(block_size) {
                let mut active = collect_block_indices(p_start, q_start, n, block_size);
                active.sort_unstable();
                active.dedup();
                if active.len() <= 1 {
                    continue;
                }

                let pivot = select_submatrix(&a, &active);
                let eig = SymmetricEigen::new(pivot);
                let mut g = DMatrix::<f64>::identity(n, n);
                for local_i in 0..active.len() {
                    for local_j in 0..active.len() {
                        g[(active[local_i], active[local_j])] =
                            eig.eigenvectors[(local_i, local_j)];
                    }
                }
                a = g.transpose() * a * g;
            }
        }
    }

    if !converged && offdiag_fro_norm(&a) > tolerance {
        return Err(AlgebraError::NumericalError(format!(
            "block Jacobi failed to converge after {max_sweeps} sweeps"
        )));
    }

    let mut eigs: Vec<f64> = (0..n).map(|i| a[(i, i)]).collect();
    eigs.sort_by(|lhs, rhs| {
        rhs.abs()
            .total_cmp(&lhs.abs())
            .then_with(|| rhs.total_cmp(lhs))
    });
    Ok(eigs)
}

fn collect_block_indices(
    p_start: usize,
    q_start: usize,
    n: usize,
    block_size: usize,
) -> Vec<usize> {
    let mut indices = Vec::with_capacity(2 * block_size);
    indices.extend(p_start..(p_start + block_size).min(n));
    indices.extend(q_start..(q_start + block_size).min(n));
    indices
}

fn select_submatrix(matrix: &DMatrix<f64>, indices: &[usize]) -> DMatrix<f64> {
    DMatrix::from_fn(indices.len(), indices.len(), |i, j| {
        matrix[(indices[i], indices[j])]
    })
}

fn offdiag_fro_norm(matrix: &DMatrix<f64>) -> f64 {
    let n = matrix.nrows();
    let mut sum = 0.0_f64;
    for i in 0..n {
        for j in 0..n {
            if i != j {
                let value = matrix[(i, j)];
                sum += value * value;
            }
        }
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::symmetric_eigenvalues_block_jacobi;

    #[test]
    fn block_jacobi_solves_diagonal_matrix() {
        let matrix = vec![vec![5.0, 0.0], vec![0.0, 3.0]];
        let eigs = symmetric_eigenvalues_block_jacobi(&matrix, 2, 4, 1.0e-12).unwrap();

        assert_eq!(eigs, vec![5.0, 3.0]);
    }

    #[test]
    fn block_jacobi_recovers_known_spectrum() {
        let matrix = vec![
            vec![4.0, 1.0, 0.5, 0.0],
            vec![1.0, 3.0, 0.2, 0.0],
            vec![0.5, 0.2, 2.0, 0.1],
            vec![0.0, 0.0, 0.1, 1.0],
        ];
        let eigs = symmetric_eigenvalues_block_jacobi(&matrix, 2, 16, 1.0e-10).unwrap();

        assert_eq!(eigs.len(), 4);
        assert!(eigs[0].abs() >= eigs[1].abs());
        assert!(eigs[1].abs() >= eigs[2].abs());
    }
}
