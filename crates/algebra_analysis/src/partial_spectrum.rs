//! Partial-spectrum solvers for symmetric matrices.

use crate::precision_policy::SpectrumObjective;
use cd_kernel::error::{AlgebraError, AlgebraResult};
use nalgebra::{DMatrix, DVector, SymmetricEigen};

/// Compute a partial spectrum for a symmetric matrix using deterministic block
/// subspace iteration with a Rayleigh-Ritz projection.
pub fn symmetric_extremal_eigenvalues(
    matrix: &[Vec<f64>],
    objective: SpectrumObjective,
    max_iterations: usize,
    tolerance: f64,
) -> AlgebraResult<Vec<f64>> {
    let n = matrix.len();
    if n == 0 {
        return Ok(Vec::new());
    }

    for row in matrix {
        if row.len() != n {
            return Err(AlgebraError::DimensionMismatch {
                expected: n,
                found: row.len(),
            });
        }
    }

    if matches!(objective, SpectrumObjective::FullSpectrum) {
        let dense = dense_matrix(matrix);
        let mut eigs: Vec<f64> = SymmetricEigen::new(dense)
            .eigenvalues
            .iter()
            .copied()
            .collect();
        sort_by_objective(&mut eigs, objective);
        return Ok(eigs);
    }

    let requested = objective.requested_count().min(n);
    if requested == 0 {
        return Ok(Vec::new());
    }

    let dense = dense_matrix(matrix);
    let fro_norm = dense.norm().max(1.0);
    let block_width = (2 * requested).max(4).min(n);
    let squared = &dense * &dense;
    let smallest_lu = match objective {
        SpectrumObjective::SmallestAbs { .. } => {
            let sigma = (1.0e-12 * fro_norm * fro_norm).max(1.0e-24);
            let shifted = squared.clone() + DMatrix::<f64>::identity(n, n) * sigma;
            Some(shifted.lu())
        }
        _ => None,
    };

    let mut q = orthonormalize_columns(&initial_subspace(n, block_width), tolerance)?;
    let mut selected = vec![0.0; requested];

    for _ in 0..max_iterations.max(1) {
        let z = match objective {
            SpectrumObjective::LargestAbs { .. } => &squared * &q,
            SpectrumObjective::SmallestAbs { .. } => smallest_lu
                .as_ref()
                .expect("guarded above")
                .solve(&q)
                .ok_or_else(|| {
                    AlgebraError::NumericalError(
                        "partial-spectrum shifted solve failed".to_string(),
                    )
                })?,
            SpectrumObjective::FullSpectrum => unreachable!("handled above"),
        };
        q = orthonormalize_columns(&z, tolerance)?;

        let projected = q.transpose() * &dense * &q;
        let eig = SymmetricEigen::new(projected);
        let ritz_vectors = &q * eig.eigenvectors;
        let ordering = ordered_indices(eig.eigenvalues.as_slice(), objective);

        let take = ordering.len().min(block_width);
        let ordered_basis = reorder_columns(&ritz_vectors, &ordering[..take]);
        q = orthonormalize_columns(&ordered_basis, tolerance)?;

        let mut max_residual = 0.0_f64;
        let mut new_selected = Vec::with_capacity(requested);
        for &idx in ordering.iter().take(requested) {
            let lambda = eig.eigenvalues[idx];
            let vector = ritz_vectors.column(idx).clone_owned();
            let residual = (&dense * &vector - vector.scale(lambda)).norm();
            max_residual = max_residual.max(residual);
            new_selected.push(lambda);
        }

        let stable = new_selected
            .iter()
            .zip(&selected)
            .all(|(lhs, rhs)| (lhs - rhs).abs() <= tolerance * fro_norm);
        selected = new_selected;

        if max_residual <= tolerance * fro_norm && stable {
            sort_by_objective(&mut selected, objective);
            return Ok(selected);
        }
    }

    sort_by_objective(&mut selected, objective);
    Ok(selected)
}

fn dense_matrix(matrix: &[Vec<f64>]) -> DMatrix<f64> {
    let n = matrix.len();
    DMatrix::from_fn(n, n, |i, j| matrix[i][j])
}

fn initial_subspace(n: usize, width: usize) -> DMatrix<f64> {
    DMatrix::from_fn(n, width, |row, col| {
        let x = (row + 1) as f64;
        let y = (col + 1) as f64;
        (x * y).sin() + (x * (y + 1.0)).cos()
    })
}

fn orthonormalize_columns(matrix: &DMatrix<f64>, tolerance: f64) -> AlgebraResult<DMatrix<f64>> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    let mut basis: Vec<DVector<f64>> = Vec::with_capacity(ncols);

    for col in 0..ncols {
        let mut vector = matrix.column(col).clone_owned();
        for basis_vector in &basis {
            let projection = basis_vector.dot(&vector);
            vector -= basis_vector * projection;
        }
        let norm = vector.norm();
        if norm > tolerance.max(1.0e-18) {
            basis.push(vector / norm);
        }
    }

    if basis.is_empty() {
        return Err(AlgebraError::NumericalError(
            "orthonormalization produced an empty basis".to_string(),
        ));
    }

    while basis.len() < ncols {
        let index = basis.len().min(nrows.saturating_sub(1));
        let mut vector = DVector::<f64>::zeros(nrows);
        vector[index] = 1.0;
        for basis_vector in &basis {
            let projection = basis_vector.dot(&vector);
            vector -= basis_vector * projection;
        }
        let norm = vector.norm();
        if norm <= tolerance.max(1.0e-18) {
            break;
        }
        basis.push(vector / norm);
    }

    let out_cols = basis.len();
    Ok(DMatrix::from_fn(nrows, out_cols, |i, j| basis[j][i]))
}

fn ordered_indices(values: &[f64], objective: SpectrumObjective) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&lhs, &rhs| compare_by_objective(values[lhs], values[rhs], objective));
    indices
}

fn reorder_columns(matrix: &DMatrix<f64>, ordering: &[usize]) -> DMatrix<f64> {
    DMatrix::from_fn(matrix.nrows(), ordering.len(), |row, col| {
        matrix[(row, ordering[col])]
    })
}

fn sort_by_objective(values: &mut [f64], objective: SpectrumObjective) {
    values.sort_by(|lhs, rhs| compare_by_objective(*lhs, *rhs, objective));
}

fn compare_by_objective(lhs: f64, rhs: f64, objective: SpectrumObjective) -> std::cmp::Ordering {
    match objective {
        SpectrumObjective::FullSpectrum | SpectrumObjective::LargestAbs { .. } => rhs
            .abs()
            .total_cmp(&lhs.abs())
            .then_with(|| rhs.total_cmp(&lhs)),
        SpectrumObjective::SmallestAbs { .. } => lhs
            .abs()
            .total_cmp(&rhs.abs())
            .then_with(|| lhs.total_cmp(&rhs)),
    }
}

#[cfg(test)]
mod tests {
    use super::symmetric_extremal_eigenvalues;
    use crate::precision_policy::SpectrumObjective;

    #[test]
    fn largest_abs_matches_diagonal_extremals() {
        let matrix = vec![
            vec![7.0, 0.0, 0.0, 0.0],
            vec![0.0, -5.0, 0.0, 0.0],
            vec![0.0, 0.0, 2.0, 0.0],
            vec![0.0, 0.0, 0.0, -1.0],
        ];

        let eigs = symmetric_extremal_eigenvalues(
            &matrix,
            SpectrumObjective::LargestAbs { k: 2 },
            32,
            1.0e-10,
        )
        .unwrap();

        assert_eq!(eigs.len(), 2);
        assert!((eigs[0] - 7.0).abs() < 1.0e-8);
        assert!((eigs[1] + 5.0).abs() < 1.0e-8);
    }

    #[test]
    fn smallest_abs_matches_diagonal_extremals() {
        let matrix = vec![
            vec![7.0, 0.0, 0.0, 0.0],
            vec![0.0, -5.0, 0.0, 0.0],
            vec![0.0, 0.0, 2.0, 0.0],
            vec![0.0, 0.0, 0.0, -1.0],
        ];

        let eigs = symmetric_extremal_eigenvalues(
            &matrix,
            SpectrumObjective::SmallestAbs { k: 2 },
            32,
            1.0e-10,
        )
        .unwrap();

        assert_eq!(eigs.len(), 2);
        assert!((eigs[0] + 1.0).abs() < 1.0e-8);
        assert!((eigs[1] - 2.0).abs() < 1.0e-8);
    }
}
