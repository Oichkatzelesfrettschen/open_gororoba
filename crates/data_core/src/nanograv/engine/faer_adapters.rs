//! Adapters between `nalgebra` and `faer` matrix types plus a few
//! solver primitives used by the joint-fit covariance pipeline.
//!
//! Functions:
//!   * `dmatrix_to_faer`            -- nalgebra DMatrix -> faer Mat
//!   * `dvector_to_faer_col`        -- nalgebra DVector -> faer single-column Mat
//!   * `faer_col_to_dvector`        -- inverse of the above
//!   * `faer_column_norms`          -- 2-norm of each column, floored at 1.0
//!   * `scale_faer_columns`         -- scale each column by a vector
//!     (or its reciprocal when `inverse` is true)
//!   * `solve_square_system_faer`   -- try Cholesky (LLT), fall back to
//!     column-pivoted QR with least-squares
//!   * `apply_inverse_covariance_to_matrix` -- Woodbury-style
//!     application of (D + U V^T)^{-1} to a RHS matrix
//!
//! All items `pub(super)`.

use anyhow::Result;
use faer::{
    Mat as FaerMat, Side,
    prelude::{Solve, SolveLstsq},
};
use nalgebra::{DMatrix, DVector};

use super::StructuredCovariance;

pub(super) fn dmatrix_to_faer(matrix: &DMatrix<f64>) -> FaerMat<f64> {
    FaerMat::from_fn(matrix.nrows(), matrix.ncols(), |row, col| {
        matrix[(row, col)]
    })
}

pub(super) fn dvector_to_faer_col(vector: &DVector<f64>) -> FaerMat<f64> {
    FaerMat::from_fn(vector.len(), 1, |row, _| vector[row])
}

pub(super) fn faer_col_to_dvector(matrix: &FaerMat<f64>) -> DVector<f64> {
    DVector::from_iterator(
        matrix.nrows(),
        (0..matrix.nrows()).map(|row| matrix[(row, 0)]),
    )
}

pub(super) fn faer_column_norms(matrix: &FaerMat<f64>) -> Vec<f64> {
    (0..matrix.ncols())
        .map(|col| {
            let sumsq = (0..matrix.nrows())
                .map(|row| matrix[(row, col)] * matrix[(row, col)])
                .sum::<f64>();
            sumsq.sqrt().max(1.0)
        })
        .collect()
}

pub(super) fn scale_faer_columns(matrix: &mut FaerMat<f64>, scales: &[f64], inverse: bool) {
    for col in 0..matrix.ncols() {
        let factor = if inverse {
            1.0 / scales[col]
        } else {
            scales[col]
        };
        for row in 0..matrix.nrows() {
            matrix[(row, col)] *= factor;
        }
    }
}

pub(super) fn solve_square_system_faer(
    system: &FaerMat<f64>,
    rhs: &FaerMat<f64>,
) -> Result<FaerMat<f64>> {
    if let Ok(cholesky) = system.llt(Side::Lower) {
        return Ok(cholesky.solve(rhs));
    }
    let qr = system.col_piv_qr();
    Ok(qr.solve_lstsq(rhs))
}

pub(super) fn apply_inverse_covariance_to_matrix(
    covariance: &StructuredCovariance,
    rhs: &FaerMat<f64>,
) -> Result<FaerMat<f64>> {
    let mut scaled = rhs.clone();
    for row in 0..scaled.nrows() {
        let factor = covariance.inv_diagonal[row];
        for col in 0..scaled.ncols() {
            scaled[(row, col)] *= factor;
        }
    }
    if covariance.low_rank.ncols() == 0 {
        return Ok(scaled);
    }
    let projected = covariance.low_rank.transpose() * &scaled;
    let solved = solve_square_system_faer(&covariance.middle, &projected)?;
    let correction = &covariance.inv_low_rank * solved;
    Ok(scaled - correction)
}
