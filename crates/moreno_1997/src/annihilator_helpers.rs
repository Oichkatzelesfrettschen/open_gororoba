//! Minimal annihilator helpers used by the Moreno (1997) paper crate.
//!
//! These are kept local so the paper crate remains self-contained and does not
//! depend on the broader `algebra_analysis` surface.

use cd_kernel::cayley_dickson::cd_multiply;
use nalgebra::{DMatrix, SVD};

/// Build the dim x dim left multiplication matrix L_a: L_a * b = a * b.
pub fn left_multiplication_matrix(a: &[f64], dim: usize) -> DMatrix<f64> {
    assert_eq!(a.len(), dim);
    let mut mat = DMatrix::zeros(dim, dim);
    for i in 0..dim {
        let mut e = vec![0.0; dim];
        e[i] = 1.0;
        let col = cd_multiply(a, &e);
        for (r, &val) in col.iter().enumerate() {
            mat[(r, i)] = val;
        }
    }
    mat
}

/// Build the dim x dim right multiplication matrix R_a: R_a * b = b * a.
pub fn right_multiplication_matrix(a: &[f64], dim: usize) -> DMatrix<f64> {
    assert_eq!(a.len(), dim);
    let mut mat = DMatrix::zeros(dim, dim);
    for i in 0..dim {
        let mut e = vec![0.0; dim];
        e[i] = 1.0;
        let col = cd_multiply(&e, a);
        for (r, &val) in col.iter().enumerate() {
            mat[(r, i)] = val;
        }
    }
    mat
}

/// Compute an orthonormal basis for the right nullspace of `mat` using SVD.
pub fn nullspace_basis(mat: &DMatrix<f64>, atol: f64) -> DMatrix<f64> {
    let n = mat.ncols();
    let svd = SVD::new(mat.clone(), false, true);
    let singular = &svd.singular_values;

    let rank = singular.iter().filter(|&&s| s > atol).count();
    if rank == n {
        return DMatrix::zeros(n, 0);
    }

    let vt = svd.v_t.expect("SVD should compute V^T");
    let nullity = n - rank;
    let mut basis = DMatrix::zeros(n, nullity);
    for col in 0..nullity {
        let row_idx = rank + col;
        for r in 0..n {
            basis[(r, col)] = vt[(row_idx, r)];
        }
    }
    basis
}
