//! Multiplication-coupling analysis (Thesis D, C-466).
//!
//! For each basis element `b` in a Cayley-Dickson algebra of
//! dimension `dim`, the multiplication table defines a permutation
//! `mu_b(c) = index of e_b * e_c`. We ask: does the encoding
//! dictionary `Phi` intertwine this permutation with a *linear* map
//! on the subspace `V = span(Phi(0), ..., Phi(dim-1))`?
//!
//! Because the lattice vectors may span only `r < 8` dimensions (due
//! to filtration constraints like `l_0 = -1`), the test is done in the
//! `r`-dimensional reduced space and the `r x r` coupling matrix is
//! reported. Two variants are computed:
//!   * **unsigned**: `Phi(mu_b(c)) = M * Phi(c)` for all c
//!   * **signed**:   `sign(b,c) * Phi(mu_b(c)) = M * Phi(c)` for all c
//!
//! Public surface:
//!   * `BasisCouplingResult`        -- per-basis residual / det record
//!   * `MultiplicationCoupling`     -- full aggregate result
//!   * `compute_multiplication_coupling` -- pipeline entry point
//!
//! Re-exported from `codebook` via `pub use` so the canonical paths
//! `algebra_analysis::codebook::BasisCouplingResult` etc. remain
//! stable for external callers.

use super::{
    EncodingDictionary,
    linear_algebra::{
        build_square_matrix, det_nxn, find_pivot_columns_reduced, gram_schmidt_basis, invert_nxn,
        mat_mul_nxn, project_to_basis,
    },
};

/// Result of attempting to compute rho(b) for a single basis element.
#[derive(Debug, Clone)]
pub struct BasisCouplingResult {
    /// The basis element index b.
    pub basis_index: usize,
    /// Whether the unsigned coupling is consistent (residual < tolerance).
    pub unsigned_consistent: bool,
    /// Whether the signed coupling is consistent.
    pub signed_consistent: bool,
    /// Determinant of unsigned coupling matrix (in the reduced space).
    pub unsigned_det: Option<f64>,
    /// Determinant of signed coupling matrix (in the reduced space).
    pub signed_det: Option<f64>,
    /// Maximum absolute residual across all verification columns (unsigned).
    pub unsigned_max_residual: f64,
    /// Maximum absolute residual across all verification columns (signed).
    pub signed_max_residual: f64,
}

/// Full multiplication coupling analysis for all basis elements of a dictionary.
#[derive(Debug, Clone)]
pub struct MultiplicationCoupling {
    /// CD algebra dimension.
    pub dim: usize,
    /// Rank of the lattice vectors (dimension of the spanned subspace).
    pub rank: usize,
    /// Per-basis results (one per basis element 0..dim).
    pub results: Vec<BasisCouplingResult>,
    /// How many bases have consistent unsigned coupling.
    pub unsigned_consistent_count: usize,
    /// How many bases have consistent signed coupling.
    pub signed_consistent_count: usize,
    /// Determinants of all unsigned coupling matrices (for structure analysis).
    pub unsigned_dets: Vec<(usize, f64)>,
    /// Determinants of all signed coupling matrices.
    pub signed_dets: Vec<(usize, f64)>,
}

/// Compute the multiplication coupling for all basis elements.
pub fn compute_multiplication_coupling(
    dict: &EncodingDictionary,
    mult_table: &cd_kernel::mult_table::CdMultTable,
) -> MultiplicationCoupling {
    let dim = dict.dim();
    assert_eq!(
        dim, mult_table.dim,
        "dictionary and multiplication table dimensions must match"
    );

    let tol = 1e-8;

    let mut phi: Vec<[f64; 8]> = vec![[0.0; 8]; dim];
    for (idx, lv) in dict.iter() {
        for k in 0..8 {
            phi[idx][k] = lv[k] as f64;
        }
    }

    let (q_basis, rank) = gram_schmidt_basis(&phi);

    let phi_r: Vec<Vec<f64>> = phi.iter().map(|p| project_to_basis(p, &q_basis)).collect();

    let pivot_cols = find_pivot_columns_reduced(&phi_r, rank);
    assert_eq!(
        pivot_cols.len(),
        rank,
        "Expected {rank} pivot columns in reduced space, got {}",
        pivot_cols.len()
    );

    let x_r = build_square_matrix(&phi_r, &pivot_cols, rank);
    let x_r_inv = invert_nxn(&x_r, rank);

    let mut results = Vec::with_capacity(dim);
    let mut unsigned_dets = Vec::new();
    let mut signed_dets = Vec::new();

    for b in 0..dim {
        let mut unsigned_max_res = 0.0f64;
        let mut signed_max_res = 0.0f64;
        let mut u_det = None;
        let mut s_det = None;

        if let Some(ref xi) = x_r_inv {
            let mut y_u = vec![vec![0.0f64; rank]; rank];
            let mut y_s = vec![vec![0.0f64; rank]; rank];

            for (j, &c) in pivot_cols.iter().enumerate() {
                let (sign, prod_idx) = mult_table.multiply_basis(b, c);
                let out_r = &phi_r[prod_idx];
                for i in 0..rank {
                    y_u[i][j] = out_r[i];
                    y_s[i][j] = (sign as f64) * out_r[i];
                }
            }

            let m_u = mat_mul_nxn(&y_u, xi, rank);
            let m_s = mat_mul_nxn(&y_s, xi, rank);

            for (c, phi_rc) in phi_r.iter().enumerate() {
                let (sign, prod_idx) = mult_table.multiply_basis(b, c);
                let phi_r_prod = &phi_r[prod_idx];

                for i in 0..rank {
                    let predicted_u: f64 =
                        m_u[i].iter().zip(phi_rc.iter()).map(|(&m, &p)| m * p).sum();
                    unsigned_max_res = unsigned_max_res.max((predicted_u - phi_r_prod[i]).abs());

                    let predicted_s: f64 =
                        m_s[i].iter().zip(phi_rc.iter()).map(|(&m, &p)| m * p).sum();
                    let target_s = (sign as f64) * phi_r_prod[i];
                    signed_max_res = signed_max_res.max((predicted_s - target_s).abs());
                }
            }

            if unsigned_max_res < tol {
                u_det = Some(det_nxn(&m_u, rank));
            }
            if signed_max_res < tol {
                s_det = Some(det_nxn(&m_s, rank));
            }
        }

        let unsigned_consistent = unsigned_max_res < tol;
        let signed_consistent = signed_max_res < tol;

        if let Some(d) = u_det {
            unsigned_dets.push((b, d));
        }
        if let Some(d) = s_det {
            signed_dets.push((b, d));
        }

        results.push(BasisCouplingResult {
            basis_index: b,
            unsigned_consistent,
            signed_consistent,
            unsigned_det: u_det,
            signed_det: s_det,
            unsigned_max_residual: unsigned_max_res,
            signed_max_residual: signed_max_res,
        });
    }

    let unsigned_consistent_count = results.iter().filter(|r| r.unsigned_consistent).count();
    let signed_consistent_count = results.iter().filter(|r| r.signed_consistent).count();

    MultiplicationCoupling {
        dim,
        rank,
        results,
        unsigned_consistent_count,
        signed_consistent_count,
        unsigned_dets,
        signed_dets,
    }
}
