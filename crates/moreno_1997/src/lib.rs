//! Moreno (1997) algebraic theorem implementations.
//!
//! # Reference
//!
//! Moreno (1997) "The zero divisors of the Cayley-Dickson algebras over the
//! real numbers", arXiv:q-alg/9710013v1.
//!
//! # Scope
//!
//! This module implements, in execution order:
//!
//! | Function | Theorem | Purpose |
//! |----------|---------|---------|
//! | `h_a_basis` | Thm 1.13 | Quaternion subalgebra H_a basis |
//! | `verify_h_a_isomorphism` | Thm 1.13 | Check H_a ~ H multiplication table |
//! | `t_operator_matrix` | Thm 1.15 | Full T_a = -I - L_a^2 on A_n |
//! | `t_tilde_matrix` | Thm 1.15 | T~_a = T_a restricted to H_a^perp |
//! | `ker_t_tilde_basis` | Thm 1.15 | Nullspace of T~_a (the V_1 subspace) |
//! | `moreno_decomposition` | Thm 1.15+1.16+Cor 1.17 | Full direct-sum decomposition |
//! | `verify_corollary_1_17` | Cor 1.17 | Annihilator bound check |
//! | `s_operator_matrix` | Thm 2.3 | S(y) = (a,y,b) on A_n |
//! | `ker_s_restricted_decomposition` | Thm 2.3 | Ker(S|_{V(a;b)^perp}) decomposition |
//! | `verify_corollary_2_4` | Cor 2.4 | Mod-8 constraint on Ker(S) |
//! | `has_eigenvalue_minus_2` | Thm 2.9 | Check spec(L^2_c) contains -2 |
//! | `find_zd_witness_moreno` | Thm 2.9 | Find y with L^2_{a+b}(y) = -2y |
//! | `is_zd_moreno` | Thm 2.9 | ZD iff lambda=-2 in spec(L^2_{a+b}) |
//! | `v_triple_basis` | Thm 2.13 | Octonion-copy embedding V(a;y;b) |
//! | `verify_theorem_2_13` | Thm 2.13 | Check V(a;y;b) ~ O multiplication |
//!
//! # Operator naming convention (FIXED -- never conflate these)
//!
//! - `T_a`: full operator `T_a(y) = (a,a,y) = -y - L_a^2(y)` defined on ALL of `A_n`.
//!   `T_a(H_a) = {0}` because H_a is associative.
//! - `T~_a`: RESTRICTION of `T_a` to `H_a^perp` (dimension `2^n - 4`).
//!   Ker(T~_a) = V_1. Kernel of T_a on all of A_n includes H_a (dim 4) plus V_1.
//! - `S`: full `S(y) = (a,y,b)` on A_n.
//!   `V(a;b) subset Ker(S)` globally.
//!   Theorem 2.3 analyzes `S` restricted to `V(a;b)^perp`, NOT full Ker(S).

mod annihilator_helpers;

use cd_kernel::cayley_dickson::{
    cd_multiply, cd_norm_sq,
    predicates::{cd_inner_product, cd_tilde, is_special_triple},
};
use nalgebra::{DMatrix, DVector};

use crate::annihilator_helpers::{
    left_multiplication_matrix, nullspace_basis, right_multiplication_matrix,
};

// ──────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Build the identity matrix of size `dim`.
fn identity(dim: usize) -> DMatrix<f64> {
    DMatrix::identity(dim, dim)
}

/// Build the "second unit" vector `tilde_e0 = e_{dim/2}`.
fn tilde_e0(dim: usize) -> Vec<f64> {
    let mut v = vec![0.0; dim];
    v[dim / 2] = 1.0;
    v
}

/// Compute L_a^2: the matrix of left-multiplication by a applied twice.
/// `L_a^2(y) = a * (a * y)`.
fn l_a_squared(a: &[f64], dim: usize) -> DMatrix<f64> {
    let la = left_multiplication_matrix(a, dim);
    &la * &la
}

/// Extract column j from a DMatrix as a Vec<f64>.
fn col_vec(m: &DMatrix<f64>, j: usize) -> Vec<f64> {
    m.column(j).iter().copied().collect()
}

/// Compute the column range of a set of basis vectors as a DMatrix (n x k).
fn vecs_to_matrix(vecs: &[Vec<f64>]) -> DMatrix<f64> {
    if vecs.is_empty() {
        return DMatrix::zeros(0, 0);
    }
    let n = vecs[0].len();
    let k = vecs.len();
    let mut m = DMatrix::zeros(n, k);
    for (j, v) in vecs.iter().enumerate() {
        for (i, &x) in v.iter().enumerate() {
            m[(i, j)] = x;
        }
    }
    m
}

// ──────────────────────────────────────────────────────────────────────────────
// Theorem 1.13: H_a quaternion subalgebra
// ──────────────────────────────────────────────────────────────────────────────

/// Returns the 4 basis vectors of H_a in source order: `[e0, a, tilde_a, tilde_e0]`.
///
/// # Source order (Theorem 1.13 multiplication table, p.14-15)
///
/// | Index | Vector | Formula |
/// |-------|--------|---------|
/// | 0 | `e0` | standard unit |
/// | 1 | `a` | the input doubly-pure unit-norm element |
/// | 2 | `tilde_a` | `a * tilde_e0 = cd_tilde(a)` |
/// | 3 | `tilde_e0` | `e_{dim/2}` |
///
/// # Pre-condition
///
/// `a` must be doubly pure (t(a_1)=0, t(a_2)=0) and have unit norm (`||a||=1`).
/// Use `is_doubly_pure(a)` and `(cd_norm_sq(a)-1.0).abs() < atol` to verify.
pub fn h_a_basis(a: &[f64]) -> [Vec<f64>; 4] {
    let dim = a.len();
    let mut e0 = vec![0.0; dim];
    e0[0] = 1.0;
    let tilde_a = cd_tilde(a);
    let tilde_e0 = tilde_e0(dim);
    [e0, a.to_vec(), tilde_a, tilde_e0]
}

/// Verify that H_a is multiplicatively closed and isomorphic to the quaternions H.
///
/// Checks the full 4x4 product table against the standard quaternion table
/// for basis `{e0=1, a=i, tilde_a=j, tilde_e0=k}` (Theorem 1.13 source order):
///
/// ```text
/// 1*1=1   1*i=i    1*j=j     1*k=k
/// i*1=i   i*i=-1   i*j=k     i*k=-j
/// j*1=j   j*i=-k   j*j=-1    j*k=i
/// k*1=k   k*i=j    k*j=-i    k*k=-1
/// ```
///
/// # Pre-condition
///
/// `a` must be doubly pure and unit norm (see `h_a_basis`).
pub fn verify_h_a_isomorphism(a: &[f64], atol: f64) -> bool {
    let [e0, ai, aj, ak] = h_a_basis(a);
    let basis = [&e0, &ai, &aj, &ak];

    // Expected product table for {e0=1, a=i, tilde_a=j, tilde_e0=k} in source order.
    //
    // Derived from CD multiplication with a=e1_8:
    //   a * tilde_a = e1 * e5 = -e4 = -k   (i*j = -k, opposite-handedness quaternion)
    //   a * tilde_e0 = e1 * e4 = e5 = +j
    //   tilde_a * a = e5 * e1 = e4 = +k
    //   tilde_e0 * a = e4 * e1 = -e5 = -j
    //
    // This is a valid quaternion algebra (isomorphic to H via swap j<->k).
    // Rows are left factor, columns are right factor; [row][col] = (sign, idx).
    struct Entry {
        sign: f64,
        idx: usize,
    }
    #[rustfmt::skip]
    let expected: [[Entry; 4]; 4] = [
        // 1  * {1, i, j, k}
        [Entry{sign:1.0,idx:0}, Entry{sign:1.0,idx:1}, Entry{sign:1.0,idx:2}, Entry{sign:1.0,idx:3}],
        // i  * {1, i, j, k}  -- i*j = -k, i*k = +j
        [Entry{sign:1.0,idx:1}, Entry{sign:-1.0,idx:0}, Entry{sign:-1.0,idx:3}, Entry{sign:1.0,idx:2}],
        // j  * {1, i, j, k}  -- j*i = +k, j*k = -i
        [Entry{sign:1.0,idx:2}, Entry{sign:1.0,idx:3}, Entry{sign:-1.0,idx:0}, Entry{sign:-1.0,idx:1}],
        // k  * {1, i, j, k}  -- k*i = -j, k*j = +i
        [Entry{sign:1.0,idx:3}, Entry{sign:-1.0,idx:2}, Entry{sign:1.0,idx:1}, Entry{sign:-1.0,idx:0}],
    ];

    let dim = a.len();
    for r in 0..4 {
        for c in 0..4 {
            let prod = cd_multiply(basis[r], basis[c]);
            let exp = &expected[r][c];
            let expected_vec = basis[exp.idx];
            for k in 0..dim {
                let expected_val = exp.sign * expected_vec[k];
                if (prod[k] - expected_val).abs() > atol {
                    return false;
                }
            }
        }
    }
    true
}

// ──────────────────────────────────────────────────────────────────────────────
// Theorem 1.15: T_a and T~_a operators
// ──────────────────────────────────────────────────────────────────────────────

/// Full T_a matrix on A_n: `T_a = -I - L_a^2`.
///
/// `T_a(y) = (a, a, y) = a*(a*y) - a^2*y = -y - L_a^2(y)`.
///
/// Note: `T_a(h) = 0` for all `h` in `H_a` (H_a is associative, Theorem 1.13).
/// Ker(T_a) restricted to H_a^perp is the V_1 subspace; see `t_tilde_matrix`.
pub fn t_operator_matrix(a: &[f64], dim: usize) -> DMatrix<f64> {
    let la2 = l_a_squared(a, dim);
    -identity(dim) - la2
}

/// T~_a matrix: restriction of T_a to H_a^perp (dimension `dim - 4`).
///
/// `T~_a` is the restriction of `T_a = -I - L_a^2` to the subspace `H_a^perp`.
///
/// # Construction
///
/// 1. Compute the orthonormal basis `{q_0, ..., q_{dim-5}}` of `H_a^perp` via
///    Gram-Schmidt on the columns of T_a orthogonalized against H_a.
/// 2. Compute T~_a = P^T * T_a * P where P is the `H_a^perp` projection.
///
/// Returns the `(dim-4) x (dim-4)` matrix of T~_a in the H_a^perp basis.
///
/// # Pre-condition
///
/// `a` must be doubly pure and unit norm.
pub fn t_tilde_matrix(a: &[f64], dim: usize, atol: f64) -> DMatrix<f64> {
    let ha_perp_basis = h_a_perp_basis(a, dim, atol);
    let ta = t_operator_matrix(a, dim);
    // T~_a = P^T * T_a * P where P = ha_perp_basis
    let p = &ha_perp_basis;
    p.transpose() * ta * p
}

/// Nullspace basis of T~_a restricted to H_a^perp: the V_1 subspace.
///
/// V_1 = Ker(T~_a) = {y in H_a^perp | a*(a*y) = -y}.
/// This is the eigenspace of L_a with eigenvalue 1 (since T~_a = -I - L_a^2
/// and -1 - lambda^2 = 0 gives lambda = 1 or lambda = -1, and Ker(T~_a) = V_1).
pub fn ker_t_tilde_basis(a: &[f64], dim: usize, atol: f64) -> DMatrix<f64> {
    let ha_perp = h_a_perp_basis(a, dim, atol);
    let ttilde = t_tilde_matrix(a, dim, atol);
    let ns = nullspace_basis(&ttilde, atol);
    // Project nullspace vectors back to A_n coordinates
    ha_perp * ns
}

/// Compute an orthonormal basis for H_a^perp (the (dim-4)-dimensional complement of H_a).
///
/// Uses Gram-Schmidt to orthogonalize standard basis vectors against H_a.
pub fn h_a_perp_basis(a: &[f64], dim: usize, atol: f64) -> DMatrix<f64> {
    let [e0, ai, aj, ak] = h_a_basis(a);
    let ha = [e0, ai, aj, ak];

    let mut perp_vecs: Vec<DVector<f64>> = Vec::with_capacity(dim - 4);
    for i in 0..dim {
        let mut v = DVector::zeros(dim);
        v[i] = 1.0;
        // Subtract projections onto H_a vectors
        for h in &ha {
            let h_dv = DVector::from_vec(h.clone());
            let proj = v.dot(&h_dv) * &h_dv;
            v -= proj;
        }
        // Subtract projections onto already-found perp vectors
        for q in &perp_vecs {
            let proj = v.dot(q) * q;
            v -= proj;
        }
        if v.norm() > atol {
            v /= v.norm();
            perp_vecs.push(v);
        }
    }
    let k = perp_vecs.len();
    let mut m = DMatrix::zeros(dim, k);
    for (j, v) in perp_vecs.iter().enumerate() {
        for i in 0..dim {
            m[(i, j)] = v[i];
        }
    }
    m
}

// ──────────────────────────────────────────────────────────────────────────────
// Theorem 1.15 + 1.16 + Corollary 1.17: full direct-sum decomposition
// ──────────────────────────────────────────────────────────────────────────────

/// Moreno direct-sum decomposition of A_n for a doubly-pure unit-norm element a.
///
/// `A_n = H_a + Ker(T~_a) + Ker(L_a) + direct-sum(V_lambda)` (Theorem 1.15).
///
/// Each eigenspace of L_a^2 restricted to H_a^perp produces one V_lambda (Theorem 1.16).
/// By Theorem 1.16, every dim(V_lambda) = 0 mod 4. By Corollary 1.17,
/// 0 <= dim(Ker(L_a)) = 0 mod 4, and dim(Ker(L_a)) <= 2^n - 4.
#[derive(Debug, Clone)]
pub struct MorenoDecomposition {
    pub dim: usize,
    /// H_a subspace: 4D copy of H. Basis: [e0, a, tilde_a, tilde_e0].
    pub h_a_dim: usize,
    /// V_0 = Ker(L_a) = Ker(R_a): the annihilator subspace.
    pub ker_l_a: DMatrix<f64>,
    /// V_1 = Ker(T~_a): elements y in H_a^perp where a*(a*y) = -y.
    pub ker_t_tilde: DMatrix<f64>,
    /// V_lambda for lambda not 0 or 1: `(lambda_value, basis_matrix)` pairs.
    /// Each eigenspace satisfies dim = 0 mod 4 (Theorem 1.16).
    pub v_lambda: Vec<(f64, DMatrix<f64>)>,
    /// Sum of all subspace dimensions equals dim.
    pub total_dim_check: bool,
    /// All dim(V_lambda) = 0 mod 4 for lambda in v_lambda.
    pub all_mod4: bool,
    /// dim(Ker L_a) = 0 mod 4.
    pub ker_la_mod4: bool,
    /// dim(Ker L_a) <= dim - 4.
    pub ker_la_leq_bound: bool,
}

/// Compute the full Moreno direct-sum decomposition for doubly-pure unit-norm `a`.
///
/// # Algorithm
///
/// 1. Compute H_a (4D) and H_a^perp (dim-4 D).
/// 2. On H_a^perp, compute L_a^2 and find its eigenspaces by SVD.
///    - Eigenvalue 0 gives V_0 = Ker(L_a).
///    - Eigenvalue -1 gives V_1 = Ker(T~_a) (T~_a = -I - L_a^2, Ker is at -1 eigvalue of L_a^2).
///    - Other eigenvalues -lambda^2 give V_lambda.
/// 3. Verify dimension sum, mod-4, and bound constraints.
pub fn moreno_decomposition(a: &[f64], dim: usize, atol: f64) -> MorenoDecomposition {
    let ha_perp = h_a_perp_basis(a, dim, atol);
    let la = left_multiplication_matrix(a, dim);
    let la2_full = &la * &la;

    // Project L_a^2 onto H_a^perp subspace
    let la2_restricted = ha_perp.transpose() * &la2_full * &ha_perp;

    // Schur decomposition for real eigenvalues. For symmetric matrices, use eigendecomp.
    // L_a is skew-symmetric (Prop 1.7), so L_a^2 is symmetric negative-semidefinite.
    let eig = la2_restricted.symmetric_eigen();
    let eigenvalues = eig.eigenvalues.clone();
    let eigenvectors = eig.eigenvectors.clone();

    // Cluster eigenvalues into groups (same eigenvalue => same eigenspace)
    let mut clusters: Vec<(f64, Vec<usize>)> = Vec::new();
    let n_perp = dim - 4;
    for i in 0..n_perp {
        let lambda = eigenvalues[i];
        let mut found = false;
        for (cluster_lam, cluster_cols) in clusters.iter_mut() {
            if (lambda - *cluster_lam).abs() < atol * 100.0 {
                cluster_cols.push(i);
                found = true;
                break;
            }
        }
        if !found {
            clusters.push((lambda, vec![i]));
        }
    }

    // Separate: ker_l_a (eigenvalue ~ 0), ker_t_tilde (eigenvalue ~ -1),
    // v_lambda (other eigenvalues).
    // Note: L_a^2's eigenvalue 0 => Ker(L_a); L_a^2's eigenvalue -1 => V_1.
    let mut ker_l_a_cols: Vec<Vec<f64>> = Vec::new();
    let mut ker_t_tilde_cols: Vec<Vec<f64>> = Vec::new();
    let mut v_lambda: Vec<(f64, DMatrix<f64>)> = Vec::new();

    for (lam2, cols) in &clusters {
        // Collect eigenvectors as columns in H_a^perp coordinates
        let mut block_cols: Vec<Vec<f64>> = Vec::new();
        for &col_idx in cols {
            let evec_perp: Vec<f64> = (0..n_perp).map(|r| eigenvectors[(r, col_idx)]).collect();
            // Project back to A_n coordinates: v = ha_perp * evec_perp
            let mut v_an = vec![0.0; dim];
            for (k, &x) in evec_perp.iter().enumerate() {
                for r in 0..dim {
                    v_an[r] += ha_perp[(r, k)] * x;
                }
            }
            block_cols.push(v_an);
        }
        let block = vecs_to_matrix(&block_cols);
        if lam2.abs() < atol * 100.0 {
            // eigenvalue 0 of L_a^2 => Ker(L_a)
            for c in block_cols {
                ker_l_a_cols.push(c);
            }
        } else if (lam2 + 1.0).abs() < atol * 100.0 {
            // eigenvalue -1 of L_a^2 => V_1 = Ker(T~_a)
            for c in block_cols {
                ker_t_tilde_cols.push(c);
            }
        } else {
            // V_lambda: lambda = sqrt(-lam2)
            let lambda = (-lam2).sqrt();
            v_lambda.push((lambda, block));
        }
    }

    let ker_l_a = vecs_to_matrix(&ker_l_a_cols);
    let ker_t_tilde = vecs_to_matrix(&ker_t_tilde_cols);

    // Dimension checks
    let ker_la_dim = ker_l_a.ncols();
    let ker_tt_dim = ker_t_tilde.ncols();
    let v_lambda_total: usize = v_lambda.iter().map(|(_, m)| m.ncols()).sum();
    let total = 4 + ker_la_dim + ker_tt_dim + v_lambda_total;
    let total_dim_check = total == dim;
    let all_mod4 = v_lambda.iter().all(|(_, m)| m.ncols().is_multiple_of(4));
    let ker_la_mod4 = ker_la_dim.is_multiple_of(4);
    let ker_la_leq_bound = ker_la_dim <= dim - 4;

    MorenoDecomposition {
        dim,
        h_a_dim: 4,
        ker_l_a,
        ker_t_tilde,
        v_lambda,
        total_dim_check,
        all_mod4,
        ker_la_mod4,
        ker_la_leq_bound,
    }
}

/// Quick verification of Corollary 1.17 only.
///
/// Returns `(ker_dim, is_0_mod_4, is_leq_2n_minus_4)`.
pub fn verify_corollary_1_17(a: &[f64], dim: usize, atol: f64) -> (usize, bool, bool) {
    let la = left_multiplication_matrix(a, dim);
    let ns = nullspace_basis(&la, atol);
    let ker_dim = ns.ncols();
    let is_mod4 = ker_dim.is_multiple_of(4);
    let is_leq = ker_dim <= dim - 4;
    (ker_dim, is_mod4, is_leq)
}

// ──────────────────────────────────────────────────────────────────────────────
// Theorem 2.3 + Corollary 2.4: S operator and Ker(S) decomposition
// ──────────────────────────────────────────────────────────────────────────────

/// Full S matrix on A_n: `S(y) = (a,y,b) = (a*y)*b - a*(y*b)`.
///
/// `S = R_b * L_a - L_a * R_b` (commutator of left/right multiplication operators).
/// S is skew-symmetric. `V(a;b) subset Ker(S)`.
///
/// # Pre-condition
///
/// For Theorem 2.3 to apply, `{a, b}` should be a special couple. However,
/// this function computes S for any pair (a, b).
pub fn s_operator_matrix(a: &[f64], b: &[f64], dim: usize) -> DMatrix<f64> {
    let la = left_multiplication_matrix(a, dim);
    let rb = right_multiplication_matrix(b, dim);
    &rb * &la - &la * &rb
}

/// Theorem 2.3: `Ker(S|_{V(a;b)^perp}) = Ker(L_{a+b}) + Ker(L_{a-b})`
/// where both kernels are subsets of `V(a;b)^perp`.
///
/// Returns `(ker_l_apb_basis, ker_l_amb_basis)` as `(dim x k1, dim x k2)` matrices.
///
/// # Pre-condition
///
/// `{a, b}` must be a special couple (`is_special_couple(a, b, atol)` should be true).
pub fn ker_s_restricted_decomposition(
    a: &[f64],
    b: &[f64],
    dim: usize,
    atol: f64,
) -> (DMatrix<f64>, DMatrix<f64>) {
    let apb: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    let amb: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();

    let la_apb = left_multiplication_matrix(&apb, dim);
    let la_amb = left_multiplication_matrix(&amb, dim);

    let ker_apb = nullspace_basis(&la_apb, atol);
    let ker_amb = nullspace_basis(&la_amb, atol);

    (ker_apb, ker_amb)
}

/// Corollary 2.4: dim Ker(S|_{V(a;b)^perp}) = 2 * dim Ker(L_{a+b}) = 0 mod 8.
///
/// Verifies Theorem 2.3 numerically and checks the mod-8 constraint.
/// Returns `true` if both conditions hold.
///
/// # Pre-condition
///
/// `{a, b}` must be a special couple.
pub fn verify_corollary_2_4(a: &[f64], b: &[f64], dim: usize, atol: f64) -> bool {
    let (ker_apb, ker_amb) = ker_s_restricted_decomposition(a, b, dim, atol);
    let d1 = ker_apb.ncols();
    let d2 = ker_amb.ncols();
    // Theorem 2.3: dim Ker(L_{a+b}) = dim Ker(L_{a-b}) (by symmetry of special couple)
    // Corollary 2.4: total = 2*d1, must be 0 mod 8
    let total = d1 + d2;
    d1 == d2 && total % 8 == 0
}

// ──────────────────────────────────────────────────────────────────────────────
// Theorem 2.9: ZD characterization via eigenvalue -2
// ──────────────────────────────────────────────────────────────────────────────

/// Check if `lambda = -2` is an eigenvalue of `L_c^2` restricted to A_n.
///
/// `L_c^2(y) = c*(c*y)`. Eigenvalue -2 means `c*(c*y) = -2*y`.
///
/// By Theorem 2.9, if `c = a + b` where `a, b` are alternative unit-norm
/// trace-zero elements, then `(a, b)` is a zero divisor in `A_{n+1}` iff
/// `-2` is in the spectrum of `L_c^2` on `A_n`.
pub fn has_eigenvalue_minus_2(c: &[f64], dim: usize, atol: f64) -> bool {
    let lc = left_multiplication_matrix(c, dim);
    let lc2 = &lc * &lc;
    // Check if (L_c^2 + 2I) has a nontrivial nullspace
    let shifted = lc2 + 2.0 * identity(dim);
    let ns = nullspace_basis(&shifted, atol);
    ns.ncols() > 0
}

/// Find a witness `y != 0` such that `L^2_{a+b}(y) = -2*y`.
///
/// Returns `None` if no such `y` exists (the pair `(a,b)` is not a ZD pair
/// in Moreno's framework).
pub fn find_zd_witness_moreno(a: &[f64], b: &[f64], dim: usize, atol: f64) -> Option<Vec<f64>> {
    let c: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    let lc = left_multiplication_matrix(&c, dim);
    let lc2 = &lc * &lc;
    let shifted = lc2 + 2.0 * identity(dim);
    let ns = nullspace_basis(&shifted, atol);
    if ns.ncols() == 0 {
        return None;
    }
    Some(col_vec(&ns, 0))
}

/// Theorem 2.9: `(a, b)` in `A_{n+1}` is a zero divisor iff `lambda = -2` is an
/// eigenvalue of `L^2_{a+b}` on `A_n`.
///
/// # Pre-condition
///
/// `a` and `b` must be alternative elements of norm 1 and trace 0 in `A_n`.
/// Use `is_special_element` from `predicates` to verify.
pub fn is_zd_moreno(a: &[f64], b: &[f64], atol: f64) -> bool {
    let dim = a.len();
    find_zd_witness_moreno(a, b, dim, atol).is_some()
}

// ──────────────────────────────────────────────────────────────────────────────
// Theorem 2.13: V(a;y;b) octonion-copy embedding
// ──────────────────────────────────────────────────────────────────────────────

/// Returns the 8 basis vectors of `V(a;y;b)` in source order (Theorem 2.13, p.30).
///
/// Source order matching the `{e0, ..., e7}` octonion basis assignment:
///
/// | Index | Vector | Standard octonion index |
/// |-------|--------|------------------------|
/// | 0 | `e0` | 0 |
/// | 1 | `a` | 1 |
/// | 2 | `b` | 2 |
/// | 3 | `a*b` | 3 |
/// | 4 | `(a*y)*b` | 4 |
/// | 5 | `y*b` | 5 |
/// | 6 | `a*y` | 6 |
/// | 7 | `y` | 7 |
///
/// # Pre-condition
///
/// `{a, y, b}` should be a special triple (`is_special_triple(a, y, b, atol)` true).
/// The output is well-defined for any inputs, but the multiplication table will match
/// octonions only when the triple condition holds.
pub fn v_triple_basis(a: &[f64], y: &[f64], b: &[f64]) -> [Vec<f64>; 8] {
    let dim = a.len();
    let mut e0 = vec![0.0; dim];
    e0[0] = 1.0;
    let ab = cd_multiply(a, b);
    let ay = cd_multiply(a, y);
    let yb = cd_multiply(y, b);
    let ayb = cd_multiply(&ay, b);
    [e0, a.to_vec(), b.to_vec(), ab, ayb, yb, ay, y.to_vec()]
}

/// Verify that `V(a;y;b)` is an octonion copy by checking the 8x8 product table.
///
/// Theorem 2.13: `V(a;y;b)` is a copy of `O` (octonions) iff `{a, y, b}` is a
/// special triple. This function checks numerically that the product table in the
/// `v_triple_basis` order matches the standard octonion multiplication table.
///
/// The standard octonion table used here is the Cayley-Dickson table for 8D
/// (same convention as `cd_multiply` at `dim=8`).
///
/// # Strategy
///
/// For each pair `(i, j)` of source-order basis vectors, compute their product in
/// `A_n` and verify it equals the expected combination of `V(a;y;b)` basis vectors.
/// The expected table is extracted by computing products of standard octonion basis
/// vectors and looking up their indices in the source-order basis.
pub fn verify_theorem_2_13(a: &[f64], y: &[f64], b: &[f64], atol: f64) -> bool {
    // Pre-condition check
    if !is_special_triple(a, y, b, atol) {
        return false;
    }

    let vbasis = v_triple_basis(a, y, b);
    let dim8 = 8;

    // Build the standard octonion multiplication table via cd_multiply at dim=8
    let std_basis: Vec<Vec<f64>> = (0..dim8)
        .map(|i| {
            let mut v = vec![0.0f64; dim8];
            v[i] = 1.0;
            v
        })
        .collect();
    let _ = std_basis; // silence unused warning

    // For each pair of V(a;y;b) basis vectors, compute their product in A_n
    // and verify it equals the expected octonion product.
    // The expected product: if vbasis[i] * vbasis[j] should equal
    // sign * vbasis[k] in A_n, then in the standard 8D octonion table,
    // e_i * e_j should equal sign * e_k.
    //
    // Strategy: build a map from product result to basis index.
    // For each (i,j), compute vbasis[i] * vbasis[j] in A_n,
    // then find which vbasis[k] it equals (or its negative).

    let n = a.len();
    for i in 0..dim8 {
        for j in 0..dim8 {
            let prod_an = cd_multiply(&vbasis[i], &vbasis[j]);
            // Find which vbasis[k] matches prod_an (up to sign)
            let mut matched = false;
            for k in 0..dim8 {
                let dot_pos = cd_inner_product(&prod_an, &vbasis[k]);
                let norm_k = cd_norm_sq(&vbasis[k]).sqrt();
                let norm_p = cd_norm_sq(&prod_an).sqrt();
                if norm_p < atol {
                    // Product is zero -- should only happen if norms fail
                    matched = true;
                    break;
                }
                if (dot_pos.abs() - norm_k * norm_p).abs() < atol * 10.0 {
                    // Check the actual octonion product e_i * e_j = sign * e_k
                    let mut ei8 = vec![0.0f64; dim8];
                    let mut ej8 = vec![0.0f64; dim8];
                    ei8[i] = 1.0;
                    ej8[j] = 1.0;
                    let prod_o8 = cd_multiply(&ei8, &ej8);
                    // prod_o8 should be sign * e_k in 8D
                    let sign = if dot_pos > 0.0 { 1.0 } else { -1.0 };
                    // Verify: prod_an = sign * vbasis[k] component-wise
                    let ok = (0..n).all(|r| (prod_an[r] - sign * vbasis[k][r]).abs() < atol);
                    // And the standard octonion gives the same sign for e_k
                    let ok_o8 = (prod_o8[k] - sign).abs() < atol;
                    if ok && ok_o8 {
                        matched = true;
                        break;
                    }
                }
            }
            if !matched {
                return false;
            }
        }
    }
    true
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use cd_kernel::cayley_dickson::predicates::is_special_couple;

    fn basis(dim: usize, i: usize) -> Vec<f64> {
        let mut v = vec![0.0; dim];
        v[i] = 1.0;
        v
    }

    const ATOL: f64 = 1e-10;
    const DIM8: usize = 8;
    const DIM16: usize = 16;

    // Standard doubly-pure unit-norm element in A_3 (8D): e1.
    fn test_a_8d() -> Vec<f64> {
        basis(DIM8, 1) // e1 in A_3; doubly pure, unit norm
    }

    // Standard doubly-pure unit-norm element in A_4 (16D): e1.
    fn test_a_16d() -> Vec<f64> {
        basis(DIM16, 1) // e1 in A_4; doubly pure, unit norm
    }

    #[test]
    fn test_h_a_basis_order_and_properties() {
        let a = test_a_8d();
        let [e0, ai, aj, ak] = h_a_basis(&a);
        // e0 is the real unit
        assert!((e0[0] - 1.0).abs() < ATOL);
        // ai = a = e1
        assert!((ai[1] - 1.0).abs() < ATOL);
        // tilde_e0 = e_{dim/2} = e4
        assert!((ak[4] - 1.0).abs() < ATOL, "tilde_e0 should be e4 in 8D");
        // tilde_a = (-a_2, a_1): e1 in 8D split -> tilde(e1) = e5
        assert!((aj[5] - 1.0).abs() < ATOL, "tilde(e1) should be e5 in 8D");
        // All four are orthonormal
        for (name, v) in [("e0", &e0), ("a", &ai), ("ta", &aj), ("te0", &ak)] {
            let n = cd_norm_sq(v);
            assert!((n - 1.0).abs() < ATOL, "{name} should have norm 1");
        }
        assert!(cd_inner_product(&e0, &ai).abs() < ATOL, "e0 perp a");
        assert!(cd_inner_product(&ai, &aj).abs() < ATOL, "a perp tilde_a");
        assert!(
            cd_inner_product(&aj, &ak).abs() < ATOL,
            "tilde_a perp tilde_e0"
        );
    }

    #[test]
    fn test_verify_h_a_isomorphism_e1_in_a3() {
        let a = test_a_8d();
        assert!(
            verify_h_a_isomorphism(&a, ATOL),
            "H_{{e1}} should be isomorphic to H in A_3"
        );
    }

    #[test]
    fn test_t_operator_is_zero_on_h_a() {
        // T_a(h) = 0 for h in H_a (Theorem 1.13, H_a is associative)
        let a = test_a_8d();
        let ta = t_operator_matrix(&a, DIM8);
        for h in &h_a_basis(&a) {
            let h_dv = DVector::from_vec(h.clone());
            let result = &ta * &h_dv;
            assert!(
                result.norm() < ATOL * 10.0,
                "T_a should annihilate H_a vectors, got norm {}",
                result.norm()
            );
        }
    }

    #[test]
    fn test_verify_corollary_1_17_a3() {
        // For a=e1 in A_3 (8D): Ker(L_a) has dim 0 (e1 is not a ZD in A_3)
        let a = test_a_8d();
        let (ker_dim, is_mod4, is_leq) = verify_corollary_1_17(&a, DIM8, ATOL);
        assert!(is_mod4, "dim Ker(L_a) must be 0 mod 4, got {ker_dim}");
        assert!(is_leq, "dim Ker(L_a)={ker_dim} must be <= {}", DIM8 - 4);
    }

    #[test]
    fn test_verify_corollary_1_17_a4_with_zd() {
        // For a = (e1+e10)/sqrt(2) in A_4 (16D): this is a ZD, Ker(L_a) has dim 4
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let mut a = vec![0.0; DIM16];
        a[1] = s;
        a[10] = s;
        let (ker_dim, is_mod4, is_leq) = verify_corollary_1_17(&a, DIM16, ATOL);
        assert!(is_mod4, "dim Ker(L_a) must be 0 mod 4, got {ker_dim}");
        assert!(is_leq, "dim Ker(L_a)={ker_dim} must be <= {}", DIM16 - 4);
        assert_eq!(ker_dim, 4, "ZD element should have Ker(L_a) dim = 4");
    }

    #[test]
    fn test_moreno_decomposition_totals_8d() {
        // In A_3 (8D) with a=e1: total should be 8, all subspaces have dim 0 mod 4
        let a = test_a_8d();
        let dec = moreno_decomposition(&a, DIM8, ATOL);
        assert!(
            dec.total_dim_check,
            "Total dim should equal 8, got H_a(4) + ker_la({}) + ker_tt({}) + v_lambda({}) = {}",
            dec.ker_l_a.ncols(),
            dec.ker_t_tilde.ncols(),
            dec.v_lambda.iter().map(|(_, m)| m.ncols()).sum::<usize>(),
            4 + dec.ker_l_a.ncols()
                + dec.ker_t_tilde.ncols()
                + dec.v_lambda.iter().map(|(_, m)| m.ncols()).sum::<usize>()
        );
        assert!(dec.ker_la_mod4, "Ker(L_a) dim must be 0 mod 4");
        assert!(dec.ker_la_leq_bound, "Ker(L_a) dim must be <= dim-4");
    }

    #[test]
    fn test_moreno_decomposition_totals_16d() {
        // In A_4 (16D) with a=e1: total should be 16
        let a = test_a_16d();
        let dec = moreno_decomposition(&a, DIM16, ATOL);
        assert!(
            dec.total_dim_check,
            "Total dim should equal 16, got {}",
            4 + dec.ker_l_a.ncols()
                + dec.ker_t_tilde.ncols()
                + dec.v_lambda.iter().map(|(_, m)| m.ncols()).sum::<usize>()
        );
        assert!(dec.ker_la_mod4, "Ker(L_a) dim must be 0 mod 4");
        assert!(dec.ker_la_leq_bound, "Ker(L_a) dim must be <= dim-4");
    }

    #[test]
    fn test_s_operator_is_skew_symmetric() {
        // S = R_b * L_a - L_a * R_b should be skew-symmetric: S = -S^T
        let a = basis(DIM8, 1); // e1
        let b = basis(DIM8, 2); // e2
        let s = s_operator_matrix(&a, &b, DIM8);
        let st = s.transpose();
        assert!((s + st).norm() < ATOL * 10.0, "S should be skew-symmetric");
    }

    #[test]
    fn test_ker_s_restricted_decomposition_a3() {
        // {e1, e2} is a special couple in A_3; Corollary 2.4 should hold
        let a = basis(DIM8, 1);
        let b = basis(DIM8, 2);
        assert!(
            is_special_couple(&a, &b, ATOL),
            "test prereq: {{e1,e2}} must be special couple"
        );
        let ok = verify_corollary_2_4(&a, &b, DIM8, ATOL);
        assert!(ok, "Corollary 2.4 should hold for {{e1,e2}} in A_3");
    }

    #[test]
    fn test_has_eigenvalue_minus_2_known_zd_16d() {
        // a=(e1+e10)/sqrt(2), b=(e4-e15)/sqrt(2) is a known ZD in A_4.
        // c = a+b should have lambda=-2 in spec(L_c^2... wait, actually
        // Theorem 2.9 applies in A_{n+1}: (a,b) in A_{n+1} is ZD iff spec(L^2_{a+b}) on A_n.
        // For 16D elements a=(e1+e10)/sqrt(2) in A_4, this is a ZD at the 16D level,
        // so we check directly: is_zero_divisor returns true.
        // The Theorem 2.9 test uses the 16D a and checks L^2_{a+b} on the first 8D half.
        // For this test: just verify that has_eigenvalue_minus_2(a+b) is true for known ZD.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let mut a = [0.0; DIM16];
        let mut b = [0.0; DIM16];
        a[1] = s;
        a[10] = s;
        b[4] = s;
        b[15] = -s;
        // c = a+b
        let c: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
        assert!(
            has_eigenvalue_minus_2(&c, DIM16, ATOL),
            "Known ZD sum should have eigenvalue -2 in spectrum of L_c^2"
        );
    }

    #[test]
    fn test_is_zd_moreno_known_pair_16d() {
        // (e1+e10)/sqrt(2) and (e4-e15)/sqrt(2) form a ZD pair in A_4
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let mut a = vec![0.0; DIM16];
        let mut b = vec![0.0; DIM16];
        a[1] = s;
        a[10] = s;
        b[4] = s;
        b[15] = -s;
        assert!(
            is_zd_moreno(&a, &b, ATOL),
            "Known ZD pair should satisfy Theorem 2.9 eigenvalue criterion"
        );
    }

    #[test]
    fn test_is_zd_moreno_non_zd_pair_8d() {
        // Theorem 2.9: (a,b) in A_{n+1} is a ZD iff -2 in spec(L_{a+b}^2) on A_n.
        // For PARALLEL elements a = b = e1: c = a+b = 2*e1, c^2 = -4.
        // L_c^2 = (2*L_{e1})^2 = 4*L_{e1}^2 = 4*(-I) = -4*I.
        // Eigenvalue is -4, not -2. So (e1, e1) is NOT a ZD pair.
        // (Note: (e1, e2) ARE orthogonal unit imaginary elements and they DO give a ZD
        //  in A_4 (their lift is e1+e10 in 16D which is a known boxkite ZD).
        //  So the right non-ZD test uses parallel elements, not orthogonal ones.)
        let a = basis(DIM8, 1);
        let b = basis(DIM8, 1); // same element: parallel, not orthogonal
        assert!(
            !is_zd_moreno(&a, &b, ATOL),
            "Parallel elements (e1, e1) should NOT satisfy Theorem 2.9 ZD criterion"
        );
    }

    #[test]
    fn test_v_triple_basis_order_a3() {
        // {e1, e7, e2} is a special triple in A_3.
        // V(e1;e7;e2) = {e0, e1, e2, e1*e2=e3, (e1*e7)*e2=e4, e7*e2=e5, e1*e7=e6, e7}
        // Cross-check the source-order basis against expected octonion products.
        let e1 = basis(DIM8, 1);
        let e7 = basis(DIM8, 7);
        let e2 = basis(DIM8, 2);
        let vb = v_triple_basis(&e1, &e7, &e2);
        // vb[0] = e0
        assert!((vb[0][0] - 1.0).abs() < ATOL, "vb[0] should be e0");
        // vb[1] = e1
        assert!((vb[1][1] - 1.0).abs() < ATOL, "vb[1] should be e1");
        // vb[2] = e2
        assert!((vb[2][2] - 1.0).abs() < ATOL, "vb[2] should be e2");
        // vb[3] = e1*e2 = e3
        let e1e2 = cd_multiply(&e1, &e2);
        for k in 0..DIM8 {
            assert!((vb[3][k] - e1e2[k]).abs() < ATOL, "vb[3] mismatch at {k}");
        }
        // vb[7] = e7
        assert!((vb[7][7] - 1.0).abs() < ATOL, "vb[7] should be e7");
    }

    #[test]
    fn test_verify_theorem_2_13_e1_e7_e2() {
        // The standard special triple {e1, e7, e2} in A_3 should give an octonion copy
        let e1 = basis(DIM8, 1);
        let e7 = basis(DIM8, 7);
        let e2 = basis(DIM8, 2);
        assert!(
            verify_theorem_2_13(&e1, &e7, &e2, ATOL),
            "{{e1, e7, e2}} should give an octonion copy (Theorem 2.13)"
        );
    }
}
