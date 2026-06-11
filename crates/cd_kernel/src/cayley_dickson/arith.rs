//! Cayley-Dickson arithmetic: conjugation, multiplication, and norms.
//!
//! # Mathematical foundation
//!
//! The Cayley-Dickson construction doubles an algebra A into A' = A x A with:
//! - Conjugation: `(a, b)* = (a*, -b)`
//! - Multiplication: `(a,b)(c,d) = (ac - d*b, da + bc*)`
//!
//! This module provides both allocating (returns `Vec<f64>`) and non-allocating
//! (writes to `&mut [f64]` workspace) variants. The non-allocating path is
//! critical for tight loops: the CP violation scan evaluates ~10K eigendecomps
//! that each call cd_multiply transitively.
//!
//! # Two API tiers
//!
//! | Tier | Function | Allocation | Use case |
//! |------|----------|-----------|----------|
//! | Allocating | `cd_multiply` | Vec per call | One-off, diagnostic |
//! | Non-allocating | `cd_multiply_into` | Zero (workspace) | Tight loops, scan |
//! | Non-allocating | `cd_conjugate_into` | Zero (buffer) | Inside workspace multiply |
//!
//! # Why recursive instead of flat SIMD
//!
//! The flat SIMD multiply (`simd.rs::sedenion_multiply_flat`) is faster for
//! fixed dims (4-16D) but requires compile-time dimension. The recursive
//! multiply works for arbitrary power-of-2 dims (used for 32D+ exploration).
//! Both paths produce identical results (validated by `test_simd_matches_scalar`).

/// Cayley-Dickson conjugation: negate all imaginary components.
///
/// `x* = (x0, -x1, -x2, ..., -x_{n-1})`
///
/// Allocates a new Vec. For non-allocating variant, see `cd_conjugate_into`.
#[inline]
pub fn cd_conjugate(x: &[f64]) -> Vec<f64> {
    let mut res = x.to_vec();
    for v in res[1..].iter_mut() {
        *v = -*v;
    }
    res
}

/// Cayley-Dickson multiplication via the doubling formula.
///
/// (a,b)(c,d) = (ac - d*b, da + bc*)
///
/// where `*` denotes conjugation. Recursion bottoms out at dim=1
/// (scalar multiplication).
pub fn cd_multiply(a: &[f64], b: &[f64]) -> Vec<f64> {
    let dim = a.len();
    debug_assert_eq!(a.len(), b.len());

    if dim == 1 {
        return vec![a[0] * b[0]];
    }

    let half = dim / 2;
    let (a_l, a_r) = a.split_at(half);
    let (c_l, c_r) = b.split_at(half);

    let conj_c_r = cd_conjugate(c_r);
    let term1 = cd_multiply(a_l, c_l);
    let term2 = cd_multiply(&conj_c_r, a_r);

    let conj_c_l = cd_conjugate(c_l);
    let term3 = cd_multiply(c_r, a_l);
    let term4 = cd_multiply(a_r, &conj_c_l);

    let mut result = Vec::with_capacity(dim);
    for i in 0..half {
        result.push(term1[i] - term2[i]);
    }
    for i in 0..half {
        result.push(term3[i] + term4[i]);
    }
    result
}

/// Cayley-Dickson multiplication with conjugated doubling (chirality inversion).
pub fn cd_multiply_conjugated(a: &[f64], b: &[f64]) -> Vec<f64> {
    let dim = a.len();
    debug_assert_eq!(a.len(), b.len());

    if dim == 1 {
        return vec![a[0] * b[0]];
    }

    let half = dim / 2;
    let (a_l, a_r) = a.split_at(half);
    let (c_l, c_r) = b.split_at(half);

    let conj_a_r = cd_conjugate(a_r);
    let term1 = cd_multiply_conjugated(a_l, c_l);
    let term2 = cd_multiply_conjugated(c_r, &conj_a_r);

    let term3 = cd_multiply_conjugated(&conj_a_r, c_l);
    let term4 = cd_multiply_conjugated(c_r, a_l);

    let mut result = Vec::with_capacity(dim);
    for i in 0..half {
        result.push(term1[i] - term2[i]);
    }
    for i in 0..half {
        result.push(term3[i] + term4[i]);
    }
    result
}

/// Squared Euclidean norm: sum of squares of all components.
#[inline]
pub fn cd_norm_sq(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum()
}

/// Check if two slices are element-wise close within tolerance.
#[inline]
pub(crate) fn allclose(a: &[f64], b: &[f64], atol: f64) -> bool {
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() <= atol)
}

/// Minimum workspace length for `cd_multiply_into` at dimension `dim`.
///
/// # Workspace layout (per recursion level)
///
/// At dimension `dim` with `half = dim/2`, each level needs:
/// - `conj[half]`: conjugate buffer (reused for both conjugates)
/// - `sub1[half]`, `sub2[half]`: two sub-product scratch slots
/// - Recursive workspace for the 4 half-dimension multiplies
///
/// Total: `4 * dim` (geometric sum of `2*dim` at each halving level).
/// Validated by `test_cd_multiply_into_sweep` for dims 1..=256.
///
/// # Callers
///
/// - `cd_multiply_into`: asserts `workspace.len() >= cd_multiply_workspace_len(dim)`
/// - `cd_multiply_mut`: allocates workspace of this size
#[inline]
pub fn cd_multiply_workspace_len(dim: usize) -> usize {
    if dim <= 1 {
        return 0;
    }
    4 * dim
}

/// Conjugate into caller-provided buffer (zero allocation).
///
/// # Non-overlap contract
///
/// `out` must have the same length as `x` and must NOT alias `x`.
///
/// # Why this exists
///
/// `cd_conjugate` allocates a Vec per call. Inside `cd_multiply_into`,
/// conjugates are stored in workspace slices, avoiding heap allocation
/// in the recursive multiply.
#[inline]
pub fn cd_conjugate_into(x: &[f64], out: &mut [f64]) {
    debug_assert_eq!(x.len(), out.len());
    out[0] = x[0];
    for i in 1..x.len() {
        out[i] = -x[i];
    }
}

/// Non-allocating Cayley-Dickson multiplication using a pre-allocated workspace.
///
/// # Doubling formula
///
/// `(a,b)(c,d) = (ac - d*b, da + bc*)`
///
/// where `*` denotes conjugation. Recursion bottoms out at dim=1.
///
/// # Non-overlap contract
///
/// - `a`, `b`: read-only inputs (may alias each other)
/// - `res`: write-only output (must NOT alias `a` or `b`)
/// - `workspace`: scratch (must NOT alias `a`, `b`, or `res`)
///
/// # Panics
///
/// Panics if `workspace.len() < cd_multiply_workspace_len(dim)`.
pub fn cd_multiply_into(a: &[f64], b: &[f64], res: &mut [f64], workspace: &mut [f64]) {
    let dim = a.len();
    assert_eq!(b.len(), dim);
    assert_eq!(res.len(), dim);
    assert!(
        workspace.len() >= cd_multiply_workspace_len(dim),
        "workspace too small: need {}, got {}",
        cd_multiply_workspace_len(dim),
        workspace.len()
    );

    if dim == 1 {
        res[0] = a[0] * b[0];
        return;
    }

    let half = dim / 2;
    let (a_l, a_r) = a.split_at(half);
    let (c_l, c_r) = b.split_at(half);

    // Workspace layout: [conj: half][sub1: half][sub2: half][recurse: ...]
    let (conj_buf, rest) = workspace.split_at_mut(half);
    let (sub1, rest2) = rest.split_at_mut(half);
    let (sub2, recurse_ws) = rest2.split_at_mut(half);

    // Left half of result: res[0..half] = a_l*c_l - conj(c_r)*a_r
    //   term1 = a_l * c_l -> sub1
    cd_multiply_into(a_l, c_l, sub1, recurse_ws);
    //   conj(c_r) -> conj_buf
    cd_conjugate_into(c_r, conj_buf);
    //   term2 = conj(c_r) * a_r -> sub2
    cd_multiply_into(conj_buf, a_r, sub2, recurse_ws);
    //   res[0..half] = sub1 - sub2
    for i in 0..half {
        res[i] = sub1[i] - sub2[i];
    }

    // Right half of result: res[half..dim] = c_r*a_l + a_r*conj(c_l)
    //   term3 = c_r * a_l -> sub1 (reuse)
    cd_multiply_into(c_r, a_l, sub1, recurse_ws);
    //   conj(c_l) -> conj_buf (reuse)
    cd_conjugate_into(c_l, conj_buf);
    //   term4 = a_r * conj(c_l) -> sub2 (reuse)
    cd_multiply_into(a_r, conj_buf, sub2, recurse_ws);
    //   res[half..dim] = sub1 + sub2
    for i in 0..half {
        res[half + i] = sub1[i] + sub2[i];
    }
}

/// In-place Cayley-Dickson multiplication.
///
/// Allocates `res[dim]` + `workspace[cd_multiply_workspace_len(dim)]` once,
/// then delegates to the non-allocating `cd_multiply_into`.
pub fn cd_multiply_mut(a: &mut Vec<f64>, b: &[f64]) {
    let dim = a.len();
    let mut res = vec![0.0; dim];
    let mut workspace = vec![0.0; cd_multiply_workspace_len(dim)];
    cd_multiply_into(a, b, &mut res, &mut workspace);
    *a = res;
}

/// Construct the left-multiplication matrix L_a where L_a`[i]``[j]` = (a * e_j)`[i]`.
pub fn left_mult_operator(a: &[f64], dim: usize) -> Vec<f64> {
    debug_assert_eq!(a.len(), dim);
    let mut matrix = vec![0.0; dim * dim];
    let mut basis = vec![0.0; dim];
    for j in 0..dim {
        if j > 0 {
            basis[j - 1] = 0.0;
        }
        basis[j] = 1.0;
        let col = cd_multiply(a, &basis);
        for i in 0..dim {
            matrix[i * dim + j] = col[i];
        }
    }
    matrix
}
