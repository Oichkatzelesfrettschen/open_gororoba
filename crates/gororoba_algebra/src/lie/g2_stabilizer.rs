//! Stabilizer subalgebras of the G2 derivation algebra of the octonions.
//!
//! Mathematical content (all statements are theorems):
//!
//! 1. Der(O) is isomorphic to the compact real Lie algebra g2 (dim 14).
//! 2. For any unit imaginary octonion e_k (k in 1..=7), the stabilizer
//!    stab(e_k) = { D in Der(O) : D(e_k) = 0 } is an 8-dim Lie subalgebra.
//! 3. Left-multiplication by e_k defines a complex structure J_k on the
//!    6-dim perpendicular complement e_k^perp in Im(O), with J_k^2 = -Id.
//! 4. Every stabilizer derivation D is skew-adjoint on e_k^perp (since D
//!    is skew-symmetric on all of O) and commutes with J_k (Leibniz +
//!    D(e_k) = 0). Together these give an embedding stab(e_k) -> u(3).
//! 5. By compact Lie algebra classification, an 8-dim compact Lie subalgebra
//!    of u(3) (which has dim 9) must be su(3). The constructive verification
//!    via 3x3 complex matrices is deferred to PR 2.
//!
//! Normalization: anti-Hermitian convention. Invariant form <A,B> = -tr(AB).
//! Structure constants: f_{abc} = <[T_a,T_b], T_c> (totally antisymmetric).
//!
//! Note on stabilizer_basis: The SVD-derived basis is orthonormal but
//! non-canonical (defined up to O(8) rotation and sign choices). Only
//! subspace-invariant and O(8)-invariant properties are stable: dimension,
//! closure, orthogonality, evaluation-map rank, residual norms, and
//! invariant contractions after fixed normalization. Raw f_{abc} tensor
//! components rotate under the basis ambiguity. Do not depend on specific
//! generator ordering or individual structure constant values in PR1.
//!
//! References:
//!   Baez, "The Octonions" (2002), Section 4.1
//!   Harvey, "Spinors and Calibrations" (1990)

use crate::construction::g2_automorphisms::{OctonionDerivation, compute_g2_basis};
use crate::construction::octonion::Octonion;
use nalgebra::DMatrix;

/// Relative tolerance factor for SVD rank determination.
/// sigma_i <= RANK_TOL_FACTOR * f64::EPSILON * max(m, n) * sigma_max
/// is treated as numerically zero. Factor 64 gives margin over bare
/// machine epsilon, which is often too optimistic in practice.
const RANK_TOL_FACTOR: f64 = 64.0;

// ---------------------------------------------------------------------------
// Invariant form
// ---------------------------------------------------------------------------

/// Invariant form on Der(O): <A, B> = -tr(A * B).
///
/// Positive definite on compact g2. Used for orthonormalization
/// and structure constant extraction.
///
/// For skew-symmetric A, B this equals the Frobenius inner product
/// sum_{i,j} A[i][j] * B[i][j], but the -tr(AB) definition is the
/// canonical Killing form and generalizes to non-skew matrices.
pub fn invariant_form(a: &OctonionDerivation, b: &OctonionDerivation) -> f64 {
    let mut trace = 0.0;
    for i in 0..8 {
        for j in 0..8 {
            trace += a.matrix[i][j] * b.matrix[j][i];
        }
    }
    -trace
}

// ---------------------------------------------------------------------------
// Orthonormalization
// ---------------------------------------------------------------------------

/// Orthonormalize a derivation basis w.r.t. the invariant form via Gram-Schmidt.
///
/// Returns orthonormal basis {T_1, ..., T_n} with <T_a, T_b> = delta_{ab}.
/// Drops vectors with invariant-form norm below `tol` (degenerate directions).
fn orthonormalize(basis: &[OctonionDerivation], tol: f64) -> Vec<OctonionDerivation> {
    let mut result: Vec<OctonionDerivation> = Vec::with_capacity(basis.len());

    for d in basis {
        let mut v = d.clone();

        // Subtract projections onto already-orthonormalized vectors
        for q in &result {
            let coeff = invariant_form(&v, q);
            for i in 0..8 {
                for j in 0..8 {
                    v.matrix[i][j] -= coeff * q.matrix[i][j];
                }
            }
        }

        // Normalize
        let norm_sq = invariant_form(&v, &v);
        if norm_sq < tol * tol {
            continue; // Linearly dependent -- drop
        }
        let inv_norm = 1.0 / norm_sq.sqrt();
        for i in 0..8 {
            for j in 0..8 {
                v.matrix[i][j] *= inv_norm;
            }
        }

        result.push(v);
    }

    result
}

// ---------------------------------------------------------------------------
// Lie bracket
// ---------------------------------------------------------------------------

/// Lie bracket [A, B] = AB - BA as 8x8 matrix commutator.
///
/// The commutator of two derivations is a derivation (closure property
/// of the Lie algebra Der(O) = g2).
pub fn derivation_bracket(
    a: &OctonionDerivation,
    b: &OctonionDerivation,
) -> OctonionDerivation {
    let mut result = [[0.0f64; 8]; 8];
    #[allow(clippy::needless_range_loop)]
    for i in 0..8 {
        for j in 0..8 {
            let mut sum = 0.0;
            for k in 0..8 {
                sum += a.matrix[i][k] * b.matrix[k][j]
                     - b.matrix[i][k] * a.matrix[k][j];
            }
            result[i][j] = sum;
        }
    }
    OctonionDerivation { matrix: result }
}

// ---------------------------------------------------------------------------
// Perpendicular basis helper
// ---------------------------------------------------------------------------

/// Returns the 6 imaginary basis indices j != fixed_unit, in increasing order.
///
/// This is the ordered basis for e_k^perp that all 6x6 matrices are expressed in.
fn perp_basis(fixed_unit: usize) -> [usize; 6] {
    assert!((1..=7).contains(&fixed_unit), "fixed_unit must be in 1..=7");
    let mut indices = [0usize; 6];
    let mut idx = 0;
    for j in 1..=7 {
        if j != fixed_unit {
            indices[idx] = j;
            idx += 1;
        }
    }
    indices
}

// ---------------------------------------------------------------------------
// Stabilizer decomposition
// ---------------------------------------------------------------------------

/// Result of the G2 stabilizer decomposition for a fixed imaginary unit e_k.
pub struct G2Stabilizer {
    /// The fixed imaginary unit index (1..=7).
    pub fixed_unit: usize,
    /// Orthonormal basis of e_k^perp (the 6 indices j != k in Im(O)).
    /// All 6x6 matrices in this module use this ordering.
    pub perp_indices: [usize; 6],
    /// Orthonormal basis for stab(e_k) = ker(E_k). Dimension: 8.
    /// Non-canonical: defined up to O(8) rotation and sign from SVD.
    pub stabilizer_basis: Vec<OctonionDerivation>,
    /// Orthonormal basis for the orthogonal complement of stab(e_k) in g2.
    /// Dimension: 6. Tangent space to S^6 = G2/SU(3).
    pub coset_complement: Vec<OctonionDerivation>,
}

/// Decompose g2 = stab(e_k) + m_k by fixing imaginary unit e_k.
///
/// Algorithm:
/// 1. Compute G2 basis via `compute_g2_basis()`, orthonormalize.
/// 2. Build 6x14 evaluation matrix E where E\[r\]\[a\] = (T_a(e_k))_{perp\[r\]}.
/// 3. SVD of E with relative threshold for rank determination.
/// 4. Right singular vectors for zero singular values span ker(E_k) = stab(e_k).
/// 5. Right singular vectors for nonzero singular values span coset complement.
///
/// Panics if `fixed_unit` is not in 1..=7.
pub fn stabilizer_decomposition(fixed_unit: usize) -> G2Stabilizer {
    assert!((1..=7).contains(&fixed_unit), "fixed_unit must be in 1..=7");

    // Step 1: Orthonormal G2 basis
    let raw_basis = compute_g2_basis();
    let g2_basis = orthonormalize(&raw_basis, 1e-12);
    let n = g2_basis.len(); // Should be 14

    // Step 2: Build evaluation matrix E (6 x n).
    // E[r][a] = (T_a(e_k)).components[perp[r]]
    // The evaluation map E: R^n -> R^6 sends coefficient vector c to
    // sum_a c_a * T_a(e_k)|_{e_k^perp}.
    // ker(E) = stab(e_k).
    let perp = perp_basis(fixed_unit);
    let m = 6;
    let ek = Octonion::basis(fixed_unit);

    let mut eval_mat = DMatrix::zeros(m, n);
    for (a, t_a) in g2_basis.iter().enumerate() {
        let d_ek = t_a.apply(&ek);
        for (r, &perp_idx) in perp.iter().enumerate() {
            eval_mat[(r, a)] = d_ek.components[perp_idx];
        }
    }

    // Step 3: Thin SVD of E (6 x 14).
    // This gives V^T of shape (min(6,14) x 14) = (6 x 14).
    // The 6 rows of V^T are the right singular vectors spanning the row space.
    // The kernel (dim 8) is the orthogonal complement of those 6 vectors in R^14.
    let svd = eval_mat.svd(false, true);
    let sigma = &svd.singular_values;
    let v_t = svd.v_t.as_ref().expect("SVD must produce V^T");

    // Relative threshold for rank
    let sigma_max = sigma.iter().copied().fold(0.0f64, f64::max);
    let tol = RANK_TOL_FACTOR * f64::EPSILON * n.max(m) as f64 * sigma_max;

    // Collect row-space vectors (nonzero singular values) as R^14 coefficient vectors.
    let mut row_space: Vec<Vec<f64>> = Vec::new();
    for row_idx in 0..v_t.nrows() {
        if row_idx < sigma.len() && sigma[row_idx] > tol {
            let v: Vec<f64> = (0..n).map(|a| v_t[(row_idx, a)]).collect();
            row_space.push(v);
        }
    }

    // Step 4: Build the kernel as the orthogonal complement of the row space.
    // Start from the standard basis of R^14, project out all row-space
    // directions (and already-accepted kernel directions), normalize.
    // Modified Gram-Schmidt, run twice for numerical stability.
    let mut kernel_coeffs: Vec<Vec<f64>> = Vec::new();

    for candidate_idx in 0..n {
        // Standard basis vector e_{candidate_idx} in R^14
        let mut v = vec![0.0f64; n];
        v[candidate_idx] = 1.0;

        // Two passes of modified Gram-Schmidt for stability
        for _pass in 0..2 {
            // Project out row-space directions
            for rs in &row_space {
                let dot: f64 = v.iter().zip(rs.iter()).map(|(a, b)| a * b).sum();
                for (vi, rsi) in v.iter_mut().zip(rs.iter()) {
                    *vi -= dot * rsi;
                }
            }
            // Project out already-accepted kernel directions
            for kv in &kernel_coeffs {
                let dot: f64 = v.iter().zip(kv.iter()).map(|(a, b)| a * b).sum();
                for (vi, kvi) in v.iter_mut().zip(kv.iter()) {
                    *vi -= dot * kvi;
                }
            }
        }

        // Normalize
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            for vi in &mut v {
                *vi /= norm;
            }
            kernel_coeffs.push(v);
        }
    }

    // Step 5: Convert coefficient vectors to derivations.
    let mut stab_derivations = Vec::with_capacity(kernel_coeffs.len());
    for coeffs in &kernel_coeffs {
        let mut d = OctonionDerivation { matrix: [[0.0; 8]; 8] };
        for (a, t_a) in g2_basis.iter().enumerate() {
            let c = coeffs[a];
            for i in 0..8 {
                for j in 0..8 {
                    d.matrix[i][j] += c * t_a.matrix[i][j];
                }
            }
        }
        stab_derivations.push(d);
    }

    let mut coset_derivations = Vec::with_capacity(row_space.len());
    for coeffs in &row_space {
        let mut d = OctonionDerivation { matrix: [[0.0; 8]; 8] };
        for (a, t_a) in g2_basis.iter().enumerate() {
            let c = coeffs[a];
            for i in 0..8 {
                for j in 0..8 {
                    d.matrix[i][j] += c * t_a.matrix[i][j];
                }
            }
        }
        coset_derivations.push(d);
    }

    G2Stabilizer {
        fixed_unit,
        perp_indices: perp,
        stabilizer_basis: stab_derivations,
        coset_complement: coset_derivations,
    }
}

// ---------------------------------------------------------------------------
// Structure constants
// ---------------------------------------------------------------------------

/// Structure constants f_{abc} = <[T_a, T_b], T_c> for a Lie subalgebra.
///
/// Precondition: `basis` must be orthonormal w.r.t. `invariant_form`.
/// With an orthonormal basis, coefficient extraction is simply the
/// invariant-form inner product of the bracket with the basis vector.
///
/// The result is totally antisymmetric in all three indices.
pub fn structure_constants(basis: &[OctonionDerivation]) -> Vec<Vec<Vec<f64>>> {
    let n = basis.len();
    let mut f = vec![vec![vec![0.0; n]; n]; n];
    for a in 0..n {
        for b in 0..n {
            if a == b {
                continue; // [T_a, T_a] = 0
            }
            let bracket = derivation_bracket(&basis[a], &basis[b]);
            for c in 0..n {
                f[a][b][c] = invariant_form(&bracket, &basis[c]);
            }
        }
    }
    f
}

// ---------------------------------------------------------------------------
// Complex structure
// ---------------------------------------------------------------------------

/// Complex structure on e_k^perp induced by left-multiplication with e_k.
///
/// J_k(v) = e_k * v for v in e_k^perp.
pub struct ComplexStructure {
    /// The 6x6 matrix of J_k in the perp_indices basis.
    pub matrix: [[f64; 6]; 6],
    /// The ordered perpendicular basis indices.
    pub perp_indices: [usize; 6],
    /// The 3 Fano-derived complex lines: (a_idx, b_idx, sign) where
    /// e_k * e_{perp\[a_idx\]} = sign * e_{perp\[b_idx\]}.
    /// Convention: a_idx < b_idx in the perp_indices ordering.
    /// These define the complex basis: v_j = e_{perp\[a_idx\]},
    /// J_k(v_j) = sign * e_{perp\[b_idx\]}.
    pub fano_pairs: [(usize, usize, i8); 3],
}

/// Compute the complex structure J_k on e_k^perp.
///
/// Discovers the Fano line structure by direct multiplication
/// (no import from physics modules).
pub fn complex_structure(fixed_unit: usize) -> ComplexStructure {
    assert!((1..=7).contains(&fixed_unit), "fixed_unit must be in 1..=7");

    let perp = perp_basis(fixed_unit);
    let ek = Octonion::basis(fixed_unit);

    // Build J_k matrix: J[r][s] = (e_k * e_{perp[s]}).components[perp[r]]
    let mut matrix = [[0.0f64; 6]; 6];
    for s in 0..6 {
        let e_s = Octonion::basis(perp[s]);
        let product = ek.multiply(&e_s);
        for r in 0..6 {
            matrix[r][s] = product.components[perp[r]];
        }
    }

    // Discover Fano pairs: find the 3 pairs (a, b) with a < b such that
    // J_k maps e_{perp[a]} to +/- e_{perp[b]}.
    let mut pairs = Vec::new();
    let mut used = [false; 6];

    for a in 0..6 {
        if used[a] {
            continue;
        }
        // Find which perp index e_k * e_{perp[a]} maps to
        for b in (a + 1)..6 {
            if used[b] {
                continue;
            }
            if matrix[b][a].abs() > 0.5 {
                // e_k * e_{perp[a]} = sign * e_{perp[b]}
                let sign = if matrix[b][a] > 0.0 { 1i8 } else { -1i8 };
                pairs.push((a, b, sign));
                used[a] = true;
                used[b] = true;
                break;
            }
        }
    }

    assert_eq!(pairs.len(), 3, "Must find exactly 3 Fano lines through e_{}", fixed_unit);

    let fano_pairs: [(usize, usize, i8); 3] = [pairs[0], pairs[1], pairs[2]];

    ComplexStructure {
        matrix,
        perp_indices: perp,
        fano_pairs,
    }
}

// ---------------------------------------------------------------------------
// Perpendicular block and skew-adjointness
// ---------------------------------------------------------------------------

/// Extract the 6x6 block of a derivation's matrix on e_k^perp coordinates.
///
/// For a GENERAL derivation, this is a projected block (the e_k^perp rows
/// and columns of the 8x8 matrix), NOT a true restriction. The subspace
/// e_k^perp is only D-invariant when D(e_k) = 0 (stabilizer derivations).
///
/// For stabilizer derivations, this block IS the genuine restricted action.
pub(crate) fn perp_block(
    derivation: &OctonionDerivation,
    perp_indices: &[usize; 6],
) -> [[f64; 6]; 6] {
    let mut block = [[0.0f64; 6]; 6];
    for r in 0..6 {
        for s in 0..6 {
            block[r][s] = derivation.matrix[perp_indices[r]][perp_indices[s]];
        }
    }
    block
}

/// Verify that a stabilizer derivation's restricted action on e_k^perp
/// is skew-adjoint: R^T + R = 0.
///
/// Precondition: derivation must be in stab(e_k) (i.e., D(e_k) = 0).
/// Only for stabilizer derivations is the perp_block a true restriction.
///
/// D is skew-symmetric on all of O (8x8), so the perp block of a skew
/// matrix is itself skew. This is structural, not numerical coincidence.
/// The test serves as defense-in-depth.
pub fn is_skew_adjoint_on_perp(
    derivation: &OctonionDerivation,
    perp_indices: &[usize; 6],
    tol: f64,
) -> bool {
    let block = perp_block(derivation, perp_indices);
    #[allow(clippy::needless_range_loop)]
    for r in 0..6 {
        for s in 0..6 {
            if (block[r][s] + block[s][r]).abs() > tol {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// J_k commutation
// ---------------------------------------------------------------------------

/// Verify that derivation D commutes with complex structure J_k on e_k^perp.
///
/// Checks: D(J_k(v)) = J_k(D(v)) for all basis vectors v in e_k^perp.
///
/// Since D(e_k) = 0 and D satisfies Leibniz:
///   D(e_k * v) = D(e_k) * v + e_k * D(v) = e_k * D(v) = J_k(D(v))
/// Therefore D(J_k(v)) = J_k(D(v)) for all v in e_k^perp.
///
/// This is an automatic consequence of Leibniz + D(e_k) = 0, not a
/// numerical coincidence. The test serves as defense-in-depth.
///
/// Combined with skew-adjointness, this establishes the u(3) embedding.
pub fn commutes_with_j(
    derivation: &OctonionDerivation,
    j: &ComplexStructure,
    tol: f64,
) -> bool {
    let perp = &j.perp_indices;
    let r_block = perp_block(derivation, perp);
    let j_mat = &j.matrix;

    // Check R * J = J * R (as 6x6 matrices)
    for i in 0..6 {
        for k in 0..6 {
            let mut rj = 0.0;
            let mut jr = 0.0;
            for m in 0..6 {
                rj += r_block[i][m] * j_mat[m][k];
                jr += j_mat[i][m] * r_block[m][k];
            }
            if (rj - jr).abs() > tol {
                return false;
            }
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    // === Dimensional tests (parameterized k=1..7) ===

    #[test]
    fn test_stabilizer_dimension() {
        for k in 1..=7 {
            let decomp = stabilizer_decomposition(k);
            assert_eq!(
                decomp.stabilizer_basis.len(), 8,
                "stab(e_{}) must have dimension 8, got {}",
                k, decomp.stabilizer_basis.len()
            );
        }
    }

    #[test]
    fn test_coset_dimension() {
        for k in 1..=7 {
            let decomp = stabilizer_decomposition(k);
            assert_eq!(
                decomp.coset_complement.len(), 6,
                "coset for e_{} must have dimension 6, got {}",
                k, decomp.coset_complement.len()
            );
        }
    }

    #[test]
    fn test_dimensions_sum_to_14() {
        for k in 1..=7 {
            let decomp = stabilizer_decomposition(k);
            let total = decomp.stabilizer_basis.len() + decomp.coset_complement.len();
            assert_eq!(total, 14, "stab + coset for e_{} must sum to 14, got {}", k, total);
        }
    }

    // === Stabilizer correctness (parameterized k=1..7) ===

    #[test]
    fn test_stabilizer_annihilates_fixed_unit() {
        for k in 1..=7 {
            let decomp = stabilizer_decomposition(k);
            let ek = Octonion::basis(k);
            for (idx, d) in decomp.stabilizer_basis.iter().enumerate() {
                let d_ek = d.apply(&ek);
                let norm = d_ek.norm();
                assert!(
                    norm < TOL,
                    "stab(e_{}) basis[{}]: D(e_{}) has norm {}, expected 0",
                    k, idx, k, norm
                );
            }
        }
    }

    #[test]
    fn test_evaluation_map_residual() {
        for k in 1..=7 {
            let decomp = stabilizer_decomposition(k);
            let ek = Octonion::basis(k);
            // Kernel elements must give near-zero when evaluation map is applied
            for (idx, d) in decomp.stabilizer_basis.iter().enumerate() {
                let d_ek = d.apply(&ek);
                let residual: f64 = d_ek.components.iter().map(|c| c * c).sum::<f64>().sqrt();
                assert!(
                    residual < TOL,
                    "e_{}: stabilizer basis[{}] evaluation residual {} exceeds tolerance",
                    k, idx, residual
                );
            }
            // Coset elements should NOT annihilate e_k
            for d in &decomp.coset_complement {
                let d_ek = d.apply(&ek);
                let norm = d_ek.norm();
                assert!(
                    norm > TOL,
                    "e_{}: coset element should NOT annihilate e_{}, but norm = {}",
                    k, k, norm
                );
            }
        }
    }

    #[test]
    fn test_stabilizer_leibniz() {
        for k in 1..=7 {
            let decomp = stabilizer_decomposition(k);
            let e1 = Octonion::basis(1);
            let e2 = Octonion::basis(2);
            let e4 = Octonion::basis(4);
            for (idx, d) in decomp.stabilizer_basis.iter().enumerate() {
                assert!(
                    d.verify_leibniz(&e1, &e2, TOL),
                    "e_{}: stab basis[{}] fails Leibniz on (e1,e2)", k, idx
                );
                assert!(
                    d.verify_leibniz(&e2, &e4, TOL),
                    "e_{}: stab basis[{}] fails Leibniz on (e2,e4)", k, idx
                );
            }
        }
    }

    #[test]
    fn test_stabilizer_closure() {
        for k in 1..=7 {
            let decomp = stabilizer_decomposition(k);
            let basis = &decomp.stabilizer_basis;
            let n = basis.len();

            for a in 0..n {
                for b in (a + 1)..n {
                    let bracket = derivation_bracket(&basis[a], &basis[b]);

                    // Construct the projection explicitly, then measure residual.
                    // This avoids catastrophic cancellation from |bracket|^2 - sum(coeff^2).
                    let mut projected = OctonionDerivation { matrix: [[0.0; 8]; 8] };
                    for c in 0..n {
                        let coeff = invariant_form(&bracket, &basis[c]);
                        for i in 0..8 {
                            for j in 0..8 {
                                projected.matrix[i][j] += coeff * basis[c].matrix[i][j];
                            }
                        }
                    }

                    // Residual = bracket - projected
                    let mut residual_sq = 0.0;
                    for i in 0..8 {
                        for j in 0..8 {
                            let diff = bracket.matrix[i][j] - projected.matrix[i][j];
                            residual_sq += diff * diff;
                        }
                    }
                    let residual = residual_sq.sqrt();

                    assert!(
                        residual < TOL,
                        "e_{}: [stab[{}], stab[{}]] leaks outside stabilizer, residual = {}",
                        k, a, b, residual
                    );
                }
            }
        }
    }

    // === Invariant form ===

    #[test]
    fn test_invariant_form_positive_definite() {
        let raw = compute_g2_basis();
        for (idx, d) in raw.iter().enumerate() {
            let norm_sq = invariant_form(d, d);
            if d.frobenius_norm() > 1e-12 {
                assert!(
                    norm_sq > 0.0,
                    "g2 basis[{}]: invariant form must be positive, got {}",
                    idx, norm_sq
                );
            }
        }
    }

    #[test]
    fn test_orthonormality_g2() {
        let raw = compute_g2_basis();
        let ortho = orthonormalize(&raw, 1e-12);
        assert_eq!(ortho.len(), 14, "g2 must have dimension 14");
        for a in 0..ortho.len() {
            for b in 0..ortho.len() {
                let ip = invariant_form(&ortho[a], &ortho[b]);
                let expected = if a == b { 1.0 } else { 0.0 };
                assert!(
                    (ip - expected).abs() < TOL,
                    "<T_{}, T_{}> = {}, expected {}",
                    a, b, ip, expected
                );
            }
        }
    }

    #[test]
    fn test_orthonormality_stabilizer() {
        for k in 1..=7 {
            let decomp = stabilizer_decomposition(k);
            let basis = &decomp.stabilizer_basis;
            for a in 0..basis.len() {
                for b in 0..basis.len() {
                    let ip = invariant_form(&basis[a], &basis[b]);
                    let expected = if a == b { 1.0 } else { 0.0 };
                    assert!(
                        (ip - expected).abs() < TOL,
                        "e_{}: <stab[{}], stab[{}]> = {}, expected {}",
                        k, a, b, ip, expected
                    );
                }
            }
        }
    }

    #[test]
    fn test_stabilizer_perp_coset() {
        for k in 1..=7 {
            let decomp = stabilizer_decomposition(k);
            for (si, s) in decomp.stabilizer_basis.iter().enumerate() {
                for (ci, c) in decomp.coset_complement.iter().enumerate() {
                    let ip = invariant_form(s, c);
                    assert!(
                        ip.abs() < TOL,
                        "e_{}: <stab[{}], coset[{}]> = {}, expected 0",
                        k, si, ci, ip
                    );
                }
            }
        }
    }

    // === Structure constants ===

    #[test]
    fn test_structure_constants_total_antisymmetry() {
        let decomp = stabilizer_decomposition(1);
        let f = structure_constants(&decomp.stabilizer_basis);
        let n = decomp.stabilizer_basis.len();

        for a in 0..n {
            for b in 0..n {
                for c in 0..n {
                    // f_{abc} = -f_{bac}
                    assert!(
                        (f[a][b][c] + f[b][a][c]).abs() < TOL,
                        "f_{{{}{}{}}} + f_{{{}{}{}}} = {} (should be 0)",
                        a, b, c, b, a, c, f[a][b][c] + f[b][a][c]
                    );
                    // f_{abc} = -f_{acb}
                    assert!(
                        (f[a][b][c] + f[a][c][b]).abs() < TOL,
                        "f_{{{}{}{}}} + f_{{{}{}{}}} = {} (should be 0)",
                        a, b, c, a, c, b, f[a][b][c] + f[a][c][b]
                    );
                }
            }
        }
    }

    #[test]
    fn test_structure_constants_jacobi() {
        // Test for k=1 (representative; structure is basis-dependent but Jacobi is invariant)
        let decomp = stabilizer_decomposition(1);
        let f = structure_constants(&decomp.stabilizer_basis);
        let n = decomp.stabilizer_basis.len();

        // Jacobi identity: sum_d (f_{bcd} f_{ade} + f_{cad} f_{bde} + f_{abd} f_{cde}) = 0
        // for each fixed (a,b,c,e). Test all ordered triples a < b < c.
        for a in 0..n {
            for b in (a + 1)..n {
                for c in (b + 1)..n {
                    for e in 0..n {
                        let mut jac = 0.0;
                        for d in 0..n {
                            jac += f[b][c][d] * f[a][d][e]
                                 + f[c][a][d] * f[b][d][e]
                                 + f[a][b][d] * f[c][d][e];
                        }
                        assert!(
                            jac.abs() < TOL,
                            "Jacobi fails for (a,b,c,e)=({},{},{},{}): {}",
                            a, b, c, e, jac
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn test_bracket_reconstruction() {
        let decomp = stabilizer_decomposition(1);
        let basis = &decomp.stabilizer_basis;
        let f = structure_constants(basis);
        let n = basis.len();

        for a in 0..n {
            for b in (a + 1)..n {
                let bracket = derivation_bracket(&basis[a], &basis[b]);

                // Reconstruct from structure constants: sum_c f_{abc} T_c
                let mut reconstructed = [[0.0f64; 8]; 8];
                for c in 0..n {
                    let coeff = f[a][b][c];
                    for i in 0..8 {
                        for j in 0..8 {
                            reconstructed[i][j] += coeff * basis[c].matrix[i][j];
                        }
                    }
                }

                // Check residual
                let mut residual_sq = 0.0;
                for i in 0..8 {
                    for j in 0..8 {
                        let diff = bracket.matrix[i][j] - reconstructed[i][j];
                        residual_sq += diff * diff;
                    }
                }
                assert!(
                    residual_sq.sqrt() < TOL,
                    "[T_{}, T_{}] reconstruction residual = {}",
                    a, b, residual_sq.sqrt()
                );
            }
        }
    }

    // === Complex structure ===

    #[test]
    fn test_j_squared_is_minus_id() {
        for k in 1..=7 {
            let cs = complex_structure(k);
            let j = &cs.matrix;

            // Compute J^2
            for i in 0..6 {
                for l in 0..6 {
                    let mut j2 = 0.0;
                    for m in 0..6 {
                        j2 += j[i][m] * j[m][l];
                    }
                    let expected = if i == l { -1.0 } else { 0.0 };
                    assert!(
                        (j2 - expected).abs() < TOL,
                        "e_{}: J^2[{}][{}] = {}, expected {}",
                        k, i, l, j2, expected
                    );
                }
            }
        }
    }

    #[test]
    fn test_j_is_orthogonal() {
        for k in 1..=7 {
            let cs = complex_structure(k);
            let j = &cs.matrix;

            // Compute J^T J
            for i in 0..6 {
                for l in 0..6 {
                    let mut jtj = 0.0;
                    for m in 0..6 {
                        jtj += j[m][i] * j[m][l]; // J^T[i][m] * J[m][l]
                    }
                    let expected = if i == l { 1.0 } else { 0.0 };
                    assert!(
                        (jtj - expected).abs() < TOL,
                        "e_{}: (J^T J)[{}][{}] = {}, expected {}",
                        k, i, l, jtj, expected
                    );
                }
            }
        }
    }

    // === u(3) embedding premises ===

    #[test]
    fn test_restricted_action_is_skew_adjoint() {
        for k in 1..=7 {
            let decomp = stabilizer_decomposition(k);
            for (idx, d) in decomp.stabilizer_basis.iter().enumerate() {
                assert!(
                    is_skew_adjoint_on_perp(d, &decomp.perp_indices, TOL),
                    "e_{}: stab basis[{}] restricted action is NOT skew-adjoint",
                    k, idx
                );
            }
        }
    }

    #[test]
    fn test_stabilizer_commutes_with_j() {
        for k in 1..=7 {
            let decomp = stabilizer_decomposition(k);
            let cs = complex_structure(k);
            for (idx, d) in decomp.stabilizer_basis.iter().enumerate() {
                assert!(
                    commutes_with_j(d, &cs, TOL),
                    "e_{}: stab basis[{}] does NOT commute with J_{}",
                    k, idx, k
                );
            }
        }
    }

    #[test]
    fn test_stabilizer_dimension_is_8() {
        // Combined with tests 17 (skew-adjoint) and 18 (J_k-commutation),
        // this establishes the computational premises for the u(3) embedding.
        // The classification theorem (8-dim compact subalgebra of u(3) = su(3))
        // is stated in module docs, not tested here.
        for k in 1..=7 {
            let decomp = stabilizer_decomposition(k);
            assert_eq!(decomp.stabilizer_basis.len(), 8,
                "e_{}: stabilizer dimension must be 8 for u(3) embedding argument", k);
        }
    }

    // === Fano discovery ===

    #[test]
    fn test_fano_lines_exactly_three() {
        for k in 1..=7 {
            let cs = complex_structure(k);
            // fano_pairs has exactly 3 entries (enforced by array type)
            // Verify each pair has distinct indices
            for &(a, b, sign) in &cs.fano_pairs {
                assert!(a < b, "e_{}: Fano pair ({}, {}) violates a < b", k, a, b);
                assert!(sign == 1 || sign == -1, "e_{}: Fano sign must be +/-1", k);
            }
        }
    }

    #[test]
    fn test_fano_pairs_cover_complement() {
        for k in 1..=7 {
            let cs = complex_structure(k);
            let mut covered = [false; 6];
            for &(a, b, _) in &cs.fano_pairs {
                assert!(!covered[a], "e_{}: perp index {} used twice", k, a);
                assert!(!covered[b], "e_{}: perp index {} used twice", k, b);
                covered[a] = true;
                covered[b] = true;
            }
            assert!(
                covered.iter().all(|&c| c),
                "e_{}: Fano pairs do not cover all 6 perp indices", k
            );
        }
    }
}
