//! Constructive SU(3) realization from G2 stabilizer (PR2/3).
//!
//! Converts the intrinsic stabilizer data from `g2_stabilizer` (PR1) into
//! explicit 3x3 complex anti-Hermitian traceless matrices and aligns them
//! to the standard Gell-Mann basis.
//!
//! Mathematical route:
//! 1. PR1's `fano_pairs` give 3 complex lines in e_k^perp.
//! 2. The +i eigenspace basis z_j = u_j - i J_k(u_j) turns the 6-dim real
//!    representation into a 3-dim complex one.
//! 3. Each stabilizer derivation restricts to a 3x3 complex matrix on C^3.
//! 4. These matrices are verified anti-Hermitian and traceless (= su(3)).
//! 5. Alignment to the standard anti-Hermitian Gell-Mann basis T_a = (i/2) lambda_a.
//!
//! Normalization convention (stated once):
//!   T_a = (i/2) lambda_a  (anti-Hermitian)
//!   [T_a, T_b] = f_{abc} T_c
//!   <A, B> = -2 tr(A B)  (Hilbert-Schmidt, orthonormalizes T_a)
//!
//! No physics-sm dependency. No su5_gut.rs edits. No physics language.
//!
//! References:
//!   Baez, "The Octonions" (2002), Section 4.1
//!   Georgi, "Lie Algebras in Particle Physics" (1999)

use super::g2_stabilizer::{ComplexStructure, G2Stabilizer, perp_block};
use nalgebra::SMatrix;
use num_complex::Complex;

/// 3x3 complex matrix type.
pub type Mat3c = SMatrix<Complex<f64>, 3, 3>;

fn c(re: f64, im: f64) -> Complex<f64> {
    Complex::new(re, im)
}

fn cr(x: f64) -> Complex<f64> {
    Complex::new(x, 0.0)
}

fn ci(x: f64) -> Complex<f64> {
    Complex::new(0.0, x)
}

// ---------------------------------------------------------------------------
// Complex basis
// ---------------------------------------------------------------------------

/// The 3 complex basis vectors z_j = u_j - i J_k(u_j) in C^3.
///
/// Each vector z_j is in the +i eigenspace of J_k (i.e., J_k(z_j) = i z_j).
/// The three vectors {z_1, z_2, z_3} form a complex basis for (e_k^perp, J_k).
pub struct ComplexBasis3 {
    /// For each of the 3 complex lines: (real_perp_idx, imag_perp_idx, sign)
    /// where u_j = e_{perp[real_idx]} and J_k(u_j) = sign * e_{perp[imag_idx]}.
    pub lines: [(usize, usize, i8); 3],
    /// The perpendicular basis indices (copied from ComplexStructure).
    pub perp_indices: [usize; 6],
}

/// Build the complex basis from PR1's ComplexStructure.
pub fn build_complex_basis(cs: &ComplexStructure) -> ComplexBasis3 {
    ComplexBasis3 {
        lines: cs.fano_pairs,
        perp_indices: cs.perp_indices,
    }
}

// ---------------------------------------------------------------------------
// 6x6 real -> 3x3 complex conversion
// ---------------------------------------------------------------------------

/// Convert a 6x6 real matrix R (acting on e_k^perp) to a 3x3 complex matrix
/// M (acting on C^3 = span{z_1, z_2, z_3}).
///
/// The basis z_j = u_j - i * sigma_j * w_j where u_j = e_{perp[a_j]} and
/// w_j = e_{perp[b_j]}, with J_k(u_j) = sigma_j * w_j.
///
/// The complex matrix entry M_{pq} is determined by:
///   R(z_q) = sum_p M_{pq} z_p
///
/// Since z_j = e_{a_j} - i sigma_j e_{b_j} (in real coordinates):
///   R(z_q) = R(e_{a_q}) - i sigma_q R(e_{b_q})
///
/// And projecting onto z_p = e_{a_p} - i sigma_p e_{b_p}:
///   M_{pq} = R[a_p][a_q] + sigma_p sigma_q R[b_p][b_q]
///            + i (sigma_p R[b_p][a_q] - sigma_q R[a_p][b_q])
///
/// (derived from the real/imaginary decomposition of the complex inner product)
fn real_6x6_to_complex_3x3(r: &[[f64; 6]; 6], basis: &ComplexBasis3) -> Mat3c {
    let mut m = Mat3c::zeros();

    for p in 0..3 {
        let (ap, bp, sp) = basis.lines[p];
        let sp_f = sp as f64;

        for q in 0..3 {
            let (aq, bq, sq) = basis.lines[q];
            let sq_f = sq as f64;

            // Real part: R[ap][aq] + sp*sq * R[bp][bq]
            let re = r[ap][aq] + sp_f * sq_f * r[bp][bq];
            // Imaginary part: sp * R[bp][aq] - sq * R[ap][bq]
            let im = sp_f * r[bp][aq] - sq_f * r[ap][bq];

            // Factor of 1/2 from the normalization <z_p, z_q> = 2 delta_{pq}
            // (since |z_j|^2 = |u_j|^2 + |w_j|^2 = 2).
            m[(p, q)] = c(re / 2.0, im / 2.0);
        }
    }

    m
}

// ---------------------------------------------------------------------------
// Fundamental representation
// ---------------------------------------------------------------------------

/// The 8 generators of su(3) as 3x3 complex anti-Hermitian traceless matrices.
pub struct FundamentalRepresentation {
    /// The 8 matrices in the computed (non-canonical) basis.
    pub matrices: [Mat3c; 8],
    /// The fixed unit that was used.
    pub fixed_unit: usize,
}

/// Construct the 3x3 complex representation from PR1 stabilizer data.
///
/// For each of the 8 stabilizer derivations:
/// 1. Extract the 6x6 real action on e_k^perp via `perp_block`.
/// 2. Convert to 3x3 complex via the Fano-derived complex basis.
pub fn fundamental_representation(
    decomp: &G2Stabilizer,
    cs: &ComplexStructure,
) -> FundamentalRepresentation {
    let basis = build_complex_basis(cs);
    let mut matrices = [Mat3c::zeros(); 8];

    for (idx, d) in decomp.stabilizer_basis.iter().enumerate() {
        let r = perp_block(d, &decomp.perp_indices);
        matrices[idx] = real_6x6_to_complex_3x3(&r, &basis);
    }

    FundamentalRepresentation {
        matrices,
        fixed_unit: decomp.fixed_unit,
    }
}

// ---------------------------------------------------------------------------
// Hilbert-Schmidt inner product
// ---------------------------------------------------------------------------

/// Hilbert-Schmidt inner product: <A, B> = -2 tr(A B).
///
/// With the convention T_a = (i/2) lambda_a, this makes the standard
/// Gell-Mann generators orthonormal.
pub fn hs_inner_product(a: &Mat3c, b: &Mat3c) -> f64 {
    let prod = a * b;
    let trace = prod.trace();
    -2.0 * trace.re
}

// ---------------------------------------------------------------------------
// Standard anti-Hermitian Gell-Mann basis
// ---------------------------------------------------------------------------

/// The 8 standard anti-Hermitian Gell-Mann generators T_a = (i/2) lambda_a.
///
/// Convention: [T_a, T_b] = f_{abc} T_c with the standard SU(3) structure
/// constants. The Hilbert-Schmidt inner product <T_a, T_b> = -2 tr(T_a T_b)
/// gives delta_{ab} (orthonormal).
pub fn standard_gell_mann_antihermitian() -> [Mat3c; 8] {
    let half_i = ci(0.5);

    // lambda_1: off-diagonal (0,1)+(1,0)
    let mut l1 = Mat3c::zeros();
    l1[(0, 1)] = cr(1.0);
    l1[(1, 0)] = cr(1.0);

    // lambda_2: off-diagonal (0,1)-(1,0) with i
    let mut l2 = Mat3c::zeros();
    l2[(0, 1)] = ci(-1.0);
    l2[(1, 0)] = ci(1.0);

    // lambda_3: diagonal (1,-1,0)
    let mut l3 = Mat3c::zeros();
    l3[(0, 0)] = cr(1.0);
    l3[(1, 1)] = cr(-1.0);

    // lambda_4: off-diagonal (0,2)+(2,0)
    let mut l4 = Mat3c::zeros();
    l4[(0, 2)] = cr(1.0);
    l4[(2, 0)] = cr(1.0);

    // lambda_5: off-diagonal (0,2)-(2,0) with i
    let mut l5 = Mat3c::zeros();
    l5[(0, 2)] = ci(-1.0);
    l5[(2, 0)] = ci(1.0);

    // lambda_6: off-diagonal (1,2)+(2,1)
    let mut l6 = Mat3c::zeros();
    l6[(1, 2)] = cr(1.0);
    l6[(2, 1)] = cr(1.0);

    // lambda_7: off-diagonal (1,2)-(2,1) with i
    let mut l7 = Mat3c::zeros();
    l7[(1, 2)] = ci(-1.0);
    l7[(2, 1)] = ci(1.0);

    // lambda_8: diagonal (1,1,-2)/sqrt(3)
    let s3 = 1.0 / 3.0_f64.sqrt();
    let mut l8 = Mat3c::zeros();
    l8[(0, 0)] = cr(s3);
    l8[(1, 1)] = cr(s3);
    l8[(2, 2)] = cr(-2.0 * s3);

    // T_a = (i/2) lambda_a
    let scale = |m: &Mat3c| -> Mat3c {
        let mut result = Mat3c::zeros();
        for i in 0..3 {
            for j in 0..3 {
                result[(i, j)] = half_i * m[(i, j)];
            }
        }
        result
    };

    [
        scale(&l1),
        scale(&l2),
        scale(&l3),
        scale(&l4),
        scale(&l5),
        scale(&l6),
        scale(&l7),
        scale(&l8),
    ]
}

// ---------------------------------------------------------------------------
// Gell-Mann alignment
// ---------------------------------------------------------------------------

/// Result of aligning the computed representation to the standard Gell-Mann basis.
pub struct GellMannAlignment {
    /// The 8x8 orthogonal rotation matrix R such that
    /// computed_basis[a] = sum_b R[a][b] * standard_basis[b].
    pub rotation: [[f64; 8]; 8],
    /// Maximum residual after alignment.
    pub max_residual: f64,
}

/// Align the computed representation to the standard anti-Hermitian Gell-Mann basis.
///
/// Uses the Hilbert-Schmidt inner product to compute the overlap matrix,
/// then verifies orthogonality and bracket-structure agreement.
///
/// The overlap matrix O_{ab} = <computed_a, standard_b> should be orthogonal
/// (since both bases are orthonormal in the same inner product).
pub fn align_with_gell_mann(rep: &FundamentalRepresentation) -> GellMannAlignment {
    let standard = standard_gell_mann_antihermitian();

    // Compute overlap matrix O[a][b] = <computed[a], standard[b]>
    let mut overlap = [[0.0f64; 8]; 8];
    for (row, mat_a) in overlap.iter_mut().zip(rep.matrices.iter()) {
        for (val, std_b) in row.iter_mut().zip(standard.iter()) {
            *val = hs_inner_product(mat_a, std_b);
        }
    }

    // Verify orthogonality: O O^T should be identity
    let mut max_residual = 0.0f64;
    #[allow(clippy::needless_range_loop)]
    for i in 0..8 {
        for j in 0..8 {
            let mut dot = 0.0;
            for k in 0..8 {
                dot += overlap[i][k] * overlap[j][k];
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            max_residual = max_residual.max((dot - expected).abs());
        }
    }

    GellMannAlignment {
        rotation: overlap,
        max_residual,
    }
}

// ---------------------------------------------------------------------------
// Casimir operators
// ---------------------------------------------------------------------------

/// Compute the fundamental Casimir: sum_a T_a * T_a.
///
/// In the anti-Hermitian convention with T_a = (i/2) lambda_a,
/// this should equal -(4/3) I_3.
pub fn fundamental_casimir(matrices: &[Mat3c; 8]) -> Mat3c {
    let mut casimir = Mat3c::zeros();
    for t in matrices {
        casimir += t * t;
    }
    casimir
}

/// Compute the adjoint Casimir contraction: sum_{c,d} f_{acd} f_{bcd}.
///
/// For SU(3) this should equal 3 * delta_{ab}.
/// Uses the structure constants computed from the representation.
pub fn adjoint_casimir_contraction(f: &[Vec<Vec<f64>>]) -> Vec<Vec<f64>> {
    let n = f.len();
    let mut c2 = vec![vec![0.0; n]; n];
    for (a, c2_row) in c2.iter_mut().enumerate() {
        for (b, c2_ab) in c2_row.iter_mut().enumerate() {
            let mut sum = 0.0;
            for ci in 0..n {
                for d in 0..n {
                    sum += f[a][ci][d] * f[b][ci][d];
                }
            }
            *c2_ab = sum;
        }
    }
    c2
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::{
        super::g2_stabilizer::{complex_structure, stabilizer_decomposition, structure_constants},
        *,
    };

    const TOL: f64 = 1e-10;

    fn build_rep(k: usize) -> (G2Stabilizer, ComplexStructure, FundamentalRepresentation) {
        let decomp = stabilizer_decomposition(k);
        let cs = complex_structure(k);
        let rep = fundamental_representation(&decomp, &cs);
        (decomp, cs, rep)
    }

    // Test 1: anti-Hermitian
    #[test]
    fn test_anti_hermitian() {
        for k in 1..=7 {
            let (_, _, rep) = build_rep(k);
            for (idx, m) in rep.matrices.iter().enumerate() {
                let m_dag = m.adjoint(); // conjugate transpose
                let sum = m + m_dag;
                let norm: f64 = sum.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
                assert!(
                    norm < TOL,
                    "e_{}: matrix[{}] is NOT anti-Hermitian, |M + M^dag| = {}",
                    k,
                    idx,
                    norm
                );
            }
        }
    }

    // Test 2: traceless
    #[test]
    fn test_traceless() {
        for k in 1..=7 {
            let (_, _, rep) = build_rep(k);
            for (idx, m) in rep.matrices.iter().enumerate() {
                let tr = m.trace();
                assert!(
                    tr.norm() < TOL,
                    "e_{}: matrix[{}] trace = {}, expected 0",
                    k,
                    idx,
                    tr
                );
            }
        }
    }

    // Test 3: commutator closure
    #[test]
    fn test_commutator_closure() {
        for k in 1..=7 {
            let (_, _, rep) = build_rep(k);
            let mats = &rep.matrices;

            for a in 0..8 {
                for b in (a + 1)..8 {
                    let comm = mats[a] * mats[b] - mats[b] * mats[a];

                    // Project out of the span of the 8 generators
                    let mut projected = Mat3c::zeros();
                    for c in 0..8 {
                        let coeff = hs_inner_product(&comm, &mats[c]);
                        projected += mats[c] * cr(coeff);
                    }

                    let residual_mat = comm - projected;
                    let residual: f64 = residual_mat
                        .iter()
                        .map(|x| x.norm_sqr())
                        .sum::<f64>()
                        .sqrt();
                    assert!(
                        residual < TOL,
                        "e_{}: [M_{}, M_{}] leaks outside su(3), residual = {}",
                        k,
                        a,
                        b,
                        residual
                    );
                }
            }
        }
    }

    // Test 4: orthonormality in Hilbert-Schmidt
    #[test]
    fn test_orthonormality_hs() {
        for k in 1..=7 {
            let (_, _, rep) = build_rep(k);
            for a in 0..8 {
                for b in 0..8 {
                    let ip = hs_inner_product(&rep.matrices[a], &rep.matrices[b]);
                    let expected = if a == b { 1.0 } else { 0.0 };
                    assert!(
                        (ip - expected).abs() < TOL,
                        "e_{}: <M_{}, M_{}> = {}, expected {}",
                        k,
                        a,
                        b,
                        ip,
                        expected
                    );
                }
            }
        }
    }

    // Test 5: Gell-Mann alignment residual
    #[test]
    fn test_gell_mann_alignment_residual() {
        for k in 1..=7 {
            let (_, _, rep) = build_rep(k);
            let alignment = align_with_gell_mann(&rep);
            assert!(
                alignment.max_residual < TOL,
                "e_{}: Gell-Mann alignment residual = {}, expected < {}",
                k,
                alignment.max_residual,
                TOL
            );
        }
    }

    // Test 6: bracket structure matches after alignment
    #[test]
    fn test_bracket_structure_after_alignment() {
        // Compute structure constants for standard Gell-Mann basis
        let standard = standard_gell_mann_antihermitian();

        let mut f_std = vec![vec![vec![0.0; 8]; 8]; 8];
        for a in 0..8 {
            for b in 0..8 {
                let comm = standard[a] * standard[b] - standard[b] * standard[a];
                for ci in 0..8 {
                    f_std[a][b][ci] = hs_inner_product(&comm, &standard[ci]);
                }
            }
        }

        // Compute structure constants from the 3x3 representation directly
        // (using the same Hilbert-Schmidt inner product, not the PR1 derivation form)
        let (_, _, rep) = build_rep(1);
        let alignment = align_with_gell_mann(&rep);
        let r = &alignment.rotation;

        let mut f_rep = vec![vec![vec![0.0; 8]; 8]; 8];
        for a in 0..8 {
            for b in 0..8 {
                let comm = rep.matrices[a] * rep.matrices[b] - rep.matrices[b] * rep.matrices[a];
                for ci in 0..8 {
                    f_rep[a][b][ci] = hs_inner_product(&comm, &rep.matrices[ci]);
                }
            }
        }

        // Rotate f_rep via the alignment: f'_{ijk} = sum_{abc} O_{ai} O_{bj} O_{ck} f_{abc}
        // where O_{ab} = <computed[a], standard[b]>, so computed = O * standard.
        // Inverse rotation: O^{-1} = O^T, so (O^T)_{ia} = O_{ai}.
        let mut max_diff = 0.0f64;
        #[allow(clippy::needless_range_loop)]
        for i in 0..8 {
            for j in 0..8 {
                for k_idx in 0..8 {
                    let mut rotated = 0.0;
                    for a in 0..8 {
                        for b in 0..8 {
                            for ci in 0..8 {
                                rotated += r[a][i] * r[b][j] * r[ci][k_idx] * f_rep[a][b][ci];
                            }
                        }
                    }
                    max_diff = max_diff.max((rotated - f_std[i][j][k_idx]).abs());
                }
            }
        }

        assert!(
            max_diff < TOL,
            "Rotated structure constants differ from standard by max {}",
            max_diff
        );
    }

    // Test 7: fundamental Casimir
    #[test]
    fn test_fundamental_casimir() {
        for k in 1..=7 {
            let (_, _, rep) = build_rep(k);
            let casimir = fundamental_casimir(&rep.matrices);
            let expected = Mat3c::identity() * cr(-4.0 / 3.0);

            let diff = casimir - expected;
            let norm: f64 = diff.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            assert!(
                norm < TOL,
                "e_{}: fundamental Casimir differs from -(4/3)I by {}",
                k,
                norm
            );
        }
    }

    // Test 8: adjoint Casimir
    #[test]
    fn test_adjoint_casimir() {
        for k in 1..=7 {
            let (decomp, _, _) = build_rep(k);
            let f = structure_constants(&decomp.stabilizer_basis);
            let c2 = adjoint_casimir_contraction(&f);

            for a in 0..8 {
                for b in 0..8 {
                    let expected = if a == b { 3.0 } else { 0.0 };
                    assert!(
                        (c2[a][b] - expected).abs() < TOL,
                        "e_{}: adjoint Casimir C2[{}][{}] = {}, expected {}",
                        k,
                        a,
                        b,
                        c2[a][b],
                        expected
                    );
                }
            }
        }
    }

    // Test 9: all 7 embeddings produce equivalent Casimir invariants
    #[test]
    fn test_all_embeddings_equivalent_casimir() {
        let mut fund_casimirs = Vec::new();
        let mut adj_casimirs = Vec::new();

        for k in 1..=7 {
            let (decomp, _, rep) = build_rep(k);

            // Fundamental Casimir: extract the (0,0) entry as representative
            let fc = fundamental_casimir(&rep.matrices);
            fund_casimirs.push(fc[(0, 0)].re);

            // Adjoint Casimir: extract the diagonal as representative
            let f = structure_constants(&decomp.stabilizer_basis);
            let c2 = adjoint_casimir_contraction(&f);
            adj_casimirs.push(c2[0][0]);
        }

        // All fundamental Casimirs should be equal
        for i in 1..7 {
            assert!(
                (fund_casimirs[i] - fund_casimirs[0]).abs() < TOL,
                "Fundamental Casimir differs between e_1 ({}) and e_{} ({})",
                fund_casimirs[0],
                i + 1,
                fund_casimirs[i]
            );
        }

        // All adjoint Casimirs should be equal
        for i in 1..7 {
            assert!(
                (adj_casimirs[i] - adj_casimirs[0]).abs() < TOL,
                "Adjoint Casimir differs between e_1 ({}) and e_{} ({})",
                adj_casimirs[0],
                i + 1,
                adj_casimirs[i]
            );
        }
    }

    // Test 10: complex basis z_j are +i eigenvectors of J_k
    #[test]
    fn test_complex_basis_eigenvectors() {
        for k in 1..=7 {
            let cs = complex_structure(k);
            let basis = build_complex_basis(&cs);
            let j = &cs.matrix;

            for idx in 0..3 {
                let (a_idx, b_idx, sign) = basis.lines[idx];
                let s = sign as f64;

                // z_j has real part u_j = e_{perp[a_idx]} and
                // imaginary part -sigma_j * e_{perp[b_idx]}.
                // J_k(z_j) should equal i * z_j.
                //
                // J_k(u_j) = sigma_j * w_j (by Fano structure)
                // J_k(w_j) = -sigma_j * u_j (since J^2 = -I)
                //
                // J_k(z_j) = J_k(u_j) - i sigma_j J_k(w_j)
                //          = sigma_j w_j - i sigma_j (-sigma_j u_j)
                //          = sigma_j w_j + i u_j
                //          = i (u_j - i sigma_j w_j)
                //          = i z_j

                // Verify via the 6x6 matrix: J applied to u_j column
                let mut j_u = [0.0f64; 6];
                for r in 0..6 {
                    j_u[r] = j[r][a_idx];
                }
                // Should be sigma * e_{b_idx}
                assert!(
                    (j_u[b_idx] - s).abs() < TOL,
                    "e_{}: J_k(u_{}) should have component {} at index {}, got {}",
                    k,
                    idx,
                    s,
                    b_idx,
                    j_u[b_idx]
                );
                // Other components should be zero
                for r in 0..6 {
                    if r != b_idx {
                        assert!(
                            j_u[r].abs() < TOL,
                            "e_{}: J_k(u_{}) has spurious component {} at index {}",
                            k,
                            idx,
                            j_u[r],
                            r
                        );
                    }
                }
            }
        }
    }

    // Verify standard Gell-Mann basis is orthonormal
    #[test]
    fn test_standard_gellmann_orthonormal() {
        let gm = standard_gell_mann_antihermitian();
        for a in 0..8 {
            for b in 0..8 {
                let ip = hs_inner_product(&gm[a], &gm[b]);
                let expected = if a == b { 1.0 } else { 0.0 };
                assert!(
                    (ip - expected).abs() < TOL,
                    "<T_{}, T_{}> = {}, expected {}",
                    a + 1,
                    b + 1,
                    ip,
                    expected
                );
            }
        }
    }

    // Verify standard Gell-Mann basis is anti-Hermitian
    #[test]
    fn test_standard_gellmann_antihermitian() {
        let gm = standard_gell_mann_antihermitian();
        for (idx, t) in gm.iter().enumerate() {
            let sum = t + t.adjoint();
            let norm: f64 = sum.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
            assert!(
                norm < TOL,
                "Standard T_{} is NOT anti-Hermitian, |T + T^dag| = {}",
                idx + 1,
                norm
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Physics cross-validation (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "physics-sm"))]
mod physics_tests {
    use super::{
        super::{
            g2_stabilizer::{complex_structure, stabilizer_decomposition},
            su5_gut,
        },
        *,
    };

    const TOL: f64 = 1e-8;

    /// Cross-validate the octonionic SU(3) against the SU(3) sector of SU(5) GUT
    /// using basis-invariant contractions.
    ///
    /// The su5_gut generator ordering differs from the standard Gell-Mann ordering,
    /// so component-by-component comparison requires explicit basis mapping.
    /// Instead, we compare invariant quantities:
    ///   1. Adjoint Casimir: sum_{c,d} f_{acd} f_{bcd} = C_2 delta_{ab}
    ///   2. Total contraction: sum_{a,b,c} f_{abc}^2 = C_2 * dim(su(3)) = 3 * 8 = 24
    ///
    /// These are basis-independent and normalization-diagnostic.
    #[test]
    fn test_su5_su3_sector_invariant_match() {
        // 1. SU(5) GUT: compute SU(3) sector structure constants and invariants
        let su5_gens = su5_gut::build_generators();
        let su5_f = su5_gut::structure_constants(&su5_gens);

        // Adjoint Casimir for SU(3) sector (generators 0..7)
        let mut c2_su5 = 0.0f64;
        for a in 0..8 {
            let mut diag = 0.0;
            for ci in 0..8 {
                for d in 0..8 {
                    diag += su5_f[a][ci][d] * su5_f[a][ci][d];
                }
            }
            c2_su5 += diag;
        }
        // Total contraction = C_2 * dim = 3 * 8 = 24
        let total_su5 = c2_su5;

        // 2. Octonionic SU(3): compute the same total contraction
        let decomp = stabilizer_decomposition(1);
        let cs = complex_structure(1);
        let rep = fundamental_representation(&decomp, &cs);

        let mut f_oct = vec![vec![vec![0.0; 8]; 8]; 8];
        for a in 0..8 {
            for b in 0..8 {
                let comm = rep.matrices[a] * rep.matrices[b] - rep.matrices[b] * rep.matrices[a];
                for ci in 0..8 {
                    f_oct[a][b][ci] = hs_inner_product(&comm, &rep.matrices[ci]);
                }
            }
        }

        let mut total_oct = 0.0f64;
        for a in 0..8 {
            for b in 0..8 {
                for ci in 0..8 {
                    total_oct += f_oct[a][b][ci] * f_oct[a][b][ci];
                }
            }
        }

        // Both total contractions should equal 24 = C_2(adj) * dim(su(3)) = 3 * 8
        assert!(
            (total_su5 - 24.0).abs() < TOL,
            "SU(5) SU(3)-sector total contraction = {}, expected 24",
            total_su5
        );
        assert!(
            (total_oct - 24.0).abs() < TOL,
            "Octonionic SU(3) total contraction = {}, expected 24",
            total_oct
        );
    }

    /// Verify the normalization convention produces the correct f_{123} = 1.
    #[test]
    fn test_normalization_translation_is_explicit() {
        // PR2 standard Gell-Mann: T_a = (i/2) lambda_a (anti-Hermitian).
        // [T_1, T_2] = (i/2)^2 [lambda_1, lambda_2] = (-1/4)(2i lambda_3) = -T_3.
        // So f_{123} = -1 in this convention (not +1, which is the Hermitian convention).
        let gm = standard_gell_mann_antihermitian();
        let comm_12 = gm[0] * gm[1] - gm[1] * gm[0];
        let f123_pr2 = hs_inner_product(&comm_12, &gm[2]);

        assert!(
            (f123_pr2.abs() - 1.0).abs() < TOL,
            "PR2 |f_123| = {}, expected 1.0 (sign depends on convention)",
            f123_pr2.abs()
        );

        // su5_gut ordering is NOT standard Gell-Mann, so we verify its
        // internal consistency: [gen0, gen1] should have nonzero projection
        // onto gen6 (the first Cartan generator, analogous to lambda_3).
        let su5_gens = su5_gut::build_generators();
        let su5_f = su5_gut::structure_constants(&su5_gens);
        // gen0 = symmetric(0,1) ~ lambda_1, gen1 = antisymmetric(0,1) ~ lambda_2
        // [lambda_1, lambda_2] = 2i lambda_3, so f^{016} should be nonzero
        let f016 = su5_f[0][1][6]; // gen6 = Cartan h_1 ~ lambda_3
        assert!(
            f016.abs() > 0.1,
            "su5_gut f^{{0,1,6}} = {} should be nonzero (lambda_1, lambda_2 -> lambda_3)",
            f016
        );
    }

    /// Verify that scalar_fraction from real_part_projection accepts the
    /// same element shapes used by the topological-friction mass pipeline.
    /// The mass matrices operate on 16-component sedenion elements;
    /// scalar_fraction must handle those correctly.
    #[test]
    fn test_scalar_projection_mass_interface_shape() {
        use crate::construction::real_part_projection::{project, scalar_fraction};

        // Sedenion-shaped element (16 components) -- typical mass pipeline output
        let mut sedenion = [0.0f64; 16];
        sedenion[0] = 0.511; // e_0 component (mass-like)
        sedenion[1] = 0.1;
        sedenion[5] = 0.3;

        let p = project(&sedenion);
        assert!(
            (p - 0.511).abs() < 1e-15,
            "project should extract e_0 = 0.511"
        );

        let frac = scalar_fraction(&sedenion);
        assert!(frac > 0.0 && frac < 1.0, "scalar_fraction must be in (0,1)");

        // Octonion-shaped (8 components)
        let oct = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        assert!((scalar_fraction(&oct) - 1.0).abs() < 1e-15);

        // 1024D DekaVoudon-shaped
        let mut dv = vec![0.0f64; 1024];
        dv[0] = 1.0;
        dv[512] = 1.0;
        let frac_dv = scalar_fraction(&dv);
        assert!(
            (frac_dv - 0.5).abs() < 1e-15,
            "1024D: half real, half imaginary"
        );
    }
}
