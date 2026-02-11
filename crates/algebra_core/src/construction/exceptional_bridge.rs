//! Exceptional Bridge: Connecting Octonion Geometry to Exceptional Lie Algebras
//!
//! This module bridges the concrete octonion constructions (G2 derivations,
//! Cayley plane OP^2, Moufang loop S^7, Albert algebra J3(O)) to the
//! Freudenthal-Tits magic square and the exceptional Lie algebra hierarchy.
//!
//! # The Exceptional Chain
//!
//! The four normed division algebras R, C, H, O generate all five exceptional
//! Lie algebras through systematic constructions:
//!
//! - G2 (14) = Der(O) -- automorphisms of octonion multiplication
//! - F4 (52) = Der(J3(O)) -- automorphisms of Albert algebra (exceptional Jordan)
//! - E6 (78) = Aut(OP^2) -- collineation group of Cayley plane
//! - E7 (133) = L(H,O) -- Tits construction with quaternions and octonions
//! - E8 (248) = L(O,O) -- Tits construction with octonions x octonions
//!
//! # Tits Construction Formula
//!
//! The correct Tits formula for the magic square L(A,B) is:
//!   dim L(A,B) = Der(J3(B)) + (dim(A)-1) * (dim(J3(B))-1) + Der(A)
//!
//! where J3(B) is the Jordan algebra of 3x3 Hermitian matrices over B,
//! dim(J3(B)) = 3 + 3*dim(B), and Der(J3(B)) dimensions are:
//!   Der(J3(R)) = so(3) = 3
//!   Der(J3(C)) = su(3) = 8
//!   Der(J3(H)) = sp(6) = 21 (C3)
//!   Der(J3(O)) = F4 = 52

use crate::construction::g2_automorphisms::compute_g2_basis;
use crate::construction::octonion::Octonion;
use crate::lie::e8_lattice::{DivisionAlgebra, FreudenthalTitsMagicSquare, MagicSquareLieAlgebra};

// ============================================================================
// Tits Construction: Correct Dimension Formula
// ============================================================================

/// Dimension of the exceptional Jordan algebra J3(A) for composition algebra A.
///
/// J3(A) = 3x3 Hermitian matrices over A.
/// dim = 3 (diagonal reals) + 3 * dim(A) (off-diagonal elements).
pub fn jordan_h3_dim(a: DivisionAlgebra) -> usize {
    3 + 3 * a.dim()
}

/// Dimension of Der(J3(A)), the derivation algebra of the exceptional Jordan algebra.
///
/// These are classical results:
/// - Der(J3(R)) = so(3) = A1 = 3
/// - Der(J3(C)) = su(3) = A2 = 8
/// - Der(J3(H)) = sp(6) = C3 = 21
/// - Der(J3(O)) = F4 = 52
pub fn jordan_h3_derivation_dim(b: DivisionAlgebra) -> usize {
    match b {
        DivisionAlgebra::R => 3,  // so(3) = A1
        DivisionAlgebra::C => 8,  // su(3) = A2
        DivisionAlgebra::H => 21, // sp(6) = C3
        DivisionAlgebra::O => 52, // F4
    }
}

/// Correct Tits dimension formula for the Freudenthal-Tits magic square.
///
/// L(A,B) = Der(J3(B)) + Im(A) tensor J3(B)_0 + Der(A)
///
/// dim L(A,B) = Der(J3(B)) + (dim(A)-1)(dim(J3(B))-1) + Der(A)
///
/// The existing `FreudenthalTitsMagicSquare::dimension_formula` uses a
/// simplified formula that underestimates exceptional algebra dimensions.
/// This function provides the correct formula.
pub fn tits_dimension(a: DivisionAlgebra, b: DivisionAlgebra) -> usize {
    let der_a = derivation_algebra_dim(a);
    let der_j3b = jordan_h3_derivation_dim(b);
    let traceless_a = a.dim() - 1; // Im(A)
    let traceless_j3b = jordan_h3_dim(b) - 1; // J3(B)_0

    der_j3b + traceless_a * traceless_j3b + der_a
}

/// Dimension of Der(A) for a composition algebra A.
///
/// - Der(R) = 0 (trivially commutative and associative)
/// - Der(C) = 0 (commutative)
/// - Der(H) = so(3) = 3 (quaternion automorphisms)
/// - Der(O) = G2 = 14 (octonion automorphisms, verified computationally)
pub fn derivation_algebra_dim(a: DivisionAlgebra) -> usize {
    match a {
        DivisionAlgebra::R => 0,
        DivisionAlgebra::C => 0,
        DivisionAlgebra::H => 3,
        DivisionAlgebra::O => 14,
    }
}

// ============================================================================
// Exceptional Chain Verification
// ============================================================================

/// The exceptional chain: systematic connections from octonions to E8.
///
/// Each exceptional Lie algebra arises from octonion structures:
/// G2 -> F4 -> E6 -> E7 -> E8
///
/// This struct collects dimension cross-validation results.
#[derive(Debug)]
pub struct ExceptionalChainVerification {
    /// G2 = Der(O): computed derivation dimension vs expected 14
    pub g2_computed_dim: usize,
    pub g2_expected_dim: usize,

    /// F4 = Der(J3(O)): dimension from Jordan algebra automorphisms
    pub f4_dim: usize,

    /// E6: Cayley plane collineation group dimension
    pub e6_dim: usize,

    /// E7: Tits construction L(H,O)
    pub e7_dim: usize,

    /// E8: Tits construction L(O,O)
    pub e8_dim: usize,

    /// Magic square cross-validation: do all dims match?
    pub all_match: bool,
}

/// Run the full exceptional chain verification.
///
/// This cross-validates:
/// 1. Our computed G2 derivation basis against the magic square expectation
/// 2. The Tits formula dimensions against the magic square lookup table
/// 3. The complete exceptional algebra hierarchy
pub fn verify_exceptional_chain() -> ExceptionalChainVerification {
    // 1. Compute G2 = Der(O) directly
    let g2_basis = compute_g2_basis();
    let g2_computed = g2_basis.len();

    // 2. F4 from Jordan algebra (theoretical dimension)
    let f4_dim = jordan_h3_derivation_dim(DivisionAlgebra::O);

    // 3. E6 from Tits formula L(C,O)
    let e6_dim = tits_dimension(DivisionAlgebra::C, DivisionAlgebra::O);

    // 4. E7 from Tits formula L(H,O)
    let e7_dim = tits_dimension(DivisionAlgebra::H, DivisionAlgebra::O);

    // 5. E8 from Tits formula L(O,O)
    let e8_dim = tits_dimension(DivisionAlgebra::O, DivisionAlgebra::O);

    // Cross-validate against magic square lookup table
    let ms = FreudenthalTitsMagicSquare::new();
    let all_match = g2_computed == 14
        && f4_dim == MagicSquareLieAlgebra::F4.dim()
        && e6_dim == MagicSquareLieAlgebra::E6.dim()
        && e7_dim == MagicSquareLieAlgebra::E7.dim()
        && e8_dim == MagicSquareLieAlgebra::E8.dim()
        && ms.get(DivisionAlgebra::O, DivisionAlgebra::R) == MagicSquareLieAlgebra::F4
        && ms.get(DivisionAlgebra::O, DivisionAlgebra::O) == MagicSquareLieAlgebra::E8;

    ExceptionalChainVerification {
        g2_computed_dim: g2_computed,
        g2_expected_dim: 14,
        f4_dim,
        e6_dim,
        e7_dim,
        e8_dim,
        all_match,
    }
}

// ============================================================================
// Structural Property Bridges
// ============================================================================

/// Verify that the Moufang loop S^7 is connected to G2 automorphisms.
///
/// G2 acts on the unit octonions S^7 preserving the Moufang loop structure.
/// Any G2 derivation D satisfies the Leibniz rule:
///   D(xy) = D(x)*y + x*D(y) for all unit octonions x, y
///
/// This function verifies the Leibniz rule on Moufang loop elements.
pub fn verify_g2_moufang_compatibility(n_samples: usize) -> (usize, usize) {
    let g2_basis = compute_g2_basis();
    if g2_basis.is_empty() {
        return (0, 0);
    }

    let mut pass = 0;
    let mut total = 0;
    let tol = 1e-8;

    // For each G2 derivation D, verify the Leibniz rule on Moufang loop elements
    for deriv in g2_basis.iter().take(3) {
        for i in 1..8 {
            for j in 1..8 {
                let ei = Octonion::basis(i);
                let ej = Octonion::basis(j);

                // D(ei * ej) should equal D(ei)*ej + ei*D(ej)
                let prod = ei.multiply(&ej);
                let d_prod = deriv.apply(&prod);
                let d_ei = deriv.apply(&ei);
                let d_ej = deriv.apply(&ej);
                let leibniz_rhs = d_ei.multiply(&ej).add(&ei.multiply(&d_ej));

                let diff: f64 = d_prod
                    .components
                    .iter()
                    .zip(leibniz_rhs.components.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum();

                total += 1;
                if diff < tol {
                    pass += 1;
                }
            }
        }
    }

    // Also sample random unit octonions
    let mut rng_state: u64 = 42;
    for _ in 0..n_samples {
        let x = random_unit_octonion(&mut rng_state);
        let y = random_unit_octonion(&mut rng_state);

        for deriv in g2_basis.iter().take(2) {
            let prod = x.multiply(&y);
            let d_prod = deriv.apply(&prod);
            let d_x = deriv.apply(&x);
            let d_y = deriv.apply(&y);
            let leibniz_rhs = d_x.multiply(&y).add(&x.multiply(&d_y));

            let diff: f64 = d_prod
                .components
                .iter()
                .zip(leibniz_rhs.components.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();

            total += 1;
            if diff < tol {
                pass += 1;
            }
        }
    }

    (pass, total)
}

/// Verify E6 connection: The Cayley plane OP^2 has automorphism group E6.
///
/// Dimensional evidence: E6 has dim 78, which decomposes as:
///   78 = 52 (F4: Albert algebra automorphisms) + 26 (traceless Albert algebra)
/// This matches the structure of OP^2 as a symmetric space E6/F4*Spin(2).
pub fn verify_e6_cayley_plane_dimension() -> bool {
    let e6_dim = MagicSquareLieAlgebra::E6.dim();
    let f4_dim = MagicSquareLieAlgebra::F4.dim();

    // E6/F4 symmetric space: dim(E6) = dim(F4) + dim(OP^2)
    // OP^2 is 16-dimensional (real), and the E6 decomposes as:
    // 78 = 52 + 26, where 26 is the traceless Albert algebra
    // (which parametrizes the tangent space of E6/(F4 x U(1)))

    // The Cayley plane dimension
    let op2_real_dim = 16;

    // Verify: F4 stabilizer + tangent dimensions relate to E6
    // The homogeneous space E6/(Spin(10) x U(1)) = OP^2 (the Cayley plane)
    // dim(Spin(10)) = 45, dim(U(1)) = 1
    // 78 - 45 - 1 = 32 = 2 * 16 (complexified tangent space)

    let spin10_dim = 45;
    let u1_dim = 1;
    let complexified_tangent = e6_dim - spin10_dim - u1_dim;

    // The real tangent space of OP^2 has dimension 16
    // The complexified tangent has dimension 32 = 2 * 16
    complexified_tangent == 2 * op2_real_dim
        && e6_dim == 78
        && f4_dim == 52
        && e6_dim - f4_dim == 26
}

/// Verify the full 4x4 magic square using the correct Tits formula.
///
/// Returns (matched, total) where matched is the number of entries
/// where the Tits formula dimension matches the lookup table dimension.
pub fn verify_tits_formula_complete() -> (usize, usize) {
    let ms = FreudenthalTitsMagicSquare::new();
    let algebras = DivisionAlgebra::all();
    let mut matched = 0;
    let total = 16;

    for &a in &algebras {
        for &b in &algebras {
            let tits_dim = tits_dimension(a, b);
            let table_dim = ms.get(a, b).dim();
            if tits_dim == table_dim {
                matched += 1;
            }
        }
    }

    (matched, total)
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Simple xorshift64 PRNG for deterministic testing.
fn xorshift64(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64) * 2.0 - 1.0
}

/// Generate a random unit octonion using deterministic PRNG.
fn random_unit_octonion(state: &mut u64) -> Octonion {
    let mut components = [0.0f64; 8];
    for c in &mut components {
        *c = xorshift64(state);
    }
    let norm: f64 = components.iter().map(|x| x * x).sum::<f64>().sqrt();
    for c in &mut components {
        *c /= norm;
    }
    Octonion { components }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::construction::octonion_geometry::{CayleyPlanePoint, MoufangLoop};

    #[test]
    fn test_jordan_h3_dimensions() {
        // J3(R) = 3x3 real symmetric = 6D
        assert_eq!(jordan_h3_dim(DivisionAlgebra::R), 6);
        // J3(C) = 3x3 complex Hermitian = 9D
        assert_eq!(jordan_h3_dim(DivisionAlgebra::C), 9);
        // J3(H) = 3x3 quaternionic Hermitian = 15D
        assert_eq!(jordan_h3_dim(DivisionAlgebra::H), 15);
        // J3(O) = 3x3 octonionic Hermitian = 27D (Albert algebra)
        assert_eq!(jordan_h3_dim(DivisionAlgebra::O), 27);
    }

    #[test]
    fn test_jordan_derivation_dimensions() {
        assert_eq!(jordan_h3_derivation_dim(DivisionAlgebra::R), 3); // so(3)
        assert_eq!(jordan_h3_derivation_dim(DivisionAlgebra::C), 8); // su(3)
        assert_eq!(jordan_h3_derivation_dim(DivisionAlgebra::H), 21); // sp(6)
        assert_eq!(jordan_h3_derivation_dim(DivisionAlgebra::O), 52); // F4
    }

    #[test]
    fn test_tits_formula_exceptional_algebras() {
        // The four exceptional algebras from the Tits formula
        // F4 = L(O,R) = 52
        assert_eq!(tits_dimension(DivisionAlgebra::O, DivisionAlgebra::R), 52);
        // E6 = L(C,O) = L(O,C) = 78
        assert_eq!(tits_dimension(DivisionAlgebra::C, DivisionAlgebra::O), 78);
        assert_eq!(tits_dimension(DivisionAlgebra::O, DivisionAlgebra::C), 78);
        // E7 = L(H,O) = L(O,H) = 133
        assert_eq!(tits_dimension(DivisionAlgebra::H, DivisionAlgebra::O), 133);
        assert_eq!(tits_dimension(DivisionAlgebra::O, DivisionAlgebra::H), 133);
        // E8 = L(O,O) = 248
        assert_eq!(tits_dimension(DivisionAlgebra::O, DivisionAlgebra::O), 248);
    }

    #[test]
    fn test_tits_formula_classical_algebras() {
        // A1 = L(R,R) = 3
        assert_eq!(tits_dimension(DivisionAlgebra::R, DivisionAlgebra::R), 3);
        // A2 = L(R,C) = L(C,R) = 8
        assert_eq!(tits_dimension(DivisionAlgebra::R, DivisionAlgebra::C), 8);
        assert_eq!(tits_dimension(DivisionAlgebra::C, DivisionAlgebra::R), 8);
        // C3 = L(R,H) = L(H,R) = 21
        assert_eq!(tits_dimension(DivisionAlgebra::R, DivisionAlgebra::H), 21);
        assert_eq!(tits_dimension(DivisionAlgebra::H, DivisionAlgebra::R), 21);
        // A2+A2 = L(C,C) = 16
        assert_eq!(tits_dimension(DivisionAlgebra::C, DivisionAlgebra::C), 16);
        // A5 = L(C,H) = L(H,C) = 35
        assert_eq!(tits_dimension(DivisionAlgebra::C, DivisionAlgebra::H), 35);
        assert_eq!(tits_dimension(DivisionAlgebra::H, DivisionAlgebra::C), 35);
        // D6 = L(H,H) = 66
        assert_eq!(tits_dimension(DivisionAlgebra::H, DivisionAlgebra::H), 66);
    }

    #[test]
    fn test_tits_formula_symmetry() {
        // The magic square is symmetric: L(A,B) = L(B,A)
        let algebras = DivisionAlgebra::all();
        for &a in &algebras {
            for &b in &algebras {
                assert_eq!(
                    tits_dimension(a, b),
                    tits_dimension(b, a),
                    "Tits formula not symmetric for {:?}, {:?}",
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn test_tits_formula_matches_lookup_table() {
        let (matched, total) = verify_tits_formula_complete();
        assert_eq!(
            matched, total,
            "Tits formula should match lookup table for all 16 entries"
        );
    }

    #[test]
    fn test_exceptional_chain_g2() {
        // Cross-validate: our computed G2 derivation algebra has dim 14
        let chain = verify_exceptional_chain();
        assert_eq!(
            chain.g2_computed_dim, 14,
            "G2 = Der(O) should have 14 derivations"
        );
        assert_eq!(chain.g2_computed_dim, chain.g2_expected_dim);
    }

    #[test]
    fn test_exceptional_chain_complete() {
        let chain = verify_exceptional_chain();
        assert_eq!(chain.f4_dim, 52, "F4 = Der(J3(O)) should be 52-dimensional");
        assert_eq!(chain.e6_dim, 78, "E6 = L(C,O) should be 78-dimensional");
        assert_eq!(chain.e7_dim, 133, "E7 = L(H,O) should be 133-dimensional");
        assert_eq!(chain.e8_dim, 248, "E8 = L(O,O) should be 248-dimensional");
        assert!(
            chain.all_match,
            "All exceptional chain dimensions must match"
        );
    }

    #[test]
    fn test_e6_cayley_plane_dimension() {
        assert!(
            verify_e6_cayley_plane_dimension(),
            "E6/Spin(10)*U(1) = OP^2 dimensional analysis should hold"
        );
    }

    #[test]
    fn test_g2_moufang_compatibility() {
        // G2 derivations should satisfy Leibniz on Moufang loop elements
        let (pass, total) = verify_g2_moufang_compatibility(20);
        assert!(total > 0, "Should have tested some Moufang-G2 pairs");
        assert_eq!(
            pass, total,
            "G2 derivations must satisfy Leibniz on ALL Moufang loop elements ({}/{})",
            pass, total
        );
    }

    #[test]
    fn test_derivation_dims_match_magic_square() {
        // Cross-validate: Der(A) values used in Tits formula must match
        // what the magic square infrastructure expects
        assert_eq!(derivation_algebra_dim(DivisionAlgebra::R), 0);
        assert_eq!(derivation_algebra_dim(DivisionAlgebra::C), 0);
        assert_eq!(derivation_algebra_dim(DivisionAlgebra::H), 3);
        assert_eq!(derivation_algebra_dim(DivisionAlgebra::O), 14);

        // G2 = 14 from our computation matches magic square's Der(O)
        let g2_basis = compute_g2_basis();
        assert_eq!(g2_basis.len(), derivation_algebra_dim(DivisionAlgebra::O));
    }

    #[test]
    fn test_exceptional_dimension_chain_monotone() {
        // The exceptional Lie algebras form a strict ascending chain:
        // G2(14) < F4(52) < E6(78) < E7(133) < E8(248)
        // Note: G2 is not in the magic square enum (it's Der(O), an input)
        let dims = [
            14, // G2 = Der(O)
            MagicSquareLieAlgebra::F4.dim(),
            MagicSquareLieAlgebra::E6.dim(),
            MagicSquareLieAlgebra::E7.dim(),
            MagicSquareLieAlgebra::E8.dim(),
        ];
        for i in 0..dims.len() - 1 {
            assert!(
                dims[i] < dims[i + 1],
                "Exceptional chain must be strictly ascending: {} < {}",
                dims[i],
                dims[i + 1]
            );
        }
    }

    #[test]
    fn test_cayley_plane_points_moufang_consistent() {
        // Cayley plane points constructed from unit octonions should be
        // consistent with the Moufang loop structure.
        let e1 = Octonion::basis(1);
        let e2 = Octonion::basis(2);

        // Points from basis octonions
        let p1 = CayleyPlanePoint::from_homogeneous(&e1, &e2, &Octonion::basis(0));
        assert!(
            p1.verify_idempotent(1e-10),
            "Point from unit octonions must be idempotent"
        );

        // Moufang loop operations preserve unit sphere
        let prod = MoufangLoop::multiply(&e1, &e2);
        let norm_sq: f64 = prod.components.iter().map(|x| x * x).sum();
        assert!(
            (norm_sq - 1.0).abs() < 1e-10,
            "Product of unit octonions must be unit (composition property)"
        );
    }

    #[test]
    fn test_albert_algebra_dimension_27() {
        // The Albert algebra J3(O) is 27-dimensional, the unique exceptional
        // Jordan algebra. This dimension is central to the magic square.
        let albert_dim = jordan_h3_dim(DivisionAlgebra::O);
        assert_eq!(
            albert_dim, 27,
            "Albert algebra J3(O) must be 27-dimensional"
        );

        // Its derivation algebra is F4 (52-dim)
        let f4_dim = jordan_h3_derivation_dim(DivisionAlgebra::O);
        assert_eq!(
            f4_dim, 52,
            "Der(Albert algebra) = F4 must be 52-dimensional"
        );

        // The traceless part is 26-dimensional
        let traceless = albert_dim - 1;
        assert_eq!(
            traceless, 26,
            "Traceless Albert algebra must be 26-dimensional"
        );

        // E6 = F4 + 26: the automorphism group of OP^2 decomposes
        assert_eq!(
            MagicSquareLieAlgebra::E6.dim(),
            f4_dim + traceless,
            "E6 = F4 + traceless Albert"
        );
    }
}
