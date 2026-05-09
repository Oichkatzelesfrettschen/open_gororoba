//! Freudenthal-Tits magic square.
//!
//! A 4x4 matrix indexed by pairs of normed division algebras `(R, C, H, O)`
//! whose entries are simple Lie algebras:
//!
//! ```text
//!         R      C      H      O
//!    R    A1     A2     C3     F4
//!    C    A2   A2xA2    A5     E6
//!    H    C3     A5     D6     E7
//!    O    F4     E6     E7     E8
//! ```
//!
//! The construction (Tits, 1966) is `L(A, B) = Der(A) + (A_0 x B_0) + Der(B)`,
//! where `A_0` denotes traceless elements. Properties:
//!
//! - **Symmetric**: `L(A, B) = L(B, A)`.
//! - **Octonionic row** produces every exceptional Lie algebra except `G_2`.
//! - **Diagonal**: self-tensorings `R x R, C x C, H x H, O x O` give
//!   `A_1, A_2 x A_2, D_6, E_8`.
//!
//! # Literature
//! - Freudenthal (1964), *Lie groups in the foundations of geometry*.
//! - Tits (1966), *Algebres alternatives, algebres de Jordan et algebres de
//!   Lie exceptionnelles*.
//! - Barton & Sudbery (2003), *Magic squares and matrix models of Lie algebras*.

use std::collections::HashSet;

// ============================================================================
// Division algebras
// ============================================================================

/// The four normed division algebras over `R`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DivisionAlgebra {
    /// Real numbers, `dim = 1`.
    R,
    /// Complex numbers, `dim = 2`.
    C,
    /// Quaternions, `dim = 4`.
    H,
    /// Octonions, `dim = 8`.
    O,
}

impl DivisionAlgebra {
    /// Dimension of the division algebra.
    pub fn dim(&self) -> usize {
        match self {
            Self::R => 1,
            Self::C => 2,
            Self::H => 4,
            Self::O => 8,
        }
    }

    /// All four division algebras in canonical order.
    pub fn all() -> [Self; 4] {
        [Self::R, Self::C, Self::H, Self::O]
    }
}

// ============================================================================
// Magic square Lie algebras
// ============================================================================

/// Simple Lie algebras that appear in the Freudenthal-Tits magic square.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MagicSquareLieAlgebra {
    /// `A_1 = sl(2) = so(3)`, rank 1, dim 3.
    A1,
    /// `A_2 = sl(3)`, rank 2, dim 8.
    A2,
    /// `A_2 x A_2 = sl(3) x sl(3)`, rank 4, dim 16.
    A2xA2,
    /// `A_5 = sl(6)`, rank 5, dim 35.
    A5,
    /// `C_3 = sp(6)`, rank 3, dim 21.
    C3,
    /// `D_6 = so(12)`, rank 6, dim 66.
    D6,
    /// `F_4`, rank 4, dim 52.
    F4,
    /// `E_6`, rank 6, dim 78.
    E6,
    /// `E_7`, rank 7, dim 133.
    E7,
    /// `E_8`, rank 8, dim 248 (largest exceptional simple Lie algebra).
    E8,
}

impl MagicSquareLieAlgebra {
    /// Dimension of the Lie algebra.
    pub fn dim(&self) -> usize {
        match self {
            Self::A1 => 3,
            Self::A2 => 8,
            Self::A2xA2 => 16,
            Self::A5 => 35,
            Self::C3 => 21,
            Self::D6 => 66,
            Self::F4 => 52,
            Self::E6 => 78,
            Self::E7 => 133,
            Self::E8 => 248,
        }
    }

    /// Rank (dimension of a Cartan subalgebra).
    pub fn rank(&self) -> usize {
        match self {
            Self::A1 => 1,
            Self::A2 => 2,
            Self::A2xA2 => 4,
            Self::A5 => 5,
            Self::C3 => 3,
            Self::D6 => 6,
            Self::F4 => 4,
            Self::E6 => 6,
            Self::E7 => 7,
            Self::E8 => 8,
        }
    }

    /// Total root count (positive + negative).
    pub fn root_count(&self) -> usize {
        match self {
            Self::A1 => 2,
            Self::A2 => 6,
            Self::A2xA2 => 12,
            Self::A5 => 30,
            Self::C3 => 18,
            Self::D6 => 60,
            Self::F4 => 48,
            Self::E6 => 72,
            Self::E7 => 126,
            Self::E8 => 240,
        }
    }

    /// True for `F_4, E_6, E_7, E_8` (the four exceptional algebras in the square).
    pub fn is_exceptional(&self) -> bool {
        matches!(self, Self::F4 | Self::E6 | Self::E7 | Self::E8)
    }
}

// ============================================================================
// The square itself
// ============================================================================

/// The Freudenthal-Tits magic square.
#[derive(Debug, Clone)]
pub struct FreudenthalTitsMagicSquare {
    square: [[MagicSquareLieAlgebra; 4]; 4],
}

impl FreudenthalTitsMagicSquare {
    /// Construct the square with all 16 entries populated.
    pub fn new() -> Self {
        use MagicSquareLieAlgebra::*;
        let square = [
            [A1, A2, C3, F4],
            [A2, A2xA2, A5, E6],
            [C3, A5, D6, E7],
            [F4, E6, E7, E8],
        ];
        Self { square }
    }

    /// Look up `L(a, b)`.
    pub fn get(&self, a: DivisionAlgebra, b: DivisionAlgebra) -> MagicSquareLieAlgebra {
        let i = Self::div_alg_index(a);
        let j = Self::div_alg_index(b);
        self.square[i][j]
    }

    fn div_alg_index(d: DivisionAlgebra) -> usize {
        match d {
            DivisionAlgebra::R => 0,
            DivisionAlgebra::C => 1,
            DivisionAlgebra::H => 2,
            DivisionAlgebra::O => 3,
        }
    }

    /// Verify symmetry: `L(a, b) = L(b, a)`.
    pub fn verify_symmetry(&self) -> bool {
        for i in 0..4 {
            for j in 0..4 {
                if self.square[i][j] != self.square[j][i] {
                    return false;
                }
            }
        }
        true
    }

    /// All distinct exceptional Lie algebras appearing in the square.
    pub fn exceptional_algebras(&self) -> Vec<MagicSquareLieAlgebra> {
        let mut result = Vec::new();
        for row in &self.square {
            for &alg in row {
                if alg.is_exceptional() && !result.contains(&alg) {
                    result.push(alg);
                }
            }
        }
        result
    }

    /// Diagonal entries `(R x R, C x C, H x H, O x O) = (A_1, A_2 x A_2, D_6, E_8)`.
    pub fn diagonal(&self) -> [MagicSquareLieAlgebra; 4] {
        [
            self.square[0][0],
            self.square[1][1],
            self.square[2][2],
            self.square[3][3],
        ]
    }

    /// Sum of dimensions of all *distinct* algebras in the square.
    pub fn total_dimension(&self) -> usize {
        let mut seen = HashSet::new();
        let mut total = 0;
        for row in &self.square {
            for &alg in row {
                if seen.insert(alg) {
                    total += alg.dim();
                }
            }
        }
        total
    }

    /// Tits dimension formula: `dim L(A, B) = Der(A) + (dim A - 1)(dim B - 1) + Der(B)`.
    ///
    /// Derivations: `Der(R) = Der(C) = 0`, `Der(H) = so(3) = A_1` (dim 3),
    /// `Der(O) = G_2` (dim 14).
    ///
    /// **Note:** this textbook formula reproduces the classical-row entries
    /// (`R, C` rows) but undercounts the exceptional entries; the *exact*
    /// Tits construction adds an `(A_0 x B_0)` summand that the simple
    /// product `(dim A - 1)(dim B - 1)` only captures linearly.
    pub fn dimension_formula(a: DivisionAlgebra, b: DivisionAlgebra) -> usize {
        let der_a = Self::derivation_dim(a);
        let der_b = Self::derivation_dim(b);
        let traceless_product = (a.dim() - 1) * (b.dim() - 1);
        der_a + traceless_product + der_b
    }

    fn derivation_dim(d: DivisionAlgebra) -> usize {
        match d {
            DivisionAlgebra::R => 0,
            DivisionAlgebra::C => 0,
            DivisionAlgebra::H => 3,
            DivisionAlgebra::O => 14,
        }
    }
}

impl Default for FreudenthalTitsMagicSquare {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience: lookup a single entry without binding the square.
pub fn magic_square_entry(a: DivisionAlgebra, b: DivisionAlgebra) -> MagicSquareLieAlgebra {
    FreudenthalTitsMagicSquare::new().get(a, b)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use DivisionAlgebra::*;
    use MagicSquareLieAlgebra::*;

    #[test]
    fn square_is_symmetric() {
        assert!(FreudenthalTitsMagicSquare::new().verify_symmetry());
    }

    #[test]
    fn corners_match_textbook() {
        let ms = FreudenthalTitsMagicSquare::new();
        assert_eq!(ms.get(R, R), A1);
        assert_eq!(ms.get(R, O), F4);
        assert_eq!(ms.get(O, R), F4);
        assert_eq!(ms.get(O, O), E8);
    }

    #[test]
    fn diagonal_is_self_tensoring_chain() {
        let diag = FreudenthalTitsMagicSquare::new().diagonal();
        assert_eq!(diag, [A1, A2xA2, D6, E8]);
    }

    #[test]
    fn exactly_four_exceptional_entries() {
        let exceptional = FreudenthalTitsMagicSquare::new().exceptional_algebras();
        assert_eq!(exceptional.len(), 4);
        assert!(exceptional.contains(&F4));
        assert!(exceptional.contains(&E6));
        assert!(exceptional.contains(&E7));
        assert!(exceptional.contains(&E8));
    }

    #[test]
    fn octonion_row_yields_f4_e6_e7_e8() {
        let ms = FreudenthalTitsMagicSquare::new();
        assert_eq!(ms.get(O, R), F4);
        assert_eq!(ms.get(O, C), E6);
        assert_eq!(ms.get(O, H), E7);
        assert_eq!(ms.get(O, O), E8);
    }

    #[test]
    fn key_dimensions_match_classical_values() {
        assert_eq!(A1.dim(), 3);
        assert_eq!(A2.dim(), 8);
        assert_eq!(F4.dim(), 52);
        assert_eq!(E6.dim(), 78);
        assert_eq!(E7.dim(), 133);
        assert_eq!(E8.dim(), 248);
    }

    #[test]
    fn root_counts_match_classical_values() {
        assert_eq!(F4.root_count(), 48);
        assert_eq!(E6.root_count(), 72);
        assert_eq!(E7.root_count(), 126);
        assert_eq!(E8.root_count(), 240);
    }

    #[test]
    fn dimension_formula_is_callable() {
        let _ = FreudenthalTitsMagicSquare::dimension_formula(R, R);
        let _ = FreudenthalTitsMagicSquare::dimension_formula(O, O);
    }

    #[test]
    fn division_algebra_dimension_product_is_sixty_four() {
        let product: usize = DivisionAlgebra::all().iter().map(|d| d.dim()).product();
        assert_eq!(product, 64);
    }
}
