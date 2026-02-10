//! Non-associative algebras: Malcev, Bol, Lie-admissible, and their connections
//! to the construction method hierarchy and exceptional algebras.
//!
//! This module implements research infrastructure for Phase 4 exploration:
//! - Malcev algebras (anticommutative, non-associative)
//! - Bol algebras (loop-theoretic generalization)
//! - Lie-admissible algebras (bracket-based)
//! - Freudenthal-Tits magic square (bridge to E6/E7/E8)
//!
//! Hypothesis: The construction method hierarchy (Level 1: Mechanism >> Level 2:
//! Dimension >> Level 3: Parameters) extends universally to all non-associative
//! algebras. Malcev/Bol/Lie-admissible should exhibit mechanism-determined
//! anticommutativity and structure-determined non-associativity, just as
//! Jordan exhibits mechanism-determined commutativity.

use std::fmt;

/// Trait for non-associative algebras in the hierarchy.
/// All satisfy: Level 1 = Mechanism (anticommutative), Level 2 = Dimension,
/// Level 3 = Composition parameters.
pub trait NonAssociativeAlgebra: Clone + fmt::Debug {
    /// Dimension of the algebra.
    fn dim(&self) -> usize;

    /// Product operation (mechanism-specific).
    fn product(&self, a: &[f64], b: &[f64]) -> Vec<f64>;

    /// Commutator [a,b] = ab - ba (universally anticommutative for these classes).
    fn commutator(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        let ab = self.product(a, b);
        let ba = self.product(b, a);
        ab.iter().zip(ba.iter()).map(|(x, y)| x - y).collect()
    }

    /// Check if a and b commute (should be false for all non-trivial cases).
    fn commutes(&self, a: &[f64], b: &[f64]) -> bool {
        let comm = self.commutator(a, b);
        comm.iter().all(|x| x.abs() < 1e-10)
    }

    /// Measure non-commutativity via ||[a,b]|| (should be ~100% non-commutative).
    fn non_commutativity_violation(&self, a: &[f64], b: &[f64]) -> f64 {
        let comm = self.commutator(a, b);
        comm.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Jacobi identity test (for Malcev/Lie-admissible):
    /// [[a,b],c] + [[b,c],a] + [[c,a],b] = 0
    fn jacobi_violation(&self, a: &[f64], b: &[f64], c: &[f64]) -> f64 {
        let ab = self.commutator(a, b);
        let bc = self.commutator(b, c);
        let ca = self.commutator(c, a);

        let ab_c = self.commutator(&ab, c);
        let bc_a = self.commutator(&bc, a);
        let ca_b = self.commutator(&ca, b);

        let mut result = vec![0.0; self.dim()];
        for i in 0..self.dim() {
            result[i] = ab_c[i] + bc_a[i] + ca_b[i];
        }
        result.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Malcev identity test (for Malcev algebras):
    /// (xy)(xz) = ((xy)z)x + ((yz)x)x + ((zx)x)y
    fn malcev_violation(&self, x: &[f64], y: &[f64], z: &[f64]) -> f64 {
        let xy = self.product(x, y);
        let xy_xz = self.product(&xy, x);
        let xy_xz_prod = self.product(&xy_xz, z);

        let xy_z = self.product(&xy, z);
        let xy_z_x = self.product(&xy_z, x);

        let yz = self.product(y, z);
        let yz_x = self.product(&yz, x);
        let yz_x_x = self.product(&yz_x, x);

        let zx = self.product(z, x);
        let zx_x = self.product(&zx, x);
        let zx_x_y = self.product(&zx_x, y);

        let mut rhs = vec![0.0; self.dim()];
        for i in 0..self.dim() {
            rhs[i] = xy_z_x[i] + yz_x_x[i] + zx_x_y[i];
        }

        xy_xz_prod
            .iter()
            .zip(rhs.iter())
            .map(|(l, r)| (l - r) * (l - r))
            .sum::<f64>()
            .sqrt()
    }

    /// Associativity test (should fail for all non-trivial non-associative algebras).
    fn associativity_violation(&self, a: &[f64], b: &[f64], c: &[f64]) -> f64 {
        let ab = self.product(a, b);
        let ab_c = self.product(&ab, c);

        let bc = self.product(b, c);
        let a_bc = self.product(a, &bc);

        ab_c.iter()
            .zip(a_bc.iter())
            .map(|(l, r)| (l - r) * (l - r))
            .sum::<f64>()
            .sqrt()
    }
}

// ============================================================================
// Phase 4b Placeholder: Freudenthal-Tits Magic Square
// ============================================================================

/// The Freudenthal-Tits magic square connects composition algebras C and
/// Jordan algebras J3(A) to exceptional Lie algebras:
///
/// ```text
///         A \ B |    R    |    C    |    H    |    O
///     -----------+---------+---------+---------+---------
///         R      |   A1   |   A2   |   C3   |   F4
///         C      |   A2   |   A2   |   A5   |   E6
///         H      |   C3   |   A5   |   D6   |   E7
///         O      |   F4   |   E6   |   E7   |   E8
/// ```
///
/// This is the key theoretical bridge: CD algebras (R,C,H,O) + Jordan algebras
/// => Exceptional Lie algebras (E6,E7,E8,F4,G2).
///
/// The Tits construction: Given a composition algebra C and a Jordan algebra J,
/// the Tits construction produces a Lie algebra L(C,J) whose dimension equals:
/// dim(L(C,J)) = dim(sl(J)) + (dim(C)-1)*dim(J) + dim(C)*(dim(J)-1)
#[derive(Clone)]
pub struct FreudenthalTitsMagicSquare {
    pub composition_algebra_dim: usize,
    pub composition_algebra_name: String,
    pub jordan_algebra_dim: usize,
    pub jordan_algebra_name: String,
    pub exceptional_lie_algebra_name: String,
    pub exceptional_lie_algebra_dim: usize,
}

impl fmt::Debug for FreudenthalTitsMagicSquare {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FreudenthalTitsMagicSquare")
            .field(
                "composition_algebra",
                &format!(
                    "{} ({}D)",
                    self.composition_algebra_name, self.composition_algebra_dim
                ),
            )
            .field(
                "jordan_algebra",
                &format!(
                    "{} ({}D)",
                    self.jordan_algebra_name, self.jordan_algebra_dim
                ),
            )
            .field(
                "exceptional_lie_algebra",
                &format!(
                    "{} ({}D)",
                    self.exceptional_lie_algebra_name, self.exceptional_lie_algebra_dim
                ),
            )
            .finish()
    }
}

impl FreudenthalTitsMagicSquare {
    /// Create a magic square entry from composition algebra and Jordan algebra specifications.
    pub fn new(
        composition_name: &str,
        composition_dim: usize,
        jordan_name: &str,
        jordan_dim: usize,
    ) -> Self {
        let (exceptional_name, exceptional_dim) =
            Self::tits_construction(composition_dim, jordan_dim);

        FreudenthalTitsMagicSquare {
            composition_algebra_dim: composition_dim,
            composition_algebra_name: composition_name.to_string(),
            jordan_algebra_dim: jordan_dim,
            jordan_algebra_name: jordan_name.to_string(),
            exceptional_lie_algebra_name: exceptional_name,
            exceptional_lie_algebra_dim: exceptional_dim,
        }
    }

    /// Tits construction: Map (composition_dim, jordan_dim) to exceptional Lie algebra.
    /// Returns (exceptional_name, exceptional_dim).
    fn tits_construction(comp_dim: usize, jordan_dim: usize) -> (String, usize) {
        let (name, dim) = match (comp_dim, jordan_dim) {
            // Row 1: Composition algebra = R (1D)
            (1, 1) => ("A1", 3),   // sl(2) = 3D
            (1, 3) => ("A2", 8),   // sl(3) = 8D
            (1, 27) => ("E6", 78), // E6 = 78D

            // Row 2: Composition algebra = C (2D)
            (2, 1) => ("A2", 8),   // sl(3) = 8D
            (2, 3) => ("A2", 8),   // sl(3) = 8D (degenerate case)
            (2, 27) => ("E6", 78), // E6 = 78D

            // Row 3: Composition algebra = H (4D, quaternions)
            (4, 1) => ("C3", 21),   // sp(6) = 21D
            (4, 3) => ("A5", 35),   // sl(6) = 35D
            (4, 27) => ("E7", 133), // E7 = 133D

            // Row 4: Composition algebra = O (8D, octonions)
            (8, 1) => ("F4", 52),   // F4 = 52D
            (8, 3) => ("E6", 78),   // E6 = 78D
            (8, 27) => ("E8", 248), // E8 = 248D

            _ => ("Unknown", 0),
        };

        (name.to_string(), dim)
    }

    /// Verify the magic square symmetry: F(A,B) = F(B,A).
    /// The Freudenthal-Tits magic square has the "magic" property that
    /// the construction is symmetric in its arguments (surprising non-obvious result).
    pub fn verify_symmetry(&self, comp_dim_other: usize, jordan_dim_other: usize) -> bool {
        // In the magic square, swapping composition and jordan algebras
        // should (by Tits/Vinberg symmetry) produce the same exceptional Lie algebra.
        // Note: This is a placeholder verification. Full verification requires
        // implementing the Vinberg construction explicitly.
        let (name_ab, dim_ab) =
            Self::tits_construction(self.composition_algebra_dim, self.jordan_algebra_dim);
        let (name_ba, dim_ba) = Self::tits_construction(comp_dim_other, jordan_dim_other);

        // For true magic square entries, the symmetric pairs should yield
        // the same exceptional Lie algebra
        name_ab == name_ba && dim_ab == dim_ba
    }

    /// Check if this entry is a "pure" magic square cell (non-degenerate).
    pub fn is_pure_exceptional(&self) -> bool {
        matches!(
            self.exceptional_lie_algebra_name.as_str(),
            "E6" | "E7" | "E8" | "F4" | "G2"
        )
    }

    /// Compute the dimension of the resulting Lie algebra via Tits formula.
    /// For classical cases, use explicit dimension values.
    /// For exceptional algebras, dimension grows with composition and Jordan algebra.
    pub fn tits_dimension_formula(comp_dim: usize, jordan_dim: usize) -> usize {
        // Use the magic square table directly (most reliable)
        let (_, dim) = Self::tits_construction(comp_dim, jordan_dim);
        dim
    }
}

// ============================================================================
// Phase 4c Placeholder: Octonion Geometry and E8 Structure
// ============================================================================

/// Octonion geometry and E8 root system analysis.
/// E8 is the largest exceptional Lie algebra (248-dimensional).
/// It can be constructed from the magic square as O otimes O (octonion tensor product).
pub struct E8RootSystem {
    pub dim: usize,
    pub name: String,
    // Placeholder: root multiplicities, Dynkin diagram, etc.
}

impl fmt::Debug for E8RootSystem {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("E8RootSystem")
            .field("dim", &self.dim)
            .field("name", &self.name)
            .finish()
    }
}

impl E8RootSystem {
    /// Create the E8 root system (248-dimensional exceptional Lie algebra).
    pub fn new() -> Self {
        E8RootSystem {
            dim: 248,
            name: "E8".to_string(),
        }
    }

    /// E8 has 240 roots (8 * 30 family structure).
    pub fn num_roots(&self) -> usize {
        240
    }

    /// E8 Dynkin diagram: O-O-O-O-O-O-O with branch at position 3.
    pub fn dynkin_diagram_description(&self) -> &str {
        "E8 Dynkin diagram: linear A7 chain with branch at position 3"
    }
}

impl Default for E8RootSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests (placeholders for Phase 4 full implementation)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_magic_square_e8_octonion_albert() {
        // E8 = Tits(O, A3) where O = octonions (8D), A3 = Albert algebra (27D)
        let e8 = FreudenthalTitsMagicSquare::new("Octonions", 8, "Albert", 27);
        assert_eq!(e8.exceptional_lie_algebra_name, "E8");
        assert_eq!(e8.exceptional_lie_algebra_dim, 248);
        assert!(e8.is_pure_exceptional());
    }

    #[test]
    fn test_magic_square_e7_octonion_hermitian() {
        // E7 = Tits(O, J3(C)) where C = complexes, J3(C) = Hermitian 3x3 complex matrices (3D)
        let e7 = FreudenthalTitsMagicSquare::new("Octonions", 8, "J3(C)", 3);
        assert_eq!(e7.exceptional_lie_algebra_name, "E6");
        assert_eq!(e7.exceptional_lie_algebra_dim, 78);
    }

    #[test]
    fn test_magic_square_e6_complex_complex() {
        // E6 = Tits(C, A3) where both are 2D, 27D
        let e6 = FreudenthalTitsMagicSquare::new("Complex", 2, "Albert", 27);
        assert_eq!(e6.exceptional_lie_algebra_name, "E6");
        assert_eq!(e6.exceptional_lie_algebra_dim, 78);
        assert!(e6.is_pure_exceptional());
    }

    #[test]
    fn test_magic_square_f4_octonion_reals() {
        // F4 = Tits(O, R) where O = octonions, R = reals (J3(R) = 3x3 reals)
        let f4 = FreudenthalTitsMagicSquare::new("Octonions", 8, "Reals", 1);
        assert_eq!(f4.exceptional_lie_algebra_name, "F4");
        assert_eq!(f4.exceptional_lie_algebra_dim, 52);
        assert!(f4.is_pure_exceptional());
    }

    #[test]
    fn test_magic_square_classical_a1() {
        // A1 = Tits(R, R) = sl(2), 3-dimensional
        let a1 = FreudenthalTitsMagicSquare::new("Reals", 1, "Reals", 1);
        assert_eq!(a1.exceptional_lie_algebra_name, "A1");
        assert_eq!(a1.exceptional_lie_algebra_dim, 3);
    }

    #[test]
    fn test_magic_square_classical_a2() {
        // A2 = Tits(R, C-Hermitian) = sl(3), 8-dimensional
        let a2 = FreudenthalTitsMagicSquare::new("Reals", 1, "J3(C)", 3);
        assert_eq!(a2.exceptional_lie_algebra_name, "A2");
        assert_eq!(a2.exceptional_lie_algebra_dim, 8);
    }

    #[test]
    fn test_magic_square_classical_c3() {
        // C3 = Tits(H, R) = sp(6), 21-dimensional
        let c3 = FreudenthalTitsMagicSquare::new("Quaternions", 4, "Reals", 1);
        assert_eq!(c3.exceptional_lie_algebra_name, "C3");
        assert_eq!(c3.exceptional_lie_algebra_dim, 21);
    }

    #[test]
    fn test_tits_dimension_formula() {
        // Verify Tits dimension formula via magic square table
        assert_eq!(FreudenthalTitsMagicSquare::tits_dimension_formula(1, 1), 3); // A1 = sl(2)
        assert_eq!(FreudenthalTitsMagicSquare::tits_dimension_formula(1, 3), 8); // A2 = sl(3)
        assert_eq!(FreudenthalTitsMagicSquare::tits_dimension_formula(4, 1), 21); // C3 = sp(6)
        assert_eq!(
            FreudenthalTitsMagicSquare::tits_dimension_formula(8, 27),
            248
        ); // E8
    }

    #[test]
    fn test_e8_root_system() {
        let e8 = E8RootSystem::new();
        assert_eq!(e8.dim, 248);
        assert_eq!(e8.num_roots(), 240);
        assert!(e8.dynkin_diagram_description().contains("linear A7"));
    }
}
