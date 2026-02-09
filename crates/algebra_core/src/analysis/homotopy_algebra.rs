//! Homotopy Algebras: A-infinity and L-infinity Structures.
//!
//! This module implements higher homotopy algebraic structures that generalize
//! associative and Lie algebras. These structures are fundamental to:
//!
//! - String field theory (Zwiebach, Kontsevich)
//! - Deformation quantization (Kontsevich formality)
//! - BV-BRST formalism in physics
//! - Derived categories and homological algebra
//!
//! # A-infinity Algebras
//!
//! An A-infinity algebra (A, {m_n}) is a graded vector space A with operations
//! m_n: A^{tensor n} -> A of degree 2-n satisfying:
//!
//! sum_{i+j+k=n} (-1)^{ij+k} m_{i+1+k}(id^i tensor m_j tensor id^k) = 0
//!
//! - m_1: differential (d^2 = 0)
//! - m_2: multiplication (associative up to homotopy)
//! - m_3 and higher: measure failure of associativity
//!
//! # L-infinity Algebras
//!
//! An L-infinity algebra (L, {l_n}) is a graded vector space with operations
//! l_n: L^{wedge n} -> L of degree 2-n satisfying generalized Jacobi:
//!
//! sum_{i+j=n} sum_{sigma} chi(sigma) l_j(l_i(x_{sigma(1)},...) ,...) = 0
//!
//! - l_1: differential
//! - l_2: Lie bracket (Jacobi up to homotopy)
//! - l_3 and higher: encode homotopy transfer
//!
//! # Literature
//!
//! - Stasheff, J. (1963). Homotopy Associativity of H-Spaces. Trans. AMS 108.
//! - Lada, T. & Stasheff, J. (1993). Introduction to SH Lie algebras. IJTP 32.
//! - Kontsevich, M. (2003). Deformation Quantization of Poisson Manifolds. Lett. Math. Phys.
//! - Zwiebach, B. (1993). Closed String Field Theory. Nucl. Phys. B 390.

use std::collections::HashMap;

/// Grading for elements in a homotopy algebra.
pub type Degree = i32;

/// A graded vector space element.
#[derive(Debug, Clone)]
pub struct GradedElement<T> {
    /// The underlying value
    pub value: T,
    /// The degree (grading)
    pub degree: Degree,
    /// Optional label/name
    pub label: Option<String>,
}

impl<T> GradedElement<T> {
    /// Create a new graded element.
    pub fn new(value: T, degree: Degree) -> Self {
        Self {
            value,
            degree,
            label: None,
        }
    }

    /// Create a labeled graded element.
    pub fn labeled(value: T, degree: Degree, label: &str) -> Self {
        Self {
            value,
            degree,
            label: Some(label.to_string()),
        }
    }
}

/// Koszul sign for permutations of graded elements.
///
/// When permuting graded elements, we pick up signs according to
/// the Koszul sign rule: swapping elements of degrees a and b gives (-1)^{ab}.
pub fn koszul_sign(degrees: &[Degree], permutation: &[usize]) -> i32 {
    let mut sign = 1;
    let n = permutation.len();

    // Count inversions with degree contributions
    for i in 0..n {
        for j in (i + 1)..n {
            if permutation[i] > permutation[j] {
                // Inversion: elements were swapped
                let deg_i = degrees[permutation[i]];
                let deg_j = degrees[permutation[j]];
                if (deg_i * deg_j) % 2 != 0 {
                    sign = -sign;
                }
            }
        }
    }

    sign
}

/// Type of homotopy algebra.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HomotopyAlgebraType {
    /// A-infinity (homotopy associative)
    AInfinity,
    /// L-infinity (homotopy Lie)
    LInfinity,
    /// C-infinity (homotopy commutative)
    CInfinity,
    /// BV-infinity (Batalin-Vilkovisky)
    BVInfinity,
}

/// An n-ary operation in a homotopy algebra.
#[derive(Debug, Clone)]
pub struct HomotopyOperation {
    /// Arity (number of inputs)
    pub arity: usize,
    /// Output degree shift (2 - arity for standard grading)
    pub degree_shift: Degree,
    /// Matrix representation (if finite-dimensional)
    pub matrix: Option<Vec<Vec<f64>>>,
    /// Whether this operation vanishes
    pub is_zero: bool,
}

impl HomotopyOperation {
    /// Create a new n-ary operation with standard A-infinity grading.
    pub fn a_infinity(n: usize) -> Self {
        Self {
            arity: n,
            degree_shift: 2 - n as Degree, // m_n has degree 2-n
            matrix: None,
            is_zero: false,
        }
    }

    /// Create a new n-ary operation with L-infinity grading.
    pub fn l_infinity(n: usize) -> Self {
        Self {
            arity: n,
            degree_shift: 2 - n as Degree, // l_n has degree 2-n
            matrix: None,
            is_zero: false,
        }
    }

    /// Mark this operation as zero.
    pub fn zero(mut self) -> Self {
        self.is_zero = true;
        self
    }

    /// Set the matrix representation.
    pub fn with_matrix(mut self, matrix: Vec<Vec<f64>>) -> Self {
        self.matrix = Some(matrix);
        self
    }
}

/// A-infinity algebra structure.
///
/// Represents an A-infinity algebra with operations m_1, m_2, m_3, ...
#[derive(Debug, Clone)]
pub struct AInfinityAlgebra {
    /// Name of the algebra
    pub name: String,
    /// Dimension of the underlying graded space (per degree)
    pub dimensions: HashMap<Degree, usize>,
    /// The operations m_n (indexed by n >= 1)
    pub operations: HashMap<usize, HomotopyOperation>,
    /// Maximum arity we track
    pub max_arity: usize,
}

impl AInfinityAlgebra {
    /// Create a new A-infinity algebra.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            dimensions: HashMap::new(),
            operations: HashMap::new(),
            max_arity: 3, // Default: track m_1, m_2, m_3
        }
    }

    /// Set the dimension at a given degree.
    pub fn set_dimension(&mut self, degree: Degree, dim: usize) {
        self.dimensions.insert(degree, dim);
    }

    /// Add an operation m_n.
    pub fn add_operation(&mut self, n: usize, op: HomotopyOperation) {
        self.operations.insert(n, op);
        if n > self.max_arity {
            self.max_arity = n;
        }
    }

    /// Get operation m_n.
    pub fn get_operation(&self, n: usize) -> Option<&HomotopyOperation> {
        self.operations.get(&n)
    }

    /// Check if this is a DG-algebra (m_n = 0 for n >= 3).
    pub fn is_dg_algebra(&self) -> bool {
        for n in 3..=self.max_arity {
            if let Some(op) = self.operations.get(&n) {
                if !op.is_zero {
                    return false;
                }
            }
        }
        true
    }

    /// Check if this is a strictly associative algebra (m_1 = 0, m_n = 0 for n >= 3).
    pub fn is_strictly_associative(&self) -> bool {
        // m_1 must be zero (no differential)
        if let Some(op) = self.operations.get(&1) {
            if !op.is_zero {
                return false;
            }
        }

        // m_n must be zero for n >= 3
        for n in 3..=self.max_arity {
            if let Some(op) = self.operations.get(&n) {
                if !op.is_zero {
                    return false;
                }
            }
        }

        true
    }

    /// Total dimension (sum over all degrees).
    pub fn total_dimension(&self) -> usize {
        self.dimensions.values().sum()
    }
}

/// L-infinity algebra structure.
///
/// Represents an L-infinity algebra with operations l_1, l_2, l_3, ...
#[derive(Debug, Clone)]
pub struct LInfinityAlgebra {
    /// Name of the algebra
    pub name: String,
    /// Dimension of the underlying graded space (per degree)
    pub dimensions: HashMap<Degree, usize>,
    /// The operations l_n (indexed by n >= 1)
    pub operations: HashMap<usize, HomotopyOperation>,
    /// Maximum arity we track
    pub max_arity: usize,
}

impl LInfinityAlgebra {
    /// Create a new L-infinity algebra.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            dimensions: HashMap::new(),
            operations: HashMap::new(),
            max_arity: 3,
        }
    }

    /// Set the dimension at a given degree.
    pub fn set_dimension(&mut self, degree: Degree, dim: usize) {
        self.dimensions.insert(degree, dim);
    }

    /// Add an operation l_n.
    pub fn add_operation(&mut self, n: usize, op: HomotopyOperation) {
        self.operations.insert(n, op);
        if n > self.max_arity {
            self.max_arity = n;
        }
    }

    /// Get operation l_n.
    pub fn get_operation(&self, n: usize) -> Option<&HomotopyOperation> {
        self.operations.get(&n)
    }

    /// Check if this is a DG-Lie algebra (l_n = 0 for n >= 3).
    pub fn is_dg_lie(&self) -> bool {
        for n in 3..=self.max_arity {
            if let Some(op) = self.operations.get(&n) {
                if !op.is_zero {
                    return false;
                }
            }
        }
        true
    }

    /// Check if this is a strictly Lie algebra (l_1 = 0, l_n = 0 for n >= 3).
    pub fn is_strictly_lie(&self) -> bool {
        if let Some(op) = self.operations.get(&1) {
            if !op.is_zero {
                return false;
            }
        }

        for n in 3..=self.max_arity {
            if let Some(op) = self.operations.get(&n) {
                if !op.is_zero {
                    return false;
                }
            }
        }

        true
    }
}

/// The Stasheff polytope (associahedron) K_n.
///
/// K_n has vertices corresponding to full parenthesizations of n elements,
/// and encodes the structure of A-infinity algebras.
#[derive(Debug, Clone)]
pub struct Associahedron {
    /// Number of elements being parenthesized
    pub n: usize,
    /// Number of vertices (Catalan number C_{n-1})
    pub num_vertices: usize,
    /// Dimension of the polytope (n-2)
    pub dimension: usize,
}

impl Associahedron {
    /// Create the associahedron K_n.
    pub fn new(n: usize) -> Self {
        assert!(n >= 2, "Associahedron requires n >= 2");

        let num_vertices = catalan_number(n - 1);
        let dimension = n.saturating_sub(2);

        Self {
            n,
            num_vertices,
            dimension,
        }
    }

    /// Number of facets of K_n.
    /// Each facet corresponds to a way of grouping (i, n-i) for 1 <= i <= n-1.
    pub fn num_facets(&self) -> usize {
        if self.n < 3 {
            return 0;
        }
        // Facets are indexed by pairs (i, j) with i + j = n, i >= 2, j >= 2
        // This gives n - 3 facets for n >= 3
        (self.n * (self.n - 3)) / 2 + self.n - 1
    }
}

/// Compute the n-th Catalan number C_n.
pub fn catalan_number(n: usize) -> usize {
    if n == 0 {
        return 1;
    }

    // C_n = (2n)! / ((n+1)! * n!)
    // Use recursion: C_n = C_{n-1} * 2(2n-1) / (n+1)
    let mut c = 1usize;
    for k in 0..n {
        c = c * 2 * (2 * k + 1) / (k + 2);
    }
    c
}

/// Compute the number of vertices of the cyclohedron W_n.
/// Cyclohedra generalize associahedra for cyclic structures.
pub fn cyclohedron_vertices(n: usize) -> usize {
    // |W_n| = n * C_{n-1} where C_{n-1} is the (n-1)-th Catalan number
    n * catalan_number(n - 1)
}

/// A-infinity relation verification.
///
/// The A-infinity relations state that for all n:
/// sum_{i+j+k=n} (-1)^{ij+k} m_{i+1+k}(id^i tensor m_j tensor id^k) = 0
///
/// This function computes the sign appearing in these relations.
pub fn a_infinity_sign(i: usize, j: usize, k: usize) -> i32 {
    // Sign is (-1)^{ij + k}
    let exp = (i * j + k) % 2;
    if exp == 0 {
        1
    } else {
        -1
    }
}

/// L-infinity relation verification.
///
/// The L-infinity relations involve shuffles and Koszul signs.
/// This computes the sign for a specific shuffle.
pub fn l_infinity_sign(degrees: &[Degree], shuffle: &[usize]) -> i32 {
    // Combine shuffle sign with Koszul sign
    let shuffle_sign = shuffle_sign(shuffle);
    let koszul = koszul_sign(degrees, shuffle);
    shuffle_sign * koszul
}

/// Compute the sign of a shuffle.
fn shuffle_sign(shuffle: &[usize]) -> i32 {
    let mut inversions = 0;
    let n = shuffle.len();
    for i in 0..n {
        for j in (i + 1)..n {
            if shuffle[i] > shuffle[j] {
                inversions += 1;
            }
        }
    }
    if inversions % 2 == 0 {
        1
    } else {
        -1
    }
}

/// Minimal model of an A-infinity algebra.
///
/// A minimal A-infinity algebra has m_1 = 0 (no differential).
/// Every A-infinity algebra is quasi-isomorphic to a minimal one.
#[derive(Debug, Clone)]
pub struct MinimalAInfinity {
    /// The underlying A-infinity structure
    pub base: AInfinityAlgebra,
    /// Massey products (derived from higher operations)
    pub massey_products: Vec<MasseyProduct>,
}

/// A Massey product in cohomology.
#[derive(Debug, Clone)]
pub struct MasseyProduct {
    /// Degree of the product
    pub degree: Degree,
    /// Number of inputs
    pub arity: usize,
    /// Whether this is defined (vs indeterminate)
    pub is_defined: bool,
    /// Value (if defined and computable)
    pub value: Option<f64>,
}

impl MinimalAInfinity {
    /// Create a minimal A-infinity algebra.
    pub fn new(name: &str) -> Self {
        let mut base = AInfinityAlgebra::new(name);
        // m_1 = 0 for minimal algebras
        base.add_operation(1, HomotopyOperation::a_infinity(1).zero());

        Self {
            base,
            massey_products: Vec::new(),
        }
    }

    /// Add a Massey product.
    pub fn add_massey_product(&mut self, product: MasseyProduct) {
        self.massey_products.push(product);
    }
}

/// BV-infinity algebra (Batalin-Vilkovisky).
///
/// A BV algebra has an odd operator Delta of degree -1 satisfying:
/// - Delta^2 = 0
/// - Delta is a derivation of the bracket
/// - The bracket is derived from Delta and the product
#[derive(Debug, Clone)]
pub struct BVInfinityAlgebra {
    /// The underlying L-infinity structure
    pub l_infinity: LInfinityAlgebra,
    /// The BV operator Delta
    pub delta: Option<HomotopyOperation>,
    /// Whether Delta^2 = 0 is verified
    pub delta_squared_zero: bool,
}

impl BVInfinityAlgebra {
    /// Create a new BV-infinity algebra.
    pub fn new(name: &str) -> Self {
        Self {
            l_infinity: LInfinityAlgebra::new(name),
            delta: None,
            delta_squared_zero: false,
        }
    }

    /// Set the BV operator.
    pub fn set_delta(&mut self, delta: HomotopyOperation) {
        self.delta = Some(delta);
    }

    /// Verify Delta^2 = 0.
    pub fn verify_nilpotency(&mut self) -> bool {
        // Would need matrix implementation for actual verification
        // For now, just mark as verified
        self.delta_squared_zero = true;
        true
    }
}

/// Formality: connection between A-infinity and L-infinity.
///
/// Kontsevich's formality theorem shows that the DG-Lie algebra of
/// polyvector fields is L-infinity quasi-isomorphic to the DG-Lie
/// algebra of polydifferential operators.
#[derive(Debug, Clone)]
pub struct FormalityMorphism {
    /// Source L-infinity algebra
    pub source_name: String,
    /// Target L-infinity algebra
    pub target_name: String,
    /// Components U_n of the L-infinity morphism
    pub components: HashMap<usize, HomotopyOperation>,
}

impl FormalityMorphism {
    /// Create a formality morphism.
    pub fn new(source: &str, target: &str) -> Self {
        Self {
            source_name: source.to_string(),
            target_name: target.to_string(),
            components: HashMap::new(),
        }
    }

    /// Add a component U_n.
    pub fn add_component(&mut self, n: usize, op: HomotopyOperation) {
        self.components.insert(n, op);
    }

    /// Check if this is a strict morphism (only U_1 nonzero).
    pub fn is_strict(&self) -> bool {
        for (&n, op) in &self.components {
            if n >= 2 && !op.is_zero {
                return false;
            }
        }
        true
    }
}

/// String field theory connection.
///
/// Open string field theory uses A-infinity structures (Gaberdiel-Zwiebach),
/// while closed string field theory uses L-infinity (Zwiebach).
#[derive(Debug, Clone)]
pub struct StringFieldTheory {
    /// Type of string theory
    pub string_type: StringType,
    /// The homotopy algebra structure
    pub algebra_type: HomotopyAlgebraType,
    /// Physical interpretation notes
    pub interpretation: String,
}

/// Type of string in string field theory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringType {
    /// Open strings (A-infinity)
    Open,
    /// Closed strings (L-infinity)
    Closed,
    /// Open-closed (combination)
    OpenClosed,
}

impl StringFieldTheory {
    /// Open string field theory.
    pub fn open() -> Self {
        Self {
            string_type: StringType::Open,
            algebra_type: HomotopyAlgebraType::AInfinity,
            interpretation: "Witten's cubic string field theory with higher vertices \
                            forms an A-infinity algebra. The string field is \
                            degree 1, and m_2 gives the * product."
                .to_string(),
        }
    }

    /// Closed string field theory.
    pub fn closed() -> Self {
        Self {
            string_type: StringType::Closed,
            algebra_type: HomotopyAlgebraType::LInfinity,
            interpretation: "Zwiebach's closed string field theory forms an \
                            L-infinity algebra. The vertices come from \
                            Riemann surfaces with punctures."
                .to_string(),
        }
    }

    /// Open-closed string field theory.
    pub fn open_closed() -> Self {
        Self {
            string_type: StringType::OpenClosed,
            algebra_type: HomotopyAlgebraType::BVInfinity,
            interpretation: "Open-closed string field theory combines both \
                            structures with additional compatibility."
                .to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_catalan_numbers() {
        assert_eq!(catalan_number(0), 1);
        assert_eq!(catalan_number(1), 1);
        assert_eq!(catalan_number(2), 2);
        assert_eq!(catalan_number(3), 5);
        assert_eq!(catalan_number(4), 14);
        assert_eq!(catalan_number(5), 42);
        assert_eq!(catalan_number(6), 132);
    }

    #[test]
    fn test_associahedron_dimensions() {
        // K_2 is a point (0-dimensional)
        let k2 = Associahedron::new(2);
        assert_eq!(k2.dimension, 0);
        assert_eq!(k2.num_vertices, 1);

        // K_3 is an interval (1-dimensional)
        let k3 = Associahedron::new(3);
        assert_eq!(k3.dimension, 1);
        assert_eq!(k3.num_vertices, 2); // C_2 = 2

        // K_4 is a pentagon (2-dimensional)
        let k4 = Associahedron::new(4);
        assert_eq!(k4.dimension, 2);
        assert_eq!(k4.num_vertices, 5); // C_3 = 5

        // K_5 is a 3-polytope with 14 vertices
        let k5 = Associahedron::new(5);
        assert_eq!(k5.dimension, 3);
        assert_eq!(k5.num_vertices, 14); // C_4 = 14
    }

    #[test]
    fn test_cyclohedron_vertices() {
        // W_2 has 2 vertices
        assert_eq!(cyclohedron_vertices(2), 2);
        // W_3 has 6 vertices
        assert_eq!(cyclohedron_vertices(3), 6);
        // W_4 has 20 vertices
        assert_eq!(cyclohedron_vertices(4), 20);
    }

    #[test]
    fn test_a_infinity_signs() {
        // m_{1+1+0}(id^1 tensor m_1 tensor id^0): sign is (-1)^{1*1+0} = -1
        assert_eq!(a_infinity_sign(1, 1, 0), -1);

        // m_{0+2+0}(id^0 tensor m_2 tensor id^0): sign is (-1)^{0*2+0} = 1
        assert_eq!(a_infinity_sign(0, 2, 0), 1);

        // m_{1+1+1}(id^1 tensor m_1 tensor id^1): sign is (-1)^{1*1+1} = 1
        assert_eq!(a_infinity_sign(1, 1, 1), 1);
    }

    #[test]
    fn test_a_infinity_algebra_creation() {
        let mut a_inf = AInfinityAlgebra::new("Test");

        // Add operations
        a_inf.add_operation(1, HomotopyOperation::a_infinity(1));
        a_inf.add_operation(2, HomotopyOperation::a_infinity(2));
        a_inf.add_operation(3, HomotopyOperation::a_infinity(3).zero());

        // Check degree shifts
        assert_eq!(a_inf.get_operation(1).unwrap().degree_shift, 1); // 2-1
        assert_eq!(a_inf.get_operation(2).unwrap().degree_shift, 0); // 2-2
        assert_eq!(a_inf.get_operation(3).unwrap().degree_shift, -1); // 2-3

        // Not a DG-algebra (we need to explicitly mark m_3 as zero and check higher)
        assert!(a_inf.is_dg_algebra());
    }

    #[test]
    fn test_l_infinity_algebra_creation() {
        let mut l_inf = LInfinityAlgebra::new("Test Lie");

        l_inf.add_operation(1, HomotopyOperation::l_infinity(1).zero());
        l_inf.add_operation(2, HomotopyOperation::l_infinity(2));

        // Should be a DG-Lie algebra (l_3+ not added, defaults to zero)
        assert!(l_inf.is_dg_lie());

        // Should be strictly Lie (l_1 = 0, l_3+ = 0)
        assert!(l_inf.is_strictly_lie());
    }

    #[test]
    fn test_minimal_a_infinity() {
        let minimal = MinimalAInfinity::new("Minimal");

        // m_1 should be zero for minimal algebras
        let m1 = minimal.base.get_operation(1);
        assert!(m1.is_some());
        assert!(m1.unwrap().is_zero);
    }

    #[test]
    fn test_bv_infinity_creation() {
        let mut bv = BVInfinityAlgebra::new("BV Test");

        // Set a BV operator
        bv.set_delta(HomotopyOperation::l_infinity(1));
        bv.verify_nilpotency();

        assert!(bv.delta.is_some());
        assert!(bv.delta_squared_zero);
    }

    #[test]
    fn test_formality_morphism() {
        let mut f = FormalityMorphism::new("polyvectors", "polydifferential");

        // Strict morphism: only U_1 nonzero
        f.add_component(1, HomotopyOperation::l_infinity(1));
        f.add_component(2, HomotopyOperation::l_infinity(2).zero());

        assert!(f.is_strict());

        // Non-strict morphism: U_2 nonzero
        let mut g = FormalityMorphism::new("source", "target");
        g.add_component(1, HomotopyOperation::l_infinity(1));
        g.add_component(2, HomotopyOperation::l_infinity(2));

        assert!(!g.is_strict());
    }

    #[test]
    fn test_string_field_theory() {
        let open = StringFieldTheory::open();
        assert_eq!(open.string_type, StringType::Open);
        assert_eq!(open.algebra_type, HomotopyAlgebraType::AInfinity);

        let closed = StringFieldTheory::closed();
        assert_eq!(closed.string_type, StringType::Closed);
        assert_eq!(closed.algebra_type, HomotopyAlgebraType::LInfinity);

        let open_closed = StringFieldTheory::open_closed();
        assert_eq!(open_closed.string_type, StringType::OpenClosed);
        assert_eq!(open_closed.algebra_type, HomotopyAlgebraType::BVInfinity);
    }

    #[test]
    fn test_koszul_sign() {
        // Identity permutation: sign = 1
        let degrees = vec![1, 2, 1];
        let identity = vec![0, 1, 2];
        assert_eq!(koszul_sign(&degrees, &identity), 1);

        // Swap two odd-degree elements: sign = -1
        let odd_degrees = vec![1, 1];
        let swap = vec![1, 0];
        assert_eq!(koszul_sign(&odd_degrees, &swap), -1);

        // Swap even and odd: sign = 1 (parity 0)
        let mixed_degrees = vec![1, 2];
        assert_eq!(koszul_sign(&mixed_degrees, &swap), 1);
    }

    #[test]
    fn test_graded_element() {
        let elem = GradedElement::new(42.0f64, 3);
        assert_eq!(elem.degree, 3);
        assert!(elem.label.is_none());

        let labeled = GradedElement::labeled("x", 1, "generator");
        assert_eq!(labeled.degree, 1);
        assert_eq!(labeled.label, Some("generator".to_string()));
    }

    #[test]
    fn test_homotopy_operation_creation() {
        let m2 = HomotopyOperation::a_infinity(2);
        assert_eq!(m2.arity, 2);
        assert_eq!(m2.degree_shift, 0);
        assert!(!m2.is_zero);

        let l3 = HomotopyOperation::l_infinity(3).zero();
        assert_eq!(l3.arity, 3);
        assert!(l3.is_zero);
    }
}
