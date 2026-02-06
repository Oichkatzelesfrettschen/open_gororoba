//! Kac-Moody Algebras: Infinite-Dimensional Generalizations of Lie Algebras.
//!
//! This module implements generalized Cartan matrices (GCMs) and their classification
//! into finite, affine, and indefinite (hyperbolic) types. Kac-Moody algebras are
//! fundamental to string theory, conformal field theory, and M-theory.
//!
//! # Classification
//!
//! A generalized Cartan matrix A is:
//! - **Finite type**: A is positive definite (classical Lie algebras A_n, D_n, E_6, E_7, E_8)
//! - **Affine type**: A is positive semi-definite with corank 1 (loop algebras, E_9 = E_8^{(1)})
//! - **Indefinite type**: A has signature (n-1, 1) or worse (E_10, E_11, etc.)
//!   - **Hyperbolic**: rank n indefinite with every proper connected subdiagram finite or affine
//!   - **Lorentzian**: signature (n-1, 1)
//!
//! # E-series Extensions
//!
//! The E-series extends beyond E8:
//! - E_9 = E_8^{(1)}: Affine extension, important in string theory
//! - E_10: Hyperbolic, conjectured to encode M-theory symmetries
//! - E_11: Even larger, proposed as hidden symmetry of supergravity
//!
//! # Literature
//!
//! - Kac, V. G. (1990). Infinite-Dimensional Lie Algebras (3rd ed.). Cambridge.
//! - Damour, T., Henneaux, M., & Nicolai, H. (2002). E10 and a 'small tension expansion' of M-theory. PRL 89.
//! - West, P. (2001). E11 and M Theory. Class. Quantum Grav. 18, 4443.
//! - Nicolai, H. & Samtleben, H. (2005). On K(E9). Q.J.Math. 56, 403-423.

use std::collections::HashSet;

/// A generalized Cartan matrix entry type.
pub type CartanEntry = i32;

/// Generalized Cartan Matrix (GCM) for Kac-Moody algebras.
///
/// A matrix A = (a_ij) is a GCM if:
/// 1. a_ii = 2 for all i
/// 2. a_ij <= 0 for i != j
/// 3. a_ij = 0 implies a_ji = 0
#[derive(Debug, Clone)]
pub struct GeneralizedCartanMatrix {
    /// The matrix entries (row-major)
    entries: Vec<Vec<CartanEntry>>,
    /// Rank of the algebra (dimension of Cartan subalgebra)
    rank: usize,
    /// Cached classification (reserved for future memoization)
    _classification: Option<KacMoodyType>,
}

/// Classification of Kac-Moody algebras.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KacMoodyType {
    /// Positive definite: finite-dimensional Lie algebra
    Finite,
    /// Positive semi-definite, corank 1: affine (loop) algebra
    Affine,
    /// Signature (n-1, 1): Lorentzian
    Lorentzian,
    /// Indefinite with all proper subdiagrams finite/affine
    Hyperbolic,
    /// General indefinite
    Indefinite,
}

/// Named Lie algebra types for convenience.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LieAlgebraType {
    A(usize),  // SL(n+1)
    B(usize),  // SO(2n+1)
    C(usize),  // Sp(2n)
    D(usize),  // SO(2n)
    E6,
    E7,
    E8,
    E9,  // Affine E8
    E10, // Hyperbolic
    E11, // Extended hyperbolic
    F4,
    G2,
}

/// A node in a Dynkin diagram.
#[derive(Debug, Clone)]
pub struct DynkinNode {
    /// Node index (0-based)
    pub index: usize,
    /// Label/name for the node
    pub label: String,
    /// Whether this is an affine extension node
    pub is_affine_extension: bool,
}

/// An edge in a Dynkin diagram.
#[derive(Debug, Clone)]
pub struct DynkinEdge {
    /// Source node index
    pub from: usize,
    /// Target node index
    pub to: usize,
    /// Bond multiplicity (1 = single, 2 = double, 3 = triple)
    pub multiplicity: u8,
    /// Arrow direction for non-simply-laced (true = from -> to)
    pub arrow_to_shorter: Option<bool>,
}

/// Dynkin diagram representation.
#[derive(Debug, Clone)]
pub struct DynkinDiagram {
    /// Nodes
    pub nodes: Vec<DynkinNode>,
    /// Edges
    pub edges: Vec<DynkinEdge>,
    /// Algebra type if known
    pub algebra_type: Option<LieAlgebraType>,
}

impl GeneralizedCartanMatrix {
    /// Create a new GCM from a 2D array.
    pub fn new(entries: Vec<Vec<CartanEntry>>) -> Result<Self, &'static str> {
        let rank = entries.len();
        if rank == 0 {
            return Err("GCM must have at least one row");
        }

        // Verify square matrix
        for row in &entries {
            if row.len() != rank {
                return Err("GCM must be square");
            }
        }

        // Verify GCM axioms
        #[allow(clippy::needless_range_loop)]
        for i in 0..rank {
            // Axiom 1: diagonal = 2
            if entries[i][i] != 2 {
                return Err("GCM diagonal entries must be 2");
            }

            #[allow(clippy::needless_range_loop)]
            for j in 0..rank {
                if i != j {
                    // Axiom 2: off-diagonal <= 0
                    if entries[i][j] > 0 {
                        return Err("GCM off-diagonal entries must be <= 0");
                    }

                    // Axiom 3: symmetry of zeros
                    if (entries[i][j] == 0) != (entries[j][i] == 0) {
                        return Err("GCM zero entries must be symmetric");
                    }
                }
            }
        }

        Ok(Self {
            entries,
            rank,
            _classification: None,
        })
    }

    /// Create from a fixed-size array (convenience for small matrices).
    pub fn from_array<const N: usize>(arr: [[CartanEntry; N]; N]) -> Result<Self, &'static str> {
        let entries: Vec<Vec<CartanEntry>> = arr.iter()
            .map(|row| row.to_vec())
            .collect();
        Self::new(entries)
    }

    /// Get the rank (number of simple roots).
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get matrix entry a_ij.
    pub fn get(&self, i: usize, j: usize) -> CartanEntry {
        self.entries[i][j]
    }

    /// Check if the matrix is symmetric (simply-laced algebra).
    pub fn is_symmetric(&self) -> bool {
        for i in 0..self.rank {
            for j in (i + 1)..self.rank {
                if self.entries[i][j] != self.entries[j][i] {
                    return false;
                }
            }
        }
        true
    }

    /// Check if simply-laced (symmetric with entries in {0, -1}).
    pub fn is_simply_laced(&self) -> bool {
        if !self.is_symmetric() {
            return false;
        }
        for i in 0..self.rank {
            for j in 0..self.rank {
                if i != j && self.entries[i][j] != 0 && self.entries[i][j] != -1 {
                    return false;
                }
            }
        }
        true
    }

    /// Compute determinant using Gaussian elimination.
    pub fn determinant(&self) -> i64 {
        // Convert to f64 for computation
        let n = self.rank;
        let mut matrix: Vec<Vec<f64>> = self.entries.iter()
            .map(|row| row.iter().map(|&x| x as f64).collect())
            .collect();

        let mut det = 1.0;

        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if matrix[k][i].abs() > matrix[max_row][i].abs() {
                    max_row = k;
                }
            }

            if matrix[max_row][i].abs() < 1e-10 {
                return 0;
            }

            if max_row != i {
                matrix.swap(i, max_row);
                det = -det;
            }

            det *= matrix[i][i];

            for k in (i + 1)..n {
                let factor = matrix[k][i] / matrix[i][i];
                #[allow(clippy::needless_range_loop)]
                for j in i..n {
                    matrix[k][j] -= factor * matrix[i][j];
                }
            }
        }

        det.round() as i64
    }

    /// Compute eigenvalues (approximate, for classification).
    pub fn eigenvalue_signs(&self) -> (usize, usize, usize) {
        // Use Sylvester's law of inertia via principal minors
        // Returns (positive, zero, negative) counts

        let det = self.determinant();

        if self.rank == 1 {
            return if det > 0 { (1, 0, 0) } else if det == 0 { (0, 1, 0) } else { (0, 0, 1) };
        }

        // Compute all principal minors to determine signature
        let mut positive = 0;
        let mut zero = 0;
        let mut negative = 0;

        // Check leading principal minors
        let mut prev_det: i64 = 1;
        for k in 1..=self.rank {
            let minor = self.leading_principal_minor(k);
            if minor > 0 && prev_det > 0 {
                positive += 1;
            } else if minor == 0 {
                zero += 1;
            } else if (minor < 0 && prev_det > 0) || (minor > 0 && prev_det < 0) {
                negative += 1;
            } else if minor < 0 && prev_det < 0 {
                positive += 1;
            }
            prev_det = if minor != 0 { minor } else { prev_det };
        }

        (positive, zero, negative)
    }

    /// Compute the k-th leading principal minor (1-indexed).
    fn leading_principal_minor(&self, k: usize) -> i64 {
        if k == 0 {
            return 1;
        }
        if k > self.rank {
            return 0;
        }

        // Extract k x k submatrix
        let submatrix: Vec<Vec<CartanEntry>> = self.entries[..k].iter()
            .map(|row| row[..k].to_vec())
            .collect();

        let sub_gcm = GeneralizedCartanMatrix {
            entries: submatrix,
            rank: k,
            _classification: None,
        };

        sub_gcm.determinant()
    }

    /// Classify this GCM.
    pub fn classify(&self) -> KacMoodyType {
        let (_positive, zero, negative) = self.eigenvalue_signs();

        // Finite type: all eigenvalues positive (positive definite)
        if negative == 0 && zero == 0 {
            return KacMoodyType::Finite;
        }

        // Affine type: exactly one zero eigenvalue, rest positive
        if negative == 0 && zero == 1 {
            return KacMoodyType::Affine;
        }

        // Lorentzian: signature (n-1, 1)
        if negative == 1 && zero == 0 {
            // Check if hyperbolic (all proper subdiagrams finite or affine)
            if self.is_hyperbolic() {
                return KacMoodyType::Hyperbolic;
            }
            return KacMoodyType::Lorentzian;
        }

        KacMoodyType::Indefinite
    }

    /// Check if this is a hyperbolic Kac-Moody algebra.
    /// Hyperbolic = Lorentzian + all proper connected subdiagrams are finite or affine.
    fn is_hyperbolic(&self) -> bool {
        // For each proper subset, check if the induced submatrix is finite or affine
        // This is expensive for large matrices, so we only check for small cases
        if self.rank <= 2 {
            return true; // All rank <= 2 Lorentzian are hyperbolic
        }

        // Check removing each node
        for removed in 0..self.rank {
            let sub_entries: Vec<Vec<CartanEntry>> = (0..self.rank)
                .filter(|&i| i != removed)
                .map(|i| {
                    (0..self.rank)
                        .filter(|&j| j != removed)
                        .map(|j| self.entries[i][j])
                        .collect()
                })
                .collect();

            if let Ok(sub_gcm) = GeneralizedCartanMatrix::new(sub_entries) {
                let sub_type = sub_gcm.classify();
                if sub_type != KacMoodyType::Finite && sub_type != KacMoodyType::Affine {
                    return false;
                }
            }
        }

        true
    }

    /// Get the Dynkin diagram for this GCM.
    pub fn dynkin_diagram(&self) -> DynkinDiagram {
        let mut nodes: Vec<DynkinNode> = (0..self.rank)
            .map(|i| DynkinNode {
                index: i,
                label: format!("{}", i + 1),
                is_affine_extension: false,
            })
            .collect();

        let mut edges = Vec::new();
        let mut connected: HashSet<(usize, usize)> = HashSet::new();

        for i in 0..self.rank {
            for j in (i + 1)..self.rank {
                if self.entries[i][j] != 0 {
                    let a_ij = self.entries[i][j];
                    let a_ji = self.entries[j][i];

                    // Multiplicity = a_ij * a_ji (always positive)
                    let mult = (a_ij * a_ji) as u8;

                    // Arrow points to shorter root if not symmetric
                    let arrow = if a_ij != a_ji {
                        Some(a_ij.abs() > a_ji.abs())
                    } else {
                        None
                    };

                    edges.push(DynkinEdge {
                        from: i,
                        to: j,
                        multiplicity: mult,
                        arrow_to_shorter: arrow,
                    });

                    connected.insert((i, j));
                }
            }
        }

        // Try to identify the algebra type
        let algebra_type = self.identify_algebra_type();

        // Mark affine extension node if applicable
        if let Some(LieAlgebraType::E9) = algebra_type {
            if self.rank == 9 {
                nodes[0].is_affine_extension = true; // Node 0 is traditionally the affine extension
            }
        }

        DynkinDiagram {
            nodes,
            edges,
            algebra_type,
        }
    }

    /// Try to identify the specific Lie algebra type.
    fn identify_algebra_type(&self) -> Option<LieAlgebraType> {
        let det = self.determinant();
        let classification = self.classify();

        match classification {
            KacMoodyType::Finite => {
                // Check E-series first (simply-laced exceptional)
                if self.is_simply_laced() {
                    match self.rank {
                        6 if det == 3 => return Some(LieAlgebraType::E6),
                        7 if det == 2 => return Some(LieAlgebraType::E7),
                        8 if det == 1 => return Some(LieAlgebraType::E8),
                        n if det == (n + 1) as i64 => return Some(LieAlgebraType::A(n)),
                        n if det == 4 && n >= 4 => return Some(LieAlgebraType::D(n)),
                        _ => {}
                    }
                }
                // Check F4, G2
                if self.rank == 4 && det == 1 && !self.is_symmetric() {
                    return Some(LieAlgebraType::F4);
                }
                if self.rank == 2 && det == 1 && !self.is_symmetric() {
                    return Some(LieAlgebraType::G2);
                }
            }
            KacMoodyType::Affine => {
                if det == 0 && self.rank == 9 && self.is_simply_laced() {
                    return Some(LieAlgebraType::E9);
                }
            }
            KacMoodyType::Hyperbolic | KacMoodyType::Lorentzian => {
                if self.rank == 10 && self.is_simply_laced() {
                    return Some(LieAlgebraType::E10);
                }
                if self.rank == 11 && self.is_simply_laced() {
                    return Some(LieAlgebraType::E11);
                }
            }
            _ => {}
        }

        None
    }
}

// === E-series Cartan matrices ===

/// E8 Cartan matrix (finite, simply-laced, exceptional).
pub fn e8_cartan() -> GeneralizedCartanMatrix {
    GeneralizedCartanMatrix::from_array([
        [ 2, -1,  0,  0,  0,  0,  0,  0],
        [-1,  2, -1,  0,  0,  0,  0,  0],
        [ 0, -1,  2, -1,  0,  0,  0, -1],
        [ 0,  0, -1,  2, -1,  0,  0,  0],
        [ 0,  0,  0, -1,  2, -1,  0,  0],
        [ 0,  0,  0,  0, -1,  2, -1,  0],
        [ 0,  0,  0,  0,  0, -1,  2,  0],
        [ 0,  0, -1,  0,  0,  0,  0,  2],
    ]).expect("E8 Cartan matrix is valid")
}

/// E9 = E8^{(1)} Cartan matrix (affine extension of E8).
///
/// E9 is the affine Kac-Moody algebra associated with E8.
/// It has rank 9 with determinant 0 (one null direction).
/// Important in heterotic string theory compactifications.
///
/// The affine extension adds node 0 connected to node 7 (the end of the long branch).
/// The highest root of E8 in terms of simple roots is:
///   theta = 2*alpha_1 + 3*alpha_2 + 4*alpha_3 + 5*alpha_4 + 6*alpha_5 + 4*alpha_6 + 2*alpha_7 + 3*alpha_8
/// Node 0 connects to nodes such that -alpha_0 = theta.
pub fn e9_cartan() -> GeneralizedCartanMatrix {
    // Standard affine E8^(1) Dynkin diagram (Kac numbering):
    //
    //         0
    //         |
    //     1 - 2 - 3 - 4 - 5 - 6 - 7 - 8
    //                     |
    //                     9  (we use 0-based, so this is index 8)
    //
    // In our 0-based indexing with E8 as nodes 1-8, node 0 is the affine extension.
    // Node 0 connects to node 2 (which is alpha_1 in Bourbaki = our node index 1)
    // Actually, for E8^(1), node 0 typically connects to the node at the end of
    // the long arm, which is node 8 in Bourbaki (our index 7).
    //
    // Using Kac's convention: node 0 connects to node 1 (which was alpha_1)
    GeneralizedCartanMatrix::from_array([
        [ 2, -1,  0,  0,  0,  0,  0,  0,  0],  // 0 (affine extension, connects to 1)
        [-1,  2, -1,  0,  0,  0,  0,  0,  0],  // 1
        [ 0, -1,  2, -1,  0,  0,  0,  0, -1],  // 2 (branching node in standard E8)
        [ 0,  0, -1,  2, -1,  0,  0,  0,  0],  // 3
        [ 0,  0,  0, -1,  2, -1,  0,  0,  0],  // 4
        [ 0,  0,  0,  0, -1,  2, -1,  0,  0],  // 5
        [ 0,  0,  0,  0,  0, -1,  2, -1,  0],  // 6
        [ 0,  0,  0,  0,  0,  0, -1,  2,  0],  // 7
        [ 0,  0, -1,  0,  0,  0,  0,  0,  2],  // 8 (the short branch)
    ]).expect("E9 Cartan matrix is valid")
}

/// E10 Cartan matrix (hyperbolic, Lorentzian signature).
///
/// E10 is conjectured to be a symmetry of M-theory (Damour-Henneaux-Nicolai).
/// The Cartan matrix has signature (9, 1).
pub fn e10_cartan() -> GeneralizedCartanMatrix {
    // E10 extends E9 by adding node 10 connected to node 0
    // Dynkin diagram (linear chain with one branch):
    //
    //     10 - 0 - 1 - 2 - 3 - 4 - 5 - 6 - 7
    //                      |
    //                      8
    //
    GeneralizedCartanMatrix::from_array([
        [ 2, -1,  0,  0,  0,  0,  0,  0,  0, -1],  // 0
        [-1,  2, -1,  0,  0,  0,  0,  0,  0,  0],  // 1
        [ 0, -1,  2, -1,  0,  0,  0,  0,  0,  0],  // 2
        [ 0,  0, -1,  2, -1,  0,  0,  0, -1,  0],  // 3
        [ 0,  0,  0, -1,  2, -1,  0,  0,  0,  0],  // 4
        [ 0,  0,  0,  0, -1,  2, -1,  0,  0,  0],  // 5
        [ 0,  0,  0,  0,  0, -1,  2, -1,  0,  0],  // 6
        [ 0,  0,  0,  0,  0,  0, -1,  2,  0,  0],  // 7
        [ 0,  0,  0, -1,  0,  0,  0,  0,  2,  0],  // 8
        [-1,  0,  0,  0,  0,  0,  0,  0,  0,  2],  // 9 (new hyperbolic extension)
    ]).expect("E10 Cartan matrix is valid")
}

/// E11 Cartan matrix (very extended E8).
///
/// E11 is proposed as a hidden symmetry of 11D supergravity (West 2001).
/// Contains E10 as a subalgebra.
pub fn e11_cartan() -> GeneralizedCartanMatrix {
    // E11 extends E10 by adding node 11 connected to node 10
    GeneralizedCartanMatrix::from_array([
        [ 2, -1,  0,  0,  0,  0,  0,  0,  0, -1,  0],  // 0
        [-1,  2, -1,  0,  0,  0,  0,  0,  0,  0,  0],  // 1
        [ 0, -1,  2, -1,  0,  0,  0,  0,  0,  0,  0],  // 2
        [ 0,  0, -1,  2, -1,  0,  0,  0, -1,  0,  0],  // 3
        [ 0,  0,  0, -1,  2, -1,  0,  0,  0,  0,  0],  // 4
        [ 0,  0,  0,  0, -1,  2, -1,  0,  0,  0,  0],  // 5
        [ 0,  0,  0,  0,  0, -1,  2, -1,  0,  0,  0],  // 6
        [ 0,  0,  0,  0,  0,  0, -1,  2,  0,  0,  0],  // 7
        [ 0,  0,  0, -1,  0,  0,  0,  0,  2,  0,  0],  // 8
        [-1,  0,  0,  0,  0,  0,  0,  0,  0,  2, -1],  // 9
        [ 0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  2],  // 10 (very extended)
    ]).expect("E11 Cartan matrix is valid")
}

/// Create the A_n Cartan matrix (SL(n+1)).
pub fn a_n_cartan(n: usize) -> GeneralizedCartanMatrix {
    let mut entries = vec![vec![0; n]; n];
    for i in 0..n {
        entries[i][i] = 2;
        if i > 0 {
            entries[i][i - 1] = -1;
        }
        if i < n - 1 {
            entries[i][i + 1] = -1;
        }
    }
    GeneralizedCartanMatrix::new(entries).expect("A_n Cartan matrix is valid")
}

/// Create the D_n Cartan matrix (SO(2n)).
pub fn d_n_cartan(n: usize) -> GeneralizedCartanMatrix {
    assert!(n >= 4, "D_n requires n >= 4");
    let mut entries = vec![vec![0; n]; n];

    // Linear chain 0 - 1 - 2 - ... - (n-3) - (n-2)
    for (i, row) in entries.iter_mut().enumerate() {
        row[i] = 2;
    }
    for i in 0..(n - 2) {
        entries[i][i + 1] = -1;
        entries[i + 1][i] = -1;
    }

    // Branch: (n-3) connected to both (n-2) and (n-1)
    entries[n - 3][n - 1] = -1;
    entries[n - 1][n - 3] = -1;

    GeneralizedCartanMatrix::new(entries).expect("D_n Cartan matrix is valid")
}

/// Weyl group information for a Kac-Moody algebra.
#[derive(Debug, Clone)]
pub struct WeylGroupInfo {
    /// Number of simple reflections (equals rank)
    pub num_generators: usize,
    /// Order of finite Weyl group (None for infinite)
    pub order: Option<u128>,
    /// Whether the Weyl group is finite
    pub is_finite: bool,
}

impl GeneralizedCartanMatrix {
    /// Get information about the Weyl group.
    pub fn weyl_group_info(&self) -> WeylGroupInfo {
        let classification = self.classify();
        let is_finite = classification == KacMoodyType::Finite;

        let order = if is_finite {
            self.compute_weyl_order()
        } else {
            None
        };

        WeylGroupInfo {
            num_generators: self.rank,
            order,
            is_finite,
        }
    }

    /// Compute Weyl group order for finite types.
    fn compute_weyl_order(&self) -> Option<u128> {
        let algebra_type = self.identify_algebra_type()?;

        match algebra_type {
            LieAlgebraType::A(n) => {
                // |W(A_n)| = (n+1)!
                Some((1..=(n + 1) as u128).product())
            }
            LieAlgebraType::D(n) => {
                // |W(D_n)| = 2^{n-1} * n!
                let factorial: u128 = (1..=n as u128).product();
                Some((1u128 << (n - 1)) * factorial)
            }
            LieAlgebraType::E6 => Some(51840),
            LieAlgebraType::E7 => Some(2903040),
            LieAlgebraType::E8 => Some(696729600),
            LieAlgebraType::F4 => Some(1152),
            LieAlgebraType::G2 => Some(12),
            _ => None, // Infinite Weyl groups
        }
    }
}

/// Root system representation for Kac-Moody algebras.
#[derive(Debug, Clone)]
pub struct KacMoodyRootSystem {
    /// Simple roots (alpha_1, ..., alpha_n) as column vectors
    pub simple_roots: Vec<Vec<f64>>,
    /// Simple coroots (alpha_1^v, ..., alpha_n^v)
    pub simple_coroots: Vec<Vec<f64>>,
    /// The GCM
    pub cartan_matrix: GeneralizedCartanMatrix,
}

impl KacMoodyRootSystem {
    /// Create a root system from a GCM using the standard realization.
    pub fn from_gcm(gcm: GeneralizedCartanMatrix) -> Self {
        let n = gcm.rank();

        // Standard realization in R^n (for finite case)
        // Simple roots are rows of the identity matrix scaled by 2
        // Coroots are adjusted for non-simply-laced cases

        let simple_roots: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut root = vec![0.0; n];
                root[i] = 1.0;
                root
            })
            .collect();

        // For simply-laced, coroots = roots
        // For non-simply-laced, need to adjust by root lengths
        let simple_coroots = simple_roots.clone();

        Self {
            simple_roots,
            simple_coroots,
            cartan_matrix: gcm,
        }
    }

    /// Apply a simple reflection s_i to a weight vector.
    pub fn simple_reflection(&self, weight: &[f64], i: usize) -> Vec<f64> {
        let n = weight.len();
        let mut result = weight.to_vec();

        // s_i(lambda) = lambda - <lambda, alpha_i^v> * alpha_i
        let pairing: f64 = weight.iter()
            .zip(&self.simple_coroots[i])
            .map(|(w, c)| w * c)
            .sum();

        #[allow(clippy::needless_range_loop)]
        for j in 0..n {
            result[j] -= pairing * self.simple_roots[i][j];
        }

        result
    }
}

// === Extended E-series Root Systems ===

/// Root type for affine and indefinite Kac-Moody algebras.
#[derive(Debug, Clone, PartialEq)]
pub struct KacMoodyRoot {
    /// Finite part in R^n (embedding of simple roots)
    pub finite_part: Vec<f64>,
    /// Affine/imaginary level (coefficient of delta)
    pub level: i32,
    /// For E10+: additional Lorentzian coordinates
    pub lorentz_coords: Vec<f64>,
    /// Real/imaginary classification
    pub root_type: RootType,
}

/// Classification of roots in Kac-Moody algebras.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootType {
    /// Real roots: have squared length > 0
    Real,
    /// Imaginary roots: have squared length <= 0
    Imaginary,
    /// Null roots (affine algebras): squared length = 0
    Null,
}

impl KacMoodyRoot {
    /// Create a real root from finite coordinates.
    pub fn real(coords: Vec<f64>) -> Self {
        Self {
            finite_part: coords,
            level: 0,
            lorentz_coords: vec![],
            root_type: RootType::Real,
        }
    }

    /// Create an affine root at a given level.
    pub fn affine(finite_part: Vec<f64>, level: i32) -> Self {
        Self {
            finite_part,
            level,
            lorentz_coords: vec![],
            root_type: if level == 0 { RootType::Real } else { RootType::Imaginary },
        }
    }

    /// Create a root with Lorentzian extension (for E10+).
    pub fn lorentzian(finite_part: Vec<f64>, level: i32, lorentz: Vec<f64>) -> Self {
        Self {
            finite_part,
            level,
            lorentz_coords: lorentz,
            root_type: RootType::Real, // Will be determined by norm
        }
    }

    /// Compute the squared norm using the appropriate inner product.
    pub fn norm_squared(&self, signature: &[i32]) -> f64 {
        let mut result = 0.0;

        // Finite part: positive definite
        for &x in &self.finite_part {
            result += x * x;
        }

        // Lorentzian part: use signature
        for (i, &x) in self.lorentz_coords.iter().enumerate() {
            if i < signature.len() {
                result += signature[i] as f64 * x * x;
            }
        }

        result
    }

    /// Add two roots.
    pub fn add(&self, other: &Self) -> Self {
        let max_finite = self.finite_part.len().max(other.finite_part.len());
        let max_lorentz = self.lorentz_coords.len().max(other.lorentz_coords.len());

        let mut finite_part = vec![0.0; max_finite];
        let mut lorentz_coords = vec![0.0; max_lorentz];

        for (i, &x) in self.finite_part.iter().enumerate() {
            finite_part[i] += x;
        }
        for (i, &x) in other.finite_part.iter().enumerate() {
            finite_part[i] += x;
        }

        for (i, &x) in self.lorentz_coords.iter().enumerate() {
            lorentz_coords[i] += x;
        }
        for (i, &x) in other.lorentz_coords.iter().enumerate() {
            lorentz_coords[i] += x;
        }

        Self {
            finite_part,
            level: self.level + other.level,
            lorentz_coords,
            root_type: RootType::Real, // Will be recomputed
        }
    }
}

/// E9 = E8^{(1)} root system (affine E8).
///
/// The root system consists of:
/// - Real roots: alpha + n*delta for alpha in E8 roots, n in Z
/// - Imaginary roots: n*delta for n != 0
///   where delta is the null root (minimal imaginary root).
#[derive(Debug, Clone)]
pub struct E9RootSystem {
    /// E8 simple roots (finite part)
    pub e8_simple_roots: Vec<KacMoodyRoot>,
    /// The null root delta
    pub delta: KacMoodyRoot,
    /// Cartan matrix
    pub cartan: GeneralizedCartanMatrix,
}

impl E9RootSystem {
    /// Create the E9 root system.
    pub fn new() -> Self {
        // E8 simple roots in R^8
        let e8_simple = vec![
            vec![1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 0.0],
            vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            vec![-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5],
        ];

        let e8_simple_roots: Vec<KacMoodyRoot> = e8_simple.into_iter()
            .map(KacMoodyRoot::real)
            .collect();

        // Delta is the null root corresponding to the affine extension
        // It represents the imaginary direction
        let delta = KacMoodyRoot {
            finite_part: vec![0.0; 8],
            level: 1,
            lorentz_coords: vec![],
            root_type: RootType::Null,
        };

        Self {
            e8_simple_roots,
            delta,
            cartan: e9_cartan(),
        }
    }

    /// Get the affine simple root (alpha_0).
    /// alpha_0 = delta - theta where theta is the highest root of E8.
    pub fn affine_simple_root(&self) -> KacMoodyRoot {
        // Highest root of E8: theta = 2*alpha_1 + 3*alpha_2 + 4*alpha_3 + ...
        // In our coordinates: (1, -1, 0, 0, 0, 0, 0, 0) + multiples
        // Simplified: use the standard highest root
        let theta = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]; // Approximate
        let neg_theta: Vec<f64> = theta.iter().map(|x| -x).collect();

        KacMoodyRoot::affine(neg_theta, 1)
    }

    /// Generate real roots up to a maximum level.
    pub fn real_roots_up_to_level(&self, max_level: u32) -> Vec<KacMoodyRoot> {
        let mut roots = Vec::new();

        // Level 0: E8 roots (240 of them)
        // We just generate a sample for now
        for root in &self.e8_simple_roots {
            roots.push(root.clone());
            // Also add negative roots
            let neg: Vec<f64> = root.finite_part.iter().map(|x| -x).collect();
            roots.push(KacMoodyRoot::real(neg));
        }

        // Higher levels: alpha + n*delta
        for level in 1..=max_level {
            for base_root in &self.e8_simple_roots {
                let mut pos = base_root.clone();
                pos.level = level as i32;
                pos.root_type = RootType::Real;
                roots.push(pos);

                let mut neg = base_root.clone();
                neg.finite_part = neg.finite_part.iter().map(|x| -x).collect();
                neg.level = -(level as i32);
                neg.root_type = RootType::Real;
                roots.push(neg);
            }
        }

        roots
    }

    /// Count of imaginary roots at level n (multiplicity).
    /// For E9, imaginary root multiplicity at level n is 8 (dimension of E8 Cartan).
    pub fn imaginary_root_multiplicity(&self, level: i32) -> usize {
        if level == 0 {
            0 // No imaginary roots at level 0
        } else {
            8 // Cartan subalgebra dimension
        }
    }
}

impl Default for E9RootSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// E10 root system (hyperbolic, Lorentzian).
///
/// E10 is the over-extended E8, conjectured to be a symmetry of M-theory.
/// Its root lattice has signature (9, 1) - Lorentzian.
#[derive(Debug, Clone)]
pub struct E10RootSystem {
    /// E9 embedded in E10
    pub e9_roots: E9RootSystem,
    /// The hyperbolic extension direction
    pub hyperbolic_coord: f64,
    /// Cartan matrix
    pub cartan: GeneralizedCartanMatrix,
}

impl E10RootSystem {
    /// Create the E10 root system.
    pub fn new() -> Self {
        Self {
            e9_roots: E9RootSystem::new(),
            hyperbolic_coord: 0.0,
            cartan: e10_cartan(),
        }
    }

    /// The signature of the E10 root lattice: (9, 1).
    pub fn signature() -> (usize, usize) {
        (9, 1) // 9 positive, 1 negative eigenvalue
    }

    /// Inner product for E10 (Lorentzian).
    pub fn inner_product(&self, a: &KacMoodyRoot, b: &KacMoodyRoot) -> f64 {
        let mut result = 0.0;

        // Finite part: positive definite (E8 inner product)
        for (x, y) in a.finite_part.iter().zip(b.finite_part.iter()) {
            result += x * y;
        }

        // Level contribution (from affine direction)
        result += (a.level * b.level) as f64;

        // Lorentzian direction: negative
        if !a.lorentz_coords.is_empty() && !b.lorentz_coords.is_empty() {
            result -= a.lorentz_coords[0] * b.lorentz_coords[0];
        }

        result
    }

    /// Generate simple roots for E10.
    pub fn simple_roots(&self) -> Vec<KacMoodyRoot> {
        let mut roots = Vec::with_capacity(10);

        // E8 simple roots (indices 1-8 in standard numbering)
        for root in &self.e9_roots.e8_simple_roots {
            roots.push(root.clone());
        }

        // E9 affine root (index 0)
        roots.push(self.e9_roots.affine_simple_root());

        // E10 hyperbolic extension (index -1 or 9)
        let hyperbolic_root = KacMoodyRoot::lorentzian(
            vec![0.0; 8],
            -1,
            vec![1.0], // Lorentzian direction
        );
        roots.push(hyperbolic_root);

        roots
    }

    /// Check if a root is timelike, spacelike, or null.
    pub fn causal_type(&self, root: &KacMoodyRoot) -> &'static str {
        let norm_sq = self.inner_product(root, root);
        if norm_sq > 1e-10 {
            "spacelike"
        } else if norm_sq < -1e-10 {
            "timelike"
        } else {
            "null"
        }
    }
}

impl Default for E10RootSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// E11 root system (very extended E8).
///
/// E11 is proposed as a hidden symmetry of 11D supergravity (West 2001).
/// Its structure is even more complex than E10.
#[derive(Debug, Clone)]
pub struct E11RootSystem {
    /// E10 embedded in E11
    pub e10_base: E10RootSystem,
    /// Cartan matrix
    pub cartan: GeneralizedCartanMatrix,
}

impl E11RootSystem {
    /// Create the E11 root system.
    pub fn new() -> Self {
        Self {
            e10_base: E10RootSystem::new(),
            cartan: e11_cartan(),
        }
    }

    /// The signature of E11.
    /// E11 has an even more indefinite signature than E10.
    pub fn signature() -> (usize, usize) {
        (9, 2) // Rough estimate - actual structure is more complex
    }

    /// Connection to 11D supergravity (informational).
    pub fn supergravity_connection() -> &'static str {
        "E11 is conjectured to be a symmetry of 11D supergravity. \
         Its level decomposition under GL(11) gives the graviton, \
         3-form, 6-form, and dual graviton representations. \
         Reference: West (2001), Class. Quantum Grav. 18, 4443."
    }

    /// Decomposition levels relevant to M-theory.
    /// Returns (level, representation dimension, interpretation).
    pub fn mtheory_level_decomposition() -> Vec<(i32, &'static str, &'static str)> {
        vec![
            (0, "SO(1,10)", "Spacetime Lorentz group"),
            (1, "A_abc", "3-form potential (M2-brane)"),
            (2, "A_abcdef", "6-form potential (M5-brane)"),
            (3, "h_a,bcdefghi", "Dual graviton"),
            // Higher levels continue...
        ]
    }
}

impl Default for E11RootSystem {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified interface for E-series root systems.
#[derive(Debug, Clone)]
pub enum ESeriesRootSystem {
    E8(Box<crate::e8_lattice::E8Lattice>),
    E9(E9RootSystem),
    E10(E10RootSystem),
    E11(E11RootSystem),
}

impl ESeriesRootSystem {
    /// Get the rank of the root system.
    pub fn rank(&self) -> usize {
        match self {
            ESeriesRootSystem::E8(_) => 8,
            ESeriesRootSystem::E9(_) => 9,
            ESeriesRootSystem::E10(_) => 10,
            ESeriesRootSystem::E11(_) => 11,
        }
    }

    /// Get the classification.
    pub fn classification(&self) -> KacMoodyType {
        match self {
            ESeriesRootSystem::E8(_) => KacMoodyType::Finite,
            ESeriesRootSystem::E9(_) => KacMoodyType::Affine,
            ESeriesRootSystem::E10(_) => KacMoodyType::Hyperbolic,
            ESeriesRootSystem::E11(_) => KacMoodyType::Indefinite,
        }
    }

    /// Is the Weyl group finite?
    pub fn has_finite_weyl_group(&self) -> bool {
        matches!(self, ESeriesRootSystem::E8(_))
    }

    /// Physics applications.
    pub fn physics_applications(&self) -> Vec<&'static str> {
        match self {
            ESeriesRootSystem::E8(_) => vec![
                "Heterotic string theory (E8 x E8)",
                "Grand unified theories",
                "Moonshine connections to Monster group",
            ],
            ESeriesRootSystem::E9(_) => vec![
                "2D conformal field theory",
                "Affine Lie algebras in string theory",
                "WZW models",
            ],
            ESeriesRootSystem::E10(_) => vec![
                "M-theory hidden symmetry (Damour-Henneaux-Nicolai)",
                "Cosmological billiards near spacelike singularities",
                "BKL dynamics of general relativity",
            ],
            ESeriesRootSystem::E11(_) => vec![
                "11D supergravity symmetry (West)",
                "M-theory duality unification",
                "Form field democracy",
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e8_cartan_finite() {
        let e8 = e8_cartan();
        assert_eq!(e8.rank(), 8);
        assert!(e8.is_simply_laced());
        assert_eq!(e8.determinant(), 1);
        assert_eq!(e8.classify(), KacMoodyType::Finite);
        assert_eq!(e8.identify_algebra_type(), Some(LieAlgebraType::E8));
    }

    #[test]
    fn test_e9_cartan_affine() {
        let e9 = e9_cartan();
        assert_eq!(e9.rank(), 9);
        assert!(e9.is_simply_laced());
        assert_eq!(e9.determinant(), 0);
        assert_eq!(e9.classify(), KacMoodyType::Affine);
        assert_eq!(e9.identify_algebra_type(), Some(LieAlgebraType::E9));
    }

    #[test]
    fn test_e10_cartan_hyperbolic() {
        let e10 = e10_cartan();
        assert_eq!(e10.rank(), 10);
        assert!(e10.is_simply_laced());
        // E10 has determinant -1 (Lorentzian signature)
        assert!(e10.determinant() < 0);
        let classification = e10.classify();
        assert!(classification == KacMoodyType::Hyperbolic
            || classification == KacMoodyType::Lorentzian);
    }

    #[test]
    fn test_e11_cartan() {
        let e11 = e11_cartan();
        assert_eq!(e11.rank(), 11);
        assert!(e11.is_simply_laced());
        // E11 is indefinite
        let classification = e11.classify();
        assert!(classification != KacMoodyType::Finite);
        assert!(classification != KacMoodyType::Affine);
    }

    #[test]
    fn test_a_n_series() {
        for n in 2..=5 {
            let a_n = a_n_cartan(n);
            assert_eq!(a_n.rank(), n);
            assert!(a_n.is_simply_laced());
            assert_eq!(a_n.determinant(), (n + 1) as i64);
            assert_eq!(a_n.classify(), KacMoodyType::Finite);
        }
    }

    #[test]
    fn test_d_n_series() {
        for n in 4..=6 {
            let d_n = d_n_cartan(n);
            assert_eq!(d_n.rank(), n);
            assert!(d_n.is_simply_laced());
            assert_eq!(d_n.determinant(), 4);
            assert_eq!(d_n.classify(), KacMoodyType::Finite);
        }
    }

    #[test]
    fn test_weyl_group_orders() {
        let e8 = e8_cartan();
        let weyl = e8.weyl_group_info();
        assert!(weyl.is_finite);
        assert_eq!(weyl.order, Some(696729600));

        let e9 = e9_cartan();
        let weyl9 = e9.weyl_group_info();
        assert!(!weyl9.is_finite);
        assert_eq!(weyl9.order, None);
    }

    #[test]
    fn test_dynkin_diagram_e8() {
        let e8 = e8_cartan();
        let diagram = e8.dynkin_diagram();

        assert_eq!(diagram.nodes.len(), 8);
        assert_eq!(diagram.edges.len(), 7); // E8 has 7 edges (tree structure)

        // All edges should be single bonds (simply-laced)
        for edge in &diagram.edges {
            assert_eq!(edge.multiplicity, 1);
            assert!(edge.arrow_to_shorter.is_none());
        }
    }

    #[test]
    fn test_root_system_reflection() {
        let a2 = a_n_cartan(2);
        let root_sys = KacMoodyRootSystem::from_gcm(a2);

        let weight = vec![1.0, 0.0];
        let reflected = root_sys.simple_reflection(&weight, 0);

        // s_1(e_1) should give something different
        assert!((reflected[0] - weight[0]).abs() > 1e-10
             || (reflected[1] - weight[1]).abs() > 1e-10);
    }

    #[test]
    fn test_e_series_hierarchy() {
        // E8 -> E9 -> E10 -> E11 form a hierarchy
        let e8 = e8_cartan();
        let e9 = e9_cartan();
        let e10 = e10_cartan();
        let e11 = e11_cartan();

        // Check ranks increase
        assert_eq!(e8.rank(), 8);
        assert_eq!(e9.rank(), 9);
        assert_eq!(e10.rank(), 10);
        assert_eq!(e11.rank(), 11);

        // Check E8 is finite, E9 affine, E10+ indefinite
        assert_eq!(e8.classify(), KacMoodyType::Finite);
        assert_eq!(e9.classify(), KacMoodyType::Affine);
        assert!(e10.classify() != KacMoodyType::Finite);
        assert!(e10.classify() != KacMoodyType::Affine);
    }

    // === E9/E10/E11 Root System Tests ===

    #[test]
    fn test_e9_root_system_creation() {
        let e9 = E9RootSystem::new();

        // Should have 8 E8 simple roots
        assert_eq!(e9.e8_simple_roots.len(), 8);

        // Delta should be at level 1
        assert_eq!(e9.delta.level, 1);
        assert_eq!(e9.delta.root_type, RootType::Null);

        // Cartan matrix should be 9x9 with determinant 0
        assert_eq!(e9.cartan.rank(), 9);
        assert_eq!(e9.cartan.determinant(), 0);
    }

    #[test]
    fn test_e9_imaginary_root_multiplicity() {
        let e9 = E9RootSystem::new();

        // Level 0 has no imaginary roots
        assert_eq!(e9.imaginary_root_multiplicity(0), 0);

        // Other levels have multiplicity 8 (E8 Cartan dimension)
        assert_eq!(e9.imaginary_root_multiplicity(1), 8);
        assert_eq!(e9.imaginary_root_multiplicity(-1), 8);
        assert_eq!(e9.imaginary_root_multiplicity(5), 8);
    }

    #[test]
    fn test_e9_real_roots_generation() {
        let e9 = E9RootSystem::new();
        let roots = e9.real_roots_up_to_level(2);

        // Should have roots at levels -2, -1, 0, 1, 2
        // Level 0: 16 (8 simple + 8 negative)
        // Levels +/-1: 16 each
        // Levels +/-2: 16 each
        assert!(roots.len() >= 16); // At minimum level 0

        // Check some roots are at different levels
        let levels: std::collections::HashSet<i32> = roots.iter()
            .map(|r| r.level)
            .collect();
        assert!(levels.contains(&0));
    }

    #[test]
    fn test_e10_root_system_creation() {
        let e10 = E10RootSystem::new();

        // Cartan matrix should be 10x10
        assert_eq!(e10.cartan.rank(), 10);

        // Signature should be (9, 1)
        let (pos, neg) = E10RootSystem::signature();
        assert_eq!(pos, 9);
        assert_eq!(neg, 1);
    }

    #[test]
    fn test_e10_simple_roots() {
        let e10 = E10RootSystem::new();
        let simple_roots = e10.simple_roots();

        // Should have 10 simple roots
        assert_eq!(simple_roots.len(), 10);

        // Last root should have Lorentzian coordinates
        let hyperbolic = &simple_roots[9];
        assert!(!hyperbolic.lorentz_coords.is_empty());
    }

    #[test]
    fn test_e10_inner_product_lorentzian() {
        let e10 = E10RootSystem::new();

        // Create a spacelike root
        let spacelike = KacMoodyRoot::real(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(e10.causal_type(&spacelike), "spacelike");

        // Create a timelike root (Lorentzian direction dominant)
        let timelike = KacMoodyRoot::lorentzian(
            vec![0.0; 8],
            0,
            vec![2.0],
        );
        assert_eq!(e10.causal_type(&timelike), "timelike");

        // Create a null root
        let null = KacMoodyRoot::lorentzian(
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            0,
            vec![1.0],
        );
        assert_eq!(e10.causal_type(&null), "null");
    }

    #[test]
    fn test_e11_root_system_creation() {
        let e11 = E11RootSystem::new();

        // Cartan matrix should be 11x11
        assert_eq!(e11.cartan.rank(), 11);

        // Should have physics connections
        let connection = E11RootSystem::supergravity_connection();
        assert!(connection.contains("11D supergravity"));
    }

    #[test]
    fn test_e11_mtheory_decomposition() {
        let decomp = E11RootSystem::mtheory_level_decomposition();

        // Should have at least the known low-level representations
        assert!(decomp.len() >= 3);

        // Level 0 should be Lorentz group
        assert_eq!(decomp[0].0, 0);
        assert!(decomp[0].1.contains("SO"));

        // Level 1 should be 3-form (M2-brane)
        assert_eq!(decomp[1].0, 1);
        assert!(decomp[1].2.contains("M2"));
    }

    #[test]
    fn test_kac_moody_root_arithmetic() {
        let a = KacMoodyRoot::affine(vec![1.0, 0.0, 0.0], 1);
        let b = KacMoodyRoot::affine(vec![0.0, 1.0, 0.0], 2);

        let sum = a.add(&b);

        assert_eq!(sum.finite_part, vec![1.0, 1.0, 0.0]);
        assert_eq!(sum.level, 3);
    }

    #[test]
    fn test_eseries_unified_interface() {
        let e9_sys = ESeriesRootSystem::E9(E9RootSystem::new());
        let e10_sys = ESeriesRootSystem::E10(E10RootSystem::new());
        let e11_sys = ESeriesRootSystem::E11(E11RootSystem::new());

        // Check ranks
        assert_eq!(e9_sys.rank(), 9);
        assert_eq!(e10_sys.rank(), 10);
        assert_eq!(e11_sys.rank(), 11);

        // Check classifications
        assert_eq!(e9_sys.classification(), KacMoodyType::Affine);
        assert_eq!(e10_sys.classification(), KacMoodyType::Hyperbolic);
        assert_eq!(e11_sys.classification(), KacMoodyType::Indefinite);

        // Only E8 has finite Weyl group
        assert!(!e9_sys.has_finite_weyl_group());
        assert!(!e10_sys.has_finite_weyl_group());
        assert!(!e11_sys.has_finite_weyl_group());
    }

    #[test]
    fn test_physics_applications() {
        let e10_sys = ESeriesRootSystem::E10(E10RootSystem::new());
        let apps = e10_sys.physics_applications();

        assert!(!apps.is_empty());
        assert!(apps.iter().any(|s| s.contains("M-theory")));
    }
}
