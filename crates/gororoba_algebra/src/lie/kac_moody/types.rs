//! Generalized Cartan matrices, Dynkin diagrams, and classification.
//!
//! Hosts:
//! - [`CartanEntry`] -- integer entry type (i32).
//! - [`GeneralizedCartanMatrix`] -- the central GCM type with rank,
//!   classification (finite/affine/Lorentzian/hyperbolic/indefinite),
//!   determinant, simply-laced predicate, Weyl-group-order lookup.
//! - [`KacMoodyType`] -- five-way classification enum.
//! - [`LieAlgebraType`] -- named Lie algebras (`A_n`, `B_n`, ..., `E_8`,
//!   `E_9`, `E_10`, `E_11`, `F_4`, `G_2`).
//! - [`DynkinNode`], [`DynkinEdge`], [`DynkinDiagram`] -- diagram representation.
//! - [`WeylGroupInfo`] -- Weyl-group cardinality and reflection count.
//!
//! Specialized E-series content (factories, root systems, and the
//! infinite-dimensional `E_9, E_10, E_11`) lives in sibling modules
//! [`super::cartans`], [`super::roots`], and [`super::e_series`].

use std::collections::HashSet;

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
    A(usize), // SL(n+1)
    B(usize), // SO(2n+1)
    C(usize), // Sp(2n)
    D(usize), // SO(2n)
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
        for (i, row) in entries.iter().enumerate() {
            // Axiom 1: diagonal = 2
            if row[i] != 2 {
                return Err("GCM diagonal entries must be 2");
            }

            for (j, &val) in row.iter().enumerate() {
                if i != j {
                    // Axiom 2: off-diagonal <= 0
                    if val > 0 {
                        return Err("GCM off-diagonal entries must be <= 0");
                    }

                    // Axiom 3: symmetry of zeros
                    if (val == 0) != (entries[j][i] == 0) {
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
        let entries: Vec<Vec<CartanEntry>> = arr.iter().map(|row| row.to_vec()).collect();
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
        let mut matrix: Vec<Vec<f64>> = self
            .entries
            .iter()
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

    /// Compute eigenvalue sign counts (positive, zero, negative) for classification.
    ///
    /// Uses Sylvester's law of inertia via leading principal minors.
    /// When intermediate minors vanish (e.g., E10 whose E9 sub-block is degenerate),
    /// we count sign changes among the non-zero entries in the sequence
    /// (1, M_1, M_2, ..., M_n) to determine the number of negative eigenvalues,
    /// then use the overall determinant to distinguish true zero eigenvalues.
    pub fn eigenvalue_signs(&self) -> (usize, usize, usize) {
        let det = self.determinant();

        if self.rank == 1 {
            return if det > 0 {
                (1, 0, 0)
            } else if det == 0 {
                (0, 1, 0)
            } else {
                (0, 0, 1)
            };
        }

        // Build the sequence (1, M_1, M_2, ..., M_n)
        let mut minors = vec![1i64];
        for k in 1..=self.rank {
            minors.push(self.leading_principal_minor(k));
        }

        // Count sign changes among non-zero entries in the minor sequence.
        // By Sylvester's law of inertia (generalized for vanishing intermediate
        // minors), the number of sign changes in the filtered non-zero subsequence
        // equals the number of negative eigenvalues.
        let non_zero: Vec<i64> = minors.iter().copied().filter(|&m| m != 0).collect();
        let mut negative: usize = 0;
        for window in non_zero.windows(2) {
            if (window[0] > 0 && window[1] < 0) || (window[0] < 0 && window[1] > 0) {
                negative += 1;
            }
        }

        // Zero eigenvalues: the matrix is singular iff det = 0.
        // The corank equals the number of trailing zeros in the minor sequence.
        let zero: usize = if det == 0 {
            // Count trailing zeros in the minor sequence (excluding position 0)
            minors[1..]
                .iter()
                .rev()
                .take_while(|&&m| m == 0)
                .count()
                .max(1)
        } else {
            0
        };

        let positive = self.rank - negative - zero;

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
        let submatrix: Vec<Vec<CartanEntry>> = self.entries[..k]
            .iter()
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
        if let Some(LieAlgebraType::E9) = algebra_type
            && self.rank == 9
        {
            nodes[0].is_affine_extension = true; // Node 0 is traditionally the affine extension
        }

        DynkinDiagram {
            nodes,
            edges,
            algebra_type,
        }
    }

    /// Try to identify the specific Lie algebra type.
    pub fn identify_algebra_type(&self) -> Option<LieAlgebraType> {
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
            KacMoodyType::Affine if det == 0 && self.rank == 9 && self.is_simply_laced() => {
                return Some(LieAlgebraType::E9);
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
