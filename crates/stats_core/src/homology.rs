//! Simplicial Homology over Z_2 (T-002, L-003).
//!
//! Computes Betti numbers of abstract simplicial complexes using GF(2) boundary
//! matrices and Gaussian elimination. This generalizes the graph-based betti_0/betti_1
//! in hypergraph.rs to arbitrary simplicial dimension.
//!
//! # Mathematics
//!
//! An abstract simplicial complex K is a collection of finite sets (simplices)
//! closed under taking subsets. The k-th boundary map d_k: C_k -> C_{k-1}
//! maps each k-simplex to the formal sum of its (k-1)-faces over Z_2.
//!
//! The k-th Betti number is:
//!   beta_k = dim ker(d_k) - dim im(d_{k+1})
//!          = nullity(d_k) - rank(d_{k+1})
//!
//! Over Z_2, all coefficients are 0 or 1 (no signs needed).
//!
//! # References
//! - Munkres, "Elements of Algebraic Topology" (1984)
//! - Edelsbrunner & Harer, "Computational Topology" (2010)

use std::collections::{BTreeSet, HashMap};

/// A simplex represented as a sorted set of vertex indices.
pub type Simplex = BTreeSet<usize>;

/// Abstract simplicial complex with boundary matrix computation.
#[derive(Debug, Clone)]
pub struct SimplicialComplex {
    /// All simplices, grouped by dimension.
    /// simplices[k] contains the k-simplices (k+1 vertices each).
    simplices: Vec<Vec<Simplex>>,
    /// Maximum dimension of any simplex.
    max_dim: usize,
}

/// Result of Betti number computation.
#[derive(Debug, Clone)]
pub struct BettiResult {
    /// Betti numbers beta_0, beta_1, ..., beta_{max_dim}.
    pub betti: Vec<usize>,
    /// Euler characteristic = sum_{k} (-1)^k * beta_k.
    pub euler_characteristic: i64,
}

impl SimplicialComplex {
    /// Create an empty simplicial complex.
    pub fn new() -> Self {
        Self {
            simplices: Vec::new(),
            max_dim: 0,
        }
    }

    /// Add a simplex and all of its faces (closure under subsets).
    ///
    /// A k-simplex has k+1 vertices. For example, a triangle {0,1,2} is a
    /// 2-simplex and generates edges {0,1}, {0,2}, {1,2} and vertices {0}, {1}, {2}.
    pub fn add_simplex(&mut self, vertices: &[usize]) {
        let simplex: Simplex = vertices.iter().copied().collect();
        let dim = simplex.len().saturating_sub(1);

        // Add all faces (subsets) recursively
        self.add_all_faces(&simplex);

        if dim > self.max_dim {
            self.max_dim = dim;
        }
    }

    /// Add a simplex and all its faces (power set minus empty set).
    fn add_all_faces(&mut self, simplex: &Simplex) {
        if simplex.is_empty() {
            return;
        }

        let dim = simplex.len() - 1;

        // Ensure we have enough dimension slots
        while self.simplices.len() <= dim {
            self.simplices.push(Vec::new());
        }

        // Add this simplex if not already present
        if !self.simplices[dim].contains(simplex) {
            self.simplices[dim].push(simplex.clone());
        }

        // Add all (dim-1)-faces: remove each vertex in turn
        if simplex.len() > 1 {
            for &v in simplex.iter() {
                let mut face = simplex.clone();
                face.remove(&v);
                self.add_all_faces(&face);
            }
        }
    }

    /// Number of k-simplices.
    pub fn count(&self, dim: usize) -> usize {
        self.simplices.get(dim).map_or(0, |s| s.len())
    }

    /// Maximum simplex dimension.
    pub fn dimension(&self) -> usize {
        self.max_dim
    }

    /// Compute the boundary matrix d_k: C_k -> C_{k-1} over Z_2.
    ///
    /// Returns a (rows x cols) binary matrix where rows index (k-1)-simplices
    /// and cols index k-simplices. Entry [i][j] = 1 iff the i-th (k-1)-simplex
    /// is a face of the j-th k-simplex.
    pub fn boundary_matrix_z2(&self, k: usize) -> Vec<Vec<u8>> {
        if k == 0 || k >= self.simplices.len() {
            return vec![];
        }

        let k_simplices = &self.simplices[k];
        let km1_simplices = &self.simplices[k - 1];

        // Build index map for (k-1)-simplices
        let km1_index: HashMap<&Simplex, usize> = km1_simplices
            .iter()
            .enumerate()
            .map(|(i, s)| (s, i))
            .collect();

        let n_rows = km1_simplices.len();
        let n_cols = k_simplices.len();
        let mut matrix = vec![vec![0u8; n_cols]; n_rows];

        for (col, sigma) in k_simplices.iter().enumerate() {
            // Each face of sigma is obtained by removing one vertex
            for &v in sigma.iter() {
                let mut face = sigma.clone();
                face.remove(&v);
                if let Some(&row) = km1_index.get(&face) {
                    // Over Z_2, all boundary coefficients are 1 (no signs)
                    matrix[row][col] = 1;
                }
            }
        }

        matrix
    }

    /// Compute rank of a Z_2 matrix via Gaussian elimination.
    ///
    /// Operates in-place on a copy using row reduction modulo 2.
    fn rank_z2(matrix: &[Vec<u8>]) -> usize {
        if matrix.is_empty() {
            return 0;
        }

        let n_rows = matrix.len();
        let n_cols = matrix[0].len();
        let mut mat: Vec<Vec<u8>> = matrix.to_vec();

        let mut rank = 0;
        let mut row = 0;
        let mut pivot_col = 0;

        // Gaussian elimination over GF(2)
        // Use while loop: row advances only on successful pivot, not on column skip.
        while row < n_rows && pivot_col < n_cols {
            // Find pivot in current column
            let mut pivot_row = None;
            #[allow(clippy::needless_range_loop)] // Index-based pivot search required
            for r in row..n_rows {
                if mat[r][pivot_col] == 1 {
                    pivot_row = Some(r);
                    break;
                }
            }

            if let Some(pr) = pivot_row {
                // Swap rows
                mat.swap(row, pr);

                // Eliminate all other 1s in this column
                for r in 0..n_rows {
                    if r != row && mat[r][pivot_col] == 1 {
                        // Two-row stencil: mat[r][c] ^= mat[row][c] requires index access
                        #[allow(clippy::needless_range_loop)]
                        for c in 0..n_cols {
                            mat[r][c] ^= mat[row][c];
                        }
                    }
                }

                rank += 1;
                row += 1;
                pivot_col += 1;
            } else {
                // No pivot in this column; try next column, same row
                pivot_col += 1;
            }
        }

        rank
    }

    /// Compute the k-th Betti number: beta_k = nullity(d_k) - rank(d_{k+1}).
    pub fn betti_k(&self, k: usize) -> usize {
        let n_k = self.count(k);
        if n_k == 0 {
            return 0;
        }

        // nullity(d_k) = dim(C_k) - rank(d_k)
        let rank_dk = if k == 0 {
            0 // d_0 is the zero map (no boundary for vertices)
        } else {
            let dk = self.boundary_matrix_z2(k);
            Self::rank_z2(&dk)
        };
        let nullity_dk = n_k - rank_dk;

        // rank(d_{k+1})
        let rank_dk1 = if k + 1 >= self.simplices.len() {
            0 // No (k+1)-simplices exist
        } else {
            let dk1 = self.boundary_matrix_z2(k + 1);
            Self::rank_z2(&dk1)
        };

        nullity_dk.saturating_sub(rank_dk1)
    }

    /// Compute all Betti numbers up to the maximum dimension.
    pub fn betti_numbers(&self) -> BettiResult {
        let betti: Vec<usize> = (0..=self.max_dim).map(|k| self.betti_k(k)).collect();

        let euler_characteristic: i64 = betti
            .iter()
            .enumerate()
            .map(|(k, &b)| if k % 2 == 0 { b as i64 } else { -(b as i64) })
            .sum();

        BettiResult {
            betti,
            euler_characteristic,
        }
    }

    /// Compute the Euler characteristic directly from simplex counts.
    ///
    /// chi = sum_{k=0}^{max_dim} (-1)^k * |K_k|
    ///
    /// This should equal sum_{k} (-1)^k * beta_k by the Euler-Poincare theorem.
    pub fn euler_characteristic(&self) -> i64 {
        self.simplices
            .iter()
            .enumerate()
            .map(|(k, simplices)| {
                let count = simplices.len() as i64;
                if k % 2 == 0 { count } else { -count }
            })
            .sum()
    }
}

impl Default for SimplicialComplex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_betti() {
        // A single point: beta_0 = 1, no higher Betti numbers
        let mut k = SimplicialComplex::new();
        k.add_simplex(&[0]);

        let result = k.betti_numbers();
        assert_eq!(result.betti, vec![1]);
        assert_eq!(result.euler_characteristic, 1);
    }

    #[test]
    fn test_triangle_boundary() {
        // Triangle boundary (3 edges, 3 vertices, no filled face):
        // {0,1}, {1,2}, {0,2} -- a 1-cycle
        // beta_0 = 1 (connected), beta_1 = 1 (one cycle)
        let mut k = SimplicialComplex::new();
        k.add_simplex(&[0, 1]);
        k.add_simplex(&[1, 2]);
        k.add_simplex(&[0, 2]);

        let result = k.betti_numbers();
        assert_eq!(result.betti[0], 1, "beta_0: connected");
        assert_eq!(result.betti[1], 1, "beta_1: one cycle (triangle boundary)");

        // Euler characteristic: V - E = 3 - 3 = 0
        assert_eq!(k.euler_characteristic(), 0);
        assert_eq!(result.euler_characteristic, 0);
    }

    #[test]
    fn test_filled_triangle() {
        // Filled triangle: {0,1,2} is a 2-simplex
        // beta_0 = 1 (connected), beta_1 = 0 (cycle filled), beta_2 = 0
        let mut k = SimplicialComplex::new();
        k.add_simplex(&[0, 1, 2]);

        assert_eq!(k.count(0), 3, "3 vertices");
        assert_eq!(k.count(1), 3, "3 edges");
        assert_eq!(k.count(2), 1, "1 face");

        let result = k.betti_numbers();
        assert_eq!(result.betti[0], 1, "beta_0: connected");
        assert_eq!(result.betti[1], 0, "beta_1: cycle killed by face");
        assert_eq!(result.betti[2], 0, "beta_2: no void");

        // Euler characteristic: V - E + F = 3 - 3 + 1 = 1
        assert_eq!(k.euler_characteristic(), 1);
    }

    #[test]
    fn test_disconnected_components() {
        // Two separate edges: {0,1} and {2,3}
        // beta_0 = 2 (two components), beta_1 = 0
        let mut k = SimplicialComplex::new();
        k.add_simplex(&[0, 1]);
        k.add_simplex(&[2, 3]);

        let result = k.betti_numbers();
        assert_eq!(result.betti[0], 2, "beta_0: two components");
        assert_eq!(result.betti[1], 0, "beta_1: no cycles");
    }

    #[test]
    fn test_torus_betti() {
        // Minimal triangulation of the torus with 7 vertices.
        // Vertices: 0..6
        // Z/7Z orbit: (i, i+1, i+3) mod 7 and (i, i+2, i+3) mod 7 for i=0..6
        // gives 14 unique triangles using all 21 edges of K7,
        // each edge in exactly 2 faces (valid closed surface).
        // Expected: beta_0 = 1, beta_1 = 2, beta_2 = 1
        let mut k = SimplicialComplex::new();
        let triangles: Vec<[usize; 3]> = vec![
            [0, 1, 3],
            [0, 1, 5],
            [0, 2, 3],
            [0, 2, 6],
            [0, 4, 5],
            [0, 4, 6],
            [1, 2, 4],
            [1, 2, 6],
            [1, 3, 4],
            [1, 5, 6],
            [2, 3, 5],
            [2, 4, 5],
            [3, 4, 6],
            [3, 5, 6],
        ];

        for tri in &triangles {
            k.add_simplex(tri);
        }

        assert_eq!(k.count(0), 7, "7 vertices");
        assert_eq!(k.count(1), 21, "21 edges");
        assert_eq!(k.count(2), 14, "14 triangles");

        let result = k.betti_numbers();
        assert_eq!(result.betti[0], 1, "beta_0: torus is connected");
        assert_eq!(
            result.betti[1], 2,
            "beta_1: torus has two independent loops"
        );
        assert_eq!(result.betti[2], 1, "beta_2: torus encloses one void");

        // Euler characteristic of torus = 0
        // V - E + F = 7 - 21 + 14 = 0
        assert_eq!(k.euler_characteristic(), 0);
        assert_eq!(result.euler_characteristic, 0);
    }
}
