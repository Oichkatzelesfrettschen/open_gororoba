//! Public types for the emanation-table subsystem.
//!
//! - `EtCell` records the per-cell product (row, col, product_index,
//!   sign, is_zero_divisor) for an emanation-table entry.
//! - `EmanationTable` is the (dim-1) x (dim-1) matrix of cells.
//! - `MandalaSummary` aggregates the sand-mandala sparsity diagnostic
//!   (fill ratio per row/col plus the global filled/total counts).
//! - `EtScaling` is the per-dimension diagnostic returned by the
//!   period-doubling scan (zd_ratio, motif component count, nodes per
//!   component).
//! - `CdGenerator` is the (dim, g=dim/2) helper used by the strut/triad
//!   identity G XOR S = X.

/// A single cell in an emanation table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EtCell {
    /// Row index (basis element index, 1..dim-1).
    pub row: usize,
    /// Column index (basis element index, 1..dim-1).
    pub col: usize,
    /// Product index: row XOR col.
    pub product_index: usize,
    /// Sign of the product: e_row * e_col = sign * e_{product_index}.
    pub sign: i32,
    /// Whether this pair participates in a diagonal zero-product
    /// (i.e., the pair (row, col) is a cross-assessor pair with at least
    /// one (s,t) solution to diag(row,s)*diag(col,t) = 0).
    pub is_zero_divisor: bool,
}

/// An emanation table for a Cayley-Dickson algebra of given dimension.
#[derive(Debug, Clone)]
pub struct EmanationTable {
    /// The Cayley-Dickson dimension (must be power of 2, >= 16).
    pub dim: usize,
    /// Table size: dim - 1 (indices 1..dim-1, square matrix).
    pub size: usize,
    /// Row-major storage: `cells[i][j]` for `i,j in 0..size`
    /// where basis index = i + 1.
    pub cells: Vec<Vec<EtCell>>,
    /// Number of cells marked as zero-divisor pairs.
    pub zd_count: usize,
    /// Total number of cells.
    pub total_cells: usize,
}

/// Summary of sand mandala sparsity analysis.
#[derive(Debug, Clone)]
pub struct MandalaSummary {
    /// Dimension of the algebra.
    pub dim: usize,
    /// Number of cross-assessor cells that ARE zero-divisor pairs.
    pub filled: usize,
    /// Total number of cross-assessor cells.
    pub total_cross: usize,
    /// Sparsity ratio: filled / total_cross.
    pub fill_ratio: f64,
    /// Row fill ratios (fraction of ZD cells per row, for cross-assessor rows).
    pub row_fill_ratios: Vec<f64>,
    /// Column fill ratios.
    pub col_fill_ratios: Vec<f64>,
}

/// Scaling data for ET period-doubling analysis.
#[derive(Debug, Clone)]
pub struct EtScaling {
    /// Cayley-Dickson dimension.
    pub dim: usize,
    /// Total cells in the ET (size^2).
    pub total_cells: usize,
    /// Number of ZD-marked cells.
    pub zd_cells: usize,
    /// Number of cross-assessor pairs.
    pub n_cross_pairs: usize,
    /// Number of cross-assessor pairs with at least one diagonal zero-product.
    pub n_zd_pairs: usize,
    /// ZD ratio: n_zd_pairs / n_cross_pairs.
    pub zd_ratio: f64,
    /// Number of motif components.
    pub n_components: usize,
    /// Number of nodes per component (constant for all components at this dim).
    pub nodes_per_component: usize,
}

/// Cayley-Dickson generator (dim, g=dim/2) used by the strut/triad
/// identity G XOR S = X.
///
/// The identity S = G XOR X holds universally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CdGenerator {
    /// Algebra dimension.
    pub dim: usize,
    /// Generator index: dim/2.
    pub g: usize,
}

impl CdGenerator {
    pub fn new(dim: usize) -> Self {
        assert!(dim >= 4 && dim.is_power_of_two());
        Self { dim, g: dim / 2 }
    }

    /// Verify the triad identity G XOR S = X for a given strut constant.
    pub fn verify_triad(&self, s: usize) -> bool {
        let x = self.g ^ s;
        x != 0 && x != self.g && x != s
    }

    /// All valid strut constants for this dimension's box-kites.
    /// At dim=16: S in {1..7} (the 7 strut signatures).
    pub fn valid_struts(&self) -> Vec<usize> {
        (1..self.g).filter(|&s| self.verify_triad(s)).collect()
    }
}
