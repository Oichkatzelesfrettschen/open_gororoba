//! Core types for the de Marrais box-kite structure.
//!
//! Defines the `Assessor` (low, high) pair (`(1..=7, 8..=15)`), the
//! `BoxKite` octahedral container, and the `BoxKiteSymmetryResult`
//! diagnostic record returned by symmetry analysis. The algorithms that
//! build, classify, and traverse box-kites live in sibling files at the
//! `algebra_analysis::boxkites` root and access these types through
//! re-exports from the parent module.

use std::collections::HashSet;

/// An assessor: pair (low, high) with low in 1..7, high in 8..15.
/// Represents a 2-plane of zero-divisors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Assessor {
    /// Low index (1-7, imaginary octonion unit)
    pub low: usize,
    /// High index (8-15, sedenion imaginary unit)
    pub high: usize,
}

impl Assessor {
    /// Create a new assessor.
    pub fn new(low: usize, high: usize) -> Self {
        debug_assert!((1..=7).contains(&low), "low must be in 1..7");
        debug_assert!((8..=15).contains(&high), "high must be in 8..15");
        Self { low, high }
    }

    /// Create diagonal zero-divisor: e_low + sign * e_high (normalized).
    pub fn diagonal(&self, sign: f64) -> Vec<f64> {
        let mut v = vec![0.0; 16];
        let norm = 2.0_f64.sqrt();
        v[self.low] = 1.0 / norm;
        v[self.high] = sign / norm;
        v
    }

    /// Unique identifier for this assessor.
    pub fn id(&self) -> usize {
        (self.low - 1) * 8 + (self.high - 8)
    }
}

/// A box-kite structure: octahedron of 6 assessors.
#[derive(Debug, Clone)]
pub struct BoxKite {
    /// The 6 assessor vertices
    pub assessors: Vec<Assessor>,
    /// Adjacency within this box-kite (indices into assessors)
    pub edges: Vec<(usize, usize)>,
    /// The 3 strut pairs (opposite vertices with no edge)
    pub struts: Vec<(usize, usize)>,
    /// Strut signature: the missing octonion index (1-7)
    pub strut_signature: usize,
    /// Unique identifier
    pub id: usize,
}

impl BoxKite {
    /// Check if this box-kite contains an assessor.
    pub fn contains(&self, a: &Assessor) -> bool {
        self.assessors.contains(a)
    }

    /// Get the sail triangles (faces with all "-" sign edges).
    pub fn sails(&self) -> Vec<[usize; 3]> {
        // In a proper octahedron, sails are the 4 faces where all 3 edges
        // have the same parity. Implementation depends on edge classification.
        // For now, return triangular faces.
        let mut faces = Vec::new();
        let n = self.assessors.len();
        if n != 6 {
            return faces;
        }

        // Build adjacency set for quick lookup
        let edge_set: HashSet<(usize, usize)> = self
            .edges
            .iter()
            .flat_map(|&(a, b)| vec![(a, b), (b, a)])
            .collect();

        // Find all triangles
        for i in 0..n {
            for j in (i + 1)..n {
                if !edge_set.contains(&(i, j)) {
                    continue;
                }
                for k in (j + 1)..n {
                    if edge_set.contains(&(i, k)) && edge_set.contains(&(j, k)) {
                        faces.push([i, j, k]);
                    }
                }
            }
        }
        faces
    }
}

/// Result of box-kite symmetry analysis.
#[derive(Debug, Clone)]
pub struct BoxKiteSymmetryResult {
    /// Number of box-kites found (should be 7 for sedenions)
    pub n_boxkites: usize,
    /// Total number of assessors (should be 42 for sedenions)
    pub n_assessors: usize,
    /// Strut signatures found (should be {1..7} for sedenions)
    pub strut_signatures: Vec<usize>,
    /// Whether the structure matches de Marrais exactly
    pub de_marrais_valid: bool,
    /// Whether PSL(2,7) symmetry is compatible
    pub psl_2_7_compatible: bool,
}
