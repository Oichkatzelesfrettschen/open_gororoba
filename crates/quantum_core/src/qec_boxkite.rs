//! Topological QEC via Sedenion Box-Kites
//!
//! This module implements a parity-check matrix representation of the
//! 7 $K_{2,2,2}$ box-kite topologies found in Sedenion zero-divisors,
//! representing a non-associative topological stabilizer code.

use crate::stabilizer_like::{CompositeCode, StabilizerLikeCode};
use std::collections::HashSet;

/// Represents a topological stabilizer built from a Sedenion Box-Kite.
#[derive(Debug, Clone)]
pub struct BoxKiteStabilizer {
    /// The index of the box kite (1 through 7)
    pub id: usize,
    /// The 6 nodes in the K_{2,2,2} graph representing the stabilizer
    pub nodes: [usize; 6],
    /// The 12 edges (zero-divisor pairs) that form the parity check
    pub parity_edges: Vec<(usize, usize)>,
}

impl BoxKiteStabilizer {
    pub fn new(id: usize, nodes: [usize; 6], edges: &[(usize, usize)]) -> Self {
        let parity_edges = edges
            .iter()
            .filter(|(u, v)| nodes.contains(u) && nodes.contains(v))
            .copied()
            .collect();

        Self {
            id,
            nodes,
            parity_edges,
        }
    }

    /// Simulates syndrome detection.
    /// In standard QEC, [S_i, S_j] = 0. Here, alternativity failure acts as a trap.
    pub fn check_syndrome(&self, error_mask: &HashSet<usize>) -> bool {
        self.parity_edges
            .iter()
            .any(|(u, v)| error_mask.contains(u) || error_mask.contains(v))
    }
}

impl StabilizerLikeCode for BoxKiteStabilizer {
    fn covered_indices(&self) -> &[usize] {
        &self.nodes
    }

    fn syndrome_strength(&self, error_mask: &HashSet<usize>) -> f64 {
        if self.check_syndrome(error_mask) {
            1.0
        } else {
            0.0
        }
    }

    fn code_distance(&self) -> usize {
        // K_{2,2,2} box-kites have distance 3 (minimum weight to flip syndrome)
        3
    }

    fn num_checks(&self) -> usize {
        self.parity_edges.len()
    }

    fn algebra_dimension(&self) -> usize {
        16 // Sedenion
    }
}

/// The full non-associative topological stabilizer code.
#[derive(Debug, Clone)]
pub struct SedenionQecCode {
    pub stabilizers: Vec<BoxKiteStabilizer>,
}

impl SedenionQecCode {
    pub fn new(stabilizers: Vec<BoxKiteStabilizer>) -> Self {
        Self { stabilizers }
    }

    /// Calculate distance metric for the non-associative code.
    pub fn compute_distance(&self) -> usize {
        self.effective_distance()
    }
}

impl CompositeCode for SedenionQecCode {
    fn num_stabilizers(&self) -> usize {
        self.stabilizers.len()
    }

    fn global_stability(&self, error_mask: &HashSet<usize>) -> f64 {
        if self.stabilizers.is_empty() {
            return 1.0;
        }
        let triggered: usize = self
            .stabilizers
            .iter()
            .filter(|s| s.syndrome_strength(error_mask) > 0.0)
            .count();
        1.0 - (triggered as f64 / self.stabilizers.len() as f64)
    }

    fn effective_distance(&self) -> usize {
        self.stabilizers
            .iter()
            .map(|s| s.code_distance())
            .min()
            .unwrap_or(0)
    }
}
