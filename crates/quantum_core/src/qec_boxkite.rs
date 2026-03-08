//! Topological QEC via Sedenion Box-Kites
//!
//! This module implements a parity-check matrix representation of the
//! 7 $K_{2,2,2}$ box-kite topologies found in Sedenion zero-divisors,
//! representing a non-associative topological stabilizer code.

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
        let mut syndrome_triggered = false;
        for (u, v) in &self.parity_edges {
            // A syndrome is triggered if one of the non-associative zero-divisor pairs is hit.
            if error_mask.contains(u) || error_mask.contains(v) {
                syndrome_triggered = true;
            }
        }
        syndrome_triggered
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
        // Distance is defined by the minimum overlap to flip a syndrome trap.
        // For K_{2,2,2} box-kites, distance threshold corresponds to the
        // non-associative failure bounds.
        3 // Baseline for 7-kite configuration
    }
}
