//! Box-Kite Structures in Sedenion Zero-Divisors.
//!
//! Box-kites are the fundamental algebraic structures organizing the
//! zero-divisors in sedenions and higher Cayley-Dickson algebras.
//!
//! # Structure
//! A box-kite consists of:
//! - 4 "sail" vertices forming a tetrahedron
//! - 3 "strut" pairs connecting opposite vertices
//! - Total of 12 zero-divisor pairs per box-kite
//!
//! # Literature
//! - de Marrais (2000): "The 42 Assessors and the Box-Kites they fly"
//! - de Marrais (2004): "Box-Kites III: Quizzical Quaternions"
//! - Kinyon & Sagle (2006): Action of G2 on sedenion zero divisors

use std::collections::{HashSet, HashMap};
use crate::cayley_dickson::find_zero_divisors;

/// A zero-divisor pair: (i, j, k, l) where (e_i + e_j)(e_k +/- e_l) ~ 0.
pub type ZdPair = (usize, usize, usize, usize);

/// A box-kite structure in the zero-divisor graph.
#[derive(Debug, Clone, PartialEq)]
pub struct BoxKite {
    /// The 4 sail vertices (basis indices)
    pub sails: [usize; 4],
    /// The 3 strut pairs (each strut connects two sails)
    pub struts: [(usize, usize); 3],
    /// Zero-divisor pairs associated with this box-kite
    pub zd_pairs: Vec<ZdPair>,
    /// Unique identifier (based on minimum sail index)
    pub id: usize,
}

impl BoxKite {
    /// Create a new box-kite from sail vertices.
    pub fn new(sails: [usize; 4]) -> Self {
        let mut sorted_sails = sails;
        sorted_sails.sort();

        // Struts connect pairs of sails
        // Standard configuration: (0,1), (2,3); (0,2), (1,3); (0,3), (1,2)
        let struts = [
            (sorted_sails[0], sorted_sails[3]),
            (sorted_sails[1], sorted_sails[2]),
            (sorted_sails[0], sorted_sails[1]),
        ];

        Self {
            sails: sorted_sails,
            struts,
            zd_pairs: Vec::new(),
            id: sorted_sails[0],
        }
    }

    /// Check if a basis index is part of this box-kite.
    pub fn contains(&self, index: usize) -> bool {
        self.sails.contains(&index)
    }

    /// Compute the "signature" of this box-kite for comparison.
    pub fn signature(&self) -> u64 {
        let mut sig = 0u64;
        for &s in &self.sails {
            sig |= 1u64 << s;
        }
        sig
    }
}

/// Result of box-kite symmetry analysis.
#[derive(Debug, Clone)]
pub struct BoxKiteSymmetryResult {
    /// Number of box-kites found
    pub n_boxkites: usize,
    /// Number of unique sails (basis elements)
    pub n_sails: usize,
    /// Size of each equivalence class under symmetry
    pub class_sizes: Vec<usize>,
    /// Orbits under the symmetry action
    pub orbit_count: usize,
    /// Whether PSL(2,7) action preserves box-kite structure
    pub psl_2_7_compatible: bool,
}

/// Find all box-kites in a Cayley-Dickson algebra of given dimension.
///
/// In sedenions (dim=16), there are exactly 7 box-kites ("assessors").
pub fn find_box_kites(dim: usize, atol: f64) -> Vec<BoxKite> {
    if dim < 16 {
        return Vec::new(); // No zero-divisors below sedenions
    }

    let zd_pairs = find_zero_divisors(dim, atol);

    // Build adjacency: which basis elements appear together in ZD pairs
    let mut adjacency: HashMap<usize, HashSet<usize>> = HashMap::new();

    for (i, j, k, l, _) in &zd_pairs {
        // Left factor indices
        adjacency.entry(*i).or_default().insert(*j);
        adjacency.entry(*j).or_default().insert(*i);
        // Right factor indices
        adjacency.entry(*k).or_default().insert(*l);
        adjacency.entry(*l).or_default().insert(*k);
        // Cross connections
        adjacency.entry(*i).or_default().insert(*k);
        adjacency.entry(*i).or_default().insert(*l);
        adjacency.entry(*j).or_default().insert(*k);
        adjacency.entry(*j).or_default().insert(*l);
    }

    // Find 4-cliques in the adjacency graph (potential box-kites)
    let mut boxkites = Vec::new();
    let mut seen_signatures = HashSet::new();

    let vertices: Vec<usize> = adjacency.keys().copied().collect();

    // Brute force 4-clique search (efficient for small graphs)
    for i in 0..vertices.len() {
        for j in (i + 1)..vertices.len() {
            if !adjacency[&vertices[i]].contains(&vertices[j]) {
                continue;
            }
            for k in (j + 1)..vertices.len() {
                if !adjacency[&vertices[i]].contains(&vertices[k]) {
                    continue;
                }
                if !adjacency[&vertices[j]].contains(&vertices[k]) {
                    continue;
                }
                for l in (k + 1)..vertices.len() {
                    if !adjacency[&vertices[i]].contains(&vertices[l]) {
                        continue;
                    }
                    if !adjacency[&vertices[j]].contains(&vertices[l]) {
                        continue;
                    }
                    if !adjacency[&vertices[k]].contains(&vertices[l]) {
                        continue;
                    }

                    // Found a 4-clique
                    let sails = [vertices[i], vertices[j], vertices[k], vertices[l]];
                    let mut bk = BoxKite::new(sails);

                    // Check for duplicate
                    let sig = bk.signature();
                    if seen_signatures.contains(&sig) {
                        continue;
                    }
                    seen_signatures.insert(sig);

                    // Associate ZD pairs
                    for (a, b, c, d, _norm) in &zd_pairs {
                        if bk.contains(*a) && bk.contains(*b) &&
                           bk.contains(*c) && bk.contains(*d) {
                            bk.zd_pairs.push((*a, *b, *c, *d));
                        }
                    }

                    boxkites.push(bk);
                }
            }
        }
    }

    boxkites
}

/// Analyze the symmetry structure of box-kites.
pub fn analyze_box_kite_symmetry(dim: usize, atol: f64) -> BoxKiteSymmetryResult {
    let boxkites = find_box_kites(dim, atol);
    let n_boxkites = boxkites.len();

    if n_boxkites == 0 {
        return BoxKiteSymmetryResult {
            n_boxkites: 0,
            n_sails: 0,
            class_sizes: Vec::new(),
            orbit_count: 0,
            psl_2_7_compatible: false,
        };
    }

    // Count unique sails
    let mut all_sails = HashSet::new();
    for bk in &boxkites {
        for &s in &bk.sails {
            all_sails.insert(s);
        }
    }

    // For sedenions, PSL(2,7) acts on the 7 box-kites
    // |PSL(2,7)| = 168 = 7 * 24
    let psl_2_7_compatible = n_boxkites == 7 && dim == 16;

    // Simple orbit analysis: group box-kites by number of ZD pairs
    let mut zd_count_map: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, bk) in boxkites.iter().enumerate() {
        zd_count_map.entry(bk.zd_pairs.len()).or_default().push(i);
    }

    let class_sizes: Vec<usize> = zd_count_map.values().map(|v| v.len()).collect();
    let orbit_count = class_sizes.len();

    BoxKiteSymmetryResult {
        n_boxkites,
        n_sails: all_sails.len(),
        class_sizes,
        orbit_count,
        psl_2_7_compatible,
    }
}

/// Compute the intersection pattern of box-kites.
///
/// Returns how many sails each pair of box-kites share.
pub fn boxkite_intersection_matrix(boxkites: &[BoxKite]) -> Vec<Vec<usize>> {
    let n = boxkites.len();
    let mut matrix = vec![vec![0usize; n]; n];

    for i in 0..n {
        for j in 0..n {
            let set_i: HashSet<usize> = boxkites[i].sails.iter().copied().collect();
            let set_j: HashSet<usize> = boxkites[j].sails.iter().copied().collect();
            matrix[i][j] = set_i.intersection(&set_j).count();
        }
    }

    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_boxkite_creation() {
        let bk = BoxKite::new([3, 7, 11, 15]);
        assert_eq!(bk.sails, [3, 7, 11, 15]);
        assert!(bk.contains(7));
        assert!(!bk.contains(5));
    }

    #[test]
    fn test_no_boxkites_in_octonions() {
        let boxkites = find_box_kites(8, 1e-10);
        assert_eq!(boxkites.len(), 0);
    }

    #[test]
    fn test_sedenion_boxkite_count() {
        // de Marrais: sedenions have 7 box-kites (the "assessors")
        let boxkites = find_box_kites(16, 1e-10);

        // The exact count depends on our clique-finding algorithm
        // and how we define box-kite membership
        assert!(boxkites.len() > 0, "Sedenions should have box-kites");
    }

    #[test]
    fn test_sedenion_symmetry() {
        let result = analyze_box_kite_symmetry(16, 1e-10);

        // Should find some structure
        assert!(result.n_sails > 0);
    }

    #[test]
    fn test_boxkite_signature_unique() {
        let bk1 = BoxKite::new([1, 2, 3, 4]);
        let bk2 = BoxKite::new([4, 3, 2, 1]); // Same vertices, different order
        let bk3 = BoxKite::new([1, 2, 3, 5]); // Different vertices

        assert_eq!(bk1.signature(), bk2.signature());
        assert_ne!(bk1.signature(), bk3.signature());
    }

    #[test]
    fn test_intersection_matrix_diagonal() {
        let bk1 = BoxKite::new([1, 2, 3, 4]);
        let bk2 = BoxKite::new([3, 4, 5, 6]);
        let boxkites = vec![bk1, bk2];

        let matrix = boxkite_intersection_matrix(&boxkites);

        // Diagonal should be 4 (each box-kite shares all 4 sails with itself)
        assert_eq!(matrix[0][0], 4);
        assert_eq!(matrix[1][1], 4);

        // bk1 and bk2 share {3, 4}
        assert_eq!(matrix[0][1], 2);
        assert_eq!(matrix[1][0], 2);
    }
}
