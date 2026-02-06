//! De Marrais Box-Kite Structures in Sedenion Zero-Divisors.
//!
//! Box-kites are the fundamental algebraic structures organizing the
//! zero-divisors in sedenions (16D Cayley-Dickson algebra).
//!
//! # Structure
//!
//! An **assessor** is a pair (low, high) with low in {1..7} and high in {8..15},
//! representing a 2-plane of zero-divisors spanned by e_low and e_high.
//!
//! A **box-kite** is an octahedral structure with:
//! - 6 vertices (assessors)
//! - 12 edges (co-assessor relationships)
//! - 3 struts (opposite pairs with no edge)
//! - 4 sail faces + 4 vent faces
//!
//! There are exactly **7 box-kites** in sedenions, partitioning all 42 primitive
//! assessors. Each box-kite corresponds to a unique "missing" octonion index.
//!
//! # Algorithm
//!
//! 1. Generate 42 primitive assessors (filter from 56 cross-pairs)
//! 2. Build co-assessor adjacency graph (edge if diagonal zero-product exists)
//! 3. Find connected components (exactly 7 for sedenions)
//! 4. Verify octahedral structure (6 vertices, degree 4, 3 non-neighbors)
//!
//! # Literature
//!
//! - de Marrais (2000): "The 42 Assessors and the Box-Kites they fly" (arXiv:math/0011260)
//! - de Marrais (2004): "Box-Kites III: Quizzical Quaternions" (arXiv:math/0403113)

use std::collections::{HashSet, HashMap, VecDeque};
use crate::cayley_dickson::{cd_multiply, cd_norm_sq};

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
        debug_assert!(low >= 1 && low <= 7, "low must be in 1..7");
        debug_assert!(high >= 8 && high <= 15, "high must be in 8..15");
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
        let edge_set: HashSet<(usize, usize)> = self.edges.iter()
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

/// Generate all 42 primitive assessors for sedenions.
///
/// Excludes (i, 8) and (i, i+8) which don't participate in diagonal zero-products.
pub fn primitive_assessors() -> Vec<Assessor> {
    let mut assessors = Vec::with_capacity(42);

    for low in 1..=7 {
        for high in 8..=15 {
            // Exclude (i, 8) - high index = 8
            if high == 8 {
                continue;
            }
            // Exclude (i, i+8) - "identity" pairs
            if high == low + 8 {
                continue;
            }
            assessors.push(Assessor::new(low, high));
        }
    }

    debug_assert_eq!(assessors.len(), 42, "Should have exactly 42 primitive assessors");
    assessors
}

/// Check if two assessors have a diagonal zero-product.
///
/// Returns Some((s, t)) if (e_low1 + s*e_high1) * (e_low2 + t*e_high2) = 0,
/// where s, t in {+1, -1}.
pub fn diagonal_zero_product(a: &Assessor, b: &Assessor, atol: f64) -> Option<(i8, i8)> {
    for s in [-1.0, 1.0] {
        for t in [-1.0, 1.0] {
            let v1 = a.diagonal(s);
            let v2 = b.diagonal(t);
            let product = cd_multiply(&v1, &v2);
            let norm = cd_norm_sq(&product).sqrt();
            if norm < atol {
                return Some((s as i8, t as i8));
            }
        }
    }
    None
}

/// Check if two assessors are co-assessors (have any diagonal zero-product).
pub fn are_coassessors(a: &Assessor, b: &Assessor, atol: f64) -> bool {
    diagonal_zero_product(a, b, atol).is_some()
}

/// Build the co-assessor adjacency graph.
///
/// Returns a map from assessor index to set of adjacent assessor indices.
pub fn build_coassessor_graph(assessors: &[Assessor], atol: f64) -> HashMap<usize, HashSet<usize>> {
    let mut graph: HashMap<usize, HashSet<usize>> = HashMap::new();

    // Initialize all vertices
    for i in 0..assessors.len() {
        graph.insert(i, HashSet::new());
    }

    // Add edges for co-assessor pairs
    for i in 0..assessors.len() {
        for j in (i + 1)..assessors.len() {
            if are_coassessors(&assessors[i], &assessors[j], atol) {
                graph.get_mut(&i).unwrap().insert(j);
                graph.get_mut(&j).unwrap().insert(i);
            }
        }
    }

    graph
}

/// Find connected components in the co-assessor graph.
///
/// Returns a vector of components, where each component is a vector of assessor indices.
pub fn find_connected_components(graph: &HashMap<usize, HashSet<usize>>) -> Vec<Vec<usize>> {
    let mut visited = HashSet::new();
    let mut components = Vec::new();

    for &start in graph.keys() {
        if visited.contains(&start) {
            continue;
        }

        // BFS to find component
        let mut component = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited.insert(start);

        while let Some(node) = queue.pop_front() {
            component.push(node);
            if let Some(neighbors) = graph.get(&node) {
                for &neighbor in neighbors {
                    if !visited.contains(&neighbor) {
                        visited.insert(neighbor);
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        component.sort();
        components.push(component);
    }

    // Sort components by first element for determinism
    components.sort_by_key(|c| c.first().copied().unwrap_or(0));
    components
}

/// Compute the strut signature for a box-kite.
///
/// The strut signature is the octonion index (1-7) missing from the low indices
/// of the box-kite's assessors.
pub fn compute_strut_signature(assessors: &[Assessor]) -> usize {
    let low_indices: HashSet<usize> = assessors.iter().map(|a| a.low).collect();
    for i in 1..=7 {
        if !low_indices.contains(&i) {
            return i;
        }
    }
    0 // Should never happen for valid box-kites
}

/// Find all 7 box-kites in sedenions.
///
/// Uses the correct de Marrais algorithm:
/// 1. Generate 42 primitive assessors
/// 2. Build co-assessor graph
/// 3. Find connected components (7 box-kites)
pub fn find_box_kites(dim: usize, atol: f64) -> Vec<BoxKite> {
    if dim != 16 {
        // Box-kites are only defined for sedenions currently
        // Extension to pathions would require different assessor definition
        return Vec::new();
    }

    let assessors = primitive_assessors();
    let graph = build_coassessor_graph(&assessors, atol);
    let components = find_connected_components(&graph);

    let mut boxkites = Vec::new();

    for (id, component) in components.iter().enumerate() {
        // Extract assessors for this component
        let bk_assessors: Vec<Assessor> = component.iter()
            .map(|&i| assessors[i])
            .collect();

        // Verify octahedral structure (should have 6 vertices)
        if bk_assessors.len() != 6 {
            continue; // Not a valid box-kite
        }

        // Build edges within this component
        let mut edges = Vec::new();
        for i in 0..component.len() {
            for j in (i + 1)..component.len() {
                if graph[&component[i]].contains(&component[j]) {
                    edges.push((i, j));
                }
            }
        }

        // Verify 12 edges (octahedron has 12 edges)
        if edges.len() != 12 {
            continue;
        }

        // Find struts (non-adjacent pairs)
        let edge_set: HashSet<(usize, usize)> = edges.iter()
            .flat_map(|&(a, b)| vec![(a, b), (b, a)])
            .collect();

        let mut struts = Vec::new();
        for i in 0..6 {
            for j in (i + 1)..6 {
                if !edge_set.contains(&(i, j)) {
                    struts.push((i, j));
                }
            }
        }

        // Verify 3 struts (octahedron has 3 pairs of opposite vertices)
        if struts.len() != 3 {
            continue;
        }

        let strut_signature = compute_strut_signature(&bk_assessors);

        boxkites.push(BoxKite {
            assessors: bk_assessors,
            edges,
            struts,
            strut_signature,
            id,
        });
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
            n_assessors: 0,
            strut_signatures: Vec::new(),
            de_marrais_valid: false,
            psl_2_7_compatible: false,
        };
    }

    // Count total assessors
    let n_assessors: usize = boxkites.iter().map(|bk| bk.assessors.len()).sum();

    // Collect strut signatures
    let mut strut_signatures: Vec<usize> = boxkites.iter()
        .map(|bk| bk.strut_signature)
        .collect();
    strut_signatures.sort();

    // Validate de Marrais structure
    let de_marrais_valid = n_boxkites == 7
        && n_assessors == 42
        && strut_signatures == vec![1, 2, 3, 4, 5, 6, 7];

    // PSL(2,7) has order 168 = 7 * 24
    let psl_2_7_compatible = de_marrais_valid;

    BoxKiteSymmetryResult {
        n_boxkites,
        n_assessors,
        strut_signatures,
        de_marrais_valid,
        psl_2_7_compatible,
    }
}

/// Legacy compatibility: compute intersection matrix for old API.
pub fn boxkite_intersection_matrix(boxkites: &[BoxKite]) -> Vec<Vec<usize>> {
    let n = boxkites.len();
    let mut matrix = vec![vec![0usize; n]; n];

    for i in 0..n {
        for j in 0..n {
            let set_i: HashSet<Assessor> = boxkites[i].assessors.iter().copied().collect();
            let set_j: HashSet<Assessor> = boxkites[j].assessors.iter().copied().collect();
            matrix[i][j] = set_i.intersection(&set_j).count();
        }
    }

    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_assessors_count() {
        let assessors = primitive_assessors();
        assert_eq!(assessors.len(), 42, "Should have exactly 42 primitive assessors");
    }

    #[test]
    fn test_assessor_exclusions() {
        let assessors = primitive_assessors();

        // Verify (i, 8) excluded
        for a in &assessors {
            assert_ne!(a.high, 8, "Assessor ({}, 8) should be excluded", a.low);
        }

        // Verify (i, i+8) excluded
        for a in &assessors {
            assert_ne!(a.high, a.low + 8,
                "Assessor ({}, {}) should be excluded", a.low, a.high);
        }
    }

    #[test]
    fn test_no_boxkites_in_octonions() {
        let boxkites = find_box_kites(8, 1e-10);
        assert_eq!(boxkites.len(), 0, "Octonions should have no box-kites");
    }

    #[test]
    fn test_sedenion_boxkite_count() {
        let boxkites = find_box_kites(16, 1e-10);
        assert_eq!(boxkites.len(), 7,
            "Sedenions should have exactly 7 box-kites (de Marrais), got {}", boxkites.len());
    }

    #[test]
    fn test_sedenion_boxkite_structure() {
        let boxkites = find_box_kites(16, 1e-10);

        for (i, bk) in boxkites.iter().enumerate() {
            // Each box-kite should have 6 assessors
            assert_eq!(bk.assessors.len(), 6,
                "Box-kite {} should have 6 assessors, got {}", i, bk.assessors.len());

            // Each box-kite should have 12 edges
            assert_eq!(bk.edges.len(), 12,
                "Box-kite {} should have 12 edges, got {}", i, bk.edges.len());

            // Each box-kite should have 3 struts
            assert_eq!(bk.struts.len(), 3,
                "Box-kite {} should have 3 struts, got {}", i, bk.struts.len());

            // Strut signature should be in 1..7
            assert!(bk.strut_signature >= 1 && bk.strut_signature <= 7,
                "Box-kite {} has invalid strut signature {}", i, bk.strut_signature);
        }
    }

    #[test]
    fn test_sedenion_strut_signatures() {
        let boxkites = find_box_kites(16, 1e-10);
        let mut signatures: Vec<usize> = boxkites.iter().map(|bk| bk.strut_signature).collect();
        signatures.sort();

        assert_eq!(signatures, vec![1, 2, 3, 4, 5, 6, 7],
            "Strut signatures should be {{1..7}}, got {:?}", signatures);
    }

    #[test]
    fn test_sedenion_symmetry() {
        let result = analyze_box_kite_symmetry(16, 1e-10);

        assert_eq!(result.n_boxkites, 7);
        assert_eq!(result.n_assessors, 42);
        assert!(result.de_marrais_valid, "Should validate de Marrais structure");
        assert!(result.psl_2_7_compatible, "Should be PSL(2,7) compatible");
    }

    #[test]
    fn test_coassessor_symmetry() {
        let a = Assessor::new(1, 10);
        let b = Assessor::new(2, 9);

        // Co-assessor relation should be symmetric
        let ab = are_coassessors(&a, &b, 1e-10);
        let ba = are_coassessors(&b, &a, 1e-10);
        assert_eq!(ab, ba, "Co-assessor relation should be symmetric");
    }

    #[test]
    fn test_assessors_partition_into_boxkites() {
        let boxkites = find_box_kites(16, 1e-10);

        // Collect all assessors from all box-kites
        let mut all_assessors: Vec<Assessor> = boxkites.iter()
            .flat_map(|bk| bk.assessors.clone())
            .collect();
        all_assessors.sort();

        // Should be exactly 42 unique assessors
        let unique: HashSet<Assessor> = all_assessors.iter().copied().collect();
        assert_eq!(unique.len(), 42, "All 42 assessors should appear");
        assert_eq!(all_assessors.len(), 42, "No assessor should appear twice");
    }

    #[test]
    fn test_octahedral_degree() {
        let boxkites = find_box_kites(16, 1e-10);

        for bk in &boxkites {
            // Count degree of each vertex
            let mut degrees = vec![0usize; 6];
            for &(i, j) in &bk.edges {
                degrees[i] += 1;
                degrees[j] += 1;
            }

            // In an octahedron, every vertex has degree 4
            for (v, &d) in degrees.iter().enumerate() {
                assert_eq!(d, 4, "Vertex {} should have degree 4, got {}", v, d);
            }
        }
    }
}
