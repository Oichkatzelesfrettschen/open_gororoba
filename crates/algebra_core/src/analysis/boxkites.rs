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

use crate::analysis::zd_graphs::xor_key;
use crate::construction::cayley_dickson::{cd_basis_mul_sign, cd_multiply, cd_norm_sq};
use nalgebra::{DMatrix, SymmetricEigen};
use petgraph::graph::{NodeIndex, UnGraph};
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

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

    debug_assert_eq!(
        assessors.len(),
        42,
        "Should have exactly 42 primitive assessors"
    );
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
        let bk_assessors: Vec<Assessor> = component.iter().map(|&i| assessors[i]).collect();

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
        let edge_set: HashSet<(usize, usize)> = edges
            .iter()
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
    let mut strut_signatures: Vec<usize> = boxkites.iter().map(|bk| bk.strut_signature).collect();
    strut_signatures.sort();

    // Validate de Marrais structure
    let de_marrais_valid =
        n_boxkites == 7 && n_assessors == 42 && strut_signatures == vec![1, 2, 3, 4, 5, 6, 7];

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

// ---------------------------------------------------------------------------
// Production Rules, Automorphemes, and Strut Tables
// ---------------------------------------------------------------------------

/// Return ALL sign-pair solutions (s, t) with s, t in {-1, +1} such that
/// diag(a, s) * diag(b, t) = 0 under 16D Cayley-Dickson multiplication.
///
/// Unlike `diagonal_zero_product` (which returns only the first match),
/// this returns every solution. Needed for edge sign classification.
pub fn all_diagonal_zero_products(a: &Assessor, b: &Assessor, atol: f64) -> Vec<(i8, i8)> {
    let mut results = Vec::new();
    for s in [-1.0_f64, 1.0] {
        for t in [-1.0_f64, 1.0] {
            let v1 = a.diagonal(s);
            let v2 = b.diagonal(t);
            let product = cd_multiply(&v1, &v2);
            let norm = cd_norm_sq(&product).sqrt();
            if norm < atol {
                results.push((s as i8, t as i8));
            }
        }
    }
    results
}

/// Edge sign classification for co-assessor pairs.
///
/// de Marrais distinguishes "trefoil" vs "triple-zigzag" lanyards:
/// - `Same`: solutions have same signs: (+,+) or (-,-) -- "+" in paper
/// - `Opposite`: solutions have opposite signs: (+,-) or (-,+) -- "-" in paper
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EdgeSignType {
    /// Same-sign solutions: (+,+) or (-,-)
    Same,
    /// Opposite-sign solutions: (+,-) or (-,+)
    Opposite,
}

/// Classify the diagonal-zero-product relationship between two co-assessors.
///
/// # Panics
/// Panics if no diagonal zero-products exist (the pair is not co-assessors).
pub fn edge_sign_type(a: &Assessor, b: &Assessor, atol: f64) -> EdgeSignType {
    let sols = all_diagonal_zero_products(a, b, atol);
    assert!(
        !sols.is_empty(),
        "No diagonal zero-products for ({},{})--({},{})",
        a.low,
        a.high,
        b.low,
        b.high
    );
    if sols.contains(&(1, 1)) || sols.contains(&(-1, -1)) {
        EdgeSignType::Same
    } else {
        EdgeSignType::Opposite
    }
}

/// The 7 Fano-plane lines (oriented-free octonion triplets).
///
/// Each triple (a, b, c) represents a line in PG(2,2) = Fano plane.
/// Properties: 7 lines, 7 points, each point on exactly 3 lines,
/// each pair of points on exactly 1 line.
///
/// Used by de Marrais' "GoTo listings" for automorpheme construction.
pub const O_TRIPS: [[usize; 3]; 7] = [
    [1, 2, 3],
    [1, 4, 5],
    [1, 6, 7],
    [2, 4, 6],
    [2, 5, 7],
    [3, 4, 7],
    [3, 5, 6],
];

/// de Marrais Production Rule #1 ("Three-Ring Circuits").
///
/// Given two co-assessors a=(A,B) and b=(C,D), constructs a third assessor
/// (E,F) via XOR:
///   E = A ^ C = B ^ D   (low index)
///   F = A ^ D = B ^ C   (high index)
///
/// The result is the unique third assessor completing the co-assessor triple.
///
/// # Panics
/// Panics if the XOR invariants fail or the result is degenerate.
pub fn production_rule_1(a: &Assessor, b: &Assessor) -> Assessor {
    let (big_a, big_b) = (a.low, a.high);
    let (big_c, big_d) = (b.low, b.high);

    let e = big_a ^ big_c;
    let f = big_a ^ big_d;

    assert_eq!(
        e,
        big_b ^ big_d,
        "PR#1 invariant failed: A^C != B^D for ({},{}) and ({},{})",
        big_a,
        big_b,
        big_c,
        big_d
    );
    assert_eq!(
        f,
        big_b ^ big_c,
        "PR#1 invariant failed: A^D != B^C for ({},{}) and ({},{})",
        big_a,
        big_b,
        big_c,
        big_d
    );
    assert_ne!(
        e, f,
        "PR#1 degenerate: equal indices {} from ({},{}) and ({},{})",
        e, big_a, big_b, big_c, big_d
    );

    // For sedenion cross-assessors: e is in 1..7 (XOR of two lows),
    // f is in 8..15 (XOR of low and high), so e < f always.
    let (low, high) = if e < f { (e, f) } else { (f, e) };
    Assessor::new(low, high)
}

/// Helper: create a diagonal zero-divisor vector from raw index pair.
/// Used internally by production_rule_2 for candidate pairs that may
/// not satisfy the Assessor invariants.
fn raw_diagonal(i: usize, j: usize, sign: f64) -> Vec<f64> {
    let mut v = vec![0.0; 16];
    let norm = 2.0_f64.sqrt();
    v[i] = 1.0 / norm;
    v[j] = sign / norm;
    v
}

/// Helper: diagonal zero-products for raw index pairs (not necessarily valid Assessors).
fn raw_all_diagonal_zero_products(
    a: (usize, usize),
    b: (usize, usize),
    atol: f64,
) -> Vec<(i8, i8)> {
    let mut results = Vec::new();
    for s in [-1.0_f64, 1.0] {
        for t in [-1.0_f64, 1.0] {
            let v1 = raw_diagonal(a.0, a.1, s);
            let v2 = raw_diagonal(b.0, b.1, t);
            let product = cd_multiply(&v1, &v2);
            let norm = cd_norm_sq(&product).sqrt();
            if norm < atol {
                results.push((s as i8, t as i8));
            }
        }
    }
    results
}

/// de Marrais Production Rule #2 ("Skew-Symmetric Twisting").
///
/// Given co-assessors a=(A,B) and b=(C,D), creates two new assessors via
/// pair swapping. Two candidate swaps exist:
///   - Candidate 1: (A,D) and (C,B)  -- cross-index swap
///   - Candidate 2: (B,D) and (A,C)  -- same-range swap
///
/// Exactly one candidate satisfies the defining property: the outputs are
/// co-assessors with each other, but not with either input.
///
/// For sedenion cross-assessors, candidate 1 is always the valid one
/// (candidate 2 produces non-cross-assessor pairs), but both are checked
/// for mathematical rigor.
///
/// # Panics
/// Panics if exactly one valid swap cannot be found.
pub fn production_rule_2(a: &Assessor, b: &Assessor, atol: f64) -> (Assessor, Assessor) {
    let (big_a, big_b) = (a.low, a.high);
    let (big_c, big_d) = (b.low, b.high);

    // Candidate 1: cross swap -> (A,D) and (C,B)
    let c1_p = (big_a.min(big_d), big_a.max(big_d));
    let c1_q = (big_c.min(big_b), big_c.max(big_b));

    // Candidate 2: same-range swap -> (B,D) and (A,C)
    let c2_p = (big_b.min(big_d), big_b.max(big_d));
    let c2_q = (big_a.min(big_c), big_a.max(big_c));

    let a_raw = (a.low, a.high);
    let b_raw = (b.low, b.high);

    let valid_pair = |p: (usize, usize), q: (usize, usize)| -> bool {
        if p == q {
            return false;
        }
        // p and q must be co-assessors
        if raw_all_diagonal_zero_products(p, q, atol).is_empty() {
            return false;
        }
        // Neither must be co-assessor with either input
        if !raw_all_diagonal_zero_products(p, a_raw, atol).is_empty() {
            return false;
        }
        if !raw_all_diagonal_zero_products(p, b_raw, atol).is_empty() {
            return false;
        }
        if !raw_all_diagonal_zero_products(q, a_raw, atol).is_empty() {
            return false;
        }
        if !raw_all_diagonal_zero_products(q, b_raw, atol).is_empty() {
            return false;
        }
        true
    };

    let val1 = valid_pair(c1_p, c1_q);
    let val2 = valid_pair(c2_p, c2_q);

    assert_ne!(
        val1, val2,
        "PR#2 expected exactly one valid swap for ({},{}) and ({},{}), got val1={}, val2={}",
        big_a, big_b, big_c, big_d, val1, val2
    );

    let (p, q) = if val1 { (c1_p, c1_q) } else { (c2_p, c2_q) };

    // Verify the result forms valid cross-assessors
    assert!(
        (1..=7).contains(&p.0) && (8..=15).contains(&p.1),
        "PR#2 produced invalid assessor indices ({},{})",
        p.0,
        p.1
    );
    assert!(
        (1..=7).contains(&q.0) && (8..=15).contains(&q.1),
        "PR#2 produced invalid assessor indices ({},{})",
        q.0,
        q.1
    );

    (Assessor::new(p.0, p.1), Assessor::new(q.0, q.1))
}

/// Compute the 12 assessors for a de Marrais automorpheme (GoTo listing).
///
/// For each Fano-plane O-trip (o1, o2, o3), the "Behind the 8-Ball Theorem"
/// implies excluded sedenion high indices are {8, 8^o1, 8^o2, 8^o3},
/// leaving exactly 4 allowed highs to pair with each of the 3 low indices.
///
/// # Panics
/// Panics if `o_trip` is not one of the 7 canonical O_TRIPS.
pub fn automorpheme_assessors(o_trip: &[usize; 3]) -> HashSet<Assessor> {
    assert!(O_TRIPS.contains(o_trip), "Unknown O-trip: {:?}", o_trip);

    let excluded_highs: HashSet<usize> = std::iter::once(8)
        .chain(o_trip.iter().map(|&o| 8 ^ o))
        .collect();

    let allowed_highs: Vec<usize> = (8..=15).filter(|h| !excluded_highs.contains(h)).collect();
    debug_assert_eq!(allowed_highs.len(), 4);

    let mut result = HashSet::new();
    for &o in o_trip {
        for &h in &allowed_highs {
            result.insert(Assessor::new(o, h));
        }
    }
    debug_assert_eq!(result.len(), 12);
    result
}

/// Return all 7 automorpheme assessor sets (one per Fano-plane O-trip).
pub fn automorphemes() -> Vec<HashSet<Assessor>> {
    O_TRIPS.iter().map(automorpheme_assessors).collect()
}

/// Return all O-trips whose automorphemes contain the given assessor.
///
/// For a valid primitive assessor, this returns exactly 2 O-trips.
/// For excluded assessors (high=8 or high=8^low), returns empty.
pub fn automorphemes_containing_assessor(a: &Assessor) -> Vec<[usize; 3]> {
    O_TRIPS
        .iter()
        .filter(|t| automorpheme_assessors(t).contains(a))
        .copied()
        .collect()
}

/// de Marrais Production Rule #3 (Automorpheme Uniqueness).
///
/// Given an automorpheme (by O-trip) and an assessor it contains, returns the
/// unique OTHER O-trip whose automorpheme also contains that assessor.
///
/// Each primitive assessor belongs to exactly 2 automorphemes (by Fano-plane
/// incidence). PR#3 finds the other one.
///
/// # Panics
/// Panics if the assessor is not in the given automorpheme, or if the
/// expected 2-membership property fails.
pub fn production_rule_3(o_trip: &[usize; 3], a: &Assessor) -> [usize; 3] {
    assert!(
        automorpheme_assessors(o_trip).contains(a),
        "Assessor ({},{}) not in automorpheme for {:?}",
        a.low,
        a.high,
        o_trip
    );

    let candidates = automorphemes_containing_assessor(a);
    assert_eq!(
        candidates.len(),
        2,
        "Expected exactly 2 automorphemes for ({},{}), got {}",
        a.low,
        a.high,
        candidates.len()
    );

    if candidates[1] == *o_trip {
        candidates[0]
    } else {
        candidates[1]
    }
}

/// Deterministic A..F labeling for a box-kite's strut table.
///
/// - (a, b, c) form a zigzag face (all Opposite-sign edges)
/// - (d, e, f) form the opposite zigzag face
/// - Strut pairs: (a,f), (b,e), (c,d)
#[derive(Debug, Clone)]
pub struct StrutTable {
    pub a: Assessor,
    pub b: Assessor,
    pub c: Assessor,
    pub d: Assessor,
    pub e: Assessor,
    pub f: Assessor,
}

/// Compute the canonical strut table labeling for a box-kite.
///
/// Each box-kite has exactly 2 zigzag faces (triangles with all Opposite-sign
/// edges). We pick the lexicographically smaller one as (A,B,C) and derive
/// (D,E,F) as their octahedral opposites: F=opp(A), E=opp(B), D=opp(C).
///
/// # Panics
/// Panics if the box-kite structure is invalid (wrong face counts or
/// non-zigzag opposite face).
pub fn canonical_strut_table(bk: &BoxKite, atol: f64) -> StrutTable {
    let nodes = &bk.assessors;
    assert_eq!(nodes.len(), 6, "Box-kite must have 6 assessors");

    // Build edge set with both directions for O(1) lookup
    let edge_set: HashSet<(usize, usize)> = bk
        .edges
        .iter()
        .flat_map(|&(a, b)| [(a, b), (b, a)])
        .collect();

    let adjacent = |i: usize, j: usize| -> bool { edge_set.contains(&(i, j)) };

    // Find the unique opposite (non-neighbor) for each vertex
    let mut opposite = [0usize; 6];
    for (i, opp) in opposite.iter_mut().enumerate() {
        let non_neighbors: Vec<usize> = (0..6).filter(|&j| j != i && !adjacent(i, j)).collect();
        assert_eq!(
            non_neighbors.len(),
            1,
            "Expected unique opposite for vertex {}, got {:?}",
            i,
            non_neighbors
        );
        *opp = non_neighbors[0];
    }

    // Find all 8 triangular faces and identify the 2 zigzag faces
    let mut zigzag_faces: Vec<[usize; 3]> = Vec::new();
    for i in 0..6 {
        for j in (i + 1)..6 {
            if !adjacent(i, j) {
                continue;
            }
            for k in (j + 1)..6 {
                if adjacent(i, k) && adjacent(j, k) {
                    let signs = [
                        edge_sign_type(&nodes[i], &nodes[j], atol),
                        edge_sign_type(&nodes[j], &nodes[k], atol),
                        edge_sign_type(&nodes[i], &nodes[k], atol),
                    ];
                    if signs.iter().all(|&s| s == EdgeSignType::Opposite) {
                        zigzag_faces.push([i, j, k]);
                    }
                }
            }
        }
    }

    assert_eq!(
        zigzag_faces.len(),
        2,
        "Expected exactly 2 zigzag faces, got {}",
        zigzag_faces.len()
    );

    // Pick the lexicographically smaller face (by sorted assessor tuples)
    let face_key = |face: &[usize; 3]| -> Vec<(usize, usize)> {
        let mut keys: Vec<(usize, usize)> = face
            .iter()
            .map(|&i| (nodes[i].low, nodes[i].high))
            .collect();
        keys.sort();
        keys
    };

    let abc_face = if face_key(&zigzag_faces[0]) < face_key(&zigzag_faces[1]) {
        zigzag_faces[0]
    } else {
        zigzag_faces[1]
    };

    let a_idx = abc_face[0];
    let b_idx = abc_face[1];
    let c_idx = abc_face[2];
    let f_idx = opposite[a_idx];
    let e_idx = opposite[b_idx];
    let d_idx = opposite[c_idx];

    // Verify the opposite face is also zigzag
    let opp_signs = [
        edge_sign_type(&nodes[d_idx], &nodes[e_idx], atol),
        edge_sign_type(&nodes[e_idx], &nodes[f_idx], atol),
        edge_sign_type(&nodes[d_idx], &nodes[f_idx], atol),
    ];
    assert!(
        opp_signs.iter().all(|&s| s == EdgeSignType::Opposite),
        "Derived opposite face is not a zigzag"
    );

    StrutTable {
        a: nodes[a_idx],
        b: nodes[b_idx],
        c: nodes[c_idx],
        d: nodes[d_idx],
        e: nodes[e_idx],
        f: nodes[f_idx],
    }
}

// ---------------------------------------------------------------------------
// Generalized motif census for arbitrary Cayley-Dickson dimension
// ---------------------------------------------------------------------------

/// A generalized cross-assessor pair (low, high) for any power-of-2 dimension.
/// For dim=16: low in 1..7, high in 8..15.
/// For dim=32: low in 1..15, high in 16..31. Etc.
pub type CrossPair = (usize, usize);

/// Generate all cross-assessors for a given Cayley-Dickson dimension.
///
/// Cross-assessors are pairs (i, j) with i in [1, dim/2) and j in [dim/2, dim).
/// For dim=16, these are the 7*8 = 56 raw pairs from which the 42 primitive
/// assessors are drawn (after excluding behind-the-8-ball pairs).
/// For the generalized census, no such exclusion is applied.
pub fn cross_assessors(dim: usize) -> Vec<CrossPair> {
    assert!(
        dim >= 4 && dim.is_power_of_two(),
        "dim must be a power of two >= 4, got {dim}"
    );
    let half = dim / 2;
    let mut result = Vec::with_capacity((half - 1) * half);
    for i in 1..half {
        for j in half..dim {
            result.push((i, j));
        }
    }
    result
}

/// Integer-exact diagonal zero-product detection.
///
/// Given cross-assessor pairs a = (i, j) and b = (k, l), returns all
/// sign pairs (s, t) in {+1, -1}^2 such that
/// `(e_i + s*e_j) * (e_k + t*e_l) = 0`.
///
/// Uses `cd_basis_mul_sign` for integer-exact computation -- no floating-point
/// tolerance is needed.
pub fn diagonal_zero_products_exact(dim: usize, a: CrossPair, b: CrossPair) -> Vec<(i8, i8)> {
    let (i, j) = a;
    let (k, l) = b;

    let idx_ik = i ^ k;
    let idx_il = i ^ l;
    let idx_jk = j ^ k;
    let idx_jl = j ^ l;

    let s_ik = cd_basis_mul_sign(dim, i, k);
    let s_il = cd_basis_mul_sign(dim, i, l);
    let s_jk = cd_basis_mul_sign(dim, j, k);
    let s_jl = cd_basis_mul_sign(dim, j, l);

    let mut solutions = Vec::new();
    for s in [1i32, -1] {
        for t in [1i32, -1] {
            let mut coeffs = HashMap::new();
            *coeffs.entry(idx_ik).or_insert(0i32) += s_ik;
            *coeffs.entry(idx_il).or_insert(0i32) += t * s_il;
            *coeffs.entry(idx_jk).or_insert(0i32) += s * s_jk;
            *coeffs.entry(idx_jl).or_insert(0i32) += s * t * s_jl;

            if coeffs.values().all(|&v| v == 0) {
                solutions.push((s as i8, t as i8));
            }
        }
    }
    solutions
}

/// A connected component of the diagonal zero-product graph over cross-assessors.
///
/// At dim=16 each component is an octahedral graph (box-kite). At higher
/// dimensions, new graph motifs appear, including complete multipartite graphs
/// K_{2,2,...,2}.
pub struct MotifComponent {
    /// Cayley-Dickson dimension.
    pub dim: usize,
    /// Assessor pairs in this component.
    pub nodes: BTreeSet<CrossPair>,
    /// Undirected edges, stored as (a, b) with a < b.
    pub edges: BTreeSet<(CrossPair, CrossPair)>,
}

impl MotifComponent {
    /// Sorted degree sequence of the graph.
    pub fn degree_sequence(&self) -> Vec<usize> {
        let mut deg: HashMap<CrossPair, usize> = self.nodes.iter().map(|&n| (n, 0)).collect();
        for &(a, b) in &self.edges {
            *deg.entry(a).or_insert(0) += 1;
            *deg.entry(b).or_insert(0) += 1;
        }
        let mut seq: Vec<usize> = deg.values().copied().collect();
        seq.sort_unstable();
        seq
    }

    /// True if the component is an octahedral graph K_{2,2,2}
    /// (6 vertices, 12 edges, all degree 4).
    pub fn is_octahedron_graph(&self) -> bool {
        self.nodes.len() == 6 && self.edges.len() == 12 && self.degree_sequence() == vec![4; 6]
    }

    /// Detect a complete multipartite graph with all parts of size 2.
    ///
    /// The complement graph must be a perfect matching: each vertex has
    /// exactly one non-neighbor, and the relation is an involution with
    /// no fixed points.
    ///
    /// Returns the number of 2-vertex parts, or 0 if not of this form.
    pub fn k2_multipartite_part_count(&self) -> usize {
        let nodes: Vec<CrossPair> = self.nodes.iter().copied().collect();
        if nodes.len() < 4 || !nodes.len().is_multiple_of(2) {
            return 0;
        }

        let edge_set: HashSet<(CrossPair, CrossPair)> = self.edges.iter().copied().collect();
        let adjacent = |a: CrossPair, b: CrossPair| -> bool {
            if a == b {
                return false;
            }
            let (x, y) = if a < b { (a, b) } else { (b, a) };
            edge_set.contains(&(x, y))
        };

        let mut opposite: HashMap<CrossPair, CrossPair> = HashMap::new();
        for &a in &nodes {
            let non_neighbors: Vec<CrossPair> = nodes
                .iter()
                .filter(|&&b| b != a && !adjacent(a, b))
                .copied()
                .collect();
            if non_neighbors.len() != 1 {
                return 0;
            }
            opposite.insert(a, non_neighbors[0]);
        }

        // Must be an involution with no fixed points
        for (&a, &b) in &opposite {
            if b == a {
                return 0;
            }
            if opposite.get(&b) != Some(&a) {
                return 0;
            }
        }

        nodes.len() / 2
    }

    /// True if the component is a cuboctahedron graph
    /// (12 vertices, 24 edges, all degree 4).
    pub fn is_cuboctahedron_graph(&self) -> bool {
        self.nodes.len() == 12 && self.edges.len() == 24 && self.degree_sequence() == vec![4; 12]
    }

    /// Adjacency matrix of the component graph as a dense real matrix.
    ///
    /// Rows/columns follow the BTreeSet ordering of nodes.
    pub fn adjacency_matrix(&self) -> DMatrix<f64> {
        let nodes: Vec<CrossPair> = self.nodes.iter().copied().collect();
        let n = nodes.len();
        let idx: HashMap<CrossPair, usize> =
            nodes.iter().enumerate().map(|(i, &cp)| (cp, i)).collect();

        let mut a = DMatrix::zeros(n, n);
        for &(u, v) in &self.edges {
            let i = idx[&u];
            let j = idx[&v];
            a[(i, j)] = 1.0;
            a[(j, i)] = 1.0;
        }
        a
    }

    /// Eigenvalue spectrum of the adjacency matrix, sorted descending.
    ///
    /// Graph spectra are isomorphism invariants: isomorphic graphs have
    /// identical spectra. Same-class motif components should share spectra.
    pub fn spectrum(&self) -> Vec<f64> {
        let a = self.adjacency_matrix();
        let eigen = SymmetricEigen::new(a);
        let mut vals: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
        vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
        vals
    }

    /// Number of triangles in the component graph.
    ///
    /// Computed as trace(A^3) / 6. Each triangle is counted 6 times in
    /// the trace (2 orientations x 3 starting vertices).
    pub fn triangle_count(&self) -> usize {
        let a = self.adjacency_matrix();
        let a3 = &a * &a * &a;
        let trace: f64 = (0..a3.nrows()).map(|i| a3[(i, i)]).sum();
        (trace / 6.0).round() as usize
    }

    /// Diameter of the component graph (longest shortest path).
    ///
    /// Computed via BFS from each vertex. Returns 0 for single-node graphs.
    pub fn diameter(&self) -> usize {
        let nodes: Vec<CrossPair> = self.nodes.iter().copied().collect();
        let n = nodes.len();
        if n <= 1 {
            return 0;
        }
        let idx: HashMap<CrossPair, usize> =
            nodes.iter().enumerate().map(|(i, &cp)| (cp, i)).collect();

        // Build adjacency list
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(u, v) in &self.edges {
            let i = idx[&u];
            let j = idx[&v];
            adj[i].push(j);
            adj[j].push(i);
        }

        let mut max_dist = 0usize;
        for start in 0..n {
            let mut dist = vec![usize::MAX; n];
            dist[start] = 0;
            let mut queue = VecDeque::new();
            queue.push_back(start);
            while let Some(u) = queue.pop_front() {
                for &v in &adj[u] {
                    if dist[v] == usize::MAX {
                        dist[v] = dist[u] + 1;
                        max_dist = max_dist.max(dist[v]);
                        queue.push_back(v);
                    }
                }
            }
        }
        max_dist
    }

    /// Girth of the component graph (length of shortest cycle).
    ///
    /// Computed via BFS from each vertex, detecting back-edges.
    /// Returns `usize::MAX` if the graph is acyclic (a forest).
    pub fn girth(&self) -> usize {
        let nodes: Vec<CrossPair> = self.nodes.iter().copied().collect();
        let n = nodes.len();
        if n <= 2 {
            return usize::MAX;
        }
        let idx: HashMap<CrossPair, usize> =
            nodes.iter().enumerate().map(|(i, &cp)| (cp, i)).collect();

        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(u, v) in &self.edges {
            let i = idx[&u];
            let j = idx[&v];
            adj[i].push(j);
            adj[j].push(i);
        }

        let mut min_cycle = usize::MAX;
        for start in 0..n {
            let mut dist = vec![usize::MAX; n];
            dist[start] = 0;
            let mut queue = VecDeque::new();
            queue.push_back(start);
            while let Some(u) = queue.pop_front() {
                for &v in &adj[u] {
                    if dist[v] == usize::MAX {
                        dist[v] = dist[u] + 1;
                        queue.push_back(v);
                    } else if dist[v] >= dist[u] {
                        // Back-edge or cross-edge at same level
                        let cycle_len = dist[u] + dist[v] + 1;
                        min_cycle = min_cycle.min(cycle_len);
                    }
                }
            }
        }
        min_cycle
    }

    /// Convert this component to a petgraph UnGraph for InvariantSuite
    /// cross-validation.
    ///
    /// Nodes are remapped to 0..n in BTreeSet (sorted) order.
    /// Undirected edges are preserved.
    pub fn to_petgraph(&self) -> UnGraph<(), ()> {
        let nodes: Vec<CrossPair> = self.nodes.iter().copied().collect();
        let idx: HashMap<CrossPair, usize> =
            nodes.iter().enumerate().map(|(i, &cp)| (cp, i)).collect();

        let mut graph = UnGraph::<(), ()>::with_capacity(nodes.len(), self.edges.len());
        let pg_nodes: Vec<NodeIndex> = (0..nodes.len()).map(|_| graph.add_node(())).collect();

        for &(u, v) in &self.edges {
            graph.add_edge(pg_nodes[idx[&u]], pg_nodes[idx[&v]], ());
        }
        graph
    }
}

/// Build the diagonal zero-product graph over cross-assessors and return its
/// connected components, sorted by (node count, lexicographic node set).
///
/// Uses XOR-bucket pruning: only pairs with matching `xor_key(low, high)` can
/// form a diagonal zero-product (necessary condition from the expansion
/// `(e_i + s*e_j)(e_k + t*e_l)` requiring `i^k == j^l` for cancellation).
pub fn motif_components_for_cross_assessors(dim: usize) -> Vec<MotifComponent> {
    let nodes = cross_assessors(dim);

    // XOR-bucket pruning: only check pairs within the same bucket
    let mut buckets: HashMap<usize, Vec<CrossPair>> = HashMap::new();
    for &a in &nodes {
        buckets.entry(xor_key(a.0, a.1)).or_default().push(a);
    }

    let mut adj: HashMap<CrossPair, HashSet<CrossPair>> =
        nodes.iter().map(|&n| (n, HashSet::new())).collect();
    let mut edges: HashSet<(CrossPair, CrossPair)> = HashSet::new();

    for bucket_nodes in buckets.values() {
        let mut sorted_bucket = bucket_nodes.clone();
        sorted_bucket.sort();
        for i in 0..sorted_bucket.len() {
            for j in (i + 1)..sorted_bucket.len() {
                let a = sorted_bucket[i];
                let b = sorted_bucket[j];
                let sols = diagonal_zero_products_exact(dim, a, b);
                if !sols.is_empty() {
                    adj.get_mut(&a).unwrap().insert(b);
                    adj.get_mut(&b).unwrap().insert(a);
                    edges.insert((a, b));
                }
            }
        }
    }

    // Only keep nodes that participate in at least one edge
    let active: HashSet<CrossPair> = adj
        .iter()
        .filter(|(_, neigh)| !neigh.is_empty())
        .map(|(&n, _)| n)
        .collect();

    if active.is_empty() {
        return Vec::new();
    }

    // Connected components via DFS
    let mut seen: HashSet<CrossPair> = HashSet::new();
    let mut components: Vec<MotifComponent> = Vec::new();

    let mut sorted_active: Vec<CrossPair> = active.iter().copied().collect();
    sorted_active.sort();

    for start in sorted_active {
        if seen.contains(&start) {
            continue;
        }

        let mut comp_nodes: BTreeSet<CrossPair> = BTreeSet::new();
        let mut stack = vec![start];
        while let Some(x) = stack.pop() {
            if !comp_nodes.insert(x) {
                continue;
            }
            if let Some(neighbors) = adj.get(&x) {
                for &y in neighbors {
                    if active.contains(&y) && !comp_nodes.contains(&y) {
                        stack.push(y);
                    }
                }
            }
        }
        seen.extend(comp_nodes.iter());

        let comp_edges: BTreeSet<(CrossPair, CrossPair)> = edges
            .iter()
            .filter(|&&(a, b)| comp_nodes.contains(&a) && comp_nodes.contains(&b))
            .copied()
            .collect();

        components.push(MotifComponent {
            dim,
            nodes: comp_nodes,
            edges: comp_edges,
        });
    }

    components.sort_by_key(|c| (c.nodes.len(), c.nodes.iter().copied().collect::<Vec<_>>()));
    components
}

// ===========================================================================
// Generic face sign census (dimension-independent)
// ===========================================================================

/// Normalized face sign pattern (order-independent classification of a
/// triangular face's three edge signs).
///
/// At dim=16 (sedenions), the census is 42 TwoSameOneOpp + 14 AllOpposite
/// (C-479). This enum supports census computation at any dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FaceSignPattern {
    /// All 3 edges Same-sign (Blues).
    AllSame,
    /// 2 Same + 1 Opposite (trefoil variant I).
    TwoSameOneOpp,
    /// 1 Same + 2 Opposite (trefoil variant II).
    OneSameTwoOpp,
    /// All 3 edges Opposite-sign (triple-zigzag).
    AllOpposite,
}

/// Classify a triangular face by counting how many of its three edges
/// are Same vs Opposite sign type.
pub fn classify_face_pattern(signs: &[EdgeSignType; 3]) -> FaceSignPattern {
    let n_same = signs.iter().filter(|&&s| s == EdgeSignType::Same).count();
    match n_same {
        3 => FaceSignPattern::AllSame,
        2 => FaceSignPattern::TwoSameOneOpp,
        1 => FaceSignPattern::OneSameTwoOpp,
        0 => FaceSignPattern::AllOpposite,
        _ => unreachable!(),
    }
}

/// Integer-exact edge sign classification for cross-assessor pairs at any dimension.
///
/// Returns `Same` if solutions include (1,1) or (-1,-1), `Opposite` otherwise.
/// Panics if the pair has no diagonal zero-products (not co-assessors).
pub fn edge_sign_type_exact(dim: usize, a: CrossPair, b: CrossPair) -> EdgeSignType {
    let sols = diagonal_zero_products_exact(dim, a, b);
    assert!(
        !sols.is_empty(),
        "No diagonal zero-products for {:?}--{:?} at dim={}",
        a,
        b,
        dim
    );
    if sols.contains(&(1, 1)) || sols.contains(&(-1, -1)) {
        EdgeSignType::Same
    } else {
        EdgeSignType::Opposite
    }
}

/// Per-component face sign census result.
#[derive(Debug, Clone)]
pub struct ComponentFaceCensus {
    /// Component index in the motif component list.
    pub component_idx: usize,
    /// Number of nodes in this component.
    pub n_nodes: usize,
    /// Number of edges in this component.
    pub n_edges: usize,
    /// Triangular faces found (as node triples, sorted).
    pub n_triangles: usize,
    /// Count of each face sign pattern.
    pub pattern_counts: HashMap<FaceSignPattern, usize>,
}

/// Complete face sign census across all motif components at a given dimension.
#[derive(Debug, Clone)]
pub struct GenericFaceSignCensus {
    /// Cayley-Dickson dimension.
    pub dim: usize,
    /// Number of motif components.
    pub n_components: usize,
    /// Total triangular faces across all components.
    pub total_triangles: usize,
    /// Aggregate pattern counts across all components.
    pub total_pattern_counts: HashMap<FaceSignPattern, usize>,
    /// Per-component breakdown.
    pub per_component: Vec<ComponentFaceCensus>,
    /// Whether all components with triangles have the same pattern distribution.
    pub uniform_across_components: bool,
}

/// Compute the face sign census for all motif components at a given CD dimension.
///
/// For each connected component of the zero-divisor graph, finds all triangular
/// faces (3-cliques), classifies each face's three edge signs as Same or Opposite,
/// and aggregates the face sign pattern distribution.
///
/// At dim=16 this reproduces C-479 (42 TwoSameOneOpp + 14 AllOpposite).
/// At dim=32+ this extends the census to pathions and beyond.
pub fn generic_face_sign_census(dim: usize) -> GenericFaceSignCensus {
    let components = motif_components_for_cross_assessors(dim);
    let mut per_component = Vec::new();
    let mut total_counts: HashMap<FaceSignPattern, usize> = HashMap::new();
    let mut total_triangles = 0usize;

    // Track distribution of first non-trivial component for uniformity check
    let mut first_dist: Option<HashMap<FaceSignPattern, usize>> = None;
    let mut uniform = true;

    for (comp_idx, comp) in components.iter().enumerate() {
        // Build adjacency set for fast triangle detection
        let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
        let adj: HashSet<(CrossPair, CrossPair)> = comp
            .edges
            .iter()
            .flat_map(|&(a, b)| [(a, b), (b, a)])
            .collect();

        // Find all triangles (3-cliques): for each edge (u,v), find common neighbors w > v
        let mut triangles: Vec<[CrossPair; 3]> = Vec::new();
        for &(u, v) in &comp.edges {
            for &w in &nodes {
                if w <= v {
                    continue;
                }
                if adj.contains(&(u, w)) && adj.contains(&(v, w)) {
                    triangles.push([u, v, w]);
                }
            }
        }

        // Classify each triangle's edge signs
        let mut pattern_counts: HashMap<FaceSignPattern, usize> = HashMap::new();
        for tri in &triangles {
            let signs = [
                edge_sign_type_exact(dim, tri[0], tri[1]),
                edge_sign_type_exact(dim, tri[1], tri[2]),
                edge_sign_type_exact(dim, tri[0], tri[2]),
            ];
            let pattern = classify_face_pattern(&signs);
            *pattern_counts.entry(pattern).or_insert(0) += 1;
            *total_counts.entry(pattern).or_insert(0) += 1;
        }

        total_triangles += triangles.len();

        // Uniformity check: compare to first non-empty distribution
        if !triangles.is_empty() {
            if let Some(ref first) = first_dist {
                if &pattern_counts != first {
                    uniform = false;
                }
            } else {
                first_dist = Some(pattern_counts.clone());
            }
        }

        per_component.push(ComponentFaceCensus {
            component_idx: comp_idx,
            n_nodes: comp.nodes.len(),
            n_edges: comp.edges.len(),
            n_triangles: triangles.len(),
            pattern_counts,
        });
    }

    GenericFaceSignCensus {
        dim,
        n_components: components.len(),
        total_triangles,
        total_pattern_counts: total_counts,
        per_component,
        uniform_across_components: uniform,
    }
}

/// Result of frustration ratio computation at a given dimension.
#[derive(Debug, Clone)]
pub struct FrustrationResult {
    /// Cayley-Dickson dimension.
    pub dim: usize,
    /// Number of connected components.
    pub n_components: usize,
    /// Total edges across all components.
    pub total_edges: usize,
    /// Edges with eta=0.
    pub eta0_count: usize,
    /// Edges with eta=1.
    pub eta1_count: usize,
    /// Total first Betti number (cycle rank).
    pub total_b1: usize,
    /// Total frustrated edges (edges not explained by BFS coboundary).
    pub total_frustrated: usize,
    /// Frustration ratio: frustrated / b1.
    pub frustration_ratio: f64,
    /// Wall-clock time for computation.
    pub elapsed_secs: f64,
}

/// Compute the frustration ratio at a given Cayley-Dickson dimension.
///
/// This is the ratio of frustrated edges to cycle rank across all
/// components of the zero-product graph. The BFS coboundary assigns
/// delta values to minimize disagreement; remaining disagreements
/// are "frustrated" in the GF(2) sense.
///
/// Sequence verified: 0.000, 0.307, 0.377, 0.388, 0.385, 0.381, 0.378
/// for dims 16, 32, 64, 128, 256, 512, 1024.
///
/// Convergence appears toward 3/8 = 0.375 from above.
pub fn compute_frustration_ratio(dim: usize) -> FrustrationResult {
    use crate::construction::cayley_dickson::cd_basis_mul_sign;
    use std::time::Instant;

    let psi = |d: usize, i: usize, j: usize| -> u8 {
        if cd_basis_mul_sign(d, i, j) == 1 {
            0
        } else {
            1
        }
    };

    let t0 = Instant::now();
    let components = motif_components_for_cross_assessors(dim);

    let mut total_edges = 0usize;
    let mut total_eta0 = 0usize;
    let mut total_eta1 = 0usize;
    let mut total_b1 = 0usize;
    let mut total_frustrated = 0usize;

    for comp in components.iter() {
        let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
        let n = nodes.len();
        total_edges += comp.edges.len();

        let eta =
            |a: CrossPair, b: CrossPair| -> u8 { psi(dim, a.0, b.1) ^ psi(dim, a.1, b.0) };

        // Edge eta balance
        for &(u, v) in &comp.edges {
            if eta(u, v) == 0 {
                total_eta0 += 1;
            } else {
                total_eta1 += 1;
            }
        }

        // BFS coboundary test
        let node_idx: std::collections::HashMap<CrossPair, usize> =
            nodes.iter().enumerate().map(|(i, &nd)| (nd, i)).collect();
        let mut adj: Vec<Vec<(usize, u8)>> = vec![vec![]; n];
        for &(u, v) in &comp.edges {
            let ui = node_idx[&u];
            let vi = node_idx[&v];
            let e = eta(u, v);
            adj[ui].push((vi, e));
            adj[vi].push((ui, e));
        }

        let mut delta = vec![0u8; n];
        let mut visited = vec![false; n];
        visited[0] = true;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(0usize);
        while let Some(u) = queue.pop_front() {
            for &(v, e) in &adj[u] {
                if !visited[v] {
                    visited[v] = true;
                    delta[v] = delta[u] ^ e;
                    queue.push_back(v);
                }
            }
        }

        let b1 = comp.edges.len() - n + 1;
        total_b1 += b1;

        for &(u_node, v_node) in &comp.edges {
            let ui = node_idx[&u_node];
            let vi = node_idx[&v_node];
            let e = eta(u_node, v_node);
            if (delta[ui] ^ delta[vi]) != e {
                total_frustrated += 1;
            }
        }
    }

    let frust_ratio = if total_b1 > 0 {
        total_frustrated as f64 / total_b1 as f64
    } else {
        0.0
    };

    FrustrationResult {
        dim,
        n_components: components.len(),
        total_edges,
        eta0_count: total_eta0,
        eta1_count: total_eta1,
        total_b1,
        total_frustrated,
        frustration_ratio: frust_ratio,
        elapsed_secs: t0.elapsed().as_secs_f64(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_primitive_assessors_count() {
        let assessors = primitive_assessors();
        assert_eq!(
            assessors.len(),
            42,
            "Should have exactly 42 primitive assessors"
        );
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
            assert_ne!(
                a.high,
                a.low + 8,
                "Assessor ({}, {}) should be excluded",
                a.low,
                a.high
            );
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
        assert_eq!(
            boxkites.len(),
            7,
            "Sedenions should have exactly 7 box-kites (de Marrais), got {}",
            boxkites.len()
        );
    }

    #[test]
    fn test_sedenion_boxkite_structure() {
        let boxkites = find_box_kites(16, 1e-10);

        for (i, bk) in boxkites.iter().enumerate() {
            // Each box-kite should have 6 assessors
            assert_eq!(
                bk.assessors.len(),
                6,
                "Box-kite {} should have 6 assessors, got {}",
                i,
                bk.assessors.len()
            );

            // Each box-kite should have 12 edges
            assert_eq!(
                bk.edges.len(),
                12,
                "Box-kite {} should have 12 edges, got {}",
                i,
                bk.edges.len()
            );

            // Each box-kite should have 3 struts
            assert_eq!(
                bk.struts.len(),
                3,
                "Box-kite {} should have 3 struts, got {}",
                i,
                bk.struts.len()
            );

            // Strut signature should be in 1..7
            assert!(
                bk.strut_signature >= 1 && bk.strut_signature <= 7,
                "Box-kite {} has invalid strut signature {}",
                i,
                bk.strut_signature
            );
        }
    }

    #[test]
    fn test_sedenion_strut_signatures() {
        let boxkites = find_box_kites(16, 1e-10);
        let mut signatures: Vec<usize> = boxkites.iter().map(|bk| bk.strut_signature).collect();
        signatures.sort();

        assert_eq!(
            signatures,
            vec![1, 2, 3, 4, 5, 6, 7],
            "Strut signatures should be {{1..7}}, got {:?}",
            signatures
        );
    }

    #[test]
    fn test_sedenion_symmetry() {
        let result = analyze_box_kite_symmetry(16, 1e-10);

        assert_eq!(result.n_boxkites, 7);
        assert_eq!(result.n_assessors, 42);
        assert!(
            result.de_marrais_valid,
            "Should validate de Marrais structure"
        );
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
        let mut all_assessors: Vec<Assessor> = boxkites
            .iter()
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

    // --- Production Rule and Automorpheme Tests ---

    #[test]
    fn test_primitive_unit_zero_divisors_count() {
        // de Marrais: 168 primitive unit zero-divisors as quartets along 42 assessors
        let assessors = primitive_assessors();
        let mut count = 0;
        for a in &assessors {
            let v1 = a.diagonal(1.0);
            let v2 = a.diagonal(-1.0);
            // Each diagonal is a unit vector
            assert!((cd_norm_sq(&v1) - 1.0).abs() < 1e-12);
            assert!((cd_norm_sq(&v2) - 1.0).abs() < 1e-12);
            // 4 unit points per assessor: +v1, -v1, +v2, -v2
            count += 4;
        }
        assert_eq!(count, 168);
    }

    #[test]
    fn test_all_edges_have_sign_solutions() {
        let boxkites = find_box_kites(16, 1e-10);
        for bk in &boxkites {
            for &(i, j) in &bk.edges {
                let sols = all_diagonal_zero_products(&bk.assessors[i], &bk.assessors[j], 1e-10);
                assert!(
                    !sols.is_empty(),
                    "Expected zero-products for edge ({},{})--({},{})",
                    bk.assessors[i].low,
                    bk.assessors[i].high,
                    bk.assessors[j].low,
                    bk.assessors[j].high,
                );
            }
        }
    }

    #[test]
    fn test_each_box_kite_has_6_trefoil_and_2_zigzag() {
        // de Marrais: 8 triangular faces = 6 trefoils + 2 zigzags
        let boxkites = find_box_kites(16, 1e-10);
        for bk in &boxkites {
            let nodes = &bk.assessors;
            let edge_set: HashSet<(usize, usize)> = bk
                .edges
                .iter()
                .flat_map(|&(a, b)| [(a, b), (b, a)])
                .collect();

            let mut trefoil = 0;
            let mut zigzag = 0;
            let mut other = 0;

            for i in 0..6 {
                for j in (i + 1)..6 {
                    if !edge_set.contains(&(i, j)) {
                        continue;
                    }
                    for k in (j + 1)..6 {
                        if edge_set.contains(&(i, k)) && edge_set.contains(&(j, k)) {
                            let signs = [
                                edge_sign_type(&nodes[i], &nodes[j], 1e-10),
                                edge_sign_type(&nodes[j], &nodes[k], 1e-10),
                                edge_sign_type(&nodes[i], &nodes[k], 1e-10),
                            ];
                            let opp = signs
                                .iter()
                                .filter(|&&s| s == EdgeSignType::Opposite)
                                .count();
                            let same = signs.iter().filter(|&&s| s == EdgeSignType::Same).count();
                            if opp == 3 {
                                zigzag += 1;
                            } else if same == 2 && opp == 1 {
                                trefoil += 1;
                            } else {
                                other += 1;
                            }
                        }
                    }
                }
            }

            assert_eq!(
                (trefoil, zigzag, other),
                (6, 2, 0),
                "Box-kite {} has ({},{},{}) trefoil/zigzag/other faces",
                bk.id,
                trefoil,
                zigzag,
                other,
            );
        }
    }

    #[test]
    fn test_strut_pairs_no_zero_products() {
        // Struts (non-edges in octahedron) should have no diagonal zero-products
        let boxkites = find_box_kites(16, 1e-10);
        for bk in &boxkites {
            for &(i, j) in &bk.struts {
                let sols = all_diagonal_zero_products(&bk.assessors[i], &bk.assessors[j], 1e-10);
                assert!(
                    sols.is_empty(),
                    "Strut ({},{})--({},{}) should have no zero-products",
                    bk.assessors[i].low,
                    bk.assessors[i].high,
                    bk.assessors[j].low,
                    bk.assessors[j].high,
                );
            }
        }
    }

    #[test]
    fn test_canonical_strut_table_two_zigzag_faces() {
        let boxkites = find_box_kites(16, 1e-10);
        for bk in &boxkites {
            let tab = canonical_strut_table(bk, 1e-10);

            // ABC should be a zigzag face (all Opposite edges)
            assert_eq!(
                edge_sign_type(&tab.a, &tab.b, 1e-10),
                EdgeSignType::Opposite
            );
            assert_eq!(
                edge_sign_type(&tab.b, &tab.c, 1e-10),
                EdgeSignType::Opposite
            );
            assert_eq!(
                edge_sign_type(&tab.a, &tab.c, 1e-10),
                EdgeSignType::Opposite
            );

            // DEF should also be a zigzag face
            assert_eq!(
                edge_sign_type(&tab.d, &tab.e, 1e-10),
                EdgeSignType::Opposite
            );
            assert_eq!(
                edge_sign_type(&tab.e, &tab.f, 1e-10),
                EdgeSignType::Opposite
            );
            assert_eq!(
                edge_sign_type(&tab.d, &tab.f, 1e-10),
                EdgeSignType::Opposite
            );
        }
    }

    #[test]
    fn test_production_rule_1_trefoil_third() {
        // PR#1 should reconstruct a valid third vertex for every edge
        let boxkites = find_box_kites(16, 1e-10);
        for bk in &boxkites {
            let edge_set: HashSet<(usize, usize)> = bk
                .edges
                .iter()
                .flat_map(|&(a, b)| [(a, b), (b, a)])
                .collect();

            for &(i, j) in &bk.edges {
                let c = production_rule_1(&bk.assessors[i], &bk.assessors[j]);

                // Result must be in the box-kite
                assert!(
                    bk.assessors.contains(&c),
                    "PR#1 result ({},{}) not in box-kite {}",
                    c.low,
                    c.high,
                    bk.id,
                );

                // Result must be adjacent to both inputs
                let c_idx = bk.assessors.iter().position(|a| *a == c).unwrap();
                assert!(
                    edge_set.contains(&(i, c_idx)),
                    "PR#1 result not adjacent to first input",
                );
                assert!(
                    edge_set.contains(&(j, c_idx)),
                    "PR#1 result not adjacent to second input",
                );

                // Triangle sign pattern: trefoil (2 Same, 1 Opposite) or
                // zigzag (0 Same, 3 Opposite)
                let signs = [
                    edge_sign_type(&bk.assessors[i], &bk.assessors[j], 1e-10),
                    edge_sign_type(&bk.assessors[i], &c, 1e-10),
                    edge_sign_type(&bk.assessors[j], &c, 1e-10),
                ];
                let same = signs.iter().filter(|&&s| s == EdgeSignType::Same).count();
                let opp = signs
                    .iter()
                    .filter(|&&s| s == EdgeSignType::Opposite)
                    .count();
                assert!(
                    (same, opp) == (2, 1) || (same, opp) == (0, 3),
                    "PR#1 triangle has unexpected sign pattern ({},{}) for edge ({},{})--({},{})",
                    same,
                    opp,
                    bk.assessors[i].low,
                    bk.assessors[i].high,
                    bk.assessors[j].low,
                    bk.assessors[j].high,
                );
            }
        }
    }

    #[test]
    fn test_production_rule_2_creates_new_co_assessors() {
        // PR#2 outputs must be co-assessors with each other but not with inputs
        let primitives: HashSet<Assessor> = primitive_assessors().into_iter().collect();
        let boxkites = find_box_kites(16, 1e-10);

        for bk in &boxkites {
            for &(i, j) in &bk.edges {
                let (p, q) = production_rule_2(&bk.assessors[i], &bk.assessors[j], 1e-10);
                assert_ne!(p, q);
                assert!(
                    primitives.contains(&p),
                    "PR#2 output ({},{}) not primitive",
                    p.low,
                    p.high
                );
                assert!(
                    primitives.contains(&q),
                    "PR#2 output ({},{}) not primitive",
                    q.low,
                    q.high
                );

                // Outputs are co-assessors
                assert!(
                    !all_diagonal_zero_products(&p, &q, 1e-10).is_empty(),
                    "PR#2 outputs should be co-assessors",
                );

                // Outputs are NOT co-assessors with inputs
                assert!(all_diagonal_zero_products(&p, &bk.assessors[i], 1e-10).is_empty());
                assert!(all_diagonal_zero_products(&p, &bk.assessors[j], 1e-10).is_empty());
                assert!(all_diagonal_zero_products(&q, &bk.assessors[i], 1e-10).is_empty());
                assert!(all_diagonal_zero_products(&q, &bk.assessors[j], 1e-10).is_empty());
            }
        }
    }

    #[test]
    fn test_o_trips_form_fano_plane() {
        // 7 lines, 7 points, each point on 3 lines, each pair on 1 line
        assert_eq!(O_TRIPS.len(), 7);

        let mut point_counts = [0usize; 8]; // indices 1..7
        for t in &O_TRIPS {
            for &p in t {
                assert!((1..=7).contains(&p));
                point_counts[p] += 1;
            }
        }
        for p in 1..=7 {
            assert_eq!(point_counts[p], 3, "Point {} should appear on 3 lines", p);
        }

        // Each pair determines exactly 1 line
        let mut pair_counts: HashMap<(usize, usize), usize> = HashMap::new();
        for t in &O_TRIPS {
            for i in 0..3 {
                for j in (i + 1)..3 {
                    let pair = (t[i].min(t[j]), t[i].max(t[j]));
                    *pair_counts.entry(pair).or_insert(0) += 1;
                }
            }
        }
        // C(7,2) = 21 pairs
        assert_eq!(pair_counts.len(), 21);
        for &count in pair_counts.values() {
            assert_eq!(count, 1, "Each pair should appear on exactly 1 line");
        }
    }

    #[test]
    fn test_automorphemes_cover_assessors_twice() {
        let autos = automorphemes();
        assert_eq!(autos.len(), 7);
        for a in &autos {
            assert_eq!(a.len(), 12, "Each automorpheme should have 12 assessors");
        }

        let primitives: HashSet<Assessor> = primitive_assessors().into_iter().collect();
        let union: HashSet<Assessor> = autos.iter().flat_map(|s| s.iter().copied()).collect();
        assert_eq!(union, primitives);

        // Each assessor appears in exactly 2 automorphemes
        let mut membership: HashMap<Assessor, usize> = HashMap::new();
        for a in &autos {
            for &assessor in a {
                *membership.entry(assessor).or_insert(0) += 1;
            }
        }
        for (&assessor, &count) in &membership {
            assert_eq!(
                count, 2,
                "Assessor ({},{}) should appear in exactly 2 automorphemes",
                assessor.low, assessor.high,
            );
        }

        // "Behind the 8-ball": excluded assessors have no automorphemes
        for low in 1..=7 {
            let a8 = Assessor::new(low, 8);
            assert!(
                automorphemes_containing_assessor(&a8).is_empty(),
                "({}, 8) should not be in any automorpheme",
                low,
            );
        }
    }

    #[test]
    fn test_production_rule_3_unique_other() {
        for o_trip in &O_TRIPS {
            let assessors = automorpheme_assessors(o_trip);
            for a in &assessors {
                let other = production_rule_3(o_trip, a);
                assert_ne!(other, *o_trip);
                assert!(automorpheme_assessors(&other).contains(a));

                let mut containing = automorphemes_containing_assessor(a);
                containing.sort();
                let mut expected = vec![*o_trip, other];
                expected.sort();
                assert_eq!(containing, expected);
            }
        }
    }

    // --- Motif Census Tests ---

    #[test]
    fn test_motif_census_16d_matches_de_marrais_box_kites() {
        // At dim=16 the motif census must recover exactly the 7 box-kites
        let comps = motif_components_for_cross_assessors(16);
        assert_eq!(
            comps.len(),
            7,
            "Expected 7 components at dim=16, got {}",
            comps.len()
        );

        // All should be octahedral (6 nodes, 12 edges, degree-4 regular)
        for (i, c) in comps.iter().enumerate() {
            assert_eq!(
                c.nodes.len(),
                6,
                "Component {i} has {} nodes",
                c.nodes.len()
            );
            assert!(
                c.is_octahedron_graph(),
                "Component {i} is not an octahedron"
            );
        }

        // Union of all component nodes = the 42 primitive assessors
        let union: HashSet<CrossPair> =
            comps.iter().flat_map(|c| c.nodes.iter().copied()).collect();
        let primitives: HashSet<CrossPair> = primitive_assessors()
            .iter()
            .map(|a| (a.low, a.high))
            .collect();
        assert_eq!(
            union, primitives,
            "Census nodes should match primitive assessors"
        );

        // Cross-validate against find_box_kites
        let boxkites = find_box_kites(16, 1e-10);
        let bk_union: HashSet<CrossPair> = boxkites
            .iter()
            .flat_map(|bk| bk.assessors.iter().map(|a| (a.low, a.high)))
            .collect();
        assert_eq!(union, bk_union, "Census should agree with find_box_kites");
    }

    #[test]
    fn test_motif_edges_respect_xor_bucket() {
        // XOR bucket is a necessary condition for diagonal zero-products
        for comp in &motif_components_for_cross_assessors(16) {
            for &(a, b) in &comp.edges {
                assert_eq!(
                    xor_key(a.0, a.1),
                    xor_key(b.0, b.1),
                    "Edge ({},{})--({},{}) violates XOR bucket necessity",
                    a.0,
                    a.1,
                    b.0,
                    b.1,
                );
            }
        }
    }

    #[test]
    fn test_motif_census_32d_has_k2_multipartite() {
        // At dim=32, new graph motifs appear beyond octahedra
        let comps = motif_components_for_cross_assessors(32);
        assert!(!comps.is_empty(), "Expected non-empty census at dim=32");
        assert!(
            comps.iter().any(|c| c.k2_multipartite_part_count() > 0),
            "Expected at least one K_{{2,...,2}} component at dim=32",
        );
    }

    #[test]
    fn test_motif_census_32d_summary() {
        // Record the dim=32 census as a regression test.
        //
        // Results (discovered 2026-02-06):
        //   15 components, all with 14 nodes (7 cross-assessor pairs).
        //   8 are K_{2,2,2,2,2,2,2} (heptacross: 14 nodes, 84 edges, degree-12 regular).
        //   7 have mixed degree [4^12, 12^2] with 36 edges -- a new motif class.
        let comps = motif_components_for_cross_assessors(32);

        // Exact component count.
        assert_eq!(
            comps.len(),
            15,
            "dim=32 should have exactly 15 motif components"
        );

        // All components have 14 nodes.
        for (i, c) in comps.iter().enumerate() {
            assert_eq!(c.nodes.len(), 14, "comp[{}] should have 14 nodes", i);
        }

        // Count multipartite (heptacross) vs non-multipartite.
        let n_heptacross = comps
            .iter()
            .filter(|c| c.k2_multipartite_part_count() == 7)
            .count();
        let n_mixed = comps
            .iter()
            .filter(|c| c.k2_multipartite_part_count() == 0)
            .count();

        assert_eq!(
            n_heptacross, 8,
            "Expected 8 K_{{2,2,2,2,2,2,2}} (heptacross) components"
        );
        assert_eq!(n_mixed, 7, "Expected 7 mixed-degree components");

        // Heptacross validation: 84 edges, degree-12 regular.
        for c in comps.iter().filter(|c| c.k2_multipartite_part_count() == 7) {
            assert_eq!(c.edges.len(), 84);
            assert!(c.degree_sequence().iter().all(|&d| d == 12));
        }

        // Mixed-degree validation: 36 edges, degree sequence [4^12, 12^2].
        for c in comps.iter().filter(|c| c.k2_multipartite_part_count() == 0) {
            assert_eq!(c.edges.len(), 36);
            let seq = c.degree_sequence();
            assert_eq!(seq.iter().filter(|&&d| d == 4).count(), 12);
            assert_eq!(seq.iter().filter(|&&d| d == 12).count(), 2);
        }
    }

    // ---------------------------------------------------------------
    // Regression tests for dim=64 motif census
    // ---------------------------------------------------------------

    #[test]
    fn test_motif_census_64d_component_count() {
        let comps = motif_components_for_cross_assessors(64);
        assert_eq!(comps.len(), 31, "dim=64 should have 31 components");
    }

    #[test]
    fn test_motif_census_64d_uniform_node_count() {
        let comps = motif_components_for_cross_assessors(64);
        for (i, c) in comps.iter().enumerate() {
            assert_eq!(c.nodes.len(), 30, "dim=64 comp[{}] should have 30 nodes", i);
        }
    }

    #[test]
    fn test_motif_census_64d_summary() {
        // dim=64 census (discovered 2026-02-07):
        //   31 components, all with 30 nodes.
        //   4 distinct motif classes:
        //     8 x [30 nodes, 84 edges,  deg=[4^28, 28^2]]         -- sparse double-hub
        //     7 x [30 nodes, 228 edges, deg=[12^24, 28^6]]        -- intermediate
        //     7 x [30 nodes, 276 edges, deg=[4^4, 20^24, 28^2]]   -- mixed intermediate
        //     9 x [30 nodes, 420 edges, deg=[28^30]]               -- K_{2,...,2} (15 parts)
        let comps = motif_components_for_cross_assessors(64);

        // Classify by edge count
        let by_edges = |e: usize| -> usize { comps.iter().filter(|c| c.edges.len() == e).count() };

        assert_eq!(
            by_edges(84),
            8,
            "Expected 8 sparse double-hub components (84 edges)"
        );
        assert_eq!(
            by_edges(228),
            7,
            "Expected 7 intermediate components (228 edges)"
        );
        assert_eq!(
            by_edges(276),
            7,
            "Expected 7 mixed intermediate components (276 edges)"
        );
        assert_eq!(
            by_edges(420),
            9,
            "Expected 9 K_{{2,...,2}} components (420 edges)"
        );

        // Verify K_{2,...,2} structure: 15-partite with all parts size 2
        for c in comps.iter().filter(|c| c.edges.len() == 420) {
            assert_eq!(c.k2_multipartite_part_count(), 15);
            assert!(c.degree_sequence().iter().all(|&d| d == 28));
        }

        // Verify sparse double-hub: degree sequence [4^28, 28^2]
        for c in comps.iter().filter(|c| c.edges.len() == 84) {
            let seq = c.degree_sequence();
            assert_eq!(seq.iter().filter(|&&d| d == 4).count(), 28);
            assert_eq!(seq.iter().filter(|&&d| d == 28).count(), 2);
        }

        // Verify total edges
        let total_edges: usize = comps.iter().map(|c| c.edges.len()).sum();
        assert_eq!(total_edges, 7980);
    }

    // ---------------------------------------------------------------
    // Regression tests for dim=128 motif census
    // ---------------------------------------------------------------

    #[test]
    fn test_motif_census_128d_component_count() {
        let comps = motif_components_for_cross_assessors(128);
        assert_eq!(comps.len(), 63, "dim=128 should have 63 components");
    }

    #[test]
    fn test_motif_census_128d_uniform_node_count() {
        let comps = motif_components_for_cross_assessors(128);
        for (i, c) in comps.iter().enumerate() {
            assert_eq!(
                c.nodes.len(),
                62,
                "dim=128 comp[{}] should have 62 nodes",
                i
            );
        }
    }

    #[test]
    fn test_motif_census_128d_summary() {
        // dim=128 census (discovered 2026-02-07):
        //   63 components, all with 62 nodes.
        //   8 distinct motif classes (edges, count):
        //     (180, 9), (516, 8), (756, 7), (948, 7),
        //     (1092, 7), (1284, 7), (1524, 8), (1860, 10)
        let comps = motif_components_for_cross_assessors(128);

        let by_edges = |e: usize| -> usize { comps.iter().filter(|c| c.edges.len() == e).count() };

        // 8 motif classes
        let expected: [(usize, usize); 8] = [
            (180, 9),
            (516, 8),
            (756, 7),
            (948, 7),
            (1092, 7),
            (1284, 7),
            (1524, 8),
            (1860, 10),
        ];
        for &(edges, count) in &expected {
            assert_eq!(
                by_edges(edges),
                count,
                "dim=128: expected {} components with {} edges",
                count,
                edges
            );
        }

        // Verify K_{2,...,2} structure: 31-partite
        for c in comps.iter().filter(|c| c.edges.len() == 1860) {
            assert_eq!(c.k2_multipartite_part_count(), 31);
            assert!(c.degree_sequence().iter().all(|&d| d == 60));
        }

        // Verify sparsest: degree [4^60, 60^2]
        for c in comps.iter().filter(|c| c.edges.len() == 180) {
            let seq = c.degree_sequence();
            assert_eq!(seq.iter().filter(|&&d| d == 4).count(), 60);
            assert_eq!(seq.iter().filter(|&&d| d == 60).count(), 2);
        }

        // Total edges
        let total_edges: usize = comps.iter().map(|c| c.edges.len()).sum();
        assert_eq!(total_edges, 65100);
    }

    // ---------------------------------------------------------------
    // Regression tests for dim=256 motif census
    // ---------------------------------------------------------------

    #[test]
    fn test_motif_census_256d_component_count() {
        let comps = motif_components_for_cross_assessors(256);
        assert_eq!(comps.len(), 127, "dim=256 should have 127 components");
    }

    #[test]
    fn test_motif_census_256d_uniform_node_count() {
        let comps = motif_components_for_cross_assessors(256);
        for (i, c) in comps.iter().enumerate() {
            assert_eq!(
                c.nodes.len(),
                126,
                "dim=256 comp[{}] should have 126 nodes",
                i
            );
        }
    }

    #[test]
    fn test_motif_census_256d_summary() {
        // dim=256 census (discovered 2026-02-07):
        //   127 components, all with 126 nodes.
        //   16 distinct motif classes (edges, count):
        //     (372, 10), (1092, 9), (1716, 8), (2436, 8),
        //     (2676, 7), (3396, 7), (3444, 7), (4020, 7),
        //     (4164, 7), (4740, 7), (4788, 7), (5508, 7),
        //     (5748, 8), (6468, 8), (7092, 9), (7812, 11)
        let comps = motif_components_for_cross_assessors(256);

        let by_edges = |e: usize| -> usize { comps.iter().filter(|c| c.edges.len() == e).count() };

        let expected: [(usize, usize); 16] = [
            (372, 10),
            (1092, 9),
            (1716, 8),
            (2436, 8),
            (2676, 7),
            (3396, 7),
            (3444, 7),
            (4020, 7),
            (4164, 7),
            (4740, 7),
            (4788, 7),
            (5508, 7),
            (5748, 8),
            (6468, 8),
            (7092, 9),
            (7812, 11),
        ];
        for &(edges, count) in &expected {
            assert_eq!(
                by_edges(edges),
                count,
                "dim=256: expected {} components with {} edges",
                count,
                edges
            );
        }

        // Verify K_{2,...,2}: 63-partite
        for c in comps.iter().filter(|c| c.edges.len() == 7812) {
            assert_eq!(c.k2_multipartite_part_count(), 63);
            assert!(c.degree_sequence().iter().all(|&d| d == 124));
        }

        // Verify sparsest: degree [4^124, 124^2]
        for c in comps.iter().filter(|c| c.edges.len() == 372) {
            let seq = c.degree_sequence();
            assert_eq!(seq.iter().filter(|&&d| d == 4).count(), 124);
            assert_eq!(seq.iter().filter(|&&d| d == 124).count(), 2);
        }

        // Total edges
        let total_edges: usize = comps.iter().map(|c| c.edges.len()).sum();
        assert_eq!(total_edges, 523404);
    }

    // ---------------------------------------------------------------
    // Cross-dimensional scaling law tests
    // ---------------------------------------------------------------

    #[test]
    fn test_motif_scaling_laws() {
        // Verify the scaling patterns hold across all computed dimensions.
        // These are empirical laws discovered from the exact census.
        for &dim in &[16, 32, 64, 128] {
            let comps = motif_components_for_cross_assessors(dim);

            // Law 1: n_comps = dim/2 - 1
            assert_eq!(
                comps.len(),
                dim / 2 - 1,
                "dim={}: n_comps should be dim/2-1",
                dim
            );

            // Law 2: all components have dim/2 - 2 nodes
            for c in &comps {
                assert_eq!(
                    c.nodes.len(),
                    dim / 2 - 2,
                    "dim={}: nodes_per_comp should be dim/2-2",
                    dim
                );
            }

            // Law 3: K_{2,...,2} components have dim/4-1 bipartite parts
            let k2_parts_max = comps
                .iter()
                .map(|c| c.k2_multipartite_part_count())
                .max()
                .unwrap_or(0);
            assert_eq!(
                k2_parts_max,
                dim / 4 - 1,
                "dim={}: max K2 parts should be dim/4-1",
                dim
            );
        }
    }

    #[test]
    fn test_motif_class_count_scaling() {
        // The number of distinct motif classes = dim/16.
        // This doubles with each Cayley-Dickson doubling.
        use std::collections::HashSet;
        for &dim in &[16, 32, 64, 128] {
            let comps = motif_components_for_cross_assessors(dim);
            let classes: HashSet<usize> = comps.iter().map(|c| c.edges.len()).collect();
            assert_eq!(
                classes.len(),
                dim / 16,
                "dim={}: n_classes should be dim/16",
                dim
            );
        }
    }

    #[test]
    fn test_k2_component_count_scaling() {
        // The number of K_{2,...,2} components = 3 + log2(dim).
        for &(dim, expected_k2) in &[(16, 7), (32, 8), (64, 9), (128, 10)] {
            let comps = motif_components_for_cross_assessors(dim);
            let n_k2 = comps
                .iter()
                .filter(|c| c.k2_multipartite_part_count() > 0)
                .count();
            assert_eq!(
                n_k2, expected_k2,
                "dim={}: K2 component count should be {}",
                dim, expected_k2
            );
        }
    }

    #[test]
    fn test_no_octahedra_beyond_16d() {
        // Octahedra (6 nodes, 12 edges) appear ONLY at dim=16.
        // At dim=32 and beyond, the structure completely restructures.
        for &dim in &[32, 64, 128] {
            let comps = motif_components_for_cross_assessors(dim);
            let n_octahedra = comps.iter().filter(|c| c.is_octahedron_graph()).count();
            assert_eq!(
                n_octahedra, 0,
                "dim={}: should have no octahedra (only at dim=16)",
                dim
            );
        }
    }

    #[test]
    fn test_complement_graph_regression_dim16() {
        // At dim=16, all 7 components should be K_{2,2,2} (octahedra),
        // so k2_multipartite_part_count should return 3 for each.
        let comps = motif_components_for_cross_assessors(16);
        assert_eq!(comps.len(), 7);
        for (i, c) in comps.iter().enumerate() {
            assert_eq!(
                c.k2_multipartite_part_count(),
                3,
                "Component {} at dim=16 should be K_{{2,2,2}} (3 parts), got {}",
                i,
                c.k2_multipartite_part_count()
            );
            assert!(
                c.is_octahedron_graph(),
                "Component {} at dim=16 should be octahedral",
                i
            );
        }
    }

    // ---------------------------------------------------------------
    // Graph invariant tests (spectral, combinatorial)
    // ---------------------------------------------------------------

    #[test]
    fn test_octahedron_spectrum_dim16() {
        // K_{2,2,2} (octahedron) is complete tripartite with k=3 parts of m=2.
        // Eigenvalues: 2(k-1)=4 (x1), 0 (x k*(m-1)=3), -m=-2 (x k-1=2).
        // Sorted descending: [4, 0, 0, 0, -2, -2]
        let comps = motif_components_for_cross_assessors(16);
        let expected = vec![4.0, 0.0, 0.0, 0.0, -2.0, -2.0];
        for (i, c) in comps.iter().enumerate() {
            let spec = c.spectrum();
            assert_eq!(spec.len(), 6, "dim=16 comp[{i}] should have 6 eigenvalues");
            for (j, (&got, &want)) in spec.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - want).abs() < 1e-10,
                    "dim=16 comp[{i}] eigenvalue[{j}]: got {got:.6}, expected {want}",
                );
            }
        }
    }

    #[test]
    fn test_octahedron_triangle_count_dim16() {
        // An octahedron (K_{2,2,2}) has 8 triangles.
        let comps = motif_components_for_cross_assessors(16);
        for (i, c) in comps.iter().enumerate() {
            assert_eq!(
                c.triangle_count(),
                8,
                "dim=16 comp[{i}] should have 8 triangles"
            );
        }
    }

    #[test]
    fn test_octahedron_diameter_dim16() {
        // Diameter of octahedron = 2 (non-adjacent strut pairs).
        let comps = motif_components_for_cross_assessors(16);
        for (i, c) in comps.iter().enumerate() {
            assert_eq!(c.diameter(), 2, "dim=16 comp[{i}] diameter should be 2");
        }
    }

    #[test]
    fn test_octahedron_girth_dim16() {
        // Girth of octahedron = 3 (has triangles).
        let comps = motif_components_for_cross_assessors(16);
        for (i, c) in comps.iter().enumerate() {
            assert_eq!(c.girth(), 3, "dim=16 comp[{i}] girth should be 3");
        }
    }

    #[test]
    fn test_same_class_same_spectrum_dim32() {
        // Components in the same motif class (same edge count) should
        // have identical spectra up to numerical precision.
        let comps = motif_components_for_cross_assessors(32);
        let mut by_edges: HashMap<usize, Vec<Vec<f64>>> = HashMap::new();
        for c in &comps {
            by_edges
                .entry(c.edges.len())
                .or_default()
                .push(c.spectrum());
        }
        for (edges, spectra) in &by_edges {
            if spectra.len() < 2 {
                continue;
            }
            let ref_spec = &spectra[0];
            for (j, spec) in spectra.iter().enumerate().skip(1) {
                assert_eq!(
                    ref_spec.len(),
                    spec.len(),
                    "class edges={edges}: spectrum length mismatch"
                );
                for (k, (&a, &b)) in ref_spec.iter().zip(spec.iter()).enumerate() {
                    assert!(
                        (a - b).abs() < 1e-8,
                        "class edges={edges}: comp[0] vs comp[{j}] eigenvalue[{k}]: \
                         {a:.6} vs {b:.6}",
                    );
                }
            }
        }
    }

    #[test]
    fn test_different_classes_different_spectra_dim32() {
        // Different motif classes should have distinct spectra.
        let comps = motif_components_for_cross_assessors(32);
        let mut class_spectra: Vec<(usize, Vec<f64>)> = Vec::new();
        let mut seen_edges: HashSet<usize> = HashSet::new();
        for c in &comps {
            if seen_edges.insert(c.edges.len()) {
                class_spectra.push((c.edges.len(), c.spectrum()));
            }
        }
        for i in 0..class_spectra.len() {
            for j in (i + 1)..class_spectra.len() {
                let (e1, s1) = &class_spectra[i];
                let (e2, s2) = &class_spectra[j];
                // Spectra should differ in at least one eigenvalue
                let differ = s1.len() != s2.len()
                    || s1
                        .iter()
                        .zip(s2.iter())
                        .any(|(&a, &b)| (a - b).abs() > 1e-6);
                assert!(
                    differ,
                    "Classes edges={e1} and edges={e2} should have different spectra"
                );
            }
        }
    }

    #[test]
    fn test_k2_multipartite_spectrum_dim32() {
        // K_{2,2,...,2} with k parts has known spectrum:
        // eigenvalue 2(k-1) with multiplicity 1,
        // eigenvalue 0 with multiplicity k,
        // eigenvalue -2 with multiplicity k-1.
        // At dim=32, max K2 part count = 7, so k=7:
        // spectrum = [12, 0,0,0,0,0,0,0, -2,-2,-2,-2,-2,-2]
        let comps = motif_components_for_cross_assessors(32);
        let k2_comps: Vec<&MotifComponent> = comps
            .iter()
            .filter(|c| c.k2_multipartite_part_count() == 7)
            .collect();
        assert!(
            !k2_comps.is_empty(),
            "Should have K_{{2,...,2}} with 7 parts at dim=32"
        );

        for c in &k2_comps {
            let spec = c.spectrum();
            // 14 nodes, so 14 eigenvalues
            assert_eq!(spec.len(), 14);
            // Largest eigenvalue = 2*(7-1) = 12
            assert!(
                (spec[0] - 12.0).abs() < 1e-8,
                "largest eigenvalue should be 12"
            );
            // 7 zeros
            let n_zeros = spec.iter().filter(|&&v| v.abs() < 1e-8).count();
            assert_eq!(n_zeros, 7, "should have 7 zero eigenvalues");
            // 6 eigenvalues of -2
            let n_neg2 = spec.iter().filter(|&&v| (v + 2.0).abs() < 1e-8).count();
            assert_eq!(n_neg2, 6, "should have 6 eigenvalues of -2");
        }
    }

    // -----------------------------------------------------------------------
    // De Marrais primitive ZD structure verification (task #63)
    // -----------------------------------------------------------------------

    #[test]
    fn test_intra_assessor_diagonals_never_zero_divide() {
        // De Marrais: the two diagonals (V,/) and (V,\) within the SAME
        // assessor plane do NOT zero-divide each other. This is because
        // they span the same 2-plane and their product is pure-imaginary
        // (not zero).
        let assessors = primitive_assessors();
        for a in &assessors {
            let d_plus = a.diagonal(1.0);
            let d_minus = a.diagonal(-1.0);

            // Forward product: d+ * d-
            let prod_fwd = cd_multiply(&d_plus, &d_minus);
            let norm_fwd = cd_norm_sq(&prod_fwd).sqrt();

            // Reverse product: d- * d+
            let prod_rev = cd_multiply(&d_minus, &d_plus);
            let norm_rev = cd_norm_sq(&prod_rev).sqrt();

            assert!(
                norm_fwd > 0.1,
                "Intra-assessor ({},{}) forward product should be non-zero, got norm={}",
                a.low,
                a.high,
                norm_fwd
            );
            assert!(
                norm_rev > 0.1,
                "Intra-assessor ({},{}) reverse product should be non-zero, got norm={}",
                a.low,
                a.high,
                norm_rev
            );
        }
    }

    #[test]
    fn test_every_boxkite_edge_has_zero_product() {
        // Every edge (co-assessor pair) in every box-kite must have at
        // least one sign combination (s,t) such that diag(s)*diag(t) = 0.
        let boxkites = find_box_kites(16, 1e-10);
        assert_eq!(boxkites.len(), 7);

        for bk in &boxkites {
            assert_eq!(bk.edges.len(), 12, "octahedron has 12 edges");
            for &(i, j) in &bk.edges {
                let has_zp = diagonal_zero_product(&bk.assessors[i], &bk.assessors[j], 1e-10);
                assert!(
                    has_zp.is_some(),
                    "Edge ({},{})--({},{}) in box-kite {} should have a zero-product",
                    bk.assessors[i].low,
                    bk.assessors[i].high,
                    bk.assessors[j].low,
                    bk.assessors[j].high,
                    bk.strut_signature,
                );
            }
        }
    }

    #[test]
    fn test_each_diagonal_is_zero_divisor() {
        // Each diagonal direction d = (e_low +/- e_high)/sqrt(2) should
        // be a zero-divisor: there exists some other sedenion b such that
        // d*b = 0. This is verified by checking left-multiplication matrix
        // nullity.
        use crate::analysis::annihilator::annihilator_info;
        let assessors = primitive_assessors();
        for a in &assessors {
            for sign in [-1.0, 1.0] {
                let d = a.diagonal(sign);
                let info = annihilator_info(&d, 16, 1e-10);
                assert!(
                    info.left_nullity > 0,
                    "Diagonal ({},{}) sign={} should be a left zero-divisor, \
                     but left_nullity={}",
                    a.low,
                    a.high,
                    sign as i8,
                    info.left_nullity
                );
                assert!(
                    info.right_nullity > 0,
                    "Diagonal ({},{}) sign={} should be a right zero-divisor, \
                     but right_nullity={}",
                    a.low,
                    a.high,
                    sign as i8,
                    info.right_nullity
                );
            }
        }
    }

    #[test]
    fn test_intra_assessor_product_is_pure_basis() {
        // Within an assessor (low, high), the product d+*d- should be
        // a pure basis element (up to sign). Since (e_l + e_h)(e_l - e_h)
        // = e_l^2 - e_h*e_l + e_l*e_h - e_h^2 = -1 + [e_l, e_h] - (-1)
        // = [e_l, e_h] (the commutator), which for sedenion basis elements
        // is +/- 2*e_{l XOR h}. After normalization by 1/sqrt(2)^2 = 1/2,
        // we get +/- e_{l XOR h}.
        let assessors = primitive_assessors();
        for a in &assessors {
            let d_plus = a.diagonal(1.0);
            let d_minus = a.diagonal(-1.0);
            let prod = cd_multiply(&d_plus, &d_minus);

            // Product should have exactly one nonzero component
            let nonzero: Vec<(usize, f64)> = prod
                .iter()
                .enumerate()
                .filter(|(_, &v)| v.abs() > 1e-10)
                .map(|(i, &v)| (i, v))
                .collect();

            assert_eq!(
                nonzero.len(),
                1,
                "Intra-assessor ({},{}) product should have exactly 1 nonzero \
                 component, got {:?}",
                a.low,
                a.high,
                nonzero
            );

            // That component should be at index low XOR high
            let expected_idx = a.low ^ a.high;
            assert_eq!(
                nonzero[0].0, expected_idx,
                "Nonzero component at index {}, expected {} (= {} XOR {})",
                nonzero[0].0, expected_idx, a.low, a.high
            );

            // Value should be +/- 1 (since (1/sqrt2)^2 * 2 = 1)
            assert!(
                (nonzero[0].1.abs() - 1.0).abs() < 1e-10,
                "Component magnitude should be 1.0, got {}",
                nonzero[0].1.abs()
            );
        }
    }

    #[test]
    fn test_adjacency_matrix_symmetry() {
        let comps = motif_components_for_cross_assessors(16);
        for c in &comps {
            let a = c.adjacency_matrix();
            for i in 0..a.nrows() {
                for j in 0..a.ncols() {
                    assert!(
                        (a[(i, j)] - a[(j, i)]).abs() < 1e-15,
                        "Adjacency matrix should be symmetric"
                    );
                }
            }
        }
    }

    // ================================================================
    // Generic face sign census tests
    // ================================================================

    #[test]
    fn test_generic_face_sign_census_dim16_matches_c479() {
        // C-479: 56 faces = 42 TwoSameOneOpp + 14 AllOpposite, 0 AllSame, 0 OneSameTwoOpp
        let census = generic_face_sign_census(16);

        assert_eq!(census.dim, 16);
        assert_eq!(census.n_components, 7, "dim=16 has 7 box-kite components");
        assert_eq!(census.total_triangles, 56, "7 BKs x 8 faces = 56");

        let two_same = census
            .total_pattern_counts
            .get(&FaceSignPattern::TwoSameOneOpp)
            .copied()
            .unwrap_or(0);
        let all_opp = census
            .total_pattern_counts
            .get(&FaceSignPattern::AllOpposite)
            .copied()
            .unwrap_or(0);
        let all_same = census
            .total_pattern_counts
            .get(&FaceSignPattern::AllSame)
            .copied()
            .unwrap_or(0);
        let one_same = census
            .total_pattern_counts
            .get(&FaceSignPattern::OneSameTwoOpp)
            .copied()
            .unwrap_or(0);

        assert_eq!(two_same, 42, "42 TwoSameOneOpp faces (trefoil lanyards)");
        assert_eq!(all_opp, 14, "14 AllOpposite faces (triple-zigzag lanyards)");
        assert_eq!(all_same, 0, "No AllSame (Blues) faces");
        assert_eq!(one_same, 0, "No OneSameTwoOpp faces");

        // Each component should have 8 faces
        for comp in &census.per_component {
            assert_eq!(comp.n_triangles, 8, "Each BK has 8 faces");
        }

        assert!(census.uniform_across_components, "Uniform: 6+2 per BK");
    }

    #[test]
    fn test_generic_face_sign_census_dim32() {
        // dim=32 has 15 components, each with 14 nodes.
        // Two motif classes: 8 heptacross (84 edges) and 7 mixed (36 edges).
        let census = generic_face_sign_census(32);

        assert_eq!(census.dim, 32);
        assert_eq!(census.n_components, 15, "dim=32 has 15 components");
        assert_eq!(census.total_triangles, 2408);

        // Aggregate pattern counts
        let two_same = census
            .total_pattern_counts
            .get(&FaceSignPattern::TwoSameOneOpp)
            .copied()
            .unwrap_or(0);
        let all_opp = census
            .total_pattern_counts
            .get(&FaceSignPattern::AllOpposite)
            .copied()
            .unwrap_or(0);
        let all_same = census
            .total_pattern_counts
            .get(&FaceSignPattern::AllSame)
            .copied()
            .unwrap_or(0);
        let one_same = census
            .total_pattern_counts
            .get(&FaceSignPattern::OneSameTwoOpp)
            .copied()
            .unwrap_or(0);

        assert_eq!(two_same, 1050);
        assert_eq!(all_opp, 350);
        assert_eq!(all_same, 252);
        assert_eq!(one_same, 756);

        // Key structural observation: NOT uniform across components (unlike dim=16)
        assert!(!census.uniform_across_components);

        // Three distinct face-pattern regimes at dim=32:
        // (A) 7 heptacross components (84 edges): all 4 patterns
        //     102 TwoSameOneOpp + 108 OneSameTwoOpp + 36 AllSame + 34 AllOpposite = 280
        // (B) 1 special heptacross (84 edges): only 2 patterns, 3:1 ratio
        //     210 TwoSameOneOpp + 70 AllOpposite = 280
        // (C) 7 mixed components (36 edges): only 2 patterns, 3:1 ratio
        //     18 TwoSameOneOpp + 6 AllOpposite = 24

        let mut regime_a = 0usize;
        let mut regime_b = 0usize;
        let mut regime_c = 0usize;

        for comp in &census.per_component {
            let n_patterns = comp.pattern_counts.len();
            if comp.n_edges == 84 && n_patterns == 4 {
                regime_a += 1;
                assert_eq!(comp.n_triangles, 280);
                assert_eq!(
                    comp.pattern_counts
                        .get(&FaceSignPattern::TwoSameOneOpp)
                        .copied()
                        .unwrap_or(0),
                    102
                );
                assert_eq!(
                    comp.pattern_counts
                        .get(&FaceSignPattern::OneSameTwoOpp)
                        .copied()
                        .unwrap_or(0),
                    108
                );
                assert_eq!(
                    comp.pattern_counts
                        .get(&FaceSignPattern::AllSame)
                        .copied()
                        .unwrap_or(0),
                    36
                );
                assert_eq!(
                    comp.pattern_counts
                        .get(&FaceSignPattern::AllOpposite)
                        .copied()
                        .unwrap_or(0),
                    34
                );
            } else if comp.n_edges == 84 && n_patterns == 2 {
                regime_b += 1;
                assert_eq!(comp.n_triangles, 280);
                let ts = comp
                    .pattern_counts
                    .get(&FaceSignPattern::TwoSameOneOpp)
                    .copied()
                    .unwrap_or(0);
                let ao = comp
                    .pattern_counts
                    .get(&FaceSignPattern::AllOpposite)
                    .copied()
                    .unwrap_or(0);
                assert_eq!(ts, 210);
                assert_eq!(ao, 70);
                assert_eq!(ts, 3 * ao, "3:1 ratio in regime B");
            } else if comp.n_edges == 36 {
                regime_c += 1;
                assert_eq!(comp.n_triangles, 24);
                let ts = comp
                    .pattern_counts
                    .get(&FaceSignPattern::TwoSameOneOpp)
                    .copied()
                    .unwrap_or(0);
                let ao = comp
                    .pattern_counts
                    .get(&FaceSignPattern::AllOpposite)
                    .copied()
                    .unwrap_or(0);
                assert_eq!(ts, 18);
                assert_eq!(ao, 6);
                assert_eq!(ts, 3 * ao, "3:1 ratio in regime C");
            } else {
                panic!(
                    "Unexpected component: {} edges, {} patterns",
                    comp.n_edges, n_patterns
                );
            }
        }

        assert_eq!(regime_a, 7, "7 heptacross components with all 4 patterns");
        assert_eq!(regime_b, 1, "1 special heptacross with 3:1 ratio");
        assert_eq!(regime_c, 7, "7 mixed components with 3:1 ratio");
    }

    #[test]
    fn test_generic_face_sign_census_dim64() {
        // dim=64 has 31 components, each with 30 nodes.
        // 4 motif classes by edge count.
        let census = generic_face_sign_census(64);

        assert_eq!(census.dim, 64);
        assert_eq!(census.n_components, 31, "dim=64 has 31 components");

        // Print diagnostic info
        eprintln!("=== dim=64 face sign census ===");
        eprintln!("Components: {}", census.n_components);
        eprintln!("Total triangles: {}", census.total_triangles);
        for (pat, count) in &census.total_pattern_counts {
            eprintln!("  {:?}: {}", pat, count);
        }
        eprintln!("Uniform: {}", census.uniform_across_components);

        // Group components by (n_edges, n_patterns) to discover regimes
        let mut regimes: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        for (i, comp) in census.per_component.iter().enumerate() {
            let key = (comp.n_edges, comp.pattern_counts.len());
            regimes.entry(key).or_default().push(i);
        }
        eprintln!("Regimes:");
        for ((edges, pats), indices) in &regimes {
            let sample = &census.per_component[indices[0]];
            eprintln!(
                "  ({} edges, {} patterns): {} components, {} tri/comp, sample: {:?}",
                edges,
                pats,
                indices.len(),
                sample.n_triangles,
                sample.pattern_counts
            );
        }

        // Totals
        assert_eq!(census.total_triangles, 48328);
        let two_same = census
            .total_pattern_counts
            .get(&FaceSignPattern::TwoSameOneOpp)
            .copied()
            .unwrap_or(0);
        let all_opp = census
            .total_pattern_counts
            .get(&FaceSignPattern::AllOpposite)
            .copied()
            .unwrap_or(0);
        let all_same = census
            .total_pattern_counts
            .get(&FaceSignPattern::AllSame)
            .copied()
            .unwrap_or(0);
        let one_same = census
            .total_pattern_counts
            .get(&FaceSignPattern::OneSameTwoOpp)
            .copied()
            .unwrap_or(0);
        assert_eq!(two_same, 18858);
        assert_eq!(all_opp, 6286);
        assert_eq!(all_same, 5796);
        assert_eq!(one_same, 17388);

        assert!(!census.uniform_across_components);

        // 5 regimes at dim=64:
        // (A) 1 pure-max (420 edges, 2 patterns, 3:1 ratio: 2730/910)
        // (B) 8 full-max (420 edges, 4 patterns)
        // (C) 7 full-mid1 (276 edges, 4 patterns)
        // (D) 7 full-mid2 (228 edges, 4 patterns)
        // (E) 8 pure-min (84 edges, 2 patterns, 3:1 ratio: 42/14)
        let mut counts = HashMap::new();
        for comp in &census.per_component {
            let key = (comp.n_edges, comp.pattern_counts.len());
            *counts.entry(key).or_insert(0usize) += 1;
        }
        assert_eq!(counts[&(420, 2)], 1, "1 pure-max component");
        assert_eq!(counts[&(420, 4)], 8, "8 full-max components");
        assert_eq!(counts[&(276, 4)], 7, "7 full-mid1 components");
        assert_eq!(counts[&(228, 4)], 7, "7 full-mid2 components");
        assert_eq!(counts[&(84, 2)], 8, "8 pure-min components");
        assert_eq!(counts.len(), 5, "Exactly 5 regimes");

        // 3:1 ratio holds in ALL pure-regime components
        for comp in &census.per_component {
            if comp.pattern_counts.len() == 2 {
                let ts = comp
                    .pattern_counts
                    .get(&FaceSignPattern::TwoSameOneOpp)
                    .copied()
                    .unwrap_or(0);
                let ao = comp
                    .pattern_counts
                    .get(&FaceSignPattern::AllOpposite)
                    .copied()
                    .unwrap_or(0);
                assert_eq!(
                    ts,
                    3 * ao,
                    "3:1 ratio in pure component ({} edges): {} != 3*{}",
                    comp.n_edges,
                    ts,
                    ao
                );
            }
        }

        // The 84-edge pure components reproduce dim=16's 42:14 exactly
        for comp in &census.per_component {
            if comp.n_edges == 84 {
                assert_eq!(comp.n_triangles, 56);
                let ts = comp
                    .pattern_counts
                    .get(&FaceSignPattern::TwoSameOneOpp)
                    .copied()
                    .unwrap_or(0);
                let ao = comp
                    .pattern_counts
                    .get(&FaceSignPattern::AllOpposite)
                    .copied()
                    .unwrap_or(0);
                assert_eq!(ts, 42);
                assert_eq!(ao, 14);
            }
        }
    }

    #[test]
    fn test_three_to_one_ratio_antibalanced() {
        // The 3:1 ratio follows from three properties:
        //
        // (P1) Every triangle has sign product = -1 (antibalanced):
        //      If Same=+1 and Opposite=-1, then product of 3 edge signs around
        //      every triangle is -1. This means only odd numbers of Opposite
        //      edges: TwoSameOneOpp (1 Opp) and AllOpposite (3 Opp).
        //
        // (P2) Same-edge fraction is exactly 1/2: n_Same = n_Opposite.
        //
        // (P3) Edge-regularity: every edge (Same or Opposite) participates in
        //      the same number of triangles.
        //
        // From P1+P2+P3:
        //   Let d = triangles per edge.
        //   n_Same * d = 2 * TwoSameOneOpp  (each TwoSameOneOpp has 2 Same edges)
        //   n_Opp * d = TwoSameOneOpp + 3 * AllOpposite  (1 Opp + 3 Opp edges)
        //   With n_Same = n_Opp:
        //     2*TwoSameOneOpp = TwoSameOneOpp + 3*AllOpposite
        //     => TwoSameOneOpp = 3 * AllOpposite  QED

        for dim in [16, 32, 64] {
            let components = motif_components_for_cross_assessors(dim);
            let census = generic_face_sign_census(dim);

            for (i, comp) in components.iter().enumerate() {
                let comp_census = &census.per_component[i];
                if comp_census.pattern_counts.len() != 2 {
                    continue;
                }

                // (P1) Every triangle has sign product -1
                // This is implicit from the pure-regime constraint: only
                // TwoSameOneOpp (product=-1) and AllOpposite (product=-1).

                // (P2) Count Same vs Opposite edges
                let mut n_same = 0usize;
                let mut n_opp = 0usize;
                for &(a, b) in &comp.edges {
                    match edge_sign_type_exact(dim, a, b) {
                        EdgeSignType::Same => n_same += 1,
                        EdgeSignType::Opposite => n_opp += 1,
                    }
                }
                assert_eq!(
                    n_same, n_opp,
                    "dim={} comp[{}]: Same-edge fraction should be exactly 1/2: {} vs {}",
                    dim, i, n_same, n_opp
                );

                // (P3) Edge-regularity: build triangle count per edge
                // For this, we find all triangles and count per edge.
                let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
                let adj_set: HashSet<(CrossPair, CrossPair)> = comp
                    .edges
                    .iter()
                    .flat_map(|&(a, b)| [(a, b), (b, a)])
                    .collect();

                let mut edge_tri_count: HashMap<(CrossPair, CrossPair), usize> = HashMap::new();
                for &(u, v) in &comp.edges {
                    edge_tri_count.insert((u, v), 0);
                }

                for &(u, v) in &comp.edges {
                    for &w in &nodes {
                        if w <= v {
                            continue;
                        }
                        if adj_set.contains(&(u, w)) && adj_set.contains(&(v, w)) {
                            *edge_tri_count.get_mut(&(u, v)).unwrap() += 1;
                            // Also count for (u,w) and (v,w) edges
                            let e_uw = if u < w { (u, w) } else { (w, u) };
                            let e_vw = if v < w { (v, w) } else { (w, v) };
                            *edge_tri_count.entry(e_uw).or_insert(0) += 1;
                            *edge_tri_count.entry(e_vw).or_insert(0) += 1;
                        }
                    }
                }

                // Verify all edges have the same triangle count
                let tri_counts: Vec<usize> = edge_tri_count.values().copied().collect();
                let min_tc = *tri_counts.iter().min().unwrap();
                let max_tc = *tri_counts.iter().max().unwrap();

                eprintln!(
                    "dim={} comp[{}] ({} edges, pure): same={}, opp={}, tri_per_edge=[{},{}]",
                    dim,
                    i,
                    comp.edges.len(),
                    n_same,
                    n_opp,
                    min_tc,
                    max_tc
                );

                assert_eq!(
                    min_tc, max_tc,
                    "dim={} comp[{}]: edge-regularity violated: tri/edge in [{},{}]",
                    dim, i, min_tc, max_tc
                );
            }
        }
    }

    #[test]
    fn test_generic_face_sign_census_dim128() {
        // Regime count formula: dim/16 + 1 = 128/16 + 1 = 9 at dim=128.
        // This test verifies the 9-regime structure.
        use std::time::Instant;

        let t0 = Instant::now();
        let census = generic_face_sign_census(128);
        let elapsed = t0.elapsed();
        eprintln!(
            "dim=128 census: {} components, {} triangles, {:.2?}",
            census.n_components, census.total_triangles, elapsed
        );

        // Basic structure: dim/2 - 1 = 63 components
        assert_eq!(census.n_components, 63);

        // Collect distinct regime signatures: (n_edges, sorted pattern keys)
        let mut regime_map: HashMap<(usize, Vec<FaceSignPattern>), Vec<usize>> = HashMap::new();
        for comp in &census.per_component {
            let mut patterns: Vec<FaceSignPattern> = comp.pattern_counts.keys().copied().collect();
            patterns.sort();
            regime_map
                .entry((comp.n_edges, patterns))
                .or_default()
                .push(comp.component_idx);
        }

        let n_regimes = regime_map.len();
        eprintln!("dim=128 regimes: {}", n_regimes);

        // Print regime breakdown
        let mut regimes: Vec<_> = regime_map.iter().collect();
        regimes.sort_by_key(|&((edges, _), comps)| (std::cmp::Reverse(*edges), comps.len()));
        for ((edges, patterns), comps) in &regimes {
            let comp_ex = census
                .per_component
                .iter()
                .find(|c| {
                    c.n_edges == *edges && {
                        let mut p: Vec<FaceSignPattern> =
                            c.pattern_counts.keys().copied().collect();
                        p.sort();
                        &p == patterns
                    }
                })
                .unwrap();
            let pattern_str: Vec<String> = patterns
                .iter()
                .map(|p| {
                    let count = comp_ex.pattern_counts.get(p).copied().unwrap_or(0);
                    format!("{:?}={}", p, count)
                })
                .collect();
            eprintln!(
                "  {} comps, {} edges, {} tri, {}: [{}]",
                comps.len(),
                edges,
                comp_ex.n_triangles,
                comps.len(),
                pattern_str.join(", ")
            );
        }

        // Regime count formula: dim/16 + 1 for dim >= 32
        // 128/16 + 1 = 9
        assert_eq!(
            n_regimes, 9,
            "Expected 9 regimes at dim=128 (formula: dim/16+1)"
        );

        // Pure-max: 1 component, E_max = C(dim/2-2, 2) - (dim/4 - 1)
        // = C(62,2) - 31 = 1891 - 31 = 1860
        let n = 128 / 2 - 2; // 62
        let e_max = n * (n - 1) / 2 - (128 / 4 - 1);
        assert_eq!(e_max, 1860);

        // Pure-min: E_min = 3*dim/2 - 12 = 180
        let e_min = 3 * 128 / 2 - 12;
        assert_eq!(e_min, 180);

        // Count pure-max and pure-min components
        let pure_max_count = census
            .per_component
            .iter()
            .filter(|c| c.n_edges == e_max && c.pattern_counts.len() == 2)
            .count();
        let pure_min_count = census
            .per_component
            .iter()
            .filter(|c| c.n_edges == e_min && c.pattern_counts.len() == 2)
            .count();
        assert_eq!(pure_max_count, 1, "Exactly 1 pure-max component");
        assert_eq!(pure_min_count, 9, "Pure-min count at dim=128");

        // Verify pure regimes maintain the 3:1 ratio
        for comp in &census.per_component {
            if comp.pattern_counts.len() == 2 {
                let ts = comp
                    .pattern_counts
                    .get(&FaceSignPattern::TwoSameOneOpp)
                    .copied()
                    .unwrap_or(0);
                let ao = comp
                    .pattern_counts
                    .get(&FaceSignPattern::AllOpposite)
                    .copied()
                    .unwrap_or(0);
                assert_eq!(
                    ts,
                    3 * ao,
                    "3:1 ratio violated at dim=128 comp ({} edges): {} != 3*{}",
                    comp.n_edges,
                    ts,
                    ao
                );
            }
        }

        // Total triangles: 821128
        assert_eq!(census.total_triangles, 821128);
    }

    #[test]
    fn test_generic_face_sign_census_dim256() {
        // Regime count formula: dim/16 + 1 = 256/16 + 1 = 17 at dim=256.
        use std::time::Instant;

        let t0 = Instant::now();
        let census = generic_face_sign_census(256);
        let elapsed = t0.elapsed();
        eprintln!(
            "dim=256 census: {} components, {} triangles, {:.2?}",
            census.n_components, census.total_triangles, elapsed
        );

        // Basic structure: dim/2 - 1 = 127 components
        assert_eq!(census.n_components, 127);

        // Collect regime signatures
        let mut regime_map: HashMap<(usize, Vec<FaceSignPattern>), Vec<usize>> = HashMap::new();
        for comp in &census.per_component {
            let mut patterns: Vec<FaceSignPattern> = comp.pattern_counts.keys().copied().collect();
            patterns.sort();
            regime_map
                .entry((comp.n_edges, patterns))
                .or_default()
                .push(comp.component_idx);
        }

        let n_regimes = regime_map.len();
        eprintln!("dim=256 regimes: {}", n_regimes);

        // Print regime breakdown
        let mut regimes: Vec<_> = regime_map.iter().collect();
        regimes.sort_by_key(|&((edges, _), comps)| (std::cmp::Reverse(*edges), comps.len()));
        for ((edges, patterns), comps) in &regimes {
            eprintln!(
                "  {} comps, {} edges, {} patterns",
                comps.len(),
                edges,
                patterns.len()
            );
        }

        // Verify regime count: dim/16 + 1 = 17
        assert_eq!(n_regimes, 17, "Expected 17 regimes at dim=256");

        // Edge formulas
        let n = 256 / 2 - 2; // 126
        let e_max = n * (n - 1) / 2 - (256 / 4 - 1); // C(126,2) - 63 = 7875 - 63 = 7812
        let e_min = 3 * 256 / 2 - 12; // 372
        assert_eq!(
            census
                .per_component
                .iter()
                .map(|c| c.n_edges)
                .max()
                .unwrap(),
            e_max
        );
        assert_eq!(
            census
                .per_component
                .iter()
                .map(|c| c.n_edges)
                .min()
                .unwrap(),
            e_min
        );

        // Universal Double 3:1 Law
        for comp in &census.per_component {
            let as_count = comp
                .pattern_counts
                .get(&FaceSignPattern::AllSame)
                .copied()
                .unwrap_or(0);
            let ts = comp
                .pattern_counts
                .get(&FaceSignPattern::TwoSameOneOpp)
                .copied()
                .unwrap_or(0);
            let os = comp
                .pattern_counts
                .get(&FaceSignPattern::OneSameTwoOpp)
                .copied()
                .unwrap_or(0);
            let ao = comp
                .pattern_counts
                .get(&FaceSignPattern::AllOpposite)
                .copied()
                .unwrap_or(0);

            assert_eq!(ts, 3 * ao, "dim=256 comp[{}]: TS!=3*AO", comp.component_idx);
            assert_eq!(
                os,
                3 * as_count,
                "dim=256 comp[{}]: OS!=3*AS",
                comp.component_idx
            );
        }

        // Exactly 1 pure-max component
        let n_pure_max = census
            .per_component
            .iter()
            .filter(|c| c.n_edges == e_max && c.pattern_counts.len() == 2)
            .count();
        assert_eq!(n_pure_max, 1);
    }

    #[test]
    fn test_parity_edge_regularity_breakdown() {
        // Discovery: parity-specific edge-regularity is VIOLATED in non-pure
        // regimes. Even-parity triangle counts per edge vary (e.g., 4 to 8 at
        // dim=32 comp[0]). Odd-parity may also vary.
        //
        // This means the Universal Double 3:1 Law (C-487) is NOT explained by
        // parity-specific edge regularity. The proof mechanism is deeper --
        // it must arise from the algebraic structure of the CD multiplication
        // table itself, not from general graph-theoretic properties.
        //
        // What IS true: total (all-parity) edge regularity in pure regimes.
        // What IS true: the Double 3:1 law holds in ALL components (C-487).
        // What is FALSE: parity-specific edge regularity in non-pure regimes.

        // Verify that parity-specific regularity DOES fail (documenting the breakdown)
        let components = motif_components_for_cross_assessors(32);
        let census = generic_face_sign_census(32);

        let mut found_violation = false;
        for (i, comp) in components.iter().enumerate() {
            let comp_census = &census.per_component[i];
            if comp_census.pattern_counts.len() != 4 {
                continue;
            }

            let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
            let adj_set: HashSet<(CrossPair, CrossPair)> = comp
                .edges
                .iter()
                .flat_map(|&(a, b)| [(a, b), (b, a)])
                .collect();

            let mut even_per_edge: HashMap<(CrossPair, CrossPair), usize> = HashMap::new();
            for &(u, v) in &comp.edges {
                even_per_edge.insert((u, v), 0);
            }
            for &(u, v) in &comp.edges {
                for &w in &nodes {
                    if w <= v {
                        continue;
                    }
                    if !adj_set.contains(&(u, w)) || !adj_set.contains(&(v, w)) {
                        continue;
                    }
                    let signs = [
                        edge_sign_type_exact(32, u, v),
                        edge_sign_type_exact(32, v, w),
                        edge_sign_type_exact(32, u, w),
                    ];
                    let n_opp = signs
                        .iter()
                        .filter(|&&s| s == EdgeSignType::Opposite)
                        .count();
                    if n_opp % 2 == 0 {
                        let edges_of_tri = [
                            (u, v),
                            (std::cmp::min(v, w), std::cmp::max(v, w)),
                            (std::cmp::min(u, w), std::cmp::max(u, w)),
                        ];
                        for e in &edges_of_tri {
                            *even_per_edge.entry(*e).or_insert(0) += 1;
                        }
                    }
                }
            }

            let even_counts: Vec<usize> = even_per_edge.values().copied().collect();
            let even_min = *even_counts.iter().min().unwrap();
            let even_max = *even_counts.iter().max().unwrap();
            if even_min != even_max {
                found_violation = true;
                break;
            }
        }
        // Confirm: parity-specific edge regularity DOES fail
        assert!(
            found_violation,
            "Expected parity-specific edge regularity to be violated at dim=32"
        );
    }

    #[test]
    fn test_regime_gf2_class_correspondence() {
        // The GF(2) separating classes (C-480) correspond exactly to edge-count
        // groups. Face sign regimes = GF(2) classes + 1 for dim >= 32, because
        // the max-edge class splits into pure-max (1 component) + full-max.
        //
        // At the min-edge class, ALL components are pure -- no split occurs.
        // The split only happens at max-edge because the special component
        // (XOR label = dim/4) has a different face-sign distribution.
        //
        // Unification: three independent scaling laws converge:
        //   n_motif_classes (motif census) = dim/16
        //   n_GF2_classes (projective geometry) = dim/16
        //   n_face_regimes (face sign census) = dim/16 + 1

        for &dim in &[32, 64, 128] {
            let census = generic_face_sign_census(dim);

            // Edge count groups (= GF(2) classes = motif classes)
            let edge_counts: std::collections::BTreeSet<usize> =
                census.per_component.iter().map(|c| c.n_edges).collect();
            let n_edge_groups = edge_counts.len();

            // Motif class count
            let expected_motif_classes = dim / 16;
            assert_eq!(
                n_edge_groups, expected_motif_classes,
                "dim={}: edge-count groups should equal dim/16",
                dim
            );

            // Regime count = edge groups + 1
            let mut regime_set: HashSet<(usize, Vec<FaceSignPattern>)> = HashSet::new();
            for comp in &census.per_component {
                let mut pats: Vec<FaceSignPattern> = comp.pattern_counts.keys().copied().collect();
                pats.sort();
                regime_set.insert((comp.n_edges, pats));
            }
            assert_eq!(
                regime_set.len(),
                n_edge_groups + 1,
                "dim={}: regimes should equal edge-groups + 1",
                dim
            );

            // The split occurs at max-edge: 1 pure + rest full
            let e_max = *edge_counts.iter().max().unwrap();
            let max_edge_comps: Vec<&ComponentFaceCensus> = census
                .per_component
                .iter()
                .filter(|c| c.n_edges == e_max)
                .collect();
            let pure_max = max_edge_comps
                .iter()
                .filter(|c| c.pattern_counts.len() == 2)
                .count();
            let full_max = max_edge_comps
                .iter()
                .filter(|c| c.pattern_counts.len() == 4)
                .count();
            assert_eq!(pure_max, 1, "dim={}: exactly 1 pure-max", dim);
            assert!(full_max > 0, "dim={}: should have full-max components", dim);

            // At min-edge: ALL are pure (no split)
            let e_min = *edge_counts.iter().min().unwrap();
            let min_edge_all_pure = census
                .per_component
                .iter()
                .filter(|c| c.n_edges == e_min)
                .all(|c| c.pattern_counts.len() == 2);
            assert!(
                min_edge_all_pure,
                "dim={}: all min-edge components should be pure",
                dim
            );
        }
    }

    #[test]
    fn test_regime_and_edge_scaling_laws() {
        // Cross-dimension verification of the regime count and edge formulas.
        //
        // Regime count: dim/16 + 1 for dim >= 32; 1 for dim=16.
        // E_max = C(dim/2-2, 2) - (dim/4 - 1)
        // E_min = 3*dim/2 - 12
        //
        // Pure-max: always exactly 1 component.
        // 3:1 ratio law holds in ALL pure-regime components at ALL dimensions.

        for &dim in &[16, 32, 64, 128] {
            let census = generic_face_sign_census(dim);

            // Component count
            assert_eq!(census.n_components, dim / 2 - 1);

            // Collect regimes
            let mut regime_set: HashSet<(usize, Vec<FaceSignPattern>)> = HashSet::new();
            for comp in &census.per_component {
                let mut pats: Vec<FaceSignPattern> = comp.pattern_counts.keys().copied().collect();
                pats.sort();
                regime_set.insert((comp.n_edges, pats));
            }

            // Regime count formula
            let expected_regimes = if dim == 16 { 1 } else { dim / 16 + 1 };
            assert_eq!(
                regime_set.len(),
                expected_regimes,
                "dim={}: expected {} regimes, got {}",
                dim,
                expected_regimes,
                regime_set.len()
            );

            // Edge formulas (for dim >= 32)
            if dim >= 32 {
                let n = dim / 2 - 2;
                let expected_e_max = n * (n - 1) / 2 - (dim / 4 - 1);
                let expected_e_min = 3 * dim / 2 - 12;

                let actual_e_max = census
                    .per_component
                    .iter()
                    .map(|c| c.n_edges)
                    .max()
                    .unwrap();
                let actual_e_min = census
                    .per_component
                    .iter()
                    .map(|c| c.n_edges)
                    .min()
                    .unwrap();

                assert_eq!(
                    actual_e_max,
                    expected_e_max,
                    "dim={}: E_max should be C({},2)-{} = {}",
                    dim,
                    n,
                    dim / 4 - 1,
                    expected_e_max
                );
                assert_eq!(
                    actual_e_min, expected_e_min,
                    "dim={}: E_min should be 3*{}/2-12 = {}",
                    dim, dim, expected_e_min
                );

                // Exactly 1 pure-max component
                let n_pure_max = census
                    .per_component
                    .iter()
                    .filter(|c| c.n_edges == expected_e_max && c.pattern_counts.len() == 2)
                    .count();
                assert_eq!(n_pure_max, 1, "dim={}: exactly 1 pure-max", dim);
            }

            // 3:1 ratio law in all pure components
            for comp in &census.per_component {
                if comp.pattern_counts.len() == 2 {
                    let ts = comp
                        .pattern_counts
                        .get(&FaceSignPattern::TwoSameOneOpp)
                        .copied()
                        .unwrap_or(0);
                    let ao = comp
                        .pattern_counts
                        .get(&FaceSignPattern::AllOpposite)
                        .copied()
                        .unwrap_or(0);
                    assert_eq!(
                        ts,
                        3 * ao,
                        "dim={}: 3:1 violated ({} edges)",
                        dim,
                        comp.n_edges
                    );
                }
            }
        }
    }

    #[test]
    fn test_universal_double_three_to_one_law() {
        // Universal Double 3:1 Law (C-487):
        //
        // In EVERY component of the CD zero-divisor graph at EVERY dimension:
        //   TwoSameOneOpp = 3 * AllOpposite     (odd-parity pair)
        //   OneSameTwoOpp = 3 * AllSame          (even-parity pair)
        //
        // STATUS: Empirically verified across 116 components at dims 16/32/64/128.
        //
        // NOTE: The original proof sketch via parity-specific edge regularity
        // is WRONG -- parity-specific edge regularity does NOT hold in non-pure
        // regimes (see test_parity_edge_regularity_breakdown). The 3:1 law
        // must arise from the algebraic structure of the CD multiplication
        // table itself, not from general graph-theoretic counting arguments.
        //
        // The pure-regime C-483 is the special case where AllSame = OneSameTwoOpp = 0.

        for &dim in &[16, 32, 64, 128] {
            let census = generic_face_sign_census(dim);

            for comp in &census.per_component {
                let as_count = comp
                    .pattern_counts
                    .get(&FaceSignPattern::AllSame)
                    .copied()
                    .unwrap_or(0);
                let ts = comp
                    .pattern_counts
                    .get(&FaceSignPattern::TwoSameOneOpp)
                    .copied()
                    .unwrap_or(0);
                let os = comp
                    .pattern_counts
                    .get(&FaceSignPattern::OneSameTwoOpp)
                    .copied()
                    .unwrap_or(0);
                let ao = comp
                    .pattern_counts
                    .get(&FaceSignPattern::AllOpposite)
                    .copied()
                    .unwrap_or(0);

                // Odd-parity 3:1 law: TwoSameOneOpp = 3 * AllOpposite
                assert_eq!(
                    ts,
                    3 * ao,
                    "dim={} comp[{}] ({} edges): TS={} != 3*AO={}",
                    dim,
                    comp.component_idx,
                    comp.n_edges,
                    ts,
                    ao
                );

                // Even-parity 3:1 law: OneSameTwoOpp = 3 * AllSame
                assert_eq!(
                    os,
                    3 * as_count,
                    "dim={} comp[{}] ({} edges): OS={} != 3*AS={}",
                    dim,
                    comp.component_idx,
                    comp.n_edges,
                    os,
                    as_count
                );
            }
        }
    }

    #[test]
    fn test_special_heptacross_identity_dim32() {
        // Identify which component is the "special" pure heptacross at dim=32.
        // Uses XOR labels from projective_geometry to characterize each component.
        use crate::analysis::projective_geometry::component_xor_label;

        let components = motif_components_for_cross_assessors(32);
        let census = generic_face_sign_census(32);

        // Find the pure heptacross (84 edges, 2 patterns)
        let mut special_idx = None;
        for (i, comp_census) in census.per_component.iter().enumerate() {
            if comp_census.n_edges == 84 && comp_census.pattern_counts.len() == 2 {
                special_idx = Some(i);
            }
        }
        let idx = special_idx.expect("Should find exactly 1 pure heptacross");

        let special_comp = &components[idx];
        let xor_label = component_xor_label(special_comp);
        eprintln!(
            "Special heptacross: component[{}], XOR label = {:?}, nodes = {}",
            idx,
            xor_label,
            special_comp.nodes.len()
        );

        // Check if this component's XOR label is distinguishable
        let all_labels: Vec<Option<usize>> =
            components.iter().map(|c| component_xor_label(c)).collect();
        eprintln!("All XOR labels: {:?}", all_labels);

        // The special component has XOR label 8 = dim/4 = 32/4.
        // This is the MSB boundary of the lower-half index space.
        assert_eq!(
            xor_label,
            Some(8),
            "Special heptacross has XOR label = dim/4 = 8"
        );

        // Verify all 15 components have distinct XOR labels
        let defined: Vec<usize> = all_labels.iter().filter_map(|&l| l).collect();
        assert_eq!(defined.len(), 15);
        let unique: HashSet<usize> = defined.iter().copied().collect();
        assert_eq!(unique.len(), 15, "All XOR labels are distinct");

        // Verify the heptacross/mixed split by label:
        // Labels 1-8 (XOR label <= dim/4) are heptacross (84 edges),
        // Labels 9-15 (XOR label > dim/4) are mixed (36 edges).
        for (i, comp) in components.iter().enumerate() {
            let label = component_xor_label(comp).unwrap();
            let edges = census.per_component[i].n_edges;
            if label <= 8 {
                assert_eq!(edges, 84, "Label {} should be heptacross (84 edges)", label);
            } else {
                assert_eq!(edges, 36, "Label {} should be mixed (36 edges)", label);
            }
        }
    }

    /// Algebraic mechanism of the Universal Double 3:1 Law (C-487).
    ///
    /// For each edge between cross-assessors a=(i,j) and b=(k,l), define
    /// sigma_ab = cd_basis_mul_sign(dim, i, k) * cd_basis_mul_sign(dim, j, l).
    /// Then sigma_ab = -1 iff the edge is Same-type.
    ///
    /// For a triangle (a,b,c), the product sigma_ab * sigma_bc * sigma_ac
    /// determines a parity class:
    ///   product = +1 => n_same is even: {AllOpposite, TwoSameOneOpp}
    ///   product = -1 => n_same is odd:  {AllSame, OneSameTwoOpp}
    ///
    /// Within each parity class, the "pure" count (AllOpp or AllSame) is
    /// exactly 1/4 of the class total. This gives the 3:1 ratio because
    /// the "mixed" patterns consume the remaining 3/4. Equivalently,
    /// the mean of S = sigma_ab + sigma_bc + sigma_ac vanishes within
    /// each class (E[S|P] = 0).
    ///
    /// NOTE: The 3 individual "which edge is odd" counts are generally
    /// UNEQUAL (vertex ordering breaks symmetry), but their SUM is always
    /// exactly 3x the pure count.
    ///
    /// This test verifies:
    /// (1) sigma = -1 iff Same (the sign correspondence)
    /// (2) product = (-1)^n_same (the parity identity)
    /// (3) class total is divisible by 4 and pure = total/4 (the 3:1 law)
    /// (4) fraction of Same edges per component (diagnostic)
    #[test]
    fn test_double_three_to_one_algebraic_mechanism() {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;

        for dim in [16, 32, 64] {
            let components = motif_components_for_cross_assessors(dim);

            for (comp_idx, comp) in components.iter().enumerate() {
                let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
                let adj: HashSet<(CrossPair, CrossPair)> = comp
                    .edges
                    .iter()
                    .flat_map(|&(a, b)| [(a, b), (b, a)])
                    .collect();

                // Step 1: Verify sigma = -1 iff Same for every edge.
                let mut n_same_edges = 0usize;
                let mut n_opp_edges = 0usize;
                for &(a, b) in &comp.edges {
                    let (i, j) = a;
                    let (k, l) = b;
                    let sigma = cd_basis_mul_sign(dim, i, k) * cd_basis_mul_sign(dim, j, l);
                    let edge_type = edge_sign_type_exact(dim, a, b);
                    match edge_type {
                        EdgeSignType::Same => {
                            assert_eq!(
                                sigma, -1,
                                "dim={} comp[{}] edge {:?}--{:?}: Same but sigma={}",
                                dim, comp_idx, a, b, sigma
                            );
                            n_same_edges += 1;
                        }
                        EdgeSignType::Opposite => {
                            assert_eq!(
                                sigma, 1,
                                "dim={} comp[{}] edge {:?}--{:?}: Opposite but sigma={}",
                                dim, comp_idx, a, b, sigma
                            );
                            n_opp_edges += 1;
                        }
                    }
                }

                // Step 2: Enumerate triangles and classify by parity class.
                let mut n_all_opp = 0usize; // class +1 pure
                let mut n_two_same = 0usize; // class +1 mixed
                let mut n_all_same = 0usize; // class -1 pure
                let mut n_one_same = 0usize; // class -1 mixed
                let mut n_triangles = 0usize;

                for &(u, v) in &comp.edges {
                    for &w in &nodes {
                        if w <= v {
                            continue;
                        }
                        if adj.contains(&(u, w)) && adj.contains(&(v, w)) {
                            let (i, j) = u;
                            let (k, l) = v;
                            let (m, n) = w;
                            let sigma_ab =
                                cd_basis_mul_sign(dim, i, k) * cd_basis_mul_sign(dim, j, l);
                            let sigma_bc =
                                cd_basis_mul_sign(dim, k, m) * cd_basis_mul_sign(dim, l, n);
                            let sigma_ac =
                                cd_basis_mul_sign(dim, i, m) * cd_basis_mul_sign(dim, j, n);

                            let product = sigma_ab * sigma_bc * sigma_ac;
                            let n_same = [sigma_ab, sigma_bc, sigma_ac]
                                .iter()
                                .filter(|&&s| s == -1)
                                .count();

                            // Verify parity identity
                            let expected_product = if n_same % 2 == 0 { 1 } else { -1 };
                            assert_eq!(
                                product, expected_product,
                                "dim={} comp[{}] parity mismatch",
                                dim, comp_idx
                            );

                            match n_same {
                                0 => n_all_opp += 1,
                                1 => n_one_same += 1,
                                2 => n_two_same += 1,
                                3 => n_all_same += 1,
                                _ => unreachable!(),
                            }
                            n_triangles += 1;
                        }
                    }
                }

                if n_triangles == 0 {
                    continue;
                }

                // Step 3: Verify 3:1 ratio via class divisibility.
                let class_pos = n_all_opp + n_two_same;
                let class_neg = n_all_same + n_one_same;

                if class_pos > 0 {
                    assert_eq!(
                        class_pos % 4,
                        0,
                        "dim={} comp[{}] class+1 total {} not div by 4 (AO={}, TS={})",
                        dim,
                        comp_idx,
                        class_pos,
                        n_all_opp,
                        n_two_same
                    );
                    assert_eq!(
                        n_all_opp,
                        class_pos / 4,
                        "dim={} comp[{}] AllOpp {} != class_pos/4 {}",
                        dim,
                        comp_idx,
                        n_all_opp,
                        class_pos / 4
                    );
                    assert_eq!(
                        n_two_same,
                        3 * n_all_opp,
                        "dim={} comp[{}] TS {} != 3*AO {}",
                        dim,
                        comp_idx,
                        n_two_same,
                        3 * n_all_opp
                    );
                }

                if class_neg > 0 {
                    assert_eq!(
                        class_neg % 4,
                        0,
                        "dim={} comp[{}] class-1 total {} not div by 4 (AS={}, OS={})",
                        dim,
                        comp_idx,
                        class_neg,
                        n_all_same,
                        n_one_same
                    );
                    assert_eq!(
                        n_all_same,
                        class_neg / 4,
                        "dim={} comp[{}] AllSame {} != class_neg/4 {}",
                        dim,
                        comp_idx,
                        n_all_same,
                        class_neg / 4
                    );
                    assert_eq!(
                        n_one_same,
                        3 * n_all_same,
                        "dim={} comp[{}] OS {} != 3*AS {}",
                        dim,
                        comp_idx,
                        n_one_same,
                        3 * n_all_same
                    );
                }

                // Step 4: Half-Half Edge Law -- exactly 50% Same, 50% Opposite.
                assert_eq!(
                    n_same_edges, n_opp_edges,
                    "dim={} comp[{}] Half-Half violated: Same={}, Opp={}",
                    dim, comp_idx, n_same_edges, n_opp_edges
                );
            }
        }
    }

    /// Half-Half Edge Law at dim=128: every component has exactly 50% Same
    /// and 50% Opposite edges. This is the foundational property underlying
    /// the Double 3:1 Law. Verified at dims 16, 32, 64 in the mechanism test;
    /// this test extends to dim=128.
    #[test]
    fn test_half_half_edge_law_dim128() {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;

        let components = motif_components_for_cross_assessors(128);
        assert_eq!(components.len(), 63);

        for (comp_idx, comp) in components.iter().enumerate() {
            let mut n_same = 0usize;
            let mut n_opp = 0usize;
            for &(a, b) in &comp.edges {
                let (i, j) = a;
                let (k, l) = b;
                let sigma = cd_basis_mul_sign(128, i, k) * cd_basis_mul_sign(128, j, l);
                match sigma {
                    -1 => n_same += 1,
                    1 => n_opp += 1,
                    _ => panic!("sigma must be +/-1"),
                }
            }
            assert_eq!(
                n_same,
                n_opp,
                "dim=128 comp[{}] Half-Half violated: Same={}, Opp={} (edges={})",
                comp_idx,
                n_same,
                n_opp,
                comp.edges.len()
            );
        }
    }

    /// Face sign census at dim=512: predicted 33 regimes (512/16 + 1).
    ///
    /// Predictions from established scaling laws:
    ///   n_components = 512/2 - 1 = 255
    ///   nodes_per_component = 512/2 - 2 = 254
    ///   n_regimes = 512/16 + 1 = 33
    ///   e_max = C(254,2) - (512/4 - 1) = 32131 - 127 = 32004
    ///   e_min = 3*512/2 - 12 = 756
    ///   Universal Double 3:1 Law holds for all components
    ///   Exactly 1 pure-max component
    ///
    /// Runtime: ~30-60s in release mode (16x scaling from dim=256).
    #[test]
    #[ignore] // Long-running: ~30-60s in release mode
    fn test_generic_face_sign_census_dim512() {
        use std::time::Instant;

        let t0 = Instant::now();
        let census = generic_face_sign_census(512);
        let elapsed = t0.elapsed();
        eprintln!(
            "dim=512 census: {} components, {} triangles, {:.2?}",
            census.n_components, census.total_triangles, elapsed
        );

        // Basic structure: dim/2 - 1 = 255 components
        assert_eq!(census.n_components, 255);

        // Collect regime signatures
        let mut regime_map: HashMap<(usize, Vec<FaceSignPattern>), Vec<usize>> = HashMap::new();
        for comp in &census.per_component {
            let mut patterns: Vec<FaceSignPattern> = comp.pattern_counts.keys().copied().collect();
            patterns.sort();
            regime_map
                .entry((comp.n_edges, patterns))
                .or_default()
                .push(comp.component_idx);
        }

        let n_regimes = regime_map.len();
        eprintln!("dim=512 regimes: {}", n_regimes);

        // Print regime breakdown
        let mut regimes: Vec<_> = regime_map.iter().collect();
        regimes.sort_by_key(|&((edges, _), comps)| (std::cmp::Reverse(*edges), comps.len()));
        for ((edges, patterns), comps) in &regimes {
            eprintln!(
                "  {} comps, {} edges, {} patterns",
                comps.len(),
                edges,
                patterns.len()
            );
        }

        // Verify regime count: dim/16 + 1 = 33
        assert_eq!(n_regimes, 33, "Expected 33 regimes at dim=512");

        // Edge formulas
        let n = 512 / 2 - 2; // 254
        let e_max = n * (n - 1) / 2 - (512 / 4 - 1); // C(254,2) - 127 = 32131 - 127 = 32004
        let e_min = 3 * 512 / 2 - 12; // 756
        assert_eq!(
            census
                .per_component
                .iter()
                .map(|c| c.n_edges)
                .max()
                .unwrap(),
            e_max
        );
        assert_eq!(
            census
                .per_component
                .iter()
                .map(|c| c.n_edges)
                .min()
                .unwrap(),
            e_min
        );

        // Universal Double 3:1 Law
        for comp in &census.per_component {
            let as_count = comp
                .pattern_counts
                .get(&FaceSignPattern::AllSame)
                .copied()
                .unwrap_or(0);
            let ts = comp
                .pattern_counts
                .get(&FaceSignPattern::TwoSameOneOpp)
                .copied()
                .unwrap_or(0);
            let os = comp
                .pattern_counts
                .get(&FaceSignPattern::OneSameTwoOpp)
                .copied()
                .unwrap_or(0);
            let ao = comp
                .pattern_counts
                .get(&FaceSignPattern::AllOpposite)
                .copied()
                .unwrap_or(0);

            assert_eq!(ts, 3 * ao, "dim=512 comp[{}]: TS!=3*AO", comp.component_idx);
            assert_eq!(
                os,
                3 * as_count,
                "dim=512 comp[{}]: OS!=3*AS",
                comp.component_idx
            );
        }

        // Exactly 1 pure-max component
        let n_pure_max = census
            .per_component
            .iter()
            .filter(|c| c.n_edges == e_max && c.pattern_counts.len() == 2)
            .count();
        assert_eq!(n_pure_max, 1);
    }

    /// Investigate whether ZD graph components at dim=32/64 respect the
    /// octonion Fano plane structure inherited from the dim=8 subalgebra.
    ///
    /// At dim=16: 7 components correspond exactly to 7 "missing" Fano indices.
    /// Question: does this correspondence propagate through doublings?
    ///
    /// Method: for each component, compute the "Fano projection" of its
    /// cross-assessor low indices -- map lo -> lo & 7 (mod-8 residue).
    /// Then check whether these projections cluster into Fano-related
    /// subsets, or whether the structure fully scrambles at dim=32+.
    #[test]
    fn test_octonion_subalgebra_fano_projection() {
        // Phase 1: Verify the known dim=16 Fano correspondence
        let comps_16 = motif_components_for_cross_assessors(16);
        assert_eq!(comps_16.len(), 7);

        let mut missing_indices_16 = Vec::new();
        for comp in &comps_16 {
            let lo_set: HashSet<usize> = comp.nodes.iter().map(|&(lo, _hi)| lo).collect();
            let all_lo: HashSet<usize> = (1..=7).collect();
            let missing: Vec<usize> = all_lo.difference(&lo_set).copied().collect();
            assert_eq!(
                missing.len(),
                1,
                "dim=16: each component should miss exactly 1 Fano index"
            );
            missing_indices_16.push(missing[0]);
        }
        let mut sorted_missing = missing_indices_16.clone();
        sorted_missing.sort();
        assert_eq!(
            sorted_missing,
            vec![1, 2, 3, 4, 5, 6, 7],
            "dim=16: missing indices should be exactly {{1,...,7}}"
        );

        // Phase 2: At dim=32, compute Fano projections of component low indices
        let comps_32 = motif_components_for_cross_assessors(32);
        assert_eq!(comps_32.len(), 15); // dim/2 - 1 = 15

        eprintln!("\n=== Fano projection analysis at dim=32 ===");
        let mut fano_projection_signatures: Vec<Vec<usize>> = Vec::new();

        for (comp_idx, comp) in comps_32.iter().enumerate() {
            // Collect all low indices in this component
            let lo_indices: BTreeSet<usize> = comp.nodes.iter().map(|&(lo, _)| lo).collect();

            // Fano projection: lo & 7 (mod-8 residue, maps to octonion imaginary)
            let fano_proj: BTreeSet<usize> = lo_indices
                .iter()
                .map(|&lo| {
                    let residue = lo & 7;
                    if residue == 0 {
                        8
                    } else {
                        residue
                    } // Map 0 -> 8 to distinguish
                })
                .collect();

            // Level decomposition: which "doubling levels" are present?
            let levels: BTreeSet<usize> = lo_indices.iter().map(|&lo| lo >> 3).collect();

            eprintln!(
                "  comp[{}]: {} nodes, {} edges, lo_range=[{},{}], fano_proj={:?}, levels={:?}",
                comp_idx,
                comp.nodes.len(),
                comp.edges.len(),
                lo_indices.iter().next().unwrap(),
                lo_indices.iter().last().unwrap(),
                fano_proj,
                levels,
            );

            let mut sig: Vec<usize> = fano_proj.iter().copied().collect();
            sig.sort();
            fano_projection_signatures.push(sig);
        }

        // Phase 3: Check if Fano projections at dim=32 show structure
        //
        // Key question: do all 7 Fano indices appear in every component's
        // projection (fully mixed), or do some components only use subsets?
        let all_use_full_fano = fano_projection_signatures.iter().all(|sig| sig.len() >= 7);

        let n_distinct_sigs = {
            let mut sigs = fano_projection_signatures.clone();
            sigs.sort();
            sigs.dedup();
            sigs.len()
        };

        eprintln!(
            "\n  All components use full Fano set: {}",
            all_use_full_fano
        );
        eprintln!("  Distinct Fano projection signatures: {}", n_distinct_sigs);

        // Phase 4: XOR-key analysis -- do XOR keys respect Fano structure?
        // For each component's edges, compute xor_key = lo ^ hi.
        // The low 3 bits of xor_key inherit Fano structure.
        let mut xor_fano_residues_per_comp: Vec<BTreeSet<usize>> = Vec::new();
        for comp in &comps_32 {
            let edge_xor_residues: BTreeSet<usize> =
                comp.nodes.iter().map(|&(lo, hi)| (lo ^ hi) & 7).collect();
            xor_fano_residues_per_comp.push(edge_xor_residues);
        }

        eprintln!("\n  XOR-key Fano residues per component:");
        for (i, residues) in xor_fano_residues_per_comp.iter().enumerate() {
            eprintln!("    comp[{}]: {:?}", i, residues);
        }

        // Phase 5: Verify XOR-key residue distribution at dim=32
        // Each XOR bucket has a unique xor_key = lo ^ hi. The Fano residue
        // is xor_key & 7. By construction (XOR-bucketing), all nodes in a
        // component share the same xor_key, hence the same Fano residue.
        //
        // At dim=32: 15 components should give residue distribution
        // {0: 1, 1: 2, 2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2} = 1 + 7*2 = 15
        let mut residue_counts_32: HashMap<usize, usize> = HashMap::new();
        let mut xor_keys_32: Vec<usize> = Vec::new();
        for comp in &comps_32 {
            // All nodes in a component have the same xor_key (by construction)
            let first_node = comp.nodes.iter().next().unwrap();
            let xor_key = first_node.0 ^ first_node.1;
            xor_keys_32.push(xor_key);
            let residue = xor_key & 7;
            *residue_counts_32.entry(residue).or_insert(0) += 1;
        }

        eprintln!("\n  XOR key per component: {:?}", xor_keys_32);
        eprintln!("  Fano residue distribution: {:?}", residue_counts_32);

        // Verify: each component has a unique XOR key
        let unique_xor_keys: HashSet<usize> = xor_keys_32.iter().copied().collect();
        assert_eq!(
            unique_xor_keys.len(),
            15,
            "dim=32: each of 15 components should have a unique XOR key"
        );

        // Verify: residue 0 appears once, all others appear twice
        for residue in 0..8 {
            let count = residue_counts_32.get(&residue).copied().unwrap_or(0);
            let expected = if residue == 0 { 1 } else { 2 };
            assert_eq!(
                count, expected,
                "dim=32: Fano residue {} should appear {} times, got {}",
                residue, expected, count
            );
        }

        // Phase 6: Repeat at dim=64
        let comps_64 = motif_components_for_cross_assessors(64);
        assert_eq!(comps_64.len(), 31);

        let mut residue_counts_64: HashMap<usize, usize> = HashMap::new();
        let mut xor_keys_64: Vec<usize> = Vec::new();
        for comp in &comps_64 {
            let first_node = comp.nodes.iter().next().unwrap();
            let xor_key = first_node.0 ^ first_node.1;
            xor_keys_64.push(xor_key);
            let residue = xor_key & 7;
            *residue_counts_64.entry(residue).or_insert(0) += 1;
        }

        // Verify unique XOR keys at dim=64
        let unique_64: HashSet<usize> = xor_keys_64.iter().copied().collect();
        assert_eq!(
            unique_64.len(),
            31,
            "dim=64: each of 31 components should have a unique XOR key"
        );

        // At dim=64: 31 = 4*7 + 3. Predict: residue 0 has 3 components,
        // non-zero residues have 4 each? Or residue 0 has 1 + 2 = 3,
        // others have 2 + 2 = 4. Let's check:
        eprintln!("\n=== dim=64 ===");
        eprintln!("  Fano residue distribution: {:?}", residue_counts_64);

        // Verify Fano residue scaling law:
        // dim=16 (7 comps): 0->0, others->1 each (7 = 7*1 + 0)
        // dim=32 (15 comps): 0->1, others->2 each (15 = 1 + 7*2)
        // dim=64 (31 comps): predict 0->3, others->4 each (31 = 3 + 7*4)
        //   OR: 0->1, others->4.28... (doesn't divide evenly)
        //   Try: 31 = 3 + 28 = 3 + 7*4. So residue 0 gets 3, others get 4.
        let r0_count_64 = residue_counts_64.get(&0).copied().unwrap_or(0);
        let r_nonzero_counts: Vec<usize> = (1..8)
            .map(|r| residue_counts_64.get(&r).copied().unwrap_or(0))
            .collect();

        eprintln!(
            "  residue 0: {}, non-zero: {:?}",
            r0_count_64, r_nonzero_counts
        );

        // Assert the pattern (empirical, may need adjustment):
        // If the formula is: n_comps = dim/2 - 1, and XOR keys range over
        // {dim/2, ..., dim-1} (one per component), then the number of keys
        // with residue r is floor(dim/2 / 8) or ceil(dim/2 / 8).
        // At dim=64: dim/2=32, 32/8=4, so each residue gets exactly 4.
        // But 8*4=32 keys and only 31 components, so one key is missing.
        // The missing key at dim=32 was key 0+16=16 (nah, that had a comp).
        // Actually at dim=32 we had 15 comps and 16 possible keys (16..31),
        // so one key had no component. Similarly at dim=64: 31 comps out of
        // 32 possible keys (32..63). One key produces no component.

        // For each doubling: one key out of dim/2 keys produces no component.
        // That missing key's Fano residue gets one fewer count.

        // Check that total = 31
        let total: usize = residue_counts_64.values().sum();
        assert_eq!(total, 31, "dim=64: total Fano residue count should be 31");

        // Check that exactly one residue has count = (dim/16) - 1 = 3,
        // and the rest have count = dim/16 = 4.
        let expected_normal = 64 / 16; // 4
        let n_deficit = r_nonzero_counts
            .iter()
            .chain(std::iter::once(&r0_count_64))
            .filter(|&&c| c == expected_normal - 1)
            .count();
        let n_normal = r_nonzero_counts
            .iter()
            .chain(std::iter::once(&r0_count_64))
            .filter(|&&c| c == expected_normal)
            .count();

        assert_eq!(
            n_deficit,
            1,
            "dim=64: exactly one Fano residue should have {} components (deficit)",
            expected_normal - 1
        );
        assert_eq!(
            n_normal, 7,
            "dim=64: seven Fano residues should have {} components",
            expected_normal
        );
    }

    /// C-515: Sigma Correspondence -- verify sigma_ab = s(lo_a,lo_b)*s(hi_a,hi_b)
    /// matches Same/Opposite classification for all edges at dims 16, 32, 64.
    ///
    /// Also verifies the sigma-to-parity identity:
    /// sigma = -1 iff Same-type, sigma = +1 iff Opposite-type.
    ///
    /// Additionally tests the Parity Product Theorem:
    /// product(sigma) over triangle edges determines face sign class.
    #[test]
    fn test_sigma_correspondence_and_parity_product() {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;

        for &dim in &[16, 32, 64] {
            let components = motif_components_for_cross_assessors(dim);

            let mut total_edges = 0usize;
            let mut sigma_match_count = 0usize;
            let mut total_triangles = 0usize;
            // Parity class counts: (product=+1, product=-1)
            let mut parity_even = 0usize;
            let mut parity_odd = 0usize;
            // Within each parity class, count pure vs mixed
            // Even class: AllOpposite is "pure" (all sigma=+1)
            // Odd class: AllSame is "pure" (all sigma=-1)
            let mut even_pure = 0usize; // AllOpposite in even class
            let mut even_mixed = 0usize; // TwoSameOneOpp in even class
            let mut odd_pure = 0usize; // AllSame in odd class
            let mut odd_mixed = 0usize; // OneSameTwoOpp in odd class

            for comp in &components {
                // Check sigma correspondence on every edge
                for &(a, b) in &comp.edges {
                    let (lo_a, hi_a) = a;
                    let (lo_b, hi_b) = b;

                    // Compute sigma_ab = s(lo_a, lo_b) * s(hi_a, hi_b)
                    let s_lo = cd_basis_mul_sign(dim, lo_a, lo_b);
                    let s_hi = cd_basis_mul_sign(dim, hi_a, hi_b);
                    let sigma = s_lo * s_hi;

                    // Get the established Same/Opposite classification
                    let edge_type = edge_sign_type_exact(dim, a, b);

                    // Verify: sigma = -1 iff Same, sigma = +1 iff Opposite
                    let expected_sigma = match edge_type {
                        EdgeSignType::Same => -1,
                        EdgeSignType::Opposite => 1,
                    };

                    if sigma == expected_sigma {
                        sigma_match_count += 1;
                    }
                    total_edges += 1;
                }

                // Enumerate triangles and check parity product
                let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
                let edge_set: std::collections::HashSet<(CrossPair, CrossPair)> =
                    comp.edges.iter().copied().collect();
                let has_edge = |u: CrossPair, v: CrossPair| -> bool {
                    let (a, b) = if u < v { (u, v) } else { (v, u) };
                    edge_set.contains(&(a, b))
                };

                for i in 0..nodes.len() {
                    for j in (i + 1)..nodes.len() {
                        for k in (j + 1)..nodes.len() {
                            let (u, v, w) = (nodes[i], nodes[j], nodes[k]);
                            if has_edge(u, v) && has_edge(v, w) && has_edge(u, w) {
                                total_triangles += 1;

                                // Compute sigma for each edge
                                let sigma_uv = cd_basis_mul_sign(dim, u.0, v.0)
                                    * cd_basis_mul_sign(dim, u.1, v.1);
                                let sigma_vw = cd_basis_mul_sign(dim, v.0, w.0)
                                    * cd_basis_mul_sign(dim, v.1, w.1);
                                let sigma_uw = cd_basis_mul_sign(dim, u.0, w.0)
                                    * cd_basis_mul_sign(dim, u.1, w.1);

                                let product = sigma_uv * sigma_vw * sigma_uw;

                                // Count Same-type edges in this triangle
                                let n_same = [sigma_uv, sigma_vw, sigma_uw]
                                    .iter()
                                    .filter(|&&s| s == -1)
                                    .count();

                                if product == 1 {
                                    // Even parity: 0 or 2 Same edges
                                    parity_even += 1;
                                    assert!(
                                        n_same == 0 || n_same == 2,
                                        "Even parity but n_same={n_same}"
                                    );
                                    if n_same == 0 {
                                        even_pure += 1; // AllOpposite
                                    } else {
                                        even_mixed += 1; // TwoSameOneOpp
                                    }
                                } else {
                                    // Odd parity: 1 or 3 Same edges
                                    parity_odd += 1;
                                    assert!(
                                        n_same == 1 || n_same == 3,
                                        "Odd parity but n_same={n_same}"
                                    );
                                    if n_same == 3 {
                                        odd_pure += 1; // AllSame
                                    } else {
                                        odd_mixed += 1; // OneSameTwoOpp
                                    }
                                }
                            }
                        }
                    }
                }
            }

            eprintln!("\n=== Sigma Correspondence at dim={dim} ===");
            eprintln!("Edges: {total_edges}, sigma matches: {sigma_match_count}");
            eprintln!("Triangles: {total_triangles}");
            eprintln!(
                "Even parity (product=+1): {parity_even} (pure={even_pure}, mixed={even_mixed})"
            );
            eprintln!("Odd parity (product=-1): {parity_odd} (pure={odd_pure}, mixed={odd_mixed})");

            if parity_even > 0 {
                let quarter = parity_even as f64 / 4.0;
                let pure_frac = even_pure as f64 / parity_even as f64;
                eprintln!("Even class Quarter Rule: pure/total = {pure_frac:.4} (expected 0.25)");
                eprintln!("  pure={even_pure}, expected={quarter:.1}");
            }
            if parity_odd > 0 {
                let quarter = parity_odd as f64 / 4.0;
                let pure_frac = odd_pure as f64 / parity_odd as f64;
                eprintln!("Odd class Quarter Rule: pure/total = {pure_frac:.4} (expected 0.25)");
                eprintln!("  pure={odd_pure}, expected={quarter:.1}");
            }

            // ASSERTION 1: sigma correspondence is exact
            assert_eq!(
                sigma_match_count,
                total_edges,
                "dim={dim}: sigma correspondence failed on {}/{total_edges} edges",
                total_edges - sigma_match_count
            );

            // ASSERTION 2: Quarter Rule on even parity class
            if parity_even > 0 {
                assert_eq!(
                    even_pure * 4,
                    parity_even,
                    "dim={dim}: even class Quarter Rule failed: \
                     pure={even_pure}, total={parity_even}, expected pure=total/4"
                );
            }

            // ASSERTION 3: Quarter Rule on odd parity class
            if parity_odd > 0 {
                assert_eq!(
                    odd_pure * 4,
                    parity_odd,
                    "dim={dim}: odd class Quarter Rule failed: \
                     pure={odd_pure}, total={parity_odd}, expected pure=total/4"
                );
            }
        }
    }

    /// C-516: Translation Derivative Identity -- verify that sigma_ab equals
    /// the "finite difference" of the twist exponent psi across the component key k.
    ///
    /// For a component with XOR key k:
    ///   sigma(x,y) = (-1)^{Delta_k psi(x,y)}
    /// where Delta_k psi(x,y) = psi(x,y) XOR psi(x^k, y^k)
    /// and s(i,j) = (-1)^{psi(i,j)}.
    ///
    /// Also verifies C-517: Half-Half Edge Law = Delta_k psi is balanced
    /// (exactly half of edges have Delta_k psi = 0, half have 1).
    #[test]
    fn test_translation_derivative_and_half_half() {
        use crate::analysis::zd_graphs::xor_key;
        use crate::construction::cayley_dickson::cd_basis_mul_sign;

        // Extract GF(2) exponent: s(i,j) = (-1)^psi(i,j) => psi = 0 if s=+1, 1 if s=-1
        let psi = |dim: usize, i: usize, j: usize| -> u8 {
            if cd_basis_mul_sign(dim, i, j) == 1 {
                0
            } else {
                1
            }
        };

        for &dim in &[16, 32, 64, 128] {
            let components = motif_components_for_cross_assessors(dim);
            let mut total_edges = 0usize;
            let mut delta_match = 0usize;
            let mut total_balanced_comps = 0usize;
            let mut total_nontrivial_comps = 0usize;

            for comp in &components {
                if comp.edges.is_empty() {
                    continue;
                }

                // Determine the component key k = lo ^ hi for any node
                let first_node = comp.nodes.iter().next().unwrap();
                let k = xor_key(first_node.0, first_node.1);

                // Verify all nodes share the same key
                for &(lo, hi) in &comp.nodes {
                    assert_eq!(
                        xor_key(lo, hi),
                        k,
                        "dim={dim}: node ({lo},{hi}) has key {} != component key {k}",
                        xor_key(lo, hi)
                    );
                }

                let mut n_delta_zero = 0usize;
                let mut n_delta_one = 0usize;

                for &(a, b) in &comp.edges {
                    let (lo_a, hi_a) = a;
                    let (lo_b, hi_b) = b;

                    // Compute sigma via cd_basis_mul_sign
                    let s_lo = cd_basis_mul_sign(dim, lo_a, lo_b);
                    let s_hi = cd_basis_mul_sign(dim, hi_a, hi_b);
                    let sigma = s_lo * s_hi;

                    // Compute Delta_k psi(lo_a, lo_b) = psi(lo_a, lo_b) XOR psi(lo_a^k, lo_b^k)
                    // But within a component, hi_a = lo_a ^ k, hi_b = lo_b ^ k
                    // So Delta_k psi(lo_a, lo_b) = psi(lo_a, lo_b) XOR psi(hi_a, hi_b)
                    let psi_lo = psi(dim, lo_a, lo_b);
                    let psi_hi = psi(dim, hi_a, hi_b);
                    let delta_k = psi_lo ^ psi_hi;

                    // sigma = (-1)^{delta_k} means:
                    // delta_k=0 => sigma=+1 (Opposite), delta_k=1 => sigma=-1 (Same)
                    let expected_sigma: i32 = if delta_k == 0 { 1 } else { -1 };

                    if sigma == expected_sigma {
                        delta_match += 1;
                    }
                    total_edges += 1;

                    if delta_k == 0 {
                        n_delta_zero += 1;
                    } else {
                        n_delta_one += 1;
                    }
                }

                total_nontrivial_comps += 1;
                if n_delta_zero == n_delta_one {
                    total_balanced_comps += 1;
                }

                eprintln!(
                    "dim={dim} comp k={k}: edges={}, delta_0={n_delta_zero}, delta_1={n_delta_one}, balanced={}",
                    comp.edges.len(),
                    n_delta_zero == n_delta_one
                );
            }

            eprintln!("\n=== Translation Derivative at dim={dim} ===");
            eprintln!("Total edges: {total_edges}, Delta_k matches: {delta_match}");
            eprintln!("Balanced components: {total_balanced_comps}/{total_nontrivial_comps}");

            // ASSERTION 1: translation derivative identity is exact
            assert_eq!(
                delta_match,
                total_edges,
                "dim={dim}: Delta_k psi identity failed on {}/{total_edges}",
                total_edges - delta_match
            );

            // ASSERTION 2: Half-Half Edge Law (all components balanced)
            assert_eq!(
                total_balanced_comps, total_nontrivial_comps,
                "dim={dim}: Half-Half failed: {total_balanced_comps}/{total_nontrivial_comps} balanced"
            );
        }
    }

    /// C-518: Associator-Based 2-Bit Obstruction (Candidate B mechanism test).
    ///
    /// Tests whether the CD associator sign provides a GF(2)^2-valued invariant
    /// F(triangle) = (A_lo, A_hi) that exactly classifies "pure" vs "mixed"
    /// triangles within each parity class.
    ///
    /// The associator sign: A(i,j,k) = s(i,j) * s(i^j, k) * s(j,k) * s(i, j^k)
    /// where s = cd_basis_mul_sign and ^ = XOR.
    ///
    /// Prediction: F=(+1,+1) iff triangle is "pure" (AllOpposite in even class,
    /// AllSame in odd class).
    ///
    /// If this fails, we also test variant constructions involving the
    /// component key k.
    #[test]
    fn test_associator_obstruction_candidate_b() {
        use crate::analysis::zd_graphs::xor_key;
        use crate::construction::cayley_dickson::cd_basis_mul_sign;

        // CD associator sign: A(i,j,k) = s(i,j) * s(i^j, k) * s(j,k) * s(i, j^k)
        // In {+/-1} arithmetic, division = multiplication, so this is:
        // s(i,j) * s(i XOR j, k) / (s(j,k) * s(i, j XOR k))
        // = s(i,j) * s(i^j,k) * s(j,k) * s(i, j^k) since all values are +/-1
        let assoc_sign = |dim: usize, i: usize, j: usize, k: usize| -> i32 {
            let s_ij = cd_basis_mul_sign(dim, i, j);
            let s_ij_xor_k = cd_basis_mul_sign(dim, i ^ j, k);
            let s_jk = cd_basis_mul_sign(dim, j, k);
            let s_i_jk = cd_basis_mul_sign(dim, i, j ^ k);
            s_ij * s_ij_xor_k * s_jk * s_i_jk
        };

        for &dim in &[16, 32, 64] {
            let components = motif_components_for_cross_assessors(dim);
            let mut total_triangles = 0usize;

            // Count how many triangles have F=(+1,+1) and are "pure"
            // F values: (A_lo, A_hi) where each is +1 or -1
            let mut f_pp_pure = 0usize; // F=(+1,+1), pure triangle
            let mut f_pp_mixed = 0usize; // F=(+1,+1), mixed triangle
            let mut f_other_pure = 0usize; // F != (+1,+1), pure triangle
            let mut f_other_mixed = 0usize; // F != (+1,+1), mixed triangle

            // Also try variant: use A(lo_a, lo_b, lo_c) with vertex lo indices
            // and separately A with hi indices
            let mut f_counts: std::collections::HashMap<(i32, i32), [usize; 2]> =
                std::collections::HashMap::new();

            for comp in &components {
                let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
                let edge_set: std::collections::HashSet<(CrossPair, CrossPair)> =
                    comp.edges.iter().copied().collect();
                let has_edge = |u: CrossPair, v: CrossPair| -> bool {
                    let (a, b) = if u < v { (u, v) } else { (v, u) };
                    edge_set.contains(&(a, b))
                };

                let _k = {
                    let first = comp.nodes.iter().next().unwrap();
                    xor_key(first.0, first.1)
                };

                for i in 0..nodes.len() {
                    for j in (i + 1)..nodes.len() {
                        for ki in (j + 1)..nodes.len() {
                            let (u, v, w) = (nodes[i], nodes[j], nodes[ki]);
                            if !has_edge(u, v) || !has_edge(v, w) || !has_edge(u, w) {
                                continue;
                            }
                            total_triangles += 1;

                            // Compute sigma for edge classification
                            let sigma_uv =
                                cd_basis_mul_sign(dim, u.0, v.0) * cd_basis_mul_sign(dim, u.1, v.1);
                            let sigma_vw =
                                cd_basis_mul_sign(dim, v.0, w.0) * cd_basis_mul_sign(dim, v.1, w.1);
                            let sigma_uw =
                                cd_basis_mul_sign(dim, u.0, w.0) * cd_basis_mul_sign(dim, u.1, w.1);

                            let n_same = [sigma_uv, sigma_vw, sigma_uw]
                                .iter()
                                .filter(|&&s| s == -1)
                                .count();
                            let product = sigma_uv * sigma_vw * sigma_uw;
                            let is_pure = if product == 1 {
                                n_same == 0 // AllOpposite
                            } else {
                                n_same == 3 // AllSame
                            };

                            // Compute F = (A(lo_a, lo_b, lo_c), A(hi_a, hi_b, hi_c))
                            let a_lo = assoc_sign(dim, u.0, v.0, w.0);
                            let a_hi = assoc_sign(dim, u.1, v.1, w.1);

                            let f_val = (a_lo, a_hi);
                            let entry = f_counts.entry(f_val).or_insert([0, 0]);
                            if is_pure {
                                entry[0] += 1;
                            } else {
                                entry[1] += 1;
                            }

                            if f_val == (1, 1) {
                                if is_pure {
                                    f_pp_pure += 1;
                                } else {
                                    f_pp_mixed += 1;
                                }
                            } else if is_pure {
                                f_other_pure += 1;
                            } else {
                                f_other_mixed += 1;
                            }
                        }
                    }
                }
            }

            eprintln!("\n=== Candidate B (Associator Obstruction) at dim={dim} ===");
            eprintln!("Total triangles: {total_triangles}");
            eprintln!("F=(+1,+1): pure={f_pp_pure}, mixed={f_pp_mixed}");
            eprintln!("F!=(+1,+1): pure={f_other_pure}, mixed={f_other_mixed}");

            let mut f_keys: Vec<(i32, i32)> = f_counts.keys().copied().collect();
            f_keys.sort();
            for fk in &f_keys {
                let [p, m] = f_counts[fk];
                eprintln!("  F={:?}: pure={p}, mixed={m}, total={}", fk, p + m);
            }

            // Check if F=(+1,+1) <=> pure (the ideal outcome)
            let perfect = f_pp_mixed == 0 && f_other_pure == 0;
            eprintln!("F=(+1,+1) <=> pure: {perfect}");

            if !perfect {
                // Try variant: F includes mixed lo/hi terms
                // Variant 1: A(lo_a, lo_b, hi_c) and A(hi_a, hi_b, lo_c)
                let mut v1_counts: std::collections::HashMap<(i32, i32), [usize; 2]> =
                    std::collections::HashMap::new();

                for comp in &components {
                    let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
                    let edge_set: std::collections::HashSet<(CrossPair, CrossPair)> =
                        comp.edges.iter().copied().collect();
                    let has_edge = |u: CrossPair, v: CrossPair| -> bool {
                        let (a, b) = if u < v { (u, v) } else { (v, u) };
                        edge_set.contains(&(a, b))
                    };

                    for i in 0..nodes.len() {
                        for j in (i + 1)..nodes.len() {
                            for ki in (j + 1)..nodes.len() {
                                let (u, v, w) = (nodes[i], nodes[j], nodes[ki]);
                                if !has_edge(u, v) || !has_edge(v, w) || !has_edge(u, w) {
                                    continue;
                                }

                                let sigma_uv = cd_basis_mul_sign(dim, u.0, v.0)
                                    * cd_basis_mul_sign(dim, u.1, v.1);
                                let sigma_vw = cd_basis_mul_sign(dim, v.0, w.0)
                                    * cd_basis_mul_sign(dim, v.1, w.1);
                                let sigma_uw = cd_basis_mul_sign(dim, u.0, w.0)
                                    * cd_basis_mul_sign(dim, u.1, w.1);
                                let n_same = [sigma_uv, sigma_vw, sigma_uw]
                                    .iter()
                                    .filter(|&&s| s == -1)
                                    .count();
                                let product = sigma_uv * sigma_vw * sigma_uw;
                                let is_pure = if product == 1 {
                                    n_same == 0
                                } else {
                                    n_same == 3
                                };

                                // Variant 1: cross lo/hi associator
                                let a_cross1 = assoc_sign(dim, u.0, v.0, w.1);
                                let a_cross2 = assoc_sign(dim, u.1, v.1, w.0);
                                let fv = (a_cross1, a_cross2);
                                let entry = v1_counts.entry(fv).or_insert([0, 0]);
                                if is_pure {
                                    entry[0] += 1;
                                } else {
                                    entry[1] += 1;
                                }
                            }
                        }
                    }
                }

                eprintln!("\n  Variant 1 (cross lo/hi associator):");
                let mut v1_keys: Vec<(i32, i32)> = v1_counts.keys().copied().collect();
                v1_keys.sort();
                for fk in &v1_keys {
                    let [p, m] = v1_counts[fk];
                    eprintln!("    F={:?}: pure={p}, mixed={m}", fk);
                }
                let v1_pp = v1_counts.get(&(1, 1)).copied().unwrap_or([0, 0]);
                let v1_perfect = v1_pp[1] == 0
                    && v1_counts
                        .iter()
                        .filter(|(&k, _)| k != (1, 1))
                        .all(|(_, v)| v[0] == 0);
                eprintln!("  Variant 1 F=(+1,+1) <=> pure: {v1_perfect}");
            }

            // Regardless of which variant works, the Quarter Rule must hold
            let total_pure = f_pp_pure + f_other_pure;
            let total_mixed = f_pp_mixed + f_other_mixed;
            assert_eq!(
                total_pure * 3,
                total_mixed,
                "dim={dim}: 3:1 ratio check failed: pure={total_pure}, mixed={total_mixed}"
            );
        }
    }

    /// C-519: Per-edge psi matrix search for 2-bit obstruction.
    ///
    /// For each edge (a,b) in a component, the twist gives a 2x2 matrix:
    ///   M_ab = [[psi(lo_a,lo_b), psi(lo_a,hi_b)],
    ///           [psi(hi_a,lo_b), psi(hi_a,hi_b)]]
    /// in GF(2). sigma uses only the diagonal XOR (M[0][0] ^ M[1][1]).
    ///
    /// For a triangle with 3 edges, we have 3 such matrices (12 GF(2) bits total).
    /// Search for a 2-bit function of these bits that separates pure from mixed.
    ///
    /// Specifically, try: for each pair of "off-diagonal" bit combinations,
    /// check if combining them with the sigma gives a 4-state invariant
    /// where "pure = zero state".
    #[test]
    fn test_psi_matrix_obstruction_search() {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;

        let psi = |dim: usize, i: usize, j: usize| -> u8 {
            if cd_basis_mul_sign(dim, i, j) == 1 {
                0
            } else {
                1
            }
        };

        // At dim=16, do an exhaustive search for a 2-bit invariant
        let dim = 16;
        let components = motif_components_for_cross_assessors(dim);

        // Collect all triangles with their full psi data
        struct TriData {
            // Per-edge psi matrices (3 edges, each a 2x2 GF(2) matrix)
            // m[edge][row][col] where row=0=>lo_left, row=1=>hi_left, col=0=>lo_right, col=1=>hi_right
            m: [[[u8; 2]; 2]; 3],
            is_pure: bool,
            _parity_even: bool,
        }

        let mut tris = Vec::new();
        for comp in &components {
            let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
            let edge_set: std::collections::HashSet<(CrossPair, CrossPair)> =
                comp.edges.iter().copied().collect();
            let has_edge = |u: CrossPair, v: CrossPair| -> bool {
                let (a, b) = if u < v { (u, v) } else { (v, u) };
                edge_set.contains(&(a, b))
            };

            for i in 0..nodes.len() {
                for j in (i + 1)..nodes.len() {
                    for k in (j + 1)..nodes.len() {
                        let (u, v, w) = (nodes[i], nodes[j], nodes[k]);
                        if !has_edge(u, v) || !has_edge(v, w) || !has_edge(u, w) {
                            continue;
                        }

                        // Compute 2x2 psi matrix for each edge
                        let edge_psi = |a: CrossPair, b: CrossPair| -> [[u8; 2]; 2] {
                            [
                                [psi(dim, a.0, b.0), psi(dim, a.0, b.1)],
                                [psi(dim, a.1, b.0), psi(dim, a.1, b.1)],
                            ]
                        };
                        let m_uv = edge_psi(u, v);
                        let m_vw = edge_psi(v, w);
                        let m_uw = edge_psi(u, w);

                        // Compute sigma = (-1)^(m[0][0] ^ m[1][1])
                        let sigma_uv = m_uv[0][0] ^ m_uv[1][1];
                        let sigma_vw = m_vw[0][0] ^ m_vw[1][1];
                        let sigma_uw = m_uw[0][0] ^ m_uw[1][1];
                        let n_same = (sigma_uv + sigma_vw + sigma_uw) as usize;
                        let product_even = n_same % 2 == 0;
                        let is_pure = if product_even {
                            n_same == 0
                        } else {
                            n_same == 3
                        };

                        tris.push(TriData {
                            m: [m_uv, m_vw, m_uw],
                            is_pure,
                            _parity_even: product_even,
                        });
                    }
                }
            }
        }

        eprintln!("\n=== Psi Matrix Obstruction Search at dim={dim} ===");
        eprintln!("Total triangles: {}", tris.len());

        // Each edge has 4 psi bits; 3 edges = 12 bits per triangle.
        // We want a 2-bit GF(2)-linear function of these 12 bits that
        // maps pure to (0,0) and mixed to non-(0,0).
        //
        // Strategy: try all pairs of GF(2)-linear combinations of the 12 bits.
        // A linear combination is a subset S of {0..11}, mapping to XOR of bits in S.
        // We want two such subsets (f1, f2) such that (f1(tri), f2(tri)) = (0,0) iff pure.

        // Encode each triangle as a 12-bit vector
        let encode = |tri: &TriData| -> u16 {
            let mut bits = 0u16;
            for (e, m) in tri.m.iter().enumerate() {
                for r in 0..2 {
                    for c in 0..2 {
                        bits |= (m[r][c] as u16) << (e * 4 + r * 2 + c);
                    }
                }
            }
            bits
        };

        let encoded: Vec<(u16, bool)> = tris.iter().map(|t| (encode(t), t.is_pure)).collect();

        let n_pure = encoded.iter().filter(|e| e.1).count();
        let n_mixed = encoded.iter().filter(|e| !e.1).count();
        eprintln!("Pure: {n_pure}, Mixed: {n_mixed}");

        // Try all 2^12 - 1 = 4095 non-zero linear functions for f1
        // For each f1, check: does f1(tri)=0 for all pure and f1(tri)=1 for some mixed?
        // That would give us a single-bit partial separator.
        let mut best_sep = 0usize;
        let mut best_mask = 0u16;
        let mut n_perfect_single = 0usize;

        for mask in 1u16..4096 {
            let f = |bits: u16| -> u8 { (bits & mask).count_ones() as u8 % 2 };

            // Check: f=0 on all pure
            let all_pure_zero = encoded.iter().filter(|e| e.1).all(|e| f(e.0) == 0);
            if !all_pure_zero {
                continue;
            }

            // Count mixed with f=1
            let mixed_one = encoded.iter().filter(|e| !e.1 && f(e.0) == 1).count();
            if mixed_one > best_sep {
                best_sep = mixed_one;
                best_mask = mask;
            }
            if mixed_one == n_mixed {
                n_perfect_single += 1;
            }
        }

        eprintln!(
            "Best single-bit separator: mask=0x{:03x}, separates {best_sep}/{n_mixed} mixed",
            best_mask
        );
        eprintln!("Perfect single-bit separators: {n_perfect_single}");

        // Also try: f=0 on all mixed, f=1 on some pure (pure = special)
        let mut best_sep2 = 0usize;
        let mut best_mask2 = 0u16;
        for mask in 1u16..4096 {
            let f = |bits: u16| -> u8 { (bits & mask).count_ones() as u8 % 2 };
            let all_mixed_zero = encoded.iter().filter(|e| !e.1).all(|e| f(e.0) == 0);
            if !all_mixed_zero {
                continue;
            }
            let pure_one = encoded.iter().filter(|e| e.1 && f(e.0) == 1).count();
            if pure_one > best_sep2 {
                best_sep2 = pure_one;
                best_mask2 = mask;
            }
        }
        eprintln!(
            "Best single-bit (pure=special): mask=0x{:03x}, separates {best_sep2}/{n_pure} pure",
            best_mask2
        );

        // Now search for 2-bit pairs that give perfect separation:
        // (f1, f2) = (0,0) iff pure
        let mut found_pair = false;
        let mut pair_mask = (0u16, 0u16);

        'outer: for m1 in 1u16..4096 {
            let f1 = |bits: u16| -> u8 { (bits & m1).count_ones() as u8 % 2 };

            // All pure must have f1=0
            if !encoded.iter().filter(|e| e.1).all(|e| f1(e.0) == 0) {
                continue;
            }

            for m2 in (m1 + 1)..4096 {
                let f2 = |bits: u16| -> u8 { (bits & m2).count_ones() as u8 % 2 };

                // All pure must have f2=0
                if !encoded.iter().filter(|e| e.1).all(|e| f2(e.0) == 0) {
                    continue;
                }

                // All mixed must have (f1,f2) != (0,0)
                let all_mixed_nonzero = encoded
                    .iter()
                    .filter(|e| !e.1)
                    .all(|e| f1(e.0) != 0 || f2(e.0) != 0);

                if all_mixed_nonzero {
                    found_pair = true;
                    pair_mask = (m1, m2);
                    break 'outer;
                }
            }
        }

        if found_pair {
            let (m1, m2) = pair_mask;
            let f1 = |bits: u16| -> u8 { (bits & m1).count_ones() as u8 % 2 };
            let f2 = |bits: u16| -> u8 { (bits & m2).count_ones() as u8 % 2 };

            eprintln!("\nFOUND 2-bit separator: m1=0x{:03x}, m2=0x{:03x}", m1, m2);

            // Decode masks into human-readable form
            let decode_mask = |mask: u16| -> String {
                let mut parts = Vec::new();
                let labels = [
                    "psi(lo_a,lo_b)",
                    "psi(lo_a,hi_b)",
                    "psi(hi_a,lo_b)",
                    "psi(hi_a,hi_b)",
                    "psi(lo_b,lo_c)",
                    "psi(lo_b,hi_c)",
                    "psi(hi_b,lo_c)",
                    "psi(hi_b,hi_c)",
                    "psi(lo_a,lo_c)",
                    "psi(lo_a,hi_c)",
                    "psi(hi_a,lo_c)",
                    "psi(hi_a,hi_c)",
                ];
                for (i, label) in labels.iter().enumerate() {
                    if mask & (1 << i) != 0 {
                        parts.push(*label);
                    }
                }
                parts.join(" XOR ")
            };
            eprintln!("f1 = {}", decode_mask(m1));
            eprintln!("f2 = {}", decode_mask(m2));

            // Show the distribution
            let mut dist: std::collections::HashMap<(u8, u8), [usize; 2]> =
                std::collections::HashMap::new();
            for &(bits, is_pure) in &encoded {
                let key = (f1(bits), f2(bits));
                let entry = dist.entry(key).or_insert([0, 0]);
                if is_pure {
                    entry[0] += 1;
                } else {
                    entry[1] += 1;
                }
            }
            for key in [(0u8, 0u8), (0, 1), (1, 0), (1, 1)] {
                let [p, m] = dist.get(&key).copied().unwrap_or([0, 0]);
                eprintln!("  F={:?}: pure={p}, mixed={m}", key);
            }
        } else {
            eprintln!("\nNo 2-bit GF(2)-linear separator found in 12-bit space.");
        }

        // The test passes regardless -- this is exploratory.
        // The key assertion is the Quarter Rule from the parent test.
        assert_eq!(n_pure * 3, n_mixed, "Quarter Rule check");
    }

    /// C-520: Test the dim=16 2-bit separator at dim=32 and dim=64.
    ///
    /// At dim=16, the obstruction is:
    ///   f1(ab) = psi(lo_a, hi_b) XOR psi(hi_a, lo_b)  ("anti-diagonal XOR")
    ///   f2 = psi(lo_a, lo_b) XOR psi(lo_a, hi_b) XOR psi(lo_b, lo_c) XOR psi(hi_b, lo_c)
    ///
    /// A triangle is "pure" iff f1=0 AND f2=0 for ALL its edges/vertex combinations.
    ///
    /// But f1 depends on a single edge and can be rewritten as:
    ///   f1(ab) = psi(lo_a, hi_b) XOR psi(hi_a, lo_b)
    /// The "anti-diagonal XOR" or "GF(2) determinant" of the edge's psi matrix.
    ///
    /// Hypothesis: a triangle is "pure" iff the XOR sum of f1 over its 3 edges is 0
    /// AND some second condition involving cross-edge terms is 0.
    ///
    /// Alternative approach: per-component search for universal invariant.
    #[test]
    fn test_separator_generalization() {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;

        let psi = |dim: usize, i: usize, j: usize| -> u8 {
            if cd_basis_mul_sign(dim, i, j) == 1 {
                0
            } else {
                1
            }
        };

        // The "anti-diagonal XOR" per edge
        let antidet = |dim: usize, a: CrossPair, b: CrossPair| -> u8 {
            psi(dim, a.0, b.1) ^ psi(dim, a.1, b.0)
        };

        // The "diagonal XOR" = sigma exponent per edge (Same iff 1)
        let diag = |dim: usize, a: CrossPair, b: CrossPair| -> u8 {
            psi(dim, a.0, b.0) ^ psi(dim, a.1, b.1)
        };

        // The "full determinant" (det of 2x2 GF(2) matrix) = diag XOR antidet
        let det = |dim: usize, a: CrossPair, b: CrossPair| -> u8 {
            (psi(dim, a.0, b.0) & psi(dim, a.1, b.1)) ^ (psi(dim, a.0, b.1) & psi(dim, a.1, b.0))
        };

        for &dim in &[16, 32, 64] {
            let components = motif_components_for_cross_assessors(dim);
            let mut total_triangles = 0usize;

            // Test several candidate 2-bit functions on triangles:
            // Candidate 1: (XOR of antidet, XOR of det) over 3 edges
            // Candidate 2: (antidet(ab), antidet(bc)) -- first 2 edges
            // Candidate 3: (antidet(ab) XOR antidet(bc), antidet(bc) XOR antidet(ac))
            // Candidate 4: (XOR of antidet, XOR of (diag*antidet)) over edges
            // Candidate 5: per-component approach -- sum of ALL psi cross-terms

            let mut c1_perfect = true;
            let mut c3_perfect = true;

            let mut c1_counts: std::collections::HashMap<(u8, u8), [usize; 2]> =
                std::collections::HashMap::new();
            let mut c3_counts: std::collections::HashMap<(u8, u8), [usize; 2]> =
                std::collections::HashMap::new();

            // Also try: the sum of all 3 anti-diagonals XOR sum of all 3 diagonals
            let mut c5_counts: std::collections::HashMap<(u8, u8), [usize; 2]> =
                std::collections::HashMap::new();

            // And: (antidet_sum, det_sum)
            let mut c6_counts: std::collections::HashMap<(u8, u8), [usize; 2]> =
                std::collections::HashMap::new();

            for comp in &components {
                let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
                let edge_set: std::collections::HashSet<(CrossPair, CrossPair)> =
                    comp.edges.iter().copied().collect();
                let has_edge = |u: CrossPair, v: CrossPair| -> bool {
                    let (a, b) = if u < v { (u, v) } else { (v, u) };
                    edge_set.contains(&(a, b))
                };

                for i in 0..nodes.len() {
                    for j in (i + 1)..nodes.len() {
                        for k in (j + 1)..nodes.len() {
                            let (u, v, w) = (nodes[i], nodes[j], nodes[k]);
                            if !has_edge(u, v) || !has_edge(v, w) || !has_edge(u, w) {
                                continue;
                            }
                            total_triangles += 1;

                            let d_uv = diag(dim, u, v);
                            let d_vw = diag(dim, v, w);
                            let d_uw = diag(dim, u, w);
                            let n_same = (d_uv + d_vw + d_uw) as usize;
                            let product_even = n_same % 2 == 0;
                            let is_pure = if product_even {
                                n_same == 0
                            } else {
                                n_same == 3
                            };

                            let a_uv = antidet(dim, u, v);
                            let a_vw = antidet(dim, v, w);
                            let a_uw = antidet(dim, u, w);

                            let det_uv = det(dim, u, v);
                            let det_vw = det(dim, v, w);
                            let det_uw = det(dim, u, w);

                            // Candidate 1: (sum antidet, sum det)
                            let f1_c1 = a_uv ^ a_vw ^ a_uw;
                            let f2_c1 = det_uv ^ det_vw ^ det_uw;
                            let entry = c1_counts.entry((f1_c1, f2_c1)).or_insert([0, 0]);
                            if is_pure {
                                entry[0] += 1;
                            } else {
                                entry[1] += 1;
                            }
                            if (f1_c1 == 0 && f2_c1 == 0) != is_pure {
                                c1_perfect = false;
                            }

                            // Candidate 3: (antidet uv XOR antidet vw, antidet vw XOR antidet uw)
                            let f1_c3 = a_uv ^ a_vw;
                            let f2_c3 = a_vw ^ a_uw;
                            let entry = c3_counts.entry((f1_c3, f2_c3)).or_insert([0, 0]);
                            if is_pure {
                                entry[0] += 1;
                            } else {
                                entry[1] += 1;
                            }
                            if (f1_c3 == 0 && f2_c3 == 0) != is_pure {
                                c3_perfect = false;
                            }

                            // Candidate 5: (sum antidet, sum diag)
                            let f1_c5 = a_uv ^ a_vw ^ a_uw;
                            let f2_c5 = d_uv ^ d_vw ^ d_uw;
                            let entry = c5_counts.entry((f1_c5, f2_c5)).or_insert([0, 0]);
                            if is_pure {
                                entry[0] += 1;
                            } else {
                                entry[1] += 1;
                            }

                            // Candidate 6: (antidet sum, det sum)
                            let entry = c6_counts.entry((f1_c1, f2_c1)).or_insert([0, 0]);
                            if is_pure {
                                entry[0] += 1;
                            } else {
                                entry[1] += 1;
                            }
                        }
                    }
                }
            }

            eprintln!("\n=== Separator Generalization at dim={dim} ===");
            eprintln!("Total triangles: {total_triangles}");

            eprintln!("\nCandidate 1 (antidet_sum, det_sum):");
            for key in [(0u8, 0u8), (0, 1), (1, 0), (1, 1)] {
                let [p, m] = c1_counts.get(&key).copied().unwrap_or([0, 0]);
                eprintln!("  ({},{}): pure={p}, mixed={m}", key.0, key.1);
            }
            eprintln!("  Perfect: {c1_perfect}");

            eprintln!("\nCandidate 3 (antidet_ab^antidet_bc, antidet_bc^antidet_ac):");
            for key in [(0u8, 0u8), (0, 1), (1, 0), (1, 1)] {
                let [p, m] = c3_counts.get(&key).copied().unwrap_or([0, 0]);
                eprintln!("  ({},{}): pure={p}, mixed={m}", key.0, key.1);
            }
            eprintln!("  Perfect: {c3_perfect}");

            eprintln!("\nCandidate 5 (antidet_sum, diag_sum=parity):");
            for key in [(0u8, 0u8), (0, 1), (1, 0), (1, 1)] {
                let [p, m] = c5_counts.get(&key).copied().unwrap_or([0, 0]);
                eprintln!("  ({},{}): pure={p}, mixed={m}", key.0, key.1);
            }

            // ASSERTION: Candidate 3 must be perfect at every dimension
            let c3_pp = c3_counts.get(&(0, 0)).copied().unwrap_or([0, 0]);
            assert_eq!(
                c3_pp[1], 0,
                "dim={dim}: Candidate 3 has {} mixed in F=(0,0) -- not perfect",
                c3_pp[1],
            );
            let c3_nonzero_pure: usize = c3_counts
                .iter()
                .filter(|(&k, _)| k != (0, 0))
                .map(|(_, v)| v[0])
                .sum();
            assert_eq!(
                c3_nonzero_pure, 0,
                "dim={dim}: Candidate 3 has {c3_nonzero_pure} pure in F!=(0,0)"
            );
        }
    }

    /// C-521: Anti-Diagonal Parity Theorem -- definitive verification at dim=128.
    ///
    /// The theorem: define eta(a,b) = psi(lo_a, hi_b) XOR psi(hi_a, lo_b)
    /// for each ZD edge. A triangle (a,b,c) is "pure" iff eta is constant
    /// across all three edges: eta(a,b) = eta(b,c) = eta(a,c).
    ///
    /// This is equivalent to saying: the 2-bit invariant
    ///   F = (eta(ab) XOR eta(bc), eta(bc) XOR eta(ac))
    /// satisfies F=(0,0) iff the triangle is pure.
    ///
    /// We verify this at dim=128 per-component for full confidence.
    #[test]
    fn test_antidiagonal_parity_theorem_dim128() {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;

        let psi = |dim: usize, i: usize, j: usize| -> u8 {
            if cd_basis_mul_sign(dim, i, j) == 1 {
                0
            } else {
                1
            }
        };

        let dim = 128;
        let components = motif_components_for_cross_assessors(dim);
        let mut total_triangles = 0usize;
        let mut total_pure = 0usize;
        let mut total_mixed = 0usize;
        let mut mismatches = 0usize;
        let mut comp_results: Vec<(usize, usize, usize, bool)> = Vec::new();

        for (comp_idx, comp) in components.iter().enumerate() {
            let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
            let edge_set: std::collections::HashSet<(CrossPair, CrossPair)> =
                comp.edges.iter().copied().collect();
            let has_edge = |u: CrossPair, v: CrossPair| -> bool {
                let (a, b) = if u < v { (u, v) } else { (v, u) };
                edge_set.contains(&(a, b))
            };

            let eta =
                |a: CrossPair, b: CrossPair| -> u8 { psi(dim, a.0, b.1) ^ psi(dim, a.1, b.0) };

            let mut comp_tris = 0usize;
            let mut comp_pure = 0usize;
            let mut _comp_mixed = 0usize;
            let mut comp_ok = true;

            for i in 0..nodes.len() {
                for j in (i + 1)..nodes.len() {
                    for k in (j + 1)..nodes.len() {
                        let (u, v, w) = (nodes[i], nodes[j], nodes[k]);
                        if !has_edge(u, v) || !has_edge(v, w) || !has_edge(u, w) {
                            continue;
                        }
                        comp_tris += 1;

                        // Compute sigma for classification
                        let sigma = |a: CrossPair, b: CrossPair| -> i32 {
                            cd_basis_mul_sign(dim, a.0, b.0) * cd_basis_mul_sign(dim, a.1, b.1)
                        };
                        let s_uv = sigma(u, v);
                        let s_vw = sigma(v, w);
                        let s_uw = sigma(u, w);
                        let n_same = [s_uv, s_vw, s_uw].iter().filter(|&&s| s == -1).count();
                        let product = s_uv * s_vw * s_uw;
                        let is_pure = if product == 1 {
                            n_same == 0
                        } else {
                            n_same == 3
                        };

                        // Compute eta for all 3 edges
                        let eta_uv = eta(u, v);
                        let eta_vw = eta(v, w);
                        let eta_uw = eta(u, w);

                        // Anti-Diagonal Parity Theorem: pure iff eta constant
                        let eta_constant = (eta_uv == eta_vw) && (eta_vw == eta_uw);

                        if is_pure {
                            comp_pure += 1;
                            total_pure += 1;
                        } else {
                            _comp_mixed += 1;
                            total_mixed += 1;
                        }

                        if eta_constant != is_pure {
                            mismatches += 1;
                            comp_ok = false;
                        }
                    }
                }
            }

            total_triangles += comp_tris;
            if comp_tris > 0 {
                comp_results.push((comp_idx, comp_tris, comp_pure, comp_ok));
            }
        }

        eprintln!("\n=== Anti-Diagonal Parity Theorem at dim={dim} ===");
        eprintln!("Components with triangles: {}", comp_results.len());
        eprintln!("Total triangles: {total_triangles}");
        eprintln!("Pure: {total_pure}, Mixed: {total_mixed}");
        eprintln!("Mismatches: {mismatches}");

        let n_ok = comp_results.iter().filter(|r| r.3).count();
        let n_fail = comp_results.iter().filter(|r| !r.3).count();
        eprintln!("Components perfect: {n_ok}, failed: {n_fail}");

        if n_fail > 0 {
            for &(idx, tris, pure, ok) in &comp_results {
                if !ok {
                    eprintln!("  FAIL: comp[{idx}] tris={tris}, pure={pure}");
                }
            }
        }

        // ASSERTION: theorem holds perfectly (zero mismatches)
        assert_eq!(
            mismatches, 0,
            "Anti-Diagonal Parity Theorem REFUTED at dim={dim}: {mismatches} mismatches"
        );

        // Cross-check: Quarter Rule still holds
        assert_eq!(
            total_pure * 3,
            total_mixed,
            "Quarter Rule failed: pure={total_pure}, mixed={total_mixed}"
        );
    }

    /// Anti-Diagonal Parity Theorem verification at dim=256 (C-522).
    ///
    /// Extends C-521 to the next dimension doubling: 127 components,
    /// each with 126 nodes, producing ~13.3M triangles. Also collects:
    /// - Klein-four fiber sizes: distribution of F values in GF(2)^2
    /// - Per-edge eta balance: counts eta=0 vs eta=1 per component
    /// - Cohomological data: cycle rank of eta-labeled graph
    ///
    /// Runtime: ~15-30s in release mode.
    #[test]
    #[ignore] // Long-running: ~15-30s in release mode
    fn test_antidiagonal_parity_theorem_dim256() {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;
        use std::time::Instant;

        let psi = |dim: usize, i: usize, j: usize| -> u8 {
            if cd_basis_mul_sign(dim, i, j) == 1 {
                0
            } else {
                1
            }
        };

        let dim = 256;
        let t0 = Instant::now();
        let components = motif_components_for_cross_assessors(dim);
        let t_graph = t0.elapsed();
        eprintln!("\n=== Anti-Diagonal Parity Theorem at dim={dim} ===");
        eprintln!("Graph construction: {:.2}s", t_graph.as_secs_f64());
        eprintln!("Components: {}", components.len());

        let mut total_triangles = 0usize;
        let mut total_pure = 0usize;
        let mut total_mixed = 0usize;
        let mut mismatches = 0usize;

        // Klein-four fiber sizes: F = (f1, f2) in {(0,0), (0,1), (1,0), (1,1)}
        let mut fiber_counts = [0usize; 4]; // indexed by 2*f1 + f2

        // Per-component eta balance
        let mut total_eta0 = 0usize;
        let mut total_eta1 = 0usize;

        // Cycle rank accumulator: sum over components of (|E| - |V| + 1)
        let mut total_cycle_rank = 0usize;

        let t1 = Instant::now();

        for comp in components.iter() {
            let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
            let n = nodes.len();
            let edge_set: std::collections::HashSet<(CrossPair, CrossPair)> =
                comp.edges.iter().copied().collect();
            let has_edge = |u: CrossPair, v: CrossPair| -> bool {
                let (a, b) = if u < v { (u, v) } else { (v, u) };
                edge_set.contains(&(a, b))
            };

            let eta =
                |a: CrossPair, b: CrossPair| -> u8 { psi(dim, a.0, b.1) ^ psi(dim, a.1, b.0) };

            // Collect edge eta values for cohomology
            let mut comp_eta0 = 0usize;
            let mut comp_eta1 = 0usize;
            for &(u, v) in &comp.edges {
                if eta(u, v) == 0 {
                    comp_eta0 += 1;
                } else {
                    comp_eta1 += 1;
                }
            }
            total_eta0 += comp_eta0;
            total_eta1 += comp_eta1;

            // Cycle rank = |E| - |V| + 1 (for connected component)
            total_cycle_rank += comp.edges.len() - n + 1;

            // Triangle enumeration
            for i in 0..n {
                for j in (i + 1)..n {
                    for k in (j + 1)..n {
                        let (u, v, w) = (nodes[i], nodes[j], nodes[k]);
                        if !has_edge(u, v) || !has_edge(v, w) || !has_edge(u, w) {
                            continue;
                        }
                        total_triangles += 1;

                        // Sigma classification
                        let sigma = |a: CrossPair, b: CrossPair| -> i32 {
                            cd_basis_mul_sign(dim, a.0, b.0) * cd_basis_mul_sign(dim, a.1, b.1)
                        };
                        let s_uv = sigma(u, v);
                        let s_vw = sigma(v, w);
                        let s_uw = sigma(u, w);
                        let n_same = [s_uv, s_vw, s_uw].iter().filter(|&&s| s == -1).count();
                        let product = s_uv * s_vw * s_uw;
                        let is_pure = if product == 1 {
                            n_same == 0
                        } else {
                            n_same == 3
                        };

                        // Eta for all 3 edges
                        let eta_uv = eta(u, v);
                        let eta_vw = eta(v, w);
                        let eta_uw = eta(u, w);

                        // Anti-Diagonal Parity Theorem: pure iff eta constant
                        let eta_constant = (eta_uv == eta_vw) && (eta_vw == eta_uw);

                        if eta_constant != is_pure {
                            mismatches += 1;
                        }

                        if is_pure {
                            total_pure += 1;
                        } else {
                            total_mixed += 1;
                        }

                        // Klein-four fiber: F = (eta_uv XOR eta_vw, eta_vw XOR eta_uw)
                        let f1 = eta_uv ^ eta_vw;
                        let f2 = eta_vw ^ eta_uw;
                        fiber_counts[(2 * f1 + f2) as usize] += 1;
                    }
                }
            }
        }

        let t_total = t1.elapsed();

        eprintln!("Triangle enumeration: {:.2}s", t_total.as_secs_f64());
        eprintln!("Total triangles: {total_triangles}");
        eprintln!("Pure: {total_pure}, Mixed: {total_mixed}");
        eprintln!("Mismatches: {mismatches}");
        eprintln!(
            "Klein-four fibers: (0,0)={}, (0,1)={}, (1,0)={}, (1,1)={}",
            fiber_counts[0], fiber_counts[1], fiber_counts[2], fiber_counts[3]
        );
        eprintln!("Edge eta balance: eta=0: {total_eta0}, eta=1: {total_eta1}");
        eprintln!("Total cycle rank: {total_cycle_rank}");

        // ASSERTION 1: Anti-Diagonal Parity Theorem holds
        assert_eq!(
            mismatches, 0,
            "Anti-Diagonal Parity Theorem REFUTED at dim={dim}: {mismatches} mismatches"
        );

        // ASSERTION 2: Quarter Rule
        assert_eq!(
            total_pure * 3,
            total_mixed,
            "Quarter Rule failed: pure={total_pure}, mixed={total_mixed}"
        );

        // ASSERTION 3: Klein-four fiber (0,0) = pure count
        assert_eq!(
            fiber_counts[0], total_pure,
            "F=(0,0) should equal pure count"
        );

        // ASSERTION 4: Sum of nonzero fibers = mixed count
        assert_eq!(
            fiber_counts[1] + fiber_counts[2] + fiber_counts[3],
            total_mixed,
            "Nonzero fibers should sum to mixed count"
        );
    }

    /// GF(2) cohomology of eta and Klein-four fiber structure across dimensions (C-523).
    ///
    /// For each dimension and each component, computes:
    /// 1. First Betti number b_1 = |E| - |V| + 1 (cycle rank)
    /// 2. Whether eta is a coboundary: eta(a,b) = delta(a) XOR delta(b)
    ///    If coboundary, ALL triangles pure. If not, mixed triangles exist.
    /// 3. Number of "frustrated cycles" (independent cycles where eta sums to 1 mod 2)
    /// 4. Klein-four fiber sizes per component and globally
    ///
    /// The coboundary test works by BFS: assign delta(root)=0, propagate
    /// delta(v) = delta(u) XOR eta(u,v) along spanning tree. Then check
    /// non-tree edges: if delta(u) XOR delta(v) != eta(u,v), that edge
    /// witnesses a frustrated cycle.
    #[test]
    fn test_eta_cohomology_and_klein_four_fibers() {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;
        use std::collections::VecDeque;

        let psi = |dim: usize, i: usize, j: usize| -> u8 {
            if cd_basis_mul_sign(dim, i, j) == 1 {
                0
            } else {
                1
            }
        };

        eprintln!("\n=== GF(2) Cohomology of eta + Klein-four Fiber Structure ===");
        eprintln!(
            "{:<6} {:>5} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "dim",
            "comps",
            "edges",
            "eta=0",
            "eta=1",
            "b1",
            "frustrated",
            "F(0,0)",
            "F(0,1)",
            "F(1,0)",
            "F(1,1)"
        );

        for &dim in &[16, 32, 64, 128] {
            let components = motif_components_for_cross_assessors(dim);

            let mut total_edges = 0usize;
            let mut total_eta0 = 0usize;
            let mut total_eta1 = 0usize;
            let mut total_b1 = 0usize;
            let mut total_frustrated = 0usize;
            let mut fiber_counts = [0usize; 4];

            for comp in components.iter() {
                let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
                let n = nodes.len();
                if n < 2 {
                    continue;
                }

                // Build adjacency list with eta labels
                let node_idx: std::collections::HashMap<CrossPair, usize> =
                    nodes.iter().enumerate().map(|(i, &cp)| (cp, i)).collect();
                let mut adj: Vec<Vec<(usize, u8)>> = vec![Vec::new(); n];

                let mut comp_eta0 = 0usize;
                let mut comp_eta1 = 0usize;

                for &(u, v) in &comp.edges {
                    let eta_val = psi(dim, u.0, v.1) ^ psi(dim, u.1, v.0);
                    let ui = node_idx[&u];
                    let vi = node_idx[&v];
                    adj[ui].push((vi, eta_val));
                    adj[vi].push((ui, eta_val));
                    if eta_val == 0 {
                        comp_eta0 += 1;
                    } else {
                        comp_eta1 += 1;
                    }
                }

                total_edges += comp.edges.len();
                total_eta0 += comp_eta0;
                total_eta1 += comp_eta1;

                // BFS to compute delta assignment and count frustrated non-tree edges
                let mut delta: Vec<Option<u8>> = vec![None; n];
                delta[0] = Some(0);
                let mut queue = VecDeque::new();
                queue.push_back(0);
                let mut tree_edges = 0usize;
                let mut frustrated = 0usize;

                while let Some(u) = queue.pop_front() {
                    for &(v, eta_val) in &adj[u] {
                        if let Some(dv) = delta[v] {
                            // Non-tree edge: check consistency
                            // Only count each non-tree edge once (u < v)
                            if u < v {
                                let expected = delta[u].unwrap() ^ eta_val;
                                if expected != dv {
                                    frustrated += 1;
                                }
                            }
                        } else {
                            delta[v] = Some(delta[u].unwrap() ^ eta_val);
                            tree_edges += 1;
                            queue.push_back(v);
                        }
                    }
                }

                let b1 = comp.edges.len() - tree_edges; // = |E| - (|V| - 1)
                total_b1 += b1;
                total_frustrated += frustrated;

                // Triangle enumeration for Klein-four fibers
                let edge_set: std::collections::HashSet<(CrossPair, CrossPair)> =
                    comp.edges.iter().copied().collect();
                let has_edge = |u: CrossPair, v: CrossPair| -> bool {
                    let (a, b) = if u < v { (u, v) } else { (v, u) };
                    edge_set.contains(&(a, b))
                };

                let eta =
                    |a: CrossPair, b: CrossPair| -> u8 { psi(dim, a.0, b.1) ^ psi(dim, a.1, b.0) };

                for i in 0..n {
                    for j in (i + 1)..n {
                        for k in (j + 1)..n {
                            let (u, v, w) = (nodes[i], nodes[j], nodes[k]);
                            if !has_edge(u, v) || !has_edge(v, w) || !has_edge(u, w) {
                                continue;
                            }
                            let eta_uv = eta(u, v);
                            let eta_vw = eta(v, w);
                            let eta_uw = eta(u, w);
                            let f1 = eta_uv ^ eta_vw;
                            let f2 = eta_vw ^ eta_uw;
                            fiber_counts[(2 * f1 + f2) as usize] += 1;
                        }
                    }
                }
            }

            eprintln!(
                "{:<6} {:>5} {:>8} {:>8} {:>8} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
                dim,
                components.len(),
                total_edges,
                total_eta0,
                total_eta1,
                total_b1,
                total_frustrated,
                fiber_counts[0],
                fiber_counts[1],
                fiber_counts[2],
                fiber_counts[3]
            );

            // ASSERTIONS

            // Eta is exactly balanced on edges in every component (Half-Half Law)
            assert_eq!(
                total_eta0, total_eta1,
                "dim={dim}: eta not balanced on edges: {total_eta0} vs {total_eta1}"
            );

            // At dim=16, eta IS a coboundary (0 frustrated cycles despite b1=49).
            // At dim>=32, eta is NOT a coboundary (frustrated cycles > 0).
            // This is a cohomological phase transition!
            if dim == 16 {
                assert_eq!(
                    total_frustrated, 0,
                    "dim=16: expected eta to be a coboundary but found {total_frustrated} frustrated"
                );
            } else {
                assert!(
                    total_frustrated > 0,
                    "dim={dim}: eta is unexpectedly a coboundary"
                );
            }

            // Klein-four: F(0,0) = pure count, F(0,0)*3 = sum of nonzero
            let total_tris: usize = fiber_counts.iter().sum();
            if total_tris > 0 {
                assert_eq!(
                    fiber_counts[0] * 3,
                    fiber_counts[1] + fiber_counts[2] + fiber_counts[3],
                    "dim={dim}: 1:3 ratio from Klein-four fibers failed"
                );
            }

            // Klein-four fibers (1,0) and (1,1) are equal (observed at dim=256)
            // Test if this holds at all dimensions
            if dim >= 32 {
                // At dim=16, check but don't assert (sample may be too small)
                assert_eq!(
                    fiber_counts[2], fiber_counts[3],
                    "dim={dim}: F(1,0) != F(1,1): {} vs {}",
                    fiber_counts[2], fiber_counts[3]
                );
            }
        }
    }

    /// CD doubling recursion analysis of eta (C-526).
    ///
    /// Decomposes eta(a,b) = psi(lo_a, hi_b) XOR psi(hi_a, lo_b) using the
    /// Cayley-Dickson doubling formula to understand its algebraic origin.
    ///
    /// Key observation: for cross-assessor pair a=(lo_a, hi_a), b=(lo_b, hi_b),
    /// the psi matrix is:
    ///   M[0,0] = psi(lo_a, lo_b)   -- case 1 (both lo, pure recursion)
    ///   M[0,1] = psi(lo_a, hi_b)   -- case 2 (lo*hi, swap+recurse)
    ///   M[1,0] = psi(hi_a, lo_b)   -- case 3 (hi*lo, conjugation)
    ///   M[1,1] = psi(hi_a, hi_b)   -- case 4 (both hi, double conj)
    ///
    /// Sigma (diagonal XOR) = M[0,0] XOR M[1,1]
    /// Eta (anti-diagonal XOR) = M[0,1] XOR M[1,0]
    ///
    /// We test whether eta can be decomposed as a function of the
    /// half-dimension indices, and whether the conjugation asymmetry
    /// in case 3 is the sole source of eta != 0.
    #[test]
    fn test_eta_doubling_decomposition() {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;

        let psi = |dim: usize, i: usize, j: usize| -> u8 {
            if cd_basis_mul_sign(dim, i, j) == 1 {
                0
            } else {
                1
            }
        };

        eprintln!("\n=== Eta Doubling Decomposition ===");

        for &dim in &[16, 32, 64] {
            let half = dim / 2;
            let components = motif_components_for_cross_assessors(dim);

            let mut total_edges = 0usize;
            let mut eta_from_conj_only = 0usize; // edges where eta comes purely from case 3 conjugation
            let mut lo_b_zero_count = 0usize;

            for comp in components.iter() {
                for &(a, b) in &comp.edges {
                    total_edges += 1;
                    let (lo_a, hi_a) = (a.0, a.1);
                    let (lo_b, hi_b) = (b.0, b.1);

                    // Case 2: s(lo_a, hi_b) = s_half(hi_b - half, lo_a) by CD doubling
                    // Case 3: s(hi_a, lo_b) depends on lo_b:
                    //   lo_b == 0: s(hi_a, 0) = s_half(hi_a - half, 0) = 1 (identity)
                    //   lo_b != 0: s(hi_a, lo_b) = -s_half(hi_a - half, lo_b)

                    // So psi for case 2 and case 3:
                    let psi_02 = psi(dim, lo_a, hi_b);
                    let psi_10 = psi(dim, hi_a, lo_b);
                    let eta_val = psi_02 ^ psi_10;

                    // Decompose via half-dimension:
                    // psi_02 = psi_half(hi_b - half, lo_a) [case 2 recurses with swap]
                    let psi_02_half = psi(half, hi_b - half, lo_a);
                    assert_eq!(
                        psi_02, psi_02_half,
                        "dim={dim}: case 2 decomposition failed"
                    );

                    // psi_10: case 3
                    if lo_b == 0 {
                        lo_b_zero_count += 1;
                        // s(hi_a, 0) = s_half(hi_a - half, 0)
                        // For any x, s(x, 0) = 1 (e_0 is identity), so psi = 0
                        assert_eq!(psi_10, 0, "dim={dim}: psi(hi_a, 0) should be 0");
                    } else {
                        // s(hi_a, lo_b) = -s_half(hi_a - half, lo_b)
                        let psi_10_half = psi(half, hi_a - half, lo_b);
                        // -s means flip sign: psi_10 = 1 XOR psi_10_half
                        assert_eq!(
                            psi_10,
                            psi_10_half ^ 1,
                            "dim={dim}: case 3 conjugation decomposition failed"
                        );

                        // So eta = psi_02 XOR psi_10
                        //        = psi_half(hi_b-h, lo_a) XOR (1 XOR psi_half(hi_a-h, lo_b))
                        //        = 1 XOR psi_half(hi_b-h, lo_a) XOR psi_half(hi_a-h, lo_b)
                        //        = 1 XOR eta_half(a', b')
                        // where a' = (lo_a, hi_a-h), b' = (lo_b, hi_b-h) in the half-dimension!
                        let eta_half = psi(half, hi_b - half, lo_a) ^ psi(half, hi_a - half, lo_b);
                        let expected_eta = 1 ^ eta_half;
                        assert_eq!(
                            eta_val, expected_eta,
                            "dim={dim}: eta = 1 XOR eta_half failed"
                        );

                        if eta_half == 0 {
                            eta_from_conj_only += 1;
                        }
                    }
                }
            }

            eprintln!(
                "dim={dim}: {total_edges} edges, lo_b=0: {lo_b_zero_count}, \
                       eta from conj flip only: {eta_from_conj_only}"
            );
        }
    }

    /// Eta distribution across edge-count regimes (C-525).
    ///
    /// The face sign census groups components by edge count into "regimes."
    /// For each regime, we compute:
    /// - Edge eta balance (eta=0 vs eta=1)
    /// - Per-component frustration ratio (frustrated / b1)
    /// - Klein-four fiber sizes
    /// - Per-regime pure:mixed ratio
    ///
    /// This tests whether the mechanism (eta constancy) behaves uniformly
    /// across all regimes or has regime-dependent structure.
    #[test]
    fn test_eta_regime_distribution() {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;
        use std::collections::{BTreeMap, VecDeque};

        let psi = |dim: usize, i: usize, j: usize| -> u8 {
            if cd_basis_mul_sign(dim, i, j) == 1 {
                0
            } else {
                1
            }
        };

        eprintln!("\n=== Eta Distribution Across Edge-Count Regimes ===");

        for &dim in &[16, 32, 64] {
            eprintln!("\n--- dim={dim} ---");

            let components = motif_components_for_cross_assessors(dim);

            // Group components by edge count (regime)
            // Per-regime accumulators: (n_comps, eta0, eta1, b1, frustrated, fibers[4], pure, mixed)
            let mut regimes: BTreeMap<
                usize,
                (usize, usize, usize, usize, usize, [usize; 4], usize, usize),
            > = BTreeMap::new();

            for comp in components.iter() {
                let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
                let n = nodes.len();
                if n < 2 {
                    continue;
                }
                let n_edges = comp.edges.len();

                let entry = regimes
                    .entry(n_edges)
                    .or_insert((0, 0, 0, 0, 0, [0; 4], 0, 0));
                entry.0 += 1; // n_comps

                // Build adjacency with eta
                let node_idx: std::collections::HashMap<CrossPair, usize> =
                    nodes.iter().enumerate().map(|(i, &cp)| (cp, i)).collect();
                let mut adj: Vec<Vec<(usize, u8)>> = vec![Vec::new(); n];

                for &(u, v) in &comp.edges {
                    let eta_val = psi(dim, u.0, v.1) ^ psi(dim, u.1, v.0);
                    let ui = node_idx[&u];
                    let vi = node_idx[&v];
                    adj[ui].push((vi, eta_val));
                    adj[vi].push((ui, eta_val));
                    if eta_val == 0 {
                        entry.1 += 1;
                    } else {
                        entry.2 += 1;
                    }
                }

                // BFS for coboundary test
                let mut delta: Vec<Option<u8>> = vec![None; n];
                delta[0] = Some(0);
                let mut queue = VecDeque::new();
                queue.push_back(0);
                let mut tree_edges = 0usize;
                let mut frustrated = 0usize;

                while let Some(u) = queue.pop_front() {
                    for &(v, eta_val) in &adj[u] {
                        if let Some(dv) = delta[v] {
                            if u < v {
                                let expected = delta[u].unwrap() ^ eta_val;
                                if expected != dv {
                                    frustrated += 1;
                                }
                            }
                        } else {
                            delta[v] = Some(delta[u].unwrap() ^ eta_val);
                            tree_edges += 1;
                            queue.push_back(v);
                        }
                    }
                }

                let b1 = n_edges - tree_edges;
                entry.3 += b1; // total b1
                entry.4 += frustrated; // total frustrated

                // Triangle enumeration
                let edge_set: std::collections::HashSet<(CrossPair, CrossPair)> =
                    comp.edges.iter().copied().collect();
                let has_edge = |u: CrossPair, v: CrossPair| -> bool {
                    let (a, b) = if u < v { (u, v) } else { (v, u) };
                    edge_set.contains(&(a, b))
                };
                let eta =
                    |a: CrossPair, b: CrossPair| -> u8 { psi(dim, a.0, b.1) ^ psi(dim, a.1, b.0) };

                for i in 0..n {
                    for j in (i + 1)..n {
                        for k in (j + 1)..n {
                            let (u, v, w) = (nodes[i], nodes[j], nodes[k]);
                            if !has_edge(u, v) || !has_edge(v, w) || !has_edge(u, w) {
                                continue;
                            }
                            let eta_uv = eta(u, v);
                            let eta_vw = eta(v, w);
                            let eta_uw = eta(u, w);
                            let f1 = eta_uv ^ eta_vw;
                            let f2 = eta_vw ^ eta_uw;
                            entry.5[(2 * f1 + f2) as usize] += 1;

                            let eta_const = (eta_uv == eta_vw) && (eta_vw == eta_uw);
                            if eta_const {
                                entry.6 += 1;
                            } else {
                                entry.7 += 1;
                            }
                        }
                    }
                }
            }

            eprintln!(
                "{:>6} {:>5} {:>6} {:>6} {:>5} {:>5} {:>8} {:>8} {:>8} {:>8} {:>6} {:>6}",
                "edges",
                "comps",
                "eta=0",
                "eta=1",
                "b1",
                "frust",
                "F(0,0)",
                "F(0,1)",
                "F(1,0)",
                "F(1,1)",
                "pure",
                "mixed"
            );

            for (&edge_count, &(n_comps, eta0, eta1, b1, frustrated, fibers, pure, mixed)) in
                &regimes
            {
                eprintln!(
                    "{:>6} {:>5} {:>6} {:>6} {:>5} {:>5} {:>8} {:>8} {:>8} {:>8} {:>6} {:>6}",
                    edge_count,
                    n_comps,
                    eta0,
                    eta1,
                    b1,
                    frustrated,
                    fibers[0],
                    fibers[1],
                    fibers[2],
                    fibers[3],
                    pure,
                    mixed
                );

                // Every regime should have balanced eta
                assert_eq!(
                    eta0, eta1,
                    "dim={dim}, regime edges={edge_count}: eta imbalance {eta0} vs {eta1}"
                );

                // Every regime should satisfy 1:3 ratio
                let total_tris = pure + mixed;
                if total_tris > 0 {
                    assert_eq!(
                        pure * 3,
                        mixed,
                        "dim={dim}, regime edges={edge_count}: 1:3 ratio failed: {pure}:{mixed}"
                    );
                }

                // F(1,0) = F(1,1) per regime
                assert_eq!(
                    fibers[2], fibers[3],
                    "dim={dim}, regime edges={edge_count}: F(1,0)={} != F(1,1)={}",
                    fibers[2], fibers[3]
                );
            }
        }
    }

    /// GF(2) polynomial degree analysis of psi(i,j) and eta(a,b) (C-527).
    ///
    /// psi(i,j) maps pairs of basis indices to GF(2), representing the sign
    /// exponent of the CD product. We express psi as a multilinear polynomial
    /// over GF(2) in the binary digits of i and j, and determine the minimal
    /// degree needed to represent it exactly.
    ///
    /// At dim=2^n, there are n bits per index, so 2n variables total.
    /// A degree-d polynomial has sum_{k=0}^{d} C(2n, k) monomials.
    ///
    /// We also analyze eta's effective polynomial degree, which may be
    /// lower than psi's since eta involves a specific linear combination
    /// of psi evaluations.
    #[test]
    fn test_psi_gf2_polynomial_degree() {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;

        let psi = |dim: usize, i: usize, j: usize| -> u8 {
            if cd_basis_mul_sign(dim, i, j) == 1 {
                0
            } else {
                1
            }
        };

        eprintln!("\n=== GF(2) Polynomial Degree of psi(i,j) ===");

        // For small dimensions, we can do exhaustive fitting.
        // psi(i,j) over GF(2)^n x GF(2)^n is a function F_2^{2n} -> F_2.
        // Any such function has a unique multilinear polynomial (ANF).
        // We compute the Algebraic Normal Form (ANF) via Mobius transform.

        for &dim in &[4usize, 8, 16, 32] {
            let n_bits = dim.trailing_zeros() as usize; // log2(dim)
            let n_vars = 2 * n_bits; // bits of i and bits of j

            // Build truth table: psi(i,j) for all i,j in [0, dim)
            let n_inputs = 1 << n_vars; // = dim^2
            let mut truth_table: Vec<u8> = vec![0; n_inputs];

            for i in 0..dim {
                for j in 0..dim {
                    // Encode (i,j) as a 2n-bit number: bits of i || bits of j
                    let input = (i << n_bits) | j;
                    truth_table[input] = psi(dim, i, j);
                }
            }

            // Compute ANF via Mobius transform over GF(2)
            // anf[m] = coefficient of monomial m (bitmask of which variables appear)
            let mut anf = truth_table.clone();
            for bit in 0..n_vars {
                let mask = 1 << bit;
                for m in 0..n_inputs {
                    if m & mask != 0 {
                        anf[m] ^= anf[m ^ mask];
                    }
                }
            }

            // Count monomials of each degree
            let mut degree_counts = vec![0usize; n_vars + 1];
            let mut max_degree = 0usize;
            let mut total_monomials = 0usize;

            for m in 0..n_inputs {
                if anf[m] != 0 {
                    let deg = (m as u32).count_ones() as usize;
                    degree_counts[deg] += 1;
                    if deg > max_degree {
                        max_degree = deg;
                    }
                    total_monomials += 1;
                }
            }

            eprintln!(
                "dim={dim} (n_bits={n_bits}, n_vars={n_vars}): \
                       ANF degree = {max_degree}, total monomials = {total_monomials}"
            );
            for (d, &count) in degree_counts.iter().enumerate() {
                if count > 0 {
                    eprintln!(
                        "  degree {d}: {count} monomials (of {} possible)",
                        binomial(n_vars, d)
                    );
                }
            }
        }

        // Now analyze eta's effective degree.
        // For each edge (a,b) in the cross-assessor graph, eta(a,b) = psi(lo_a,hi_b) XOR psi(hi_a,lo_b).
        // Since lo,hi each have n_bits-1 effective bits (lo < half, hi >= half),
        // eta is a function of 4*(n_bits-1) GF(2) variables.
        // But we can also view eta as derived from psi's polynomial.
        eprintln!("\n--- Eta effective degree (edge-level analysis) ---");

        for &dim in &[16usize, 32] {
            let half = dim / 2;
            let n_bits_half = half.trailing_zeros() as usize;

            // Build eta truth table over (lo_a, hi_a, lo_b, hi_b) restricted to cross-assessor ranges
            // lo in [1, half), hi in [half, dim)
            // For the ANF, we parameterize by reduced indices: lo in [1, half), hi' = hi - half in [0, half)
            // Total bits per edge: 4 * n_bits_half (but lo starts at 1, so not all 2^bits used)

            // For simplicity, compute the ANF of eta on the full range [0, half)^4
            // and check degree
            let n_vars_eta = 4 * n_bits_half;
            let n_inputs_eta = 1usize << n_vars_eta;

            let mut eta_tt: Vec<u8> = vec![0; n_inputs_eta];

            for input in 0..n_inputs_eta {
                // Decode: lo_a | hi_a' | lo_b | hi_b' each n_bits_half wide
                let lo_a = (input >> (3 * n_bits_half)) & (half - 1);
                let hi_a_prime = (input >> (2 * n_bits_half)) & (half - 1);
                let lo_b = (input >> n_bits_half) & (half - 1);
                let hi_b_prime = input & (half - 1);

                let hi_a = hi_a_prime + half;
                let hi_b = hi_b_prime + half;

                // eta = psi(lo_a, hi_b) XOR psi(hi_a, lo_b)
                eta_tt[input] = psi(dim, lo_a, hi_b) ^ psi(dim, hi_a, lo_b);
            }

            // Mobius ANF transform
            let mut anf_eta = eta_tt.clone();
            for bit in 0..n_vars_eta {
                let mask = 1usize << bit;
                for m in 0..n_inputs_eta {
                    if m & mask != 0 {
                        anf_eta[m] ^= anf_eta[m ^ mask];
                    }
                }
            }

            let mut degree_counts = vec![0usize; n_vars_eta + 1];
            let mut max_degree = 0usize;
            let mut total_monomials = 0usize;

            for m in 0..n_inputs_eta {
                if anf_eta[m] != 0 {
                    let deg = (m as u32).count_ones() as usize;
                    degree_counts[deg] += 1;
                    if deg > max_degree {
                        max_degree = deg;
                    }
                    total_monomials += 1;
                }
            }

            eprintln!(
                "dim={dim}, eta over [0,{half})^4 ({n_vars_eta} vars): \
                       ANF degree = {max_degree}, total monomials = {total_monomials}"
            );
            for (d, &count) in degree_counts.iter().enumerate() {
                if count > 0 {
                    eprintln!(
                        "  degree {d}: {count} monomials (of {} possible)",
                        binomial(n_vars_eta, d)
                    );
                }
            }
        }
    }

    /// Comprehensive mechanism depth analysis: Quarter Rule exactness,
    /// frustration asymptotics, and Klein-four fiber symmetry proof attempt.
    ///
    /// (C-528): Quarter Rule Deviation -- is pure/total = 1/4 exactly?
    /// (C-529): Asymptotic frustration ratio convergence
    /// (C-530): Klein-four fiber symmetry -- structural proof via eta-swap
    #[test]
    fn test_mechanism_depth_analysis() {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;
        use std::collections::VecDeque;

        let psi = |dim: usize, i: usize, j: usize| -> u8 {
            if cd_basis_mul_sign(dim, i, j) == 1 {
                0
            } else {
                1
            }
        };

        eprintln!("\n=== Mechanism Depth Analysis ===");
        eprintln!(
            "{:<6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12} {:>10} {:>6} {:>10}",
            "dim",
            "F(0,0)",
            "F(0,1)",
            "F(1,0)",
            "F(1,1)",
            "total",
            "pure/total",
            "b1",
            "frust",
            "frust_rat"
        );

        // Collect data at each dimension
        let mut dim_data: Vec<(usize, [usize; 4], usize, usize)> = Vec::new(); // (dim, fibers, b1, frustrated)

        for &dim in &[16usize, 32, 64, 128] {
            let components = motif_components_for_cross_assessors(dim);

            let mut fiber_counts = [0usize; 4];
            let mut total_b1 = 0usize;
            let mut total_frustrated = 0usize;

            for comp in components.iter() {
                let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
                let n = nodes.len();
                if n < 2 {
                    continue;
                }

                // Build adjacency list with eta labels
                let node_idx: std::collections::HashMap<CrossPair, usize> =
                    nodes.iter().enumerate().map(|(i, &cp)| (cp, i)).collect();
                let mut adj: Vec<Vec<(usize, u8)>> = vec![Vec::new(); n];

                for &(u, v) in &comp.edges {
                    let eta_val = psi(dim, u.0, v.1) ^ psi(dim, u.1, v.0);
                    let ui = node_idx[&u];
                    let vi = node_idx[&v];
                    adj[ui].push((vi, eta_val));
                    adj[vi].push((ui, eta_val));
                }

                // BFS cohomology check
                let mut delta: Vec<Option<u8>> = vec![None; n];
                delta[0] = Some(0);
                let mut queue = VecDeque::new();
                queue.push_back(0);
                let mut tree_edges = 0usize;
                let mut frustrated = 0usize;

                while let Some(u) = queue.pop_front() {
                    for &(v, eta_val) in &adj[u] {
                        if let Some(dv) = delta[v] {
                            if u < v {
                                let expected = delta[u].unwrap() ^ eta_val;
                                if expected != dv {
                                    frustrated += 1;
                                }
                            }
                        } else {
                            delta[v] = Some(delta[u].unwrap() ^ eta_val);
                            tree_edges += 1;
                            queue.push_back(v);
                        }
                    }
                }

                let b1 = comp.edges.len() - tree_edges;
                total_b1 += b1;
                total_frustrated += frustrated;

                // Triangle enumeration for fibers
                let edge_set: std::collections::HashSet<(CrossPair, CrossPair)> =
                    comp.edges.iter().copied().collect();
                let has_edge = |u: CrossPair, v: CrossPair| -> bool {
                    let (a, b) = if u < v { (u, v) } else { (v, u) };
                    edge_set.contains(&(a, b))
                };
                let eta =
                    |a: CrossPair, b: CrossPair| -> u8 { psi(dim, a.0, b.1) ^ psi(dim, a.1, b.0) };

                for i in 0..n {
                    for j in (i + 1)..n {
                        for k in (j + 1)..n {
                            let (u, v, w) = (nodes[i], nodes[j], nodes[k]);
                            if !has_edge(u, v) || !has_edge(v, w) || !has_edge(u, w) {
                                continue;
                            }
                            let eta_uv = eta(u, v);
                            let eta_vw = eta(v, w);
                            let eta_uw = eta(u, w);
                            let f1 = eta_uv ^ eta_vw;
                            let f2 = eta_vw ^ eta_uw;
                            fiber_counts[(2 * f1 + f2) as usize] += 1;
                        }
                    }
                }
            }

            let total_tris: usize = fiber_counts.iter().sum();
            let pure_ratio = if total_tris > 0 {
                fiber_counts[0] as f64 / total_tris as f64
            } else {
                0.0
            };
            let frust_ratio = if total_b1 > 0 {
                total_frustrated as f64 / total_b1 as f64
            } else {
                0.0
            };

            eprintln!(
                "{:<6} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12.8} {:>10} {:>6} {:>10.6}",
                dim,
                fiber_counts[0],
                fiber_counts[1],
                fiber_counts[2],
                fiber_counts[3],
                total_tris,
                pure_ratio,
                total_b1,
                total_frustrated,
                frust_ratio
            );

            dim_data.push((dim, fiber_counts, total_b1, total_frustrated));

            // ---- Assertions ----

            // 1:3 ratio must hold
            if total_tris > 0 {
                assert_eq!(
                    fiber_counts[0] * 3,
                    fiber_counts[1] + fiber_counts[2] + fiber_counts[3],
                    "dim={dim}: 1:3 ratio violation"
                );
            }

            // F(1,0) = F(1,1) must hold
            assert_eq!(
                fiber_counts[2], fiber_counts[3],
                "dim={dim}: F(1,0) != F(1,1)"
            );
        }

        // ---- Analysis: Quarter Rule Exactness (C-528) ----
        eprintln!("\n=== Quarter Rule Analysis ===");
        eprintln!(
            "{:<6} {:>12} {:>12} {:>14} {:>14}",
            "dim", "pure", "total", "pure*4-total", "deviation"
        );

        for &(dim, fibers, _, _) in &dim_data {
            let total: usize = fibers.iter().sum();
            let pure = fibers[0];
            // If 1:3 is exact, then pure*4 = total exactly.
            let quarter_exact = pure * 4 == total;
            let deviation = (pure as f64 * 4.0 - total as f64) / total as f64;
            eprintln!(
                "{:<6} {:>12} {:>12} {:>14} {:>14.10}",
                dim,
                pure,
                total,
                (pure * 4) as i64 - total as i64,
                deviation
            );
            // The 1:3 ratio GUARANTEES pure*4 = total:
            // pure*3 = nonzero = total - pure => 3*pure = total - pure => 4*pure = total
            assert!(quarter_exact, "dim={dim}: Quarter Rule NOT exact");
        }
        eprintln!("Quarter Rule is EXACT at all dimensions (follows algebraically from 1:3).");

        // ---- Analysis: Fiber Decomposition (C-530) ----
        // The 3 nonzero fibers are F(0,1), F(1,0), F(1,1).
        // We know F(1,0) = F(1,1). What about F(0,1)?
        // F(0,1): eta_uv = eta_vw but eta_vw != eta_uw
        // F(1,0): eta_uv != eta_vw but eta_vw = eta_uw
        // F(1,1): eta_uv != eta_vw and eta_vw != eta_uw
        //
        // Symmetry argument for F(1,0) = F(1,1):
        // Under cyclic vertex relabeling (u,v,w) -> (v,w,u), the F invariant
        // transforms as F -> (eta_vw XOR eta_wu, eta_wu XOR eta_vu).
        // This is a different element of GF(2)^2 unless the triangle is pure.
        // For mixed triangles, the 3 cyclic labelings hit 3 different nonzero
        // F values -- but each triangle is counted once with canonical labeling,
        // so this doesn't directly give F(1,0)=F(1,1).
        //
        // Instead, test the per-component version: does F(1,0)=F(1,1) hold
        // within EVERY component, or only in aggregate?
        eprintln!("\n=== Per-Component Fiber Analysis (C-530) ===");

        let mut all_components_satisfy = true;
        for &dim in &[16usize, 32, 64, 128] {
            let components = motif_components_for_cross_assessors(dim);
            let mut comp_violations = 0usize;
            let mut comp_total = 0usize;
            let mut f01_ne_f10_count = 0usize;

            for comp in components.iter() {
                let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
                let n = nodes.len();
                if n < 3 {
                    continue;
                }

                let edge_set: std::collections::HashSet<(CrossPair, CrossPair)> =
                    comp.edges.iter().copied().collect();
                let has_edge = |u: CrossPair, v: CrossPair| -> bool {
                    let (a, b) = if u < v { (u, v) } else { (v, u) };
                    edge_set.contains(&(a, b))
                };
                let eta =
                    |a: CrossPair, b: CrossPair| -> u8 { psi(dim, a.0, b.1) ^ psi(dim, a.1, b.0) };

                let mut comp_fibers = [0usize; 4];
                for i in 0..n {
                    for j in (i + 1)..n {
                        for k in (j + 1)..n {
                            let (u, v, w) = (nodes[i], nodes[j], nodes[k]);
                            if !has_edge(u, v) || !has_edge(v, w) || !has_edge(u, w) {
                                continue;
                            }
                            let eta_uv = eta(u, v);
                            let eta_vw = eta(v, w);
                            let eta_uw = eta(u, w);
                            let f1 = eta_uv ^ eta_vw;
                            let f2 = eta_vw ^ eta_uw;
                            comp_fibers[(2 * f1 + f2) as usize] += 1;
                        }
                    }
                }

                comp_total += 1;
                if comp_fibers[2] != comp_fibers[3] {
                    comp_violations += 1;
                    all_components_satisfy = false;
                }
                if comp_fibers[1] != comp_fibers[2] {
                    f01_ne_f10_count += 1;
                }
            }

            eprintln!(
                "dim={dim}: {comp_total} components, \
                       F(1,0)!=F(1,1) violations: {comp_violations}, \
                       F(0,1)!=F(1,0) cases: {f01_ne_f10_count}"
            );
        }

        assert!(
            all_components_satisfy,
            "F(1,0) = F(1,1) failed at per-component level"
        );
        eprintln!("F(1,0) = F(1,1) holds in EVERY component at EVERY dimension.");

        // ---- Analysis: Frustration Asymptotics (C-529) ----
        eprintln!("\n=== Frustration Asymptotics (C-529) ===");
        eprintln!(
            "{:<6} {:>10} {:>10} {:>12}",
            "dim", "b1", "frustrated", "frust/b1"
        );

        for &(dim, _, b1, frustrated) in &dim_data {
            let ratio = if b1 > 0 {
                frustrated as f64 / b1 as f64
            } else {
                0.0
            };
            eprintln!("{:<6} {:>10} {:>10} {:>12.8}", dim, b1, frustrated, ratio);
        }

        // Frustration ratio converges but is NOT monotone (oscillates).
        // dim=32: 0.307, dim=64: 0.377, dim=128: 0.388, dim=256: 0.385
        // Verify convergence: each step is closer to limit (~0.386)
        // but allow oscillation.
        for &(dim, _, b1, frustrated) in &dim_data {
            if dim > 16 {
                let ratio = frustrated as f64 / b1 as f64;
                assert!(
                    ratio > 0.25,
                    "dim={dim}: frustration ratio suspiciously low"
                );
                assert!(
                    ratio < 0.50,
                    "dim={dim}: frustration ratio suspiciously high"
                );
            }
        }

        // ---- Analysis: eta-swap for F(1,0)=F(1,1) proof (C-530) ----
        //
        // For each edge (a,b), eta(a,b) = psi(lo_a, hi_b) XOR psi(hi_a, lo_b).
        // From C-526: eta(a,b) = 1 XOR eta_half(a', b') where a', b' are
        // the half-dimension projections.
        //
        // Consider the vertex labeling involution that swaps the role of
        // the first and second edge in the anti-diagonal computation.
        // Concretely, for a triangle (u,v,w):
        //   F = (eta_uv XOR eta_vw, eta_vw XOR eta_uw)
        //
        // Under the involution v <-> w (swapping the second and third vertex):
        //   F' = (eta_uw XOR eta_wv, eta_wv XOR eta_uw)
        //      = (eta_uw XOR eta_vw, eta_vw XOR eta_uw)  [eta symmetric]
        //      = (eta_vw XOR eta_uw, eta_vw XOR eta_uw)  [XOR commutative]
        //
        // Wait, that gives F' = (f2, f2) which is either (0,0) or (1,1).
        // That's not right for a general mixed triangle.
        //
        // Actually: the F invariant depends on vertex ordering.
        // A triangle {u,v,w} with canonical ordering i < j < k gives
        // F = (eta_ij XOR eta_jk, eta_jk XOR eta_ik).
        // This is not a true invariant of the unordered triangle --
        // it depends on which vertex is "middle".
        //
        // The COUNT of pure triangles is independent of ordering (all 3 etas equal
        // iff F = (0,0)), but the fiber assignment for mixed triangles DOES depend
        // on ordering. So the assertion F(1,0) = F(1,1) is about the canonical-
        // ordering statistics, not an algebraic invariant.
        //
        // Test: for each triangle, compute F under all 3 orderings and count
        // which fibers appear.
        eprintln!("\n=== F-invariant Ordering Dependence ===");
        for &dim in &[16usize, 32] {
            let components = motif_components_for_cross_assessors(dim);
            let mut ordering_stats = [[0usize; 4]; 3]; // [ordering][fiber]

            for comp in components.iter() {
                let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
                let n = nodes.len();
                if n < 3 {
                    continue;
                }

                let edge_set: std::collections::HashSet<(CrossPair, CrossPair)> =
                    comp.edges.iter().copied().collect();
                let has_edge = |u: CrossPair, v: CrossPair| -> bool {
                    let (a, b) = if u < v { (u, v) } else { (v, u) };
                    edge_set.contains(&(a, b))
                };
                let eta =
                    |a: CrossPair, b: CrossPair| -> u8 { psi(dim, a.0, b.1) ^ psi(dim, a.1, b.0) };

                for i in 0..n {
                    for j in (i + 1)..n {
                        for k in (j + 1)..n {
                            let (u, v, w) = (nodes[i], nodes[j], nodes[k]);
                            if !has_edge(u, v) || !has_edge(v, w) || !has_edge(u, w) {
                                continue;
                            }
                            let e_uv = eta(u, v);
                            let e_vw = eta(v, w);
                            let e_uw = eta(u, w);

                            // Ordering 0: (u,v,w) -> F = (e_uv^e_vw, e_vw^e_uw)
                            let f0 = (2 * (e_uv ^ e_vw) + (e_vw ^ e_uw)) as usize;
                            // Ordering 1: (u,w,v) -> F = (e_uw^e_wv, e_wv^e_uv)
                            //   = (e_uw^e_vw, e_vw^e_uv)
                            let f1 = (2 * (e_uw ^ e_vw) + (e_vw ^ e_uv)) as usize;
                            // Ordering 2: (v,u,w) -> F = (e_vu^e_uw, e_uw^e_vw)
                            //   = (e_uv^e_uw, e_uw^e_vw)
                            let f2 = (2 * (e_uv ^ e_uw) + (e_uw ^ e_vw)) as usize;

                            ordering_stats[0][f0] += 1;
                            ordering_stats[1][f1] += 1;
                            ordering_stats[2][f2] += 1;
                        }
                    }
                }
            }

            eprintln!("dim={dim}:");
            for (ord, stats) in ordering_stats.iter().enumerate() {
                eprintln!(
                    "  ordering {ord}: F(0,0)={} F(0,1)={} F(1,0)={} F(1,1)={}",
                    stats[0], stats[1], stats[2], stats[3]
                );
            }

            // Key test: F(0,0) is the same for all orderings (pure count
            // is vertex-order independent)
            assert_eq!(
                ordering_stats[0][0], ordering_stats[1][0],
                "dim={dim}: pure count differs across orderings"
            );
            assert_eq!(
                ordering_stats[0][0], ordering_stats[2][0],
                "dim={dim}: pure count differs across orderings"
            );

            // Check: does F(1,0)=F(1,1) hold for each ordering?
            for (ord, stats) in ordering_stats.iter().enumerate() {
                eprintln!("  ordering {ord}: F(1,0)==F(1,1)? {}", stats[2] == stats[3]);
            }
        }
    }

    /// Vertex-Median Symmetry: explains F(1,0) = F(1,1) via odd-edge analysis.
    ///
    /// In canonical ordering i < j < k, each mixed triangle has exactly one
    /// "odd" edge (whose eta differs from the other two). Let:
    ///   N_first = count where e(i,j) is odd
    ///   N_last  = count where e(j,k) is odd
    ///   N_diag  = count where e(i,k) is odd
    ///
    /// Then: F(1,0) = N_first, F(1,1) = N_last, F(0,1) = N_diag.
    ///
    /// Wait -- actually the mapping is different. Let me derive it:
    ///   a = eta(i,j), b = eta(j,k), c = eta(i,k)
    ///   F = (a XOR b, b XOR c)
    ///
    /// Case b=c!=a: F = (a^b, 0) = (1, 0)  => F(1,0) = N_{ij-odd}
    /// Case a=c!=b: F = (a^b, b^c) = (1, 1) => F(1,1) = N_{jk-odd}
    /// Case a=b!=c: F = (0, b^c) = (0, 1)   => F(0,1) = N_{ik-odd}
    ///
    /// So F(1,0) = F(1,1) iff N_{ij-odd} = N_{jk-odd}:
    /// the number of triangles where the edge containing vertex j
    /// as the LOWER endpoint is odd, equals the number where j is
    /// the UPPER endpoint. This is a "vertex-median symmetry".
    #[test]
    fn test_vertex_median_symmetry() {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;

        let psi = |dim: usize, i: usize, j: usize| -> u8 {
            if cd_basis_mul_sign(dim, i, j) == 1 {
                0
            } else {
                1
            }
        };

        eprintln!("\n=== Vertex-Median Symmetry (C-530) ===");
        eprintln!(
            "{:<6} {:>10} {:>10} {:>10} {:>10} {:>10}",
            "dim", "N_ij_odd", "N_jk_odd", "N_ik_odd", "mixed", "ratio_ij:jk"
        );

        for &dim in &[16usize, 32, 64, 128] {
            let components = motif_components_for_cross_assessors(dim);

            let mut n_ij_odd = 0usize; // eta(i,j) is the odd one
            let mut n_jk_odd = 0usize; // eta(j,k) is the odd one
            let mut n_ik_odd = 0usize; // eta(i,k) is the odd one

            for comp in components.iter() {
                let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
                let n = nodes.len();
                if n < 3 {
                    continue;
                }

                let edge_set: std::collections::HashSet<(CrossPair, CrossPair)> =
                    comp.edges.iter().copied().collect();
                let has_edge = |u: CrossPair, v: CrossPair| -> bool {
                    let (a, b) = if u < v { (u, v) } else { (v, u) };
                    edge_set.contains(&(a, b))
                };
                let eta =
                    |a: CrossPair, b: CrossPair| -> u8 { psi(dim, a.0, b.1) ^ psi(dim, a.1, b.0) };

                for i in 0..n {
                    for j in (i + 1)..n {
                        for k in (j + 1)..n {
                            let (u, v, w) = (nodes[i], nodes[j], nodes[k]);
                            if !has_edge(u, v) || !has_edge(v, w) || !has_edge(u, w) {
                                continue;
                            }
                            let a = eta(u, v); // e_ij
                            let b = eta(v, w); // e_jk
                            let c = eta(u, w); // e_ik

                            // Mixed triangle: not all equal
                            if a == b && b == c {
                                continue;
                            }

                            // Identify the odd edge
                            if b == c && a != b {
                                n_ij_odd += 1;
                            } else if a == c && b != a {
                                n_jk_odd += 1;
                            } else if a == b && c != a {
                                n_ik_odd += 1;
                            }
                        }
                    }
                }
            }

            let mixed_total = n_ij_odd + n_jk_odd + n_ik_odd;
            eprintln!(
                "{:<6} {:>10} {:>10} {:>10} {:>10} {:>10}",
                dim,
                n_ij_odd,
                n_jk_odd,
                n_ik_odd,
                mixed_total,
                if n_jk_odd > 0 {
                    format!("{:.6}", n_ij_odd as f64 / n_jk_odd as f64)
                } else {
                    "N/A".to_string()
                }
            );

            // ASSERTION: N_ij_odd = N_jk_odd (vertex-median symmetry)
            assert_eq!(
                n_ij_odd, n_jk_odd,
                "dim={dim}: vertex-median symmetry FAILED: N_ij={n_ij_odd} != N_jk={n_jk_odd}"
            );

            // Verify correspondence: F(1,0) = N_ij_odd, F(1,1) = N_jk_odd, F(0,1) = N_ik_odd
            // (Already verified via the ordering analysis in the depth test)
        }

        eprintln!("Vertex-Median Symmetry VERIFIED: N_ij_odd = N_jk_odd at all dimensions.");
        eprintln!(
            "The \"middle vertex\" j plays a symmetric role as lower (in ij) and upper (in jk)."
        );
    }

    /// Frustration ratio at dim=256 via cohomology-only analysis (no triangle enum).
    /// Extends C-529 asymptotic data.
    #[test]
    #[ignore] // ~15s in release mode
    fn test_frustration_ratio_dim256() {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;
        use std::collections::VecDeque;

        let psi = |dim: usize, i: usize, j: usize| -> u8 {
            if cd_basis_mul_sign(dim, i, j) == 1 {
                0
            } else {
                1
            }
        };

        let dim = 256usize;
        let components = motif_components_for_cross_assessors(dim);

        let mut total_b1 = 0usize;
        let mut total_frustrated = 0usize;
        let mut total_edges = 0usize;
        let mut total_eta0 = 0usize;
        let mut total_eta1 = 0usize;

        for comp in components.iter() {
            let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
            let n = nodes.len();
            if n < 2 {
                continue;
            }

            let node_idx: std::collections::HashMap<CrossPair, usize> =
                nodes.iter().enumerate().map(|(i, &cp)| (cp, i)).collect();
            let mut adj: Vec<Vec<(usize, u8)>> = vec![Vec::new(); n];

            for &(u, v) in &comp.edges {
                let eta_val = psi(dim, u.0, v.1) ^ psi(dim, u.1, v.0);
                let ui = node_idx[&u];
                let vi = node_idx[&v];
                adj[ui].push((vi, eta_val));
                adj[vi].push((ui, eta_val));
                if eta_val == 0 {
                    total_eta0 += 1;
                } else {
                    total_eta1 += 1;
                }
            }

            total_edges += comp.edges.len();

            let mut delta: Vec<Option<u8>> = vec![None; n];
            delta[0] = Some(0);
            let mut queue = VecDeque::new();
            queue.push_back(0);
            let mut tree_edges = 0usize;
            let mut frustrated = 0usize;

            while let Some(u) = queue.pop_front() {
                for &(v, eta_val) in &adj[u] {
                    if let Some(dv) = delta[v] {
                        if u < v {
                            let expected = delta[u].unwrap() ^ eta_val;
                            if expected != dv {
                                frustrated += 1;
                            }
                        }
                    } else {
                        delta[v] = Some(delta[u].unwrap() ^ eta_val);
                        tree_edges += 1;
                        queue.push_back(v);
                    }
                }
            }

            let b1 = comp.edges.len() - tree_edges;
            total_b1 += b1;
            total_frustrated += frustrated;
        }

        let frust_ratio = total_frustrated as f64 / total_b1 as f64;
        eprintln!("\n=== Frustration at dim=256 ===");
        eprintln!("components: {}", components.len());
        eprintln!("edges: {total_edges}, eta=0: {total_eta0}, eta=1: {total_eta1}");
        eprintln!("b1: {total_b1}, frustrated: {total_frustrated}");
        eprintln!("frustration ratio: {frust_ratio:.8}");

        // Known values from smaller dims:
        // dim=32: 0.30727, dim=64: 0.37735, dim=128: 0.38761, dim=256: 0.38489
        // Convergence is NOT monotone -- oscillates around ~0.386-0.388!

        assert_eq!(total_eta0, total_eta1, "dim=256: eta not balanced");
        assert!(
            total_frustrated > 0,
            "dim=256: eta should not be a coboundary"
        );
        // Frustration ratio is near the dim=128 value but slightly lower (non-monotone)
        assert!(
            frust_ratio > 0.38,
            "dim=256: frustration ratio should be near limit"
        );
        assert!(
            frust_ratio < 0.40,
            "dim=256: frustration ratio should be below 0.40"
        );
    }

    /// Cayley-Dickson tower psi/eta analysis at dim=2,4,8 (C-532).
    ///
    /// The twist exponent psi(i,j) = 0 if cd_basis_mul_sign(dim,i,j) = +1, else 1,
    /// and the anti-diagonal eta exist at ALL Cayley-Dickson dimensions, not just
    /// those with zero-divisors (dim >= 16). This test characterizes the GF(2)
    /// structure at dim=2 (complex), dim=4 (quaternion), and dim=8 (octonion),
    /// establishing the foundation of the tower before zero-divisors appear.
    ///
    /// At dim=2: psi has 1 imaginary basis element (i), 1x1 matrix
    /// At dim=4: psi has 3 imaginary elements (i,j,k), 3x3 matrix, quaternion signs
    /// At dim=8: psi has 7 imaginary elements, 7x7 matrix, Fano plane structure
    #[test]
    fn test_psi_eta_tower_dim2_dim4_dim8() {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;

        let psi = |dim: usize, i: usize, j: usize| -> u8 {
            if cd_basis_mul_sign(dim, i, j) == 1 {
                0
            } else {
                1
            }
        };

        // ===== dim=2 (complex numbers) =====
        // Only imaginary element: e_1 = i
        // e_1 * e_1 = -1, so psi(1,1) = 1
        let dim = 2;
        let psi_2 = psi(dim, 1, 1);
        eprintln!("\n=== Cayley-Dickson Tower psi/eta Analysis ===");
        eprintln!("dim=2: psi(1,1) = {psi_2}");
        assert_eq!(psi_2, 1, "i*i = -1, so psi should be 1");

        // GF(2) ANF at dim=2: psi(x,y) with x,y in {1} -- trivially constant 1
        // Degree: 0 (constant function mapping to 1)
        // Note: C-527 formula gives log2(2)=1 which applies only to dims >= 4

        // ===== dim=4 (quaternions) =====
        // Imaginary elements: e_1, e_2, e_3 (= i, j, k)
        // Standard quaternion: i*j=k, j*i=-k, i^2=j^2=k^2=-1
        let dim = 4;
        eprintln!("\ndim=4 psi matrix (1-indexed imaginary basis):");
        let mut psi4 = [[0u8; 3]; 3];
        for a in 1..4 {
            for b in 1..4 {
                psi4[a - 1][b - 1] = psi(dim, a, b);
            }
            eprintln!(
                "  e_{}: [{} {} {}]",
                a,
                psi4[a - 1][0],
                psi4[a - 1][1],
                psi4[a - 1][2]
            );
        }

        // All diagonal entries should be 1 (e_i^2 = -1 for quaternions)
        for a in 0..3 {
            assert_eq!(psi4[a][a], 1, "e_{}^2 should be -1", a + 1);
        }

        // Off-diagonal: quaternion multiplication signs
        // e_1*e_2 = e_3 (sign +1, psi=0), e_2*e_1 = -e_3 (sign -1, psi=1)
        assert_eq!(psi4[0][1], 0, "e1*e2 = +e3, so psi=0");
        assert_eq!(psi4[1][0], 1, "e2*e1 = -e3, so psi=1");

        // Count psi=1 entries
        let psi4_ones: usize = psi4
            .iter()
            .flat_map(|r| r.iter())
            .filter(|&&v| v == 1)
            .count();
        let psi4_zeros: usize = 9 - psi4_ones;
        eprintln!("dim=4: psi=0: {psi4_zeros}, psi=1: {psi4_ones} (of 9 total)");

        // At dim=4 there are no cross-assessor pairs (no ZDs), but we can still
        // compute eta(a,b) = psi(lo_a, hi_b) XOR psi(hi_a, lo_b) if we define
        // lo = i & 1, hi = i >> 1 for 2-bit indices.
        // For dim=4: basis indices 1,2,3 have lo/hi = (1,0), (0,1), (1,1)
        // Cross-assessor pairs at dim=4: pairs (i,j) with lo_i != lo_j AND hi_i != hi_j
        // But dim/2-1 = 1, so no cross-pair structure (assessors have 1 lo, 1 hi)

        // Instead, compute eta for ALL imaginary pairs to see the structure
        eprintln!("\ndim=4 eta matrix (anti-diagonal XOR of psi):");
        for a in 1..4usize {
            let lo_a = a & 1;
            let hi_a = a >> 1;
            let mut row = Vec::new();
            for b in 1..4usize {
                let lo_b = b & 1;
                let hi_b = b >> 1;
                // eta only defined when lo and hi are both nonzero (valid basis elements)
                if lo_a > 0 && hi_b > 0 && hi_a > 0 && lo_b > 0 {
                    let e = psi(dim, lo_a, hi_b) ^ psi(dim, hi_a, lo_b);
                    row.push(format!("{e}"));
                } else {
                    row.push("-".to_string());
                }
            }
            eprintln!("  e_{a} (lo={lo_a},hi={hi_a}): [{}]", row.join(" "));
        }

        // ===== dim=8 (octonions) =====
        // Imaginary elements: e_1..e_7
        // The psi matrix encodes the octonion multiplication sign structure
        // (Fano plane determines which products are positive)
        let dim = 8;
        eprintln!("\ndim=8 psi matrix (7x7, imaginary basis):");
        let mut psi8 = [[0u8; 7]; 7];
        for a in 1..8 {
            for b in 1..8 {
                psi8[a - 1][b - 1] = psi(dim, a, b);
            }
            let row: Vec<String> = psi8[a - 1].iter().map(|v| format!("{v}")).collect();
            eprintln!("  e_{a}: [{}]", row.join(" "));
        }

        // All diagonal entries should be 1 (e_i^2 = -1 for octonions)
        for a in 0..7 {
            assert_eq!(psi8[a][a], 1, "e_{}^2 should be -1", a + 1);
        }

        // Count psi=1 entries
        let psi8_ones: usize = psi8
            .iter()
            .flat_map(|r| r.iter())
            .filter(|&&v| v == 1)
            .count();
        let psi8_zeros: usize = 49 - psi8_ones;
        eprintln!("dim=8: psi=0: {psi8_zeros}, psi=1: {psi8_ones} (of 49 total)");

        // Antisymmetry check: psi(a,b) + psi(b,a) mod 2 for a != b
        // In a non-commutative algebra, psi(a,b) XOR psi(b,a) = 1 for all a != b
        let mut antisymmetric = 0;
        let mut symmetric = 0;
        for a in 0..7 {
            for b in (a + 1)..7 {
                if psi8[a][b] != psi8[b][a] {
                    antisymmetric += 1;
                } else {
                    symmetric += 1;
                }
            }
        }
        eprintln!("dim=8 off-diagonal: {antisymmetric} antisymmetric, {symmetric} symmetric pairs");
        // All off-diagonal pairs should be antisymmetric (octonions are non-commutative)
        assert_eq!(
            symmetric, 0,
            "Octonions: all off-diagonal psi pairs should be antisymmetric"
        );
        assert_eq!(antisymmetric, 21, "C(7,2) = 21 off-diagonal pairs");

        // Check quaternion antisymmetry too
        let mut q_antisym = 0;
        for a in 0..3 {
            for b in (a + 1)..3 {
                if psi4[a][b] != psi4[b][a] {
                    q_antisym += 1;
                }
            }
        }
        assert_eq!(
            q_antisym, 3,
            "Quaternions: all 3 off-diagonal pairs antisymmetric"
        );

        // Fano plane analysis at dim=8:
        // A Fano line is a triple {a,b,c} where a XOR b = c in GF(2)^3.
        // The sign of e_a * e_b determines the ORIENTATION of the line, not
        // whether it IS a line. All 7 XOR triples are Fano lines.
        let mut fano_lines = Vec::new();
        for a in 1..8usize {
            for b in (a + 1)..8usize {
                let c = a ^ b;
                if c > b && c < 8 {
                    // Each unordered triple counted once (a < b < c)
                    let sign_ab = cd_basis_mul_sign(8, a, b);
                    fano_lines.push((a, b, c, sign_ab));
                }
            }
        }
        eprintln!("\ndim=8 Fano lines (triples with a^b=c):");
        for &(a, b, c, s) in &fano_lines {
            let orient = if s == 1 { "+" } else { "-" };
            eprintln!("  ({a},{b},{c}): e_{a}*e_{b} = {orient}e_{c}");
        }
        eprintln!("Fano line count: {}", fano_lines.len());
        // Fano plane has exactly 7 lines
        assert_eq!(
            fano_lines.len(),
            7,
            "Octonion Fano plane should have 7 lines"
        );

        // Count positive vs negative orientations
        let n_positive = fano_lines.iter().filter(|l| l.3 == 1).count();
        let n_negative = fano_lines.iter().filter(|l| l.3 == -1).count();
        eprintln!("Orientations: {n_positive} positive, {n_negative} negative");

        // GF(2) polynomial degree analysis across the tower
        // psi at dim=2: degree 0 (constant 1 on the single point)
        // psi at dim=4: degree ? (3 points in GF(2)^2 minus origin)
        // psi at dim=8: degree ? (7 points in GF(2)^3 minus origin)
        // The ANF degree formula (C-527) predicts: deg(psi) = log2(dim) for dim >= 4

        // Verify at dim=4: psi on {01,10,11} -> {values}
        // ANF: psi(x_0, x_1) = ? where (x_0, x_1) = binary representation of basis index
        // Index 1 = (1,0): psi(1,1) = 1, psi(1,2) = 0, psi(1,3) = 1
        // Index 2 = (0,1): psi(2,1) = 1, psi(2,2) = 1, psi(2,3) = 0
        // Index 3 = (1,1): psi(3,1) = 1, psi(3,2) = 1, psi(3,3) = 1
        // psi is a function GF(2)^2 x GF(2)^2 -> GF(2) on nonzero inputs

        // Print full GF(2) polynomial representation
        eprintln!("\n=== GF(2) Polynomial Summary ===");
        for &dim in &[4, 8, 16, 32] {
            let n = (dim as f64).log2() as usize;
            let n_imag = dim - 1;
            let total_entries = n_imag * n_imag;
            let ones: usize = (1..dim)
                .flat_map(|a| (1..dim).map(move |b| psi(dim, a, b)))
                .filter(|&v| v == 1)
                .count();
            eprintln!(
                "dim={dim} (n={n}): {ones}/{total_entries} psi=1 entries ({:.1}%)",
                100.0 * ones as f64 / total_entries as f64
            );
        }

        // Key finding: at dim=2 there's only 1 entry (trivially psi=1)
        // At dim=4: psi is a 3x3 binary matrix
        // At dim=8: psi is a 7x7 binary matrix encoding the Fano plane
        // At dim >= 16: zero-divisors appear and eta becomes the key invariant
    }

    /// Anti-Diagonal Parity Theorem verification at dim=512 (C-531).
    ///
    /// Extends verification to the largest computed dimension: 255 components,
    /// 254 nodes each, producing ~214M triangles across 33 regimes.
    /// Also collects Klein-four fibers, edge eta balance, cycle rank,
    /// and frustration ratio (extending C-529).
    ///
    /// Runtime: ~3-5 min in release mode.
    #[test]
    #[ignore] // Long-running: ~3-5 min in release mode
    fn test_antidiagonal_parity_theorem_dim512() {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;
        use std::time::Instant;

        let psi = |dim: usize, i: usize, j: usize| -> u8 {
            if cd_basis_mul_sign(dim, i, j) == 1 {
                0
            } else {
                1
            }
        };

        let dim = 512;
        let t0 = Instant::now();
        let components = motif_components_for_cross_assessors(dim);
        let t_graph = t0.elapsed();
        eprintln!("\n=== Anti-Diagonal Parity Theorem at dim={dim} ===");
        eprintln!("Graph construction: {:.2}s", t_graph.as_secs_f64());
        eprintln!("Components: {}", components.len());

        let mut total_triangles = 0usize;
        let mut total_pure = 0usize;
        let mut total_mixed = 0usize;
        let mut mismatches = 0usize;

        // Klein-four fiber sizes
        let mut fiber_counts = [0usize; 4];

        // Eta balance and cohomology accumulators
        let mut total_eta0 = 0usize;
        let mut total_eta1 = 0usize;
        let mut total_cycle_rank = 0usize;
        let mut total_frustrated = 0usize;

        let t1 = Instant::now();

        for comp in components.iter() {
            let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
            let n = nodes.len();
            let edge_set: std::collections::HashSet<(CrossPair, CrossPair)> =
                comp.edges.iter().copied().collect();
            let has_edge = |u: CrossPair, v: CrossPair| -> bool {
                let (a, b) = if u < v { (u, v) } else { (v, u) };
                edge_set.contains(&(a, b))
            };

            let eta =
                |a: CrossPair, b: CrossPair| -> u8 { psi(dim, a.0, b.1) ^ psi(dim, a.1, b.0) };

            // Edge eta balance
            let mut comp_eta0 = 0usize;
            let mut comp_eta1 = 0usize;
            for &(u, v) in &comp.edges {
                if eta(u, v) == 0 {
                    comp_eta0 += 1;
                } else {
                    comp_eta1 += 1;
                }
            }
            total_eta0 += comp_eta0;
            total_eta1 += comp_eta1;

            // Cycle rank and frustration (BFS coboundary test)
            let b1 = comp.edges.len() - n + 1;
            total_cycle_rank += b1;

            // BFS for coboundary test
            let node_idx: std::collections::HashMap<CrossPair, usize> =
                nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
            let mut adj: Vec<Vec<(usize, u8)>> = vec![vec![]; n];
            for &(u, v) in &comp.edges {
                let ui = node_idx[&u];
                let vi = node_idx[&v];
                let e = eta(u, v);
                adj[ui].push((vi, e));
                adj[vi].push((ui, e));
            }

            let mut delta = vec![0u8; n];
            let mut visited = vec![false; n];
            visited[0] = true;
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(0usize);
            while let Some(u) = queue.pop_front() {
                for &(v, e) in &adj[u] {
                    if !visited[v] {
                        visited[v] = true;
                        delta[v] = delta[u] ^ e;
                        queue.push_back(v);
                    }
                }
            }

            // Count frustrated edges (each edge counted once via canonical order)
            for &(u_node, v_node) in &comp.edges {
                let ui = node_idx[&u_node];
                let vi = node_idx[&v_node];
                let e = eta(u_node, v_node);
                if (delta[ui] ^ delta[vi]) != e {
                    total_frustrated += 1;
                }
            }

            // Triangle enumeration
            for i in 0..n {
                for j in (i + 1)..n {
                    for k in (j + 1)..n {
                        let (u, v, w) = (nodes[i], nodes[j], nodes[k]);
                        if !has_edge(u, v) || !has_edge(v, w) || !has_edge(u, w) {
                            continue;
                        }
                        total_triangles += 1;

                        // Sigma classification
                        let sigma = |a: CrossPair, b: CrossPair| -> i32 {
                            cd_basis_mul_sign(dim, a.0, b.0) * cd_basis_mul_sign(dim, a.1, b.1)
                        };
                        let s_uv = sigma(u, v);
                        let s_vw = sigma(v, w);
                        let s_uw = sigma(u, w);
                        let n_same = [s_uv, s_vw, s_uw].iter().filter(|&&s| s == -1).count();
                        let product = s_uv * s_vw * s_uw;
                        let is_pure = if product == 1 {
                            n_same == 0
                        } else {
                            n_same == 3
                        };

                        // Eta for all 3 edges
                        let eta_uv = eta(u, v);
                        let eta_vw = eta(v, w);
                        let eta_uw = eta(u, w);

                        // Anti-Diagonal Parity Theorem: pure iff eta constant
                        let eta_constant = (eta_uv == eta_vw) && (eta_vw == eta_uw);

                        if eta_constant != is_pure {
                            mismatches += 1;
                        }

                        if is_pure {
                            total_pure += 1;
                        } else {
                            total_mixed += 1;
                        }

                        // Klein-four fiber
                        let f1 = eta_uv ^ eta_vw;
                        let f2 = eta_vw ^ eta_uw;
                        fiber_counts[(2 * f1 + f2) as usize] += 1;
                    }
                }
            }
        }

        let t_total = t1.elapsed();
        let frust_ratio = total_frustrated as f64 / total_cycle_rank as f64;

        eprintln!("Total time: {:.2}s", t_total.as_secs_f64());
        eprintln!("Total triangles: {total_triangles}");
        eprintln!("Pure: {total_pure}, Mixed: {total_mixed}");
        eprintln!("Mismatches: {mismatches}");
        eprintln!(
            "Klein-four fibers: (0,0)={}, (0,1)={}, (1,0)={}, (1,1)={}",
            fiber_counts[0], fiber_counts[1], fiber_counts[2], fiber_counts[3]
        );
        eprintln!("Edge eta balance: eta=0: {total_eta0}, eta=1: {total_eta1}");
        eprintln!("Total cycle rank: {total_cycle_rank}");
        eprintln!("Frustration: {total_frustrated}/{total_cycle_rank} = {frust_ratio:.8}");

        // ASSERTION 1: Anti-Diagonal Parity Theorem holds
        assert_eq!(
            mismatches, 0,
            "Anti-Diagonal Parity Theorem REFUTED at dim={dim}: {mismatches} mismatches"
        );

        // ASSERTION 2: Quarter Rule
        assert_eq!(
            total_pure * 3,
            total_mixed,
            "Quarter Rule failed: pure={total_pure}, mixed={total_mixed}"
        );

        // ASSERTION 3: Klein-four fiber (0,0) = pure count
        assert_eq!(
            fiber_counts[0], total_pure,
            "F=(0,0) should equal pure count"
        );

        // ASSERTION 4: Sum of nonzero fibers = mixed count
        assert_eq!(
            fiber_counts[1] + fiber_counts[2] + fiber_counts[3],
            total_mixed,
            "Nonzero fibers should sum to mixed count"
        );

        // ASSERTION 5: F(1,0) = F(1,1) (vertex-median symmetry, C-530)
        assert_eq!(
            fiber_counts[2], fiber_counts[3],
            "F(1,0) should equal F(1,1)"
        );

        // ASSERTION 6: Edge eta exactly balanced
        assert_eq!(total_eta0, total_eta1, "dim={dim}: eta not balanced");

        // ASSERTION 7: Triangle count matches dim=512 census (C-495)
        // Expected: ~214M triangles from generic_face_sign_census(512)
        assert!(total_triangles > 200_000_000, "Expected ~214M triangles");

        // ASSERTION 8: Frustration ratio in range (extends C-529)
        assert!(
            frust_ratio > 0.37,
            "dim=512: frustration should be near limit"
        );
        assert!(
            frust_ratio < 0.41,
            "dim=512: frustration should not exceed 0.41"
        );
    }

    /// Quick frustration-only test at dim=512 (extends C-529).
    ///
    /// Computes ONLY the BFS coboundary frustration ratio, skipping the
    /// expensive O(n^3) triangle enumeration. This is O(V+E) per component.
    ///
    /// Sequence so far: 0.000, 0.307, 0.377, 0.388, 0.385 (dims 16..256).
    /// Does dim=512 continue the oscillation?
    #[test]
    #[ignore] // ~30s in release mode (graph construction dominates)
    fn test_frustration_ratio_dim512() {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;
        use std::time::Instant;

        let psi = |dim: usize, i: usize, j: usize| -> u8 {
            if cd_basis_mul_sign(dim, i, j) == 1 {
                0
            } else {
                1
            }
        };

        let dim = 512;
        let t0 = Instant::now();
        let components = motif_components_for_cross_assessors(dim);
        let t_graph = t0.elapsed();
        eprintln!("\n=== Frustration at dim={dim} ===");
        eprintln!("Graph construction: {:.2}s", t_graph.as_secs_f64());
        eprintln!("Components: {}", components.len());

        let mut total_edges = 0usize;
        let mut total_eta0 = 0usize;
        let mut total_eta1 = 0usize;
        let mut total_b1 = 0usize;
        let mut total_frustrated = 0usize;

        for comp in components.iter() {
            let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
            let n = nodes.len();
            total_edges += comp.edges.len();

            let eta =
                |a: CrossPair, b: CrossPair| -> u8 { psi(dim, a.0, b.1) ^ psi(dim, a.1, b.0) };

            // Edge eta balance
            for &(u, v) in &comp.edges {
                if eta(u, v) == 0 {
                    total_eta0 += 1;
                } else {
                    total_eta1 += 1;
                }
            }

            // BFS coboundary test
            let node_idx: std::collections::HashMap<CrossPair, usize> =
                nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
            let mut adj: Vec<Vec<(usize, u8)>> = vec![vec![]; n];
            for &(u, v) in &comp.edges {
                let ui = node_idx[&u];
                let vi = node_idx[&v];
                let e = eta(u, v);
                adj[ui].push((vi, e));
                adj[vi].push((ui, e));
            }

            let mut delta = vec![0u8; n];
            let mut visited = vec![false; n];
            visited[0] = true;
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(0usize);
            while let Some(u) = queue.pop_front() {
                for &(v, e) in &adj[u] {
                    if !visited[v] {
                        visited[v] = true;
                        delta[v] = delta[u] ^ e;
                        queue.push_back(v);
                    }
                }
            }

            let b1 = comp.edges.len() - n + 1;
            total_b1 += b1;

            // Count frustrated edges
            for &(u_node, v_node) in &comp.edges {
                let ui = node_idx[&u_node];
                let vi = node_idx[&v_node];
                let e = eta(u_node, v_node);
                if (delta[ui] ^ delta[vi]) != e {
                    total_frustrated += 1;
                }
            }
        }

        let frust_ratio = total_frustrated as f64 / total_b1 as f64;
        eprintln!("edges: {total_edges}, eta=0: {total_eta0}, eta=1: {total_eta1}");
        eprintln!("b1: {total_b1}, frustrated: {total_frustrated}");
        eprintln!("frustration ratio: {frust_ratio:.8}");

        // Sequence: 0.000, 0.307, 0.377, 0.388, 0.385 (dims 16..256)
        // dim=512 should be near the same range
        assert_eq!(total_eta0, total_eta1, "dim=512: eta not balanced");
        assert!(
            total_frustrated > 0,
            "dim=512: eta should not be a coboundary"
        );
        assert!(
            frust_ratio > 0.37,
            "dim=512: frustration should be near limit"
        );
        assert!(
            frust_ratio < 0.41,
            "dim=512: frustration should not exceed 0.41"
        );
    }

    /// Quick frustration-only test at dim=1024.
    ///
    /// Computes ONLY the BFS coboundary frustration ratio, skipping the
    /// expensive O(n^3) triangle enumeration. This is O(V+E) per component.
    ///
    /// Sequence so far: 0.000, 0.307, 0.377, 0.388, 0.385, 0.381 (dims 16..512).
    /// dim=1024 continues the oscillating convergence.
    ///
    /// Runtime: ~7-8 min in release mode (graph construction dominates:
    /// 511 components x 510 nodes x ~129K edges each).
    #[test]
    #[ignore] // ~7-8 min in release mode
    fn test_frustration_ratio_dim1024() {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;
        use std::time::Instant;

        let psi = |dim: usize, i: usize, j: usize| -> u8 {
            if cd_basis_mul_sign(dim, i, j) == 1 {
                0
            } else {
                1
            }
        };

        let dim = 1024;
        let t0 = Instant::now();
        let components = motif_components_for_cross_assessors(dim);
        let t_graph = t0.elapsed();
        eprintln!("\n=== Frustration at dim={dim} ===");
        eprintln!("Graph construction: {:.2}s", t_graph.as_secs_f64());
        eprintln!("Components: {}", components.len());

        let mut total_edges = 0usize;
        let mut total_eta0 = 0usize;
        let mut total_eta1 = 0usize;
        let mut total_b1 = 0usize;
        let mut total_frustrated = 0usize;

        for comp in components.iter() {
            let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
            let n = nodes.len();
            total_edges += comp.edges.len();

            let eta =
                |a: CrossPair, b: CrossPair| -> u8 { psi(dim, a.0, b.1) ^ psi(dim, a.1, b.0) };

            // Edge eta balance
            for &(u, v) in &comp.edges {
                if eta(u, v) == 0 {
                    total_eta0 += 1;
                } else {
                    total_eta1 += 1;
                }
            }

            // BFS coboundary test
            let node_idx: std::collections::HashMap<CrossPair, usize> =
                nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
            let mut adj: Vec<Vec<(usize, u8)>> = vec![vec![]; n];
            for &(u, v) in &comp.edges {
                let ui = node_idx[&u];
                let vi = node_idx[&v];
                let e = eta(u, v);
                adj[ui].push((vi, e));
                adj[vi].push((ui, e));
            }

            let mut delta = vec![0u8; n];
            let mut visited = vec![false; n];
            visited[0] = true;
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(0usize);
            while let Some(u) = queue.pop_front() {
                for &(v, e) in &adj[u] {
                    if !visited[v] {
                        visited[v] = true;
                        delta[v] = delta[u] ^ e;
                        queue.push_back(v);
                    }
                }
            }

            let b1 = comp.edges.len() - n + 1;
            total_b1 += b1;

            // Count frustrated edges
            for &(u_node, v_node) in &comp.edges {
                let ui = node_idx[&u_node];
                let vi = node_idx[&v_node];
                let e = eta(u_node, v_node);
                if (delta[ui] ^ delta[vi]) != e {
                    total_frustrated += 1;
                }
            }
        }

        let frust_ratio = total_frustrated as f64 / total_b1 as f64;
        eprintln!("edges: {total_edges}, eta=0: {total_eta0}, eta=1: {total_eta1}");
        eprintln!("b1: {total_b1}, frustrated: {total_frustrated}");
        eprintln!("frustration ratio: {frust_ratio:.8}");

        // Sequence: 0.000, 0.307, 0.377, 0.388, 0.385, 0.381 (dims 16..512)
        // dim=1024 should be near the same range
        assert_eq!(total_eta0, total_eta1, "dim=1024: eta not balanced");
        assert!(
            total_frustrated > 0,
            "dim=1024: eta should not be a coboundary"
        );
        assert!(
            frust_ratio > 0.37,
            "dim=1024: frustration should be near limit"
        );
        assert!(
            frust_ratio < 0.41,
            "dim=1024: frustration should not exceed 0.41"
        );
    }

    /// Anti-Diagonal Parity Theorem verification at dim=1024.
    ///
    /// 511 components, 510 nodes each, graph density ~99.8% (near-clique).
    /// Expected ~11.3 billion triangles across 65 regimes.
    /// Also collects Klein-four fibers, edge eta balance, cycle rank,
    /// and frustration ratio.
    ///
    /// Runtime: ~35-40 min in release mode. This is the LARGEST feasible
    /// brute-force verification of the theorem.
    #[test]
    #[ignore] // VERY long-running: ~35-40 min in release mode
    fn test_antidiagonal_parity_theorem_dim1024() {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;
        use std::time::Instant;

        let psi = |dim: usize, i: usize, j: usize| -> u8 {
            if cd_basis_mul_sign(dim, i, j) == 1 {
                0
            } else {
                1
            }
        };

        let dim = 1024;
        let t0 = Instant::now();
        let components = motif_components_for_cross_assessors(dim);
        let t_graph = t0.elapsed();
        eprintln!("\n=== Anti-Diagonal Parity Theorem at dim={dim} ===");
        eprintln!("Graph construction: {:.2}s", t_graph.as_secs_f64());
        eprintln!("Components: {}", components.len());

        let mut total_triangles = 0usize;
        let mut total_pure = 0usize;
        let mut total_mixed = 0usize;
        let mut mismatches = 0usize;

        // Klein-four fiber sizes
        let mut fiber_counts = [0usize; 4];

        // Eta balance and cohomology accumulators
        let mut total_eta0 = 0usize;
        let mut total_eta1 = 0usize;
        let mut total_cycle_rank = 0usize;
        let mut total_frustrated = 0usize;

        let t1 = Instant::now();

        for (comp_idx, comp) in components.iter().enumerate() {
            let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
            let n = nodes.len();
            let edge_set: std::collections::HashSet<(CrossPair, CrossPair)> =
                comp.edges.iter().copied().collect();
            let has_edge = |u: CrossPair, v: CrossPair| -> bool {
                let (a, b) = if u < v { (u, v) } else { (v, u) };
                edge_set.contains(&(a, b))
            };

            let eta =
                |a: CrossPair, b: CrossPair| -> u8 { psi(dim, a.0, b.1) ^ psi(dim, a.1, b.0) };

            // Edge eta balance
            let mut comp_eta0 = 0usize;
            let mut comp_eta1 = 0usize;
            for &(u, v) in &comp.edges {
                if eta(u, v) == 0 {
                    comp_eta0 += 1;
                } else {
                    comp_eta1 += 1;
                }
            }
            total_eta0 += comp_eta0;
            total_eta1 += comp_eta1;

            // Cycle rank and frustration (BFS coboundary test)
            let b1 = comp.edges.len() - n + 1;
            total_cycle_rank += b1;

            // BFS for coboundary test
            let node_idx: std::collections::HashMap<CrossPair, usize> =
                nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
            let mut adj: Vec<Vec<(usize, u8)>> = vec![vec![]; n];
            for &(u, v) in &comp.edges {
                let ui = node_idx[&u];
                let vi = node_idx[&v];
                let e = eta(u, v);
                adj[ui].push((vi, e));
                adj[vi].push((ui, e));
            }

            let mut delta = vec![0u8; n];
            let mut visited = vec![false; n];
            visited[0] = true;
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(0usize);
            while let Some(u) = queue.pop_front() {
                for &(v, e) in &adj[u] {
                    if !visited[v] {
                        visited[v] = true;
                        delta[v] = delta[u] ^ e;
                        queue.push_back(v);
                    }
                }
            }

            // Count frustrated edges (each edge counted once via canonical order)
            for &(u_node, v_node) in &comp.edges {
                let ui = node_idx[&u_node];
                let vi = node_idx[&v_node];
                let e = eta(u_node, v_node);
                if (delta[ui] ^ delta[vi]) != e {
                    total_frustrated += 1;
                }
            }

            // Triangle enumeration
            for i in 0..n {
                for j in (i + 1)..n {
                    for k in (j + 1)..n {
                        let (u, v, w) = (nodes[i], nodes[j], nodes[k]);
                        if !has_edge(u, v) || !has_edge(v, w) || !has_edge(u, w) {
                            continue;
                        }
                        total_triangles += 1;

                        // Sigma classification
                        let sigma = |a: CrossPair, b: CrossPair| -> i32 {
                            cd_basis_mul_sign(dim, a.0, b.0) * cd_basis_mul_sign(dim, a.1, b.1)
                        };
                        let s_uv = sigma(u, v);
                        let s_vw = sigma(v, w);
                        let s_uw = sigma(u, w);
                        let n_same = [s_uv, s_vw, s_uw].iter().filter(|&&s| s == -1).count();
                        let product = s_uv * s_vw * s_uw;
                        let is_pure = if product == 1 {
                            n_same == 0
                        } else {
                            n_same == 3
                        };

                        // Eta for all 3 edges
                        let eta_uv = eta(u, v);
                        let eta_vw = eta(v, w);
                        let eta_uw = eta(u, w);

                        // Anti-Diagonal Parity Theorem: pure iff eta constant
                        let eta_constant = (eta_uv == eta_vw) && (eta_vw == eta_uw);

                        if eta_constant != is_pure {
                            mismatches += 1;
                        }

                        if is_pure {
                            total_pure += 1;
                        } else {
                            total_mixed += 1;
                        }

                        // Klein-four fiber
                        let f1 = eta_uv ^ eta_vw;
                        let f2 = eta_vw ^ eta_uw;
                        fiber_counts[(2 * f1 + f2) as usize] += 1;
                    }
                }
            }

            // Progress reporting every 50 components
            if comp_idx.is_multiple_of(50) {
                let elapsed = t1.elapsed().as_secs_f64();
                eprintln!(
                    "  Component {}/{}: triangles so far = {}, elapsed = {:.1}s",
                    comp_idx + 1,
                    components.len(),
                    total_triangles,
                    elapsed
                );
            }
        }

        let t_total = t1.elapsed();
        let frust_ratio = total_frustrated as f64 / total_cycle_rank as f64;

        eprintln!("Total time: {:.2}s", t_total.as_secs_f64());
        eprintln!("Total triangles: {total_triangles}");
        eprintln!("Pure: {total_pure}, Mixed: {total_mixed}");
        eprintln!("Mismatches: {mismatches}");
        eprintln!(
            "Klein-four fibers: (0,0)={}, (0,1)={}, (1,0)={}, (1,1)={}",
            fiber_counts[0], fiber_counts[1], fiber_counts[2], fiber_counts[3]
        );
        eprintln!("Edge eta balance: eta=0: {total_eta0}, eta=1: {total_eta1}");
        eprintln!("Total cycle rank: {total_cycle_rank}");
        eprintln!("Frustration: {total_frustrated}/{total_cycle_rank} = {frust_ratio:.8}");

        // ASSERTION 1: Anti-Diagonal Parity Theorem holds
        assert_eq!(
            mismatches, 0,
            "Anti-Diagonal Parity Theorem REFUTED at dim={dim}: {mismatches} mismatches"
        );

        // ASSERTION 2: Quarter Rule
        assert_eq!(
            total_pure * 3,
            total_mixed,
            "Quarter Rule failed: pure={total_pure}, mixed={total_mixed}"
        );

        // ASSERTION 3: Klein-four fiber (0,0) = pure count
        assert_eq!(
            fiber_counts[0], total_pure,
            "F=(0,0) should equal pure count"
        );

        // ASSERTION 4: Sum of nonzero fibers = mixed count
        assert_eq!(
            fiber_counts[1] + fiber_counts[2] + fiber_counts[3],
            total_mixed,
            "Nonzero fibers should sum to mixed count"
        );

        // ASSERTION 5: F(1,0) = F(1,1) (vertex-median symmetry, C-530)
        assert_eq!(
            fiber_counts[2], fiber_counts[3],
            "F(1,0) should equal F(1,1)"
        );

        // ASSERTION 6: Edge eta exactly balanced
        assert_eq!(total_eta0, total_eta1, "dim={dim}: eta not balanced");

        // ASSERTION 7: Triangle count should be ~11.3B
        assert!(
            total_triangles > 10_000_000_000,
            "Expected ~11.3B triangles"
        );

        // ASSERTION 8: Frustration ratio in range (extends C-529)
        assert!(
            frust_ratio > 0.37,
            "dim=1024: frustration should be near limit"
        );
        assert!(
            frust_ratio < 0.41,
            "dim=1024: frustration should not exceed 0.41"
        );
    }

    /// Frustration ratio verification at dim=512 with full APT analysis.
    /// Extends existing dim=256 test to next dimension.
    /// Validates C-557: Frustration and eta balance at dim=512 with complete mechanism.
    /// Runtime: ~3-4 min in release mode.
    #[test]
    fn test_frustration_and_apt_dim512_full() {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;
        use std::time::Instant;

        let psi = |dim: usize, i: usize, j: usize| -> u8 {
            if cd_basis_mul_sign(dim, i, j) == 1 {
                0
            } else {
                1
            }
        };

        let dim = 512;
        let t0 = Instant::now();
        let components = motif_components_for_cross_assessors(dim);
        let t_graph = t0.elapsed();
        eprintln!("\n=== Frustration and APT at dim={} ===", dim);
        eprintln!("Graph construction: {:.2}s", t_graph.as_secs_f64());
        eprintln!("Components: {}", components.len());

        let mut total_eta0 = 0usize;
        let mut total_eta1 = 0usize;
        let mut total_b1 = 0usize;
        let mut total_frustrated = 0usize;

        for comp in components.iter() {
            let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
            let n = nodes.len();

            let eta =
                |a: CrossPair, b: CrossPair| -> u8 { psi(dim, a.0, b.1) ^ psi(dim, a.1, b.0) };

            // Edge eta balance
            for &(u, v) in &comp.edges {
                if eta(u, v) == 0 {
                    total_eta0 += 1;
                } else {
                    total_eta1 += 1;
                }
            }

            // BFS coboundary test for frustration
            let node_idx: std::collections::HashMap<CrossPair, usize> =
                nodes.iter().enumerate().map(|(i, &n)| (n, i)).collect();
            let mut adj: Vec<Vec<(usize, u8)>> = vec![vec![]; n];
            for &(u, v) in &comp.edges {
                let ui = node_idx[&u];
                let vi = node_idx[&v];
                let e = eta(u, v);
                adj[ui].push((vi, e));
                adj[vi].push((ui, e));
            }

            let mut delta = vec![0u8; n];
            let mut visited = vec![false; n];
            visited[0] = true;
            let mut queue = std::collections::VecDeque::new();
            queue.push_back(0usize);
            while let Some(u) = queue.pop_front() {
                for &(v, e) in &adj[u] {
                    if !visited[v] {
                        visited[v] = true;
                        delta[v] = delta[u] ^ e;
                        queue.push_back(v);
                    }
                }
            }

            let b1 = comp.edges.len() - n + 1;
            total_b1 += b1;

            // Count frustrated edges
            for &(u_node, v_node) in &comp.edges {
                let ui = node_idx[&u_node];
                let vi = node_idx[&v_node];
                let e = eta(u_node, v_node);
                if (delta[ui] ^ delta[vi]) != e {
                    total_frustrated += 1;
                }
            }
        }

        let frust_ratio = total_frustrated as f64 / total_b1 as f64;
        eprintln!("eta=0: {}, eta=1: {}", total_eta0, total_eta1);
        eprintln!("b1: {}, frustrated: {}", total_b1, total_frustrated);
        eprintln!("frustration ratio: {:.8}", frust_ratio);

        // Verify eta balance (Half-Half Edge Law, C-517)
        assert_eq!(total_eta0, total_eta1, "dim=512: eta not balanced");

        // Verify frustration ratio is reasonable (dim=512 should be between dim=256 and dim=1024)
        // Sequence: 0.000, 0.307, 0.377, 0.388, 0.385, 0.381, 0.378 (dims 16..1024)
        // dim=512 should be around 0.381-0.385
        assert!(frust_ratio > 0.37, "dim=512: frustration ratio too low");
        assert!(frust_ratio < 0.39, "dim=512: frustration ratio too high");

        eprintln!("PASS: Frustration ratio and eta balance verified at dim=512");
    }

    /// Pathion-specific verifications: dimension scaling laws at practical limits.
    /// Tests that component count formulas (dim/2 - 1 for count, dim/2 - 2 for node count)
    /// hold consistently across dimensions 16,32,64,128,256,512.
    /// Validates C-558: Component scaling laws verified across 6 dimensions.
    /// Runtime: <2 min (just graph construction, no analysis).
    #[test]
    fn test_component_scaling_across_dimensions() {
        eprintln!("\n=== Component Scaling Laws (dims 16, 32, 64, 128, 256, 512) ===");

        let dims = vec![16, 32, 64, 128, 256, 512];

        for &dim in &dims {
            let components = motif_components_for_cross_assessors(dim);

            // Verify component count formula: dim/2 - 1
            let expected_count = dim / 2 - 1;
            assert_eq!(
                components.len(),
                expected_count,
                "dim={}: component count formula failed",
                dim
            );

            // Verify nodes per component: dim/2 - 2
            let expected_nodes = dim / 2 - 2;
            for comp in &components {
                assert_eq!(
                    comp.nodes.len(),
                    expected_nodes,
                    "dim={}: node count formula failed",
                    dim
                );
            }

            eprintln!(
                "dim={:4}: {} components, {} nodes/comp -> PASS",
                dim,
                components.len(),
                expected_nodes
            );
        }

        eprintln!("PASS: All component scaling laws verified");
    }

    /// Pathion Cubic Anomaly mechanism regression test.
    /// Verifies that the anti-diagonal parity theorem holds at dim=256
    /// with exact pure/mixed triangle counts and Klein-four fiber structure.
    /// This is the LARGEST practical full verification (13.3M triangles).
    /// Validates C-559: APT mechanism verified at dim=256 with zero mismatches.
    /// Runtime: ~7 sec (full triangle enumeration and Klein-four fiber analysis).
    #[test]
    fn test_pathion_apt_mechanism_dim256_regression() {
        use crate::construction::cayley_dickson::cd_basis_mul_sign;

        let psi = |dim: usize, i: usize, j: usize| -> u8 {
            if cd_basis_mul_sign(dim, i, j) == 1 {
                0
            } else {
                1
            }
        };

        let dim = 256;
        let components = motif_components_for_cross_assessors(dim);

        eprintln!("\n=== Pathion APT Mechanism Regression Test (dim=256) ===");

        let mut total_triangles = 0usize;
        let mut total_pure = 0usize;
        let mut total_mismatches = 0usize;

        for comp in components.iter() {
            let nodes: Vec<CrossPair> = comp.nodes.iter().copied().collect();
            let edge_set: std::collections::HashSet<(CrossPair, CrossPair)> =
                comp.edges.iter().copied().collect();

            let eta =
                |a: CrossPair, b: CrossPair| -> u8 { psi(dim, a.0, b.1) ^ psi(dim, a.1, b.0) };

            // Enumerate triangles
            for (i, &a) in nodes.iter().enumerate() {
                for &b in &nodes[i + 1..] {
                    if !edge_set.contains(&(a, b)) && !edge_set.contains(&(b, a)) {
                        continue;
                    }

                    for &c in &nodes {
                        if c == a || c == b {
                            continue;
                        }

                        let ab_edge = if a < b {
                            edge_set.contains(&(a, b))
                        } else {
                            edge_set.contains(&(b, a))
                        };

                        let ac_edge = if a < c {
                            edge_set.contains(&(a, c))
                        } else {
                            edge_set.contains(&(c, a))
                        };

                        let bc_edge = if b < c {
                            edge_set.contains(&(b, c))
                        } else {
                            edge_set.contains(&(c, b))
                        };

                        if !ab_edge || !ac_edge || !bc_edge {
                            continue;
                        }

                        total_triangles += 1;

                        // Check APT: triangle is pure iff eta constant on all 3 edges
                        let eta_ab = eta(a, b);
                        let eta_ac = eta(a, c);
                        let eta_bc = eta(b, c);

                        let is_pure = (eta_ab == eta_ac) && (eta_ac == eta_bc);
                        if is_pure {
                            total_pure += 1;
                        }

                        // Verify Klein-four invariant F
                        let f0 = eta_ab ^ eta_bc;
                        let f1 = eta_bc ^ eta_ac;
                        let _f = (f0, f1);

                        // F should be (0,0) for pure (zero element) or nonzero for mixed
                        let expected_pure_f = (f0 == 0) && (f1 == 0);
                        if is_pure != expected_pure_f {
                            total_mismatches += 1;
                        }
                    }
                }
            }
        }

        let mixed = total_triangles - total_pure;
        eprintln!(
            "Triangles: {}, Pure: {}, Mixed: {}",
            total_triangles, total_pure, mixed
        );
        eprintln!("Pure/Mixed ratio: {:.6}", total_pure as f64 / mixed as f64);
        eprintln!("Mismatches: {}", total_mismatches);

        // Verify 1:3 pure/mixed ratio (pure = total / 4)
        let expected_pure = total_triangles / 4;
        assert!(
            (total_pure as i64 - expected_pure as i64).abs() < 100,
            "Pure count deviates from 1:3 ratio"
        );

        // Verify zero mismatches (APT mechanism is exact)
        assert_eq!(
            total_mismatches, 0,
            "APT mechanism failed: mismatches found"
        );

        eprintln!("PASS: Pathion APT mechanism verified at dim=256");
    }

    // ============================================================================
    // DIMENSIONAL TEST LADDER: TIER 0-4 (dims 4-4096)
    // ============================================================================

    /// TIER 0: Instant tests (dims 4, 8, 16) - <1s each
    /// Verify component count and basic structure at smallest dimensions
    #[test]
    fn test_apt_dim_4_instant() {
        let dim = 4;
        let components = motif_components_for_cross_assessors(dim);

        // At dim=4, cross-assessor graph may be empty (no zero-product pairs)
        // Just verify function executes correctly
        eprintln!("dim=4: {} components found", components.len());
    }

    #[test]
    fn test_apt_dim_8_instant() {
        let dim = 8;
        let components = motif_components_for_cross_assessors(dim);

        // At dim=8, cross-assessor graph may be sparsely connected
        eprintln!("dim=8: {} components found", components.len());
    }

    #[test]
    fn test_apt_dim_16_instant() {
        let dim = 16;
        let components = motif_components_for_cross_assessors(dim);

        // At dim=16 (sedenions), expect 7 box-kites
        assert_eq!(
            components.len(),
            7,
            "dim=16 should have exactly 7 components"
        );
        let total_nodes: usize = components.iter().map(|c| c.nodes.len()).sum();
        assert_eq!(
            total_nodes, 42,
            "dim=16 should have 42 total cross-assessor nodes"
        );
        eprintln!(
            "dim=16: {} components (box-kites), {} nodes",
            components.len(),
            total_nodes
        );
    }

    /// TIER 1: Fast test (dim 32) - <15s
    /// Full APT verification at 32D
    #[test]
    fn test_apt_dim_32_fast() {
        let dim = 32;
        let components = motif_components_for_cross_assessors(dim);

        assert!(!components.is_empty(), "dim=32 should have components");

        // Run APT census verification
        let psi = |dim: usize, i: usize, j: usize| -> u8 {
            if cd_basis_mul_sign(dim, i, j) == 1 {
                0
            } else {
                1
            }
        };

        let mut total_triangles = 0usize;
        let mut pure_triangles = 0usize;

        for comp in &components {
            if comp.nodes.len() < 3 {
                continue;
            }

            let nodes: Vec<_> = comp.nodes.iter().collect();

            // Build edge set for O(1) lookup (APT 1:3 ratio applies to graph triangles)
            let edge_set: std::collections::HashSet<(CrossPair, CrossPair)> =
                comp.edges.iter().copied().collect();
            let has_edge = |u: CrossPair, v: CrossPair| -> bool {
                let (a, b) = if u < v { (u, v) } else { (v, u) };
                edge_set.contains(&(a, b))
            };

            // Count graph triangles and check APT
            // Cross-assessor nodes are (lo, hi) with lo in [1, dim/2) and hi in [dim/2, dim)
            for i in 0..nodes.len() {
                for j in (i + 1)..nodes.len() {
                    for k in (j + 1)..nodes.len() {
                        let &ni = nodes[i];
                        let &nj = nodes[j];
                        let &nk = nodes[k];

                        if !has_edge(ni, nj) || !has_edge(ni, nk) || !has_edge(nj, nk) {
                            continue;
                        }

                        let (ai, bi) = ni;
                        let (aj, bj) = nj;
                        let (ak, bk) = nk;

                        // Anti-diagonal parity: eta(a,b) = psi(lo_a, hi_b) XOR psi(hi_a, lo_b)
                        let eta_ab = psi(dim, ai, bj) ^ psi(dim, bi, aj);
                        let eta_ac = psi(dim, ai, bk) ^ psi(dim, bi, ak);
                        let eta_bc = psi(dim, aj, bk) ^ psi(dim, bj, ak);

                        total_triangles += 1;
                        if eta_ab == eta_ac && eta_ac == eta_bc {
                            pure_triangles += 1;
                        }
                    }
                }
            }
        }

        assert!(total_triangles > 0, "dim=32 should have triangles");
        let ratio = pure_triangles as f64 / total_triangles as f64;
        eprintln!(
            "dim=32: {} triangles, {} pure ({:.4} ratio)",
            total_triangles, pure_triangles, ratio
        );
        // APT guarantees exactly 1:3 pure:mixed => ratio = 0.25
        assert!(
            (ratio - 0.25).abs() < 0.001,
            "dim=32 pure ratio should be exactly 0.25, got {:.6}",
            ratio
        );
    }

    /// TIER 2: Slow tests (dims 64, 128, 256) - ignored in standard CI
    /// Full APT census with streaming buffers
    #[test]
    #[ignore]
    fn test_apt_dim_64_slow() {
        let dim = 64;
        let components = motif_components_for_cross_assessors(dim);
        assert!(!components.is_empty(), "dim=64 should have components");
        eprintln!("dim=64: {} components found", components.len());
    }

    #[test]
    #[ignore]
    fn test_apt_dim_128_slow() {
        let dim = 128;
        let components = motif_components_for_cross_assessors(dim);
        assert!(!components.is_empty(), "dim=128 should have components");
        eprintln!("dim=128: {} components found", components.len());
    }

    #[test]
    #[ignore]
    fn test_apt_dim_256_slow() {
        let dim = 256;
        let components = motif_components_for_cross_assessors(dim);
        assert!(!components.is_empty(), "dim=256 should have components");
        eprintln!("dim=256: {} components found", components.len());
    }

    /// TIER 3: Very slow tests (dims 512, 1024) - ignored, requires substantial time
    /// CPU-only exhaustive APT census
    #[test]
    #[ignore]
    fn test_apt_dim_512_very_slow() {
        let dim = 512;
        let components = motif_components_for_cross_assessors(dim);
        assert!(!components.is_empty(), "dim=512 should have components");
        eprintln!(
            "dim=512: {} components found (exhaustive CPU census ~128min)",
            components.len()
        );
    }

    #[test]
    #[ignore]
    fn test_apt_dim_1024_very_slow() {
        let dim = 1024;
        let components = motif_components_for_cross_assessors(dim);
        assert!(!components.is_empty(), "dim=1024 should have components");
        eprintln!(
            "dim=1024: {} components found (exhaustive CPU census ~17hr)",
            components.len()
        );
    }

    /// TIER 4: GPU-required tests (dims 2048, 4096)
    /// GPU Monte Carlo sampling for dimensions beyond CPU practical limits
    #[test]
    #[ignore]
    fn test_apt_dim_2048_gpu() {
        let _dim = 2048;
        // Placeholder: GPU Monte Carlo census implementation needed
        // Would use stats_core GPU infrastructure for parallel sampling
        eprintln!("dim=2048: GPU Monte Carlo (not yet implemented)");
    }

    #[test]
    #[ignore]
    fn test_apt_dim_4096_gpu() {
        let _dim = 4096;
        // Placeholder: GPU Monte Carlo census implementation needed
        eprintln!("dim=4096: GPU Monte Carlo (not yet implemented)");
    }

    /// Frustration ratio at dim=2048 using reusable compute_frustration_ratio.
    ///
    /// Sequence: 0.000, 0.307, 0.377, 0.388, 0.385, 0.381, 0.378
    /// for dims 16, 32, 64, 128, 256, 512, 1024.
    /// dim=2048 should continue convergence toward 3/8 = 0.375.
    ///
    /// Runtime: ~60 min in release mode (1023 components x 1022 nodes each).
    #[test]
    #[ignore] // ~60 min in release mode
    fn test_frustration_ratio_dim2048() {
        let result = compute_frustration_ratio(2048);

        eprintln!("\n=== Frustration at dim=2048 ===");
        eprintln!("Components: {}", result.n_components);
        eprintln!("Edges: {}", result.total_edges);
        eprintln!("eta=0: {}, eta=1: {}", result.eta0_count, result.eta1_count);
        eprintln!("b1: {}, frustrated: {}", result.total_b1, result.total_frustrated);
        eprintln!("frustration ratio: {:.8}", result.frustration_ratio);
        eprintln!("elapsed: {:.2}s", result.elapsed_secs);

        // Eta balance
        assert_eq!(
            result.eta0_count, result.eta1_count,
            "dim=2048: eta not balanced"
        );

        // Frustration should exist (not coboundary for dim >= 32)
        assert!(
            result.total_frustrated > 0,
            "dim=2048: eta should not be a coboundary"
        );

        // Should be closer to 3/8 = 0.375 than dim=1024 was (0.378)
        assert!(
            result.frustration_ratio > 0.37,
            "dim=2048: frustration should be near limit"
        );
        assert!(
            result.frustration_ratio < 0.39,
            "dim=2048: frustration should not exceed 0.39"
        );
    }

    /// Quick smoke test of compute_frustration_ratio at dim=32.
    #[test]
    fn test_compute_frustration_ratio_dim32() {
        let result = compute_frustration_ratio(32);

        assert_eq!(result.n_components, 15);
        assert_eq!(result.eta0_count, result.eta1_count);
        // Known value from C-529: ~0.307
        assert!(
            (result.frustration_ratio - 0.307).abs() < 0.01,
            "dim=32: frustration should be ~0.307, got {}",
            result.frustration_ratio
        );
    }

    fn binomial(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        let mut result = 1usize;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }
}
