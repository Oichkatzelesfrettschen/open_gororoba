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

use crate::zd_graphs::xor_key;
use cd_kernel::cayley_dickson::{cd_multiply, cd_norm_sq};
use nalgebra::{DMatrix, SymmetricEigen};
use petgraph::graph::{NodeIndex, UnGraph};
use rayon::prelude::*;
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

// Core types (Assessor, BoxKite, BoxKiteSymmetryResult) live in the
// `types` submodule. Re-exports preserve the public API at
// algebra_analysis::boxkites::*.
pub mod types;
pub use types::{Assessor, BoxKite, BoxKiteSymmetryResult};

// Co-assessor graph construction and connected-component analysis
// (primitive_assessors, diagonal_zero_product, are_coassessors,
// build_coassessor_graph, find_connected_components,
// compute_strut_signature) live in the `coassessor_graph` submodule.
pub mod coassessor_graph;
pub use coassessor_graph::{
    are_coassessors, build_coassessor_graph, compute_strut_signature, diagonal_zero_product,
    find_connected_components, primitive_assessors,
};

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

/// Cached sedenion box-kites (dim=16, atol=1e-10).
/// Safe for concurrent test access: OnceLock guarantees single initialization.
/// In test builds, also initializes the Rayon pool with physical core pinning
/// before computing box-kites (which triggers parallel ZD search via cd_kernel).
pub fn cached_sedenion_boxkites() -> &'static Vec<BoxKite> {
    use std::sync::OnceLock;
    static SEDENION_BOXKITES: OnceLock<Vec<BoxKite>> = OnceLock::new();
    SEDENION_BOXKITES.get_or_init(|| {
        #[cfg(test)]
        crate::test_support::init_physical_rayon_pool();
        find_box_kites(16, 1e-10)
    })
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

// de Marrais production rules (O_TRIPS const, production_rule_1,
// production_rule_2, production_rule_3) and automorpheme machinery
// (automorpheme_assessors, automorphemes,
// automorphemes_containing_assessor) live in the `production_rules`
// submodule.
pub mod production_rules;
pub use production_rules::{
    O_TRIPS, automorpheme_assessors, automorphemes, automorphemes_containing_assessor,
    production_rule_1, production_rule_2, production_rule_3,
};

// StrutTable struct + canonical_strut_table function live in the
// `strut_table` submodule.
pub mod strut_table;
pub use strut_table::{StrutTable, canonical_strut_table};

// Generalized cross-assessor enumeration and integer-exact diagonal
// zero-product detection (CrossPair type alias, cross_assessors,
// diagonal_zero_products_exact) live in the `cross_assessors` submodule.
pub mod cross_assessors;
pub use cross_assessors::{CrossPair, cross_assessors, diagonal_zero_products_exact};

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
        vals.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
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

    // Parallel bucket processing: each bucket's pairwise comparisons are
    // independent. Collect edges from all buckets in parallel, then merge.
    let bucket_list: Vec<Vec<CrossPair>> = buckets.into_values().collect();
    let all_edges: Vec<(CrossPair, CrossPair)> = bucket_list
        .par_iter()
        .flat_map(|bucket_nodes| {
            let mut sorted_bucket = bucket_nodes.clone();
            sorted_bucket.sort();
            let mut local_edges = Vec::new();
            for i in 0..sorted_bucket.len() {
                for j in (i + 1)..sorted_bucket.len() {
                    let a = sorted_bucket[i];
                    let b = sorted_bucket[j];
                    let sols = diagonal_zero_products_exact(dim, a, b);
                    if !sols.is_empty() {
                        local_edges.push((a, b));
                    }
                }
            }
            local_edges
        })
        .collect();

    // Build adjacency from collected edges
    let mut adj: HashMap<CrossPair, HashSet<CrossPair>> =
        nodes.iter().map(|&n| (n, HashSet::new())).collect();
    let mut edges: HashSet<(CrossPair, CrossPair)> = HashSet::with_capacity(all_edges.len());
    for (a, b) in all_edges {
        adj.get_mut(&a).unwrap().insert(b);
        adj.get_mut(&b).unwrap().insert(a);
        edges.insert((a, b));
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

// Triangular-face sign-pattern classification + per-component census
// (FaceSignPattern, classify_face_pattern, edge_sign_type_exact,
// ComponentFaceCensus, GenericFaceSignCensus, generic_face_sign_census)
// live in the `face_sign_census` submodule.
pub mod face_sign_census;
pub use face_sign_census::{
    ComponentFaceCensus, FaceSignPattern, GenericFaceSignCensus, classify_face_pattern,
    edge_sign_type_exact, generic_face_sign_census,
};

// Imbalance ratio (frustrated edges / cycle rank b1) over the zero-product
// graph at any CD dimension. ImbalanceResult + compute_imbalance_ratio
// live in the `imbalance` submodule.
pub mod imbalance;
pub use imbalance::{ImbalanceResult, compute_imbalance_ratio};


// Test block relocated to sibling boxkites/tests.rs.
// The 7030-line cfg(test) section made the parent file 7665 lines
// with only 633 lines of production code -- moving tests out keeps
// the parent's surface area focused on the actual API.
#[cfg(test)]
mod tests;
