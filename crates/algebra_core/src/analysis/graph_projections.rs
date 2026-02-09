//! Graph projections of Cayley-Dickson algebras.
//!
//! Implements named graph predicates (Layer 3 of the monograph abstraction)
//! and the invariant suite for fingerprinting.
//!
//! # Named Predicates
//! - **P_ZD**: Zero-divisor adjacency (parity cliques, K_{n/2} U K_{n/2})
//! - **P_match**: XOR matching adjacency (perfect matching, partner = i XOR (N/16))
//!
//! Each predicate implements the [`GraphPredicate`] trait, providing a uniform
//! interface for graph construction, invariant computation, and testing.

use std::collections::{HashMap, HashSet, VecDeque};
use petgraph::graph::{UnGraph, NodeIndex};
use petgraph::algo::{connected_components, dijkstra};
use nalgebra::DMatrix;

/// Comprehensive graph invariants for motif fingerprinting (A3).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GraphInvariants {
    pub n_nodes: usize,
    pub n_edges: usize,
    pub n_components: usize,
    /// Sorted degree multiset.
    pub degrees: Vec<usize>,
    /// Adjacency spectrum (eigenvalues).
    pub spectrum: Vec<f64>,
    pub triangle_count: usize,
    /// Diameter of the largest component.
    pub diameter: Option<usize>,
    pub girth: Option<usize>,
}

/// Compute the full suite of invariants for an undirected graph.
pub fn compute_graph_invariants(graph: &UnGraph<(), ()>) -> GraphInvariants {
    let n = graph.node_count();
    let e = graph.edge_count();
    let n_comp = connected_components(graph);

    let mut degrees: Vec<usize> = graph.node_indices()
        .map(|i| graph.neighbors(i).count())
        .collect();
    degrees.sort_unstable();

    // Adjacency Matrix for spectrum and triangles
    let mut adj = DMatrix::<f64>::zeros(n, n);
    for edge in graph.edge_indices() {
        let (u, v) = graph.edge_endpoints(edge).unwrap();
        adj[(u.index(), v.index())] = 1.0;
        adj[(v.index(), u.index())] = 1.0;
    }

    // Triangle count: Tr(A^3) / 6 (compute before eigendecomposition consumes adj)
    let adj2 = &adj * &adj;
    let adj3 = &adj2 * &adj;
    let trace: f64 = (0..n).map(|i| adj3[(i, i)]).sum();
    let triangle_count = (trace / 6.0).round() as usize;

    // Spectrum via nalgebra (symmetric_eigen consumes adj, so compute after adj products)
    let eigen = adj.symmetric_eigen();
    let mut spectrum: Vec<f64> = eigen.eigenvalues.iter().cloned().collect();
    spectrum.sort_by(|a: &f64, b: &f64| a.partial_cmp(b).unwrap());

    // Diameter (BFS-based for unweighted)
    let mut max_diam = 0;
    for i in graph.node_indices() {
        let paths = dijkstra(graph, i, None, |_| 1);
        for &d in paths.values() {
            if d > max_diam { max_diam = d; }
        }
    }

    // Girth: BFS from each node to find shortest cycle
    let mut min_cycle = None;
    for start in graph.node_indices() {
        let mut dist = vec![None; n];
        let mut parent = vec![None; n];
        let mut q = std::collections::VecDeque::new();
        
        dist[start.index()] = Some(0);
        q.push_back(start);
        
        while let Some(u) = q.pop_front() {
            let d_u = dist[u.index()].unwrap();
            for v in graph.neighbors(u) {
                if dist[v.index()].is_none() {
                    dist[v.index()] = Some(d_u + 1);
                    parent[v.index()] = Some(u);
                    q.push_back(v);
                } else if Some(v) != parent[u.index()] {
                    // Cycle found
                    let cycle_len = d_u + dist[v.index()].unwrap() + 1;
                    min_cycle = match min_cycle {
                        None => Some(cycle_len),
                        Some(m) => Some(m.min(cycle_len)),
                    };
                }
            }
        }
    }

    GraphInvariants {
        n_nodes: n,
        n_edges: e,
        n_components: n_comp,
        degrees,
        spectrum,
        triangle_count,
        diameter: if max_diam > 0 { Some(max_diam) } else { None },
        girth: min_cycle,
    }
}

/// Generate the "Pathion Adjacency" graph (Perfect Matching).
///
/// Rule: A(i,j) = 1 iff j = i XOR (N/16).
/// Valid for dim N >= 64.
pub fn generate_pathion_matching(dim: usize) -> UnGraph<(), ()> {
    let mut graph = UnGraph::<(), ()>::with_capacity(dim, dim / 2);
    let nodes: Vec<NodeIndex> = (0..dim).map(|_| graph.add_node(())).collect();

    let xor_partner = dim / 16;

    for i in 0..dim {
        let j = i ^ xor_partner;
        if i < j {
            graph.add_edge(nodes[i], nodes[j], ());
        }
    }
    graph
}

/// Generate the "Zero-Divisor Adjacency" graph (Parity Cliques).
///
/// Rule: A(i,j) = 1 iff i != j AND i%2 == j%2.
/// Creates two disjoint cliques (Evens and Odds).
pub fn generate_zd_parity_cliques(dim: usize) -> UnGraph<(), ()> {
    let mut graph = UnGraph::<(), ()>::with_capacity(dim, dim * (dim/2 - 1));
    let nodes: Vec<NodeIndex> = (0..dim).map(|_| graph.add_node(())).collect();

    // Even clique
    for i in (0..dim).step_by(2) {
        for j in (i + 2..dim).step_by(2) {
            graph.add_edge(nodes[i], nodes[j], ());
        }
    }

    // Odd clique
    for i in (1..dim).step_by(2) {
        for j in (i + 2..dim).step_by(2) {
            graph.add_edge(nodes[i], nodes[j], ());
        }
    }

    graph
}

// ============================================================================
// Layer 3: Named Graph Predicates
// ============================================================================

/// A named predicate on pairs of node indices that defines a graph.
///
/// Predicates are the building blocks for constructing graphs from
/// CD algebra structure. Each predicate has a name, a dimension
/// requirement, and a test function.
pub trait GraphPredicate {
    /// Human-readable name (e.g., "P_ZD", "P_match").
    fn name(&self) -> &str;

    /// Minimum dimension for which this predicate is defined.
    fn min_dim(&self) -> usize;

    /// Test whether nodes i and j are adjacent under this predicate.
    /// Assumes i < j.
    fn test(&self, i: usize, j: usize) -> bool;

    /// Build the full graph for a given number of nodes.
    fn build_graph(&self, n_nodes: usize) -> UnGraph<(), ()> {
        let mut graph = UnGraph::<(), ()>::with_capacity(n_nodes, 0);
        let nodes: Vec<NodeIndex> = (0..n_nodes).map(|_| graph.add_node(())).collect();
        for i in 0..n_nodes {
            for j in (i + 1)..n_nodes {
                if self.test(i, j) {
                    graph.add_edge(nodes[i], nodes[j], ());
                }
            }
        }
        graph
    }

    /// Build the graph and compute its full invariant suite.
    fn invariants(&self, n_nodes: usize) -> GraphInvariants {
        let graph = self.build_graph(n_nodes);
        compute_graph_invariants(&graph)
    }
}

/// P_ZD: Zero-divisor adjacency predicate (parity cliques).
///
/// Two nodes are adjacent iff they have the same parity (both even or both odd).
/// This creates K_{n/2} U K_{n/2} -- two disjoint cliques.
pub struct ZeroDivisorPredicate;

impl GraphPredicate for ZeroDivisorPredicate {
    fn name(&self) -> &str { "P_ZD" }
    fn min_dim(&self) -> usize { 16 }

    fn test(&self, i: usize, j: usize) -> bool {
        i % 2 == j % 2
    }

    fn build_graph(&self, n_nodes: usize) -> UnGraph<(), ()> {
        generate_zd_parity_cliques(n_nodes)
    }
}

/// P_match: XOR matching adjacency predicate.
///
/// Two nodes are adjacent iff j = i XOR (dim/16).
/// This creates a perfect matching of dim/2 edges.
pub struct XorMatchPredicate {
    /// The XOR partner mask: dim/16.
    pub xor_mask: usize,
}

impl XorMatchPredicate {
    /// Create for a given dimension (must be >= 64).
    pub fn new(dim: usize) -> Self {
        Self { xor_mask: dim / 16 }
    }
}

impl GraphPredicate for XorMatchPredicate {
    fn name(&self) -> &str { "P_match" }
    fn min_dim(&self) -> usize { 64 }

    fn test(&self, i: usize, j: usize) -> bool {
        (i ^ j) == self.xor_mask
    }

    fn build_graph(&self, n_nodes: usize) -> UnGraph<(), ()> {
        let mut graph = UnGraph::<(), ()>::with_capacity(n_nodes, n_nodes / 2);
        let nodes: Vec<NodeIndex> = (0..n_nodes).map(|_| graph.add_node(())).collect();
        for i in 0..n_nodes {
            let j = i ^ self.xor_mask;
            if i < j && j < n_nodes {
                graph.add_edge(nodes[i], nodes[j], ());
            }
        }
        graph
    }
}

/// A predicate built from an explicit adjacency matrix (for cross-validation
/// against external CSV data or computed ZD graphs).
pub struct MatrixPredicate {
    name: String,
    /// Row-major adjacency matrix. adj[i * n + j] = true if edge (i,j).
    adj: Vec<bool>,
    n: usize,
}

impl MatrixPredicate {
    /// Build from a boolean adjacency matrix.
    pub fn from_bool_matrix(name: &str, adj: &[Vec<bool>]) -> Self {
        let n = adj.len();
        let mut flat = vec![false; n * n];
        for (i, row) in adj.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                flat[i * n + j] = val;
            }
        }
        Self { name: name.to_string(), adj: flat, n }
    }

    /// Build from a u8 adjacency matrix (nonzero = edge).
    pub fn from_u8_matrix(name: &str, adj: &[Vec<u8>]) -> Self {
        let n = adj.len();
        let mut flat = vec![false; n * n];
        for (i, row) in adj.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                flat[i * n + j] = val != 0;
            }
        }
        Self { name: name.to_string(), adj: flat, n }
    }
}

impl GraphPredicate for MatrixPredicate {
    fn name(&self) -> &str { &self.name }
    fn min_dim(&self) -> usize { self.n }

    fn test(&self, i: usize, j: usize) -> bool {
        if i < self.n && j < self.n {
            self.adj[i * self.n + j]
        } else {
            false
        }
    }
}

// ============================================================================
// Layer 4: Unified Invariant Suite
// ============================================================================

/// A hashable invariant fingerprint for motif classification.
///
/// Two components with identical fingerprints are candidate isomorphs.
/// The fingerprint captures the combinatorial skeleton (degree sequence,
/// edge count, triangles, girth, diameter) but NOT the spectrum, since
/// floating-point eigenvalues are not reliably hashable.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MotifFingerprint {
    pub n_nodes: usize,
    pub n_edges: usize,
    pub degrees: Vec<usize>,
    pub triangle_count: usize,
    pub girth: Option<usize>,
    pub diameter: Option<usize>,
}

impl MotifFingerprint {
    /// Extract a hashable fingerprint from full invariants.
    pub fn from_invariants(inv: &GraphInvariants) -> Self {
        Self {
            n_nodes: inv.n_nodes,
            n_edges: inv.n_edges,
            degrees: inv.degrees.clone(),
            triangle_count: inv.triangle_count,
            girth: inv.girth,
            diameter: inv.diameter,
        }
    }
}

/// Per-component invariants for motif fingerprinting.
#[derive(Debug, Clone)]
pub struct ComponentInvariant {
    /// Index within the sorted list of components.
    pub component_id: usize,
    /// Node indices in the original graph (sorted).
    pub original_nodes: Vec<usize>,
    /// Full graph invariants for this component.
    pub invariants: GraphInvariants,
    /// Hashable fingerprint for classification.
    pub fingerprint: MotifFingerprint,
}

/// Complete invariant suite for a predicate-defined graph.
///
/// Combines whole-graph invariants with per-component analysis
/// and motif classification.  This is the primary output of
/// Layer 4 in the monograph abstraction stack.
#[derive(Debug, Clone)]
pub struct InvariantSuite {
    /// Predicate name (e.g., "P_ZD").
    pub predicate_name: String,
    /// Number of nodes in the full graph.
    pub n_nodes: usize,
    /// Whole-graph invariants.
    pub global: GraphInvariants,
    /// Per-component invariants, sorted by component size (descending).
    pub components: Vec<ComponentInvariant>,
    /// Motif classes: each entry is (fingerprint, list of component_ids).
    /// Sorted by class size (descending).
    pub motif_classes: Vec<(MotifFingerprint, Vec<usize>)>,
}

impl InvariantSuite {
    /// Number of distinct motif isomorphism classes.
    pub fn n_motif_classes(&self) -> usize {
        self.motif_classes.len()
    }

    /// Size of the largest connected component.
    pub fn max_component_size(&self) -> usize {
        self.components.first().map_or(0, |c| c.invariants.n_nodes)
    }

    /// Size of the smallest connected component.
    pub fn min_component_size(&self) -> usize {
        self.components.last().map_or(0, |c| c.invariants.n_nodes)
    }
}

/// Extract connected components as ordered sets of node indices.
///
/// Returns components sorted by size (largest first), with node indices
/// sorted within each component.
fn extract_components(graph: &UnGraph<(), ()>) -> Vec<Vec<NodeIndex>> {
    let n = graph.node_count();
    let mut visited = vec![false; n];
    let mut components = Vec::new();

    for start in graph.node_indices() {
        if visited[start.index()] {
            continue;
        }
        let mut component = Vec::new();
        let mut queue = VecDeque::new();
        queue.push_back(start);
        visited[start.index()] = true;

        while let Some(node) = queue.pop_front() {
            component.push(node);
            for neighbor in graph.neighbors(node) {
                if !visited[neighbor.index()] {
                    visited[neighbor.index()] = true;
                    queue.push_back(neighbor);
                }
            }
        }
        component.sort_by_key(|n| n.index());
        components.push(component);
    }

    // Sort by size descending, then by smallest node index for stability
    components.sort_by(|a, b| {
        b.len().cmp(&a.len())
            .then_with(|| a[0].index().cmp(&b[0].index()))
    });
    components
}

/// Build an induced subgraph from a set of node indices.
///
/// Creates a new graph containing only the specified nodes and their
/// mutual edges.  Node indices in the subgraph are remapped to 0..n.
fn induced_subgraph(graph: &UnGraph<(), ()>, nodes: &[NodeIndex]) -> UnGraph<(), ()> {
    let node_set: HashSet<NodeIndex> = nodes.iter().copied().collect();
    let mut subgraph = UnGraph::with_capacity(nodes.len(), 0);
    let mut old_to_new: HashMap<NodeIndex, NodeIndex> = HashMap::new();

    for &old in nodes {
        let new = subgraph.add_node(());
        old_to_new.insert(old, new);
    }

    for &old in nodes {
        for neighbor in graph.neighbors(old) {
            if neighbor.index() > old.index() && node_set.contains(&neighbor) {
                subgraph.add_edge(old_to_new[&old], old_to_new[&neighbor], ());
            }
        }
    }
    subgraph
}

/// Compute the complete invariant suite for a graph built from a predicate.
///
/// This is the primary entry point for Layer 4 analysis.  It builds
/// the graph, decomposes it into connected components, computes invariants
/// for each component, and classifies components into motif isomorphism
/// classes based on their combinatorial fingerprints.
pub fn compute_invariant_suite(
    predicate: &dyn GraphPredicate,
    n_nodes: usize,
) -> InvariantSuite {
    let graph = predicate.build_graph(n_nodes);
    compute_invariant_suite_from_graph(predicate.name(), &graph)
}

/// Compute the invariant suite directly from an already-built graph.
pub fn compute_invariant_suite_from_graph(
    name: &str,
    graph: &UnGraph<(), ()>,
) -> InvariantSuite {
    let n = graph.node_count();
    let global = compute_graph_invariants(graph);

    // Decompose into connected components
    let raw_components = extract_components(graph);

    // Compute per-component invariants
    let components: Vec<ComponentInvariant> = raw_components
        .iter()
        .enumerate()
        .map(|(id, nodes)| {
            let subgraph = induced_subgraph(graph, nodes);
            let inv = compute_graph_invariants(&subgraph);
            let fingerprint = MotifFingerprint::from_invariants(&inv);
            ComponentInvariant {
                component_id: id,
                original_nodes: nodes.iter().map(|n| n.index()).collect(),
                invariants: inv,
                fingerprint,
            }
        })
        .collect();

    // Classify into motif classes by fingerprint
    let mut class_map: HashMap<MotifFingerprint, Vec<usize>> = HashMap::new();
    for comp in &components {
        class_map
            .entry(comp.fingerprint.clone())
            .or_default()
            .push(comp.component_id);
    }
    let mut motif_classes: Vec<(MotifFingerprint, Vec<usize>)> =
        class_map.into_iter().collect();
    // Sort by class size descending, then by n_nodes for stability
    motif_classes.sort_by(|a, b| {
        b.1.len().cmp(&a.1.len())
            .then_with(|| b.0.n_nodes.cmp(&a.0.n_nodes))
    });

    InvariantSuite {
        predicate_name: name.to_string(),
        n_nodes: n,
        global,
        components,
        motif_classes,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pathion_matching_invariants_64d() {
        let graph = generate_pathion_matching(64);
        let inv = compute_graph_invariants(&graph);

        assert_eq!(inv.n_nodes, 64);
        assert_eq!(inv.n_edges, 32);
        assert_eq!(inv.n_components, 32);
        assert!(inv.degrees.iter().all(|&d| d == 1));
        
        // Spectrum should be +/- 1 (32 each)
        let n_pos = inv.spectrum.iter().filter(|&&e| (e - 1.0).abs() < 1e-10).count();
        let n_neg = inv.spectrum.iter().filter(|&&e| (e + 1.0).abs() < 1e-10).count();
        assert_eq!(n_pos, 32);
        assert_eq!(n_neg, 32);
        
        assert_eq!(inv.triangle_count, 0);
        assert_eq!(inv.girth, None);
    }

    // ── GraphPredicate trait tests ──────────────────────────────────────

    #[test]
    fn test_zd_predicate_matches_direct_function() {
        let pred = ZeroDivisorPredicate;
        assert_eq!(pred.name(), "P_ZD");
        assert_eq!(pred.min_dim(), 16);

        // Trait-dispatched graph should have identical invariants to the direct function
        let trait_inv = pred.invariants(32);
        let direct_inv = compute_graph_invariants(&generate_zd_parity_cliques(32));

        assert_eq!(trait_inv.n_nodes, direct_inv.n_nodes);
        assert_eq!(trait_inv.n_edges, direct_inv.n_edges);
        assert_eq!(trait_inv.n_components, direct_inv.n_components);
        assert_eq!(trait_inv.degrees, direct_inv.degrees);
        assert_eq!(trait_inv.triangle_count, direct_inv.triangle_count);
        assert_eq!(trait_inv.girth, direct_inv.girth);
    }

    #[test]
    fn test_xor_match_predicate_matches_direct_function() {
        let pred = XorMatchPredicate::new(64);
        assert_eq!(pred.name(), "P_match");
        assert_eq!(pred.min_dim(), 64);
        assert_eq!(pred.xor_mask, 4); // 64/16 = 4

        let trait_inv = pred.invariants(64);
        let direct_inv = compute_graph_invariants(&generate_pathion_matching(64));

        assert_eq!(trait_inv.n_nodes, direct_inv.n_nodes);
        assert_eq!(trait_inv.n_edges, direct_inv.n_edges);
        assert_eq!(trait_inv.n_components, direct_inv.n_components);
        assert_eq!(trait_inv.degrees, direct_inv.degrees);
        assert_eq!(trait_inv.triangle_count, direct_inv.triangle_count);
    }

    #[test]
    fn test_zd_predicate_test_fn() {
        let pred = ZeroDivisorPredicate;
        // Same parity -> adjacent
        assert!(pred.test(0, 2));
        assert!(pred.test(1, 3));
        assert!(pred.test(4, 6));
        // Different parity -> not adjacent
        assert!(!pred.test(0, 1));
        assert!(!pred.test(2, 3));
    }

    #[test]
    fn test_xor_match_predicate_test_fn() {
        let pred = XorMatchPredicate::new(64);
        // j = i XOR 4
        assert!(pred.test(0, 4));
        assert!(pred.test(1, 5));
        assert!(pred.test(10, 14));
        // Non-partners
        assert!(!pred.test(0, 1));
        assert!(!pred.test(0, 8));
    }

    #[test]
    fn test_matrix_predicate_from_bool() {
        // 4-node path graph: 0-1-2-3
        let adj = vec![
            vec![false, true, false, false],
            vec![true, false, true, false],
            vec![false, true, false, true],
            vec![false, false, true, false],
        ];
        let pred = MatrixPredicate::from_bool_matrix("P_path4", &adj);
        assert_eq!(pred.name(), "P_path4");
        assert_eq!(pred.min_dim(), 4);

        assert!(pred.test(0, 1));
        assert!(pred.test(1, 2));
        assert!(pred.test(2, 3));
        assert!(!pred.test(0, 2));
        assert!(!pred.test(0, 3));
        assert!(!pred.test(1, 3));

        let inv = pred.invariants(4);
        assert_eq!(inv.n_nodes, 4);
        assert_eq!(inv.n_edges, 3);
        assert_eq!(inv.n_components, 1);
        assert_eq!(inv.diameter, Some(3));
        assert_eq!(inv.triangle_count, 0);
    }

    #[test]
    fn test_matrix_predicate_from_u8() {
        // 3-node triangle (K3)
        let adj: Vec<Vec<u8>> = vec![
            vec![0, 1, 1],
            vec![1, 0, 1],
            vec![1, 1, 0],
        ];
        let pred = MatrixPredicate::from_u8_matrix("P_K3", &adj);

        let inv = pred.invariants(3);
        assert_eq!(inv.n_nodes, 3);
        assert_eq!(inv.n_edges, 3);
        assert_eq!(inv.n_components, 1);
        assert_eq!(inv.triangle_count, 1);
        assert_eq!(inv.girth, Some(3));
        assert_eq!(inv.diameter, Some(1));
    }

    #[test]
    fn test_matrix_predicate_out_of_bounds() {
        let adj = vec![vec![false, true], vec![true, false]];
        let pred = MatrixPredicate::from_bool_matrix("P_tiny", &adj);
        // Out-of-bounds indices should return false
        assert!(!pred.test(0, 5));
        assert!(!pred.test(5, 0));
    }

    // ── Original invariant tests ────────────────────────────────────────

    #[test]
    fn test_zd_parity_cliques_invariants_32d() {
        let graph = generate_zd_parity_cliques(32);
        let inv = compute_graph_invariants(&graph);

        assert_eq!(inv.n_nodes, 32);
        assert_eq!(inv.n_components, 2);
        // Each clique size 16 has 16*15/2 = 120 edges. Total 240.
        assert_eq!(inv.n_edges, 240);
        assert!(inv.degrees.iter().all(|&d| d == 15));
        
        // Spectrum: 15 (mult 2), -1 (mult 30)
        let n_15 = inv.spectrum.iter().filter(|&&e| (e - 15.0).abs() < 1e-10).count();
        let n_m1 = inv.spectrum.iter().filter(|&&e| (e + 1.0).abs() < 1e-10).count();
        assert_eq!(n_15, 2);
        assert_eq!(n_m1, 30);
        
        // Triangles: 2 * C(16,3) = 2 * (16*15*14 / 6) = 2 * 560 = 1120
        assert_eq!(inv.triangle_count, 1120);
        assert_eq!(inv.girth, Some(3));
    }

    // ── Layer 4: InvariantSuite tests ───────────────────────────────────

    #[test]
    fn test_invariant_suite_zd_32d() {
        let pred = ZeroDivisorPredicate;
        let suite = compute_invariant_suite(&pred, 32);

        assert_eq!(suite.predicate_name, "P_ZD");
        assert_eq!(suite.n_nodes, 32);
        assert_eq!(suite.global.n_components, 2);

        // Should decompose into exactly 2 components (even clique + odd clique)
        assert_eq!(suite.components.len(), 2);
        // Both components are K_16
        for comp in &suite.components {
            assert_eq!(comp.invariants.n_nodes, 16);
            assert_eq!(comp.invariants.n_edges, 120); // C(16,2)
            assert!(comp.invariants.degrees.iter().all(|&d| d == 15));
            assert_eq!(comp.invariants.triangle_count, 560); // C(16,3)
            assert_eq!(comp.invariants.girth, Some(3));
            assert_eq!(comp.invariants.diameter, Some(1));
        }

        // Both components are isomorphic -> 1 motif class
        assert_eq!(suite.n_motif_classes(), 1);
        assert_eq!(suite.motif_classes[0].1.len(), 2);
    }

    #[test]
    fn test_invariant_suite_xor_matching_64d() {
        let pred = XorMatchPredicate::new(64);
        let suite = compute_invariant_suite(&pred, 64);

        assert_eq!(suite.predicate_name, "P_match");
        assert_eq!(suite.n_nodes, 64);
        assert_eq!(suite.global.n_components, 32);

        // 32 components, each a single edge (K_2)
        assert_eq!(suite.components.len(), 32);
        for comp in &suite.components {
            assert_eq!(comp.invariants.n_nodes, 2);
            assert_eq!(comp.invariants.n_edges, 1);
            assert_eq!(comp.invariants.triangle_count, 0);
        }

        // All components are K_2 -> 1 motif class
        assert_eq!(suite.n_motif_classes(), 1);
        assert_eq!(suite.motif_classes[0].1.len(), 32);
    }

    #[test]
    fn test_invariant_suite_disconnected_graph() {
        // Build a graph with 2 different component types: K3 + K2
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());
        let n3 = graph.add_node(());
        let n4 = graph.add_node(());

        // K3: 0-1-2-0
        graph.add_edge(n0, n1, ());
        graph.add_edge(n1, n2, ());
        graph.add_edge(n0, n2, ());
        // K2: 3-4
        graph.add_edge(n3, n4, ());

        let suite = compute_invariant_suite_from_graph("test_mixed", &graph);

        assert_eq!(suite.components.len(), 2);
        // Largest component first (K3 has 3 nodes)
        assert_eq!(suite.components[0].invariants.n_nodes, 3);
        assert_eq!(suite.components[1].invariants.n_nodes, 2);

        // Two different motif classes: K3 and K2
        assert_eq!(suite.n_motif_classes(), 2);
        assert_eq!(suite.max_component_size(), 3);
        assert_eq!(suite.min_component_size(), 2);
    }

    #[test]
    fn test_invariant_suite_single_component() {
        // Complete graph K4
        let mut graph = UnGraph::<(), ()>::new_undirected();
        let nodes: Vec<_> = (0..4).map(|_| graph.add_node(())).collect();
        for i in 0..4 {
            for j in (i + 1)..4 {
                graph.add_edge(nodes[i], nodes[j], ());
            }
        }

        let suite = compute_invariant_suite_from_graph("K4", &graph);

        assert_eq!(suite.components.len(), 1);
        assert_eq!(suite.n_motif_classes(), 1);
        assert_eq!(suite.components[0].invariants.n_nodes, 4);
        assert_eq!(suite.components[0].invariants.n_edges, 6);
        assert_eq!(suite.components[0].invariants.triangle_count, 4); // C(4,3)
        assert_eq!(suite.components[0].invariants.diameter, Some(1));
    }

    #[test]
    fn test_motif_fingerprint_equality() {
        // Two K_16 cliques should produce the same fingerprint
        let pred = ZeroDivisorPredicate;
        let suite = compute_invariant_suite(&pred, 32);

        let fp0 = &suite.components[0].fingerprint;
        let fp1 = &suite.components[1].fingerprint;
        assert_eq!(fp0, fp1);
    }

    #[test]
    fn test_extract_components_empty_graph() {
        let graph = UnGraph::<(), ()>::new_undirected();
        let comps = extract_components(&graph);
        assert!(comps.is_empty());
    }

    #[test]
    fn test_extract_components_isolated_nodes() {
        let mut graph = UnGraph::<(), ()>::new_undirected();
        for _ in 0..5 {
            graph.add_node(());
        }
        let comps = extract_components(&graph);
        assert_eq!(comps.len(), 5);
        for c in &comps {
            assert_eq!(c.len(), 1);
        }
    }

    // ====================================================================
    // Cross-validation: InvariantSuite vs MotifComponent
    //
    // These tests convert each MotifComponent to a petgraph UnGraph, then
    // compute GraphInvariants via compute_graph_invariants.  The results
    // must match the MotifComponent's own computed invariants (spectrum,
    // triangles, diameter, girth, degree sequence).
    // ====================================================================

    use crate::analysis::boxkites::motif_components_for_cross_assessors;

    /// Helper: compare two sorted spectra within tolerance.
    fn spectra_match(a: &[f64], b: &[f64], tol: f64) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn test_cross_validate_dim16() {
        let comps = motif_components_for_cross_assessors(16);
        assert_eq!(comps.len(), 7, "dim=16 must have 7 components");

        for (i, mc) in comps.iter().enumerate() {
            let graph = mc.to_petgraph();
            let inv = compute_graph_invariants(&graph);

            // Structural counts
            assert_eq!(inv.n_nodes, mc.nodes.len(),
                "dim=16 comp {i}: node count mismatch");
            assert_eq!(inv.n_edges, mc.edges.len(),
                "dim=16 comp {i}: edge count mismatch");
            assert_eq!(inv.n_components, 1,
                "dim=16 comp {i}: must be single connected component");

            // Degree sequence
            assert_eq!(inv.degrees, mc.degree_sequence(),
                "dim=16 comp {i}: degree sequence mismatch");

            // Triangle count
            assert_eq!(inv.triangle_count, mc.triangle_count(),
                "dim=16 comp {i}: triangle count mismatch");

            // Diameter
            let mc_diam = mc.diameter();
            assert_eq!(inv.diameter, Some(mc_diam),
                "dim=16 comp {i}: diameter mismatch");

            // Girth
            let mc_girth = mc.girth();
            let expected_girth = if mc_girth == usize::MAX { None } else { Some(mc_girth) };
            assert_eq!(inv.girth, expected_girth,
                "dim=16 comp {i}: girth mismatch");

            // Spectrum (ascending in GraphInvariants, descending in MotifComponent)
            let mut mc_spec = mc.spectrum();
            mc_spec.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert!(spectra_match(&inv.spectrum, &mc_spec, 1e-8),
                "dim=16 comp {i}: spectrum mismatch");
        }
    }

    #[test]
    fn test_cross_validate_dim32() {
        let comps = motif_components_for_cross_assessors(32);
        assert_eq!(comps.len(), 15, "dim=32 must have 15 components");

        for (i, mc) in comps.iter().enumerate() {
            let graph = mc.to_petgraph();
            let inv = compute_graph_invariants(&graph);

            assert_eq!(inv.n_nodes, mc.nodes.len(),
                "dim=32 comp {i}: node count");
            assert_eq!(inv.n_edges, mc.edges.len(),
                "dim=32 comp {i}: edge count");
            assert_eq!(inv.n_components, 1,
                "dim=32 comp {i}: connected");
            assert_eq!(inv.degrees, mc.degree_sequence(),
                "dim=32 comp {i}: degrees");
            assert_eq!(inv.triangle_count, mc.triangle_count(),
                "dim=32 comp {i}: triangles");
            assert_eq!(inv.diameter, Some(mc.diameter()),
                "dim=32 comp {i}: diameter");

            let mc_girth = mc.girth();
            let expected_girth = if mc_girth == usize::MAX { None } else { Some(mc_girth) };
            assert_eq!(inv.girth, expected_girth,
                "dim=32 comp {i}: girth");

            let mut mc_spec = mc.spectrum();
            mc_spec.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert!(spectra_match(&inv.spectrum, &mc_spec, 1e-8),
                "dim=32 comp {i}: spectrum");
        }
    }

    #[test]
    fn test_cross_validate_dim64() {
        let comps = motif_components_for_cross_assessors(64);
        assert_eq!(comps.len(), 31, "dim=64 must have 31 components");

        for (i, mc) in comps.iter().enumerate() {
            let graph = mc.to_petgraph();
            let inv = compute_graph_invariants(&graph);

            assert_eq!(inv.n_nodes, mc.nodes.len(),
                "dim=64 comp {i}: node count");
            assert_eq!(inv.n_edges, mc.edges.len(),
                "dim=64 comp {i}: edge count");
            assert_eq!(inv.n_components, 1,
                "dim=64 comp {i}: connected");
            assert_eq!(inv.degrees, mc.degree_sequence(),
                "dim=64 comp {i}: degrees");
            assert_eq!(inv.triangle_count, mc.triangle_count(),
                "dim=64 comp {i}: triangles");
            assert_eq!(inv.diameter, Some(mc.diameter()),
                "dim=64 comp {i}: diameter");

            let mc_girth = mc.girth();
            let expected_girth = if mc_girth == usize::MAX { None } else { Some(mc_girth) };
            assert_eq!(inv.girth, expected_girth,
                "dim=64 comp {i}: girth");

            let mut mc_spec = mc.spectrum();
            mc_spec.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert!(spectra_match(&inv.spectrum, &mc_spec, 1e-8),
                "dim=64 comp {i}: spectrum");
        }
    }

    #[test]
    fn test_cross_validate_motif_classes_dim32() {
        // Verify InvariantSuite's motif classification matches MotifComponent's
        // own spectral grouping at dim=32.
        let comps = motif_components_for_cross_assessors(32);

        // Build full graph from all components
        let mut full_graph = UnGraph::<(), ()>::new_undirected();
        let mut all_nodes: HashMap<(usize, usize), NodeIndex> = HashMap::new();

        // Collect all unique nodes across all components
        for mc in &comps {
            for &cp in &mc.nodes {
                if !all_nodes.contains_key(&cp) {
                    let idx = full_graph.add_node(());
                    all_nodes.insert(cp, idx);
                }
            }
        }
        // Add all edges
        for mc in &comps {
            for &(u, v) in &mc.edges {
                full_graph.add_edge(all_nodes[&u], all_nodes[&v], ());
            }
        }

        let suite = compute_invariant_suite_from_graph("cross_assessor_32", &full_graph);

        // 15 components, 2 motif classes at dim=32
        assert_eq!(suite.components.len(), 15);
        assert_eq!(suite.n_motif_classes(), 2,
            "dim=32 should have 2 motif classes");

        // Verify class sizes: 8 heptacross + 7 mixed-degree
        let class_sizes: Vec<usize> = suite.motif_classes.iter()
            .map(|(_, ids)| ids.len())
            .collect();
        assert!(class_sizes.contains(&8), "should have class of size 8");
        assert!(class_sizes.contains(&7), "should have class of size 7");
    }

    #[test]
    fn test_cross_validate_motif_classes_dim64() {
        // dim=64: 4 motif classes
        let comps = motif_components_for_cross_assessors(64);

        let mut full_graph = UnGraph::<(), ()>::new_undirected();
        let mut all_nodes: HashMap<(usize, usize), NodeIndex> = HashMap::new();

        for mc in &comps {
            for &cp in &mc.nodes {
                if !all_nodes.contains_key(&cp) {
                    let idx = full_graph.add_node(());
                    all_nodes.insert(cp, idx);
                }
            }
        }
        for mc in &comps {
            for &(u, v) in &mc.edges {
                full_graph.add_edge(all_nodes[&u], all_nodes[&v], ());
            }
        }

        let suite = compute_invariant_suite_from_graph("cross_assessor_64", &full_graph);

        assert_eq!(suite.components.len(), 31);
        assert_eq!(suite.n_motif_classes(), 4,
            "dim=64 should have 4 motif classes");

        // Class sizes should sum to 31
        let total: usize = suite.motif_classes.iter()
            .map(|(_, ids)| ids.len())
            .sum();
        assert_eq!(total, 31);
    }
}
