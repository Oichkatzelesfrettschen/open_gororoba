//! Zero-divisor interaction graphs for Cayley-Dickson algebras.
//!
//! This module uses petgraph to model the algebraic structure of zero-divisors
//! as graphs, enabling network analysis of the multiplication table structure.
//!
//! # Literature Context
//! - de Marrais (2001): Box-kite structure of sedenion zero-divisors
//! - Moreno & Recht (2007): Algebraic automorphisms and ZD families
//! - Project claim C-078: General-form ZD spectrum diversity
//!
//! # Graph Models
//! 1. **ZD Interaction Graph**: Nodes are ZD pairs, edges connect pairs sharing basis elements
//! 2. **Basis Participation Graph**: Nodes are basis indices, edges weighted by ZD involvement
//! 3. **Associator Triad Graph**: Nodes are basis elements, edges are non-zero associators

use petgraph::graph::{DiGraph, UnGraph};
use petgraph::algo::{connected_components, dijkstra, tarjan_scc};
use std::collections::{HashMap, HashSet};
use crate::cayley_dickson::{find_zero_divisors, cd_associator_norm};

/// A zero-divisor pair: indices (i, j, k, l) where (e_i + e_j) * (e_k +/- e_l) ~ 0.
pub type ZdPair = (usize, usize, usize, usize, f64);

/// Result of ZD graph analysis.
#[derive(Debug, Clone)]
pub struct ZdGraphAnalysis {
    /// Number of nodes (ZD pairs)
    pub n_nodes: usize,
    /// Number of edges (shared basis elements)
    pub n_edges: usize,
    /// Number of connected components (ZD families)
    pub n_components: usize,
    /// Size of largest component
    pub largest_component_size: usize,
    /// Average degree (connections per ZD pair)
    pub avg_degree: f64,
    /// Maximum degree (most connected ZD pair)
    pub max_degree: usize,
    /// Clustering coefficient (local clustering)
    pub clustering_coefficient: f64,
    /// Basis elements ranked by participation count
    pub basis_participation: Vec<(usize, usize)>,
}

/// Result of basis participation analysis.
#[derive(Debug, Clone)]
pub struct BasisParticipationResult {
    /// Basis index -> number of ZD pairs it participates in
    pub counts: Vec<(usize, usize)>,
    /// Shannon entropy of participation distribution
    pub entropy: f64,
    /// Gini coefficient (inequality measure)
    pub gini: f64,
    /// Indices of "hub" bases (>= 2x mean participation)
    pub hub_indices: Vec<usize>,
}

/// Result of associator graph analysis.
#[derive(Debug, Clone)]
pub struct AssociatorGraphResult {
    /// Number of basis pairs with non-zero associators
    pub n_nonzero_pairs: usize,
    /// Number of strongly connected components
    pub n_scc: usize,
    /// Mean associator norm
    pub mean_norm: f64,
    /// Max associator norm
    pub max_norm: f64,
    /// Pairs with largest associator norms
    pub top_pairs: Vec<((usize, usize, usize), f64)>,
}

/// Build the ZD interaction graph where nodes are ZD pairs and edges
/// connect pairs that share at least one basis element.
pub fn build_zd_interaction_graph(dim: usize, atol: f64) -> UnGraph<ZdPair, f64> {
    let zd_pairs = find_zero_divisors(dim, atol);
    let n = zd_pairs.len();

    let mut graph = UnGraph::<ZdPair, f64>::new_undirected();
    let mut node_indices = Vec::with_capacity(n);

    // Add nodes
    for pair in &zd_pairs {
        node_indices.push(graph.add_node(*pair));
    }

    // Add edges for pairs sharing basis elements
    for i in 0..n {
        let (i1, j1, k1, l1, _) = zd_pairs[i];
        let set1: HashSet<usize> = [i1, j1, k1, l1].into_iter().collect();

        for j in (i + 1)..n {
            let (i2, j2, k2, l2, _) = zd_pairs[j];
            let set2: HashSet<usize> = [i2, j2, k2, l2].into_iter().collect();

            let shared = set1.intersection(&set2).count();
            if shared > 0 {
                // Weight by number of shared elements
                graph.add_edge(node_indices[i], node_indices[j], shared as f64);
            }
        }
    }

    graph
}

/// Analyze the ZD interaction graph structure.
pub fn analyze_zd_graph(dim: usize, atol: f64) -> ZdGraphAnalysis {
    let graph = build_zd_interaction_graph(dim, atol);
    let n_nodes = graph.node_count();
    let n_edges = graph.edge_count();

    if n_nodes == 0 {
        return ZdGraphAnalysis {
            n_nodes: 0,
            n_edges: 0,
            n_components: 0,
            largest_component_size: 0,
            avg_degree: 0.0,
            max_degree: 0,
            clustering_coefficient: 0.0,
            basis_participation: Vec::new(),
        };
    }

    // Connected components
    let n_components = connected_components(&graph);

    // Degree statistics
    let degrees: Vec<usize> = graph.node_indices()
        .map(|n| graph.edges(n).count())
        .collect();

    let max_degree = *degrees.iter().max().unwrap_or(&0);
    let avg_degree = degrees.iter().sum::<usize>() as f64 / n_nodes as f64;

    // Clustering coefficient (ratio of triangles to possible triangles)
    let mut total_triangles = 0usize;
    let mut total_triples = 0usize;

    for node in graph.node_indices() {
        let neighbors: Vec<_> = graph.neighbors(node).collect();
        let k = neighbors.len();
        if k >= 2 {
            total_triples += k * (k - 1) / 2;
            // Count triangles
            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    if graph.find_edge(neighbors[i], neighbors[j]).is_some() {
                        total_triangles += 1;
                    }
                }
            }
        }
    }

    let clustering_coefficient = if total_triples > 0 {
        total_triangles as f64 / total_triples as f64
    } else {
        0.0
    };

    // Find largest component via BFS
    let mut largest = 0usize;
    let mut global_visited = HashSet::new();
    for start in graph.node_indices() {
        if global_visited.contains(&start) {
            continue;
        }
        let mut component_size = 0;
        let mut stack = vec![start];
        while let Some(n) = stack.pop() {
            if global_visited.insert(n) {
                component_size += 1;
                for neighbor in graph.neighbors(n) {
                    if !global_visited.contains(&neighbor) {
                        stack.push(neighbor);
                    }
                }
            }
        }
        largest = largest.max(component_size);
    }

    // Basis participation
    let zd_pairs = find_zero_divisors(dim, atol);
    let mut basis_counts: HashMap<usize, usize> = HashMap::new();
    for (i, j, k, l, _) in &zd_pairs {
        *basis_counts.entry(*i).or_insert(0) += 1;
        *basis_counts.entry(*j).or_insert(0) += 1;
        *basis_counts.entry(*k).or_insert(0) += 1;
        *basis_counts.entry(*l).or_insert(0) += 1;
    }
    let mut basis_participation: Vec<_> = basis_counts.into_iter().collect();
    basis_participation.sort_by_key(|x| std::cmp::Reverse(x.1));

    ZdGraphAnalysis {
        n_nodes,
        n_edges,
        n_components,
        largest_component_size: largest,
        avg_degree,
        max_degree,
        clustering_coefficient,
        basis_participation,
    }
}

/// Analyze which basis elements participate most in zero-divisors.
pub fn analyze_basis_participation(dim: usize, atol: f64) -> BasisParticipationResult {
    let zd_pairs = find_zero_divisors(dim, atol);

    // Count participation for each basis element
    let mut counts = vec![0usize; dim];
    for (i, j, k, l, _) in &zd_pairs {
        counts[*i] += 1;
        counts[*j] += 1;
        counts[*k] += 1;
        counts[*l] += 1;
    }

    let total: usize = counts.iter().sum();

    // Shannon entropy
    let entropy = if total > 0 {
        let total_f = total as f64;
        counts.iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / total_f;
                -p * p.ln()
            })
            .sum()
    } else {
        0.0
    };

    // Gini coefficient
    let mut sorted_counts = counts.clone();
    sorted_counts.sort();
    let n = sorted_counts.len() as f64;
    let gini = if total > 0 {
        let sum_indexed: f64 = sorted_counts.iter()
            .enumerate()
            .map(|(i, &c)| (i + 1) as f64 * c as f64)
            .sum();
        let sum_counts: f64 = sorted_counts.iter().map(|&c| c as f64).sum();
        (2.0 * sum_indexed) / (n * sum_counts) - (n + 1.0) / n
    } else {
        0.0
    };

    // Hub indices
    let mean_count = if dim > 0 { total as f64 / dim as f64 } else { 0.0 };
    let hub_indices: Vec<usize> = counts.iter()
        .enumerate()
        .filter(|(_, &c)| c as f64 >= 2.0 * mean_count)
        .map(|(i, _)| i)
        .collect();

    // Sorted (index, count) pairs
    let mut indexed_counts: Vec<(usize, usize)> = counts.into_iter().enumerate().collect();
    indexed_counts.sort_by_key(|x| std::cmp::Reverse(x.1));

    BasisParticipationResult {
        counts: indexed_counts,
        entropy,
        gini,
        hub_indices,
    }
}

/// Build directed graph of associator relationships.
/// Edge (i,j,k) -> (j,k,l) if both triples have non-zero associators.
pub fn build_associator_graph(dim: usize, threshold: f64) -> DiGraph<usize, f64> {
    let mut graph = DiGraph::<usize, f64>::new();

    // Add nodes for each basis element
    let node_indices: Vec<_> = (0..dim).map(|i| graph.add_node(i)).collect();

    // Create basis vectors
    let mut basis_vectors: Vec<Vec<f64>> = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut v = vec![0.0; dim];
        v[i] = 1.0;
        basis_vectors.push(v);
    }

    // Check all triples for non-zero associators
    let mut associators: Vec<((usize, usize, usize), f64)> = Vec::new();

    for i in 1..dim {  // Skip e_0 = 1
        for j in 1..dim {
            if j == i { continue; }
            for k in 1..dim {
                if k == i || k == j { continue; }

                let norm = cd_associator_norm(&basis_vectors[i], &basis_vectors[j], &basis_vectors[k]);
                if norm > threshold {
                    associators.push(((i, j, k), norm));
                    // Add edge from j to k weighted by associator norm
                    graph.add_edge(node_indices[j], node_indices[k], norm);
                }
            }
        }
    }

    graph
}

/// Analyze the associator structure as a directed graph.
pub fn analyze_associator_graph(dim: usize, threshold: f64) -> AssociatorGraphResult {
    let graph = build_associator_graph(dim, threshold);

    // Create basis vectors for associator computation
    let mut basis_vectors: Vec<Vec<f64>> = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut v = vec![0.0; dim];
        v[i] = 1.0;
        basis_vectors.push(v);
    }

    // Collect all non-zero associators
    let mut associators: Vec<((usize, usize, usize), f64)> = Vec::new();

    for i in 1..dim {
        for j in 1..dim {
            if j == i { continue; }
            for k in 1..dim {
                if k == i || k == j { continue; }

                let norm = cd_associator_norm(&basis_vectors[i], &basis_vectors[j], &basis_vectors[k]);
                if norm > threshold {
                    associators.push(((i, j, k), norm));
                }
            }
        }
    }

    let n_nonzero = associators.len();

    // Statistics
    let mean_norm = if n_nonzero > 0 {
        associators.iter().map(|(_, n)| n).sum::<f64>() / n_nonzero as f64
    } else {
        0.0
    };

    let max_norm = associators.iter().map(|(_, n)| *n).fold(0.0, f64::max);

    // Top pairs
    associators.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_pairs: Vec<_> = associators.into_iter().take(10).collect();

    // Strongly connected components
    let scc = tarjan_scc(&graph);
    let n_scc = scc.len();

    AssociatorGraphResult {
        n_nonzero_pairs: n_nonzero,
        n_scc,
        mean_norm,
        max_norm,
        top_pairs,
    }
}

/// Find shortest path between two ZD pairs in the interaction graph.
/// Returns (distance, path) or None if disconnected.
pub fn zd_shortest_path(
    dim: usize,
    pair1_idx: usize,
    pair2_idx: usize,
    atol: f64,
) -> Option<(f64, Vec<usize>)> {
    let graph = build_zd_interaction_graph(dim, atol);
    let node_indices: Vec<_> = graph.node_indices().collect();

    if pair1_idx >= node_indices.len() || pair2_idx >= node_indices.len() {
        return None;
    }

    let start = node_indices[pair1_idx];
    let end = node_indices[pair2_idx];

    // Use Dijkstra with inverse weights (more shared = shorter)
    let distances = dijkstra(&graph, start, Some(end), |e| 1.0 / e.weight());

    distances.get(&end).map(|&d| (d, vec![pair1_idx, pair2_idx]))
}

/// Compute the "algebraic diameter" of the ZD graph (max shortest path).
pub fn zd_graph_diameter(dim: usize, atol: f64) -> usize {
    let graph = build_zd_interaction_graph(dim, atol);
    let nodes: Vec<_> = graph.node_indices().collect();

    if nodes.is_empty() {
        return 0;
    }

    let mut max_dist = 0usize;

    // Sample pairs for large graphs
    let sample_size = nodes.len().min(50);

    for &start in nodes.iter().take(sample_size) {
        let distances = dijkstra(&graph, start, None, |_| 1);
        for &dist in distances.values() {
            max_dist = max_dist.max(dist as usize);
        }
    }

    max_dist
}

// ---------------------------------------------------------------------------
// XOR-bucket heuristics for Cayley-Dickson zero-product detection
// ---------------------------------------------------------------------------

/// XOR-bucket key for a Cayley-Dickson basis pair.
///
/// In CD algebras, `e_i * e_j = +/- e_{i^j}`. The XOR of two basis indices
/// determines the target basis element. This key is used for grouping
/// assessors that could potentially annihilate.
#[inline]
pub fn xor_key(i: usize, j: usize) -> usize {
    i ^ j
}

/// Necessary (not sufficient) condition for a 2-blade zero-product:
/// `(e_i + e_j) * (e_k + e_l) = 0` requires `(i ^ j) == (k ^ l)`.
///
/// This is a fast O(1) pre-filter that eliminates most non-zero-product pairs.
#[inline]
pub fn xor_bucket_necessary_for_two_blade(i: usize, j: usize, k: usize, l: usize) -> bool {
    xor_key(i, j) == xor_key(k, l)
}

/// Check if a 4-tuple is XOR-balanced: `a ^ b ^ c ^ d == 0`.
///
/// This is a common necessary constraint for 4-term cancellation patterns
/// in XOR-indexed algebras.
#[inline]
pub fn xor_balanced_four_tuple(a: usize, b: usize, c: usize, d: usize) -> bool {
    (a ^ b ^ c ^ d) == 0
}

/// For a XOR-balanced 4-tuple, return the 3 pairing bucket values.
///
/// The 3 perfect matchings of {a,b,c,d} into pairs give shared XOR buckets:
///   (a,b)-(c,d): bucket = a^b = c^d
///   (a,c)-(b,d): bucket = a^c = b^d
///   (a,d)-(b,c): bucket = a^d = b^c
///
/// Returns an array of 3 bucket values (NOT necessarily distinct).
///
/// # Panics
/// Panics if the 4-tuple is not XOR-balanced.
pub fn xor_pairing_buckets(a: usize, b: usize, c: usize, d: usize) -> [usize; 3] {
    assert!(
        xor_balanced_four_tuple(a, b, c, d),
        "Expected XOR-balanced 4-tuple (a^b^c^d == 0), got {}", a ^ b ^ c ^ d,
    );
    [a ^ b, a ^ c, a ^ d]
}

/// Necessary condition for a 2-blade to be compatible with a XOR-balanced 4-blade.
///
/// The 2-blade bucket `(i ^ j)` must match one of the 3 pairing buckets
/// induced by the balanced 4-tuple `(a, b, c, d)`.
///
/// Returns false if the 4-tuple is not XOR-balanced.
pub fn xor_bucket_necessary_2v4(
    i: usize, j: usize,
    a: usize, b: usize, c: usize, d: usize,
) -> bool {
    if !xor_balanced_four_tuple(a, b, c, d) {
        return false;
    }
    let bucket = xor_key(i, j);
    let pairings = xor_pairing_buckets(a, b, c, d);
    pairings.contains(&bucket)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sedenion_zd_graph() {
        let analysis = analyze_zd_graph(16, 1e-10);

        // Sedenions have known ZD pairs
        assert!(analysis.n_nodes > 0, "Should find ZD pairs in sedenions");
        assert!(analysis.n_edges > 0, "ZD pairs should share basis elements");

        // Should form connected structure (de Marrais box-kites)
        assert!(analysis.n_components > 0);
        assert!(analysis.clustering_coefficient >= 0.0);
        assert!(analysis.clustering_coefficient <= 1.0);
    }

    #[test]
    fn test_octonion_no_zd() {
        // Octonions have no zero-divisors
        let analysis = analyze_zd_graph(8, 1e-10);
        assert_eq!(analysis.n_nodes, 0);
        assert_eq!(analysis.n_edges, 0);
    }

    #[test]
    fn test_basis_participation() {
        let result = analyze_basis_participation(16, 1e-10);

        // Should have non-zero counts for basis elements involved in ZDs
        let total_participation: usize = result.counts.iter().map(|(_, c)| c).sum();
        assert!(total_participation > 0);

        // Entropy should be positive for non-uniform distribution
        assert!(result.entropy > 0.0);

        // Gini coefficient in [0, 1]
        assert!(result.gini >= 0.0);
        assert!(result.gini <= 1.0);
    }

    #[test]
    fn test_associator_graph_quaternions() {
        // Quaternions are associative, no non-zero associators
        let result = analyze_associator_graph(4, 1e-10);
        assert_eq!(result.n_nonzero_pairs, 0);
        assert_eq!(result.mean_norm, 0.0);
    }

    #[test]
    fn test_associator_graph_octonions() {
        // Octonions are non-associative
        let result = analyze_associator_graph(8, 1e-10);
        assert!(result.n_nonzero_pairs > 0, "Octonions should have non-zero associators");
        assert!(result.mean_norm > 0.0);
    }

    #[test]
    fn test_associator_graph_sedenions() {
        // Sedenions have even more non-associativity
        let result_8 = analyze_associator_graph(8, 1e-10);
        let result_16 = analyze_associator_graph(16, 1e-10);

        // More non-zero associators in higher dimension
        assert!(result_16.n_nonzero_pairs > result_8.n_nonzero_pairs);
    }

    #[test]
    fn test_zd_graph_diameter() {
        let diameter = zd_graph_diameter(16, 1e-10);
        // Diameter should be finite for connected graph
        assert!(diameter > 0);
        assert!(diameter < 100); // Reasonable upper bound
    }

    // --- XOR heuristic tests ---

    #[test]
    fn test_xor_balanced_four_tuple() {
        // 1^2^4^7 = 0b001 ^ 0b010 ^ 0b100 ^ 0b111 = 0
        assert!(xor_balanced_four_tuple(1, 2, 4, 7));
        // 1^2^3^0 = 0 (since 1^2=3, 3^3=0, 0^0=0)
        assert!(xor_balanced_four_tuple(1, 2, 3, 0));
        // Non-balanced
        assert!(!xor_balanced_four_tuple(1, 2, 3, 4));
    }

    #[test]
    fn test_xor_pairing_buckets() {
        let (a, b, c, d) = (1, 2, 4, 7);
        let buckets = xor_pairing_buckets(a, b, c, d);
        // a^b = 3, a^c = 5, a^d = 6
        assert_eq!(buckets, [3, 5, 6]);
        // Verify each pairing has matching XOR
        assert_eq!(a ^ b, c ^ d); // 3 == 3
        assert_eq!(a ^ c, b ^ d); // 5 == 5
        assert_eq!(a ^ d, b ^ c); // 6 == 6
    }

    #[test]
    fn test_xor_bucket_necessary_2v4() {
        let (a, b, c, d) = (1, 2, 4, 7);
        // 2-blades matching one of the 3 pairing buckets (3, 5, 6)
        assert!(xor_bucket_necessary_2v4(1, 2, a, b, c, d)); // bucket 3
        assert!(xor_bucket_necessary_2v4(1, 4, a, b, c, d)); // bucket 5
        assert!(xor_bucket_necessary_2v4(1, 7, a, b, c, d)); // bucket 6
        // Non-matching bucket
        assert!(!xor_bucket_necessary_2v4(1, 3, a, b, c, d)); // bucket 2
    }

    #[test]
    fn test_xor_bucket_necessary_two_blade() {
        // (i^j) must equal (k^l) for zero-product possibility
        assert!(xor_bucket_necessary_for_two_blade(1, 10, 2, 9));  // 11 == 11
        assert!(!xor_bucket_necessary_for_two_blade(1, 10, 2, 10)); // 11 != 8
    }
}
