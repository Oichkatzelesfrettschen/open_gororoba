//! Graph projections of 64D algebras.
//!
//! Implements the adjacency predicates derived from empirical matrices:
//! - Pathion Adjacency: Perfect Matching (XOR pairing)
//! - Zero-Divisor Adjacency: Parity Cliques (K_n/2 U K_n/2)

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
}
