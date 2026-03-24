//! Topological analysis for lattice filtrations.
//!
//! Provides graph-based persistence metrics and connectivity analysis
//! for zero-divisor structures and lattice embeddings.
//!
//! Migrated from src/topology_analysis.py.

use petgraph::{algo::connected_components, graph::UnGraph};
use std::collections::HashMap;

/// Result of a topological connectivity analysis.
#[derive(Debug, Clone)]
pub struct TopologyStats {
    pub n_nodes: usize,
    pub n_edges: usize,
    pub mean_degree: f64,
    pub diameter_estimate: Option<usize>,
    pub n_components: usize,
}

/// Analyze connectivity of an edge-list representing a lattice filtration.
pub fn analyze_lattice_topology(edges: &[(usize, usize)]) -> TopologyStats {
    let mut g = UnGraph::<usize, ()>::default();
    let mut node_map = HashMap::new();

    for &(u, v) in edges {
        let u_idx = *node_map.entry(u).or_insert_with(|| g.add_node(u));
        let v_idx = *node_map.entry(v).or_insert_with(|| g.add_node(v));
        g.add_edge(u_idx, v_idx, ());
    }

    let n_nodes = g.node_count();
    let n_edges = g.edge_count();
    let mean_degree = if n_nodes > 0 {
        2.0 * n_edges as f64 / n_nodes as f64
    } else {
        0.0
    };

    let n_components = connected_components(&g);

    TopologyStats {
        n_nodes,
        n_edges,
        mean_degree,
        diameter_estimate: None,
        n_components,
    }
}

/// Compute a persistent homology estimate (Betti-0) for a filtered graph.
///
/// Returns a sequence of component counts as a function of filtration step.
pub fn compute_graph_persistence_b0(
    _nodes: &[usize],
    edges: &[(usize, usize, f64)], // (u, v, filtration_value)
    steps: &[f64],
) -> Vec<usize> {
    let mut results = Vec::new();
    for &threshold in steps {
        let active_edges: Vec<(usize, usize)> = edges
            .iter()
            .filter(|&&(_, _, val)| val <= threshold)
            .map(|&(u, v, _)| (u, v))
            .collect();
        let stats = analyze_lattice_topology(&active_edges);
        results.push(stats.n_nodes - active_edges.len()); // Simple estimate
    }
    results
}
