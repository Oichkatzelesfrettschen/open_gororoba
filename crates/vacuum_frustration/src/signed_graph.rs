//! Signed Graph construction and manipulation.
//!
//! A signed graph represents edges with +1 or -1 signs, derived from the
//! Cayley-Dickson basis multiplication signs (psi matrix).

use petgraph::graph::{NodeIndex, UnGraph};
use std::collections::HashMap;

/// Signed graph built from Cayley-Dickson psi matrix.
///
/// Each edge in the graph carries a sign (+1 or -1) determined by the
/// multiplication table of the corresponding algebra.
pub struct SignedGraph {
    /// Underlying petgraph UnGraph with node = basis index, edge = sign
    pub graph: UnGraph<usize, i32>,
    /// Dimension of the algebra
    pub dim: usize,
    /// Map from basis index to node index (for fast lookup)
    index_to_node: HashMap<usize, NodeIndex>,
}

impl SignedGraph {
    /// Create empty signed graph.
    pub fn new(dim: usize) -> Self {
        Self {
            graph: UnGraph::new_undirected(),
            dim,
            index_to_node: HashMap::new(),
        }
    }

    /// Build signed graph from Cayley-Dickson psi matrix.
    ///
    /// Constructs graph where:
    /// - Nodes = basis elements (0..dim)
    /// - Edges = non-zero products in multiplication table
    /// - Edge weights = sign of the product (±1)
    ///
    /// Returns a connected graph where e_i * e_j creates edge (i,j) with sign psi(i,j).
    ///
    /// # Arguments
    /// * `dim` - Dimension (must be power of 2)
    /// * `psi` - Function that computes cd_basis_mul_sign(dim, i, j)
    pub fn from_psi_matrix<F>(dim: usize, psi: F) -> Self
    where
        F: Fn(usize, usize) -> i32,
    {
        assert!(dim.is_power_of_two(), "Dimension must be power of 2");

        let mut graph = UnGraph::new_undirected();
        let mut index_to_node = HashMap::new();

        // Create all nodes
        for i in 0..dim {
            let node = graph.add_node(i);
            index_to_node.insert(i, node);
        }

        // Add edges from psi matrix
        // Only add edges where i < j to avoid duplicates in undirected graph
        for i in 0..dim {
            for j in (i + 1)..dim {
                let sign = psi(i, j); // Use forward multiplication sign

                let node_i = index_to_node[&i];
                let node_j = index_to_node[&j];
                graph.add_edge(node_i, node_j, sign);
            }
        }

        Self {
            graph,
            dim,
            index_to_node,
        }
    }

    /// Get node index for a basis element.
    pub fn node_for_basis(&self, basis_idx: usize) -> Option<NodeIndex> {
        self.index_to_node.get(&basis_idx).copied()
    }

    /// Get basis index for a node.
    pub fn basis_for_node(&self, node: NodeIndex) -> usize {
        self.graph[node]
    }

    /// Number of nodes (basis elements).
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Find all cycles in the graph using DFS.
    ///
    /// Returns fundamental cycle basis (linearly independent cycles).
    /// For a graph with V vertices and E edges, cyclomatic complexity is E - V + 1.
    pub fn fundamental_cycles(&self) -> Vec<Vec<(NodeIndex, NodeIndex, i32)>> {
        let mut cycles = Vec::new();

        // Use DFS to find cycles
        // For simplicity, we find all minimal cycles
        for start_node in self.graph.node_indices() {
            let mut visited = std::collections::HashSet::new();
            let mut path = Vec::new();
            self.dfs_find_cycles(start_node, start_node, &mut visited, &mut path, &mut cycles);
        }

        cycles
    }

    /// DFS helper to find cycles.
    fn dfs_find_cycles(
        &self,
        current: NodeIndex,
        start: NodeIndex,
        visited: &mut std::collections::HashSet<NodeIndex>,
        path: &mut Vec<(NodeIndex, NodeIndex, i32)>,
        cycles: &mut Vec<Vec<(NodeIndex, NodeIndex, i32)>>,
    ) {
        visited.insert(current);

        for neighbor in self.graph.neighbors(current) {
            if neighbor == start && !path.is_empty() {
                // Found a cycle
                cycles.push(path.clone());
            } else if !visited.contains(&neighbor) {
                let edge_sign = self
                    .graph
                    .find_edge(current, neighbor)
                    .map(|e| self.graph[e])
                    .unwrap_or(1);
                path.push((current, neighbor, edge_sign));
                self.dfs_find_cycles(neighbor, start, visited, path, cycles);
                path.pop();
            }
        }
    }

    /// Check if a cycle is balanced (product of edge signs = +1).
    pub fn cycle_is_balanced(&self, cycle: &[(NodeIndex, NodeIndex, i32)]) -> bool {
        let product: i32 = cycle.iter().map(|(_, _, sign)| sign).product();
        product == 1
    }

    /// Count balanced and unbalanced cycles.
    pub fn count_balanced_cycles(&self) -> (usize, usize) {
        let cycles = self.fundamental_cycles();
        let balanced = cycles.iter().filter(|c| self.cycle_is_balanced(c)).count();
        let unbalanced = cycles.len() - balanced;
        (balanced, unbalanced)
    }

    /// All edge signs as vector.
    pub fn edge_signs(&self) -> Vec<i32> {
        self.graph.edge_weights().copied().collect()
    }

    /// Total number of edges.
    pub fn total_edges(&self) -> usize {
        self.graph.edge_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signed_graph_creation() {
        // Build for dim=4 (quaternions)
        let dim = 4;
        let graph = SignedGraph::from_psi_matrix(dim, |i, j| {
            // Dummy psi: alternating +1/-1
            if (i + j) % 2 == 0 { 1 } else { -1 }
        });

        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.dim, 4);
    }

    #[test]
    fn test_signed_graph_node_lookup() {
        let dim = 8;
        let graph = SignedGraph::from_psi_matrix(dim, |i, j| {
            if (i * j) % 2 == 0 { 1 } else { -1 }
        });

        for i in 0..dim {
            let node = graph.node_for_basis(i).unwrap();
            assert_eq!(graph.basis_for_node(node), i);
        }
    }

    #[test]
    fn test_edge_count() {
        let dim = 4;
        let graph = SignedGraph::from_psi_matrix(dim, |_i, _j| 1);

        // For dim=4, we have 4 nodes and (4*3)/2 = 6 edges (fully connected)
        assert_eq!(graph.edge_count(), 6);
    }

    #[test]
    fn test_cycle_balance_check() {
        let dim = 4;
        // Construct a simple balanced subgraph
        let graph = SignedGraph::from_psi_matrix(dim, |_i, _j| {
            // Make all positive edges
            1
        });

        let (_balanced, unbalanced) = graph.count_balanced_cycles();
        // With all positive edges, all cycles should be balanced
        assert!(unbalanced == 0);
    }

    #[test]
    fn test_all_positive_edges_balanced() {
        let dim = 4;
        let graph = SignedGraph::from_psi_matrix(dim, |_i, _j| 1);

        let (_balanced, unbalanced) = graph.count_balanced_cycles();
        assert_eq!(unbalanced, 0, "All-positive graph should have all balanced cycles");
    }
}
