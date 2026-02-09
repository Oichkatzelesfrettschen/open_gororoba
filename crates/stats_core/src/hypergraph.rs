//! Hypergraph Metrics for Triad Interaction Networks.
//!
//! Provides topological and statistical analysis for hypergraphs
//! where edges are 3-tuples (triads). The primary metric is the
//! clustering coefficient of the projected simple graph.
//!
//! # Projection
//! A 3-uniform hypergraph is projected to a simple graph via clique
//! expansion: each hyperedge {u, v, w} becomes three edges {u,v},
//! {v,w}, {w,u}. The clustering coefficient is then computed on
//! this projected graph.

use std::collections::{HashMap, HashSet};

/// A 3-uniform hypergraph for triad interaction networks.
///
/// Vertices are represented by usize IDs. Each hyperedge is a sorted
/// triple [a, b, c] stored in canonical order for deduplication.
#[derive(Debug, Clone)]
pub struct TriadHypergraph {
    pub edges: HashSet<[usize; 3]>,
    pub vertices: HashSet<usize>,
}

impl Default for TriadHypergraph {
    fn default() -> Self {
        Self::new()
    }
}

impl TriadHypergraph {
    pub fn new() -> Self {
        Self {
            edges: HashSet::new(),
            vertices: HashSet::new(),
        }
    }

    /// Add a triad (hyperedge) with three vertex IDs.
    ///
    /// The triple is stored in sorted canonical order to avoid duplicates.
    pub fn add_triad(&mut self, k: usize, p: usize, q: usize) {
        let mut triad = [k, p, q];
        triad.sort();
        self.edges.insert(triad);
        self.vertices.insert(k);
        self.vertices.insert(p);
        self.vertices.insert(q);
    }

    /// Number of vertices in the hypergraph.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Number of hyperedges (triads) in the hypergraph.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Degree of a vertex (number of hyperedges containing it).
    pub fn degree(&self, v: usize) -> usize {
        self.edges.iter().filter(|e| e.contains(&v)).count()
    }

    /// Degree distribution as a sorted vector of (degree, count) pairs.
    pub fn degree_distribution(&self) -> Vec<(usize, usize)> {
        let mut deg_counts: HashMap<usize, usize> = HashMap::new();
        for &v in &self.vertices {
            let d = self.degree(v);
            *deg_counts.entry(d).or_default() += 1;
        }
        let mut dist: Vec<_> = deg_counts.into_iter().collect();
        dist.sort();
        dist
    }

    /// Compute clustering coefficient of the projected simple graph.
    ///
    /// The hypergraph is projected to a simple graph via clique expansion
    /// (each triad becomes a triangle). The clustering coefficient is:
    ///   C = (number of triangles found) / (number of connected triplets)
    /// where each vertex contributes deg*(deg-1)/2 connected triplets.
    pub fn clustering_coefficient(&self) -> f64 {
        let adj = self.project_to_graph();
        let mut triangles = 0usize;
        let mut triplets = 0usize;

        for v in &self.vertices {
            if let Some(neighbors) = adj.get(v) {
                let deg = neighbors.len();
                if deg < 2 {
                    continue;
                }

                triplets += deg * (deg - 1) / 2;

                let neighbors_vec: Vec<_> = neighbors.iter().collect();
                for i in 0..deg {
                    for j in (i + 1)..deg {
                        let u = neighbors_vec[i];
                        let w = neighbors_vec[j];
                        if adj.get(u).is_some_and(|n| n.contains(w)) {
                            triangles += 1;
                        }
                    }
                }
            }
        }

        if triplets == 0 {
            0.0
        } else {
            // Each triangle is counted once per vertex (3 times total),
            // and each connected triplet that forms a triangle is counted
            // once per vertex. The ratio gives the clustering coefficient.
            (triangles as f64) / (triplets as f64)
        }
    }

    /// Project hypergraph to simple graph via clique expansion.
    ///
    /// Each hyperedge {u, v, w} becomes three undirected edges
    /// {u,v}, {v,w}, {w,u} in the projected graph.
    fn project_to_graph(&self) -> HashMap<usize, HashSet<usize>> {
        let mut adj: HashMap<usize, HashSet<usize>> = HashMap::new();
        for &[u, v, w] in &self.edges {
            for &(a, b) in &[(u, v), (v, w), (w, u)] {
                adj.entry(a).or_default().insert(b);
                adj.entry(b).or_default().insert(a);
            }
        }
        adj
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_hypergraph() {
        let hg = TriadHypergraph::new();
        assert_eq!(hg.vertex_count(), 0);
        assert_eq!(hg.edge_count(), 0);
        assert_eq!(hg.clustering_coefficient(), 0.0);
    }

    #[test]
    fn test_single_triad() {
        let mut hg = TriadHypergraph::new();
        hg.add_triad(0, 1, 2);
        assert_eq!(hg.vertex_count(), 3);
        assert_eq!(hg.edge_count(), 1);

        // A single triad projects to a triangle K3.
        // Each vertex has degree 2, triplets per vertex = 1.
        // The triangle {0,1,2} is closed, so clustering = 1.0.
        assert!((hg.clustering_coefficient() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_canonical_ordering() {
        let mut hg = TriadHypergraph::new();
        // Adding same triad in different orders should not duplicate
        hg.add_triad(2, 0, 1);
        hg.add_triad(1, 2, 0);
        hg.add_triad(0, 1, 2);
        assert_eq!(hg.edge_count(), 1);
    }

    #[test]
    fn test_two_disjoint_triads() {
        let mut hg = TriadHypergraph::new();
        hg.add_triad(0, 1, 2);
        hg.add_triad(3, 4, 5);
        assert_eq!(hg.vertex_count(), 6);
        assert_eq!(hg.edge_count(), 2);
        // Two disjoint triangles: each vertex has degree 2, triplet = 1,
        // triangle = 1, so C = 1.0 (locally each is a complete triangle)
        assert!((hg.clustering_coefficient() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_shared_edge_triads() {
        let mut hg = TriadHypergraph::new();
        // Two triads sharing edge {0,1}: {0,1,2} and {0,1,3}
        hg.add_triad(0, 1, 2);
        hg.add_triad(0, 1, 3);
        assert_eq!(hg.vertex_count(), 4);
        assert_eq!(hg.edge_count(), 2);

        // Projected graph: 0-1, 0-2, 1-2, 0-3, 1-3
        // Missing edge: 2-3
        // Vertex 0: neighbors {1,2,3}, deg=3, triplets=3, triangles: {1,2}=yes, {1,3}=yes, {2,3}=no => 2
        // Vertex 1: neighbors {0,2,3}, deg=3, triplets=3, triangles: {0,2}=yes, {0,3}=yes, {2,3}=no => 2
        // Vertex 2: neighbors {0,1}, deg=2, triplets=1, triangles: {0,1}=yes => 1
        // Vertex 3: neighbors {0,1}, deg=2, triplets=1, triangles: {0,1}=yes => 1
        // Total: 6/8 = 0.75
        let c = hg.clustering_coefficient();
        assert!((c - 0.75).abs() < 1e-14, "Expected 0.75, got {}", c);
    }

    #[test]
    fn test_degree_distribution() {
        let mut hg = TriadHypergraph::new();
        hg.add_triad(0, 1, 2);
        hg.add_triad(0, 1, 3);
        // Vertex 0 and 1 each appear in 2 triads, vertices 2 and 3 in 1 each
        let dist = hg.degree_distribution();
        assert_eq!(dist, vec![(1, 2), (2, 2)]);
    }

    #[test]
    fn test_degree_single_vertex() {
        let mut hg = TriadHypergraph::new();
        hg.add_triad(0, 1, 2);
        assert_eq!(hg.degree(0), 1);
        assert_eq!(hg.degree(1), 1);
        assert_eq!(hg.degree(99), 0); // non-existent vertex
    }

    #[test]
    fn test_k4_tetrahedron() {
        let mut hg = TriadHypergraph::new();
        // All 4 faces of a tetrahedron (K4 complete hypergraph on 4 vertices)
        hg.add_triad(0, 1, 2);
        hg.add_triad(0, 1, 3);
        hg.add_triad(0, 2, 3);
        hg.add_triad(1, 2, 3);
        assert_eq!(hg.vertex_count(), 4);
        assert_eq!(hg.edge_count(), 4);

        // Projected graph is K4 (complete graph on 4 vertices).
        // Clustering coefficient of K4 = 1.0
        assert!((hg.clustering_coefficient() - 1.0).abs() < 1e-14);
    }
}
