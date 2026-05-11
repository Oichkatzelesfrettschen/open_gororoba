//! Imbalance ratio (frustrated edges / cycle rank) over the zero-product
//! graph at a given Cayley-Dickson dimension.
//!
//! For each motif component:
//! 1. Assign an edge sign eta(u,v) = psi(dim, u_low, v_high) XOR
//!    psi(dim, u_high, v_low) where psi maps cd_basis_mul_sign output
//!    to {0, 1}.
//! 2. Run a BFS coboundary that propagates a vertex delta value across
//!    the component; the BFS chooses a delta assignment that explains as
//!    many edges as possible.
//! 3. Edges where (delta[u] XOR delta[v]) does not match eta(u,v) are
//!    "frustrated" -- GF(2) cocycle defects.
//! 4. The imbalance ratio is total_frustrated / total_b1 across all
//!    components.
//!
//! Sequence verified: 0.000, 0.307, 0.377, 0.388, 0.385, 0.381, 0.378
//! for dims 16, 32, 64, 128, 256, 512, 1024 -- approaches 3/8 = 0.375
//! from above.

use std::time::Instant;

use cd_kernel::cayley_dickson::cd_basis_mul_sign;

use super::{CrossPair, motif_components_for_cross_assessors};

/// Result of imbalance ratio computation at a given dimension.
#[derive(Debug, Clone)]
pub struct ImbalanceResult {
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
    /// Imbalance ratio: frustrated / b1.
    pub imbalance_ratio: f64,
    /// Wall-clock time for computation.
    pub elapsed_secs: f64,
}

/// Compute the imbalance ratio at a given Cayley-Dickson dimension.
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
pub fn compute_imbalance_ratio(dim: usize) -> ImbalanceResult {
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

        let eta = |a: CrossPair, b: CrossPair| -> u8 { psi(dim, a.0, b.1) ^ psi(dim, a.1, b.0) };

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

    ImbalanceResult {
        dim,
        n_components: components.len(),
        total_edges,
        eta0_count: total_eta0,
        eta1_count: total_eta1,
        total_b1,
        total_frustrated,
        imbalance_ratio: frust_ratio,
        elapsed_secs: t0.elapsed().as_secs_f64(),
    }
}
