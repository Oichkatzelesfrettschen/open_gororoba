//! Thesis G: spectral fingerprints for motif classification.
//!
//! `spectral_fingerprint_from_adjacency` consumes a ZD adjacency
//! matrix and returns a `MotifFingerprint` summarizing:
//!   * sorted degree sequence
//!   * adjacency-matrix eigenvalues (descending)
//!   * triangle count
//!   * diameter (BFS from each vertex)
//!   * girth (shortest cycle via BFS, 0 if acyclic)
//!   * vertex / edge counts
//!
//! Re-exported from `cd_external` via `pub use` so external paths
//! algebra_experimental::cd_external::{MotifFingerprint,
//! spectral_fingerprint_from_adjacency} remain stable.

/// Spectral fingerprint of a graph component.
#[derive(Debug, Clone)]
pub struct MotifFingerprint {
    /// Sorted degree sequence.
    pub degree_sequence: Vec<usize>,
    /// Eigenvalues of the adjacency matrix, sorted in descending order.
    pub eigenvalues: Vec<f64>,
    /// Number of triangles in the graph.
    pub triangle_count: usize,
    /// Graph diameter.
    pub diameter: usize,
    /// Graph girth (shortest cycle length, 0 if acyclic).
    pub girth: usize,
    /// Number of vertices.
    pub n_vertices: usize,
    /// Number of edges.
    pub n_edges: usize,
}

/// Compute spectral fingerprints from a ZD adjacency matrix.
///
/// For the parity-clique graph K_m union K_m:
///   spectrum = {(m-1) with multiplicity 2, (-1) with multiplicity 2(m-1)}
///
/// For the matching graph r*K_2:
///   spectrum = {+1 with multiplicity r, -1 with multiplicity r}
pub fn spectral_fingerprint_from_adjacency(adj: &[Vec<u8>]) -> MotifFingerprint {
    let n = adj.len();

    // Degree sequence
    let mut degrees: Vec<usize> = (0..n)
        .map(|i| adj[i].iter().map(|&x| x as usize).sum())
        .collect();
    degrees.sort();

    // Edge count
    let n_edges: usize = adj
        .iter()
        .flat_map(|row| row.iter())
        .map(|&x| x as usize)
        .sum::<usize>()
        / 2;

    // Triangle count: count triples (i,j,k) with all three edges present
    let mut triangles = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if adj[i][j] == 1 {
                for (k, adj_k) in adj.iter().enumerate().skip(j + 1).take(n - j - 1) {
                    if adj[i][k] == 1 && adj_k[j] == 1 {
                        triangles += 1;
                    }
                }
            }
        }
    }

    // Diameter via BFS from each vertex
    let mut diameter = 0usize;
    for start in 0..n {
        let mut dist = vec![usize::MAX; n];
        dist[start] = 0;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        while let Some(u) = queue.pop_front() {
            for v in 0..n {
                if adj[u][v] == 1 && dist[v] == usize::MAX {
                    dist[v] = dist[u] + 1;
                    queue.push_back(v);
                }
            }
        }
        let max_d = dist
            .iter()
            .filter(|&&d| d != usize::MAX)
            .copied()
            .max()
            .unwrap_or(0);
        if max_d > diameter {
            diameter = max_d;
        }
    }

    // Girth via BFS
    let mut girth = 0usize;
    for start in 0..n {
        let mut dist = vec![usize::MAX; n];
        let mut parent = vec![usize::MAX; n];
        dist[start] = 0;
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        while let Some(u) = queue.pop_front() {
            for v in 0..n {
                if adj[u][v] == 1 {
                    if dist[v] == usize::MAX {
                        dist[v] = dist[u] + 1;
                        parent[v] = u;
                        queue.push_back(v);
                    } else if parent[u] != v && parent[v] != u {
                        let cycle_len = dist[u] + dist[v] + 1;
                        if girth == 0 || cycle_len < girth {
                            girth = cycle_len;
                        }
                    }
                }
            }
        }
    }

    // Eigenvalues: build f64 matrix and compute via nalgebra
    let mat = nalgebra::DMatrix::<f64>::from_fn(n, n, |i, j| adj[i][j] as f64);
    let eigen = mat.symmetric_eigen();
    let mut eigenvalues: Vec<f64> = eigen.eigenvalues.iter().copied().collect();
    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    MotifFingerprint {
        degree_sequence: degrees,
        eigenvalues,
        triangle_count: triangles,
        diameter,
        girth,
        n_vertices: n,
        n_edges,
    }
}
