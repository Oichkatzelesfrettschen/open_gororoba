//! GPU-accelerated frustration ratio computation via parallel BFS.
//!
//! The frustration ratio measures the fraction of edges that violate the coboundary
//! condition. Computing this requires BFS traversal and delta assignment.
//!
//! GPU acceleration: Parallel BFS from multiple starting vertices can be significantly
//! faster than serial BFS, especially for large components (500+ nodes).

use cudarc::driver::CudaContext;
use std::sync::Arc;

/// GPU-accelerated frustration computation.
pub struct FrustrationGpu {
    _device: Arc<CudaContext>,
}

/// Result of frustration computation
#[derive(Debug, Clone)]
pub struct FrustrationResult {
    pub total_edges: usize,
    pub total_eta0: usize,
    pub total_eta1: usize,
    pub cycle_rank: usize,
    pub frustrated_count: usize,
    pub frustration_ratio: f64,
}

impl FrustrationGpu {
    /// Create new GPU frustration computer.
    pub fn new(device: Arc<CudaContext>) -> Result<Self, String> {
        Ok(Self { _device: device })
    }

    /// Compute frustration ratio using GPU-accelerated BFS.
    ///
    /// # Arguments
    /// * `edges` - List of edges
    /// * `n_nodes` - Number of nodes in component
    /// * `eta_values` - eta value for each edge
    ///
    /// # Returns
    /// FrustrationResult with all metrics
    pub fn compute_frustration_gpu(
        &self,
        edges: &[(usize, usize)],
        n_nodes: usize,
        eta_values: &[u8],
    ) -> Result<FrustrationResult, String> {
        // For now, use CPU implementation with GPU fallback
        // Full GPU BFS would require complex kernel
        eprintln!("GPU frustration: using CPU fallback (kernel not yet implemented)");

        Self::compute_frustration_cpu(edges, n_nodes, eta_values)
    }

    /// CPU fallback: compute frustration via BFS.
    pub fn compute_frustration_cpu(
        edges: &[(usize, usize)],
        n_nodes: usize,
        eta_values: &[u8],
    ) -> Result<FrustrationResult, String> {
        use std::collections::VecDeque;

        let mut total_eta0 = 0usize;
        let mut total_eta1 = 0usize;

        for &eta in eta_values {
            if eta == 0 {
                total_eta0 += 1;
            } else {
                total_eta1 += 1;
            }
        }

        // Build adjacency list
        let mut adj: Vec<Vec<(usize, u8)>> = vec![Vec::new(); n_nodes];
        for (i, &(u, v)) in edges.iter().enumerate() {
            if u < n_nodes && v < n_nodes && i < eta_values.len() {
                let eta = eta_values[i];
                adj[u].push((v, eta));
                adj[v].push((u, eta));
            }
        }

        // BFS coboundary test
        let mut delta = vec![0u8; n_nodes];
        let mut visited = vec![false; n_nodes];
        visited[0] = true;

        let mut queue = VecDeque::new();
        queue.push_back(0usize);

        while let Some(u) = queue.pop_front() {
            for &(v, eta) in &adj[u] {
                if !visited[v] {
                    visited[v] = true;
                    delta[v] = delta[u] ^ eta;
                    queue.push_back(v);
                }
            }
        }

        // Count frustrated edges
        let mut frustrated_count = 0usize;
        for (i, &(u, v)) in edges.iter().enumerate() {
            if u < n_nodes && v < n_nodes && i < eta_values.len() {
                let expected_eta = eta_values[i];
                let actual_eta = delta[u] ^ delta[v];
                if expected_eta != actual_eta {
                    frustrated_count += 1;
                }
            }
        }

        let cycle_rank = edges.len().saturating_sub(n_nodes - 1);
        let frustration_ratio = frustrated_count as f64 / cycle_rank.max(1) as f64;

        Ok(FrustrationResult {
            total_edges: edges.len(),
            total_eta0,
            total_eta1,
            cycle_rank,
            frustrated_count,
            frustration_ratio,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_frustration_simple() {
        // Simple test: line graph (no cycles, no frustration)
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        let eta_values = vec![0, 0, 0]; // All zeros
        let n_nodes = 4;

        let result = FrustrationGpu::compute_frustration_cpu(&edges, n_nodes, &eta_values);
        assert!(result.is_ok());

        let result = result.unwrap();
        eprintln!("Test graph: {} frustrated of {} edges", result.frustrated_count, result.total_edges);
        assert_eq!(result.cycle_rank, 0, "Line graph should have no cycles");
    }

    #[test]
    fn test_cpu_frustration_triangle() {
        // Triangle graph (1 cycle)
        let edges = vec![(0, 1), (1, 2), (2, 0)];
        let eta_values = vec![0, 0, 1]; // Last edge has eta=1, causes frustration
        let n_nodes = 3;

        let result = FrustrationGpu::compute_frustration_cpu(&edges, n_nodes, &eta_values);
        assert!(result.is_ok());

        let result = result.unwrap();
        eprintln!(
            "Triangle: frustrated={}, cycle_rank={}, ratio={:.4}",
            result.frustrated_count, result.cycle_rank, result.frustration_ratio
        );
        assert!(result.frustrated_count > 0, "Triangle should have frustration");
    }
}
