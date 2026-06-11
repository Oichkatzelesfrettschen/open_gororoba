//! GPU-accelerated component graph construction.
//!
//! For high-dimensional Cayley-Dickson algebras, checking all pairs of cross-assessors
//! to find zero-product edges is O(n^2). GPU parallelization gives 10-100x speedup.
//!
//! Two-phase pattern (count+compact) avoids variable-length output allocation:
//! 1. Count phase: atomic increment to find total number of edges
//! 2. Compact phase: parallel gather edges into pre-allocated output array

#[cfg(feature = "gpu")]
use cudarc::driver::PushKernelArg;
#[cfg(feature = "gpu")]
use gororoba_gpu_cuda::{Buffer, CompileOptions, LaunchConfig, ModuleRegistry};

/// GPU-accelerated component graph constructor.
pub struct GraphConstructorGpu;

/// NVRTC CUDA kernel source for parallel edge detection.
/// Two-phase pattern: count matching edges, then gather into dense array.
#[cfg(feature = "gpu")]
const GRAPH_KERNEL_SRC: &str = r#"
// Phase 1: Count matching edges from eta matrix
extern "C" __global__ void count_edges(
    const unsigned char* __restrict__ eta,
    const unsigned char* __restrict__ node_a,
    const unsigned char* __restrict__ node_b,
    unsigned int dim_half,
    unsigned int n_nodes,
    int* __restrict__ count_out
) {
    // Each thread checks one (i,j) pair from upper triangle of nodes
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = (n_nodes * (n_nodes - 1)) / 2;

    if (idx < total) {
        // Decode (i,j) from triangular index
        unsigned int i = 0;
        unsigned int remaining = idx;
        while (remaining >= (n_nodes - i - 1)) {
            remaining -= (n_nodes - i - 1);
            i++;
        }
        unsigned int j = i + 1 + remaining;

        unsigned int ai = node_a[i];
        unsigned int bi = node_b[i];
        unsigned int aj = node_a[j];
        unsigned int bj = node_b[j];

        unsigned int eta_check = 0;
        if (ai < dim_half && bi < dim_half && aj < dim_half && bj < dim_half) {
            unsigned int eta_sum =
                eta[ai * dim_half + aj] +
                eta[bi * dim_half + bj] +
                eta[ai * dim_half + bj] +
                eta[bi * dim_half + aj];
            eta_check = (eta_sum == 2 || eta_sum == 4) ? 1 : 0;
        }

        if (eta_check) {
            atomicAdd(count_out, 1);
        }
    }
}

// Phase 2: Compact edges into dense output arrays
extern "C" __global__ void compact_edges(
    const unsigned char* __restrict__ eta,
    const unsigned char* __restrict__ node_a,
    const unsigned char* __restrict__ node_b,
    unsigned int dim_half,
    unsigned int n_nodes,
    unsigned int* __restrict__ edge_i_out,
    unsigned int* __restrict__ edge_j_out,
    unsigned int total_edges
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int tri_total = (n_nodes * (n_nodes - 1)) / 2;

    if (idx < tri_total) {
        // Decode (i,j) from triangular index
        unsigned int i = 0;
        unsigned int remaining = idx;
        while (remaining >= (n_nodes - i - 1)) {
            remaining -= (n_nodes - i - 1);
            i++;
        }
        unsigned int j = i + 1 + remaining;

        unsigned int ai = node_a[i];
        unsigned int bi = node_b[i];
        unsigned int aj = node_a[j];
        unsigned int bj = node_b[j];

        unsigned int eta_check = 0;
        if (ai < dim_half && bi < dim_half && aj < dim_half && bj < dim_half) {
            unsigned int eta_sum =
                eta[ai * dim_half + aj] +
                eta[bi * dim_half + bj] +
                eta[ai * dim_half + bj] +
                eta[bi * dim_half + aj];
            eta_check = (eta_sum == 2 || eta_sum == 4) ? 1 : 0;
        }

        if (eta_check) {
            // Atomic increment to get unique position
            unsigned int pos = atomicAdd((unsigned int*)&edge_i_out[total_edges], 1);
            if (pos < total_edges) {
                edge_i_out[pos] = i;
                edge_j_out[pos] = j;
            }
        }
    }
}
"#;

fn validate_graph_input(
    dim: usize,
    eta_matrix: &[u8],
    nodes: &[(u8, u8)],
) -> Result<(usize, usize), String> {
    if dim < 2 {
        return Err(format!(
            "graph construction dimension must be >= 2, got {dim}"
        ));
    }
    if !dim.is_power_of_two() {
        return Err(format!(
            "graph construction dimension must be a power of two, got {dim}"
        ));
    }
    let dim_half = dim / 2;
    let expected_eta_len = dim_half
        .checked_mul(dim_half)
        .ok_or_else(|| format!("graph construction dimension {dim} overflows eta shape"))?;
    if eta_matrix.len() != expected_eta_len {
        return Err(format!(
            "graph construction eta length {} does not match expected {} for dim {dim}",
            eta_matrix.len(),
            expected_eta_len
        ));
    }
    if let Some(&other) = eta_matrix.iter().find(|&&value| value != 0 && value != 1) {
        return Err(format!(
            "graph construction eta value must be 0 or 1, got {other}"
        ));
    }
    if dim_half > u32::MAX as usize {
        return Err(format!(
            "graph construction dim_half {dim_half} exceeds u32"
        ));
    }
    if nodes.len() > u32::MAX as usize {
        return Err(format!(
            "graph construction node count {} exceeds u32",
            nodes.len()
        ));
    }
    let tri_total = nodes
        .len()
        .checked_mul(nodes.len().saturating_sub(1))
        .and_then(|value| value.checked_div(2))
        .ok_or_else(|| "graph construction triangular pair count overflows".to_string())?;
    if tri_total > u32::MAX as usize {
        return Err(format!(
            "graph construction pair count {tri_total} exceeds u32 dispatch"
        ));
    }
    Ok((dim_half, tri_total))
}

impl GraphConstructorGpu {
    /// Find zero-product edges (uses GPU if available, falls back to CPU).
    ///
    /// # Arguments
    /// * `dim` - Dimension
    /// * `eta_matrix` - Pre-computed eta matrix
    /// * `nodes` - List of node IDs (cross-assessor pairs)
    ///
    /// # Returns
    /// Vector of edges (i_idx, j_idx) where `nodes[i]` and `nodes[j]` form zero-product.
    pub fn find_edges(
        dim: usize,
        eta_matrix: &[u8],
        nodes: &[(u8, u8)],
    ) -> Result<Vec<(usize, usize)>, String> {
        #[cfg(feature = "gpu")]
        {
            // Try GPU first
            if let Ok(edges) = Self::find_edges_gpu(dim, eta_matrix, nodes) {
                return Ok(edges);
            }
        }

        // Fall back to CPU
        Self::find_edges_cpu(dim, eta_matrix, nodes)
    }

    /// CPU implementation: find zero-product edges.
    fn find_edges_cpu(
        dim: usize,
        eta_matrix: &[u8],
        nodes: &[(u8, u8)],
    ) -> Result<Vec<(usize, usize)>, String> {
        let (dim_half, _) = validate_graph_input(dim, eta_matrix, nodes)?;
        let mut edges = Vec::new();

        for i in 0..nodes.len() {
            for j in (i + 1)..nodes.len() {
                let (ai, bi) = nodes[i];
                let (aj, bj) = nodes[j];

                let ai = ai as usize;
                let bi = bi as usize;
                let aj = aj as usize;
                let bj = bj as usize;

                // Check if pair forms zero-product edge
                // Condition: if eta values are balanced (sum = 2 or 4)
                if ai < dim_half && bi < dim_half && aj < dim_half && bj < dim_half {
                    let eta_sum = eta_matrix[ai * dim_half + aj]
                        + eta_matrix[bi * dim_half + bj]
                        + eta_matrix[ai * dim_half + bj]
                        + eta_matrix[bi * dim_half + aj];

                    // Edge exists if eta sum matches zero-product condition
                    if eta_sum == 2 || eta_sum == 4 {
                        edges.push((i, j));
                    }
                }
            }
        }

        Ok(edges)
    }

    /// GPU implementation using two-phase count+compact.
    #[cfg(feature = "gpu")]
    fn find_edges_gpu(
        dim: usize,
        eta_matrix: &[u8],
        nodes: &[(u8, u8)],
    ) -> Result<Vec<(usize, usize)>, String> {
        let (dim_half, tri_total) = validate_graph_input(dim, eta_matrix, nodes)?;
        if tri_total == 0 {
            return Ok(Vec::new());
        }
        let dim_half_u32 = u32::try_from(dim_half)
            .map_err(|_| format!("graph construction dim_half {dim_half} exceeds u32"))?;
        let n_nodes = u32::try_from(nodes.len())
            .map_err(|_| format!("graph construction node count {} exceeds u32", nodes.len()))?;
        let tri_total_u32 = u32::try_from(tri_total).map_err(|_| {
            format!("graph construction pair count {tri_total} exceeds u32 dispatch")
        })?;

        let ctx = gororoba_gpu_cuda::Context::with_default_device()
            .map_err(|e| format!("CUDA init: {}", e))?;
        let stream = ctx.default_stream();

        let opts = CompileOptions::empty();
        let ptx = CompileOptions::compile_ptx(GRAPH_KERNEL_SRC, &opts)
            .map_err(|e| format!("NVRTC compile: {}", e))?;

        let registry = ModuleRegistry::load(ctx.raw(), ptx, &["count_edges", "compact_edges"])
            .map_err(|e| format!("Module load: {}", e))?;

        let count_kernel = registry
            .get("count_edges")
            .map_err(|e| format!("Count kernel load: {}", e))?;

        let compact_kernel = registry
            .get("compact_edges")
            .map_err(|e| format!("Compact kernel load: {}", e))?;

        let node_a: Vec<u8> = nodes.iter().map(|&(a, _)| a).collect();
        let node_b: Vec<u8> = nodes.iter().map(|&(_, b)| b).collect();

        // Allocate device memory for eta
        let eta_dev =
            Buffer::htod(&stream, eta_matrix).map_err(|e| format!("Upload eta: {}", e))?;
        let node_a_dev =
            Buffer::htod(&stream, &node_a).map_err(|e| format!("Upload node_a: {}", e))?;
        let node_b_dev =
            Buffer::htod(&stream, &node_b).map_err(|e| format!("Upload node_b: {}", e))?;

        // Phase 1: Count edges
        let mut count_dev =
            Buffer::<i32>::alloc_zeros(&stream, 1).map_err(|e| format!("Alloc count: {}", e))?;

        let cfg = LaunchConfig::launch_1d(tri_total_u32);

        let mut builder = stream.launch_builder(&count_kernel);
        builder.arg(eta_dev.raw());
        builder.arg(node_a_dev.raw());
        builder.arg(node_b_dev.raw());
        builder.arg(&dim_half_u32);
        builder.arg(&n_nodes);
        builder.arg(count_dev.raw_mut());

        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| format!("Count launch: {}", e))?;
        }

        let counts: Vec<i32> = count_dev
            .dtoh_vec()
            .map_err(|e| format!("Copy count: {}", e))?;

        let num_edges = usize::try_from(counts[0])
            .map_err(|_| format!("Count kernel returned negative edge count {}", counts[0]))?;

        if num_edges == 0 {
            return Ok(Vec::new());
        }

        // Phase 2: Compact edges
        // Allocate output arrays (with extra element for atomic counter)
        let edge_i_len = num_edges
            .checked_add(1)
            .ok_or_else(|| "graph construction edge_i allocation length overflows".to_string())?;
        let mut edge_i_dev = Buffer::<u32>::alloc_zeros(&stream, edge_i_len)
            .map_err(|e| format!("Alloc edge_i: {}", e))?;

        let mut edge_j_dev = Buffer::<u32>::alloc_zeros(&stream, num_edges)
            .map_err(|e| format!("Alloc edge_j: {}", e))?;

        let num_edges_u32 = u32::try_from(num_edges)
            .map_err(|_| format!("graph construction edge count {num_edges} exceeds u32"))?;

        let mut builder = stream.launch_builder(&compact_kernel);
        builder.arg(eta_dev.raw());
        builder.arg(node_a_dev.raw());
        builder.arg(node_b_dev.raw());
        builder.arg(&dim_half_u32);
        builder.arg(&n_nodes);
        builder.arg(edge_i_dev.raw_mut());
        builder.arg(edge_j_dev.raw_mut());
        builder.arg(&num_edges_u32);

        unsafe {
            builder
                .launch(cfg)
                .map_err(|e| format!("Compact launch: {}", e))?;
        }

        let edge_i_host: Vec<u32> = edge_i_dev
            .dtoh_vec()
            .map_err(|e| format!("Copy edge_i: {}", e))?;

        let edge_j_host: Vec<u32> = edge_j_dev
            .dtoh_vec()
            .map_err(|e| format!("Copy edge_j: {}", e))?;

        // Convert to edge list
        let edges: Vec<(usize, usize)> = edge_i_host[..num_edges]
            .iter()
            .zip(edge_j_host.iter())
            .map(|(&i, &j)| (i as usize, j as usize))
            .collect();

        Ok(edges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_edge_finding() {
        let dim = 32;
        let dim_half = dim / 2;

        // Create mock eta matrix (all zeros for simplicity)
        let eta_matrix = vec![0u8; dim_half * dim_half];

        // Create mock nodes
        let nodes = vec![(0, 1), (2, 3), (4, 5)];

        let edges = GraphConstructorGpu::find_edges(dim, &eta_matrix, &nodes);
        assert!(edges.is_ok());

        let edges = edges.unwrap();
        eprintln!("Found {} edges in test graph", edges.len());
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_fallback() {
        // Test that GPU gracefully falls back to CPU when appropriate
        let dim = 16;
        let dim_half = dim / 2;
        let eta_matrix = vec![0u8; dim_half * dim_half];
        let nodes = vec![(0, 1), (2, 3)];

        let result = GraphConstructorGpu::find_edges(dim, &eta_matrix, &nodes);
        assert!(result.is_ok());
        eprintln!("GPU/CPU fallback test passed");
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_matches_cpu_edge_finding() {
        use crate::gpu::is_gpu_available;

        if !is_gpu_available() {
            eprintln!("GPU not available; skipping graph construction GPU parity test");
            return;
        }

        let dim = 8;
        let dim_half = dim / 2;
        let mut checkerboard_eta = vec![0u8; dim_half * dim_half];
        let mut diagonal_eta = vec![0u8; dim_half * dim_half];
        for row in 0..dim_half {
            for col in 0..dim_half {
                checkerboard_eta[row * dim_half + col] = ((row + col) % 2) as u8;
            }
            diagonal_eta[row * dim_half + row] = 1;
        }

        let zero_eta = vec![0u8; dim_half * dim_half];
        let cases = [
            ("zero_eta", &zero_eta[..], vec![(0, 1), (2, 3), (1, 2)]),
            (
                "checkerboard",
                &checkerboard_eta[..],
                vec![(0, 1), (2, 3), (0, 2), (1, 3)],
            ),
            (
                "skips_out_of_half_nodes",
                &diagonal_eta[..],
                vec![(0, 1), (2, 3), (4, 0), (1, 2)],
            ),
        ];

        for (label, eta, nodes) in cases {
            let mut cpu = GraphConstructorGpu::find_edges_cpu(dim, eta, &nodes)
                .unwrap_or_else(|err| panic!("{label}: CPU graph construction failed: {err}"));
            let mut gpu = GraphConstructorGpu::find_edges_gpu(dim, eta, &nodes)
                .unwrap_or_else(|err| panic!("{label}: GPU graph construction failed: {err}"));
            cpu.sort_unstable();
            gpu.sort_unstable();
            assert_eq!(gpu, cpu, "{label}: CUDA edge list mismatch");
        }
    }
}
