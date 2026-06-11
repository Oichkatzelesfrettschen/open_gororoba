//! cubecl-wgpu minimax-path distances for ultrametric graph analysis.
//!
//! The bottleneck distance between two graph nodes is the minimum, over all
//! paths, of the maximum edge weight on the path. Single-linkage cophenetic
//! distances use the same minimax mechanism on the minimum spanning tree.
//! This module keeps a CPU reference and a cubecl row kernel over a dense
//! adjacency matrix so parity has a compact, deterministic surface.

#![cfg(feature = "cubecl")]

use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

pub const INF_DISTANCE: f32 = 1.0e30;

#[cube(launch_unchecked)]
pub fn minimax_rows_kernel(
    adjacency: &Array<f32>,
    distances: &mut Array<f32>,
    visited: &mut Array<u32>,
    #[comptime] n_nodes: u32,
) {
    let source = ABSOLUTE_POS;
    if source >= n_nodes as usize {
        terminate!();
    }

    let n_nodes_usize = n_nodes as usize;
    let row_base = source * n_nodes_usize;
    let mut target = 0usize;
    while target < n_nodes_usize {
        distances[row_base + target] = adjacency[row_base + target];
        visited[row_base + target] = 0u32;
        target += 1usize;
    }
    distances[row_base + source] = 0.0f32;

    let mut step: u32 = 0u32;
    while step < n_nodes {
        let mut best_node: usize = 0usize;
        let mut best_distance: f32 = 1.0e30f32;
        let mut has_best: u32 = 0u32;
        let mut candidate = 0usize;
        while candidate < n_nodes_usize {
            let candidate_distance = distances[row_base + candidate];
            if visited[row_base + candidate] == 0u32
                && (has_best == 0u32 || candidate_distance < best_distance)
            {
                best_distance = candidate_distance;
                best_node = candidate;
                has_best = 1u32;
            }
            candidate += 1usize;
        }

        if has_best != 0u32 {
            visited[row_base + best_node] = 1u32;
            let mut neighbor = 0usize;
            while neighbor < n_nodes_usize {
                let edge_weight = adjacency[best_node * n_nodes_usize + neighbor];
                let mut path_weight = best_distance;
                if edge_weight > path_weight {
                    path_weight = edge_weight;
                }
                if path_weight < distances[row_base + neighbor] {
                    distances[row_base + neighbor] = path_weight;
                }
                neighbor += 1usize;
            }
        }

        step += 1u32;
    }
}

pub struct UltrametricCubeclKernel;

impl UltrametricCubeclKernel {
    pub fn is_available() -> bool {
        ultrametric_cubecl_available()
    }

    pub fn minimax_distance_matrix(adjacency: &[f32], n_nodes: usize) -> Result<Vec<f32>, String> {
        let prepared = PreparedMinimaxInput::new(adjacency, n_nodes)?;
        if prepared.n_nodes == 0 {
            return Ok(Vec::new());
        }
        if !Self::is_available() {
            return Err("ultrametric cubecl adapter unavailable".to_string());
        }

        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);

        let adjacency_bytes = encode_f32_slice(prepared.adjacency)?;
        let adjacency_handle = client.create_from_slice(&adjacency_bytes);
        let distances_handle = client.empty(prepared.matrix_len * std::mem::size_of::<f32>());
        let visited_handle = client.empty(prepared.matrix_len * std::mem::size_of::<u32>());
        let distances_readback = distances_handle.clone();

        let cube_dim = CubeDim::new_1d(256);
        let cube_count = CubeCount::new_1d(prepared.n_nodes.div_ceil(256));

        // SAFETY: launch_unchecked requires exact buffer-size proof. All three
        // dense matrices have n_nodes * n_nodes entries, and each source-row
        // thread only reads or writes its own output and visited row.
        unsafe {
            minimax_rows_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(adjacency_handle, prepared.matrix_len),
                ArrayArg::from_raw_parts(distances_handle, prepared.matrix_len),
                ArrayArg::from_raw_parts(visited_handle, prepared.matrix_len),
                prepared.n_nodes,
            );
        }

        let bytes = client.read_one_unchecked(distances_readback);
        decode_f32_output(&bytes, prepared.matrix_len, "minimax_distances")
    }
}

pub fn ultrametric_cubecl_available() -> bool {
    gororoba_gpu_cubecl::Runtime::probe()
}

pub fn minimax_distance_matrix_cpu(adjacency: &[f32], n_nodes: usize) -> Result<Vec<f32>, String> {
    let prepared = PreparedMinimaxInput::new(adjacency, n_nodes)?;
    let n = prepared.n_nodes as usize;
    let mut output = vec![INF_DISTANCE; prepared.matrix_len];

    for source in 0..n {
        let row_base = source * n;
        let mut distances = prepared.adjacency[row_base..row_base + n].to_vec();
        let mut visited = vec![false; n];
        distances[source] = 0.0;

        for _ in 0..n {
            let mut best_node = None;
            let mut best_distance = INF_DISTANCE;
            for candidate in 0..n {
                if !visited[candidate] && distances[candidate] < best_distance {
                    best_distance = distances[candidate];
                    best_node = Some(candidate);
                }
            }

            let Some(node) = best_node else {
                break;
            };
            visited[node] = true;

            for (neighbor, distance) in distances.iter_mut().enumerate().take(n) {
                let path_weight = best_distance.max(prepared.adjacency[node * n + neighbor]);
                if path_weight < *distance {
                    *distance = path_weight;
                }
            }
        }

        output[row_base..row_base + n].copy_from_slice(&distances);
    }

    Ok(output)
}

struct PreparedMinimaxInput<'a> {
    adjacency: &'a [f32],
    n_nodes: u32,
    matrix_len: usize,
}

impl<'a> PreparedMinimaxInput<'a> {
    fn new(adjacency: &'a [f32], n_nodes: usize) -> Result<Self, String> {
        if n_nodes > u32::MAX as usize {
            return Err(format!(
                "ultrametric minimax node count {n_nodes} exceeds u32"
            ));
        }
        let matrix_len = n_nodes
            .checked_mul(n_nodes)
            .ok_or_else(|| format!("ultrametric minimax node count {n_nodes} overflows matrix"))?;
        if adjacency.len() != matrix_len {
            return Err(format!(
                "ultrametric minimax adjacency length {} does not match {matrix_len}",
                adjacency.len()
            ));
        }
        if matrix_len > u32::MAX as usize {
            return Err(format!(
                "ultrametric minimax matrix length {matrix_len} exceeds u32 dispatch"
            ));
        }
        for (index, &weight) in adjacency.iter().enumerate() {
            if weight.is_nan() || weight.is_sign_negative() {
                return Err(format!(
                    "ultrametric minimax adjacency weight at {index} must be nonnegative and not NaN"
                ));
            }
            if weight > INF_DISTANCE {
                return Err(format!(
                    "ultrametric minimax adjacency weight at {index} exceeds INF_DISTANCE"
                ));
            }
        }

        Ok(Self {
            adjacency,
            n_nodes: n_nodes as u32,
            matrix_len,
        })
    }
}

fn encode_f32_slice(values: &[f32]) -> Result<Vec<u8>, String> {
    let byte_len = values
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "ultrametric cubecl buffer size overflows".to_string())?;
    let mut bytes = Vec::with_capacity(byte_len);
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    Ok(bytes)
}

fn decode_f32_output(bytes: &[u8], output_len: usize, label: &str) -> Result<Vec<f32>, String> {
    let expected_bytes = output_len
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("ultrametric cubecl {label} length overflows bytes"))?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "ultrametric cubecl {label} readback returned {} bytes, expected {expected_bytes}",
            bytes.len()
        ));
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ultrametric_cubecl_available_does_not_panic() {
        let _ = UltrametricCubeclKernel::is_available();
    }

    #[test]
    fn minimax_cpu_prefers_lower_bottleneck_path() {
        let adjacency = vec![0.0, 9.0, 2.0, 9.0, 0.0, 3.0, 2.0, 3.0, 0.0];
        let distances = minimax_distance_matrix_cpu(&adjacency, 3).unwrap();
        assert_eq!(distances[1], 3.0);
        assert_eq!(distances[3], 3.0);
    }

    #[test]
    fn minimax_rejects_invalid_input() {
        assert!(PreparedMinimaxInput::new(&[0.0, 1.0], 2).is_err());
        assert!(PreparedMinimaxInput::new(&[f32::NAN], 1).is_err());
        assert!(PreparedMinimaxInput::new(&[-1.0], 1).is_err());
    }

    #[test]
    fn minimax_decodes_f32_output() {
        let bytes = [0.0f32, 1.5, INF_DISTANCE]
            .into_iter()
            .flat_map(f32::to_ne_bytes)
            .collect::<Vec<_>>();
        assert_eq!(
            decode_f32_output(&bytes, 3, "distances").unwrap(),
            vec![0.0, 1.5, INF_DISTANCE]
        );
    }
}
