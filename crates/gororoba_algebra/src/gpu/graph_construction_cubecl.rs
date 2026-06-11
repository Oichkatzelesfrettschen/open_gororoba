//! cubecl-wgpu pair-validation launcher for component graph construction.
//!
//! The cubecl path checks each upper-triangle node pair in parallel. It writes
//! `(i, j)` for edge pairs and `u32::MAX` sentinels for non-edges, then the
//! host filters and sorts the compact edge list.

#![cfg(feature = "cubecl")]

use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

const NO_EDGE: u32 = u32::MAX;

#[cube(launch_unchecked)]
pub fn graph_edge_flags_kernel(
    eta_matrix: &Array<u32>,
    packed_nodes: &Array<u32>,
    edge_i_out: &mut Array<u32>,
    edge_j_out: &mut Array<u32>,
    #[comptime] dim_half: u32,
    #[comptime] n_nodes: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= edge_i_out.len() {
        terminate!();
    }

    let n_nodes_u = n_nodes as usize;
    let mut i = 0usize;
    let mut remaining = idx;
    loop {
        let row_len = n_nodes_u - i - 1usize;
        if remaining < row_len {
            break;
        }
        remaining -= row_len;
        i += 1usize;
    }
    let j = i + 1usize + remaining;

    let node_a = packed_nodes[i];
    let node_b = packed_nodes[j];
    let ai = node_a & 0xffffu32;
    let bi = (node_a >> 16u32) & 0xffffu32;
    let aj = node_b & 0xffffu32;
    let bj = (node_b >> 16u32) & 0xffffu32;

    if ai < dim_half && bi < dim_half && aj < dim_half && bj < dim_half {
        let dim_half_u = dim_half as usize;
        let eta_sum = eta_matrix[(ai as usize) * dim_half_u + (aj as usize)]
            + eta_matrix[(bi as usize) * dim_half_u + (bj as usize)]
            + eta_matrix[(ai as usize) * dim_half_u + (bj as usize)]
            + eta_matrix[(bi as usize) * dim_half_u + (aj as usize)];
        if eta_sum == 2u32 || eta_sum == 4u32 {
            edge_i_out[idx] = i as u32;
            edge_j_out[idx] = j as u32;
        }
    }
}

pub struct GraphConstructionCubeclKernel;

impl GraphConstructionCubeclKernel {
    pub fn is_available() -> bool {
        graph_construction_cubecl_available()
    }

    pub fn find_edges(
        dim: usize,
        eta_matrix: &[u8],
        nodes: &[(u8, u8)],
    ) -> Result<Vec<(usize, usize)>, String> {
        let prepared = PreparedGraphInput::new(dim, eta_matrix, nodes)?;
        if prepared.tri_total == 0 {
            return Ok(Vec::new());
        }
        if !Self::is_available() {
            return Err("graph construction cubecl adapter unavailable".to_string());
        }

        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);

        let eta_bytes = encode_u32_slice(&prepared.eta)?;
        let nodes_bytes = encode_u32_slice(&prepared.nodes)?;
        let sentinel_values = vec![NO_EDGE; prepared.tri_total];
        let sentinel_bytes = encode_u32_slice(&sentinel_values)?;
        let eta_handle = client.create_from_slice(&eta_bytes);
        let nodes_handle = client.create_from_slice(&nodes_bytes);
        let edge_i_handle = client.create_from_slice(&sentinel_bytes);
        let edge_j_handle = client.create_from_slice(&sentinel_bytes);
        let edge_i_readback = edge_i_handle.clone();
        let edge_j_readback = edge_j_handle.clone();

        let cube_dim = CubeDim::new_1d(256);
        let cube_count = CubeCount::new_1d(prepared.tri_total.div_ceil(256) as u32);

        // SAFETY: launch_unchecked requires exact buffer-size proof. The eta
        // buffer has `(dim / 2)^2` u32 entries, packed_nodes has nodes.len()
        // entries, and both output arrays have tri_total entries. The kernel's
        // upper-triangle decoder only emits node indices below nodes.len().
        unsafe {
            graph_edge_flags_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(eta_handle, prepared.eta.len()),
                ArrayArg::from_raw_parts(nodes_handle, prepared.nodes.len()),
                ArrayArg::from_raw_parts(edge_i_handle, prepared.tri_total),
                ArrayArg::from_raw_parts(edge_j_handle, prepared.tri_total),
                prepared.dim_half,
                prepared.n_nodes,
            );
        }

        let edge_i_bytes = client.read_one_unchecked(edge_i_readback);
        let edge_j_bytes = client.read_one_unchecked(edge_j_readback);
        let edge_i = decode_u32_output(&edge_i_bytes, prepared.tri_total, "edge_i")?;
        let edge_j = decode_u32_output(&edge_j_bytes, prepared.tri_total, "edge_j")?;
        let mut edges = Vec::new();
        for (&i, &j) in edge_i.iter().zip(edge_j.iter()) {
            match (i, j) {
                (NO_EDGE, NO_EDGE) => {}
                (NO_EDGE, _) | (_, NO_EDGE) => {
                    return Err(format!(
                        "graph construction cubecl wrote mismatched edge sentinel ({i}, {j})"
                    ));
                }
                _ => edges.push((i as usize, j as usize)),
            }
        }
        edges.sort_unstable();
        Ok(edges)
    }
}

pub fn graph_construction_cubecl_available() -> bool {
    gororoba_gpu_cubecl::Runtime::probe()
}

struct PreparedGraphInput {
    dim_half: u32,
    n_nodes: u32,
    eta: Vec<u32>,
    nodes: Vec<u32>,
    tri_total: usize,
}

impl PreparedGraphInput {
    fn new(dim: usize, eta_matrix: &[u8], nodes: &[(u8, u8)]) -> Result<Self, String> {
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

        let mut eta = Vec::with_capacity(eta_matrix.len());
        for &value in eta_matrix {
            match value {
                0 => eta.push(0),
                1 => eta.push(1),
                other => {
                    return Err(format!(
                        "graph construction eta value must be 0 or 1, got {other}"
                    ));
                }
            }
        }
        let packed_nodes = nodes
            .iter()
            .map(|&(a, b)| u32::from(a) | (u32::from(b) << 16))
            .collect();

        Ok(Self {
            dim_half: dim_half as u32,
            n_nodes: nodes.len() as u32,
            eta,
            nodes: packed_nodes,
            tri_total,
        })
    }
}

fn encode_u32_slice(values: &[u32]) -> Result<Vec<u8>, String> {
    let byte_len = values
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| "graph construction cubecl buffer size overflows".to_string())?;
    let mut bytes = Vec::with_capacity(byte_len);
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    Ok(bytes)
}

fn decode_u32_output(bytes: &[u8], output_len: usize, label: &str) -> Result<Vec<u32>, String> {
    let expected_bytes = output_len
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| format!("graph construction cubecl {label} length overflows bytes"))?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "graph construction cubecl {label} readback returned {} bytes, expected {expected_bytes}",
            bytes.len()
        ));
    }

    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_cubecl_available_does_not_panic() {
        let _ = GraphConstructionCubeclKernel::is_available();
    }

    #[test]
    fn graph_cubecl_prepares_inputs() {
        let eta = vec![0u8; 16];
        let nodes = vec![(0u8, 1u8), (2, 3), (4, 5)];
        let prepared = PreparedGraphInput::new(8, &eta, &nodes).unwrap();
        assert_eq!(prepared.nodes, vec![0x0001_0000, 0x0003_0002, 0x0005_0004]);
        assert_eq!(prepared.dim_half, 4);
        assert_eq!(prepared.n_nodes, 3);
        assert_eq!(prepared.tri_total, 3);
    }

    #[test]
    fn graph_cubecl_rejects_invalid_input() {
        assert!(PreparedGraphInput::new(0, &[], &[]).is_err());
        assert!(PreparedGraphInput::new(3, &[0; 1], &[]).is_err());
        assert!(PreparedGraphInput::new(8, &[0; 15], &[]).is_err());
        assert!(PreparedGraphInput::new(8, &[2; 16], &[]).is_err());
    }

    #[test]
    fn graph_cubecl_encodes_and_decodes_u32() {
        let values = [0u32, 1, NO_EDGE];
        let bytes = encode_u32_slice(&values).unwrap();
        assert_eq!(
            decode_u32_output(&bytes, values.len(), "test").unwrap(),
            values
        );
    }
}
