//! cubecl-wgpu edge-validation launcher for imbalance ratio computation.
//!
//! The serial graph-labeling step stays on the CPU. The cubecl kernel writes
//! one binary frustrated-edge flag per input edge, and the host sums those
//! flags to avoid relying on backend-specific atomic support.

#![cfg(feature = "cubecl")]

use std::collections::VecDeque;

use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

use super::ImbalanceResult;

#[cube(launch_unchecked)]
pub fn imbalance_edge_flags_kernel(
    edge_u: &Array<u32>,
    edge_v: &Array<u32>,
    eta_values: &Array<u32>,
    delta_values: &Array<u32>,
    frustrated_flags: &mut Array<u32>,
) {
    let idx = ABSOLUTE_POS;
    if idx >= eta_values.len() {
        terminate!();
    }

    let u = edge_u[idx] as usize;
    let v = edge_v[idx] as usize;
    let computed = delta_values[u] ^ delta_values[v];
    let mut flag = 0u32;
    if computed != eta_values[idx] {
        flag = 1u32;
    }
    frustrated_flags[idx] = flag;
}

pub struct ImbalanceCubeclKernel;

impl ImbalanceCubeclKernel {
    pub fn is_available() -> bool {
        imbalance_cubecl_available()
    }

    pub fn compute(
        edges: &[(usize, usize)],
        n_nodes: usize,
        eta_values: &[u8],
    ) -> Result<ImbalanceResult, String> {
        let prepared = PreparedImbalanceInput::new(edges, n_nodes, eta_values)?;
        if prepared.edge_u.is_empty() {
            return Ok(prepared.empty_result());
        }
        if !Self::is_available() {
            return Err("imbalance cubecl adapter unavailable".to_string());
        }

        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);

        let edge_u_bytes = encode_u32_slice(&prepared.edge_u)?;
        let edge_v_bytes = encode_u32_slice(&prepared.edge_v)?;
        let eta_bytes = encode_u32_slice(&prepared.eta)?;
        let delta_bytes = encode_u32_slice(&prepared.delta)?;

        let edge_u_handle = client.create_from_slice(&edge_u_bytes);
        let edge_v_handle = client.create_from_slice(&edge_v_bytes);
        let eta_handle = client.create_from_slice(&eta_bytes);
        let delta_handle = client.create_from_slice(&delta_bytes);
        let flags_handle = client.empty(prepared.edge_u.len() * std::mem::size_of::<u32>());
        let flags_handle_readback = flags_handle.clone();

        let cube_dim = CubeDim::new_1d(256);
        let cube_count = CubeCount::new_1d(prepared.edge_u.len().div_ceil(256) as u32);

        // SAFETY: launch_unchecked requires exact buffer-size proof. Each
        // input edge array and the output flag array have edges.len() u32
        // slots. `delta_values` has n_nodes entries, and input validation
        // rejects edges outside that node range.
        unsafe {
            imbalance_edge_flags_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(edge_u_handle, prepared.edge_u.len()),
                ArrayArg::from_raw_parts(edge_v_handle, prepared.edge_v.len()),
                ArrayArg::from_raw_parts(eta_handle, prepared.eta.len()),
                ArrayArg::from_raw_parts(delta_handle, prepared.delta.len()),
                ArrayArg::from_raw_parts(flags_handle, prepared.edge_u.len()),
            );
        }

        let bytes = client.read_one_unchecked(flags_handle_readback);
        let flags = decode_u32_output(&bytes, prepared.edge_u.len(), "frustrated_flags")?;
        let frustrated_count = flags.into_iter().try_fold(0usize, |acc, flag| match flag {
            0 => Ok(acc),
            1 => Ok(acc + 1),
            other => Err(format!(
                "imbalance cubecl shader wrote non-binary flag {other}"
            )),
        })?;
        Ok(prepared.result(frustrated_count))
    }
}

pub fn imbalance_cubecl_available() -> bool {
    gororoba_gpu_cubecl::Runtime::probe()
}

struct PreparedImbalanceInput {
    edge_u: Vec<u32>,
    edge_v: Vec<u32>,
    eta: Vec<u32>,
    delta: Vec<u32>,
    total_eta0: usize,
    total_eta1: usize,
    cycle_rank: usize,
}

impl PreparedImbalanceInput {
    fn new(edges: &[(usize, usize)], n_nodes: usize, eta_values: &[u8]) -> Result<Self, String> {
        if n_nodes == 0 {
            return Err("imbalance graph must have at least one node".to_string());
        }
        if edges.len() != eta_values.len() {
            return Err(format!(
                "imbalance edge count {} does not match eta count {}",
                edges.len(),
                eta_values.len()
            ));
        }
        if edges.len() > u32::MAX as usize {
            return Err(format!(
                "imbalance edge count {} exceeds u32 dispatch",
                edges.len()
            ));
        }
        if n_nodes > u32::MAX as usize {
            return Err(format!(
                "imbalance node count {n_nodes} exceeds u32 buffers"
            ));
        }

        let mut edge_u = Vec::with_capacity(edges.len());
        let mut edge_v = Vec::with_capacity(edges.len());
        for &(u, v) in edges {
            if u >= n_nodes || v >= n_nodes {
                return Err(format!(
                    "imbalance edge ({u}, {v}) exceeds node count {n_nodes}"
                ));
            }
            edge_u.push(u as u32);
            edge_v.push(v as u32);
        }

        let mut eta = Vec::with_capacity(eta_values.len());
        let mut total_eta0 = 0usize;
        let mut total_eta1 = 0usize;
        for &value in eta_values {
            match value {
                0 => {
                    eta.push(0);
                    total_eta0 += 1;
                }
                1 => {
                    eta.push(1);
                    total_eta1 += 1;
                }
                other => {
                    return Err(format!("imbalance eta value must be 0 or 1, got {other}"));
                }
            }
        }

        let delta = assign_delta(edges, n_nodes, eta_values);
        let cycle_rank = edges.len().saturating_sub(n_nodes - 1);

        Ok(Self {
            edge_u,
            edge_v,
            eta,
            delta,
            total_eta0,
            total_eta1,
            cycle_rank,
        })
    }

    fn empty_result(&self) -> ImbalanceResult {
        self.result(0)
    }

    fn result(&self, frustrated_count: usize) -> ImbalanceResult {
        ImbalanceResult {
            total_edges: self.edge_u.len(),
            total_eta0: self.total_eta0,
            total_eta1: self.total_eta1,
            cycle_rank: self.cycle_rank,
            frustrated_count,
            imbalance_ratio: frustrated_count as f64 / self.cycle_rank.max(1) as f64,
        }
    }
}

fn assign_delta(edges: &[(usize, usize)], n_nodes: usize, eta_values: &[u8]) -> Vec<u32> {
    let mut delta = vec![0u32; n_nodes];
    let mut visited = vec![false; n_nodes];
    let mut adj: Vec<Vec<(usize, u8)>> = vec![Vec::new(); n_nodes];

    for (idx, &(u, v)) in edges.iter().enumerate() {
        let eta = eta_values[idx];
        adj[u].push((v, eta));
        adj[v].push((u, eta));
    }

    visited[0] = true;
    let mut queue = VecDeque::new();
    queue.push_back(0usize);

    while let Some(u) = queue.pop_front() {
        for &(v, eta) in &adj[u] {
            if !visited[v] {
                visited[v] = true;
                delta[v] = delta[u] ^ u32::from(eta);
                queue.push_back(v);
            }
        }
    }

    delta
}

fn encode_u32_slice(values: &[u32]) -> Result<Vec<u8>, String> {
    let byte_len = values
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| "imbalance cubecl buffer size overflows".to_string())?;
    let mut bytes = Vec::with_capacity(byte_len);
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    Ok(bytes)
}

fn decode_u32_output(bytes: &[u8], output_len: usize, label: &str) -> Result<Vec<u32>, String> {
    let expected_bytes = output_len
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| format!("imbalance cubecl {label} length overflows bytes"))?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "imbalance cubecl {label} readback returned {} bytes, expected {expected_bytes}",
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
    fn imbalance_cubecl_available_does_not_panic() {
        let _ = ImbalanceCubeclKernel::is_available();
    }

    #[test]
    fn imbalance_cubecl_prepares_triangle_delta() {
        let edges = [(0usize, 1usize), (1, 2), (2, 0)];
        let eta = [0u8, 0, 1];
        let prepared = PreparedImbalanceInput::new(&edges, 3, &eta).unwrap();
        assert_eq!(prepared.edge_u, vec![0, 1, 2]);
        assert_eq!(prepared.edge_v, vec![1, 2, 0]);
        assert_eq!(prepared.eta, vec![0, 0, 1]);
        assert_eq!(prepared.delta, vec![0, 0, 1]);
        assert_eq!(prepared.cycle_rank, 1);
    }

    #[test]
    fn imbalance_cubecl_rejects_invalid_input() {
        assert!(PreparedImbalanceInput::new(&[(0, 1)], 0, &[0]).is_err());
        assert!(PreparedImbalanceInput::new(&[(0, 1)], 2, &[]).is_err());
        assert!(PreparedImbalanceInput::new(&[(0, 2)], 2, &[0]).is_err());
        assert!(PreparedImbalanceInput::new(&[(0, 1)], 2, &[2]).is_err());
    }

    #[test]
    fn imbalance_cubecl_encodes_and_decodes_u32() {
        let values = [0u32, 1, 255, u32::MAX];
        let bytes = encode_u32_slice(&values).unwrap();
        assert_eq!(
            decode_u32_output(&bytes, values.len(), "test").unwrap(),
            values
        );
    }
}
