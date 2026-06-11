//! cubecl-wgpu classifier for dimensional APT census samples.
//!
//! Sample triples are prepared on the CPU to match the scalar and Vulkan
//! paths. The cubecl kernel classifies each sample into one class code, and
//! the host reduces those codes into the public `GpuAptResult` counters.

#![cfg(feature = "cubecl")]

use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

use super::GpuAptResult;

const INVALID_CLASS: u32 = u32::MAX;

#[cube(launch_unchecked)]
pub fn apt_classify_kernel(
    packed_nodes: &Array<u32>,
    sample_i: &Array<u32>,
    sample_j: &Array<u32>,
    sample_k: &Array<u32>,
    class_out: &mut Array<u32>,
    #[comptime] dim: u32,
) {
    let idx = ABSOLUTE_POS;
    if idx >= class_out.len() {
        terminate!();
    }

    let node_i = packed_nodes[sample_i[idx] as usize];
    let node_j = packed_nodes[sample_j[idx] as usize];
    let node_k = packed_nodes[sample_k[idx] as usize];

    let ai = node_i & 0xffffu32;
    let bi = (node_i >> 16u32) & 0xffffu32;
    let aj = node_j & 0xffffu32;
    let bj = (node_j >> 16u32) & 0xffffu32;
    let ak = node_k & 0xffffu32;
    let bk = (node_k >> 16u32) & 0xffffu32;

    if ai < dim && bi < dim && aj < dim && bj < dim && ak < dim && bk < dim {
        let eta_ij = psi(dim, ai, bj) ^ psi(dim, aj, bi);
        let eta_ik = psi(dim, ai, bk) ^ psi(dim, ak, bi);
        let eta_jk = psi(dim, aj, bk) ^ psi(dim, ak, bj);

        if eta_ij == eta_ik && eta_ik == eta_jk {
            class_out[idx] = 0u32;
        } else {
            let f0 = eta_ij ^ eta_jk;
            let f1 = eta_jk ^ eta_ik;
            class_out[idx] = (f0 << 1u32) | f1;
        }
    }
}

#[cube]
fn psi(dim: u32, i: u32, j: u32) -> u32 {
    cd_basis_mul_is_negative(dim, i, j)
}

#[cube]
fn cd_basis_mul_is_negative(dim: u32, p_input: u32, q_input: u32) -> u32 {
    let mut is_negative = 0u32;
    let mut p = p_input;
    let mut q = q_input;
    let mut half = dim / 2u32;

    while half > 0u32 {
        let p_hi = p >= half;
        let q_hi = q >= half;
        let mut next_half = half >> 1u32;

        if !p_hi && q_hi {
            let qh = q - half;
            q = p;
            p = qh;
        } else if p_hi && !q_hi {
            p -= half;
            if q != 0u32 {
                is_negative = 1u32 - is_negative;
            }
        } else if p_hi && q_hi {
            let qh = q - half;
            let ph = p - half;
            if qh == 0u32 {
                is_negative = 1u32 - is_negative;
                next_half = 0u32;
            } else {
                p = qh;
                q = ph;
            }
        }

        half = next_half;
    }

    is_negative
}

pub struct DimensionalCubeclKernel;

impl DimensionalCubeclKernel {
    pub fn is_available() -> bool {
        dimensional_cubecl_available()
    }

    pub fn compute_apt(
        dim: usize,
        nodes: &[(u8, u8)],
        n_samples: usize,
        seed: u64,
    ) -> Result<GpuAptResult, String> {
        let prepared = PreparedDimensionalInput::new(dim, nodes, n_samples, seed)?;
        if prepared.n_samples == 0 {
            return prepared.result_from_classes(&[]);
        }
        if !Self::is_available() {
            return Err("dimensional cubecl adapter unavailable".to_string());
        }

        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);

        let nodes_bytes = encode_u32_slice(&prepared.nodes)?;
        let sample_i_bytes = encode_u32_slice(&prepared.sample_i)?;
        let sample_j_bytes = encode_u32_slice(&prepared.sample_j)?;
        let sample_k_bytes = encode_u32_slice(&prepared.sample_k)?;
        let sentinel_values = vec![INVALID_CLASS; prepared.n_samples];
        let sentinel_bytes = encode_u32_slice(&sentinel_values)?;

        let nodes_handle = client.create_from_slice(&nodes_bytes);
        let sample_i_handle = client.create_from_slice(&sample_i_bytes);
        let sample_j_handle = client.create_from_slice(&sample_j_bytes);
        let sample_k_handle = client.create_from_slice(&sample_k_bytes);
        let class_handle = client.create_from_slice(&sentinel_bytes);
        let class_readback = class_handle.clone();

        let cube_dim = CubeDim::new_1d(256);
        let cube_count = CubeCount::new_1d(prepared.n_samples.div_ceil(256) as u32);

        // SAFETY: launch_unchecked requires exact buffer-size proof. The sample
        // arrays and class output all have n_samples entries. Sample generation
        // only emits indices below nodes.len(), and packed_nodes has nodes.len()
        // entries.
        unsafe {
            apt_classify_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(nodes_handle, prepared.nodes.len()),
                ArrayArg::from_raw_parts(sample_i_handle, prepared.sample_i.len()),
                ArrayArg::from_raw_parts(sample_j_handle, prepared.sample_j.len()),
                ArrayArg::from_raw_parts(sample_k_handle, prepared.sample_k.len()),
                ArrayArg::from_raw_parts(class_handle, prepared.n_samples),
                prepared.dim as u32,
            );
        }

        let class_bytes = client.read_one_unchecked(class_readback);
        let classes = decode_u32_output(&class_bytes, prepared.n_samples, "classes")?;
        prepared.result_from_classes(&classes)
    }
}

pub fn dimensional_cubecl_available() -> bool {
    gororoba_gpu_cubecl::Runtime::probe()
}

struct PreparedDimensionalInput {
    dim: usize,
    n_nodes: usize,
    n_samples: usize,
    nodes: Vec<u32>,
    sample_i: Vec<u32>,
    sample_j: Vec<u32>,
    sample_k: Vec<u32>,
}

impl PreparedDimensionalInput {
    fn new(dim: usize, nodes: &[(u8, u8)], n_samples: usize, seed: u64) -> Result<Self, String> {
        if dim < 2 {
            return Err(format!("dimensional APT dimension must be >= 2, got {dim}"));
        }
        if !dim.is_power_of_two() {
            return Err(format!(
                "dimensional APT dimension must be a power of two, got {dim}"
            ));
        }
        if dim > u16::MAX as usize {
            return Err(format!(
                "dimensional APT dimension {dim} exceeds packed node range"
            ));
        }
        if nodes.len() < 3 && n_samples != 0 {
            return Err(format!(
                "dimensional APT needs at least 3 nodes for sampling, got {}",
                nodes.len()
            ));
        }
        if nodes.len() > u32::MAX as usize {
            return Err(format!(
                "dimensional APT node count {} exceeds u32",
                nodes.len()
            ));
        }
        if n_samples > u32::MAX as usize {
            return Err(format!(
                "dimensional APT sample count {n_samples} exceeds u32 dispatch"
            ));
        }

        let packed_nodes = nodes
            .iter()
            .map(|&(a, b)| u32::from(a) | (u32::from(b) << 16))
            .collect();
        let (sample_i, sample_j, sample_k) = sample_triples(nodes.len(), n_samples, seed);

        Ok(Self {
            dim,
            n_nodes: nodes.len(),
            n_samples,
            nodes: packed_nodes,
            sample_i,
            sample_j,
            sample_k,
        })
    }

    fn result_from_classes(&self, classes: &[u32]) -> Result<GpuAptResult, String> {
        if classes.len() != self.n_samples {
            return Err(format!(
                "dimensional cubecl class count {} does not match n_samples {}",
                classes.len(),
                self.n_samples
            ));
        }

        let mut pure_count = 0usize;
        let mut mixed_count = 0usize;
        let mut fiber_00 = 0usize;
        let mut fiber_01 = 0usize;
        let mut fiber_10 = 0usize;
        let mut fiber_11 = 0usize;

        for &class in classes {
            match class {
                0 => {
                    pure_count += 1;
                    fiber_00 += 1;
                }
                1 => {
                    mixed_count += 1;
                    fiber_01 += 1;
                }
                2 => {
                    mixed_count += 1;
                    fiber_10 += 1;
                }
                3 => {
                    mixed_count += 1;
                    fiber_11 += 1;
                }
                INVALID_CLASS => {}
                other => {
                    return Err(format!(
                        "dimensional cubecl shader wrote invalid class {other}"
                    ));
                }
            }
        }

        Ok(GpuAptResult {
            dim: self.dim,
            n_nodes: self.n_nodes,
            n_samples: self.n_samples,
            pure_count,
            mixed_count,
            fiber_00,
            fiber_01,
            fiber_10,
            fiber_11,
            pure_ratio: pure_count as f64 / self.n_samples.max(1) as f64,
        })
    }
}

fn sample_triples(n_nodes: usize, n_samples: usize, seed: u64) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut rng_state = seed;
    let mut sample_i = Vec::with_capacity(n_samples);
    let mut sample_j = Vec::with_capacity(n_samples);
    let mut sample_k = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let i = (next_rng(&mut rng_state) as usize) % n_nodes;
        let mut j = (next_rng(&mut rng_state) as usize) % n_nodes;
        while j == i {
            j = (next_rng(&mut rng_state) as usize) % n_nodes;
        }
        let mut k = (next_rng(&mut rng_state) as usize) % n_nodes;
        while k == i || k == j {
            k = (next_rng(&mut rng_state) as usize) % n_nodes;
        }
        sample_i.push(i as u32);
        sample_j.push(j as u32);
        sample_k.push(k as u32);
    }

    (sample_i, sample_j, sample_k)
}

fn next_rng(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e3779b97f4a7c15);
    let z = *state ^ (*state >> 30);
    let z_mul = z.wrapping_mul(0xbf58476d1ce4e5b9);
    z_mul ^ (z_mul >> 27)
}

fn encode_u32_slice(values: &[u32]) -> Result<Vec<u8>, String> {
    let byte_len = values
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| "dimensional cubecl buffer size overflows".to_string())?;
    let mut bytes = Vec::with_capacity(byte_len);
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    Ok(bytes)
}

fn decode_u32_output(bytes: &[u8], output_len: usize, label: &str) -> Result<Vec<u32>, String> {
    let expected_bytes = output_len
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| format!("dimensional cubecl {label} length overflows bytes"))?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "dimensional cubecl {label} readback returned {} bytes, expected {expected_bytes}",
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
    fn dimensional_cubecl_available_does_not_panic() {
        let _ = DimensionalCubeclKernel::is_available();
    }

    #[test]
    fn dimensional_cubecl_prepares_cpu_samples() {
        let nodes = [(1u8, 8u8), (2, 9), (3, 10), (4, 11)];
        let prepared = PreparedDimensionalInput::new(16, &nodes, 8, 42).unwrap();
        assert_eq!(
            prepared.nodes,
            vec![0x0008_0001, 0x0009_0002, 0x000a_0003, 0x000b_0004]
        );
        assert_eq!(prepared.sample_i.len(), 8);
        assert_eq!(prepared.sample_j.len(), 8);
        assert_eq!(prepared.sample_k.len(), 8);
        for idx in 0..8 {
            assert_ne!(prepared.sample_i[idx], prepared.sample_j[idx]);
            assert_ne!(prepared.sample_i[idx], prepared.sample_k[idx]);
            assert_ne!(prepared.sample_j[idx], prepared.sample_k[idx]);
        }
    }

    #[test]
    fn dimensional_cubecl_rejects_invalid_input() {
        assert!(PreparedDimensionalInput::new(0, &[], 0, 42).is_err());
        assert!(PreparedDimensionalInput::new(3, &[], 0, 42).is_err());
        assert!(PreparedDimensionalInput::new(8, &[(1, 4), (2, 5)], 1, 42).is_err());
    }

    #[test]
    fn dimensional_cubecl_reduces_classes() {
        let nodes = [(1u8, 8u8), (2, 9), (3, 10)];
        let prepared = PreparedDimensionalInput::new(16, &nodes, 4, 42).unwrap();
        let result = prepared.result_from_classes(&[0, 1, 2, 3]).unwrap();
        assert_eq!(result.pure_count, 1);
        assert_eq!(result.mixed_count, 3);
        assert_eq!(result.fiber_00, 1);
        assert_eq!(result.fiber_01, 1);
        assert_eq!(result.fiber_10, 1);
        assert_eq!(result.fiber_11, 1);
    }
}
