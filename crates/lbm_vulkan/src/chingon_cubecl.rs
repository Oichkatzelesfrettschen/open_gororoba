//! cubecl-wgpu backend for the Chingon AVT bilinear tensor contraction.
//!
//! WHY: Closes another cell of the cubecl + Vulkan + CUDA parity matrix
//! (#136 Phase 2). Vulkan and CUDA chingon paths already exist; this
//! adds the cubecl path for portable Metal/DX12/WebGPU coverage.
//!
//! WHAT: A `#[cube]` kernel that computes the per-violation contribution
//! and writes it into a per-thread `(m_index, contribution)` output pair.
//! The host then scatters into the final `force` vector. This avoids
//! the atomic-f32 CAS loop used by the WGSL path (Vulkan supports
//! atomicCompareExchangeWeak; cubecl 0.10's atomic surface is more
//! limited and not portable across all backends).
//!
//! HOW:
//!   1. validate inputs (`v_nd.len() == dim`, `packed_avt.len() == n_violations`).
//!   2. open cubecl-wgpu device / client.
//!   3. upload `v_nd` (f32) and `packed_avt` (u32).
//!   4. allocate two per-thread outputs: `m_out: u32[n_viol]` and
//!      `contrib_out: f32[n_viol]`.
//!   5. launch the kernel: 256 threads per cube, ceil(n_viol / 256) cubes.
//!   6. read back both arrays.
//!   7. scatter-sum on host: `force[m_out[v]] += contrib_out[v]` (f64 accum).

#![cfg(feature = "cubecl")]

use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

use crate::chingon_cpu::ChingonInputs;

/// Per-violation contribution kernel: one thread per violation.
///
/// Unpacks `(m, j, i, sign)` from `packed_avt[v]`, computes the
/// contribution `alpha * v[i] * v[j] * sign * inv_n_viol`, and writes
/// `(m, contribution)` to per-thread output slots.
#[cube(launch_unchecked)]
pub fn chingon_kernel(
    packed_avt: &Array<u32>,
    v_nd: &Array<f32>,
    m_out: &mut Array<u32>,
    contrib_out: &mut Array<f32>,
    #[comptime] n_violations: u32,
    #[comptime] index_bits: u32,
    #[comptime] mask: u32,
    #[comptime] shift_j: u32,
    #[comptime] shift_i: u32,
    #[comptime] shift_s: u32,
    #[comptime] alpha_bits: u32,
    #[comptime] inv_n_bits: u32,
) {
    let _ = index_bits;
    let v_idx = ABSOLUTE_POS;
    let n_viol_usize = n_violations as usize;
    if v_idx >= n_viol_usize {
        terminate!();
    }
    let alpha = f32::reinterpret(alpha_bits);
    let inv_n_viol = f32::reinterpret(inv_n_bits);

    let packed = packed_avt[v_idx];
    let m_idx = packed & mask;
    let j_idx = (packed >> shift_j) & mask;
    let i_idx = (packed >> shift_i) & mask;
    let sign_positive = (packed >> shift_s) & 1u32;

    // Map sign in {0, 1} to {-2.0, +2.0} via affine transform to keep the
    // arithmetic path branchless (cubecl 0.10's if-else over comptime values
    // does not always produce the same generated code across backends).
    let sign_val = (sign_positive as f32) * 4.0_f32 - 2.0_f32;

    let vi = v_nd[i_idx as usize];
    let vj = v_nd[j_idx as usize];
    let contrib = alpha * vi * vj * sign_val * inv_n_viol;

    m_out[v_idx] = m_idx;
    contrib_out[v_idx] = contrib;
}

/// Probe whether a cubecl-wgpu adapter is reachable on this host.
pub fn is_available() -> bool {
    std::panic::catch_unwind(|| {
        let device = WgpuDevice::default();
        let _client = WgpuRuntime::client(&device);
    })
    .is_ok()
}

#[derive(Debug, thiserror::Error)]
pub enum CubeclChingonError {
    #[error("v_nd length {got} != dim {expected}")]
    VLengthMismatch { got: usize, expected: usize },
    #[error("index_bits {got} too small for dim {dim}; need at least ceil(log2(dim))")]
    IndexBitsTooSmall { got: u32, dim: u32 },
    #[error("cubecl adapter not available on this host")]
    AdapterUnavailable,
}

/// Compute the chingon AVT contraction via cubecl-wgpu.
///
/// Returns a fresh `force` vector of length `inputs.dim` matching the
/// CPU reference (`chingon_contract_cpu`) within single-precision
/// floating-point tolerance.
pub fn chingon_contract_cubecl(inputs: &ChingonInputs) -> Result<Vec<f32>, CubeclChingonError> {
    if inputs.v_nd.len() != inputs.dim as usize {
        return Err(CubeclChingonError::VLengthMismatch {
            got: inputs.v_nd.len(),
            expected: inputs.dim as usize,
        });
    }
    let needed_bits = if inputs.dim <= 1 {
        0
    } else {
        32 - (inputs.dim - 1).leading_zeros()
    };
    if inputs.index_bits < needed_bits {
        return Err(CubeclChingonError::IndexBitsTooSmall {
            got: inputs.index_bits,
            dim: inputs.dim,
        });
    }
    if !is_available() {
        return Err(CubeclChingonError::AdapterUnavailable);
    }

    let n_viol = inputs.packed_avt.len();
    if n_viol == 0 {
        return Ok(vec![0.0_f32; inputs.dim as usize]);
    }

    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);

    let packed_bytes: &[u8] = bytemuck::cast_slice(&inputs.packed_avt);
    let v_bytes: &[u8] = bytemuck::cast_slice(&inputs.v_nd);
    let packed_handle = client.create_from_slice(packed_bytes);
    let v_handle = client.create_from_slice(v_bytes);
    let m_handle = client.empty(n_viol * std::mem::size_of::<u32>());
    let contrib_handle = client.empty(n_viol * std::mem::size_of::<f32>());
    let m_handle_readback = m_handle.clone();
    let contrib_handle_readback = contrib_handle.clone();

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = CubeCount::new_1d(n_viol.div_ceil(256) as u32);

    // SAFETY: launch_unchecked is unsafe because cubecl cannot verify
    // buffer sizes. We honour the contract: packed_avt.len() == n_viol
    // matches the kernel's `Array<u32>` first arg; v_nd.len() == dim
    // matches; both output handles are sized to n_viol u32 / f32 entries.
    // f32 scalars are smuggled through `comptime u32` via to_bits() to
    // avoid the ScalarArg API removed from cubecl 0.10 prelude.
    let ib = inputs.index_bits;
    let mask: u32 = (1u32 << ib) - 1u32;
    unsafe {
        chingon_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(packed_handle, n_viol),
            ArrayArg::from_raw_parts(v_handle, inputs.v_nd.len()),
            ArrayArg::from_raw_parts(m_handle, n_viol),
            ArrayArg::from_raw_parts(contrib_handle, n_viol),
            n_viol as u32,
            ib,
            mask,
            ib,           // shift_j
            2u32 * ib,    // shift_i
            3u32 * ib,    // shift_s
            inputs.alpha.to_bits(),
            inputs.inv_n_viol.to_bits(),
        );
    }

    let m_bytes = client.read_one_unchecked(m_handle_readback);
    let contrib_bytes = client.read_one_unchecked(contrib_handle_readback);
    let m_arr: &[u32] = bytemuck::cast_slice(&m_bytes);
    let contrib_arr: &[f32] = bytemuck::cast_slice(&contrib_bytes);

    // Host scatter-sum with f64 accumulators to match CPU reference precision.
    let dim = inputs.dim as usize;
    let mut accum = vec![0.0_f64; dim];
    for (idx, &m) in m_arr.iter().enumerate() {
        accum[m as usize] += contrib_arr[idx] as f64;
    }
    Ok(accum.into_iter().map(|x| x as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_available_does_not_panic() {
        let _ = is_available();
    }

    #[test]
    fn rejects_short_v_nd() {
        let bad = ChingonInputs {
            v_nd: vec![1.0; 4],
            packed_avt: vec![],
            alpha: 1.0,
            inv_n_viol: 1.0,
            index_bits: 3,
            dim: 8,
        };
        match chingon_contract_cubecl(&bad) {
            Err(CubeclChingonError::VLengthMismatch { got: 4, expected: 8 }) => {}
            other => panic!("expected VLengthMismatch, got {:?}", other),
        }
    }

    #[test]
    fn rejects_too_few_index_bits() {
        let bad = ChingonInputs {
            v_nd: vec![1.0; 256],
            packed_avt: vec![],
            alpha: 1.0,
            inv_n_viol: 1.0,
            index_bits: 4, // need 8 for dim=256
            dim: 256,
        };
        match chingon_contract_cubecl(&bad) {
            Err(CubeclChingonError::IndexBitsTooSmall { got: 4, dim: 256 }) => {}
            other => panic!("expected IndexBitsTooSmall, got {:?}", other),
        }
    }

    /// Empty packed_avt returns all-zero force without touching the GPU.
    #[test]
    fn empty_violations_returns_zero_force() {
        let inputs = ChingonInputs {
            v_nd: vec![1.0; 8],
            packed_avt: vec![],
            alpha: 1.0,
            inv_n_viol: 1.0,
            index_bits: 3,
            dim: 8,
        };
        let force = chingon_contract_cubecl(&inputs).unwrap();
        assert_eq!(force, vec![0.0; 8]);
    }
}
