// Copyright (c) 2026 Terascale Functionalists
// SPDX-License-Identifier: GPL-2.0-or-later

//! cubecl-wgpu backend for the Besag-Clifford approximate PCG shuffle and
//! viscosity transform sub-kernels.
//!
//! Two GPU kernels cover the embarrassingly parallel steps of the permutation
//! test pipeline:
//!
//!   1. `pcg_shuffle_kernel`: each output cell i receives
//!      imbalance_in[pcg_hash(seed ^ batch_idx ^ i) % n_cells], which is an
//!      approximate (collision-permissive) random permutation matching the
//!      statistical quality of besag_clifford.wgsl::shuffle_imbalance.
//!
//!   2. `transform_viscosity_kernel` (re-used from transform_viscosity_cubecl):
//!      `tau[i] = 3 * nu_base * exp(lambda * (phi[i] - phi_mean)) + 0.5`.
//!
//! The regional-correlation reduction and count_extreme comparison run on CPU:
//! cubecl 0.10 does not support mutable f32 phi-nodes in the expand IR, making
//! the strided-accumulation loops non-trivial to express.

#![cfg(feature = "cubecl")]

use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

use crate::{
    transform_viscosity_cpu::ViscosityTransformInputs,
    transform_viscosity_cubecl::{CubeclViscosityError, transform_viscosity_cubecl},
};

#[derive(Debug, thiserror::Error)]
pub enum BcCubeclError {
    #[error("cubecl adapter not available on this host")]
    AdapterUnavailable,
    #[error("viscosity transform failed: {0}")]
    TransformFailed(#[from] CubeclViscosityError),
}

/// Approximate parallel PCG shuffle kernel.
///
/// Output cell `pos` receives `imbalance_in[pcg_hash(seed_xor_batch ^ pos) % n_cells]`.
/// `seed_xor_batch` = `seed ^ batch_idx` (pre-computed by the host so each batch
/// derives a distinct permutation from the same kernel specialization).
///
/// This is a direct-read shuffle (each thread reads from the original input),
/// which is statistically equivalent to the WGSL scratch-copy swap: both are
/// approximate Fisher-Yates with PCG-driven source selection.
#[cube(launch_unchecked)]
pub fn pcg_shuffle_kernel(
    imbalance_in: &Array<f32>,
    shuffled_out: &mut Array<f32>,
    #[comptime] n_cells: u32,
    #[comptime] seed_xor_batch: u32,
) {
    let pos = ABSOLUTE_POS;
    if pos >= n_cells as usize {
        terminate!();
    }
    let tid = pos as u32;
    let rng_in = seed_xor_batch ^ tid;
    // PCG hash matching pcg_hash in besag_clifford.wgsl
    let state = rng_in * 747796405_u32 + 2891336453_u32;
    let word = ((state >> ((state >> 28_u32) + 4_u32)) ^ state) * 277803737_u32;
    let hashed = (word >> 22_u32) ^ word;
    let partner = (hashed % n_cells) as usize;
    shuffled_out[pos] = imbalance_in[partner];
}

/// Probe whether a cubecl-wgpu adapter is reachable on this host.
pub fn is_available() -> bool {
    gororoba_gpu_cubecl::Runtime::probe()
}

/// Run the PCG shuffle on `imbalance` and return the shuffled field.
///
/// Each element i in the output is `imbalance[pcg_hash(seed ^ batch_idx ^ i) % n]`,
/// which is an approximate random permutation of the input values (possible
/// collisions/omissions, same as the WGSL path).
pub fn shuffle_imbalance_cubecl(
    imbalance: &[f32],
    seed: u32,
    batch_idx: u32,
) -> Result<Vec<f32>, BcCubeclError> {
    let n = imbalance.len();
    if n == 0 {
        return Ok(vec![]);
    }
    if !is_available() {
        return Err(BcCubeclError::AdapterUnavailable);
    }

    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);

    let src_bytes: &[u8] = bytemuck::cast_slice(imbalance);
    let imbalance_handle = client.create_from_slice(src_bytes);
    let shuffled_handle = client.empty(std::mem::size_of_val(imbalance));
    let shuffled_readback = shuffled_handle.clone();

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = CubeCount::new_1d(n.div_ceil(256) as u32);

    // SAFETY: imbalance_handle and shuffled_handle each cover n f32 elements.
    // n_cells and seed_xor_batch are compile-time constants baked into the shader.
    unsafe {
        pcg_shuffle_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(imbalance_handle, n),
            ArrayArg::from_raw_parts(shuffled_handle, n),
            n as u32,
            seed ^ batch_idx,
        );
    }

    let shuffled_bytes = client.read_one_unchecked(shuffled_readback);
    let shuffled: &[f32] = bytemuck::cast_slice(&shuffled_bytes);
    Ok(shuffled.to_vec())
}

/// Run the GPU shuffle and viscosity transform for one permutation batch.
///
/// Returns the per-cell tau field after: (1) PCG shuffle of `imbalance`,
/// (2) tau = 3 * nu_base * exp(lambda * (phi - phi_mean)) + 0.5.
/// Regional-correlation and count_extreme steps are left to the caller.
pub fn shuffle_and_transform_cubecl(
    imbalance: &[f32],
    nu_base: f32,
    lambda: f32,
    phi_mean: f32,
    seed: u32,
    batch_idx: u32,
) -> Result<Vec<f32>, BcCubeclError> {
    let shuffled = shuffle_imbalance_cubecl(imbalance, seed, batch_idx)?;
    let inputs = ViscosityTransformInputs {
        phi: shuffled,
        nu_base,
        lambda,
        phi_mean,
    };
    let tau = transform_viscosity_cubecl(&inputs)?;
    Ok(tau)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_available_does_not_panic() {
        let _ = is_available();
    }

    #[test]
    fn empty_input_returns_empty() {
        let result = shuffle_imbalance_cubecl(&[], 42, 0);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }
}
