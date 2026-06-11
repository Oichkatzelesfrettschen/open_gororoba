// Copyright (c) 2026 Terascale Functionalists
// SPDX-License-Identifier: GPL-2.0-or-later

//! cubecl-wgpu ZD viscosity hash for the dark-halo detector.
//!
//! The cubecl path ports `dark_halo_viscosity.wgsl`: each GPU thread writes
//! the deterministic tau value for one cell. The LBM-sensitive velocity and
//! density criteria stay on CPU-side callers; this module exposes the isolated
//! ZD proxy count used by the Vulkan parity lane.

#![cfg(feature = "cubecl")]

use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

use crate::dark_halo_vulkan::DarkHaloConfig;

#[derive(Clone, Debug, PartialEq)]
pub struct DarkHaloCubeclResult {
    pub halo_count: u32,
    pub n_cells: usize,
    pub halo_fraction: f32,
}

#[derive(Debug, thiserror::Error)]
pub enum DarkHaloCubeclError {
    #[error("cubecl adapter not available on this host")]
    AdapterUnavailable,
    #[error("grid dimensions must all be positive (got nx={nx}, ny={ny}, nz={nz})")]
    EmptyGrid { nx: usize, ny: usize, nz: usize },
    #[error("grid too large for u32 cell indexing: nx*ny*nz = {0}")]
    GridTooLarge(u64),
}

#[cube(launch_unchecked)]
pub fn dark_halo_viscosity_kernel(
    tau_out: &mut Array<f32>,
    #[comptime] nx: u32,
    #[comptime] ny: u32,
    #[comptime] seed: u32,
    #[comptime] tau_base_bits: u32,
    #[comptime] tau_amp_bits: u32,
    #[comptime] lambda_bits: u32,
) {
    let cell = ABSOLUTE_POS;
    if cell >= tau_out.len() {
        terminate!();
    }

    let cell_u32 = cell as u32;
    let x = cell_u32 % nx;
    let y = (cell_u32 / nx) % ny;
    let z = cell_u32 / (nx * ny);
    let tau_base = f32::reinterpret(tau_base_bits);
    let tau_amp = f32::reinterpret(tau_amp_bits);
    let lambda = f32::reinterpret(lambda_bits);
    let hash = pos_hash(seed, x, y, z);
    let noise = (hash & 0xffffu32) as f32 / 65_535.0f32;
    tau_out[cell] = tau_base + tau_amp * f32::sin(lambda * noise);
}

#[cube]
fn pos_hash(seed: u32, x: u32, y: u32, z: u32) -> u32 {
    let mut hash = seed ^ (x * 73_856_093u32) ^ (y * 19_349_663u32);
    hash ^= z * 83_492_791u32;
    hash = (hash >> 13u32) ^ hash;
    hash *= 0x5bd1_e995u32;
    (hash >> 15u32) ^ hash
}

pub fn is_available() -> bool {
    gororoba_gpu_cubecl::Runtime::probe()
}

pub fn compute_tau_field_cubecl(
    nx: usize,
    ny: usize,
    nz: usize,
    config: &DarkHaloConfig,
) -> Result<Vec<f32>, DarkHaloCubeclError> {
    let n_cells = validate_grid(nx, ny, nz)?;
    if !is_available() {
        return Err(DarkHaloCubeclError::AdapterUnavailable);
    }

    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);
    let tau_handle = client.empty(n_cells * std::mem::size_of::<f32>());
    let tau_readback = tau_handle.clone();
    let cube_dim = CubeDim::new_1d(256);
    let cube_count = CubeCount::new_1d(n_cells.div_ceil(256) as u32);
    let lambda = (config.k_dim as f32).ln();

    // SAFETY: launch_unchecked requires the host to prove buffer sizes.
    // tau_handle has exactly nx * ny * nz f32 elements, and the kernel exits
    // for any thread whose absolute position exceeds that length.
    unsafe {
        dark_halo_viscosity_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(tau_handle, n_cells),
            nx as u32,
            ny as u32,
            config.seed,
            config.tau_base.to_bits(),
            config.tau_amp.to_bits(),
            lambda.to_bits(),
        );
    }

    let tau_bytes = client.read_one_unchecked(tau_readback);
    let tau: &[f32] = bytemuck::cast_slice(&tau_bytes);
    Ok(tau.to_vec())
}

pub fn count_zd_proxy(tau: &[f32], tau_base: f32, tau_amp: f32, zd_threshold: f32) -> u32 {
    tau.iter()
        .map(|&value| {
            let proxy = if tau_amp > 0.0 {
                (value - tau_base) / tau_amp
            } else {
                0.0
            };
            u32::from(proxy > zd_threshold)
        })
        .sum()
}

pub fn count_zd_cubecl(
    nx: usize,
    ny: usize,
    nz: usize,
    config: &DarkHaloConfig,
) -> Result<DarkHaloCubeclResult, DarkHaloCubeclError> {
    let tau = compute_tau_field_cubecl(nx, ny, nz, config)?;
    let halo_count = count_zd_proxy(&tau, config.tau_base, config.tau_amp, config.zd_threshold);
    Ok(DarkHaloCubeclResult {
        halo_count,
        n_cells: tau.len(),
        halo_fraction: halo_count as f32 / tau.len() as f32,
    })
}

fn validate_grid(nx: usize, ny: usize, nz: usize) -> Result<usize, DarkHaloCubeclError> {
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(DarkHaloCubeclError::EmptyGrid { nx, ny, nz });
    }
    let n_cells_u64 = (nx as u64) * (ny as u64) * (nz as u64);
    if n_cells_u64 > u32::MAX as u64 {
        return Err(DarkHaloCubeclError::GridTooLarge(n_cells_u64));
    }
    Ok(n_cells_u64 as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_available_does_not_panic() {
        let _ = is_available();
    }

    #[test]
    fn rejects_empty_grid() {
        assert!(matches!(
            validate_grid(0, 1, 1),
            Err(DarkHaloCubeclError::EmptyGrid { .. })
        ));
    }

    #[test]
    fn counts_zd_proxy_like_vulkan_parity_reference() {
        let tau = [0.7, 1.0, 1.3];
        assert_eq!(count_zd_proxy(&tau, 1.0, 0.3, 0.0), 1);
        assert_eq!(count_zd_proxy(&tau, 1.0, 0.3, -2.0), 3);
        assert_eq!(count_zd_proxy(&tau, 1.0, 0.3, 2.0), 0);
    }
}
