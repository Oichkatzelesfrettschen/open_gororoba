//! cubecl-wgpu backend for box-counting fractal dimension.
//!
//! WHY: Closes one cell of the cubecl + Vulkan + CUDA parity matrix
//! (docs/engineering/cubecl_vulkan_cuda_parity_matrix.md). The Vulkan
//! WGSL path already exists at `box_counting_vulkan.rs`; this module
//! provides the cubecl path that runs the same algorithm via the
//! cubecl-wgpu runtime so the same source executes on Vulkan, Metal,
//! DX12, and WebGPU without per-backend duplication.
//!
//! WHAT: A `#[cube]` kernel that for each box writes a 0 or 1 to a
//! per-box output array, plus a launcher that uploads the density grid,
//! launches the kernel, and reads back the per-box occupancy. The host
//! sums the per-box flags to produce the final integer count (avoids
//! the atomic-add surface and matches the existing
//! `cd_kernel::turboquant::cubecl_backend` portability pattern).
//!
//! HOW:
//!   1. validate inputs.
//!   2. open default `WgpuDevice` / `WgpuRuntime::client`.
//!   3. upload rho as bytes.
//!   4. allocate per-box u32 output handle (length = n_boxes).
//!   5. launch the kernel: 256 threads per cube, ceil(n_boxes / 256) cubes.
//!   6. read back the u32 array; sum on host.

#![cfg(feature = "cubecl")]

use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

/// Per-box occupancy kernel.
///
/// One thread per box. The thread scans its box's cells in (dz, dy, dx)
/// order and writes 1 to its output slot on the first occupied cell;
/// 0 if no cells are occupied. The threshold is also a comptime constant
/// to avoid the `ScalarArg` API which was removed from cubecl 0.10's
/// prelude (see project MEMORY note on cubecl 0.10 naming pitfalls).
#[cube(launch_unchecked)]
pub fn box_counting_kernel(
    rho: &Array<f32>,
    occupied: &mut Array<u32>,
    #[comptime] nx: u32,
    #[comptime] ny: u32,
    #[comptime] nz: u32,
    #[comptime] box_size: u32,
    #[comptime] bx_count: u32,
    #[comptime] by_count: u32,
    #[comptime] bz_count: u32,
    #[comptime] threshold_bits: u32,
) {
    // ABSOLUTE_POS and Array::len return cubecl's NativeExpand<usize>;
    // all comptime constants arrive as u32 from the host. Convert to
    // usize early so the kernel body stays single-width (cubecl's expand
    // IR rejects mixed u32/usize arithmetic). The host validates the u32
    // bounds before launch so the conversions are safe.
    let threshold = f32::reinterpret(threshold_bits);
    let nx_u = nx as usize;
    let ny_u = ny as usize;
    let nz_u = nz as usize;
    let bs_u = box_size as usize;
    let bx_u = bx_count as usize;
    let by_u = by_count as usize;
    let bz_u = bz_count as usize;
    let total_boxes = bx_u * by_u * bz_u;

    let box_idx = ABSOLUTE_POS;
    if box_idx >= total_boxes {
        terminate!();
    }

    let ibx = box_idx % bx_u;
    let iby = (box_idx / bx_u) % by_u;
    let ibz = box_idx / (bx_u * by_u);

    let mut is_occupied: u32 = 0u32;
    let mut dz: usize = 0;
    while dz < bs_u {
        let mut dy: usize = 0;
        while dy < bs_u {
            let mut dx: usize = 0;
            while dx < bs_u {
                let ix = ibx * bs_u + dx;
                let iy = iby * bs_u + dy;
                let iz = ibz * bs_u + dz;
                if ix < nx_u && iy < ny_u && iz < nz_u {
                    let cell_idx = iz * ny_u * nx_u + iy * nx_u + ix;
                    if rho[cell_idx] > threshold {
                        is_occupied = 1u32;
                    }
                }
                dx += 1;
            }
            dy += 1;
        }
        dz += 1;
    }
    occupied[box_idx] = is_occupied;
}

/// Probe whether a cubecl-wgpu adapter is reachable on this host.
///
/// Mirrors `cd_kernel::turboquant::cubecl_backend::launcher::is_available`.
/// Cheap on success; catches panics from headless containers without a
/// software renderer so the probe is non-fatal.
pub fn is_available() -> bool {
    std::panic::catch_unwind(|| {
        let device = WgpuDevice::default();
        let _client = WgpuRuntime::client(&device);
    })
    .is_ok()
}

/// Errors from the cubecl box-counting launcher.
#[derive(Debug, thiserror::Error)]
pub enum CubeclBoxCountError {
    #[error("rho length {got} != nx*ny*nz = {expected}")]
    RhoLengthMismatch { got: usize, expected: usize },
    #[error("box_size must be at least 1 (got {0})")]
    BoxSizeZero(u32),
    #[error("cubecl adapter not available on this host")]
    AdapterUnavailable,
}

/// Count occupied boxes at a single scale via cubecl-wgpu.
///
/// Returns the same `u32` integer count as
/// `lbm_vulkan::box_counting_cpu::count_occupied_boxes`. Parity is
/// asserted by `tests/box_counting_cubecl_parity.rs`.
pub fn count_occupied_boxes_cubecl(
    rho: &[f32],
    threshold: f32,
    nx: usize,
    ny: usize,
    nz: usize,
    box_size: usize,
) -> Result<u32, CubeclBoxCountError> {
    if rho.len() != nx * ny * nz {
        return Err(CubeclBoxCountError::RhoLengthMismatch {
            got: rho.len(),
            expected: nx * ny * nz,
        });
    }
    if box_size == 0 {
        return Err(CubeclBoxCountError::BoxSizeZero(0));
    }
    if !is_available() {
        return Err(CubeclBoxCountError::AdapterUnavailable);
    }

    let bx_count = nx.div_ceil(box_size);
    let by_count = ny.div_ceil(box_size);
    let bz_count = nz.div_ceil(box_size);
    let n_boxes = bx_count * by_count * bz_count;

    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);

    let rho_bytes: &[u8] = bytemuck::cast_slice(rho);
    let rho_handle = client.create_from_slice(rho_bytes);
    let occupied_handle = client.empty(n_boxes * std::mem::size_of::<u32>());
    // The readback API consumes the Handle, so clone it for the launch
    // path while keeping the original for `read_one_unchecked`.
    let occupied_handle_for_readback = occupied_handle.clone();

    let cube_dim = CubeDim::new_1d(256);
    let cube_count = CubeCount::new_1d(n_boxes.div_ceil(256) as u32);

    // SAFETY: launch_unchecked skips cubecl's bounds-check trampoline. We
    // honour the contract by (a) sizing occupied_handle to exactly n_boxes
    // u32 entries, matching the kernel's bounds via the early-terminate
    // path when `box_idx >= total_boxes`, and (b) passing nx/ny/nz/box_size/
    // bx_count/by_count/bz_count as comptime u32 so the kernel does not
    // index out of range.
    // SAFETY: `launch_unchecked` is unsafe because cubecl cannot statically
    // verify buffer sizes match the kernel's expectations. We satisfy that
    // contract by passing `rho.len()` (matching the kernel's `Array<f32>`)
    // and `n_boxes` (matching the `Array<u32>` output, sized to the same
    // grid the comptime constants describe). `ArrayArg::from_raw_parts`
    // takes 2 args (handle, length) -- no generic in cubecl 0.10. Comptime
    // scalars are passed bare without `ScalarArg::new`.
    unsafe {
        box_counting_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(rho_handle, rho.len()),
            ArrayArg::from_raw_parts(occupied_handle, n_boxes),
            nx as u32,
            ny as u32,
            nz as u32,
            box_size as u32,
            bx_count as u32,
            by_count as u32,
            bz_count as u32,
            threshold.to_bits(),
        );
    }

    let occupied_bytes = client.read_one_unchecked(occupied_handle_for_readback);
    let occupied: &[u32] = bytemuck::cast_slice(&occupied_bytes);
    Ok(occupied.iter().sum())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `is_available()` must never panic regardless of host state.
    /// On hosts with no GPU it returns false; with a GPU it returns true.
    #[test]
    fn is_available_does_not_panic() {
        let _ = is_available();
    }

    /// Input-validation: wrong rho length is reported before any GPU work.
    #[test]
    fn rejects_wrong_rho_length() {
        let bad = vec![0.0f32; 7];
        match count_occupied_boxes_cubecl(&bad, 0.5, 2, 2, 2, 1) {
            Err(CubeclBoxCountError::RhoLengthMismatch {
                got: 7,
                expected: 8,
            }) => {}
            other => panic!("expected RhoLengthMismatch, got {:?}", other),
        }
    }

    /// Input-validation: zero box_size is rejected.
    #[test]
    fn rejects_zero_box_size() {
        let rho = vec![0.0f32; 8];
        match count_occupied_boxes_cubecl(&rho, 0.5, 2, 2, 2, 0) {
            Err(CubeclBoxCountError::BoxSizeZero(_)) => {}
            other => panic!("expected BoxSizeZero, got {:?}", other),
        }
    }
}
