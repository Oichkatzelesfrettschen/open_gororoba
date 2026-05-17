//! D3Q19 lattice Boltzmann method, cubecl-wgpu compute backend.
//!
//! Mirrors the WGSL kernel in `shaders/lbm_d3q19.wgsl` (Vulkan path)
//! and the CPU reference `lbm_3d::LbmSolver3D` (BGK + periodic). The
//! cubecl `#[cube(launch_unchecked)]` macro compiles the kernel
//! through cubecl-wgpu's runtime so the same Rust source runs on
//! Vulkan, Metal, DX12, and WebGPU without per-backend duplication.
//!
//! Buffer layout (SoA): `f[i * n_cells + cell]` where
//! `cell = z*nx*ny + y*nx + x` and `i` runs 0..19 over D3Q19 channels.
//! Ping-pong between two device handles each timestep.
//!
//! cubecl 0.10 pitfalls honoured (per
//! `reference_cubecl_010_naming_pitfalls`):
//! - `ABSOLUTE_POS` and `Array::len` return `NativeExpand<usize>`;
//!   convert comptime `u32` params to `usize` at the top so the
//!   kernel body stays single-width (cubecl's expand IR rejects
//!   mixed u32/usize arithmetic).
//! - `CubeDim::new_3d` is the explicit `(x, y, z)` constructor;
//!   `CubeDim::new` is an auto-config helper we do not want here.
//! - `ArrayArg::from_raw_parts` is not generic; takes (handle, len).
//! - f32 scalars are smuggled through `#[comptime] u32` via
//!   `to_bits()` (`ScalarArg` was removed from the cubecl 0.10
//!   prelude).

#![cfg(feature = "cubecl")]

use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

const D3Q19_CHANNELS: u32 = 19;
const THREADS_PER_CUBE: u32 = 64;

/// One BGK + periodic-stream timestep in the PUSH scheme. Each
/// thread owns one cell, reads its 19 own-cell distribution values,
/// computes (rho, ux, uy, uz), builds f_eq, applies BGK, then
/// writes the post-collision value to the periodic-neighbor cell
/// for each of the 19 directions.
///
/// For any fixed direction i, `src -> src + c_i mod (nx, ny, nz)`
/// is a bijection on cells, so the 19 scattered writes per thread
/// never collide (each (channel, dst) pair is written by exactly
/// one source thread). cubecl writes through `Array<f32>` without
/// atomics for this reason.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn lbm_d3q19_step_kernel(
    f_in: &Array<f32>,
    f_out: &mut Array<f32>,
    #[comptime] nx: u32,
    #[comptime] ny: u32,
    #[comptime] nz: u32,
    #[comptime] inv_tau_bits: u32,
) {
    // Promote every grid quantity to usize so the kernel body stays
    // single-width. cubecl's expand IR rejects mixed u32 vs usize
    // arithmetic, and ABSOLUTE_POS + Array indexing are usize.
    let nx_u = nx as usize;
    let ny_u = ny as usize;
    let nz_u = nz as usize;
    let n_cells = nx_u * ny_u * nz_u;

    let cell = ABSOLUTE_POS;
    if cell >= n_cells {
        terminate!();
    }

    // Decompose cell into (x, y, z) in usize first; promote to i32
    // for the periodic-wrap arithmetic (we need a signed type so
    // (-1) % nx works correctly).
    let xu = cell % nx_u;
    let yu = (cell / nx_u) % ny_u;
    let zu = cell / (nx_u * ny_u);
    let x = xu as i32;
    let y = yu as i32;
    let z = zu as i32;
    let nxi = nx_u as i32;
    let nyi = ny_u as i32;
    let nzi = nz_u as i32;

    // Read 19 own-cell f values.
    let f0 = f_in[cell];
    let f1 = f_in[n_cells + cell];
    let f2 = f_in[2 * n_cells + cell];
    let f3 = f_in[3 * n_cells + cell];
    let f4 = f_in[4 * n_cells + cell];
    let f5 = f_in[5 * n_cells + cell];
    let f6 = f_in[6 * n_cells + cell];
    let f7 = f_in[7 * n_cells + cell];
    let f8 = f_in[8 * n_cells + cell];
    let f9 = f_in[9 * n_cells + cell];
    let f10 = f_in[10 * n_cells + cell];
    let f11 = f_in[11 * n_cells + cell];
    let f12 = f_in[12 * n_cells + cell];
    let f13 = f_in[13 * n_cells + cell];
    let f14 = f_in[14 * n_cells + cell];
    let f15 = f_in[15 * n_cells + cell];
    let f16 = f_in[16 * n_cells + cell];
    let f17 = f_in[17 * n_cells + cell];
    let f18 = f_in[18 * n_cells + cell];

    // Macroscopic moments (rho, mx, my, mz -> ux, uy, uz).
    let rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 + f11 + f12 + f13 + f14
        + f15 + f16 + f17 + f18;
    let inv_rho = 1.0_f32 / rho;
    let mx = f1 - f2 + f7 - f8 + f9 - f10 + f11 - f12 + f13 - f14;
    let my = f3 - f4 + f7 - f8 - f9 + f10 + f15 - f16 + f17 - f18;
    let mz = f5 - f6 + f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18;
    let ux = mx * inv_rho;
    let uy = my * inv_rho;
    let uz = mz * inv_rho;
    let u_sq = ux * ux + uy * uy + uz * uz;

    // BGK collision with the standard D3Q19 equilibrium:
    //   f_eq[i] = w_i * rho * (1 + 3 c.u + 4.5 (c.u)^2 - 1.5 u^2)
    //   f_post[i] = f[i] - inv_tau * (f[i] - f_eq[i])
    let inv_tau = f32::reinterpret(inv_tau_bits);
    let base = 1.0_f32 - 1.5_f32 * u_sq;
    let w0 = 1.0_f32 / 3.0_f32;
    let w1 = 1.0_f32 / 18.0_f32;
    let w2 = 1.0_f32 / 36.0_f32;

    let eq0 = w0 * rho * base;
    let p0 = f0 - inv_tau * (f0 - eq0);

    let cu1 = ux;
    let eq1 = w1 * rho * (base + 3.0 * cu1 + 4.5 * cu1 * cu1);
    let p1 = f1 - inv_tau * (f1 - eq1);
    let cu2 = -ux;
    let eq2 = w1 * rho * (base + 3.0 * cu2 + 4.5 * cu2 * cu2);
    let p2 = f2 - inv_tau * (f2 - eq2);
    let cu3 = uy;
    let eq3 = w1 * rho * (base + 3.0 * cu3 + 4.5 * cu3 * cu3);
    let p3 = f3 - inv_tau * (f3 - eq3);
    let cu4 = -uy;
    let eq4 = w1 * rho * (base + 3.0 * cu4 + 4.5 * cu4 * cu4);
    let p4 = f4 - inv_tau * (f4 - eq4);
    let cu5 = uz;
    let eq5 = w1 * rho * (base + 3.0 * cu5 + 4.5 * cu5 * cu5);
    let p5 = f5 - inv_tau * (f5 - eq5);
    let cu6 = -uz;
    let eq6 = w1 * rho * (base + 3.0 * cu6 + 4.5 * cu6 * cu6);
    let p6 = f6 - inv_tau * (f6 - eq6);

    let cu7 = ux + uy;
    let eq7 = w2 * rho * (base + 3.0 * cu7 + 4.5 * cu7 * cu7);
    let p7 = f7 - inv_tau * (f7 - eq7);
    let cu8 = -ux - uy;
    let eq8 = w2 * rho * (base + 3.0 * cu8 + 4.5 * cu8 * cu8);
    let p8 = f8 - inv_tau * (f8 - eq8);
    let cu9 = ux - uy;
    let eq9 = w2 * rho * (base + 3.0 * cu9 + 4.5 * cu9 * cu9);
    let p9 = f9 - inv_tau * (f9 - eq9);
    let cu10 = -ux + uy;
    let eq10 = w2 * rho * (base + 3.0 * cu10 + 4.5 * cu10 * cu10);
    let p10 = f10 - inv_tau * (f10 - eq10);
    let cu11 = ux + uz;
    let eq11 = w2 * rho * (base + 3.0 * cu11 + 4.5 * cu11 * cu11);
    let p11 = f11 - inv_tau * (f11 - eq11);
    let cu12 = -ux - uz;
    let eq12 = w2 * rho * (base + 3.0 * cu12 + 4.5 * cu12 * cu12);
    let p12 = f12 - inv_tau * (f12 - eq12);
    let cu13 = ux - uz;
    let eq13 = w2 * rho * (base + 3.0 * cu13 + 4.5 * cu13 * cu13);
    let p13 = f13 - inv_tau * (f13 - eq13);
    let cu14 = -ux + uz;
    let eq14 = w2 * rho * (base + 3.0 * cu14 + 4.5 * cu14 * cu14);
    let p14 = f14 - inv_tau * (f14 - eq14);
    let cu15 = uy + uz;
    let eq15 = w2 * rho * (base + 3.0 * cu15 + 4.5 * cu15 * cu15);
    let p15 = f15 - inv_tau * (f15 - eq15);
    let cu16 = -uy - uz;
    let eq16 = w2 * rho * (base + 3.0 * cu16 + 4.5 * cu16 * cu16);
    let p16 = f16 - inv_tau * (f16 - eq16);
    let cu17 = uy - uz;
    let eq17 = w2 * rho * (base + 3.0 * cu17 + 4.5 * cu17 * cu17);
    let p17 = f17 - inv_tau * (f17 - eq17);
    let cu18 = -uy + uz;
    let eq18 = w2 * rho * (base + 3.0 * cu18 + 4.5 * cu18 * cu18);
    let p18 = f18 - inv_tau * (f18 - eq18);

    // PUSH the 19 post-collision values to periodic-neighbor cells.
    // periodic_neighbor returns a usize index in row-major
    // (z, y, x) order matching the SoA channel base.
    f_out[cell] = p0;
    f_out[n_cells + periodic_neighbor(x, y, z, 1, 0, 0, nxi, nyi, nzi, nx_u)] = p1;
    f_out[2 * n_cells + periodic_neighbor(x, y, z, -1, 0, 0, nxi, nyi, nzi, nx_u)] = p2;
    f_out[3 * n_cells + periodic_neighbor(x, y, z, 0, 1, 0, nxi, nyi, nzi, nx_u)] = p3;
    f_out[4 * n_cells + periodic_neighbor(x, y, z, 0, -1, 0, nxi, nyi, nzi, nx_u)] = p4;
    f_out[5 * n_cells + periodic_neighbor(x, y, z, 0, 0, 1, nxi, nyi, nzi, nx_u)] = p5;
    f_out[6 * n_cells + periodic_neighbor(x, y, z, 0, 0, -1, nxi, nyi, nzi, nx_u)] = p6;
    f_out[7 * n_cells + periodic_neighbor(x, y, z, 1, 1, 0, nxi, nyi, nzi, nx_u)] = p7;
    f_out[8 * n_cells + periodic_neighbor(x, y, z, -1, -1, 0, nxi, nyi, nzi, nx_u)] = p8;
    f_out[9 * n_cells + periodic_neighbor(x, y, z, 1, -1, 0, nxi, nyi, nzi, nx_u)] = p9;
    f_out[10 * n_cells + periodic_neighbor(x, y, z, -1, 1, 0, nxi, nyi, nzi, nx_u)] = p10;
    f_out[11 * n_cells + periodic_neighbor(x, y, z, 1, 0, 1, nxi, nyi, nzi, nx_u)] = p11;
    f_out[12 * n_cells + periodic_neighbor(x, y, z, -1, 0, -1, nxi, nyi, nzi, nx_u)] = p12;
    f_out[13 * n_cells + periodic_neighbor(x, y, z, 1, 0, -1, nxi, nyi, nzi, nx_u)] = p13;
    f_out[14 * n_cells + periodic_neighbor(x, y, z, -1, 0, 1, nxi, nyi, nzi, nx_u)] = p14;
    f_out[15 * n_cells + periodic_neighbor(x, y, z, 0, 1, 1, nxi, nyi, nzi, nx_u)] = p15;
    f_out[16 * n_cells + periodic_neighbor(x, y, z, 0, -1, -1, nxi, nyi, nzi, nx_u)] = p16;
    f_out[17 * n_cells + periodic_neighbor(x, y, z, 0, 1, -1, nxi, nyi, nzi, nx_u)] = p17;
    f_out[18 * n_cells + periodic_neighbor(x, y, z, 0, -1, 1, nxi, nyi, nzi, nx_u)] = p18;
}

/// Periodic neighbor index in row-major (z, y, x) order, returned
/// as a `usize` ready for SoA channel-base addressing. The caller
/// passes nx_u so the helper can return usize directly without an
/// extra cast site at every call.
#[cube]
#[allow(clippy::too_many_arguments)]
fn periodic_neighbor(
    x: i32,
    y: i32,
    z: i32,
    cx: i32,
    cy: i32,
    cz: i32,
    nx: i32,
    ny: i32,
    nz: i32,
    nx_u: usize,
) -> usize {
    let mut nx_i = x + cx;
    let mut ny_i = y + cy;
    let mut nz_i = z + cz;
    nx_i -= nx * (nx_i / nx);
    if nx_i < 0 {
        nx_i += nx;
    }
    ny_i -= ny * (ny_i / ny);
    if ny_i < 0 {
        ny_i += ny;
    }
    nz_i -= nz * (nz_i / nz);
    if nz_i < 0 {
        nz_i += nz;
    }
    let nx_area = nx_u * (ny as usize);
    (nz_i as usize) * nx_area + (ny_i as usize) * nx_u + (nx_i as usize)
}

/// Probe whether a cubecl-wgpu adapter is reachable. Thin delegate
/// to the consolidated `gororoba_gpu_cubecl::Runtime::probe()`.
pub fn is_available() -> bool {
    gororoba_gpu_cubecl::Runtime::probe()
}

#[derive(Debug, thiserror::Error)]
pub enum CubeclLbmError {
    #[error("grid dimensions must all be positive (got nx={nx}, ny={ny}, nz={nz})")]
    EmptyGrid { nx: usize, ny: usize, nz: usize },
    #[error(
        "grid too large for u32 cell indexing: nx*ny*nz = {n_cells} (max = {})",
        u32::MAX
    )]
    GridTooLarge { n_cells: u64 },
    #[error("input f slice length {got} does not match nx*ny*nz*19 = {expected}")]
    LengthMismatch { got: usize, expected: usize },
    #[error("tau must satisfy tau > 0.5 for BGK stability (got {0})")]
    UnstableTau(f32),
    #[error("cubecl adapter not available on this host")]
    AdapterUnavailable,
}

/// Run `num_steps` BGK + periodic-stream timesteps on the cubecl-wgpu
/// runtime starting from `f_init` (SoA `[channel * n_cells + cell]`,
/// length `nx*ny*nz*19`).
///
/// Returns the post-evolve `f` array in the same SoA layout, length
/// `nx*ny*nz*19`. The function is single-shot: it opens its own wgpu
/// client per call. For batched evolution, callers should plumb the
/// client through and call the kernel directly.
pub fn evolve_d3q19_cubecl(
    nx: usize,
    ny: usize,
    nz: usize,
    tau: f32,
    f_init: &[f32],
    num_steps: usize,
) -> Result<Vec<f32>, CubeclLbmError> {
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(CubeclLbmError::EmptyGrid { nx, ny, nz });
    }
    if tau.is_nan() || tau <= 0.5 {
        return Err(CubeclLbmError::UnstableTau(tau));
    }
    // u64 arithmetic to detect overflow before narrowing. The kernel
    // launch signature takes u32 grid dimensions and a u32 cube count,
    // so a usize-sized grid that wraps during `as u32` would launch
    // for a truncated cell range while the buffers still describe the
    // full domain -- a silent-corruption mode we'd rather refuse.
    let n_cells_u64 = (nx as u64) * (ny as u64) * (nz as u64);
    if n_cells_u64 > u32::MAX as u64 {
        return Err(CubeclLbmError::GridTooLarge {
            n_cells: n_cells_u64,
        });
    }
    let n_cells = nx * ny * nz;
    let expected = n_cells * D3Q19_CHANNELS as usize;
    if f_init.len() != expected {
        return Err(CubeclLbmError::LengthMismatch {
            got: f_init.len(),
            expected,
        });
    }

    // Empty-input fast path: no kernel launch required. The
    // `n_cells == 0` case is guarded by EmptyGrid above, but keep
    // the no-steps fast path here so callers that pass num_steps=0
    // can read the input back identically without an adapter probe.
    if num_steps == 0 {
        return Ok(f_init.to_vec());
    }

    if !is_available() {
        return Err(CubeclLbmError::AdapterUnavailable);
    }

    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);

    let f_bytes: &[u8] = bytemuck::cast_slice(f_init);
    let mut handle_a = client.create_from_slice(f_bytes);
    let mut handle_b = client.empty(std::mem::size_of_val(f_init));

    let cube_dim = CubeDim::new_3d(THREADS_PER_CUBE, 1, 1);
    let cube_count = CubeCount::new_1d((n_cells as u32).div_ceil(THREADS_PER_CUBE));

    let inv_tau_bits = (1.0_f32 / tau).to_bits();
    let nx_u = nx as u32;
    let ny_u = ny as u32;
    let nz_u = nz as u32;

    for _ in 0..num_steps {
        let handle_a_for_launch = handle_a.clone();
        let handle_b_for_launch = handle_b.clone();
        // SAFETY: launch_unchecked skips cubecl's bounds-check
        // trampoline. The kernel's own `if cell >= n_cells_u`
        // early-terminate covers oversubscribed workgroups, and the
        // two array lengths are exactly `n_cells * 19` f32 elements
        // matching the kernel's SoA addressing.
        unsafe {
            lbm_d3q19_step_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                cube_count.clone(),
                cube_dim,
                ArrayArg::from_raw_parts(handle_a_for_launch, f_init.len()),
                ArrayArg::from_raw_parts(handle_b_for_launch, f_init.len()),
                nx_u,
                ny_u,
                nz_u,
                inv_tau_bits,
            );
        }
        std::mem::swap(&mut handle_a, &mut handle_b);
    }

    let out_bytes = client.read_one_unchecked(handle_a);
    let out: &[f32] = bytemuck::cast_slice(&out_bytes);
    Ok(out.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_steps_returns_input_unchanged() {
        let f = vec![1.0_f32; 4 * 4 * 4 * 19];
        let out = evolve_d3q19_cubecl(4, 4, 4, 1.0, &f, 0).unwrap();
        assert_eq!(out, f);
    }

    #[test]
    fn rejects_unstable_tau() {
        let f = vec![1.0_f32; 4 * 4 * 4 * 19];
        match evolve_d3q19_cubecl(4, 4, 4, 0.4, &f, 1) {
            Err(CubeclLbmError::UnstableTau(_)) => {}
            other => panic!("expected UnstableTau, got {other:?}"),
        }
    }

    #[test]
    fn rejects_length_mismatch() {
        let bad = vec![1.0_f32; 7];
        match evolve_d3q19_cubecl(2, 2, 2, 1.0, &bad, 1) {
            Err(CubeclLbmError::LengthMismatch { got: 7, expected: 152 }) => {}
            other => panic!("expected LengthMismatch, got {other:?}"),
        }
    }

    /// Refuse grids whose total cell count would overflow u32, before
    /// any allocation. Tested by picking three dimensions whose product
    /// strictly exceeds u32::MAX without allocating the full buffer.
    #[test]
    fn rejects_grid_too_large_for_u32_indexing() {
        // 2^16 * 2^16 * 2 = 2^33 cells -- well beyond u32::MAX = 2^32 - 1.
        let nx = 1usize << 16;
        let ny = 1usize << 16;
        let nz = 2usize;
        // Empty slice so the length-mismatch arm would fire if the
        // grid-size guard let the call through. The guard must reject
        // BEFORE reading f_init.len().
        let f: Vec<f32> = Vec::new();
        match evolve_d3q19_cubecl(nx, ny, nz, 1.0, &f, 1) {
            Err(CubeclLbmError::GridTooLarge { n_cells }) => {
                assert_eq!(n_cells, (nx as u64) * (ny as u64) * (nz as u64));
            }
            other => panic!("expected GridTooLarge, got {other:?}"),
        }
    }
}
