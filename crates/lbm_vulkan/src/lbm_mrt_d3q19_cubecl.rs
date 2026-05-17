//! D3Q19 lattice Boltzmann method, MRT collision, cubecl-wgpu backend.
//!
//! Mirrors the WGSL kernel at `shaders/lbm_mrt_d3q19.wgsl` and the CPU
//! reference `lbm_3d::LbmSolver3D::new_mrt`. The collision operator is
//! d'Humieres (2002) MRT with fixed rates S_E=1.19, S_EPS=1.40, S_Q=1.20,
//! S_GHOST=1.00 and physical rate s_nu=1/tau (stress moments).
//!
//! cubecl 0.10 pitfalls honoured:
//! - ABSOLUTE_POS and Array indexing are usize; convert comptime u32
//!   params to usize at the top of the kernel body.
//! - CubeDim::new_3d is the explicit (x,y,z) constructor.
//! - ArrayArg::from_raw_parts is not generic; takes (handle, len).
//! - f32 scalars that vary at launch (inv_tau) are smuggled as
//!   `#[comptime] u32` via to_bits(); fixed mathematical constants
//!   are written as f32 literals directly.

#![cfg(feature = "cubecl")]

use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

const D3Q19_CHANNELS: u32 = 19;
const THREADS_PER_CUBE: u32 = 64;

/// One MRT + periodic-stream timestep in the PUSH scheme. Each thread owns
/// one cell, performs the d'Humieres D3Q19 MRT collision (8 phases), then
/// pushes post-collision values to the 19 periodic-neighbor cells.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn lbm_mrt_d3q19_step_kernel(
    f_in: &Array<f32>,
    f_out: &mut Array<f32>,
    #[comptime] nx: u32,
    #[comptime] ny: u32,
    #[comptime] nz: u32,
    #[comptime] inv_tau_bits: u32,
) {
    let nx_u = nx as usize;
    let ny_u = ny as usize;
    let nz_u = nz as usize;
    let n_cells = nx_u * ny_u * nz_u;

    let cell = ABSOLUTE_POS;
    if cell >= n_cells {
        terminate!();
    }

    let xu = cell % nx_u;
    let yu = (cell / nx_u) % ny_u;
    let zu = cell / (nx_u * ny_u);
    let x = xu as i32;
    let y = yu as i32;
    let z = zu as i32;
    let nxi = nx_u as i32;
    let nyi = ny_u as i32;
    let nzi = nz_u as i32;

    // Phase 1: read 19 own-cell f values.
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

    // Phase 2: macroscopic moments.
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

    // Phase 3: forward transform m = M * f (d'Humieres D3Q19 orthogonal basis).
    let m0 = rho;
    let m1 = -30.0_f32 * f0
        - 11.0_f32 * (f1 + f2 + f3 + f4 + f5 + f6)
        + 8.0_f32 * (f7 + f8 + f9 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18);
    let m2 = 12.0_f32 * f0 - 4.0_f32 * (f1 + f2 + f3 + f4 + f5 + f6)
        + (f7 + f8 + f9 + f10 + f11 + f12 + f13 + f14 + f15 + f16 + f17 + f18);
    let m3 = f1 - f2 + f7 - f8 + f9 - f10 + f11 - f12 + f13 - f14;
    let m4 = -4.0_f32 * (f1 - f2) + f7 - f8 + f9 - f10 + f11 - f12 + f13 - f14;
    let m5 = f3 - f4 + f7 - f8 - f9 + f10 + f15 - f16 + f17 - f18;
    let m6 = -4.0_f32 * (f3 - f4) + f7 - f8 - f9 + f10 + f15 - f16 + f17 - f18;
    let m7 = f5 - f6 + f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18;
    let m8 = -4.0_f32 * (f5 - f6) + f11 - f12 - f13 + f14 + f15 - f16 - f17 + f18;
    let m9 = 2.0_f32 * (f1 + f2) - (f3 + f4 + f5 + f6) + f7 + f8 + f9 + f10 + f11 + f12
        + f13 + f14
        - 2.0_f32 * (f15 + f16 + f17 + f18);
    let m10 = -2.0_f32 * (f1 + f2) + (f3 + f4 + f5 + f6) + f7 + f8 + f9 + f10 + f11 + f12
        + f13 + f14
        - 2.0_f32 * (f15 + f16 + f17 + f18);
    let m11 = (f3 + f4) - (f5 + f6) + f7 + f8 + f9 + f10 - (f11 + f12 + f13 + f14);
    let m12 = -(f3 + f4) + (f5 + f6) + f7 + f8 + f9 + f10 - (f11 + f12 + f13 + f14);
    let m13 = f7 + f8 - f9 - f10;
    let m14 = f11 + f12 - f13 - f14;
    let m15 = f15 + f16 - f17 - f18;
    let m16 = f7 - f8 - f9 + f10 - f11 + f12 + f13 - f14;
    let m17 = -f7 + f8 - f9 + f10 + f15 - f16 + f17 - f18;
    let m18 = f11 - f12 + f13 - f14 - f15 + f16 + f17 - f18;

    // Phase 4: equilibrium moments.
    let m1_eq = rho * (-11.0_f32 + 19.0_f32 * u_sq);
    let m2_eq = rho * (3.0_f32 - 5.5_f32 * u_sq);
    let m4_eq = (-2.0_f32 / 3.0_f32) * rho * ux;
    let m6_eq = (-2.0_f32 / 3.0_f32) * rho * uy;
    let m8_eq = (-2.0_f32 / 3.0_f32) * rho * uz;
    let m9_eq = rho * (2.0_f32 * ux * ux - uy * uy - uz * uz);
    let m10_eq = -0.5_f32 * rho * (2.0_f32 * ux * ux - uy * uy - uz * uz);
    let m11_eq = rho * (uy * uy - uz * uz);
    let m12_eq = -0.5_f32 * rho * (uy * uy - uz * uz);
    let m13_eq = rho * ux * uy;
    let m14_eq = rho * ux * uz;
    let m15_eq = rho * uy * uz;

    // Phase 5: relax ms = m - S*(m - m_eq) with 5 distinct rates.
    let s_nu = f32::reinterpret(inv_tau_bits); // 1/tau
    let s_e: f32 = 1.19_f32;
    let s_eps: f32 = 1.40_f32;
    let s_q: f32 = 1.20_f32;
    let s_ghost: f32 = 1.00_f32;

    let ms0 = m0;
    let ms1 = m1 - s_e * (m1 - m1_eq);
    let ms2 = m2 - s_eps * (m2 - m2_eq);
    let ms3 = m3;
    let ms4 = m4 - s_q * (m4 - m4_eq);
    let ms5 = m5;
    let ms6 = m6 - s_q * (m6 - m6_eq);
    let ms7 = m7;
    let ms8 = m8 - s_q * (m8 - m8_eq);
    let ms9 = m9 - s_nu * (m9 - m9_eq);
    let ms10 = m10 - s_ghost * (m10 - m10_eq);
    let ms11 = m11 - s_nu * (m11 - m11_eq);
    let ms12 = m12 - s_ghost * (m12 - m12_eq);
    let ms13 = m13 - s_nu * (m13 - m13_eq);
    let ms14 = m14 - s_nu * (m14 - m14_eq);
    let ms15 = m15 - s_nu * (m15 - m15_eq);
    let ms16 = m16 - s_ghost * m16;
    let ms17 = m17 - s_ghost * m17;
    let ms18 = m18 - s_ghost * m18;

    // Phase 6: scale by reciprocal row norms-squared.
    let s0 = ms0 * (1.0_f32 / 19.0_f32);
    let s1 = ms1 * (1.0_f32 / 2394.0_f32);
    let s2 = ms2 * (1.0_f32 / 252.0_f32);
    let s3 = ms3 * (1.0_f32 / 10.0_f32);
    let s4 = ms4 * (1.0_f32 / 40.0_f32);
    let s5 = ms5 * (1.0_f32 / 10.0_f32);
    let s6 = ms6 * (1.0_f32 / 40.0_f32);
    let s7 = ms7 * (1.0_f32 / 10.0_f32);
    let s8 = ms8 * (1.0_f32 / 40.0_f32);
    let s9 = ms9 * (1.0_f32 / 36.0_f32);
    let s10 = ms10 * (1.0_f32 / 36.0_f32);
    let s11 = ms11 * (1.0_f32 / 12.0_f32);
    let s12 = ms12 * (1.0_f32 / 12.0_f32);
    let s13 = ms13 * (1.0_f32 / 4.0_f32);
    let s14 = ms14 * (1.0_f32 / 4.0_f32);
    let s15 = ms15 * (1.0_f32 / 4.0_f32);
    let s16 = ms16 * (1.0_f32 / 8.0_f32);
    let s17 = ms17 * (1.0_f32 / 8.0_f32);
    let s18 = ms18 * (1.0_f32 / 8.0_f32);

    // Phase 7: inverse transform f_post = M^T_scaled * ms.
    let p0 = s0 - 30.0_f32 * s1 + 12.0_f32 * s2;
    let p1 = s0 - 11.0_f32 * s1 - 4.0_f32 * s2 + s3 - 4.0_f32 * s4 + 2.0_f32 * s9
        - 2.0_f32 * s10;
    let p2 = s0 - 11.0_f32 * s1 - 4.0_f32 * s2 - s3 + 4.0_f32 * s4 + 2.0_f32 * s9
        - 2.0_f32 * s10;
    let p3 = s0 - 11.0_f32 * s1 - 4.0_f32 * s2 + s5 - 4.0_f32 * s6 - s9 + s10 + s11 - s12;
    let p4 = s0 - 11.0_f32 * s1 - 4.0_f32 * s2 - s5 + 4.0_f32 * s6 - s9 + s10 + s11 - s12;
    let p5 = s0 - 11.0_f32 * s1 - 4.0_f32 * s2 + s7 - 4.0_f32 * s8 - s9 + s10 - s11 + s12;
    let p6 = s0 - 11.0_f32 * s1 - 4.0_f32 * s2 - s7 + 4.0_f32 * s8 - s9 + s10 - s11 + s12;
    let p7 = s0 + 8.0_f32 * s1 + s2 + s3 + s4 + s5 + s6 + s9 + s10 + s11 + s12 + s13 + s16
        - s17;
    let p8 = s0 + 8.0_f32 * s1 + s2 - s3 - s4 - s5 - s6 + s9 + s10 + s11 + s12 + s13 - s16
        + s17;
    let p9 = s0 + 8.0_f32 * s1 + s2 + s3 + s4 - s5 - s6 + s9 + s10 + s11 + s12 - s13 - s16
        - s17;
    let p10 = s0 + 8.0_f32 * s1 + s2 - s3 - s4 + s5 + s6 + s9 + s10 + s11 + s12 - s13 + s16
        + s17;
    let p11 = s0 + 8.0_f32 * s1 + s2 + s3 + s4 + s7 + s8 + s9 + s10 - s11 - s12 + s14 - s16
        + s18;
    let p12 = s0 + 8.0_f32 * s1 + s2 - s3 - s4 - s7 - s8 + s9 + s10 - s11 - s12 + s14 + s16
        - s18;
    let p13 = s0 + 8.0_f32 * s1 + s2 + s3 + s4 - s7 - s8 + s9 + s10 - s11 - s12 - s14 + s16
        + s18;
    let p14 = s0 + 8.0_f32 * s1 + s2 - s3 - s4 + s7 + s8 + s9 + s10 - s11 - s12 - s14 - s16
        - s18;
    let p15 = s0 + 8.0_f32 * s1 + s2 + s5 + s6 + s7 + s8 - 2.0_f32 * s9 - 2.0_f32 * s10
        + s15
        + s17
        - s18;
    let p16 = s0 + 8.0_f32 * s1 + s2 - s5 - s6 - s7 - s8 - 2.0_f32 * s9 - 2.0_f32 * s10
        + s15
        - s17
        + s18;
    let p17 = s0 + 8.0_f32 * s1 + s2 + s5 + s6 - s7 - s8 - 2.0_f32 * s9 - 2.0_f32 * s10
        - s15
        - s17
        - s18;
    let p18 = s0 + 8.0_f32 * s1 + s2 - s5 - s6 + s7 + s8 - 2.0_f32 * s9 - 2.0_f32 * s10
        - s15
        + s17
        + s18;

    // Phase 8: PUSH post-collision values to periodic-neighbor cells.
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

/// Periodic neighbor index in row-major (z, y, x) order.
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

/// Probe whether a cubecl-wgpu adapter is reachable.
pub fn is_available() -> bool {
    gororoba_gpu_cubecl::Runtime::probe()
}

/// Errors emitted by the D3Q19 MRT cubecl launcher.
#[derive(Debug, thiserror::Error)]
pub enum CubeclMrtLbmError {
    #[error("grid dimensions must all be positive (got nx={nx}, ny={ny}, nz={nz})")]
    EmptyGrid { nx: usize, ny: usize, nz: usize },
    #[error(
        "grid too large for u32 cell indexing: nx*ny*nz = {n_cells} (max = {})",
        u32::MAX
    )]
    GridTooLarge { n_cells: u64 },
    #[error("input f slice length {got} does not match nx*ny*nz*19 = {expected}")]
    LengthMismatch { got: usize, expected: usize },
    #[error("tau must satisfy tau > 0.5 for MRT stability (got {0})")]
    UnstableTau(f32),
    #[error("cubecl adapter not available on this host")]
    AdapterUnavailable,
}

/// Run `num_steps` MRT + periodic-stream timesteps on the cubecl-wgpu
/// runtime starting from `f_init` (SoA `[channel * n_cells + cell]`,
/// length `nx*ny*nz*19`). Returns the post-evolve `f` array in the same
/// SoA layout.
pub fn evolve_mrt_d3q19_cubecl(
    nx: usize,
    ny: usize,
    nz: usize,
    tau: f32,
    f_init: &[f32],
    num_steps: usize,
) -> Result<Vec<f32>, CubeclMrtLbmError> {
    if nx == 0 || ny == 0 || nz == 0 {
        return Err(CubeclMrtLbmError::EmptyGrid { nx, ny, nz });
    }
    if tau.is_nan() || tau <= 0.5 {
        return Err(CubeclMrtLbmError::UnstableTau(tau));
    }
    let n_cells_u64 = (nx as u64)
        .checked_mul(ny as u64)
        .and_then(|p| p.checked_mul(nz as u64))
        .unwrap_or(u64::MAX);
    if n_cells_u64 > u32::MAX as u64 {
        return Err(CubeclMrtLbmError::GridTooLarge {
            n_cells: n_cells_u64,
        });
    }
    let n_cells = nx * ny * nz;
    let expected = n_cells * D3Q19_CHANNELS as usize;
    if f_init.len() != expected {
        return Err(CubeclMrtLbmError::LengthMismatch {
            got: f_init.len(),
            expected,
        });
    }

    if num_steps == 0 {
        return Ok(f_init.to_vec());
    }

    if !is_available() {
        return Err(CubeclMrtLbmError::AdapterUnavailable);
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
        let ha = handle_a.clone();
        let hb = handle_b.clone();
        // SAFETY: launch_unchecked skips the cubecl bounds-check trampoline.
        // The kernel guards against oversubscribed workgroups with the
        // `if cell >= n_cells` early-terminate, and both arrays are
        // exactly `n_cells * 19` f32 elements matching the SoA addressing.
        unsafe {
            lbm_mrt_d3q19_step_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                cube_count.clone(),
                cube_dim,
                ArrayArg::from_raw_parts(ha, f_init.len()),
                ArrayArg::from_raw_parts(hb, f_init.len()),
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
        let out = evolve_mrt_d3q19_cubecl(4, 4, 4, 1.0, &f, 0).unwrap();
        assert_eq!(out, f);
    }

    #[test]
    fn rejects_unstable_tau() {
        let f = vec![1.0_f32; 4 * 4 * 4 * 19];
        match evolve_mrt_d3q19_cubecl(4, 4, 4, 0.4, &f, 1) {
            Err(CubeclMrtLbmError::UnstableTau(_)) => {}
            other => panic!("expected UnstableTau, got {other:?}"),
        }
    }

    #[test]
    fn rejects_length_mismatch() {
        let bad = vec![1.0_f32; 7];
        match evolve_mrt_d3q19_cubecl(2, 2, 2, 1.0, &bad, 1) {
            Err(CubeclMrtLbmError::LengthMismatch { got: 7, expected: 152 }) => {}
            other => panic!("expected LengthMismatch, got {other:?}"),
        }
    }

    #[test]
    fn rejects_grid_too_large_for_u32_indexing() {
        let nx = 1usize << 16;
        let ny = 1usize << 16;
        let nz = 2usize;
        let f: Vec<f32> = Vec::new();
        match evolve_mrt_d3q19_cubecl(nx, ny, nz, 1.0, &f, 1) {
            Err(CubeclMrtLbmError::GridTooLarge { n_cells }) => {
                assert_eq!(n_cells, (nx as u64) * (ny as u64) * (nz as u64));
            }
            other => panic!("expected GridTooLarge, got {other:?}"),
        }
    }
}
