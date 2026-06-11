//! cubecl-wgpu sparse D3Q19 direct active-brick backend.

#![cfg(feature = "cubecl")]

use cubecl::prelude::*;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

use crate::sparse_lbm_common::{SparseLbmError, SparseLbmPlan};

const THREADS_PER_CUBE: u32 = 256;

#[derive(Debug, thiserror::Error)]
pub enum SparseCubeclError {
    #[error(transparent)]
    Sparse(#[from] SparseLbmError),
    #[error("cubecl adapter not available on this host")]
    AdapterUnavailable,
}

pub fn is_available() -> bool {
    gororoba_gpu_cubecl::Runtime::probe()
}

pub fn evolve_sparse_d3q19_cubecl(
    plan: &SparseLbmPlan,
    tau: f32,
    f_init: &[f32],
    num_steps: usize,
) -> Result<Vec<f32>, SparseCubeclError> {
    if tau.is_nan() || tau <= 0.5 {
        return Err(SparseLbmError::UnstableTau(tau).into());
    }
    if plan.n_active_bricks() == 0 {
        return Err(SparseLbmError::NoActiveBricks.into());
    }
    plan.validate_f_len(f_init)?;
    if num_steps == 0 {
        return Ok(f_init.to_vec());
    }
    if !is_available() {
        return Err(SparseCubeclError::AdapterUnavailable);
    }

    let device = WgpuDevice::default();
    let client = WgpuRuntime::client(&device);
    let f_handle = client.create_from_slice(bytemuck::cast_slice(f_init));
    let active_handle = client.create_from_slice(bytemuck::cast_slice(&plan.active_brick_ids));
    let indirect_handle = client.create_from_slice(bytemuck::cast_slice(&plan.indirect_table));

    let n_active_cells = plan.n_active_cells();
    let cube_dim = CubeDim::new_1d(THREADS_PER_CUBE);
    let cube_count = CubeCount::new_1d((n_active_cells as u32).div_ceil(THREADS_PER_CUBE));
    let inv_tau_bits = (1.0_f32 / tau).to_bits();

    for step in 0..num_steps {
        let f_for_launch = f_handle.clone();
        let active_for_launch = active_handle.clone();
        let indirect_for_launch = indirect_handle.clone();
        unsafe {
            sparse_d3q19_step_kernel::launch_unchecked::<WgpuRuntime>(
                &client,
                cube_count.clone(),
                cube_dim,
                ArrayArg::from_raw_parts(f_for_launch, f_init.len()),
                ArrayArg::from_raw_parts(active_for_launch, plan.active_brick_ids.len()),
                ArrayArg::from_raw_parts(indirect_for_launch, plan.indirect_table.len()),
                plan.nx as u32,
                plan.ny as u32,
                plan.nz as u32,
                plan.bricks_x as u32,
                plan.bricks_y as u32,
                plan.bricks_z as u32,
                n_active_cells as u32,
                (step & 1) as u32,
                inv_tau_bits,
            );
        }
    }

    let out_bytes = client.read_one_unchecked(f_handle);
    let out: &[f32] = bytemuck::cast_slice(&out_bytes);
    Ok(out.to_vec())
}

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn sparse_d3q19_step_kernel(
    f: &mut Array<f32>,
    active_brick_ids: &Array<u32>,
    indirect_table: &Array<i32>,
    #[comptime] nx: u32,
    #[comptime] ny: u32,
    #[comptime] nz: u32,
    #[comptime] bricks_x: u32,
    #[comptime] bricks_y: u32,
    #[comptime] _bricks_z: u32,
    #[comptime] n_active_cells: u32,
    #[comptime] parity: u32,
    #[comptime] inv_tau_bits: u32,
) {
    let tid = ABSOLUTE_POS;
    let n_active = n_active_cells as usize;
    if tid >= n_active {
        terminate!();
    }

    let bricks_x_u = bricks_x as usize;
    let bricks_y_u = bricks_y as usize;
    let pool_idx = tid / 512usize;
    let local_idx = tid % 512usize;
    let global_brick = active_brick_ids[pool_idx] as usize;
    let bx = global_brick % bricks_x_u;
    let by = (global_brick / bricks_x_u) % bricks_y_u;
    let bz = global_brick / (bricks_x_u * bricks_y_u);
    let lx = local_idx % 8usize;
    let ly = (local_idx / 8usize) % 8usize;
    let lz = local_idx / 64usize;
    let x = bx * 8usize + lx;
    let y = by * 8usize + ly;
    let z = bz * 8usize + lz;
    if x >= nx as usize || y >= ny as usize || z >= nz as usize {
        terminate!();
    }

    let mut rho = 0.0_f32;
    let mut mx = 0.0_f32;
    let mut my = 0.0_f32;
    let mut mz = 0.0_f32;
    let mut local = Array::<f32>::new(19usize);
    let mut i = 0usize;
    while i < 19usize {
        let mut read_dir = i;
        let mut src_tid = tid;
        if parity != 0u32 {
            let candidate = neighbor_tid(
                x as i32,
                y as i32,
                z as i32,
                -cx(i as u32),
                -cy(i as u32),
                -cz(i as u32),
                nx as i32,
                ny as i32,
                nz as i32,
                bricks_x_u,
                bricks_y_u,
                indirect_table,
            );
            if candidate >= 0i32 {
                read_dir = opp(i as u32) as usize;
                src_tid = candidate as usize;
            }
        }
        let value = f[read_dir * n_active + src_tid];
        let fi = finite_or_zero(value);
        local[i] = fi;
        rho += fi;
        mx += (cx(i as u32) as f32) * fi;
        my += (cy(i as u32) as f32) * fi;
        mz += (cz(i as u32) as f32) * fi;
        i += 1usize;
    }

    let mut ux = 0.0_f32;
    let mut uy = 0.0_f32;
    let mut uz = 0.0_f32;
    if rho > 1.0e-20_f32 {
        let inv_rho = 1.0_f32 / rho;
        ux = mx * inv_rho;
        uy = my * inv_rho;
        uz = mz * inv_rho;
    } else {
        rho = 1.0_f32;
    }
    let inv_tau = f32::reinterpret(inv_tau_bits);
    let u_sq = ux * ux + uy * uy + uz * uz;
    let base = 1.0_f32 - 1.5_f32 * u_sq;
    let mut j = 0usize;
    while j < 19usize {
        let eu =
            (cx(j as u32) as f32) * ux + (cy(j as u32) as f32) * uy + (cz(j as u32) as f32) * uz;
        let f_eq = weight(j as u32) * rho * (base + 3.0_f32 * eu + 4.5_f32 * eu * eu);
        local[j] = local[j] - (local[j] - f_eq) * inv_tau;
        j += 1usize;
    }

    let mut k = 0usize;
    while k < 19usize {
        if parity == 0u32 {
            let candidate = neighbor_tid(
                x as i32,
                y as i32,
                z as i32,
                cx(k as u32),
                cy(k as u32),
                cz(k as u32),
                nx as i32,
                ny as i32,
                nz as i32,
                bricks_x_u,
                bricks_y_u,
                indirect_table,
            );
            let write_tid = if candidate >= 0i32 {
                candidate as usize
            } else {
                tid
            };
            f[(opp(k as u32) as usize) * n_active + write_tid] = local[k];
        } else {
            f[k * n_active + tid] = local[k];
        }
        k += 1usize;
    }
}

#[cube]
#[allow(clippy::too_many_arguments)]
fn neighbor_tid(
    x: i32,
    y: i32,
    z: i32,
    dx: i32,
    dy: i32,
    dz: i32,
    nx: i32,
    ny: i32,
    nz: i32,
    bricks_x: usize,
    bricks_y: usize,
    indirect_table: &Array<i32>,
) -> i32 {
    let xn = wrap_i32(x + dx, nx) as usize;
    let yn = wrap_i32(y + dy, ny) as usize;
    let zn = wrap_i32(z + dz, nz) as usize;
    let bx = xn / 8usize;
    let by = yn / 8usize;
    let bz = zn / 8usize;
    let brick = bx + bricks_x * (by + bricks_y * bz);
    let pool = indirect_table[brick];
    let mut result = pool;
    if pool >= 0i32 {
        let lx = xn % 8usize;
        let ly = yn % 8usize;
        let lz = zn % 8usize;
        result = pool * 512i32 + (lx + 8usize * (ly + 8usize * lz)) as i32;
    }
    result
}

#[cube]
fn wrap_i32(value: i32, modulus: i32) -> i32 {
    let mut wrapped = value % modulus;
    if wrapped < 0i32 {
        wrapped += modulus;
    }
    wrapped
}

#[cube]
#[allow(
    clippy::eq_op,
    clippy::excessive_precision,
    clippy::manual_range_contains
)]
fn finite_or_zero(value: f32) -> f32 {
    let mut result = value;
    if !(value == value && value <= 3.402823466e38_f32 && value >= -3.402823466e38_f32) {
        result = 0.0_f32;
    }
    result
}

#[cube]
fn cx(i: u32) -> i32 {
    let mut result = 0i32;
    if i == 1u32 || i == 7u32 || i == 9u32 || i == 11u32 || i == 13u32 {
        result = 1i32;
    }
    if i == 2u32 || i == 8u32 || i == 10u32 || i == 12u32 || i == 14u32 {
        result = -1i32;
    }
    result
}

#[cube]
fn cy(i: u32) -> i32 {
    let mut result = 0i32;
    if i == 3u32 || i == 7u32 || i == 10u32 || i == 15u32 || i == 17u32 {
        result = 1i32;
    }
    if i == 4u32 || i == 8u32 || i == 9u32 || i == 16u32 || i == 18u32 {
        result = -1i32;
    }
    result
}

#[cube]
fn cz(i: u32) -> i32 {
    let mut result = 0i32;
    if i == 5u32 || i == 11u32 || i == 14u32 || i == 15u32 || i == 18u32 {
        result = 1i32;
    }
    if i == 6u32 || i == 12u32 || i == 13u32 || i == 16u32 || i == 17u32 {
        result = -1i32;
    }
    result
}

#[cube]
fn opp(i: u32) -> u32 {
    let mut result = 0u32;
    if i > 0u32 {
        result = i - 1u32;
        if i % 2u32 == 1u32 {
            result = i + 1u32;
        }
    }
    result
}

#[cube]
fn weight(i: u32) -> f32 {
    let mut result = (i as f32) * 0.0_f32 + (1.0_f32 / 36.0_f32);
    if i == 0u32 {
        result = 1.0_f32 / 3.0_f32;
    }
    if i == 1u32 || i == 2u32 || i == 3u32 || i == 4u32 || i == 5u32 || i == 6u32 {
        result = 1.0_f32 / 18.0_f32;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_lbm_common::evolve_sparse_d3q19_cpu;

    #[test]
    fn zero_steps_returns_input() {
        let mask = vec![1u8; 8 * 8 * 8];
        let plan = SparseLbmPlan::from_geometry_mask(8, 8, 8, &mask).unwrap();
        let f0 = plan.equilibrium_at_rest();
        let out = evolve_sparse_d3q19_cubecl(&plan, 1.0, &f0, 0).unwrap();
        assert_eq!(out, f0);
    }

    #[test]
    #[ignore = "cubecl-wgpu adapter required"]
    fn cubecl_matches_cpu_sparse_equilibrium() {
        if !is_available() {
            return;
        }
        let mask = vec![1u8; 8 * 8 * 8];
        let plan = SparseLbmPlan::from_geometry_mask(8, 8, 8, &mask).unwrap();
        let f0 = plan.equilibrium_at_rest();
        let cpu = evolve_sparse_d3q19_cpu(&plan, 1.0, &f0, 3).unwrap();
        let gpu = evolve_sparse_d3q19_cubecl(&plan, 1.0, &f0, 3).unwrap();
        assert_eq!(gpu.len(), cpu.len());
        for (idx, (got, expected)) in gpu.iter().zip(cpu.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1.0e-6,
                "idx={idx} got={got} expected={expected}"
            );
        }
    }
}
