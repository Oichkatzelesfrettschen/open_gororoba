//! cubecl-wgpu compute path for the CUDA-style GRMHD conservative advance.
//!
//! The kernel mirrors the staged Vulkan path and works over the same 8-channel
//! primitive/conserved SoA buffers. Storage uses FP32, so parity checks compare
//! device outputs with canonical CPU calculations at FP32-appropriate tolerances.

#![cfg(feature = "cubecl")]

use cubecl::{client::ComputeClient, prelude::*, server::Handle};
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

pub use crate::vulkan::{GrmhdVulkanConfig as GrmhdCubeclConfig, NCONS, NPRIM};

const WORKGROUP_SIZE: u32 = 256;
const OP_PRECOMPUTE_METRIC: u32 = 0;
const OP_PRIM2CON: u32 = 1;
const OP_COMPUTE_FLUX: u32 = 2;
const OP_FLUX_DIVERGENCE: u32 = 3;
const OP_EULER_UPDATE: u32 = 4;

#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)] // CubeCL launch ABI keeps SoA buffers and comptime scalars as separate kernel IR values.
pub fn grmhd_cubecl_step_kernel(
    prims: &Array<f32>,
    cons: &mut Array<f32>,
    cons_new: &mut Array<f32>,
    flux: &mut Array<f32>,
    rhs: &mut Array<f32>,
    gcov: &mut Array<f32>,
    sqrt_g: &mut Array<f32>,
    #[comptime] n1: u32,
    #[comptime] n2: u32,
    #[comptime] n3: u32,
    #[comptime] r_min_bits: u32,
    #[comptime] log_r_ratio_bits: u32,
    #[comptime] kerr_a_bits: u32,
    #[comptime] gam_m1_bits: u32,
    #[comptime] dx1_bits: u32,
    #[comptime] dx2_bits: u32,
    #[comptime] dx3_bits: u32,
    #[comptime] dt_bits: u32,
    #[comptime] dir: u32,
    #[comptime] op: u32,
) {
    let idx = ABSOLUTE_POS;
    let n1_u = n1 as usize;
    let n2_u = n2 as usize;
    let n3_u = n3 as usize;
    let n_total = n1_u * n2_u * n3_u;

    if op == OP_PRECOMPUTE_METRIC {
        precompute_metric(
            idx,
            gcov,
            sqrt_g,
            n1_u,
            n2_u,
            r_min_bits,
            log_r_ratio_bits,
            kerr_a_bits,
        );
    } else if op == OP_PRIM2CON {
        prim2con(
            idx,
            prims,
            cons,
            gcov,
            sqrt_g,
            n2_u,
            n3_u,
            n_total,
            gam_m1_bits,
        );
    } else if op == OP_COMPUTE_FLUX {
        compute_flux(
            idx,
            prims,
            flux,
            gcov,
            sqrt_g,
            n2_u,
            n3_u,
            n_total,
            gam_m1_bits,
            dir,
        );
    } else if op == OP_FLUX_DIVERGENCE {
        flux_divergence(
            idx, flux, rhs, n1_u, n2_u, n3_u, n_total, dx1_bits, dx2_bits, dx3_bits, dir,
        );
    } else if op == OP_EULER_UPDATE {
        euler_update(idx, cons, cons_new, rhs, n_total, dt_bits);
    }
}

#[cube]
#[allow(clippy::too_many_arguments)] // Metric precompute takes buffer handles and Kerr grid scalars as separate device arguments.
fn precompute_metric(
    cell: usize,
    gcov: &mut Array<f32>,
    sqrt_g: &mut Array<f32>,
    n1: usize,
    n2: usize,
    r_min_bits: u32,
    log_r_ratio_bits: u32,
    kerr_a_bits: u32,
) {
    let total = n1 * n2;
    if cell >= total {
        terminate!();
    }

    let i = cell / n2;
    let j = cell % n2;
    let r_min = f32::reinterpret(r_min_bits);
    let log_r_ratio = f32::reinterpret(log_r_ratio_bits);
    let kerr_a = f32::reinterpret(kerr_a_bits);
    let xi = (i as f32) / ((n1 - 1) as f32);
    let r = r_min * f32::exp(xi * log_r_ratio);
    let th = std::f32::consts::PI * ((j as f32) + 0.5_f32) / (n2 as f32);
    let sth = f32::sin(th);
    let cth = f32::cos(th);
    let sigma = r * r + kerr_a * kerr_a * cth * cth;
    let delta = r * r - 2.0_f32 * r + kerr_a * kerr_a;
    let a_met =
        (r * r + kerr_a * kerr_a) * (r * r + kerr_a * kerr_a) - delta * kerr_a * kerr_a * sth * sth;
    let base = 5 * cell;
    gcov[base] = -(1.0_f32 - 2.0_f32 * r / sigma);
    gcov[base + 1] = sigma / delta;
    gcov[base + 2] = sigma;
    gcov[base + 3] = a_met * sth * sth / sigma;
    gcov[base + 4] = -2.0_f32 * kerr_a * r * sth * sth / sigma;
    sqrt_g[cell] = sigma * abs_f32(sth);
}

#[cube]
fn prim2con(
    cell: usize,
    prims: &Array<f32>,
    cons: &mut Array<f32>,
    gcov: &Array<f32>,
    sqrt_g: &Array<f32>,
    n2: usize,
    n3: usize,
    n_total: usize,
    gam_m1_bits: u32,
) {
    if cell >= n_total {
        terminate!();
    }

    let met_idx = metric_index(cell, n2, n3);
    let rho = prims[soa_index(0, cell, n_total)];
    let u = prims[soa_index(1, cell, n_total)];
    let v1 = prims[soa_index(2, cell, n_total)];
    let v2 = prims[soa_index(3, cell, n_total)];
    let v3 = prims[soa_index(4, cell, n_total)];
    let b1 = prims[soa_index(5, cell, n_total)];
    let b2 = prims[soa_index(6, cell, n_total)];
    let b3 = prims[soa_index(7, cell, n_total)];
    let pressure = f32::reinterpret(gam_m1_bits) * u;
    let base = 5 * met_idx;
    let g_tt = gcov[base];
    let g_rr = gcov[base + 1];
    let g_thth = gcov[base + 2];
    let g_phph = gcov[base + 3];
    let g_tph = gcov[base + 4];
    let sg = sqrt_g[met_idx];

    let vsq = g_rr * v1 * v1 + g_thth * v2 * v2 + g_phph * v3 * v3;
    let alpha_sq = -(g_tt + 2.0_f32 * g_tph * v3 + vsq);
    let mut ut = 1.0_f32;
    if alpha_sq > 1.0e-20_f32 {
        ut = 1.0_f32 / f32::sqrt(alpha_sq);
    }
    let inv_ut = 1.0_f32 / ut;
    let u_r = g_rr * ut * v1;
    let u_th = g_thth * ut * v2;
    let u_ph = g_phph * ut * v3 + g_tph * ut;
    let u_t = g_tt * ut + g_tph * ut * v3;
    let bt = b1 * u_r + b2 * u_th + b3 * u_ph;
    let b_up_1 = (b1 + bt * ut * v1) * inv_ut;
    let b_up_2 = (b2 + bt * ut * v2) * inv_ut;
    let b_up_3 = (b3 + bt * ut * v3) * inv_ut;
    let b_t_cov = g_tt * bt + g_tph * b_up_3;
    let b_r_cov = g_rr * b_up_1;
    let b_th_cov = g_thth * b_up_2;
    let b_ph_cov = g_tph * bt + g_phph * b_up_3;
    let bsq_raw = bt * b_t_cov + b_up_1 * b_r_cov + b_up_2 * b_th_cov + b_up_3 * b_ph_cov;
    let bsq = max_f32(bsq_raw, 0.0_f32);
    let w = rho + u + pressure + bsq;
    let ptot = pressure + 0.5_f32 * bsq;

    cons[soa_index(0, cell, n_total)] = sg * rho * ut;
    let t_t_t = w * ut * u_t + ptot - bt * b_t_cov;
    cons[soa_index(1, cell, n_total)] = sg * (t_t_t + rho * ut);
    cons[soa_index(2, cell, n_total)] = sg * (w * ut * u_r - bt * b_r_cov);
    cons[soa_index(3, cell, n_total)] = sg * (w * ut * u_th - bt * b_th_cov);
    cons[soa_index(4, cell, n_total)] = sg * (w * ut * u_ph - bt * b_ph_cov);
    cons[soa_index(5, cell, n_total)] = sg * b1;
    cons[soa_index(6, cell, n_total)] = sg * b2;
    cons[soa_index(7, cell, n_total)] = sg * b3;
}

#[cube]
#[allow(clippy::too_many_arguments)] // Flux evaluation keeps primitive, metric, grid, and direction lanes explicit in device IR.
fn compute_flux(
    cell: usize,
    prims: &Array<f32>,
    flux: &mut Array<f32>,
    gcov: &Array<f32>,
    sqrt_g: &Array<f32>,
    n2: usize,
    n3: usize,
    n_total: usize,
    gam_m1_bits: u32,
    dir: u32,
) {
    if cell >= n_total {
        terminate!();
    }

    let met_idx = metric_index(cell, n2, n3);
    let rho = prims[soa_index(0, cell, n_total)];
    let u = prims[soa_index(1, cell, n_total)];
    let v1 = prims[soa_index(2, cell, n_total)];
    let v2 = prims[soa_index(3, cell, n_total)];
    let v3 = prims[soa_index(4, cell, n_total)];
    let b1 = prims[soa_index(5, cell, n_total)];
    let b2 = prims[soa_index(6, cell, n_total)];
    let b3 = prims[soa_index(7, cell, n_total)];
    let pressure = f32::reinterpret(gam_m1_bits) * u;
    let base = 5 * met_idx;
    let g_tt = gcov[base];
    let g_rr = gcov[base + 1];
    let g_thth = gcov[base + 2];
    let g_phph = gcov[base + 3];
    let g_tph = gcov[base + 4];
    let sg = sqrt_g[met_idx];

    let vsq = g_rr * v1 * v1 + g_thth * v2 * v2 + g_phph * v3 * v3;
    let alpha_sq = -(g_tt + 2.0_f32 * g_tph * v3 + vsq);
    let mut ut = 1.0_f32;
    if alpha_sq > 1.0e-20_f32 {
        ut = 1.0_f32 / f32::sqrt(alpha_sq);
    }
    let inv_ut = 1.0_f32 / ut;

    let u_r = g_rr * ut * v1;
    let u_th = g_thth * ut * v2;
    let u_ph = g_phph * ut * v3 + g_tph * ut;
    let u_t = g_tt * ut + g_tph * ut * v3;
    let bt = b1 * u_r + b2 * u_th + b3 * u_ph;
    let b_up_1 = (b1 + bt * ut * v1) * inv_ut;
    let b_up_2 = (b2 + bt * ut * v2) * inv_ut;
    let b_up_3 = (b3 + bt * ut * v3) * inv_ut;
    let b_t_cov = g_tt * bt + g_tph * b_up_3;
    let b_r_cov = g_rr * b_up_1;
    let b_th_cov = g_thth * b_up_2;
    let b_ph_cov = g_tph * bt + g_phph * b_up_3;
    let bsq_raw = bt * b_t_cov + b_up_1 * b_r_cov + b_up_2 * b_th_cov + b_up_3 * b_ph_cov;
    let bsq = max_f32(bsq_raw, 0.0_f32);
    let w = rho + u + pressure + bsq;
    let ptot = pressure + 0.5_f32 * bsq;

    let v_dir = select3(v1, v2, v3, dir);
    let b_dir = select3(b1, b2, b3, dir);
    let u_up_dir = ut * v_dir;
    let b_up_dir = select3(b_up_1, b_up_2, b_up_3, dir);

    flux[soa_index(0, cell, n_total)] = sg * rho * u_up_dir;
    let b_cov_t = g_tt * bt + g_tph * b_up_3;
    let t_dir_t = w * u_up_dir * u_t - b_up_dir * b_cov_t;
    flux[soa_index(1, cell, n_total)] = sg * (t_dir_t + rho * u_up_dir);

    let mut jj = 0u32;
    while jj < 3 {
        let u_cov_j = select3(u_r, u_th, u_ph, jj);
        let g_diag_j = select3(g_rr, g_thth, g_phph, jj);
        let b_up_j = select3(b_up_1, b_up_2, b_up_3, jj);
        let mut b_cov_j = g_diag_j * b_up_j;
        if jj == 2 {
            b_cov_j += g_tph * bt;
        }
        let mut delta_jd = 0.0_f32;
        if jj == dir {
            delta_jd = 1.0_f32;
        }
        flux[soa_index(2 + (jj as usize), cell, n_total)] =
            sg * (w * u_up_dir * u_cov_j + ptot * delta_jd - b_up_dir * b_cov_j);
        jj += 1;
    }

    let mut jj2 = 0u32;
    while jj2 < 3 {
        let v_j2 = select3(v1, v2, v3, jj2);
        let b_j2 = select3(b1, b2, b3, jj2);
        flux[soa_index(5 + (jj2 as usize), cell, n_total)] = sg * (v_dir * b_j2 - v_j2 * b_dir);
        jj2 += 1;
    }
}

#[cube]
#[allow(clippy::too_many_arguments)] // Divergence stencil keeps grid extents and spacing bits explicit for CubeCL indexing.
fn flux_divergence(
    cell: usize,
    flux: &Array<f32>,
    rhs: &mut Array<f32>,
    n1: usize,
    n2: usize,
    n3: usize,
    n_total: usize,
    dx1_bits: u32,
    dx2_bits: u32,
    dx3_bits: u32,
    dir: u32,
) {
    if cell >= n_total {
        terminate!();
    }

    let k = cell % n3;
    let tmp1 = cell / n3;
    let j = tmp1 % n2;
    let i = tmp1 / n2;
    let mut at_boundary = false;
    let mut idx_m = cell;
    let mut idx_p = cell;

    if dir == 0 {
        let stride = n2 * n3;
        at_boundary = i == 0 || i >= n1 - 1;
        if !at_boundary {
            idx_m = cell - stride;
            idx_p = cell + stride;
        }
    } else if dir == 1 {
        let stride = n3;
        at_boundary = j == 0 || j >= n2 - 1;
        if !at_boundary {
            idx_m = cell - stride;
            idx_p = cell + stride;
        }
    } else {
        if k == 0 {
            idx_m = cell + n3 - 1;
        } else {
            idx_m = cell - 1;
        }
        if k == n3 - 1 {
            idx_p = cell + 1 - n3;
        } else {
            idx_p = cell + 1;
        }
    }

    if at_boundary {
        terminate!();
    }

    let inv_dx = select3(
        1.0_f32 / f32::reinterpret(dx1_bits),
        1.0_f32 / f32::reinterpret(dx2_bits),
        1.0_f32 / f32::reinterpret(dx3_bits),
        dir,
    );
    let mut var_idx = 0usize;
    while var_idx < NCONS {
        let offset = var_idx * n_total;
        let f_p = flux[offset + idx_p];
        let f_m = flux[offset + idx_m];
        rhs[offset + cell] = rhs[offset + cell] - 0.5_f32 * inv_dx * (f_p - f_m);
        var_idx += 1;
    }
}

#[cube]
fn euler_update(
    idx: usize,
    cons: &Array<f32>,
    cons_new: &mut Array<f32>,
    rhs: &Array<f32>,
    n_total: usize,
    dt_bits: u32,
) {
    let total_vars = NCONS * n_total;
    if idx >= total_vars {
        terminate!();
    }
    cons_new[idx] = cons[idx] + f32::reinterpret(dt_bits) * rhs[idx];
}

#[cube]
fn soa_index(channel: usize, cell: usize, n_total: usize) -> usize {
    channel * n_total + cell
}

#[cube]
fn metric_index(cell: usize, n2: usize, n3: usize) -> usize {
    let tmp1 = cell / n3;
    let j = tmp1 % n2;
    let i = tmp1 / n2;
    i * n2 + j
}

#[cube]
fn select3(a: f32, b: f32, c: f32, idx: u32) -> f32 {
    let mut value = a;
    if idx == 1 {
        value = b;
    } else if idx == 2 {
        value = c;
    }
    value
}

#[cube]
fn abs_f32(value: f32) -> f32 {
    let mut out = value;
    if value < 0.0_f32 {
        out = -value;
    }
    out
}

#[cube]
fn max_f32(a: f32, b: f32) -> f32 {
    let mut out = a;
    if b > a {
        out = b;
    }
    out
}

pub struct GrmhdCubeclKernel;

impl GrmhdCubeclKernel {
    pub fn is_available() -> bool {
        grmhd_cubecl_available()
    }

    pub fn advance_conserved(
        config: GrmhdCubeclConfig,
        prims_soa: &[f32],
        dt: f32,
        steps: usize,
    ) -> Result<Vec<f32>, String> {
        validate_run_inputs(config, prims_soa, dt, steps)?;
        if !Self::is_available() {
            return Err("GRMHD cubecl adapter unavailable".to_string());
        }

        let n_total = config.n_total();
        let cons_len = NCONS * n_total;

        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        let buffers = CubeclRunBuffers::new(&client, config, prims_soa)?;

        launch_metric_precompute(&client, &buffers, config, dt)?;
        for _ in 0..steps {
            launch_step(&client, &buffers, config, dt)?;
        }

        decode_f32_output(
            &client.read_one_unchecked(buffers.cons_new_readback.clone()),
            cons_len,
            "cons_new",
        )
    }
}

pub fn grmhd_cubecl_available() -> bool {
    static CUBECL_AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CUBECL_AVAILABLE.get_or_init(|| {
        if !crate::vulkan::GrmhdVulkanKernel::is_available() {
            return false;
        }
        gororoba_gpu_cubecl::Runtime::probe()
    })
}

pub fn advance_conserved_cubecl(
    config: GrmhdCubeclConfig,
    prims_soa: &[f32],
    dt: f32,
    steps: usize,
) -> Result<Vec<f32>, String> {
    GrmhdCubeclKernel::advance_conserved(config, prims_soa, dt, steps)
}

struct CubeclRunBuffers {
    prims_handle: Handle,
    cons_handle: Handle,
    cons_new_handle: Handle,
    cons_new_readback: Handle,
    flux_handle: Handle,
    gcov_handle: Handle,
    sqrt_handle: Handle,
    zero_cons: Vec<f32>,
}

impl CubeclRunBuffers {
    fn new(
        client: &ComputeClient<WgpuRuntime>,
        config: GrmhdCubeclConfig,
        prims_soa: &[f32],
    ) -> Result<Self, String> {
        let n_total = config.n_total();
        let zero_cons = vec![0.0f32; NCONS * n_total];
        let zero_metric = vec![0.0f32; 5 * config.n1 * config.n2];
        let zero_sqrt = vec![0.0f32; config.n1 * config.n2];
        let cons_new_handle = create_buffer(client, &zero_cons)?;

        Ok(Self {
            prims_handle: create_buffer(client, prims_soa)?,
            cons_handle: create_buffer(client, &zero_cons)?,
            cons_new_readback: cons_new_handle.clone(),
            cons_new_handle,
            flux_handle: create_buffer(client, &zero_cons)?,
            gcov_handle: create_buffer(client, &zero_metric)?,
            sqrt_handle: create_buffer(client, &zero_sqrt)?,
            zero_cons,
        })
    }

    fn zero_rhs(&self, client: &ComputeClient<WgpuRuntime>) -> Result<Handle, String> {
        create_buffer(client, &self.zero_cons)
    }
}

fn create_buffer(client: &ComputeClient<WgpuRuntime>, values: &[f32]) -> Result<Handle, String> {
    Ok(client.create_from_slice(&encode_f32_slice(values)?))
}

fn launch_metric_precompute(
    client: &ComputeClient<WgpuRuntime>,
    buffers: &CubeclRunBuffers,
    config: GrmhdCubeclConfig,
    dt: f32,
) -> Result<(), String> {
    let rhs_handle = buffers.zero_rhs(client)?;
    launch_op(
        client,
        &buffers.prims_handle,
        &buffers.cons_handle,
        &buffers.cons_new_handle,
        &buffers.flux_handle,
        &rhs_handle,
        &buffers.gcov_handle,
        &buffers.sqrt_handle,
        config,
        dt,
        0,
        OP_PRECOMPUTE_METRIC,
        config.n1 * config.n2,
    )
}

fn launch_step(
    client: &ComputeClient<WgpuRuntime>,
    buffers: &CubeclRunBuffers,
    config: GrmhdCubeclConfig,
    dt: f32,
) -> Result<(), String> {
    let rhs_handle = buffers.zero_rhs(client)?;
    let n_total = config.n_total();
    launch_op(
        client,
        &buffers.prims_handle,
        &buffers.cons_handle,
        &buffers.cons_new_handle,
        &buffers.flux_handle,
        &rhs_handle,
        &buffers.gcov_handle,
        &buffers.sqrt_handle,
        config,
        dt,
        0,
        OP_PRIM2CON,
        n_total,
    )?;
    for dir in 0..3u32 {
        launch_flux_direction(client, buffers, &rhs_handle, config, dt, dir)?;
    }
    launch_op(
        client,
        &buffers.prims_handle,
        &buffers.cons_handle,
        &buffers.cons_new_handle,
        &buffers.flux_handle,
        &rhs_handle,
        &buffers.gcov_handle,
        &buffers.sqrt_handle,
        config,
        dt,
        0,
        OP_EULER_UPDATE,
        NCONS * n_total,
    )
}

fn launch_flux_direction(
    client: &ComputeClient<WgpuRuntime>,
    buffers: &CubeclRunBuffers,
    rhs_handle: &Handle,
    config: GrmhdCubeclConfig,
    dt: f32,
    dir: u32,
) -> Result<(), String> {
    let n_total = config.n_total();
    launch_op(
        client,
        &buffers.prims_handle,
        &buffers.cons_handle,
        &buffers.cons_new_handle,
        &buffers.flux_handle,
        rhs_handle,
        &buffers.gcov_handle,
        &buffers.sqrt_handle,
        config,
        dt,
        dir,
        OP_COMPUTE_FLUX,
        n_total,
    )?;
    launch_op(
        client,
        &buffers.prims_handle,
        &buffers.cons_handle,
        &buffers.cons_new_handle,
        &buffers.flux_handle,
        rhs_handle,
        &buffers.gcov_handle,
        &buffers.sqrt_handle,
        config,
        dt,
        dir,
        OP_FLUX_DIVERGENCE,
        n_total,
    )
}

#[allow(clippy::too_many_arguments)] // Host launcher mirrors the CubeCL kernel ABI and descriptor ordering.
fn launch_op(
    client: &ComputeClient<WgpuRuntime>,
    prims_handle: &Handle,
    cons_handle: &Handle,
    cons_new_handle: &Handle,
    flux_handle: &Handle,
    rhs_handle: &Handle,
    gcov_handle: &Handle,
    sqrt_handle: &Handle,
    config: GrmhdCubeclConfig,
    dt: f32,
    dir: u32,
    op: u32,
    work_items: usize,
) -> Result<(), String> {
    let cube_dim = CubeDim::new_1d(WORKGROUP_SIZE);
    let cube_count = CubeCount::new_1d(work_items.div_ceil(WORKGROUP_SIZE as usize) as u32);
    let dx = config.dx();
    let log_r_ratio = (config.r_max / config.r_min).ln();

    // SAFETY: every ArrayArg length matches its underlying f32 element count.
    // Each operation exits threads whose absolute position exceeds its active
    // work-item range.
    unsafe {
        grmhd_cubecl_step_kernel::launch_unchecked::<WgpuRuntime>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(prims_handle.clone(), NPRIM * config.n_total()),
            ArrayArg::from_raw_parts(cons_handle.clone(), NCONS * config.n_total()),
            ArrayArg::from_raw_parts(cons_new_handle.clone(), NCONS * config.n_total()),
            ArrayArg::from_raw_parts(flux_handle.clone(), NCONS * config.n_total()),
            ArrayArg::from_raw_parts(rhs_handle.clone(), NCONS * config.n_total()),
            ArrayArg::from_raw_parts(gcov_handle.clone(), 5 * config.n1 * config.n2),
            ArrayArg::from_raw_parts(sqrt_handle.clone(), config.n1 * config.n2),
            config.n1 as u32,
            config.n2 as u32,
            config.n3 as u32,
            config.r_min.to_bits(),
            log_r_ratio.to_bits(),
            config.kerr_a.to_bits(),
            (config.gamma - 1.0).to_bits(),
            dx[0].to_bits(),
            dx[1].to_bits(),
            dx[2].to_bits(),
            dt.to_bits(),
            dir,
            op,
        );
    }
    Ok(())
}

fn validate_run_inputs(
    config: GrmhdCubeclConfig,
    prims_soa: &[f32],
    dt: f32,
    steps: usize,
) -> Result<(), String> {
    config.validate()?;
    let expected = NPRIM * config.n_total();
    if prims_soa.len() != expected {
        return Err(format!(
            "primitive SoA length mismatch: got {}, expected {}",
            prims_soa.len(),
            expected
        ));
    }
    if !(dt.is_finite() && dt > 0.0) {
        return Err(format!("dt must be finite and positive, got {dt}"));
    }
    if steps == 0 {
        return Err("steps must be at least 1".to_string());
    }
    Ok(())
}

fn encode_f32_slice(values: &[f32]) -> Result<Vec<u8>, String> {
    let byte_len = values
        .len()
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| "GRMHD cubecl buffer size overflows".to_string())?;
    let mut bytes = Vec::with_capacity(byte_len);
    for &value in values {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    Ok(bytes)
}

fn decode_f32_output(bytes: &[u8], output_len: usize, label: &str) -> Result<Vec<f32>, String> {
    let expected_bytes = output_len
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| format!("GRMHD cubecl {label} length overflows bytes"))?;
    if bytes.len() != expected_bytes {
        return Err(format!(
            "GRMHD cubecl {label} readback returned {} bytes, expected {expected_bytes}",
            bytes.len()
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vulkan::canonical_prim2con_and_flux_f32;

    fn fixture_config() -> GrmhdCubeclConfig {
        GrmhdCubeclConfig::new(6, 5, 4, 2.5, 20.0, 0.0, 4.0 / 3.0).unwrap()
    }

    fn parity_config() -> GrmhdCubeclConfig {
        GrmhdCubeclConfig::new(4, 3, 2, 4.0, 12.0, 0.7, 4.0 / 3.0).unwrap()
    }

    fn soa_index(n_total: usize, channel: usize, cell: usize) -> usize {
        channel * n_total + cell
    }

    fn fixture_prims(config: GrmhdCubeclConfig) -> Vec<f32> {
        let n_total = config.n_total();
        let mut prims = vec![0.0f32; NPRIM * n_total];
        for cell in 0..n_total {
            let k = cell % config.n3;
            prims[soa_index(n_total, 0, cell)] = 1.0 + 0.01 * (cell % 7) as f32;
            prims[soa_index(n_total, 1, cell)] = 0.02 + 0.001 * (cell % 5) as f32;
            prims[soa_index(n_total, 2, cell)] = 0.0003 * (cell % 3) as f32;
            prims[soa_index(n_total, 3, cell)] = -0.0002 * (cell % 4) as f32;
            prims[soa_index(n_total, 4, cell)] = 0.0001 * k as f32;
            prims[soa_index(n_total, 5, cell)] = 0.001;
            prims[soa_index(n_total, 6, cell)] = 0.0005;
            prims[soa_index(n_total, 7, cell)] = -0.00025;
        }
        prims
    }

    fn moving_magnetized_prims(config: GrmhdCubeclConfig) -> Vec<f32> {
        let n_total = config.n_total();
        let mut prims = vec![0.0f32; NPRIM * n_total];
        for cell in 0..n_total {
            let i = cell / (config.n2 * config.n3);
            let xi = i as f32 / (config.n1 - 1) as f32;
            let radius = config.r_min * (xi * (config.r_max / config.r_min).ln()).exp();
            let v1 = 0.04 + 0.003 * (cell % 3) as f32;
            let v2 = -0.025 + 0.002 * (cell % 4) as f32;
            let v3 = 0.45 / radius;
            for (channel, value) in [
                1.0 + 0.01 * (cell % 5) as f32,
                0.03 + 0.001 * (cell % 3) as f32,
                v1,
                v2,
                v3,
                2.3 * v1,
                2.3 * v2,
                2.3 * v3,
            ]
            .into_iter()
            .enumerate()
            {
                prims[soa_index(n_total, channel, cell)] = value;
            }
        }
        prims
    }

    fn assert_fp32_parity(expected: &[f32], observed: &[f32], label: &str) {
        assert_eq!(expected.len(), observed.len());
        for (index, (&expected, &observed)) in expected.iter().zip(observed).enumerate() {
            let scale = expected.abs().max(1.0);
            let relative_or_absolute_error = (expected - observed).abs() / scale;
            assert!(
                relative_or_absolute_error < 5.0e-5,
                "{label} mismatch at {index}: CPU={expected}, device={observed}, normalized error={relative_or_absolute_error}"
            );
        }
    }

    #[test]
    fn grmhd_cubecl_device_probe_does_not_panic() {
        let _device_guard = crate::vulkan::GRMHD_DEVICE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let _ = GrmhdCubeclKernel::is_available();
    }

    #[test]
    fn invalid_inputs_are_rejected_before_dispatch() {
        let config = fixture_config();
        let prims = fixture_prims(config);
        assert!(GrmhdCubeclKernel::advance_conserved(config, &prims, 0.0, 1).is_err());
        assert!(GrmhdCubeclKernel::advance_conserved(config, &prims, 0.1, 0).is_err());
        assert!(
            GrmhdCubeclKernel::advance_conserved(config, &prims[..prims.len() - 1], 0.1, 1)
                .is_err()
        );
    }

    #[test]
    fn grmhd_cubecl_prim2con_and_flux_match_canonical_cpu() {
        let _device_guard = crate::vulkan::GRMHD_DEVICE_TEST_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !GrmhdCubeclKernel::is_available() {
            eprintln!("No CubeCL wgpu adapter, skipping GRMHD parity test");
            return;
        }
        let config = parity_config();
        let prims = moving_magnetized_prims(config);
        let (expected_cons, expected_flux) = canonical_prim2con_and_flux_f32(config, &prims);
        let device = WgpuDevice::default();
        let client = WgpuRuntime::client(&device);
        let buffers = CubeclRunBuffers::new(&client, config, &prims).unwrap_or_else(|error| {
            panic!("CubeCL buffers failed on an available device: {error}")
        });
        launch_metric_precompute(&client, &buffers, config, 0.1)
            .unwrap_or_else(|error| panic!("CubeCL metric precompute failed: {error}"));
        let rhs_handle = buffers
            .zero_rhs(&client)
            .unwrap_or_else(|error| panic!("CubeCL RHS allocation failed: {error}"));
        launch_op(
            &client,
            &buffers.prims_handle,
            &buffers.cons_handle,
            &buffers.cons_new_handle,
            &buffers.flux_handle,
            &rhs_handle,
            &buffers.gcov_handle,
            &buffers.sqrt_handle,
            config,
            0.1,
            0,
            OP_PRIM2CON,
            config.n_total(),
        )
        .unwrap_or_else(|error| panic!("CubeCL prim2con failed on an available device: {error}"));
        let observed_cons = decode_f32_output(
            &client.read_one_unchecked(buffers.cons_handle.clone()),
            NCONS * config.n_total(),
            "cons",
        )
        .unwrap_or_else(|error| panic!("CubeCL prim2con readback failed: {error}"));
        assert_fp32_parity(&expected_cons, &observed_cons, "CubeCL prim2con");

        for dir in 0..3u32 {
            launch_op(
                &client,
                &buffers.prims_handle,
                &buffers.cons_handle,
                &buffers.cons_new_handle,
                &buffers.flux_handle,
                &rhs_handle,
                &buffers.gcov_handle,
                &buffers.sqrt_handle,
                config,
                0.1,
                dir,
                OP_COMPUTE_FLUX,
                config.n_total(),
            )
            .unwrap_or_else(|error| {
                panic!("CubeCL direction-{dir} flux failed on an available device: {error}")
            });
            let observed_flux = decode_f32_output(
                &client.read_one_unchecked(buffers.flux_handle.clone()),
                NCONS * config.n_total(),
                "flux",
            )
            .unwrap_or_else(|error| panic!("CubeCL direction-{dir} flux readback failed: {error}"));
            assert_fp32_parity(
                &expected_flux[dir as usize],
                &observed_flux,
                &format!("CubeCL direction-{dir} flux"),
            );
        }
    }
}
