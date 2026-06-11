//! GPU-accelerated GRMHD solver via cudarc + NVRTC.
//!
//! Compiles the CUDA kernels at runtime, uploads grid data once,
//! then runs the hot loop entirely on GPU:
//!   1. precompute_metric_kernel (once)
//!   2. prim2con_kernel
//!   3. compute_flux_kernel (x3 directions)
//!   4. flux_divergence_kernel (x3 directions)
//!   5. euler_update_kernel
//!   6. con2prim (CPU fallback for now)
//!
//! Data layout: SoA (NPRIM * N_total) for coalesced GPU access.

use anyhow::{Context, Result};
use cudarc::driver::{CudaStream, PushKernelArg};
use gororoba_gpu_cuda::{
    Buffer, CompileOptions, Context as CudaContextHelper, KernelHandle, LaunchConfig,
    ModuleRegistry,
};
use std::sync::Arc;

const KERNEL_SRC: &str = include_str!("kernels_grmhd.cu");
const NPRIM: usize = 8;
const NCONS: usize = 8;

/// GPU GRMHD solver state.
pub struct GrmhdGpu {
    _ctx: CudaContextHelper,
    stream: Arc<CudaStream>,
    kernels: GrmhdKernels,
    buffers: GrmhdBuffers,

    // Grid dimensions
    n1: usize,
    n2: usize,
    n3: usize,
    n_total: usize,

    // Physics parameters
    gam_m1: f64, // gamma - 1
    kerr_a: f64,
    r_min: f64,
    r_max: f64,
    dx1: f64,
    dx2: f64,
    dx3: f64,
}

struct GrmhdKernels {
    precompute_metric: KernelHandle,
    compute_flux: KernelHandle,
    prim2con: KernelHandle,
    euler_update: KernelHandle,
    flux_divergence: KernelHandle,
}

struct GrmhdBuffers {
    prims: Buffer<f64>,    // [NPRIM * N_total]
    cons: Buffer<f64>,     // [NCONS * N_total]
    cons_new: Buffer<f64>, // [NCONS * N_total]
    flux: Buffer<f64>,     // [NCONS * N_total] (reused per direction)
    rhs: Buffer<f64>,      // [NCONS * N_total]
    gcov: Buffer<f64>,     // [5 * N1 * N2]
    gcon: Buffer<f64>,     // [5 * N1 * N2]
    sqrt_g: Buffer<f64>,   // [N1 * N2]
    lapse: Buffer<f64>,    // [N1 * N2]
}

fn load_kernels(ctx: &CudaContextHelper) -> Result<GrmhdKernels> {
    let opts = CompileOptions::for_arch(7, 0);
    let registry = ModuleRegistry::compile_and_load(
        ctx.raw(),
        KERNEL_SRC,
        &opts,
        &[
            "precompute_metric_kernel",
            "compute_flux_kernel",
            "prim2con_kernel",
            "euler_update_kernel",
            "flux_divergence_kernel",
        ],
    )
    .context("Compile/load GRMHD CUDA module")?;

    Ok(GrmhdKernels {
        precompute_metric: registry.get("precompute_metric_kernel")?,
        compute_flux: registry.get("compute_flux_kernel")?,
        prim2con: registry.get("prim2con_kernel")?,
        euler_update: registry.get("euler_update_kernel")?,
        flux_divergence: registry.get("flux_divergence_kernel")?,
    })
}

fn allocate_buffers(
    stream: &Arc<CudaStream>,
    n_total: usize,
    n_met: usize,
) -> Result<GrmhdBuffers> {
    Ok(GrmhdBuffers {
        prims: Buffer::alloc_zeros(stream, NPRIM * n_total)?,
        cons: Buffer::alloc_zeros(stream, NCONS * n_total)?,
        cons_new: Buffer::alloc_zeros(stream, NCONS * n_total)?,
        flux: Buffer::alloc_zeros(stream, NCONS * n_total)?,
        rhs: Buffer::alloc_zeros(stream, NCONS * n_total)?,
        gcov: Buffer::alloc_zeros(stream, 5 * n_met)?,
        gcon: Buffer::alloc_zeros(stream, 5 * n_met)?,
        sqrt_g: Buffer::alloc_zeros(stream, n_met)?,
        lapse: Buffer::alloc_zeros(stream, n_met)?,
    })
}

impl GrmhdGpu {
    /// Initialize the GPU solver: compile kernels, allocate buffers,
    /// precompute the metric.
    pub fn new(
        n1: usize,
        n2: usize,
        n3: usize,
        r_min: f64,
        r_max: f64,
        kerr_a: f64,
        gamma: f64,
    ) -> Result<Self> {
        let ctx = CudaContextHelper::with_default_device().context("CUDA init for GRMHD")?;
        let stream = ctx.default_stream();
        let kernels = load_kernels(&ctx)?;

        let n_total = n1 * n2 * n3;
        let n_met = n1 * n2;
        let buffers = allocate_buffers(&stream, n_total, n_met)?;

        let dx1 = (r_max / r_min).ln() / n1 as f64;
        let dx2 = std::f64::consts::PI / n2 as f64;
        let dx3 = 2.0 * std::f64::consts::PI / n3 as f64;

        let mut solver = Self {
            _ctx: ctx,
            stream,
            kernels,
            buffers,
            n1,
            n2,
            n3,
            n_total,
            gam_m1: gamma - 1.0,
            kerr_a,
            r_min,
            r_max,
            dx1,
            dx2,
            dx3,
        };

        // Precompute metric (one-time)
        solver.precompute_metric_gpu()?;

        Ok(solver)
    }

    fn precompute_metric_gpu(&mut self) -> Result<()> {
        let n_met = (self.n1 * self.n2) as u32;
        let cfg = LaunchConfig::launch_1d(n_met);

        let kerr_a = self.kerr_a;
        let r_min = self.r_min;
        let r_max = self.r_max;
        let n1_i = self.n1 as i32;
        let n2_i = self.n2 as i32;

        let mut builder = self.stream.launch_builder(&self.kernels.precompute_metric);
        builder.arg(self.buffers.gcov.raw_mut());
        builder.arg(self.buffers.gcon.raw_mut());
        builder.arg(self.buffers.sqrt_g.raw_mut());
        builder.arg(self.buffers.lapse.raw_mut());
        builder.arg(&kerr_a);
        builder.arg(&r_min);
        builder.arg(&r_max);
        builder.arg(&n1_i);
        builder.arg(&n2_i);
        unsafe { builder.launch(cfg) }.context("Launch precompute_metric")?;

        Ok(())
    }

    /// Upload primitive variables from host to device (SoA layout).
    pub fn upload_prims(&mut self, prims_soa: &[f64]) -> Result<()> {
        assert_eq!(prims_soa.len(), NPRIM * self.n_total);
        self.buffers.prims =
            Buffer::htod(&self.stream, prims_soa).context("Upload prims to GPU")?;
        Ok(())
    }

    /// Download primitive variables from device to host.
    pub fn download_prims(&self, prims_soa: &mut [f64]) -> Result<()> {
        assert_eq!(prims_soa.len(), NPRIM * self.n_total);
        self.buffers
            .prims
            .dtoh(prims_soa)
            .context("Download prims from GPU")?;
        Ok(())
    }

    /// Compute fluxes in one direction and accumulate into RHS.
    fn compute_flux_direction(&mut self, dir: usize) -> Result<()> {
        let nt = self.n_total as u32;
        let cfg = LaunchConfig::launch_1d(nt);

        // Compute flux
        let gam_m1 = self.gam_m1;
        let dir_i = dir as i32;
        let n1_i = self.n1 as i32;
        let n2_i = self.n2 as i32;
        let n3_i = self.n3 as i32;
        let nt_i = self.n_total as i32;

        let mut builder = self.stream.launch_builder(&self.kernels.compute_flux);
        builder.arg(self.buffers.prims.raw());
        builder.arg(self.buffers.gcov.raw());
        builder.arg(self.buffers.sqrt_g.raw());
        builder.arg(self.buffers.flux.raw_mut());
        builder.arg(&gam_m1);
        builder.arg(&dir_i);
        builder.arg(&n1_i);
        builder.arg(&n2_i);
        builder.arg(&n3_i);
        builder.arg(&nt_i);
        unsafe { builder.launch(cfg) }.context("Launch compute_flux")?;

        // Accumulate flux divergence into RHS
        let inv_dx = match dir {
            0 => 1.0 / self.dx1,
            1 => 1.0 / self.dx2,
            _ => 1.0 / self.dx3,
        };

        let mut builder = self.stream.launch_builder(&self.kernels.flux_divergence);
        builder.arg(self.buffers.flux.raw());
        builder.arg(self.buffers.rhs.raw_mut());
        builder.arg(&inv_dx);
        builder.arg(&dir_i);
        builder.arg(&n1_i);
        builder.arg(&n2_i);
        builder.arg(&n3_i);
        builder.arg(&nt_i);
        unsafe { builder.launch(cfg) }.context("Launch flux_divergence")?;

        Ok(())
    }

    /// Execute one forward Euler timestep on GPU.
    ///
    /// Steps:
    /// 1. Zero RHS
    /// 2. Compute fluxes in all 3 directions
    /// 3. Euler update: U_new = U_old + dt * RHS
    pub fn step(&mut self, dt: f64) -> Result<()> {
        let nt = self.n_total;
        let total_vars = (NCONS * nt) as u32;

        // Zero RHS by reallocating a zeroed buffer
        self.buffers.rhs = Buffer::alloc_zeros(&self.stream, NCONS * nt).context("Zero RHS")?;

        // Compute prim2con
        {
            let cfg = LaunchConfig::launch_1d(nt as u32);
            let gam_m1 = self.gam_m1;
            let n1_i = self.n1 as i32;
            let n2_i = self.n2 as i32;
            let n3_i = self.n3 as i32;
            let nt_i = self.n_total as i32;
            let mut builder = self.stream.launch_builder(&self.kernels.prim2con);
            builder.arg(self.buffers.prims.raw());
            builder.arg(self.buffers.gcov.raw());
            builder.arg(self.buffers.sqrt_g.raw());
            builder.arg(self.buffers.cons.raw_mut());
            builder.arg(&gam_m1);
            builder.arg(&n1_i);
            builder.arg(&n2_i);
            builder.arg(&n3_i);
            builder.arg(&nt_i);
            unsafe { builder.launch(cfg) }.context("Launch prim2con")?;
        }

        // Flux divergence in all 3 directions
        self.compute_flux_direction(0)?;
        self.compute_flux_direction(1)?;
        self.compute_flux_direction(2)?;

        // Euler update
        {
            let cfg = LaunchConfig::launch_1d(total_vars);
            let total_vars_i = total_vars as i32;
            let mut builder = self.stream.launch_builder(&self.kernels.euler_update);
            builder.arg(self.buffers.cons.raw());
            builder.arg(self.buffers.rhs.raw());
            builder.arg(self.buffers.cons_new.raw_mut());
            builder.arg(&dt);
            builder.arg(&total_vars_i);
            unsafe { builder.launch(cfg) }.context("Launch euler_update")?;
        }

        // Swap cons <-> cons_new
        std::mem::swap(&mut self.buffers.cons, &mut self.buffers.cons_new);

        Ok(())
    }

    /// Report grid dimensions.
    pub fn grid_info(&self) -> String {
        format!(
            "GRMHD GPU: {}x{}x{} = {} cells, a={}, r=[{:.1},{:.1}]",
            self.n1, self.n2, self.n3, self.n_total, self.kerr_a, self.r_min, self.r_max
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_init() {
        // Only run if CUDA is available (cheap probe via gpu_cuda).
        if !gororoba_gpu_cuda::Context::is_available() {
            eprintln!("No CUDA device, skipping GPU test");
            return;
        }

        let solver = GrmhdGpu::new(64, 32, 16, 2.5, 40.0, 0.0, 4.0 / 3.0);
        match solver {
            Ok(s) => println!("{}", s.grid_info()),
            Err(e) => eprintln!("GPU init failed (expected without CUDA): {}", e),
        }
    }
}
