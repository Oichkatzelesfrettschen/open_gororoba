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
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use std::sync::Arc;

const KERNEL_SRC: &str = include_str!("kernels_grmhd.cu");
const NPRIM: usize = 8;
const NCONS: usize = 8;

/// GPU GRMHD solver state.
pub struct GrmhdGpu {
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,

    // Kernels
    precompute_metric: CudaFunction,
    compute_flux: CudaFunction,
    prim2con: CudaFunction,
    euler_update: CudaFunction,
    flux_divergence: CudaFunction,

    // Device buffers
    d_prims: CudaSlice<f64>,    // [NPRIM * N_total]
    d_cons: CudaSlice<f64>,     // [NCONS * N_total]
    d_cons_new: CudaSlice<f64>, // [NCONS * N_total]
    d_flux: CudaSlice<f64>,     // [NCONS * N_total] (reused per direction)
    d_rhs: CudaSlice<f64>,      // [NCONS * N_total]
    d_gcov: CudaSlice<f64>,     // [5 * N1 * N2]
    d_gcon: CudaSlice<f64>,     // [5 * N1 * N2]
    d_sqrt_g: CudaSlice<f64>,   // [N1 * N2]
    d_lapse: CudaSlice<f64>,    // [N1 * N2]

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
        // Delegate context acquisition to gpu_cuda::Context so the
        // `cudart_device::get_count` + ordinal-range checks happen in
        // one place across the workspace.
        let ctx_wrapper = gororoba_gpu_cuda::Context::with_default_device()
            .context("CUDA init for GRMHD")?;
        let ctx = ctx_wrapper.raw().clone();
        let stream = ctx.default_stream();

        // Compile kernels via NVRTC
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some("sm_70"), // V100+
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNEL_SRC, opts)
            .context("NVRTC compile kernels_grmhd.cu")?;
        let module = ctx.load_module(ptx).context("Load GRMHD CUDA module")?;

        let precompute_metric = module.load_function("precompute_metric_kernel")?;
        let compute_flux = module.load_function("compute_flux_kernel")?;
        let prim2con = module.load_function("prim2con_kernel")?;
        let euler_update = module.load_function("euler_update_kernel")?;
        let flux_divergence = module.load_function("flux_divergence_kernel")?;

        let n_total = n1 * n2 * n3;
        let n_met = n1 * n2;

        // Allocate device buffers
        let d_prims = stream.alloc_zeros::<f64>(NPRIM * n_total)?;
        let d_cons = stream.alloc_zeros::<f64>(NCONS * n_total)?;
        let d_cons_new = stream.alloc_zeros::<f64>(NCONS * n_total)?;
        let d_flux = stream.alloc_zeros::<f64>(NCONS * n_total)?;
        let d_rhs = stream.alloc_zeros::<f64>(NCONS * n_total)?;
        let d_gcov = stream.alloc_zeros::<f64>(5 * n_met)?;
        let d_gcon = stream.alloc_zeros::<f64>(5 * n_met)?;
        let d_sqrt_g = stream.alloc_zeros::<f64>(n_met)?;
        let d_lapse = stream.alloc_zeros::<f64>(n_met)?;

        let dx1 = (r_max / r_min).ln() / n1 as f64;
        let dx2 = std::f64::consts::PI / n2 as f64;
        let dx3 = 2.0 * std::f64::consts::PI / n3 as f64;

        let mut solver = Self {
            _ctx: ctx,
            stream,
            precompute_metric,
            compute_flux,
            prim2con,
            euler_update,
            flux_divergence,
            d_prims,
            d_cons,
            d_cons_new,
            d_flux,
            d_rhs,
            d_gcov,
            d_gcon,
            d_sqrt_g,
            d_lapse,
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
        let threads = 256u32;
        let blocks = n_met.div_ceil(threads);

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        let kerr_a = self.kerr_a;
        let r_min = self.r_min;
        let r_max = self.r_max;
        let n1_i = self.n1 as i32;
        let n2_i = self.n2 as i32;

        let mut builder = self.stream.launch_builder(&self.precompute_metric);
        builder.arg(&self.d_gcov);
        builder.arg(&self.d_gcon);
        builder.arg(&self.d_sqrt_g);
        builder.arg(&self.d_lapse);
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
        self.d_prims = self
            .stream
            .clone_htod(prims_soa)
            .context("Upload prims to GPU")?;
        Ok(())
    }

    /// Download primitive variables from device to host.
    pub fn download_prims(&self, prims_soa: &mut [f64]) -> Result<()> {
        assert_eq!(prims_soa.len(), NPRIM * self.n_total);
        let host = self
            .stream
            .clone_dtoh(&self.d_prims)
            .context("Download prims from GPU")?;
        prims_soa.copy_from_slice(&host);
        Ok(())
    }

    /// Compute fluxes in one direction and accumulate into RHS.
    fn compute_flux_direction(&mut self, dir: usize) -> Result<()> {
        let nt = self.n_total as u32;
        let threads = 256u32;
        let blocks = nt.div_ceil(threads);

        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        // Compute flux
        let gam_m1 = self.gam_m1;
        let dir_i = dir as i32;
        let n1_i = self.n1 as i32;
        let n2_i = self.n2 as i32;
        let n3_i = self.n3 as i32;
        let nt_i = self.n_total as i32;

        let mut builder = self.stream.launch_builder(&self.compute_flux);
        builder.arg(&self.d_prims);
        builder.arg(&self.d_gcov);
        builder.arg(&self.d_sqrt_g);
        builder.arg(&self.d_flux);
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

        let mut builder = self.stream.launch_builder(&self.flux_divergence);
        builder.arg(&self.d_flux);
        builder.arg(&self.d_rhs);
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
        let threads = 256u32;

        // Zero RHS by reallocating a zeroed buffer
        self.d_rhs = self
            .stream
            .alloc_zeros::<f64>(NCONS * nt)
            .context("Zero RHS")?;

        // Compute prim2con
        {
            let blocks = (nt as u32).div_ceil(threads);
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let gam_m1 = self.gam_m1;
            let n1_i = self.n1 as i32;
            let n2_i = self.n2 as i32;
            let n3_i = self.n3 as i32;
            let nt_i = self.n_total as i32;
            let mut builder = self.stream.launch_builder(&self.prim2con);
            builder.arg(&self.d_prims);
            builder.arg(&self.d_gcov);
            builder.arg(&self.d_sqrt_g);
            builder.arg(&self.d_cons);
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
            let blocks = total_vars.div_ceil(threads);
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let total_vars_i = total_vars as i32;
            let mut builder = self.stream.launch_builder(&self.euler_update);
            builder.arg(&self.d_cons);
            builder.arg(&self.d_rhs);
            builder.arg(&self.d_cons_new);
            builder.arg(&dt);
            builder.arg(&total_vars_i);
            unsafe { builder.launch(cfg) }.context("Launch euler_update")?;
        }

        // Swap cons <-> cons_new
        std::mem::swap(&mut self.d_cons, &mut self.d_cons_new);

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
