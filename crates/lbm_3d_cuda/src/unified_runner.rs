//! Unified Memory LBM runner for 1024^3+ grids.
//!
//! Uses `cuMemAllocManaged` for the distribution buffer to page between
//! VRAM and system RAM transparently. The macroscopic fields (rho, u, tau)
//! are not allocated -- this runner uses the ephemeral macro kernel that
//! computes rho/u in registers only.
//!
//! The slice extraction kernel (`read_slice_int8_soa`) runs on-the-fly
//! from the unified distribution buffer.
//!
//! # VRAM Budget at 1024^3 INT8
//!
//! - Distribution buffer (unified): 19 * 1.07B = 20.3 GB (pages to system RAM)
//! - tau field: 4.3 GB (GPU-only, uniform value -> can use constant memory)
//! - Slice output: 2 MB (per-frame transient)
//! - Total GPU-resident: depends on occupancy; ~7-10 GB typical
//!
//! # Limitations
//!
//! - No full macroscopic readback (ephemeral only)
//! - No force field (null-pointer bypass)
//! - Uniform tau only (no per-cell viscosity field)
//! - INT8 precision only

use crate::bench_kernels::KERNEL_SLICE_SRC;
use anyhow::Result;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg, UnifiedSlice,
};
use std::sync::Arc;

/// Minimal INT8 MRT solver using Unified Memory for 1024^3+ grids.
pub struct UnifiedInt8Runner {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    /// Distribution buffer in Unified Memory (pages between VRAM and system RAM).
    d_f: UnifiedSlice<u8>,
    /// Tau field in Unified Memory (uniform value, pages on demand).
    d_tau: UnifiedSlice<f32>,
    /// Step kernel function.
    step_kernel: CudaFunction,
    /// Slice extraction kernel.
    slice_kernel: Option<CudaFunction>,
    /// Grid dimensions.
    pub nx: i32,
    pub ny: i32,
    pub nz: i32,
    n_cells: usize,
    /// Step parity for ping-pong (even though we use single buffer,
    /// the init writes equilibrium distributions).
    step_count: u64,
}

impl UnifiedInt8Runner {
    /// Create a new unified memory INT8 MRT runner.
    ///
    /// The distribution buffer is allocated via `cuMemAllocManaged` and
    /// can exceed GPU VRAM -- the driver pages transparently.
    pub fn new(nx: usize, ny: usize, nz: usize, tau: f32) -> Result<Self> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();

        // Compile INT8 SoA MRT kernel
        let src = include_str!("kernels_int8_soa.cu");
        let arch = crate::probe_cuda_device_props()
            .map(|p| p.compile_arch())
            .unwrap_or("sm_75");
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(arch),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(src, opts)?;
        let module = ctx.load_module(ptx)?;
        // A-A streaming with ephemeral macroscopic fields: single-buffer
        // parity-toggle, rho/u computed in registers only (no global writes).
        // This eliminates the need for 17.2 GB of rho/u buffers at 1024^3.
        let step_kernel = module.load_function("lbm_step_int8_soa_mrt_aa_ephemeral_kernel")?;

        let n_cells = nx * ny * nz;
        let f_bytes = n_cells * 19; // 1 byte per distribution (INT8)

        // Allocate distribution buffer in Unified Memory
        // Safety: the buffer is used exclusively by CUDA kernels (no concurrent
        // host access during kernel execution).
        let mut d_f = unsafe { ctx.alloc_unified::<u8>(f_bytes, true) }?;

        // Prefetch to device for initial write
        d_f.prefetch()?;

        // Tau buffer in unified memory (pages on demand at 1024^3 = 4.3 GB)
        let mut d_tau = unsafe { ctx.alloc_unified::<f32>(n_cells, true) }?;

        // Lightweight init: writes only f and tau (no temporary rho/u buffers).
        // At 1024^3 this saves 17.2 GB of GPU allocations vs the full init kernel.
        let init_kernel = module.load_function("initialize_int8_soa_ephemeral_kernel")?;
        let (nx_i, ny_i, nz_i) = (nx as i32, ny as i32, nz as i32);

        let blocks = (n_cells as u32).div_ceil(128);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        {
            let mut b = stream.launch_builder(&init_kernel);
            b.arg(&mut d_f)
                .arg(&mut d_tau)
                .arg(&tau)
                .arg(&nx_i)
                .arg(&ny_i)
                .arg(&nz_i);
            unsafe { b.launch(cfg) }?;
        }
        ctx.synchronize()?;

        Ok(Self {
            ctx,
            stream,
            d_f,
            d_tau,
            step_kernel,
            slice_kernel: None,
            nx: nx_i,
            ny: ny_i,
            nz: nz_i,
            n_cells,
            step_count: 0,
        })
    }

    /// Run N LBM steps.
    ///
    /// Uses the INT8 MRT kernel with the unified distribution buffer.
    /// The kernel sees the same device pointer -- Unified Memory is
    /// transparently GPU-accessible.
    pub fn step_n(&mut self, n: usize) -> Result<()> {
        let blocks = (self.n_cells as u32).div_ceil(128);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };

        let force_null: u64 = 0; // null pointer -- kernel checks force != NULL

        // A-A streaming with ephemeral macroscopic fields.
        // Kernel signature: f, tau, force, nx, ny, nz, parity
        for _ in 0..n {
            let parity = (self.step_count & 1) as i32;
            let mut b = self.stream.launch_builder(&self.step_kernel);
            b.arg(&mut self.d_f)   // single f buffer (read+write in-place)
                .arg(&self.d_tau)
                .arg(&force_null)    // null force pointer (kernel guards)
                .arg(&self.nx)
                .arg(&self.ny)
                .arg(&self.nz)
                .arg(&parity);
            unsafe { b.launch(cfg) }?;
            self.step_count += 1;
        }
        Ok(())
    }

    /// Extract a 2D slice of rho and velocity magnitude on-the-fly.
    pub fn read_slice(
        &mut self,
        slice_axis: i32,
        slice_idx: i32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        // Lazy-compile slice kernel
        if self.slice_kernel.is_none() {
            let arch = crate::probe_cuda_device_props()
                .map(|p| p.compile_arch())
                .unwrap_or("sm_75");
            let opts = cudarc::nvrtc::CompileOptions {
                arch: Some(arch),
                ..Default::default()
            };
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNEL_SLICE_SRC, opts)?;
            let module = self.ctx.load_module(ptx)?;
            self.slice_kernel = Some(module.load_function("read_slice_int8_soa")?);
        }

        let (sw, sh) = match slice_axis {
            0 => (self.ny as usize, self.nz as usize),
            1 => (self.nx as usize, self.nz as usize),
            _ => (self.nx as usize, self.ny as usize),
        };
        let slice_size = sw * sh;

        let mut d_rho_slice = self.stream.alloc_zeros::<f32>(slice_size)?;
        let mut d_vel_slice = self.stream.alloc_zeros::<f32>(slice_size)?;

        let blocks = (slice_size as u32).div_ceil(128);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        {
            let mut b = self.stream.launch_builder(self.slice_kernel.as_ref().unwrap());
            b.arg(&self.d_f)
                .arg(&mut d_rho_slice)
                .arg(&mut d_vel_slice)
                .arg(&self.nx)
                .arg(&self.ny)
                .arg(&self.nz)
                .arg(&slice_axis)
                .arg(&slice_idx);
            unsafe { b.launch(cfg) }?;
        }

        let rho_slice = self.stream.clone_dtoh(&d_rho_slice)?;
        let vel_slice = self.stream.clone_dtoh(&d_vel_slice)?;
        Ok((rho_slice, vel_slice))
    }

    /// Total distribution buffer size in bytes.
    pub fn dist_bytes(&self) -> usize {
        self.n_cells * 19
    }

    /// Steps executed so far.
    pub fn steps(&self) -> u64 {
        self.step_count
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_unified_runner_struct_size() {
        // Verify the struct exists and is non-zero sized (compile-time check).
        assert!(std::mem::size_of::<super::UnifiedInt8Runner>() > 0);
    }
}
