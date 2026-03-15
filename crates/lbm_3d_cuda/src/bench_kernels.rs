// Benchmark kernel runners for extended precision tiers.
// Compiled and launched entirely at runtime (NVRTC), bypassing LbmSolver3DCuda.
// WHY: New precision kernels (FP16, FP8, INT8, DD, TensorCore) are benchmark-only;
//   they should not pollute the production solver's kernel-loading path.
// HOW: Each runner creates a fresh CudaContext, compiles its kernel source via
//   NVRTC, allocates device buffers, and exposes step_n() for timed execution.

use crate::probe_cuda_device_props;
use anyhow::{Context as _, Result};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::CompileOptions,
};
use std::sync::Arc;

// CUDA source for each extended precision kernel tier.
const KERNEL_FP16_SRC: &str = include_str!("kernels_fp16.cu");
const KERNEL_FP8_SRC: &str = include_str!("kernels_fp8.cu");
const KERNEL_INT8_SRC: &str = include_str!("kernels_int8.cu");
const KERNEL_DD_SRC: &str = include_str!("kernels_dd.cu");
const KERNEL_TC_SRC: &str = include_str!("kernels_tensor_core.cu");

fn arch_static() -> &'static str {
    probe_cuda_device_props()
        .map(|p| p.compile_arch())
        .unwrap_or("sm_75")
}

/// Compile a CUDA source string to two kernel functions (step + init).
/// Returns (step_kernel, init_kernel).
fn compile_and_load(
    ctx: &Arc<CudaContext>,
    src: &str,
    cuda_include: bool,
    arch: &'static str,
    step_name: &str,
    init_name: &str,
) -> Result<(CudaFunction, CudaFunction)> {
    let opts = CompileOptions {
        include_paths: if cuda_include {
            vec!["/opt/cuda/include".to_string()]
        } else {
            vec![]
        },
        arch: Some(arch),
        ..Default::default()
    };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(src, opts)?;
    let module = ctx.load_module(ptx)?;
    let step_k = module
        .load_function(step_name)
        .with_context(|| format!("load step kernel '{step_name}'"))?;
    let init_k = module
        .load_function(init_name)
        .with_context(|| format!("load init kernel '{init_name}'"))?;
    Ok((step_k, init_k))
}

fn launch_cfg_1d(n_cells: usize, threads: u32) -> LaunchConfig {
    let blocks = (n_cells as u32).div_ceil(threads).max(1);
    LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

// ============================================================================
// BenchKernelRunner -- FP16 / FP8 / INT8 AoS single-buffer kernels
// ============================================================================

/// Unified benchmark runner for precision tiers with AoS layout:
/// FP16 (2 bytes/dist), FP8 e4m3 (1 byte/dist), INT8 (1 byte/dist).
///
/// All three step kernels share this argument layout (as void pointers on device):
///   (f_in: *byte, f_out: *byte, rho: *f32, u: *f32, force: *f32, tau: *f32,
///    nx: i32, ny: i32, nz: i32)
/// Tau and macroscopic fields are always FP32 regardless of distribution precision.
pub struct BenchKernelRunner {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    d_f_a: CudaSlice<u8>,     // ping distribution buffer (n_cells * 20 * elem_bytes, stride 20)
    d_f_b: CudaSlice<u8>,     // pong distribution buffer (same size)
    d_rho: CudaSlice<f32>,    // n_cells
    d_u: CudaSlice<f32>,      // n_cells * 3
    d_force: CudaSlice<f32>,  // n_cells * 3 (uniform zero)
    d_tau: CudaSlice<f32>,    // n_cells (uniform 0.7)
    step_kernel: CudaFunction,
    nx: i32,
    ny: i32,
    nz: i32,
    pub n_cells: usize,
    pub elem_bytes: usize,
    pub precision_label: &'static str,
    threads_per_block: u32,
}

impl BenchKernelRunner {
    // Private constructor -- argument count reflects distinct kernel configuration axes
    // (source, step/init names, grid dimensions, precision, include flag, thread count).
    // A wrapping struct would not meaningfully reduce complexity here.
    #[allow(clippy::too_many_arguments)]
    fn build(
        src: &str,
        step_kernel_name: &str,
        init_kernel_name: &str,
        nx: usize,
        ny: usize,
        nz: usize,
        elem_bytes: usize,
        precision_label: &'static str,
        cuda_include: bool,
        threads_per_block: u32,
    ) -> Result<Self> {
        let ctx = CudaContext::new(0).context("CUDA device 0 not available")?;
        let stream = ctx.default_stream();
        let arch = arch_static();

        let (step_kernel, init_kernel) =
            compile_and_load(&ctx, src, cuda_include, arch, step_kernel_name, init_kernel_name)?;

        let n_cells = nx * ny * nz;
        // Stride 20: fp16/fp8/int8 kernels use 20-element padded AoS to guarantee
        // 4-byte alignment for half2/uchar4/int32 vectorized loads at every idx.
        let f_bytes = n_cells * 20 * elem_bytes;

        let mut d_f_a = stream.alloc_zeros::<u8>(f_bytes)?;
        let d_f_b = stream.alloc_zeros::<u8>(f_bytes)?;
        let mut d_rho = stream.alloc_zeros::<f32>(n_cells)?;
        let mut d_u = stream.alloc_zeros::<f32>(n_cells * 3)?;
        let d_force = stream.alloc_zeros::<f32>(n_cells * 3)?;
        // Tau: uniform 0.7 FP32
        let mut d_tau: CudaSlice<f32> = stream.clone_htod(&vec![0.7_f32; n_cells])?;

        let (nx_i, ny_i, nz_i) = (nx as i32, ny as i32, nz as i32);
        let rho_init = 1.0_f32;
        let ux_init = 0.0_f32;
        let uy_init = 0.0_f32;
        let uz_init = 0.0_f32;
        let tau_val = 0.7_f32;
        let cfg = launch_cfg_1d(n_cells, threads_per_block);
        {
            let mut b = stream.launch_builder(&init_kernel);
            b.arg(&mut d_f_a)
                .arg(&mut d_rho)
                .arg(&mut d_u)
                .arg(&mut d_tau)
                .arg(&rho_init)
                .arg(&ux_init)
                .arg(&uy_init)
                .arg(&uz_init)
                .arg(&tau_val)
                .arg(&nx_i)
                .arg(&ny_i)
                .arg(&nz_i);
            unsafe { b.launch(cfg) }?;
        }
        ctx.synchronize()?;

        Ok(Self {
            ctx,
            stream,
            d_f_a,
            d_f_b,
            d_rho,
            d_u,
            d_force,
            d_tau,
            step_kernel,
            nx: nx as i32,
            ny: ny as i32,
            nz: nz as i32,
            n_cells,
            elem_bytes,
            precision_label,
            threads_per_block,
        })
    }

    pub fn new_fp16(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        Self::build(
            KERNEL_FP16_SRC,
            "lbm_step_fused_fp16_kernel",
            "initialize_uniform_fp16_kernel",
            nx,
            ny,
            nz,
            2,
            "FP16",
            true,
            1024,
        )
    }

    pub fn new_fp8(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        // FP8 requires SM 8.9 (Ada Lovelace). Skip on older devices.
        let arch = arch_static();
        if !arch.contains("sm_89") && !arch.starts_with("sm_9") {
            anyhow::bail!(
                "FP8 requires SM 8.9+ (Ada Lovelace). Current arch: {arch}"
            );
        }
        Self::build(
            KERNEL_FP8_SRC,
            "lbm_step_fused_fp8_kernel",
            "initialize_uniform_fp8_kernel",
            nx,
            ny,
            nz,
            1,
            "FP8_e4m3",
            true,
            1024,
        )
    }

    pub fn new_int8(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        Self::build(
            KERNEL_INT8_SRC,
            "lbm_step_fused_int8_kernel",
            "initialize_uniform_int8_kernel",
            nx,
            ny,
            nz,
            1,
            "INT8",
            false,
            1024,
        )
    }

    /// Run `n` LBM steps. Alternates d_f_a/d_f_b via std::mem::swap.
    pub fn step_n(&mut self, n: usize) -> Result<()> {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let cfg = launch_cfg_1d(self.n_cells, self.threads_per_block);

        for _ in 0..n {
            let mut b = self.stream.launch_builder(&self.step_kernel);
            b.arg(&self.d_f_a)
                .arg(&mut self.d_f_b)
                .arg(&mut self.d_rho)
                .arg(&mut self.d_u)
                .arg(&self.d_force)
                .arg(&self.d_tau)
                .arg(&nx)
                .arg(&ny)
                .arg(&nz);
            unsafe { b.launch(cfg) }?;
            std::mem::swap(&mut self.d_f_a, &mut self.d_f_b);
        }
        Ok(())
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// VRAM used by distribution buffers (ping + pong) in bytes.
    /// Stride is 20 (padded from 19) for 4-byte alignment of vectorized loads.
    pub fn vram_dist_bytes(&self) -> usize {
        self.n_cells * 20 * self.elem_bytes * 2
    }
}

// ============================================================================
// DdBenchSolver -- Double-Double (FP128 emulation)
// ============================================================================

/// Double-double LBM benchmark solver.
/// Each distribution stored as (hi: f64, lo: f64) -- 16 bytes/value.
/// Layout: i-major SoA. f_hi[i*n_cells + idx] -- coalesced reads for fixed i.
/// Four distribution buffers: f_hi_a, f_lo_a (ping), f_hi_b, f_lo_b (pong).
pub struct DdBenchSolver {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    d_f_hi_a: CudaSlice<u8>,   // 19 * n_cells * 8 bytes (FP64 hi, ping, i-major SoA)
    d_f_lo_a: CudaSlice<u8>,   // 19 * n_cells * 8 bytes (FP64 lo, ping, i-major SoA)
    d_f_hi_b: CudaSlice<u8>,   // pong hi
    d_f_lo_b: CudaSlice<u8>,   // pong lo
    d_rho: CudaSlice<f32>,
    d_u: CudaSlice<f32>,
    d_force: CudaSlice<f32>,
    d_tau: CudaSlice<f32>,
    step_kernel: CudaFunction,
    nx: i32,
    ny: i32,
    nz: i32,
    pub n_cells: usize,
}

impl DdBenchSolver {
    pub fn new(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        let ctx = CudaContext::new(0).context("CUDA device 0 not available")?;
        let stream = ctx.default_stream();
        let arch = arch_static();

        let opts = CompileOptions {
            arch: Some(arch),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNEL_DD_SRC, opts)?;
        let module = ctx.load_module(ptx)?;
        let step_kernel = module.load_function("lbm_step_fused_dd_kernel")?;
        let init_kernel = module.load_function("initialize_uniform_dd_kernel")?;

        let n_cells = nx * ny * nz;
        let f64_buf_bytes = n_cells * 19 * 8; // doubles

        let mut d_f_hi_a = stream.alloc_zeros::<u8>(f64_buf_bytes)?;
        let mut d_f_lo_a = stream.alloc_zeros::<u8>(f64_buf_bytes)?;
        let d_f_hi_b = stream.alloc_zeros::<u8>(f64_buf_bytes)?;
        let d_f_lo_b = stream.alloc_zeros::<u8>(f64_buf_bytes)?;
        let mut d_rho = stream.alloc_zeros::<f32>(n_cells)?;
        let mut d_u = stream.alloc_zeros::<f32>(n_cells * 3)?;
        let d_force = stream.alloc_zeros::<f32>(n_cells * 3)?;
        let mut d_tau: CudaSlice<f32> = stream.clone_htod(&vec![0.7_f32; n_cells])?;

        let (nx_i, ny_i, nz_i) = (nx as i32, ny as i32, nz as i32);
        let rho_init = 1.0_f32;
        let u_init = 0.0_f32;
        let tau_val = 0.7_f32;
        let cfg = launch_cfg_1d(n_cells, 128); // 128 threads: FP64 register pressure
        {
            let mut b = stream.launch_builder(&init_kernel);
            b.arg(&mut d_f_hi_a)
                .arg(&mut d_f_lo_a)
                .arg(&mut d_rho)
                .arg(&mut d_u)
                .arg(&mut d_tau)
                .arg(&rho_init)
                .arg(&u_init)
                .arg(&u_init)
                .arg(&u_init)
                .arg(&tau_val)
                .arg(&nx_i)
                .arg(&ny_i)
                .arg(&nz_i);
            unsafe { b.launch(cfg) }?;
        }
        ctx.synchronize()?;

        Ok(Self {
            ctx,
            stream,
            d_f_hi_a,
            d_f_lo_a,
            d_f_hi_b,
            d_f_lo_b,
            d_rho,
            d_u,
            d_force,
            d_tau,
            step_kernel,
            nx: nx as i32,
            ny: ny as i32,
            nz: nz as i32,
            n_cells,
        })
    }

    /// Run `n` DD LBM steps. Uses 128 threads/block (FP64 register pressure).
    pub fn step_n(&mut self, n: usize) -> Result<()> {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let cfg = launch_cfg_1d(self.n_cells, 128);

        for _ in 0..n {
            let mut b = self.stream.launch_builder(&self.step_kernel);
            b.arg(&self.d_f_hi_a)
                .arg(&self.d_f_lo_a)
                .arg(&mut self.d_f_hi_b)
                .arg(&mut self.d_f_lo_b)
                .arg(&mut self.d_rho)
                .arg(&mut self.d_u)
                .arg(&self.d_force)
                .arg(&self.d_tau)
                .arg(&nx)
                .arg(&ny)
                .arg(&nz);
            unsafe { b.launch(cfg) }?;
            std::mem::swap(&mut self.d_f_hi_a, &mut self.d_f_hi_b);
            std::mem::swap(&mut self.d_f_lo_a, &mut self.d_f_lo_b);
        }
        Ok(())
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// VRAM used by distribution buffers (4 buffers: hi_a, lo_a, hi_b, lo_b) in bytes.
    pub fn vram_dist_bytes(&self) -> usize {
        self.n_cells * 19 * 8 * 4
    }
}

// ============================================================================
// TensorCoreProbe -- WMMA proxy benchmark
// ============================================================================

/// Measures raw Tensor Core throughput (GFLOPS) via WMMA.
/// Not a LBM step -- reports GFLOPS, not MLUPS.
pub struct TensorCoreProbe {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    d_a_tf32: Option<CudaSlice<u8>>,
    d_b_tf32: Option<CudaSlice<u8>>,
    d_c_tf32: Option<CudaSlice<u8>>,
    d_a_fp16: Option<CudaSlice<u8>>,
    d_b_fp16: Option<CudaSlice<u8>>,
    d_c_fp16: Option<CudaSlice<u8>>,
    d_a_int8: Option<CudaSlice<u8>>,
    d_b_int8: Option<CudaSlice<u8>>,
    d_c_int8: Option<CudaSlice<u8>>,
    d_a_int4: Option<CudaSlice<u8>>,
    d_b_int4: Option<CudaSlice<u8>>,
    d_c_int4: Option<CudaSlice<u8>>,
    tf32_kernel: Option<CudaFunction>,
    fp16_kernel: Option<CudaFunction>,
    int8_kernel: Option<CudaFunction>,
    int4_kernel: Option<CudaFunction>,
    n_warps: usize,
}

impl TensorCoreProbe {
    pub fn new() -> Result<Self> {
        let ctx = CudaContext::new(0).context("CUDA device 0 not available")?;
        let stream = ctx.default_stream();
        let arch = arch_static();

        let module_result: anyhow::Result<_> = (|| {
            let opts = CompileOptions {
                include_paths: vec!["/opt/cuda/include".to_string()],
                arch: Some(arch),
                ..Default::default()
            };
            let ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNEL_TC_SRC, opts)?;
            Ok(ctx.load_module(ptx)?)
        })();

        let (tf32_kernel, fp16_kernel, int8_kernel, int4_kernel) = match module_result {
            Ok(ref m) => (
                m.load_function("tensor_core_tf32_proxy").ok(),
                m.load_function("tensor_core_fp16_proxy").ok(),
                m.load_function("tensor_core_int8_proxy").ok(),
                m.load_function("tensor_core_int4_proxy").ok(),
            ),
            Err(ref e) => {
                eprintln!("  [TC] WMMA compile failed ({e}); TC benchmark skipped");
                (None, None, None, None)
            }
        };

        let n_warps = 256usize;

        // Helper closures for typed buffer allocation.
        let alloc_f32_buf = |n: usize| -> Result<CudaSlice<u8>> {
            let bytes: Vec<u8> = vec![1.0_f32; n]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect();
            Ok(stream.clone_htod(&bytes)?)
        };
        let alloc_f16_buf = |n: usize| -> Result<CudaSlice<u8>> {
            // 1.0 in FP16 = 0x3C00
            let bytes: Vec<u8> = std::iter::repeat_n(0x3C00_u16, n)
                .flat_map(|x| x.to_le_bytes())
                .collect();
            Ok(stream.clone_htod(&bytes)?)
        };
        let alloc_ones_i8 = |n: usize| -> Result<CudaSlice<u8>> {
            Ok(stream.clone_htod(&vec![1u8; n])?)
        };
        let alloc_zero_buf = |n: usize| -> Result<CudaSlice<u8>> {
            Ok(stream.alloc_zeros::<u8>(n)?)
        };

        let (d_a_tf32, d_b_tf32, d_c_tf32) = if tf32_kernel.is_some() {
            (
                Some(alloc_f32_buf(16 * 8)?),
                Some(alloc_f32_buf(8 * 16)?),
                Some(alloc_zero_buf(n_warps * 16 * 16 * 4)?),
            )
        } else {
            (None, None, None)
        };
        let (d_a_fp16, d_b_fp16, d_c_fp16) = if fp16_kernel.is_some() {
            (
                Some(alloc_f16_buf(16 * 16)?),
                Some(alloc_f16_buf(16 * 16)?),
                Some(alloc_zero_buf(n_warps * 16 * 16 * 4)?),
            )
        } else {
            (None, None, None)
        };
        let (d_a_int8, d_b_int8, d_c_int8) = if int8_kernel.is_some() {
            (
                Some(alloc_ones_i8(16 * 16)?),
                Some(alloc_ones_i8(16 * 16)?),
                Some(alloc_zero_buf(n_warps * 16 * 16 * 4)?),
            )
        } else {
            (None, None, None)
        };
        let (d_a_int4, d_b_int4, d_c_int4) = if int4_kernel.is_some() {
            (
                Some(alloc_zero_buf(8 * 32 / 8)?),
                Some(alloc_zero_buf(32 * 8 / 8)?),
                Some(alloc_zero_buf(n_warps * 8 * 8 * 4)?),
            )
        } else {
            (None, None, None)
        };

        ctx.synchronize()?;
        Ok(Self {
            ctx,
            stream,
            d_a_tf32,
            d_b_tf32,
            d_c_tf32,
            d_a_fp16,
            d_b_fp16,
            d_c_fp16,
            d_a_int8,
            d_b_int8,
            d_c_int8,
            d_a_int4,
            d_b_int4,
            d_c_int4,
            tf32_kernel,
            fp16_kernel,
            int8_kernel,
            int4_kernel,
            n_warps,
        })
    }

    /// Run TF32 WMMA proxy. Returns total FLOP count (2 * M*N*K * iters * warps).
    pub fn run_tf32(&mut self, n_iters: i32) -> Result<f64> {
        let kernel = self
            .tf32_kernel
            .as_ref()
            .context("TF32 TC kernel not available")?;
        let d_a = self.d_a_tf32.as_ref().context("TF32 A buffer")?;
        let d_b = self.d_b_tf32.as_ref().context("TF32 B buffer")?;
        let d_c = self.d_c_tf32.as_mut().context("TF32 C buffer")?;
        let cfg = LaunchConfig {
            grid_dim: (self.n_warps as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(kernel);
        b.arg(d_a).arg(d_b).arg(d_c).arg(&n_iters);
        unsafe { b.launch(cfg) }?;
        Ok(2.0 * (16 * 16 * 8) as f64 * n_iters as f64 * self.n_warps as f64)
    }

    pub fn run_fp16(&mut self, n_iters: i32) -> Result<f64> {
        let kernel = self
            .fp16_kernel
            .as_ref()
            .context("FP16 TC kernel not available")?;
        let d_a = self.d_a_fp16.as_ref().context("FP16 A")?;
        let d_b = self.d_b_fp16.as_ref().context("FP16 B")?;
        let d_c = self.d_c_fp16.as_mut().context("FP16 C")?;
        let cfg = LaunchConfig {
            grid_dim: (self.n_warps as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(kernel);
        b.arg(d_a).arg(d_b).arg(d_c).arg(&n_iters);
        unsafe { b.launch(cfg) }?;
        Ok(2.0 * (16 * 16 * 16) as f64 * n_iters as f64 * self.n_warps as f64)
    }

    pub fn run_int8(&mut self, n_iters: i32) -> Result<f64> {
        let kernel = self
            .int8_kernel
            .as_ref()
            .context("INT8 TC kernel not available")?;
        let d_a = self.d_a_int8.as_ref().context("INT8 A")?;
        let d_b = self.d_b_int8.as_ref().context("INT8 B")?;
        let d_c = self.d_c_int8.as_mut().context("INT8 C")?;
        let cfg = LaunchConfig {
            grid_dim: (self.n_warps as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(kernel);
        b.arg(d_a).arg(d_b).arg(d_c).arg(&n_iters);
        unsafe { b.launch(cfg) }?;
        Ok(2.0 * (16 * 16 * 16) as f64 * n_iters as f64 * self.n_warps as f64)
    }

    pub fn run_int4(&mut self, n_iters: i32) -> Result<f64> {
        let kernel = self
            .int4_kernel
            .as_ref()
            .context("INT4 TC kernel not available")?;
        let d_a = self.d_a_int4.as_ref().context("INT4 A")?;
        let d_b = self.d_b_int4.as_ref().context("INT4 B")?;
        let d_c = self.d_c_int4.as_mut().context("INT4 C")?;
        let cfg = LaunchConfig {
            grid_dim: (self.n_warps as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(kernel);
        b.arg(d_a).arg(d_b).arg(d_c).arg(&n_iters);
        unsafe { b.launch(cfg) }?;
        Ok(2.0 * (8 * 8 * 32) as f64 * n_iters as f64 * self.n_warps as f64)
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    pub fn has_tf32(&self) -> bool {
        self.tf32_kernel.is_some()
    }
    pub fn has_fp16(&self) -> bool {
        self.fp16_kernel.is_some()
    }
    pub fn has_int8(&self) -> bool {
        self.int8_kernel.is_some()
    }
    pub fn has_int4(&self) -> bool {
        self.int4_kernel.is_some()
    }
}
