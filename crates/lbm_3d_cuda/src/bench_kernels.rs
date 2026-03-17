//! Benchmark kernel runners for extended precision tiers.
//!
//! Each runner is compiled and launched entirely at runtime via NVRTC, bypassing
//! [`crate::LbmSolver3DCuda`]. New precision kernels (FP16, FP8, INT8, DD,
//! TensorCore, INT4, FP4) live here so they do not pollute the production
//! solver's kernel-loading path.
//!
//! # Design
//! Each runner owns a fresh [`cudarc::driver::CudaContext`], compiles its kernel
//! source via NVRTC, allocates device buffers with equilibrium initialization,
//! and exposes `step_n()` for timed execution.
//!
//! # Measured throughput at 128^3 on RTX 4070 Ti (Ada SM 8.9, 504 GB/s peak)
//!
//! | Runner              | MLUPS  | BW% peak | VRAM (MB) | Physics valid? |
//! |---------------------|--------|----------|-----------|----------------|
//! | INT8 SoA            | 5643   | 51.5%    | 76        | Yes (tau>0.51) |
//! | FP8_e4m3 SoA        | 5408   | 49.4%    | 76        | Yes            |
//! | FP8_e5m2 SoA        | 5280   | 48.2%    | 76        | Yes            |
//! | INT4 SoA            | 6148   | 28.1%    | 38        | No (1/36->0)   |
//! | FP4 E2M1            | 4727   | 21.6%    | 38        | No             |
//! | FP16 SoA half2 ILP  | 3803   | 69.4%    | 152       | Yes            |
//! | INT16 SoA           | 3569   | 65.1%    | 152       | Yes            |
//! | FP16 SoA            | 3463   | 63.2%    | 152       | Yes            |
//! | FP16 AoS            | 3459   | 63.1%    | 152       | Yes            |
//! | INT16 AoS           | 3446   | 62.8%    | 152       | Yes            |
//! | BF16 SoA            | 3204   | 58.5%    | 152       | Yes (tau>0.55) |
//! | FP8 e5m2 AoS        | 2931   | 26.8%    | 76        | Yes            |
//! | FP32 SoA CS         | 2062   | 75.4%    | 304       | Yes            |
//! | FP64 SoA            | 406    | 29.7%    | 608       | Yes (reference)|
//! | DD (FP128 emul.)    | 58     | 8.4%     | 1215      | Yes (reference)|
//!
//! CS = cache-streaming stores (`__stcs`, Ada only). Measured +3.1% vs baseline FP32 SoA.

use crate::probe_cuda_device_props;
use anyhow::{Context as _, Result};
use cudarc::{
    driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg},
    nvrtc::CompileOptions,
};
use std::sync::Arc;

// MRT collision device function -- single source of truth.
// Prepend to any kernel source before NVRTC compilation to add MRT support.
#[allow(dead_code)]
pub const MRT_COLLISION_SRC: &str = include_str!("mrt_collision_d3q19.cu");

// On-the-fly 2D slice extraction kernel.
pub const KERNEL_SLICE_SRC: &str = include_str!("kernels_slice.cu");

// CUDA source for each extended precision kernel tier.
const KERNEL_FP16_SRC: &str = include_str!("kernels_fp16.cu");
const KERNEL_FP8_SRC: &str = include_str!("kernels_fp8.cu");
const KERNEL_FP8_E5M2_SRC: &str = include_str!("kernels_fp8_e5m2.cu");
const KERNEL_INT8_SRC: &str = include_str!("kernels_int8.cu");
const KERNEL_INT4_SRC: &str = include_str!("kernels_int4.cu");
const KERNEL_INT16_SRC: &str = include_str!("kernels_int16.cu");
const KERNEL_FP16_SOA_SRC: &str = include_str!("kernels_fp16_soa.cu");
const KERNEL_FP8_SOA_SRC: &str = include_str!("kernels_fp8_soa.cu");
const KERNEL_FP8_E5M2_SOA_SRC: &str = include_str!("kernels_fp8_e5m2_soa.cu");
const KERNEL_INT8_SOA_SRC: &str = include_str!("kernels_int8_soa.cu");
const KERNEL_INT16_SOA_SRC: &str = include_str!("kernels_int16_soa.cu");
const KERNEL_BF16_SOA_SRC: &str = include_str!("kernels_bf16_soa.cu");
const KERNEL_FP64_SOA_SRC: &str = include_str!("kernels_fp64_soa.cu");
const KERNEL_FP32_SOA_CS_SRC: &str = include_str!("kernels_fp32_soa_cs.cu");
const KERNEL_FP16_SOA_HALF2_SRC: &str = include_str!("kernels_fp16_soa_half2.cu");
const KERNEL_FP4_SRC: &str = include_str!("kernels_fp4.cu");
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
    d_f_a: CudaSlice<u8>, // ping distribution buffer (n_cells * 20 * elem_bytes, stride 20)
    d_f_b: CudaSlice<u8>, // pong distribution buffer (same size)
    d_rho: CudaSlice<f32>, // n_cells
    d_u: CudaSlice<f32>,  // n_cells * 3
    d_force: CudaSlice<f32>, // n_cells * 3 (uniform zero)
    d_tau: CudaSlice<f32>, // n_cells (uniform 0.7)
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

        let (step_kernel, init_kernel) = compile_and_load(
            &ctx,
            src,
            cuda_include,
            arch,
            step_kernel_name,
            init_kernel_name,
        )?;

        let n_cells = nx * ny * nz;
        // Stride 20: fp16/fp8/int8 kernels use 20-element padded AoS to guarantee
        // 4-byte alignment for half2/uchar4/int32 vectorized loads at every idx.
        let f_bytes = n_cells * 20 * elem_bytes;

        // VRAM check: estimate total allocation before committing.
        // ping-pong + macro + 1.5 GB NVRTC compilation overhead
        let required_bytes = f_bytes * 2 + n_cells * 32 + 1_500_000_000;
        let free_vram = {
            let mut free: usize = 0;
            let mut total: usize = 0;
            unsafe {
                cudarc::driver::sys::cuMemGetInfo_v2(
                    &mut free as *mut usize,
                    &mut total as *mut usize,
                );
            }
            free
        };
        if free_vram > 0 && required_bytes > (free_vram as f64 * 0.95) as usize {
            anyhow::bail!(
                "VRAM insufficient for {precision_label} at {nx}x{ny}x{nz}: \
                 need {} MB, free {} MB (90% threshold). Skipping.",
                required_bytes / (1024 * 1024),
                free_vram / (1024 * 1024),
            );
        }

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

    /// FP16 AoS stride-20 D3Q19 LBM. 2 bytes/dist. SM 5.0+.
    ///
    /// Measured: 3459 MLUPS at 128^3 on Ada (63.1% of 504 GB/s peak).
    /// 128 threads/block: FP16 register pressure (~152 bytes/thread) prevents
    /// full 1024 threads/block without spill on Ada's 65536-reg SM.
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
            128,
        )
    }

    /// FP8 e4m3 AoS stride-20 D3Q19 LBM. 1 byte/dist. Requires SM 8.9+ (Ada).
    ///
    /// e4m3: 4-bit exponent, 3-bit mantissa, range [-448, 448].
    /// Measured: 5408 MLUPS at 128^3 (SoA variant; AoS measured separately).
    pub fn new_fp8(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        // FP8 requires SM 8.9 (Ada Lovelace). Skip on older devices.
        let arch = arch_static();
        if !arch.contains("sm_89") && !arch.starts_with("sm_9") {
            anyhow::bail!("FP8 requires SM 8.9+ (Ada Lovelace). Current arch: {arch}");
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
            128,
        )
    }

    /// INT8 AoS stride-20 D3Q19 LBM. 1 byte/dist. All SM versions.
    ///
    /// DIST_SCALE=64: range [-2, 2), LSB=0.016. Stable for tau >= 0.51 with
    /// rho near 1. Physics-valid, lowest-VRAM production candidate.
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
            128,
        )
    }

    /// FP8 e5m2 AoS stride-20 D3Q19 LBM. 1 byte/dist. Requires SM 8.9+ (Ada).
    ///
    /// e5m2: 5-bit exponent, 2-bit mantissa. Range ~57344 (4x e4m3 range);
    /// 1-bit less mantissa precision. Measured 2.4% below e4m3 SoA at 128^3
    /// (e5m2 store path marginally slower on Ada). Same VRAM as e4m3.
    pub fn new_fp8_e5m2(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        // FP8 e5m2: same SM 8.9 requirement as e4m3.
        // 5-bit exponent, 2-bit mantissa. Range: ~57344 (4x e4m3 range).
        // Same stride-20 AoS layout and 1-byte element size as e4m3.
        let arch = arch_static();
        if !arch.contains("sm_89") && !arch.starts_with("sm_9") {
            anyhow::bail!("FP8 e5m2 requires SM 8.9+ (Ada Lovelace). Current arch: {arch}");
        }
        Self::build(
            KERNEL_FP8_E5M2_SRC,
            "lbm_step_fused_fp8e5m2_kernel",
            "initialize_uniform_fp8e5m2_kernel",
            nx,
            ny,
            nz,
            1,
            "FP8_e5m2",
            true,
            128,
        )
    }

    /// INT16 AoS stride-20 D3Q19 LBM.
    /// DIST_SCALE=16384 -> range [-2, 2), LSB=6.1e-5 (vs INT8 LSB=0.016).
    /// Better precision than INT8 for moderate-Re flows; same VRAM cost as FP16.
    pub fn new_int16(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        Self::build(
            KERNEL_INT16_SRC,
            "lbm_step_int16_kernel",
            "initialize_uniform_int16_kernel",
            nx,
            ny,
            nz,
            2,
            "INT16",
            false,
            128,
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
// SoaBenchRunner -- i-major SoA pull kernels (FP16/FP8/INT8)
// ============================================================================

/// Benchmark runner for i-major SoA precision tiers with pull-scheme streaming.
/// Covers: FP16 SoA, FP8 e4m3 SoA, INT8 SoA.
///
/// Key difference from BenchKernelRunner (AoS):
///   - Buffer layout: f[i * n_cells + idx] instead of f[idx * 20 + i]
///   - No per-cell padding (stride 19, not 20)
///   - Buffer size: n_cells * 19 * elem_bytes per buffer
///   - Force/velocity fields are SoA (3*n_cells) matching kernels_fp16_soa.cu
pub struct SoaBenchRunner {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    d_f_a: CudaSlice<u8>,
    d_f_b: CudaSlice<u8>,
    d_rho: CudaSlice<f32>,
    d_u: CudaSlice<f32>,
    d_force: CudaSlice<f32>,
    d_tau: CudaSlice<f32>,
    step_kernel: CudaFunction,
    nx: i32,
    ny: i32,
    nz: i32,
    pub n_cells: usize,
    pub elem_bytes: usize,
    pub precision_label: &'static str,
    /// Cells per thread in the step kernel (1 for standard, 2 for half2 ILP).
    cells_per_thread: usize,
}

/// Compile-time parameters for [`SoaBenchRunner::build`].
struct SoaBuildSpec {
    src: &'static str,
    step_kernel_name: &'static str,
    init_kernel_name: &'static str,
    elem_bytes: usize,
    precision_label: &'static str,
    cuda_include: bool,
    /// Cells processed per thread in the step kernel.
    /// 1 for all standard variants; 2 for half2 ILP variant.
    /// Only affects step_n() launch grid -- init always uses 1 thread/cell.
    cells_per_thread: usize,
}

impl SoaBenchRunner {
    fn build(spec: SoaBuildSpec, nx: usize, ny: usize, nz: usize) -> Result<Self> {
        let SoaBuildSpec {
            src,
            step_kernel_name,
            init_kernel_name,
            elem_bytes,
            precision_label,
            cuda_include,
            cells_per_thread,
        } = spec;
        let ctx = CudaContext::new(0).context("CUDA device 0 not available")?;
        let stream = ctx.default_stream();
        let arch = arch_static();

        let (step_kernel, init_kernel) = compile_and_load(
            &ctx,
            src,
            cuda_include,
            arch,
            step_kernel_name,
            init_kernel_name,
        )?;

        let n_cells = nx * ny * nz;
        // i-major SoA: no padding, stride 19 distributions per buffer.
        let f_bytes = n_cells * 19 * elem_bytes;

        // VRAM check: estimate total allocation before committing.
        // 2 * f_bytes (ping-pong) + rho(4*N) + u(12*N) + force(12*N) + tau(4*N) = 2f + 32*N
        let required_bytes = f_bytes * 2 + n_cells * 32;
        let free_vram = {
            let mut free: usize = 0;
            let mut total: usize = 0;
            unsafe {
                cudarc::driver::sys::cuMemGetInfo_v2(
                    &mut free as *mut usize,
                    &mut total as *mut usize,
                );
            }
            free
        };
        if free_vram > 0 && required_bytes > (free_vram as f64 * 0.95) as usize {
            anyhow::bail!(
                "VRAM insufficient for {precision_label} at {nx}x{ny}x{nz}: \
                 need {} MB, free {} MB (90% threshold). Skipping.",
                required_bytes / (1024 * 1024),
                free_vram / (1024 * 1024),
            );
        }

        let mut d_f_a = stream.alloc_zeros::<u8>(f_bytes)?;
        let d_f_b = stream.alloc_zeros::<u8>(f_bytes)?;
        let mut d_rho = stream.alloc_zeros::<f32>(n_cells)?;
        // u_out is SoA: [3 * n_cells] for the SoA kernels.
        let mut d_u = stream.alloc_zeros::<f32>(n_cells * 3)?;
        // force is SoA: [3 * n_cells], all zeros for benchmark.
        let d_force = stream.alloc_zeros::<f32>(n_cells * 3)?;
        let mut d_tau = stream.clone_htod(&vec![0.7_f32; n_cells])?;

        let (nx_i, ny_i, nz_i) = (nx as i32, ny as i32, nz as i32);
        let rho_init = 1.0_f32;
        let u_zero = 0.0_f32;
        let tau_val = 0.7_f32;
        let cfg = launch_cfg_1d(n_cells, 128);
        {
            let mut b = stream.launch_builder(&init_kernel);
            b.arg(&mut d_f_a)
                .arg(&mut d_rho)
                .arg(&mut d_u)
                .arg(&mut d_tau)
                .arg(&rho_init)
                .arg(&u_zero)
                .arg(&u_zero)
                .arg(&u_zero)
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
            cells_per_thread,
        })
    }

    /// FP16 i-major SoA D3Q19 LBM. 2 bytes/dist. Pull scheme, coalesced 128-byte reads.
    ///
    /// Measured: 3463 MLUPS at 128^3 on Ada (63.2% peak). Baseline SoA reference tier.
    pub fn new_fp16_soa(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        Self::build(
            SoaBuildSpec {
                src: KERNEL_FP16_SOA_SRC,
                step_kernel_name: "lbm_step_fp16_soa_kernel",
                init_kernel_name: "initialize_uniform_fp16_soa_kernel",
                elem_bytes: 2,
                precision_label: "FP16_SoA",
                cuda_include: true,
                cells_per_thread: 1,
            },
            nx,
            ny,
            nz,
        )
    }

    /// FP8 e4m3 i-major SoA D3Q19 LBM. 1 byte/dist. Requires SM 8.9+ (Ada).
    ///
    /// Measured: 5408 MLUPS at 128^3 (49.4% peak). Pareto-optimal physics-valid tier
    /// jointly with INT8 SoA. 4x VRAM reduction vs FP32.
    pub fn new_fp8_soa(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        let arch = arch_static();
        if !arch.contains("sm_89") && !arch.starts_with("sm_9") {
            anyhow::bail!("FP8 SoA requires SM 8.9+ (Ada Lovelace). Current arch: {arch}");
        }
        Self::build(
            SoaBuildSpec {
                src: KERNEL_FP8_SOA_SRC,
                step_kernel_name: "lbm_step_fp8_soa_kernel",
                init_kernel_name: "initialize_uniform_fp8_soa_kernel",
                elem_bytes: 1,
                precision_label: "FP8_SoA",
                cuda_include: true,
                cells_per_thread: 1,
            },
            nx,
            ny,
            nz,
        )
    }

    /// INT8 i-major SoA D3Q19 LBM. 1 byte/dist. All SM versions.
    ///
    /// Pareto-optimal physics-valid tier: 5643 MLUPS at 128^3 (51.5% peak, 76 MB VRAM).
    /// DIST_SCALE=64; stable for tau >= 0.51. Preferred tier for VRAM-limited deployments.
    pub fn new_int8_soa(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        Self::build(
            SoaBuildSpec {
                src: KERNEL_INT8_SOA_SRC,
                step_kernel_name: "lbm_step_int8_soa_kernel",
                init_kernel_name: "initialize_uniform_int8_soa_kernel",
                elem_bytes: 1,
                precision_label: "INT8_SoA",
                cuda_include: false,
                cells_per_thread: 1,
            },
            nx,
            ny,
            nz,
        )
    }

    /// INT16 i-major SoA D3Q19 LBM. 2 bytes/dist. All SM versions.
    ///
    /// DIST_SCALE=16384: range [-2, 2), LSB=6.1e-5 (vs INT8 LSB=0.016).
    /// Measured: 3569 MLUPS at 128^3 (+3.0% over FP16 SoA; integer load path avoids
    /// FP16 conversion pipeline). Better numerical precision than INT8 at same VRAM cost.
    pub fn new_int16_soa(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        Self::build(
            SoaBuildSpec {
                src: KERNEL_INT16_SOA_SRC,
                step_kernel_name: "lbm_step_int16_soa_kernel",
                init_kernel_name: "initialize_uniform_int16_soa_kernel",
                elem_bytes: 2,
                precision_label: "INT16_SoA",
                cuda_include: false,
                cells_per_thread: 1,
            },
            nx,
            ny,
            nz,
        )
    }

    /// BF16 i-major SoA D3Q19 LBM. 2 bytes/dist. Requires SM 8.0+ (Ampere/Ada).
    ///
    /// BF16: 8-bit exponent (same as FP32), 7-bit mantissa. Larger dynamic range
    /// than FP16 (good for density-contrast flows) but lower mantissa precision.
    /// Measured: 3204 MLUPS at 128^3 (7.5% *below* FP16 SoA despite equal element size;
    /// Ada SM 8.9 BF16 scalar load latency is higher than FP16 load latency).
    pub fn new_bf16_soa(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        // BF16 requires SM 8.0+ (Ampere and later). All Ada (SM 8.9) qualifies.
        let arch = arch_static();
        if !arch.contains("sm_80")
            && !arch.contains("sm_86")
            && !arch.contains("sm_89")
            && !arch.starts_with("sm_9")
        {
            anyhow::bail!("BF16 SoA requires SM 8.0+ (Ampere or later). Current arch: {arch}");
        }
        Self::build(
            SoaBuildSpec {
                src: KERNEL_BF16_SOA_SRC,
                step_kernel_name: "lbm_step_bf16_soa_kernel",
                init_kernel_name: "initialize_uniform_bf16_soa_kernel",
                elem_bytes: 2,
                precision_label: "BF16_SoA",
                cuda_include: true,
                cells_per_thread: 1,
            },
            nx,
            ny,
            nz,
        )
    }

    /// FP8 e5m2 i-major SoA D3Q19 LBM. 1 byte/dist. Requires SM 8.9+ (Ada).
    ///
    /// e5m2 vs e4m3: wider exponent range (~57344 vs ~448), but 1-bit less mantissa.
    /// Measured: 5280 MLUPS at 128^3 (2.4% below e4m3 SoA). Marginally slower store path on Ada.
    pub fn new_fp8_e5m2_soa(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        let arch = arch_static();
        if !arch.contains("sm_89") && !arch.starts_with("sm_9") {
            anyhow::bail!("FP8 e5m2 SoA requires SM 8.9+ (Ada Lovelace). Current arch: {arch}");
        }
        Self::build(
            SoaBuildSpec {
                src: KERNEL_FP8_E5M2_SOA_SRC,
                step_kernel_name: "lbm_step_fp8_e5m2_soa_kernel",
                init_kernel_name: "initialize_uniform_fp8_e5m2_soa_kernel",
                elem_bytes: 1,
                precision_label: "FP8_e5m2_SoA",
                cuda_include: true,
                cells_per_thread: 1,
            },
            nx,
            ny,
            nz,
        )
    }

    /// FP64 i-major SoA -- primary use is numerical validation / reference.
    /// ~8x VRAM cost vs FP32 SoA; compute-bound on gaming GPUs (~1/64 FP64 throughput).
    pub fn new_fp64_soa(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        Self::build(
            SoaBuildSpec {
                src: KERNEL_FP64_SOA_SRC,
                step_kernel_name: "lbm_step_fp64_soa_kernel",
                init_kernel_name: "initialize_uniform_fp64_soa_kernel",
                elem_bytes: 8,
                precision_label: "FP64_SoA",
                cuda_include: false,
                cells_per_thread: 1,
            },
            nx,
            ny,
            nz,
        )
    }

    /// FP32 i-major SoA with Ada cache-streaming stores (`__stcs`). 4 bytes/dist.
    ///
    /// Ping reads: `__ldg()` (read-only L1 path).
    /// Pong writes: `__stcs()` (PTX `st.global.cs`, L2 evict-first bypass).
    ///
    /// Measured: +3.1% at 128^3 on Ada SM 8.9 vs baseline FP32 SoA.
    /// Limited gain because 304 MB ping buffer >> 48 MB Ada L2; evict-first
    /// only helps when the working set fits in L2 (effective at <= 64^3).
    pub fn new_fp32_cs(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        Self::build(
            SoaBuildSpec {
                src: KERNEL_FP32_SOA_CS_SRC,
                step_kernel_name: "lbm_step_fp32_soa_cs_kernel",
                init_kernel_name: "initialize_uniform_fp32_soa_cs_kernel",
                elem_bytes: 4,
                precision_label: "FP32_SoA_CS",
                cuda_include: false,
                cells_per_thread: 1,
            },
            nx,
            ny,
            nz,
        )
    }

    /// FP16 i-major SoA with half2 ILP: 2 cells per thread.
    /// Uses `__half2` for velocity moment accumulation (2 FP16 FMAs per instruction).
    /// Grid: `ceil(n_cells / 2)` threads -- reflected in step_n() via cells_per_thread=2.
    pub fn new_fp16_half2(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        Self::build(
            SoaBuildSpec {
                src: KERNEL_FP16_SOA_HALF2_SRC,
                step_kernel_name: "lbm_step_fp16_soa_half2_kernel",
                init_kernel_name: "initialize_uniform_fp16_soa_half2_kernel",
                elem_bytes: 2,
                precision_label: "FP16_SoA_H2",
                cuda_include: true,
                cells_per_thread: 2,
            },
            nx,
            ny,
            nz,
        )
    }

    // ====================================================================
    // MRT collision variants -- identical buffer layout and launch config
    // as their BGK counterparts, but dispatch to the MRT kernel name.
    // The MRT kernels live in the same .cu source files as the BGK kernels.
    // ====================================================================

    /// BF16 i-major SoA MRT D3Q19 LBM. 2 bytes/dist. Requires SM 8.0+ (Ampere/Ada).
    pub fn new_bf16_soa_mrt(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        let arch = arch_static();
        if !arch.contains("sm_80")
            && !arch.contains("sm_86")
            && !arch.contains("sm_89")
            && !arch.starts_with("sm_9")
        {
            anyhow::bail!("BF16 SoA MRT requires SM 8.0+ (Ampere or later). Current arch: {arch}");
        }
        Self::build(
            SoaBuildSpec {
                src: KERNEL_BF16_SOA_SRC,
                step_kernel_name: "lbm_step_bf16_soa_mrt_kernel",
                init_kernel_name: "initialize_uniform_bf16_soa_kernel",
                elem_bytes: 2,
                precision_label: "BF16_SoA",
                cuda_include: true,
                cells_per_thread: 1,
            },
            nx,
            ny,
            nz,
        )
    }

    /// FP16 i-major SoA MRT D3Q19 LBM. 2 bytes/dist. SM 5.0+.
    pub fn new_fp16_soa_mrt(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        Self::build(
            SoaBuildSpec {
                src: KERNEL_FP16_SOA_SRC,
                step_kernel_name: "lbm_step_fp16_soa_mrt_kernel",
                init_kernel_name: "initialize_uniform_fp16_soa_kernel",
                elem_bytes: 2,
                precision_label: "FP16_SoA",
                cuda_include: true,
                cells_per_thread: 1,
            },
            nx,
            ny,
            nz,
        )
    }

    /// FP16 i-major SoA half2 ILP MRT D3Q19 LBM. 2 cells/thread. SM 5.0+.
    pub fn new_fp16_half2_mrt(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        Self::build(
            SoaBuildSpec {
                src: KERNEL_FP16_SOA_HALF2_SRC,
                step_kernel_name: "lbm_step_fp16_soa_half2_mrt_kernel",
                init_kernel_name: "initialize_uniform_fp16_soa_half2_kernel",
                elem_bytes: 2,
                precision_label: "FP16_SoA_H2",
                cuda_include: true,
                cells_per_thread: 2,
            },
            nx,
            ny,
            nz,
        )
    }

    /// FP8 e4m3 i-major SoA MRT D3Q19 LBM. 1 byte/dist. Requires SM 8.9+ (Ada).
    pub fn new_fp8_soa_mrt(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        let arch = arch_static();
        if !arch.contains("sm_89") && !arch.starts_with("sm_9") {
            anyhow::bail!("FP8 SoA MRT requires SM 8.9+ (Ada Lovelace). Current arch: {arch}");
        }
        Self::build(
            SoaBuildSpec {
                src: KERNEL_FP8_SOA_SRC,
                step_kernel_name: "lbm_step_fp8_soa_mrt_kernel",
                init_kernel_name: "initialize_uniform_fp8_soa_kernel",
                elem_bytes: 1,
                precision_label: "FP8_SoA",
                cuda_include: true,
                cells_per_thread: 1,
            },
            nx,
            ny,
            nz,
        )
    }

    /// FP8 e5m2 i-major SoA MRT D3Q19 LBM. 1 byte/dist. Requires SM 8.9+ (Ada).
    pub fn new_fp8_e5m2_soa_mrt(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        let arch = arch_static();
        if !arch.contains("sm_89") && !arch.starts_with("sm_9") {
            anyhow::bail!(
                "FP8 e5m2 SoA MRT requires SM 8.9+ (Ada Lovelace). Current arch: {arch}"
            );
        }
        Self::build(
            SoaBuildSpec {
                src: KERNEL_FP8_E5M2_SOA_SRC,
                step_kernel_name: "lbm_step_fp8_e5m2_soa_mrt_kernel",
                init_kernel_name: "initialize_uniform_fp8_e5m2_soa_kernel",
                elem_bytes: 1,
                precision_label: "FP8_e5m2_SoA",
                cuda_include: true,
                cells_per_thread: 1,
            },
            nx,
            ny,
            nz,
        )
    }

    /// INT8 i-major SoA MRT D3Q19 LBM. 1 byte/dist. All SM versions.
    pub fn new_int8_soa_mrt(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        Self::build(
            SoaBuildSpec {
                src: KERNEL_INT8_SOA_SRC,
                step_kernel_name: "lbm_step_int8_soa_mrt_kernel",
                init_kernel_name: "initialize_uniform_int8_soa_kernel",
                elem_bytes: 1,
                precision_label: "INT8_SoA",
                cuda_include: false,
                cells_per_thread: 1,
            },
            nx,
            ny,
            nz,
        )
    }

    /// INT16 i-major SoA MRT D3Q19 LBM. 2 bytes/dist. All SM versions.
    pub fn new_int16_soa_mrt(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        Self::build(
            SoaBuildSpec {
                src: KERNEL_INT16_SOA_SRC,
                step_kernel_name: "lbm_step_int16_soa_mrt_kernel",
                init_kernel_name: "initialize_uniform_int16_soa_kernel",
                elem_bytes: 2,
                precision_label: "INT16_SoA",
                cuda_include: false,
                cells_per_thread: 1,
            },
            nx,
            ny,
            nz,
        )
    }

    /// FP64 i-major SoA MRT D3Q19 LBM. 8 bytes/dist. Reference precision.
    pub fn new_fp64_soa_mrt(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        Self::build(
            SoaBuildSpec {
                src: KERNEL_FP64_SOA_SRC,
                step_kernel_name: "lbm_step_fp64_soa_mrt_kernel",
                init_kernel_name: "initialize_uniform_fp64_soa_kernel",
                elem_bytes: 8,
                precision_label: "FP64_SoA",
                cuda_include: false,
                cells_per_thread: 1,
            },
            nx,
            ny,
            nz,
        )
    }

    /// Run `n` LBM steps, alternating ping/pong buffers.
    pub fn step_n(&mut self, n: usize) -> Result<()> {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        // Half2 variant: ceil(n_cells / 2) threads; standard: n_cells threads.
        let launch_n = self.n_cells.div_ceil(self.cells_per_thread);
        let cfg = launch_cfg_1d(launch_n, 128);

        for _ in 0..n {
            let mut b = self.stream.launch_builder(&self.step_kernel);
            b.arg(&self.d_f_a)
                .arg(&mut self.d_f_b)
                .arg(&mut self.d_rho)
                .arg(&mut self.d_u)
                .arg(&self.d_tau)
                .arg(&self.d_force)
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

    /// Extract a 2D slice of rho and velocity magnitude from the current
    /// distribution buffer on the fly, without materializing the full 3D
    /// macroscopic field. Returns `(rho_slice, vel_mag_slice)` as `Vec<f32>`.
    ///
    /// This enables live viewing of 512^3 INT8 simulations within 12 GB:
    /// only ~2 MB is read back per frame instead of multi-GB macroscopic.
    pub fn read_slice(
        &self,
        slice_axis: i32,
        slice_idx: i32,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let arch = arch_static();
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some(arch),
            ..Default::default()
        };
        let kernel_name = if self.elem_bytes == 1 {
            "read_slice_int8_soa"
        } else {
            "read_slice_fp32_soa"
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNEL_SLICE_SRC, opts)?;
        let module = self.ctx.load_module(ptx)?;
        let slice_kernel = module.load_function(kernel_name)?;

        let (sw, sh) = match slice_axis {
            0 => (self.ny as usize, self.nz as usize),
            1 => (self.nx as usize, self.nz as usize),
            _ => (self.nx as usize, self.ny as usize),
        };
        let slice_size = sw * sh;

        let mut d_rho_slice = self.stream.alloc_zeros::<f32>(slice_size)?;
        let mut d_vel_slice = self.stream.alloc_zeros::<f32>(slice_size)?;

        let cfg = launch_cfg_1d(slice_size, 128);
        {
            let mut b = self.stream.launch_builder(&slice_kernel);
            b.arg(&self.d_f_a)
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

    /// Grid dimensions.
    pub fn grid_dim(&self) -> (usize, usize, usize) {
        (self.nx as usize, self.ny as usize, self.nz as usize)
    }

    /// VRAM used by distribution buffers (ping + pong) in bytes.
    /// SoA uses stride-19 (no padding), so n_cells * 19 * elem_bytes * 2.
    pub fn vram_dist_bytes(&self) -> usize {
        self.n_cells * 19 * self.elem_bytes * 2
    }
}

// ============================================================================
// Int4BenchRunner -- INT4 nibble-packed i-major SoA kernel
// ============================================================================

/// Benchmark runner for INT4 nibble-packed LBM -- bandwidth ceiling test.
///
/// Physics note: INT4 storage quantizes 1/36 weight to 0, corrupting diagonal
/// velocity populations. This runner measures bandwidth ceiling only.
/// Buffer: [19 * half_cells] bytes where half_cells = n_cells / 2.
/// Each thread in the kernel handles 2 cells (nibble pair per byte per direction).
pub struct Int4BenchRunner {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    d_f_a: CudaSlice<u8>,
    d_f_b: CudaSlice<u8>,
    d_rho: CudaSlice<f32>,
    d_u: CudaSlice<f32>,
    d_force: CudaSlice<f32>,
    d_tau: CudaSlice<f32>,
    step_kernel: CudaFunction,
    nx: i32,
    ny: i32,
    nz: i32,
    pub n_cells: usize,
    half_cells: usize,
}

impl Int4BenchRunner {
    /// Construct INT4 benchmark runner for an `nx x ny x nz` grid.
    ///
    /// `n_cells` must be even (power-of-2 grids always satisfy this).
    /// Thread k handles cells `2k` and `2k+1` via nibble pair extraction.
    /// Measured: 6148 MLUPS at 128^3 (bandwidth ceiling only; physics are broken).
    pub fn new(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        assert!(
            (nx * ny * nz).is_multiple_of(2),
            "INT4 kernel requires even n_cells (power-of-2 grids always satisfy this)"
        );
        let ctx = CudaContext::new(0).context("CUDA device 0 not available")?;
        let stream = ctx.default_stream();
        let arch = arch_static();

        let opts = CompileOptions {
            arch: Some(arch),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNEL_INT4_SRC, opts)?;
        let module = ctx.load_module(ptx)?;
        let step_kernel = module.load_function("lbm_step_fused_int4_kernel")?;
        let init_kernel = module.load_function("initialize_uniform_int4_kernel")?;

        let n_cells = nx * ny * nz;
        let half_cells = n_cells / 2;
        // Buffer: 19 nibble-pair bytes per cell-pair.
        let f_bytes = 19 * half_cells;

        let mut d_f_a = stream.alloc_zeros::<u8>(f_bytes)?;
        let d_f_b = stream.alloc_zeros::<u8>(f_bytes)?;
        let mut d_rho = stream.alloc_zeros::<f32>(n_cells)?;
        let mut d_u = stream.alloc_zeros::<f32>(n_cells * 3)?;
        let d_force = stream.alloc_zeros::<f32>(n_cells * 3)?;
        let mut d_tau = stream.clone_htod(&vec![0.7_f32; n_cells])?;

        let (nx_i, ny_i, nz_i) = (nx as i32, ny as i32, nz as i32);
        let rho_init = 1.0_f32;
        let u_zero = 0.0_f32;
        let tau_val = 0.7_f32;
        // One thread per cell-pair (half_cells threads total).
        let cfg = launch_cfg_1d(half_cells, 128);
        {
            let mut b = stream.launch_builder(&init_kernel);
            b.arg(&mut d_f_a)
                .arg(&mut d_rho)
                .arg(&mut d_u)
                .arg(&mut d_tau)
                .arg(&rho_init)
                .arg(&u_zero)
                .arg(&u_zero)
                .arg(&u_zero)
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
            half_cells,
        })
    }

    /// Run `n` INT4 LBM steps.
    pub fn step_n(&mut self, n: usize) -> Result<()> {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let cfg = launch_cfg_1d(self.half_cells, 128);

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

    /// VRAM for distribution buffers (ping + pong): 19 * (n_cells/2) * 2 bytes.
    pub fn vram_dist_bytes(&self) -> usize {
        19 * self.half_cells * 2
    }
}

// ============================================================================
// Fp4BenchRunner -- FP4 E2M1 nibble-packed bandwidth ceiling benchmark
// ============================================================================

/// FP4 E2M1 nibble-packed LBM -- bandwidth ceiling test.
///
/// FP4 E2M1 is NOT physically viable for D3Q19 LBM (rest weight 1/3 quantizes to 0.5,
/// 50% error). This runner measures the bandwidth ceiling assuming Blackwell-class
/// FP4 native decoding, emulated on Ada via the same nibble packing as INT4.
///
/// Buffer: [19 * half_cells] bytes where half_cells = ceil(n_cells / 2).
/// Thread k handles cells 2k and 2k+1 (lo nibble = cell 0, hi nibble = cell 1).
pub struct Fp4BenchRunner {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    d_f_a: CudaSlice<u8>,
    d_f_b: CudaSlice<u8>,
    d_rho: CudaSlice<f32>,
    d_u: CudaSlice<f32>,
    d_force: CudaSlice<f32>,
    d_tau: CudaSlice<f32>,
    step_kernel: CudaFunction,
    nx: i32,
    ny: i32,
    nz: i32,
    pub n_cells: usize,
    half_cells: usize,
}

impl Fp4BenchRunner {
    /// Construct FP4 E2M1 bandwidth ceiling runner for an `nx x ny x nz` grid.
    ///
    /// Thread k handles cells `2k` and `2k+1` via lo/hi nibble extraction.
    /// Half-cell count: `ceil(n_cells / 2)`.
    /// Measured: 4727 MLUPS at 128^3 (bandwidth ceiling; physics broken).
    /// 23% slower than INT4 due to FP4_DECODE lookup table overhead per direction.
    pub fn new(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        let ctx = CudaContext::new(0).context("CUDA device 0 not available")?;
        let stream = ctx.default_stream();
        let arch = arch_static();

        let opts = CompileOptions {
            arch: Some(arch),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNEL_FP4_SRC, opts)?;
        let module = ctx.load_module(ptx)?;
        let step_kernel = module.load_function("lbm_step_fp4_kernel")?;
        let init_kernel = module.load_function("initialize_uniform_fp4_kernel")?;

        let n_cells = nx * ny * nz;
        let half_cells = n_cells.div_ceil(2);
        let f_bytes = 19 * half_cells;

        let mut d_f_a = stream.alloc_zeros::<u8>(f_bytes)?;
        let d_f_b = stream.alloc_zeros::<u8>(f_bytes)?;
        let mut d_rho = stream.alloc_zeros::<f32>(n_cells)?;
        let mut d_u = stream.alloc_zeros::<f32>(n_cells * 3)?;
        let d_force = stream.alloc_zeros::<f32>(n_cells * 3)?;
        let mut d_tau = stream.clone_htod(&vec![0.7_f32; n_cells])?;

        let (nx_i, ny_i, nz_i) = (nx as i32, ny as i32, nz as i32);
        let rho_init = 1.0_f32;
        let u_zero = 0.0_f32;
        let tau_val = 0.7_f32;
        let cfg = launch_cfg_1d(half_cells, 128);
        {
            let mut b = stream.launch_builder(&init_kernel);
            b.arg(&mut d_f_a)
                .arg(&mut d_rho)
                .arg(&mut d_u)
                .arg(&mut d_tau)
                .arg(&rho_init)
                .arg(&u_zero)
                .arg(&u_zero)
                .arg(&u_zero)
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
            half_cells,
        })
    }

    /// Run `n` FP4 LBM steps (bandwidth ceiling, physics broken).
    pub fn step_n(&mut self, n: usize) -> Result<()> {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let cfg = launch_cfg_1d(self.half_cells, 128);

        for _ in 0..n {
            let mut b = self.stream.launch_builder(&self.step_kernel);
            b.arg(&self.d_f_a)
                .arg(&mut self.d_f_b)
                .arg(&mut self.d_rho)
                .arg(&mut self.d_u)
                .arg(&self.d_tau)
                .arg(&self.d_force)
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

    /// VRAM for distribution buffers (ping + pong): 19 * ceil(n_cells/2) * 2 bytes.
    pub fn vram_dist_bytes(&self) -> usize {
        19 * self.half_cells * 2
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
    d_f_hi_a: CudaSlice<u8>, // 19 * n_cells * 8 bytes (FP64 hi, ping, i-major SoA)
    d_f_lo_a: CudaSlice<u8>, // 19 * n_cells * 8 bytes (FP64 lo, ping, i-major SoA)
    d_f_hi_b: CudaSlice<u8>, // pong hi
    d_f_lo_b: CudaSlice<u8>, // pong lo
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
    /// Construct double-double benchmark solver for an `nx x ny x nz` grid.
    ///
    /// Each distribution is stored as `(hi: f64, lo: f64)` -- 16 bytes/value.
    /// Layout: i-major SoA with 4 distribution buffers (hi_a, lo_a, hi_b, lo_b).
    /// Measured: 58 MLUPS at 128^3 (8.4% peak). Use for reference validation only.
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
    d_a_bf16: Option<CudaSlice<u8>>,
    d_b_bf16: Option<CudaSlice<u8>>,
    d_c_bf16: Option<CudaSlice<u8>>,
    tf32_kernel: Option<CudaFunction>,
    fp16_kernel: Option<CudaFunction>,
    int8_kernel: Option<CudaFunction>,
    int4_kernel: Option<CudaFunction>,
    bf16_kernel: Option<CudaFunction>,
    n_warps: usize,
}

impl TensorCoreProbe {
    /// Construct the Tensor Core throughput probe.
    ///
    /// Compiles `kernels_tensor_core.cu` via NVRTC; if compilation fails
    /// (e.g. old CUDA toolkit without WMMA headers), all kernels are set to
    /// `None` and the probe silently skips TC measurements.
    ///
    /// Five WMMA variants are probed: TF32, FP16, INT8, INT4, BF16.
    /// Results are GFLOPS, not MLUPS (Tensor Core WMMA is matrix-multiply,
    /// not LBM step throughput).
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

        let (tf32_kernel, fp16_kernel, int8_kernel, int4_kernel, bf16_kernel) = match module_result
        {
            Ok(ref m) => (
                m.load_function("tensor_core_tf32_proxy").ok(),
                m.load_function("tensor_core_fp16_proxy").ok(),
                m.load_function("tensor_core_int8_proxy").ok(),
                m.load_function("tensor_core_int4_proxy").ok(),
                m.load_function("tensor_core_bf16_proxy").ok(),
            ),
            Err(ref e) => {
                eprintln!("  [TC] WMMA compile failed ({e}); TC benchmark skipped");
                (None, None, None, None, None)
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
        let alloc_ones_i8 =
            |n: usize| -> Result<CudaSlice<u8>> { Ok(stream.clone_htod(&vec![1u8; n])?) };
        let alloc_zero_buf =
            |n: usize| -> Result<CudaSlice<u8>> { Ok(stream.alloc_zeros::<u8>(n)?) };

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
        // BF16: M=16, N=16, K=16 (same shape as FP16 WMMA).
        // BF16 1.0 = 0x3F80 (upper 16 bits of FP32 1.0 = 0x3F800000).
        let alloc_bf16_buf = |n: usize| -> Result<CudaSlice<u8>> {
            let bytes: Vec<u8> = std::iter::repeat_n(0x3F80_u16, n)
                .flat_map(|x| x.to_le_bytes())
                .collect();
            Ok(stream.clone_htod(&bytes)?)
        };
        let (d_a_bf16, d_b_bf16, d_c_bf16) = if bf16_kernel.is_some() {
            (
                Some(alloc_bf16_buf(16 * 16)?),
                Some(alloc_bf16_buf(16 * 16)?),
                // C accumulator is FP32 (4 bytes per element).
                Some(alloc_zero_buf(n_warps * 16 * 16 * 4)?),
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
            d_a_bf16,
            d_b_bf16,
            d_c_bf16,
            tf32_kernel,
            fp16_kernel,
            int8_kernel,
            int4_kernel,
            bf16_kernel,
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

    pub fn run_bf16(&mut self, n_iters: i32) -> Result<f64> {
        let kernel = self
            .bf16_kernel
            .as_ref()
            .context("BF16 TC kernel not available")?;
        let d_a = self.d_a_bf16.as_ref().context("BF16 A")?;
        let d_b = self.d_b_bf16.as_ref().context("BF16 B")?;
        let d_c = self.d_c_bf16.as_mut().context("BF16 C")?;
        let cfg = LaunchConfig {
            grid_dim: (self.n_warps as u32, 1, 1),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut b = self.stream.launch_builder(kernel);
        b.arg(d_a).arg(d_b).arg(d_c).arg(&n_iters);
        unsafe { b.launch(cfg) }?;
        // BF16 WMMA: M=16, N=16, K=16 -- same FMA count as FP16.
        Ok(2.0 * (16 * 16 * 16) as f64 * n_iters as f64 * self.n_warps as f64)
    }

    pub fn has_bf16(&self) -> bool {
        self.bf16_kernel.is_some()
    }
}
