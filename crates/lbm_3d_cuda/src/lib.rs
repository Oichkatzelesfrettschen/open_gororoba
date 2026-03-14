// GPU-accelerated Lattice Boltzmann Method (D3Q19) with CUDA
// Runtime kernel compilation via cudarc NVRTC

pub mod box_counting_gpu;
pub mod chingon_gpu;

use anyhow::{Context, Result, ensure};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaGraph, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
    PushKernelArg,
};
use std::sync::Arc;

// CudaGraph contains raw *mut pointers that are !Send+!Sync.
// CUDA graph operations are internally synchronized by the runtime
// when invoked from the owning device context, so this is safe.
struct SendSyncGraph(CudaGraph);
// SAFETY: CudaGraph is only used from a single device context and
// the CUDA runtime serializes all graph operations on that context.
unsafe impl Send for SendSyncGraph {}
unsafe impl Sync for SendSyncGraph {}

// Re-export GPU backend selection types for downstream consumers.
pub use gororoba_gpu_bridge::{ComputeBackend, HardwareCaps};

/// Probe whether CUDA is available on this machine.
///
/// Attempts to create a CUDA context on device 0. Returns true on
/// success, false on any error (no driver, no device, incompatible).
pub fn probe_cuda_available() -> bool {
    CudaContext::new(0).is_ok()
}

/// Bit-compatible wrapper for Complex32 to satisfy CUDA traits.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct ComplexDevice {
    pub re: f32,
    pub im: f32,
}

unsafe impl cudarc::driver::DeviceRepr for ComplexDevice {}
unsafe impl cudarc::driver::ValidAsZeroBits for ComplexDevice {}

#[cfg(feature = "cufft")]
use cudarc::cufft::result as cufft;

const KERNEL_SRC: &str = include_str!("kernels.cu");
const KERNEL_BF16_SRC: &str = include_str!("kernels_bf16.cu");
const KERNEL_FP64_SRC: &str = include_str!("kernels_fp64.cu");
const KERNEL_SOA_SRC: &str = include_str!("kernels_soa.cu");

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Precision {
    FP32,
    BF16,
    FP64,
}

/// GPU-accelerated D3Q19 LBM solver with mixed-precision support (FP32/BF16)
pub struct LbmSolver3DCuda {
    nx: usize,
    ny: usize,
    nz: usize,
    n_cells: usize,
    pub precision: Precision,
    _ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    d_f: CudaSlice<u8>,
    d_f_tmp: CudaSlice<u8>,
    d_rho: CudaSlice<u8>,
    d_u: CudaSlice<u8>,
    d_tau: CudaSlice<u8>,
    d_force: CudaSlice<u8>,
    initialize_uniform_kernel: CudaFunction,
    initialize_custom_kernel: CudaFunction,
    lbm_step_fused_kernel: CudaFunction,
    lbm_step_fused_4d_kernel: Option<CudaFunction>,
    enstrophy_cell_kernel: CudaFunction,
    reduce_sum_rho_kernel: CudaFunction,
    reduce_sum_f32_kernel: CudaFunction,
    zero_kernel: CudaFunction,
    apply_mask_kernel: CudaFunction,
    convert_real_to_complex_kernel: CudaFunction,
    convert_complex_to_real_kernel: CudaFunction,
    #[allow(dead_code)]
    update_tau_from_voudon_kernel: CudaFunction,
    // SoA kernel functions (FP32 only, loaded from kernels_soa.cu)
    soa_step_kernel: Option<CudaFunction>,
    soa_init_uniform_kernel: Option<CudaFunction>,
    soa_init_custom_kernel: Option<CudaFunction>,
    soa_smagorinsky_kernel: Option<CudaFunction>,
    /// Shared-memory tiled Smagorinsky kernel (8x8x4 tile + 1-cell halo).
    soa_smagorinsky_tiled_kernel: Option<CudaFunction>,
    #[allow(dead_code)]
    soa_batch_step_kernel: Option<CudaFunction>,
    #[allow(dead_code)]
    soa_batch_init_kernel: Option<CudaFunction>,
    soa_mrt_step_kernel: Option<CudaFunction>,
    /// Thread-coarsened BGK kernel (1 thread = 2 cells, Phase 8).
    soa_coarsened_step_kernel: Option<CudaFunction>,
    /// Thread-coarsened MRT kernel (1 thread = 2 cells, Phase 8).
    soa_mrt_coarsened_step_kernel: Option<CudaFunction>,
    /// GPU max-speed reduction kernel (Phase 12: Mach telemetry).
    reduce_max_speed_kernel: Option<CudaFunction>,
    /// Pull-streaming BGK kernel (Phase 5: coalesced writes).
    soa_pull_step_kernel: Option<CudaFunction>,
    /// Pull-streaming MRT kernel (Phase 5: coalesced writes).
    soa_mrt_pull_step_kernel: Option<CudaFunction>,
    /// Shared-memory tiled BGK kernel (8x8x4 tile + halo, pull-scheme).
    soa_tiled_step_kernel: Option<CudaFunction>,
    /// Shared-memory tiled MRT kernel (8x8x4 tile + halo, pull-scheme).
    soa_mrt_tiled_step_kernel: Option<CudaFunction>,
    /// A-A (Alternating-Address) BGK kernel -- single buffer, no ping-pong.
    soa_aa_step_kernel: Option<CudaFunction>,
    /// A-A MRT kernel -- single buffer, no ping-pong.
    soa_mrt_aa_step_kernel: Option<CudaFunction>,
    /// Whether the solver uses SoA layout (default true for FP32).
    use_soa: bool,
    /// Whether to use MRT collision instead of BGK (FP32 SoA only).
    use_mrt: bool,
    /// Whether to use thread-coarsened kernels (1 thread = 2 cells).
    use_coarsening: bool,
    /// Whether to use pull-streaming (scattered reads, coalesced writes).
    /// Mutually exclusive with coarsening (pull breaks float2 vectorization).
    use_pull_stream: bool,
    /// Whether to use shared-memory tiled kernels (8x8x4 tile).
    /// Highest priority dispatch: tiling > pull > coarsening > standard.
    use_tiling: bool,
    /// Whether to use A-A (Alternating-Address) single-buffer streaming.
    /// Halves VRAM for f by eliminating ping-pong. Incompatible with graph capture.
    use_aa: bool,
    /// A-A parity: toggles 0/1 each step. Even=push-to-opposite, odd=pull-from-opposite.
    aa_parity: i32,
    lbm_block_dim: (u32, u32, u32),
    #[cfg(feature = "cufft")]
    fft_plan: Option<cudarc::cufft::sys::cufftHandle>,
    d_reduction_out: CudaSlice<f32>,
    d_reduction_out_f64: Option<CudaSlice<f64>>,
    d_reduction_buffer_f32: Option<CudaSlice<f32>>,
    d_reduction_buffer_f64: Option<CudaSlice<f64>>,
    pub d_u_hat: Option<CudaSlice<ComplexDevice>>,
    pub d_u_hat_out: Option<CudaSlice<ComplexDevice>>,
    pub rho: Vec<f32>,
    pub u: Vec<[f32; 3]>,
    /// Captured CUDA graph for 2-step pair (A->B->A). Created lazily on
    /// first `step_graph_pair()` call. Amortizes launch overhead for bulk
    /// stepping via `step_n()`.
    step_graph_cache: Option<SendSyncGraph>,
    /// Whether L2 cache pinning is active for the rho field.
    /// When true, re-pin after d_rho reallocation (e.g. initialize_custom).
    l2_pinned: bool,
}

impl LbmSolver3DCuda {
    fn parse_block_dim_env() -> Option<(u32, u32, u32)> {
        let raw = std::env::var("GOROROBA_LBM_BLOCK_DIM").ok()?;
        let cleaned = raw
            .trim()
            .replace(',', "x")
            .replace(' ', "")
            .to_ascii_lowercase();
        let parts: Vec<&str> = cleaned.split('x').filter(|p| !p.is_empty()).collect();
        if parts.len() != 3 {
            return None;
        }
        let bx: u32 = parts[0].parse().ok()?;
        let by: u32 = parts[1].parse().ok()?;
        let bz: u32 = parts[2].parse().ok()?;
        if bx == 0 || by == 0 || bz == 0 {
            return None;
        }
        if (bx as u64) * (by as u64) * (bz as u64) > 1024 {
            return None;
        }
        Some((bx, by, bz))
    }

    /// LaunchConfig safe for FP64 kernels: uses 128 threads/block instead of
    /// cudarc's default 1024. D3Q19 FP64 kernels need ~48+ 32-bit registers
    /// per thread (19 doubles + macroscopic vars); 1024 threads exceeds the
    /// 65536 registers/SM limit on Ada Lovelace.
    fn launch_config_1d(n_elems: u32, precision: Precision) -> LaunchConfig {
        let threads = if precision == Precision::FP64 {
            128u32
        } else {
            1024u32
        };
        let blocks = n_elems.div_ceil(threads).max(1);
        LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// Probe hardware and construct a CUDA solver only if the backend
    /// is available and the problem size warrants GPU dispatch.
    ///
    /// Uses `gororoba_gpu_bridge` for runtime detection: probes CUDA
    /// availability, SIMD capabilities, and problem size threshold.
    /// Returns `Err` if CUDA is not the best backend (caller can fall
    /// back to the CPU solver).
    pub fn detect_and_select(
        nx: usize,
        ny: usize,
        nz: usize,
        tau: f64,
        precision: Precision,
    ) -> Result<Self> {
        let problem_size = nx * ny * nz;
        let simd = gororoba_gpu_bridge::probe_simd();
        let caps = HardwareCaps {
            cuda_available: probe_cuda_available(),
            vulkan_available: false,
            simd,
        };
        let backend = gororoba_gpu_bridge::detect_best_backend(&caps, problem_size);

        ensure!(
            backend == ComputeBackend::Cuda,
            "CUDA not selected: detected backend = {backend:?} \
             (cuda_available={}, problem_size={problem_size}, simd={simd})",
            caps.cuda_available,
        );

        Self::new(nx, ny, nz, tau, precision)
    }

    pub fn new(nx: usize, ny: usize, nz: usize, tau: f64, precision: Precision) -> Result<Self> {
        let n_cells = nx * ny * nz;
        let ctx = CudaContext::new(0).context("CUDA Init Failed")?;
        let stream = ctx.default_stream();
        let src = match precision {
            Precision::FP32 => KERNEL_SRC,
            Precision::BF16 => KERNEL_BF16_SRC,
            Precision::FP64 => KERNEL_FP64_SRC,
        };

        use cudarc::nvrtc::CompileOptions;
        let opts = if precision == Precision::BF16 {
            CompileOptions {
                include_paths: vec!["/opt/cuda/include".to_string()],
                arch: Some("sm_89"),
                ..Default::default()
            }
        } else {
            CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            }
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(src, opts)?;
        let module = ctx.load_module(ptx)?;

        let lbm_step_fused_kernel = module.load_function(match precision {
            Precision::BF16 => "lbm_step_fused_bf16_kernel",
            Precision::FP64 => "lbm_step_fused_fp64_kernel",
            Precision::FP32 => "lbm_step_fused_kernel",
        })?;
        let lbm_step_fused_4d_kernel = if precision == Precision::BF16 {
            Some(module.load_function("lbm_step_fused_bf16_4d_batch_kernel")?)
        } else {
            None
        };
        let initialize_uniform_kernel = module.load_function(match precision {
            Precision::BF16 => "initialize_uniform_bf16_kernel",
            Precision::FP64 => "initialize_uniform_fp64_kernel",
            Precision::FP32 => "initialize_uniform_kernel",
        })?;
        let initialize_custom_kernel = module.load_function(match precision {
            Precision::BF16 => "initialize_custom_bf16_kernel",
            Precision::FP64 => "initialize_custom_fp64_kernel",
            Precision::FP32 => "initialize_custom_kernel",
        })?;
        let enstrophy_cell_kernel = module.load_function(if precision == Precision::FP64 {
            "compute_enstrophy_cell_fp64_kernel"
        } else {
            "compute_enstrophy_cell_kernel"
        })?;
        // FP64: reduce_sum operates on double*, uses reduce_sum_fp64_kernel
        let reduce_sum_rho_kernel = module.load_function(match precision {
            Precision::BF16 => "reduce_sum_bf16_to_f32_kernel",
            Precision::FP64 => "reduce_sum_fp64_kernel",
            Precision::FP32 => "reduce_sum_kernel",
        })?;
        // FP32 reduction kernel -- FP64 module has no reduce_sum_kernel, use fp64 variant
        let reduce_sum_f32_kernel = if precision == Precision::FP64 {
            module.load_function("reduce_sum_fp64_kernel")?
        } else {
            module.load_function("reduce_sum_kernel")?
        };
        let zero_kernel = module.load_function(match precision {
            Precision::BF16 => "zero_f32_kernel",
            Precision::FP64 => "zero_fp64_kernel",
            Precision::FP32 => "zero_kernel",
        })?;
        let apply_mask_kernel = module.load_function(if precision == Precision::FP64 {
            "apply_spectral_mask_fp64_kernel"
        } else {
            "apply_spectral_mask_kernel"
        })?;
        let convert_real_to_complex_kernel = module.load_function(match precision {
            Precision::BF16 => "convert_real_bf16_to_complex_f32_kernel",
            Precision::FP64 => "convert_real_fp64_to_complex_f32_kernel",
            Precision::FP32 => "convert_real_to_complex_kernel",
        })?;
        let convert_complex_to_real_kernel = module.load_function(match precision {
            Precision::BF16 => "convert_complex_f32_to_real_bf16_kernel",
            Precision::FP64 => "convert_complex_f32_to_real_fp64_kernel",
            Precision::FP32 => "convert_complex_to_real_kernel",
        })?;
        let update_tau_from_voudon_kernel =
            module.load_function("update_tau_from_voudon_frustration_kernel")?;

        // Compile SoA kernels for FP32 (production path with coalesced memory)
        let (
            soa_step_kernel,
            soa_init_uniform_kernel,
            soa_init_custom_kernel,
            soa_smagorinsky_kernel,
            soa_smagorinsky_tiled_kernel,
            soa_batch_step_kernel,
            soa_batch_init_kernel,
            soa_mrt_step_kernel,
            soa_coarsened_step_kernel,
            soa_mrt_coarsened_step_kernel,
            reduce_max_speed_kernel,
            soa_pull_step_kernel,
            soa_mrt_pull_step_kernel,
            soa_tiled_step_kernel,
            soa_mrt_tiled_step_kernel,
            soa_aa_step_kernel,
            soa_mrt_aa_step_kernel,
        ) = if precision == Precision::FP32 {
            let soa_opts = cudarc::nvrtc::CompileOptions {
                arch: Some("sm_89"),
                ..Default::default()
            };
            let soa_ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNEL_SOA_SRC, soa_opts)?;
            let soa_module = ctx.load_module(soa_ptx)?;
            (
                Some(soa_module.load_function("lbm_step_soa_fused")?),
                Some(soa_module.load_function("initialize_uniform_soa_kernel")?),
                Some(soa_module.load_function("initialize_custom_soa_kernel")?),
                Some(soa_module.load_function("compute_smagorinsky_tau_kernel")?),
                Some(soa_module.load_function("compute_smagorinsky_tau_tiled")?),
                Some(soa_module.load_function("lbm_step_soa_batch_kernel")?),
                Some(soa_module.load_function("initialize_custom_soa_batch_kernel")?),
                Some(soa_module.load_function("lbm_step_soa_mrt_fused")?),
                Some(soa_module.load_function("lbm_step_soa_coarsened")?),
                Some(soa_module.load_function("lbm_step_soa_mrt_coarsened")?),
                Some(soa_module.load_function("reduce_max_speed_f32")?),
                Some(soa_module.load_function("lbm_step_soa_pull")?),
                Some(soa_module.load_function("lbm_step_soa_mrt_pull")?),
                Some(soa_module.load_function("lbm_step_soa_tiled")?),
                Some(soa_module.load_function("lbm_step_soa_mrt_tiled")?),
                Some(soa_module.load_function("lbm_step_soa_aa")?),
                Some(soa_module.load_function("lbm_step_soa_mrt_aa")?),
            )
        } else {
            (
                None, None, None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None,
            )
        };
        let use_soa = precision == Precision::FP32;

        let es = match precision {
            Precision::FP32 => 4,
            Precision::BF16 => 2,
            Precision::FP64 => 8,
        };
        let mut d_f = stream.alloc_zeros::<u8>(19 * n_cells * es)?;
        let d_f_tmp = stream.alloc_zeros::<u8>(19 * n_cells * es)?;
        let mut d_rho = stream.alloc_zeros::<u8>(n_cells * es)?;
        let mut d_u = stream.alloc_zeros::<u8>(3 * n_cells * es)?;
        let d_force = stream.alloc_zeros::<u8>(3 * n_cells * es)?;
        let d_tau = match precision {
            Precision::FP32 => {
                let v = vec![tau as f32; n_cells];
                let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                stream.clone_htod(&b)?
            }
            Precision::BF16 => {
                let v = vec![half::bf16::from_f32(tau as f32).to_bits(); n_cells];
                let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                stream.clone_htod(&b)?
            }
            Precision::FP64 => {
                let v = vec![tau; n_cells];
                let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                stream.clone_htod(&b)?
            }
        };

        let d_reduction_out = stream.alloc_zeros::<f32>(1)?;
        let d_reduction_out_f64 = if precision == Precision::FP64 {
            Some(stream.alloc_zeros::<f64>(1)?)
        } else {
            None
        };
        let d_reduction_buffer_f32 = Some(stream.alloc_zeros::<f32>(n_cells)?);
        let d_reduction_buffer_f64 = if precision == Precision::FP64 {
            Some(stream.alloc_zeros::<f64>(n_cells)?)
        } else {
            None
        };
        let d_u_hat = Some(stream.alloc_zeros::<ComplexDevice>(n_cells)?);
        let d_u_hat_out = Some(stream.alloc_zeros::<ComplexDevice>(n_cells)?);

        // Initialize: FP64 kernel takes double args, FP32/BF16 take float args
        let (nx_i, ny_i, nz_i) = (nx as i32, ny as i32, nz as i32);
        if precision == Precision::FP64 {
            let rho_init = 1.0_f64;
            let u_init = 0.0_f64;
            let mut init = stream.launch_builder(&initialize_uniform_kernel);
            init.arg(&mut d_f)
                .arg(&mut d_rho)
                .arg(&mut d_u)
                .arg(&rho_init)
                .arg(&u_init)
                .arg(&u_init)
                .arg(&u_init)
                .arg(&nx_i)
                .arg(&ny_i)
                .arg(&nz_i);
            unsafe { init.launch(Self::launch_config_1d(n_cells as u32, precision)) }?;
        } else if use_soa {
            // SoA uniform init for FP32 (writes f/rho/u in SoA layout)
            let soa_kernel = soa_init_uniform_kernel
                .as_ref()
                .context("SoA uniform init kernel not loaded during new()")?;
            let rho_init = 1.0_f32;
            let u_init = 0.0_f32;
            let threads = 128u32;
            let blocks = (n_cells as u32).div_ceil(threads);
            let config = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut init = stream.launch_builder(soa_kernel);
            init.arg(&mut d_f)
                .arg(&mut d_rho)
                .arg(&mut d_u)
                .arg(&rho_init)
                .arg(&u_init)
                .arg(&u_init)
                .arg(&u_init)
                .arg(&nx_i)
                .arg(&ny_i)
                .arg(&nz_i);
            unsafe { init.launch(config) }?;
        } else {
            let rho_init = 1.0_f32;
            let u_init = 0.0_f32;
            let mut init = stream.launch_builder(&initialize_uniform_kernel);
            init.arg(&mut d_f)
                .arg(&mut d_rho)
                .arg(&mut d_u)
                .arg(&rho_init)
                .arg(&u_init)
                .arg(&u_init)
                .arg(&u_init)
                .arg(&nx_i)
                .arg(&ny_i)
                .arg(&nz_i);
            unsafe { init.launch(Self::launch_config_1d(n_cells as u32, precision)) }?;
        }

        // Default kernel geometry. Override with:
        //   GOROROBA_LBM_BLOCK_DIM=8x4x4 (or 8,4,4)
        let default_block_dim = (4u32, 4u32, 4u32);
        let lbm_block_dim = Self::parse_block_dim_env().unwrap_or(default_block_dim);

        Ok(Self {
            nx,
            ny,
            nz,
            n_cells,
            precision,
            _ctx: ctx,
            stream,
            d_f,
            d_f_tmp,
            d_rho,
            d_u,
            d_tau,
            d_force,
            initialize_uniform_kernel,
            initialize_custom_kernel,
            lbm_step_fused_kernel,
            lbm_step_fused_4d_kernel,
            enstrophy_cell_kernel,
            reduce_sum_rho_kernel,
            reduce_sum_f32_kernel,
            zero_kernel,
            apply_mask_kernel,
            convert_real_to_complex_kernel,
            convert_complex_to_real_kernel,
            update_tau_from_voudon_kernel,
            soa_step_kernel,
            soa_init_uniform_kernel,
            soa_init_custom_kernel,
            soa_smagorinsky_kernel,
            soa_smagorinsky_tiled_kernel,
            soa_batch_step_kernel,
            soa_batch_init_kernel,
            soa_mrt_step_kernel,
            soa_coarsened_step_kernel,
            soa_mrt_coarsened_step_kernel,
            reduce_max_speed_kernel,
            soa_pull_step_kernel,
            soa_mrt_pull_step_kernel,
            soa_tiled_step_kernel,
            soa_mrt_tiled_step_kernel,
            soa_aa_step_kernel,
            soa_mrt_aa_step_kernel,
            use_soa,
            use_mrt: false,
            use_coarsening: false,
            use_pull_stream: false,
            use_tiling: false,
            use_aa: false,
            aa_parity: 0,
            lbm_block_dim,
            #[cfg(feature = "cufft")]
            fft_plan: None,
            d_reduction_out,
            d_reduction_out_f64,
            d_reduction_buffer_f32,
            d_reduction_buffer_f64,
            d_u_hat,
            d_u_hat_out,
            rho: vec![1.0; n_cells],
            u: vec![[0.0; 3]; n_cells],
            step_graph_cache: None,
            l2_pinned: false,
        })
    }

    /// Create an MRT (Multiple-Relaxation-Time) GPU solver.
    ///
    /// Identical to `new()` but dispatches the d'Humieres MRT collision kernel
    /// instead of BGK. MRT extends the practical Mach stability limit from ~0.3
    /// to ~1.5, enabling lower density floors and higher dynamic range.
    ///
    /// MRT is only supported for FP32 precision (the SoA production path).
    pub fn new_mrt(
        nx: usize,
        ny: usize,
        nz: usize,
        tau: f64,
        precision: Precision,
    ) -> Result<Self> {
        ensure!(
            precision == Precision::FP32,
            "MRT collision is only supported for FP32 (SoA) precision"
        );
        let mut solver = Self::new(nx, ny, nz, tau, precision)?;
        solver.use_mrt = true;
        Ok(solver)
    }

    fn encode_f32_to_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
    }

    fn encode_bf16_to_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .map(|v| half::bf16::from_f32(*v).to_bits())
            .flat_map(|bits| bits.to_le_bytes())
            .collect::<Vec<u8>>()
    }

    fn encode_f64_to_bytes(values: &[f64]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()
    }

    pub fn step_4d(&mut self, nw: usize) -> Result<()> {
        let (nx, ny, nz_total) = (self.nx as i32, self.ny as i32, self.nz as i32);
        let nw_i = nw as i32;
        ensure!(nw > 0, "nw must be > 0");
        let nz_sub = nz_total / nw_i;
        ensure!(
            nz_total % nw_i == 0,
            "Total Z dimension {} must be divisible by nw {}",
            nz_total,
            nw
        );

        let kernel = self
            .lbm_step_fused_4d_kernel
            .as_ref()
            .context("4D kernel not available (requires BF16)")?;

        let (bx, by, bz) = self.lbm_block_dim;
        let config = LaunchConfig {
            grid_dim: (
                self.nx.div_ceil(bx as usize) as u32,
                self.ny.div_ceil(by as usize) as u32,
                self.nz.div_ceil(bz as usize) as u32,
            ),
            block_dim: (bx, by, bz),
            shared_mem_bytes: 0,
        };

        let mut b = self.stream.launch_builder(kernel);
        b.arg(&self.d_f)
            .arg(&mut self.d_f_tmp)
            .arg(&mut self.d_rho)
            .arg(&mut self.d_u)
            .arg(&self.d_force)
            .arg(&self.d_tau)
            .arg(&nx)
            .arg(&ny)
            .arg(&nz_sub)
            .arg(&nw_i);
        unsafe { b.launch(config) }?;
        std::mem::swap(&mut self.d_f, &mut self.d_f_tmp);
        Ok(())
    }

    pub fn initialize_uniform(&mut self, rho: f32, u: [f32; 3]) -> Result<()> {
        let (nx_i, ny_i, nz_i) = (self.nx as i32, self.ny as i32, self.nz as i32);
        if self.precision == Precision::FP64 {
            // FP64 kernel expects double args -- widen f32 to f64
            let rho_d = rho as f64;
            let (ux_d, uy_d, uz_d) = (u[0] as f64, u[1] as f64, u[2] as f64);
            let mut init = self.stream.launch_builder(&self.initialize_uniform_kernel);
            init.arg(&mut self.d_f)
                .arg(&mut self.d_rho)
                .arg(&mut self.d_u)
                .arg(&rho_d)
                .arg(&ux_d)
                .arg(&uy_d)
                .arg(&uz_d)
                .arg(&nx_i)
                .arg(&ny_i)
                .arg(&nz_i);
            unsafe { init.launch(Self::launch_config_1d(self.n_cells as u32, self.precision)) }?;
        } else if self.use_soa {
            // SoA uniform init: same signature, writes f/rho/u in SoA layout
            let soa_kernel = self
                .soa_init_uniform_kernel
                .as_ref()
                .context("SoA uniform init kernel not loaded")?;
            let rho_init = rho;
            let (ux, uy, uz) = (u[0], u[1], u[2]);
            let threads = 128u32;
            let blocks = (self.n_cells as u32).div_ceil(threads);
            let config = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut init = self.stream.launch_builder(soa_kernel);
            init.arg(&mut self.d_f)
                .arg(&mut self.d_rho)
                .arg(&mut self.d_u)
                .arg(&rho_init)
                .arg(&ux)
                .arg(&uy)
                .arg(&uz)
                .arg(&nx_i)
                .arg(&ny_i)
                .arg(&nz_i);
            unsafe { init.launch(config) }?;
        } else {
            let rho_init = rho;
            let (ux, uy, uz) = (u[0], u[1], u[2]);
            let mut init = self.stream.launch_builder(&self.initialize_uniform_kernel);
            init.arg(&mut self.d_f)
                .arg(&mut self.d_rho)
                .arg(&mut self.d_u)
                .arg(&rho_init)
                .arg(&ux)
                .arg(&uy)
                .arg(&uz)
                .arg(&nx_i)
                .arg(&ny_i)
                .arg(&nz_i);
            unsafe { init.launch(Self::launch_config_1d(self.n_cells as u32, self.precision)) }?;
        }
        Ok(())
    }

    pub fn initialize_custom(&mut self, rho: &[f64], u: &[[f64; 3]]) -> Result<()> {
        ensure!(
            rho.len() == self.n_cells,
            "rho length mismatch: got {}, expected {}",
            rho.len(),
            self.n_cells
        );
        ensure!(
            u.len() == self.n_cells,
            "u length mismatch: got {}, expected {}",
            u.len(),
            self.n_cells
        );

        if self.precision == Precision::FP64 {
            // FP64: store doubles directly
            let mut u_flat = Vec::with_capacity(self.n_cells * 3);
            for v in u {
                u_flat.push(v[0]);
                u_flat.push(v[1]);
                u_flat.push(v[2]);
            }
            let rho_bytes = Self::encode_f64_to_bytes(rho);
            let u_bytes = Self::encode_f64_to_bytes(&u_flat);
            let d_rho_in = self.stream.clone_htod(&rho_bytes)?;
            let d_u_in = self.stream.clone_htod(&u_bytes)?;

            let (nx_i, ny_i, nz_i) = (self.nx as i32, self.ny as i32, self.nz as i32);
            let mut init = self.stream.launch_builder(&self.initialize_custom_kernel);
            init.arg(&mut self.d_f)
                .arg(&mut self.d_rho)
                .arg(&mut self.d_u)
                .arg(&d_rho_in)
                .arg(&d_u_in)
                .arg(&nx_i)
                .arg(&ny_i)
                .arg(&nz_i);
            unsafe { init.launch(Self::launch_config_1d(self.n_cells as u32, self.precision)) }?;
        } else {
            let mut rho_flat = Vec::with_capacity(self.n_cells);
            for &v in rho {
                rho_flat.push(v as f32);
            }
            // Velocity always uploaded as AoS (both SoA and AoS init kernels
            // accept AoS input -- the SoA kernel transposes internally)
            let mut u_flat = Vec::with_capacity(self.n_cells * 3);
            for v in u {
                u_flat.push(v[0] as f32);
                u_flat.push(v[1] as f32);
                u_flat.push(v[2] as f32);
            }
            let rho_bytes = match self.precision {
                Precision::FP32 => Self::encode_f32_to_bytes(&rho_flat),
                Precision::BF16 => Self::encode_bf16_to_bytes(&rho_flat),
                Precision::FP64 => unreachable!("FP64 handled in outer branch"),
            };
            let u_bytes = match self.precision {
                Precision::FP32 => Self::encode_f32_to_bytes(&u_flat),
                Precision::BF16 => Self::encode_bf16_to_bytes(&u_flat),
                Precision::FP64 => unreachable!("FP64 handled in outer branch"),
            };
            let d_rho_in = self.stream.clone_htod(&rho_bytes)?;
            let d_u_in = self.stream.clone_htod(&u_bytes)?;

            let (nx_i, ny_i, nz_i) = (self.nx as i32, self.ny as i32, self.nz as i32);

            if self.use_soa {
                // SoA custom init: kernel reads AoS input, writes SoA on device
                let soa_kernel = self
                    .soa_init_custom_kernel
                    .as_ref()
                    .context("SoA custom init kernel not loaded")?;
                let threads = 128u32;
                let blocks = (self.n_cells as u32).div_ceil(threads);
                let config = LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut init = self.stream.launch_builder(soa_kernel);
                init.arg(&mut self.d_f)
                    .arg(&mut self.d_rho)
                    .arg(&mut self.d_u)
                    .arg(&d_rho_in)
                    .arg(&d_u_in)
                    .arg(&nx_i)
                    .arg(&ny_i)
                    .arg(&nz_i);
                unsafe { init.launch(config) }?;
            } else {
                let mut init = self.stream.launch_builder(&self.initialize_custom_kernel);
                init.arg(&mut self.d_f)
                    .arg(&mut self.d_rho)
                    .arg(&mut self.d_u)
                    .arg(&d_rho_in)
                    .arg(&d_u_in)
                    .arg(&nx_i)
                    .arg(&ny_i)
                    .arg(&nz_i);
                unsafe {
                    init.launch(Self::launch_config_1d(self.n_cells as u32, self.precision))
                }?;
            }
        }
        Ok(())
    }

    pub fn update_tau_from_voudon(
        &mut self,
        frustration: &CudaSlice<f32>,
        tau_base: f32,
        alpha_voudon: f32,
    ) -> Result<()> {
        let n_cells = self.n_cells as u32;
        let n_cells_i = n_cells as i32;
        let mut b = self
            .stream
            .launch_builder(&self.update_tau_from_voudon_kernel);
        b.arg(&mut self.d_tau)
            .arg(frustration)
            .arg(&tau_base)
            .arg(&alpha_voudon)
            .arg(&n_cells_i);

        unsafe { b.launch(Self::launch_config_1d(n_cells, self.precision)) }?;
        Ok(())
    }

    pub fn set_force_field(&mut self, force_field: &[[f64; 3]]) -> Result<()> {
        ensure!(
            force_field.len() == self.n_cells,
            "force field length mismatch: got {}, expected {}",
            force_field.len(),
            self.n_cells
        );

        if self.precision == Precision::FP64 {
            let mut force_flat = Vec::with_capacity(self.n_cells * 3);
            for v in force_field {
                force_flat.push(v[0]);
                force_flat.push(v[1]);
                force_flat.push(v[2]);
            }
            let bytes = Self::encode_f64_to_bytes(&force_flat);
            self.d_force = self.stream.clone_htod(&bytes)?;
        } else if self.use_soa {
            // SoA force layout: [fx_0..fx_N, fy_0..fy_N, fz_0..fz_N]
            let n = self.n_cells;
            let mut force_soa = vec![0.0f32; n * 3];
            for (i, v) in force_field.iter().enumerate() {
                force_soa[i] = v[0] as f32; // fx at offset 0*N+i
                force_soa[n + i] = v[1] as f32; // fy at offset 1*N+i
                force_soa[2 * n + i] = v[2] as f32; // fz at offset 2*N+i
            }
            let bytes = Self::encode_f32_to_bytes(&force_soa);
            self.d_force = self.stream.clone_htod(&bytes)?;
        } else {
            let mut force_flat = Vec::with_capacity(self.n_cells * 3);
            for v in force_field {
                force_flat.push(v[0] as f32);
                force_flat.push(v[1] as f32);
                force_flat.push(v[2] as f32);
            }
            let bytes = match self.precision {
                Precision::FP32 => Self::encode_f32_to_bytes(&force_flat),
                Precision::BF16 => Self::encode_bf16_to_bytes(&force_flat),
                Precision::FP64 => unreachable!("FP64 handled in outer branch"),
            };
            self.d_force = self.stream.clone_htod(&bytes)?;
        }
        Ok(())
    }

    pub fn set_viscosity_field(&mut self, viscosity_field: &[f64]) -> Result<()> {
        ensure!(
            viscosity_field.len() == self.n_cells,
            "viscosity field length mismatch: got {}, expected {}",
            viscosity_field.len(),
            self.n_cells
        );

        // LBM lattice units (D3Q19 BGK): nu = (tau - 0.5) / 3  =>  tau = 0.5 + 3*nu
        if self.precision == Precision::FP64 {
            let tau_flat: Vec<f64> = viscosity_field.iter().map(|&nu| 0.5 + 3.0 * nu).collect();
            let bytes = Self::encode_f64_to_bytes(&tau_flat);
            self.d_tau = self.stream.clone_htod(&bytes)?;
        } else {
            let mut tau_flat = Vec::with_capacity(self.n_cells);
            for &nu in viscosity_field {
                tau_flat.push((0.5 + 3.0 * nu) as f32);
            }
            let bytes = match self.precision {
                Precision::FP32 => Self::encode_f32_to_bytes(&tau_flat),
                Precision::BF16 => Self::encode_bf16_to_bytes(&tau_flat),
                Precision::FP64 => unreachable!("FP64 handled in outer branch"),
            };
            self.d_tau = self.stream.clone_htod(&bytes)?;
        }
        Ok(())
    }

    /// Compute Smagorinsky LES subgrid viscosity from current velocity field.
    ///
    /// Launches the GPU kernel that computes the strain rate tensor S_ij
    /// from central differences on the SoA velocity field, then sets:
    ///   tau(x) = tau_base + 3 * (C_s * dx)^2 * |S(x)|
    /// clamped to [0.505, 5.0] for stability.
    ///
    /// Call between LBM step() iterations (e.g., every 10 steps) to adapt
    /// the local viscosity to velocity gradients. High-gradient regions
    /// (galaxy cores) get higher tau, preserving structure against diffusion.
    pub fn update_smagorinsky_tau(&mut self, cs: f64, dx: f64, tau_base: f64) -> Result<()> {
        let (nx_i, ny_i, nz_i) = (self.nx as i32, self.ny as i32, self.nz as i32);
        let cs_sq_dx_sq = (cs * cs * dx * dx) as f32;
        let tau_base_f32 = tau_base as f32;

        if self.use_tiling {
            // Tiled path: 3D grid of 8x8x4 tiles, shared-memory velocity gradients.
            let soa_kernel = self
                .soa_smagorinsky_tiled_kernel
                .as_ref()
                .context("Tiled Smagorinsky kernel requires SoA mode (FP32)")?;
            let gx = (self.nx as u32).div_ceil(8);
            let gy = (self.ny as u32).div_ceil(8);
            let gz = (self.nz as u32).div_ceil(4);
            let config = LaunchConfig {
                grid_dim: (gx, gy, gz),
                block_dim: (8, 8, 4),
                shared_mem_bytes: 0, // statically allocated __shared__
            };
            let mut b = self.stream.launch_builder(soa_kernel);
            b.arg(&self.d_u)
                .arg(&mut self.d_tau)
                .arg(&tau_base_f32)
                .arg(&cs_sq_dx_sq)
                .arg(&nx_i)
                .arg(&ny_i)
                .arg(&nz_i);
            unsafe { b.launch(config) }?;
        } else {
            // Untiled path: 1D grid, scattered global reads.
            let soa_kernel = self
                .soa_smagorinsky_kernel
                .as_ref()
                .context("Smagorinsky kernel requires SoA mode (FP32)")?;
            let threads = 128u32;
            let blocks = (self.n_cells as u32).div_ceil(threads);
            let config = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut b = self.stream.launch_builder(soa_kernel);
            b.arg(&self.d_u)
                .arg(&mut self.d_tau)
                .arg(&tau_base_f32)
                .arg(&cs_sq_dx_sq)
                .arg(&nx_i)
                .arg(&ny_i)
                .arg(&nz_i);
            unsafe { b.launch(config) }?;
        }
        Ok(())
    }

    pub fn evolve(&mut self, steps: usize) -> Result<()> {
        for _ in 0..steps {
            self.step()?;
        }
        self.sync_to_host()?;
        Ok(())
    }

    fn decode_f32_from_bytes(bytes: &[u8], out: &mut [f32]) -> Result<()> {
        ensure!(
            bytes.len() == out.len() * 4,
            "unexpected f32 byte length: got {}, expected {}",
            bytes.len(),
            out.len() * 4
        );
        for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
            *dst = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        Ok(())
    }

    fn decode_bf16_from_bytes(bytes: &[u8], out: &mut [f32]) -> Result<()> {
        ensure!(
            bytes.len() == out.len() * 2,
            "unexpected bf16 byte length: got {}, expected {}",
            bytes.len(),
            out.len() * 2
        );
        for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(2)) {
            let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
            *dst = half::bf16::from_bits(bits).to_f32();
        }
        Ok(())
    }

    /// Decode FP64 GPU bytes into f32 host buffers (narrowing for API compat).
    fn decode_f64_from_bytes_to_f32(bytes: &[u8], out: &mut [f32]) -> Result<()> {
        ensure!(
            bytes.len() == out.len() * 8,
            "unexpected f64 byte length: got {}, expected {}",
            bytes.len(),
            out.len() * 8
        );
        for (dst, chunk) in out.iter_mut().zip(bytes.chunks_exact(8)) {
            let mut arr = [0u8; 8];
            arr.copy_from_slice(chunk);
            *dst = f64::from_le_bytes(arr) as f32;
        }
        Ok(())
    }

    /// Launch the LBM step kernel (d_f -> d_f_tmp) without swapping buffers.
    ///
    /// Extracted from `step()` so that `step_graph_pair()` can capture two
    /// consecutive launches (A->B, B->A) into a single CUDA graph without
    /// the host-side `std::mem::swap` that would be invisible to the graph.
    fn launch_step_kernel(&mut self) -> Result<()> {
        let (nx, ny, nz) = (self.nx as i32, self.ny as i32, self.nz as i32);

        if self.use_soa {
            if self.use_aa {
                // A-A streaming: single-buffer, parity-dependent direction remap.
                // Highest priority: overrides all other streaming modes since it
                // changes the fundamental buffer strategy (one f, no f_tmp).
                let threads = 128u32;
                let soa_kernel = if self.use_mrt {
                    self.soa_mrt_aa_step_kernel
                        .as_ref()
                        .context("A-A MRT SoA step kernel not loaded")?
                } else {
                    self.soa_aa_step_kernel
                        .as_ref()
                        .context("A-A BGK SoA step kernel not loaded")?
                };
                let blocks = (self.n_cells as u32).div_ceil(threads);
                let config = LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                };
                let parity = self.aa_parity;
                let mut b = self.stream.launch_builder(soa_kernel);
                b.arg(&mut self.d_f)
                    .arg(&mut self.d_rho)
                    .arg(&mut self.d_u)
                    .arg(&self.d_tau)
                    .arg(&self.d_force)
                    .arg(&nx)
                    .arg(&ny)
                    .arg(&nz)
                    .arg(&parity);
                unsafe { b.launch(config) }?;
                return Ok(());
            } else if self.use_tiling {
                // Tiled path: 3D grid of 8x8x4 tiles, shared-memory pull scheme.
                // Highest priority: overrides coarsening and pull-stream flags.
                let soa_kernel = if self.use_mrt {
                    self.soa_mrt_tiled_step_kernel
                        .as_ref()
                        .context("Tiled MRT SoA step kernel not loaded")?
                } else {
                    self.soa_tiled_step_kernel
                        .as_ref()
                        .context("Tiled BGK SoA step kernel not loaded")?
                };
                let config = LaunchConfig {
                    grid_dim: (
                        (self.nx as u32).div_ceil(8),
                        (self.ny as u32).div_ceil(8),
                        (self.nz as u32).div_ceil(4),
                    ),
                    block_dim: (8, 8, 4),
                    shared_mem_bytes: 45_600,
                };
                let mut b = self.stream.launch_builder(soa_kernel);
                b.arg(&self.d_f)
                    .arg(&mut self.d_f_tmp)
                    .arg(&mut self.d_rho)
                    .arg(&mut self.d_u)
                    .arg(&self.d_tau)
                    .arg(&self.d_force)
                    .arg(&nx)
                    .arg(&ny)
                    .arg(&nz);
                unsafe { b.launch(config) }?;
            } else if self.use_coarsening {
                let threads = 128u32;
                let soa_kernel = if self.use_mrt {
                    self.soa_mrt_coarsened_step_kernel
                        .as_ref()
                        .context("Coarsened MRT SoA step kernel not loaded")?
                } else {
                    self.soa_coarsened_step_kernel
                        .as_ref()
                        .context("Coarsened BGK SoA step kernel not loaded")?
                };
                let half_cells = (self.n_cells as u32).div_ceil(2);
                let blocks = half_cells.div_ceil(threads);
                let config = LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut b = self.stream.launch_builder(soa_kernel);
                b.arg(&self.d_f)
                    .arg(&mut self.d_f_tmp)
                    .arg(&mut self.d_rho)
                    .arg(&mut self.d_u)
                    .arg(&self.d_tau)
                    .arg(&self.d_force)
                    .arg(&nx)
                    .arg(&ny)
                    .arg(&nz);
                unsafe { b.launch(config) }?;
            } else if self.use_pull_stream {
                let threads = 128u32;
                let soa_kernel = if self.use_mrt {
                    self.soa_mrt_pull_step_kernel
                        .as_ref()
                        .context("Pull MRT SoA step kernel not loaded")?
                } else {
                    self.soa_pull_step_kernel
                        .as_ref()
                        .context("Pull BGK SoA step kernel not loaded")?
                };
                let blocks = (self.n_cells as u32).div_ceil(threads);
                let config = LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut b = self.stream.launch_builder(soa_kernel);
                b.arg(&self.d_f)
                    .arg(&mut self.d_f_tmp)
                    .arg(&mut self.d_rho)
                    .arg(&mut self.d_u)
                    .arg(&self.d_tau)
                    .arg(&self.d_force)
                    .arg(&nx)
                    .arg(&ny)
                    .arg(&nz);
                unsafe { b.launch(config) }?;
            } else {
                let threads = 128u32;
                let soa_kernel = if self.use_mrt {
                    self.soa_mrt_step_kernel
                        .as_ref()
                        .context("MRT SoA step kernel not loaded")?
                } else {
                    self.soa_step_kernel
                        .as_ref()
                        .context("SoA step kernel not loaded")?
                };
                let blocks = (self.n_cells as u32).div_ceil(threads);
                let config = LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut b = self.stream.launch_builder(soa_kernel);
                b.arg(&self.d_f)
                    .arg(&mut self.d_f_tmp)
                    .arg(&mut self.d_rho)
                    .arg(&mut self.d_u)
                    .arg(&self.d_tau)
                    .arg(&self.d_force)
                    .arg(&nx)
                    .arg(&ny)
                    .arg(&nz);
                unsafe { b.launch(config) }?;
            }
        } else {
            let (bx, by, bz) = self.lbm_block_dim;
            let config = LaunchConfig {
                grid_dim: (
                    self.nx.div_ceil(bx as usize) as u32,
                    self.ny.div_ceil(by as usize) as u32,
                    self.nz.div_ceil(bz as usize) as u32,
                ),
                block_dim: (bx, by, bz),
                shared_mem_bytes: 0,
            };
            let mut b = self.stream.launch_builder(&self.lbm_step_fused_kernel);
            b.arg(&self.d_f)
                .arg(&mut self.d_f_tmp)
                .arg(&mut self.d_rho)
                .arg(&mut self.d_u)
                .arg(&self.d_force)
                .arg(&self.d_tau)
                .arg(&nx)
                .arg(&ny)
                .arg(&nz);
            unsafe { b.launch(config) }?;
        }
        Ok(())
    }

    pub fn step(&mut self) -> Result<()> {
        self.launch_step_kernel()?;
        if self.use_aa {
            // A-A: toggle parity, no buffer swap needed (single buffer)
            self.aa_parity = 1 - self.aa_parity;
        } else {
            std::mem::swap(&mut self.d_f, &mut self.d_f_tmp);
        }
        Ok(())
    }

    /// Graph-accelerated double-step: captures kernel(A->B) + kernel(B->A)
    /// into a single CUDA graph, fixing the double-buffer bug in the old
    /// single-step graph which always read from buffer A on replay.
    ///
    /// Each call advances the simulation by 2 timesteps. The graph bakes
    /// in both device pointer directions, so no host-side swap is needed on
    /// replay -- data starts and ends in buffer A.
    ///
    /// For odd total step counts, use `step_n()` which handles the remainder.
    pub fn step_graph_pair(&mut self) -> Result<()> {
        use cudarc::driver::sys;

        ensure!(
            self.use_soa,
            "step_graph_pair() requires SoA layout (use_soa=true)"
        );

        if self.step_graph_cache.is_none() {
            self.stream
                .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_GLOBAL)?;

            // Step 1: A -> B
            self.launch_step_kernel()?;
            std::mem::swap(&mut self.d_f, &mut self.d_f_tmp);

            // Step 2: B -> A (after swap, d_f is B, d_f_tmp is A)
            self.launch_step_kernel()?;
            std::mem::swap(&mut self.d_f, &mut self.d_f_tmp);

            let graph = self
                .stream
                .end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                )?
                .context("Failed to capture 2-step kernel pair into CUDA graph")?;
            self.step_graph_cache = Some(SendSyncGraph(graph));
        }

        let graph = self
            .step_graph_cache
            .as_ref()
            .context("step_graph_cache unexpectedly empty after capture")?;
        graph.0.launch()?;
        Ok(())
    }

    /// Run `n` LBM timesteps, using graph-pair acceleration when available.
    ///
    /// SoA: uses `step_graph_pair()` for pairs (amortizing launch overhead),
    /// plus one regular `step()` for odd remainders.
    /// AoS: falls back to `n` individual `step()` calls.
    pub fn step_n(&mut self, n: usize) -> Result<()> {
        if self.use_aa {
            // A-A: no graph acceleration (graph bakes in buffer pointers,
            // but A-A uses one buffer with parity-dependent semantics)
            for _ in 0..n {
                self.step()?;
            }
        } else if self.use_soa {
            let pairs = n / 2;
            let remainder = n % 2;
            for _ in 0..pairs {
                self.step_graph_pair()?;
            }
            if remainder == 1 {
                self.step()?;
            }
        } else {
            for _ in 0..n {
                self.step()?;
            }
        }
        Ok(())
    }

    /// Invalidate the cached CUDA graph (call after changing tau, force, or grid size).
    pub fn invalidate_graph(&mut self) {
        self.step_graph_cache = None;
    }

    pub fn calculate_enstrophy(&mut self) -> Result<f32> {
        let (nx, ny, nz, n) = (
            self.nx as i32,
            self.ny as i32,
            self.nz as i32,
            self.n_cells as i32,
        );
        let enstrophy_config = LaunchConfig {
            grid_dim: (
                self.nx.div_ceil(2) as u32,
                self.ny.div_ceil(2) as u32,
                self.nz.div_ceil(2) as u32,
            ),
            block_dim: (2, 2, 2),
            shared_mem_bytes: 0,
        };
        let grid_size = self.n_cells.div_ceil(256) as u32;
        let reduce_config = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };

        if self.precision == Precision::FP64 {
            // FP64: enstrophy kernel writes double*, reduce uses double buffers
            let d_out = self
                .d_reduction_out_f64
                .as_mut()
                .context("FP64 reduction output not allocated")?;
            let (out_ptr, _) = d_out.device_ptr(&self.stream);
            let mut bz = self.stream.launch_builder(&self.zero_kernel);
            bz.arg(&out_ptr);
            unsafe { bz.launch(LaunchConfig::for_num_elems(1)) }?;

            let d_u_ptr = self.d_u.device_ptr(&self.stream).0;
            let d_buf = self
                .d_reduction_buffer_f64
                .as_ref()
                .context("FP64 reduction buffer not allocated")?;
            let d_enstrophy_buffer_ptr = d_buf.device_ptr(&self.stream).0;

            let mut b = self.stream.launch_builder(&self.enstrophy_cell_kernel);
            b.arg(&d_u_ptr)
                .arg(&d_enstrophy_buffer_ptr)
                .arg(&nx)
                .arg(&ny)
                .arg(&nz);
            unsafe { b.launch(enstrophy_config) }?;

            let d_buf = self.d_reduction_buffer_f64.as_ref().unwrap();
            let d_out = self.d_reduction_out_f64.as_mut().unwrap();
            let mut b_reduce = self.stream.launch_builder(&self.reduce_sum_f32_kernel);
            b_reduce.arg(d_buf).arg(d_out).arg(&n);
            unsafe { b_reduce.launch(reduce_config) }?;

            let res = self
                .stream
                .clone_dtoh(self.d_reduction_out_f64.as_ref().unwrap())?;
            Ok((res[0] / self.n_cells as f64) as f32)
        } else {
            let (out_ptr, _) = self.d_reduction_out.device_ptr(&self.stream);
            let mut bz = self.stream.launch_builder(&self.zero_kernel);
            bz.arg(&out_ptr);
            unsafe { bz.launch(LaunchConfig::for_num_elems(1)) }?;

            let d_u_ptr = self.d_u.device_ptr(&self.stream).0;
            let d_enstrophy_buffer_ptr = self
                .d_reduction_buffer_f32
                .as_ref()
                .unwrap()
                .device_ptr(&self.stream)
                .0;

            let mut b = self.stream.launch_builder(&self.enstrophy_cell_kernel);
            b.arg(&d_u_ptr)
                .arg(&d_enstrophy_buffer_ptr)
                .arg(&nx)
                .arg(&ny)
                .arg(&nz);
            unsafe { b.launch(enstrophy_config) }?;

            let mut b_reduce = self.stream.launch_builder(&self.reduce_sum_f32_kernel);
            b_reduce
                .arg(self.d_reduction_buffer_f32.as_ref().unwrap())
                .arg(&mut self.d_reduction_out)
                .arg(&n);
            unsafe { b_reduce.launch(reduce_config) }?;

            let res = self.stream.clone_dtoh(&self.d_reduction_out)?;
            Ok(res[0] / self.n_cells as f32)
        }
    }

    pub fn calculate_mean_density(&mut self) -> Result<f32> {
        let n = self.n_cells as i32;
        let grid_size = self.n_cells.div_ceil(256) as u32;

        if self.precision == Precision::FP64 {
            // FP64: zero/reduce/read use double buffers
            let d_out = self
                .d_reduction_out_f64
                .as_mut()
                .context("FP64 reduction buffer not allocated")?;
            let (out_ptr, _) = d_out.device_ptr(&self.stream);
            let mut bz = self.stream.launch_builder(&self.zero_kernel);
            bz.arg(&out_ptr);
            unsafe { bz.launch(LaunchConfig::for_num_elems(1)) }?;

            let d_out = self.d_reduction_out_f64.as_mut().unwrap();
            let mut b = self.stream.launch_builder(&self.reduce_sum_rho_kernel);
            b.arg(&self.d_rho).arg(d_out).arg(&n);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }?;

            let res = self
                .stream
                .clone_dtoh(self.d_reduction_out_f64.as_ref().unwrap())?;
            Ok((res[0] / self.n_cells as f64) as f32)
        } else {
            let (out_ptr, _) = self.d_reduction_out.device_ptr(&self.stream);
            let mut bz = self.stream.launch_builder(&self.zero_kernel);
            bz.arg(&out_ptr);
            unsafe { bz.launch(LaunchConfig::for_num_elems(1)) }?;

            let mut b = self.stream.launch_builder(&self.reduce_sum_rho_kernel);
            b.arg(&self.d_rho).arg(&mut self.d_reduction_out).arg(&n);
            unsafe {
                b.launch(LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
            }?;

            let res = self.stream.clone_dtoh(&self.d_reduction_out)?;
            Ok(res[0] / self.n_cells as f32)
        }
    }

    /// Maximum Mach number from host-side velocity buffer.
    ///
    /// Requires a prior `sync_to_host()` call to populate `self.u`.
    /// Ma = max(|u|) / c_s where c_s = 1/sqrt(3) for D3Q19.
    pub fn max_mach_number(&self) -> f32 {
        let cs = (1.0_f32 / 3.0).sqrt();
        let v_max = self
            .u
            .iter()
            .map(|ui| (ui[0] * ui[0] + ui[1] * ui[1] + ui[2] * ui[2]).sqrt())
            .fold(0.0_f32, f32::max);
        v_max / cs
    }

    /// Maximum Mach number computed entirely on GPU (Phase 12).
    ///
    /// Two-pass max-reduction over d_u SoA velocity field. Reads back a single
    /// f32 (4 bytes PCIe) instead of the full 24 MB velocity buffer.
    /// Cost: ~10us kernel + 4 bytes readback vs 2ms for sync_to_host.
    ///
    /// Only available for FP32 SoA precision.
    pub fn max_mach_number_device(&mut self) -> Result<f32> {
        let kernel = self
            .reduce_max_speed_kernel
            .as_ref()
            .context("reduce_max_speed_f32 kernel not loaded (requires FP32)")?;

        let n = self.n_cells as i32;
        let threads = 128u32;
        let blocks_pass1 = (self.n_cells as u32).div_ceil(threads);

        // Pass 1: N cells -> blocks_pass1 per-block maxima (written to d_buf).
        let config1 = LaunchConfig {
            grid_dim: (blocks_pass1, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        {
            let d_buf = self
                .d_reduction_buffer_f32
                .as_mut()
                .context("reduction buffer not allocated")?;
            let mut b1 = self.stream.launch_builder(kernel);
            b1.arg(&self.d_u).arg(d_buf).arg(&n);
            unsafe { b1.launch(config1) }?;
        }

        // Pass 2: read back per-block maxima and reduce on host.
        // blocks_pass1 floats = 64 KB at 128^3 (vs 24 MB for full sync_to_host).
        // The reduce_max_speed_f32 kernel computes sqrt(ux^2+uy^2+uz^2) internally,
        // so pass-2 would need a separate scalar-max kernel. Host reduction of
        // 16K floats is ~5us -- negligible compared to the PCIe transfer.
        let d_buf = self
            .d_reduction_buffer_f32
            .as_ref()
            .context("reduction buffer not allocated")?;
        let block_maxima = self.stream.clone_dtoh(d_buf)?;
        let max_speed = block_maxima[..blocks_pass1 as usize]
            .iter()
            .fold(0.0_f32, |a, &b| f32::max(a, b));

        let cs = (1.0_f32 / 3.0).sqrt();
        Ok(max_speed / cs)
    }

    /// Enable or disable thread-coarsened kernels (1 thread = 2 cells).
    ///
    /// When enabled, the SoA step kernels use float2 vectorized loads and
    /// process 2 contiguous cells per thread for instruction-level parallelism.
    /// Only affects FP32 SoA path.
    pub fn set_coarsening(&mut self, enabled: bool) {
        self.use_coarsening = enabled;
    }

    /// Whether thread coarsening is currently enabled.
    pub fn use_coarsening(&self) -> bool {
        self.use_coarsening
    }

    /// Enable or disable pull-streaming (Phase 5: coalesced writes).
    ///
    /// Pull-streaming reverses the data flow: each thread reads from
    /// opposite-direction neighbors (scattered reads) and writes to itself
    /// (coalesced writes). On Ada Lovelace, coalesced writes typically
    /// outperform coalesced reads for bandwidth-limited kernels.
    ///
    /// Mutually exclusive with coarsening: enabling pull disables coarsening
    /// (pull breaks float2 vectorized loads because source cells for
    /// contiguous idx are non-contiguous for cy/cz != 0).
    pub fn set_pull_streaming(&mut self, enabled: bool) {
        self.use_pull_stream = enabled;
        if enabled {
            self.use_coarsening = false;
        }
    }

    /// Whether pull-streaming is currently enabled.
    pub fn use_pull_streaming(&self) -> bool {
        self.use_pull_stream
    }

    /// Enable or disable shared-memory tiled kernels (8x8x4 tile + halo).
    ///
    /// Tiling has the highest dispatch priority: when enabled, it overrides
    /// both coarsening and pull-streaming flags. The tiled kernel uses
    /// cooperative shared-memory loading with a 1-cell halo and pull-scheme
    /// streaming for coalesced global writes.
    pub fn set_tiling(&mut self, enabled: bool) {
        self.use_tiling = enabled;
        if enabled {
            self.invalidate_graph();
        }
    }

    /// Whether shared-memory tiling is currently enabled.
    pub fn use_tiling(&self) -> bool {
        self.use_tiling
    }

    /// Enable A-A (Alternating-Address) single-buffer streaming.
    ///
    /// Halves VRAM for f by eliminating the ping-pong buffer. Each step uses
    /// a parity-dependent direction remap via the OPP[] LUT. Incompatible
    /// with CUDA graph capture (graph bakes in buffer pointers, but A-A
    /// reads/writes the same buffer with varying semantics).
    ///
    /// The d_f_tmp allocation is NOT freed (would require re-creation to
    /// switch back). It simply goes unused when A-A is active.
    pub fn set_aa_streaming(&mut self, enabled: bool) {
        self.use_aa = enabled;
        self.aa_parity = 0;
        if enabled {
            self.invalidate_graph();
        }
    }

    /// Whether A-A streaming is currently enabled.
    pub fn use_aa_streaming(&self) -> bool {
        self.use_aa
    }

    /// Grid dimension along X.
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Grid dimension along Y.
    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Grid dimension along Z.
    pub fn nz(&self) -> usize {
        self.nz
    }

    /// Total number of cells (nx * ny * nz).
    pub fn n_cells(&self) -> usize {
        self.n_cells
    }

    pub fn sync_to_host(&mut self) -> Result<()> {
        let rho_bytes = self.stream.clone_dtoh(&self.d_rho)?;
        let u_bytes = self.stream.clone_dtoh(&self.d_u)?;

        match self.precision {
            Precision::FP32 => {
                Self::decode_f32_from_bytes(&rho_bytes, &mut self.rho)?;
                let mut u_flat = vec![0.0f32; self.n_cells * 3];
                Self::decode_f32_from_bytes(&u_bytes, &mut u_flat)?;
                if self.use_soa {
                    // SoA velocity: u_flat = [ux_0..ux_N, uy_0..uy_N, uz_0..uz_N]
                    let n = self.n_cells;
                    for idx in 0..n {
                        self.u[idx] = [u_flat[idx], u_flat[n + idx], u_flat[2 * n + idx]];
                    }
                } else {
                    // AoS velocity: u_flat = [ux_0, uy_0, uz_0, ux_1, ...]
                    for idx in 0..self.n_cells {
                        let base = idx * 3;
                        self.u[idx] = [u_flat[base], u_flat[base + 1], u_flat[base + 2]];
                    }
                }
            }
            Precision::BF16 => {
                Self::decode_bf16_from_bytes(&rho_bytes, &mut self.rho)?;
                let mut u_flat = vec![0.0f32; self.n_cells * 3];
                Self::decode_bf16_from_bytes(&u_bytes, &mut u_flat)?;
                for idx in 0..self.n_cells {
                    let base = idx * 3;
                    self.u[idx] = [u_flat[base], u_flat[base + 1], u_flat[base + 2]];
                }
            }
            Precision::FP64 => {
                Self::decode_f64_from_bytes_to_f32(&rho_bytes, &mut self.rho)?;
                let mut u_flat = vec![0.0f32; self.n_cells * 3];
                Self::decode_f64_from_bytes_to_f32(&u_bytes, &mut u_flat)?;
                for idx in 0..self.n_cells {
                    let base = idx * 3;
                    self.u[idx] = [u_flat[base], u_flat[base + 1], u_flat[base + 2]];
                }
            }
        }

        Ok(())
    }

    #[cfg(feature = "cufft")]
    fn ensure_fft_plan(&mut self) -> Result<cudarc::cufft::sys::cufftHandle> {
        if let Some(handle) = self.fft_plan {
            return Ok(handle);
        }
        let handle = cufft::plan_3d(
            self.nx as i32,
            self.ny as i32,
            self.nz as i32,
            cudarc::cufft::sys::cufftType::CUFFT_C2C,
        )
        .context("cufftPlan3d failed")?;
        unsafe {
            cufft::set_stream(handle, self.stream.cu_stream() as _)
                .context("cufftSetStream failed")?;
        }
        self.fft_plan = Some(handle);
        Ok(handle)
    }

    /// Out-of-place complex-to-complex FFT using cuFFT.
    ///
    /// `direction` is `-1` for forward and `1` for inverse.
    #[cfg(feature = "cufft")]
    pub fn fft_3d_c2c_into(
        &mut self,
        input: &CudaSlice<ComplexDevice>,
        output: &mut CudaSlice<ComplexDevice>,
        direction: i32,
    ) -> Result<()> {
        let handle = self.ensure_fft_plan()?;
        let (i_ptr, _) = input.device_ptr(&self.stream);
        let (o_ptr, _) = output.device_ptr(&self.stream);
        unsafe {
            cufft::exec_c2c(handle, i_ptr as *mut _, o_ptr as *mut _, direction)
                .context("cufftExecC2C failed")?;
        }
        Ok(())
    }

    pub fn apply_spectral_mask(
        &self,
        u_hat: &mut CudaSlice<ComplexDevice>,
        mask: &CudaSlice<f32>,
        damping: f32,
    ) -> Result<()> {
        let n = self.n_cells as i32;
        let mut b = self.stream.launch_builder(&self.apply_mask_kernel);
        b.arg(u_hat).arg(mask).arg(&damping).arg(&n);
        unsafe { b.launch(Self::launch_config_1d(self.n_cells as u32, self.precision)) }?;
        Ok(())
    }

    pub fn convert_real_to_complex(
        &self,
        d_u_hat: &mut CudaSlice<ComplexDevice>,
        component: usize,
    ) -> Result<()> {
        let (c, n) = (component as i32, self.n_cells as i32);
        let mut b = self
            .stream
            .launch_builder(&self.convert_real_to_complex_kernel);
        b.arg(&self.d_u).arg(d_u_hat).arg(&c).arg(&n);
        unsafe { b.launch(Self::launch_config_1d(self.n_cells as u32, self.precision)) }?;
        Ok(())
    }

    pub fn convert_complex_to_real(
        &mut self,
        d_u_hat: &CudaSlice<ComplexDevice>,
        component: usize,
        scale: f32,
    ) -> Result<()> {
        let (c, n) = (component as i32, self.n_cells as i32);
        let mut b = self
            .stream
            .launch_builder(&self.convert_complex_to_real_kernel);
        b.arg(d_u_hat)
            .arg(&mut self.d_u)
            .arg(&c)
            .arg(&scale)
            .arg(&n);
        unsafe { b.launch(Self::launch_config_1d(self.n_cells as u32, self.precision)) }?;
        Ok(())
    }

    /// Access the underlying CUDA stream for external synchronization.
    ///
    /// **Multi-stream pipeline pattern:**
    /// ```ignore
    /// // Create solver A on default stream
    /// let solver_a = LbmSolver3DCuda::new(nx, ny, nz, tau, Precision::FP32)?;
    ///
    /// // Fork stream for parallel work
    /// let stream_b = solver_a.stream().fork()?;
    ///
    /// // Use events for cross-stream synchronization:
    /// // 1. Stream A: LBM step for galaxy K
    /// solver_a.step()?;
    /// let event_a = solver_a.stream().record_event(None)?;
    ///
    /// // 2. Stream B waits for stream A, then box-counts galaxy K
    /// stream_b.wait(&event_a)?;
    /// // (box-counting on stream_b here)
    ///
    /// // 3. CPU prepares galaxy K+1 while GPU is busy
    /// ```
    ///
    /// For euclid_df_sweep: CPU prep (~5ms) overlaps with GPU LBM (~10ms)
    /// -> ~40% throughput gain for N=1000 galaxies.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// CUDA context handle (for creating auxiliary GPU objects like GpuBoxCounter).
    pub fn context(&self) -> &Arc<CudaContext> {
        &self._ctx
    }

    /// Create a new CUDA stream on the same context for multi-stream pipelining.
    ///
    /// The returned stream runs concurrently with the solver's default stream.
    /// Use event synchronization to order dependencies between streams.
    pub fn fork_stream(&self) -> Result<Arc<CudaStream>> {
        Ok(self._ctx.new_stream()?)
    }

    /// Device-resident density buffer (raw bytes, precision-dependent layout).
    ///
    /// For FP32: nx*ny*nz f32 values in row-major order.
    /// Used by GpuBoxCounter for zero-copy box-counting on GPU-evolved density.
    pub fn d_rho_bytes(&self) -> &CudaSlice<u8> {
        &self.d_rho
    }

    /// Pin the rho field in L2 cache via CUDA access policy window.
    ///
    /// Reserves 16 MB of L2 set-aside budget via `cuCtxSetLimit` and marks
    /// the rho buffer as PERSISTING. Only effective on SM 8.0+ (Ampere+).
    pub fn pin_rho_in_l2(&self) -> Result<()> {
        use cudarc::driver::sys::{
            CUaccessPolicyWindow_st, CUaccessProperty_enum, CUlimit_enum, CUstreamAttrID,
            CUstreamAttrValue, cuCtxSetLimit, cuStreamSetAttribute,
        };

        // Reserve 32 MB of L2 for persistent data.
        // Rho is pinned via access policy window; tau benefits from the enlarged
        // budget via LRU (rho + tau = 16 MB at 128^3, fits in 32 MB with headroom).
        // Only ONE access policy window per stream -- tau stays LRU-warm.
        let budget: usize = 32 * 1024 * 1024;
        let limit_result =
            unsafe { cuCtxSetLimit(CUlimit_enum::CU_LIMIT_PERSISTING_L2_CACHE_SIZE, budget) };
        if limit_result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            eprintln!(
                "    L2 budget reservation failed: {:?} (non-fatal, pinning may be ineffective)",
                limit_result
            );
        }

        let rho_bytes = self.d_rho.len(); // CudaSlice<u8>.len() = byte count
        let (rho_ptr, _guard) = self.d_rho.device_ptr(&self.stream);

        let policy = CUaccessPolicyWindow_st {
            base_ptr: rho_ptr as *mut std::ffi::c_void,
            num_bytes: rho_bytes,
            hitRatio: 1.0,
            hitProp: CUaccessProperty_enum::CU_ACCESS_PROPERTY_PERSISTING,
            missProp: CUaccessProperty_enum::CU_ACCESS_PROPERTY_NORMAL,
        };
        let attr_value = CUstreamAttrValue {
            accessPolicyWindow: policy,
        };
        let result = unsafe {
            cuStreamSetAttribute(
                self.stream.cu_stream(),
                CUstreamAttrID::CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                &attr_value,
            )
        };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            anyhow::bail!("cuStreamSetAttribute failed: {:?}", result);
        }

        eprintln!(
            "    L2 residency: pinned {} KB rho field (hitRatio=1.0, PERSISTING, budget={}MB)",
            rho_bytes / 1024,
            budget / (1024 * 1024),
        );
        Ok(())
    }

    /// Remove L2 access policy (reset to NORMAL hit behavior).
    fn unpin_l2(&self) -> Result<()> {
        use cudarc::driver::sys::{
            CUaccessPolicyWindow_st, CUaccessProperty_enum, CUstreamAttrID, CUstreamAttrValue,
            cuStreamSetAttribute,
        };

        let rho_bytes = self.d_rho.len();
        let (rho_ptr, _guard) = self.d_rho.device_ptr(&self.stream);

        let policy = CUaccessPolicyWindow_st {
            base_ptr: rho_ptr as *mut std::ffi::c_void,
            num_bytes: rho_bytes,
            hitRatio: 0.0,
            hitProp: CUaccessProperty_enum::CU_ACCESS_PROPERTY_NORMAL,
            missProp: CUaccessProperty_enum::CU_ACCESS_PROPERTY_NORMAL,
        };
        let attr_value = CUstreamAttrValue {
            accessPolicyWindow: policy,
        };
        let result = unsafe {
            cuStreamSetAttribute(
                self.stream.cu_stream(),
                CUstreamAttrID::CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                &attr_value,
            )
        };
        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            anyhow::bail!("cuStreamSetAttribute (unpin) failed: {:?}", result);
        }
        Ok(())
    }

    /// Enable or disable L2 cache pinning for the rho field.
    ///
    /// When enabled, reserves 32 MB of L2 set-aside budget and marks the rho
    /// buffer as PERSISTING. Tau benefits from the enlarged budget via LRU.
    /// When disabled, resets to NORMAL access policy.
    /// Re-pins automatically if d_rho is reallocated (e.g. via initialize_custom).
    pub fn set_l2_pinning(&mut self, enabled: bool) -> Result<()> {
        if enabled {
            self.pin_rho_in_l2()?;
            self.l2_pinned = true;
        } else {
            self.unpin_l2()?;
            self.l2_pinned = false;
        }
        Ok(())
    }
}

impl Drop for LbmSolver3DCuda {
    fn drop(&mut self) {
        #[cfg(feature = "cufft")]
        if let Some(handle) = self.fft_plan.take() {
            // Best-effort cleanup. Avoid panicking in Drop.
            let _ = unsafe { cufft::destroy(handle) };
        }
    }
}

#[cfg(test)]
impl LbmSolver3DCuda {
    fn set_all_distributions_for_test(&mut self, value: f32) -> Result<()> {
        let count = 19 * self.n_cells;
        let bytes: Vec<u8> = match self.precision {
            Precision::FP32 => {
                let values = vec![value; count];
                values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>()
            }
            Precision::BF16 => {
                let bf = half::bf16::from_f32(value).to_bits();
                let values = vec![bf; count];
                values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>()
            }
            Precision::FP64 => {
                let values = vec![value as f64; count];
                values
                    .iter()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>()
            }
        };
        self.d_f = self.stream.clone_htod(&bytes)?;
        Ok(())
    }
}

// ============================================================================
// Dark Halo Hunt CUDA solver (SoA layout, optimized for Ada Lovelace)
// ============================================================================

const KERNEL_DARK_HALO_SRC: &str = include_str!("kernels_dark_halo.cu");

/// Result of a single k-value dark halo scan via CUDA.
#[derive(Debug, Clone)]
pub struct CudaDarkHaloResult {
    /// CD algebra dimension parameter.
    pub k: usize,
    /// Volume fraction of halo cells (0.0 = no halos, 1.0 = all halos).
    pub volume_fraction: f64,
    /// Total number of halo cells detected.
    pub halo_count: u32,
    /// Total cells in the grid.
    pub n_cells: u32,
    /// Whether convergence early-out triggered.
    pub early_stopped: bool,
    /// Number of LBM steps actually executed.
    pub steps_run: u32,
}

/// CUDA-accelerated dark halo hunt solver with SoA memory layout.
///
/// Uses NVRTC runtime compilation of optimized CUDA kernels:
/// - `lbm_step_soa_fused`: Fused collision + streaming with SoA layout
/// - `dark_halo_detector`: Warp-level reduction for halo cell counting
/// - `zd_viscosity_modulation`: ZD-imbalance viscosity field generation
/// - `convergence_check`: Delta-rho early stopping
///
/// # Performance vs Vulkan path
/// Expected ~2-5x speedup from SoA coalescing + fused kernel.
/// Additional ~2x from f16 storage (TODO).
///
/// # Hardware requirements
/// - CUDA compute capability >= 7.0 (Volta+)
/// - Optimized for SM 8.9 (Ada Lovelace, RTX 4070 Ti)
pub struct DarkHaloCudaSolver {
    nx: usize,
    ny: usize,
    nz: usize,
    n_cells: usize,
    stream: Arc<CudaStream>,
    // SoA distribution function buffers: [19 * N] floats
    d_f: CudaSlice<f32>,
    d_f_tmp: CudaSlice<f32>,
    // Macroscopic fields
    d_rho: CudaSlice<f32>,
    d_rho_prev: CudaSlice<f32>, // for convergence check
    d_u: CudaSlice<f32>,        // [3 * N] SoA: ux[N], uy[N], uz[N]
    d_tau: CudaSlice<f32>,
    // Atomic counter for halo detection
    d_halo_count: CudaSlice<u32>,
    #[expect(dead_code)] // TODO: wire convergence_check kernel
    d_delta_sum: CudaSlice<f32>,
    // Kernels
    lbm_step_kernel: CudaFunction,
    halo_detector_kernel: CudaFunction,
    zd_viscosity_kernel: CudaFunction,
    #[expect(dead_code)] // TODO: wire convergence_check kernel
    convergence_kernel: CudaFunction,
}

impl DarkHaloCudaSolver {
    /// Create a new CUDA dark halo solver.
    ///
    /// Compiles kernels via NVRTC and allocates SoA buffers.
    ///
    /// # VRAM budget (f32 ping-pong)
    /// - 256^3: ~2.8 GB (fits 4+ GB cards)
    /// - 384^3: ~9.5 GB (fits 12 GB cards)
    /// - 512^3: ~23.6 GB (REQUIRES f16 + A-A streaming, or >24 GB card)
    pub fn new(nx: usize, ny: usize, nz: usize) -> Result<Self> {
        let ctx = cudarc::driver::CudaContext::new(0).context("CUDA context creation")?;
        // Check VRAM budget before allocating
        let n_cells = nx * ny * nz;
        let required_bytes = (19 * n_cells * 4 * 2) // f, f_tmp (ping-pong)
            + (n_cells * 4 * 2)                      // rho, rho_prev
            + (3 * n_cells * 4)                       // u (SoA)
            + (n_cells * 4)                            // tau
            + 8; // counters
        let total_mem = {
            let mut free: usize = 0;
            let mut total: usize = 0;
            unsafe {
                cudarc::driver::sys::cuMemGetInfo_v2(
                    &mut free as *mut usize,
                    &mut total as *mut usize,
                );
            }
            total
        };
        if total_mem > 0 && required_bytes > (total_mem as f64 * 0.9) as usize {
            anyhow::bail!(
                "VRAM insufficient: need {} MB, have {} MB (90% of {} MB). \
                 Grid {}^3 requires f16+A-A streaming or a larger GPU.",
                required_bytes / (1024 * 1024),
                (total_mem as f64 * 0.9) as usize / (1024 * 1024),
                total_mem / (1024 * 1024),
                nx,
            );
        }
        let stream = ctx.default_stream();

        // Compile dark halo kernels via NVRTC
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some("sm_89"),
            prec_div: Some(false),
            prec_sqrt: Some(false),
            ftz: Some(true),
            fmad: Some(true),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(KERNEL_DARK_HALO_SRC, opts)
            .context("NVRTC compilation of dark halo kernels")?;
        let module = ctx.load_module(ptx).context("Load dark halo CUDA module")?;

        let lbm_step_kernel = module
            .load_function("lbm_step_soa_fused")
            .context("Load lbm_step_soa_fused")?;
        let halo_detector_kernel = module
            .load_function("dark_halo_detector")
            .context("Load dark_halo_detector")?;
        let zd_viscosity_kernel = module
            .load_function("zd_viscosity_modulation")
            .context("Load zd_viscosity_modulation")?;
        let convergence_kernel = module
            .load_function("convergence_check")
            .context("Load convergence_check")?;

        // Allocate SoA buffers (cudarc 0.19: allocations go through CudaStream)
        let d_f = stream
            .alloc_zeros::<f32>(19 * n_cells)
            .context("Alloc d_f")?;
        let d_f_tmp = stream
            .alloc_zeros::<f32>(19 * n_cells)
            .context("Alloc d_f_tmp")?;
        let d_rho = stream.alloc_zeros::<f32>(n_cells).context("Alloc d_rho")?;
        let d_rho_prev = stream
            .alloc_zeros::<f32>(n_cells)
            .context("Alloc d_rho_prev")?;
        let d_u = stream
            .alloc_zeros::<f32>(3 * n_cells)
            .context("Alloc d_u")?;
        let d_tau = stream.alloc_zeros::<f32>(n_cells).context("Alloc d_tau")?;
        let d_halo_count = stream.alloc_zeros::<u32>(1).context("Alloc d_halo_count")?;
        let d_delta_sum = stream.alloc_zeros::<f32>(1).context("Alloc d_delta_sum")?;

        let vram_mb = (
            19 * n_cells * 4 * 2 // f, f_tmp
            + n_cells * 4 * 2               // rho, rho_prev
            + 3 * n_cells * 4               // u
            + n_cells * 4
            // tau
        ) / (1024 * 1024);
        eprintln!(
            "    CUDA dark halo solver: {}x{}x{} = {} cells, {} MB VRAM (SoA)",
            nx, ny, nz, n_cells, vram_mb
        );

        Ok(Self {
            nx,
            ny,
            nz,
            n_cells,
            stream,
            d_f,
            d_f_tmp,
            d_rho,
            d_rho_prev,
            d_u,
            d_tau,
            d_halo_count,
            d_delta_sum,
            lbm_step_kernel,
            halo_detector_kernel,
            zd_viscosity_kernel,
            convergence_kernel,
        })
    }

    /// VRAM usage in bytes for SoA layout.
    pub fn vram_bytes(&self) -> usize {
        (19 * self.n_cells * 4 * 2)
            + (self.n_cells * 4 * 2)
            + (3 * self.n_cells * 4)
            + (self.n_cells * 4)
            + 8 // counters
    }

    /// Pin the rho field in the Ada Lovelace L2 cache (48 MB) via access policy window.
    ///
    /// Uses cuStreamSetAttribute with CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW
    /// to mark the rho buffer as PERSISTING. This keeps the most-accessed
    /// macroscopic field in L2 across kernel launches, avoiding VRAM re-fetch.
    ///
    /// Only effective on SM 8.0+ (Ampere/Ada Lovelace). No-op on older GPUs.
    pub fn pin_rho_in_l2(&self) -> Result<()> {
        use cudarc::driver::sys::{
            CUaccessPolicyWindow_st, CUaccessProperty_enum, CUlimit_enum, CUstreamAttrID,
            CUstreamAttrValue, cuCtxSetLimit, cuStreamSetAttribute,
        };

        // Reserve 32 MB of L2 for persistent data.
        // Rho is pinned via access policy window; tau benefits from the enlarged
        // budget via LRU (rho + tau = 16 MB at 128^3, fits in 32 MB with headroom).
        // Only ONE access policy window per stream -- tau stays LRU-warm.
        let budget: usize = 32 * 1024 * 1024;
        let limit_result =
            unsafe { cuCtxSetLimit(CUlimit_enum::CU_LIMIT_PERSISTING_L2_CACHE_SIZE, budget) };
        if limit_result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            eprintln!(
                "    L2 budget reservation failed: {:?} (non-fatal, pinning may be ineffective)",
                limit_result
            );
        }

        let rho_bytes = self.n_cells * std::mem::size_of::<f32>();
        let (rho_ptr, _guard) = self.d_rho.device_ptr(&self.stream);

        let policy = CUaccessPolicyWindow_st {
            base_ptr: rho_ptr as *mut std::ffi::c_void,
            num_bytes: rho_bytes,
            hitRatio: 1.0,
            hitProp: CUaccessProperty_enum::CU_ACCESS_PROPERTY_PERSISTING,
            missProp: CUaccessProperty_enum::CU_ACCESS_PROPERTY_NORMAL,
        };

        // In CUDA 12+, CUstreamAttrValue = CUlaunchAttributeValue (union).
        // CUstreamAttrID = CUlaunchAttributeID. Variant is CU_LAUNCH_ATTRIBUTE_*.
        let attr_value = CUstreamAttrValue {
            accessPolicyWindow: policy,
        };

        let result = unsafe {
            cuStreamSetAttribute(
                self.stream.cu_stream(),
                CUstreamAttrID::CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                &attr_value,
            )
        };

        if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            anyhow::bail!("cuStreamSetAttribute failed: {:?}", result);
        }

        eprintln!(
            "    L2 residency: pinned {} KB rho field (hitRatio=1.0, PERSISTING, budget={}MB)",
            rho_bytes / 1024,
            budget / (1024 * 1024),
        );
        Ok(())
    }

    /// Count halo cells on the CPU using portable SIMD on pinned host memory.
    ///
    /// Pipeline: GPU d_rho -> PCIe DMA -> pinned host buffer -> L3 V-Cache -> wide::f32x8
    ///
    /// Uses rayon par_chunks(8) with `wide::f32x8` for 8-wide threshold comparison.
    /// The `wide` crate compiles to AVX2 on x86_64, NEON on aarch64, and scalar
    /// fallback elsewhere -- no `#[cfg(target_arch)]` needed.
    ///
    /// On the 5600X3D with 96 MB L3 V-Cache, the 8.4 MB rho field (128^3) sits
    /// entirely in L3 for ~200 GB/s internal bandwidth.
    ///
    /// Returns (halo_count, total_cells).
    pub fn count_halos_cpu_simd(
        &self,
        rho_threshold: f32,
        velocity_epsilon: f32,
    ) -> Result<(u32, u32)> {
        use rayon::prelude::*;
        use wide::{CmpGt, f32x8};

        // Readback rho to pinned memory
        let pinned_rho = self.readback_rho_pinned()?;
        let rho_slice = pinned_rho
            .as_slice()
            .map_err(|e| anyhow::anyhow!("Pinned slice sync: {e}"))?;

        // Also readback velocity for the velocity threshold check
        let u_host = self
            .stream
            .clone_dtoh(&self.d_u)
            .map_err(|e| anyhow::anyhow!("Clone u dtoh: {e}"))?;
        let n = self.n_cells;

        // Parallel SIMD counting: process 8 cells at a time
        let vel_eps_sq = velocity_epsilon * velocity_epsilon;

        let halo_count: usize = rho_slice
            .par_chunks(8)
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let base = chunk_idx * 8;
                let mut count = 0usize;

                if chunk.len() == 8 {
                    // SIMD path: 8-wide comparison via wide crate
                    let rho_vec = f32x8::new(chunk.try_into().unwrap());
                    let threshold_vec = f32x8::splat(rho_threshold);
                    let cmp_mask = rho_vec.cmp_gt(threshold_vec);
                    let mask_bits = cmp_mask.to_array();

                    // Check velocity for each cell where rho > threshold
                    for (bit, &mask_val) in mask_bits.iter().enumerate() {
                        // wide cmp_gt returns 0.0 for false, all-bits-1 (NaN) for true
                        if mask_val != 0.0 {
                            let idx = base + bit;
                            if idx < n {
                                let ux = u_host[idx];
                                let uy = u_host[n + idx];
                                let uz = u_host[2 * n + idx];
                                let u_sq = ux * ux + uy * uy + uz * uz;
                                if u_sq < vel_eps_sq {
                                    count += 1;
                                }
                            }
                        }
                    }
                } else {
                    // Tail elements: scalar
                    for (i, &rho) in chunk.iter().enumerate() {
                        let idx = base + i;
                        if rho > rho_threshold && idx < n {
                            let ux = u_host[idx];
                            let uy = u_host[n + idx];
                            let uz = u_host[2 * n + idx];
                            let u_sq = ux * ux + uy * uy + uz * uz;
                            if u_sq < vel_eps_sq {
                                count += 1;
                            }
                        }
                    }
                }
                count
            })
            .sum();

        Ok((halo_count as u32, n as u32))
    }

    /// Legacy alias for `count_halos_cpu_simd`.
    #[cfg(target_arch = "x86_64")]
    pub fn count_halos_cpu_avx2(
        &self,
        rho_threshold: f32,
        velocity_epsilon: f32,
    ) -> Result<(u32, u32)> {
        self.count_halos_cpu_simd(rho_threshold, velocity_epsilon)
    }

    /// Readback the rho field to page-locked (pinned) host memory via PCIe DMA.
    ///
    /// Returns a `PinnedHostSlice<f32>` that the GPU DMA controller wrote directly to,
    /// bypassing OS virtual memory paging. The data lives at a physical address
    /// that can be pulled into the CPU L3 V-Cache by hardware prefetch.
    ///
    /// # Safety
    /// The returned slice is valid after synchronization (handled internally by cudarc).
    pub fn readback_rho_pinned(&self) -> Result<cudarc::driver::PinnedHostSlice<f32>> {
        let ctx = self.stream.context();
        let mut pinned = unsafe {
            ctx.alloc_pinned::<f32>(self.n_cells)
                .map_err(|e| anyhow::anyhow!("Pinned alloc failed: {e}"))?
        };
        self.stream
            .memcpy_dtoh(&self.d_rho, &mut pinned)
            .map_err(|e| anyhow::anyhow!("DMA dtoh failed: {e}"))?;
        Ok(pinned)
    }

    /// Run dark halo hunt for a single k-value.
    ///
    /// 1. Generate ZD viscosity field for dimension k
    /// 2. Initialize LBM at equilibrium
    /// 3. Run LBM steps (with optional convergence early-out)
    /// 4. Count halo cells via warp-level atomic reduction
    ///
    /// # Arguments
    /// * `k` - CD algebra dimension parameter
    /// * `steps` - Maximum number of LBM steps
    /// * `tau_base` - Base relaxation time
    /// * `tau_amp` - Amplitude of ZD-modulated viscosity
    /// * `rho_threshold` - Density threshold for halo classification
    /// * `velocity_epsilon` - Velocity threshold for halo classification
    /// * `convergence_tol` - If > 0, enable early stopping when delta_rho < tol
    /// * `check_interval` - Steps between convergence checks
    #[expect(clippy::too_many_arguments)]
    pub fn run_k_value(
        &mut self,
        k: usize,
        steps: u32,
        tau_base: f32,
        tau_amp: f32,
        rho_threshold: f32,
        velocity_epsilon: f32,
        convergence_tol: f32,
        check_interval: u32,
    ) -> Result<CudaDarkHaloResult> {
        let n = self.n_cells;
        let nx = self.nx as i32;
        let ny = self.ny as i32;
        let nz = self.nz as i32;
        let lambda = (k as f32).ln();
        let threads = 128u32;
        let blocks = (n as u32).div_ceil(threads);
        let cfg = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        // Step 1: Generate ZD viscosity field
        let seed = 42u32;
        let k_dim = k as i32;
        unsafe {
            let mut b = self.stream.launch_builder(&self.zd_viscosity_kernel);
            b.arg(&mut self.d_tau)
                .arg(&nx)
                .arg(&ny)
                .arg(&nz)
                .arg(&tau_base)
                .arg(&tau_amp)
                .arg(&lambda)
                .arg(&seed)
                .arg(&k_dim);
            b.launch(cfg).context("ZD viscosity modulation")?;
        }

        // Step 2: Initialize LBM at equilibrium (rho=1, u=0)
        // For SoA layout, f[dir * N + idx] = W[dir] for equilibrium at rho=1, u=0
        let equilibrium: Vec<f32> = (0..19)
            .flat_map(|dir| {
                let w = match dir {
                    0 => 1.0 / 3.0,
                    1..=6 => 1.0 / 18.0,
                    _ => 1.0 / 36.0,
                };
                std::iter::repeat_n(w, n)
            })
            .collect();
        self.d_f = self
            .stream
            .clone_htod(&equilibrium)
            .context("Init f equilibrium")?;
        self.d_f_tmp = self
            .stream
            .clone_htod(&equilibrium)
            .context("Init f_tmp equilibrium")?;

        // Step 2b: Pin rho in L2 cache (Ada Lovelace, best-effort)
        if let Err(e) = self.pin_rho_in_l2() {
            eprintln!("    L2 pinning skipped: {e}");
        }

        // Step 3: Run LBM steps with ping-pong via swap
        let mut steps_run = 0u32;
        let early_stopped = false;
        // NULL pointer for no forcing -- pass as a raw device pointer value
        let null_force: u64 = 0;

        for step in 0..steps {
            unsafe {
                let mut b = self.stream.launch_builder(&self.lbm_step_kernel);
                b.arg(&self.d_f)
                    .arg(&mut self.d_f_tmp)
                    .arg(&mut self.d_rho)
                    .arg(&mut self.d_u)
                    .arg(&self.d_tau)
                    .arg(&null_force)
                    .arg(&nx)
                    .arg(&ny)
                    .arg(&nz);
                b.launch(cfg).context("LBM step")?;
            }
            std::mem::swap(&mut self.d_f, &mut self.d_f_tmp);
            steps_run += 1;

            // Convergence check every check_interval steps
            if convergence_tol > 0.0 && check_interval > 0 && step > 0 && step % check_interval == 0
            {
                // TODO: implement delta-rho convergence via convergence_check kernel
                // For now, just run all steps (stub)
                let _ = &self.d_rho_prev;
            }
        }

        // Step 4: Compute actual rho_mean from evolved field, then detect dark halos
        //
        // The caller's rho_threshold is a "density_factor" multiplier. The CUDA
        // kernel checks `rho > threshold` (overdense wells, matching the WGSL
        // dark_halo_detector.wgsl criterion 3). We compute the actual mean
        // density and set threshold = density_factor * rho_mean so that
        // density_factor=1.5 finds cells at >= 150% of mean density.
        let rho_host = self
            .stream
            .clone_dtoh(&self.d_rho)
            .context("Readback rho for mean")?;
        let rho_sum: f64 = rho_host.iter().map(|&r| r as f64).sum();
        let rho_mean = (rho_sum / n as f64) as f32;
        let effective_threshold = rho_threshold * rho_mean;

        let zero_count = vec![0u32; 1];
        self.d_halo_count = self
            .stream
            .clone_htod(&zero_count)
            .context("Zero halo count")?;

        let n_i32 = n as i32;
        unsafe {
            let mut b = self.stream.launch_builder(&self.halo_detector_kernel);
            b.arg(&self.d_rho)
                .arg(&self.d_u)
                .arg(&mut self.d_halo_count)
                .arg(&effective_threshold)
                .arg(&velocity_epsilon)
                .arg(&n_i32);
            b.launch(cfg).context("Halo detection")?;
        }

        // Read back result
        let halo_count = self
            .stream
            .clone_dtoh(&self.d_halo_count)
            .context("Read halo count")?;

        let count = halo_count[0];
        let vf = count as f64 / n as f64;

        Ok(CudaDarkHaloResult {
            k,
            volume_fraction: vf,
            halo_count: count,
            n_cells: n as u32,
            early_stopped,
            steps_run,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn gpu_available() -> bool {
        CudaContext::new(0).is_ok()
    }

    fn maybe_solver(precision: Precision) -> Option<LbmSolver3DCuda> {
        if !gpu_available() {
            eprintln!("Skipping GPU test: CUDA device unavailable");
            return None;
        }
        match LbmSolver3DCuda::new(4, 4, 4, 0.6, precision) {
            Ok(solver) => Some(solver),
            Err(err) => {
                eprintln!("Skipping GPU test: failed to initialize solver ({err})");
                None
            }
        }
    }

    #[test]
    fn init_and_first_step_are_finite_fp32() {
        let Some(mut solver) = maybe_solver(Precision::FP32) else {
            return;
        };
        let mean0 = solver
            .calculate_mean_density()
            .expect("mean density should compute");
        assert!(mean0.is_finite());
        assert_abs_diff_eq!(mean0, 1.0, epsilon = 1.0e-3);

        solver.step().expect("first step should succeed");
        let mean1 = solver
            .calculate_mean_density()
            .expect("mean density after first step");
        assert!(mean1.is_finite());
        assert_abs_diff_eq!(mean1, 1.0, epsilon = 1.0e-2);
    }

    #[test]
    fn mean_density_is_mean_not_sum() {
        let Some(mut solver) = maybe_solver(Precision::FP32) else {
            return;
        };
        // With uniform rho=1 init, true mean must be ~1 regardless of cell count.
        let mean = solver
            .calculate_mean_density()
            .expect("mean density should compute");
        assert!(mean.is_finite());
        assert!(mean < 2.0, "expected mean close to 1, got {mean}");
    }

    #[test]
    fn sync_to_host_populates_fresh_buffers_bf16() {
        let Some(mut solver) = maybe_solver(Precision::BF16) else {
            return;
        };
        solver.step().expect("step should succeed");
        solver.sync_to_host().expect("sync_to_host should succeed");

        assert_eq!(solver.rho.len(), solver.n_cells);
        assert_eq!(solver.u.len(), solver.n_cells);
        assert!(solver.rho.iter().all(|v| v.is_finite()));
        assert!(
            solver
                .u
                .iter()
                .all(|v| v[0].is_finite() && v[1].is_finite() && v[2].is_finite())
        );
    }

    #[test]
    fn init_and_first_step_are_finite_fp64() {
        let Some(mut solver) = maybe_solver(Precision::FP64) else {
            return;
        };
        let mean0 = solver
            .calculate_mean_density()
            .expect("mean density should compute");
        assert!(mean0.is_finite());
        assert_abs_diff_eq!(mean0, 1.0, epsilon = 1.0e-5);

        solver.step().expect("first step should succeed");
        let mean1 = solver
            .calculate_mean_density()
            .expect("mean density after first step");
        assert!(mean1.is_finite());
        assert_abs_diff_eq!(mean1, 1.0, epsilon = 1.0e-4);

        solver.sync_to_host().expect("sync_to_host should succeed");
        assert!(solver.rho.iter().all(|v| v.is_finite()));
        assert!(
            solver
                .u
                .iter()
                .all(|v| v[0].is_finite() && v[1].is_finite() && v[2].is_finite())
        );
    }

    #[test]
    fn zero_density_guard_prevents_nan_propagation_fp32() {
        let Some(mut solver) = maybe_solver(Precision::FP32) else {
            return;
        };
        solver
            .set_all_distributions_for_test(0.0)
            .expect("test setup should succeed");
        solver
            .step()
            .expect("step should succeed with zeroed distributions");

        let mean = solver
            .calculate_mean_density()
            .expect("mean density should compute");
        assert!(mean.is_finite());
        assert!(mean > 0.0);

        solver.sync_to_host().expect("sync_to_host should succeed");
        assert!(solver.rho.iter().all(|v| v.is_finite()));
    }

    // ---- CPU-side tests (no GPU required) ----

    #[test]
    fn launch_config_1d_fp32_uses_1024_threads() {
        let cfg = LbmSolver3DCuda::launch_config_1d(2048, Precision::FP32);
        assert_eq!(cfg.block_dim, (1024, 1, 1));
        assert_eq!(cfg.grid_dim, (2, 1, 1));
    }

    #[test]
    fn launch_config_1d_fp64_uses_128_threads() {
        let cfg = LbmSolver3DCuda::launch_config_1d(2048, Precision::FP64);
        assert_eq!(cfg.block_dim, (128, 1, 1));
        assert_eq!(cfg.grid_dim, (16, 1, 1));
    }

    #[test]
    fn launch_config_1d_single_element() {
        let cfg = LbmSolver3DCuda::launch_config_1d(1, Precision::FP32);
        assert_eq!(cfg.grid_dim.0, 1, "at least 1 block for 1 element");
    }

    #[test]
    fn launch_config_1d_bf16_uses_1024_threads() {
        let cfg = LbmSolver3DCuda::launch_config_1d(512, Precision::BF16);
        assert_eq!(cfg.block_dim, (1024, 1, 1));
        assert_eq!(cfg.grid_dim, (1, 1, 1));
    }

    #[test]
    fn encode_f32_roundtrip() {
        let values = vec![1.0f32, -2.5, 7.25, 0.0];
        let bytes = LbmSolver3DCuda::encode_f32_to_bytes(&values);
        assert_eq!(bytes.len(), 16);
        let mut decoded = vec![0.0f32; 4];
        LbmSolver3DCuda::decode_f32_from_bytes(&bytes, &mut decoded).unwrap();
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-7);
        }
    }

    #[test]
    fn encode_bf16_roundtrip() {
        let values = vec![1.0f32, -2.0, 0.5, 0.0];
        let bytes = LbmSolver3DCuda::encode_bf16_to_bytes(&values);
        assert_eq!(bytes.len(), 8);
        let mut decoded = vec![0.0f32; 4];
        LbmSolver3DCuda::decode_bf16_from_bytes(&bytes, &mut decoded).unwrap();
        for (a, b) in values.iter().zip(decoded.iter()) {
            // BF16 has limited precision (~3 decimal digits)
            assert_abs_diff_eq!(a, b, epsilon = 0.02);
        }
    }

    #[test]
    fn encode_f64_to_f32_roundtrip() {
        let values = vec![1.0f64, -2.5, 7.25, 0.0];
        let bytes = LbmSolver3DCuda::encode_f64_to_bytes(&values);
        assert_eq!(bytes.len(), 32);
        let mut decoded = vec![0.0f32; 4];
        LbmSolver3DCuda::decode_f64_from_bytes_to_f32(&bytes, &mut decoded).unwrap();
        for (a, b) in values.iter().zip(decoded.iter()) {
            assert_abs_diff_eq!(*a as f32, *b, epsilon = 1e-6);
        }
    }

    #[test]
    fn decode_f32_rejects_wrong_length() {
        let bytes = vec![0u8; 5]; // not a multiple of 4
        let mut out = vec![0.0f32; 2];
        assert!(LbmSolver3DCuda::decode_f32_from_bytes(&bytes, &mut out).is_err());
    }

    #[test]
    fn decode_bf16_rejects_wrong_length() {
        let bytes = vec![0u8; 3]; // not a multiple of 2
        let mut out = vec![0.0f32; 2];
        assert!(LbmSolver3DCuda::decode_bf16_from_bytes(&bytes, &mut out).is_err());
    }
}
