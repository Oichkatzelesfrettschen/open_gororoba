// ============================================================================
// Dark Halo Hunt CUDA solver (SoA layout, optimized for Ada Lovelace)
// ============================================================================

use std::sync::Arc;

use anyhow::{Context, Result};
use cudarc::driver::{CudaFunction, CudaStream, DevicePtr, PushKernelArg};
use gororoba_gpu_cuda::{
    Buffer, CompileOptions, Context as CudaContextHelper, LaunchConfig, ModuleRegistry,
};
use gororoba_gpu_readback::{
    ReadbackBufferShape, ReadbackDescriptor, ReadbackElementType, ReadbackLayout, ReadbackResidency,
};

use crate::preferred_cuda_arch;

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
    _ctx: CudaContextHelper,
    stream: Arc<CudaStream>,
    _module_registry: ModuleRegistry,
    // SoA distribution function buffers: [19 * N] floats
    d_f: Buffer<f32>,
    d_f_tmp: Buffer<f32>,
    // Macroscopic fields
    d_rho: Buffer<f32>,
    d_rho_prev: Buffer<f32>, // for convergence check
    d_u: Buffer<f32>,        // [3 * N] SoA: ux[N], uy[N], uz[N]
    d_tau: Buffer<f32>,
    // Atomic counter for halo detection
    d_halo_count: Buffer<u32>,
    d_delta_sum: Buffer<f32>,
    // Kernels
    lbm_step_kernel: CudaFunction,
    halo_detector_kernel: CudaFunction,
    zd_viscosity_kernel: CudaFunction,
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
        let ctx = CudaContextHelper::with_default_device().context("CUDA context creation")?;
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
            // SAFETY: cuMemGetInfo_v2 is a read-only CUDA driver API call that
            // writes to stack-local variables. The CUDA context is valid
            // (established by CudaDevice::new in the enclosing constructor).
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

        let opts = CompileOptions::with_arch(preferred_cuda_arch())
            .prec_div(false)
            .prec_sqrt(false)
            .ftz(true)
            .fmad(true);
        let ptx = CompileOptions::compile_ptx(KERNEL_DARK_HALO_SRC, &opts)
            .context("NVRTC compilation of dark halo kernels")?;
        let module_registry = ModuleRegistry::load(
            ctx.raw(),
            ptx,
            &[
                "lbm_step_soa_fused",
                "dark_halo_detector",
                "zd_viscosity_modulation",
                "convergence_check",
            ],
        )
        .context("Load dark halo CUDA module")?;

        let lbm_step_kernel = module_registry.get("lbm_step_soa_fused")?;
        let halo_detector_kernel = module_registry.get("dark_halo_detector")?;
        let zd_viscosity_kernel = module_registry.get("zd_viscosity_modulation")?;
        let convergence_kernel = module_registry.get("convergence_check")?;

        // Allocate SoA buffers (cudarc 0.19: allocations go through CudaStream)
        let d_f = Buffer::alloc_zeros(&stream, 19 * n_cells).context("Alloc d_f")?;
        let d_f_tmp = Buffer::alloc_zeros(&stream, 19 * n_cells).context("Alloc d_f_tmp")?;
        let d_rho = Buffer::alloc_zeros(&stream, n_cells).context("Alloc d_rho")?;
        let d_rho_prev = Buffer::alloc_zeros(&stream, n_cells).context("Alloc d_rho_prev")?;
        let d_u = Buffer::alloc_zeros(&stream, 3 * n_cells).context("Alloc d_u")?;
        let d_tau = Buffer::alloc_zeros(&stream, n_cells).context("Alloc d_tau")?;
        let d_halo_count = Buffer::alloc_zeros(&stream, 1).context("Alloc d_halo_count")?;
        let d_delta_sum = Buffer::alloc_zeros(&stream, 1).context("Alloc d_delta_sum")?;

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
            _ctx: ctx,
            stream,
            _module_registry: module_registry,
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
        let (rho_ptr, _guard) = self.d_rho.raw().device_ptr(&self.stream);

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
            .d_u
            .dtoh_vec()
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
                    let cmp_mask = rho_vec.simd_gt(threshold_vec);
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
            .memcpy_dtoh(self.d_rho.raw(), &mut pinned)
            .map_err(|e| anyhow::anyhow!("DMA dtoh failed: {e}"))?;
        Ok(pinned)
    }

    /// Descriptor for the pinned host-side density readback surface.
    #[must_use]
    pub fn rho_readback_descriptor(&self) -> ReadbackDescriptor {
        ReadbackDescriptor {
            backend_name: "CUDA".to_string(),
            label: "rho".to_string(),
            shape: ReadbackBufferShape {
                width: self.nx as u32,
                height: self.ny as u32,
                depth: self.nz as u32,
                elements_per_point: 1,
            },
            element_type: ReadbackElementType::F32,
            layout: ReadbackLayout::Packed,
            residency: ReadbackResidency::PinnedHost,
        }
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
        let cfg = LaunchConfig::launch_blocks_1d(blocks, threads);

        // Step 1: Generate ZD viscosity field
        let seed = 42u32;
        let k_dim = k as i32;
        unsafe {
            let mut b = self.stream.launch_builder(&self.zd_viscosity_kernel);
            b.arg(self.d_tau.raw_mut())
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
        self.d_f = Buffer::htod(&self.stream, &equilibrium).context("Init f equilibrium")?;
        self.d_f_tmp =
            Buffer::htod(&self.stream, &equilibrium).context("Init f_tmp equilibrium")?;

        // Step 2b: Pin rho in L2 cache (Ada Lovelace, best-effort)
        if let Err(e) = self.pin_rho_in_l2() {
            eprintln!("    L2 pinning skipped: {e}");
        }

        // Step 3: Run LBM steps with ping-pong via swap
        let mut steps_run = 0u32;
        let mut early_stopped = false;
        // NULL pointer for no forcing -- pass as a raw device pointer value
        let null_force: u64 = 0;

        for step in 0..steps {
            unsafe {
                let mut b = self.stream.launch_builder(&self.lbm_step_kernel);
                b.arg(self.d_f.raw())
                    .arg(self.d_f_tmp.raw_mut())
                    .arg(self.d_rho.raw_mut())
                    .arg(self.d_u.raw_mut())
                    .arg(self.d_tau.raw())
                    .arg(&null_force)
                    .arg(&nx)
                    .arg(&ny)
                    .arg(&nz);
                b.launch(cfg).context("LBM step")?;
            }
            std::mem::swap(&mut self.d_f, &mut self.d_f_tmp);
            steps_run += 1;

            // Convergence check every check_interval steps.
            // Compares d_rho against the snapshot taken at the previous interval
            // (or zeros on the first interval -- that check always fails, which
            // is correct since the simulation has not yet reached steady state).
            if convergence_tol > 0.0 && check_interval > 0 && step > 0 && step % check_interval == 0
            {
                // Zero the atomic accumulator before each launch.
                self.d_delta_sum =
                    Buffer::htod(&self.stream, &[0.0f32]).context("Zero d_delta_sum")?;

                let n_i32 = n as i32;
                unsafe {
                    let mut b = self.stream.launch_builder(&self.convergence_kernel);
                    b.arg(self.d_rho.raw())
                        .arg(self.d_rho_prev.raw())
                        .arg(self.d_delta_sum.raw_mut())
                        .arg(&n_i32);
                    b.launch(cfg).context("Convergence check kernel")?;
                }

                // dtoh_vec synchronizes the stream before returning.
                let delta_buf = self.d_delta_sum.dtoh_vec().context("Read d_delta_sum")?;
                let delta_mean = delta_buf[0] / n as f32;

                // Update the reference snapshot for the next interval.
                self.stream
                    .memcpy_dtod(self.d_rho.raw(), self.d_rho_prev.raw_mut())
                    .context("Update rho_prev")?;

                if delta_mean < convergence_tol {
                    early_stopped = true;
                    break;
                }
            }
        }

        // Step 4: Compute actual rho_mean from evolved field, then detect dark halos
        //
        // The caller's rho_threshold is a "density_factor" multiplier. The CUDA
        // kernel checks `rho > threshold` (overdense wells, matching the WGSL
        // dark_halo_detector.wgsl criterion 3). We compute the actual mean
        // density and set threshold = density_factor * rho_mean so that
        // density_factor=1.5 finds cells at >= 150% of mean density.
        let rho_host = self.d_rho.dtoh_vec().context("Readback rho for mean")?;
        let rho_sum: f64 = rho_host.iter().map(|&r| r as f64).sum();
        let rho_mean = (rho_sum / n as f64) as f32;
        let effective_threshold = rho_threshold * rho_mean;

        let zero_count = vec![0u32; 1];
        self.d_halo_count = Buffer::htod(&self.stream, &zero_count).context("Zero halo count")?;

        let n_i32 = n as i32;
        unsafe {
            let mut b = self.stream.launch_builder(&self.halo_detector_kernel);
            b.arg(self.d_rho.raw())
                .arg(self.d_u.raw())
                .arg(self.d_halo_count.raw_mut())
                .arg(&effective_threshold)
                .arg(&velocity_epsilon)
                .arg(&n_i32);
            b.launch(cfg).context("Halo detection")?;
        }

        // Read back result
        let halo_count = self.d_halo_count.dtoh_vec().context("Read halo count")?;

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
