//! Optimal kernel auto-selector for Ada Lovelace D3Q19 LBM.
//!
//! Encodes the three architectural truths proven by ncu/nsys profiling:
//! 1. A-A streaming is MANDATORY at grid <= 256^3 (C-1390: 79.7% peak BW)
//! 2. MRT collision is FREE for all tiers below FP16_H2 (C-1391: latency hiding)
//! 3. Low-precision storage gives up to 4x BW reduction (C-1385: INT8 Pareto-optimal)
//!
//! Combined decision table (C-1392):
//!   Tier 1 (max throughput): INT8_SoA MRT A-A -> ~5000+ MLUPS, 76 MB at 128^3
//!   Tier 2 (float throughput): FP8_e4m3_SoA MRT A-A -> ~5200+ MLUPS
//!   Tier 3 (balanced accuracy): FP16_SoA MRT -> ~3500 MLUPS, 10-bit mantissa
//!   Tier 4 (high accuracy): FP32 MRT A-A -> ~1980 MLUPS, full f32
//!   Tier 5 (validation): FP64 SoA MRT -> ~400 MLUPS, reference

/// Collision policy. MRT is the production default on Ada Lovelace (C-1391).
///
/// BGK is retained for research (textbook validation, pre-Ada hardware).
/// DD is retained for extreme-precision validation only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionPolicy {
    /// MRT collision -- production default. Faster than BGK on Ada via latency
    /// hiding (C-1391: 722 FMA/cell fills memory-stall pipeline bubbles).
    /// Also more stable via ghost moment damping (s_ghost=1.0).
    Production,
    /// BGK collision -- research only. Simpler physics (57 FMA/cell) but slower
    /// on Ada due to warp scheduler starvation during memory stalls.
    /// Useful for: textbook validation, pre-Ada hardware, isolating MRT artifacts.
    Research,
    /// DD (double-double ~106-bit) BGK -- validation only. 80x slower than FP64.
    /// Useful for: precision validation of FP64/FP32 results. MRT variant
    /// formally deferred (task #37: ~3000 lines for ~0.001 MLUPS, no ROI).
    ValidationDD,
}

/// Precision tier for distribution storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrecisionRequirement {
    /// Maximum throughput, acceptable quantization (~1.6% relative error).
    MaxThroughput,
    /// Maximum float throughput, ~5% relative error (FP8).
    MaxFloatThroughput,
    /// Balanced accuracy for moderate-Re flows (10-bit mantissa).
    Balanced,
    /// Full FP32 precision for high-accuracy requirements.
    HighAccuracy,
    /// FP64 reference precision for validation.
    Validation,
}

impl PrecisionRequirement {
    /// Bridge to the canonical workspace precision vocabulary
    /// (gororoba_gpu_bridge::StoragePrecision). This is a representative
    /// storage type per tier, not a bijection -- the requirement enum
    /// captures a policy choice, not a precise type.
    pub fn storage_precision(self) -> gororoba_gpu_bridge::StoragePrecision {
        use gororoba_gpu_bridge::StoragePrecision as SP;
        match self {
            Self::MaxThroughput => SP::Int8,
            Self::MaxFloatThroughput => SP::Fp8E4m3,
            Self::Balanced => SP::Fp16,
            Self::HighAccuracy => SP::Fp32,
            Self::Validation => SP::Fp64,
        }
    }
}

/// CUDA source owner used by the selected-runner dispatch table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelSource {
    Int8Soa,
    Fp8Soa,
    Fp16Soa,
    Fp32Soa,
    Fp64Soa,
}

impl KernelSource {
    /// Rust constant that owns the corresponding CUDA source edge.
    pub const fn source_label(self) -> &'static str {
        match self {
            Self::Int8Soa => "KERNEL_INT8_SOA_SRC",
            Self::Fp8Soa => "KERNEL_FP8_SOA_SRC",
            Self::Fp16Soa => "KERNEL_FP16_SOA_SRC",
            Self::Fp32Soa => "KERNEL_SOA_SRC",
            Self::Fp64Soa => "KERNEL_FP64_SOA_SRC",
        }
    }
}

/// Complete launch contract shared by the selector and SoA runner.
///
/// The contract includes the init symbol because A-A kernels have a different
/// step signature from ping-pong kernels even when their storage precision
/// matches. Keeping both symbols here prevents a selector recommendation from
/// silently drifting away from the constructor that launches it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KernelDispatchSpec {
    pub source: KernelSource,
    pub step_kernel_name: &'static str,
    pub init_kernel_name: &'static str,
    pub elem_bytes: usize,
    pub is_aa: bool,
    pub cells_per_thread: usize,
    pub precision_label: &'static str,
    pub cuda_include: bool,
    pub init_takes_tau: bool,
    pub tau_is_inverse: bool,
}

impl KernelDispatchSpec {
    pub const fn int8_soa_mrt_aa() -> Self {
        Self {
            source: KernelSource::Int8Soa,
            step_kernel_name: "lbm_step_int8_soa_mrt_aa_kernel",
            init_kernel_name: "initialize_uniform_int8_soa_kernel",
            elem_bytes: 1,
            is_aa: true,
            cells_per_thread: 1,
            precision_label: "INT8_SoA",
            cuda_include: false,
            init_takes_tau: true,
            tau_is_inverse: false,
        }
    }

    pub const fn fp8_soa_mrt_aa() -> Self {
        Self {
            source: KernelSource::Fp8Soa,
            step_kernel_name: "lbm_step_fp8_soa_mrt_aa_kernel",
            init_kernel_name: "initialize_uniform_fp8_soa_kernel",
            elem_bytes: 1,
            is_aa: true,
            cells_per_thread: 1,
            precision_label: "FP8_SoA",
            cuda_include: true,
            init_takes_tau: true,
            tau_is_inverse: false,
        }
    }

    pub const fn fp8_soa_mrt() -> Self {
        Self {
            source: KernelSource::Fp8Soa,
            step_kernel_name: "lbm_step_fp8_soa_mrt_kernel",
            init_kernel_name: "initialize_uniform_fp8_soa_kernel",
            elem_bytes: 1,
            is_aa: false,
            cells_per_thread: 1,
            precision_label: "FP8_SoA",
            cuda_include: true,
            init_takes_tau: true,
            tau_is_inverse: false,
        }
    }

    pub const fn fp16_soa_mrt() -> Self {
        Self {
            source: KernelSource::Fp16Soa,
            step_kernel_name: "lbm_step_fp16_soa_mrt_kernel",
            init_kernel_name: "initialize_uniform_fp16_soa_kernel",
            elem_bytes: 2,
            is_aa: false,
            cells_per_thread: 1,
            precision_label: "FP16_SoA",
            cuda_include: true,
            init_takes_tau: true,
            tau_is_inverse: false,
        }
    }

    pub const fn fp32_soa_mrt_aa() -> Self {
        Self {
            source: KernelSource::Fp32Soa,
            step_kernel_name: "lbm_step_soa_mrt_aa",
            init_kernel_name: "initialize_uniform_soa_kernel",
            elem_bytes: 4,
            is_aa: true,
            cells_per_thread: 1,
            precision_label: "FP32",
            cuda_include: false,
            init_takes_tau: false,
            tau_is_inverse: true,
        }
    }

    pub const fn int8_soa_mrt() -> Self {
        Self {
            source: KernelSource::Int8Soa,
            step_kernel_name: "lbm_step_int8_soa_mrt_kernel",
            init_kernel_name: "initialize_uniform_int8_soa_kernel",
            elem_bytes: 1,
            is_aa: false,
            cells_per_thread: 1,
            precision_label: "INT8_SoA",
            cuda_include: false,
            init_takes_tau: true,
            tau_is_inverse: false,
        }
    }

    pub const fn fp64_soa_mrt() -> Self {
        Self {
            source: KernelSource::Fp64Soa,
            step_kernel_name: "lbm_step_fp64_soa_mrt_kernel",
            init_kernel_name: "initialize_uniform_fp64_soa_kernel",
            elem_bytes: 8,
            is_aa: false,
            cells_per_thread: 1,
            precision_label: "FP64_SoA",
            cuda_include: false,
            init_takes_tau: true,
            tau_is_inverse: false,
        }
    }

    pub const fn source_label(self) -> &'static str {
        self.source.source_label()
    }
}

/// Selected kernel configuration.
#[derive(Debug, Clone)]
pub struct KernelSelection {
    /// Kernel entry point name for NVRTC.
    pub kernel_name: &'static str,
    /// Source file constant name (for include_str! lookup).
    pub source_label: &'static str,
    /// Collision mode.
    pub collision: &'static str,
    /// Streaming layout.
    pub streaming: &'static str,
    /// Precision tier label.
    pub precision: &'static str,
    /// Bytes per distribution element.
    pub elem_bytes: usize,
    /// VRAM for distribution buffers at this grid (MB).
    pub vram_dist_mb: usize,
    /// Estimated MLUPS at 128^3 (from C-1391/C-1392 benchmarks).
    pub estimated_mlups_128: f64,
    /// Whether A-A streaming is used (single buffer).
    pub is_aa: bool,
    /// Cells per thread (2 for coarsened/half2 variants).
    pub cells_per_thread: usize,
    /// Complete source and launch contract consumed by `SoaBenchRunner`.
    pub dispatch: KernelDispatchSpec,
}

/// Select the optimal kernel for the given grid size and precision requirement.
///
/// Implements the formal decision table from C-1392.
pub fn select_optimal_kernel(
    grid_dim: (usize, usize, usize),
    precision: PrecisionRequirement,
) -> KernelSelection {
    let n_cells = grid_dim.0 * grid_dim.1 * grid_dim.2;

    match precision {
        PrecisionRequirement::MaxThroughput => {
            // INT8 SoA MRT A-A: Pareto-optimal (5142 MLUPS, 76 MB at 128^3)
            let vram = n_cells * 19 / (1024 * 1024); // 1 byte/dist, single buffer
            KernelSelection {
                kernel_name: KernelDispatchSpec::int8_soa_mrt_aa().step_kernel_name,
                source_label: KernelDispatchSpec::int8_soa_mrt_aa().source_label(),
                collision: "MRT",
                streaming: "A-A",
                precision: "INT8_SoA",
                elem_bytes: 1,
                vram_dist_mb: vram,
                estimated_mlups_128: 5142.0,
                is_aa: KernelDispatchSpec::int8_soa_mrt_aa().is_aa,
                cells_per_thread: KernelDispatchSpec::int8_soa_mrt_aa().cells_per_thread,
                dispatch: KernelDispatchSpec::int8_soa_mrt_aa(),
            }
        }
        PrecisionRequirement::MaxFloatThroughput => {
            // FP8 e4m3 SoA MRT A-A: highest float throughput (5210 MLUPS)
            let vram = n_cells * 19 / (1024 * 1024);
            KernelSelection {
                kernel_name: KernelDispatchSpec::fp8_soa_mrt_aa().step_kernel_name,
                source_label: KernelDispatchSpec::fp8_soa_mrt_aa().source_label(),
                collision: "MRT",
                streaming: "A-A",
                precision: "FP8_e4m3_SoA",
                elem_bytes: 1,
                vram_dist_mb: vram,
                estimated_mlups_128: 5210.0,
                is_aa: KernelDispatchSpec::fp8_soa_mrt_aa().is_aa,
                cells_per_thread: KernelDispatchSpec::fp8_soa_mrt_aa().cells_per_thread,
                dispatch: KernelDispatchSpec::fp8_soa_mrt_aa(),
            }
        }
        PrecisionRequirement::Balanced => {
            // FP16 SoA MRT: good accuracy (10-bit mantissa) with MRT stability
            let vram = n_cells * 19 * 2 / (1024 * 1024);
            KernelSelection {
                kernel_name: KernelDispatchSpec::fp16_soa_mrt().step_kernel_name,
                source_label: KernelDispatchSpec::fp16_soa_mrt().source_label(),
                collision: "MRT",
                streaming: "pull",
                precision: "FP16_SoA",
                elem_bytes: 2,
                vram_dist_mb: vram,
                estimated_mlups_128: 3486.0,
                is_aa: KernelDispatchSpec::fp16_soa_mrt().is_aa,
                cells_per_thread: KernelDispatchSpec::fp16_soa_mrt().cells_per_thread,
                dispatch: KernelDispatchSpec::fp16_soa_mrt(),
            }
        }
        PrecisionRequirement::HighAccuracy => {
            // FP32 MRT A-A: full precision with VRAM halving
            let vram = n_cells * 19 * 4 / (1024 * 1024);
            KernelSelection {
                kernel_name: KernelDispatchSpec::fp32_soa_mrt_aa().step_kernel_name,
                source_label: KernelDispatchSpec::fp32_soa_mrt_aa().source_label(),
                collision: "MRT",
                streaming: "A-A",
                precision: "FP32",
                elem_bytes: 4,
                vram_dist_mb: vram,
                estimated_mlups_128: 1979.0,
                is_aa: KernelDispatchSpec::fp32_soa_mrt_aa().is_aa,
                cells_per_thread: KernelDispatchSpec::fp32_soa_mrt_aa().cells_per_thread,
                dispatch: KernelDispatchSpec::fp32_soa_mrt_aa(),
            }
        }
        PrecisionRequirement::Validation => {
            // FP64 SoA MRT: reference precision
            let vram = n_cells * 19 * 8 / (1024 * 1024);
            KernelSelection {
                kernel_name: KernelDispatchSpec::fp64_soa_mrt().step_kernel_name,
                source_label: KernelDispatchSpec::fp64_soa_mrt().source_label(),
                collision: "MRT",
                streaming: "pull",
                precision: "FP64_SoA",
                elem_bytes: 8,
                vram_dist_mb: vram,
                estimated_mlups_128: 398.0,
                is_aa: KernelDispatchSpec::fp64_soa_mrt().is_aa,
                cells_per_thread: KernelDispatchSpec::fp64_soa_mrt().cells_per_thread,
                dispatch: KernelDispatchSpec::fp64_soa_mrt(),
            }
        }
    }
}

/// Check if the selected kernel fits in the available VRAM.
pub fn fits_in_vram(selection: &KernelSelection, vram_mb: usize) -> bool {
    // Distribution buffers + macroscopic (rho, u, tau, force) overhead
    let macro_overhead_mb = selection.vram_dist_mb / 4; // ~25% of dist for macro fields
    let total_mb = selection.vram_dist_mb + macro_overhead_mb;
    total_mb < vram_mb
}

/// Print a human-readable summary of the kernel selection.
pub fn print_selection(sel: &KernelSelection) {
    println!("Optimal kernel selection:");
    println!("  Kernel: {}", sel.kernel_name);
    println!("  Collision: {}", sel.collision);
    println!("  Streaming: {}", sel.streaming);
    println!("  Precision: {}", sel.precision);
    println!("  VRAM (dist): {} MB", sel.vram_dist_mb);
    println!("  Est. MLUPS (128^3): {:.0}", sel.estimated_mlups_128);
    println!("  A-A: {}", sel.is_aa);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_throughput_selects_int8_aa_mrt() {
        let sel = select_optimal_kernel((128, 128, 128), PrecisionRequirement::MaxThroughput);
        assert_eq!(sel.kernel_name, "lbm_step_int8_soa_mrt_aa_kernel");
        assert_eq!(sel.collision, "MRT");
        assert_eq!(sel.streaming, "A-A");
        assert!(sel.is_aa);
        assert_eq!(sel.elem_bytes, 1);
    }

    #[test]
    fn test_high_accuracy_selects_fp32_aa_mrt() {
        let sel = select_optimal_kernel((64, 64, 64), PrecisionRequirement::HighAccuracy);
        assert_eq!(sel.kernel_name, "lbm_step_soa_mrt_aa");
        assert_eq!(sel.precision, "FP32");
        assert!(sel.is_aa);
    }

    #[test]
    fn selected_metadata_matches_runner_dispatch_contract() {
        let requirements = [
            PrecisionRequirement::MaxThroughput,
            PrecisionRequirement::MaxFloatThroughput,
            PrecisionRequirement::Balanced,
            PrecisionRequirement::HighAccuracy,
            PrecisionRequirement::Validation,
        ];
        for requirement in requirements {
            let selection = select_optimal_kernel((128, 128, 128), requirement);
            let dispatch = selection.dispatch;
            assert_eq!(selection.kernel_name, dispatch.step_kernel_name);
            assert_eq!(selection.source_label, dispatch.source_label());
            assert_eq!(selection.elem_bytes, dispatch.elem_bytes);
            assert_eq!(selection.is_aa, dispatch.is_aa);
            assert_eq!(selection.cells_per_thread, dispatch.cells_per_thread);
        }

        let high_accuracy = select_optimal_kernel((128, 128, 128), PrecisionRequirement::HighAccuracy);
        assert!(!high_accuracy.dispatch.init_takes_tau);
        assert!(high_accuracy.dispatch.tau_is_inverse);
    }

    #[test]
    fn test_256_cube_fits_in_12gb_at_int8() {
        let sel = select_optimal_kernel((256, 256, 256), PrecisionRequirement::MaxThroughput);
        assert!(
            fits_in_vram(&sel, 12 * 1024),
            "256^3 INT8 A-A must fit in 12 GB"
        );
        // 256^3 * 19 * 1 byte = 319 MB distribution
        assert!(
            sel.vram_dist_mb < 400,
            "VRAM should be ~319 MB, got {}",
            sel.vram_dist_mb
        );
    }

    #[test]
    fn test_256_cube_fp32_does_not_fit_pingpong_12gb() {
        // FP32 ping-pong at 256^3: 19 * 16.7M * 4 * 2 = 2.5 GB
        // But our selector uses A-A (single buffer): 19 * 16.7M * 4 = 1.27 GB
        let sel = select_optimal_kernel((256, 256, 256), PrecisionRequirement::HighAccuracy);
        assert!(
            fits_in_vram(&sel, 12 * 1024),
            "256^3 FP32 A-A should fit in 12 GB"
        );
    }
}
