//! GPU backend selection and dispatch for gororoba simulation workloads.
//!
//! Provides a runtime-switchable backend chooser that selects between CPU
//! and GPU execution based on problem size and hardware availability.
//!
//! Also provides CPU core-pinning utilities. Uses `core_affinity` for
//! cross-platform, topology-agnostic core pinning that works correctly
//! on multi-socket servers, hybrid architectures (BIG.little), and
//! cloud VMs where CPU IDs may not follow simple even/odd patterns.

/// Compute backend for a workload.
///
/// Ordered by preference: CUDA > Vulkan > CpuSimd > CpuScalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ComputeBackend {
    CpuScalar,
    CpuSimd,
    Vulkan,
    Cuda,
}

/// Distribution-storage or working-field precision tier.
///
/// This is intentionally broader than any one backend. CUDA, Vulkan, and CPU
/// callers can use the same vocabulary when describing storage policy even if a
/// given backend only supports a subset of these modes in production.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StoragePrecision {
    Int4,
    Int8,
    Int16,
    Fp8E4m3,
    Fp8E5m2,
    Fp16,
    Bf16,
    Fp32,
    Fp64,
    DdFp128,
}

/// Memory layout policy for simulation buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum BufferLayout {
    /// Array-of-structures. Commonly favored on CPU fallback paths.
    Aos,
    /// Structure-of-arrays. Commonly favored on GPU hot paths for coalescing.
    Soa,
}

/// Residency policy for large simulation buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum MemoryResidency {
    /// Device-local memory only.
    DeviceLocal,
    /// Unified/managed memory for the full working set.
    ManagedUnified,
    /// Unified/managed memory with tile-prefetch sequencing.
    ManagedUnifiedTiled,
    /// Host-resident or host-visible path.
    HostVisible,
}

/// Frontend-visible frame mode for interactive viewers or inspection tools.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum FrameMode {
    /// Full 3D volume rendering.
    Volume3d,
    /// 2D slice view through a larger volume.
    Slice2d,
    /// Particle or tracer visualization.
    ParticleTrace,
    /// Scalar or tabular dashboard view.
    ScalarField,
}

/// Shared execution-profile descriptor for backend-facing applications.
///
/// This is not meant to replace domain-specific runtime configuration. It is a
/// compact common vocabulary that lets viewers, profilers, and planners talk
/// about backend choice, storage policy, and frontend presentation in one
/// stable type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionProfile {
    pub backend: ComputeBackend,
    pub storage: StoragePrecision,
    pub layout: BufferLayout,
    pub residency: MemoryResidency,
    pub frame_mode: FrameMode,
}

/// Legacy two-tier backend selection (kept for backward compatibility).
pub fn choose_backend(problem_size: usize, gpu_available: bool) -> ComputeBackend {
    if gpu_available && problem_size >= 10_000 {
        ComputeBackend::Cuda
    } else {
        ComputeBackend::CpuScalar
    }
}

/// Hardware capability flags for backend detection.
///
/// Callers probe their own environment (cudarc, ash, cpuid) and pass
/// the results here. This keeps gororoba_gpu_bridge free of heavy
/// GPU dependencies.
#[derive(Debug, Clone, Copy, Default)]
pub struct HardwareCaps {
    /// CUDA device available with compute capability >= 8.0 (Ampere+).
    pub cuda_available: bool,
    /// CUDA device available with compute capability >= 8.9 (Ada Lovelace).
    pub cuda_ada_available: bool,
    /// CUDA compute capability major version of the selected device.
    pub cuda_compute_major: u32,
    /// CUDA compute capability minor version of the selected device.
    pub cuda_compute_minor: u32,
    /// CUDA L2 cache size in bytes.
    pub cuda_l2_bytes: usize,
    /// CUDA shared memory available per block in bytes.
    pub cuda_shared_mem_per_block: usize,
    /// CUDA device advertises native BF16 support.
    pub cuda_bf16_native: bool,
    /// CUDA device is a good fit for sparse shared-tile kernels.
    pub cuda_sparse_tile_preferred: bool,
    /// Vulkan device available.
    pub vulkan_available: bool,
    /// CPU SIMD capabilities (granular feature detection).
    pub simd: SimdCaps,
}

/// Select the best available backend given hardware capabilities and
/// problem size.
///
/// For GPU backends, requires `problem_size >= 10_000` to justify
/// the kernel launch and data transfer overhead.
pub fn detect_best_backend(caps: &HardwareCaps, problem_size: usize) -> ComputeBackend {
    let gpu_worthy = problem_size >= 10_000;

    if gpu_worthy && caps.cuda_available {
        ComputeBackend::Cuda
    } else if gpu_worthy && caps.vulkan_available {
        ComputeBackend::Vulkan
    } else if caps.simd.any() {
        ComputeBackend::CpuSimd
    } else {
        ComputeBackend::CpuScalar
    }
}

/// Granular SIMD feature detection for the current CPU.
///
/// Reports the full feature ladder so callers can select the widest
/// available instruction set. The `wide` crate auto-dispatches based
/// on compile-time features, but this struct lets us log capabilities
/// and make runtime decisions (e.g., choosing f32x8 vs f32x4 paths).
#[derive(Debug, Clone, Copy, Default)]
pub struct SimdCaps {
    // x86_64 features (ascending width/capability)
    pub sse2: bool,
    pub sse4_1: bool,
    pub avx: bool,
    pub avx2: bool,
    pub fma: bool,
    pub avx512f: bool,

    // ARM features
    pub neon: bool,
    pub sve: bool,
}

impl SimdCaps {
    /// Returns true if any SIMD extension is available.
    pub fn any(&self) -> bool {
        self.sse2 || self.avx || self.avx2 || self.avx512f || self.neon || self.sve
    }

    /// Returns the widest available register width in bits.
    pub fn widest_bits(&self) -> u32 {
        if self.avx512f {
            512
        } else if self.avx2 || self.avx {
            256
        } else if self.sse2 || self.neon || self.sve {
            // SSE2 = 128-bit, NEON = 128-bit, SVE >= 128-bit (variable)
            128
        } else {
            0
        }
    }
}

impl std::fmt::Display for SimdCaps {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut features = Vec::new();
        if self.sse2 {
            features.push("SSE2");
        }
        if self.sse4_1 {
            features.push("SSE4.1");
        }
        if self.avx {
            features.push("AVX");
        }
        if self.avx2 {
            features.push("AVX2");
        }
        if self.fma {
            features.push("FMA");
        }
        if self.avx512f {
            features.push("AVX-512F");
        }
        if self.neon {
            features.push("NEON");
        }
        if self.sve {
            features.push("SVE");
        }
        if features.is_empty() {
            write!(f, "scalar (no SIMD)")
        } else {
            write!(f, "{}", features.join("+"))
        }
    }
}

/// Probe SIMD availability on the current CPU.
///
/// Returns a `SimdCaps` struct with all detected features.
pub fn probe_simd() -> SimdCaps {
    let mut caps = SimdCaps::default();

    #[cfg(target_arch = "x86_64")]
    {
        caps.sse2 = std::arch::is_x86_feature_detected!("sse2");
        caps.sse4_1 = std::arch::is_x86_feature_detected!("sse4.1");
        caps.avx = std::arch::is_x86_feature_detected!("avx");
        caps.avx2 = std::arch::is_x86_feature_detected!("avx2");
        caps.fma = std::arch::is_x86_feature_detected!("fma");
        caps.avx512f = std::arch::is_x86_feature_detected!("avx512f");
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is mandatory on all AArch64 implementations
        caps.neon = true;
        // SVE detection: std::arch::is_aarch64_feature_detected is unstable,
        // so we conservatively report false. When stabilized, add:
        // caps.sve = std::arch::is_aarch64_feature_detected!("sve");
    }

    caps
}

/// Pin the current thread to one of the available CPU cores.
///
/// Uses `core_affinity` for cross-platform, hardware-agnostic core
/// selection. Unlike raw `sched_setaffinity` with even-numbered ID
/// heuristics, this works correctly on:
/// - Multi-socket servers (non-contiguous core numbering)
/// - Hybrid architectures (Intel BIG.little, Apple Silicon)
/// - Cloud VMs (hypervisor-scrambled virtual CPU IDs)
///
/// `max_cores` limits how many cores are eligible. If 0, all
/// available cores are eligible.
///
/// Returns the list of `CoreId`s selected (for use with rayon or
/// manual thread pinning), or an empty vec if discovery failed.
pub fn select_cores(max_cores: usize) -> Vec<core_affinity::CoreId> {
    let core_ids = match core_affinity::get_core_ids() {
        Some(ids) if !ids.is_empty() => ids,
        _ => return Vec::new(),
    };

    let limit = if max_cores == 0 {
        core_ids.len()
    } else {
        max_cores.min(core_ids.len())
    };

    core_ids[..limit].to_vec()
}

/// Pin the current thread to the first available core.
///
/// Convenience wrapper for single-threaded workloads or main-thread
/// pinning before spawning a thread pool.
///
/// `max_cores` limits the selection pool (0 = all available).
///
/// Returns the number of cores pinned (0 or 1).
pub fn pin_to_physical_cores(max_cores: usize) -> usize {
    let cores = select_cores(max_cores);
    match cores.first() {
        Some(&core_id) if core_affinity::set_for_current(core_id) => cores.len(),
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_choose_backend_threshold() {
        assert_eq!(choose_backend(500, true), ComputeBackend::CpuScalar);
        assert_eq!(choose_backend(20_000, true), ComputeBackend::Cuda);
        assert_eq!(choose_backend(20_000, false), ComputeBackend::CpuScalar);
    }

    #[test]
    fn test_detect_best_backend_preference_order() {
        let simd = SimdCaps {
            avx2: true,
            ..Default::default()
        };
        let all_caps = HardwareCaps {
            cuda_available: true,
            cuda_ada_available: false,
            cuda_compute_major: 8,
            cuda_compute_minor: 0,
            cuda_l2_bytes: 0,
            cuda_shared_mem_per_block: 0,
            cuda_bf16_native: true,
            cuda_sparse_tile_preferred: false,
            vulkan_available: true,
            simd,
        };
        // CUDA preferred over Vulkan for large problems
        assert_eq!(detect_best_backend(&all_caps, 20_000), ComputeBackend::Cuda);

        let vulkan_only = HardwareCaps {
            cuda_available: false,
            cuda_ada_available: false,
            cuda_compute_major: 0,
            cuda_compute_minor: 0,
            cuda_l2_bytes: 0,
            cuda_shared_mem_per_block: 0,
            cuda_bf16_native: false,
            cuda_sparse_tile_preferred: false,
            vulkan_available: true,
            simd,
        };
        assert_eq!(
            detect_best_backend(&vulkan_only, 20_000),
            ComputeBackend::Vulkan
        );

        // Small problem -> CPU path even with GPU available
        assert_eq!(detect_best_backend(&all_caps, 500), ComputeBackend::CpuSimd);

        let no_gpu = HardwareCaps {
            cuda_available: false,
            cuda_ada_available: false,
            cuda_compute_major: 0,
            cuda_compute_minor: 0,
            cuda_l2_bytes: 0,
            cuda_shared_mem_per_block: 0,
            cuda_bf16_native: false,
            cuda_sparse_tile_preferred: false,
            vulkan_available: false,
            simd,
        };
        assert_eq!(
            detect_best_backend(&no_gpu, 20_000),
            ComputeBackend::CpuSimd
        );

        let nothing = HardwareCaps::default();
        assert_eq!(
            detect_best_backend(&nothing, 20_000),
            ComputeBackend::CpuScalar
        );
    }

    #[test]
    fn test_probe_simd() {
        let caps = probe_simd();
        #[cfg(target_arch = "x86_64")]
        {
            assert!(caps.sse2, "SSE2 should be available on x86_64");
            assert!(caps.avx2, "AVX2 should be available on modern x86_64");
            assert!(caps.any());
            assert!(caps.widest_bits() >= 256);
        }
        #[cfg(target_arch = "aarch64")]
        {
            assert!(caps.neon, "NEON should be available on aarch64");
            assert!(caps.any());
        }
    }

    #[test]
    fn test_simd_caps_display() {
        let caps = SimdCaps {
            sse2: true,
            avx2: true,
            fma: true,
            ..Default::default()
        };
        let s = format!("{caps}");
        assert!(s.contains("SSE2"), "display should include SSE2");
        assert!(s.contains("AVX2"), "display should include AVX2");
        assert!(s.contains("FMA"), "display should include FMA");
    }

    #[test]
    fn test_simd_caps_widest_bits() {
        assert_eq!(SimdCaps::default().widest_bits(), 0);
        assert_eq!(
            SimdCaps {
                sse2: true,
                ..Default::default()
            }
            .widest_bits(),
            128
        );
        assert_eq!(
            SimdCaps {
                avx2: true,
                ..Default::default()
            }
            .widest_bits(),
            256
        );
        assert_eq!(
            SimdCaps {
                avx512f: true,
                ..Default::default()
            }
            .widest_bits(),
            512
        );
        assert_eq!(
            SimdCaps {
                neon: true,
                ..Default::default()
            }
            .widest_bits(),
            128
        );
    }

    #[test]
    fn test_backend_ordering() {
        // Ord should reflect preference: CpuScalar < CpuSimd < Vulkan < Cuda
        assert!(ComputeBackend::CpuScalar < ComputeBackend::CpuSimd);
        assert!(ComputeBackend::CpuSimd < ComputeBackend::Vulkan);
        assert!(ComputeBackend::Vulkan < ComputeBackend::Cuda);
    }

    #[test]
    fn test_core_ids_available() {
        let ids = core_affinity::get_core_ids();
        assert!(ids.is_some(), "get_core_ids() returned None");
        let ids = ids.unwrap();
        assert!(!ids.is_empty(), "no core IDs available");
    }

    #[test]
    fn test_select_cores_returns_nonempty() {
        let cores = select_cores(0);
        assert!(!cores.is_empty(), "expected at least 1 core");
    }

    #[test]
    fn test_select_cores_with_limit() {
        let cores = select_cores(2);
        assert!(
            cores.len() <= 2,
            "should not exceed limit of 2, got {}",
            cores.len()
        );
        assert!(!cores.is_empty(), "should have at least 1 core");
    }

    #[test]
    fn test_pin_to_physical_cores_returns_nonzero() {
        let pinned = pin_to_physical_cores(0);
        assert!(pinned > 0, "expected at least 1 core, got 0");
    }

    #[test]
    fn test_pin_with_limit() {
        let pinned = pin_to_physical_cores(2);
        assert!(pinned <= 2, "should not exceed limit of 2, got {}", pinned);
        assert!(pinned > 0, "should pin at least 1 core");
    }
}
