//! CUDA device probing and architecture selection.
//!
//! Ported from `lbm_3d_cuda/src/lib.rs:probe_cuda_device_props` and
//! `lbm_3d_cuda/src/kernel_selector.rs`.

/// CUDA device properties relevant to TurboQuant kernel selection.
#[derive(Clone, Debug)]
pub struct CudaDeviceProps {
    /// Compute capability major version.
    pub major: u32,
    /// Compute capability minor version.
    pub minor: u32,
    /// L2 cache size in bytes.
    pub l2_bytes: usize,
    /// Shared memory per block in bytes.
    pub shared_mem_per_block: usize,
    /// Total global memory in bytes.
    pub total_global_mem: usize,
    /// Whether BF16 is natively supported (SM 8.0+).
    pub bf16_native: bool,
    /// Whether FP8 is natively supported (SM 8.9+).
    pub fp8_native: bool,
    /// Whether TMA (Tensor Memory Accelerator) is available (SM 9.0+).
    pub tma_available: bool,
    /// Device name.
    pub name: String,
}

impl CudaDeviceProps {
    /// Return the NVRTC compile target architecture string.
    ///
    /// From `lbm_3d_cuda/lib.rs:compile_arch`.
    pub fn compile_arch(&self) -> &'static str {
        match (self.major, self.minor) {
            (9, _) => "sm_90",     // Hopper
            (8, 9) => "sm_89",     // Ada Lovelace
            (8, 6..=8) => "sm_86", // Ampere (GA10x)
            (8, _) => "sm_80",     // Ampere (GA100)
            (7, 5..) => "sm_75",   // Turing
            _ => "sm_52",          // Maxwell fallback
        }
    }

    /// Whether this is Ada Lovelace (RTX 40xx).
    pub fn is_ada(&self) -> bool {
        self.major == 8 && self.minor == 9
    }

    /// Whether this is Hopper (H100, etc).
    pub fn is_hopper(&self) -> bool {
        self.major >= 9
    }

    /// Recommended TurboQuant kernel tier based on hardware.
    ///
    /// From steinmarder's kernel_selector.rs pattern: the optimal kernel
    /// depends on hardware capabilities, not just the algorithm.
    pub fn recommended_tier(&self) -> KernelTier {
        if self.is_hopper() {
            KernelTier::HopperFused // TMA + warp-specialization
        } else if self.is_ada() {
            KernelTier::AdaOptimized // FP8 indices, 128KB L1
        } else if self.bf16_native {
            KernelTier::AmpereBf16 // BF16 codebook storage
        } else {
            KernelTier::Generic // FP32 scalar fallback
        }
    }

    /// Estimate available VRAM after a safety margin.
    ///
    /// Pattern from `lbm_3d_cuda/lib.rs` VRAM pre-check.
    pub fn usable_vram_bytes(&self) -> usize {
        (self.total_global_mem as f64 * 0.90) as usize
    }
}

/// TurboQuant CUDA kernel optimization tier.
///
/// From steinmarder's precision-tier kernel selector (24 kernel variants
/// across 10 tiers).  TurboQuant has fewer tiers but the dispatch pattern
/// is the same: probe -> select -> compile -> launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelTier {
    /// FP32 scalar, no special hardware features.  Always works.
    Generic,
    /// Ampere BF16: store codebook as BF16, promote to FP32 for dot product.
    AmpereBf16,
    /// Ada Lovelace: FP8 index storage, 128KB L1 for KV cache residency.
    AdaOptimized,
    /// Hopper: TMA async loads, warp-specialized dequant+attention fusion.
    HopperFused,
}

impl std::fmt::Display for KernelTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelTier::Generic => write!(f, "generic-fp32"),
            KernelTier::AmpereBf16 => write!(f, "ampere-bf16"),
            KernelTier::AdaOptimized => write!(f, "ada-fp8"),
            KernelTier::HopperFused => write!(f, "hopper-fused"),
        }
    }
}

/// Probe the first CUDA device for properties.
///
/// Returns None if CUDA is not available (no GPU, no driver, etc).
/// This function does NOT require the `cuda` feature -- it probes
/// availability at runtime.
///
/// When the `cuda` feature is enabled, this uses cudarc to query
/// actual device properties.  Without it, returns None.
#[cfg(feature = "cuda")]
pub fn probe_device() -> Option<CudaDeviceProps> {
    use cudarc::runtime::result::device as cudart_device;

    // Get device count -- if zero or error, no CUDA available
    let count = cudart_device::get_count().ok()?;
    if count == 0 {
        return None;
    }

    // Get device properties for device 0
    let props = cudart_device::get_device_prop(0).ok()?;
    let major = props.major as u32;
    let minor = props.minor as u32;

    Some(CudaDeviceProps {
        major,
        minor,
        l2_bytes: props.l2CacheSize as usize,
        shared_mem_per_block: props.sharedMemPerBlock as usize,
        total_global_mem: props.totalGlobalMem,
        bf16_native: major >= 8,
        fp8_native: (major == 8 && minor >= 9) || major >= 9,
        tma_available: major >= 9,
        name: {
            let bytes: Vec<u8> = props.name.iter().map(|&c| c as u8).collect();
            String::from_utf8_lossy(&bytes)
                .trim_end_matches('\0')
                .to_string()
        },
    })
}

/// Stub: returns None when CUDA feature is not enabled.
#[cfg(not(feature = "cuda"))]
pub fn probe_device() -> Option<CudaDeviceProps> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_arch_mapping() {
        let hopper = CudaDeviceProps {
            major: 9,
            minor: 0,
            l2_bytes: 0,
            shared_mem_per_block: 0,
            total_global_mem: 0,
            bf16_native: true,
            fp8_native: true,
            tma_available: true,
            name: "H100".into(),
        };
        assert_eq!(hopper.compile_arch(), "sm_90");
        assert!(hopper.is_hopper());
        assert_eq!(hopper.recommended_tier(), KernelTier::HopperFused);

        let ada = CudaDeviceProps {
            major: 8,
            minor: 9,
            l2_bytes: 0,
            shared_mem_per_block: 0,
            total_global_mem: 0,
            bf16_native: true,
            fp8_native: true,
            tma_available: false,
            name: "RTX 4070 Ti".into(),
        };
        assert_eq!(ada.compile_arch(), "sm_89");
        assert!(ada.is_ada());
        assert_eq!(ada.recommended_tier(), KernelTier::AdaOptimized);

        let ampere = CudaDeviceProps {
            major: 8,
            minor: 0,
            l2_bytes: 0,
            shared_mem_per_block: 0,
            total_global_mem: 0,
            bf16_native: true,
            fp8_native: false,
            tma_available: false,
            name: "A100".into(),
        };
        assert_eq!(ampere.compile_arch(), "sm_80");
        assert_eq!(ampere.recommended_tier(), KernelTier::AmpereBf16);

        let turing = CudaDeviceProps {
            major: 7,
            minor: 5,
            l2_bytes: 0,
            shared_mem_per_block: 0,
            total_global_mem: 0,
            bf16_native: false,
            fp8_native: false,
            tma_available: false,
            name: "RTX 2080".into(),
        };
        assert_eq!(turing.compile_arch(), "sm_75");
        assert_eq!(turing.recommended_tier(), KernelTier::Generic);
    }

    #[test]
    fn test_probe_device_no_crash() {
        // Should return Some or None without panicking
        let _props = probe_device();
    }

    #[test]
    fn test_kernel_tier_display() {
        assert_eq!(format!("{}", KernelTier::Generic), "generic-fp32");
        assert_eq!(format!("{}", KernelTier::HopperFused), "hopper-fused");
    }
}
