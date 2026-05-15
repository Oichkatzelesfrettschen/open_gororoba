//! Canonical CUDA device probe.
//!
//! Consolidates `cd_kernel::turboquant::cuda::device::CudaDeviceProps`
//! and `lbm_3d_cuda::probe_cuda_device_props` into a single source of
//! truth. Bridges to `gororoba_gpu_bridge::HardwareCaps` so callers can
//! convert between the canonical type vocabulary and the cudarc-specific
//! probe.

use cudarc::runtime::result::device as cudart_device;
use gororoba_gpu_bridge::HardwareCaps;

use crate::error::{CudaError, Result};

/// CUDA device properties relevant to kernel selection.
#[derive(Clone, Debug, Default)]
pub struct DeviceProbe {
    /// Compute capability major version.
    pub major: u32,
    /// Compute capability minor version.
    pub minor: u32,
    /// L2 cache size in bytes.
    pub l2_bytes: usize,
    /// Shared memory per block in bytes (uses `sharedMemPerBlockOptin`
    /// when larger).
    pub shared_mem_per_block: usize,
    /// Total global memory in bytes.
    pub total_global_mem: usize,
    /// Native BF16 support (SM 8.0+, Ampere onward).
    pub bf16_native: bool,
    /// Native FP8 support (SM 8.9+, Ada Lovelace onward).
    pub fp8_native: bool,
    /// Tensor Memory Accelerator available (SM 9.0+, Hopper onward).
    pub tma_available: bool,
    /// Device name (verbatim from `cudaDeviceProp.name`).
    pub name: String,
}

impl DeviceProbe {
    /// Probe CUDA device `ordinal` (default 0) via cudarc runtime.
    ///
    /// Returns `Err(CudaError::NoDevice)` when no CUDA device is present.
    pub fn query() -> Result<Self> {
        Self::query_ordinal(0)
    }

    /// Probe a specific device ordinal.
    pub fn query_ordinal(ordinal: usize) -> Result<Self> {
        let count = cudart_device::get_count().unwrap_or(0).max(0) as usize;
        if count == 0 {
            return Err(CudaError::NoDevice);
        }
        if ordinal >= count {
            return Err(CudaError::OrdinalOutOfRange { ordinal, count });
        }
        let i32_ord = ordinal.try_into().unwrap_or(0);
        let prop = cudart_device::get_device_prop(i32_ord)
            .map_err(|e| CudaError::Nvml(format!("get_device_prop({}): {:?}", ordinal, e)))?;
        let major = prop.major.max(0) as u32;
        let minor = prop.minor.max(0) as u32;
        let l2_bytes = prop.l2CacheSize.max(0) as usize;
        let shared_mem_per_block = prop.sharedMemPerBlockOptin.max(prop.sharedMemPerBlock);
        let total_global_mem = prop.totalGlobalMem;

        let bf16_native = major >= 8;
        let fp8_native = major > 8 || (major == 8 && minor >= 9);
        let tma_available = major >= 9;

        let name_bytes = prop.name.as_slice();
        let nul = name_bytes
            .iter()
            .position(|b| *b == 0)
            .unwrap_or(name_bytes.len());
        let name_u8: &[u8] =
            unsafe { std::slice::from_raw_parts(name_bytes.as_ptr().cast::<u8>(), nul) };
        let name = String::from_utf8_lossy(name_u8).into_owned();

        Ok(Self {
            major,
            minor,
            l2_bytes,
            shared_mem_per_block,
            total_global_mem,
            bf16_native,
            fp8_native,
            tma_available,
            name,
        })
    }

    /// NVRTC `--gpu-architecture` target string for this device.
    pub fn compile_arch(&self) -> &'static str {
        match (self.major, self.minor) {
            (9, _) => "sm_90",
            (8, 9) => "sm_89",
            (8, m) if (6..=8).contains(&m) => "sm_86",
            (8, _) => "sm_80",
            (7, m) if m >= 5 => "sm_75",
            _ => "sm_52",
        }
    }

    /// Ada Lovelace (SM 8.9, RTX 40-series).
    pub fn is_ada(&self) -> bool {
        self.major == 8 && self.minor == 9
    }

    /// Hopper (SM 9.0, H100).
    pub fn is_hopper(&self) -> bool {
        self.major >= 9
    }

    /// 90% of total global memory (workspace heuristic from
    /// lbm_3d_cuda).
    pub fn usable_vram_bytes(&self) -> usize {
        (self.total_global_mem as f64 * 0.90) as usize
    }

    /// Bridge to the canonical `gororoba_gpu_bridge::HardwareCaps` shape.
    pub fn to_hardware_caps(&self) -> HardwareCaps {
        HardwareCaps {
            cuda_available: true,
            cuda_ada_available: self.is_ada(),
            cuda_compute_major: self.major,
            cuda_compute_minor: self.minor,
            cuda_l2_bytes: self.l2_bytes,
            cuda_shared_mem_per_block: self.shared_mem_per_block,
            cuda_bf16_native: self.bf16_native,
            cuda_sparse_tile_preferred: self.shared_mem_per_block >= 128 * 1024,
            vulkan_available: false,
            simd: Default::default(),
        }
    }
}
