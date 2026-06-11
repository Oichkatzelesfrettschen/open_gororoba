//! GPU acceleration for Cayley-Dickson algebra computations.
//!
//! This module provides GPU-accelerated implementations of computationally intensive
//! operations, particularly for high-dimensional zero-divisor analysis.
//!
//! Features (when compiled with `gpu` feature):
//! - Eta matrix computation (parallel XOR operations)
//! - Component graph construction (parallel edge detection)
//! - Imbalance ratio computation (parallel BFS)
//! - Triangle enumeration (parallel triple checking)
//!
//! All GPU operations have CPU fallback implementations.

// dimensional module is always available (has CPU fallback + wide-index API)
pub mod avt_pack;
pub mod dimensional;
#[cfg(feature = "cubecl")]
mod dimensional_cubecl;
#[cfg(feature = "vulkan")]
mod dimensional_vulkan;
#[cfg(feature = "gpu")]
mod eta_matrix;
#[cfg(feature = "cubecl")]
mod eta_matrix_cubecl;
#[cfg(feature = "vulkan")]
mod eta_matrix_vulkan;
#[cfg(any(feature = "gpu", feature = "vulkan", feature = "cubecl"))]
mod graph_construction;
#[cfg(feature = "cubecl")]
mod graph_construction_cubecl;
#[cfg(feature = "vulkan")]
mod graph_construction_vulkan;
#[cfg(any(feature = "gpu", feature = "vulkan", feature = "cubecl"))]
mod imbalance;
#[cfg(feature = "cubecl")]
mod imbalance_cubecl;
#[cfg(feature = "vulkan")]
mod imbalance_vulkan;
pub mod tensor_avt;
pub mod voudon;
#[cfg(feature = "cubecl")]
mod voudon_cubecl;
#[cfg(feature = "vulkan")]
mod voudon_vulkan;

pub use avt_pack::GpuPackableAvt;
pub use dimensional::{GpuAptResult, GpuAptResultWide, GpuDimensionalEngine};
#[cfg(feature = "cubecl")]
pub use dimensional_cubecl::{DimensionalCubeclKernel, dimensional_cubecl_available};
#[cfg(feature = "vulkan")]
pub use dimensional_vulkan::{
    DIMENSIONAL_VULKAN_ENTRY_POINT, DIMENSIONAL_VULKAN_WGSL, DimensionalVulkanKernel,
    DimensionalVulkanPipeline,
};
#[cfg(feature = "gpu")]
pub use eta_matrix::EtaMatrixGpu;
#[cfg(feature = "cubecl")]
pub use eta_matrix_cubecl::{EtaMatrixCubeclKernel, eta_matrix_cubecl_available};
#[cfg(feature = "vulkan")]
pub use eta_matrix_vulkan::{
    ETA_MATRIX_VULKAN_ENTRY_POINT, ETA_MATRIX_VULKAN_WGSL, EtaMatrixVulkanKernel,
    EtaMatrixVulkanPipeline,
};
pub use gororoba_gpu_bridge::ComputeBackend;
#[cfg(any(feature = "gpu", feature = "vulkan", feature = "cubecl"))]
pub use graph_construction::GraphConstructorGpu;
#[cfg(feature = "cubecl")]
pub use graph_construction_cubecl::{
    GraphConstructionCubeclKernel, graph_construction_cubecl_available,
};
#[cfg(feature = "vulkan")]
pub use graph_construction_vulkan::{
    GRAPH_CONSTRUCTION_VULKAN_ENTRY_POINT, GRAPH_CONSTRUCTION_VULKAN_WGSL,
    GraphConstructionVulkanKernel, GraphConstructionVulkanPipeline,
};
#[cfg(any(feature = "gpu", feature = "vulkan", feature = "cubecl"))]
pub use imbalance::{ImbalanceGpu, ImbalanceResult};
#[cfg(feature = "cubecl")]
pub use imbalance_cubecl::{ImbalanceCubeclKernel, imbalance_cubecl_available};
#[cfg(feature = "vulkan")]
pub use imbalance_vulkan::{
    IMBALANCE_VULKAN_ENTRY_POINT, IMBALANCE_VULKAN_WGSL, ImbalanceVulkanKernel,
    ImbalanceVulkanPipeline,
};
pub use tensor_avt::{
    TensorAVT, TensorAvtAutoConfig, TensorAvtAutoResult, TensorAvtCalibrationMode,
    TensorAvtMulSession, TensorAvtNormSession, TensorAvtThresholdOverrides,
};
#[cfg(feature = "cubecl")]
pub use tensor_avt::{TensorAvtCubeclKernel, tensor_avt_cubecl_available};
#[cfg(feature = "vulkan")]
pub use tensor_avt::{TensorAvtVulkanKernel, tensor_avt_vulkan_available};
#[cfg(feature = "cubecl")]
pub use voudon_cubecl::{VoudonCubeclKernel, voudon_cubecl_available};
#[cfg(feature = "vulkan")]
pub use voudon_vulkan::{
    VOUDON_VULKAN_ENTRY_POINT, VOUDON_VULKAN_WGSL, VoudonVulkanKernel, VoudonVulkanPipeline,
};

/// GPU device initialization and error handling.
#[cfg(feature = "gpu")]
pub mod device {
    use cudarc::driver::CudaContext;
    use std::sync::Arc;

    /// Initialize GPU device for computation.
    ///
    /// Routes through the consolidated
    /// `gororoba_gpu_cuda::Context::with_default_device` helper so the
    /// `cudart_device::get_count` + ordinal-range check lives in one
    /// place across 13 workspace crates. The returned
    /// `Arc<CudaContext>` is byte-compatible with the prior direct
    /// `CudaContext::new(0)` return.
    ///
    /// # Returns
    /// A handle to GPU context for device 0, or an error if CUDA is unavailable.
    pub fn init_gpu() -> Result<Arc<CudaContext>, String> {
        let ctx_wrapper = gororoba_gpu_cuda::Context::with_default_device()
            .map_err(|e| format!("CUDA device initialization failed: {}", e))?;
        Ok(ctx_wrapper.raw().clone())
    }

    /// Check if GPU is available without initializing.
    ///
    /// Delegates to `gororoba_gpu_cuda::Context::is_available` which
    /// probes `cudart_device::get_count` without paying for context
    /// creation.
    pub fn is_gpu_available() -> bool {
        gororoba_gpu_cuda::Context::is_available()
    }
}

#[cfg(not(feature = "gpu"))]
pub mod device {
    /// GPU stubs when compiled without gpu feature.
    pub fn is_gpu_available() -> bool {
        false
    }
}

// Re-export common types
pub use device::is_gpu_available;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_availability_check() {
        // This test just ensures the availability check compiles and runs
        let available = is_gpu_available();
        if available {
            eprintln!("GPU is available for acceleration");
        } else {
            eprintln!("GPU not available; CPU-only mode");
        }
    }
}
