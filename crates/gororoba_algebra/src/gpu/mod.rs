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
#[cfg(feature = "gpu")]
mod eta_matrix;
#[cfg(feature = "gpu")]
mod graph_construction;
#[cfg(feature = "gpu")]
mod imbalance;
pub mod tensor_avt;
pub mod voudon;

pub use avt_pack::GpuPackableAvt;
pub use dimensional::{GpuAptResult, GpuAptResultWide, GpuDimensionalEngine};
#[cfg(feature = "gpu")]
pub use eta_matrix::EtaMatrixGpu;
pub use gororoba_gpu_bridge::ComputeBackend;
#[cfg(feature = "gpu")]
pub use graph_construction::GraphConstructorGpu;
#[cfg(feature = "gpu")]
pub use imbalance::ImbalanceGpu;
pub use tensor_avt::{
    TensorAVT, TensorAvtAutoConfig, TensorAvtAutoResult, TensorAvtCalibrationMode,
    TensorAvtMulSession, TensorAvtNormSession, TensorAvtThresholdOverrides,
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
