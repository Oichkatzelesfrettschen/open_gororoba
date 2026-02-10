//! GPU acceleration for Cayley-Dickson algebra computations.
//!
//! This module provides GPU-accelerated implementations of computationally intensive
//! operations, particularly for high-dimensional zero-divisor analysis.
//!
//! Features (when compiled with `gpu` feature):
//! - Eta matrix computation (parallel XOR operations)
//! - Component graph construction (parallel edge detection)
//! - Frustration ratio computation (parallel BFS)
//! - Triangle enumeration (parallel triple checking)
//!
//! All GPU operations have CPU fallback implementations.

#[cfg(feature = "gpu")]
mod eta_matrix;
#[cfg(feature = "gpu")]
mod graph_construction;
#[cfg(feature = "gpu")]
mod frustration;

#[cfg(feature = "gpu")]
pub use eta_matrix::EtaMatrixGpu;
#[cfg(feature = "gpu")]
pub use graph_construction::GraphConstructorGpu;
#[cfg(feature = "gpu")]
pub use frustration::FrustrationGpu;

/// GPU device initialization and error handling.
#[cfg(feature = "gpu")]
pub mod device {
    use cudarc::driver::CudaDevice;
    use std::sync::Arc;

    /// Initialize GPU device for computation.
    ///
    /// # Returns
    /// A handle to GPU device 0, or an error if CUDA is unavailable.
    pub fn init_gpu() -> Result<Arc<CudaDevice>, String> {
        CudaDevice::new(0).map_err(|e| format!("CUDA device initialization failed: {}", e))
    }

    /// Check if GPU is available without initializing.
    pub fn is_gpu_available() -> bool {
        CudaDevice::new(0).is_ok()
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
