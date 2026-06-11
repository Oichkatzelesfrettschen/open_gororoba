//! Error type for CUDA helper failures.
//!
//! WHY: 13 cudarc-using crates have three competing error patterns
//! (`anyhow::Result`, `Result<T, String>`, `Option<T>`). This module
//! consolidates them under a single error type with `From` impls so
//! callers see one error surface.

use thiserror::Error;

/// All errors produced by the CUDA helpers.
#[derive(Debug, Error)]
pub enum CudaError {
    /// Wraps cudarc driver errors (CUDA_ERROR_* codes).
    #[error("CUDA driver error: {0}")]
    Driver(#[from] cudarc::driver::DriverError),

    /// Wraps cudarc NVRTC compilation errors.
    #[error("NVRTC compilation error: {0}")]
    Nvrtc(#[from] cudarc::nvrtc::CompileError),

    /// No CUDA-capable device present.
    #[error("no CUDA device available (count = 0)")]
    NoDevice,

    /// Device ordinal out of range.
    #[error("CUDA device ordinal {ordinal} out of range (have {count} devices)")]
    OrdinalOutOfRange { ordinal: usize, count: usize },

    /// Compute capability is below the requested minimum.
    #[error(
        "CUDA device compute capability {found_major}.{found_minor} is below required {needed_major}.{needed_minor}"
    )]
    InsufficientComputeCapability {
        found_major: u32,
        found_minor: u32,
        needed_major: u32,
        needed_minor: u32,
    },

    /// Kernel-function lookup miss inside a loaded module.
    #[error("kernel function {name} not found in module")]
    KernelNotFound { name: String },

    /// NVML telemetry failure.
    #[error("NVML telemetry error: {0}")]
    Nvml(String),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, CudaError>;
