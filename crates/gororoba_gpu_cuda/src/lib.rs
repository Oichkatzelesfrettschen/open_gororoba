//! Shared CUDA helpers for the open_gororoba workspace.
//!
//! WHY: 13 crates currently depend on cudarc 0.19 directly. 48+ sites
//! independently spin up a `CudaContext::new(0)`, 35+ sites repeat the
//! same NVRTC compile-options pattern, 28+ sites load PTX + functions,
//! 60+ allocate buffers, 40+ memcpy. Three competing error patterns
//! coexist (anyhow::Result, Result<T, String>, Option<T>) and two
//! competing device-property structs (`cd_kernel::turboquant::cuda::CudaDeviceProps`
//! and `gororoba_gpu_bridge::HardwareCaps`).
//!
//! WHAT: Eleven helper modules behind feature `cudarc` (default off):
//!   - error.rs:    `CudaError` unifying DriverError + nvrtc::CompileError
//!   - context.rs:  `Context::with_default_device` / `::with_device(ordinal)`
//!   - probe.rs:    `DeviceProbe::query` -- canonical device-cap struct
//!   - nvrtc.rs:    `CompileOptions::for_arch(major, minor)` builder
//!   - module.rs:   `ModuleRegistry` -- PTX cache + multi-function loader
//!   - buffer.rs:   `Buffer<T>::alloc_zeros` / `::htod` / `::dtoh`
//!   - managed.rs:  `ManagedBuffer<T>` -- unified memory + prefetch
//!   - launch.rs:   `LaunchConfig::launch_1d` / `::launch_2d` (arch-tuned)
//!   - nvml.rs:     `nvml::Telemetry::sample`
//!   - optix.rs:    `optix::PipelineBuilder` (device-context + module half)
//!
//! Re-exports `gororoba_gpu_bridge::{ComputeBackend, HardwareCaps,
//! StoragePrecision}` and `gororoba_optix` types so consumers have one
//! import surface.
//!
//! HOW: Default-features build pulls only the bridge re-exports.
//! Enabling `--features cudarc` activates cudarc + nvml-wrapper +
//! thiserror + log via `{ workspace = true, optional = true }`.
//! Mirrors the cd_kernel + lbm_3d_cuda + sign_imbalance feature-gating
//! pattern.

pub use gororoba_gpu_bridge::{ComputeBackend, HardwareCaps, StoragePrecision};
pub use gororoba_optix as optix_ffi;

#[cfg(feature = "cudarc")]
mod buffer;
#[cfg(feature = "cudarc")]
mod context;
#[cfg(feature = "cudarc")]
mod error;
#[cfg(feature = "cudarc")]
mod launch;
#[cfg(feature = "cudarc")]
mod managed;
#[cfg(feature = "cudarc")]
mod module;
#[cfg(feature = "cudarc")]
mod nvml;
#[cfg(feature = "cudarc")]
mod nvrtc;
#[cfg(feature = "cudarc")]
mod optix;
#[cfg(feature = "cudarc")]
mod probe;

#[cfg(feature = "cudarc")]
pub use buffer::Buffer;
#[cfg(feature = "cudarc")]
pub use context::Context;
#[cfg(feature = "cudarc")]
pub use error::{CudaError, Result};
#[cfg(feature = "cudarc")]
pub use launch::LaunchConfig;
#[cfg(feature = "cudarc")]
pub use managed::ManagedBuffer;
#[cfg(feature = "cudarc")]
pub use module::{KernelHandle, ModuleRegistry};
#[cfg(feature = "cudarc")]
pub use nvrtc::CompileOptions;
#[cfg(feature = "cudarc")]
pub use probe::DeviceProbe;

#[cfg(feature = "cudarc")]
pub mod telemetry {
    pub use crate::nvml::Telemetry;
}

#[cfg(feature = "cudarc")]
pub mod optix_helpers {
    pub use crate::optix::PipelineBuilder;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn re_exports_compile() {
        let _: ComputeBackend = ComputeBackend::CpuScalar;
        let _: StoragePrecision = StoragePrecision::Fp32;
    }
}
