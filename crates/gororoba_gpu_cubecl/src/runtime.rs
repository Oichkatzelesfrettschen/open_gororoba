//! cubecl-wgpu Runtime acquisition + probe helpers.
//!
//! Consolidates the WgpuDevice::default() + WgpuRuntime::client() +
//! panic-safe `is_available()` probe pattern duplicated across:
//!   - cd_kernel/src/turboquant/cubecl_backend/launcher.rs:140-203
//!   - lbm_vulkan/src/box_counting_cubecl.rs:104-165
//!   - lbm_vulkan/src/chingon_cubecl.rs:80-137
//!   - lbm_vulkan/src/transform_viscosity_cubecl.rs:38-94
//!
//! All four sites used identical code. This module exposes it once.

use cubecl::Runtime as _;
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

/// Top-level entry point for cubecl-wgpu runtime access.
///
/// All construction is free-function (no Runtime struct instance) because
/// `WgpuDevice::default()` is the canonical handle and a singleton in
/// cubecl-wgpu 0.10's model. The struct exists as a namespace for the
/// helpers.
pub struct Runtime;

impl Runtime {
    /// Returns true iff a default wgpu device + cubecl client can be
    /// constructed without panicking.
    ///
    /// Consolidates 4 identical 11-line implementations across the
    /// workspace. The `std::panic::catch_unwind` wraps cubecl's
    /// `WgpuRuntime::client` which panics on adapter-acquisition failure
    /// (e.g. no Vulkan/Metal/DX12 driver available, no compatible
    /// adapter, headless WebGPU without browser).
    pub fn probe() -> bool {
        std::panic::catch_unwind(|| {
            let device = WgpuDevice::default();
            let _client = WgpuRuntime::client(&device);
        })
        .is_ok()
    }
}
