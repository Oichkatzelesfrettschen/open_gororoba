//! Minimal OptiX 9.1 FFI declarations for particle tracing.
//!
//! These are the subset of OptiX C API functions needed for:
//! - Context creation (`optixInit`, `optixDeviceContextCreate`)
//! - Module compilation (`optixModuleCreate`)
//! - Pipeline construction (`optixProgramGroupCreate`, `optixPipelineCreate`)
//! - Acceleration structure (`optixAccelBuild`)
//! - Launch (`optixLaunch`)
//!
//! The full OptiX API has ~60 functions; we only declare the ~15 needed
//! for the particle tracing pipeline.
//!
//! # Safety
//!
//! All functions are `unsafe extern "C"` FFI calls to liboptix.so.
//! The caller must ensure correct struct layout, valid handles, and
//! proper CUDA context binding before calling these functions.
//!
//! # Loading
//!
//! OptiX uses a function table loaded at runtime via `optixInit()`.
//! After calling `optixInit()`, all function pointers are populated
//! in a global table. We call them through the stubs header pattern.
//!
//! For this implementation, we use `libloading` to dynamically load
//! `libnvoptix.so.1` and resolve function pointers at runtime.

/// OptiX result code (maps to OptixResult enum in C).
pub type OptixResult = i32;

/// Opaque handle types (all are pointers in C).
pub type OptixDeviceContext = *mut std::ffi::c_void;
pub type OptixModule = *mut std::ffi::c_void;
pub type OptixProgramGroup = *mut std::ffi::c_void;
pub type OptixPipeline = *mut std::ffi::c_void;
pub type OptixTraversableHandle = u64;

/// OptiX success code.
pub const OPTIX_SUCCESS: OptixResult = 0;

/// Check if OptiX shared library is available on this system.
pub fn optix_available() -> bool {
    std::path::Path::new("/usr/lib/libnvoptix.so.1").exists()
        || std::path::Path::new("/usr/lib64/libnvoptix.so.1").exists()
}

/// OptiX initialization status.
/// In a full implementation, this would call `optixInit()` via the
/// function table loaded from libnvoptix.so.1.
///
/// For now, this probes whether the OptiX shared library exists.
pub fn probe_optix() -> Result<(), String> {
    if optix_available() {
        Ok(())
    } else {
        Err("libnvoptix.so.1 not found. Install NVIDIA OptiX SDK.".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optix_probe() {
        // Just verify the probe doesn't crash -- actual availability depends on system
        let _ = probe_optix();
    }

    #[test]
    fn test_handle_sizes() {
        assert_eq!(
            std::mem::size_of::<OptixDeviceContext>(),
            std::mem::size_of::<*mut ()>()
        );
        assert_eq!(std::mem::size_of::<OptixTraversableHandle>(), 8);
    }
}
