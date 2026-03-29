//! cubecl unified GPU backend for TurboQuant.
//!
//! Replaces the separate CUDA (.cu) and Vulkan (.comp) kernel definitions
//! with a single set of cubecl kernel definitions that dispatch to:
//! - CUDA (via cubecl CUDA backend)
//! - Vulkan/SPIR-V (via cubecl Vulkan backend)
//! - WebGPU (via cubecl WGPU backend)
//! - Metal (via cubecl Metal backend)
//!
//! This closes the "WGPU/Burn GPU" gap vs the turboquant crate.
//!
//! # Architecture
//!
//! cubecl kernels are written in Rust using the `#[cube]` macro.
//! They look like regular Rust code but compile to GPU-specific backends
//! via JIT at runtime.
//!
//! # Current status: stub module
//!
//! The kernel implementations will be filled in as cubecl reaches 1.0.
//! For now, this module documents the planned kernel signatures and
//! provides the Backend::CubeCL variant.

/// cubecl backend capabilities.
#[derive(Clone, Debug)]
pub struct CubeclCapabilities {
    /// Available cubecl backends.
    pub backends: Vec<String>,
    /// Whether CUDA is available via cubecl.
    pub cuda_available: bool,
    /// Whether Vulkan is available via cubecl.
    pub vulkan_available: bool,
    /// Whether WebGPU is available via cubecl.
    pub wgpu_available: bool,
}

/// Probe cubecl backend availability.
///
/// Returns None if cubecl feature is not enabled.
#[cfg(feature = "cubecl")]
pub fn probe_cubecl() -> Option<CubeclCapabilities> {
    // cubecl probing will be implemented when the API stabilizes.
    // For now, return a placeholder.
    Some(CubeclCapabilities {
        backends: vec!["cuda".into(), "vulkan".into(), "wgpu".into()],
        cuda_available: true,
        vulkan_available: true,
        wgpu_available: true,
    })
}

#[cfg(not(feature = "cubecl"))]
pub fn probe_cubecl() -> Option<CubeclCapabilities> {
    None
}

/// Planned cubecl kernel signatures (to be implemented with #[cube] macro):
///
/// 1. `cubecl_quantize_boundary`: boundary-search quantization
///    Input: f32 values, f32 boundaries
///    Output: u8 indices
///    Maps to: turboquant_quantize_boundary in .cu / quantize.comp
///
/// 2. `cubecl_dequant_dot`: storage-compute split dequant + dot product
///    Input: u8 key_indices (SoA), f32 centroids, f32 queries
///    Output: f32 attention scores
///    Maps to: turboquant_dequant_dot in .cu / dequant_dot.comp
///
/// 3. `cubecl_sign_dot`: QJL sign-sketch inner product
///    Input: u32 packed_signs, f32 s_matrix, f32 query
///    Output: f32 correction terms
///    Maps to: turboquant_sign_dot in .cu
///
/// 4. `cubecl_fast_jl_rotate`: Walsh-Hadamard + Rademacher rotation
///    Input: f32 data, f32 d1, f32 d2
///    Output: f32 rotated data (in-place)
///    Maps to: turboquant_fast_jl_rotate in .cu
///
/// 5. `cubecl_dequant_dot_q16`: Q16.16 exact dequant + dot
///    Input: u8 key_indices, i32 centroids_q16, i32 queries_q16
///    Output: f32 attention scores
///    Maps to: turboquant_dequant_dot_q16 in .cu
pub const PLANNED_KERNELS: usize = 5;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_probe_cubecl() {
        let result = probe_cubecl();
        #[cfg(feature = "cubecl")]
        {
            assert!(result.is_some());
            let caps = result.unwrap();
            assert!(!caps.backends.is_empty());
        }
        #[cfg(not(feature = "cubecl"))]
        {
            assert!(result.is_none());
        }
    }

    #[test]
    fn test_planned_kernel_count() {
        assert_eq!(PLANNED_KERNELS, 5);
    }
}
