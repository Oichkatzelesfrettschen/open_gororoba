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
//! # Current status: WGPU launcher complete; other backends planned
//!
//! As of 2026-05-10 the `quantize` kernel is wired up end-to-end against
//! the cubecl-wgpu runtime via `launcher::quantize` (parity-verified
//! against the CPU reference on 1024 samples). The remaining four
//! planned kernels (dequant_dot, sign_dot, fast_jl_rotate,
//! dequant_dot_q16) and the CUDA / Vulkan / Metal cubecl backends
//! are still scheduled to land as cubecl reaches 1.0.

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
/// Returns `None` when the `cubecl` Cargo feature is disabled. When
/// enabled, returns a [`CubeclCapabilities`] populated as follows:
///
/// - `wgpu_available`: probed live via [`launcher::is_available`],
///   which constructs a `WgpuDevice::default()` + a `WgpuRuntime`
///   client and returns `true` iff that handshake succeeds. This is the
///   only backend with a real probe today.
/// - `cuda_available` / `vulkan_available`: reported as `false` until
///   the corresponding cubecl backends are wired up. They are intentionally
///   conservative -- callers that route on these flags will fall back to
///   the existing CUDA (cudarc) and Vulkan (ash) backends, which remain
///   the production path for those targets.
///
/// `backends` lists the labels of every probe-positive backend, so a
/// downstream feature-detect loop can iterate without having to know
/// about each flag separately.
#[cfg(feature = "cubecl")]
pub fn probe_cubecl() -> Option<CubeclCapabilities> {
    let wgpu_available = launcher::is_available();
    let mut backends = Vec::new();
    if wgpu_available {
        backends.push("wgpu".to_string());
    }
    Some(CubeclCapabilities {
        backends,
        cuda_available: false,
        vulkan_available: false,
        wgpu_available,
    })
}

#[cfg(not(feature = "cubecl"))]
pub fn probe_cubecl() -> Option<CubeclCapabilities> {
    None
}

#[cfg(feature = "cubecl")]
pub mod quantize_kernel;

/// Host-side launcher that bridges `quantize_kernel` to a runnable Rust
/// function via the cubecl-wgpu runtime. See `launcher::quantize` for
/// the public API and `launcher::is_available` for the runtime probe.
#[cfg(feature = "cubecl")]
pub mod launcher;

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
