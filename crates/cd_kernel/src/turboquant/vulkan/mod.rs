//! Vulkan compute backend for TurboQuant.
//!
//! First-class compute backend, not a CUDA fallback.  Ported from
//! steinmarder's Vulkan subsystem (`src/vulkan/`) which treats Vulkan
//! compute shaders as a serious workload path.
//!
//! # Design principles (from steinmarder Vulkan analysis)
//!
//! - `layout(local_size_x = 256)` for 1D compute (quantize/dequant)
//! - std430 buffer layout for packed data (no std140 alignment waste)
//! - Push constants < 256 bytes for kernel parameters (bits, d, n_vectors)
//! - Subgroup shuffles for warp-level cooperation (Vulkan 1.1+)
//! - buffer_device_address for true pointer access in shaders (Vulkan 1.2+)
//!
//! # Feature gates
//!
//! - Portable baseline: core subgroup features only (subgroupBroadcast, subgroupAdd)
//! - Optional cooperative-matrix: behind VK_KHR_cooperative_matrix feature check
//! - Subgroup extended types: 8/16-bit integer and 16-bit float in subgroup ops
//!
//! # Shader inventory (to be compiled from GLSL -> SPIR-V)
//!
//! - `quantize.comp`: boundary-search quantization, SoA output
//! - `dequant_dot.comp`: codebook lookup + dot product accumulation
//! - `fused_attention.comp`: dequant + attention score in one dispatch
//!
//! # Memory layout
//!
//! SoA: `indices[coord * n_vectors + vec_idx]` for coalesced subgroup reads.
//! Signs packed as u32 words (32 signs per word) for subgroup popcount.

pub mod context;
pub mod shaders;

pub use context::VulkanCapabilities;
