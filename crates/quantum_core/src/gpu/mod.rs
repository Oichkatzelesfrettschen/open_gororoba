//! GPU-accelerated tensor contractions for quantum tensor networks.
//!
//! The CUDA feature keeps the FP64 PEPS contraction path. The cubecl and
//! Vulkan features add portable PEPS row kernels with explicit FP32 precision
//! contracts.
//!
//! # Feature Gate
//!
//! GPU support is optional. Compile with `--features gpu` for CUDA,
//! `--features cubecl` for the cubecl-wgpu path, or `--features vulkan` for
//! Vulkan compute.
//!
//! # Architecture
//!
//! - CUDA uses cudarc with dynamic loading and NVRTC runtime compilation.
//! - cubecl uses cubecl-wgpu runtime compilation.
//! - Vulkan uses the workspace ash helper crate.
//! - Both paths preserve CPU fallback when no GPU device is available.
//! - Per-module feature gating isolates backend code.

#![cfg(any(feature = "gpu", feature = "cubecl", feature = "vulkan"))]

#[cfg(feature = "gpu")]
pub mod peps;
#[cfg(feature = "cubecl")]
pub mod peps_cubecl;
#[cfg(feature = "vulkan")]
pub mod peps_vulkan;

#[cfg(feature = "gpu")]
pub use peps::gpu_contract_rows_peps;
#[cfg(feature = "cubecl")]
pub use peps_cubecl::cubecl_contract_rows_peps_fp32;
#[cfg(feature = "vulkan")]
pub use peps_vulkan::vulkan_contract_rows_peps_fp32;
