//! GPU-accelerated tensor contractions for quantum tensor networks.
//!
//! This module provides CUDA-based acceleration for expensive tensor operations,
//! with automatic CPU fallback when no GPU is available.
//!
//! # Feature Gate
//!
//! GPU support is optional and controlled by the `gpu` feature flag.
//! Compile with `--features gpu` to enable GPU acceleration.
//!
//! # Architecture
//!
//! - Uses cudarc with dynamic CUDA loading (no build-time CUDA requirement)
//! - NVRTC compiles kernels at runtime
//! - Graceful fallback to CPU implementations if no device found
//! - Per-module feature gating to isolate GPU code

#![cfg(feature = "gpu")]

pub mod peps;

pub use peps::gpu_contract_rows_peps;
