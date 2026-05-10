//! cubecl `#[cube]` quantize kernel -- Phase A scaffold.
//!
//! Phase A captures the intended kernel signature + algorithm in pure
//! Rust prose. Phase B (when cubecl reaches 1.0 and the meta-crate
//! lands in workspace deps) will replace this with the real
//! #[cube(launch_unchecked)] annotated function.
//!
//! # Why this is documentation-only in Phase A
//!
//! - cubecl 0.10.0-pre.2 is the workspace-pinned version, but the
//!   meta-crate `cubecl` (which re-exports `prelude::*` and the
//!   `#[cube]` macro context) isn't fully wired into this workspace's
//!   dep graph yet. cubecl-core alone exposes the IR but the macro
//!   expansion needs the meta-crate as a sibling import.
//! - cubecl 0.10's atomic API is in flux (the AtomicU32::or signature
//!   changed between rc.1 and rc.2; rc.3 may move it again).
//! - Wiring the cubecl runtime + ComputeClient + per-runtime backend
//!   probe (CUDA / wgpu / metal) is a multi-day integration that
//!   warrants its own focused sprint.
//!
//! # Intended kernel (Phase B)
//!
//! ```ignore
//! use cubecl::prelude::*;
//!
//! #[cube(launch_unchecked)]
//! pub fn quantize_kernel(
//!     values: &Array<f32>,
//!     boundaries: &Array<f32>,
//!     indices: &mut Array<u32>,
//!     #[comptime] n_boundaries: u32,
//! ) {
//!     let gid = ABSOLUTE_POS;
//!     if gid >= values.len() { terminate!(); }
//!     let v = values[gid];
//!     let mut count: u32 = 0;
//!     for b in 0..n_boundaries {
//!         if v > boundaries[b] { count += 1; }
//!     }
//!     indices[gid] = count;
//! }
//! ```
//!
//! The kernel body matches the boundary-count algorithm in
//! vulkan/shaders/quantize.comp; per-thread output is a u32 that the
//! CPU-side wrapper packs into u8 bytes after readback (avoiding the
//! cubecl 0.10 atomic-API churn). The corresponding wrapper would
//! call:
//!
//! ```ignore
//! quantize_kernel::launch_unchecked::<R>(
//!     &client,
//!     CubeCount::Static(workgroup_count, 1, 1),
//!     CubeDim::new(256, 1, 1),
//!     ArrayArg::from_raw_parts::<f32>(&values_handle, n_values, 1),
//!     ArrayArg::from_raw_parts::<f32>(&boundaries_handle, n_boundaries, 1),
//!     ArrayArg::from_raw_parts::<u32>(&indices_handle, n_values, 1),
//!     ScalarArg::new(n_boundaries),
//! );
//! ```
//!
//! # Phase B prerequisites
//!
//! 1. Add `cubecl = { workspace = true, optional = true }` to
//!    cd_kernel/Cargo.toml deps and to the cubecl feature.
//! 2. Pick a runtime: `cubecl::cuda::CudaRuntime` for NVIDIA-only,
//!    `cubecl::wgpu::WgpuRuntime` for cross-platform.
//! 3. Validate the AtomicU32 / Array<u32> output approach against the
//!    final cubecl 1.0 atomic API.
//! 4. Add a parity test against Backend::Cpu output for a 1024-element
//!    sample.
//!
//! Until then, Backend::CubeCL falls back to CPU at runtime (see
//! turboquant/backend.rs Backend::CubeCL match arm).

#![cfg(feature = "cubecl")]
