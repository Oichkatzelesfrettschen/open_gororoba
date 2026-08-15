//! Warp domain lattice lanes and the modules they share.
//!
//! The `warp` dispatcher binary owns the fourteen lanes under `src/bin/warp/`.
//! Three modules live here because more than one lane calls them.
//!
//! `warp_runner` drives `lbm_3d_cuda` and reaches `cudarc` directly for CUDA
//! event timing, and `warp_precision_suite_ops` types its whole configuration
//! on `lbm_3d_cuda::Precision` and calls into the runner. Both therefore carry
//! `#[cfg(feature = "gpu")]`. Without it the two `use lbm_3d_cuda` lines fail to
//! resolve and take the library down with them, which made `gpu` a permanently
//! on flag rather than the opt-in tier its `[features]` table advertised.
//! `warp_telemetry` gates its NVML sampler internally and builds either way.

#[cfg(feature = "gpu")]
pub mod warp_precision_suite_ops;
#[cfg(feature = "gpu")]
pub mod warp_runner;
pub mod warp_telemetry;
