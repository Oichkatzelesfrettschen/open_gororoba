//! CUDA backend for TurboQuant: hardware probing, NVRTC JIT, kernel dispatch.
//!
//! Ported from open_gororoba's `lbm_3d_cuda` crate which implements the same
//! pattern: probe hardware -> select architecture -> JIT compile -> dispatch.
//!
//! # Architecture selection (from steinmarder SASS measurements)
//!
//! | Architecture | SM   | Key Feature       | TurboQuant Impact           |
//! |--------------|------|-------------------|-----------------------------|
//! | Ampere       | sm80 | BF16 tensor cores | BF16 codebook storage       |
//! | Ada Lovelace | sm89 | FP8, 128KB L1     | FP8 indices, L1-resident kv |
//! | Hopper       | sm90 | TMA, warp-special | Async dequant+attention     |
//!
//! # SASS-informed design rules (steinmarder RTX 4070 Ti measurements)
//!
//! - FFMA: 4.54 cyc latency, 44.6 ops/clk -- all CD multiply leaf ops
//! - MUFU.RCP: 41.53 cyc -- AVOID transcendentals, pre-compute norms
//! - LDG L2 miss: 92-123 cyc -- minimize global loads, use L1 (128KB on Ada)
//! - POPC: 7-8 cyc -- fast enough for inline sign-vector popcount
//! - FTZ: zero overhead on Ada -- f32 denormals free
//!
//! # Kernel design (storage-compute split, from steinmarder INT8 SoA LBM)
//!
//! Store quantized indices as u8, promote to f32 centroid values only at
//! the point of arithmetic (dot product accumulation).  SoA layout
//! (indices[coord_idx * n_vectors + vector_idx]) for coalesced access.

pub mod device;
pub mod jit;

pub use device::{CudaDeviceProps, probe_device};
