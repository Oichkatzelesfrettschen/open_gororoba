//! TurboQuant KV cache quantization pipeline.
//!
//! Pure Rust port of the TurboQuant (ICLR 2026) two-stage pipeline:
//!   Stage 1 (MSE): Random rotation + per-coordinate Lloyd-Max scalar quantization
//!   Stage 2 (QJL): Quantized Johnson-Lindenstrauss 1-bit sign sketches for
//!                   unbiased inner product estimation on the residual
//!
//! The rotation decorrelates coordinates so the marginal distribution is
//! approximately N(0, 1/d), enabling a single precomputed codebook per
//! (dimension, bits) pair.  The QJL correction debiases the inner product
//! estimator at the cost of 1 extra bit per coordinate.
//!
//! # Measured results (2026-03-28, Rust release mode)
//!
//! | Config (d=128) | Throughput | MSE/coord | Cosine | Compression |
//! |----------------|-----------|-----------|--------|-------------|
//! | 2-bit WHT      | 79 kvec/s | 1.395     | 0.377  | 7.1x        |
//! | 3-bit WHT      | 77 kvec/s | 1.444     | 0.446  | 4.9x        |
//! | 4-bit WHT      | 72 kvec/s | 1.540     | 0.452  | 3.8x        |
//! | 3-bit Haar     | 25 kvec/s | 1.451     | 0.441  | 4.9x        |
//!
//! WHT is 3x faster than Haar and gives 0.5% better MSE.  Sign packing
//! via [`sign_pack::BitPackedSigns`] provides 8x memory reduction for QJL
//! storage (128 bytes -> 16 bytes per sign vector at d=128).
//!
//! # Sources
//!
//! Ported from:
//! - `~/Github/turboquant-pytorch/turboquant.py` (Haar rotation, QJL, pipeline)
//! - `~/Github/TurboQuant/wht_rotation.py` (Walsh-Hadamard butterfly)
//! - `~/Github/TurboQuant/compressors.py` (CompressorV2, CompressorMSE)
//!
//! Hardware patterns from:
//! - `~/Github/steinmarder/src/cuda_lbm/` (storage-compute split, SoA layout)
//! - `~/Github/steinmarder/src/nerf/nerf_simd.c` (AVX2 dispatch)
//!
//! Paper references:
//! - TurboQuant (ICLR 2026, arXiv 2504.19874)
//! - QJL (AAAI 2025, arXiv 2406.03482)
//! - PolarQuant (arXiv 2502.02617) -- theory basis for Lloyd-Max codebook
//! - Ailon-Chazelle (STOC 2006) -- fast JL construction D*WHT*D'
//! - HadaCore (arXiv 2412.08832) -- GPU Tensor Core FWHT (future CUDA path)

pub mod adaptive_bits;
pub mod albert_algebra;
pub mod autotune;
pub mod backend;
pub mod baselines;
pub mod cd_fidelity;
pub mod compressor;
pub mod compute_stack;
pub mod config;
#[cfg(feature = "cubecl")]
pub mod cubecl_backend;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod dispatch;
pub mod dot_product_bench;
pub mod e8_rotation;
pub mod e8_validation;
pub mod exceptional_roots;
pub mod f4_rotation;
pub mod fixed_codebook;
pub mod fixed_point;
pub mod gpu_evaluation;
pub mod grouping;
pub mod inline_cd;
pub mod hierarchical;
pub mod hybrid;
pub mod optimized;
pub mod pipeline;
pub mod polar;
pub mod qjl;
pub mod rotation;
pub mod rotation_diagnostics;
pub mod sign_pack;
pub mod simd_codebook;
pub mod simd_evaluation;
pub mod simsimd_bridge;
pub mod synthesized;
pub mod wht_crate_scope;
pub mod workspace;
pub mod zd_bias;
#[cfg(feature = "vulkan")]
pub mod vulkan;
