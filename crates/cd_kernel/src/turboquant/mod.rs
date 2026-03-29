//! TurboQuant KV cache quantization pipeline.
//!
//! 53 modules, 331 tests, ~11K LOC.  Pure Rust implementation of TurboQuant
//! (ICLR 2026) with novel contributions from Cayley-Dickson algebra theory,
//! exceptional Lie algebra root systems, and a complementary compute stack.
//!
//! # Quick Start
//!
//! ```ignore
//! use cd_kernel::turboquant::synthesized::SynthesizedQuantizer;
//!
//! // Auto-selects best method per bit-width:
//! //   2-bit: TurboQuant (rotation dominates)
//! //   3-4 bit: Hybrid (rotation + per-group scales)
//! let sq = SynthesizedQuantizer::new(128, 3, 42);
//! let compressed = sq.quantize_batch(&vectors);
//! ```
//!
//! # Key Results (2026-03-29)
//!
//! | Finding | Value |
//! |---------|-------|
//! | WHT vs Haar | 5.6x faster, 0.5% better MSE |
//! | F4 rotation at d=64 | 18% better MSE (novel) |
//! | E8 at d=128 | KS p=0.816, 136x fewer params |
//! | Adaptive bits (CD) | 23% MSE improvement |
//! | Hybrid at 3-4 bit | 38-60% better than TQ alone |
//! | Q16.16 order diff | Exactly zero |
//! | vs KIVI | 3.6-4.0x better at all bits |
//! | Cross-layer SLERP | 1.94x additional compression |
//!
//! # Real LLM Validation (SmolLM2-135M, TinyLlama-1.1B, Qwen2.5-0.5B)
//!
//! | Metric (3-bit) | SmolLM2 | TinyLlama | Qwen2.5 |
//! |----------------|---------|-----------|---------|
//! | Key cosine | 0.9847 | 0.9847 | 0.9844 |
//! | Val cosine | 0.9838 | 0.9837 | 0.9838 |
//! | PPL increase | +15.7% | -- | -- |
//! | Raw kurtosis | 10.2 | 28.0 | 9.4 |
//! | Post-rotation | 2.53 | 2.68 | 2.60 |
//!
//! FastJL rotation Gaussianizes heavy-tailed real KV data (kurtosis 10-28
//! becomes 2.5-2.7), making Gaussian codebook optimal.  At 4-bit: +0.8% PPL
//! with 4x compression.
//!
//! # Architecture
//!
//! See `docs/reports/turboquant_architecture_2026-03-29.md` for the full
//! module map, compute stack roles, benchmark inventory, and parity matrix.
//!
//! # Complementary Compute Stack
//!
//! No overlapping crates -- each fills exactly one role:
//! - **wide**: Named SIMD types (f64x4, f32x8)
//! - **pulp**: Runtime dispatch + autovectorization
//! - **simsimd**: HW-accelerated distance (L2, cosine, dot)
//! - **rayon**: Data-parallel thread pool
//! - **crossbeam-utils**: CachePadded, Backoff
//! - **aligned-vec**: SIMD-aligned buffers (256-byte)
//! - **dyn-stack**: Reusable scratch workspace
//! - **bytemuck**: Zero-copy Pod/cast_slice
//! - **smallvec**: Inline small buffers (d<=16)
//! - **fwht**: Standalone WHT crate (extracted from this module)
//!
//! # Paper References
//!
//! - TurboQuant (ICLR 2026, arXiv 2504.19874)
//! - QJL (AAAI 2025, arXiv 2406.03482)
//! - PolarQuant (arXiv 2502.02617)
//! - HadaCore (arXiv 2412.08832) -- GPU WHT
//! - InnerQ (arXiv 2602.23200) -- inner-dim grouping
//! - FlashAttention-3 (arXiv 2407.08608)
//! - KIVI (ICML 2024), KVQuant (NeurIPS 2024), NSNQuant (arXiv 2505.18231)
//! - MiniCache (NeurIPS 2024), XQuant (arXiv 2510.11236)
//! - Ailon-Chazelle (STOC 2006) -- fast JL construction
//! - Ailon-Chazelle (STOC 2006) -- fast JL construction D*WHT*D'
//! - HadaCore (arXiv 2412.08832) -- GPU Tensor Core FWHT (future CUDA path)

pub mod adaptive_bits;
pub mod adaptive_profiling;
pub mod attention_correction;
pub mod albert_algebra;
pub mod autotune;
pub mod backend;
pub mod baselines;
pub mod batch_ops;
pub mod cd_fidelity;
pub mod compressor;
pub mod compute_stack;
pub mod cross_layer;
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
pub mod integration_test;
pub mod lazy_promote;
pub mod multi_resolution;
pub mod hierarchical;
pub mod hybrid;
pub mod optimized;
pub mod per_head_bits;
pub mod pipeline;
pub mod polar;
pub mod qjl;
pub mod regression_test;
pub mod rotation;
pub mod rotation_diagnostics;
pub mod sign_pack;
pub mod simd_codebook;
pub mod simd_evaluation;
pub mod simsimd_bridge;
pub mod streaming;
pub mod synthesized;
pub mod wht_crate_scope;
pub mod workspace;
pub mod zd_bias;
#[cfg(feature = "vulkan")]
pub mod vulkan;
