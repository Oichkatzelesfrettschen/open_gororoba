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
//! # Real LLM Validation (6 models, 135M-8B, WikiText-2)
//!
//! | Model | Params | 3b cos | 4b cos | PPL +3b | PPL +4b | Kurt raw->rot |
//! |-------|--------|--------|--------|---------|---------|---------------|
//! | SmolLM2 | 135M | 0.9847 | 0.9958 | +13.9% | +1.3% | 10.2 -> 2.53 |
//! | TinyLlama | 1.1B | 0.9847 | 0.9958 | -- | -- | 28.0 -> 2.68 |
//! | Qwen2.5 | 3.1B | 0.9839 | 0.9956 | +101% | +8.1% | 18.1 -> 2.97 |
//! | Mistral | 7.2B | 0.9836 | 0.9955 | +2.4% | -0.6% | 6.9 -> 2.91 |
//! | DeepSeek-R1 | 8.0B | 0.9837 | 0.9956 | -13.6% | +0.2% | 5.6 -> 2.93 |
//!
//! Published SOTA comparison (RotateKV, Llama-2-7B, WT2/4096):
//!   4-bit +0.2%, 3-bit +1.6%, 2-bit +7.4%
//! Our Mistral-7B (WT2/128): 4-bit -0.6%, 3-bit +2.4%, 2-bit +13.8%
//!
//! Gap vs SOTA at 2-bit from: (1) per-group metadata overhead,
//! (2) random vs calibrated channel reordering, (3) shorter eval context.
//!
//! CDAQ module (`cdaq.rs`): E8-structured channel reordering + per-group
//! scales + precision windows + outlier retention.  Improves cosine by
//! +0.7% on real data (0.9422 -> 0.9494 on Mistral-7B 2-bit).
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
pub mod albert_algebra;
pub mod attention_correction;
pub mod autotune;
pub mod backend;
pub mod baselines;
pub mod batch_ops;
pub mod cd_fidelity;
pub mod cdaq;
pub mod compressor;
pub mod compute_stack;
pub mod config;
pub mod cross_layer;
#[cfg(feature = "cubecl")]
pub mod cubecl_backend;
#[cfg(feature = "cuda")]
pub mod cuda;
mod dequant_contract;
pub mod dispatch;
pub mod dot_product_bench;
pub mod e8_rotation;
pub mod e8_validation;
pub mod exceptional_roots;
pub mod f4_rotation;
pub mod fixed_codebook;
pub mod fixed_point;
pub mod fp8;
pub mod gpu_evaluation;
pub mod grouping;
pub mod hierarchical;
pub mod hybrid;
pub mod inline_cd;
pub mod integration_test;
pub mod lazy_promote;
pub mod multi_precision;
pub mod multi_resolution;
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
#[cfg(feature = "vulkan")]
pub mod vulkan;
pub mod wht_crate_scope;
pub mod workspace;
pub mod zd_bias;
