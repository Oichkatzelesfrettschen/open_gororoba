# TurboQuant x Cayley-Dickson Optimization Analysis

**Source**: tonbistudio/turboquant-pytorch (ICLR 2026, arXiv:2504.19874)
**Analysis date**: 2026-03-28

## TurboQuant Architecture

Two-stage vector quantization for LLM KV cache compression:
- Stage 1 (MSE): Haar random rotation + per-coordinate Lloyd-Max scalar quantization
- Stage 2 (QJL): 1-bit Quantized Johnson-Lindenstrauss on residuals for unbiased inner products
- Claims: 5x compression at 3-bit with 99.5% attention fidelity
- KV Cache: Keys use TurboQuantProd (inner products for attention), Values use TurboQuantMSE

## Optimization Opportunities from CD Tower

### 1. Walsh-Hadamard Transform = CD Doubling Butterfly (19x fewer FLOPs)

**Current**: Dense d x d Haar rotation via QR decomposition. At d=128: 134M FLOPs per rotation.
**Proposed**: Replace with randomized Hadamard transform (RHT): `Pi_rht = D * H_d * D'` where
D, D' are random sign matrices and H_d is the Walsh-Hadamard transform.
**CD connection**: The WHT at d=128 is a 7-level butterfly network. Each level is structurally
identical to the CD doubling rule `(a,b)(c,d) = (ac - d*b, da + bc*)`. The `implement_cd_algebra!`
macro in `gororoba_algebra::construction::cd_tower` generates these butterfly structures at
compile time. The d=128 WHT maps precisely to the CD tower from CdPairH (4D) through Pathion (32D)
up to 128D.
**Result**: O(d log d) = 128 * 7 = 896 multiply-adds vs O(d^2) = 16,384. **19x fewer FLOPs**.
**Risk**: RHT is not identical to Haar; distortion bound changes by constant factor. Paper's
framework allows structured rotations (see OpenReview supplementary).

### 2. QJL Sign Packing via cd_sign_fuel (8x memory reduction)

**Current**: Each QJL sign vector is d=128 int8 values (128 bytes per vector).
**Proposed**: Pack into single u128 using `cd_sign_fuel` bit-indexing infrastructure.
Inner product via popcount: `<a,b> = 2 * sum_{i: a_i=+1} b_i - sum(b_i)`.
**At scale**: 8192 tokens * 36 layers * 32 heads * 128 bytes = 1.2 GB for signs alone.
After packing: 150 MB. **8x reduction** in QJL storage.

### 3. SIMD Codebook Lookup (4-8x faster)

**Current**: PyTorch broadcast subtract + abs + argmin materializing (N, d, 2^b) tensor.
At N=8192, d=128, b=3: 32 MB allocation per layer.
**Proposed**: Rust wide::f32x8 vectorized boundary search. At 3-bit, 8 centroids fit
in one f32x8 SIMD register. Single-pass distance + argmin.
**Memory**: Eliminates the (N, d, 2^b) intermediate entirely.

### 4. CD Associator as Per-Token Residual Quality Score (novel)

**Current**: No per-token quality metric for QJL approximation quality.
**Proposed**: Compute `||[r, e_1, e_2]||` on the QJL residual r for each token.
High associator = residual has structure that sign projections capture poorly.
**Application**: Drives adaptive bit allocation -- give more bits to high-associator tokens.
Connects heliospheric boundary detection (CD identifies transition layers)
to KV cache compression (CD identifies quantization-vulnerable tokens).

### 5. Zero-Divisor Analysis for Systematic Bias Detection

**Current**: Known bias factor 2/pi = 0.637 for 1-bit QJL.
**Proposed**: Apply Moreno (1997) `is_zd_moreno` to QJL residual distributions.
Identifies structural directions where TurboQuant systematically underperforms.
Analytically grounded approach to the sign-quantization bias problem.

### 6. NVRTC Kernel Fusion (halve HBM bandwidth)

**Current**: Three separate cuBLAS matmuls in asymmetric_attention_scores.
**Proposed**: Fuse using existing cudarc 0.19.1 NVRTC infrastructure (same path
as GPU fractal dimension E-166, Cd256FrustrationKernel).
Load query q once from HBM, compute all three terms in single pass.
**Result**: 2x reduction in HBM bandwidth requirement.

### 7. Lloyd-Max Codebook Caching (bug fix)

**Current**: validate.py creates 108 compressor instances, each re-solving Lloyd-Max.
**Fix**: Cache codebook per (d, bits) pair. Free speedup.

## Integration Priority

| # | Opportunity | Effort | Impact | Risk |
|---|-----------|--------|--------|------|
| 1 | WHT butterfly rotation | Moderate | 19x FLOP reduction | Low (paper allows) |
| 2 | QJL sign packing (u128) | Low | 8x memory reduction | None |
| 3 | SIMD codebook (Rust) | Moderate | 4-8x codebook speedup | None |
| 4 | CD residual quality score | Low | Novel diagnostic | Experimental |
| 5 | ZD bias analysis | Low | Novel insight | Experimental |
| 6 | NVRTC kernel fusion | High | 2x bandwidth | Low |
| 7 | Codebook caching | Trivial | Startup speedup | None |

## Key Files in open_gororoba

- `cd_kernel/src/` -- cd_sign_fuel, basis indexing, popcount infrastructure
- `gororoba_algebra/src/construction/cd_tower.rs` -- implement_cd_algebra! butterfly macro
- `gororoba_algebra/src/gpu/avt_pack.rs` -- GpuPackableAvt, NVRTC kernel path
- `algebra_analysis/src/moreno.rs` -- is_zd_moreno, zero-divisor eigenvalue analysis
- `lbm_3d_cuda/` -- cudarc 0.19.1 runtime compilation patterns

## References

- [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) -- TurboQuant paper
- [Triton implementation](https://dejan.ai/blog/turboquant/) -- kernel fusion patterns
- Moreno (1997) -- sedenion eigenvalue structure for rotation bounds
