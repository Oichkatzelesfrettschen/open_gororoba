# TurboQuant x Cayley-Dickson Optimization Analysis

**Source**: tonbistudio/turboquant-pytorch (ICLR 2026)
**Analysis date**: 2026-03-28

## TurboQuant Architecture

Two-stage vector quantization for LLM KV cache compression:
- Stage 1 (MSE): Haar random rotation + per-coordinate Lloyd-Max
- Stage 2 (QJL): 1-bit Quantized Johnson-Lindenstrauss on residuals
- Claims: 5x compression at 3-bit with 99.5% attention fidelity

## Optimization Opportunities from CD Tower

### 1. Sedenion-Structured Rotation (128x fewer ops)

**Current**: d x d Haar rotation (16,384 multiply-adds for d=128)
**Proposed**: Decompose into d/16 sedenion left-multiplications (128 multiply-adds)
**Mechanism**: Unit sedenion a in S^15 defines orthogonal map L_a: R^16 -> R^16.
For d=128, use 8 independent sedenion multiplications on 16D blocks.
**Advantage**: 128x fewer arithmetic ops, SIMD-friendly (our wide::f64x4 pattern)
**Risk**: Sedenion rotation may not achieve same decorrelation as Haar. Test needed.

### 2. SIMD Codebook Lookup (4-8x faster)

**Current**: PyTorch broadcast subtract + abs + argmin (generic tensor ops)
**Proposed**: Rust wide::f32x8 vectorized boundary search
**Mechanism**: At 3-bit, 8 centroids fit in one f32x8 SIMD register.
Subtract all 8 simultaneously, take abs, find min index in one pass.
**Advantage**: 4-8x speedup over PyTorch for small codebook sizes

### 3. CD Associator as Fidelity Metric

**Current**: Cosine similarity / L2 distortion
**Proposed**: 32D CD associator on pre/post-quantization attention keys
**Mechanism**: Embed attention weight vectors in pathion space via Takens delay.
A_post/A_pre ratio measures phase-geometry distortion invisible to scalar metrics.
**Advantage**: Detects non-linear relationship distortion (our MAST results show
CD captures dynamics that L2 misses)

### 4. Algebraic Distortion Bounds

**Current**: Statistical distortion estimates from Gaussian approximation
**Proposed**: Exact bounds from CD algebra theory
**Mechanism**: The sedenion multiplication table has known eigenvalue structure
(Moreno 1997 theorem). The distortion of a sedenion rotation is bounded by
the spectrum of the left-multiplication operator.
**Advantage**: Provable worst-case guarantees instead of average-case estimates

## Implementation Priority

1. **CD fidelity metric** (low effort, high insight) -- measure quantization quality
2. **SIMD codebook** (moderate effort, guaranteed speedup) -- port to Rust
3. **Sedenion rotation** (high effort, potentially transformative) -- needs theory + experiment
