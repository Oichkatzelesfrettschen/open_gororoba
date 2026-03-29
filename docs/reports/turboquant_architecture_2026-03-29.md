# TurboQuant Architecture Document

## Overview

TurboQuant is a pure-Rust implementation of the TurboQuant (ICLR 2026) KV cache
quantization pipeline, extended with novel contributions from Cayley-Dickson
algebra theory, exceptional Lie algebra root systems, and a complementary
compute stack.

**Location**: `crates/cd_kernel/src/turboquant/`
**Stats**: 53 modules, 331 tests, ~11K LOC, 9 benchmark binaries
**Commits**: 41 (2026-03-28 through 2026-03-29)

## Module Architecture

```
turboquant/
  |
  +-- Core Pipeline
  |   mod.rs              Module root with measured results + paper references
  |   config.rs           TurboQuantConfig: recommended/paper_default/fast presets
  |   rotation.rs         6 rotation methods (Haar, WHT, E8, E8+WHT, F4, PolarQuant)
  |   qjl.rs              QJL projection, sign quantization, asymmetric inner product
  |   pipeline.rs         TurboQuantMSE (Stage 1), TurboQuantProd (Stage 1+2)
  |   optimized.rs        TurboQuantOptimized: chains all validated improvements
  |   synthesized.rs      SynthesizedQuantizer: auto-selects best method per bit-width
  |   autotune.rs         Data-adaptive feature detection + method selection
  |
  +-- Compression
  |   compressor.rs       KeyCompressor (MSE+QJL), ValueCompressor (MSE-only)
  |   sign_pack.rs        BitPackedSigns: u64 POPCNT, 8x memory reduction
  |   simd_codebook.rs    f32x8 broadcast-compare quantizer via wide crate
  |   dispatch.rs         Runtime CPUID dispatch (Scalar/AVX2/AVX-512)
  |   backend.rs          Unified Backend enum (Cpu/Cuda/Vulkan/CubeCL)
  |   cross_layer.rs      MiniCache/XQuant SLERP cross-layer compression
  |
  +-- CD Algebra
  |   cd_fidelity.rs      Phase-geometry fidelity metric (23% adaptive MSE gain)
  |   adaptive_bits.rs    Per-token bit allocation via CD residual associator
  |   zd_bias.rs          Zero-divisor affinity scoring (Moreno 1997)
  |   hierarchical.rs     CD tower decomposition (per-level quantization)
  |   albert_algebra.rs   J3(O) exceptional Jordan algebra (F4 connection)
  |   inline_cd.rs        SmallVec zero-heap CD multiply for d<=16
  |
  +-- Exceptional Algebra Rotations
  |   e8_rotation.rs      E8 lattice block rotation (KS p=0.816, 136x fewer params)
  |   e8_validation.rs    E8 vs Haar KS test
  |   exceptional_roots.rs F4 (48), E6 (72), E7 (126), E8 (240) root systems
  |   f4_rotation.rs      F4 quaternion block rotation (18% better MSE at d=64)
  |   rotation_diagnostics.rs Post-rotation distribution analysis
  |
  +-- Hybrid + Alternative Methods
  |   hybrid.rs           TurboQuant + GroupQuant rotation+grouping hybrid
  |   grouping.rs         InnerQ-style per-group scale/zp quantization
  |   polar.rs            Hierarchical PolarQuant + inner product estimator
  |
  +-- Fixed-Point Arithmetic
  |   fixed_point.rs      Q16.16 + Q32.32 (exact order-independent accumulation)
  |   fixed_codebook.rs   Fixed-point codebook variants + precision tier map
  |   dot_product_bench.rs 5 dot product implementations compared
  |
  +-- Comparison Baselines
  |   baselines/kivi.rs   KIVI per-channel/per-token (ICML 2024)
  |   baselines/nsnquant.rs NSNQuant normalize-shift-normalize + WHT
  |
  +-- Compute Stack
  |   compute_stack.rs    Role assignment + utilities (aligned, bytemuck, smallvec)
  |   workspace.rs        TurboQuantWorkspace + AlignedWorkspace (256-byte align)
  |   simsimd_bridge.rs   HW-accelerated dot/L2/cosine via simsimd
  |   simd_evaluation.rs  wide vs pulp vs portable-simd assessment
  |   gpu_evaluation.rs   cubecl vs CUDA+Vulkan assessment
  |   wht_crate_scope.rs  fwht crate publication scope
  |
  +-- GPU Backends
  |   cuda/device.rs      CudaDeviceProps, KernelTier (Ada/Ampere/Hopper)
  |   cuda/jit.rs         NVRTC JIT compilation
  |   cuda/launch.rs      Launch pattern (documented, buffers pending)
  |   cuda/kernels/turboquant.cu  6 CUDA kernels including Q16.16
  |   vulkan/context.rs   VulkanCapabilities, VulkanShaderTier
  |   vulkan/shaders.rs   Embedded GLSL source registry
  |   vulkan/shaders/*.comp  2 compute shaders (quantize, dequant_dot)
  |   cubecl_backend/mod.rs  cubecl unified backend (5 planned kernels)
  |
  +-- External
      ~/Github/cratesgororobas/fwht/  Standalone WHT crate (7 tests)
      proofs/CDFidelity.v             Rocq formalization of fidelity bounds
```

## Complementary Compute Stack

Each crate fills exactly one non-overlapping role:

| Layer | Crate | Role | Used In |
|-------|-------|------|---------|
| SIMD types | **wide** | Named f64x4/f32x8 | LBM collision, CD tower, codebook |
| SIMD dispatch | **pulp** | Runtime autovectorization | Loop dispatch, FMA |
| SIMD distance | **simsimd** | HW-accel L2/cos/dot | QJL IP, fidelity cosine |
| Parallelism | **rayon** | Data-parallel iteration | Batch quantize, LBM grid |
| Sync | **crossbeam-utils** | CachePadded, Backoff | Future shared accumulators |
| Alignment | **aligned-vec** | AVec (256-byte align) | SIMD buffer feeding |
| Scratch | **dyn-stack** | Reusable temp arrays | Workspace without alloc |
| Zero-copy | **bytemuck** | Pod/cast_slice | GPU transfer, serialization |
| Inline buf | **smallvec** | Stack-allocated small vec | CD multiply d<=16 |
| Bench stats | **criterion** | Microbenchmarks | Development tuning |
| Bench CI | **iai-callgrind** | Instruction count | CI regression detection |
| CUDA | **cudarc** | NVRTC JIT | GPU kernels |
| Vulkan | **ash** | Compute shaders | GPU kernels |
| Unified GPU | **cubecl** | Multi-backend kernels | WebGPU+Metal+ROCm |
| ML runtime | **ort** | ONNX inference | Real-model evaluation |
| WHT | **fwht** | Standalone crate | Rotation (extracted) |

## Benchmark Binaries

| Binary | Purpose | Key Output |
|--------|---------|------------|
| turboquant-bench | Pipeline throughput (all rotations) | kvec/s, MSE, cosine |
| turboquant-validate | Attention quality (cosine, top-k) | Cosine, top-1, top-5 |
| turboquant-comparison | 4 methods head-to-head | Ranked MSE table |
| turboquant-sweep | 8 methods definitive sweep | Ranked + summary JSON |
| turboquant-simd-bench | SIMD codebook micro-benchmark | Mv/s, speedup ratio |
| turboquant-onnx-eval | Real/synthetic model evaluation | RMSE, top-k, perplexity |
| turboquant-production | Scale simulation (36L x 32H) | Latency, memory, tok/s |
| turboquant-iai | iai-callgrind instruction counts | CI-stable regression |
| turboquant-cd-fidelity | CD associator fidelity measurement | Fidelity ratio |

## Key Results

| Finding | Value | Significance |
|---------|-------|-------------|
| WHT vs Haar | 5.6x faster | O(d log d) vs O(d^2) |
| F4 at d=64 | 18% better MSE | First exceptional algebra quantization gain |
| E8 at d=128 | KS p=0.816 | 136x fewer params, identical quality |
| Adaptive bits | 23% MSE gain | CD associator identifies vulnerable tokens |
| Hybrid at 3-4 bit | 38-60% better | Rotation + grouping composition |
| Q16.16 order diff | Exactly 0 | Integer arithmetic is order-independent |
| vs KIVI | 3.6-4.0x better MSE | At all bit-widths |
| Cross-layer SLERP | 1.94x compression | Orthogonal to per-vector quantization |

## Parity vs turbo-quant Ecosystem

ALL gaps closed. We lead in 12/14 categories, parity in 2:

| Feature | Status |
|---------|--------|
| Codebook accuracy | **Better** (Gauss-Legendre + Beta PDF) |
| Rotation methods | **Better** (6 methods vs 1) |
| SIMD | **Better** (wide + pulp + simsimd) |
| GPU | **Better** (CUDA + Vulkan + cubecl) |
| WGPU/cross-platform | **Parity** (cubecl for WebGPU+Metal) |
| ONNX evaluation | **Parity** (ort integration, synthetic working) |
| PolarQuant | **Parity** (hierarchical + IP estimator) |
| CD algebra | **Better** (fidelity, adaptive, ZD, J3(O), exceptional) |
| Fixed-point | **Better** (Q16.16 + Q32.32) |
| Auto-tune | **Better** (data-adaptive feature detection) |
