//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/research_narratives.toml -->
//!
//! # Scope: Ultra-High-Throughput Warp Ring Simulation (128^3 @ 2500 Steps / 40s)
//!
//! ## 1. Objective
//! Achieve **62.5 steps per second** (16ms/step) at $128^3$ resolution on RTX 4070 Ti.
//! This corresponds to **~131 MLUPS** (Mega Lattice Updates Per Second).
//! *Theoretical Peak for RTX 4070 Ti:* ~3000-5000 MLUPS (Memory Bandwidth Limited: 504 GB/s).
//! *Required Bandwidth:* $128^3 \times 19 \times 8 \text{ bytes} \times 2 \text{ (RW)} \approx 640 \text{ MB per step}$.
//! $640 \text{ MB} \times 62.5 \approx 40 \text{ GB/s}$.
//! **Feasibility:** 40 GB/s is < 10% of 504 GB/s. We are **latency bound**, not throughput bound.
//!
//! ## 2. Optimization Strategy
//!
//! ### 2.1 GPU-Resident Loop (Implemented)
//! *   **Action:** Move `mean_density` and `enstrophy` reductions to GPU.
//! *   **Status:** Implemented in `lbm_3d_cuda` (kernels 9 & 10).
//! *   **Gain:** Eliminates 16MB transfer per check. Reduces PCI-e latency.
//!
//! ### 2.2 Advanced GPU Hardware Utilization (Ada Lovelace / SM 8.9)
//! The RTX 4070 Ti features 4th Gen Tensor Cores and 3rd Gen RT Cores. Standard CUDA kernels (SIMT) are the baseline; the following opportunities target the hardware "ceiling":
//!
//! #### 2.2.1 FP8 / INT8 Quantization (Tensor Cores)
//! *   **Opportunity:** Use NVIDIA's **Transformer Engine** (FP8) for the distribution functions ($f_i$). 
//! *   **Benefit:** 2x throughput over FP16, 4x over FP32. Reduced VRAM footprint allows $256^3$ or $512^3$ grids.
//! *   **Mechanism:** LBM collision can be cast as a **Fused Multiply-Add (FMA)** operation. 4th Gen Tensor Cores perform asynchronous FP8 matrix-multiply-accumulate (MMA) at massive rates.
//! *   **Risk:** Numerical stability of the E7 spectral forcing. Requires a "dynamic range" manager to prevent underflow in high-frequency spectral modes.
//!
//! #### 2.2.2 cuFFT Callbacks (Kernel Fusion)
//! *   **Opportunity:** Implement **cuFFT L2 Callbacks** to apply the Gaussian E7 mask.
//! *   **Benefit:** Eliminates the need for a separate `apply_mask_kernel` launch and its associated global memory round-trip.
//! *   **Mechanism:** The spectral sieve is applied in-flight as cuFFT writes elements back to global memory (or reads them). Data never leaves the L2 cache during the transformation.
//!
//! #### 2.2.3 Asynchronous Execution (CUDA Graphs)
//! *   **Opportunity:** Capture the entire `step()` loop into a **CUDA Graph**.
//! *   **Benefit:** Reduces CPU-side launch overhead from ~10us per kernel to nearly zero.
//! *   **Context:** For a 16ms step target, spending 50us on launch overhead (6 kernels + 2 FFTs) is >3% waste.
//!
//! ### 2.3 CPU / Cache / RAM Optimization (Zen 3 / 5600X3D)
//! *   **L3 Cache (3D V-Cache):** The 5600X3D has a massive 96MB L3 cache.
//! *   **Temporal Blocking:** For CPU fallback, tile the 128^3 grid into 32^3 blocks that fit entirely in L3. This allows multiple time-steps to be computed within the cache before writing back to main RAM.
//! *   **AVX2 + FMA:** Ensure all algebra/curvature code uses FMA3 instructions (supported by Zen 3) to perform $a*b + c$ in a single cycle.
//!
//! ## 3. Data Contract & Interpretation
//! *   **Schema:** `WarpRingExperiment` (in `gororoba_contracts`).
//! *   **Artifacts:** 
//! >   *   `warp_ring_trace.h5`: Time-series scalars (Enstrophy, Density).
//! >   *   `warp_field_step_N.h5`: Full 3D velocity field [nx, ny, nz, 3] for topological verification.
//! *   **Goal:** Provide a "Gold Standard" dataset for lambda-gororoba cross-validation.
//!
//! ## 4. Execution Plan
//! 1.  **Validate:** Finish current `warp_optimized_run` (1500 steps).
//! 2.  **Benchmark:** Measure steps/sec.
//! 3.  **Scale:** If > 40 steps/s, bump to 2500 steps.
//! 4.  **Contract:** Wrap final binary with `gororoba_contracts`.
//!
//! ## 6. Synthesis: Hardware-Specific "Knowledge Interpretation"
//! We are moving beyond standard CUDA into **Ada Lovelace Deep Features**:
//!
//! ### 6.1 Tensor Core & Precision Strategy (Knowledge Interpretation)
//! *   **FP8 (E4M3/E5M2):** **REJECTED** for LBM state.
//! >   *   *Reason:* LBM relies on the non-equilibrium part $f^{neq} = f - f_{eq}$, which is often $O(10^{-3})$. FP8-E4M3 has a machine epsilon of $2^{-3} = 0.125$. The entire signal would be lost to quantization noise.
//! >   *   *Verdict:* FP8 is "Not Detailed Enough".
//! *   **BFloat16 (BF16):** **ACCEPTED** for High-Performance State.
//! >   *   *Reason:* BF16 preserves the 8-bit exponent of FP32, maintaining the dynamic range required for density variations. The 7-bit mantissa is low precision, but sufficient for "visual" or "topological" LBM where exact conservation to $10^{-15}$ is not required (unlike engineering CFD).
//! >   *   *Benefit:* **50% Bandwidth Reduction** (20 GB/s vs 40 GB/s). Fits $256^3$ in same VRAM.
//! *   **Tensor Cores:**
//! >   *   *Usage:* Use `mma.m16n8k16` instructions if re-writing in raw PTX, but for CUDA C++, standard `hmma` is complex for stencil.
//! >   *   *Pivot:* Focus on **BF16 Memory + FP32 Compute** (Mixed Precision). This uses the Tensor Core's accumulators implicitly in modern CUDA compilers.
//!
//! ## 7. Performance Baseline ($128^3$)
//! *   **FP32 (Fused):** ~20-30 steps/s (Projected).
//! *   **BF16 (Fused):** ~40-60 steps/s (Projected). **Targeting 40s run.**
//!
