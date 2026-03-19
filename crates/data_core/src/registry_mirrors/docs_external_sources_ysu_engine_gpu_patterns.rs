//! <!-- AUTO-GENERATED: READ-ONLY COMPATIBILITY EXPORT. -->
//! <!-- Source of truth: registry/external_sources.toml -->
//! <!-- Canonical write path: registry/canonical/control_plane.sqlite3 -->
//! <!-- Source label: XS-023 -->
//! <!-- Regenerate with: cargo run -p gororoba_cli_data --bin provenance -- export-external-sources -->
//!
//! # YSU-engine GPU Optimization Patterns: Technique Reference
//!
//! ## Source Attribution
//!
//! - Project: YSU-engine by Umut Korkmaz
//! - License: MIT
//! - Repository: https://github.com/ismail0098-lang/YSU-engine
//! - Architecture: C11/Vulkan rendering + physics engine
//! - Date extracted: 2026-03-10
//! - Purpose: Transferable GPU optimization patterns for LBM CUDA kernels (lbm_3d_cuda crate)
//! - Note: YSU-engine is NOT CUDA-based. We extract general GPU optimization principles
//! that apply to our NVRTC-compiled D3Q19 LBM kernels targeting Ada Lovelace SM 8.9.
//!
//! ## SASS Instruction Latency Table (Ada Lovelace SM 8.9)
//!
//! | Instruction                          | Latency (cycles) | Throughput (ops/cycle/SM) | Relevance to D3Q19                            |
//! |---------------------------------------|-------------------|---------------------------|------------------------------------------------|
//! | FFMA (fused multiply-add)             | 4.54              | 128                       | Dominates equilibrium computation              |
//! | MUFU.EX2 (SFU -- exp2)               | 17.56             | 16                        | Used in transcendental approximations          |
//! | MUFU.SIN (SFU -- sin)                | ~17               | 16                        | Used in voudon tau modulation                  |
//! | IADD3 (integer add, 3-input)         | 4                 | 64                        | Used in index computation                      |
//! | LDG (global load, cached)            | ~200-400 (miss)   | --                        | Dominates streaming; ~30 cy on L1 hit          |
//! | Global memory pointer chase          | 92.29             | --                        | Worst case for dependent loads                 |
//! | STS/LDS (shared memory store/load)   | ~28               | --                        | 3x faster than global for neighbor access      |
//! | ATOM (atomic add)                    | ~20 (same warp)   | --                        | ~100 cy under cross-warp contention            |
//!
//! FFMA throughput is the key metric for equilibrium kernels. Each SM on Ada Lovelace
//! can retire 128 FFMA operations per cycle, meaning that FMA-bound kernels scale
//! linearly with SM count until memory bandwidth saturates. Memory latency hiding
//! requires 4+ active warps per SM to keep the FFMA pipeline fed. The 92.29 cy global
//! pointer chase cost motivates shared memory tiling for neighbor access patterns where
//! coalescing alone is insufficient.
//!
//! ## FMA Chain Scheduling for D3Q19 Equilibrium
//!
//! The equilibrium distribution function for D3Q19 is:
//!
//! ```ignore
//! f_eq[i] = W[i] * rho * (1.0 + c_dot_u / cs2 + c_dot_u^2 / (2*cs4) - u_sq / (2*cs2))
//! ```ignore
//!
//! With cs2 = 1/3, the reciprocals become constants: inv_cs2 = 3.0, inv_2cs4 = 4.5,
//! half_inv_cs2 = 1.5. This allows an algebraic rearrangement into a Horner-like FMA
//! chain that the compiler cannot always discover on its own.
//!
//! Current form (separate mul+add, ~6 FP operations per direction):
//!
//! ```ignore
//! float term1 = c_dot_u * 3.0f;
//! float term2 = c_dot_u * c_dot_u * 4.5f;
//! float term3 = u_sq * 1.5f;
//! float bracket = 1.0f + term1 + term2 - term3;
//! f_eq[i] = W[i] * rho * bracket;
//! ```ignore
//!
//! FMA-optimized form (Horner-like, 3 FMA + 1 MUL):
//!
//! ```ignore
//! float base = fmaf(-u_sq, 1.5f, 1.0f);               // 1 FMA: 1.0 - u_sq * 1.5
//! float poly = fmaf(fmaf(c_dot_u, 4.5f, 3.0f),         // 2 nested FMA
//! >                 c_dot_u, base);
//! f_eq[i] = (W[i] * rho) * poly;                       // 1 MUL (w_rho precomputed)
//! ```ignore
//!
//! This collapses from ~6 FP operations to 3 FMA + 1 MUL. At 4.54 cy per FFMA, the
//! saving is approximately 2 FFMA cycles per direction per cell, which sums to 38
//! cycles per cell across all 19 lattice directions.
//!
//! YSU-engine reported a 3.16x MLP speedup via hand-scheduled FMA sequences in their
//! Vulkan compute shaders. Note: nvcc with --use_fast_math may already contract
//! multiply-add pairs to FMA instructions, but explicit fmaf() calls guarantee the
//! schedule regardless of optimization level and prevent the compiler from reordering
//! operations that would break the FMA chain. This is particularly important for
//! NVRTC runtime compilation where optimization flags may vary.
//!
//! ## Shared Memory Strategy for Lattice Neighbor Access
//!
//! D3Q19 streaming reads 19 neighbors per cell from global memory. Even with SoA
//! layout providing coalesced access, each cell still issues 19 separate global loads
//! during the streaming step. A shared memory tiling strategy can reduce this cost
//! significantly for kernels that access spatial neighbors.
//!
//! **Tiling approach:**
//!
//! A 3x3x3 tile in shared memory covers all D3Q19 neighbor directions (the D3Q19
//! velocity set spans at most +/-1 in each axis). The cooperative loading protocol
//! is:
//!
//! 1. Each thread loads its own cell data from global memory into shared memory.
//! 2. Boundary threads additionally load halo cells (1-cell border).
//! 3. Call __syncthreads() to ensure all loads complete.
//! 4. Each thread reads its 19 neighbors from shared memory instead of global.
//!
//! **Latency comparison:**
//!
//! - Shared memory read: ~28 cy per access
//! - Global memory read: ~92-200 cy per access (depending on cache behavior)
//! - Improvement: 3-7x per read operation
//!
//! **Caveat for our codebase:** The dark_halo kernel already uses SoA layout which
//! provides coalesced global reads. Shared memory tiling is most beneficial for the
//! collision kernel with force coupling, which reads rho, velocity, and force vectors
//! from neighbors for gradient computation (enstrophy kernel, MHD source terms).
//!
//! **SM 8.9 shared memory budget:**
//!
//! Ada Lovelace SM 8.9 has 100 KB of configurable L1/shared memory. With 48 KB
//! allocated to shared memory:
//!
//! - Capacity: 48K / (4 bytes * 19 directions) = ~631 cells per block
//! - At 128 threads/block: each thread has ~5 cells of shared memory buffer
//! - This is sufficient for a 3D tile with halo, provided block dimensions are
//! chosen to minimize the surface-to-volume ratio of the tile (e.g., 8x4x4
//! rather than 128x1x1)
//!
//! ## Occupancy Culling for Sparse DM Density Regions
//!
//! NFW dark matter density profiles concentrate mass at the galactic center and fall
//! off as 1/r at small radii and 1/r^3 at large radii. In our heliospheric simulation
//! domain (1-157 AU), approximately 60% of cells at outer radii (>50 AU) have DM
//! density below the threshold where drag forces are physically significant.
//!
//! **Early-exit strategy:**
//!
//! In the Guo forcing kernel, add an early-exit check before the 19-iteration forcing
//! loop:
//!
//! ```ignore
//! float force_mag = sqrtf(fx*fx + fy*fy + fz*fz);
//! if (force_mag < FORCE_EPSILON) return;  // skip Guo source term
//! ```ignore
//!
//! This avoids the 19-iteration Guo forcing loop for cells with negligible DM drag,
//! saving approximately 19 * (2 FMA + 1 MUL) = ~95 FP operations per skipped cell.
//!
//! **Analogy to YSU-engine:** YSU uses similar culling for empty voxels in volume
//! rendering -- transparent cells skip the shading pipeline entirely. The principle
//! is identical: avoid computation in regions where the result is below the noise
//! floor.
//!
//! **Warp divergence tradeoff:**
//!
//! Early-exit creates divergence within warps when some threads exit and others
//! continue. However, the net throughput gain exceeds the divergence penalty when
//! more than 50% of cells in a warp qualify for the early exit. At outer heliosphere
//! distances, the skip fraction typically exceeds 60%, making this a net win.
//!
//! Use __ballot_sync(0xFFFFFFFF, force_mag < FORCE_EPSILON) to count active lanes
//! within a warp. This enables adaptive behavior: if ALL 32 lanes would skip, the
//! entire warp can return early without any divergence penalty. Our dark_halo_detector
//! kernel already uses __ballot_sync + __popc for warp-level reduction (the same
//! pattern).
//!
//! **Expected speedup:** Approximately 2x for the Guo forcing kernel at outer
//! heliosphere distances (>50 AU), decreasing to ~1.1x near 1 AU where DM density
//! is more uniform above threshold.
//!
//! ## Polynomial Transcendental Approximation
//!
//! The MUFU special function unit on Ada Lovelace costs 17.56 cy for exp2 and ~17 cy
//! for sin, compared to 4.54 cy for FFMA -- a 3.9x latency ratio. Additionally, SFU
//! throughput is only 16 ops/cycle/SM versus 128 for FFMA, an 8x throughput gap.
//!
//! **Where transcendentals appear in our LBM kernels:**
//!
//! 1. Voudon frustration kernel: sinf() for phase modulation
//! 2. Zero-divisor viscosity modulation: sinf() for oscillatory tau adjustment
//! 3. Potential future DM cross-section: exp() for temperature-dependent sigma
//!
//! **D3Q19 equilibrium is already polynomial** -- no transcendentals needed. The FMA
//! chain optimization in Section 2 applies directly without any approximation error.
//!
//! **Polynomial replacement options:**
//!
//! - Degree-4 Chebyshev polynomial for exp(x) on [-2, 2]: ~1e-5 relative error,
//! cost = 4 FMA = ~18.2 cy (comparable to MUFU.EX2 at 17.56 cy -- marginal gain)
//! - Degree-5 minimax polynomial for sin(x): ~1e-6 accuracy, cost = 4 FMA + 1 MUL
//! = ~23 cy (worse than MUFU.SIN at ~17 cy)
//!
//! **Recommendation:** Only apply polynomial transcendental approximation when
//! profiling shows that transcendentals consume more than 10% of total kernel time.
//! For current D3Q19 kernels, the FMA chain optimization (Section 2) and shared
//! memory tiling (Section 3) give larger payoffs. The polynomial approach becomes
//! worthwhile only if a future kernel introduces transcendentals in the hot inner
//! loop (e.g., per-cell temperature-dependent cross sections evaluated every
//! timestep).
//!
//! ## CUDA Assembly Optimization Reference
//!
//! SASS (Streaming ASSembler) analysis provides ground-truth performance data that
//! complements high-level profiling. Two key tools:
//!
//! - `cuobjdump --dump-sass <file>.cubin` -- extracts SASS assembly from compiled
//! kernels
//! - `nvdisasm <file>.cubin` -- provides control flow graph visualization and
//! register allocation details
//!
//! **Register pressure analysis for D3Q19:**
//!
//! | Allocation               | Registers |
//! |---------------------------|-----------|
//! | f_local[19]               | 19        |
//! | u_local[3]                | 3         |
//! | rho, tau, inv_tau, etc.   | 5         |
//! | Loop temporaries          | ~8        |
//! | **Total (conservative)**  | **~35**   |
//!
//! **Occupancy calculation for SM 8.9:**
//!
//! - Register file: 65536 registers per SM, 32 threads per warp
//! - At 35 registers/thread: max 65536 / 35 = 1872 threads/SM = 58 warps
//! - SM 8.9 hardware cap: 48 warps per SM
//! - Conclusion: our D3Q19 kernels are NOT register-limited
//!
//! **Latency hiding requirements:**
//!
//! The FFMA pipeline depth of 4.54 cy with 1 issue per cycle means we need at least
//! 4-5 active warps per SM to keep the pipeline saturated. At our standard block size
//! of 128 threads (4 warps per block), we need 2+ resident blocks per SM for good
//! occupancy. This is easily achieved given the register budget above.
//!
//! **Hardware parameters (RTX 4070 Ti, Ada Lovelace):**
//!
//! - 128 CUDA cores per SM, 60 SMs = 7680 cores total
//! - Memory bandwidth: 504 GB/s
//! - At 128 threads/block (our D3Q19 standard): 4 warps/block
//!
//! **Bandwidth analysis for D3Q19 at 128^3 grid:**
//!
//! - Data per full sweep: 19 directions * 128^3 cells * 4 bytes = ~160 MB
//! - Read + write = ~320 MB per timestep
//! - At 504 GB/s: theoretical max ~1575 timesteps/s
//! - Practical throughput (accounting for overhead): ~800-1000 timesteps/s
//! - Bandwidth-limited regime begins above ~3000 steps/s (smaller grids)
//!
//! **Key insight:** FMA optimization reduces compute cost but does not change memory
//! bandwidth requirements. For 128^3 and larger grids, our kernels are bandwidth-bound.
//! Shared memory tiling reduces effective bandwidth pressure for stencil patterns by
//! converting redundant global reads (where multiple threads read the same neighbor
//! cell) into a single global read followed by shared memory broadcast.
//!
