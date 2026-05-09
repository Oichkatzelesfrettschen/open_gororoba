# Unsafe-block SAFETY-coverage methodology (DEBT-UNSAFE-1 reframe)

This document refreshes the original DEBT-UNSAFE-1 task scope using the
anchored repo-audit numbers and proposes a risk-weighted methodology for
adding SAFETY comments. The Stage A 2026-04-30 baseline reported "442
unsafe blocks across 54 files; 16.5% SAFETY coverage" as a flat metric;
the anchored audit reveals a more nuanced picture.

## Anchored counts (debt-baseline-v1, 2026-05-09)

| Tree            | Rust files | unsafe blocks | SAFETY comments |
|-----------------|-----------:|--------------:|----------------:|
| crates/         |       2131 |           392 |              65 |
| proofs/         |        130 |          3420 |               0 |

The proofs/ tree's 3420 unsafe blocks are in scaffolding (test harnesses,
verifiers, ad-hoc reproducers). They are categorically different from the
crates/ unsafe -- they are not part of the gate-tested API surface.

Stage A's "16.5%" is an aggregate over all 54 files in crates/ that contain
any unsafe. This averages high-coverage files (UnsafeAoSoAPtr in
lbm_3d/solver.rs is 100% covered) with no-coverage files. The aggregate
is misleading.

## Top crates by unsafe-block density (crates/ only)

| File                                          | unsafe blocks | risk band  |
|-----------------------------------------------|--------------:|------------|
| crates/lbm_vulkan/src/compute.rs               |            59 | high       |
| crates/lbm_vulkan/src/besag_clifford_vulkan.rs |            52 | high       |
| crates/lbm_3d_cuda/src/lib.rs                  |            49 | high       |
| crates/cd_kernel/src/avx2_primitives.rs        |            26 | medium     |
| crates/lbm_3d_cuda/src/bench_kernels.rs        |            18 | medium     |
| crates/grmhd_core/src/flux.rs                  |            18 | medium     |
| crates/lbm_vulkan/src/alignment_vulkan.rs      |            17 | medium     |
| crates/cd_kernel/src/x87_ext80.rs              |            17 | medium     |
| crates/lbm_vulkan/src/lib.rs                   |            10 | medium     |
| crates/sign_imbalance/src/kubo_transport_gpu.rs |            8 | low-medium |
| crates/gororoba_cli_physics/src/bin/kerr_pathion_gpu.rs |     8 | low-medium |
| crates/cd_kernel/src/x87_primitives.rs         |             8 | low-medium |
| crates/lbm_3d/src/solver.rs                    |             7 | (verified) |
| crates/lbm_3d_cuda/src/sparse/mod.rs           |             7 | low-medium |
| crates/gororoba_cli_physics/src/bin/dark_halo_hunt.rs   |     7 | low-medium |

The top 4 files (lbm_vulkan/compute, lbm_vulkan/besag_clifford_vulkan,
lbm_3d_cuda/lib, cd_kernel/avx2_primitives) account for 186 of the 392
unsafe blocks (47%). Adding SAFETY comments there is the highest-leverage
work.

## Risk-weighted classification

Each unsafe block falls into one of five risk bands. The band determines
whether SAFETY annotation is **required** vs **recommended**.

### Band: VK-FFI (high)

Vulkan dispatches via `ash` raw FFI. Wrong invariants here can cause GPU
hangs, host segfaults, or silent corruption. SAFETY comments are
**required**: every block must explain the synchronization, lifetime,
and exclusivity argument for the resource handles being passed.

Files: lbm_vulkan/{compute,besag_clifford_vulkan,alignment_vulkan,lib}.rs.

### Band: CUDA-FFI (high)

CUDA dispatches via `cudarc::driver` raw FFI. Same risk profile as
VK-FFI. SAFETY required.

Files: lbm_3d_cuda/{lib,bench_kernels,sparse/mod}.rs,
sign_imbalance/kubo_transport_gpu.rs,
gororoba_cli_physics/{kerr_pathion_gpu,dark_halo_hunt}.rs.

### Band: SIMD-Intrinsics (medium)

`std::arch::x86_64` AVX2 / SSE2 / FMA intrinsics. Memory layout and
alignment are local invariants; mistakes typically cause panics or
latent precision loss but not host RCE. SAFETY recommended; one-line
form acceptable: `// SAFETY: ptr is 32-byte aligned and points to 8
contiguous f32`.

Files: cd_kernel/avx2_primitives.rs, cd_kernel/x87_*.rs.

### Band: AoSoA-Pointer (medium-low)

The `UnsafeAoSoAPtr` pattern from lbm_3d/solver.rs. Already 100%
SAFETY-annotated; serves as the canonical example. Other files using
similar manual aliasing should mirror its docstring style.

Files: lbm_3d/solver.rs (verified), grmhd_core/flux.rs (audit pending).

### Band: Lifetime-Pun (low)

`std::mem::transmute` between `&[u8]` and `&[T]` for binary parsing or
zero-copy casts. SAFETY recommended; the comment should cite the
alignment and size invariants and the source of the bytes.

Files: cd_kernel/turboquant/cuda/launch.rs, data_core/spice/daf.rs.

## Recommended top-10 SAFETY targets

In risk-weighted order (file, approximate block count, band):

1. lbm_vulkan/compute.rs (59, VK-FFI) -- start here; biggest impact.
2. lbm_vulkan/besag_clifford_vulkan.rs (52, VK-FFI).
3. lbm_3d_cuda/lib.rs (49, CUDA-FFI).
4. cd_kernel/avx2_primitives.rs (26, SIMD-Intrinsics).
5. lbm_3d_cuda/bench_kernels.rs (18, CUDA-FFI).
6. grmhd_core/flux.rs (18, AoSoA-Pointer; mirror lbm_3d/solver.rs style).
7. lbm_vulkan/alignment_vulkan.rs (17, VK-FFI).
8. cd_kernel/x87_ext80.rs (17, SIMD-Intrinsics).
9. lbm_vulkan/lib.rs (10, VK-FFI).
10. sign_imbalance/kubo_transport_gpu.rs (8, CUDA-FFI).

Closing the top 10 raises crates/ SAFETY coverage from
65 / 392 = 16.6% to roughly (65 + 274) / 392 = 86.5%.

## Acceptance criteria

DEBT-UNSAFE-1 closes when:

- `make repo-audit-strict` reports `safety_comments` >= 350 in crates/
  and the unsafe-block count has not grown.
- Every unsafe block in the high-risk bands (VK-FFI, CUDA-FFI) has a
  SAFETY comment whose first line names the invariant.
- The proofs/ unsafe-block count is documented as a separate concern
  (DEBT-UNSAFE-PROOFS-SCAFFOLDING) and not included in the closure
  criterion.

## See also

- `data/output/debt_baseline_2026_05_09.toml` (anchored counts).
- `crates/gororoba_cli_data/src/bin/repo_audit.rs` (audit binary).
- `crates/lbm_3d/src/solver.rs` UnsafeAoSoAPtr (canonical SAFETY style).
- Stage B B-G2-PRECON (now obsolete; Phase A.3 generator change covers it).
