# cubecl + Vulkan + CUDA Parity Matrix (task #136)

Date: 2026-05-17
Issue: User asked for "cubecl+vulkan items in sync for CUDA+Vulkan at full-feature-parity".
Scope: enumerate every GPU compute kernel in the workspace and identify which
backends (CPU reference, CUDA via cudarc, Vulkan via ash, cubecl) are present
vs missing.

WHY: Today the workspace has three GPU paths -- raw cudarc, raw ash/Vulkan,
and cubecl (wgpu-backed). Many kernels exist in only one path. The user
wants Vulkan and CUDA to reach feature parity so that a host without an
NVIDIA driver still has the full numerical experiment-running capability,
and so that cubecl provides a portable third lane.

WHAT: This document is the source of truth for which kernels exist where
and which gaps need filling. It will be turned into a parity-test
dispatch table in `crates/gpu_parity_harness/` (proposed; not yet
created) that runs CPU vs each available backend across the matrix.

HOW: Phase 1 scope. Phase 2 (subsequent commits) creates a cubecl /
Vulkan implementation per missing cell; Phase 3 wires parity tests
following the `turboquant_cubecl_parity.rs` template already in place
for `cd_kernel`.

## Existing Parity Template (cd_kernel)

`crates/cd_kernel/tests/turboquant_cubecl_parity.rs` and
`turboquant_vulkan_parity.rs` already verify the 3-way invariant
(CPU == Vulkan == cubecl, bit-identical u8 output) for the TurboQuant
quantizer. This is the gold standard pattern: gated `#[ignore = "gpu"]`,
runtime adapter probe, ChaCha20-seeded inputs, byte-exact comparison.

## Parity Matrix

| Kernel / Subsystem                 | Crate                             | CPU | CUDA | Vulkan | cubecl | Parity Test                                                                                                                              |
| ---------------------------------- | --------------------------------- | --- | ---- | ------ | ------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| TurboQuant quantize (3-bit, 128-d) | cd_kernel                         | YES | n/a  | YES    | YES    | YES (both)                                                                                                                               |
| LBM D3Q19 stream + collide         | lbm_vulkan / lbm_3d               | YES | YES  | YES    | YES    | YES (CPU-vs-Vulkan, CPU-vs-cubecl, 3-way in lbm_d3q19_parity.rs)                                                                         |
| LBM MRT collision                  | lbm_vulkan / lbm_3d_cuda / lbm_3d | YES | YES  | YES    | YES    | YES (CPU-vs-Vulkan in lbm_mrt_d3q19_vulkan_parity.rs, CPU-vs-cubecl in lbm_mrt_d3q19_cubecl_parity.rs, 3-way in lbm_mrt_d3q19_parity.rs) |
| Sparse-grid LBM direct active bricks | lbm_3d_cuda + lbm_vulkan         | YES | YES  | YES    | YES    | YES (CPU-vs-Vulkan in sparse_lbm_vulkan_parity.rs, CPU-vs-cubecl in sparse_lbm_cubecl_parity.rs)                                        |
| Box-counting fractal dimension     | lbm_3d_cuda + lbm_vulkan          | YES | YES  | YES    | YES    | YES (CPU vs cubecl in lbm_vulkan; CUDA + Vulkan oracles share box_counting_cpu)                                                          |
| Chingon (anisotropy operator)      | lbm_3d_cuda + lbm_vulkan          | YES | YES  | YES    | YES    | YES (CPU oracle + cubecl in lbm_vulkan)                                                                                                  |
| Alignment / orientation projection | lbm_3d_cuda + lbm_vulkan          | YES | YES  | YES    | YES    | YES (CPU-vs-cubecl in lbm_vulkan/tests/alignment_cubecl_parity.rs)                                                                       |
| Besag-Clifford GMRF                | sign_imbalance + lbm_vulkan       | YES | YES  | YES    | YES    | YES (CPU-vs-cubecl in besag_clifford_cubecl_parity.rs: exact PCG shuffle + 1e-5 transform)                                               |
| Dark-halo ZD viscosity/count       | lbm_3d_cuda + lbm_vulkan          | YES | YES  | YES    | YES    | YES (CPU-ZD-oracle vs Vulkan in dark_halo_vulkan_parity.rs, CPU-vs-cubecl in dark_halo_cubecl_parity.rs)                                 |
| Kubo transport conductivity        | sign_imbalance                    | YES | YES  | n/a    | n/a    | NO (cuSOLVER/cuBLAS; not a portable compute shader)                                                                                      |
| Algebraic lensing                  | optics_core                       | YES | YES  | YES    | YES    | YES (CPU-vs-Vulkan in algebraic_lensing_gpu.rs; CPU-vs-cubecl in algebraic_lensing_cubecl_parity.rs)                                    |
| Voudon stabilizer                  | algebra_experimental              | YES | YES  | YES    | YES    | YES (CPU-vs-Vulkan and CPU-vs-cubecl row counts in voudon_vulkan_parity.rs and voudon_cubecl_parity.rs)                                  |
| GRMHD GPU advance                  | grmhd_core                        | YES | YES  | YES    | YES    | YES (CPU-vs-Vulkan and CPU-vs-cubecl in grmhd_core parity tests)                                                                         |
| Tensor AVT dense multiply/norm     | gororoba_algebra                  | YES | YES  | YES    | YES    | YES (CPU-vs-Vulkan and CPU-vs-cubecl in tensor_avt_vulkan_parity.rs and tensor_avt_cubecl_parity.rs)                                     |
| Coop-matrix probe (Vulkan-only)    | lbm_vulkan                        | n/a | NO   | YES    | NO     | n/a (Vulkan-only feature; no equivalent in cudarc)                                                                                       |
| OptiX BVH ray-tracing              | lbm_3d_cuda + gororoba_optix      | n/a | YES  | NO     | NO     | n/a (OptiX is NVIDIA-only by design)                                                                                                     |

Legend:

- YES = implementation exists and is wired into the public API.
- NO = implementation absent; needs to be written for parity.
- n/a = the kernel does not make sense for this backend.

## Quantitative Gap Summary

- Total kernels enumerated: 16
- CPU + Vulkan + cubecl parity rows: TurboQuant, LBM D3Q19, LBM MRT,
  sparse-grid LBM direct active bricks, box-counting, Chingon, alignment,
  Besag-Clifford, dark-halo, algebraic lensing, Voudon stabilizer, GRMHD,
  and Tensor AVT.
- CUDA + Vulkan present without cubecl: none in the parity rows.
- See docs/engineering/issue_136_phase2_finalization.md for the
  per-cell deferral rationale.
- CUDA-only custom-compute row: none in the parity rows.
  Kubo is n/a for Vulkan/cubecl (uses cuSOLVER symmetric eigensolver + cuBLAS DGEMM,
  not portable compute shaders; implementing from scratch is out of scope for #136).
- Vulkan only: 1 / 16 (coop-matrix probe; structurally NVIDIA-incompatible)
- OptiX (NVIDIA-only, expected): 1 / 16

## Phase 2 Recommended Build Order

Highest expected ROI (closes parity for the most-used kernels first):

1. **LBM D3Q19 stream + collide (Vulkan)**
   - Most-used kernel in the repo. Already has CPU reference in `lbm_3d`.
   - Existing `lbm_vulkan` crate has compute scaffolding (`compute.rs`,
     `precision_dispatch.rs`) but lacks the stream+collide compute shader.
   - Action: port the cudarc kernel from `lbm_3d_cuda/src/lib.rs` D3Q19
     loop into a WGSL or GLSL compute shader; wire through `naga` (already
     a dep). Reference grid: 32^3 to validate at PR time; ramp to 128^3.

2. **LBM MRT collision (Vulkan + cubecl)** -- COMPLETE (Wave G1)
   - `lbm_vulkan/shaders/lbm_mrt_d3q19.wgsl`: d'Humieres D3Q19 MRT WGSL shader.
   - `lbm_vulkan/src/lbm_mrt_d3q19_vulkan.rs`: Vulkan solver using gpu_vulkan helpers.
   - `lbm_vulkan/src/lbm_mrt_d3q19_cubecl.rs`: cubecl #[cube] MRT kernel.
   - Parity tests: CPU-vs-Vulkan, CPU-vs-cubecl, 3-way (all gated #[ignore = "gpu"]).
   - abs_tol=2e-3 / rel_tol=2e-2 (wider than BGK due to higher MRT FMA count).

3. **Box-counting fractal dimension cubecl backend**
   - Already has CUDA + Vulkan paths; add cubecl path so we have a 3-way
     parity test mirroring `turboquant_cubecl_parity.rs`.
   - The cubecl Backend launcher work (task #128) directly enables this.

4. **Chingon / alignment cubecl backend**
   - Same pattern as (3): existing CUDA + Vulkan; add cubecl + parity test.

5. **Dark-halo ZD Vulkan port -- COMPLETE**
   - Three-pipeline Vulkan backend in `dark_halo_vulkan.rs`:
     (1) deterministic ZD viscosity (dark_halo_viscosity.wgsl, Murmur-inspired hash),
     (2) D3Q19 BGK PUSH with per-cell tau (dark_halo_lbm_step.wgsl),
     (3) 3-criterion cell classifier (dark_halo_detector.wgsl).
   - Parity tests at `tests/dark_halo_vulkan_parity.rs` (deterministic_zd_count,
     all_pass_all_fail).
   - Note: the CUDA path uses cuRAND "Monte Carlo" label but the ZD viscosity field
     is deterministic (spatial hash, no true RNG). The Vulkan port implements the
     deterministic variant.

6. **Kubo transport conductivity -- n/a for Vulkan/cubecl**
   - The CUDA path calls cuSOLVER (symmetric eigendecomposition) and cuBLAS (DGEMM).
     These are vendor-library calls, not custom compute shaders. Porting to Vulkan
     would require implementing full Jacobi or QR iteration from scratch -- out of
     scope for #136. Marked n/a in the parity matrix.

Lower-priority (more specialized):

7. No portable custom-compute rows remain below the direct active-brick sparse
   LBM surface. CUDA-only sparse residency variants, OptiX ray tracing, and
   vendor BLAS/eigensolver calls remain separate API or memory-policy surfaces.

## Implementation Notes

### cubecl Backend Launcher (task #128)

The work for task #128 is the foundation: `cd_kernel::turboquant::cubecl_backend::launcher`
already exists and is wired into the 3-way parity test. The same `launcher`
pattern (probe `is_available()` at runtime, fall through to CPU otherwise)
should be extracted into a top-level `gpu_parity_harness` crate so the
LBM / dark-halo / kubo paths can reuse it without re-implementing
adapter discovery.

### Vulkan Compute-Shader Toolchain

The `lbm_vulkan` crate already pulls in `naga` (workspace dep) with
`wgsl-in` and `spv-out` features. The path is:

1. Author WGSL compute shader.
2. `naga` transpiles WGSL -> SPIR-V at build time (or runtime via
   `naga::front::wgsl::parse_str`).
3. `ash` consumes the SPIR-V via `vk::ShaderModuleCreateInfo`.

WGSL is preferred over GLSL for new kernels because (a) it has stricter
typing (catches bugs at parse time), (b) the same source can be reused
by cubecl-wgpu without modification, and (c) WGSL has clearer rules
around shared memory and barriers.

### CPU Reference Pattern

Every new kernel must ship a CPU reference path in pure Rust (no GPU
deps) which is the parity oracle. Existing precedent: `cd_kernel`'s
`turboquant::backend::Backend::Cpu(SimdLevel::Scalar)` and `LbmSolver3D`
(no `Cuda` suffix) in `lbm_3d` are both reference implementations that
the GPU paths are compared against.

## Acceptance Criteria for "full feature parity"

For each kernel in the matrix, parity is achieved when:

1. CPU reference path compiles without any GPU feature flag.
2. CUDA path (where applicable) produces output matching CPU within
   tolerance (bit-exact for integer kernels; relative error < 1e-5 for
   single-precision float; < 1e-9 for double-precision float).
3. Vulkan path produces output matching CPU within the same tolerance.
4. cubecl path produces output matching CPU within the same tolerance.
5. A `crates/<crate>/tests/<kernel>_<backend>_parity.rs` test exists for
   each backend, gated `#[ignore = "gpu"]`, that probes adapter
   availability at runtime and skips cleanly if absent.
6. Test inputs are seeded (ChaCha20Rng with a documented seed) for
   reproducibility.

## Out of Scope for #136

- Performance parity (CUDA vs Vulkan absolute throughput). The objective
  is _feature_ parity (correctness across backends), not equal-FLOPS
  optimisation.
- OptiX BVH-tracing pipelines. The OptiX pipeline in `gororoba_optix`
  is by design NVIDIA-only; reproducing its ray-tracing semantics in
  Vulkan would require a separate ray-tracing extension (VK_KHR_ray_tracing
  with acceleration structures), which is a multi-week scope.
- WebGPU / browser-side cubecl deployment. cubecl currently targets the
  desktop wgpu backend; browser-target build is a separate concern.

## Next Concrete Steps (Phase 2 tasks)

Each becomes its own task once #136 is signed off:

- T-LBM-VULKAN-1: COMPLETE (PR #51, Wave D). D3Q19 WGSL PUSH BGK shipped
  in `crates/lbm_vulkan/shaders/lbm_d3q19.wgsl` + `lbm_d3q19_vulkan.rs`.
  Parity test at `tests/lbm_d3q19_vulkan_parity.rs` (CPU vs Vulkan).
  3-way parity test at `tests/lbm_d3q19_parity.rs` (PR #54, Wave F).
- T-LBM-VULKAN-2: Port MRT collision matrix to WGSL.
  Acceptance: parity vs `LbmSolver3DCuda::new_mrt().step()`.
- T-LBM-CUBECL-1: cubecl backend for box-counting fractal dimension.
  Acceptance: 3-way parity test mirroring `turboquant_cubecl_parity.rs`.
- T-LBM-CUBECL-2: COMPLETE. cubecl backend for alignment (box-kite orientation scan).
  `lbm_vulkan/src/alignment_cubecl.rs` (per-(v,o)-pair kernel, immutable-only IR).
  Parity test at `tests/alignment_cubecl_parity.rs` (CPU-vs-cubecl, 64 vectors).
- T-DARK-HALO-VULKAN-1: COMPLETE. Three-pipeline dark-halo Vulkan backend:
  viscosity (dark_halo_viscosity.wgsl), LBM step with per-cell tau
  (dark_halo_lbm_step.wgsl), classifier (dark_halo_detector.wgsl).
  Parity tests in `tests/dark_halo_vulkan_parity.rs`.
- T-KUBO-VULKAN-1: n/a. Kubo uses cuSOLVER + cuBLAS; not a portable compute shader.

Each phase-2 task delivers exactly one cell in the parity matrix going
from NO to YES, with its accompanying parity test. The matrix in this
document is the live tracking artefact.
