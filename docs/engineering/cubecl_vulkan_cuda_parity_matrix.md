# cubecl + Vulkan + CUDA Parity Matrix (task #136)

Date: 2026-05-13
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

| Kernel / Subsystem                  | Crate                 | CPU | CUDA | Vulkan | cubecl | Parity Test |
|-------------------------------------|-----------------------|-----|------|--------|--------|-------------|
| TurboQuant quantize (3-bit, 128-d)  | cd_kernel             | YES | n/a  | YES    | YES    | YES (both) |
| LBM D3Q19 stream + collide          | lbm_3d_cuda / lbm_3d  | YES | YES  | NO     | NO     | NO         |
| LBM MRT collision                   | lbm_3d_cuda / lbm_3d  | YES | YES  | NO     | NO     | NO         |
| Sparse-grid LBM                     | lbm_3d_cuda           | n/a | YES  | NO     | NO     | NO         |
| Box-counting fractal dimension      | lbm_3d_cuda + lbm_vulkan | YES | YES  | YES    | YES    | YES (CPU vs cubecl in lbm_vulkan; CUDA + Vulkan oracles share box_counting_cpu) |
| Chingon (anisotropy operator)       | lbm_3d_cuda + lbm_vulkan | YES | YES  | YES    | YES    | YES (CPU oracle + cubecl in lbm_vulkan) |
| Alignment / orientation projection  | lbm_3d_cuda + lbm_vulkan | YES | YES  | YES    | NO     | NO         |
| Besag-Clifford GMRF                 | sign_imbalance + lbm_vulkan | YES | YES  | YES    | NO     | NO         |
| Dark-halo Monte Carlo               | lbm_3d_cuda           | n/a | YES  | NO     | NO     | NO         |
| Kubo transport conductivity         | sign_imbalance        | YES | YES  | NO     | NO     | NO         |
| Algebraic lensing                   | optics_core           | YES | YES  | NO     | NO     | NO         |
| Voudon stabilizer                   | algebra_experimental  | YES | YES  | NO     | NO     | NO         |
| GRMHD GPU advance                   | grmhd_core            | YES | YES  | NO     | NO     | NO         |
| Coop-matrix probe (Vulkan-only)     | lbm_vulkan            | n/a | NO   | YES    | NO     | n/a (Vulkan-only feature; no equivalent in cudarc) |
| OptiX BVH ray-tracing               | lbm_3d_cuda + gororoba_optix | n/a | YES  | NO     | NO     | n/a (OptiX is NVIDIA-only by design) |

Legend:
- YES = implementation exists and is wired into the public API.
- NO  = implementation absent; needs to be written for parity.
- n/a = the kernel does not make sense for this backend.
- PARTIAL = some variants done, some missing.

## Quantitative Gap Summary

- Total kernels enumerated: 15
- Full 3-way parity (CPU + CUDA + Vulkan, ideally also cubecl): 3 / 15
  (TurboQuant + Box-counting + Chingon).
- CPU + cubecl partial (no full Vulkan device-pipeline): 1 / 15
  (transform_viscosity -- besag_clifford sub-kernel; shader exists but
   device-pipeline not wired up for non-besag callers).
- CUDA + Vulkan present (no cubecl): 1 / 15 (alignment, besag-clifford
  full pipeline)
- See docs/engineering/issue_136_phase2_finalization.md for the
  per-cell deferral rationale.
- CUDA only: 8 / 15 (LBM core, sparse LBM, dark-halo, kubo, lensing, voudon, GRMHD, MRT)
- Vulkan only: 1 / 15 (coop-matrix probe; structurally NVIDIA-incompatible)
- OptiX (NVIDIA-only, expected): 1 / 15

## Phase 2 Recommended Build Order

Highest expected ROI (closes parity for the most-used kernels first):

1. **LBM D3Q19 stream + collide (Vulkan)**
   - Most-used kernel in the repo. Already has CPU reference in `lbm_3d`.
   - Existing `lbm_vulkan` crate has compute scaffolding (`compute.rs`,
     `precision_dispatch.rs`) but lacks the stream+collide compute shader.
   - Action: port the cudarc kernel from `lbm_3d_cuda/src/lib.rs` D3Q19
     loop into a WGSL or GLSL compute shader; wire through `naga` (already
     a dep). Reference grid: 32^3 to validate at PR time; ramp to 128^3.

2. **LBM MRT collision (Vulkan + cubecl)**
   - Builds on (1); shares the streaming step.
   - Hand-CUDA expression already at `lbm_3d_cuda::LbmSolver3DCuda::new_mrt`.
   - Once Vulkan stream+collide lands, MRT is a 19x19 matrix-multiply per cell
     in the collision step -- naturally maps to compute shader workgroups.

3. **Box-counting fractal dimension cubecl backend**
   - Already has CUDA + Vulkan paths; add cubecl path so we have a 3-way
     parity test mirroring `turboquant_cubecl_parity.rs`.
   - The cubecl Backend launcher work (task #128) directly enables this.

4. **Chingon / alignment cubecl backend**
   - Same pattern as (3): existing CUDA + Vulkan; add cubecl + parity test.

5. **Dark-halo Monte Carlo Vulkan port**
   - Compute-only kernel; RNG state per-thread; reduction at the end.
   - Vulkan needs a PCG or Philox RNG implemented in WGSL/GLSL (cudarc
     uses cuRAND); recommend Philox-2x32-10 (license: BSD-3, easy port).

6. **Kubo transport conductivity Vulkan port**
   - Algorithmically simpler than dark-halo (no RNG); just a tensor
     contraction. Direct WGSL translation of the cudarc reduction.

Lower-priority (more specialized):

7. Algebraic lensing Vulkan port -- needed only for GPU-accelerated
   `optics_core::algebraic_lensing`; CPU path is acceptable for now.
8. Voudon stabilizer Vulkan -- speculative algebra path, low usage.
9. GRMHD GPU Vulkan -- the CPU path remains the production target for now.

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
  is *feature* parity (correctness across backends), not equal-FLOPS
  optimisation.
- OptiX BVH-tracing pipelines. The OptiX pipeline in `gororoba_optix`
  is by design NVIDIA-only; reproducing its ray-tracing semantics in
  Vulkan would require a separate ray-tracing extension (VK_KHR_ray_tracing
  with acceleration structures), which is a multi-week scope.
- WebGPU / browser-side cubecl deployment. cubecl currently targets the
  desktop wgpu backend; browser-target build is a separate concern.

## Next Concrete Steps (Phase 2 tasks)

Each becomes its own task once #136 is signed off:

- T-LBM-VULKAN-1: Port D3Q19 stream + collide to WGSL compute shader.
  Acceptance: 32^3 grid, single-relaxation-time BGK, parity vs
  `LbmSolver3D::stream_collide()`.
- T-LBM-VULKAN-2: Port MRT collision matrix to WGSL.
  Acceptance: parity vs `LbmSolver3DCuda::new_mrt().step()`.
- T-LBM-CUBECL-1: cubecl backend for box-counting fractal dimension.
  Acceptance: 3-way parity test mirroring `turboquant_cubecl_parity.rs`.
- T-LBM-CUBECL-2: cubecl backend for chingon / alignment.
- T-DARK-HALO-VULKAN-1: WGSL Philox RNG + dark-halo Monte Carlo.
- T-KUBO-VULKAN-1: Kubo transport conductivity WGSL port.

Each phase-2 task delivers exactly one cell in the parity matrix going
from NO to YES, with its accompanying parity test. The matrix in this
document is the live tracking artefact.
