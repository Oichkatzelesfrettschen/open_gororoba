# GPU Compute Roadmap -- open_gororoba

This document describes the current GPU backend coverage across the workspace,
the recommended build order for closing parity gaps, and the strategy for
dimensional-kernel specialization. It is the companion to
`docs/engineering/cubecl_vulkan_cuda_parity_matrix.md` (the per-kernel audit)
and the `gororoba_gpu_*` foundation crates.

## Quick orientation

The workspace runs three portable GPU paths:

- **Vulkan** via `ash` + `naga` (WGSL shaders compiled to SPIR-V at runtime).
  Foundation crate: `gororoba_gpu_vulkan`.
- **CUDA** via `cudarc` 0.19 (NVRTC runtime PTX compilation).
  Foundation crate: `gororoba_gpu_cuda`.
- **cubecl** via `cubecl-wgpu` 0.10 (`#[cube]` kernels compiled to WGSL).
  Foundation crate: `gororoba_gpu_cubecl`.

CPU reference paths live in pure-Rust crates (no GPU feature flags). Each GPU
path is gated by an optional feature (`ash`, `cudarc`, `cubecl`) so the
workspace builds on CPU-only hosts without SDK dependencies.

All three foundation crates re-export the canonical types from
`gororoba_gpu_bridge` (`ComputeBackend`, `StoragePrecision`, `HardwareCaps`).

## Extended parity matrix (all kernels)

Rows 1-15 are the original scope from `cubecl_vulkan_cuda_parity_matrix.md`;
rows 16-23 cover the additional algebra / physics kernels discovered in the
full audit.

| #  | Kernel / Subsystem                 | Crate(s)                     | Dims     | CPU | CUDA    | Vulkan | cubecl | Parity test           |
| -- | ---------------------------------- | ---------------------------- | -------- | --- | ------- | ------ | ------ | --------------------- |
| 1  | TurboQuant quantize (3-bit, 128-d) | cd_kernel                    | 128      | YES | YES     | YES    | YES    | YES (VK/cubecl; CUDA contract) |
| 2  | LBM D3Q19 stream + collide (BGK)   | lbm_vulkan / lbm_3d          | 3D grid  | YES | YES     | YES    | YES    | YES (3-way)           |
| 3  | LBM MRT collision                  | lbm_vulkan / lbm_3d_cuda     | 3D grid  | YES | YES     | YES    | YES    | YES (3-way)           |
| 4  | Sparse-grid LBM (direct bricks)    | lbm_3d_cuda + lbm_vulkan     | 8^3+     | YES | YES     | YES    | YES    | YES (CPU/VK/cubecl)   |
| 5  | Box-counting fractal dimension     | lbm_3d_cuda + lbm_vulkan     | 3D grid  | YES | YES     | YES    | YES    | YES                   |
| 6  | Chingon (anisotropy operator)      | lbm_3d_cuda + lbm_vulkan     | 3D grid  | YES | YES     | YES    | YES    | YES                   |
| 7  | Alignment / orientation projection | lbm_3d_cuda + lbm_vulkan     | 3D grid  | YES | YES     | YES    | YES    | YES                   |
| 8  | Besag-Clifford GMRF shuffle+xform  | sign_imbalance + lbm_vulkan  | n_cells  | YES | YES     | YES    | YES    | YES                   |
| 9  | Dark-halo ZD viscosity/count       | lbm_3d_cuda + lbm_vulkan     | 3D grid  | YES | YES     | YES    | YES    | YES                   |
| 10 | Kubo transport conductivity        | sign_imbalance               | NxN      | YES | YES     | n/a    | n/a    | NO                    |
| 11 | Algebraic lensing                  | optics_core                  | 3D rays  | YES | YES     | YES    | YES    | YES (CPU/VK/cubecl)   |
| 12 | Voudon stabilizer                  | algebra_experimental         | 256D     | YES | YES     | YES    | YES    | YES (CPU/VK/cubecl)   |
| 13 | GRMHD GPU advance                  | grmhd_core                   | 3D grid  | YES | YES     | YES    | YES    | YES (CPU/VK/cubecl)   |
| 14 | Coop-matrix probe (Vulkan-only)    | lbm_vulkan                   | tile     | n/a | NO      | YES    | NO     | n/a                   |
| 15 | OptiX BVH ray-tracing              | lbm_3d_cuda + gororoba_optix | scene    | n/a | YES     | NO     | NO     | n/a                   |
| 16 | Eta matrix (ZD analysis)           | gororoba_algebra             | 8..1024D | YES | YES     | YES    | YES    | YES (3-way)           |
| 17 | ZD imbalance ratio                 | gororoba_algebra             | 8..256D  | YES | YES     | YES    | YES    | YES (3-way)           |
| 18 | Voudon kernel (algebra)            | gororoba_algebra             | 256D     | YES | YES     | YES    | YES    | YES (3-way)           |
| 19 | APT census / dimensional scan      | gororoba_algebra             | 8..256D  | YES | YES     | YES    | YES    | YES (3-way)           |
| 20 | ZD graph construction              | gororoba_algebra             | 8..256D  | YES | YES     | YES    | YES    | YES (3-way)           |
| 21 | Tensor AVT                         | gororoba_algebra             | 16/32D   | YES | YES     | YES    | YES    | YES (CPU/VK/cubecl)   |
| 22 | PEPS row contraction               | quantum_core                 | n_sites  | YES | YES     | YES    | YES    | YES (CPU/VK/cubecl)   |
| 23 | Ultrametric minimax graph          | stats_core                   | 64 nodes | YES | YES     | YES    | YES    | YES (CPU/VK/cubecl)   |

Legend: YES = wired and tested. NO = absent. STUB = module exists, returns
"not implemented". n/a = backend does not apply to this kernel.

Row 1 marks CUDA as wired because `Backend::Cuda` dispatches quantize through
`TurboQuantCudaKernels::quantize_batch`. The direct quantize parity tests cover
CPU == Vulkan and CPU == cubecl; CUDA coverage is the runtime-guarded fused
dequant-dot contract smoke.

## What NOT to port

These CUDA paths have no portable Vulkan/cubecl equivalent by design:

- **Kubo transport** (`sign_imbalance`): calls cuSOLVER `syevd` (symmetric
  eigendecomposition) and cuBLAS `dgemm`. Implementing these from scratch
  would require a full iterative eigensolver in WGSL -- multi-week scope
  with no clear correctness oracle. Remain CUDA-only.
- **OptiX BVH ray-tracing** (`gororoba_optix`): NVIDIA proprietary API.
  Vulkan ray-tracing requires `VK_KHR_ray_tracing_pipeline` + acceleration
  structures, a separate multi-week effort.
- **cuFFT in `lbm_3d_cuda`**: The streaming FFT path uses cuFFT for spectral
  forcing. `vk_fft` covers the Vulkan equivalent but is not yet a workspace
  dependency. Defer until there is a second consumer.
- **Sparse-grid LBM CUDA residency variants** (`lbm_3d_cuda`): the portable
  direct active-brick D3Q19 stepping surface is implemented in `lbm_vulkan`.
  CUDA managed-memory oversubscription, tile prefetch, shared-halo staging,
  and peer-to-peer inter-GPU brick transfer remain CUDA memory-policy and
  kernel-variant choices. Port those only when a Vulkan external-memory or
  buffer-device-address design is required by a concrete caller.

## Dimensional-kernel strategy

Several kernels (eta_matrix, imbalance, APT scan, graph_construction) run over
Cayley-Dickson algebras at dim = 8, 16, 32, 64, 128, 256. The CPU and CUDA
paths pass `dim` at runtime; the NVRTC compiler re-specialises per call. For
Vulkan and cubecl the correct pattern is:

**WGSL (Vulkan path):**

```wgsl
@id(0) override CD_DIM: u32;   // set via VkSpecializationMapEntry at pipeline creation
const HALF: u32 = CD_DIM / 2u;
```

Use `gororoba_gpu_vulkan::ShaderModule::from_wgsl_with_u32_overrides(...)`
for WGSL sources that carry `@id(n) override CD_DIM: u32;`. The helper validates
the explicit override metadata, lowers selected u32 overrides into concrete
WGSL `const` declarations before `naga` SPIR-V emission, and produces one
shader/pipeline pair per `dim` value. `ComputePipelineBuilder::with_override`
remains available for precompiled SPIR-V modules that already contain Vulkan
specialization constants.

**cubecl (`#[cube]` path):**

```rust
#[cube(launch_unchecked)]
pub fn cd_eta_kernel(
    ...,
    #[comptime] cd_dim: u32,
) { ... }
```

Pass `dim as u32` as the comptime argument. cubecl specialises one shader
module per value, caching automatically. The caller must pass a consistent
compile-time value per launch; runtime dispatch sits outside `#[cube]`.

**CUDA (existing):**
NVRTC `#define CD_DIM <dim>` is emitted as `-DCD_DIM=<dim>` in compile options.
The `gororoba_gpu_cuda::CompileOptions::define("CD_DIM", dim)` builder covers this.
Calls at new dim values incur a one-time PTX recompile; the ModuleRegistry
caches the result.

## Recommended build order (Wave H+)

### H1 -- TurboQuant cubecl launcher

Status: DONE.

Evidence:

- `crates/cd_kernel/src/turboquant/cubecl_backend/launcher.rs` probes the
  cubecl-wgpu adapter, allocates buffers, dispatches
  `quantize_kernel::launch_unchecked`, and reads back results.
- `crates/cd_kernel/src/turboquant/cubecl_backend/mod.rs` exposes the
  launcher through the cubecl backend API.
- `crates/cd_kernel/tests/turboquant_cubecl_parity.rs` is the ignored GPU
  parity test for CPU == cubecl output.

Remaining validation: run the ignored test on a host with a wgpu adapter:
`cargo test -p cd_kernel --features cubecl --release turboquant_cubecl_parity -- --ignored --nocapture`.

### H2 -- Complete CUDA helper migration

Status: OPEN (raw CUDA boundary wrapper follow-up).

The old "13 direct-cudarc import sites" count was stale. A current sweep with
`rg -n '^use cudarc|CudaContext::new|cudarc::' crates --glob '*.rs'` and the
stricter `rg -n 'cudarc::nvrtc|compile_ptx_with_opts|load_module|load_function|= LaunchConfig \{|launch\(LaunchConfig \{|CudaContext::new|nvrtc::CompileOptions'`
direct-surface sweep no longer finds actionable context acquisition, NVRTC
compile, module/function loading, or literal launch-geometry duplication outside
`gororoba_gpu_cuda`. Raw stream, pointer, launch-builder, `CudaSlice`,
`UnifiedSlice`, helper-exported `Ptx`, cuBLAS, cuSOLVER, cuFFT,
pinned-host-memory, and driver FFI boundaries remain direct by design until a
typed helper exists for those exact CUDA APIs.

Migration pattern for future direct-helper duplication: replace
`CudaContext::new(0)?` and inline NVRTC compile with
`gororoba_gpu_cuda::Context::with_default_device()` and
`ModuleRegistry::compile_and_load(ctx.raw(), src, &opts, &["kernel_name"])`.
Use `CompileOptions::compile_ptx(...)` plus `ModuleRegistry::load(...)` only
when a caller must inspect, cache, or transform PTX before module loading.

Completed slices:

- `crates/gororoba_gpu_cuda/src/module.rs`: `ModuleRegistry::compile_and_load`
  now folds workspace NVRTC options and module/function registry loading into
  one helper for single-step call sites.
- `crates/lbm_3d_cuda/tests/test_cudarc_alloc.rs` and
  `crates/lbm_3d_cuda/benches/gpu_sparse_1024.rs`: CUDA context acquisition,
  inline NVRTC, direct module/function loading, and literal launch geometry now
  route through `gororoba_gpu_cuda`. Direct cudarc use remains only at the
  managed-memory allocation, raw stream, sparse solver ABI, and `PushKernelArg`
  launch-builder boundaries. The release-gate checks, clippy, MCP source
  search, ctags, lizard, and the three-test sparse managed CUDA smoke passed.
- `crates/optics_core/src/algebraic_lensing_gpu.rs`: CUDA context
  acquisition, NVRTC compile options, PTX compile, module/function loading,
  1D launch configuration, and device-buffer upload/readback now route
  through `gororoba_gpu_cuda`. Direct cudarc use remains only at the raw
  stream, device-trait, and launch-builder boundary that the helper crate
  intentionally still exposes.
- `crates/grmhd_core/src/gpu.rs`: CUDA context acquisition, `sm_70` NVRTC
  options, PTX compile, module/function loading, device-buffer ownership,
  upload/readback, and 1D launch configuration now route through
  `gororoba_gpu_cuda`. Direct cudarc use remains only at the raw stream and
  launch-builder boundary. The existing runtime smoke still reports
  `NVRTC compile kernels_grmhd.cu` on this host and treats that as a non-fatal
  unavailable-GPU path, so this slice proves launcher/helper migration and
  compile-time integration rather than a successful GRMHD CUDA run.
- `crates/sign_imbalance/src/besag_clifford_cuda.rs`: CUDA context
  acquisition, default NVRTC compile, module/function loading, device-buffer
  ownership, upload/readback, and 1D launch configuration now route through
  `gororoba_gpu_cuda`. Direct cudarc use remains only at the raw stream and
  launch-builder boundary.
- `crates/sign_imbalance/src/kubo_transport_gpu.rs`: CUDA context acquisition
  and stream-attached buffer upload/readback now route through
  `gororoba_gpu_cuda`. Direct cudarc use remains at the cuBLAS/cuSOLVER FFI,
  raw pointer-trait, and stream boundary because the Kubo path calls vendor
  eigensolver and GEMM APIs directly rather than launching workspace kernels.
  The local GPU eigendecomposition and transport parity tests passed for the
  quaternion, octonion, and J1-J2 chain models; package GPU check,
  all-targets clippy, and format checks passed.
- `crates/cd_kernel/src/turboquant/cuda/`: CUDA device probing, NVRTC compile
  options, PTX compile, context acquisition, module/function loading, module
  lifetime ownership, stream-attached buffer upload/readback, and 1D/2D launch
  configuration now route through `gororoba_gpu_cuda`. Direct cudarc use
  remains only at the raw stream, launch-builder, `PushKernelArg`, and PTX type
  boundary. The local TurboQuant CUDA test slice compiled for detected `sm_89`,
  initialized the CUDA kernels, and passed; package CUDA check, all-targets
  clippy, and format checks passed.
- `crates/gororoba_engine/examples/`: CUDA example context acquisition now
  routes through `gororoba_gpu_cuda::Context::with_default_device`. The local
  GPU example check and all-targets example clippy passed for the Voudon bridge,
  Voudon stabilizer, and phase-transition examples.
- `crates/lbm_3d_cuda/src/alignment_gpu.rs`: CUDA context acquisition,
  default NVRTC compile, module/function loading, stream-attached
  upload/readback, and 1D launch configuration now route through
  `gororoba_gpu_cuda`. Direct cudarc use remains only at the raw stream and
  launch-builder boundary. The crate release-gate check, all-targets clippy,
  targeted source inventory, and lizard complexity pass succeeded.
- `crates/lbm_3d_cuda/src/unified_runner.rs`: CUDA context acquisition,
  detected-arch NVRTC compile options, step/init and lazy slice module/function
  loading, module lifetime ownership, slice-buffer allocation/readback, and
  128-thread 1D launch configuration now route through `gororoba_gpu_cuda`.
  Direct cudarc use remains only at the raw stream, launch-builder, and
  `UnifiedSlice` boundary because the runner intentionally owns managed memory.
  The crate release-gate check, all-targets clippy, targeted source inventory,
  and lizard complexity pass succeeded.
- `crates/lbm_3d_cuda/src/box_counting_gpu.rs`: default NVRTC compile,
  module/function loading, module lifetime ownership, scratch-buffer
  allocation/readback, host-density upload, zero/count/histogram launches, and
  fixed-block reduction launch configuration now route through
  `gororoba_gpu_cuda`. Direct cudarc use remains at the raw context parameter,
  device-density `CudaSlice` input, stream, and launch-builder boundary because
  existing solver and sweep callers pass an owned solver context into the
  counter. The local ignored CUDA tests passed for uniform, single-cell,
  plane, empty-field, and CPU-agreement cases; crate check, all-targets clippy,
  targeted source inventory, and lizard complexity pass succeeded.
- `crates/lbm_3d_cuda/src/chingon_gpu.rs`: CUDA context acquisition, selected
  architecture NVRTC compile options, PTX compile, four-kernel module/function
  loading, module lifetime ownership, packed-AVT upload, working-buffer
  ownership/readback, zero/build/contraction/projection launch configuration,
  and fixed dynamic-shared-memory launch setup now route through
  `gororoba_gpu_cuda`. Direct cudarc use remains only at the raw stream,
  launch-builder, and `PushKernelArg` boundary because the Chingon pipeline
  passes a long scalar argument list into pre-existing CUDA kernels. The crate
  release-gate check, all-targets clippy, MCP source search, MCP lizard
  complexity, clangd CUDA kernel check, ctags, cscope, and cflow probes passed
  on the migrated slice.
- `crates/lbm_3d_cuda/src/dark_halo.rs`: CUDA context acquisition, selected
  architecture NVRTC precision flags, PTX compile, four-kernel module/function
  loading, module lifetime ownership, SoA buffer allocation, equilibrium
  upload, readback, convergence-counter reset, halo-counter reset, and 128-thread
  launch configuration now route through `gororoba_gpu_cuda`. Direct cudarc use
  remains at the raw stream, L2 access-policy driver calls, pinned-host-memory
  readback, and `PushKernelArg` boundary because those paths exercise CUDA
  residency and DMA APIs directly. The crate release-gate check, all-targets
  clippy, MCP source search, clangd CUDA kernel check, ctags, cscope, and cflow
  probes passed. MCP lizard ran and reported high-complexity warnings for the
  existing `DarkHaloCudaSolver::new` and `DarkHaloCudaSolver::run_k_value`
  functions; that complexity remains a separate refactor item from the helper
  migration.
- `crates/lbm_3d_cuda/src/sparse/mod.rs`: selected-architecture NVRTC compile
  options, PTX compile, sparse map module/function loading, sparse LBM
  module/function loading, module lifetime ownership for the stored LBM step
  kernel, 1D launch configuration, and caller-selected 3D map launch
  configuration now route through `gororoba_gpu_cuda`. Direct cudarc use remains
  at the raw context, stream, `CudaSlice`, `UnifiedSlice`, device pointer,
  managed prefetch, event, and launch-builder boundary because the sparse API
  exposes raw device buffers and owns unified-memory tile movement. The crate
  release-gate check, all-targets clippy, live sparse managed/tiled CUDA smoke
  tests, MCP source search, MCP lizard complexity, ctags, cscope, cflow, and
  direct clang CUDA syntax checks passed. `clangd --check` parsed
  `kernels_sparse_map.cu` with 0 errors; on `kernels_sparse_lbm.cu`, clangd
  reached semantic feature testing and failed its internal ExtractFunction tweak
  on CUDA loop control, so clang syntax checking is the stronger evidence for
  that kernel.
- `crates/lbm_3d_cuda/src/lib.rs`: production solver context acquisition,
  selected-architecture and BF16 include-path NVRTC options, PTX compile,
  core module/function loading, FP32 SoA module/function loading, 1D helper
  launches, 3D LBM launches, tiled Smagorinsky launches, enstrophy reductions,
  density reductions, and device Mach pass-1 launch configuration now route
  through `gororoba_gpu_cuda`. Direct cudarc use remains at the raw context,
  stream, slice, graph, device-pointer, launch-builder, one-element
  `LaunchConfig::for_num_elems(1)`, cuFFT, L2 access-policy driver call, and
  `PushKernelArg` boundaries because the production solver still exposes or
  calls those CUDA APIs directly. Format, release-gate check, all-targets
  clippy, focused `launch_config_1d` tests, broad CUDA-dependent crate check,
  MCP source search, MCP lizard complexity, ctags, cscope, cflow, direct clang
  CUDA syntax checking, and MCP clangd LSP checking passed on this slice.
  Lizard still reports existing complexity in `new`, `initialize_custom`,
  `launch_step_kernel`, and several SoA CUDA kernels; that is the next
  refactor pressure after helper migration, not evidence that the helper
  migration failed.
- `crates/lbm_3d_cuda/src/bench_kernels.rs`: AoS, SoA, INT4, FP4,
  double-double, on-demand slice, and TensorCore benchmark context acquisition,
  selected-architecture NVRTC options, PTX compile, module/function loading,
  module lifetime ownership, 1D benchmark launches, and TensorCore fixed-warp
  launches now route through `gororoba_gpu_cuda`. Direct cudarc use remains at
  raw stream, `CudaSlice`, launch-builder, `PushKernelArg`, and the typed
  launch-config return boundary because the benchmark runners still own raw
  device buffers and launch builders directly. Format, release-gate check,
  all-targets clippy, broad CUDA-dependent crate check, benchmark-filtered
  lib test compilation, MCP source search, MCP lizard complexity, ctags,
  cscope, cflow, direct clang CUDA syntax checking, and MCP clangd LSP checking
  passed on this slice. TensorCore syntax/LSP checks require an SM 8.9 compile
  target because the kernel uses TF32/BF16 WMMA; the SM 7.5 probe fails as
  expected for that architecture surface. Lizard still reports existing
  complexity in the double-double and TensorCore constructors; that is a
  benchmark refactor item, not a helper-migration blocker.
- `crates/lbm_3d_cuda/src/aot_cubins.rs`,
  `crates/lbm_3d_cuda/src/optix_pipeline.rs`, and
  `crates/cd_kernel/src/turboquant/cuda/jit.rs`: stale direct NVRTC,
  module-load, context-probe, and raw cudarc `Ptx` references now route through
  the helper vocabulary. `gororoba_gpu_cuda` re-exports the PTX type so JIT
  producers and `ModuleRegistry` consumers share one import surface while still
  using cudarc's underlying PTX representation. Format, focused release-gate
  checking across `gororoba_gpu_cuda`, `cd_kernel/cuda`, and `lbm_3d_cuda`, and
  the direct-surface sweep passed.
- Workspace `cudarc` dependency: updated from the locked `0.19.4` line to
  `0.19.7`, the current crates.io release. Broad CUDA-dependent crate checking
  passed across `lbm_3d_cuda`, `cd_kernel/cuda`, `gororoba_algebra/gpu`,
  `quantum_core/gpu`, `stats_core/gpu`, `algebra_experimental/gpu`,
  `sign_imbalance/gpu`, `optics_core/gpu`, `grmhd_core/gpu`, and
  `gororoba_engine/gpu`.
- `crates/quantum_core/src/gpu/peps.rs`: CUDA context acquisition, default
  NVRTC compile, module/function loading, device-buffer upload/readback, and
  1D launch configuration now route through `gororoba_gpu_cuda`. Direct cudarc
  use remains only at the raw stream and launch-builder boundary. The PEPS GPU
  bench target also imports `std::hint::black_box` so all-targets clippy can
  compile the benchmark with `gpu` enabled.
- `crates/stats_core/src/ultrametric/gpu.rs`: CUDA context acquisition,
  default NVRTC compile, module/function loading, device-buffer upload/readback,
  and 1D launch configuration now route through `gororoba_gpu_cuda`. The GPU
  test surface ran on local CUDA and passed initialization, CPU comparison,
  multi-epsilon monotonicity, and full permutation-test checks.
- `crates/algebra_experimental/src/voudon_stabilizer.rs`: CUDA context
  acquisition, default NVRTC compile, module/function loading, zeroed
  device-buffer ownership, readback, and fixed 256-thread launch configuration
  now route through `gororoba_gpu_cuda`. The crate GPU check, all-targets clippy,
  full library test surface, and local source-analysis probes passed.
- `crates/gororoba_algebra/src/gpu/voudon.rs`: CUDA context acquisition,
  default NVRTC compile, module/function loading, zeroed device-buffer ownership,
  readback, and 1D launch configuration now route through `gororoba_gpu_cuda`.
  `gororoba_algebra/gpu` now includes the `analysis` and `physics` feature
  closure needed by the CUDA module graph, so
  `cargo check -p gororoba_algebra --no-default-features --features gpu`
  passes under `release-gate`. The Voudon-focused library tests, all-targets
  clippy, and local source-analysis probes passed.
- `crates/gororoba_algebra/src/gpu/eta_matrix.rs`: CUDA context acquisition,
  default NVRTC compile, module/function loading, zeroed device-buffer ownership,
  readback, and 1D launch configuration now route through `gororoba_gpu_cuda`.
  The local CUDA parity tests passed against the CPU eta matrix for dimensions
  16, 32, 64, 128, 256, and 512; all-targets clippy and local source-analysis
  probes passed.
- `crates/gororoba_algebra/src/gpu/imbalance.rs`: CUDA context acquisition,
  default NVRTC compile, module/function loading, device-buffer upload/readback,
  and 1D launch configuration now route through `gororoba_gpu_cuda`. The GPU
  validation path rejects truncated eta arrays and invalid edge endpoints so
  `compute_imbalance_gpu` keeps the CPU fallback semantics for non-kernel-safe
  inputs. The local CUDA simple and CPU-parity imbalance tests, all-targets
  clippy, and source-analysis probes passed.
- `crates/gororoba_algebra/src/gpu/dimensional.rs`: CUDA context acquisition,
  default NVRTC compile, module/function loading, node-buffer upload, six
  zeroed counter buffers, readback, and 1D launch configuration now route
  through `gororoba_gpu_cuda`. The CPU and CUDA sampling paths now reject
  invalid dimensions and undersized sampled node sets before modulo/retry-loop
  sampling. The local CUDA dim=32 Monte Carlo test, wide-index CPU tests,
  all-targets clippy, and source-analysis probes passed.
- `crates/gororoba_algebra/src/gpu/graph_construction.rs`: CUDA context
  acquisition, default NVRTC compile, two-kernel module/function loading,
  eta/node-buffer upload, count and edge-output buffer ownership, readback, and
  1D launch configuration now route through `gororoba_gpu_cuda`. The CUDA
  kernels now use the same eta-sum edge predicate as the CPU path instead of
  accepting every upper-triangle pair. The local CUDA-vs-CPU graph edge parity
  test, all-targets clippy, and source-analysis probes passed.
- `crates/gororoba_algebra/src/gpu/tensor_avt/cuda.rs`: CUDA context
  acquisition, SM-count probing, `sm_89` NVRTC compile options with CUDA SDK
  include path and C++14 mode, PTX compile/cache, five-kernel module/function
  loading, one-shot buffer upload/readback, and 1D/2D launch configuration now
  route through `gororoba_gpu_cuda`. Direct cudarc use remains at the raw
  stream, `CudaSlice`, and launch-builder boundary because the module exposes
  reusable device workspaces as part of its public GPU API. The tensor AVT
  library slice and the ignored local CUDA identity/cross-validation tests
  passed; clangd, ctags/readtags, cscope, cflow, lizard, and source-analysis
  MCP probes were run on the migrated CUDA surface.

Acceptance: `cargo build --workspace --features gpu` clean; all existing
GPU-gated tests still pass when `--ignored` GPU tests are run.

### H3 -- gororoba_algebra Vulkan paths + override helper

Status: DONE.

For each of the 5 algebra modules (eta_matrix, imbalance, voudon,
dimensional, graph_construction):

1. Write a WGSL compute shader. Use `@id(n) override CD_DIM: u32;` for kernels
   whose work shape depends on Cayley-Dickson dimension at pipeline creation.
   DONE for `eta_matrix_vulkan`; DONE for `imbalance_vulkan` without a
   `CD_DIM` override because it validates a prepared edge list and eta label
   stream. DONE for `graph_construction_vulkan` without a `CD_DIM` override
   because it consumes a prepared eta matrix plus node list. DONE for
   `dimensional_vulkan` without a `CD_DIM` override because it classifies
   CPU-prepared APT sample triples. DONE for `voudon_vulkan` without a
   `CD_DIM` override because the Voudon field generator is fixed at 256D.
2. Add a `<module>_vulkan.rs` launcher in `gororoba_algebra/src/gpu/` that
   builds a pipeline per dim value via `gororoba_gpu_vulkan::ShaderModule`
   and `ComputePipelineBuilder`.
   DONE for `eta_matrix_vulkan`, including descriptor allocation,
   host-visible output storage, dispatch, readback, and binary-output
   validation. DONE for `imbalance_vulkan`, including CPU BFS delta
   preparation, five storage-buffer bindings, atomic frustrated-edge
   accumulation, and result readback. DONE for `graph_construction_vulkan`,
   including eta/node upload, upper-triangle pair decoding, atomic edge
   compaction, sorted readback, and CPU parity. DONE for
   `dimensional_vulkan`, including deterministic CPU sample preparation,
   node/sample upload, shader-side Cayley-Dickson sign classification, atomic
   APT counters, and CPU parity. DONE for `voudon_vulkan`, including 256D
   basis-sign evaluation, deterministic spatial hashing, integer violation
   counts, checked readback, and CPU parity.
3. DONE: add `with_override(name, value)`, `with_override_id(id, value)`,
   and `build_specialised(overrides)` to `ComputePipelineBuilder` in
   `gororoba_gpu_vulkan`; add `ShaderModule::from_wgsl_with_u32_overrides(...)`
   for WGSL sources. WGSL shaders must annotate named overrides with explicit
   `@id(n)` metadata so the lowering helper can validate stable names.
4. Wire into a distinct `vulkan` feature dispatch in each module. The existing
   `gpu` feature remains the CUDA compatibility surface. DONE for
   `eta_matrix_vulkan`, `imbalance_vulkan`, `graph_construction_vulkan`,
   `dimensional_vulkan`, and `voudon_vulkan`.

Acceptance: CPU == Vulkan for dim in {8, 16, 32, 64} in a new parity test
`crates/gororoba_algebra/tests/<module>_vulkan_parity.rs`. DONE for
`eta_matrix_vulkan_parity.rs`; DONE for `imbalance_vulkan_parity.rs` on
representative graph cases; DONE for `graph_construction_vulkan_parity.rs` on
eta/node graph cases; DONE for `dimensional_vulkan_parity.rs` on exact APT
sample-classification cases; DONE for `voudon_vulkan_parity.rs` on 256D
field-generation cases.

Validation evidence for the gororoba_algebra Vulkan paths:

```bash
CARGO_TARGET_DIR=.cache/gate-target cargo test -p gororoba_algebra \
  --no-default-features --features vulkan --test eta_matrix_vulkan_parity \
  --profile release-gate -- --ignored --nocapture

CARGO_TARGET_DIR=.cache/gate-target cargo test -p gororoba_algebra \
  --no-default-features --features vulkan --test imbalance_vulkan_parity \
  --profile release-gate -- --ignored --nocapture

CARGO_TARGET_DIR=.cache/gate-target cargo test -p gororoba_algebra \
  --no-default-features --features vulkan \
  --test graph_construction_vulkan_parity \
  --profile release-gate -- --ignored --nocapture

CARGO_TARGET_DIR=.cache/gate-target cargo test -p gororoba_algebra \
  --no-default-features --features vulkan --test dimensional_vulkan_parity \
  --profile release-gate -- --ignored --nocapture

CARGO_TARGET_DIR=.cache/gate-target cargo test -p gororoba_algebra \
  --no-default-features --features vulkan --test voudon_vulkan_parity \
  --profile release-gate -- --ignored --nocapture
```

### H4 -- gororoba_algebra cubecl paths + 3-way parity (~2-3 hours)

Status: DONE. `eta_matrix_cubecl`, `imbalance_cubecl`,
`graph_construction_cubecl`, `dimensional_cubecl`, and `voudon_cubecl`
are implemented and covered by CPU == Vulkan == cubecl parity tests.

After H3, add cubecl `#[cube]` kernels for the same 5 modules using
`#[comptime] cd_dim: u32`. Combine with H3 Vulkan results into 3-way
parity tests (CPU == Vulkan == cubecl).

Completed:

- `crates/gororoba_algebra/src/gpu/eta_matrix_cubecl.rs` implements the
  eta-matrix cubecl launcher and binary-output readback.
- `crates/gororoba_algebra/tests/eta_matrix_cubecl_parity.rs` checks
  CPU == Vulkan == cubecl for dimensions 8, 16, 32, and 64.
- `crates/gororoba_algebra/src/gpu/imbalance_cubecl.rs` implements the
  CPU-BFS plus cubecl edge-flag validation path for imbalance ratios.
- `crates/gororoba_algebra/tests/imbalance_cubecl_parity.rs` checks
  CPU == Vulkan == cubecl on representative graph cases.
- `crates/gororoba_algebra/src/gpu/graph_construction_cubecl.rs` implements
  cubecl upper-triangle pair validation with sentinel readback and host
  compaction.
- `crates/gororoba_algebra/tests/graph_construction_cubecl_parity.rs` checks
  CPU == Vulkan == cubecl on eta/node graph cases.
- `crates/gororoba_algebra/src/gpu/dimensional_cubecl.rs` implements cubecl
  APT sample classification with CPU-prepared sample triples and host
  reduction.
- `crates/gororoba_algebra/tests/dimensional_cubecl_parity.rs` checks
  CPU == Vulkan == cubecl on representative 16D and 32D APT census cases.
- `crates/gororoba_algebra/src/gpu/voudon_cubecl.rs` implements cubecl
  256D Voudon frustration-count generation with host conversion to the
  public field values.
- `crates/gororoba_algebra/tests/voudon_cubecl_parity.rs` checks
  CPU == Vulkan == cubecl on the Voudon field cases from the Vulkan parity
  lane.

Validation evidence for the gororoba_algebra cubecl paths:

```bash
CARGO_TARGET_DIR=.cache/gate-target cargo test -p gororoba_algebra \
  --no-default-features --features cubecl --lib --profile release-gate \
  eta_cubecl -- --nocapture

CARGO_TARGET_DIR=.cache/gate-target cargo test -p gororoba_algebra \
  --no-default-features --features 'vulkan cubecl' \
  --test eta_matrix_cubecl_parity \
  --profile release-gate -- --ignored --nocapture

CARGO_TARGET_DIR=.cache/gate-target cargo test -p gororoba_algebra \
  --no-default-features --features cubecl --lib --profile release-gate \
  imbalance_cubecl -- --nocapture

CARGO_TARGET_DIR=.cache/gate-target cargo test -p gororoba_algebra \
  --no-default-features --features 'vulkan cubecl' \
  --test imbalance_cubecl_parity \
  --profile release-gate -- --ignored --nocapture

CARGO_TARGET_DIR=.cache/gate-target cargo test -p gororoba_algebra \
  --no-default-features --features cubecl --lib --profile release-gate \
  graph_cubecl -- --nocapture

CARGO_TARGET_DIR=.cache/gate-target cargo test -p gororoba_algebra \
  --no-default-features --features 'vulkan cubecl' \
  --test graph_construction_cubecl_parity \
  --profile release-gate -- --ignored --nocapture

CARGO_TARGET_DIR=.cache/gate-target cargo test -p gororoba_algebra \
  --no-default-features --features cubecl --lib --profile release-gate \
  dimensional_cubecl -- --nocapture

CARGO_TARGET_DIR=.cache/gate-target cargo test -p gororoba_algebra \
  --no-default-features --features 'vulkan cubecl' \
  --test dimensional_cubecl_parity \
  --profile release-gate -- --ignored --nocapture

CARGO_TARGET_DIR=.cache/gate-target cargo test -p gororoba_algebra \
  --no-default-features --features cubecl --lib --profile release-gate \
  voudon_cubecl -- --nocapture

CARGO_TARGET_DIR=.cache/gate-target cargo test -p gororoba_algebra \
  --no-default-features --features 'vulkan cubecl' \
  --test voudon_cubecl_parity \
  --profile release-gate -- --ignored --nocapture
```

### H5 -- stats_core ultrametric Vulkan + cubecl

Status: DONE. `crates/stats_core/src/ultrametric/ultrametric_cubecl.rs`
and `crates/stats_core/src/ultrametric/ultrametric_vulkan.rs` implement CPU,
cubecl, and Vulkan minimax-path bottleneck distances over a dense adjacency
matrix. Both portable launchers use a flat source-row dispatch: each thread
computes one Dijkstra-style bottleneck row. The ignored parity tests check
CPU == cubecl and CPU == Vulkan for a 64-node ChaCha20-seeded graph.

Validation evidence:

```bash
CARGO_TARGET_DIR=.cache/gate-target cargo test -p stats_core \
  --no-default-features --features cubecl --lib --profile release-gate \
  ultrametric_cubecl -- --nocapture

CARGO_TARGET_DIR=.cache/gate-target cargo test -p stats_core \
  --no-default-features --features cubecl \
  --test ultrametric_cubecl_parity \
  --profile release-gate -- --ignored --nocapture
CARGO_TARGET_DIR=.cache/gate-target cargo check -p stats_core \
  --no-default-features --features vulkan --profile release-gate
CARGO_TARGET_DIR=.cache/gate-target cargo clippy -p stats_core \
  --no-default-features --features vulkan --all-targets \
  --profile release-gate -- -D warnings
CARGO_TARGET_DIR=.cache/gate-target cargo test -p stats_core \
  --no-default-features --features vulkan --lib ultrametric_vulkan \
  --profile release-gate -- --nocapture --test-threads=1
CARGO_TARGET_DIR=.cache/gate-target cargo test -p stats_core \
  --no-default-features --features vulkan \
  --test ultrametric_vulkan_parity \
  --profile release-gate -- --ignored --nocapture --test-threads=1
CARGO_TARGET_DIR=.cache/gate-target cargo check -p stats_core \
  --no-default-features --features 'vulkan cubecl' --profile release-gate
CARGO_TARGET_DIR=.cache/gate-target cargo clippy -p stats_core \
  --no-default-features --features 'vulkan cubecl' --all-targets \
  --profile release-gate -- -D warnings
```

### H6 -- Dark-halo ZD cubecl (~2-3 hours)

Status: DONE. `crates/lbm_vulkan/src/dark_halo_cubecl.rs` ports the
deterministic ZD viscosity hash (`dark_halo_viscosity.wgsl`) to a cubecl
kernel. The cubecl path writes the per-cell tau field; the ZD proxy count
runs on the CPU side, matching the split used by the Besag-Clifford cubecl
lane.

Acceptance is covered by `crates/lbm_vulkan/tests/dark_halo_cubecl_parity.rs`:
CPU ZD count == cubecl ZD count within the same 1% tolerance used by
`dark_halo_vulkan_parity.rs`.

Validation evidence:

```bash
CARGO_TARGET_DIR=.cache/gate-target cargo test -p lbm_vulkan \
  --features cubecl --lib --profile release-gate \
  dark_halo_cubecl -- --nocapture

CARGO_TARGET_DIR=.cache/gate-target cargo test -p lbm_vulkan \
  --features cubecl --test dark_halo_cubecl_parity \
  --profile release-gate -- --ignored --nocapture
```

### H7 -- GRMHD Vulkan + cubecl advance step (~3-4 hours)

Status: DONE. `crates/grmhd_core/src/vulkan.rs` ports the CUDA-style
conservative advance path to WGSL, and `crates/grmhd_core/src/cubecl.rs`
ports the same staged path to cubecl-wgpu: metric precompute,
primitive-to-conserved conversion, flux construction, centered flux
divergence, and forward Euler update over the crate's 8-channel SoA layout
(`rho, u, v1, v2, v3, B1, B2, B3`). The original roadmap text said HLLD and
"6-component" while listing seven fields; the crate's implemented solver is
HLL-oriented and uses the same 8 primitive/conserved channels as
`crates/grmhd_core/src/gpu.rs`.

Acceptance: CPU == Vulkan and CPU == cubecl for a 32^3 domain, 10 steps,
relative tolerance 1e-4. The Vulkan path is covered by
`vulkan::tests::grmhd_vulkan_matches_cpu_reference_for_cuda_style_advance`.
The cubecl path is covered by the module-local
`cubecl::tests::grmhd_cubecl_matches_cpu_reference_for_cuda_style_advance`
and the integration test
`crates/grmhd_core/tests/grmhd_cubecl_parity.rs`.

Validation evidence for the GRMHD Vulkan + cubecl advance path:

```bash
CARGO_TARGET_DIR=.cache/gate-target cargo test -p grmhd_core \
  --no-default-features --features vulkan --lib --profile release-gate \
  vulkan -- --nocapture
CARGO_TARGET_DIR=.cache/gate-target cargo test -p grmhd_core \
  --no-default-features --features vulkan --lib --profile release-gate \
  grmhd_vulkan_matches_cpu_reference_for_cuda_style_advance \
  -- --ignored --nocapture
CARGO_TARGET_DIR=.cache/gate-target cargo clippy -p grmhd_core \
  --no-default-features --features vulkan --all-targets \
  --profile release-gate -- -D warnings
CARGO_TARGET_DIR=.cache/gate-target cargo check -p grmhd_core \
  --no-default-features --features cubecl --profile release-gate
CARGO_TARGET_DIR=.cache/gate-target cargo clippy -p grmhd_core \
  --no-default-features --features cubecl --all-targets \
  --profile release-gate -- -D warnings
CARGO_TARGET_DIR=.cache/gate-target cargo test -p grmhd_core \
  --no-default-features --features cubecl --lib --profile release-gate \
  cubecl -- --nocapture
CARGO_TARGET_DIR=.cache/gate-target cargo test -p grmhd_core \
  --no-default-features --features cubecl --test grmhd_cubecl_parity \
  --profile release-gate -- --ignored --nocapture
lizard crates/grmhd_core/src/gpu.rs crates/grmhd_core/src/vulkan.rs \
  crates/grmhd_core/src/cubecl.rs crates/grmhd_core/src/kernels_grmhd.cu
cloc crates/grmhd_core/src/gpu.rs crates/grmhd_core/src/vulkan.rs \
  crates/grmhd_core/src/cubecl.rs crates/grmhd_core/src/kernels_grmhd.cu
cppcheck --enable=warning,style,performance --language=c++ --std=c++14 \
  --force crates/grmhd_core/src/kernels_grmhd.cu
```

### H8 -- Sparse-grid LBM portable direct active bricks -- DONE

The portable direct active-brick D3Q19 stepping surface is implemented in
`lbm_vulkan`; see the canonical H8 evidence block below. The CUDA-only
managed-memory, tile-prefetch, shared-halo, and peer-transfer variants remain
scoped as memory-policy surfaces under "What NOT to port".

### H9 -- Optics algebraic lensing Vulkan -- DONE

The implementation surface is the CUDA GRIN ray marcher in
`crates/optics_core/src/algebraic_lensing_grin.cu`, not a 2D convolution.
`crates/optics_core/src/algebraic_lensing_gpu.rs` now exposes a Vulkan
WGSL backend for the same 3D density-field RK4 ray trace: trilinear density
sampling, refractive-index gradient, ray-curvature update, and final
position/direction readback.

Validation:

```bash
cargo test -p optics_core --no-default-features --features vulkan --lib --profile release-gate algebraic_lensing -- --nocapture
cargo test -p optics_core --no-default-features --features vulkan --lib --profile release-gate algebraic_lensing_vulkan -- --ignored --nocapture
lizard crates/optics_core/src/algebraic_lensing_gpu.rs crates/gororoba_gpu_vulkan/src/buffer.rs
```

### H10 -- quantum PEPS Vulkan + cubecl

Status: DONE for the portable PEPS row-contraction slice. The original H10
title also named Voudon Vulkan, but that claim is stale: the algebra Voudon
Vulkan and cubecl paths are covered by H3/H4 (`voudon_vulkan`,
`voudon_cubecl`, and the CPU == Vulkan == cubecl parity test). The remaining
confirmed gap was `quantum_core` PEPS portable backend support.

Implemented evidence:

- `crates/quantum_core/Cargo.toml` adds separate `cubecl` and `vulkan`
  features. The existing `gpu` feature remains CUDA-only.
- `crates/quantum_core/src/gpu/peps_cubecl.rs` implements a cubecl-wgpu
  element-wise complex row product kernel and runtime launcher.
- `crates/quantum_core/src/gpu/peps_vulkan.rs` implements the matching Vulkan
  WGSL element-wise complex row product kernel and runtime launcher through
  `gororoba_gpu_vulkan`.
- `crates/quantum_core/src/peps.rs` dispatches large PEPS row products through
  CUDA when `gpu` is enabled, cubecl when `cubecl` is enabled without CUDA, and
  Vulkan when `vulkan` is enabled without CUDA or cubecl. CPU fallback remains
  available for adapter failure or invalid input.
- The cubecl and Vulkan paths have an explicit portable FP32 precision contract:
  they narrow `faer::c64` real and imaginary components to `f32` before dispatch
  and widen readback to `c64`. CUDA remains the FP64 backend for PEPS row
  contraction.
- `crates/quantum_core/tests/peps_cubecl_parity.rs` adds the ignored adapter
  parity test for CPU == cubecl on a representative complex row product.
- `crates/quantum_core/tests/peps_vulkan_parity.rs` adds the ignored adapter
  parity test for CPU == Vulkan on the same representative complex row product.
- `crates/quantum_core/benches/gpu_peps_bench.rs` now benchmarks the cubecl and
  Vulkan PEPS FP32 paths under their backend features.

Validation evidence:

```bash
CARGO_TARGET_DIR=.cache/gate-target cargo check -p quantum_core \
  --no-default-features --features cubecl --profile release-gate
CARGO_TARGET_DIR=.cache/gate-target cargo clippy -p quantum_core \
  --no-default-features --features cubecl --all-targets \
  --profile release-gate -- -D warnings
CARGO_TARGET_DIR=.cache/gate-target cargo test -p quantum_core \
  --no-default-features --features cubecl --lib peps_cubecl \
  --profile release-gate
CARGO_TARGET_DIR=.cache/gate-target cargo test -p quantum_core \
  --no-default-features --features cubecl --test peps_cubecl_parity \
  --profile release-gate
CARGO_TARGET_DIR=.cache/gate-target cargo check -p quantum_core \
  --no-default-features --features 'gpu cubecl' --profile release-gate
CARGO_TARGET_DIR=.cache/gate-target cargo clippy -p quantum_core \
  --no-default-features --features 'gpu cubecl' --all-targets \
  --profile release-gate -- -D warnings
CARGO_TARGET_DIR=.cache/gate-target cargo check -p quantum_core \
  --no-default-features --features vulkan --profile release-gate
CARGO_TARGET_DIR=.cache/gate-target cargo clippy -p quantum_core \
  --no-default-features --features vulkan --all-targets \
  --profile release-gate -- -D warnings
CARGO_TARGET_DIR=.cache/gate-target cargo test -p quantum_core \
  --no-default-features --features vulkan --lib peps_vulkan \
  --profile release-gate -- --nocapture --test-threads=1
CARGO_TARGET_DIR=.cache/gate-target cargo test -p quantum_core \
  --no-default-features --features vulkan --test peps_vulkan_parity \
  --profile release-gate -- --ignored --nocapture --test-threads=1
CARGO_TARGET_DIR=.cache/gate-target cargo check -p quantum_core \
  --no-default-features --features 'vulkan cubecl' --profile release-gate
CARGO_TARGET_DIR=.cache/gate-target cargo clippy -p quantum_core \
  --no-default-features --features 'vulkan cubecl' --all-targets \
  --profile release-gate -- -D warnings
```

### H11 -- Optics algebraic lensing cubecl

Status: DONE. Row 11's cubecl `NO` was still true before this slice:
`optics_core` had CUDA and Vulkan algebraic-lensing paths but no cubecl feature,
launcher, public export, or parity test. The new cubecl path mirrors the Vulkan
3D RK4 ray trace over a density field: trilinear density sampling,
refractive-index gradient, ray-curvature update, and final position/direction
readback.

Implemented evidence:

- `crates/optics_core/Cargo.toml` adds a separate `cubecl` feature with
  `cubecl`, `cubecl-wgpu`, and `gororoba_gpu_cubecl` dependencies.
- `crates/optics_core/src/lib.rs` exports `AlgebraicLensingCubecl`,
  `AlgebraicLensingCubeclConfig`, and `trace_rays_cpu_reference_cubecl` when
  `cubecl` is enabled.
- `crates/optics_core/src/algebraic_lensing_gpu.rs` implements the cubecl-wgpu
  kernel, host launcher, input validation, CPU reference, and adapter-gated
  unit parity.
- `crates/optics_core/tests/algebraic_lensing_cubecl_parity.rs` adds the
  ignored adapter parity target for CPU == cubecl on the same gradient-density
  ray fixture used by the Vulkan path.

Validation evidence:

```bash
CARGO_TARGET_DIR=.cache/gate-target cargo check -p optics_core \
  --no-default-features --features cubecl --profile release-gate
CARGO_TARGET_DIR=.cache/gate-target cargo clippy -p optics_core \
  --no-default-features --features cubecl --all-targets \
  --profile release-gate -- -D warnings
CARGO_TARGET_DIR=.cache/gate-target cargo test -p optics_core \
  --no-default-features --features cubecl --lib algebraic_lensing \
  --profile release-gate
CARGO_TARGET_DIR=.cache/gate-target cargo test -p optics_core \
  --no-default-features --features cubecl \
  --test algebraic_lensing_cubecl_parity --profile release-gate
CARGO_TARGET_DIR=.cache/gate-target cargo check -p optics_core \
  --no-default-features --features 'vulkan cubecl' --profile release-gate
CARGO_TARGET_DIR=.cache/gate-target cargo clippy -p optics_core \
  --no-default-features --features 'vulkan cubecl' --all-targets \
  --profile release-gate -- -D warnings
```

### H12 -- algebra_experimental Voudon stabilizer Vulkan + cubecl

Status: DONE for the portable row-count slice. The remaining
`algebra_experimental` Voudon gap was not the separate `gororoba_algebra`
Voudon path from H3/H4; row 12 still lacked Vulkan/cubecl implementations and
portable parity tests for the 256D stabilizer predicate.

Implemented evidence:

- `crates/algebra_experimental/Cargo.toml` adds separate `cubecl` and `vulkan`
  features. The existing `gpu` feature remains CUDA-only.
- `crates/algebra_experimental/src/voudon_stabilizer.rs` now exposes a CPU
  stabilizer predicate, deterministic per-row count oracle, Vulkan row-count
  WGSL kernel, cubecl-wgpu row-count kernel, and host reconstruction of the
  first stable triples.
- The Vulkan and cubecl paths count stable-cycle rows rather than appending
  triples with a GPU atomic counter. That keeps parity deterministic and
  compares the portable predicate surface directly against the CPU oracle.
- `crates/algebra_experimental/tests/voudon_cubecl_parity.rs` adds the ignored
  adapter parity target for CPU == cubecl row counts across all 256 rows.
- `crates/algebra_experimental/tests/voudon_vulkan_parity.rs` adds the ignored
  adapter parity target for CPU == Vulkan row counts across all 256 rows.
- The Vulkan availability probe uses the same match-binding shape as the
  existing `gororoba_algebra` Vulkan probes. On the local NVIDIA 610.43.02
  stack, the shorter `build_vulkan_context().is_ok()` temporary-drop shape
  triggered a driver-thread segfault during probe teardown.

Validation evidence:

```bash
CARGO_TARGET_DIR=.cache/gate-target cargo clippy -p algebra_experimental \
  --no-default-features --features cubecl --all-targets \
  --profile release-gate -- -D warnings
CARGO_TARGET_DIR=.cache/gate-target cargo test -p algebra_experimental \
  --no-default-features --features cubecl --lib voudon_stabilizer \
  --profile release-gate
CARGO_TARGET_DIR=.cache/gate-target cargo test -p algebra_experimental \
  --no-default-features --features cubecl --test voudon_cubecl_parity \
  --profile release-gate -- --ignored
CARGO_TARGET_DIR=.cache/gate-target cargo check -p algebra_experimental \
  --no-default-features --features vulkan --profile release-gate
CARGO_TARGET_DIR=.cache/gate-target cargo clippy -p algebra_experimental \
  --no-default-features --features vulkan --all-targets \
  --profile release-gate -- -D warnings
CARGO_TARGET_DIR=.cache/gate-target cargo test -p algebra_experimental \
  --no-default-features --features vulkan --lib voudon_stabilizer \
  --profile release-gate -- --test-threads=1 --nocapture
CARGO_TARGET_DIR=.cache/gate-target cargo test -p algebra_experimental \
  --no-default-features --features vulkan --test voudon_vulkan_parity \
  --profile release-gate -- --ignored --nocapture --test-threads=1
CARGO_TARGET_DIR=.cache/gate-target cargo check -p algebra_experimental \
  --no-default-features --features 'gpu cubecl vulkan' --profile release-gate
CARGO_TARGET_DIR=.cache/gate-target cargo clippy -p algebra_experimental \
  --no-default-features --features 'gpu cubecl vulkan' --all-targets \
  --profile release-gate -- -D warnings
CARGO_TARGET_DIR=.cache/gate-target cargo test -p algebra_experimental \
  --no-default-features --features 'cubecl vulkan' --lib voudon_stabilizer \
  --profile release-gate -- --test-threads=1 --nocapture
```

### H13 -- gororoba_algebra Tensor AVT Vulkan + cubecl

Status: DONE for the portable dense-multiply and norm-squared slice. Row 21
now covers the same public Tensor AVT operations across CUDA, Vulkan, and
cubecl: single dense CD multiply, batched dense CD multiply, and batched
norm-squared.

Implemented evidence:

- `crates/gororoba_algebra/src/gpu/tensor_avt/vulkan.rs` now implements a
  Vulkan WGSL backend for Tensor AVT dense multiply and norm-squared.
  `ComputeBackend::Vulkan` sessions route through the Vulkan workspace.
- `crates/gororoba_algebra/src/gpu/tensor_avt/cubecl.rs` now implements the
  matching cubecl-wgpu kernels and explicit `compute_*_cubecl` APIs.
- `crates/gororoba_algebra/src/gpu/tensor_avt/sessions.rs` carries boxed
  Vulkan workspaces so the session enum does not grow to the size of the
  Vulkan buffer/device state.
- `crates/gororoba_algebra/tests/tensor_avt_vulkan_parity.rs` validates
  CPU == Vulkan for single multiply, batched multiply, and norm-squared over
  16D and 32D fixtures.
- `crates/gororoba_algebra/tests/tensor_avt_cubecl_parity.rs` validates
  CPU == cubecl for the same operations and fixtures.
- The Tensor AVT Vulkan runtime stores buffers and pipeline/device state in a
  drop order that avoids the local NVIDIA teardown segfault observed when a
  temporary runtime probe dropped the instance before child Vulkan objects.

Validation evidence:

```bash
CARGO_TARGET_DIR=.cache/gate-target cargo check -p gororoba_algebra \
  --no-default-features --features vulkan --profile release-gate
CARGO_TARGET_DIR=.cache/gate-target cargo clippy -p gororoba_algebra \
  --no-default-features --features vulkan --all-targets \
  --profile release-gate -- -D warnings
CARGO_TARGET_DIR=.cache/gate-target cargo test -p gororoba_algebra \
  --no-default-features --features vulkan --lib tensor_avt \
  --profile release-gate -- --nocapture --test-threads=1
CARGO_TARGET_DIR=.cache/gate-target cargo test -p gororoba_algebra \
  --no-default-features --features vulkan --test tensor_avt_vulkan_parity \
  --profile release-gate -- --ignored --nocapture --test-threads=1
CARGO_TARGET_DIR=.cache/gate-target cargo check -p gororoba_algebra \
  --no-default-features --features cubecl --profile release-gate
CARGO_TARGET_DIR=.cache/gate-target cargo clippy -p gororoba_algebra \
  --no-default-features --features cubecl --all-targets \
  --profile release-gate -- -D warnings
CARGO_TARGET_DIR=.cache/gate-target cargo test -p gororoba_algebra \
  --no-default-features --features cubecl --lib tensor_avt \
  --profile release-gate -- --nocapture --test-threads=1
CARGO_TARGET_DIR=.cache/gate-target cargo test -p gororoba_algebra \
  --no-default-features --features cubecl --test tensor_avt_cubecl_parity \
  --profile release-gate -- --ignored --nocapture --test-threads=1
CARGO_TARGET_DIR=.cache/gate-target cargo check -p gororoba_algebra \
  --no-default-features --features 'vulkan cubecl' --profile release-gate
CARGO_TARGET_DIR=.cache/gate-target cargo clippy -p gororoba_algebra \
  --no-default-features --features 'vulkan cubecl' --all-targets \
  --profile release-gate -- -D warnings
```

## Shared component strengthening

The three foundation crates need these additions to enable H3+:

### gororoba_gpu_vulkan additions needed

- DONE: `ComputePipelineBuilder::with_override(name: &str, value: u32)`: sets a
  numeric Vulkan specialization constant via `VkSpecializationMapEntry` before
  creating `VkPipeline`. Use this for SPIR-V modules that already contain
  specialization constants.
- DONE: `ComputePipelineBuilder::build_specialised(overrides: &[(String, u32)])`:
  batch variant for kernels with multiple override constants.
- DONE: `ShaderModule::from_wgsl_with_u32_overrides(...)`: validates explicit
  named WGSL overrides and lowers selected u32 values to concrete constants
  before `naga` SPIR-V emission. This is required because `naga` 29 validates
  WGSL overrides but rejects `Expression::Override` in the SPIR-V backend.
- DONE: `HostVisibleBuffer`: owns a host-visible, host-coherent `VkBuffer` and
  `VkDeviceMemory` pair with checked u32 upload/readback helpers.
- DONE: `DescriptorPool` and `DescriptorSet`: allocate descriptor sets from a
  layout and write storage-buffer bindings without repeating raw Vulkan
  boilerplate in each algebra launcher.

### gororoba_gpu_cubecl additions needed

- `skip_if_unavailable!()` macro is already present. No additions required
  for H1-H6.

### H8 -- lbm_vulkan sparse-grid LBM direct active bricks

Status: DONE for the direct active-brick D3Q19 BGK sparse stepping surface.

Implemented evidence:

- `crates/lbm_vulkan/src/sparse_lbm_common.rs` builds backend-neutral sparse
  brick metadata from a dense geometry mask and provides the CPU oracle for the
  direct active-brick A-A stepping model.
- `crates/lbm_vulkan/src/sparse_lbm_vulkan.rs` implements the Vulkan compute
  path over the same `active_brick_ids` and `indirect_table` ABI used by the
  CUDA sparse path.
- `crates/lbm_vulkan/src/sparse_lbm_cubecl.rs` implements the cubecl-wgpu
  path for the same direct active-brick stepping model.
- `crates/lbm_vulkan/tests/sparse_lbm_vulkan_parity.rs` and
  `crates/lbm_vulkan/tests/sparse_lbm_cubecl_parity.rs` compare each portable
  backend against the CPU sparse oracle.

Validation:

```bash
CARGO_TARGET_DIR=.cache/gate-target cargo check -p lbm_vulkan \
  --no-default-features --features cubecl --profile release-gate
CARGO_TARGET_DIR=.cache/gate-target cargo clippy -p lbm_vulkan \
  --no-default-features --features cubecl --all-targets \
  --profile release-gate -- -D warnings
CARGO_TARGET_DIR=.cache/gate-target cargo test -p lbm_vulkan \
  --no-default-features --features cubecl --lib sparse_lbm \
  --profile release-gate -- --nocapture --test-threads=1
CARGO_TARGET_DIR=.cache/gate-target cargo test -p lbm_vulkan \
  --no-default-features --features cubecl --test sparse_lbm_vulkan_parity \
  --profile release-gate -- --ignored --nocapture --test-threads=1
CARGO_TARGET_DIR=.cache/gate-target cargo test -p lbm_vulkan \
  --no-default-features --features cubecl --test sparse_lbm_cubecl_parity \
  --profile release-gate -- --ignored --nocapture --test-threads=1
```

The portable scope intentionally does not claim CUDA managed-memory
oversubscription, tile prefetch, or shared-halo staging. Those remain CUDA
memory-policy and kernel-variant choices rather than backend parity gaps for
the direct active-brick D3Q19 stepping surface.

### gororoba_gpu_cuda additions needed

- DONE: `CompileOptions::define(name: &str, value: impl Display)`: adds
  `-DNAME=value` to the NVRTC compile flags. Needed by H2 migration to
  replace ad-hoc string formatting in the remaining direct cudarc sites.
- DONE: `CompileOptions::include_path(path)`: carries NVRTC header search
  paths through the helper, needed by CUDA sources that include SDK headers
  such as `cuda_runtime.h`.
- DONE: `CompileOptions::for_arch(7, 0)` maps to `sm_70`, preserving the old
  GRMHD CUDA target instead of falling through to the generic fallback.
- DONE: `CompileOptions::option(...)`: adds an arbitrary NVRTC command-line
  option for CUDA sources that need flags other than defines, such as
  `-std=c++14` for WMMA-heavy tensor AVT kernels.
- DONE: `LaunchConfig::launch_blocks_1d(...)` and
  `LaunchConfig::launch_blocks_2d(...)`: preserve kernels that choose explicit
  grid/block shapes before launch instead of deriving the grid from an element
  count.
- DONE: `DeviceProbe::sm_count`: exposes the CUDA multiprocessor count through
  the canonical device probe, replacing direct driver-attribute queries in
  launch-shape heuristics.
- DONE: Workspace `cudarc` dependency is pinned to the current latest
  `0.19.7` release. docs.rs for that release documents CUDA 13.0 support in
  addition to the existing CUDA 11.x and 12.x ranges.

## Testing conventions

All GPU parity tests follow the pattern established in
`crates/cd_kernel/tests/turboquant_cubecl_parity.rs`:

1. Gated `#[ignore = "gpu (backend description)"]`.
2. Runtime adapter probe at the start; `eprintln!("skip: ...")` and `return`
   if unavailable (never `panic` or `assert`).
3. ChaCha20-seeded inputs (`RNG_SEED: u64 = 0x...`).
4. Tolerance hierarchy: integer kernels require bit-exact output;
   f32 exp/trig kernels use `abs_tol=1e-5, rel_tol=1e-4`; MRT/FMA-heavy
   kernels use `abs_tol=2e-3, rel_tol=2e-2`.
5. On failure: print the first 3-5 mismatching cells with values and errors.

Run GPU tests with:

```
cargo test -p <crate> --features <backend> --release \
  <test_module> -- --ignored --nocapture
```

## Status summary (2026-05-17)

| Wave | PR  | Description                                      | Status   |
| ---- | --- | ------------------------------------------------ | -------- |
| A    | #21 | snake_case + iridate + cordierite + 2D test      | MERGED   |
| B1   | #22 | gororoba_gpu_vulkan foundation                   | MERGED   |
| B2   | #23 | gororoba_gpu_cubecl foundation                   | MERGED   |
| B3   | #24 | gororoba_gpu_cuda foundation                     | MERGED   |
| B4   | #25 | StoragePrecision bridges + deprecations          | MERGED   |
| C1   | #26 | Migrate lbm_vulkan                               | MERGED   |
| C2   | #27 | Migrate cd_kernel turboquant                     | MERGED   |
| C3   | #28 | Migrate lbm_3d_cuda + sign_imbalance             | MERGED   |
| C4   | #29 | Migrate algebra/grmhd/quantum/stats              | MERGED   |
| C5   | #30 | Migrate Vulkan CLI bins + OptiX consumers        | MERGED   |
| C-t  | #31 | Delete deprecated Precision enums                | MERGED   |
| D    | #51 | LBM D3Q19 Vulkan (BGK PUSH shader)               | MERGED   |
| E    | #52 | LBM D3Q19 cubecl path                            | MERGED   |
| F    | #54 | 3-way LBM D3Q19 parity test                      | MERGED   |
| G1   | #55 | LBM MRT Vulkan + cubecl + 3-way parity           | MERGED   |
| G2   | #56 | Alignment cubecl parity                          | MERGED   |
| G3   | #57 | Dark-halo ZD Vulkan (3-pipeline)                 | MERGED   |
| G5   | #58 | Besag-Clifford cubecl shuffle + transform        | MERGED   |
| H1   | --  | TurboQuant cubecl launcher + parity              | DONE     |
| H2   | --  | CUDA helper migration inventory                  | OPEN     |
| H3   | --  | gororoba_algebra Vulkan + CD_DIM override helper | DONE     |
| H4   | --  | gororoba_algebra cubecl + 3-way parity           | DONE     |
| H5   | --  | stats_core ultrametric Vulkan + cubecl           | DONE     |
| H6   | --  | Dark-halo ZD cubecl                              | DONE     |
| H7   | --  | GRMHD Vulkan + cubecl advance step               | DONE     |
| H8   | --  | lbm_vulkan sparse-grid direct active bricks      | DONE     |
| H9   | --  | Optics algebraic lensing Vulkan                  | DONE     |
| H10  | --  | quantum_core PEPS Vulkan + cubecl row contraction | DONE     |
| H11  | --  | optics_core algebraic lensing cubecl             | DONE     |
| H12  | --  | algebra_experimental Voudon Vulkan + cubecl rows | DONE     |
| H13  | --  | gororoba_algebra Tensor AVT Vulkan + cubecl      | DONE     |
