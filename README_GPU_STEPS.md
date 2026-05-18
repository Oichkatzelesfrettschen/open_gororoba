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

| # | Kernel / Subsystem                  | Crate(s)                    | Dims      | CPU | CUDA | Vulkan | cubecl | Parity test |
|---|-------------------------------------|-----------------------------|-----------|-----|------|--------|--------|-------------|
| 1 | TurboQuant quantize (3-bit, 128-d)  | cd_kernel                   | 128       | YES | n/a  | YES    | YES    | YES (both) |
| 2 | LBM D3Q19 stream + collide (BGK)    | lbm_vulkan / lbm_3d         | 3D grid   | YES | YES  | YES    | YES    | YES (3-way) |
| 3 | LBM MRT collision                   | lbm_vulkan / lbm_3d_cuda    | 3D grid   | YES | YES  | YES    | YES    | YES (3-way) |
| 4 | Sparse-grid LBM                     | lbm_3d_cuda                 | 3D grid   | n/a | YES  | NO     | NO     | NO |
| 5 | Box-counting fractal dimension      | lbm_3d_cuda + lbm_vulkan    | 3D grid   | YES | YES  | YES    | YES    | YES |
| 6 | Chingon (anisotropy operator)       | lbm_3d_cuda + lbm_vulkan    | 3D grid   | YES | YES  | YES    | YES    | YES |
| 7 | Alignment / orientation projection  | lbm_3d_cuda + lbm_vulkan    | 3D grid   | YES | YES  | YES    | YES    | YES |
| 8 | Besag-Clifford GMRF shuffle+xform   | sign_imbalance + lbm_vulkan | n_cells   | YES | YES  | YES    | YES    | YES |
| 9 | Dark-halo Monte Carlo (ZD Vulkan)   | lbm_3d_cuda + lbm_vulkan    | 3D grid   | n/a | YES  | YES    | NO     | YES (Vulkan) |
|10 | Kubo transport conductivity         | sign_imbalance              | NxN       | YES | YES  | n/a    | n/a    | NO |
|11 | Algebraic lensing                   | optics_core                 | 2D        | YES | YES  | NO     | NO     | NO |
|12 | Voudon stabilizer                   | algebra_experimental        | 16D       | YES | YES  | NO     | NO     | NO |
|13 | GRMHD GPU advance                   | grmhd_core                  | 3D grid   | YES | YES  | NO     | NO     | NO |
|14 | Coop-matrix probe (Vulkan-only)     | lbm_vulkan                  | tile      | n/a | NO   | YES    | NO     | n/a |
|15 | OptiX BVH ray-tracing               | lbm_3d_cuda + gororoba_optix| scene     | n/a | YES  | NO     | NO     | n/a |
|16 | Eta matrix (ZD analysis)            | gororoba_algebra            | 8..1024D  | YES | YES  | NO     | NO     | NO |
|17 | ZD imbalance ratio                  | gororoba_algebra            | 8..256D   | YES | YES  | NO     | NO     | NO |
|18 | Voudon kernel (algebra)             | gororoba_algebra            | 16D       | YES | YES  | NO     | NO     | NO |
|19 | APT census / dimensional scan       | gororoba_algebra            | 8..256D   | YES | YES  | NO     | NO     | NO |
|20 | ZD graph construction               | gororoba_algebra            | 8..256D   | YES | YES  | NO     | NO     | NO |
|21 | Tensor AVT                          | gororoba_algebra            | 16/32D    | YES | YES  | STUB   | NO     | NO |
|22 | PEPS row contraction                | quantum_core                | n_sites   | YES | YES  | NO     | NO     | NO |
|23 | Ultrametric triples                 | stats_core                  | n_sites   | YES | YES  | NO     | NO     | NO |

Legend: YES = wired and tested. NO = absent. STUB = module exists, returns
"not implemented". n/a = backend does not apply to this kernel.

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
- **Sparse-grid LBM brick-map** (`lbm_3d_cuda`): The peer-to-peer inter-GPU
  brick transfer relies on `cudarc` device-to-device memcpy and UVA. A
  Vulkan port would require `VK_KHR_buffer_device_address` + external
  memory handles. Defer.

## Dimensional-kernel strategy

Several kernels (eta_matrix, imbalance, APT scan, graph_construction) run over
Cayley-Dickson algebras at dim = 8, 16, 32, 64, 128, 256. The CPU and CUDA
paths pass `dim` at runtime; the NVRTC compiler re-specialises per call. For
Vulkan and cubecl the correct pattern is:

**WGSL (Vulkan path):**
```wgsl
override CD_DIM: u32;          // set via VkSpecializationMapEntry at pipeline creation
const HALF: u32 = CD_DIM / 2u;
```

Use `gororoba_gpu_vulkan::shader::ComputePipelineBuilder::with_override("CD_DIM", dim)`
(not yet implemented; see Wave H3 below). Each distinct `dim` value produces
a separate `VkPipeline` object; the builder caches them keyed on the
override map.

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
NVRTC `#define CD_DIM <dim>` via `-DCDIM=<dim>` in compile options. The
`gororoba_gpu_cuda::CompileOptions::define("CD_DIM", dim)` builder covers this.
Calls at new dim values incur a one-time PTX recompile; the ModuleRegistry
caches the result.

## Recommended build order (Wave H+)

### H1 -- TurboQuant cubecl launcher (~1-2 hours)

The `#[cube]` quantize kernel already exists at
`crates/cd_kernel/src/turboquant/cubecl_backend/quantize_kernel.rs` but has
no launcher.rs integration and no parity test.

Steps:
1. Write `turboquant/cubecl_backend/launcher.rs` -- probe adapter, allocate
   buffers, dispatch `quantize_kernel::launch_unchecked`, readback.
2. Wire into `cd_kernel::turboquant::cubecl_backend` public API.
3. Add `crates/cd_kernel/tests/turboquant_cubecl_parity.rs` (gated
   `#[ignore = "gpu"]`): CPU == cubecl output byte-exact (u8 codes).

Acceptance: existing `turboquant_vulkan_parity.rs` still passes; new cubecl
test passes when `--features cubecl` and a wgpu adapter is present.

### H2 -- Complete Wave C migration (13 direct-cudarc import sites)

Eight crates still import `use cudarc::` directly instead of routing through
`gororoba_gpu_cuda`. This is the migration left unfinished in Wave C:

- `crates/gororoba_algebra/src/gpu/` (6 modules): eta_matrix, imbalance,
  voudon, dimensional, graph_construction, tensor_avt/cuda
- `crates/lbm_3d_cuda/src/` (2 sites): inline context init, managed_memory
- `crates/cd_kernel/src/turboquant/cuda/` (1 site): device.rs
- `crates/optics_core/src/` (1 site)
- `crates/quantum_core/src/gpu/` (1 site): peps.rs
- `crates/stats_core/src/ultrametric/` (1 site): gpu.rs
- `crates/algebra_experimental/` (1 site)

Pattern for each: replace `CudaContext::new(0)?` and inline NVRTC compile
with `gororoba_gpu_cuda::Context::with_default_device()` and
`ModuleRegistry::compile_and_load(src, &["kernel_name"])`.

Acceptance: `cargo build --workspace --features gpu` clean; all existing
GPU-gated tests still pass when `--ignored` GPU tests are run.

### H3 -- gororoba_algebra Vulkan paths + override helper (~3-4 hours)

For each of the 5 algebra modules (eta_matrix, imbalance, voudon,
dimensional, graph_construction):

1. Write a WGSL compute shader using `override CD_DIM: u32;`.
2. Add a `<module>_vulkan.rs` launcher in `gororoba_algebra/src/gpu/` that
   builds a pipeline per dim value via `gororoba_gpu_vulkan::ComputePipelineBuilder`.
3. Add `with_override(name, value)` to `ComputePipelineBuilder` in
   `gororoba_gpu_vulkan` (the helper needed by all five shaders).
4. Wire into the existing `gpu` feature dispatch in each module.

Acceptance: CPU == Vulkan for dim in {8, 16, 32, 64} in a new parity test
`crates/gororoba_algebra/tests/<module>_vulkan_parity.rs`.

### H4 -- gororoba_algebra cubecl paths + 3-way parity (~2-3 hours)

After H3, add cubecl `#[cube]` kernels for the same 5 modules using
`#[comptime] cd_dim: u32`. Combine with H3 Vulkan results into 3-way
parity tests (CPU == Vulkan == cubecl).

### H5 -- stats_core ultrametric cubecl (~1-2 hours)

`crates/stats_core/src/ultrametric/gpu.rs` has a CUDA path but no Vulkan or
cubecl. The operation (Dijkstra-variant bottleneck distance) is
embarrassingly parallel per source node.

1. Write `ultrametric_cubecl.rs` using `#[cube]` with a flat thread-per-row
   dispatch.
2. Parity test: CPU == cubecl for a 64-node graph (ChaCha20-seeded).

### H6 -- Dark-halo ZD cubecl (~2-3 hours)

`lbm_vulkan::dark_halo_vulkan` exists and passes parity tests. Add a cubecl
path by porting the deterministic ZD viscosity hash (`dark_halo_viscosity.wgsl`)
to a `#[cube]` kernel. The LBM step and detector run on CPU for the cubecl
path (same split as Besag-Clifford).

Acceptance: CPU ZD count == cubecl ZD count within 1% tolerance (matching the
existing Vulkan tolerance in `dark_halo_vulkan_parity.rs`).

### H7 -- GRMHD Vulkan advance step (~3-4 hours)

`crates/grmhd_core/src/gpu.rs` has a CUDA advance kernel. Port the HLLD
Riemann solver loop to WGSL. The shader needs only `override NX/NY/NZ: u32`
and a 6-component field array (`rho, vx, vy, vz, Bx, By, Bz`).

Acceptance: CPU == Vulkan for a 32^3 domain, 10 steps, relative tolerance
1e-4 (wider than LBM BGK due to HLLD multi-condition branching).

### H8 -- Sparse-grid LBM Vulkan (deferred, complex)

See "What NOT to port" above. Requires `VK_KHR_buffer_device_address` and
an implementation of the brick-map peer transfer without UVA. Estimated 1-2
weeks of effort. Defer until a concrete use case requires Vulkan on multi-GPU.

### H9 -- Optics algebraic lensing Vulkan (~2-3 hours)

`crates/optics_core/src/` has a CUDA algebraic lensing kernel. Port to WGSL.
The kernel is a 2D convolution with a CD-algebra-derived PSF.

### H10 -- Voudon Vulkan / quantum PEPS cubecl (low priority)

Low-usage paths. Port after H3-H7 if coverage targets require it.

## Shared component strengthening

The three foundation crates need these additions to enable H3+:

### gororoba_gpu_vulkan additions needed

- `ComputePipelineBuilder::with_override(name: &str, value: u32)`: sets a
  WGSL pipeline-override constant via `VkSpecializationMapEntry` before
  creating `VkPipeline`. Caches compiled pipelines keyed on
  `(spirv_hash, override_map)`.
- `ComputePipelineBuilder::build_specialised(overrides: &[(String, u32)])`:
  batch variant for kernels with multiple override constants.

### gororoba_gpu_cubecl additions needed

- `skip_if_unavailable!()` macro is already present. No additions required
  for H1-H6.

### gororoba_gpu_cuda additions needed

- `CompileOptions::define(name: &str, value: impl Display)`: adds
  `-DNAME=value` to the NVRTC compile flags. Needed by H2 migration to
  replace ad-hoc string formatting in the 13 direct-cudarc sites.

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

| Wave | PR  | Description                                       | Status   |
|------|-----|---------------------------------------------------|----------|
| A    | #21 | snake_case + iridate + cordierite + 2D test       | MERGED   |
| B1   | #22 | gororoba_gpu_vulkan foundation                    | MERGED   |
| B2   | #23 | gororoba_gpu_cubecl foundation                    | MERGED   |
| B3   | #24 | gororoba_gpu_cuda foundation                      | MERGED   |
| B4   | #25 | StoragePrecision bridges + deprecations           | MERGED   |
| C1   | #26 | Migrate lbm_vulkan                                | MERGED   |
| C2   | #27 | Migrate cd_kernel turboquant                      | MERGED   |
| C3   | #28 | Migrate lbm_3d_cuda + sign_imbalance              | MERGED   |
| C4   | #29 | Migrate algebra/grmhd/quantum/stats               | MERGED   |
| C5   | #30 | Migrate Vulkan CLI bins + OptiX consumers         | MERGED   |
| C-t  | #31 | Delete deprecated Precision enums                 | MERGED   |
| D    | #51 | LBM D3Q19 Vulkan (BGK PUSH shader)                | MERGED   |
| E    | #52 | LBM D3Q19 cubecl path                             | MERGED   |
| F    | #54 | 3-way LBM D3Q19 parity test                       | MERGED   |
| G1   | #55 | LBM MRT Vulkan + cubecl + 3-way parity            | MERGED   |
| G2   | #56 | Alignment cubecl parity                           | MERGED   |
| G3   | #57 | Dark-halo ZD Vulkan (3-pipeline)                  | MERGED   |
| G5   | #58 | Besag-Clifford cubecl shuffle + transform         | MERGED   |
| H1   | --  | TurboQuant cubecl launcher + parity               | OPEN     |
| H2   | --  | Wave C migration (13 remaining cudarc sites)      | OPEN     |
| H3   | --  | gororoba_algebra Vulkan + CD_DIM override helper  | OPEN     |
| H4   | --  | gororoba_algebra cubecl + 3-way parity            | OPEN     |
| H5   | --  | stats_core ultrametric cubecl                     | OPEN     |
| H6   | --  | Dark-halo ZD cubecl                               | OPEN     |
| H7   | --  | GRMHD Vulkan advance step                         | OPEN     |
| H8   | --  | Sparse-grid LBM Vulkan (deferred -- complex)      | DEFERRED |
| H9   | --  | Optics algebraic lensing Vulkan                   | OPEN     |
