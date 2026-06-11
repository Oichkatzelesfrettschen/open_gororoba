# LBM D3Q19 Vulkan + cubecl Port Scope (#136 follow-up)

Date: 2026-05-13
Parent: #136 "cubecl + Vulkan + CUDA feature parity sync"
Status: NOT STARTED -- this document scopes the next major #136 sub-PR.

## WHY

Of the 15 GPU kernels enumerated in
`docs/engineering/cubecl_vulkan_cuda_parity_matrix.md`, the LBM D3Q19
stream + collide is the most-used in the workspace (every fluid-dynamics
experiment runs it) and currently exists only in CUDA via `cudarc`. Closing
the Vulkan + cubecl cells lets the workspace run end-to-end LBM on:

- Linux without an NVIDIA card (Vulkan).
- macOS arm64 via cubecl-wgpu -> Metal.
- Windows DX12 via cubecl-wgpu.
- WebGPU in browsers via cubecl-wgpu.

## Algorithm in 30 lines

D3Q19 LBM has 19 discrete velocities `c_i` in 3D, with weights `w_i`:

```
c_0 = (0, 0, 0),                              w_0 = 1/3
c_1..6 = +/-axes (6 directions),              w_i = 1/18
c_7..18 = face diagonals (12 directions),     w_i = 1/36
```

State: per-cell distribution functions `f_i[x,y,z]` (19 channels). Per step:

```
1. Collide (BGK):  f_i_post = f_i - omega * (f_i - f_eq_i(rho, u))
2. Stream:         f_i[x + c_i] = f_i_post[x]
```

where `f_eq_i = w_i * rho * (1 + 3*c_i.u + 4.5*(c_i.u)^2 - 1.5*u.u)`.

## Scope tree

### Tier 1: CPU reference (already exists)

- `crates/lbm_3d/src/solver.rs` -- pure-Rust f64 reference.
  No work needed. Validation oracle.

### Tier 2: Vulkan WGSL shader + ash device pipeline (~700 lines)

- New `crates/lbm_vulkan/shaders/lbm_d3q19.wgsl` (~80 lines compute shader).
- New `crates/lbm_vulkan/src/lbm_d3q19_vulkan.rs` -- builds compute pipeline,
  binds f-distribution buffers + uniforms, dispatches per timestep.
- Parity test `crates/lbm_vulkan/tests/lbm_d3q19_vulkan_parity.rs` gated
  `#[ignore = "gpu"]`. Compares CPU oracle vs Vulkan after N=10 steps
  on a 32^3 grid; tolerance ~1e-6 relative per cell.

### Tier 3: cubecl `#[cube]` kernel + launcher (~400 lines)

- New `crates/lbm_vulkan/src/lbm_d3q19_cubecl.rs` -- `#[cube]` kernel
  mirroring the WGSL math; launcher using `cubecl-wgpu` runtime.
- The `f_distributions` array layout must match between paths: AoSoA
  `[cell_idx][channel]` or SoA `[channel][cell_idx]`. SoA is the standard
  LBM choice for stream-step locality.

### Tier 4: 3-way parity test

- `crates/lbm_vulkan/tests/lbm_d3q19_parity.rs` compares CPU vs Vulkan
  vs cubecl on the same seeded initial state, N steps, byte-checks the
  rho + u_x + u_y + u_z derived moments.

## Open design decisions

1. **fp32 vs fp64**: production CUDA uses fp64 for fluid-stability reasons.
   Vulkan compute shaders support fp64 only with `shader-f64` capability;
   cubecl-wgpu supports it on adapters that expose it but it's not
   universal. Recommendation: fp32 default, fp64 feature-gated.

2. **Boundary conditions**: at minimum need periodic + bounce-back.
   Periodic is simple modular addressing; bounce-back requires per-face
   no-slip handling. Defer to a follow-up if the Vulkan first pass is
   periodic-only.

3. **Streaming approach**: pull-streaming (each thread reads from neighbors)
   vs push-streaming (each thread writes to neighbors). Pull is the
   conventional WGSL choice (no atomic writes); CUDA codebase uses pull.

4. **AA pattern vs AB pattern**: AA-pattern uses one f-buffer (half the
   memory) but requires careful swap; AB-pattern uses two buffers. The
   existing CUDA code uses AB; recommend matching for parity.

5. **fp32 numerical drift over many steps**: byte-exact CPU-Vulkan parity
   is unrealistic past ~100 steps (f32 accumulates ~1e-5 per step).
   Tolerance must scale with `sqrt(n_steps) * eps_f32`.

## Estimated effort

- Tier 2 (Vulkan WGSL + ash pipeline): 3-5 days.
- Tier 3 (cubecl kernel + launcher): 1-2 days (lots of cube-kernel
  pitfalls already documented from box-counting + chingon ports).
- Tier 4 (3-way parity): 1 day (template exists).

Total: ~1 week. Each tier is independently merge-able; the scoping
strategy is to land Tier 2 first (closes the "Vulkan" column), then
Tier 3 (closes the "cubecl" column), then Tier 4 (the integration
test). Once Tier 4 lands, this kernel transitions from "deferred"
in the parity matrix to "full 3-way parity" alongside TurboQuant +
box-counting + chingon, lifting the score from 3/15 to 4/15.

## What's NOT in scope

- Multi-GPU LBM (spatial decomposition + halo exchange).
- Refinement (multi-grid LBM).
- MRT collision (separate kernel; followup-of-followup).
- LBM-flavored MHD / turbulence closures.

## Acceptance criteria for the Tier 2 + 3 + 4 PR sequence

1. CPU oracle in `lbm_3d::solver` unchanged.
2. WGSL shader compiles via `naga::front::wgsl::parse_str`.
3. cubecl `#[cube]` kernel compiles + launches.
4. Parity test passes within tolerance on at least 5 distinct seeds.
5. `lbm_vulkan` test count grows from 79 (current) to >=85.
6. Workspace clippy --all-targets clean with -D warnings.
7. Workspace inheritance preserved in any Cargo.toml edits.

## When this lands

Score in `docs/engineering/cubecl_vulkan_cuda_parity_matrix.md` updates:

- "LBM D3Q19 stream + collide" row: YES on all four columns + parity test.
- Summary count: 4 of 15 kernels with full 3-way parity (was 3 after #18).
