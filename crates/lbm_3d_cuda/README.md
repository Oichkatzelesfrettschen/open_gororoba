# lbm_3d_cuda

GPU D3Q19 Lattice Boltzmann backend for `open_gororoba`.

This crate uses CUDA runtime compilation through `cudarc` and is API-aligned with
CPU-side LBM paths used in the workspace.

## Backend Policy (Human + LLM)

- Prefer `cudarc` as the CUDA integration layer.
- Prefer runtime NVRTC/PTX loading through `cudarc` over build-time `nvcc` pipelines.
- Do not introduce parallel CUDA stacks unless explicitly required by scope.
- Keep GPU lattice constants aligned with the CPU reference in `crates/lbm_3d`.

## Current Status

- GPU/CPU equivalence tests pass for uniform, gradient, high-contrast, and large-grid cases.
- Deterministic behavior is verified in repeated GPU runs.
- D3Q19 direction tables are synchronized with CPU expectations.

## Notes on Physics Terms

- `tau` is relaxation time.
- `nu` is kinematic viscosity.
- Standard conversion in lattice units is: `nu = (tau - 0.5) / 3.0`.

## Performance Data

Canonical benchmark snapshots should be recorded in:

- `registry/lbm_gpu_performance.toml`

Optional exhaustive benchmarking is available via Criterion benches in this crate.

## Required Validation Before Merge

- `cargo clippy -p lbm_3d_cuda -- -D warnings`
- `cargo test -p lbm_3d_cuda`
- `cargo test -p lbm_3d_cuda --test test_gpu_cpu_equivalence -- --nocapture`
- `make ascii-check`

## Troubleshooting

If parity regresses, verify in this order:

1. D3Q19 direction vectors and weights in CUDA kernels.
2. Collision and streaming phase semantics (CPU vs GPU ordering).
3. Memory layout and index mapping between host and device buffers.
4. Viscosity field packing and unpacking paths.

Do not loosen tolerances before direction tables and phase logic are confirmed.
