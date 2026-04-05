<!-- AUTO-GENERATED: DO NOT EDIT -->
<!-- Source of truth: registry/requirements.toml; registry/requirements_narrative.toml -->

# Requirements: lbm_3d_cuda

`lbm_3d_cuda` is the CUDA-accelerated D3Q19 lattice Boltzmann lane. It is an
optional GPU crate and should be treated as a crate-scoped capability rather
than part of the mandatory host baseline.

Primary verification commands:

```ignore
cargo check -p lbm_3d_cuda
cargo clippy -p lbm_3d_cuda --all-targets -- -D warnings
```

Host assumptions:

- NVIDIA GPU and driver stack visible to the process
- CUDA runtime/NVRTC support compatible with `cudarc`
- Rust nightly toolchain matching the workspace policy

Notes:

- Runtime kernel compilation is performed through `cudarc` NVRTC paths.
- Production and benchmark precision tiers are intentionally separate surfaces.
- Keep GPU validation as an explicit lane so CPU-only hosts can still validate
  the rest of the workspace.
