# `gororoba_sparse_grid` Extraction Scope

## Purpose

This scope defines a lightweight, backend-neutral crate for sparse grid
metadata, occupancy planning, and tile-window bookkeeping.

The crate should help CUDA, CPU, Vulkan, and science-planning code share the
same sparse-grid vocabulary without importing density-threshold semantics,
kernel launch policy, or LBM-specific storage rules.

## Why Extract It

The repo already uses the same sparse concepts in several places:

- CUDA sparse LBM active-brick execution
- OptiX brick occupancy and BVH rebuild policy
- heliosphere feature-cube sparse-memory planning
- managed tiled fallback planning for large grids

That is enough repeated structure to justify a small metadata crate.

## Candidate Public Surface

### Core geometry

- `BrickShape3d`
  - edge length in cells
  - halo edge length in cells
- `LogicalGrid3d`
  - `nx`, `ny`, `nz`
- `BrickGrid3d`
  - bricks per axis
  - total brick count

### Occupancy metadata

- `OccupancyBitsetStats`
  - total bricks
  - active bricks
  - occupancy fraction
- `IndirectBrickTableShape`
  - entry count
  - bytes per entry
- `ActiveBrickWindow`
  - active-brick start
  - active-brick count
  - active-cell start
  - active-cell count

### Planning metadata

- `SparseTilePlan`
  - number of windows
  - peak active bricks per window
  - recommended tile bytes
  - whether metadata hotset fits L2
- `SparseMetadataFootprint`
  - occupancy-bitset MiB
  - indirect-table MiB
  - active-brick-ID MiB

## Good Extraction Candidates

### From `lbm_3d_cuda::sparse`

- brick-grid dimension math
- active-brick window bookkeeping
- metadata-footprint estimation

### From `data_core::heliosphere_feature_cube`

- sparse memory planning structs that describe brick/halo geometry
- hardware-agnostic occupancy and tile-count planning

### From OptiX support

- generic brick-count and occupancy bookkeeping
- not the tracer payloads or density-threshold logic

## Must Stay Local

### Solver-local

- D3Q19 storage bytes-per-cell assumptions
- BF16 vs FP32 sparse-state formulas tied to current solver layout
- kernel launch and tile-prefetch policy
- density/velocity thresholds that determine "active fluid"

### OptiX-local

- BVH build and SBT ownership
- tracer-specific brick AABB payload layouts

### Science-local

- heliosphere event-mask rules
- product-aware occupancy weights

## Extraction Order

1. Move geometry and occupancy-footprint types first.
2. Move active-brick window bookkeeping next.
3. Keep solver-specific byte formulas behind adapters until a second workload
   proves they belong in the shared crate.

## Success Criteria

This extraction is worth doing only if the crate remains:

- free of CUDA runtime bindings
- free of Vulkan runtime bindings
- free of solver-specific physical semantics
- useful to at least two owners, such as CUDA sparse LBM and heliosphere sparse
  planning

If it starts absorbing threshold policy or kernel launch logic, it has gone too
far.
