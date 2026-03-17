# GPU Modularization Task List

## Status Legend

- `done`: implemented and verified
- `active`: currently being executed
- `queued`: planned next
- `defer`: keep local until another workload proves reuse

## Cross-Crate Task List

- `done` Extract generic OptiX runtime/FFI boundary into `gororoba_optix`.
- `done` Add lightweight shared execution vocabulary in `gororoba_gpu_bridge`.
- `done` Record the current GPU stack modularization plan and OptiX boundary in
  engineering docs.
- `done` Add a backend-neutral viewer contract crate, `gororoba_view_core`.
- `done` Add CPU and CUDA `ViewerFrameSource` adapters for the live viewer.
- `done` Add an OptiX particle `ViewerFrameSource` adapter at the frame-contract
  layer.
- `done` Inventory OptiX pipeline-building logic into reusable vs solver-local
  pieces.
- `done` Decide which Vulkan helpers belong in reusable substrate vs
  `lbm_vulkan`.
- `done` Split `lbm-live-viewer` responsibilities into frontend loop,
  camera/input state, frame transport, and backend adapter layers.
- `done` Introduce backend adapters that implement
  `gororoba_view_core::ViewerFrameSource`.
- `done` Rework the live viewer to consume the new viewer contract instead of
  one concrete CUDA+Vulkan path.
- `defer` Extract generic Vulkan capability and image/readback helpers into a
  lighter reusable module or crate.
- `queued` Keep managed sparse SoA and managed tiled fallback working while the
  crate split continues.
- `active` Add rustdoc coverage to each public reusable GPU/viewer contract as
  new crates or modules appear.
- `done` Record a concrete `lbm_*` shared-substrate audit and extraction order.
- `done` Extract shared slice/LUT raster logic into `gororoba_view_raster`.
- `done` Add backend-neutral particle metadata to `gororoba_view_core` so
  particle-producing adapters can describe semantics, coordinate spaces, and
  bounds without backend-specific UI logic.
- `queued` Publish a viewer adapter capability matrix covering CPU, CUDA,
  OptiX, and deferred Vulkan modes.
- `queued` Scope a lightweight `gororoba_gpu_readback` crate for reusable
  CUDA/Vulkan host-staging and copy contracts.
- `queued` Scope a lightweight `gororoba_sparse_grid` crate for reusable sparse
  occupancy, active-brick, and tile-window metadata.

## Seam Inventory

### OptiX <-> CUDA runtime seam

- current owner
  - `gororoba_optix`
- seam purpose
  - load OptiX, query function table, create/destroy current-context device
    runtime
- status
  - `done`
- next work
  - keep generic runtime ownership here
  - only extract more if another workload needs generic pipeline building

### CUDA solver <-> viewer seam

- current owner
  - split between `lbm_3d_cuda` and `lbm-live-viewer`
- current problem
  - viewer binary knows too much about one concrete solver
- target seam
  - `gororoba_view_core::ViewerFrameSource`
- status
  - `done`
- next work
  - first CUDA volume adapter is implemented
  - follow with CPU and OptiX particle adapters

### Vulkan renderer <-> viewer seam

- current owner
  - split between `lbm_vulkan` and `lbm-live-viewer`
- current problem
  - `lbm_vulkan` still owns LBM-specific command submission and render/readback
    policy
- target seam
  - backend adapter that yields `SliceRgba8` or `VolumeF32`
- status
  - `done`
- next work
  - keep `GororobaEngine`, WGSL selection, and render command submission local
  - only extract generic Vulkan readback helpers when a second renderer exists

### CPU fallback <-> viewer seam

- current owner
  - mostly implicit, not yet formalized
- current problem
  - no first-class adapter contract for CPU-only interactive viewing
- target seam
  - CPU adapter implementing `ViewerFrameSource`
- status
  - `done`
- next work
  - expand beyond dense volume if a second CPU-facing frame mode is needed
  - keep CPU metadata aligned with the shared viewer contract

### Shared metadata/frame seam

- current owner
  - `gororoba_view_core` plus `gororoba_gpu_bridge`
- seam purpose
  - unify backend name, precision, layout, residency, frame mode, grid shape,
    and frame payload vocabulary
- status
  - `done`
- next work
  - keep public rustdoc complete as new metadata fields appear
  - add a capability matrix so frontend limits are documented, not implied

### OptiX particle view <-> viewer seam

- current owner
  - not yet implemented as an adapter
- current problem
  - particle/tracer data exists, but there is no backend-neutral frontend
    presentation contract around it yet
- target seam
  - `ViewerFramePacket::Particles`
- status
  - `done`
- next work
  - enrich particle metadata without leaking solver-local semantics
  - keep live OptiX launch/readback orchestration solver-local for now

## OptiX Inventory

### Reusable now

- `gororoba_optix::OptixApi`
- `gororoba_optix::OptixRuntime`
- generic OptiX FFI handles and structs
- OptiX shared-library probing
- current-CUDA-context device-context creation/destruction

### Solver-local for now

- `lbm_3d_cuda::optix_tracer::OptiXTracerConfig`
- `lbm_3d_cuda::optix_tracer::LbmSbtData`
- `lbm_3d_cuda::optix_tracer::BrickAabb`
- `lbm_3d_cuda::optix_tracer::BrickResult`
- density-threshold occupancy logic
- particle advection semantics over LBM velocity fields
- `optix_tracer.cu`
- `optix_brick_scan.cu`
- `EulerianLagrangianOrchestrator`

### Candidate future extraction

- `OptiXCompileOptions`
  - `defer`
  - generic in shape, but still only used for `optix_tracer.cu`
- live pipeline assembly helpers
  - `defer`
  - only extract if a second non-LBM OptiX workload appears
- BVH build utility helpers
  - `defer`
  - current implementation is tightly coupled to LBM density-driven brick sets

## Viewer Frame Contract

The new viewer contract lives in `gororoba_view_core`.

### Current shared types

- `GridShape3d`
- `ScalarFieldKind`
- `FrameMetadata`
- `VolumeFrameF32`
- `SliceFrameRgba8`
- `ParticleFrame`
- `ParticleSemantic`
- `CoordinateSpace3d`
- `ParticleFrameMetadata`
- `ViewerFramePacket`
- `ViewerFrameSource`

### Required backend adapters

- `CudaVolumeAdapter`
  - adapts `LbmSolver3DCuda` or sparse CUDA solver outputs
  - status: `done` for the first dense CUDA adapter in `lbm-live-viewer`
- `VulkanVolumeAdapter`
  - adapts `lbm_vulkan` render/slice outputs
  - status: `queued`
- `CpuFallbackAdapter`
  - supports CPU-only stepping and inspection
  - status: `done` for the first dense CPU adapter in `lbm-live-viewer`
- `OptixParticleAdapter`
  - surfaces particle/tracer views through the same frontend contract
  - status: `done` at the frame-contract layer

## Vulkan Extraction Decisions

### Reusable substrate candidates

- Vulkan hardware capability probing
- device-local VRAM sizing/tiering helpers
- generic image readback helpers
- generic command-pool / command-buffer setup helpers

### Decision

- `VulkanContext` capability probing stays a candidate reusable substrate seam
- `GororobaEngine` command-pool setup, command-buffer submission, and render
  readback stay LBM-local for now
- the first viewer adapter now uses the real CUDA solver volume path instead of
  a Vulkan-owned shadow simulation

### Stay in `lbm_vulkan`

- `GororobaEngine`
- LBM collision/render WGSL selection
- LBM-specific precision dispatch
- LBM-specific accumulation and entropy buffers
- LBM-specific compute pipeline assembly

## `lbm-live-viewer` Split

### To move out of the binary

- camera orbit state
- input mapping
- frontend frame loop
- status panel metadata formatting
- backend-neutral viewer transport

### To keep as application wiring

- command-line parsing
- choosing which backend adapter to instantiate
- top-level app startup and shutdown

## Immediate Execution Order

1. Publish the viewer adapter capability matrix for CPU, CUDA, OptiX, and
   deferred Vulkan paths.
2. Scope `gororoba_gpu_readback` around reusable host-staging contracts instead
   of solver- or renderer-owned submission logic.
3. Scope `gororoba_sparse_grid` around backend-neutral occupancy/tile metadata
   instead of density-threshold semantics.
4. Revisit Vulkan helper extraction only when a second renderer needs the same
   command/readback substrate.
5. Keep shared viewer/frame contracts rustdoc-complete as adapters evolve.
