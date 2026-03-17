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
- `active` Inventory OptiX pipeline-building logic into reusable vs solver-local
  pieces.
- `active` Decide which Vulkan helpers belong in reusable substrate vs
  `lbm_vulkan`.
- `active` Split `lbm-live-viewer` responsibilities into frontend loop,
  camera/input state, frame transport, and backend adapter layers.
- `queued` Introduce backend adapters that implement
  `gororoba_view_core::ViewerFrameSource`.
- `queued` Rework the live viewer to consume the new viewer contract instead of
  one concrete CUDA+Vulkan path.
- `queued` Extract generic Vulkan capability and image/readback helpers into a
  lighter reusable module or crate.
- `queued` Keep managed sparse SoA and managed tiled fallback working while the
  crate split continues.
- `queued` Add rustdoc coverage to each public reusable GPU/viewer contract as
  new crates or modules appear.

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
  - `active`
- next work
  - add first CUDA backend adapter
  - move readback and metadata formatting out of the binary

### Vulkan renderer <-> viewer seam

- current owner
  - split between `lbm_vulkan` and `lbm-live-viewer`
- current problem
  - viewer binary owns command-pool setup and direct render/readback loop
- target seam
  - backend adapter that yields `SliceRgba8` or `VolumeF32`
- status
  - `active`
- next work
  - identify generic image/readback helpers
  - keep LBM shader selection in `lbm_vulkan`

### CPU fallback <-> viewer seam

- current owner
  - mostly implicit, not yet formalized
- current problem
  - no first-class adapter contract for CPU-only interactive viewing
- target seam
  - CPU adapter implementing `ViewerFrameSource`
- status
  - `queued`
- next work
  - define minimum CPU viewer capabilities
  - support smaller grids and slice/volume fallback paths

### Shared metadata/frame seam

- current owner
  - `gororoba_view_core` plus `gororoba_gpu_bridge`
- seam purpose
  - unify backend name, precision, layout, residency, frame mode, grid shape,
    and frame payload vocabulary
- status
  - `done`
- next work
  - migrate current viewer and adapters onto these shared contracts

### OptiX particle view <-> viewer seam

- current owner
  - not yet implemented as an adapter
- current problem
  - particle/tracer data exists, but there is no backend-neutral frontend
    presentation contract around it yet
- target seam
  - `ViewerFramePacket::Particles`
- status
  - `queued`
- next work
  - build an `OptixParticleAdapter`
  - expose tracer positions/velocities through `gororoba_view_core`

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
- `ViewerFramePacket`
- `ViewerFrameSource`

### Required backend adapters

- `CudaVolumeAdapter`
  - adapts `LbmSolver3DCuda` or sparse CUDA solver outputs
- `VulkanVolumeAdapter`
  - adapts `lbm_vulkan` render/slice outputs
- `CpuFallbackAdapter`
  - supports CPU-only stepping and inspection
- `OptixParticleAdapter`
  - surfaces particle/tracer views through the same frontend contract

## Vulkan Extraction Decisions

### Reusable substrate candidates

- Vulkan hardware capability probing
- device-local VRAM sizing/tiering helpers
- generic image readback helpers
- generic command-pool / command-buffer setup helpers

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

1. Implement the first backend adapter against `gororoba_view_core`.
2. Move reusable camera/input/frame-loop pieces out of `lbm-live-viewer`.
3. Extract generic Vulkan capability/readback helpers only after the first
   adapter proves the seam.
