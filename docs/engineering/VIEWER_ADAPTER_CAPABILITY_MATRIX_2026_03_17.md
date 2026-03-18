# Viewer Adapter Capability Matrix

## Purpose

This matrix records what each currently implemented or planned viewer backend
can actually provide through `gororoba_view_core`, so the frontend contract
stays explicit and reviewable.

The goal is not to promise every mode everywhere. The goal is to keep the
interactive viewer honest about which backends provide volume frames, particle
frames, reset support, host readback, and CPU-safe fallbacks.

## Shared Contract Surface

The viewer frontend consumes these shared contracts:

- `gororoba_view_core::ViewerFrameSource`
- `gororoba_view_core::FrameMetadata`
- `gororoba_view_core::ViewerFramePacket`
- `gororoba_view_core::VolumeFrameF32`
- `gororoba_view_core::SliceFrameRgba8`
- `gororoba_view_core::ParticleFrame`
- `gororoba_view_core::ParticleFrameMetadata`

## Capability Matrix

| Adapter | Owner | Status | Frame modes | Reset | Host readback path | Particle metadata | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `CpuLbmVolumeAdapter` | `lbm-live-viewer/backend.rs` | implemented | `Volume3d` | yes | direct host memory | n/a | CPU-safe fallback; dense AoS state stays solver-local |
| `CudaLbmVolumeAdapter` | `lbm-live-viewer/backend.rs` | implemented | `Volume3d` | yes | solver-owned CUDA readback | none | Real CUDA density viewer; currently volume-only |
| `OptixParticleAdapter` | `lbm-live-viewer/backend.rs` | implemented | `ParticleTrace` | yes | orchestrator host buffers | yes | Particle contract-layer adapter; live launch/pipeline control remains solver-local |
| `VulkanVolumeAdapter` | deferred | planned | likely `Slice2d` or `Volume3d` | unknown | renderer-owned image/readback | n/a | Deferred until a second renderer proves the seam |
| `CpuParticleAdapter` | deferred | planned | `ParticleTrace` | likely yes | direct host memory | yes | Only needed if a non-OptiX particle producer appears |

## Metadata Expectations

### Volume backends

Volume-producing adapters should always populate:

- `title`
- `backend_name`
- `grid`
- `step`
- `sim_time`
- `preferred_frame_mode`
- `execution`
- `readback` when the backend copies from device-visible or host-visible state

They should set:

- `particle_metadata = None`

unless the adapter also provides a particle-capable frame mode.

### Particle backends

Particle-producing adapters should populate `particle_metadata` with:

- semantic role, such as tracer or seed
- position and velocity coordinate spaces
- current particle count
- optional bounds
- optional snapshot interval

They should also populate `readback` when the particle snapshot comes from a
host-visible staging surface or direct host-owned buffers.

This lets the frontend remain backend-neutral even when the particle source is
OptiX-specific underneath.

## Current Boundary Decisions

### Keep local to solver or renderer

- CUDA kernel launch ownership
- CUDA sparse residency policy
- OptiX pipeline assembly, SBT wiring, and launch parameters
- Vulkan command submission and image ownership

### Keep shared in viewer substrate

- frame packet types
- grid metadata
- execution profile vocabulary
- particle semantic/bounds metadata
- slice raster contracts

## Follow-on Work

1. Add a second particle-producing adapter before extracting any generic
   particle-view frontend crate.
2. Revisit Vulkan extraction only if a renderer outside `lbm_vulkan` needs the
   same readback and frame transport seam.
3. Keep this matrix updated whenever a new adapter or frame mode lands.
