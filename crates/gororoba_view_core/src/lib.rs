//! Backend-neutral viewer contracts for gororoba simulation frontends.
//!
//! This crate is intentionally lightweight. It does not know about CUDA,
//! Vulkan, OptiX, LBM, or any specific science lane. It only defines the
//! shared vocabulary needed by interactive frontends and backend adapters:
//!
//! - grid shape
//! - frame metadata
//! - volume, slice, and particle frame packets
//! - a trait for stepping a simulation and copying a frame

use anyhow::Result;
use gororoba_gpu_bridge::{ExecutionProfile, FrameMode};

pub use gororoba_gpu_readback::{
    ReadbackBufferShape, ReadbackDescriptor, ReadbackElementType, ReadbackLayout,
    ReadbackResidency,
};

/// Runtime grid dimensions for a 3D simulation or data volume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GridShape3d {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
}

impl GridShape3d {
    /// Total number of cells in the grid.
    #[must_use]
    pub fn cell_count(self) -> u64 {
        self.nx as u64 * self.ny as u64 * self.nz as u64
    }
}

/// Logical scalar field carried by a volume frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScalarFieldKind {
    Density,
    Entropy,
    Speed,
    Temperature,
    MagneticMagnitude,
    Custom,
}

/// Shared runtime metadata for a viewer session.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameMetadata {
    pub title: String,
    pub backend_name: String,
    pub grid: GridShape3d,
    pub step: u64,
    pub sim_time: f64,
    pub preferred_frame_mode: FrameMode,
    pub execution: ExecutionProfile,
    pub fps_hint: Option<f32>,
    pub mlups_hint: Option<f64>,
    pub readback: Option<ReadbackDescriptor>,
    pub particle_metadata: Option<ParticleFrameMetadata>,
}

/// Dense 3D scalar field for volume-oriented viewers.
#[derive(Debug, Clone, PartialEq)]
pub struct VolumeFrameF32 {
    pub grid: GridShape3d,
    pub field: ScalarFieldKind,
    pub values: Vec<f32>,
}

/// 2D color frame for slice or rendered-image viewers.
#[derive(Debug, Clone, PartialEq)]
pub struct SliceFrameRgba8 {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

/// Particle/tracer frame for OptiX or CPU/GPU particle overlays.
#[derive(Debug, Clone, PartialEq)]
pub struct ParticleFrame {
    pub positions: Vec<[f32; 3]>,
    pub velocities: Vec<[f32; 3]>,
}

/// Semantic role of a particle frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParticleSemantic {
    Tracer,
    Seed,
    Sample,
    Generic,
}

/// Coordinate-space interpretation for particle positions or velocities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoordinateSpace3d {
    GridIndex,
    World,
    NormalizedUnitCube,
    Unknown,
}

/// Optional metadata that lets frontends and tooling interpret particle-frame
/// payloads without backend-specific assumptions.
#[derive(Debug, Clone, PartialEq)]
pub struct ParticleFrameMetadata {
    pub semantic: ParticleSemantic,
    pub position_space: CoordinateSpace3d,
    pub velocity_space: CoordinateSpace3d,
    pub particle_count: usize,
    pub bounds_min: Option<[f32; 3]>,
    pub bounds_max: Option<[f32; 3]>,
    pub snapshot_interval_steps: Option<u64>,
}

/// Backend-neutral frame payload returned to interactive frontends.
#[derive(Debug, Clone, PartialEq)]
pub enum ViewerFramePacket {
    VolumeF32(VolumeFrameF32),
    SliceRgba8(SliceFrameRgba8),
    Particles(ParticleFrame),
}

/// Trait implemented by simulation backends that can drive an interactive
/// viewer.
///
/// Frontends use this trait to:
/// - query execution/runtime metadata
/// - advance the simulation
/// - request a frame in a supported presentation mode
pub trait ViewerFrameSource {
    /// Execution profile describing backend, precision, layout, and residency.
    fn execution_profile(&self) -> ExecutionProfile;

    /// Frame modes this source can currently provide.
    fn supported_frame_modes(&self) -> Vec<FrameMode>;

    /// Session metadata suitable for status panels and overlays.
    fn frame_metadata(&self) -> FrameMetadata;

    /// Advance the simulation or data source by `n_steps`.
    fn step_simulation(&mut self, n_steps: usize) -> Result<()>;

    /// Copy a frame in the requested presentation mode.
    fn copy_frame(&mut self, requested_mode: FrameMode) -> Result<ViewerFramePacket>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use gororoba_gpu_bridge::{
        BufferLayout, ComputeBackend, ExecutionProfile, FrameMode, MemoryResidency,
        StoragePrecision,
    };

    #[test]
    fn grid_shape_counts_cells() {
        let grid = GridShape3d {
            nx: 16,
            ny: 8,
            nz: 4,
        };
        assert_eq!(grid.cell_count(), 512);
    }

    #[test]
    fn metadata_carries_execution_profile() {
        let profile = ExecutionProfile {
            backend: ComputeBackend::Cuda,
            storage: StoragePrecision::Fp32,
            layout: BufferLayout::Soa,
            residency: MemoryResidency::DeviceLocal,
            frame_mode: FrameMode::Volume3d,
        };
        let metadata = FrameMetadata {
            title: "test".to_string(),
            backend_name: "CUDA".to_string(),
            grid: GridShape3d {
                nx: 8,
                ny: 8,
                nz: 8,
            },
            step: 7,
            sim_time: 0.125,
            preferred_frame_mode: FrameMode::Volume3d,
            execution: profile,
            fps_hint: Some(60.0),
            mlups_hint: Some(120.0),
            readback: None,
            particle_metadata: None,
        };
        assert_eq!(metadata.execution.backend, ComputeBackend::Cuda);
        assert_eq!(metadata.preferred_frame_mode, FrameMode::Volume3d);
    }

    #[test]
    fn particle_metadata_can_describe_tracer_bounds() {
        let metadata = ParticleFrameMetadata {
            semantic: ParticleSemantic::Tracer,
            position_space: CoordinateSpace3d::World,
            velocity_space: CoordinateSpace3d::World,
            particle_count: 1024,
            bounds_min: Some([0.0, 0.0, 0.0]),
            bounds_max: Some([64.0, 64.0, 64.0]),
            snapshot_interval_steps: Some(1),
        };
        assert_eq!(metadata.semantic, ParticleSemantic::Tracer);
        assert_eq!(metadata.particle_count, 1024);
    }
}
