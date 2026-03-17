//! Backend adapters for `lbm-live-viewer`.
//!
//! This module owns the first concrete `ViewerFrameSource` implementation for
//! the interactive viewer. It adapts the CUDA LBM solver into a backend-neutral
//! volume source and keeps reset/initial-condition policy local to the viewer
//! application.

use anyhow::Result;
use gororoba_gpu_bridge::{
    BufferLayout, ComputeBackend, ExecutionProfile, FrameMode, MemoryResidency, StoragePrecision,
};
use gororoba_view_core::{
    CoordinateSpace3d, FrameMetadata, GridShape3d, ParticleFrameMetadata,
    ParticleSemantic, ScalarFieldKind, ViewerFramePacket, ViewerFrameSource, VolumeFrameF32,
};
use std::f64::consts::PI;
use std::time::Instant;

use lbm_3d::solver::LbmSolver3D;
#[cfg(feature = "gpu")]
use lbm_3d_cuda::optix_orchestrator::{EulerianLagrangianOrchestrator, OrchestratorConfig};
#[cfg(feature = "gpu")]
use lbm_3d_cuda::optix_tracer::OptiXTracerConfig;
#[cfg(feature = "gpu")]
use lbm_3d_cuda::{LbmSolver3DCuda, Precision};

/// Runtime backend selector for the live viewer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewerBackendKind {
    Cpu,
    Cuda,
    OptixParticles,
}

/// Viewer-local extension that adds reset semantics on top of the shared frame
/// source contract.
pub trait ResettableViewerSource: ViewerFrameSource {
    /// Reset the simulation to its initial conditions.
    fn reset_simulation(&mut self) -> Result<()>;
}

/// Build the requested viewer backend as a boxed trait object.
pub fn build_viewer_source(
    backend: ViewerBackendKind,
    grid: usize,
    tau: f64,
    use_mrt: bool,
) -> Result<Box<dyn ResettableViewerSource>> {
    match backend {
        ViewerBackendKind::Cpu => Ok(Box::new(CpuLbmVolumeAdapter::new(grid, tau, use_mrt))),
        ViewerBackendKind::Cuda => build_cuda_viewer_source(grid, tau, use_mrt),
        ViewerBackendKind::OptixParticles => build_optix_particle_source(grid),
    }
}

#[cfg(feature = "gpu")]
fn build_cuda_viewer_source(
    grid: usize,
    tau: f64,
    use_mrt: bool,
) -> Result<Box<dyn ResettableViewerSource>> {
    Ok(Box::new(CudaLbmVolumeAdapter::new(grid, tau, use_mrt)?))
}

#[cfg(not(feature = "gpu"))]
fn build_cuda_viewer_source(
    _grid: usize,
    _tau: f64,
    _use_mrt: bool,
) -> Result<Box<dyn ResettableViewerSource>> {
    anyhow::bail!("CUDA viewer backend requires the 'gpu' feature")
}

#[cfg(feature = "gpu")]
fn build_optix_particle_source(grid: usize) -> Result<Box<dyn ResettableViewerSource>> {
    Ok(Box::new(OptixParticleAdapter::new(grid)))
}

#[cfg(not(feature = "gpu"))]
fn build_optix_particle_source(_grid: usize) -> Result<Box<dyn ResettableViewerSource>> {
    anyhow::bail!("OptiX particle viewer backend requires the 'gpu' feature")
}

/// CPU-backed dense volume adapter for the interactive viewer.
///
/// This adapter exists to prove the frontend contract is not CUDA-specific and
/// to provide a low-dependency fallback path for development and testing.
pub struct CpuLbmVolumeAdapter {
    solver: LbmSolver3D,
    grid: GridShape3d,
    use_mrt: bool,
    init_rho: Vec<f64>,
    init_u: Vec<[f64; 3]>,
    step: u64,
    sim_time: f64,
    mlups_hint: Option<f64>,
}

impl CpuLbmVolumeAdapter {
    /// Construct a new CPU viewer adapter using Taylor-Green initial
    /// conditions.
    #[must_use]
    pub fn new(grid: usize, tau: f64, use_mrt: bool) -> Self {
        let mut solver = if use_mrt {
            LbmSolver3D::new_mrt(grid, grid, grid, tau)
        } else {
            LbmSolver3D::new(grid, grid, grid, tau)
        };
        let (init_rho, init_u) = taylor_green_initial_conditions(grid);
        solver.rho.clone_from(&init_rho);
        solver.u.clone_from(&init_u);
        solver.reinitialize_from_macroscopic();
        Self {
            solver,
            grid: GridShape3d {
                nx: grid as u32,
                ny: grid as u32,
                nz: grid as u32,
            },
            use_mrt,
            init_rho,
            init_u,
            step: 0,
            sim_time: 0.0,
            mlups_hint: None,
        }
    }

    fn density_frame(&self) -> VolumeFrameF32 {
        VolumeFrameF32 {
            grid: self.grid,
            field: ScalarFieldKind::Density,
            values: self.solver.rho.iter().map(|&v| v as f32).collect(),
        }
    }
}

impl ViewerFrameSource for CpuLbmVolumeAdapter {
    fn execution_profile(&self) -> ExecutionProfile {
        ExecutionProfile {
            backend: ComputeBackend::CpuScalar,
            storage: StoragePrecision::Fp64,
            layout: BufferLayout::Aos,
            residency: MemoryResidency::HostVisible,
            frame_mode: FrameMode::Volume3d,
        }
    }

    fn supported_frame_modes(&self) -> Vec<FrameMode> {
        vec![FrameMode::Volume3d]
    }

    fn frame_metadata(&self) -> FrameMetadata {
        FrameMetadata {
            title: format!(
                "CPU {} density viewer",
                if self.use_mrt { "MRT" } else { "BGK" }
            ),
            backend_name: "CPU LBM".to_string(),
            grid: self.grid,
            step: self.step,
            sim_time: self.sim_time,
            preferred_frame_mode: FrameMode::Volume3d,
            execution: self.execution_profile(),
            fps_hint: None,
            mlups_hint: self.mlups_hint,
            particle_metadata: None,
        }
    }

    fn step_simulation(&mut self, n_steps: usize) -> Result<()> {
        let start = Instant::now();
        self.solver.evolve(n_steps);
        let elapsed = start.elapsed().as_secs_f64();
        self.step += n_steps as u64;
        self.sim_time = self.step as f64;
        if elapsed > 0.0 {
            self.mlups_hint =
                Some(self.grid.cell_count() as f64 * n_steps as f64 / elapsed / 1.0e6);
        }
        Ok(())
    }

    fn copy_frame(&mut self, requested_mode: FrameMode) -> Result<ViewerFramePacket> {
        match requested_mode {
            FrameMode::Volume3d => Ok(ViewerFramePacket::VolumeF32(self.density_frame())),
            _ => anyhow::bail!("CPU viewer adapter only supports FrameMode::Volume3d"),
        }
    }
}

impl ResettableViewerSource for CpuLbmVolumeAdapter {
    fn reset_simulation(&mut self) -> Result<()> {
        self.solver.rho.clone_from(&self.init_rho);
        self.solver.u.clone_from(&self.init_u);
        self.solver.reinitialize_from_macroscopic();
        self.step = 0;
        self.sim_time = 0.0;
        self.mlups_hint = None;
        Ok(())
    }
}

/// OptiX-backed particle-frame adapter.
///
/// This adapter surfaces the orchestrator's host-visible particle buffers
/// through the shared `ParticleFrame` contract. It is intentionally limited to
/// frame transport and viewer integration; live OptiX launch coupling remains in
/// the solver/orchestrator layer.
#[cfg(feature = "gpu")]
pub struct OptixParticleAdapter {
    orchestrator: EulerianLagrangianOrchestrator,
    grid: GridShape3d,
}

#[cfg(feature = "gpu")]
impl OptixParticleAdapter {
    #[must_use]
    pub fn new(grid: usize) -> Self {
        let tracer = OptiXTracerConfig {
            grid_dim: (grid as u32, grid as u32, grid as u32),
            ..OptiXTracerConfig::default()
        };
        let orchestrator = EulerianLagrangianOrchestrator::new(OrchestratorConfig {
            tracer,
            n_particles: (grid * grid).max(256) as u32,
            tracing_enabled: true,
            snapshot_interval: 1,
            velocity_device_ptr: 0,
            density_device_ptr: 0,
        });
        Self {
            orchestrator,
            grid: GridShape3d {
                nx: grid as u32,
                ny: grid as u32,
                nz: grid as u32,
            },
        }
    }
}

#[cfg(feature = "gpu")]
impl ViewerFrameSource for OptixParticleAdapter {
    fn execution_profile(&self) -> ExecutionProfile {
        ExecutionProfile {
            backend: ComputeBackend::Cuda,
            storage: StoragePrecision::Fp32,
            layout: BufferLayout::Soa,
            residency: MemoryResidency::DeviceLocal,
            frame_mode: FrameMode::ParticleTrace,
        }
    }

    fn supported_frame_modes(&self) -> Vec<FrameMode> {
        vec![FrameMode::ParticleTrace]
    }

    fn frame_metadata(&self) -> FrameMetadata {
        FrameMetadata {
            title: "OptiX particle buffer viewer".to_string(),
            backend_name: "OptiX tracer".to_string(),
            grid: self.grid,
            step: self.orchestrator.lbm_step,
            sim_time: self.orchestrator.lbm_step as f64,
            preferred_frame_mode: FrameMode::ParticleTrace,
            execution: self.execution_profile(),
            fps_hint: None,
            mlups_hint: None,
            particle_metadata: Some(ParticleFrameMetadata {
                semantic: ParticleSemantic::Tracer,
                position_space: CoordinateSpace3d::World,
                velocity_space: CoordinateSpace3d::World,
                particle_count: self.orchestrator.particle_positions.len(),
                bounds_min: Some([0.0, 0.0, 0.0]),
                bounds_max: Some([
                    self.grid.nx as f32,
                    self.grid.ny as f32,
                    self.grid.nz as f32,
                ]),
                snapshot_interval_steps: Some(self.orchestrator.snapshot_interval),
            }),
        }
    }

    fn step_simulation(&mut self, n_steps: usize) -> Result<()> {
        for _ in 0..n_steps {
            self.orchestrator.advance_step();
        }
        Ok(())
    }

    fn copy_frame(&mut self, requested_mode: FrameMode) -> Result<ViewerFramePacket> {
        match requested_mode {
            FrameMode::ParticleTrace => Ok(ViewerFramePacket::Particles(
                gororoba_view_core::ParticleFrame {
                    positions: self.orchestrator.particle_positions.clone(),
                    velocities: self.orchestrator.particle_velocities.clone(),
                },
            )),
            _ => anyhow::bail!("OptiX particle adapter only supports ParticleTrace frames"),
        }
    }
}

#[cfg(feature = "gpu")]
impl ResettableViewerSource for OptixParticleAdapter {
    fn reset_simulation(&mut self) -> Result<()> {
        *self = Self::new(self.grid.nx as usize);
        Ok(())
    }
}

/// CUDA-backed density-volume adapter for the interactive viewer.
///
/// The adapter uses the real `LbmSolver3DCuda` state, synchronizes the density
/// field to host on demand, and emits a backend-neutral `VolumeFrameF32`.
#[cfg(feature = "gpu")]
pub struct CudaLbmVolumeAdapter {
    solver: LbmSolver3DCuda,
    grid: GridShape3d,
    use_mrt: bool,
    init_rho: Vec<f64>,
    init_u: Vec<[f64; 3]>,
    step: u64,
    sim_time: f64,
    mlups_hint: Option<f64>,
}

#[cfg(feature = "gpu")]
impl CudaLbmVolumeAdapter {
    /// Construct a new CUDA viewer adapter using Taylor-Green initial
    /// conditions.
    pub fn new(grid: usize, tau: f64, use_mrt: bool) -> Result<Self> {
        let mut solver = if use_mrt {
            LbmSolver3DCuda::new_mrt(grid, grid, grid, tau, Precision::FP32)?
        } else {
            LbmSolver3DCuda::new(grid, grid, grid, tau, Precision::FP32)?
        };
        let (init_rho, init_u) = taylor_green_initial_conditions(grid);
        solver.initialize_custom(&init_rho, &init_u)?;
        Ok(Self {
            solver,
            grid: GridShape3d {
                nx: grid as u32,
                ny: grid as u32,
                nz: grid as u32,
            },
            use_mrt,
            init_rho,
            init_u,
            step: 0,
            sim_time: 0.0,
            mlups_hint: None,
        })
    }

    fn density_frame(&mut self) -> Result<VolumeFrameF32> {
        self.solver.sync_to_host()?;
        Ok(VolumeFrameF32 {
            grid: self.grid,
            field: ScalarFieldKind::Density,
            values: self.solver.rho_host().to_vec(),
        })
    }
}

#[cfg(feature = "gpu")]
impl ViewerFrameSource for CudaLbmVolumeAdapter {
    fn execution_profile(&self) -> ExecutionProfile {
        let storage = match self.solver.precision() {
            Precision::FP32 => StoragePrecision::Fp32,
            Precision::BF16 => StoragePrecision::Bf16,
            Precision::FP64 => StoragePrecision::Fp64,
        };
        ExecutionProfile {
            backend: ComputeBackend::Cuda,
            storage,
            layout: if self.solver.uses_soa_layout() {
                BufferLayout::Soa
            } else {
                BufferLayout::Aos
            },
            residency: MemoryResidency::DeviceLocal,
            frame_mode: FrameMode::Volume3d,
        }
    }

    fn supported_frame_modes(&self) -> Vec<FrameMode> {
        vec![FrameMode::Volume3d]
    }

    fn frame_metadata(&self) -> FrameMetadata {
        FrameMetadata {
            title: format!(
                "CUDA {} density viewer",
                if self.use_mrt { "MRT" } else { "BGK" }
            ),
            backend_name: "CUDA LBM".to_string(),
            grid: self.grid,
            step: self.step,
            sim_time: self.sim_time,
            preferred_frame_mode: FrameMode::Volume3d,
            execution: self.execution_profile(),
            fps_hint: None,
            mlups_hint: self.mlups_hint,
            particle_metadata: None,
        }
    }

    fn step_simulation(&mut self, n_steps: usize) -> Result<()> {
        let start = Instant::now();
        for _ in 0..n_steps {
            self.solver.step()?;
        }
        let elapsed = start.elapsed().as_secs_f64();
        self.step += n_steps as u64;
        self.sim_time = self.step as f64;
        if elapsed > 0.0 {
            self.mlups_hint =
                Some(self.grid.cell_count() as f64 * n_steps as f64 / elapsed / 1.0e6);
        }
        Ok(())
    }

    fn copy_frame(&mut self, requested_mode: FrameMode) -> Result<ViewerFramePacket> {
        match requested_mode {
            FrameMode::Volume3d => Ok(ViewerFramePacket::VolumeF32(self.density_frame()?)),
            _ => anyhow::bail!("CUDA viewer adapter only supports FrameMode::Volume3d"),
        }
    }
}

#[cfg(feature = "gpu")]
impl ResettableViewerSource for CudaLbmVolumeAdapter {
    fn reset_simulation(&mut self) -> Result<()> {
        self.solver
            .initialize_custom(&self.init_rho, &self.init_u)?;
        self.step = 0;
        self.sim_time = 0.0;
        self.mlups_hint = None;
        Ok(())
    }
}

fn taylor_green_initial_conditions(grid: usize) -> (Vec<f64>, Vec<[f64; 3]>) {
    let n_cells = grid * grid * grid;
    let u0 = 0.04;
    let kx = 2.0 * PI / grid as f64;
    let ky = 2.0 * PI / grid as f64;
    let mut rho = vec![1.0f64; n_cells];
    let mut u = vec![[0.0f64; 3]; n_cells];
    for z in 0..grid {
        for y in 0..grid {
            for x in 0..grid {
                let idx = z * grid * grid + y * grid + x;
                let ux = u0 * (kx * x as f64).cos() * (ky * y as f64).sin();
                let uy = -u0 * (kx * x as f64).sin() * (ky * y as f64).cos();
                rho[idx] =
                    1.0 + 0.01 * (kx * x as f64).cos() * (ky * y as f64).cos();
                u[idx] = [ux, uy, 0.0];
            }
        }
    }
    (rho, u)
}
