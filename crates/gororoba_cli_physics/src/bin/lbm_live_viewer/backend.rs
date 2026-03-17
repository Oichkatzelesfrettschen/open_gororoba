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
    FrameMetadata, GridShape3d, ScalarFieldKind, ViewerFramePacket, ViewerFrameSource,
    VolumeFrameF32,
};
use lbm_3d_cuda::{LbmSolver3DCuda, Precision};
use std::f64::consts::PI;
use std::time::Instant;

/// Viewer-local extension that adds reset semantics on top of the shared frame
/// source contract.
pub trait ResettableViewerSource: ViewerFrameSource {
    /// Reset the simulation to its initial conditions.
    fn reset_simulation(&mut self) -> Result<()>;
}

/// CUDA-backed density-volume adapter for the interactive viewer.
///
/// The adapter uses the real `LbmSolver3DCuda` state, synchronizes the density
/// field to host on demand, and emits a backend-neutral `VolumeFrameF32`.
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
