use anyhow::Result;
use gororoba_algebra::physics::octonion_field::FieldParams;
use lbm_3d_cuda::{LbmSolver3DCuda, Precision};

/// Configuration for 4D simulation.
#[derive(Debug, Clone)]
pub struct SimulationConfig4D {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub nw: usize, // Number of 3D worlds (slices)
    pub tau: f64,
    pub precision: Precision,
    pub algebra_params: FieldParams,
    pub coupling_fluid_algebra: f32,
    pub coupling_algebra_fluid: f32,
}

/// 4D LBM Backend (GPU only).
///
/// Wraps LbmSolver3DCuda but interprets the Z dimension as (z + nz * w).
#[allow(clippy::large_enum_variant)]
pub enum LbmBackend4D {
    Gpu(LbmSolver3DCuda),
}

impl LbmBackend4D {
    pub fn step(&mut self, nw: usize) -> Result<()> {
        match self {
            LbmBackend4D::Gpu(solver) => solver.step_4d(nw),
        }
    }

    pub fn calculate_mean_density(&mut self) -> Result<f32> {
        match self {
            LbmBackend4D::Gpu(solver) => solver.calculate_mean_density(),
        }
    }
}

pub struct SimulationState4D {
    pub fluid: LbmBackend4D,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub nw: usize,
    pub current_step: usize,
}

impl SimulationState4D {
    pub fn new(config: SimulationConfig4D) -> Result<Self> {
        #[cfg(not(feature = "gpu"))]
        {
            let _ = config;
            anyhow::bail!("4D simulation requires GPU feature");
        }

        #[cfg(feature = "gpu")]
        {
            let (nx, ny, nz, nw) = (config.nx, config.ny, config.nz, config.nw);
            let nz_total = nz * nw;
            let solver = LbmSolver3DCuda::new(nx, ny, nz_total, config.tau, config.precision)
                .map_err(|e| anyhow::anyhow!("failed to initialize GPU LBM solver for 4D: {e}"))?;
            let fluid = LbmBackend4D::Gpu(solver);
            Ok(Self {
                fluid,
                nx,
                ny,
                nz,
                nw,
                current_step: 0,
            })
        }
    }

    pub fn step(&mut self) -> Result<()> {
        self.fluid.step(self.nw)?;
        self.current_step += 1;
        Ok(())
    }
}
