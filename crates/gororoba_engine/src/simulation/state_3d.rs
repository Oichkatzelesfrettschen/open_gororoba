#[cfg(feature = "gpu")]
use anyhow::Context;
use anyhow::Result;
use lbm_3d_cuda::{LbmSolver3DCuda, Precision};
use lbm_3d::solver::LbmSolver3D;
use algebra_core::physics::octonion_field::FieldParams;
use ndarray::Array3;

/// Configuration for the simulation, used to create `SimulationState3D`.
#[derive(Debug, Clone)]
pub struct SimulationConfig3D {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub tau: f64,
    pub use_gpu: bool,
    pub precision: Precision,
    pub algebra_params: FieldParams,
    pub coupling_fluid_algebra: f32,
    pub coupling_algebra_fluid: f32,
}

/// Represents the state of the 3D LBM simulation.
#[allow(clippy::large_enum_variant)]
pub enum LbmBackend3D {
    Cpu(LbmSolver3D),
    Gpu(LbmSolver3DCuda),
}

/// Trait for a 3D frustration field.
pub trait FrustrationField3D: Send + Sync {
    fn apply(&mut self, fluid: &mut LbmBackend3D, nx: usize, ny: usize, nz: usize) -> Result<()>;
}

/// Trait for a 3D algebraic field.
pub trait AlgebraicField3D {
}

impl LbmBackend3D {
    pub fn step(&mut self) -> Result<()> {
        match self {
            LbmBackend3D::Cpu(solver) => {
                solver.evolve_one_step();
                Ok(())
            },
            LbmBackend3D::Gpu(solver) => solver.step(),
        }
    }

    pub fn try_mean_density(&mut self) -> Result<f64> {
        match self {
            LbmBackend3D::Cpu(solver) => {
                Ok(solver.total_mass() / (solver.nx * solver.ny * solver.nz) as f64)
            }
            #[cfg(feature = "gpu")]
            LbmBackend3D::Gpu(solver) => Ok(
                solver
                    .calculate_mean_density()
                    .context("failed to compute GPU mean density")? as f64,
            ),
            #[cfg(not(feature = "gpu"))]
            LbmBackend3D::Gpu(_) => Err(anyhow::anyhow!(
                "GPU backend unavailable: build gororoba_engine with --features gpu"
            )),
        }
    }

    pub fn mean_density(&mut self) -> f64 {
        self.try_mean_density().unwrap_or(1.0)
    }

    pub fn try_velocity(
        &mut self,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> Result<(Array3<f64>, Array3<f64>, Array3<f64>)> {
        let mut ux = Array3::zeros((nx, ny, nz));
        let mut uy = Array3::zeros((nx, ny, nz));
        let mut uz = Array3::zeros((nx, ny, nz));

        match self {
            LbmBackend3D::Cpu(solver) => {
                for (idx, vel) in solver.u.iter().enumerate() {
                    let z = idx / (nx * ny); let y = (idx % (nx * ny)) / nx; let x = idx % nx;
                    ux[[x, y, z]] = vel[0]; uy[[x, y, z]] = vel[1]; uz[[x, y, z]] = vel[2];
                }
            }
            #[cfg(feature = "gpu")]
            LbmBackend3D::Gpu(solver) => {
                solver
                    .sync_to_host()
                    .context("failed to sync GPU velocity field to host")?;
                for (idx, vel) in solver.u.iter().enumerate() {
                    let z = idx / (nx * ny); let y = (idx % (nx * ny)) / nx; let x = idx % nx;
                    ux[[x, y, z]] = vel[0] as f64; uy[[x, y, z]] = vel[1] as f64; uz[[x, y, z]] = vel[2] as f64;
                }
            }
            #[cfg(not(feature = "gpu"))]
            LbmBackend3D::Gpu(_) => {
                return Err(anyhow::anyhow!(
                    "GPU backend unavailable: build gororoba_engine with --features gpu"
                ));
            }
        }
        Ok((ux, uy, uz))
    }

    pub fn get_velocity(
        &mut self,
        nx: usize,
        ny: usize,
        nz: usize,
    ) -> (Array3<f64>, Array3<f64>, Array3<f64>) {
        self.try_velocity(nx, ny, nz).unwrap_or_else(|_| {
            (
                Array3::zeros((nx, ny, nz)),
                Array3::zeros((nx, ny, nz)),
                Array3::zeros((nx, ny, nz)),
            )
        })
    }

    #[cfg(feature = "gpu")]
    pub fn apply_spectral_mask(&self, u_hat: &mut cudarc::driver::CudaSlice<lbm_3d_cuda::ComplexDevice>, mask: &cudarc::driver::CudaSlice<f32>, damping: f32) -> anyhow::Result<()> {
        match self {
            LbmBackend3D::Cpu(_) => Err(anyhow::anyhow!("Spectral mask GPU not available on CPU backend")),
            LbmBackend3D::Gpu(solver) => solver.apply_spectral_mask(u_hat, mask, damping),
        }
    }

    #[cfg(feature = "gpu")]
    pub fn convert_real_to_complex(&self, d_u_hat: &mut cudarc::driver::CudaSlice<lbm_3d_cuda::ComplexDevice>, component: usize) -> anyhow::Result<()> {
        match self {
            LbmBackend3D::Cpu(_) => Err(anyhow::anyhow!("Convert real to complex GPU not available on CPU backend")),
            LbmBackend3D::Gpu(solver) => solver.convert_real_to_complex(d_u_hat, component),
        }
    }

    #[cfg(feature = "gpu")]
    pub fn convert_complex_to_real(&mut self, d_u_hat: &cudarc::driver::CudaSlice<lbm_3d_cuda::ComplexDevice>, component: usize, scale: f32) -> anyhow::Result<()> {
        match self {
            LbmBackend3D::Cpu(_) => Err(anyhow::anyhow!("Convert complex to real GPU not available on CPU backend")),
            LbmBackend3D::Gpu(solver) => solver.convert_complex_to_real(d_u_hat, component, scale),
        }
    }

    #[cfg(feature = "gpu")]
    pub fn stream(&self) -> &std::sync::Arc<cudarc::driver::CudaStream> {
        match self {
            LbmBackend3D::Cpu(_) => panic!("CPU backend does not have a CUDA stream"),
            LbmBackend3D::Gpu(solver) => solver.stream(),
        }
    }
    
    #[cfg(feature = "gpu")]
    pub fn calculate_enstrophy(&mut self) -> anyhow::Result<f32> {
        match self {
            LbmBackend3D::Cpu(_) => Err(anyhow::anyhow!("Enstrophy calculation GPU not available on CPU backend")),
            LbmBackend3D::Gpu(solver) => solver.calculate_enstrophy(),
        }
    }
}

pub struct SimulationState3D {
    pub fluid: LbmBackend3D,
    pub frustration: Option<Box<dyn FrustrationField3D>>,
    pub curvature_field: Vec<f64>,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub current_step: usize,
}

impl SimulationState3D {
    pub fn new(config: SimulationConfig3D) -> Result<Self> {
        let (nx, ny, nz) = (config.nx, config.ny, config.nz);
        let n_cells = nx * ny * nz;

        // Minkowski curvature is zero everywhere (analytic).
        let curvature_field = vec![0.0; n_cells];

        let fluid = if config.use_gpu {
            #[cfg(feature = "gpu")]
            { 
                let solver = LbmSolver3DCuda::new(nx, ny, nz, config.tau, config.precision)
                    .map_err(|e| anyhow::anyhow!("failed to initialize GPU LBM solver: {e}"))?;
                LbmBackend3D::Gpu(solver)
            }
            #[cfg(not(feature = "gpu"))]
            { return Err(anyhow::anyhow!("GPU features not enabled. Build with --features gpu")) }
        } else {
            LbmBackend3D::Cpu(LbmSolver3D::new(nx, ny, nz, config.tau))
        };

        Ok(Self {
            fluid,
            frustration: None,
            curvature_field,
            nx,
            ny,
            nz,
            current_step: 0,
        })
    }

    pub fn step(&mut self) -> Result<()> {
        self.fluid.step()?;
        if let Some(ref mut f) = self.frustration {
            f.apply(&mut self.fluid, self.nx, self.ny, self.nz)?;
        }
        self.current_step += 1;
        Ok(())
    }
}
