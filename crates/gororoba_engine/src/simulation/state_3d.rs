#[cfg(feature = "gpu")]
use anyhow::Context;
use anyhow::Result;
use gororoba_algebra::physics::octonion_field::FieldParams;
use lbm_3d::solver::LbmSolver3D;
use lbm_3d_cuda::{LbmSolver3DCuda, Precision};
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

/// Trait for a 3D imbalance field.
pub trait ImbalanceField3D: Send + Sync {
    fn apply(&mut self, fluid: &mut LbmBackend3D, nx: usize, ny: usize, nz: usize) -> Result<()>;

    /// Optional scalar diagnostic exported in warp traces as `algebra_norm`.
    ///
    /// Implementations may return a physically motivated coupling-strength proxy.
    fn trace_algebra_norm(&self) -> f64 {
        0.0
    }
}

/// Trait for a 3D algebraic field.
pub trait AlgebraicField3D {}

impl LbmBackend3D {
    pub fn step(&mut self) -> Result<()> {
        match self {
            LbmBackend3D::Cpu(solver) => {
                solver.evolve_one_step();
                Ok(())
            }
            LbmBackend3D::Gpu(solver) => solver.step(),
        }
    }

    pub fn try_mean_density(&mut self) -> Result<f64> {
        match self {
            LbmBackend3D::Cpu(solver) => {
                Ok(solver.total_mass() / (solver.nx * solver.ny * solver.nz) as f64)
            }
            #[cfg(feature = "gpu")]
            LbmBackend3D::Gpu(solver) => Ok(solver
                .calculate_mean_density()
                .context("failed to compute GPU mean density")?
                as f64),
            #[cfg(not(feature = "gpu"))]
            LbmBackend3D::Gpu(_) => Err(anyhow::anyhow!(
                "GPU backend unavailable: build gororoba_engine with --features gpu"
            )),
        }
    }

    pub fn mean_density(&mut self) -> f64 {
        self.try_mean_density().unwrap_or(1.0)
    }

    pub fn try_velocity_rms(&mut self, nx: usize, ny: usize, nz: usize) -> Result<f64> {
        fn rms_from_velocity_field<T: Copy + Into<f64>>(u: &[[T; 3]], n_cells: usize) -> f64 {
            if n_cells == 0 || u.is_empty() {
                return 0.0;
            }
            let sum_sq: f64 = u
                .iter()
                .map(|v| {
                    let ux: f64 = v[0].into();
                    let uy: f64 = v[1].into();
                    let uz: f64 = v[2].into();
                    ux * ux + uy * uy + uz * uz
                })
                .sum();
            (sum_sq / n_cells as f64).sqrt()
        }

        let expected_n_cells = nx.saturating_mul(ny).saturating_mul(nz);
        match self {
            LbmBackend3D::Cpu(solver) => {
                let n_cells = if expected_n_cells == solver.u.len() {
                    expected_n_cells
                } else {
                    solver.u.len()
                };
                Ok(rms_from_velocity_field(&solver.u, n_cells))
            }
            #[cfg(feature = "gpu")]
            LbmBackend3D::Gpu(solver) => {
                solver
                    .sync_to_host()
                    .context("failed to sync GPU velocity field for RMS trace")?;
                let n_cells = if expected_n_cells == solver.u.len() {
                    expected_n_cells
                } else {
                    solver.u.len()
                };
                Ok(rms_from_velocity_field(&solver.u, n_cells))
            }
            #[cfg(not(feature = "gpu"))]
            LbmBackend3D::Gpu(_) => Err(anyhow::anyhow!(
                "GPU backend unavailable: build gororoba_engine with --features gpu"
            )),
        }
    }

    pub fn try_enstrophy(&mut self) -> Result<f64> {
        fn enstrophy_from_cpu_velocity(solver: &LbmSolver3D) -> f64 {
            let (nx, ny, nz) = (solver.nx, solver.ny, solver.nz);
            let idx = |x: usize, y: usize, z: usize| -> usize { x + nx * (y + ny * z) };
            let mut enstrophy = 0.0f64;
            for z in 0..nz {
                for y in 0..ny {
                    for x in 0..nx {
                        let xp = (x + 1) % nx;
                        let xm = (x + nx - 1) % nx;
                        let yp = (y + 1) % ny;
                        let ym = (y + ny - 1) % ny;
                        let zp = (z + 1) % nz;
                        let zm = (z + nz - 1) % nz;
                        let duz_dy =
                            (solver.u[idx(x, yp, z)][2] - solver.u[idx(x, ym, z)][2]) * 0.5;
                        let duy_dz =
                            (solver.u[idx(x, y, zp)][1] - solver.u[idx(x, y, zm)][1]) * 0.5;
                        let dux_dz =
                            (solver.u[idx(x, y, zp)][0] - solver.u[idx(x, y, zm)][0]) * 0.5;
                        let duz_dx =
                            (solver.u[idx(xp, y, z)][2] - solver.u[idx(xm, y, z)][2]) * 0.5;
                        let duy_dx =
                            (solver.u[idx(xp, y, z)][1] - solver.u[idx(xm, y, z)][1]) * 0.5;
                        let dux_dy =
                            (solver.u[idx(x, yp, z)][0] - solver.u[idx(x, ym, z)][0]) * 0.5;
                        let wx = duz_dy - duy_dz;
                        let wy = dux_dz - duz_dx;
                        let wz = duy_dx - dux_dy;
                        enstrophy += wx * wx + wy * wy + wz * wz;
                    }
                }
            }
            enstrophy / (nx * ny * nz) as f64
        }

        match self {
            LbmBackend3D::Cpu(solver) => Ok(enstrophy_from_cpu_velocity(solver)),
            #[cfg(feature = "gpu")]
            LbmBackend3D::Gpu(solver) => Ok(solver
                .calculate_enstrophy()
                .context("failed to compute GPU enstrophy")?
                as f64),
            #[cfg(not(feature = "gpu"))]
            LbmBackend3D::Gpu(_) => Err(anyhow::anyhow!(
                "GPU backend unavailable: build gororoba_engine with --features gpu"
            )),
        }
    }

    pub fn try_set_force_field(&mut self, force_field: &[[f64; 3]]) -> Result<()> {
        match self {
            LbmBackend3D::Cpu(solver) => solver
                .set_force_field(force_field.to_vec())
                .map_err(|e| anyhow::anyhow!("failed to set CPU force field: {e}")),
            #[cfg(feature = "gpu")]
            LbmBackend3D::Gpu(solver) => solver
                .set_force_field(force_field)
                .context("failed to set GPU force field"),
            #[cfg(not(feature = "gpu"))]
            LbmBackend3D::Gpu(_) => Err(anyhow::anyhow!(
                "GPU backend unavailable: build gororoba_engine with --features gpu"
            )),
        }
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
                    let z = idx / (nx * ny);
                    let y = (idx % (nx * ny)) / nx;
                    let x = idx % nx;
                    ux[[x, y, z]] = vel[0];
                    uy[[x, y, z]] = vel[1];
                    uz[[x, y, z]] = vel[2];
                }
            }
            #[cfg(feature = "gpu")]
            LbmBackend3D::Gpu(solver) => {
                solver
                    .sync_to_host()
                    .context("failed to sync GPU velocity field to host")?;
                for (idx, vel) in solver.u.iter().enumerate() {
                    let z = idx / (nx * ny);
                    let y = (idx % (nx * ny)) / nx;
                    let x = idx % nx;
                    ux[[x, y, z]] = vel[0] as f64;
                    uy[[x, y, z]] = vel[1] as f64;
                    uz[[x, y, z]] = vel[2] as f64;
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
    pub fn apply_spectral_mask(
        &self,
        u_hat: &mut cudarc::driver::CudaSlice<lbm_3d_cuda::ComplexDevice>,
        mask: &cudarc::driver::CudaSlice<f32>,
        damping: f32,
    ) -> anyhow::Result<()> {
        match self {
            LbmBackend3D::Cpu(_) => Err(anyhow::anyhow!(
                "Spectral mask GPU not available on CPU backend"
            )),
            LbmBackend3D::Gpu(solver) => solver.apply_spectral_mask(u_hat, mask, damping),
        }
    }

    #[cfg(feature = "gpu")]
    pub fn convert_real_to_complex(
        &self,
        d_u_hat: &mut cudarc::driver::CudaSlice<lbm_3d_cuda::ComplexDevice>,
        component: usize,
    ) -> anyhow::Result<()> {
        match self {
            LbmBackend3D::Cpu(_) => Err(anyhow::anyhow!(
                "Convert real to complex GPU not available on CPU backend"
            )),
            LbmBackend3D::Gpu(solver) => solver.convert_real_to_complex(d_u_hat, component),
        }
    }

    #[cfg(feature = "gpu")]
    pub fn convert_complex_to_real(
        &mut self,
        d_u_hat: &cudarc::driver::CudaSlice<lbm_3d_cuda::ComplexDevice>,
        component: usize,
        scale: f32,
    ) -> anyhow::Result<()> {
        match self {
            LbmBackend3D::Cpu(_) => Err(anyhow::anyhow!(
                "Convert complex to real GPU not available on CPU backend"
            )),
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
            LbmBackend3D::Cpu(_) => Err(anyhow::anyhow!(
                "Enstrophy calculation GPU not available on CPU backend"
            )),
            LbmBackend3D::Gpu(solver) => solver.calculate_enstrophy(),
        }
    }
}

pub struct SimulationState3D {
    pub fluid: LbmBackend3D,
    pub imbalance: Option<Box<dyn ImbalanceField3D>>,
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
            {
                return Err(anyhow::anyhow!(
                    "GPU features not enabled. Build with --features gpu"
                ));
            }
        } else {
            LbmBackend3D::Cpu(LbmSolver3D::new(nx, ny, nz, config.tau))
        };

        Ok(Self {
            fluid,
            imbalance: None,
            curvature_field,
            nx,
            ny,
            nz,
            current_step: 0,
        })
    }

    pub fn step(&mut self) -> Result<()> {
        self.fluid.step()?;
        if let Some(ref mut f) = self.imbalance {
            f.apply(&mut self.fluid, self.nx, self.ny, self.nz)?;
        }
        self.current_step += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{LbmBackend3D, LbmSolver3D};

    #[test]
    fn velocity_rms_matches_uniform_field() {
        let mut solver = LbmSolver3D::new(4, 4, 4, 0.6);
        solver.initialize_uniform(1.0, [0.1, -0.2, 0.3]);
        let mut backend = LbmBackend3D::Cpu(solver);
        let rms = backend
            .try_velocity_rms(4, 4, 4)
            .expect("CPU velocity RMS should compute");
        let expected = (0.1f64 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3).sqrt();
        assert!(
            (rms - expected).abs() < 1.0e-12,
            "rms={rms}, expected={expected}"
        );
    }
}
