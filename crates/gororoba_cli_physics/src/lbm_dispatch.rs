//! Unified LBM solver dispatcher: bridges CPU (AVX2/AoSoA) and GPU (CUDA/SoA)
//! solvers behind a common interface.
//!
//! The dispatcher handles f64<->f32 precision conversion, layout differences
//! (AoSoA on CPU, SoA on GPU), and fallible vs infallible API bridging
//! transparently. Callers interact with a single `LbmBackend` enum that
//! auto-detects hardware at construction time.

use anyhow::Result;
use lbm_3d::solver::{CollisionMode, LbmSolver3D};

/// Runtime-dispatched LBM solver backend.
///
/// Wraps either a CPU solver (AVX2/AoSoA, f64) or a GPU solver (CUDA/SoA,
/// f32) behind a unified interface. All methods accept and return f64;
/// the GPU path converts internally.
///
/// The GPU variant is boxed to avoid inflating the enum size -- the CUDA
/// solver carries ~20 kernel handles, device buffers, and host arrays.
pub enum LbmBackend {
    /// CPU solver with wide::f64x4 SIMD and AoSoA memory layout.
    Avx2(Box<LbmSolver3D>),
    /// GPU solver with CUDA FP32 SoA memory layout.
    #[cfg(feature = "gpu")]
    Cuda(Box<lbm_3d_cuda::LbmSolver3DCuda>),
}

impl LbmBackend {
    /// Auto-detect the best available backend and construct a solver.
    ///
    /// Probes CUDA availability first; falls back to CPU AVX2. The GPU
    /// path uses FP32 precision (SoA layout); the CPU path uses f64
    /// (AoSoA layout with wide::f64x4 SIMD).
    pub fn auto_detect(
        nx: usize,
        ny: usize,
        nz: usize,
        tau: f64,
        mode: CollisionMode,
    ) -> Result<Self> {
        #[cfg(feature = "gpu")]
        {
            if lbm_3d_cuda::probe_cuda_available() {
                let solver = match mode {
                    CollisionMode::Bgk => lbm_3d_cuda::LbmSolver3DCuda::new(
                        nx,
                        ny,
                        nz,
                        tau,
                        lbm_3d_cuda::Precision::FP32,
                    )?,
                    CollisionMode::Mrt => lbm_3d_cuda::LbmSolver3DCuda::new_mrt(
                        nx,
                        ny,
                        nz,
                        tau,
                        lbm_3d_cuda::Precision::FP32,
                    )?,
                };
                return Ok(Self::Cuda(Box::new(solver)));
            }
        }
        Ok(Self::cpu(nx, ny, nz, tau, mode))
    }

    /// Force CPU backend (AVX2/AoSoA).
    pub fn cpu(nx: usize, ny: usize, nz: usize, tau: f64, mode: CollisionMode) -> Self {
        let solver = match mode {
            CollisionMode::Bgk => LbmSolver3D::new(nx, ny, nz, tau),
            CollisionMode::Mrt => LbmSolver3D::new_mrt(nx, ny, nz, tau),
        };
        Self::Avx2(Box::new(solver))
    }

    /// Force GPU backend (CUDA FP32/SoA).
    #[cfg(feature = "gpu")]
    pub fn cuda(
        nx: usize,
        ny: usize,
        nz: usize,
        tau: f64,
        mode: CollisionMode,
    ) -> Result<Self> {
        let solver = match mode {
            CollisionMode::Bgk => lbm_3d_cuda::LbmSolver3DCuda::new(
                nx,
                ny,
                nz,
                tau,
                lbm_3d_cuda::Precision::FP32,
            )?,
            CollisionMode::Mrt => lbm_3d_cuda::LbmSolver3DCuda::new_mrt(
                nx,
                ny,
                nz,
                tau,
                lbm_3d_cuda::Precision::FP32,
            )?,
        };
        Ok(Self::Cuda(Box::new(solver)))
    }

    /// Advance one LBM time step.
    pub fn step(&mut self) -> Result<()> {
        match self {
            Self::Avx2(s) => {
                s.evolve_one_step();
                Ok(())
            }
            #[cfg(feature = "gpu")]
            Self::Cuda(s) => s.step(),
        }
    }

    /// Maximum Mach number across all cells.
    ///
    /// GPU path uses device-side reduction (no PCIe readback of the full
    /// velocity field). CPU path scans the host array directly.
    pub fn max_mach_number(&mut self) -> Result<f64> {
        match self {
            Self::Avx2(s) => Ok(s.max_mach_number()),
            #[cfg(feature = "gpu")]
            Self::Cuda(s) => Ok(s.max_mach_number_device()? as f64),
        }
    }

    /// Initialize from density and velocity fields (f64 interface).
    ///
    /// The GPU path converts f64 -> f32 internally during upload.
    /// The CPU path copies into the solver's AoSoA-indexed arrays and
    /// recomputes the equilibrium distribution.
    pub fn initialize_custom(&mut self, rho: &[f64], u: &[[f64; 3]]) -> Result<()> {
        match self {
            Self::Avx2(s) => {
                s.rho.copy_from_slice(rho);
                for (sv, uv) in s.u.iter_mut().zip(u.iter()) {
                    *sv = *uv;
                }
                s.reinitialize_from_macroscopic();
                Ok(())
            }
            #[cfg(feature = "gpu")]
            Self::Cuda(s) => s.initialize_custom(rho, u),
        }
    }

    /// Set the external force field (f64 interface).
    ///
    /// CPU solver takes ownership (stores in AoSoA layout internally).
    /// GPU solver borrows for upload then releases.
    pub fn set_force_field(&mut self, force: Vec<[f64; 3]>) -> Result<()> {
        match self {
            Self::Avx2(s) => s.set_force_field(force).map_err(Into::into),
            #[cfg(feature = "gpu")]
            Self::Cuda(s) => s.set_force_field(&force),
        }
    }

    /// Total mass (sum of density across all cells).
    ///
    /// GPU path uses device-side mean-density reduction and multiplies
    /// by cell count. CPU path sums the host array directly.
    pub fn total_mass(&mut self) -> Result<f64> {
        match self {
            Self::Avx2(s) => Ok(s.total_mass()),
            #[cfg(feature = "gpu")]
            Self::Cuda(s) => {
                let mean = s.calculate_mean_density()? as f64;
                let n = (s.nx() * s.ny() * s.nz()) as f64;
                Ok(mean * n)
            }
        }
    }

    /// Enable/disable thread-coarsened GPU kernels (no-op on CPU).
    pub fn set_coarsening(&mut self, enabled: bool) {
        match self {
            Self::Avx2(_) => {
                let _ = enabled;
            }
            #[cfg(feature = "gpu")]
            Self::Cuda(s) => s.set_coarsening(enabled),
        }
    }

    /// Enable/disable pull-streaming GPU kernels (no-op on CPU).
    ///
    /// Mutually exclusive with coarsening on GPU: enabling pull
    /// automatically disables coarsening (pull breaks float2 loads).
    pub fn set_pull_streaming(&mut self, enabled: bool) {
        match self {
            Self::Avx2(_) => {
                let _ = enabled;
            }
            #[cfg(feature = "gpu")]
            Self::Cuda(s) => s.set_pull_streaming(enabled),
        }
    }

    /// Human-readable backend identifier for logging.
    pub fn backend_name(&self) -> &'static str {
        match self {
            Self::Avx2(_) => "CPU (AVX2/AoSoA f64)",
            #[cfg(feature = "gpu")]
            Self::Cuda(_) => "CUDA (FP32/SoA)",
        }
    }
}
