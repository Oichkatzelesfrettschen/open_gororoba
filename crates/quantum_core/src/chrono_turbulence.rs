use lbm_core::{D2Q9, viscosity_with_chrono_complex};
use num_complex::Complex;
use ndarray::Array2;

/// Chrono-Turbulence Solver: Hydrodynamics of the Complex Temporal Manifold.
pub struct ChronoTurbulenceSolver {
    pub sim: D2Q9,
    pub alpha_chrono: f64,
}

impl ChronoTurbulenceSolver {
    pub fn new(nx: usize, ny: usize, tau_base: f64, alpha_chrono: f64) -> Self {
        Self {
            sim: D2Q9::new(nx, ny, tau_base),
            alpha_chrono,
        }
    }

    /// Evolve the temporal fluid forward by one complex time step.
    pub fn step(&mut self, d_tau: Complex<f64>) {
        // Imaginary component of complex time acts as temporal flux pressure
        let epsilon = d_tau.im;
        
        // Modulate viscosity based on temporal flux
        let nu_base = self.sim.viscosity();
        let nu_eff = viscosity_with_chrono_complex(nu_base, self.alpha_chrono, epsilon);
        
        // Update simulation relaxation time
        self.sim.tau = 3.0 * nu_eff + 0.5;
        
        // Execute LBM cycle for the temporal manifold
        self.sim.collide(0, self.sim.ny);
        self.sim.stream();
    }

    /// Returns the "Temporal Vorticity" field (Causal Eddies).
    pub fn temporal_vorticity(&self) -> Array2<f64> {
        let (_, ux, uy) = self.sim.macroscopic();
        let (nx, ny) = ux.dim();
        let mut vorticity = Array2::zeros((nx, ny));
        
        for x in 0..nx {
            for y in 0..ny {
                let xp = (x + 1) % nx;
                let yp = (y + 1) % ny;
                let duy_dx = (uy[[xp, y]] - uy[[(x + nx - 1) % nx, y]]) / 2.0;
                let dux_dy = (ux[[x, yp]] - ux[[x, (y + ny - 1) % ny]]) / 2.0;
                vorticity[[x, y]] = duy_dx - dux_dy;
            }
        }
        vorticity
    }
}
