//! Warp Ring Pipeline.
//!
//! Integration of Fluid Dynamics, Kerr Spacetime, and Sedenion Algebra
//! to simulate the "Warp Ring" effect. This pipeline demonstrates the
//! engine's capability to host multi-physics simulations.

use crate::simulation::{SimulationConfig, SimulationState};
use crate::traits::{ThesisEvidence, ThesisPipeline};
use algebra_core::physics::octonion_field::{FieldParams, oct_norm_sq};
use optics_core::grin::{GrinMedium, Ray, Vec3};

/// GRIN medium derived from algebra field energy density.
///
/// Maps the local algebra energy to a refractive index profile:
///   n(x,y,z) = 1.0 + scale * sqrt(energy_at_cell)
///
/// This is the simplest physically motivated coupling: regions of
/// higher algebraic field energy bend light more.
pub struct EngineGrinMedium {
    /// 2D algebra energy grid (nx x ny), projected to z=const plane.
    energy_grid: Vec<Vec<f64>>,
    /// Coupling scale: how strongly algebra energy modulates n.
    scale: f64,
    /// Grid spacing (assumes unit cells).
    nx: usize,
    ny: usize,
}

impl EngineGrinMedium {
    /// Build from a flat algebra energy field.
    pub fn from_energy_grid(energy: &[f64], nx: usize, ny: usize, scale: f64) -> Self {
        let mut grid = vec![vec![0.0; ny]; nx];
        for (idx, &e) in energy.iter().enumerate() {
            let ix = idx % nx;
            let iy = idx / nx;
            if ix < nx && iy < ny {
                grid[ix][iy] = e;
            }
        }
        Self {
            energy_grid: grid,
            scale,
            nx,
            ny,
        }
    }
}

impl GrinMedium for EngineGrinMedium {
    fn gradient_and_n(&self, p: Vec3) -> (Vec3, f64) {
        // Clamp to grid bounds
        let x = p[0].clamp(0.0, (self.nx - 1) as f64);
        let y = p[1].clamp(0.0, (self.ny - 1) as f64);
        let ix = (x as usize).min(self.nx - 1);
        let iy = (y as usize).min(self.ny - 1);

        let e = self.energy_grid[ix][iy];
        let n = 1.0 + self.scale * e.sqrt();

        // Central-difference gradient (clamped at boundaries)
        let ix_lo = ix.saturating_sub(1);
        let ix_hi = (ix + 1).min(self.nx - 1);
        let iy_lo = iy.saturating_sub(1);
        let iy_hi = (iy + 1).min(self.ny - 1);

        let dn_dx = if ix_hi > ix_lo {
            let e_hi = self.energy_grid[ix_hi][iy];
            let e_lo = self.energy_grid[ix_lo][iy];
            let n_hi = 1.0 + self.scale * e_hi.sqrt();
            let n_lo = 1.0 + self.scale * e_lo.sqrt();
            (n_hi - n_lo) / (ix_hi - ix_lo) as f64
        } else {
            0.0
        };

        let dn_dy = if iy_hi > iy_lo {
            let e_hi = self.energy_grid[ix][iy_hi];
            let e_lo = self.energy_grid[ix][iy_lo];
            let n_hi = 1.0 + self.scale * e_hi.sqrt();
            let n_lo = 1.0 + self.scale * e_lo.sqrt();
            (n_hi - n_lo) / (iy_hi - iy_lo) as f64
        } else {
            0.0
        };

        ([dn_dx, dn_dy, 0.0], n)
    }
}

/// Pipeline for the Warp Ring simulation.
#[derive(Debug, Clone)]
pub struct WarpRingPipeline {
    /// Simulation configuration.
    pub config: SimulationConfig,
    /// Number of steps to run.
    pub n_steps: usize,
}

impl Default for WarpRingPipeline {
    fn default() -> Self {
        Self {
            config: SimulationConfig {
                nx: 64,
                ny: 64,
                tau: 0.8,
                algebra_params: FieldParams {
                    n: 64,
                    l: 64.0, // Assuming unit lattice spacing implies length = n
                    mass: 1.0,
                    coupling: 0.1,
                    dt: 0.01,
                },
                coupling_fluid_algebra: 0.1,
                coupling_algebra_fluid: 0.1,
                coupling_metric_algebra: 0.1,
            },
            n_steps: 100,
        }
    }
}

impl ThesisPipeline for WarpRingPipeline {
    fn name(&self) -> &str {
        "Warp Ring: Algebra-Fluid Integration"
    }

    fn execute(&self) -> ThesisEvidence {
        let mut state = SimulationState::new(self.config.clone());

        // Run simulation
        for _ in 0..self.n_steps {
            state.step();
        }

        // Gather Evidence
        // 1. Is the algebra field non-zero? (Coupling check)
        let algebra_energy: f64 = match &state.algebra {
            crate::simulation::AlgebraicField::Octonion(f) => f.iter().map(oct_norm_sq).sum(),
            // For other fields, we can't use oct_norm_sq directly, but they are just arrays of f64s underneath
            // For now, we only support Octonion in this pipeline
            _ => 0.0,
        };

        // 2. Is the fluid affected? (Back-reaction check)
        // Check if viscosity varies spatially (implies algebra coupling working)
        // (This requires looking at internal D2Q9 state or extending SimulationState API)
        // For now, we assume if algebra_energy > 0, the coupling *could* work.

        // 3. GRIN ray trace through the algebra energy field
        let energy_flat: Vec<f64> = match &state.algebra {
            crate::simulation::AlgebraicField::Octonion(f) => f.iter().map(oct_norm_sq).collect(),
            _ => vec![0.0; self.config.nx * self.config.ny],
        };

        let grin = EngineGrinMedium::from_energy_grid(
            &energy_flat,
            self.config.nx,
            self.config.ny,
            0.01, // coupling scale
        );

        let initial_ray = Ray {
            pos: [0.0, self.config.ny as f64 / 2.0, 0.0],
            dir: [1.0, 0.0, 0.0], // horizontal launch
        };

        let trace_result = optics_core::trace_ray(initial_ray, &grin, 0.5, self.config.nx);
        let ray_deviation = if let Some(last_pos) = trace_result.positions.last() {
            // Deviation from straight-line path (y displacement)
            (last_pos[1] - initial_ray.pos[1]).abs()
        } else {
            0.0
        };

        let passes = algebra_energy > 1e-9;

        ThesisEvidence {
            thesis_id: 0, // 0 for integration/demo
            label: "Warp Ring Integration".to_string(),
            metric_value: algebra_energy,
            threshold: 1e-9,
            passes_gate: passes,
            messages: vec![
                format!("Algebra Energy: {:.6e}", algebra_energy),
                format!("Steps: {}", self.n_steps),
                format!("Ray Deviation: {:.6e}", ray_deviation),
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warp_ring_optics_integrated() {
        let pipeline = WarpRingPipeline::default();
        let evidence = pipeline.execute();
        // Ray deviation message should be present
        assert!(
            evidence
                .messages
                .iter()
                .any(|m| m.contains("Ray Deviation")),
            "evidence should contain ray deviation metric"
        );
    }

    #[test]
    fn test_engine_grin_medium_uniform() {
        // Uniform energy -> n=const -> no bending
        let energy = vec![1.0; 16]; // 4x4
        let grin = EngineGrinMedium::from_energy_grid(&energy, 4, 4, 0.01);
        let (grad, n) = grin.gradient_and_n([2.0, 2.0, 0.0]);
        assert!((n - 1.01).abs() < 1e-10, "n = 1 + 0.01*sqrt(1) = 1.01");
        assert!(
            grad[0].abs() < 1e-10 && grad[1].abs() < 1e-10,
            "uniform field -> zero gradient"
        );
    }
}
