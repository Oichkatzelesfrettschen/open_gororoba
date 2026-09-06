//! Shared post-step D3Q19 population diagnostics.

use crate::lbm_dispatch::LbmBackend;

#[derive(Debug, Clone)]
pub struct PopulationObservation {
    pub finite: bool,
    pub minimum_density: f64,
    pub mass: f64,
    pub mach: f64,
    pub minimum_population: f64,
    pub density: Vec<f64>,
}

pub fn inspect_populations(populations: &[f64]) -> anyhow::Result<PopulationObservation> {
    anyhow::ensure!(
        !populations.is_empty() && populations.len().is_multiple_of(19),
        "complete cell-major D3Q19 populations required"
    );
    let lattice = lbm_3d::lattice::D3Q19Lattice::new();
    let mut observation = PopulationObservation {
        finite: true,
        minimum_density: f64::INFINITY,
        mass: 0.0,
        mach: 0.0,
        minimum_population: f64::INFINITY,
        density: Vec::with_capacity(populations.len() / 19),
    };
    for cell in populations.chunks_exact(19) {
        observation.finite &= cell.iter().all(|value| value.is_finite());
        for &population in cell {
            observation.minimum_population = observation.minimum_population.min(population);
        }
        let density: f64 = cell.iter().sum();
        let mut momentum = [0.0; 3];
        for (direction, &population) in cell.iter().enumerate() {
            let velocity = lattice.velocity(direction);
            for axis in 0..3 {
                momentum[axis] += population * f64::from(velocity[axis]);
            }
        }
        let mach = momentum
            .into_iter()
            .map(|value| (value / density).powi(2))
            .sum::<f64>()
            .sqrt()
            * 3.0_f64.sqrt();
        observation.finite &= density.is_finite() && mach.is_finite();
        observation.minimum_density = observation.minimum_density.min(density);
        observation.mass += density;
        observation.density.push(density);
        observation.mach = observation.mach.max(mach);
    }
    observation.finite &= observation.mass.is_finite();
    Ok(observation)
}

impl PopulationObservation {
    /// Raw moment velocity excludes a force half-step convention on both backends.
    pub fn require_stable(
        &self,
        initial_mass: f64,
        mass_budget: f64,
        mach_budget: f64,
    ) -> anyhow::Result<()> {
        anyhow::ensure!(
            initial_mass.is_finite() && initial_mass > 0.0,
            "positive finite initial mass required"
        );
        anyhow::ensure!(
            mass_budget.is_finite()
                && mass_budget >= 0.0
                && mach_budget.is_finite()
                && mach_budget > 0.0,
            "finite nonnegative mass and positive Mach budgets required"
        );
        anyhow::ensure!(
            self.finite && self.minimum_density > 0.0 && self.minimum_population >= 0.0,
            "nonfinite, nonpositive density or negative population"
        );
        anyhow::ensure!(
            (self.mass / initial_mass - 1.0).abs() <= mass_budget,
            "relative mass budget exceeded"
        );
        anyhow::ensure!(
            self.mach <= mach_budget,
            "raw population-moment Mach budget exceeded"
        );
        Ok(())
    }
}

pub fn inspect_fields(backend: &mut LbmBackend) -> anyhow::Result<PopulationObservation> {
    let populations: Vec<f64> = match backend {
        LbmBackend::Avx2(solver) => (0..solver.nx * solver.ny * solver.nz)
            .flat_map(|cell| {
                let source = &solver.f;
                (0..19).map(move |direction| source[lbm_3d::solver::aosoa_idx(cell, direction)])
            })
            .collect(),
        #[cfg(feature = "gpu")]
        LbmBackend::Cuda(solver) => solver
            .read_populations_fp32()?
            .into_iter()
            .map(f64::from)
            .collect(),
    };
    inspect_populations(&populations)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn raw_population_admission_rejects_negative_nonfinite_and_exploded_states() {
        let equilibrium = lbm_3d::solver::BgkCollision::initialize_rest(
            1.0,
            &lbm_3d::lattice::D3Q19Lattice::new(),
        );
        let baseline = inspect_populations(&equilibrium).unwrap();
        assert!(baseline.require_stable(1.0, 1e-5, 0.3).is_ok());
        for invalid in [f64::NAN, f64::INFINITY, -0.01, 1e113] {
            let mut changed = equilibrium;
            changed[18] = invalid;
            assert!(
                inspect_populations(&changed)
                    .unwrap()
                    .require_stable(1.0, 1e-5, 0.3)
                    .is_err()
            );
        }
        assert!(inspect_populations(&[]).is_err());
        assert!(inspect_populations(&[1.0; 18]).is_err());
    }
}
