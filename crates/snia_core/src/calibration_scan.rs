use crate::yield_model::NickelYieldModel;
use serde::{Deserialize, Serialize};

/// Reduced DDT parameter set for calibration scans.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DdtParameterSet {
    pub transition_density: f64,
    pub turbulence_intensity: f64,
    pub ignition_offset_km: f64,
}

/// Scan output row.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CalibrationResult {
    pub parameter_set: DdtParameterSet,
    pub predicted_nickel56_msun: f64,
    pub objective: f64,
}

/// Score a batch of parameter sets against a target Ni56 mass.
#[must_use]
pub fn scan_parameter_sets(
    parameter_sets: &[DdtParameterSet],
    burned_mass_msun: f64,
    target_nickel56_msun: f64,
    yield_model: NickelYieldModel,
) -> Vec<CalibrationResult> {
    parameter_sets
        .iter()
        .copied()
        .map(|set| {
            let detonation_strength = (set.turbulence_intensity
                * (set.transition_density / 2.0e7).sqrt())
            .clamp(0.1, 3.0);
            let predicted = yield_model.estimate_nickel56_mass(
                burned_mass_msun,
                set.transition_density,
                detonation_strength,
            );
            let objective = (predicted - target_nickel56_msun).abs();
            CalibrationResult {
                parameter_set: set,
                predicted_nickel56_msun: predicted,
                objective,
            }
        })
        .collect()
}

/// Returns the best parameter set in a scan (minimum objective).
#[must_use]
pub fn best_parameter_set(results: &[CalibrationResult]) -> Option<CalibrationResult> {
    results
        .iter()
        .copied()
        .min_by(|a, b| a.objective.total_cmp(&b.objective))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn best_parameter_selection_works() {
        let grid = [
            DdtParameterSet {
                transition_density: 1.8e7,
                turbulence_intensity: 0.7,
                ignition_offset_km: 80.0,
            },
            DdtParameterSet {
                transition_density: 2.2e7,
                turbulence_intensity: 1.0,
                ignition_offset_km: 60.0,
            },
            DdtParameterSet {
                transition_density: 2.8e7,
                turbulence_intensity: 1.2,
                ignition_offset_km: 40.0,
            },
        ];
        let out = scan_parameter_sets(&grid, 1.0, 0.6, NickelYieldModel::default());
        assert_eq!(out.len(), grid.len());
        let best = best_parameter_set(&out).expect("best");
        assert!(best.objective >= 0.0);
    }
}
