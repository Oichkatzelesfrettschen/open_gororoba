use crate::error::SniaError;
use crate::solver::SimulationResult;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationSnapshot {
    pub final_time_s: f64,
    pub n_steps: usize,
    pub burned_mass_msun: f64,
    pub nickel56_mass_msun: f64,
    pub peak_density: f64,
    pub density_profile: Vec<f64>,
    pub velocity_profile: Vec<f64>,
    pub pressure_profile: Vec<f64>,
    pub carbon_profile: Vec<f64>,
    pub nickel56_profile: Vec<f64>,
}

impl From<&SimulationResult> for SimulationSnapshot {
    fn from(result: &SimulationResult) -> Self {
        Self {
            final_time_s: result.final_time_s,
            n_steps: result.n_steps,
            burned_mass_msun: result.burned_mass_msun,
            nickel56_mass_msun: result.nickel56_mass_msun,
            peak_density: result.peak_density,
            density_profile: result.cells.iter().map(|c| c.hydro.density).collect(),
            velocity_profile: result.cells.iter().map(|c| c.hydro.velocity).collect(),
            pressure_profile: result.cells.iter().map(|c| c.hydro.pressure).collect(),
            carbon_profile: result
                .cells
                .iter()
                .map(|c| c.burn.carbon_mass_fraction)
                .collect(),
            nickel56_profile: result
                .cells
                .iter()
                .map(|c| c.burn.nickel56_mass_fraction)
                .collect(),
        }
    }
}

pub fn write_snapshot_toml<P: AsRef<Path>>(
    path: P,
    snapshot: &SimulationSnapshot,
) -> Result<(), SniaError> {
    let serialized = toml::to_string_pretty(snapshot)?;
    fs::write(path, serialized)?;
    Ok(())
}

pub fn read_snapshot_toml<P: AsRef<Path>>(path: P) -> Result<SimulationSnapshot, SniaError> {
    let content = fs::read_to_string(path)?;
    Ok(toml::from_str(&content)?)
}

#[cfg(feature = "hdf5-export")]
pub fn write_snapshot_hdf5<P: AsRef<Path>>(
    path: P,
    snapshot: &SimulationSnapshot,
) -> Result<(), SniaError> {
    let file = hdf5::File::create(path)?;
    file.new_dataset_builder()
        .with_data(&snapshot.density_profile)
        .create("density_profile")?;
    file.new_dataset_builder()
        .with_data(&snapshot.velocity_profile)
        .create("velocity_profile")?;
    file.new_dataset_builder()
        .with_data(&snapshot.pressure_profile)
        .create("pressure_profile")?;
    file.new_dataset_builder()
        .with_data(&snapshot.carbon_profile)
        .create("carbon_profile")?;
    file.new_dataset_builder()
        .with_data(&snapshot.nickel56_profile)
        .create("nickel56_profile")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{BurnState, HydroState1D};

    #[test]
    fn toml_roundtrip_preserves_snapshot() {
        let temp_dir = tempfile::tempdir().expect("tempdir");
        let path = temp_dir.path().join("snapshot.toml");

        let result = SimulationResult {
            final_time_s: 0.1,
            n_steps: 3,
            burned_mass_msun: 0.2,
            nickel56_mass_msun: 0.1,
            peak_density: 2.5e7,
            detonation_events: vec![],
            cells: vec![crate::solver::CellState {
                hydro: HydroState1D {
                    density: 2.5e7,
                    velocity: 1.0e6,
                    pressure: 3.0e23,
                    specific_internal_energy: 1.0e17,
                },
                burn: BurnState::default(),
                temperature: 1.0e9,
            }],
        };
        let snapshot = SimulationSnapshot::from(&result);

        write_snapshot_toml(&path, &snapshot).expect("write");
        let loaded = read_snapshot_toml(&path).expect("read");
        assert_eq!(loaded.n_steps, snapshot.n_steps);
        assert_eq!(loaded.density_profile.len(), 1);
    }
}
