//! `snia_core` provides a Rust-native Type Ia DDT solver scaffold.
//!
//! This crate focuses on rebuilding core physics components from the Pantheon
//! workflow as first-class Rust modules:
//! - White dwarf equation of state (`eos`)
//! - 1D finite-volume hydro fluxes (`hydro`)
//! - Carbon burning model and coupling (`reaction`)
//! - Coupled time integration (`solver`)
//! - Yield calibration and light-curve synthesis (`calibration_scan`, `lightcurve`)
//! - TOML-first output and optional HDF5 export (`output`)

#![forbid(unsafe_code)]

pub mod alpha_chain;
pub mod calibration_scan;
pub mod eos;
pub mod error;
pub mod hydro;
pub mod lightcurve;
pub mod output;
pub mod reaction;
pub mod solver;
pub mod types;
pub mod yield_model;

pub use alpha_chain::{AlphaChainNetwork, AlphaChainState};
pub use calibration_scan::{
    best_parameter_set, scan_parameter_sets, CalibrationResult, DdtParameterSet,
};
pub use eos::{CachedEos, EosComposition, EosState, WhiteDwarfEos};
pub use error::SniaError;
pub use hydro::{BoundaryCondition, ConservativeState, HllcFlux1D, LimiterKind};
pub use lightcurve::{LightCurveModel, LightCurveSample};
pub use output::{read_snapshot_toml, write_snapshot_toml, SimulationSnapshot};
pub use reaction::{BurnStepResult, CarbonBurnModel};
pub use solver::{CellState, DetonationEvent, SimulationResult, SniaCoreSolver, SolverConfig};
pub use types::{BurnState, HydroState1D, ThermoState};
pub use yield_model::NickelYieldModel;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_models_are_constructible() {
        let eos = WhiteDwarfEos::default();
        let burner = CarbonBurnModel::default();
        let yield_model = NickelYieldModel::default();

        let p = eos.pressure(1.0e7, 1.0e9).expect("pressure");
        assert!(p.is_finite());
        assert!(burner.q_value > 0.0);
        assert!(yield_model.reference_density > 0.0);
    }
}
