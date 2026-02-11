use snia_core::{
    write_snapshot_toml, BurnState, CarbonBurnModel, HllcFlux1D, HydroState1D, NickelYieldModel,
    SimulationSnapshot, SniaCoreSolver, SolverConfig, WhiteDwarfEos,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = SolverConfig {
        n_cells: 64,
        final_time_s: 2.0e-3,
        max_steps: 256,
        ..SolverConfig::default()
    };
    let initial_hydro = HydroState1D {
        density: 2.6e7,
        velocity: 1.2e6,
        pressure: 3.1e23,
        specific_internal_energy: 1.2e17,
    };

    let mut solver = SniaCoreSolver::new(
        config,
        WhiteDwarfEos::default(),
        HllcFlux1D::default(),
        CarbonBurnModel::default(),
        NickelYieldModel::default(),
        initial_hydro,
        1.8e9,
        BurnState::default(),
    )?;

    let result = solver.run()?;
    let snapshot = SimulationSnapshot::from(&result);
    write_snapshot_toml("snia_snapshot.toml", &snapshot)?;

    println!(
        "SN Ia scaffold run complete: steps={}, t={:.4e} s, Ni56={:.4e} Msun",
        result.n_steps, result.final_time_s, result.nickel56_mass_msun
    );
    Ok(())
}
