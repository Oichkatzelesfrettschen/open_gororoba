use snia_core::{
    read_snapshot_toml, write_snapshot_toml, BurnState, CarbonBurnModel, HllcFlux1D, HydroState1D,
    SimulationSnapshot, SniaCoreSolver, SolverConfig, WhiteDwarfEos,
};

#[test]
fn short_pipeline_produces_snapshot() {
    let config = SolverConfig {
        n_cells: 12,
        final_time_s: 8.0e-4,
        max_steps: 64,
        ..SolverConfig::default()
    };

    let mut solver = SniaCoreSolver::new(
        config,
        WhiteDwarfEos::default(),
        HllcFlux1D::default(),
        CarbonBurnModel::default(),
        snia_core::NickelYieldModel::default(),
        HydroState1D {
            density: 2.2e7,
            velocity: 8.0e5,
            pressure: 2.8e23,
            specific_internal_energy: 1.0e17,
        },
        1.2e9,
        BurnState::default(),
    )
    .expect("solver init");

    let result = solver.run().expect("run");
    assert!(result.n_steps > 0);
    assert!(result.nickel56_mass_msun >= 0.0);

    let snapshot = SimulationSnapshot::from(&result);
    let temp = tempfile::tempdir().expect("tempdir");
    let path = temp.path().join("snia_snapshot.toml");
    write_snapshot_toml(&path, &snapshot).expect("write toml");
    let loaded = read_snapshot_toml(&path).expect("read toml");
    assert_eq!(loaded.density_profile.len(), result.cells.len());
}
