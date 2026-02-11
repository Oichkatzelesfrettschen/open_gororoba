use snia_core::{
    BurnState, CarbonBurnModel, HllcFlux1D, HydroState1D, NickelYieldModel, SniaCoreSolver,
    SolverConfig, WhiteDwarfEos,
};

fn run_once() -> snia_core::SimulationResult {
    let config = SolverConfig {
        n_cells: 24,
        final_time_s: 8.0e-4,
        max_steps: 96,
        ..SolverConfig::default()
    };
    let hydro = HydroState1D {
        density: 2.2e7,
        velocity: 8.5e5,
        pressure: 2.7e23,
        specific_internal_energy: 1.05e17,
    };
    let burn = BurnState::default();

    let mut solver = SniaCoreSolver::new(
        config,
        WhiteDwarfEos::default(),
        HllcFlux1D::default(),
        CarbonBurnModel::default(),
        NickelYieldModel::default(),
        hydro,
        1.4e9,
        burn,
    )
    .expect("solver init");
    solver.run().expect("run")
}

#[test]
fn deterministic_replay_matches_exactly() {
    let a = run_once();
    let b = run_once();

    assert_eq!(a.n_steps, b.n_steps);
    assert_eq!(a.cells.len(), b.cells.len());
    assert_eq!(a.detonation_events.len(), b.detonation_events.len());
    assert_eq!(a.final_time_s.to_bits(), b.final_time_s.to_bits());
    assert_eq!(
        a.nickel56_mass_msun.to_bits(),
        b.nickel56_mass_msun.to_bits()
    );
    assert_eq!(a.peak_density.to_bits(), b.peak_density.to_bits());

    for (ca, cb) in a.cells.iter().zip(b.cells.iter()) {
        assert_eq!(ca.hydro.density.to_bits(), cb.hydro.density.to_bits());
        assert_eq!(ca.hydro.velocity.to_bits(), cb.hydro.velocity.to_bits());
        assert_eq!(ca.hydro.pressure.to_bits(), cb.hydro.pressure.to_bits());
        assert_eq!(
            ca.hydro.specific_internal_energy.to_bits(),
            cb.hydro.specific_internal_energy.to_bits()
        );
        assert_eq!(
            ca.burn.carbon_mass_fraction.to_bits(),
            cb.burn.carbon_mass_fraction.to_bits()
        );
        assert_eq!(
            ca.burn.nickel56_mass_fraction.to_bits(),
            cb.burn.nickel56_mass_fraction.to_bits()
        );
    }
}
