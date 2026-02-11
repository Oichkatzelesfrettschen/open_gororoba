use serde::Deserialize;
use snia_core::{
    BurnState, CarbonBurnModel, HllcFlux1D, HydroState1D, NickelYieldModel, SniaCoreSolver,
    SolverConfig, WhiteDwarfEos,
};

#[derive(Debug, Deserialize)]
struct Fixture {
    config: FixtureConfig,
    initial_hydro: FixtureHydro,
    initial_burn: FixtureBurn,
    expected: FixtureExpected,
}

#[derive(Debug, Deserialize)]
struct FixtureConfig {
    n_cells: usize,
    dx_cm: f64,
    cfl: f64,
    final_time_s: f64,
    max_steps: usize,
    cell_volume_cm3: f64,
    detonation_burn_fraction_threshold: f64,
    detonation_cj_fraction_threshold: f64,
}

#[derive(Debug, Deserialize)]
struct FixtureHydro {
    density: f64,
    velocity: f64,
    pressure: f64,
    specific_internal_energy: f64,
    temperature: f64,
}

#[derive(Debug, Deserialize)]
struct FixtureBurn {
    carbon_mass_fraction: f64,
    oxygen_mass_fraction: f64,
    nickel56_mass_fraction: f64,
}

#[derive(Debug, Deserialize)]
struct FixtureExpected {
    min_steps: usize,
    max_steps: usize,
    nickel56_min_msun: f64,
    nickel56_max_msun: f64,
    peak_density_min: f64,
    peak_density_max: f64,
}

#[test]
fn pantheon_baseline_window_compatibility() {
    let fixture_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/pantheon_baseline_windows.toml");
    let fixture_text = std::fs::read_to_string(&fixture_path).expect("fixture read");
    let fixture: Fixture = toml::from_str(&fixture_text).expect("fixture parse");

    let config = SolverConfig {
        n_cells: fixture.config.n_cells,
        dx_cm: fixture.config.dx_cm,
        cfl: fixture.config.cfl,
        final_time_s: fixture.config.final_time_s,
        max_steps: fixture.config.max_steps,
        cell_volume_cm3: fixture.config.cell_volume_cm3,
        detonation_burn_fraction_threshold: fixture.config.detonation_burn_fraction_threshold,
        detonation_cj_fraction_threshold: fixture.config.detonation_cj_fraction_threshold,
    };
    let hydro = HydroState1D {
        density: fixture.initial_hydro.density,
        velocity: fixture.initial_hydro.velocity,
        pressure: fixture.initial_hydro.pressure,
        specific_internal_energy: fixture.initial_hydro.specific_internal_energy,
    };
    let burn = BurnState {
        carbon_mass_fraction: fixture.initial_burn.carbon_mass_fraction,
        oxygen_mass_fraction: fixture.initial_burn.oxygen_mass_fraction,
        nickel56_mass_fraction: fixture.initial_burn.nickel56_mass_fraction,
        specific_nuclear_energy: 0.0,
    };

    let mut solver = SniaCoreSolver::new(
        config,
        WhiteDwarfEos::default(),
        HllcFlux1D::default(),
        CarbonBurnModel::default(),
        NickelYieldModel::default(),
        hydro,
        fixture.initial_hydro.temperature,
        burn,
    )
    .expect("solver init");
    let result = solver.run().expect("run");

    assert!(result.n_steps >= fixture.expected.min_steps);
    assert!(result.n_steps <= fixture.expected.max_steps);
    assert!(result.nickel56_mass_msun >= fixture.expected.nickel56_min_msun);
    assert!(result.nickel56_mass_msun <= fixture.expected.nickel56_max_msun);
    assert!(result.peak_density >= fixture.expected.peak_density_min);
    assert!(result.peak_density <= fixture.expected.peak_density_max);
}
