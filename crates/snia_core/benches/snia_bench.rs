use criterion::{criterion_group, criterion_main, Criterion};
use snia_core::{
    BurnState, CarbonBurnModel, HllcFlux1D, HydroState1D, NickelYieldModel, SniaCoreSolver,
    SolverConfig, WhiteDwarfEos,
};

fn bench_solver_step(c: &mut Criterion) {
    c.bench_function("snia_core.step", |b| {
        b.iter(|| {
            let config = SolverConfig {
                n_cells: 24,
                final_time_s: 2.0e-4,
                max_steps: 8,
                ..SolverConfig::default()
            };
            let mut solver = SniaCoreSolver::new(
                config,
                WhiteDwarfEos::default(),
                HllcFlux1D::default(),
                CarbonBurnModel::default(),
                NickelYieldModel::default(),
                HydroState1D {
                    density: 2.0e7,
                    velocity: 1.0e6,
                    pressure: 3.0e23,
                    specific_internal_energy: 1.1e17,
                },
                1.4e9,
                BurnState::default(),
            )
            .expect("solver init");
            let _ = solver.step().expect("step");
        })
    });
}

criterion_group!(benches, bench_solver_step);
criterion_main!(benches);
