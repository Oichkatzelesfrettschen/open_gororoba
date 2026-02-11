use criterion::{black_box, criterion_group, criterion_main, Criterion};
use snia_core::{BoundaryCondition, HllcFlux1D, HydroState1D, LimiterKind};

fn bench_hydro_rk2_update(c: &mut Criterion) {
    let h = HllcFlux1D {
        gamma: 1.4,
        ..HllcFlux1D::default()
    };
    let n = 256usize;
    let dx = 1.0 / n as f64;
    let dt = 1.0e-4;
    let mut cells = Vec::with_capacity(n);
    for i in 0..n {
        let x = (i as f64 + 0.5) / n as f64;
        if x < 0.5 {
            cells.push(HydroState1D {
                density: 1.0,
                velocity: 0.0,
                pressure: 1.0,
                specific_internal_energy: 2.5,
            });
        } else {
            cells.push(HydroState1D {
                density: 0.125,
                velocity: 0.0,
                pressure: 0.1,
                specific_internal_energy: 2.0,
            });
        }
    }

    c.bench_function("hydro_rk2_update_256", |b| {
        b.iter(|| {
            let _ = h
                .rk2_update(
                    black_box(&cells),
                    black_box(dx),
                    black_box(dt),
                    black_box(LimiterKind::MonotonizedCentral),
                    black_box(BoundaryCondition::Outflow),
                )
                .expect("rk2");
        })
    });
}

criterion_group!(benches, bench_hydro_rk2_update);
criterion_main!(benches);
