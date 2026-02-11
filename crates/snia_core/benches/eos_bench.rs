use criterion::{black_box, criterion_group, criterion_main, Criterion};
use snia_core::{EosComposition, WhiteDwarfEos};

fn bench_eos_from_rho_t(c: &mut Criterion) {
    let eos = WhiteDwarfEos::default();
    let comp = EosComposition::default();
    c.bench_function("eos_from_rho_t", |b| {
        b.iter(|| {
            let _ = eos
                .eos_from_rho_t(black_box(2.5e7), black_box(1.3e9), black_box(comp))
                .expect("eos");
        })
    });
}

fn bench_temperature_inversion(c: &mut Criterion) {
    let eos = WhiteDwarfEos::default();
    let comp = EosComposition::default();
    let state = eos.eos_from_rho_t(2.5e7, 1.3e9, comp).expect("state");
    let e = state.thermo.specific_internal_energy;
    c.bench_function("temperature_from_rho_e", |b| {
        b.iter(|| {
            let _ = eos
                .temperature_from_rho_e(black_box(2.5e7), black_box(e), black_box(comp))
                .expect("invert");
        })
    });
}

criterion_group!(benches, bench_eos_from_rho_t, bench_temperature_inversion);
criterion_main!(benches);
