use algebra_experimental::{
    cayley_dickson_structs::Sedenion, su_n_generators::construct_su5_generators_algebraic,
};
use criterion::{Criterion, criterion_group, criterion_main};

fn su5_construction_benchmark(c: &mut Criterion) {
    let mut basis = [Sedenion::default(); 16];
    for i in 0..16 {
        let mut components = [0.0; 16];
        components[i] = 1.0;
        basis[i] = Sedenion::from_slice(&components);
    }
    let complex_structure = basis[15];

    c.bench_function("construct_su5_generators_algebraic", |b| {
        b.iter(|| construct_su5_generators_algebraic(&basis, &complex_structure))
    });
}

criterion_group!(benches, su5_construction_benchmark);
criterion_main!(benches);
