use criterion::{Criterion, criterion_group, criterion_main};
use fwht::{fast_jl_rotate, generate_rademacher_diagonals, wht_inplace};
use std::hint::black_box;

fn bench_wht_128(c: &mut Criterion) {
    let mut data: Vec<f64> = (0..128).map(|i| (i as f64 * 0.1).sin()).collect();
    c.bench_function("wht_inplace_128", |b| {
        b.iter(|| {
            wht_inplace(black_box(&mut data));
        })
    });
}

fn bench_fast_jl_128(c: &mut Criterion) {
    let d = 128;
    let (d1, d2) = generate_rademacher_diagonals(d, 42);
    let x: Vec<f64> = (0..d).map(|i| (i as f64 * 0.1).sin()).collect();
    let mut buf = vec![0.0; d];
    let mut out = vec![0.0; d];
    c.bench_function("fast_jl_rotate_128", |b| {
        b.iter(|| {
            fast_jl_rotate(black_box(&x), &d1, &d2, &mut buf, &mut out);
        })
    });
}

criterion_group!(benches, bench_wht_128, bench_fast_jl_128);
criterion_main!(benches);
