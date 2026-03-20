use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use std::hint::black_box;

use cd_kernel::cayley_dickson::{
    cd_multiply, cd_multiply_simd, octonion_multiply_flat, quaternion_multiply_flat,
    sedenion_multiply_flat,
};

fn bench_quaternion_multiply(c: &mut Criterion) {
    let a: [f64; 4] = [1.0, 0.2, -0.3, 0.4];
    let b: [f64; 4] = [0.5, -0.1, 0.3, 0.7];
    let a_slice: Vec<f64> = a.to_vec();
    let b_slice: Vec<f64> = b.to_vec();

    let mut group = c.benchmark_group("quaternion_4d");
    group.bench_function("cd_multiply_scalar", |bench| {
        bench.iter(|| black_box(cd_multiply(black_box(&a_slice), black_box(&b_slice))))
    });
    group.bench_function("cd_multiply_simd", |bench| {
        bench.iter(|| black_box(cd_multiply_simd(black_box(&a_slice), black_box(&b_slice))))
    });
    group.bench_function("flat_avx2", |bench| {
        bench.iter(|| black_box(quaternion_multiply_flat(black_box(&a), black_box(&b))))
    });
    group.finish();
}

fn bench_octonion_multiply(c: &mut Criterion) {
    let a: [f64; 8] = [1.0, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7, 0.8];
    let b: [f64; 8] = [0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8];
    let a_slice: Vec<f64> = a.to_vec();
    let b_slice: Vec<f64> = b.to_vec();

    let mut group = c.benchmark_group("octonion_8d");
    group.bench_function("cd_multiply_scalar", |bench| {
        bench.iter(|| black_box(cd_multiply(black_box(&a_slice), black_box(&b_slice))))
    });
    group.bench_function("cd_multiply_simd", |bench| {
        bench.iter(|| black_box(cd_multiply_simd(black_box(&a_slice), black_box(&b_slice))))
    });
    group.bench_function("flat_avx2", |bench| {
        bench.iter(|| black_box(octonion_multiply_flat(black_box(&a), black_box(&b))))
    });
    group.finish();
}

fn bench_sedenion_multiply(c: &mut Criterion) {
    let a: [f64; 16] = [
        1.0, 0.1, -0.2, 0.3, 0.4, -0.5, 0.6, 0.7,
        -0.8, 0.9, 0.1, -0.2, 0.3, 0.4, -0.5, 0.6,
    ];
    let b: [f64; 16] = [
        0.5, -0.1, 0.2, -0.3, 0.4, 0.5, -0.6, 0.7,
        0.8, -0.9, 0.1, 0.2, -0.3, 0.4, 0.5, -0.6,
    ];
    let a_slice: Vec<f64> = a.to_vec();
    let b_slice: Vec<f64> = b.to_vec();

    let mut group = c.benchmark_group("sedenion_16d");
    group.bench_function("cd_multiply_scalar", |bench| {
        bench.iter(|| black_box(cd_multiply(black_box(&a_slice), black_box(&b_slice))))
    });
    group.bench_function("cd_multiply_simd", |bench| {
        bench.iter(|| black_box(cd_multiply_simd(black_box(&a_slice), black_box(&b_slice))))
    });
    group.bench_function("flat_avx2", |bench| {
        bench.iter(|| black_box(sedenion_multiply_flat(black_box(&a), black_box(&b))))
    });
    group.finish();
}

fn bench_dimension_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("cd_multiply_scaling");
    for dim in [4, 8, 16, 32, 64] {
        let a: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.123).sin()).collect();
        let b: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.456).cos()).collect();
        group.bench_with_input(BenchmarkId::new("scalar", dim), &dim, |bench, _| {
            bench.iter(|| black_box(cd_multiply(black_box(&a), black_box(&b))))
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_quaternion_multiply,
    bench_octonion_multiply,
    bench_sedenion_multiply,
    bench_dimension_scaling,
);
criterion_main!(benches);
