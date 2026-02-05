//! Benchmarks for Cayley-Dickson algebra operations.

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rand::prelude::*;
use rand::rngs::StdRng;

// Import from the library (without python feature)
use gororoba_kernels::algebra::{
    cd_multiply, cd_conjugate, cd_norm_sq, cd_associator_norm,
    batch_associator_norms, batch_associator_norms_parallel,
    find_zero_divisors,
};
use gororoba_kernels::clifford::{gamma_matrices_cl8, verify_clifford_relation};

fn random_element(dim: usize, rng: &mut StdRng) -> Vec<f64> {
    (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

fn bench_cd_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("cd_multiply");
    let mut rng = StdRng::seed_from_u64(42);

    for dim in [4, 8, 16, 32, 64] {
        let a = random_element(dim, &mut rng);
        let b = random_element(dim, &mut rng);

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| cd_multiply(black_box(&a), black_box(&b)))
        });
    }
    group.finish();
}

fn bench_cd_conjugate(c: &mut Criterion) {
    let mut group = c.benchmark_group("cd_conjugate");
    let mut rng = StdRng::seed_from_u64(42);

    for dim in [4, 8, 16, 32, 64] {
        let a = random_element(dim, &mut rng);

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| cd_conjugate(black_box(&a)))
        });
    }
    group.finish();
}

fn bench_cd_norm_sq(c: &mut Criterion) {
    let mut group = c.benchmark_group("cd_norm_sq");
    let mut rng = StdRng::seed_from_u64(42);

    for dim in [4, 8, 16, 32, 64] {
        let a = random_element(dim, &mut rng);

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| cd_norm_sq(black_box(&a)))
        });
    }
    group.finish();
}

fn bench_associator_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("cd_associator_norm");
    let mut rng = StdRng::seed_from_u64(42);

    for dim in [4, 8, 16, 32] {
        let a = random_element(dim, &mut rng);
        let b = random_element(dim, &mut rng);
        let x = random_element(dim, &mut rng);

        group.bench_with_input(BenchmarkId::from_parameter(dim), &dim, |bench, _| {
            bench.iter(|| cd_associator_norm(black_box(&a), black_box(&b), black_box(&x)))
        });
    }
    group.finish();
}

fn bench_batch_associator_sequential_vs_parallel(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_associator");
    let mut rng = StdRng::seed_from_u64(42);

    let dim = 16;
    let n_triples = 1000;

    let a_flat: Vec<f64> = (0..dim * n_triples).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let b_flat: Vec<f64> = (0..dim * n_triples).map(|_| rng.gen_range(-1.0..1.0)).collect();
    let c_flat: Vec<f64> = (0..dim * n_triples).map(|_| rng.gen_range(-1.0..1.0)).collect();

    group.bench_function("sequential_1000", |bench| {
        bench.iter(|| {
            batch_associator_norms(
                black_box(&a_flat),
                black_box(&b_flat),
                black_box(&c_flat),
                dim,
                n_triples,
            )
        })
    });

    group.bench_function("parallel_1000", |bench| {
        bench.iter(|| {
            batch_associator_norms_parallel(
                black_box(&a_flat),
                black_box(&b_flat),
                black_box(&c_flat),
                dim,
                n_triples,
            )
        })
    });

    group.finish();
}

fn bench_find_zero_divisors(c: &mut Criterion) {
    let mut group = c.benchmark_group("find_zero_divisors");
    group.sample_size(10); // Expensive operation

    group.bench_function("sedenion_16", |bench| {
        bench.iter(|| find_zero_divisors(black_box(16), black_box(1e-10)))
    });

    group.finish();
}

fn bench_clifford_gamma(c: &mut Criterion) {
    let mut group = c.benchmark_group("clifford");

    group.bench_function("gamma_matrices_cl8", |bench| {
        bench.iter(|| gamma_matrices_cl8())
    });

    let gammas = gamma_matrices_cl8();
    group.bench_function("verify_clifford_relation", |bench| {
        bench.iter(|| verify_clifford_relation(black_box(&gammas), 1e-10))
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_cd_multiply,
    bench_cd_conjugate,
    bench_cd_norm_sq,
    bench_associator_norm,
    bench_batch_associator_sequential_vs_parallel,
    bench_find_zero_divisors,
    bench_clifford_gamma,
);

criterion_main!(benches);
