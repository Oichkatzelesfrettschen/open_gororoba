//! Criterion benchmarks for algebra_core.
//!
//! Benchmarks the core computational routines:
//! - Cayley-Dickson multiplication at various dimensions (8, 16, 32, 64)
//! - Associator computation (single and batch)
//! - Zero-divisor search
//! - Octonion field evolution

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use algebra_core::{
    cd_multiply, cd_associator, cd_associator_norm,
    batch_associator_norms, batch_associator_norms_parallel,
    find_zero_divisors,
    oct_multiply, stormer_verlet_step, gaussian_wave_packet, FieldParams,
};

/// Benchmark Cayley-Dickson multiplication at various dimensions.
fn bench_cd_multiply(c: &mut Criterion) {
    let mut group = c.benchmark_group("cd_multiply");

    for dim in [8, 16, 32, 64] {
        // Create random-ish test vectors
        let a: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.1).sin()).collect();
        let b: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.2).cos()).collect();

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |bench, _| {
            bench.iter(|| {
                cd_multiply(black_box(&a), black_box(&b))
            });
        });
    }

    group.finish();
}

/// Benchmark associator computation.
fn bench_associator(c: &mut Criterion) {
    let mut group = c.benchmark_group("cd_associator");

    for dim in [8, 16, 32] {
        let a: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.1).sin()).collect();
        let b: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.2).cos()).collect();
        let d: Vec<f64> = (0..dim).map(|i| (i as f64 * 0.3).sin()).collect();

        group.bench_with_input(BenchmarkId::new("single/dim", dim), &dim, |bench, _| {
            bench.iter(|| {
                cd_associator(black_box(&a), black_box(&b), black_box(&d))
            });
        });

        group.bench_with_input(BenchmarkId::new("norm/dim", dim), &dim, |bench, _| {
            bench.iter(|| {
                cd_associator_norm(black_box(&a), black_box(&b), black_box(&d))
            });
        });
    }

    group.finish();
}

/// Benchmark batch associator computation.
fn bench_batch_associator(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_associator");

    let dim = 16;
    let n_triples = 1000;

    // Generate flat test arrays (contiguous memory for SIMD)
    let a_flat: Vec<f64> = (0..(dim * n_triples))
        .map(|i| (i as f64 * 0.01).sin())
        .collect();
    let b_flat: Vec<f64> = (0..(dim * n_triples))
        .map(|i| (i as f64 * 0.02).cos())
        .collect();
    let c_flat: Vec<f64> = (0..(dim * n_triples))
        .map(|i| (i as f64 * 0.03).sin())
        .collect();

    group.bench_function("sequential/1000x16", |bench| {
        bench.iter(|| {
            batch_associator_norms(
                black_box(&a_flat),
                black_box(&b_flat),
                black_box(&c_flat),
                black_box(dim),
                black_box(n_triples),
            )
        });
    });

    group.bench_function("parallel/1000x16", |bench| {
        bench.iter(|| {
            batch_associator_norms_parallel(
                black_box(&a_flat),
                black_box(&b_flat),
                black_box(&c_flat),
                black_box(dim),
                black_box(n_triples),
            )
        });
    });

    group.finish();
}

/// Benchmark zero-divisor search.
fn bench_zero_divisor_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("zero_divisor_search");
    group.sample_size(20); // ZD search is slow, reduce samples

    for dim in [16, 32] {
        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |bench, &dim| {
            bench.iter(|| {
                find_zero_divisors(black_box(dim), black_box(1e-10))
            });
        });
    }

    group.finish();
}

/// Benchmark octonion multiplication.
fn bench_octonion_multiply(c: &mut Criterion) {
    let a = [1.0, 0.5, 0.3, 0.2, 0.1, 0.4, 0.6, 0.7];
    let b = [0.8, 0.2, 0.4, 0.5, 0.3, 0.1, 0.2, 0.9];

    c.bench_function("oct_multiply", |bench| {
        bench.iter(|| {
            oct_multiply(black_box(&a), black_box(&b))
        });
    });
}

/// Benchmark Stormer-Verlet field evolution step.
fn bench_stormer_verlet(c: &mut Criterion) {
    let mut group = c.benchmark_group("stormer_verlet");

    for n_sites in [32, 64, 128] {
        let params = FieldParams {
            n: n_sites,
            l: n_sites as f64 * 0.1,
            mass: 1.0,
            coupling: 0.1,
            dt: 0.01,
        };

        let (phi, pi) = gaussian_wave_packet(&params);

        group.bench_with_input(BenchmarkId::new("n_sites", n_sites), &n_sites, |bench, _| {
            let mut phi_mut = phi.clone();
            let mut pi_mut = pi.clone();
            bench.iter(|| {
                stormer_verlet_step(
                    black_box(&mut phi_mut),
                    black_box(&mut pi_mut),
                    black_box(&params),
                );
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_cd_multiply,
    bench_associator,
    bench_batch_associator,
    bench_zero_divisor_search,
    bench_octonion_multiply,
    bench_stormer_verlet,
);

criterion_main!(benches);
