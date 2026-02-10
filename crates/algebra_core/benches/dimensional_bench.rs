//! Criterion benchmarks for dimensional analysis of Cayley-Dickson zero-divisor structures.
//!
//! Benchmarks cover the three computational stages of the APT verification pipeline:
//! 1. Component extraction (motif_components_for_cross_assessors)
//! 2. Eta matrix computation (cd_basis_mul_sign-based psi/eta)
//! 3. Triangle enumeration & APT census
//!
//! Run with: cargo bench --bench dimensional_bench -p algebra_core
//! GPU benchmarks: cargo bench --bench dimensional_bench -p algebra_core --features gpu

use algebra_core::analysis::boxkites::{cross_assessors, motif_components_for_cross_assessors};
use algebra_core::cd_basis_mul_sign;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::time::Duration;

/// Benchmark component extraction at various dimensions.
/// This is Stage 1 of the pipeline: building the cross-assessor graph and finding
/// connected components via zero-product edge detection.
fn bench_component_extraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("component_extraction");
    group.sample_size(10);

    for dim in [16, 32, 64, 128] {
        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |bench, &dim| {
            bench.iter(|| motif_components_for_cross_assessors(black_box(dim)));
        });
    }

    group.finish();
}

/// Benchmark cross-assessor enumeration (input to component extraction).
fn bench_cross_assessor_enumeration(c: &mut Criterion) {
    let mut group = c.benchmark_group("cross_assessors");

    for dim in [16, 32, 64, 128, 256] {
        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |bench, &dim| {
            bench.iter(|| cross_assessors(black_box(dim)));
        });
    }

    group.finish();
}

/// Benchmark cd_basis_mul_sign at various dimensions.
/// This is the core primitive: O(log dim) per call, called O(n^2) times for eta matrix.
fn bench_basis_mul_sign(c: &mut Criterion) {
    let mut group = c.benchmark_group("cd_basis_mul_sign");

    for dim in [16, 32, 64, 128, 256, 512] {
        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |bench, &dim| {
            bench.iter(|| {
                // Compute psi for a representative set of (i,j) pairs
                let half = dim / 2;
                let mut sum = 0i64;
                for i in 1..half.min(32) {
                    for j in half..dim.min(half + 32) {
                        sum += cd_basis_mul_sign(black_box(dim), black_box(i), black_box(j)) as i64;
                    }
                }
                sum
            });
        });
    }

    group.finish();
}

/// Benchmark eta matrix computation (CPU).
/// eta(i,j) = psi(i, j+dim/2) XOR psi(j, i+dim/2) for i,j in [0, dim/2).
fn bench_eta_matrix_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("eta_matrix_cpu");
    group.sample_size(10);

    let psi = |dim: usize, i: usize, j: usize| -> u8 {
        if cd_basis_mul_sign(dim, i, j) == 1 {
            0
        } else {
            1
        }
    };

    for dim in [16, 32, 64, 128, 256] {
        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |bench, &dim| {
            bench.iter(|| {
                let half = dim / 2;
                let mut eta = vec![0u8; half * half];
                for i in 0..half {
                    for j in 0..half {
                        let psi_ij = psi(dim, i, j + half);
                        let psi_ji = psi(dim, j, i + half);
                        eta[i * half + j] = psi_ij ^ psi_ji;
                    }
                }
                eta
            });
        });
    }

    group.finish();
}

/// Benchmark APT triangle census (exhaustive, CPU).
/// Enumerates all triangles in each component and classifies as pure/mixed.
fn bench_apt_census_exhaustive(c: &mut Criterion) {
    let mut group = c.benchmark_group("apt_census_exhaustive");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(15));

    let psi = |dim: usize, i: usize, j: usize| -> u8 {
        if cd_basis_mul_sign(dim, i, j) == 1 {
            0
        } else {
            1
        }
    };

    // Only small dimensions for exhaustive census
    for dim in [16, 32, 64] {
        let components = motif_components_for_cross_assessors(dim);

        group.bench_with_input(BenchmarkId::new("dim", dim), &dim, |bench, &dim| {
            bench.iter(|| {
                let mut total = 0usize;
                let mut pure = 0usize;

                for comp in &components {
                    let nodes: Vec<_> = comp.nodes.iter().collect();
                    for i in 0..nodes.len() {
                        for j in (i + 1)..nodes.len() {
                            for k in (j + 1)..nodes.len() {
                                let &(ai, bi) = nodes[i];
                                let &(aj, bj) = nodes[j];
                                let &(ak, bk) = nodes[k];

                                // Anti-diagonal parity: eta(a,b) = psi(lo_a, hi_b) XOR psi(hi_a, lo_b)
                                let eta_ij = psi(dim, ai, bj) ^ psi(dim, bi, aj);
                                let eta_ik = psi(dim, ai, bk) ^ psi(dim, bi, ak);
                                let eta_jk = psi(dim, aj, bk) ^ psi(dim, bj, ak);

                                total += 1;
                                if eta_ij == eta_ik && eta_ik == eta_jk {
                                    pure += 1;
                                }
                            }
                        }
                    }
                }
                (total, pure)
            });
        });
    }

    group.finish();
}

/// Benchmark APT census via Monte Carlo sampling (CPU).
/// Samples random triangles for large-dimension performance comparison.
fn bench_apt_census_monte_carlo(c: &mut Criterion) {
    let mut group = c.benchmark_group("apt_census_monte_carlo");
    group.sample_size(10);

    let psi = |dim: usize, i: usize, j: usize| -> u8 {
        if cd_basis_mul_sign(dim, i, j) == 1 {
            0
        } else {
            1
        }
    };

    for dim in [32, 64, 128] {
        let components = motif_components_for_cross_assessors(dim);
        let all_nodes: Vec<(usize, usize)> = components
            .iter()
            .flat_map(|comp| comp.nodes.iter().copied())
            .collect();

        let n_samples = 10_000usize;

        group.bench_with_input(
            BenchmarkId::new("dim", dim),
            &(dim, &all_nodes),
            |bench, &(dim, ref nodes)| {
                bench.iter(|| {
                    let mut rng_state = 42u64;
                    let next_rng = |state: &mut u64| -> u64 {
                        *state = state.wrapping_add(0x9e3779b97f4a7c15);
                        let z = *state ^ (*state >> 30);
                        let z_mul = z.wrapping_mul(0xbf58476d1ce4e5b9);
                        z_mul ^ (z_mul >> 27)
                    };

                    let mut pure = 0usize;
                    let n = nodes.len();

                    for _ in 0..n_samples {
                        let i = (next_rng(&mut rng_state) as usize) % n;
                        let mut j = (next_rng(&mut rng_state) as usize) % n;
                        while j == i {
                            j = (next_rng(&mut rng_state) as usize) % n;
                        }
                        let mut k = (next_rng(&mut rng_state) as usize) % n;
                        while k == i || k == j {
                            k = (next_rng(&mut rng_state) as usize) % n;
                        }

                        let (ai, bi) = nodes[i];
                        let (aj, bj) = nodes[j];
                        let (ak, bk) = nodes[k];

                        // Anti-diagonal parity: eta(a,b) = psi(lo_a, hi_b) XOR psi(hi_a, lo_b)
                        let eta_ij = psi(dim, ai, bj) ^ psi(dim, bi, aj);
                        let eta_ik = psi(dim, ai, bk) ^ psi(dim, bi, ak);
                        let eta_jk = psi(dim, aj, bk) ^ psi(dim, bj, ak);

                        if eta_ij == eta_ik && eta_ik == eta_jk {
                            pure += 1;
                        }
                    }
                    pure
                });
            },
        );
    }

    group.finish();
}

/// Benchmark scaling: component extraction across wide dimensional range.
/// Reports time per dimension for log-log scaling analysis.
fn bench_scaling_analysis(c: &mut Criterion) {
    let mut group = c.benchmark_group("scaling_analysis");
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(20));

    // Test O(n^2) scaling of cross-assessor enumeration
    for dim in [16, 32, 64, 128] {
        group.bench_with_input(
            BenchmarkId::new("cross_assessors", dim),
            &dim,
            |bench, &dim| {
                bench.iter(|| cross_assessors(black_box(dim)));
            },
        );
    }

    // Test O(n^3) scaling of exhaustive triangle count (dims 16, 32 only)
    for dim in [16, 32] {
        let components = motif_components_for_cross_assessors(dim);
        group.bench_with_input(
            BenchmarkId::new("triangle_count", dim),
            &dim,
            |bench, &_dim| {
                bench.iter(|| {
                    let mut total = 0usize;
                    for comp in &components {
                        let n = comp.nodes.len();
                        total += n * (n - 1) * (n - 2) / 6;
                    }
                    total
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_component_extraction,
    bench_cross_assessor_enumeration,
    bench_basis_mul_sign,
    bench_eta_matrix_cpu,
    bench_apt_census_exhaustive,
    bench_apt_census_monte_carlo,
    bench_scaling_analysis,
);

criterion_main!(benches);
