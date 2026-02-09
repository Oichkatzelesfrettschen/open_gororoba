//! Criterion benchmarks for quantum_core.
//!
//! Benchmarks the core quantum simulation routines:
//! - Matrix Product State operations (gates, entropy)
//! - Harper-Hofstadter eigenspectrum
//! - FHS Chern number calculation

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use quantum_core::{fhs_chern_numbers, harper_hamiltonian, MatrixProductState};

/// Benchmark MPS single-qubit gate application.
fn bench_mps_single_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps_single_gate");

    for n_qubits in [4, 8, 12] {
        let mps = MatrixProductState::new_zero_state(n_qubits);

        group.bench_with_input(
            BenchmarkId::new("hadamard/n", n_qubits),
            &n_qubits,
            |bench, _| {
                let mut mps_mut = mps.clone();
                bench.iter(|| {
                    mps_mut.apply_hadamard(black_box(0));
                });
            },
        );

        group.bench_with_input(BenchmarkId::new("x/n", n_qubits), &n_qubits, |bench, _| {
            let mut mps_mut = mps.clone();
            bench.iter(|| {
                mps_mut.apply_x(black_box(0));
            });
        });
    }

    group.finish();
}

/// Benchmark MPS two-qubit gate (CNOT) application.
fn bench_mps_two_qubit_gate(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps_two_qubit_gate");

    for n_qubits in [4, 8, 12] {
        // Create a state with some entanglement
        let mut mps = MatrixProductState::new_zero_state(n_qubits);
        mps.apply_hadamard(0);

        group.bench_with_input(
            BenchmarkId::new("cnot/n", n_qubits),
            &n_qubits,
            |bench, _| {
                let mut mps_mut = mps.clone();
                bench.iter(|| {
                    mps_mut.apply_cnot(black_box(0), black_box(1));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark MPS entropy measurement.
fn bench_mps_entropy(c: &mut Criterion) {
    let mut group = c.benchmark_group("mps_entropy");

    for n_qubits in [4, 8, 12] {
        // Create an entangled state (Bell-like chain)
        let mut mps = MatrixProductState::new_zero_state(n_qubits);
        mps.apply_hadamard(0);
        for i in 0..(n_qubits - 1) {
            mps.apply_cnot(i, i + 1);
        }

        group.bench_with_input(BenchmarkId::new("n", n_qubits), &n_qubits, |bench, _| {
            bench.iter(|| mps.measure_entropy(black_box(n_qubits / 2)));
        });
    }

    group.finish();
}

/// Benchmark Harper Hamiltonian construction.
fn bench_harper_hamiltonian(c: &mut Criterion) {
    let mut group = c.benchmark_group("harper_hamiltonian");

    for q in [4, 8, 16] {
        group.bench_with_input(BenchmarkId::new("q", q), &q, |bench, &q| {
            bench.iter(|| {
                harper_hamiltonian(black_box(0.5), black_box(0.3), black_box(1), black_box(q))
            });
        });
    }

    group.finish();
}

/// Benchmark FHS Chern number calculation.
fn bench_fhs_chern(c: &mut Criterion) {
    let mut group = c.benchmark_group("fhs_chern");
    group.sample_size(10); // FHS is slow, reduce samples

    for n_grid in [11, 17] {
        group.bench_with_input(
            BenchmarkId::new("grid", n_grid),
            &n_grid,
            |bench, &n_grid| {
                bench.iter(|| fhs_chern_numbers(black_box(1), black_box(4), black_box(n_grid)));
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_mps_single_gate,
    bench_mps_two_qubit_gate,
    bench_mps_entropy,
    bench_harper_hamiltonian,
    bench_fhs_chern,
);

criterion_main!(benches);
