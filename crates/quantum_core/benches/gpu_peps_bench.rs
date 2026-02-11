//! GPU vs CPU benchmarks for PEPS row contraction.
//!
//! Compares performance of GPU-accelerated vs CPU PEPS row contraction
//! for varying tensor sizes. Compiled with `cargo bench --bench gpu_peps_bench --features gpu`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use faer::complex_native::c64;

/// Benchmark GPU vs CPU for varying tensor sizes (elements).
fn benchmark_peps_row_contraction(c: &mut Criterion) {
    let mut group = c.benchmark_group("peps_row_contraction");

    // Test sizes: 1K, 10K, 100K, 1M elements
    let sizes = vec![1_000, 10_000, 100_000, 1_000_000];

    for size in sizes {
        // Create test data
        let upper: Vec<c64> = (0..size)
            .map(|i| {
                let angle = (i as f64) * 0.001;
                c64::new(angle.cos(), angle.sin())
            })
            .collect();

        let lower: Vec<c64> = (0..size)
            .map(|i| {
                let angle = (i as f64) * 0.002;
                c64::new(angle.cos() * 0.5, angle.sin() * 0.5)
            })
            .collect();

        // Benchmark: element-wise complex multiplication
        // (This is what contract_rows does internally)
        group.bench_with_input(BenchmarkId::new("cpu_multiply", size), &size, |b, &_| {
            b.iter(|| {
                upper
                    .iter()
                    .zip(lower.iter())
                    .map(|(a, b)| c64::new(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re))
                    .collect::<Vec<_>>()
            })
        });

        // For GPU: only benchmark if feature is enabled
        #[cfg(feature = "gpu")]
        {
            group.bench_with_input(BenchmarkId::new("gpu_multiply", size), &size, |b, &_| {
                b.iter(|| {
                    quantum_core::gpu::peps::gpu_contract_rows_peps(
                        black_box(&upper),
                        black_box(&lower),
                    )
                })
            });
        }
    }

    group.finish();
}

/// Benchmark memory transfer overhead for GPU kernels.
#[cfg(feature = "gpu")]
fn benchmark_gpu_memory_transfer(c: &mut Criterion) {
    let mut group = c.benchmark_group("gpu_memory_transfer");

    // Test sizes: measure upload + download + kernel overhead
    let sizes = vec![10_000, 100_000, 1_000_000];

    for size in sizes {
        let data: Vec<f64> = (0..size).map(|i| (i as f64) * 0.001).collect();

        group.bench_with_input(
            BenchmarkId::new("host_to_device_to_host", size),
            &size,
            |b, &_| {
                b.iter(|| {
                    // Simulated transfer: copy to Vec and back
                    // In practice, this would be cudarc clone_htod + clone_dtoh
                    let _copy = black_box(&data).to_vec();
                    let _verify = _copy.len();
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_peps_row_contraction,);

// Conditionally include GPU memory transfer benchmark
#[cfg(feature = "gpu")]
criterion_group! {
    name = gpu_benches;
    config = Criterion::default().sample_size(10);
    targets = benchmark_gpu_memory_transfer
}

#[cfg(feature = "gpu")]
criterion_main!(benches, gpu_benches);

#[cfg(not(feature = "gpu"))]
criterion_main!(benches);
