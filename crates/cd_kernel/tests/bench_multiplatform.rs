//! Multi-platform CD kernel benchmark for MICRO26 paper.
//!
//! Measures throughput of batch_sliding_associator_norms_f32 at each
//! power-of-two dimension from 8D to 256D. Outputs structured results
//! for cross-platform comparison.
//!
//! Run: cargo test --release -p cd_kernel -- bench_multiplatform --nocapture

use std::time::Instant;

fn random_vectors_f32(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
                })
                .collect()
        })
        .collect()
}

fn random_vectors_f64(n: usize, dim: usize, seed: u64) -> Vec<Vec<f64>> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            (0..dim)
                .map(|_| {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    ((state >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0
                })
                .collect()
        })
        .collect()
}

#[test]
fn bench_multiplatform_f32() {
    let n = 2000;
    let warmup = 500;
    let iters = 3;

    println!("\n=== CD Kernel Multi-Platform Benchmark (f32) ===");
    println!("  Vectors per run: {}, Warmup: {}, Iterations: {}", n, warmup, iters);
    println!("  {:>6} {:>12} {:>12} {:>12} {:>12}",
        "Dim", "Mean (ms)", "Min (ms)", "kvec/s", "norms/run");

    for &dim in &[8, 16, 32, 64, 128, 256] {
        let vectors = random_vectors_f32(n, dim, 42);

        // Warmup
        let warmup_vecs = random_vectors_f32(warmup, dim, 99);
        let _ = cd_kernel::batch_sliding_associator_norms_f32(&warmup_vecs, dim);

        // Timed runs
        let mut times_ms = Vec::new();
        for _ in 0..iters {
            let t0 = Instant::now();
            let norms = cd_kernel::batch_sliding_associator_norms_f32(&vectors, dim);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            times_ms.push(ms);
            assert!(!norms.is_empty());
        }

        let mean_ms = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
        let min_ms = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
        let kvecs = n as f64 / mean_ms;
        let norms_count = n - 2;

        println!("  {:>6} {:>12.3} {:>12.3} {:>12.1} {:>12}",
            dim, mean_ms, min_ms, kvecs, norms_count);
    }
}

#[test]
fn bench_multiplatform_f64() {
    let n = 2000;
    let warmup = 500;
    let iters = 3;

    println!("\n=== CD Kernel Multi-Platform Benchmark (f64) ===");
    println!("  Vectors per run: {}, Warmup: {}, Iterations: {}", n, warmup, iters);
    println!("  {:>6} {:>12} {:>12} {:>12} {:>12}",
        "Dim", "Mean (ms)", "Min (ms)", "kvec/s", "norms/run");

    for &dim in &[8, 16, 32, 64, 128, 256] {
        let vectors = random_vectors_f64(n, dim, 42);

        // Warmup
        let warmup_vecs = random_vectors_f64(warmup, dim, 99);
        let _ = cd_kernel::batch_sliding_associator_norms_dispatch(&warmup_vecs, dim, "f64");

        // Timed runs
        let mut times_ms = Vec::new();
        for _ in 0..iters {
            let t0 = Instant::now();
            let norms = cd_kernel::batch_sliding_associator_norms_dispatch(&vectors, dim, "f64");
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            times_ms.push(ms);
            assert!(!norms.is_empty());
        }

        let mean_ms = times_ms.iter().sum::<f64>() / times_ms.len() as f64;
        let min_ms = times_ms.iter().cloned().fold(f64::INFINITY, f64::min);
        let kvecs = n as f64 / mean_ms;
        let norms_count = n - 2;

        println!("  {:>6} {:>12.3} {:>12.3} {:>12.1} {:>12}",
            dim, mean_ms, min_ms, kvecs, norms_count);
    }
}

#[test]
fn bench_multiplatform_ablation() {
    // Ablation: measure each layer's contribution by testing at different
    // vector counts to separate compute-bound from memory-bound behavior
    let dim = 32;

    println!("\n=== Ablation: Throughput vs Batch Size at 32D (f32) ===");
    println!("  {:>8} {:>12} {:>12}", "N", "ms", "kvec/s");

    for &n in &[100, 500, 1000, 2000, 5000, 10000] {
        let vectors = random_vectors_f32(n, dim, 42);
        let t0 = Instant::now();
        let _ = cd_kernel::batch_sliding_associator_norms_f32(&vectors, dim);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let kvecs = n as f64 / ms;
        println!("  {:>8} {:>12.3} {:>12.1}", n, ms, kvecs);
    }
}
