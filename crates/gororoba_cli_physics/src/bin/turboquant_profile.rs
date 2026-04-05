//! TurboQuant profiling binary for perf/flamegraph analysis.
//!
//! Runs the full quantization pipeline in a tight loop to generate
//! profiling data.  Use with:
//!   cargo build --release -p gororoba_cli_physics --bin turboquant-profile
//!   perf record -g ./target/release/turboquant-profile --iterations 100
//!   perf script | inferno-collapse-perf | inferno-flamegraph > flame.svg
//!
//! Or with cargo-flamegraph:
//!   cargo flamegraph --bin turboquant-profile -- --iterations 100

use std::{hint::black_box, time::Instant};

use cd_kernel::turboquant::synthesized::SynthesizedQuantizer;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let iterations: usize = args
        .get(1)
        .and_then(|s| s.strip_prefix("--iterations=").or(Some(s)))
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    let d = 128;
    let bits = 3;
    let n_vectors = 1000;

    println!(
        "TurboQuant profiling: d={}, bits={}, vectors={}, iterations={}",
        d, bits, n_vectors, iterations
    );

    // Generate data once
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};
    let mut rng = ChaCha20Rng::seed_from_u64(42);
    let normal = StandardNormal;
    let vectors: Vec<Vec<f64>> = (0..n_vectors)
        .map(|_| (0..d).map(|_| normal.sample(&mut rng)).collect())
        .collect();

    // Warm up
    let sq = SynthesizedQuantizer::new_fast(d, bits, 42);
    let mut buf = vec![0.0f64; 3 * d];
    for v in &vectors[..10] {
        black_box(sq.quantize(v, &mut buf));
    }

    // Profile loop
    println!("Starting profile loop...");
    let t0 = Instant::now();

    for iter in 0..iterations {
        // Full pipeline: quantize all vectors
        for v in &vectors {
            black_box(sq.quantize(v, &mut buf));
        }

        if (iter + 1) % 10 == 0 {
            let elapsed = t0.elapsed().as_secs_f64();
            let total_vecs = (iter + 1) * n_vectors;
            println!(
                "  iter {}/{}: {:.0} kvec/s",
                iter + 1,
                iterations,
                total_vecs as f64 / elapsed / 1000.0
            );
        }
    }

    let total = t0.elapsed().as_secs_f64();
    let total_vecs = iterations * n_vectors;
    println!(
        "\nProfile complete: {} vectors in {:.1}s ({:.0} kvec/s)",
        total_vecs,
        total,
        total_vecs as f64 / total / 1000.0
    );
}
