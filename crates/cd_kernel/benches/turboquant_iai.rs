//! iai-callgrind benchmarks for TurboQuant instruction-count regression detection.
//!
//! These benchmarks measure instruction counts (not wall-clock time), making
//! them CI-stable and immune to CPU frequency scaling, thermal throttling,
//! and system load.
//!
//! Run with: cargo bench --bench turboquant_iai
//! Requires: valgrind installed (iai-callgrind runs under callgrind)

use cd_kernel::{lloyd_max, turboquant::rotation::wht_inplace};
use iai_callgrind::{library_benchmark, library_benchmark_group, main};

fn setup_wht_data() -> Vec<f64> {
    (0..128).map(|i| (i as f64 * 0.1).sin()).collect()
}

fn setup_codebook() -> (Vec<f32>, Vec<f32>) {
    let cb = lloyd_max::get_codebook(128, 3);
    (cb.centroids.clone(), cb.boundaries.clone())
}

#[library_benchmark]
#[bench::d128(setup_wht_data())]
fn bench_wht_128(mut data: Vec<f64>) {
    wht_inplace(&mut data);
}

#[library_benchmark]
fn bench_codebook_solve() {
    let _cb = lloyd_max::solve_lloyd_max(128, 3);
}

#[library_benchmark]
#[bench::d128(setup_codebook())]
fn bench_boundary_quantize((_centroids, boundaries): (Vec<f32>, Vec<f32>)) {
    let sigma = 1.0 / (128.0f32).sqrt();
    let values: Vec<f32> = (0..128)
        .map(|i| ((i as f32 * 0.618) % 1.0 - 0.5) * 7.0 * sigma)
        .collect();
    for &v in &values {
        let mut idx = 0u8;
        for &b in boundaries.iter() {
            if v > b {
                idx += 1;
            } else {
                break;
            }
        }
        std::hint::black_box(idx);
    }
}

library_benchmark_group!(
    name = turboquant_iai;
    benchmarks = bench_wht_128, bench_codebook_solve, bench_boundary_quantize
);

main!(library_benchmark_groups = turboquant_iai);
