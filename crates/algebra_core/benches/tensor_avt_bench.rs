//! Criterion benchmarks for tensor_avt CPU and GPU paths.

use algebra_core::gpu::TensorAVT;
#[cfg(feature = "gpu")]
use algebra_core::gpu::is_gpu_available;
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

fn values(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|idx| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407 + idx as u64);
            let unit = ((state >> 32) as u32) as f32 / (u32::MAX as f32);
            (unit * 2.0) - 1.0
        })
        .collect()
}

fn bench_tensor_avt_single(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_avt_single");
    for dim in [16usize, 256, 512, 1024] {
        let avt = TensorAVT::new(dim);
        let a = values(dim, 0xA11CE + dim as u64);
        let x = values(dim, 0xBADC0DE + dim as u64);
        group.bench_with_input(BenchmarkId::new("cpu", dim), &dim, |bench, _| {
            bench.iter(|| avt.compute_cd_mul(black_box(&a), black_box(&x)).unwrap());
        });
    }
    group.finish();
}

fn bench_tensor_avt_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_avt_batch");
    for (dim, batch_size) in [(16usize, 1usize), (256, 16), (512, 16), (1024, 64)] {
        let avt = TensorAVT::new(dim);
        let a = values(dim, 0xCAFE + dim as u64);
        let x_batch = values(dim * batch_size, 0xFACEFEED + batch_size as u64);
        let label = format!("cpu-dim{dim}-batch{batch_size}");
        group.bench_function(label, |bench| {
            bench.iter(|| {
                avt.compute_cd_mul_batch(black_box(&a), black_box(&x_batch), black_box(batch_size))
                    .unwrap()
            });
        });
    }
    group.finish();
}

fn bench_tensor_avt_norm(c: &mut Criterion) {
    let mut group = c.benchmark_group("tensor_avt_norm_sq");
    for (dim, batch_size) in [(16usize, 1usize), (256, 16), (512, 16), (1024, 64)] {
        let avt = TensorAVT::new(dim);
        let vectors = values(dim * batch_size, 0xDEADBEEF + batch_size as u64);
        let label = format!("cpu-dim{dim}-batch{batch_size}");
        group.bench_function(label, |bench| {
            bench.iter(|| {
                avt.compute_norm_sq_batch(black_box(&vectors), black_box(batch_size))
                    .unwrap()
            });
        });
    }
    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_tensor_avt_single_gpu(c: &mut Criterion) {
    if !is_gpu_available() {
        return;
    }
    let mut group = c.benchmark_group("tensor_avt_single_gpu");
    for dim in [16usize, 256, 512, 1024] {
        let avt = TensorAVT::new(dim);
        let a = values(dim, 0x5151 + dim as u64);
        let x = values(dim, 0x6161 + dim as u64);
        group.bench_with_input(BenchmarkId::new("gpu", dim), &dim, |bench, _| {
            bench.iter(|| avt.compute_cd_mul(black_box(&a), black_box(&x)).unwrap());
        });
    }
    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_tensor_avt_batch_gpu(c: &mut Criterion) {
    if !is_gpu_available() {
        return;
    }
    let mut group = c.benchmark_group("tensor_avt_batch_gpu");
    for (dim, batch_size) in [(16usize, 1usize), (256, 16), (512, 16), (1024, 64)] {
        let avt = TensorAVT::new(dim);
        let a = values(dim, 0x7171 + dim as u64);
        let x_batch = values(dim * batch_size, 0x8181 + batch_size as u64);
        let label = format!("gpu-dim{dim}-batch{batch_size}");
        group.bench_function(label, |bench| {
            bench.iter(|| {
                avt.compute_cd_mul_batch(black_box(&a), black_box(&x_batch), black_box(batch_size))
                    .unwrap()
            });
        });
    }
    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_tensor_avt_norm_gpu(c: &mut Criterion) {
    if !is_gpu_available() {
        return;
    }
    let mut group = c.benchmark_group("tensor_avt_norm_sq_gpu");
    for (dim, batch_size) in [(16usize, 1usize), (256, 16), (512, 16), (1024, 64)] {
        let avt = TensorAVT::new(dim);
        let vectors = values(dim * batch_size, 0x9191 + batch_size as u64);
        let label = format!("gpu-dim{dim}-batch{batch_size}");
        group.bench_function(label, |bench| {
            bench.iter(|| {
                avt.compute_norm_sq_batch(black_box(&vectors), black_box(batch_size))
                    .unwrap()
            });
        });
    }
    group.finish();
}

#[cfg(not(feature = "gpu"))]
fn bench_tensor_avt_single_gpu(_: &mut Criterion) {}

#[cfg(not(feature = "gpu"))]
fn bench_tensor_avt_batch_gpu(_: &mut Criterion) {}

#[cfg(not(feature = "gpu"))]
fn bench_tensor_avt_norm_gpu(_: &mut Criterion) {}

criterion_group!(
    benches,
    bench_tensor_avt_single,
    bench_tensor_avt_batch,
    bench_tensor_avt_norm,
    bench_tensor_avt_single_gpu,
    bench_tensor_avt_batch_gpu,
    bench_tensor_avt_norm_gpu,
);
criterion_main!(benches);
