//! Criterion benchmarks for tensor_avt CPU and GPU paths.

use std::hint::black_box;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
#[cfg(feature = "gpu")]
use gororoba_algebra::gpu::ComputeBackend;
use gororoba_algebra::gpu::TensorAVT;
#[cfg(feature = "gpu")]
use gororoba_algebra::gpu::is_gpu_available;
#[cfg(feature = "gpu")]
use std::time::Instant;

fn bench_large_enabled() -> bool {
    matches!(
        std::env::var("TENSOR_AVT_BENCH_LARGE").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn single_dims() -> Vec<usize> {
    let mut dims = vec![16usize, 256, 512, 1024];
    if bench_large_enabled() {
        dims.extend([2048, 4096]);
    }
    dims
}

fn batch_shapes() -> Vec<(usize, usize)> {
    let mut shapes = vec![(16usize, 1usize), (256, 16), (512, 16), (1024, 64)];
    if bench_large_enabled() {
        shapes.extend([(2048, 16), (4096, 8)]);
    }
    shapes
}

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
    for dim in single_dims() {
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
    for (dim, batch_size) in batch_shapes() {
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
    for (dim, batch_size) in batch_shapes() {
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
    for dim in single_dims() {
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
fn bench_tensor_avt_single_gpu_resident(c: &mut Criterion) {
    if !is_gpu_available() {
        return;
    }
    let mut group = c.benchmark_group("tensor_avt_single_gpu_resident");
    for dim in single_dims() {
        let avt = TensorAVT::new(dim);
        let a = values(dim, 0xA551 + dim as u64);
        let x = values(dim, 0xB661 + dim as u64);
        let mut session = avt
            .new_mul_session(ComputeBackend::Cuda, 1)
            .expect("create CUDA mul session");
        session.load_left(&a).expect("load left input");
        session.load_right(&x, 1).expect("load right input");
        group.bench_with_input(
            BenchmarkId::new("gpu-resident", dim),
            &dim,
            move |bench, _| {
                bench.iter_custom(|iters| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        session.run_single(&avt).expect("run resident single");
                    }
                    let output = session
                        .download_output(dim)
                        .expect("download resident single");
                    black_box(output);
                    start.elapsed()
                });
            },
        );
    }
    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_tensor_avt_batch_gpu(c: &mut Criterion) {
    if !is_gpu_available() {
        return;
    }
    let mut group = c.benchmark_group("tensor_avt_batch_gpu");
    for (dim, batch_size) in batch_shapes() {
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
fn bench_tensor_avt_batch_gpu_resident(c: &mut Criterion) {
    if !is_gpu_available() {
        return;
    }
    let mut group = c.benchmark_group("tensor_avt_batch_gpu_resident");
    for (dim, batch_size) in batch_shapes() {
        let avt = TensorAVT::new(dim);
        let a = values(dim, 0xC771 + dim as u64);
        let x_batch = values(dim * batch_size, 0xD881 + batch_size as u64);
        let len = dim * batch_size;
        let label = format!("gpu-resident-dim{dim}-batch{batch_size}");
        let mut session = avt
            .new_mul_session(ComputeBackend::Cuda, batch_size)
            .expect("create CUDA batch session");
        session.load_left(&a).expect("load batch left input");
        session
            .load_right(&x_batch, batch_size)
            .expect("load batch right input");
        group.bench_function(label, move |bench| {
            bench.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    session
                        .run_batch(&avt, batch_size)
                        .expect("run resident batch");
                }
                let output = session
                    .download_output(len)
                    .expect("download resident batch");
                black_box(output);
                start.elapsed()
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
    for (dim, batch_size) in batch_shapes() {
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

#[cfg(feature = "gpu")]
fn bench_tensor_avt_norm_gpu_resident(c: &mut Criterion) {
    if !is_gpu_available() {
        return;
    }
    let mut group = c.benchmark_group("tensor_avt_norm_sq_gpu_resident");
    for (dim, batch_size) in batch_shapes() {
        let avt = TensorAVT::new(dim);
        let vectors = values(dim * batch_size, 0xE991 + batch_size as u64);
        let label = format!("gpu-resident-dim{dim}-batch{batch_size}");
        let mut session = avt
            .new_norm_session(ComputeBackend::Cuda, batch_size)
            .expect("create CUDA norm session");
        session
            .load_vectors(&vectors, batch_size)
            .expect("load norm vectors");
        group.bench_function(label, move |bench| {
            bench.iter_custom(|iters| {
                let start = Instant::now();
                for _ in 0..iters {
                    session
                        .run_norms(&avt, batch_size)
                        .expect("run resident norms");
                }
                let output = session
                    .download_norms(batch_size)
                    .expect("download resident norms");
                black_box(output);
                start.elapsed()
            });
        });
    }
    group.finish();
}

#[cfg(not(feature = "gpu"))]
fn bench_tensor_avt_single_gpu(_: &mut Criterion) {}

#[cfg(not(feature = "gpu"))]
fn bench_tensor_avt_single_gpu_resident(_: &mut Criterion) {}

#[cfg(not(feature = "gpu"))]
fn bench_tensor_avt_batch_gpu(_: &mut Criterion) {}

#[cfg(not(feature = "gpu"))]
fn bench_tensor_avt_batch_gpu_resident(_: &mut Criterion) {}

#[cfg(not(feature = "gpu"))]
fn bench_tensor_avt_norm_gpu(_: &mut Criterion) {}

#[cfg(not(feature = "gpu"))]
fn bench_tensor_avt_norm_gpu_resident(_: &mut Criterion) {}

criterion_group!(
    benches,
    bench_tensor_avt_single,
    bench_tensor_avt_batch,
    bench_tensor_avt_norm,
    bench_tensor_avt_single_gpu,
    bench_tensor_avt_single_gpu_resident,
    bench_tensor_avt_batch_gpu,
    bench_tensor_avt_batch_gpu_resident,
    bench_tensor_avt_norm_gpu,
    bench_tensor_avt_norm_gpu_resident,
);
criterion_main!(benches);
