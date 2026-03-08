//! Deterministic callgrind benches for tensor_avt host-side costs.

use algebra_core::gpu::TensorAVT;
#[cfg(feature = "gpu")]
use algebra_core::gpu::is_gpu_available;
use iai_callgrind::{library_benchmark, library_benchmark_group, main};
use std::hint::black_box;

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

#[library_benchmark]
fn cd_mul_cpu_256() -> Vec<f32> {
    let dim = 256;
    let avt = TensorAVT::new(dim);
    let a = values(dim, 0xA11CE);
    let x = values(dim, 0xBADC0DE);
    avt.compute_cd_mul(black_box(&a), black_box(&x))
        .expect("single")
}

#[library_benchmark]
fn cd_mul_cpu_1024() -> Vec<f32> {
    let dim = 1024;
    let avt = TensorAVT::new(dim);
    let a = values(dim, 0xA11CE + dim as u64);
    let x = values(dim, 0xBADC0DE + dim as u64);
    avt.compute_cd_mul(black_box(&a), black_box(&x))
        .expect("single")
}

#[library_benchmark]
fn cd_mul_batch_cpu_256x64() -> Vec<f32> {
    let dim = 256;
    let batch_size = 64;
    let avt = TensorAVT::new(dim);
    let a = values(dim, 0xCAFE);
    let x_batch = values(dim * batch_size, 0xFACEFEED);
    avt.compute_cd_mul_batch(black_box(&a), black_box(&x_batch), black_box(batch_size))
        .expect("batch")
}

#[library_benchmark]
fn cd_mul_batch_cpu_4096x8() -> Vec<f32> {
    let dim = 4096;
    let batch_size = 8;
    let avt = TensorAVT::new(dim);
    let a = values(dim, 0xCAFE + dim as u64);
    let x_batch = values(dim * batch_size, 0xFACEFEED + batch_size as u64);
    avt.compute_cd_mul_batch(black_box(&a), black_box(&x_batch), black_box(batch_size))
        .expect("batch")
}

#[library_benchmark]
fn norm_sq_batch_cpu_256x64() -> Vec<f32> {
    let dim = 256;
    let batch_size = 64;
    let avt = TensorAVT::new(dim);
    let vectors = values(dim * batch_size, 0xDEADBEEF);
    avt.compute_norm_sq_batch(black_box(&vectors), black_box(batch_size))
        .expect("norms")
}

#[library_benchmark]
fn norm_sq_batch_cpu_4096x8() -> Vec<f32> {
    let dim = 4096;
    let batch_size = 8;
    let avt = TensorAVT::new(dim);
    let vectors = values(dim * batch_size, 0xDEADBEEF + dim as u64);
    avt.compute_norm_sq_batch(black_box(&vectors), black_box(batch_size))
        .expect("norms")
}

library_benchmark_group!(
    name = cpu_tensor_avt;
    benchmarks =
        cd_mul_cpu_256,
        cd_mul_cpu_1024,
        cd_mul_batch_cpu_256x64,
        cd_mul_batch_cpu_4096x8,
        norm_sq_batch_cpu_256x64,
        norm_sq_batch_cpu_4096x8
);

#[cfg(feature = "gpu")]
#[library_benchmark]
fn cd_mul_gpu_host_256() -> usize {
    if !is_gpu_available() {
        return 0;
    }
    let dim = 256;
    let avt = TensorAVT::new(dim);
    let a = values(dim, 0x5151);
    let x = values(dim, 0x6161);
    avt.compute_cd_mul(black_box(&a), black_box(&x))
        .expect("gpu single")
        .len()
}

#[cfg(feature = "gpu")]
#[library_benchmark]
fn cd_mul_batch_gpu_host_256x64() -> usize {
    if !is_gpu_available() {
        return 0;
    }
    let dim = 256;
    let batch_size = 64;
    let avt = TensorAVT::new(dim);
    let a = values(dim, 0x7171);
    let x_batch = values(dim * batch_size, 0x8181);
    avt.compute_cd_mul_batch(black_box(&a), black_box(&x_batch), black_box(batch_size))
        .expect("gpu batch")
        .len()
}

#[cfg(feature = "gpu")]
#[library_benchmark]
fn norm_sq_batch_gpu_host_256x64() -> usize {
    if !is_gpu_available() {
        return 0;
    }
    let dim = 256;
    let batch_size = 64;
    let avt = TensorAVT::new(dim);
    let vectors = values(dim * batch_size, 0x9191);
    avt.compute_norm_sq_batch(black_box(&vectors), black_box(batch_size))
        .expect("gpu norms")
        .len()
}

#[cfg(feature = "gpu")]
library_benchmark_group!(
    name = gpu_tensor_avt_host;
    benchmarks = cd_mul_gpu_host_256, cd_mul_batch_gpu_host_256x64, norm_sq_batch_gpu_host_256x64
);

#[cfg(feature = "gpu")]
main!(
    library_benchmark_groups = cpu_tensor_avt,
    gpu_tensor_avt_host
);

#[cfg(not(feature = "gpu"))]
main!(library_benchmark_groups = cpu_tensor_avt);
