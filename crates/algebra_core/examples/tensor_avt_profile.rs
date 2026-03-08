use algebra_core::gpu::{TensorAVT, is_gpu_available};
use std::{env, hint::black_box, process, time::Instant};

#[derive(Clone, Copy)]
enum PathKind {
    Single,
    Batch,
    Norm,
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum BackendKind {
    Cpu,
    Gpu,
}

struct Config {
    path: PathKind,
    backend: BackendKind,
    dim: usize,
    batch_size: usize,
    iters: usize,
    warmup: usize,
    seed: u64,
}

fn usage() -> ! {
    eprintln!(
        "usage: cargo run --example tensor_avt_profile -- [--path single|batch|norm] [--backend cpu|gpu] [--dim N] [--batch-size N] [--iters N] [--warmup N] [--seed N]"
    );
    process::exit(2);
}

fn parse_args() -> Config {
    let mut config = Config {
        path: PathKind::Single,
        backend: BackendKind::Cpu,
        dim: 256,
        batch_size: 16,
        iters: 64,
        warmup: 8,
        seed: 7,
    };
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        let value = args.next().unwrap_or_else(|| usage());
        match arg.as_str() {
            "--path" => {
                config.path = match value.as_str() {
                    "single" => PathKind::Single,
                    "batch" => PathKind::Batch,
                    "norm" => PathKind::Norm,
                    _ => usage(),
                };
            }
            "--backend" => {
                config.backend = match value.as_str() {
                    "cpu" => BackendKind::Cpu,
                    "gpu" => BackendKind::Gpu,
                    _ => usage(),
                };
            }
            "--dim" => config.dim = value.parse().unwrap_or_else(|_| usage()),
            "--batch-size" => config.batch_size = value.parse().unwrap_or_else(|_| usage()),
            "--iters" => config.iters = value.parse().unwrap_or_else(|_| usage()),
            "--warmup" => config.warmup = value.parse().unwrap_or_else(|_| usage()),
            "--seed" => config.seed = value.parse().unwrap_or_else(|_| usage()),
            _ => usage(),
        }
    }
    if config.dim < 16 || !config.dim.is_power_of_two() {
        eprintln!("--dim must be a power of two >= 16");
        process::exit(2);
    }
    if config.iters == 0 || config.batch_size == 0 {
        eprintln!("--iters and --batch-size must be > 0");
        process::exit(2);
    }
    config
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

fn run_once(config: &Config, avt: &TensorAVT) -> f32 {
    match config.path {
        PathKind::Single => {
            let a = values(config.dim, config.seed ^ 0xA11CE);
            let x = values(config.dim, config.seed ^ 0xBADC0DE);
            let out = avt.compute_cd_mul(black_box(&a), black_box(&x)).unwrap();
            black_box(out.iter().copied().sum())
        }
        PathKind::Batch => {
            let a = values(config.dim, config.seed ^ 0xCAFE);
            let x_batch = values(config.dim * config.batch_size, config.seed ^ 0xFACEFEED);
            let out = avt
                .compute_cd_mul_batch(
                    black_box(&a),
                    black_box(&x_batch),
                    black_box(config.batch_size),
                )
                .unwrap();
            black_box(out.iter().copied().sum())
        }
        PathKind::Norm => {
            let vectors = values(config.dim * config.batch_size, config.seed ^ 0xDEADBEEF);
            let out = avt
                .compute_norm_sq_batch(black_box(&vectors), black_box(config.batch_size))
                .unwrap();
            black_box(out.iter().copied().sum())
        }
    }
}

fn main() {
    let config = parse_args();
    let compiled_gpu = cfg!(feature = "gpu");
    match config.backend {
        BackendKind::Cpu if compiled_gpu => {
            eprintln!("cpu backend requires building the example without --features gpu");
            process::exit(2);
        }
        BackendKind::Gpu if !compiled_gpu => {
            eprintln!("gpu backend requires building the example with --features gpu");
            process::exit(2);
        }
        BackendKind::Gpu if !is_gpu_available() => {
            eprintln!("gpu backend requested but CUDA device is not visible");
            process::exit(2);
        }
        _ => {}
    }

    let avt = TensorAVT::new(config.dim);
    let mut checksum = 0.0f32;
    for _ in 0..config.warmup {
        checksum += run_once(&config, &avt);
    }

    let start = Instant::now();
    for _ in 0..config.iters {
        checksum += run_once(&config, &avt);
    }
    let elapsed = start.elapsed();
    let backend = match config.backend {
        BackendKind::Cpu => "cpu",
        BackendKind::Gpu => "gpu",
    };
    let path = match config.path {
        PathKind::Single => "single",
        PathKind::Batch => "batch",
        PathKind::Norm => "norm",
    };

    println!("backend={backend}");
    println!("compiled_gpu={compiled_gpu}");
    println!("gpu_visible={}", is_gpu_available());
    println!("path={path}");
    println!("dim={}", config.dim);
    println!("batch_size={}", config.batch_size);
    println!("iters={}", config.iters);
    println!("warmup={}", config.warmup);
    println!("seconds={:.9}", elapsed.as_secs_f64());
    println!(
        "ns_per_iter={:.3}",
        elapsed.as_nanos() as f64 / config.iters as f64
    );
    println!("checksum={checksum:.6}");
}
