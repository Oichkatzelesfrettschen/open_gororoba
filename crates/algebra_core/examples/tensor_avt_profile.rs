#[cfg(feature = "gpu")]
use algebra_core::gpu::tensor_avt::{TensorAvtMulGpuWorkspace, TensorAvtNormGpuWorkspace};
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

#[derive(Clone, Copy)]
enum GpuMode {
    Host,
    Workspace,
    Resident,
}

struct Config {
    path: PathKind,
    backend: BackendKind,
    gpu_mode: GpuMode,
    dim: usize,
    batch_size: usize,
    iters: usize,
    warmup: usize,
    seed: u64,
}

struct Inputs {
    a: Vec<f32>,
    x: Vec<f32>,
    x_batch: Vec<f32>,
    vectors: Vec<f32>,
}

fn usage() -> ! {
    eprintln!(
        "usage: cargo run --example tensor_avt_profile -- [--path single|batch|norm] [--backend cpu|gpu] [--gpu-mode host|workspace|resident] [--dim N] [--batch-size N] [--iters N] [--warmup N] [--seed N]"
    );
    process::exit(2);
}

fn parse_args() -> Config {
    let mut config = Config {
        path: PathKind::Single,
        backend: BackendKind::Cpu,
        gpu_mode: GpuMode::Host,
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
            "--gpu-mode" => {
                config.gpu_mode = match value.as_str() {
                    "host" => GpuMode::Host,
                    "workspace" => GpuMode::Workspace,
                    "resident" => GpuMode::Resident,
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

fn make_inputs(config: &Config) -> Inputs {
    Inputs {
        a: values(config.dim, config.seed ^ 0xA11CE),
        x: values(config.dim, config.seed ^ 0xBADC0DE),
        x_batch: values(config.dim * config.batch_size, config.seed ^ 0xFACEFEED),
        vectors: values(config.dim * config.batch_size, config.seed ^ 0xDEADBEEF),
    }
}

fn path_name(path: PathKind) -> &'static str {
    match path {
        PathKind::Single => "single",
        PathKind::Batch => "batch",
        PathKind::Norm => "norm",
    }
}

fn backend_name(backend: BackendKind) -> &'static str {
    match backend {
        BackendKind::Cpu => "cpu",
        BackendKind::Gpu => "gpu",
    }
}

fn gpu_mode_name(mode: GpuMode) -> &'static str {
    match mode {
        GpuMode::Host => "host",
        GpuMode::Workspace => "workspace",
        GpuMode::Resident => "resident",
    }
}

fn run_once_cpu(config: &Config, avt: &TensorAVT, inputs: &Inputs) -> f32 {
    match config.path {
        PathKind::Single => {
            let out = avt
                .compute_cd_mul(black_box(&inputs.a), black_box(&inputs.x))
                .unwrap();
            black_box(out.iter().copied().sum::<f32>())
        }
        PathKind::Batch => {
            let out = avt
                .compute_cd_mul_batch(
                    black_box(&inputs.a),
                    black_box(&inputs.x_batch),
                    black_box(config.batch_size),
                )
                .unwrap();
            black_box(out.iter().copied().sum::<f32>())
        }
        PathKind::Norm => {
            let out = avt
                .compute_norm_sq_batch(black_box(&inputs.vectors), black_box(config.batch_size))
                .unwrap();
            black_box(out.iter().copied().sum::<f32>())
        }
    }
}

#[cfg(feature = "gpu")]
enum GpuWorkspace {
    Mul(TensorAvtMulGpuWorkspace),
    Norm(TensorAvtNormGpuWorkspace),
}

#[cfg(feature = "gpu")]
fn make_gpu_workspace(config: &Config, avt: &TensorAVT) -> Result<GpuWorkspace, String> {
    match config.path {
        PathKind::Single => Ok(GpuWorkspace::Mul(avt.new_gpu_mul_workspace(1)?)),
        PathKind::Batch => Ok(GpuWorkspace::Mul(
            avt.new_gpu_mul_workspace(config.batch_size)?,
        )),
        PathKind::Norm => Ok(GpuWorkspace::Norm(
            avt.new_gpu_norm_workspace(config.batch_size)?,
        )),
    }
}

#[cfg(feature = "gpu")]
fn upload_resident_inputs(
    config: &Config,
    inputs: &Inputs,
    workspace: &mut GpuWorkspace,
) -> Result<(), String> {
    match (config.path, workspace) {
        (PathKind::Single, GpuWorkspace::Mul(workspace)) => {
            workspace.upload_a(&inputs.a)?;
            workspace.upload_x(&inputs.x, 1, config.dim)
        }
        (PathKind::Batch, GpuWorkspace::Mul(workspace)) => {
            workspace.upload_a(&inputs.a)?;
            workspace.upload_x(&inputs.x_batch, config.batch_size, config.dim)
        }
        (PathKind::Norm, GpuWorkspace::Norm(workspace)) => {
            workspace.upload_vectors(&inputs.vectors, config.batch_size, config.dim)
        }
        _ => Err("workspace kind does not match profiling path".into()),
    }
}

#[cfg(feature = "gpu")]
fn run_once_gpu_workspace(
    config: &Config,
    avt: &TensorAVT,
    inputs: &Inputs,
    workspace: &mut GpuWorkspace,
) -> Result<f32, String> {
    match (config.path, workspace) {
        (PathKind::Single, GpuWorkspace::Mul(workspace)) => avt
            .compute_cd_mul_with_workspace(&inputs.a, &inputs.x, workspace)
            .map(|out| black_box(out.iter().copied().sum::<f32>())),
        (PathKind::Batch, GpuWorkspace::Mul(workspace)) => avt
            .compute_cd_mul_batch_with_workspace(
                &inputs.a,
                &inputs.x_batch,
                config.batch_size,
                workspace,
            )
            .map(|out| black_box(out.iter().copied().sum::<f32>())),
        (PathKind::Norm, GpuWorkspace::Norm(workspace)) => avt
            .compute_norm_sq_batch_with_workspace(&inputs.vectors, config.batch_size, workspace)
            .map(|out| black_box(out.iter().copied().sum::<f32>())),
        _ => Err("workspace kind does not match profiling path".into()),
    }
}

#[cfg(feature = "gpu")]
fn run_once_gpu_resident(
    config: &Config,
    avt: &TensorAVT,
    workspace: &mut GpuWorkspace,
) -> Result<(), String> {
    match (config.path, workspace) {
        (PathKind::Single, GpuWorkspace::Mul(workspace)) => {
            avt.launch_cd_mul_with_workspace(workspace)?;
            workspace
                .stream()
                .synchronize()
                .map_err(|e| format!("Synchronize single stream: {e}"))
        }
        (PathKind::Batch, GpuWorkspace::Mul(workspace)) => {
            avt.launch_cd_mul_batch_with_workspace(config.batch_size, workspace)?;
            workspace
                .stream()
                .synchronize()
                .map_err(|e| format!("Synchronize batch stream: {e}"))
        }
        (PathKind::Norm, GpuWorkspace::Norm(workspace)) => {
            avt.launch_norm_sq_batch_with_workspace(config.batch_size, workspace)?;
            workspace
                .stream()
                .synchronize()
                .map_err(|e| format!("Synchronize norm stream: {e}"))
        }
        _ => Err("workspace kind does not match profiling path".into()),
    }
}

#[cfg(feature = "gpu")]
fn resident_checksum(config: &Config, workspace: &mut GpuWorkspace) -> Result<f32, String> {
    match (config.path, workspace) {
        (PathKind::Single, GpuWorkspace::Mul(workspace)) => workspace
            .download_y(config.dim)
            .map(|out| black_box(out.iter().copied().sum::<f32>())),
        (PathKind::Batch, GpuWorkspace::Mul(workspace)) => workspace
            .download_y(config.batch_size * config.dim)
            .map(|out| black_box(out.iter().copied().sum::<f32>())),
        (PathKind::Norm, GpuWorkspace::Norm(workspace)) => workspace
            .download_norms(config.batch_size)
            .map(|out| black_box(out.iter().copied().sum::<f32>())),
        _ => Err("workspace kind does not match profiling path".into()),
    }
}

#[cfg(feature = "gpu")]
fn run_gpu_profile(
    config: &Config,
    avt: &TensorAVT,
    inputs: &Inputs,
) -> Result<(std::time::Duration, f32), String> {
    match config.gpu_mode {
        GpuMode::Host => match config.path {
            PathKind::Single | PathKind::Batch | PathKind::Norm => {
                let mut checksum = 0.0f32;
                for _ in 0..config.warmup {
                    checksum = black_box(match config.path {
                        PathKind::Single => avt
                            .compute_cd_mul(black_box(&inputs.a), black_box(&inputs.x))
                            .map(|out| black_box(out.iter().copied().sum::<f32>()))?,
                        PathKind::Batch => avt
                            .compute_cd_mul_batch(
                                black_box(&inputs.a),
                                black_box(&inputs.x_batch),
                                black_box(config.batch_size),
                            )
                            .map(|out| black_box(out.iter().copied().sum::<f32>()))?,
                        PathKind::Norm => avt
                            .compute_norm_sq_batch(
                                black_box(&inputs.vectors),
                                black_box(config.batch_size),
                            )
                            .map(|out| black_box(out.iter().copied().sum::<f32>()))?,
                    });
                }
                let start = Instant::now();
                for _ in 0..config.iters {
                    checksum = black_box(match config.path {
                        PathKind::Single => avt
                            .compute_cd_mul(black_box(&inputs.a), black_box(&inputs.x))
                            .map(|out| black_box(out.iter().copied().sum::<f32>()))?,
                        PathKind::Batch => avt
                            .compute_cd_mul_batch(
                                black_box(&inputs.a),
                                black_box(&inputs.x_batch),
                                black_box(config.batch_size),
                            )
                            .map(|out| black_box(out.iter().copied().sum::<f32>()))?,
                        PathKind::Norm => avt
                            .compute_norm_sq_batch(
                                black_box(&inputs.vectors),
                                black_box(config.batch_size),
                            )
                            .map(|out| black_box(out.iter().copied().sum::<f32>()))?,
                    });
                }
                Ok((start.elapsed(), checksum))
            }
        },
        GpuMode::Workspace => {
            let mut workspace = make_gpu_workspace(config, avt)?;
            let mut checksum = 0.0f32;
            for _ in 0..config.warmup {
                checksum = black_box(run_once_gpu_workspace(config, avt, inputs, &mut workspace)?);
            }
            let start = Instant::now();
            for _ in 0..config.iters {
                checksum = black_box(run_once_gpu_workspace(config, avt, inputs, &mut workspace)?);
            }
            Ok((start.elapsed(), checksum))
        }
        GpuMode::Resident => {
            let mut workspace = make_gpu_workspace(config, avt)?;
            upload_resident_inputs(config, inputs, &mut workspace)?;
            for _ in 0..config.warmup {
                run_once_gpu_resident(config, avt, &mut workspace)?;
            }
            let start = Instant::now();
            for _ in 0..config.iters {
                run_once_gpu_resident(config, avt, &mut workspace)?;
            }
            let checksum = resident_checksum(config, &mut workspace)?;
            Ok((start.elapsed(), checksum))
        }
    }
}

fn emit_results(config: &Config, elapsed: std::time::Duration, checksum: f32) {
    println!("backend={}", backend_name(config.backend));
    println!("compiled_gpu={}", cfg!(feature = "gpu"));
    println!("gpu_visible={}", is_gpu_available());
    println!("gpu_mode={}", gpu_mode_name(config.gpu_mode));
    println!("path={}", path_name(config.path));
    println!("dim={}", config.dim);
    println!("batch_size={}", config.batch_size);
    println!("iters={}", config.iters);
    println!("warmup={}", config.warmup);
    println!("inputs_precomputed=true");
    println!("seconds={:.9}", elapsed.as_secs_f64());
    println!(
        "ns_per_iter={:.3}",
        elapsed.as_nanos() as f64 / config.iters as f64
    );
    println!("checksum={checksum:.6}");
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
    let inputs = make_inputs(&config);

    match config.backend {
        BackendKind::Cpu => {
            let mut checksum = 0.0f32;
            for _ in 0..config.warmup {
                checksum = black_box(run_once_cpu(&config, &avt, &inputs));
            }
            let start = Instant::now();
            for _ in 0..config.iters {
                checksum = black_box(run_once_cpu(&config, &avt, &inputs));
            }
            let elapsed = start.elapsed();
            emit_results(&config, elapsed, checksum);
        }
        BackendKind::Gpu => {
            #[cfg(feature = "gpu")]
            {
                match run_gpu_profile(&config, &avt, &inputs) {
                    Ok((elapsed, checksum)) => emit_results(&config, elapsed, checksum),
                    Err(err) => {
                        eprintln!("{err}");
                        process::exit(1);
                    }
                }
            }
            #[cfg(not(feature = "gpu"))]
            {
                let _ = (&avt, &inputs, compiled_gpu);
                unreachable!("validated gpu feature above");
            }
        }
    }
}
