#[cfg(feature = "gpu")]
use gororoba_algebra::gpu::{ComputeBackend, TensorAvtMulSession, TensorAvtNormSession};
use gororoba_algebra::gpu::{TensorAVT, is_gpu_available};
use std::{env, fs, hint::black_box, path::PathBuf, process, time::Instant};
use tracing::info_span;
#[cfg(feature = "tracy-profile")]
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

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
    out: Option<PathBuf>,
}

struct Inputs {
    a: Vec<f32>,
    x: Vec<f32>,
    x_batch: Vec<f32>,
    vectors: Vec<f32>,
}

#[cfg(feature = "gpu")]
enum GpuSession {
    Mul(TensorAvtMulSession),
    Norm(TensorAvtNormSession),
}

fn usage() -> ! {
    eprintln!(
        "usage: cargo run -p gororoba_cli_algebra --bin tensor-avt-profile-suite -- [--path single|batch|norm] [--backend cpu|gpu] [--gpu-mode host|workspace|resident] [--dim N] [--batch-size N] [--iters N] [--warmup N] [--seed N] [--out DIR]"
    );
    process::exit(2);
}

fn init_tracing() {
    #[cfg(feature = "tracy-profile")]
    {
        let tracy_layer = tracing_tracy::TracyLayer::default();
        tracing_subscriber::registry().with(tracy_layer).init();
    }
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
        out: None,
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
            "--out" => config.out = Some(PathBuf::from(value)),
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
    let _span = info_span!("inputs.generate").entered();
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
    let _span = info_span!("tensor_avt.iteration", backend = "cpu").entered();
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
fn make_gpu_session(config: &Config, avt: &TensorAVT) -> Result<GpuSession, String> {
    let _span = info_span!("session.create", backend = "cuda").entered();
    match config.path {
        PathKind::Single => Ok(GpuSession::Mul(
            avt.new_mul_session(ComputeBackend::Cuda, 1)?,
        )),
        PathKind::Batch => Ok(GpuSession::Mul(
            avt.new_mul_session(ComputeBackend::Cuda, config.batch_size)?,
        )),
        PathKind::Norm => Ok(GpuSession::Norm(
            avt.new_norm_session(ComputeBackend::Cuda, config.batch_size)?,
        )),
    }
}

#[cfg(feature = "gpu")]
fn load_session_inputs(
    config: &Config,
    inputs: &Inputs,
    session: &mut GpuSession,
) -> Result<(), String> {
    let _span = info_span!("cuda.upload").entered();
    match (config.path, session) {
        (PathKind::Single, GpuSession::Mul(session)) => {
            session.load_left(&inputs.a)?;
            session.load_right(&inputs.x, 1)
        }
        (PathKind::Batch, GpuSession::Mul(session)) => {
            session.load_left(&inputs.a)?;
            session.load_right(&inputs.x_batch, config.batch_size)
        }
        (PathKind::Norm, GpuSession::Norm(session)) => {
            session.load_vectors(&inputs.vectors, config.batch_size)
        }
        _ => Err("session kind does not match profiling path".into()),
    }
}

#[cfg(feature = "gpu")]
fn run_once_gpu_workspace(
    config: &Config,
    avt: &TensorAVT,
    inputs: &Inputs,
    session: &mut GpuSession,
) -> Result<f32, String> {
    load_session_inputs(config, inputs, session)?;
    let _span = info_span!(
        "tensor_avt.iteration",
        backend = "cuda",
        gpu_mode = "workspace"
    )
    .entered();
    match (config.path, session) {
        (PathKind::Single, GpuSession::Mul(session)) => {
            let _launch = info_span!("cuda.launch", path = "single").entered();
            session.run_single(avt)?;
            drop(_launch);
            let _download = info_span!("cuda.download").entered();
            session
                .download_output(config.dim)
                .map(|out| black_box(out.iter().copied().sum::<f32>()))
        }
        (PathKind::Batch, GpuSession::Mul(session)) => {
            let _launch = info_span!("cuda.launch", path = "batch").entered();
            session.run_batch(avt, config.batch_size)?;
            drop(_launch);
            let _download = info_span!("cuda.download").entered();
            session
                .download_output(config.batch_size * config.dim)
                .map(|out| black_box(out.iter().copied().sum::<f32>()))
        }
        (PathKind::Norm, GpuSession::Norm(session)) => {
            let _launch = info_span!("cuda.launch", path = "norm").entered();
            session.run_norms(avt, config.batch_size)?;
            drop(_launch);
            let _download = info_span!("cuda.download").entered();
            session
                .download_norms(config.batch_size)
                .map(|out| black_box(out.iter().copied().sum::<f32>()))
        }
        _ => Err("session kind does not match profiling path".into()),
    }
}

#[cfg(feature = "gpu")]
fn run_once_gpu_resident(
    config: &Config,
    avt: &TensorAVT,
    session: &mut GpuSession,
) -> Result<(), String> {
    let _span = info_span!(
        "tensor_avt.iteration",
        backend = "cuda",
        gpu_mode = "resident"
    )
    .entered();
    let _launch = info_span!("cuda.launch", path = path_name(config.path)).entered();
    match (config.path, session) {
        (PathKind::Single, GpuSession::Mul(session)) => session.run_single(avt),
        (PathKind::Batch, GpuSession::Mul(session)) => session.run_batch(avt, config.batch_size),
        (PathKind::Norm, GpuSession::Norm(session)) => session.run_norms(avt, config.batch_size),
        _ => Err("session kind does not match profiling path".into()),
    }
}

#[cfg(feature = "gpu")]
fn resident_checksum(config: &Config, session: &mut GpuSession) -> Result<f32, String> {
    let _span = info_span!("cuda.download").entered();
    match (config.path, session) {
        (PathKind::Single, GpuSession::Mul(session)) => session
            .download_output(config.dim)
            .map(|out| black_box(out.iter().copied().sum::<f32>())),
        (PathKind::Batch, GpuSession::Mul(session)) => session
            .download_output(config.batch_size * config.dim)
            .map(|out| black_box(out.iter().copied().sum::<f32>())),
        (PathKind::Norm, GpuSession::Norm(session)) => session
            .download_norms(config.batch_size)
            .map(|out| black_box(out.iter().copied().sum::<f32>())),
        _ => Err("session kind does not match profiling path".into()),
    }
}

#[cfg(feature = "gpu")]
fn run_gpu_profile(
    config: &Config,
    avt: &TensorAVT,
    inputs: &Inputs,
) -> Result<(std::time::Duration, f32), String> {
    let _total = info_span!("tensor_avt.total", backend = "cuda").entered();
    match config.gpu_mode {
        GpuMode::Host => {
            let mut checksum = 0.0f32;
            for _ in 0..config.warmup {
                checksum = black_box(match config.path {
                    PathKind::Single => {
                        let _span =
                            info_span!("tensor_avt.iteration", backend = "cuda", gpu_mode = "host")
                                .entered();
                        avt.compute_cd_mul(black_box(&inputs.a), black_box(&inputs.x))
                            .map(|out| black_box(out.iter().copied().sum::<f32>()))?
                    }
                    PathKind::Batch => {
                        let _span =
                            info_span!("tensor_avt.iteration", backend = "cuda", gpu_mode = "host")
                                .entered();
                        avt.compute_cd_mul_batch(
                            black_box(&inputs.a),
                            black_box(&inputs.x_batch),
                            black_box(config.batch_size),
                        )
                        .map(|out| black_box(out.iter().copied().sum::<f32>()))?
                    }
                    PathKind::Norm => {
                        let _span =
                            info_span!("tensor_avt.iteration", backend = "cuda", gpu_mode = "host")
                                .entered();
                        avt.compute_norm_sq_batch(
                            black_box(&inputs.vectors),
                            black_box(config.batch_size),
                        )
                        .map(|out| black_box(out.iter().copied().sum::<f32>()))?
                    }
                });
            }
            let start = Instant::now();
            for _ in 0..config.iters {
                checksum = black_box(match config.path {
                    PathKind::Single => {
                        let _span =
                            info_span!("tensor_avt.iteration", backend = "cuda", gpu_mode = "host")
                                .entered();
                        avt.compute_cd_mul(black_box(&inputs.a), black_box(&inputs.x))
                            .map(|out| black_box(out.iter().copied().sum::<f32>()))?
                    }
                    PathKind::Batch => {
                        let _span =
                            info_span!("tensor_avt.iteration", backend = "cuda", gpu_mode = "host")
                                .entered();
                        avt.compute_cd_mul_batch(
                            black_box(&inputs.a),
                            black_box(&inputs.x_batch),
                            black_box(config.batch_size),
                        )
                        .map(|out| black_box(out.iter().copied().sum::<f32>()))?
                    }
                    PathKind::Norm => {
                        let _span =
                            info_span!("tensor_avt.iteration", backend = "cuda", gpu_mode = "host")
                                .entered();
                        avt.compute_norm_sq_batch(
                            black_box(&inputs.vectors),
                            black_box(config.batch_size),
                        )
                        .map(|out| black_box(out.iter().copied().sum::<f32>()))?
                    }
                });
            }
            Ok((start.elapsed(), checksum))
        }
        GpuMode::Workspace => {
            let mut session = make_gpu_session(config, avt)?;
            let mut checksum = 0.0f32;
            for _ in 0..config.warmup {
                checksum = black_box(run_once_gpu_workspace(config, avt, inputs, &mut session)?);
            }
            let start = Instant::now();
            for _ in 0..config.iters {
                checksum = black_box(run_once_gpu_workspace(config, avt, inputs, &mut session)?);
            }
            Ok((start.elapsed(), checksum))
        }
        GpuMode::Resident => {
            let mut session = make_gpu_session(config, avt)?;
            load_session_inputs(config, inputs, &mut session)?;
            for _ in 0..config.warmup {
                run_once_gpu_resident(config, avt, &mut session)?;
            }
            let _ = resident_checksum(config, &mut session)?;
            let start = Instant::now();
            for _ in 0..config.iters {
                run_once_gpu_resident(config, avt, &mut session)?;
            }
            let checksum = resident_checksum(config, &mut session)?;
            Ok((start.elapsed(), checksum))
        }
    }
}

fn emit_results(config: &Config, elapsed: std::time::Duration, checksum: f32) -> String {
    let mut out = String::new();
    out.push_str(&format!("backend = \"{}\"\n", backend_name(config.backend)));
    out.push_str(&format!("compiled_gpu = {}\n", cfg!(feature = "gpu")));
    out.push_str(&format!("gpu_visible = {}\n", is_gpu_available()));
    out.push_str(&format!(
        "gpu_mode = \"{}\"\n",
        gpu_mode_name(config.gpu_mode)
    ));
    out.push_str(&format!("path = \"{}\"\n", path_name(config.path)));
    out.push_str(&format!("dim = {}\n", config.dim));
    out.push_str(&format!("batch_size = {}\n", config.batch_size));
    out.push_str(&format!("iters = {}\n", config.iters));
    out.push_str(&format!("warmup = {}\n", config.warmup));
    out.push_str("inputs_precomputed = true\n");
    out.push_str(&format!("seconds = {:.9}\n", elapsed.as_secs_f64()));
    out.push_str(&format!(
        "ns_per_iter = {:.3}\n",
        elapsed.as_nanos() as f64 / config.iters as f64
    ));
    out.push_str(&format!("checksum = {:.6}\n", checksum));
    out
}

fn maybe_write_output(config: &Config, rendered: &str) -> Result<(), String> {
    if let Some(dir) = &config.out {
        fs::create_dir_all(dir).map_err(|e| format!("create output dir: {e}"))?;
        fs::write(dir.join("summary.toml"), rendered).map_err(|e| format!("write summary: {e}"))?;
    }
    Ok(())
}

fn main() {
    init_tracing();
    let config = parse_args();
    let compiled_gpu = cfg!(feature = "gpu");
    match config.backend {
        BackendKind::Gpu if !compiled_gpu => {
            eprintln!("gpu backend requires building gororoba_cli_algebra with --features gpu");
            process::exit(2);
        }
        BackendKind::Gpu if !is_gpu_available() => {
            eprintln!("gpu backend requested but CUDA device is not visible");
            process::exit(2);
        }
        _ => {}
    }

    let _total = info_span!("tensor_avt.total").entered();
    let avt = {
        let _span = info_span!("session.create", backend = "shared").entered();
        TensorAVT::new(config.dim)
    };
    let inputs = make_inputs(&config);

    let rendered = match config.backend {
        BackendKind::Cpu => {
            let mut checksum = 0.0f32;
            for _ in 0..config.warmup {
                checksum = black_box(run_once_cpu(&config, &avt, &inputs));
            }
            let start = Instant::now();
            for _ in 0..config.iters {
                checksum = black_box(run_once_cpu(&config, &avt, &inputs));
            }
            emit_results(&config, start.elapsed(), checksum)
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
    };

    if let Err(err) = maybe_write_output(&config, &rendered) {
        eprintln!("{err}");
        process::exit(1);
    }
    print!("{rendered}");
}
