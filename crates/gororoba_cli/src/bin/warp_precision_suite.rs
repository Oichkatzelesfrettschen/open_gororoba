mod warp_precision_suite_ops;
mod warp_runner;

use std::error::Error;
use std::path::PathBuf;
use warp_precision_suite_ops::{
    PrecisionSuiteConfig, default_timing_mode_for_backend, parse_backend, parse_bool01, parse_csv,
    parse_precision, parse_timing_mode, run_precision_suite,
};
use warp_runner::{BackendKind, TimingMode};

fn usage() -> &'static str {
    "Usage:
  warp-precision-suite bench <res> <FP32|BF16> [duration_s=20] [trace_stride=10] [h5_out] [backend=gpu|cpu] [timing_mode=auto|launch_only|stream_sync_each_step|cuda_events] [gate=1] [write_summary=1]
  warp-precision-suite matrix [duration_s=20] [trace_stride=10] [sizes_csv=8,16,32,64] [precisions_csv=FP32,BF16] [out_dir=data/h5/precision_bench] [backend=gpu|cpu] [timing_mode=auto|launch_only|stream_sync_each_step|cuda_events] [detailed_names=0] [gate=1] [write_summary=1]"
}

fn parse_backend_with_default(input: Option<&String>) -> Result<BackendKind, Box<dyn Error>> {
    match input {
        Some(s) => parse_backend(s)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "backend must be one of: gpu, cuda, cpu",
                )
            })
            .map_err(Into::into),
        None => Ok(BackendKind::Gpu),
    }
}

fn parse_timing_mode_with_default(
    input: Option<&String>,
    backend: BackendKind,
) -> Result<TimingMode, Box<dyn Error>> {
    match input {
        Some(raw) if raw.eq_ignore_ascii_case("auto") => {
            Ok(default_timing_mode_for_backend(backend))
        }
        Some(raw) => parse_timing_mode(raw).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "timing_mode must be one of: auto, launch_only, stream_sync_each_step, cuda_events",
            )
            .into()
        }),
        None => Ok(default_timing_mode_for_backend(backend)),
    }
}

fn parse_bool_with_default(
    input: Option<&String>,
    field: &str,
    default: bool,
) -> Result<bool, Box<dyn Error>> {
    match input {
        Some(s) => parse_bool01(s)
            .ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("{} must be one of: 0,1,true,false,yes,no", field),
                )
            })
            .map_err(Into::into),
        None => Ok(default),
    }
}

fn run_bench_mode(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() < 4 {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, usage()).into());
    }
    let resolution: usize = args[2].parse()?;
    let precision = parse_precision(&args[3]).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "precision must be FP32 or BF16",
        )
    })?;
    let duration_secs = args.get(4).map_or(Ok(20.0), |s| s.parse())?;
    let trace_stride = args.get(5).map_or(Ok(10usize), |s| s.parse())?;
    let h5_output = args.get(6).map(PathBuf::from);
    let backend = parse_backend_with_default(args.get(7))?;
    let timing_mode = parse_timing_mode_with_default(args.get(8), backend)?;
    let gate = parse_bool_with_default(args.get(9), "gate", true)?;
    let write_summary = parse_bool_with_default(args.get(10), "write_summary", true)?;

    run_precision_suite(PrecisionSuiteConfig {
        backend,
        timing_mode,
        duration_secs,
        trace_stride,
        sizes: vec![resolution],
        precisions: vec![precision],
        out_dir: None,
        single_output: h5_output,
        detailed_names: false,
        gate,
        write_summary,
    })
}

fn run_matrix_mode(args: &[String]) -> Result<(), Box<dyn Error>> {
    let duration_secs: f64 = args.get(2).map_or(Ok(20.0), |s| s.parse())?;
    let trace_stride: usize = args.get(3).map_or(Ok(10usize), |s| s.parse())?;
    let sizes_csv = args.get(4).map(String::as_str).unwrap_or("8,16,32,64");
    let precisions_csv = args.get(5).map(String::as_str).unwrap_or("FP32,BF16");
    let out_dir = args
        .get(6)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/h5/precision_bench"));
    let backend = parse_backend_with_default(args.get(7))?;
    let timing_mode = parse_timing_mode_with_default(args.get(8), backend)?;
    let detailed_names = parse_bool_with_default(args.get(9), "detailed_names", false)?;
    let gate = parse_bool_with_default(args.get(10), "gate", true)?;
    let write_summary = parse_bool_with_default(args.get(11), "write_summary", true)?;

    let sizes = parse_csv(sizes_csv, |s| s.parse::<usize>().ok(), "size")?;
    let precisions = parse_csv(precisions_csv, parse_precision, "precision")?;
    std::fs::create_dir_all(&out_dir)?;
    println!("=== Warp Precision Suite / Matrix Mode ===");
    println!(
        "duration_s={}, trace_stride={}, sizes={}, precisions={}, backend={:?}, timing_mode={:?}, detailed_names={}, out_dir={}",
        duration_secs,
        trace_stride,
        sizes_csv,
        precisions_csv,
        backend,
        timing_mode,
        detailed_names,
        out_dir.display()
    );
    if backend == BackendKind::Cpu && precisions.len() > 1 {
        eprintln!(
            "[WARN] CPU backend currently executes FP64 path; precision tokens are kept for matrix parity."
        );
    }

    run_precision_suite(PrecisionSuiteConfig {
        backend,
        timing_mode,
        duration_secs,
        trace_stride,
        sizes,
        precisions,
        out_dir: Some(out_dir),
        single_output: None,
        detailed_names,
        gate,
        write_summary,
    })?;
    println!("DONE: matrix complete");
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();
    let mode = args
        .get(1)
        .map(|s| s.to_ascii_lowercase())
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, usage()))?;

    match mode.as_str() {
        "bench" => run_bench_mode(&args),
        "matrix" => run_matrix_mode(&args),
        _ => Err(std::io::Error::new(std::io::ErrorKind::InvalidInput, usage()).into()),
    }
}
