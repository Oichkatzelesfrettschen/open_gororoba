mod warp_runner;

use lbm_3d_cuda::Precision;
use std::error::Error;
use std::path::PathBuf;
use warp_runner::{
    gate_h5_outputs, print_case_report, run_case, write_step_timing_report, BackendKind, BenchCase,
    TimingMode,
};

fn parse_precision(input: &str) -> Option<Precision> {
    match input.to_ascii_uppercase().as_str() {
        "FP32" => Some(Precision::FP32),
        "BF16" => Some(Precision::BF16),
        _ => None,
    }
}

fn parse_csv<T>(
    input: &str,
    parse_one: impl Fn(&str) -> Option<T>,
    what: &str,
) -> Result<Vec<T>, Box<dyn Error>> {
    let mut out = Vec::new();
    for token in input.split(',') {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value = parse_one(trimmed).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("invalid {} token: {}", what, trimmed),
            )
        })?;
        out.push(value);
    }
    if out.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{} list cannot be empty", what),
        )
        .into());
    }
    Ok(out)
}

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();
    let duration_secs: f64 = args.get(1).map_or(Ok(20.0), |s| s.parse())?;
    let trace_stride: usize = args.get(2).map_or(Ok(10usize), |s| s.parse())?;
    let sizes_csv = args.get(3).map(String::as_str).unwrap_or("8,16,32,64");
    let precisions_csv = args.get(4).map(String::as_str).unwrap_or("FP32,BF16");
    let out_dir = args
        .get(5)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/h5/precision_bench"));

    if duration_secs <= 0.0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "duration_s must be > 0",
        )
        .into());
    }
    if trace_stride == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "trace_stride must be >= 1",
        )
        .into());
    }

    let sizes = parse_csv(
        sizes_csv,
        |s| s.parse::<usize>().ok(),
        "size",
    )?;
    let precisions = parse_csv(precisions_csv, parse_precision, "precision")?;

    std::fs::create_dir_all(&out_dir)?;
    println!("=== Warp Precision Matrix ===");
    println!(
        "duration_s={}, trace_stride={}, sizes={}, precisions={}, out_dir={}",
        duration_secs,
        trace_stride,
        sizes_csv,
        precisions_csv,
        out_dir.display()
    );

    let mut artifacts = Vec::new();
    for &size in &sizes {
        for &precision in &precisions {
            let out_path = out_dir.join(format!("warp_ring_{}_{}.h5", size, match precision {
                Precision::FP32 => "FP32",
                Precision::BF16 => "BF16",
            }));
            println!();
            println!(
                "[RUN] size={}, precision={:?}, out={}",
                size,
                precision,
                out_path.display()
            );
            let report = run_case(&BenchCase {
                resolution: size,
                precision,
                backend: BackendKind::Gpu,
                timing_mode: TimingMode::CudaEvents,
                duration_secs,
                trace_stride,
                h5_output: Some(out_path.clone()),
            })?;
            print_case_report(&report);
            let precision_tag = match precision {
                Precision::FP32 => "FP32",
                Precision::BF16 => "BF16",
            };
            let timing_out = out_dir.join(format!(
                "timing_{}_{}_{}s.toml",
                size,
                precision_tag,
                duration_secs.round() as u64
            ));
            write_step_timing_report(&timing_out, &report)?;
            println!("BENCH_COMPLETE");
            artifacts.push(out_path);
        }
    }

    println!();
    println!("[GATE] Post-run finite check on /simulation/trace/rho_mean");
    gate_h5_outputs(&artifacts)?;
    println!("DONE: matrix complete and finite gate passed");
    Ok(())
}
