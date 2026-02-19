mod warp_runner;
mod warp_telemetry;

use lbm_3d_cuda::Precision;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Duration;
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

fn parse_timing_mode(input: &str) -> Option<TimingMode> {
    match input.to_ascii_lowercase().as_str() {
        "launch_only" | "launch" => Some(TimingMode::LaunchOnly),
        "stream_sync_each_step" | "sync_each_step" | "sync" => Some(TimingMode::StreamSyncEachStep),
        "cuda_events" | "events" => Some(TimingMode::CudaEvents),
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

fn precision_tag(precision: Precision) -> &'static str {
    match precision {
        Precision::FP32 => "FP32",
        Precision::BF16 => "BF16",
        Precision::FP64 => "FP64",
    }
}

fn timing_mode_tag(mode: TimingMode) -> &'static str {
    match mode {
        TimingMode::LaunchOnly => "launch_only",
        TimingMode::StreamSyncEachStep => "stream_sync_each_step",
        TimingMode::CudaEvents => "cuda_events",
    }
}

fn backend_tag(backend: BackendKind) -> &'static str {
    match backend {
        BackendKind::Cpu => "CPU",
        BackendKind::Gpu => "GPU",
    }
}

fn write_basemark_summary(
    out_path: &Path,
    timestamp: &str,
    out_dir: &Path,
    duration_secs: f64,
    trace_stride: usize,
    timing_mode: TimingMode,
    cases: &[(PathBuf, PathBuf, warp_runner::BenchCaseReport)],
) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = String::new();
    out.push_str("# Warp basemark summary (Rust-native, TOML-first)\n");
    out.push_str("schema = \"open_gororoba.warp_basemark.v1\"\n");
    out.push_str(&format!("timestamp = \"{}\"\n", timestamp));
    out.push_str(&format!("out_dir = \"{}\"\n", out_dir.display()));
    out.push_str(&format!("duration_secs = {:.6}\n", duration_secs));
    out.push_str(&format!("trace_stride = {}\n", trace_stride));
    out.push_str(&format!(
        "timing_mode = \"{}\"\n",
        timing_mode_tag(timing_mode)
    ));
    out.push('\n');

    for (h5_path, timing_path, report) in cases {
        out.push_str("[[case]]\n");
        out.push_str(&format!("resolution = {}\n", report.resolution));
        out.push_str(&format!("backend = \"{}\"\n", backend_tag(report.backend)));
        out.push_str(&format!(
            "precision = \"{}\"\n",
            match report.backend {
                BackendKind::Cpu => "FP64",
                BackendKind::Gpu => precision_tag(report.precision),
            }
        ));
        out.push_str(&format!(
            "timing_mode = \"{}\"\n",
            timing_mode_tag(report.timing_mode)
        ));
        out.push_str(&format!("elapsed_secs = {:.6}\n", report.elapsed_secs));
        out.push_str(&format!("steps = {}\n", report.steps));
        out.push_str(&format!("steps_per_sec = {:.6}\n", report.steps_per_sec));
        out.push_str(&format!("mlups = {:.6}\n", report.mlups));
        out.push_str(&format!("samples = {}\n", report.sample_count));
        out.push_str(&format!("rho_final = {:.9}\n", report.quality.final_value));
        out.push_str(&format!(
            "rho_abs_drift_final = {:.9e}\n",
            report.quality.abs_drift_final
        ));
        out.push_str(&format!("rho_std = {:.9e}\n", report.quality.std_dev));
        out.push_str(&format!(
            "step_timing_p50_us = {:.6}\n",
            report.step_timing.p50_us
        ));
        out.push_str(&format!(
            "step_timing_p90_us = {:.6}\n",
            report.step_timing.p90_us
        ));
        out.push_str(&format!(
            "step_timing_p99_us = {:.6}\n",
            report.step_timing.p99_us
        ));
        out.push_str(&format!("h5_output = \"{}\"\n", h5_path.display()));
        out.push_str(&format!("timing_output = \"{}\"\n", timing_path.display()));
        out.push('\n');
    }

    std::fs::write(out_path, out)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();

    // CLI (simple, positional to keep parity with existing warp binaries):
    //   warp-basemark-suite [duration_s=45] [trace_stride=10] [sizes_csv=32,128,256] [precisions_csv=FP32,BF16] [timing_mode=cuda_events] [out_dir]
    let args: Vec<String> = std::env::args().collect();
    let duration_secs: f64 = args.get(1).map_or(Ok(45.0), |s| s.parse())?;
    let trace_stride: usize = args.get(2).map_or(Ok(10usize), |s| s.parse())?;
    let sizes_csv = args.get(3).map(String::as_str).unwrap_or("32,128,256");
    let precisions_csv = args.get(4).map(String::as_str).unwrap_or("FP32,BF16");
    let timing_mode = args
        .get(5)
        .map(String::as_str)
        .map(parse_timing_mode)
        .unwrap_or(Some(TimingMode::CudaEvents))
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "timing_mode must be one of: launch_only, stream_sync_each_step, cuda_events",
            )
        })?;

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

    let sizes = parse_csv(sizes_csv, |s| s.parse::<usize>().ok(), "size")?;
    let precisions = parse_csv(precisions_csv, parse_precision, "precision")?;

    let timestamp = chrono::Utc::now().format("%Y%m%d-%H%M%S").to_string();
    let out_dir = args
        .get(6)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("data/h5/basemarks/{}", timestamp)));
    std::fs::create_dir_all(&out_dir)?;

    let telemetry_path = out_dir.join("gpu_telemetry.csv");
    let telemetry_handle =
        warp_telemetry::spawn_gpu_telemetry_sampler(telemetry_path.clone(), Duration::from_secs(1));

    println!("=== Warp Basemark Suite ===");
    println!(
        "timestamp={}, duration_s={:.2}, trace_stride={}, sizes={}, precisions={}, timing_mode={}, out_dir={}",
        timestamp,
        duration_secs,
        trace_stride,
        sizes_csv,
        precisions_csv,
        timing_mode_tag(timing_mode),
        out_dir.display()
    );

    let mut artifacts = Vec::new();
    let mut reports: Vec<(PathBuf, PathBuf, warp_runner::BenchCaseReport)> = Vec::new();

    // CPU baseline: only for the smallest size in the provided list.
    // Note: CPU backend currently runs FP64 (lbm_3d::solver::LbmSolver3D). We label it as CPU_FP64.
    if let Some(&min_size) = sizes.iter().min() {
        let out_path = out_dir.join(format!(
            "warp_ring_{}_CPU_FP64_{:.0}s.h5",
            min_size, duration_secs
        ));
        let timing_out = out_dir.join(format!(
            "timing_{}_CPU_FP64_{:.0}s.toml",
            min_size, duration_secs
        ));
        println!();
        println!(
            "[RUN] size={} backend=CPU precision=FP64 duration_s={:.2} out={}",
            min_size,
            duration_secs,
            out_path.display()
        );
        let report = run_case(&BenchCase {
            resolution: min_size,
            precision: Precision::FP32, // ignored on CPU backend
            backend: BackendKind::Cpu,
            timing_mode: TimingMode::LaunchOnly,
            duration_secs,
            trace_stride,
            h5_output: Some(out_path.clone()),
        })?;
        print_case_report(&report);
        write_step_timing_report(&timing_out, &report)?;
        artifacts.push(out_path.clone());
        reports.push((out_path, timing_out, report));
    }

    for &size in &sizes {
        for &precision in &precisions {
            let out_path = out_dir.join(format!(
                "warp_ring_{}_{}_{}_{:.0}s.h5",
                size,
                backend_tag(BackendKind::Gpu),
                precision_tag(precision),
                duration_secs
            ));
            let timing_out = out_dir.join(format!(
                "timing_{}_{}_{}_{:.0}s.toml",
                size,
                backend_tag(BackendKind::Gpu),
                precision_tag(precision),
                duration_secs
            ));
            println!();
            println!(
                "[RUN] size={} backend=GPU precision={:?} duration_s={:.2} out={}",
                size,
                precision,
                duration_secs,
                out_path.display()
            );
            let report = run_case(&BenchCase {
                resolution: size,
                precision,
                backend: BackendKind::Gpu,
                timing_mode,
                duration_secs,
                trace_stride,
                h5_output: Some(out_path.clone()),
            })?;
            print_case_report(&report);
            write_step_timing_report(&timing_out, &report)?;
            artifacts.push(out_path.clone());
            reports.push((out_path, timing_out, report));
        }
    }

    println!();
    println!("[GATE] Post-run warp-acceptance gate on basemark artifacts");
    gate_h5_outputs(&artifacts)?;

    if let Some((stop, handle)) = telemetry_handle {
        stop.store(true, Ordering::Relaxed);
        if let Err(err) = handle.join() {
            eprintln!("[WARN] telemetry sampler join failed: {:?}", err);
        }
        println!("TELEMETRY_CSV: {}", telemetry_path.display());
    }

    let summary_out = PathBuf::from(format!(
        "reports/warp_basemark_report_{}.toml",
        chrono::Utc::now().format("%Y_%m_%d")
    ));
    write_basemark_summary(
        &summary_out,
        &timestamp,
        &out_dir,
        duration_secs,
        trace_stride,
        timing_mode,
        &reports,
    )?;
    println!("SUMMARY_REPORT: {}", summary_out.display());
    println!("DONE: basemark suite complete and acceptance gate passed");
    Ok(())
}
