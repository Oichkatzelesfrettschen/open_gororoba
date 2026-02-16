use lbm_3d_cuda::Precision;
use std::error::Error;
use std::path::{Path, PathBuf};

use crate::warp_runner::{
    gate_h5_outputs, print_case_report, run_case, write_step_timing_report, BackendKind, BenchCase,
    BenchCaseReport, TimingMode,
};

#[derive(Debug, Clone)]
pub struct PrecisionSuiteConfig {
    pub backend: BackendKind,
    pub timing_mode: TimingMode,
    pub duration_secs: f64,
    pub trace_stride: usize,
    pub sizes: Vec<usize>,
    pub precisions: Vec<Precision>,
    pub out_dir: Option<PathBuf>,
    pub single_output: Option<PathBuf>,
    pub detailed_names: bool,
    pub gate: bool,
    pub write_summary: bool,
}

pub fn parse_precision(input: &str) -> Option<Precision> {
    match input.to_ascii_uppercase().as_str() {
        "FP32" => Some(Precision::FP32),
        "BF16" => Some(Precision::BF16),
        _ => None,
    }
}

#[allow(dead_code)]
pub fn parse_backend(input: &str) -> Option<BackendKind> {
    match input.to_ascii_lowercase().as_str() {
        "cpu" => Some(BackendKind::Cpu),
        "gpu" | "cuda" => Some(BackendKind::Gpu),
        _ => None,
    }
}

#[allow(dead_code)]
pub fn parse_timing_mode(input: &str) -> Option<TimingMode> {
    match input.to_ascii_lowercase().as_str() {
        "launch_only" | "launch" => Some(TimingMode::LaunchOnly),
        "stream_sync_each_step" | "sync_each_step" | "sync" => Some(TimingMode::StreamSyncEachStep),
        "cuda_events" | "events" => Some(TimingMode::CudaEvents),
        _ => None,
    }
}

#[allow(dead_code)]
pub fn parse_bool01(input: &str) -> Option<bool> {
    match input {
        "1" | "true" | "TRUE" | "yes" | "YES" => Some(true),
        "0" | "false" | "FALSE" | "no" | "NO" => Some(false),
        _ => None,
    }
}

#[allow(dead_code)]
pub fn parse_csv<T>(
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

#[allow(dead_code)]
pub fn default_timing_mode_for_backend(backend: BackendKind) -> TimingMode {
    match backend {
        BackendKind::Cpu => TimingMode::LaunchOnly,
        BackendKind::Gpu => TimingMode::CudaEvents,
    }
}

fn precision_tag(precision: Precision) -> &'static str {
    match precision {
        Precision::FP32 => "FP32",
        Precision::BF16 => "BF16",
    }
}

fn backend_tag(backend: BackendKind) -> &'static str {
    match backend {
        BackendKind::Cpu => "CPU",
        BackendKind::Gpu => "GPU",
    }
}

fn timing_mode_tag(mode: TimingMode) -> &'static str {
    match mode {
        TimingMode::LaunchOnly => "launch_only",
        TimingMode::StreamSyncEachStep => "stream_sync_each_step",
        TimingMode::CudaEvents => "cuda_events",
    }
}

fn case_h5_name(resolution: usize, backend: BackendKind, precision: Precision, duration_secs: f64, detailed: bool) -> String {
    if detailed {
        format!(
            "warp_ring_{}_{}_{}_{}s.h5",
            resolution,
            backend_tag(backend),
            precision_tag(precision),
            duration_secs.round() as u64
        )
    } else {
        format!("warp_ring_{}_{}.h5", resolution, precision_tag(precision))
    }
}

fn timing_name(resolution: usize, backend: BackendKind, precision: Precision, duration_secs: f64, detailed: bool) -> String {
    if detailed {
        format!(
            "timing_{}_{}_{}_{}s.toml",
            resolution,
            backend_tag(backend),
            precision_tag(precision),
            duration_secs.round() as u64
        )
    } else {
        format!(
            "timing_{}_{}_{}s.toml",
            resolution,
            precision_tag(precision),
            duration_secs.round() as u64
        )
    }
}

fn default_summary_path(out_dir: Option<&Path>, single_output: Option<&Path>) -> Option<PathBuf> {
    if let Some(dir) = out_dir {
        return Some(dir.join("warp_precision_suite_summary.toml"));
    }
    if let Some(path) = single_output {
        if let Some(parent) = path.parent() {
            return Some(parent.join("warp_precision_suite_summary.toml"));
        }
    }
    None
}

fn write_summary(
    out_path: &Path,
    cfg: &PrecisionSuiteConfig,
    cases: &[(PathBuf, PathBuf, BenchCaseReport)],
) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let timestamp = chrono::Utc::now().format("%Y%m%d-%H%M%S").to_string();

    let mut out = String::new();
    out.push_str("# Warp precision suite summary\n");
    out.push_str("schema = \"open_gororoba.warp_precision_suite.v1\"\n");
    out.push_str(&format!("timestamp = \"{}\"\n", timestamp));
    out.push_str(&format!("backend = \"{}\"\n", backend_tag(cfg.backend)));
    out.push_str(&format!(
        "timing_mode = \"{}\"\n",
        timing_mode_tag(cfg.timing_mode)
    ));
    out.push_str(&format!("duration_secs = {:.6}\n", cfg.duration_secs));
    out.push_str(&format!("trace_stride = {}\n", cfg.trace_stride));
    out.push_str(&format!("detailed_names = {}\n", cfg.detailed_names));
    if let Some(dir) = cfg.out_dir.as_deref() {
        out.push_str(&format!("out_dir = \"{}\"\n", dir.display()));
    }
    if let Some(path) = cfg.single_output.as_deref() {
        out.push_str(&format!("single_output = \"{}\"\n", path.display()));
    }
    out.push('\n');

    for (h5_path, timing_path, report) in cases {
        out.push_str("[[case]]\n");
        out.push_str(&format!("resolution = {}\n", report.resolution));
        out.push_str(&format!("precision = \"{}\"\n", precision_tag(report.precision)));
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
        out.push_str(&format!(
            "timing_output = \"{}\"\n",
            timing_path.display()
        ));
        out.push('\n');
    }

    std::fs::write(out_path, out)?;
    Ok(())
}

pub fn run_precision_suite(cfg: PrecisionSuiteConfig) -> Result<(), Box<dyn Error>> {
    if cfg.duration_secs <= 0.0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "duration_s must be > 0",
        )
        .into());
    }
    if cfg.trace_stride == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "trace_stride must be >= 1",
        )
        .into());
    }
    if cfg.sizes.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "sizes list cannot be empty",
        )
        .into());
    }
    if cfg.precisions.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "precisions list cannot be empty",
        )
        .into());
    }

    if cfg.single_output.is_some() && (cfg.sizes.len() != 1 || cfg.precisions.len() != 1) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "single_output requires exactly one size and one precision",
        )
        .into());
    }

    if let Some(dir) = cfg.out_dir.as_deref() {
        std::fs::create_dir_all(dir)?;
    }
    if let Some(path) = cfg.single_output.as_deref() {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
    }

    let mut artifacts: Vec<PathBuf> = Vec::new();
    let mut cases_with_paths: Vec<(PathBuf, PathBuf, BenchCaseReport)> = Vec::new();

    for &size in &cfg.sizes {
        for &precision in &cfg.precisions {
            let h5_output = if let Some(path) = cfg.single_output.as_ref() {
                Some(path.clone())
            } else if let Some(dir) = cfg.out_dir.as_ref() {
                Some(dir.join(case_h5_name(
                    size,
                    cfg.backend,
                    precision,
                    cfg.duration_secs,
                    cfg.detailed_names,
                )))
            } else {
                None
            };

            println!();
            println!(
                "[RUN] size={}, backend={:?}, precision={:?}, duration_s={:.2}, timing_mode={:?}",
                size, cfg.backend, precision, cfg.duration_secs, cfg.timing_mode
            );
            if let Some(path) = h5_output.as_deref() {
                println!("      out={}", path.display());
            }

            let report = run_case(&BenchCase {
                resolution: size,
                precision,
                backend: cfg.backend,
                timing_mode: cfg.timing_mode,
                duration_secs: cfg.duration_secs,
                trace_stride: cfg.trace_stride,
                h5_output: h5_output.clone(),
            })?;
            print_case_report(&report);

            if let Some(h5_path) = h5_output.as_ref() {
                let timing_path = if let Some(dir) = cfg.out_dir.as_deref() {
                    dir.join(timing_name(
                        size,
                        cfg.backend,
                        precision,
                        cfg.duration_secs,
                        cfg.detailed_names,
                    ))
                } else {
                    let parent = h5_path.parent().map_or_else(|| PathBuf::from("."), PathBuf::from);
                    parent.join(timing_name(
                        size,
                        cfg.backend,
                        precision,
                        cfg.duration_secs,
                        cfg.detailed_names,
                    ))
                };
                write_step_timing_report(&timing_path, &report)?;
                println!("TIMING_REPORT: {}", timing_path.display());

                artifacts.push(h5_path.clone());
                cases_with_paths.push((h5_path.clone(), timing_path, report));
            }
            println!("BENCH_COMPLETE");
        }
    }

    if cfg.gate && !artifacts.is_empty() {
        println!();
        println!("[GATE] Post-run warp acceptance gate");
        gate_h5_outputs(&artifacts)?;
    }

    if cfg.write_summary && !cases_with_paths.is_empty() {
        if let Some(path) = default_summary_path(cfg.out_dir.as_deref(), cfg.single_output.as_deref()) {
            write_summary(&path, &cfg, &cases_with_paths)?;
            println!("SUMMARY_REPORT: {}", path.display());
        }
    }

    Ok(())
}

#[allow(dead_code)]
pub fn run_bench_compat_args(args: &[String]) -> Result<(), Box<dyn Error>> {
    if args.len() < 3 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "Usage: warp-bench-precision <res> <FP32|BF16> [duration_s=20] [trace_stride=10] [h5_out]",
        )
        .into());
    }

    let resolution: usize = args[1].parse()?;
    let precision = parse_precision(&args[2]).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "precision must be FP32 or BF16",
        )
    })?;
    let duration_secs = if args.len() >= 4 {
        args[3].parse()?
    } else {
        20.0
    };
    let trace_stride = if args.len() >= 5 {
        args[4].parse()?
    } else {
        10usize
    };
    let h5_output = args.get(5).map(PathBuf::from);

    run_precision_suite(PrecisionSuiteConfig {
        backend: BackendKind::Gpu,
        timing_mode: TimingMode::CudaEvents,
        duration_secs,
        trace_stride,
        sizes: vec![resolution],
        precisions: vec![precision],
        out_dir: None,
        single_output: h5_output,
        detailed_names: false,
        gate: true,
        write_summary: true,
    })
}

#[allow(dead_code)]
pub fn run_matrix_compat_args(args: &[String]) -> Result<(), Box<dyn Error>> {
    let duration_secs: f64 = args.get(1).map_or(Ok(20.0), |s| s.parse())?;
    let trace_stride: usize = args.get(2).map_or(Ok(10usize), |s| s.parse())?;
    let sizes_csv = args.get(3).map(String::as_str).unwrap_or("8,16,32,64");
    let precisions_csv = args.get(4).map(String::as_str).unwrap_or("FP32,BF16");
    let out_dir = args
        .get(5)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/h5/precision_bench"));

    let sizes = parse_csv(sizes_csv, |s| s.parse::<usize>().ok(), "size")?;
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

    run_precision_suite(PrecisionSuiteConfig {
        backend: BackendKind::Gpu,
        timing_mode: TimingMode::CudaEvents,
        duration_secs,
        trace_stride,
        sizes,
        precisions,
        out_dir: Some(out_dir),
        single_output: None,
        detailed_names: false,
        gate: true,
        write_summary: true,
    })?;
    println!("DONE: matrix complete");
    Ok(())
}
