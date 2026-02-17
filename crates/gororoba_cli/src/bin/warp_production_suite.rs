mod warp_runner;
mod warp_telemetry;

use lbm_3d_cuda::Precision;
use std::error::Error;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::Duration;
use warp_runner::{
    gate_h5_outputs_with_profile, print_case_report, run_case, write_step_timing_report,
    BackendKind, BenchCase, GateProfile, TimingMode,
};

fn parse_precision(input: &str) -> Option<Precision> {
    match input.to_ascii_uppercase().as_str() {
        "FP32" => Some(Precision::FP32),
        "BF16" => Some(Precision::BF16),
        _ => None,
    }
}

fn parse_bool01(input: &str) -> Option<bool> {
    match input {
        "1" | "true" | "TRUE" | "yes" | "YES" => Some(true),
        "0" | "false" | "FALSE" | "no" | "NO" => Some(false),
        _ => None,
    }
}

fn precision_label(precision: Precision) -> &'static str {
    match precision {
        Precision::FP32 => "FP32",
        Precision::BF16 => "BF16",
    }
}

fn env_or_default(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| format!("<default:{default}>"))
}

fn run_case_and_collect(
    out_dir: &std::path::Path,
    artifacts: &mut Vec<PathBuf>,
    resolution: usize,
    run_precision: Precision,
    duration_secs: f64,
    trace_stride: usize,
) -> Result<(), Box<dyn Error>> {
    let out_path = out_dir.join(format!(
        "warp_ring_{}_{}_{:.0}s.h5",
        resolution,
        precision_label(run_precision),
        duration_secs
    ));
    let timing_out = out_dir.join(format!(
        "timing_{}_{}_{:.0}s.toml",
        resolution,
        precision_label(run_precision),
        duration_secs
    ));
    println!();
    println!(
        "[RUN] size={}, precision={:?}, duration_s={:.2}, out={}",
        resolution,
        run_precision,
        duration_secs,
        out_path.display()
    );
    let report = run_case(&BenchCase {
        resolution,
        precision: run_precision,
        backend: BackendKind::Gpu,
        timing_mode: TimingMode::CudaEvents,
        duration_secs,
        trace_stride,
        h5_output: Some(out_path.clone()),
    })?;
    print_case_report(&report);
    write_step_timing_report(&timing_out, &report)?;
    println!("TIMING_REPORT: {}", timing_out.display());
    println!("BENCH_COMPLETE");
    artifacts.push(out_path);
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();
    let duration_128_s: f64 = args.get(1).map_or(Ok(60.0), |s| s.parse())?;
    let duration_256_s: f64 = args.get(2).map_or(Ok(60.0), |s| s.parse())?;
    let trace_stride: usize = args.get(3).map_or(Ok(10usize), |s| s.parse())?;
    let precision = args
        .get(4)
        .map_or(Some(Precision::BF16), |s| parse_precision(s))
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "precision must be FP32 or BF16",
            )
        })?;
    let run_fp32_reference = args
        .get(5)
        .map_or(Some(false), |s| parse_bool01(s))
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "run_fp32_reference must be one of: 0,1,true,false,yes,no",
            )
        })?;
    let telemetry_csv = args.get(6).map(PathBuf::from);

    if duration_128_s <= 0.0 || duration_256_s <= 0.0 {
        return Err(
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "durations must be > 0").into(),
        );
    }
    if trace_stride == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "trace_stride must be >= 1",
        )
        .into());
    }

    let timestamp = chrono::Utc::now().format("%Y%m%d-%H%M%S").to_string();
    let out_dir = PathBuf::from(format!("data/h5/production/{}", timestamp));
    std::fs::create_dir_all(&out_dir)?;
    let telemetry_path = telemetry_csv
        .clone()
        .unwrap_or_else(|| out_dir.join("gpu_telemetry.csv"));
    let telemetry_handle =
        warp_telemetry::spawn_gpu_telemetry_sampler(telemetry_path.clone(), Duration::from_secs(1));

    println!("=== Warp Production Suite ===");
    println!(
        "timestamp={}, precision={:?}, duration_128_s={}, duration_256_s={}, trace_stride={}, run_fp32_reference={}, out_dir={}, telemetry_csv={}",
        timestamp,
        precision,
        duration_128_s,
        duration_256_s,
        trace_stride,
        run_fp32_reference,
        out_dir.display(),
        telemetry_path.display()
    );
    println!(
        "forcing_env: GOROROBA_KOLMO_MODE_Y={}, GOROROBA_KOLMO_RE_TARGET={}, GOROROBA_KOLMO_MAX_MACH={}, GOROROBA_KOLMO_RHO0={}",
        env_or_default("GOROROBA_KOLMO_MODE_Y", "1"),
        env_or_default("GOROROBA_KOLMO_RE_TARGET", "64.0"),
        env_or_default("GOROROBA_KOLMO_MAX_MACH", "0.08"),
        env_or_default("GOROROBA_KOLMO_RHO0", "1.0")
    );
    println!(
        "forcing_env: GOROROBA_BF16_MIN_DELTA_ULP_RATIO={}, GOROROBA_BF16_ENFORCE_MODE_FLOOR={}, GOROROBA_ENFORCE_MODE_FLOOR_ALL_PRECISIONS={}",
        env_or_default("GOROROBA_BF16_MIN_DELTA_ULP_RATIO", "1.0e-2"),
        env_or_default("GOROROBA_BF16_ENFORCE_MODE_FLOOR", "true"),
        env_or_default("GOROROBA_ENFORCE_MODE_FLOOR_ALL_PRECISIONS", "false")
    );

    let mut artifacts = Vec::new();

    let run_result = (|| -> Result<(), Box<dyn Error>> {
        run_case_and_collect(
            &out_dir,
            &mut artifacts,
            128,
            precision,
            duration_128_s,
            trace_stride,
        )?;
        run_case_and_collect(
            &out_dir,
            &mut artifacts,
            256,
            precision,
            duration_256_s,
            trace_stride,
        )?;

        if run_fp32_reference {
            run_case_and_collect(
                &out_dir,
                &mut artifacts,
                128,
                Precision::FP32,
                duration_128_s,
                trace_stride,
            )?;
            run_case_and_collect(
                &out_dir,
                &mut artifacts,
                256,
                Precision::FP32,
                duration_256_s,
                trace_stride,
            )?;
        }

        println!();
        println!("[GATE] Post-run canonical_300s_measured artifact gate");
        gate_h5_outputs_with_profile(&artifacts, GateProfile::Canonical300sMeasured)?;
        Ok(())
    })();

    if let Some((stop, handle)) = telemetry_handle {
        stop.store(true, Ordering::Relaxed);
        if let Err(err) = handle.join() {
            eprintln!("[WARN] GPU telemetry worker join failed: {:?}", err);
        } else {
            println!("GPU_TELEMETRY: {}", telemetry_path.display());
        }
    }
    run_result?;
    println!("DONE: production suite complete and canonical_300s_measured gate passed");
    Ok(())
}
