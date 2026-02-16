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

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();
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

    let report = run_case(&BenchCase {
        resolution,
        precision,
        backend: BackendKind::Gpu,
        timing_mode: TimingMode::CudaEvents,
        duration_secs,
        trace_stride,
        h5_output,
    })?;
    print_case_report(&report);
    let timing_out = report.h5_output.as_ref().map(|h5| {
        let parent = h5.parent().map_or_else(|| PathBuf::from("."), PathBuf::from);
        parent.join(format!(
            "timing_{}_{}_{}s.toml",
            resolution,
            match precision {
                Precision::FP32 => "FP32",
                Precision::BF16 => "BF16",
            },
            duration_secs.round() as usize
        ))
    });
    if let Some(path) = timing_out.as_deref() {
        write_step_timing_report(path, &report)?;
        println!("TIMING_REPORT: {}", path.display());
    }
    if let Some(path) = report.h5_output.clone() {
        gate_h5_outputs(&[path])?;
    }
    println!("BENCH_COMPLETE");
    Ok(())
}
