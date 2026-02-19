mod warp_runner;

use lbm_3d_cuda::Precision;
use std::error::Error;
use std::path::PathBuf;
use warp_runner::{
    gate_h5_outputs, print_case_report, run_case, write_step_timing_report, BackendKind, BenchCase,
    TimingMode,
};

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();
    let duration_secs: f64 = args.get(1).map_or(Ok(5.0), |s| s.parse())?;
    let trace_stride: usize = args.get(2).map_or(Ok(5usize), |s| s.parse())?;
    let out_dir = args
        .get(3)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/h5/smoke_bench"));

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

    std::fs::create_dir_all(&out_dir)?;
    println!("=== Warp GPU Smoke ===");
    println!(
        "duration_s={}, trace_stride={}, sizes=8,16, precisions=FP32,BF16, out_dir={}",
        duration_secs,
        trace_stride,
        out_dir.display()
    );

    let sizes = [8usize, 16usize];
    let precisions = [Precision::FP32, Precision::BF16];
    let mut artifacts = Vec::new();

    for &size in &sizes {
        for &precision in &precisions {
            let out_path = out_dir.join(format!(
                "warp_ring_{}_{}.h5",
                size,
                match precision {
                    Precision::FP32 => "FP32",
                    Precision::BF16 => "BF16",
                    Precision::FP64 => "FP64",
                }
            ));
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
                Precision::FP64 => "FP64",
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
    println!("[GATE] Post-run warp acceptance gate");
    gate_h5_outputs(&artifacts)?;
    println!("DONE: smoke suite complete and finite gate passed");
    Ok(())
}
