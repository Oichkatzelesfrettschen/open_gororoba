use algebra_core::physics::octonion_field::FieldParams;
#[cfg(feature = "hdf5-export")]
use data_core::hdf5_export::{
    export_experiment_contract, export_rho_quality_metrics, export_simulation_trace,
    read_simulation_trace_component, scan_hdf5_numeric_datasets, NumericDatasetScanStatus,
};
use data_core::quality::{validate_rho_trace, RhoQualityThresholds, RhoTraceQuality};
use gororoba_engine::simulation::{E7SpectralFilter, SimulationConfig3D, SimulationState3D};
#[cfg(feature = "hdf5-export")]
use gororoba_contracts::{WarpRingConfig, WarpRingExperiment, WarpRingResults};
use lbm_3d_cuda::Precision;
use std::error::Error;
use std::path::{Path, PathBuf};

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    Cpu,
    Gpu,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimingMode {
    /// Wall-clock time around `state.step()` without any GPU synchronization.
    ///
    /// For GPU runs this mostly measures kernel-launch overhead and will significantly
    /// under-report true device time.
    LaunchOnly,
    /// Wall-clock time around `state.step()` plus a stream synchronization after each step.
    ///
    /// This is simple and truthful, but can perturb performance.
    StreamSyncEachStep,
    /// CUDA event timing around `state.step()` on the solver stream.
    ///
    /// This measures GPU time (and implicitly synchronizes to read event timestamps).
    CudaEvents,
}

#[derive(Debug, Clone)]
pub struct BenchCase {
    pub resolution: usize,
    pub precision: Precision,
    pub backend: BackendKind,
    pub timing_mode: TimingMode,
    pub duration_secs: f64,
    pub trace_stride: usize,
    pub h5_output: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub struct StepTimingBin {
    pub lower_us: f64,
    pub upper_us: f64,
    pub count: usize,
}

#[derive(Debug, Clone)]
pub struct StepTimingStats {
    pub sample_count: usize,
    pub min_us: f64,
    pub max_us: f64,
    pub mean_us: f64,
    pub p50_us: f64,
    pub p90_us: f64,
    pub p99_us: f64,
    pub bins: Vec<StepTimingBin>,
}

#[derive(Debug, Clone)]
pub struct BenchCaseReport {
    pub resolution: usize,
    pub precision: Precision,
    pub backend: BackendKind,
    pub timing_mode: TimingMode,
    pub elapsed_secs: f64,
    pub steps: usize,
    pub steps_per_sec: f64,
    pub mlups: f64,
    pub sample_count: usize,
    pub quality: RhoTraceQuality,
    pub step_timing: StepTimingStats,
    pub h5_output: Option<PathBuf>,
}

fn percentile_sorted(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let clamped = percentile.clamp(0.0, 1.0);
    let pos = clamped * (values.len().saturating_sub(1)) as f64;
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;
    if lower == upper {
        values[lower]
    } else {
        let w = pos - lower as f64;
        values[lower] * (1.0 - w) + values[upper] * w
    }
}

fn build_histogram(values: &[f64], bin_count: usize) -> Vec<StepTimingBin> {
    if values.is_empty() {
        return Vec::new();
    }
    let bins = bin_count.max(1);
    let min = values
        .iter()
        .copied()
        .fold(f64::INFINITY, |acc, v| if v < acc { v } else { acc });
    let max = values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |acc, v| if v > acc { v } else { acc });
    let range = max - min;
    if range <= f64::EPSILON {
        return vec![StepTimingBin {
            lower_us: min,
            upper_us: max,
            count: values.len(),
        }];
    }

    let width = range / bins as f64;
    let mut counts = vec![0usize; bins];
    for value in values {
        let mut idx = ((*value - min) / width).floor() as usize;
        if idx >= bins {
            idx = bins - 1;
        }
        counts[idx] += 1;
    }

    let mut out = Vec::with_capacity(bins);
    for (idx, count) in counts.into_iter().enumerate() {
        let lower = min + idx as f64 * width;
        let upper = if idx + 1 == bins {
            max
        } else {
            min + (idx as f64 + 1.0) * width
        };
        out.push(StepTimingBin {
            lower_us: lower,
            upper_us: upper,
            count,
        });
    }
    out
}

fn compute_step_timing_stats(step_times_us: &[f64], histogram_bins: usize) -> StepTimingStats {
    if step_times_us.is_empty() {
        return StepTimingStats {
            sample_count: 0,
            min_us: 0.0,
            max_us: 0.0,
            mean_us: 0.0,
            p50_us: 0.0,
            p90_us: 0.0,
            p99_us: 0.0,
            bins: Vec::new(),
        };
    }

    let mut sorted = step_times_us.to_vec();
    sorted.sort_by(f64::total_cmp);
    let sample_count = sorted.len();
    let min_us = sorted[0];
    let max_us = sorted[sample_count - 1];
    let sum: f64 = sorted.iter().sum();
    let mean_us = sum / sample_count as f64;
    let p50_us = percentile_sorted(&sorted, 0.50);
    let p90_us = percentile_sorted(&sorted, 0.90);
    let p99_us = percentile_sorted(&sorted, 0.99);
    let bins = build_histogram(&sorted, histogram_bins);

    StepTimingStats {
        sample_count,
        min_us,
        max_us,
        mean_us,
        p50_us,
        p90_us,
        p99_us,
        bins,
    }
}

pub fn write_step_timing_report(path: &Path, report: &BenchCaseReport) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let precision = match report.backend {
        BackendKind::Cpu => "FP64",
        BackendKind::Gpu => match report.precision {
            Precision::FP32 => "FP32",
            Precision::BF16 => "BF16",
        },
    };
    let mut out = String::new();
    out.push_str("# Per-step timing histogram summary generated by warp_runner.\n");
    out.push_str("[case]\n");
    out.push_str(&format!("resolution = {}\n", report.resolution));
    out.push_str(&format!("precision = \"{}\"\n", precision));
    out.push_str(&format!(
        "backend = \"{}\"\n",
        match report.backend {
            BackendKind::Cpu => "cpu",
            BackendKind::Gpu => "gpu",
        }
    ));
    out.push_str(&format!(
        "timing_mode = \"{}\"\n",
        match report.timing_mode {
            TimingMode::LaunchOnly => "launch_only",
            TimingMode::StreamSyncEachStep => "stream_sync_each_step",
            TimingMode::CudaEvents => "cuda_events",
        }
    ));
    out.push_str(&format!("elapsed_secs = {:.6}\n", report.elapsed_secs));
    out.push_str(&format!("steps = {}\n", report.steps));
    out.push_str(&format!("steps_per_sec = {:.6}\n", report.steps_per_sec));
    out.push_str(&format!("mlups = {:.6}\n", report.mlups));
    if let Some(h5) = report.h5_output.as_deref() {
        out.push_str(&format!("h5_output = \"{}\"\n", h5.display()));
    }
    out.push_str("\n[timing]\n");
    out.push_str(&format!("sample_count = {}\n", report.step_timing.sample_count));
    out.push_str(&format!("min_us = {:.6}\n", report.step_timing.min_us));
    out.push_str(&format!("max_us = {:.6}\n", report.step_timing.max_us));
    out.push_str(&format!("mean_us = {:.6}\n", report.step_timing.mean_us));
    out.push_str(&format!("p50_us = {:.6}\n", report.step_timing.p50_us));
    out.push_str(&format!("p90_us = {:.6}\n", report.step_timing.p90_us));
    out.push_str(&format!("p99_us = {:.6}\n", report.step_timing.p99_us));
    for bin in &report.step_timing.bins {
        out.push_str("\n[[timing.histogram_bin]]\n");
        out.push_str(&format!("lower_us = {:.6}\n", bin.lower_us));
        out.push_str(&format!("upper_us = {:.6}\n", bin.upper_us));
        out.push_str(&format!("count = {}\n", bin.count));
    }
    std::fs::write(path, out)?;
    Ok(())
}

#[cfg(feature = "hdf5-export")]
fn export_bench_trace(
    out_path: &Path,
    resolution: usize,
    backend: BackendKind,
    precision: Precision,
    total_steps: usize,
    elapsed_secs: f64,
    time_hist: &[f64],
    rho_hist: &[f64],
    quality: &RhoTraceQuality,
    thresholds: RhoQualityThresholds,
) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let contract = WarpRingExperiment {
        experiment_id: format!(
            "BENCH-WARP-{}-{}-{}",
            resolution,
            match backend {
                BackendKind::Cpu => "cpu_fp64",
                BackendKind::Gpu => match precision {
                    Precision::FP32 => "gpu_fp32",
                    Precision::BF16 => "gpu_bf16",
                },
            },
            chrono::Utc::now().format("%Y%m%d-%H%M%S"),
        ),
        config: WarpRingConfig {
            resolution,
            steps: total_steps,
            tau: 0.6,
            forcing_type: "E7_Bench_Filter".to_string(),
            coupling_lambda: 0.95,
            initial_condition: "Uniform_Rho1_U0".to_string(),
        },
        results: WarpRingResults {
            final_enstrophy: 0.0,
            mean_density: *rho_hist.last().unwrap_or(&f64::NAN),
            betti_1_persistence: None,
            execution_time_s: elapsed_secs,
            steps_per_second: total_steps as f64 / elapsed_secs.max(1.0e-12),
            mlups: (total_steps as f64 / elapsed_secs.max(1.0e-12))
                * (resolution * resolution * resolution) as f64
                / 1.0e6,
            artifact_path: out_path.display().to_string(),
        },
    };
    export_experiment_contract(out_path, &contract)?;
    let enstrophy = vec![0.0; time_hist.len()];
    let algebra_norm = vec![0.0; time_hist.len()];
    export_simulation_trace(out_path, time_hist, rho_hist, &enstrophy, &algebra_norm)?;
    export_rho_quality_metrics(out_path, quality, thresholds)?;
    Ok(())
}

pub fn run_case(case: &BenchCase) -> Result<BenchCaseReport, Box<dyn Error>> {
    let config = SimulationConfig3D {
        nx: case.resolution,
        ny: case.resolution,
        nz: case.resolution,
        tau: 0.6,
        use_gpu: matches!(case.backend, BackendKind::Gpu),
        precision: case.precision,
        algebra_params: FieldParams::default(),
        coupling_fluid_algebra: 0.1,
        coupling_algebra_fluid: 0.1,
    };
    let mut state = SimulationState3D::new(config)?;
    state.frustration = Some(Box::new(E7SpectralFilter::new(
        case.resolution,
        case.resolution,
        case.resolution,
        0.95,
    )));

    state.step()?;
    let start = std::time::Instant::now();
    let mut steps = 0usize;
    let mut time_hist = Vec::new();
    let mut rho_hist = Vec::new();
    let mut step_times_us = Vec::new();

    while start.elapsed().as_secs_f64() < case.duration_secs {
        let step_start = std::time::Instant::now();
        match case.timing_mode {
            TimingMode::LaunchOnly => {
                state.step()?;
                step_times_us.push(step_start.elapsed().as_secs_f64() * 1.0e6);
            }
            TimingMode::StreamSyncEachStep => {
                state.step()?;
                #[cfg(feature = "gpu")]
                if matches!(case.backend, BackendKind::Gpu) {
                    state.fluid.stream().synchronize()?;
                }
                step_times_us.push(step_start.elapsed().as_secs_f64() * 1.0e6);
            }
            TimingMode::CudaEvents => {
                #[cfg(feature = "gpu")]
                {
                    if !matches!(case.backend, BackendKind::Gpu) {
                        // CPU backend has no CUDA stream; fall back to wall-clock.
                        state.step()?;
                        step_times_us.push(step_start.elapsed().as_secs_f64() * 1.0e6);
                    } else {
                        let stream = state.fluid.stream().clone();
                        let flags = cudarc::driver::sys::CUevent_flags::CU_EVENT_BLOCKING_SYNC;
                        let start_evt = stream.record_event(Some(flags))?;
                        state.step()?;
                        let end_evt = stream.record_event(Some(flags))?;
                        let ms = start_evt.elapsed_ms(&end_evt)?;
                        step_times_us.push(ms as f64 * 1000.0);
                    }
                }
                #[cfg(not(feature = "gpu"))]
                {
                    state.step()?;
                    step_times_us.push(step_start.elapsed().as_secs_f64() * 1.0e6);
                }
            }
        }
        steps += 1;
        if steps % case.trace_stride == 0 {
            time_hist.push(start.elapsed().as_secs_f64());
            rho_hist.push(state.fluid.try_mean_density()?);
        }
    }
    if time_hist.is_empty() {
        time_hist.push(start.elapsed().as_secs_f64());
        rho_hist.push(state.fluid.try_mean_density()?);
    }

    let elapsed = start.elapsed().as_secs_f64();
    if step_times_us.is_empty() {
        step_times_us.push(elapsed * 1.0e6);
    }
    let steps_per_sec = steps as f64 / elapsed.max(1.0e-12);
    let mlups = (steps_per_sec * (case.resolution * case.resolution * case.resolution) as f64) / 1.0e6;
    let thresholds = RhoQualityThresholds::default();
    let quality = validate_rho_trace(&rho_hist, thresholds)
        .map_err(|e| std::io::Error::other(format!("rho quality gate failed: {e}")))?;
    let step_timing = compute_step_timing_stats(&step_times_us, 20);

    if let Some(path) = case.h5_output.as_deref() {
        #[cfg(feature = "hdf5-export")]
        {
            export_bench_trace(
                path,
                case.resolution,
                case.backend,
                case.precision,
                steps,
                elapsed,
                &time_hist,
                &rho_hist,
                &quality,
                thresholds,
            )?;
        }
        #[cfg(not(feature = "hdf5-export"))]
        {
            return Err(std::io::Error::other(format!(
                "HDF5 export requested for {}, but gororoba_cli was built without hdf5-export feature",
                path.display()
            ))
            .into());
        }
    }

    Ok(BenchCaseReport {
        resolution: case.resolution,
        precision: case.precision,
        backend: case.backend,
        timing_mode: case.timing_mode,
        elapsed_secs: elapsed,
        steps,
        steps_per_sec,
        mlups,
        sample_count: rho_hist.len(),
        quality,
        step_timing,
        h5_output: case.h5_output.clone(),
    })
}

pub fn print_case_report(report: &BenchCaseReport) {
    let precision_display = match report.backend {
        BackendKind::Cpu => "FP64".to_string(),
        BackendKind::Gpu => format!("{:?}", report.precision),
    };
    println!(
        "{:4} | {:?}/{}/{:?} | {:6.2}s | {:10.2} steps/s | {:8.2} MLUPS | steps={} | samples={} | rho_final={:.6} | drift={:.3e} | std={:.3e}",
        report.resolution,
        report.backend,
        precision_display,
        report.timing_mode,
        report.elapsed_secs,
        report.steps_per_sec,
        report.mlups,
        report.steps,
        report.sample_count,
        report.quality.final_value,
        report.quality.abs_drift_final,
        report.quality.std_dev
    );
    println!(
        "     step_timing_us: mean={:.2}, p50={:.2}, p90={:.2}, p99={:.2}, min={:.2}, max={:.2}, n={}",
        report.step_timing.mean_us,
        report.step_timing.p50_us,
        report.step_timing.p90_us,
        report.step_timing.p99_us,
        report.step_timing.min_us,
        report.step_timing.max_us,
        report.step_timing.sample_count
    );
    if let Some(path) = report.h5_output.as_deref() {
        println!("HDF5_TRACE: {}", path.display());
    }
}

pub fn gate_h5_outputs(paths: &[PathBuf]) -> Result<(), Box<dyn Error>> {
    if paths.is_empty() {
        return Ok(());
    }

    #[cfg(feature = "hdf5-export")]
    {
        fn all_finite(values: &[f64]) -> bool {
            values.iter().all(|v| v.is_finite())
        }

        fn nondecreasing(values: &[f64]) -> bool {
            values.windows(2).all(|w| w[1] >= w[0])
        }

        let thresholds = RhoQualityThresholds::default();
        for path in paths {
            let report = scan_hdf5_numeric_datasets(path)?;
            for entry in &report.entries {
                if entry.status == NumericDatasetScanStatus::Checked
                    && (entry.nan_count > 0 || entry.inf_count > 0)
                {
                    return Err(std::io::Error::other(format!(
                        "{}: non-finite numeric dataset {} (dtype={}, nan={}, inf={})",
                        path.display(),
                        entry.path,
                        entry.dtype,
                        entry.nan_count,
                        entry.inf_count
                    ))
                    .into());
                }
                if entry.status == NumericDatasetScanStatus::UnsupportedNumericLayout {
                    return Err(std::io::Error::other(format!(
                        "{}: unsupported numeric dataset layout {} (dtype={}); fail-closed",
                        path.display(),
                        entry.path,
                        entry.dtype
                    ))
                    .into());
                }
            }

            let time = read_simulation_trace_component(path, "time")?;
            let rho = read_simulation_trace_component(path, "rho_mean")?;
            let enstrophy = read_simulation_trace_component(path, "enstrophy")?;
            let algebra_norm = read_simulation_trace_component(path, "algebra_norm")?;
            let n = rho.len();
            if n == 0 {
                return Err(std::io::Error::other(format!(
                    "{}: empty rho_mean trace",
                    path.display()
                ))
                .into());
            }
            if time.len() != n || enstrophy.len() != n || algebra_norm.len() != n {
                return Err(std::io::Error::other(format!(
                    "{}: trace length mismatch time={} rho={} enstrophy={} algebra_norm={}",
                    path.display(),
                    time.len(),
                    rho.len(),
                    enstrophy.len(),
                    algebra_norm.len()
                ))
                .into());
            }
            if !all_finite(&time)
                || !all_finite(&rho)
                || !all_finite(&enstrophy)
                || !all_finite(&algebra_norm)
            {
                return Err(std::io::Error::other(format!(
                    "{}: non-finite value detected in simulation/trace datasets",
                    path.display()
                ))
                .into());
            }
            if !nondecreasing(&time) {
                return Err(std::io::Error::other(format!(
                    "{}: time trace is not monotonic nondecreasing",
                    path.display()
                ))
                .into());
            }

            let quality = validate_rho_trace(&rho, thresholds).map_err(|e| {
                std::io::Error::other(format!(
                    "{}: rho quality gate failed: {e}",
                    path.display()
                ))
            })?;
            println!(
                "[OK]   {}: samples={}, drift={:.3e}, std={:.3e}, datasets_total={}, numeric_checked={}, unsupported={}, non_finite_numeric_datasets={}",
                path.display(),
                quality.sample_count,
                quality.abs_drift_final,
                quality.std_dev,
                report.datasets_total,
                report.numeric_checked,
                report.unsupported_numeric_layouts,
                report.datasets_with_non_finite
            );
        }
        println!(
            "WARP_ACCEPTANCE_GATE: PASS (files={}, rho_drift<= {:.3e}, rho_std<= {:.3e})",
            paths.len(),
            thresholds.max_abs_drift_final,
            thresholds.max_std_dev
        );
        return Ok(());
    }

    #[cfg(not(feature = "hdf5-export"))]
    {
        let _ = paths;
        return Err(std::io::Error::other("cannot run warp acceptance gate without hdf5-export feature")
            .into());
    }
}
