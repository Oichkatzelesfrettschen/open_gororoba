mod warp_runner;
mod warp_telemetry;

#[cfg(feature = "hdf5-export")]
use data_core::hdf5_export::read_simulation_trace_component;
use gororoba_cli::warp_forcing_policy::{apply_warp_forcing_env, load_warp_forcing_profile};
use lbm_3d_cuda::Precision;
use stats_core::helpers::{mean as arithmetic_mean, std_dev};
use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::Ordering;
use std::time::Duration;
use warp_runner::{
    BackendKind, BenchCase, BenchCaseReport, GateProfile, TimingMode, gate_h5_outputs_with_profile,
    print_case_report, run_case, write_step_timing_report,
};

#[derive(Debug, Clone)]
struct TimingSnapshot {
    steps_per_sec: f64,
    mlups: f64,
    p50_us: f64,
    p90_us: f64,
    p99_us: f64,
}

#[derive(Debug, Clone)]
struct CandidateScore {
    block_dim: (u32, u32, u32),
    per_resolution: BTreeMap<usize, ResolutionRepeatStats>,
    geom_mlups_mean: f64,
    geom_mlups_lcb95: f64,
    geom_mlups_ucb95: f64,
}

#[derive(Debug, Clone)]
struct ResolutionRepeatStats {
    repeats: usize,
    mlups_values: Vec<f64>,
    mean_mlups: f64,
    std_mlups: f64,
    ci95_halfwidth: f64,
}

#[derive(Debug, Clone)]
struct RunArtifact {
    h5_path: PathBuf,
    timing_path: PathBuf,
    report: BenchCaseReport,
}

fn usage() -> &'static str {
    "Usage:
  warp-fastpath-suite [sweep_duration_s=45] [prod_duration_s=300] [trace_stride=10] [block_dims_csv=16x4x2,8x8x2,8x4x4,4x4x8,4x4x4] [resolutions_csv=128,256] [baseline_dir=data/h5/production/20260215-211758_precision_suite_rerun] [legacy_dir=data/h5/production/20260215-200556] [out_tag=bf16_fastpath] [forcing_profile=<env/default>] [sweep_repeats=3] [min_improvement_pct=1.0]

Runs:
1) BF16 block-dim sweep (cuda_events) at selected resolutions
2) Picks fastest block dim by geometric-mean MLUPS
3) Reruns BF16 production lane at selected resolutions and duration
4) Emits comparison report TOML in reports/"
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
                format!("invalid {what} token: {trimmed}"),
            )
        })?;
        out.push(value);
    }
    if out.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{what} list cannot be empty"),
        )
        .into());
    }
    Ok(out)
}

fn parse_block_dim_token(token: &str) -> Option<(u32, u32, u32)> {
    let cleaned = token
        .trim()
        .replace(',', "x")
        .replace(' ', "")
        .to_ascii_lowercase();
    let parts: Vec<&str> = cleaned.split('x').filter(|p| !p.is_empty()).collect();
    if parts.len() != 3 {
        return None;
    }
    let bx: u32 = parts[0].parse().ok()?;
    let by: u32 = parts[1].parse().ok()?;
    let bz: u32 = parts[2].parse().ok()?;
    if bx == 0 || by == 0 || bz == 0 {
        return None;
    }
    if (bx as u64) * (by as u64) * (bz as u64) > 1024 {
        return None;
    }
    Some((bx, by, bz))
}

fn block_dim_tag(dim: (u32, u32, u32)) -> String {
    format!("{}x{}x{}", dim.0, dim.1, dim.2)
}

fn geometric_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut log_sum = 0.0;
    let mut n = 0usize;
    for &v in values {
        if v > 0.0 && v.is_finite() {
            log_sum += v.ln();
            n += 1;
        }
    }
    if n == 0 {
        0.0
    } else {
        (log_sum / n as f64).exp()
    }
}

fn ci95_halfwidth(values: &[f64], std: f64) -> f64 {
    if values.len() < 2 {
        0.0
    } else {
        1.96 * std / (values.len() as f64).sqrt()
    }
}

fn pct_delta(new: f64, old: f64) -> f64 {
    if !old.is_finite() || old.abs() <= f64::EPSILON {
        0.0
    } else {
        (new / old - 1.0) * 100.0
    }
}

fn read_cpu_model() -> Option<String> {
    let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("model name\t: ") {
            return Some(rest.trim().to_string());
        }
    }
    None
}

fn read_gpu_name() -> Option<String> {
    let out = Command::new("nvidia-smi")
        .args(["--query-gpu=name", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout)
        .ok()?
        .lines()
        .next()
        .map(|s| s.trim().to_string())
}

fn read_cuda_version() -> Option<String> {
    let out = Command::new("nvidia-smi").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8(out.stdout).ok()?;
    for line in text.lines() {
        if let Some(idx) = line.find("CUDA Version:") {
            let rest = &line[idx + "CUDA Version:".len()..];
            if let Some(v) = rest.split_whitespace().next() {
                return Some(v.to_string());
            }
        }
    }
    None
}

fn parse_toml_f64(
    table: &toml::value::Table,
    key: &str,
    section: &str,
) -> Result<f64, Box<dyn Error>> {
    let value = table.get(key).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("missing key '{key}' in [{section}]"),
        )
    })?;
    if let Some(v) = value.as_float() {
        return Ok(v);
    }
    if let Some(v) = value.as_integer() {
        return Ok(v as f64);
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("key '{key}' in [{section}] is not numeric"),
    )
    .into())
}

fn parse_timing_snapshot(path: &Path) -> Result<TimingSnapshot, Box<dyn Error>> {
    let text = std::fs::read_to_string(path)?;
    let value = text.parse::<toml::Value>()?;
    let case = value
        .get("case")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "missing [case] table")
        })?;
    let timing = value
        .get("timing")
        .and_then(toml::Value::as_table)
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "missing [timing] table")
        })?;

    Ok(TimingSnapshot {
        steps_per_sec: parse_toml_f64(case, "steps_per_sec", "case")?,
        mlups: parse_toml_f64(case, "mlups", "case")?,
        p50_us: parse_toml_f64(timing, "p50_us", "timing")?,
        p90_us: parse_toml_f64(timing, "p90_us", "timing")?,
        p99_us: parse_toml_f64(timing, "p99_us", "timing")?,
    })
}

fn find_timing_file(
    dir: &Path,
    resolution: usize,
    duration_secs: f64,
) -> Result<PathBuf, Box<dyn Error>> {
    let duration_tag = format!("_{}s.toml", duration_secs.round() as u64);
    let exact_gpu = format!(
        "timing_{}_GPU_BF16_{}s.toml",
        resolution,
        duration_secs.round() as u64
    );
    let exact_legacy = format!(
        "timing_{}_BF16_{}s.toml",
        resolution,
        duration_secs.round() as u64
    );
    let mut matches = Vec::new();

    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let name = match path.file_name().and_then(|s| s.to_str()) {
            Some(v) => v,
            None => continue,
        };
        if !name.starts_with("timing_") || !name.ends_with(".toml") {
            continue;
        }
        if !name.contains("BF16") {
            continue;
        }
        if !name.contains(&duration_tag) {
            continue;
        }
        if !name.contains(&format!("{resolution}")) {
            continue;
        }
        matches.push(path);
    }

    if matches.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "no timing TOML found in {} for resolution={} duration={}s",
                dir.display(),
                resolution,
                duration_secs.round() as u64
            ),
        )
        .into());
    }

    matches.sort();
    if let Some(path) = matches
        .iter()
        .find(|p| p.file_name().and_then(|s| s.to_str()) == Some(exact_gpu.as_str()))
    {
        return Ok(path.clone());
    }
    if let Some(path) = matches
        .iter()
        .find(|p| p.file_name().and_then(|s| s.to_str()) == Some(exact_legacy.as_str()))
    {
        return Ok(path.clone());
    }
    Ok(matches[0].clone())
}

fn run_bf16_case(
    resolution: usize,
    duration_secs: f64,
    trace_stride: usize,
    out_h5: PathBuf,
    timing_path: PathBuf,
    forcing_profile_name: Option<&str>,
) -> Result<RunArtifact, Box<dyn Error>> {
    let forcing_profile = load_warp_forcing_profile(forcing_profile_name, resolution)?;
    apply_warp_forcing_env(&forcing_profile.values);
    println!(
        "    [FORCING_PROFILE] name={}, source={}, resolution={}, per_resolution_override={}, mode_y={}, Re_target={:.3}, Ma_max={:.3}, rho0={:.3}, bf16_min_delta_ulp_ratio={:.3}, enforce_mode_floor={}, enforce_mode_floor_all_precisions={}",
        forcing_profile.profile_name,
        forcing_profile.source_path.display(),
        forcing_profile.resolution,
        forcing_profile.used_resolution_override,
        forcing_profile.values.mode_y_requested,
        forcing_profile.values.re_target,
        forcing_profile.values.max_mach,
        forcing_profile.values.rho0,
        forcing_profile.values.bf16_min_delta_ulp_ratio,
        forcing_profile.values.bf16_enforce_mode_floor,
        forcing_profile.values.enforce_mode_floor_all_precisions
    );

    let report = run_case(&BenchCase {
        resolution,
        precision: Precision::BF16,
        backend: BackendKind::Gpu,
        timing_mode: TimingMode::CudaEvents,
        duration_secs,
        trace_stride,
        h5_output: Some(out_h5.clone()),
    })?;
    print_case_report(&report);
    write_step_timing_report(&timing_path, &report)?;
    Ok(RunArtifact {
        h5_path: out_h5,
        timing_path,
        report,
    })
}

fn high_latency_tail_count(report: &BenchCaseReport) -> usize {
    // Trace extraction happens every `trace_stride` steps. In current lanes this is 10,
    // so the periodic overhead often occupies roughly the upper decile of per-step times.
    let threshold = report.step_timing.p90_us;
    report
        .step_timing
        .bins
        .iter()
        .filter(|bin| bin.lower_us >= threshold)
        .map(|bin| bin.count)
        .sum()
}

#[cfg(feature = "hdf5-export")]
fn trace_component_all_zero(path: &Path, component: &str) -> Result<bool, Box<dyn Error>> {
    let data = read_simulation_trace_component(path, component)?;
    Ok(data.iter().all(|v| v.abs() <= 1.0e-30))
}

#[cfg(not(feature = "hdf5-export"))]
fn trace_component_all_zero(_path: &Path, _component: &str) -> Result<bool, Box<dyn Error>> {
    Ok(false)
}

fn unique_report_path() -> PathBuf {
    let date = chrono::Utc::now().format("%Y_%m_%d").to_string();
    let stamp = chrono::Utc::now().format("%H%M%S").to_string();
    PathBuf::from(format!(
        "reports/warp_bf16_fastpath_comparison_report_{}_{}.toml",
        date, stamp
    ))
}

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();
    let sweep_duration_secs: f64 = args.get(1).map_or(Ok(45.0), |s| s.parse())?;
    let prod_duration_secs: f64 = args.get(2).map_or(Ok(300.0), |s| s.parse())?;
    let trace_stride: usize = args.get(3).map_or(Ok(10usize), |s| s.parse())?;
    let block_dims_csv = args
        .get(4)
        .map(String::as_str)
        .unwrap_or("16x4x2,8x8x2,8x4x4,4x4x8,4x4x4");
    let resolutions_csv = args.get(5).map(String::as_str).unwrap_or("128,256");
    let baseline_dir = args.get(6).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("data/h5/production/20260215-211758_precision_suite_rerun")
    });
    let legacy_dir = args
        .get(7)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("data/h5/production/20260215-200556"));
    let out_tag = args.get(8).map(String::as_str).unwrap_or("bf16_fastpath");
    let forcing_profile_name = args.get(9).map(String::as_str);
    let sweep_repeats: usize = args.get(10).map_or(Ok(3usize), |s| s.parse())?;
    let min_improvement_pct: f64 = args.get(11).map_or(Ok(1.0f64), |s| s.parse())?;

    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{}", usage());
        return Ok(());
    }
    if sweep_duration_secs <= 0.0 || prod_duration_secs <= 0.0 {
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
    if sweep_repeats == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "sweep_repeats must be >= 1",
        )
        .into());
    }
    if !min_improvement_pct.is_finite() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "min_improvement_pct must be finite",
        )
        .into());
    }

    let block_dims = parse_csv(block_dims_csv, parse_block_dim_token, "block_dim")?;
    let resolutions = parse_csv(resolutions_csv, |s| s.parse::<usize>().ok(), "resolution")?;
    if resolutions.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "resolutions list cannot be empty",
        )
        .into());
    }

    let timestamp = chrono::Utc::now().format("%Y%m%d-%H%M%S").to_string();
    let sweep_root = PathBuf::from(format!(
        "data/h5/basemarks/{}_bf16_blockdim_sweep",
        timestamp
    ));
    std::fs::create_dir_all(&sweep_root)?;

    println!("=== Warp BF16 Fastpath Suite ===");
    println!(
        "sweep_duration_s={:.2}, prod_duration_s={:.2}, trace_stride={}, block_dims={}, resolutions={}, sweep_repeats={}, min_improvement_pct={:.3}",
        sweep_duration_secs,
        prod_duration_secs,
        trace_stride,
        block_dims_csv,
        resolutions_csv,
        sweep_repeats,
        min_improvement_pct
    );
    println!("baseline_dir={}", baseline_dir.display());
    println!("legacy_dir={}", legacy_dir.display());
    println!("sweep_root={}", sweep_root.display());
    println!(
        "forcing_profile_selector: cli_arg={:?}, env_GOROROBA_WARP_FORCING_PROFILE={}",
        forcing_profile_name,
        std::env::var("GOROROBA_WARP_FORCING_PROFILE").unwrap_or_else(|_| "<unset>".to_string())
    );

    let mut candidate_scores = Vec::new();

    for &block_dim in &block_dims {
        let dim_tag = block_dim_tag(block_dim);
        // SAFETY: Called single-threaded in binary main before spawning work.
        unsafe {
            std::env::set_var("GOROROBA_LBM_BLOCK_DIM", &dim_tag);
        }
        let candidate_dir = sweep_root.join(format!("blockdim_{}", dim_tag.replace('x', "_")));
        std::fs::create_dir_all(&candidate_dir)?;
        println!();
        println!("[SWEEP] block_dim={}", dim_tag);

        let mut artifacts = Vec::new();
        let mut per_resolution = BTreeMap::new();
        let mut geom_input_mean = Vec::new();
        let mut geom_input_lcb = Vec::new();
        let mut geom_input_ucb = Vec::new();

        for &res in &resolutions {
            let mut mlups_values = Vec::new();
            for repeat_idx in 0..sweep_repeats {
                let out_h5 = candidate_dir.join(format!(
                    "warp_ring_{}_GPU_BF16_{}s_r{}.h5",
                    res,
                    sweep_duration_secs.round() as u64,
                    repeat_idx + 1
                ));
                let timing_path = candidate_dir.join(format!(
                    "timing_{}_GPU_BF16_{}s_r{}.toml",
                    res,
                    sweep_duration_secs.round() as u64,
                    repeat_idx + 1
                ));
                println!(
                    "  [RUN] res={} repeat={}/{} out={} timing={}",
                    res,
                    repeat_idx + 1,
                    sweep_repeats,
                    out_h5.display(),
                    timing_path.display()
                );
                let artifact = run_bf16_case(
                    res,
                    sweep_duration_secs,
                    trace_stride,
                    out_h5.clone(),
                    timing_path,
                    forcing_profile_name,
                )?;
                artifacts.push(out_h5);
                mlups_values.push(artifact.report.mlups);
            }
            let mean_mlups = arithmetic_mean(&mlups_values);
            let std_mlups = std_dev(&mlups_values, mean_mlups);
            let ci95 = ci95_halfwidth(&mlups_values, std_mlups);
            println!(
                "  [STATS] res={} mean_mlups={:.6} std_mlups={:.6} ci95_halfwidth={:.6}",
                res, mean_mlups, std_mlups, ci95
            );
            per_resolution.insert(
                res,
                ResolutionRepeatStats {
                    repeats: sweep_repeats,
                    mlups_values: mlups_values.clone(),
                    mean_mlups,
                    std_mlups,
                    ci95_halfwidth: ci95,
                },
            );
            geom_input_mean.push(mean_mlups);
            geom_input_lcb.push((mean_mlups - ci95).max(1.0e-12));
            geom_input_ucb.push((mean_mlups + ci95).max(1.0e-12));
        }

        gate_h5_outputs_with_profile(&artifacts, GateProfile::Canonical300s)?;
        let geom_mlups_mean = geometric_mean(&geom_input_mean);
        let geom_mlups_lcb95 = geometric_mean(&geom_input_lcb);
        let geom_mlups_ucb95 = geometric_mean(&geom_input_ucb);
        println!(
            "  [SCORE] block_dim={} geom_mlups_mean={:.6} geom_mlups_lcb95={:.6} geom_mlups_ucb95={:.6}",
            dim_tag, geom_mlups_mean, geom_mlups_lcb95, geom_mlups_ucb95
        );
        candidate_scores.push(CandidateScore {
            block_dim,
            per_resolution,
            geom_mlups_mean,
            geom_mlups_lcb95,
            geom_mlups_ucb95,
        });
    }

    candidate_scores.sort_by(|a, b| {
        b.geom_mlups_lcb95
            .total_cmp(&a.geom_mlups_lcb95)
            .then_with(|| b.geom_mlups_mean.total_cmp(&a.geom_mlups_mean))
    });
    let winner = candidate_scores
        .first()
        .ok_or_else(|| std::io::Error::other("no candidate scores computed for block-dim sweep"))?;
    let winning_dim_tag = block_dim_tag(winner.block_dim);
    let baseline_block_dim = std::env::var("GOROROBA_FASTPATH_BASELINE_BLOCK_DIM")
        .ok()
        .and_then(|v| parse_block_dim_token(&v))
        .unwrap_or((4, 4, 4));
    let baseline_score = candidate_scores
        .iter()
        .find(|c| c.block_dim == baseline_block_dim);
    if let Some(base) = baseline_score {
        let required_lcb = base.geom_mlups_ucb95 * (1.0 + min_improvement_pct / 100.0);
        if winner.geom_mlups_lcb95 < required_lcb {
            return Err(std::io::Error::other(format!(
                "winner {} is not statistically above baseline {} with min_improvement_pct={:.3}: winner_lcb95={:.6} < required={:.6} (baseline_ucb95={:.6})",
                winning_dim_tag,
                block_dim_tag(baseline_block_dim),
                min_improvement_pct,
                winner.geom_mlups_lcb95,
                required_lcb,
                base.geom_mlups_ucb95
            ))
            .into());
        }
    } else {
        eprintln!(
            "[WARN] baseline block dim {} not present in sweep candidate list; skipping statistical superiority check",
            block_dim_tag(baseline_block_dim)
        );
    }
    println!();
    println!("[WINNER] GOROROBA_LBM_BLOCK_DIM={}", winning_dim_tag);

    let mut sweep_score_text = String::new();
    sweep_score_text.push_str("# BF16 block-dim sweep scores (sorted desc by geom_mlups)\n");
    sweep_score_text.push_str(&format!(
        "# generated_at_utc={}\n\n",
        chrono::Utc::now().to_rfc3339()
    ));
    for score in &candidate_scores {
        let tag = block_dim_tag(score.block_dim);
        sweep_score_text.push_str(&format!(
            "block_dim={} geom_mlups_mean={:.6} geom_mlups_lcb95={:.6} geom_mlups_ucb95={:.6}",
            tag, score.geom_mlups_mean, score.geom_mlups_lcb95, score.geom_mlups_ucb95
        ));
        for (res, stats) in &score.per_resolution {
            sweep_score_text.push_str(&format!(
                " mlups_mean_{}={:.6} mlups_std_{}={:.6} mlups_ci95_halfwidth_{}={:.6}",
                res, stats.mean_mlups, res, stats.std_mlups, res, stats.ci95_halfwidth
            ));
        }
        sweep_score_text.push('\n');
    }
    let sweep_scores_path = sweep_root.join("blockdim_sweep_scores.txt");
    std::fs::write(&sweep_scores_path, sweep_score_text)?;
    println!("SWEEP_SCORES: {}", sweep_scores_path.display());

    // SAFETY: Called single-threaded in binary main before spawning work.
    unsafe {
        std::env::set_var("GOROROBA_LBM_BLOCK_DIM", &winning_dim_tag);
    }
    let production_dir = PathBuf::from(format!(
        "data/h5/production/{}_{}_{}",
        timestamp,
        out_tag,
        winning_dim_tag.replace('x', "_")
    ));
    std::fs::create_dir_all(&production_dir)?;

    println!();
    println!(
        "[PRODUCTION] out_dir={} (block_dim={})",
        production_dir.display(),
        winning_dim_tag
    );

    let telemetry_path = production_dir.join("gpu_telemetry.csv");
    let telemetry_handle =
        warp_telemetry::spawn_gpu_telemetry_sampler(telemetry_path.clone(), Duration::from_secs(1));

    let mut production_artifacts = Vec::new();
    let mut production_records: BTreeMap<usize, RunArtifact> = BTreeMap::new();
    let prod_result = (|| -> Result<(), Box<dyn Error>> {
        for &res in &resolutions {
            let out_h5 = production_dir.join(format!(
                "warp_ring_{}_GPU_BF16_{}s.h5",
                res,
                prod_duration_secs.round() as u64
            ));
            let timing_path = production_dir.join(format!(
                "timing_{}_GPU_BF16_{}s.toml",
                res,
                prod_duration_secs.round() as u64
            ));
            println!(
                "  [RUN] res={} out={} timing={}",
                res,
                out_h5.display(),
                timing_path.display()
            );
            let artifact = run_bf16_case(
                res,
                prod_duration_secs,
                trace_stride,
                out_h5.clone(),
                timing_path,
                forcing_profile_name,
            )?;
            production_artifacts.push(out_h5);
            production_records.insert(res, artifact);
        }
        gate_h5_outputs_with_profile(&production_artifacts, GateProfile::Canonical300sMeasured)?;
        Ok(())
    })();

    if let Some((stop, handle)) = telemetry_handle {
        stop.store(true, Ordering::Relaxed);
        if let Err(err) = handle.join() {
            eprintln!("[WARN] telemetry thread join failed: {:?}", err);
        } else {
            println!("GPU_TELEMETRY: {}", telemetry_path.display());
        }
    }
    prod_result?;

    let mut baseline_by_res = BTreeMap::new();
    let mut legacy_by_res = BTreeMap::new();
    for &res in &resolutions {
        let baseline_timing = find_timing_file(&baseline_dir, res, prod_duration_secs)?;
        let baseline_snapshot = parse_timing_snapshot(&baseline_timing)?;
        baseline_by_res.insert(res, baseline_snapshot);

        if let Ok(path) = find_timing_file(&legacy_dir, res, prod_duration_secs)
            && let Ok(snapshot) = parse_timing_snapshot(&path)
        {
            legacy_by_res.insert(res, snapshot);
        }
    }

    let mut aggregate_untuned_mlups = Vec::new();
    let mut aggregate_tuned_mlups = Vec::new();
    let mut comparison_rows = Vec::new();
    for &res in &resolutions {
        let baseline = baseline_by_res.get(&res).ok_or_else(|| {
            std::io::Error::other(format!("missing baseline snapshot for resolution {}", res))
        })?;
        let tuned = production_records.get(&res).ok_or_else(|| {
            std::io::Error::other(format!("missing production record for resolution {}", res))
        })?;
        aggregate_untuned_mlups.push(baseline.mlups);
        aggregate_tuned_mlups.push(tuned.report.mlups);
        comparison_rows.push((
            res,
            baseline.clone(),
            tuned.report.steps_per_sec,
            tuned.report.mlups,
            tuned.report.step_timing.p50_us,
            tuned.report.step_timing.p90_us,
            tuned.report.step_timing.p99_us,
        ));
    }
    comparison_rows.sort_by_key(|(res, _, _, _, _, _, _)| Reverse(*res));

    let mut sorted_res = resolutions.clone();
    sorted_res.sort_unstable();
    let scaling_exp_untuned = if sorted_res.len() >= 2 {
        let lo = sorted_res[0];
        let hi = sorted_res[sorted_res.len() - 1];
        let lo_sps = baseline_by_res
            .get(&lo)
            .map(|s| s.steps_per_sec)
            .unwrap_or(0.0);
        let hi_sps = baseline_by_res
            .get(&hi)
            .map(|s| s.steps_per_sec)
            .unwrap_or(0.0);
        if lo_sps > 0.0 && hi_sps > 0.0 && hi > lo {
            (lo_sps / hi_sps).ln() / (hi as f64 / lo as f64).ln()
        } else {
            0.0
        }
    } else {
        0.0
    };
    let scaling_exp_tuned = if sorted_res.len() >= 2 {
        let lo = sorted_res[0];
        let hi = sorted_res[sorted_res.len() - 1];
        let lo_sps = production_records
            .get(&lo)
            .map(|r| r.report.steps_per_sec)
            .unwrap_or(0.0);
        let hi_sps = production_records
            .get(&hi)
            .map(|r| r.report.steps_per_sec)
            .unwrap_or(0.0);
        if lo_sps > 0.0 && hi_sps > 0.0 && hi > lo {
            (lo_sps / hi_sps).ln() / (hi as f64 / lo as f64).ln()
        } else {
            0.0
        }
    } else {
        0.0
    };

    let geom_untuned = geometric_mean(&aggregate_untuned_mlups);
    let geom_tuned = geometric_mean(&aggregate_tuned_mlups);

    let timing_regime_signature = if legacy_by_res.len() >= 2 && production_records.len() >= 2 {
        let l128 = legacy_by_res.get(&128).map(|s| s.p50_us).unwrap_or(0.0);
        let l256 = legacy_by_res.get(&256).map(|s| s.p50_us).unwrap_or(0.0);
        let t128 = production_records
            .get(&128)
            .map(|r| r.report.step_timing.p50_us)
            .unwrap_or(0.0);
        let t256 = production_records
            .get(&256)
            .map(|r| r.report.step_timing.p50_us)
            .unwrap_or(0.0);
        format!(
            "Legacy {} appears launch-like (p50 {:.3}us/{:.3}us); tuned run {} is kernel-like (p50 {:.3}us/{:.3}us).",
            legacy_dir.display(),
            l128,
            l256,
            production_dir.display(),
            t128,
            t256
        )
    } else {
        "Legacy vs tuned timing regime split unavailable (legacy timing snapshots missing)."
            .to_string()
    };

    let mut tail_lines = Vec::new();
    for (&res, artifact) in &production_records {
        let tail_count = high_latency_tail_count(&artifact.report);
        let expected_trace_events = artifact.report.steps / trace_stride;
        let ratio = if expected_trace_events > 0 {
            tail_count as f64 / expected_trace_events as f64
        } else {
            0.0
        };
        tail_lines.push(format!(
            "{}^3: tail_count={} expected_trace_events={} ratio={:.3}",
            res, tail_count, expected_trace_events, ratio
        ));
    }
    let periodic_tail_signature = format!(
        "High-latency tail vs trace cadence (trace_stride={}): {}",
        trace_stride,
        tail_lines.join("; ")
    );

    let mut rho_ok = true;
    let mut rho_ranges = Vec::new();
    for (&res, artifact) in &production_records {
        let q = &artifact.report.quality;
        if (q.final_value - 1.0).abs() > 1.0e-12 || q.abs_drift_final > 0.0 || q.std_dev > 0.0 {
            rho_ok = false;
        }
        rho_ranges.push(format!(
            "{}^3: rho_final={:.12}, drift={:.3e}, std={:.3e}",
            res, q.final_value, q.abs_drift_final, q.std_dev
        ));
    }
    let invariant_signature = if rho_ok {
        format!(
            "rho_mean remained exactly 1.0 with zero drift/std across tuned production runs ({})",
            rho_ranges.join("; ")
        )
    } else {
        format!("rho_mean invariants observed: {}", rho_ranges.join("; "))
    };

    let mut enstrophy_zero_all = true;
    let mut algebra_zero_all = true;
    let mut quantization_lines = Vec::new();
    for (&res, artifact) in &production_records {
        if !trace_component_all_zero(&artifact.h5_path, "enstrophy")? {
            enstrophy_zero_all = false;
        }
        if !trace_component_all_zero(&artifact.h5_path, "algebra_norm")? {
            algebra_zero_all = false;
        }
        quantization_lines.push(format!(
            "{}^3 delta_f_over_ulp={:.3e}",
            res, artifact.report.forcing.bf16_delta_to_ulp_ratio
        ));
    }
    let trace_content_signature = if enstrophy_zero_all && algebra_zero_all {
        format!(
            "enstrophy and algebra_norm traces are zero-filled in current exports; likely BF16 forcing quantization-limited ({})",
            quantization_lines.join(", ")
        )
    } else if enstrophy_zero_all {
        format!(
            "enstrophy traces are zero-filled but algebra_norm has non-zero structure ({}).",
            quantization_lines.join(", ")
        )
    } else if algebra_zero_all {
        format!(
            "algebra_norm traces are zero-filled but enstrophy has non-zero structure ({}).",
            quantization_lines.join(", ")
        )
    } else {
        format!(
            "enstrophy and algebra_norm traces both contain non-zero structure ({}).",
            quantization_lines.join(", ")
        )
    };

    let report_path = unique_report_path();
    if let Some(parent) = report_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut out = String::new();
    out.push_str("schema = \"open_gororoba.warp_bf16_fastpath_comparison.v1\"\n");
    out.push_str(&format!(
        "generated_at_utc = \"{}\"\n",
        chrono::Utc::now().to_rfc3339()
    ));
    out.push_str("status = \"pass\"\n\n");

    out.push_str("[system]\n");
    out.push_str(&format!(
        "cpu = \"{}\"\n",
        read_cpu_model().unwrap_or_else(|| "unknown".to_string())
    ));
    out.push_str(&format!(
        "gpu = \"{}\"\n",
        read_gpu_name().unwrap_or_else(|| "unknown".to_string())
    ));
    out.push_str(&format!(
        "cuda = \"{}\"\n\n",
        read_cuda_version().unwrap_or_else(|| "unknown".to_string())
    ));

    out.push_str("[inputs]\n");
    out.push_str(&format!("baseline_dir = \"{}\"\n", baseline_dir.display()));
    out.push_str(&format!("legacy_dir = \"{}\"\n", legacy_dir.display()));
    out.push_str(&format!("sweep_root = \"{}\"\n", sweep_root.display()));
    out.push_str(&format!(
        "sweep_scores = \"{}\"\n",
        sweep_scores_path.display()
    ));
    out.push_str(&format!(
        "production_dir = \"{}\"\n\n",
        production_dir.display()
    ));

    out.push_str("[sweep]\n");
    out.push_str("method = \"BF16 matrix sweep at selected resolutions with cuda_events; confidence-aware winner by geometric mean LCB95 across per-resolution repeats\"\n");
    out.push_str(&format!("winning_block_dim = \"{}\"\n", winning_dim_tag));
    out.push_str(&format!("sweep_repeats = {}\n", sweep_repeats));
    out.push_str(&format!(
        "baseline_block_dim = \"{}\"\n",
        block_dim_tag(baseline_block_dim)
    ));
    out.push_str(&format!(
        "min_improvement_pct = {:.6}\n",
        min_improvement_pct
    ));
    out.push_str("env_key = \"GOROROBA_LBM_BLOCK_DIM\"\n\n");
    for score in &candidate_scores {
        out.push_str("[[sweep.candidate]]\n");
        out.push_str(&format!(
            "block_dim = \"{}\"\n",
            block_dim_tag(score.block_dim)
        ));
        for (res, stats) in &score.per_resolution {
            out.push_str(&format!("repeats_{} = {}\n", res, stats.repeats));
            out.push_str(&format!("mlups_mean_{} = {:.6}\n", res, stats.mean_mlups));
            out.push_str(&format!("mlups_std_{} = {:.6}\n", res, stats.std_mlups));
            out.push_str(&format!(
                "mlups_ci95_halfwidth_{} = {:.6}\n",
                res, stats.ci95_halfwidth
            ));
            for (idx, value) in stats.mlups_values.iter().enumerate() {
                out.push_str(&format!("mlups_{}_r{} = {:.6}\n", res, idx + 1, value));
            }
        }
        out.push_str(&format!("geom_mlups_mean = {:.6}\n", score.geom_mlups_mean));
        out.push_str(&format!(
            "geom_mlups_lcb95 = {:.6}\n",
            score.geom_mlups_lcb95
        ));
        out.push_str(&format!(
            "geom_mlups_ucb95 = {:.6}\n\n",
            score.geom_mlups_ucb95
        ));
    }

    out.push_str("[comparison]\n");
    out.push_str(&format!(
        "untuned_run_dir = \"{}\"\n",
        baseline_dir.display()
    ));
    out.push_str(&format!(
        "tuned_run_dir = \"{}\"\n\n",
        production_dir.display()
    ));
    for (res, baseline, tuned_sps, tuned_mlups, tuned_p50, tuned_p90, tuned_p99) in &comparison_rows
    {
        out.push_str("[[comparison.case]]\n");
        out.push_str(&format!("resolution = {}\n", res));
        out.push_str(&format!(
            "untuned_steps_per_sec = {:.6}\n",
            baseline.steps_per_sec
        ));
        out.push_str(&format!("untuned_mlups = {:.6}\n", baseline.mlups));
        out.push_str(&format!("untuned_p50_us = {:.6}\n", baseline.p50_us));
        out.push_str(&format!("untuned_p90_us = {:.6}\n", baseline.p90_us));
        out.push_str(&format!("untuned_p99_us = {:.6}\n", baseline.p99_us));
        out.push_str(&format!("tuned_steps_per_sec = {:.6}\n", tuned_sps));
        out.push_str(&format!("tuned_mlups = {:.6}\n", tuned_mlups));
        out.push_str(&format!("tuned_p50_us = {:.6}\n", tuned_p50));
        out.push_str(&format!("tuned_p90_us = {:.6}\n", tuned_p90));
        out.push_str(&format!("tuned_p99_us = {:.6}\n", tuned_p99));
        out.push_str(&format!(
            "delta_steps_per_sec_pct = {:.4}\n",
            pct_delta(*tuned_sps, baseline.steps_per_sec)
        ));
        out.push_str(&format!(
            "delta_mlups_pct = {:.4}\n",
            pct_delta(*tuned_mlups, baseline.mlups)
        ));
        out.push_str(&format!(
            "delta_p50_us_pct = {:.4}\n",
            pct_delta(*tuned_p50, baseline.p50_us)
        ));
        out.push_str(&format!(
            "delta_p90_us_pct = {:.4}\n",
            pct_delta(*tuned_p90, baseline.p90_us)
        ));
        out.push_str(&format!(
            "delta_p99_us_pct = {:.4}\n\n",
            pct_delta(*tuned_p99, baseline.p99_us)
        ));
    }

    out.push_str("[comparison.aggregate]\n");
    out.push_str(&format!("geom_mlups_untuned = {:.6}\n", geom_untuned));
    out.push_str(&format!("geom_mlups_tuned = {:.6}\n", geom_tuned));
    out.push_str(&format!(
        "delta_geom_mlups_pct = {:.4}\n",
        pct_delta(geom_tuned, geom_untuned)
    ));
    out.push_str(&format!(
        "scaling_exponent_steps_per_sec_untuned = {:.4}\n",
        scaling_exp_untuned
    ));
    out.push_str(&format!(
        "scaling_exponent_steps_per_sec_tuned = {:.4}\n\n",
        scaling_exp_tuned
    ));

    out.push_str("[analysis_signatures]\n");
    out.push_str(&format!(
        "timing_regime_signature = \"{}\"\n",
        timing_regime_signature.replace('"', "'")
    ));
    out.push_str(&format!(
        "trace_periodic_tail_signature = \"{}\"\n",
        periodic_tail_signature.replace('"', "'")
    ));
    out.push_str(&format!(
        "invariant_signature = \"{}\"\n",
        invariant_signature.replace('"', "'")
    ));
    out.push_str(&format!(
        "trace_content_signature = \"{}\"\n\n",
        trace_content_signature.replace('"', "'")
    ));

    out.push_str("[artifacts]\n");
    out.push_str(&format!(
        "telemetry_csv = \"{}\"\n",
        telemetry_path.display()
    ));
    for (&res, artifact) in &production_records {
        out.push_str(&format!(
            "run_tuned_{}_h5 = \"{}\"\n",
            res,
            artifact.h5_path.display()
        ));
        out.push_str(&format!(
            "run_tuned_{}_timing = \"{}\"\n",
            res,
            artifact.timing_path.display()
        ));
    }
    out.push('\n');

    out.push_str("[gate]\n");
    out.push_str("result = \"pass\"\n");
    out.push_str("notes = \"warp acceptance gate passed for sweep candidates and tuned production outputs\"\n");

    std::fs::write(&report_path, out)?;
    println!();
    println!("FASTPATH_REPORT: {}", report_path.display());
    println!("DONE: fastpath suite completed and report emitted");
    Ok(())
}
