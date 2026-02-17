#![cfg_attr(not(feature = "hdf5-export"), allow(dead_code, unused_imports))]

#[cfg(feature = "hdf5-export")]
use data_core::hdf5_export::{read_simulation_spectral_component, read_simulation_trace_component};
use data_core::quality::{validate_rho_trace, RhoQualityThresholds, RhoTraceQuality};
#[cfg(feature = "hdf5-export")]
use gororoba_cli::warp_gate_policy::load_warp_gate_policy;
#[cfg(feature = "hdf5-export")]
use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use stats_core::helpers::{mean, std_dev};

#[derive(Debug, Clone)]
struct CliArgs {
    production_dir: PathBuf,
    baseline_dir: Option<PathBuf>,
    legacy_dir: Option<PathBuf>,
    trace_stride: usize,
    resolutions: Vec<usize>,
    report_out: Option<PathBuf>,
}

#[derive(Debug, Clone)]
struct HistogramBin {
    lower_us: f64,
    count: usize,
}

#[derive(Debug, Clone)]
struct TimingSnapshot {
    steps: usize,
    steps_per_sec: f64,
    mlups: f64,
    p50_us: f64,
    p90_us: f64,
    p99_us: f64,
    histogram: Vec<HistogramBin>,
}

#[derive(Debug, Clone)]
struct ScalarSummary {
    mean: f64,
    std_dev: f64,
    nonzero_fraction: f64,
}

#[derive(Debug, Clone)]
struct FidelitySummary {
    enstrophy_measured_present: bool,
    enstrophy_nonzero_fraction: f64,
    enstrophy_measured_nonzero_fraction: Option<f64>,
    enstrophy_fallback_fraction: Option<f64>,
    enstrophy_both_zero_fraction: Option<f64>,
    algebra_nonzero_fraction: f64,
    u_rms_nonzero_fraction: f64,
    u_rms_model_present: bool,
    u_rms_model_lock_fraction: Option<f64>,
    spectral_total_power_present: bool,
    spectral_total_power_nonzero_fraction: Option<f64>,
    spectral_total_power_std_dev: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
enum TimingRegime {
    LaunchLike,
    KernelLike,
    Transitional,
}

impl TimingRegime {
    fn as_str(self) -> &'static str {
        match self {
            Self::LaunchLike => "launch_like",
            Self::KernelLike => "kernel_like",
            Self::Transitional => "transitional",
        }
    }
}

#[derive(Debug, Clone)]
struct TimingRegimeSummary {
    regime: TimingRegime,
    mean_step_us: f64,
    p90_to_p50: f64,
    p99_to_p50: f64,
    rationale: String,
}

#[derive(Debug, Clone)]
struct OverheadDecomposition {
    expected_trace_events: usize,
    observed_total_us: f64,
    base_step_total_us: f64,
    estimated_reduction_overhead_us: f64,
    estimated_dtoh_overhead_us: f64,
    estimated_total_us: f64,
    closure_error_pct: f64,
}

#[derive(Debug, Clone)]
struct SpectralFeatures {
    sample_count: usize,
    dt_seconds: f64,
    dominant_frequency_hz: f64,
    dominant_period_s: f64,
    dominant_power: f64,
    total_power: f64,
    harmonic2_ratio: f64,
    loglog_slope: f64,
}

#[derive(Debug, Clone)]
struct MomentPack {
    mean: f64,
    std_dev: f64,
    skewness: f64,
    kurtosis_excess: f64,
    lag1_autocorr: f64,
    shannon_entropy_bits: f64,
}

#[derive(Debug, Clone)]
struct InvariantPack {
    rho: MomentPack,
    enstrophy: MomentPack,
    algebra_norm: MomentPack,
    u_rms: MomentPack,
    corr_rho_enstrophy: f64,
    corr_rho_u_rms: f64,
    corr_enstrophy_u_rms: f64,
    lag1_corr_enstrophy_u_rms: f64,
}

#[derive(Debug, Clone)]
struct CaseAnalysis {
    resolution: usize,
    h5_path: PathBuf,
    timing_path: PathBuf,
    timing: TimingSnapshot,
    tail_count: usize,
    tail_to_trace_ratio: f64,
    rho_quality: RhoTraceQuality,
    rho_summary: ScalarSummary,
    fidelity: FidelitySummary,
    timing_regime: TimingRegimeSummary,
    overhead: OverheadDecomposition,
    spectral_rho: SpectralFeatures,
    spectral_enstrophy: SpectralFeatures,
    spectral_u_rms: SpectralFeatures,
    invariants: InvariantPack,
}

fn usage() -> &'static str {
    "Usage:
  warp-fastpath-analyze <production_dir> [baseline_dir=data/h5/production/20260215-211758_precision_suite_rerun] [legacy_dir=data/h5/production/20260215-200556] [trace_stride=10] [resolutions_csv=128,256] [report_out]

Analyzes existing BF16 fastpath production outputs and emits a TOML report with:
- timing regime split (legacy launch-like vs event-timed kernel-like)
- periodic tail fingerprint vs trace cadence
- measured-vs-model trace fidelity ratios
- spectral signal richness diagnostics
"
}

fn parse_csv_usize(input: &str, field: &str) -> Result<Vec<usize>, Box<dyn Error>> {
    let mut out = Vec::new();
    for token in input.split(',') {
        let t = token.trim();
        if t.is_empty() {
            continue;
        }
        out.push(t.parse::<usize>().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("invalid {field} token '{t}': {e}"),
            )
        })?);
    }
    if out.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("{field} cannot be empty"),
        )
        .into());
    }
    Ok(out)
}

fn parse_args() -> Result<CliArgs, Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{}", usage());
        std::process::exit(0);
    }
    let production_dir = args.get(1).map(PathBuf::from).ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "missing <production_dir>")
    })?;
    let baseline_dir = args.get(2).map(PathBuf::from).or_else(|| {
        Some(PathBuf::from(
            "data/h5/production/20260215-211758_precision_suite_rerun",
        ))
    });
    let legacy_dir = args
        .get(3)
        .map(PathBuf::from)
        .or_else(|| Some(PathBuf::from("data/h5/production/20260215-200556")));
    let trace_stride = args.get(4).map_or(Ok(10usize), |s| s.parse())?;
    if trace_stride == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "trace_stride must be >= 1",
        )
        .into());
    }
    let resolutions = parse_csv_usize(
        args.get(5).map(String::as_str).unwrap_or("128,256"),
        "resolutions_csv",
    )?;
    let report_out = args.get(6).map(PathBuf::from);
    Ok(CliArgs {
        production_dir,
        baseline_dir,
        legacy_dir,
        trace_stride,
        resolutions,
        report_out,
    })
}

fn parse_toml_number(
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

fn parse_toml_usize(
    table: &toml::value::Table,
    key: &str,
    section: &str,
) -> Result<usize, Box<dyn Error>> {
    let value = table.get(key).ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("missing key '{key}' in [{section}]"),
        )
    })?;
    if let Some(v) = value.as_integer() {
        return Ok(v.max(0) as usize);
    }
    Err(std::io::Error::new(
        std::io::ErrorKind::InvalidData,
        format!("key '{key}' in [{section}] is not integer"),
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
    let mut histogram = Vec::new();
    if let Some(entries) = timing.get("histogram_bin").and_then(toml::Value::as_array) {
        for entry in entries {
            let table = entry.as_table().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "timing.histogram_bin entry must be a table",
                )
            })?;
            let lower_us = parse_toml_number(table, "lower_us", "timing.histogram_bin")?;
            let count = parse_toml_usize(table, "count", "timing.histogram_bin")?;
            histogram.push(HistogramBin { lower_us, count });
        }
    }
    Ok(TimingSnapshot {
        steps: parse_toml_usize(case, "steps", "case")?,
        steps_per_sec: parse_toml_number(case, "steps_per_sec", "case")?,
        mlups: parse_toml_number(case, "mlups", "case")?,
        p50_us: parse_toml_number(timing, "p50_us", "timing")?,
        p90_us: parse_toml_number(timing, "p90_us", "timing")?,
        p99_us: parse_toml_number(timing, "p99_us", "timing")?,
        histogram,
    })
}

fn parse_duration_from_filename(name: &str) -> u64 {
    for token in name.split('_').rev() {
        let digits: String = token.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !digits.is_empty() && token.contains('s')
            && let Ok(value) = digits.parse::<u64>() {
                return value;
            }
    }
    0
}

fn find_h5_file(dir: &Path, resolution: usize) -> Result<PathBuf, Box<dyn Error>> {
    let mut candidates: Vec<(u64, PathBuf)> = Vec::new();
    let prefixes = [
        format!("warp_ring_{}_GPU_BF16_", resolution),
        format!("warp_ring_{}_BF16_", resolution),
    ];
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if name.ends_with(".h5") && prefixes.iter().any(|prefix| name.starts_with(prefix)) {
            candidates.push((parse_duration_from_filename(name), path));
        }
    }
    if candidates.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "no H5 artifact found in {} for resolution {}",
                dir.display(),
                resolution
            ),
        )
        .into());
    }
    candidates.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    Ok(candidates[candidates.len() - 1].1.clone())
}

fn find_timing_file(dir: &Path, resolution: usize) -> Result<PathBuf, Box<dyn Error>> {
    let mut candidates: Vec<(u64, PathBuf)> = Vec::new();
    let prefixes = [
        format!("timing_{}_GPU_BF16_", resolution),
        format!("timing_{}_BF16_", resolution),
    ];
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        if !name.ends_with(".toml") {
            continue;
        }
        if prefixes.iter().any(|prefix| name.starts_with(prefix)) {
            candidates.push((parse_duration_from_filename(name), path));
        }
    }
    if candidates.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "no timing TOML found in {} for resolution {}",
                dir.display(),
                resolution
            ),
        )
        .into());
    }
    candidates.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    Ok(candidates[candidates.len() - 1].1.clone())
}

fn lag1_autocorr(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let x = &values[..values.len() - 1];
    let y = &values[1..];
    pearson_corr(x, y)
}

fn shannon_entropy_bits(values: &[f64], bins: usize) -> f64 {
    if values.is_empty() || bins == 0 {
        return 0.0;
    }
    let min = values.iter().copied().fold(f64::INFINITY, f64::min);
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !min.is_finite() || !max.is_finite() || (max - min).abs() <= f64::EPSILON {
        return 0.0;
    }
    let width = (max - min) / bins as f64;
    let mut counts = vec![0usize; bins];
    for &v in values {
        let mut idx = ((v - min) / width).floor() as usize;
        if idx >= bins {
            idx = bins - 1;
        }
        counts[idx] += 1;
    }
    let n = values.len() as f64;
    let mut h = 0.0f64;
    for c in counts {
        if c == 0 {
            continue;
        }
        let p = c as f64 / n;
        h -= p * p.log2();
    }
    h
}

fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }
    let mx = mean(&x[..n]);
    let my = mean(&y[..n]);
    let mut cov = 0.0f64;
    let mut vx = 0.0f64;
    let mut vy = 0.0f64;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx <= f64::EPSILON || vy <= f64::EPSILON {
        0.0
    } else {
        cov / (vx.sqrt() * vy.sqrt())
    }
}

fn lag_corr(x: &[f64], y: &[f64], lag: usize) -> f64 {
    if lag >= x.len() || lag >= y.len() {
        return 0.0;
    }
    let x0 = &x[..x.len() - lag];
    let y1 = &y[lag..];
    pearson_corr(x0, y1)
}

fn moment_pack(values: &[f64]) -> MomentPack {
    let m = mean(values);
    let sd = std_dev(values, m);
    if sd <= f64::EPSILON {
        return MomentPack {
            mean: m,
            std_dev: sd,
            skewness: 0.0,
            kurtosis_excess: 0.0,
            lag1_autocorr: lag1_autocorr(values),
            shannon_entropy_bits: shannon_entropy_bits(values, 32),
        };
    }
    let mut s3 = 0.0f64;
    let mut s4 = 0.0f64;
    for &v in values {
        let z = (v - m) / sd;
        s3 += z.powi(3);
        s4 += z.powi(4);
    }
    let n = values.len().max(1) as f64;
    MomentPack {
        mean: m,
        std_dev: sd,
        skewness: s3 / n,
        kurtosis_excess: s4 / n - 3.0,
        lag1_autocorr: lag1_autocorr(values),
        shannon_entropy_bits: shannon_entropy_bits(values, 32),
    }
}

fn finite_mean_dt(time: &[f64]) -> f64 {
    if time.len() < 2 {
        return 1.0;
    }
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for w in time.windows(2) {
        let dt = w[1] - w[0];
        if dt.is_finite() && dt > 0.0 {
            sum += dt;
            n += 1;
        }
    }
    if n == 0 { 1.0 } else { sum / n as f64 }
}

fn fit_loglog_slope(freq: &[f64], power: &[f64]) -> f64 {
    let mut n = 0usize;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut sxx = 0.0f64;
    let mut sxy = 0.0f64;
    for (&f, &p) in freq.iter().zip(power) {
        if f > 0.0 && p > 0.0 && f.is_finite() && p.is_finite() {
            let x = f.ln();
            let y = p.ln();
            n += 1;
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
    }
    if n < 2 {
        return 0.0;
    }
    let n_f = n as f64;
    let denom = n_f * sxx - sx * sx;
    if denom.abs() <= f64::EPSILON {
        0.0
    } else {
        (n_f * sxy - sx * sy) / denom
    }
}

fn dft_power_features(values: &[f64], dt_seconds: f64) -> SpectralFeatures {
    let n = values.len();
    if n < 4 {
        return SpectralFeatures {
            sample_count: n,
            dt_seconds,
            dominant_frequency_hz: 0.0,
            dominant_period_s: 0.0,
            dominant_power: 0.0,
            total_power: 0.0,
            harmonic2_ratio: 0.0,
            loglog_slope: 0.0,
        };
    }
    let mean_v = mean(values);
    let demeaned: Vec<f64> = values.iter().map(|v| v - mean_v).collect();
    let half = n / 2;
    let norm = (n as f64).powi(2);
    let mut freq = Vec::new();
    let mut power = Vec::new();
    for k in 1..=half {
        let mut re = 0.0f64;
        let mut im = 0.0f64;
        for (t, &x) in demeaned.iter().enumerate() {
            let theta = std::f64::consts::TAU * (k as f64) * (t as f64) / (n as f64);
            re += x * theta.cos();
            im -= x * theta.sin();
        }
        let p = (re * re + im * im) / norm.max(f64::EPSILON);
        let f = k as f64 / (n as f64 * dt_seconds.max(1.0e-12));
        freq.push(f);
        power.push(p);
    }
    let mut dom_idx = 0usize;
    for i in 1..power.len() {
        if power[i] > power[dom_idx] {
            dom_idx = i;
        }
    }
    let dom_freq = freq[dom_idx];
    let dom_power = power[dom_idx];
    let total_power: f64 = power.iter().sum();
    let dom_k = dom_idx + 1;
    let harmonic2_ratio = if dom_k * 2 <= power.len() && dom_power > 0.0 {
        power[dom_k * 2 - 1] / dom_power
    } else {
        0.0
    };
    SpectralFeatures {
        sample_count: n,
        dt_seconds,
        dominant_frequency_hz: dom_freq,
        dominant_period_s: if dom_freq > 0.0 { 1.0 / dom_freq } else { 0.0 },
        dominant_power: dom_power,
        total_power,
        harmonic2_ratio,
        loglog_slope: fit_loglog_slope(&freq, &power),
    }
}

fn classify_timing_regime(timing: &TimingSnapshot) -> TimingRegimeSummary {
    let mean_step_us = if timing.steps_per_sec > 0.0 {
        1.0e6 / timing.steps_per_sec
    } else {
        0.0
    };
    let p90_to_p50 = if timing.p50_us > 0.0 {
        timing.p90_us / timing.p50_us
    } else {
        0.0
    };
    let p99_to_p50 = if timing.p50_us > 0.0 {
        timing.p99_us / timing.p50_us
    } else {
        0.0
    };
    let (regime, rationale) =
        if timing.p50_us <= 50.0 && mean_step_us <= 200.0 && p99_to_p50 >= 10.0 {
            (
                TimingRegime::LaunchLike,
                "sub-50us p50 with heavy-tail indicates launch-dominated timing".to_string(),
            )
        } else if timing.p50_us >= 500.0 && mean_step_us >= 500.0 {
            (
                TimingRegime::KernelLike,
                ">=500us p50 and mean step implies kernel-time-dominated timing".to_string(),
            )
        } else {
            (
                TimingRegime::Transitional,
                "between launch-like and kernel-like envelopes".to_string(),
            )
        };
    TimingRegimeSummary {
        regime,
        mean_step_us,
        p90_to_p50,
        p99_to_p50,
        rationale,
    }
}

fn overhead_decomposition(timing: &TimingSnapshot, trace_stride: usize) -> OverheadDecomposition {
    let expected_trace_events = timing.steps.checked_div(trace_stride).unwrap_or(0);
    let observed_total_us = if timing.steps_per_sec > 0.0 {
        timing.steps as f64 * (1.0e6 / timing.steps_per_sec)
    } else {
        0.0
    };
    let base_step_total_us = timing.p50_us.max(0.0) * timing.steps as f64;
    let reduction_per_event = (timing.p90_us - timing.p50_us).max(0.0);
    let dtoh_per_event = (timing.p99_us - timing.p90_us).max(0.0);
    let estimated_reduction_overhead_us = reduction_per_event * expected_trace_events as f64;
    let estimated_dtoh_overhead_us = dtoh_per_event * expected_trace_events as f64;
    let estimated_total_us =
        base_step_total_us + estimated_reduction_overhead_us + estimated_dtoh_overhead_us;
    let closure_error_pct = if observed_total_us > f64::EPSILON {
        (estimated_total_us / observed_total_us - 1.0) * 100.0
    } else {
        0.0
    };
    OverheadDecomposition {
        expected_trace_events,
        observed_total_us,
        base_step_total_us,
        estimated_reduction_overhead_us,
        estimated_dtoh_overhead_us,
        estimated_total_us,
        closure_error_pct,
    }
}

fn nonzero_fraction(values: &[f64], eps: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let nz = values.iter().filter(|v| v.abs() > eps).count();
    nz as f64 / values.len() as f64
}

fn model_lock_fraction(measured: &[f64], model: &[f64], abs_eps: f64, rel_eps: f64) -> Option<f64> {
    let n = measured.len().min(model.len());
    if n == 0 {
        return None;
    }
    let mut locked = 0usize;
    for i in 0..n {
        let tol = abs_eps.max(rel_eps * model[i].abs());
        if (measured[i] - model[i]).abs() <= tol {
            locked += 1;
        }
    }
    Some(locked as f64 / n as f64)
}

fn fallback_fraction(enstrophy: &[f64], measured: &[f64], eps: f64) -> Option<f64> {
    let n = enstrophy.len().min(measured.len());
    if n == 0 {
        return None;
    }
    let mut count = 0usize;
    for i in 0..n {
        if measured[i].abs() <= eps && enstrophy[i].abs() > eps {
            count += 1;
        }
    }
    Some(count as f64 / n as f64)
}

fn both_zero_fraction(primary: &[f64], measured: &[f64], eps: f64) -> Option<f64> {
    let n = primary.len().min(measured.len());
    if n == 0 {
        return None;
    }
    let mut count = 0usize;
    for i in 0..n {
        if primary[i].abs() <= eps && measured[i].abs() <= eps {
            count += 1;
        }
    }
    Some(count as f64 / n as f64)
}

fn high_latency_tail_count(timing: &TimingSnapshot) -> usize {
    timing
        .histogram
        .iter()
        .filter(|bin| bin.lower_us >= timing.p90_us)
        .map(|bin| bin.count)
        .sum()
}

fn geometric_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut log_sum = 0.0;
    let mut n = 0usize;
    for &v in values {
        if v.is_finite() && v > 0.0 {
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

fn pct_delta(new: f64, old: f64) -> f64 {
    if !old.is_finite() || old.abs() <= f64::EPSILON {
        0.0
    } else {
        (new / old - 1.0) * 100.0
    }
}

#[cfg(feature = "hdf5-export")]
fn read_trace_optional(path: &Path, component: &str) -> Option<Vec<f64>> {
    read_simulation_trace_component(path, component).ok()
}

#[cfg(feature = "hdf5-export")]
fn read_spectral_optional(path: &Path, component: &str) -> Option<Vec<f64>> {
    read_simulation_spectral_component(path, component).ok()
}

fn unique_report_path() -> PathBuf {
    let date = chrono::Utc::now().format("%Y_%m_%d").to_string();
    let stamp = chrono::Utc::now().format("%H%M%S_%3f").to_string();
    PathBuf::from(format!(
        "reports/warp_fastpath_analysis_{}_{}.toml",
        date, stamp
    ))
}

#[cfg(feature = "hdf5-export")]
fn run() -> Result<(), Box<dyn Error>> {
    let cli = parse_args()?;
    if !cli.production_dir.exists() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!(
                "production_dir does not exist: {}",
                cli.production_dir.display()
            ),
        )
        .into());
    }
    let loaded_gate_policy = load_warp_gate_policy()?;
    let gate_policy = loaded_gate_policy.policy;

    let mut cases = Vec::new();
    let mut baseline_snapshots = BTreeMap::<usize, TimingSnapshot>::new();
    let mut legacy_snapshots = BTreeMap::<usize, TimingSnapshot>::new();

    for &res in &cli.resolutions {
        let h5_path = find_h5_file(&cli.production_dir, res)?;
        let timing_path = find_timing_file(&cli.production_dir, res)?;
        let timing = parse_timing_snapshot(&timing_path)?;

        if let Some(dir) = cli.baseline_dir.as_deref() {
            if dir.exists() {
                if let Ok(path) = find_timing_file(dir, res) {
                    if let Ok(s) = parse_timing_snapshot(&path) {
                        baseline_snapshots.insert(res, s);
                    }
                }
            }
        }
        if let Some(dir) = cli.legacy_dir.as_deref() {
            if dir.exists() {
                if let Ok(path) = find_timing_file(dir, res) {
                    if let Ok(s) = parse_timing_snapshot(&path) {
                        legacy_snapshots.insert(res, s);
                    }
                }
            }
        }

        let time_trace = read_simulation_trace_component(&h5_path, "time")?;
        let rho = read_simulation_trace_component(&h5_path, "rho_mean")?;
        let enstrophy = read_simulation_trace_component(&h5_path, "enstrophy")?;
        let algebra_norm = read_simulation_trace_component(&h5_path, "algebra_norm")?;
        let u_rms = read_simulation_trace_component(&h5_path, "u_rms")?;
        let u_rms_model = read_trace_optional(&h5_path, "u_rms_model");
        let enstrophy_measured = read_trace_optional(&h5_path, "enstrophy_measured");
        let spectral_total_power = read_spectral_optional(&h5_path, "total_power");

        let thresholds = RhoQualityThresholds::default();
        let rho_quality = validate_rho_trace(&rho, thresholds).map_err(|e| {
            std::io::Error::other(format!(
                "{}: rho quality gate failed: {e}",
                h5_path.display()
            ))
        })?;
        let rho_mean = mean(&rho);
        let rho_std = std_dev(&rho, rho_mean);

        let enstrophy_fallback_fraction = enstrophy_measured
            .as_ref()
            .and_then(|m| fallback_fraction(&enstrophy, m, 1.0e-20));
        let enstrophy_both_zero_fraction = enstrophy_measured
            .as_ref()
            .and_then(|m| both_zero_fraction(&enstrophy, m, 1.0e-20));
        let u_rms_model_lock_fraction = u_rms_model
            .as_ref()
            .and_then(|m| model_lock_fraction(&u_rms, m, 1.0e-14, 1.0e-6));
        let spectral_total_power_nonzero_fraction = spectral_total_power
            .as_ref()
            .map(|v| nonzero_fraction(v, 1.0e-30));
        let spectral_total_power_std_dev = spectral_total_power.as_ref().map(|v| {
            let m = mean(v);
            std_dev(v, m)
        });
        let dt_seconds = finite_mean_dt(&time_trace);
        let spectral_rho = dft_power_features(&rho, dt_seconds);
        let spectral_enstrophy = dft_power_features(&enstrophy, dt_seconds);
        let spectral_u_rms = dft_power_features(&u_rms, dt_seconds);
        let invariants = InvariantPack {
            rho: moment_pack(&rho),
            enstrophy: moment_pack(&enstrophy),
            algebra_norm: moment_pack(&algebra_norm),
            u_rms: moment_pack(&u_rms),
            corr_rho_enstrophy: pearson_corr(&rho, &enstrophy),
            corr_rho_u_rms: pearson_corr(&rho, &u_rms),
            corr_enstrophy_u_rms: pearson_corr(&enstrophy, &u_rms),
            lag1_corr_enstrophy_u_rms: lag_corr(&enstrophy, &u_rms, 1),
        };

        let tail_count = high_latency_tail_count(&timing);
        let expected_trace_events = timing.steps / cli.trace_stride;
        let tail_to_trace_ratio = if expected_trace_events > 0 {
            tail_count as f64 / expected_trace_events as f64
        } else {
            0.0
        };
        let timing_regime = classify_timing_regime(&timing);
        let overhead = overhead_decomposition(&timing, cli.trace_stride);

        cases.push(CaseAnalysis {
            resolution: res,
            h5_path,
            timing_path,
            timing,
            tail_count,
            tail_to_trace_ratio,
            rho_quality,
            rho_summary: ScalarSummary {
                mean: rho_mean,
                std_dev: rho_std,
                nonzero_fraction: nonzero_fraction(&rho, 1.0e-30),
            },
            fidelity: FidelitySummary {
                enstrophy_measured_present: enstrophy_measured.is_some(),
                enstrophy_nonzero_fraction: nonzero_fraction(&enstrophy, 1.0e-30),
                enstrophy_measured_nonzero_fraction: enstrophy_measured
                    .as_ref()
                    .map(|v| nonzero_fraction(v, 1.0e-30)),
                enstrophy_fallback_fraction,
                enstrophy_both_zero_fraction,
                algebra_nonzero_fraction: nonzero_fraction(&algebra_norm, 1.0e-30),
                u_rms_nonzero_fraction: nonzero_fraction(&u_rms, 1.0e-30),
                u_rms_model_present: u_rms_model.is_some(),
                u_rms_model_lock_fraction,
                spectral_total_power_present: spectral_total_power.is_some(),
                spectral_total_power_nonzero_fraction,
                spectral_total_power_std_dev,
            },
            timing_regime,
            overhead,
            spectral_rho,
            spectral_enstrophy,
            spectral_u_rms,
            invariants,
        });
    }

    cases.sort_by_key(|c| c.resolution);
    let tuned_mlups: Vec<f64> = cases.iter().map(|c| c.timing.mlups).collect();
    let baseline_mlups: Vec<f64> = cases
        .iter()
        .filter_map(|c| baseline_snapshots.get(&c.resolution).map(|s| s.mlups))
        .collect();

    let geom_tuned = geometric_mean(&tuned_mlups);
    let geom_baseline = geometric_mean(&baseline_mlups);
    let delta_geom = if baseline_mlups.len() == cases.len() && !baseline_mlups.is_empty() {
        Some(pct_delta(geom_tuned, geom_baseline))
    } else {
        None
    };

    let legacy_128_p50 = legacy_snapshots.get(&128).map(|s| s.p50_us);
    let legacy_256_p50 = legacy_snapshots.get(&256).map(|s| s.p50_us);
    let tuned_128_p50 = cases
        .iter()
        .find(|c| c.resolution == 128)
        .map(|c| c.timing.p50_us);
    let tuned_256_p50 = cases
        .iter()
        .find(|c| c.resolution == 256)
        .map(|c| c.timing.p50_us);

    let timing_regime_signature = match (legacy_128_p50, legacy_256_p50, tuned_128_p50, tuned_256_p50)
    {
        (Some(l128), Some(l256), Some(t128), Some(t256)) => format!(
            "Legacy p50 ({l128:.3}us/{l256:.3}us) is launch-like while tuned p50 ({t128:.3}us/{t256:.3}us) is kernel-like."
        ),
        _ => "Legacy/tuned regime split unavailable for one or more resolutions.".to_string(),
    };

    let periodic_tail_signature = cases
        .iter()
        .map(|c| {
            format!(
                "{}^3 tail_count={} tail_to_trace_ratio={:.6}",
                c.resolution, c.tail_count, c.tail_to_trace_ratio
            )
        })
        .collect::<Vec<_>>()
        .join("; ");

    let rho_ok = cases.iter().all(|c| {
        (c.rho_quality.final_value - 1.0).abs() <= 1.0e-12
            && c.rho_quality.abs_drift_final == 0.0
            && c.rho_quality.std_dev == 0.0
    });
    let invariant_signature = if rho_ok {
        "rho_mean remained exactly 1.0 with zero drift/std in all analyzed runs.".to_string()
    } else {
        let details = cases
            .iter()
            .map(|c| {
                format!(
                    "{}^3 rho_final={:.12} drift={:.3e} std={:.3e}",
                    c.resolution,
                    c.rho_quality.final_value,
                    c.rho_quality.abs_drift_final,
                    c.rho_quality.std_dev
                )
            })
            .collect::<Vec<_>>()
            .join("; ");
        format!("rho invariants diverged: {details}")
    };

    let fidelity_lines = cases
        .iter()
        .map(|c| {
            let enstrophy_measured_nonzero = c
                .fidelity
                .enstrophy_measured_nonzero_fraction
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "na".to_string());
            let fallback_fraction = c
                .fidelity
                .enstrophy_fallback_fraction
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "na".to_string());
            let both_zero_fraction = c
                .fidelity
                .enstrophy_both_zero_fraction
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "na".to_string());
            let u_rms_model_lock = c
                .fidelity
                .u_rms_model_lock_fraction
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "na".to_string());
            let spectral_nonzero = c
                .fidelity
                .spectral_total_power_nonzero_fraction
                .map(|v| format!("{v:.3}"))
                .unwrap_or_else(|| "na".to_string());
            format!(
                "{}^3 enstrophy_measured_present={}, u_rms_model_present={}, spectral_total_power_present={}, enstrophy_measured_nonzero={}, fallback_fraction={}, both_zero_fraction={}, u_rms_model_lock={}, spectral_total_power_nonzero={}",
                c.resolution,
                c.fidelity.enstrophy_measured_present,
                c.fidelity.u_rms_model_present,
                c.fidelity.spectral_total_power_present,
                enstrophy_measured_nonzero,
                fallback_fraction,
                both_zero_fraction,
                u_rms_model_lock,
                spectral_nonzero
            )
        })
        .collect::<Vec<_>>()
        .join("; ");
    let measurement_fidelity_signature =
        format!("Measured-vs-model fidelity summary: {fidelity_lines}");

    let measured_activity_gate_pass = cases.iter().all(|c| {
        c.fidelity.enstrophy_measured_nonzero_fraction.unwrap_or(0.0)
            >= gate_policy.measured_enstrophy_nonzero_fraction_min
            && c.fidelity.u_rms_nonzero_fraction >= gate_policy.measured_u_rms_nonzero_fraction_min
            && c.fidelity.algebra_nonzero_fraction
                >= gate_policy.measured_algebra_norm_nonzero_fraction_min
            && c.fidelity.spectral_total_power_nonzero_fraction.unwrap_or(0.0)
                >= gate_policy.measured_spectral_total_power_nonzero_fraction_min
            && c.fidelity.u_rms_model_lock_fraction.unwrap_or(1.0)
                < gate_policy.u_rms_model_lock_max_fraction
    });

    let report_path = cli.report_out.unwrap_or_else(unique_report_path);
    if let Some(parent) = report_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let mut out = String::new();
    out.push_str("schema = \"open_gororoba.warp_fastpath_analysis.v3\"\n");
    out.push_str(&format!(
        "generated_at_utc = \"{}\"\n\n",
        chrono::Utc::now().to_rfc3339()
    ));
    out.push_str("[inputs]\n");
    out.push_str(&format!(
        "production_dir = \"{}\"\n",
        cli.production_dir.display()
    ));
    if let Some(dir) = cli.baseline_dir.as_deref() {
        out.push_str(&format!("baseline_dir = \"{}\"\n", dir.display()));
    }
    if let Some(dir) = cli.legacy_dir.as_deref() {
        out.push_str(&format!("legacy_dir = \"{}\"\n", dir.display()));
    }
    out.push_str(&format!("trace_stride = {}\n\n", cli.trace_stride));

    out.push_str("[methods]\n");
    out.push_str("timing_tail_method = \"count histogram mass with lower_us >= p90_us and compare to steps/trace_stride\"\n");
    out.push_str("fidelity_method = \"quantify measured-vs-model lock, fallback fractions, and spectral nonzero richness\"\n");
    out.push_str("timing_regime_classifier = \"launch_like if p50<=50us && mean_step<=200us && p99/p50>=10; kernel_like if p50>=500us && mean_step>=500us; else transitional\"\n");
    out.push_str("overhead_decomposition_method = \"estimate base(p50) + per-trace reduction(p90-p50) + per-trace dtoh tail(p99-p90)\"\n");
    out.push_str("trace_spectral_method = \"naive DFT on demeaned trace for rho/enstrophy/u_rms, dominant frequency/harmonic/slope extraction\"\n");
    out.push_str("invariant_pack_method = \"moment pack (mean/std/skew/kurtosis/lag1/entropy) + cross-correlation matrix\"\n");
    out.push_str(&format!(
        "measured_gate_policy_source = \"{}\"\n",
        loaded_gate_policy.source_path.display()
    ));
    out.push_str(
        "invariant_method = \"rho_mean final/drift/std using default rho quality thresholds\"\n\n",
    );

    for case in &cases {
        out.push_str("[[case]]\n");
        out.push_str(&format!("resolution = {}\n", case.resolution));
        out.push_str(&format!("h5_path = \"{}\"\n", case.h5_path.display()));
        out.push_str(&format!(
            "timing_path = \"{}\"\n",
            case.timing_path.display()
        ));
        out.push_str(&format!("steps = {}\n", case.timing.steps));
        out.push_str(&format!(
            "steps_per_sec = {:.6}\n",
            case.timing.steps_per_sec
        ));
        out.push_str(&format!("mlups = {:.6}\n", case.timing.mlups));
        out.push_str(&format!("p50_us = {:.6}\n", case.timing.p50_us));
        out.push_str(&format!("p90_us = {:.6}\n", case.timing.p90_us));
        out.push_str(&format!("p99_us = {:.6}\n", case.timing.p99_us));
        out.push_str(&format!(
            "timing_regime = \"{}\"\n",
            case.timing_regime.regime.as_str()
        ));
        out.push_str(&format!(
            "timing_regime_mean_step_us = {:.6}\n",
            case.timing_regime.mean_step_us
        ));
        out.push_str(&format!(
            "timing_regime_p90_to_p50 = {:.6}\n",
            case.timing_regime.p90_to_p50
        ));
        out.push_str(&format!(
            "timing_regime_p99_to_p50 = {:.6}\n",
            case.timing_regime.p99_to_p50
        ));
        out.push_str(&format!(
            "timing_regime_rationale = \"{}\"\n",
            case.timing_regime.rationale.replace('"', "'")
        ));
        if let Some(snapshot) = baseline_snapshots.get(&case.resolution) {
            out.push_str(&format!("baseline_mlups = {:.6}\n", snapshot.mlups));
            out.push_str(&format!(
                "delta_mlups_vs_baseline_pct = {:.4}\n",
                pct_delta(case.timing.mlups, snapshot.mlups)
            ));
        }
        out.push_str(&format!("tail_count = {}\n", case.tail_count));
        out.push_str(&format!(
            "tail_to_trace_ratio = {:.6}\n",
            case.tail_to_trace_ratio
        ));
        out.push_str(&format!(
            "overhead_expected_trace_events = {}\n",
            case.overhead.expected_trace_events
        ));
        out.push_str(&format!(
            "overhead_observed_total_us = {:.6}\n",
            case.overhead.observed_total_us
        ));
        out.push_str(&format!(
            "overhead_base_step_total_us = {:.6}\n",
            case.overhead.base_step_total_us
        ));
        out.push_str(&format!(
            "overhead_est_reduction_total_us = {:.6}\n",
            case.overhead.estimated_reduction_overhead_us
        ));
        out.push_str(&format!(
            "overhead_est_dtoh_total_us = {:.6}\n",
            case.overhead.estimated_dtoh_overhead_us
        ));
        out.push_str(&format!(
            "overhead_est_total_us = {:.6}\n",
            case.overhead.estimated_total_us
        ));
        out.push_str(&format!(
            "overhead_closure_error_pct = {:.6}\n",
            case.overhead.closure_error_pct
        ));
        out.push_str(&format!(
            "rho_final = {:.12}\n",
            case.rho_quality.final_value
        ));
        out.push_str(&format!(
            "rho_abs_drift_final = {:.9e}\n",
            case.rho_quality.abs_drift_final
        ));
        out.push_str(&format!("rho_std_dev = {:.9e}\n", case.rho_quality.std_dev));
        out.push_str(&format!("rho_mean = {:.12}\n", case.rho_summary.mean));
        out.push_str(&format!(
            "rho_trace_std = {:.9e}\n",
            case.rho_summary.std_dev
        ));
        out.push_str(&format!(
            "rho_nonzero_fraction = {:.6}\n",
            case.rho_summary.nonzero_fraction
        ));
        out.push_str(&format!(
            "enstrophy_nonzero_fraction = {:.6}\n",
            case.fidelity.enstrophy_nonzero_fraction
        ));
        out.push_str(&format!(
            "enstrophy_measured_present = {}\n",
            case.fidelity.enstrophy_measured_present
        ));
        if let Some(v) = case.fidelity.enstrophy_measured_nonzero_fraction {
            out.push_str(&format!("enstrophy_measured_nonzero_fraction = {:.6}\n", v));
        }
        if let Some(v) = case.fidelity.enstrophy_fallback_fraction {
            out.push_str(&format!("enstrophy_fallback_fraction = {:.6}\n", v));
        }
        if let Some(v) = case.fidelity.enstrophy_both_zero_fraction {
            out.push_str(&format!("enstrophy_both_zero_fraction = {:.6}\n", v));
        }
        out.push_str(&format!(
            "algebra_norm_nonzero_fraction = {:.6}\n",
            case.fidelity.algebra_nonzero_fraction
        ));
        out.push_str(&format!(
            "u_rms_nonzero_fraction = {:.6}\n",
            case.fidelity.u_rms_nonzero_fraction
        ));
        out.push_str(&format!(
            "u_rms_model_present = {}\n",
            case.fidelity.u_rms_model_present
        ));
        if let Some(v) = case.fidelity.u_rms_model_lock_fraction {
            out.push_str(&format!("u_rms_model_lock_fraction = {:.6}\n", v));
        }
        out.push_str(&format!(
            "spectral_total_power_present = {}\n",
            case.fidelity.spectral_total_power_present
        ));
        if let Some(v) = case.fidelity.spectral_total_power_nonzero_fraction {
            out.push_str(&format!(
                "spectral_total_power_nonzero_fraction = {:.6}\n",
                v
            ));
        }
        if let Some(v) = case.fidelity.spectral_total_power_std_dev {
            out.push_str(&format!("spectral_total_power_std_dev = {:.9e}\n", v));
        }
        out.push_str(&format!(
            "trace_spectral_sample_count = {}\n",
            case.spectral_rho.sample_count
        ));
        out.push_str(&format!(
            "trace_spectral_dt_seconds = {:.9e}\n",
            case.spectral_rho.dt_seconds
        ));
        out.push_str(&format!(
            "trace_spectral_rho_dom_freq_hz = {:.9e}\n",
            case.spectral_rho.dominant_frequency_hz
        ));
        out.push_str(&format!(
            "trace_spectral_rho_dom_period_s = {:.9e}\n",
            case.spectral_rho.dominant_period_s
        ));
        out.push_str(&format!(
            "trace_spectral_rho_dom_power = {:.9e}\n",
            case.spectral_rho.dominant_power
        ));
        out.push_str(&format!(
            "trace_spectral_rho_total_power = {:.9e}\n",
            case.spectral_rho.total_power
        ));
        out.push_str(&format!(
            "trace_spectral_rho_harmonic2_ratio = {:.9e}\n",
            case.spectral_rho.harmonic2_ratio
        ));
        out.push_str(&format!(
            "trace_spectral_rho_loglog_slope = {:.9e}\n",
            case.spectral_rho.loglog_slope
        ));
        out.push_str(&format!(
            "trace_spectral_enstrophy_dom_freq_hz = {:.9e}\n",
            case.spectral_enstrophy.dominant_frequency_hz
        ));
        out.push_str(&format!(
            "trace_spectral_enstrophy_dom_period_s = {:.9e}\n",
            case.spectral_enstrophy.dominant_period_s
        ));
        out.push_str(&format!(
            "trace_spectral_enstrophy_dom_power = {:.9e}\n",
            case.spectral_enstrophy.dominant_power
        ));
        out.push_str(&format!(
            "trace_spectral_enstrophy_total_power = {:.9e}\n",
            case.spectral_enstrophy.total_power
        ));
        out.push_str(&format!(
            "trace_spectral_enstrophy_harmonic2_ratio = {:.9e}\n",
            case.spectral_enstrophy.harmonic2_ratio
        ));
        out.push_str(&format!(
            "trace_spectral_enstrophy_loglog_slope = {:.9e}\n",
            case.spectral_enstrophy.loglog_slope
        ));
        out.push_str(&format!(
            "trace_spectral_u_rms_dom_freq_hz = {:.9e}\n",
            case.spectral_u_rms.dominant_frequency_hz
        ));
        out.push_str(&format!(
            "trace_spectral_u_rms_dom_period_s = {:.9e}\n",
            case.spectral_u_rms.dominant_period_s
        ));
        out.push_str(&format!(
            "trace_spectral_u_rms_dom_power = {:.9e}\n",
            case.spectral_u_rms.dominant_power
        ));
        out.push_str(&format!(
            "trace_spectral_u_rms_total_power = {:.9e}\n",
            case.spectral_u_rms.total_power
        ));
        out.push_str(&format!(
            "trace_spectral_u_rms_harmonic2_ratio = {:.9e}\n",
            case.spectral_u_rms.harmonic2_ratio
        ));
        out.push_str(&format!(
            "trace_spectral_u_rms_loglog_slope = {:.9e}\n",
            case.spectral_u_rms.loglog_slope
        ));
        out.push_str(&format!(
            "invariant_rho_mean = {:.12}\n",
            case.invariants.rho.mean
        ));
        out.push_str(&format!(
            "invariant_rho_std = {:.9e}\n",
            case.invariants.rho.std_dev
        ));
        out.push_str(&format!(
            "invariant_rho_skew = {:.9e}\n",
            case.invariants.rho.skewness
        ));
        out.push_str(&format!(
            "invariant_rho_kurtosis_excess = {:.9e}\n",
            case.invariants.rho.kurtosis_excess
        ));
        out.push_str(&format!(
            "invariant_rho_lag1_autocorr = {:.9e}\n",
            case.invariants.rho.lag1_autocorr
        ));
        out.push_str(&format!(
            "invariant_rho_entropy_bits = {:.9e}\n",
            case.invariants.rho.shannon_entropy_bits
        ));
        out.push_str(&format!(
            "invariant_enstrophy_mean = {:.12}\n",
            case.invariants.enstrophy.mean
        ));
        out.push_str(&format!(
            "invariant_enstrophy_std = {:.9e}\n",
            case.invariants.enstrophy.std_dev
        ));
        out.push_str(&format!(
            "invariant_enstrophy_skew = {:.9e}\n",
            case.invariants.enstrophy.skewness
        ));
        out.push_str(&format!(
            "invariant_enstrophy_kurtosis_excess = {:.9e}\n",
            case.invariants.enstrophy.kurtosis_excess
        ));
        out.push_str(&format!(
            "invariant_enstrophy_lag1_autocorr = {:.9e}\n",
            case.invariants.enstrophy.lag1_autocorr
        ));
        out.push_str(&format!(
            "invariant_enstrophy_entropy_bits = {:.9e}\n",
            case.invariants.enstrophy.shannon_entropy_bits
        ));
        out.push_str(&format!(
            "invariant_algebra_norm_mean = {:.12}\n",
            case.invariants.algebra_norm.mean
        ));
        out.push_str(&format!(
            "invariant_algebra_norm_std = {:.9e}\n",
            case.invariants.algebra_norm.std_dev
        ));
        out.push_str(&format!(
            "invariant_algebra_norm_skew = {:.9e}\n",
            case.invariants.algebra_norm.skewness
        ));
        out.push_str(&format!(
            "invariant_algebra_norm_kurtosis_excess = {:.9e}\n",
            case.invariants.algebra_norm.kurtosis_excess
        ));
        out.push_str(&format!(
            "invariant_algebra_norm_lag1_autocorr = {:.9e}\n",
            case.invariants.algebra_norm.lag1_autocorr
        ));
        out.push_str(&format!(
            "invariant_algebra_norm_entropy_bits = {:.9e}\n",
            case.invariants.algebra_norm.shannon_entropy_bits
        ));
        out.push_str(&format!(
            "invariant_u_rms_mean = {:.12}\n",
            case.invariants.u_rms.mean
        ));
        out.push_str(&format!(
            "invariant_u_rms_std = {:.9e}\n",
            case.invariants.u_rms.std_dev
        ));
        out.push_str(&format!(
            "invariant_u_rms_skew = {:.9e}\n",
            case.invariants.u_rms.skewness
        ));
        out.push_str(&format!(
            "invariant_u_rms_kurtosis_excess = {:.9e}\n",
            case.invariants.u_rms.kurtosis_excess
        ));
        out.push_str(&format!(
            "invariant_u_rms_lag1_autocorr = {:.9e}\n",
            case.invariants.u_rms.lag1_autocorr
        ));
        out.push_str(&format!(
            "invariant_u_rms_entropy_bits = {:.9e}\n",
            case.invariants.u_rms.shannon_entropy_bits
        ));
        out.push_str(&format!(
            "invariant_corr_rho_enstrophy = {:.9e}\n",
            case.invariants.corr_rho_enstrophy
        ));
        out.push_str(&format!(
            "invariant_corr_rho_u_rms = {:.9e}\n",
            case.invariants.corr_rho_u_rms
        ));
        out.push_str(&format!(
            "invariant_corr_enstrophy_u_rms = {:.9e}\n",
            case.invariants.corr_enstrophy_u_rms
        ));
        out.push_str(&format!(
            "invariant_lag1_corr_enstrophy_u_rms = {:.9e}\n",
            case.invariants.lag1_corr_enstrophy_u_rms
        ));
        out.push('\n');
    }

    out.push_str("[aggregate]\n");
    out.push_str(&format!("case_count = {}\n", cases.len()));
    out.push_str(&format!("geom_mlups_tuned = {:.6}\n", geom_tuned));
    if let Some(delta) = delta_geom {
        out.push_str(&format!("geom_mlups_baseline = {:.6}\n", geom_baseline));
        out.push_str(&format!(
            "delta_geom_mlups_vs_baseline_pct = {:.4}\n",
            delta
        ));
    }
    out.push_str(&format!("rho_gate_pass = {}\n", rho_ok));
    let launch_like_count = cases
        .iter()
        .filter(|c| matches!(c.timing_regime.regime, TimingRegime::LaunchLike))
        .count();
    let kernel_like_count = cases
        .iter()
        .filter(|c| matches!(c.timing_regime.regime, TimingRegime::KernelLike))
        .count();
    let transitional_count = cases
        .iter()
        .filter(|c| matches!(c.timing_regime.regime, TimingRegime::Transitional))
        .count();
    let mean_tail_to_trace_ratio = if cases.is_empty() {
        0.0
    } else {
        cases.iter().map(|c| c.tail_to_trace_ratio).sum::<f64>() / cases.len() as f64
    };
    out.push_str(&format!("launch_like_case_count = {}\n", launch_like_count));
    out.push_str(&format!("kernel_like_case_count = {}\n", kernel_like_count));
    out.push_str(&format!("transitional_case_count = {}\n", transitional_count));
    out.push_str(&format!(
        "mean_tail_to_trace_ratio = {:.6}\n",
        mean_tail_to_trace_ratio
    ));
    out.push_str(&format!(
        "measured_activity_gate_pass = {}\n\n",
        measured_activity_gate_pass
    ));

    out.push_str("[signatures]\n");
    out.push_str(&format!(
        "timing_regime_split = \"{}\"\n",
        timing_regime_signature.replace('"', "'")
    ));
    out.push_str(&format!(
        "periodic_trace_overhead_fingerprint = \"{}\"\n",
        periodic_tail_signature.replace('"', "'")
    ));
    out.push_str(&format!(
        "invariant_signature = \"{}\"\n",
        invariant_signature.replace('"', "'")
    ));
    out.push_str(&format!(
        "measurement_fidelity_signature = \"{}\"\n\n",
        measurement_fidelity_signature.replace('"', "'")
    ));

    out.push_str("[[conjecture]]\n");
    out.push_str("id = \"WFA-C1\"\n");
    out.push_str(
        "statement = \"High-latency tail mass is primarily deterministic trace cadence overhead, not random jitter.\"\n",
    );
    out.push_str("falsifiable_test = \"Run identical 300s lanes at trace_stride=5,10,20 and test if tail_to_trace_ratio remains approximately 1 while absolute tail count scales with 1/stride.\"\n\n");

    out.push_str("[[conjecture]]\n");
    out.push_str("id = \"WFA-C2\"\n");
    out.push_str(
        "statement = \"Low measured enstrophy activity in BF16 can be instrumentation/model-lock dominated even when throughput and rho invariants look healthy.\"\n",
    );
    out.push_str("falsifiable_test = \"Run matched FP32/BF16 with identical forcing and compare enstrophy_measured_nonzero_fraction and u_rms_model_lock_fraction under canonical_300s gating.\"\n\n");

    out.push_str("[[conjecture]]\n");
    out.push_str("id = \"WFA-C3\"\n");
    out.push_str(
        "statement = \"A physically valid fastpath requires both performance gain and measured activity gain (nonzero spectral and enstrophy_measured channels).\"\n",
    );
    out.push_str("falsifiable_test = \"Adopt measured_activity_gate_pass as a required gate before claiming production-grade physics fidelity.\"\n");

    std::fs::write(&report_path, out)?;
    println!("FASTPATH_ANALYSIS_REPORT: {}", report_path.display());
    Ok(())
}

#[cfg(not(feature = "hdf5-export"))]
fn run() -> Result<(), Box<dyn Error>> {
    Err(std::io::Error::other(
        "warp-fastpath-analyze requires hdf5-export feature: cargo run -p gororoba_cli --features 'gpu,hdf5-export' --bin warp-fastpath-analyze -- <production_dir>",
    )
    .into())
}

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();
    run()
}
