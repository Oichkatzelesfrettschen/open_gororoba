#![cfg_attr(not(feature = "hdf5-export"), allow(dead_code, unused_imports))]

#[cfg(feature = "hdf5-export")]
use data_core::hdf5_export::{read_simulation_spectral_component, read_simulation_trace_component};
use data_core::quality::{validate_rho_trace, RhoQualityThresholds, RhoTraceQuality};
#[cfg(feature = "hdf5-export")]
use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};

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
        if !digits.is_empty() && token.contains('s') {
            if let Ok(value) = digits.parse::<u64>() {
                return value;
            }
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

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn std_dev(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
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

        let tail_count = high_latency_tail_count(&timing);
        let expected_trace_events = timing.steps / cli.trace_stride;
        let tail_to_trace_ratio = if expected_trace_events > 0 {
            tail_count as f64 / expected_trace_events as f64
        } else {
            0.0
        };

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

    let timing_regime_signature = match (legacy_128_p50, legacy_256_p50, tuned_128_p50, tuned_256_p50) {
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
        c.fidelity
            .enstrophy_measured_nonzero_fraction
            .unwrap_or(0.0)
            > 0.0
            && c.fidelity
                .spectral_total_power_nonzero_fraction
                .unwrap_or(0.0)
                > 0.0
            && c.fidelity.u_rms_model_lock_fraction.unwrap_or(1.0) < 0.999
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
