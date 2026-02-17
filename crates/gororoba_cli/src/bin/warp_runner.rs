use algebra_core::physics::octonion_field::FieldParams;
#[cfg(feature = "hdf5-export")]
use data_core::hdf5_export::{
    export_experiment_contract, export_rho_quality_metrics, export_simulation_spectral_summary,
    export_simulation_trace_bundle, read_simulation_spectral_component,
    read_simulation_trace_component, scan_hdf5_numeric_datasets, NumericDatasetScanStatus,
    SimulationTraceBundle, SpectralSummarySeries,
};
use data_core::quality::{validate_rho_trace, RhoQualityThresholds, RhoTraceQuality};
#[cfg(feature = "hdf5-export")]
use data_core::quality::{validate_scalar_trace_signal, ScalarTraceThresholds};
#[cfg(feature = "hdf5-export")]
use gororoba_cli::warp_gate_policy::{
    load_warp_gate_policy, CANONICAL_REQUIRED_SPECTRAL_CHANNELS, CANONICAL_REQUIRED_TRACE_CHANNELS,
};
#[cfg(feature = "hdf5-export")]
use gororoba_contracts::{WarpRingConfig, WarpRingExperiment, WarpRingResults};
use gororoba_engine::simulation::{E7SpectralFilter, SimulationConfig3D, SimulationState3D};
use lbm_3d_cuda::Precision;
use lbm_core::turbulence::{extract_dominant_triads, power_spectrum, triad_clustering_coefficient};
use ndarray::Array2;
#[cfg(feature = "hdf5-export")]
use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};

const LBM_TAU_BENCH: f64 = 0.6;

#[cfg(not(feature = "hdf5-export"))]
#[derive(Debug, Clone, Default)]
struct SpectralSummarySeries {
    time: Vec<f64>,
    k_peak: Vec<f64>,
    power_peak: Vec<f64>,
    total_power: Vec<f64>,
    slope: Vec<f64>,
    triad_count: Vec<f64>,
    triad_clustering: Vec<f64>,
}

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

#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateProfile {
    Research,
    Canonical300s,
    Canonical300sMeasured,
}

#[allow(dead_code)]
impl GateProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Research => "research",
            Self::Canonical300s => "canonical_300s",
            Self::Canonical300sMeasured => "canonical_300s_measured",
        }
    }
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

#[derive(Debug, Clone, Copy)]
pub struct KolmogorovForcingSpec {
    pub tau: f64,
    pub nu: f64,
    pub cs: f64,
    pub rho0: f64,
    pub mode_y_requested: usize,
    pub mode_y: usize,
    pub k_y: f64,
    pub re_target: f64,
    pub re_effective: f64,
    pub max_mach: f64,
    pub mach_effective: f64,
    pub u_target: f64,
    pub acceleration_amplitude: f64,
    pub body_force_density_amplitude: f64,
    pub viscous_time_steps: f64,
    pub power_injection_density: f64,
    pub bf16_distribution_ulp: f64,
    pub bf16_delta_f_estimate: f64,
    pub bf16_delta_to_ulp_ratio: f64,
    pub bf16_mode_floor_target_ratio: f64,
    pub bf16_mode_floor_applied: bool,
    pub mode_floor_all_precisions: bool,
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
    pub forcing: KolmogorovForcingSpec,
    pub h5_output: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy)]
struct SpectralSignaturePoint {
    k_peak: f64,
    power_peak: f64,
    total_power: f64,
    slope: f64,
    triad_count: f64,
    triad_clustering: f64,
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

fn parse_env_f64(name: &str, default: f64) -> Result<f64, Box<dyn Error>> {
    match std::env::var(name) {
        Ok(raw) => raw.parse::<f64>().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("{name} must be a finite float, got '{raw}': {e}"),
            )
            .into()
        }),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(e) => Err(std::io::Error::other(format!("failed to read {name}: {e}")).into()),
    }
}

fn parse_env_usize(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    match std::env::var(name) {
        Ok(raw) => raw.parse::<usize>().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("{name} must be a positive integer, got '{raw}': {e}"),
            )
            .into()
        }),
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(e) => Err(std::io::Error::other(format!("failed to read {name}: {e}")).into()),
    }
}

fn parse_env_bool(name: &str, default: bool) -> Result<bool, Box<dyn Error>> {
    match std::env::var(name) {
        Ok(raw) => match raw.to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "on" => Ok(true),
            "0" | "false" | "no" | "off" => Ok(false),
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("{name} must be bool-like (0/1/true/false), got '{raw}'"),
            )
            .into()),
        },
        Err(std::env::VarError::NotPresent) => Ok(default),
        Err(e) => Err(std::io::Error::other(format!("failed to read {name}: {e}")).into()),
    }
}

fn kolmogorov_u_rms_model(forcing: &KolmogorovForcingSpec) -> f64 {
    forcing.u_target.abs() / f64::sqrt(2.0)
}

fn kolmogorov_enstrophy_model(forcing: &KolmogorovForcingSpec) -> f64 {
    // For u_x(y) = U0*sin(k y), vorticity magnitude is |omega_z| = |k*U0*cos(k y)|.
    // Volume-mean enstrophy proxy is 0.5*(k*U0)^2 in lattice units.
    0.5 * (forcing.k_y * forcing.u_target).powi(2)
}

fn evaluate_kolmogorov_mode(
    ny: usize,
    mode_y: usize,
    tau: f64,
    re_target: f64,
    max_mach: f64,
    rho0: f64,
) -> Result<KolmogorovForcingSpec, Box<dyn Error>> {
    if mode_y == 0 {
        return Err(
            std::io::Error::new(std::io::ErrorKind::InvalidInput, "mode_y must be >= 1").into(),
        );
    }
    let cs2: f64 = 1.0 / 3.0;
    let cs = f64::sqrt(cs2);
    let nu = cs2 * (tau - 0.5);
    let k_y = std::f64::consts::TAU * mode_y as f64 / ny as f64;
    let u_from_re = re_target * nu * k_y;
    let u_from_mach = max_mach * cs;
    let u_target = u_from_re.min(u_from_mach);
    let re_effective = u_target / (nu * k_y);
    let mach_effective = u_target / cs;
    let acceleration_amplitude = nu * k_y * k_y * u_target;
    let body_force_density_amplitude = rho0 * acceleration_amplitude;
    let viscous_time_steps = 1.0 / (nu * k_y * k_y);
    let power_injection_density = 0.5 * body_force_density_amplitude * u_target;
    let bf16_relative_precision = 2.0f64.powi(-7);
    let d3q19_axis_weight = 1.0 / 18.0;
    let guo_prefactor = (1.0 - 1.0 / (2.0 * tau)).abs();
    let bf16_distribution_ulp = (rho0 * d3q19_axis_weight).abs() * bf16_relative_precision;
    let bf16_delta_f_estimate =
        guo_prefactor * 3.0 * d3q19_axis_weight * acceleration_amplitude.abs();
    let bf16_delta_to_ulp_ratio = if bf16_distribution_ulp > 0.0 {
        bf16_delta_f_estimate / bf16_distribution_ulp
    } else {
        0.0
    };

    Ok(KolmogorovForcingSpec {
        tau,
        nu,
        cs,
        rho0,
        mode_y_requested: mode_y,
        mode_y,
        k_y,
        re_target,
        re_effective,
        max_mach,
        mach_effective,
        u_target,
        acceleration_amplitude,
        body_force_density_amplitude,
        viscous_time_steps,
        power_injection_density,
        bf16_distribution_ulp,
        bf16_delta_f_estimate,
        bf16_delta_to_ulp_ratio,
        bf16_mode_floor_target_ratio: 0.0,
        bf16_mode_floor_applied: false,
        mode_floor_all_precisions: false,
    })
}

fn fit_loglog_slope(k_axis: &[f64], power: &[f64]) -> f64 {
    let mut n = 0usize;
    let mut sx = 0.0f64;
    let mut sy = 0.0f64;
    let mut sxx = 0.0f64;
    let mut sxy = 0.0f64;
    for (&k, &p) in k_axis.iter().zip(power) {
        if k > 0.0 && p > 0.0 && k.is_finite() && p.is_finite() {
            let x = k.ln();
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

fn compute_midplane_spectral_signature(
    state: &mut SimulationState3D,
) -> Result<Option<SpectralSignaturePoint>, Box<dyn Error>> {
    let (ux, uy, uz) = state.fluid.try_velocity(state.nx, state.ny, state.nz)?;
    let z_mid = state.nz / 2;
    let mut ux_plane = Array2::<f64>::zeros((state.nx, state.ny));
    let mut uy_plane = Array2::<f64>::zeros((state.nx, state.ny));
    let mut speed_plane = Array2::<f64>::zeros((state.nx, state.ny));
    for x in 0..state.nx {
        for y in 0..state.ny {
            let vx = ux[[x, y, z_mid]];
            let vy = uy[[x, y, z_mid]];
            let vz = uz[[x, y, z_mid]];
            ux_plane[[x, y]] = vx;
            uy_plane[[x, y]] = vy;
            speed_plane[[x, y]] = (vx * vx + vy * vy + vz * vz).sqrt();
        }
    }

    let (k_axis, power) = power_spectrum(&speed_plane);
    if power.is_empty() {
        return Ok(None);
    }
    let mut peak_idx = 0usize;
    for idx in 1..power.len() {
        if power[idx] > power[peak_idx] {
            peak_idx = idx;
        }
    }
    let total_power: f64 = power.iter().sum();
    let slope = fit_loglog_slope(&k_axis, &power);
    let triads = extract_dominant_triads(&ux_plane, &uy_plane, 1.0e-12);
    let triad_count = triads.len() as f64;
    let triad_clustering = if triads.is_empty() {
        0.0
    } else {
        triad_clustering_coefficient(&triads)
    };

    Ok(Some(SpectralSignaturePoint {
        k_peak: k_axis[peak_idx],
        power_peak: power[peak_idx],
        total_power,
        slope,
        triad_count,
        triad_clustering,
    }))
}

/// Derive sinusoidal Kolmogorov forcing from the steady incompressible NS balance.
///
/// We model:
///   u_x(y) = U0 * sin(k y),   f_x(y) = F0 * sin(k y)
/// and use:
///   0 = nu * d2(u_x)/dy2 + f_x  =>  F0 = nu * k^2 * U0
///
/// In lattice units:
///   nu = c_s^2 * (tau - 0.5), c_s^2 = 1/3, dt = dx = rho0 = 1 (default).
///
/// The characteristic velocity U0 is set by a target forcing-scale Reynolds number
/// and clipped by a low-Mach cap:
///   Re_target = U0 / (nu * k),  U0 <= Ma_max * c_s.
fn derive_kolmogorov_forcing_spec(
    ny: usize,
    tau: f64,
    precision: Precision,
) -> Result<KolmogorovForcingSpec, Box<dyn Error>> {
    if ny == 0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "ny must be > 0 for Kolmogorov forcing",
        )
        .into());
    }
    if tau <= 0.5 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("tau must be > 0.5 for positive viscosity, got {tau}"),
        )
        .into());
    }

    let mode_y_requested = parse_env_usize("GOROROBA_KOLMO_MODE_Y", 1)?.max(1);
    let re_target = parse_env_f64("GOROROBA_KOLMO_RE_TARGET", 64.0)?;
    let max_mach = parse_env_f64("GOROROBA_KOLMO_MAX_MACH", 0.08)?;
    let rho0 = parse_env_f64("GOROROBA_KOLMO_RHO0", 1.0)?;
    let bf16_mode_floor_target_ratio =
        parse_env_f64("GOROROBA_BF16_MIN_DELTA_ULP_RATIO", 1.0e-2)?.max(0.0);
    let bf16_mode_floor_enabled = parse_env_bool("GOROROBA_BF16_ENFORCE_MODE_FLOOR", true)?;
    let mode_floor_all_precisions =
        parse_env_bool("GOROROBA_ENFORCE_MODE_FLOOR_ALL_PRECISIONS", false)?;
    if !re_target.is_finite() || re_target <= 0.0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("GOROROBA_KOLMO_RE_TARGET must be finite and > 0, got {re_target}"),
        )
        .into());
    }
    if !max_mach.is_finite() || max_mach <= 0.0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("GOROROBA_KOLMO_MAX_MACH must be finite and > 0, got {max_mach}"),
        )
        .into());
    }
    if !rho0.is_finite() || rho0 <= 0.0 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("GOROROBA_KOLMO_RHO0 must be finite and > 0, got {rho0}"),
        )
        .into());
    }

    let mut spec = evaluate_kolmogorov_mode(ny, mode_y_requested, tau, re_target, max_mach, rho0)?;
    let mut mode_floor_applied = false;
    let max_mode = (ny / 2).saturating_sub(1).max(1);

    let apply_mode_floor = matches!(precision, Precision::BF16) || mode_floor_all_precisions;
    if apply_mode_floor
        && bf16_mode_floor_enabled
        && bf16_mode_floor_target_ratio > 0.0
        && spec.bf16_delta_to_ulp_ratio < bf16_mode_floor_target_ratio
        && mode_y_requested < max_mode
    {
        for mode in (mode_y_requested + 1)..=max_mode {
            let candidate = evaluate_kolmogorov_mode(ny, mode, tau, re_target, max_mach, rho0)?;
            spec = candidate;
            if spec.bf16_delta_to_ulp_ratio >= bf16_mode_floor_target_ratio {
                mode_floor_applied = true;
                break;
            }
        }
        mode_floor_applied = mode_floor_applied || spec.mode_y != mode_y_requested;
    }

    spec.mode_y_requested = mode_y_requested;
    spec.bf16_mode_floor_target_ratio = bf16_mode_floor_target_ratio;
    spec.bf16_mode_floor_applied = mode_floor_applied;
    spec.mode_floor_all_precisions = mode_floor_all_precisions;
    Ok(spec)
}

fn build_kolmogorov_force_field(
    nx: usize,
    ny: usize,
    nz: usize,
    mode_y: usize,
    acceleration_amplitude: f64,
) -> Vec<[f64; 3]> {
    let mut force = vec![[0.0f64; 3]; nx * ny * nz];
    for z in 0..nz {
        for y in 0..ny {
            let phase = std::f64::consts::TAU * mode_y as f64 * (y as f64) / ny as f64;
            let fx = acceleration_amplitude * phase.sin();
            for x in 0..nx {
                let idx = x + nx * (y + ny * z);
                force[idx] = [fx, 0.0, 0.0];
            }
        }
    }
    force
}

pub fn write_step_timing_report(
    path: &Path,
    report: &BenchCaseReport,
) -> Result<(), Box<dyn Error>> {
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
    out.push_str("\n[forcing]\n");
    out.push_str("model = \"kolmogorov_ns_balance\"\n");
    out.push_str(&format!("tau = {:.6}\n", report.forcing.tau));
    out.push_str(&format!("nu = {:.9}\n", report.forcing.nu));
    out.push_str(&format!("cs = {:.9}\n", report.forcing.cs));
    out.push_str(&format!("rho0 = {:.9}\n", report.forcing.rho0));
    out.push_str(&format!(
        "mode_y_requested = {}\n",
        report.forcing.mode_y_requested
    ));
    out.push_str(&format!("mode_y = {}\n", report.forcing.mode_y));
    out.push_str(&format!("k_y = {:.9}\n", report.forcing.k_y));
    out.push_str(&format!("re_target = {:.6}\n", report.forcing.re_target));
    out.push_str(&format!(
        "re_effective = {:.6}\n",
        report.forcing.re_effective
    ));
    out.push_str(&format!("max_mach = {:.6}\n", report.forcing.max_mach));
    out.push_str(&format!(
        "mach_effective = {:.6}\n",
        report.forcing.mach_effective
    ));
    out.push_str(&format!("u_target = {:.9}\n", report.forcing.u_target));
    out.push_str(&format!(
        "acceleration_amplitude = {:.9e}\n",
        report.forcing.acceleration_amplitude
    ));
    out.push_str(&format!(
        "body_force_density_amplitude = {:.9e}\n",
        report.forcing.body_force_density_amplitude
    ));
    out.push_str(&format!(
        "viscous_time_steps = {:.3}\n",
        report.forcing.viscous_time_steps
    ));
    out.push_str(&format!(
        "power_injection_density = {:.9e}\n",
        report.forcing.power_injection_density
    ));
    out.push_str(&format!(
        "bf16_distribution_ulp = {:.9e}\n",
        report.forcing.bf16_distribution_ulp
    ));
    out.push_str(&format!(
        "bf16_delta_f_estimate = {:.9e}\n",
        report.forcing.bf16_delta_f_estimate
    ));
    out.push_str(&format!(
        "bf16_delta_to_ulp_ratio = {:.9e}\n",
        report.forcing.bf16_delta_to_ulp_ratio
    ));
    out.push_str(&format!(
        "bf16_mode_floor_target_ratio = {:.9e}\n",
        report.forcing.bf16_mode_floor_target_ratio
    ));
    out.push_str(&format!(
        "bf16_mode_floor_applied = {}\n",
        report.forcing.bf16_mode_floor_applied
    ));
    out.push_str(&format!(
        "mode_floor_all_precisions = {}\n",
        report.forcing.mode_floor_all_precisions
    ));
    out.push_str(
        "notes = \"F0 = nu*k^2*U0; U0=min(Re_target*nu*k, max_mach*cs); mode floor can raise mode_y to meet min delta_f/ulp target (BF16-only by default or all precisions via env); f_x=F0*sin(k*y)\"\n",
    );
    out.push_str("\n[timing]\n");
    out.push_str(&format!(
        "sample_count = {}\n",
        report.step_timing.sample_count
    ));
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
#[allow(clippy::too_many_arguments)]
fn export_bench_trace(
    out_path: &Path,
    resolution: usize,
    backend: BackendKind,
    precision: Precision,
    forcing: &KolmogorovForcingSpec,
    total_steps: usize,
    elapsed_secs: f64,
    time_hist: &[f64],
    rho_hist: &[f64],
    enstrophy_hist: &[f64],
    enstrophy_measured_hist: &[f64],
    algebra_norm_hist: &[f64],
    u_rms_hist: &[f64],
    u_rms_model_hist: &[f64],
    mach_eff_hist: &[f64],
    re_eff_hist: &[f64],
    power_injection_proxy_hist: &[f64],
    dissipation_proxy_hist: &[f64],
    gpu_u_rms_sync_cadence_secs: f64,
    spectral_summary: &SpectralSummarySeries,
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
            tau: forcing.tau,
            forcing_type: format!(
                "Kolmogorov_NS_Derived(mode_y={},Re_target={:.3},Ma_max={:.3},F0={:.3e})",
                forcing.mode_y, forcing.re_target, forcing.max_mach, forcing.acceleration_amplitude
            ),
            coupling_lambda: 0.95,
            initial_condition: "Uniform_Rho1_U0".to_string(),
        },
        results: WarpRingResults {
            final_enstrophy: *enstrophy_hist.last().unwrap_or(&0.0),
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
    let mut channels = BTreeMap::new();
    channels.insert("rho_mean".to_string(), rho_hist.to_vec());
    channels.insert("enstrophy".to_string(), enstrophy_hist.to_vec());
    channels.insert(
        "enstrophy_measured".to_string(),
        enstrophy_measured_hist.to_vec(),
    );
    channels.insert("algebra_norm".to_string(), algebra_norm_hist.to_vec());
    channels.insert("u_rms".to_string(), u_rms_hist.to_vec());
    channels.insert("u_rms_model".to_string(), u_rms_model_hist.to_vec());
    channels.insert("mach_eff".to_string(), mach_eff_hist.to_vec());
    channels.insert("re_eff".to_string(), re_eff_hist.to_vec());
    channels.insert(
        "power_injection_proxy".to_string(),
        power_injection_proxy_hist.to_vec(),
    );
    channels.insert(
        "dissipation_proxy".to_string(),
        dissipation_proxy_hist.to_vec(),
    );
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "trace_profile".to_string(),
        "tiered_evidence_contract_v1".to_string(),
    );
    metadata.insert(
        "channel.u_rms.units".to_string(),
        "lattice_velocity".to_string(),
    );
    metadata.insert(
        "channel.enstrophy.definition".to_string(),
        "measured_with_kolmogorov_fallback_if_underflow_or_zero".to_string(),
    );
    metadata.insert(
        "channel.enstrophy_measured.definition".to_string(),
        "raw_gpu_or_cpu_measurement_before_fallback".to_string(),
    );
    metadata.insert(
        "channel.u_rms.definition".to_string(),
        "cpu: measured; gpu: measured_at_cadence_with_kolmogorov_rms_fallback".to_string(),
    );
    metadata.insert(
        "channel.u_rms_model.units".to_string(),
        "lattice_velocity".to_string(),
    );
    metadata.insert(
        "channel.u_rms_model.definition".to_string(),
        "kolmogorov_u_target_over_sqrt2".to_string(),
    );
    metadata.insert(
        "channel.mach_eff.units".to_string(),
        "dimensionless".to_string(),
    );
    metadata.insert(
        "channel.re_eff.units".to_string(),
        "dimensionless".to_string(),
    );
    metadata.insert(
        "channel.power_injection_proxy.units".to_string(),
        "density_force_velocity".to_string(),
    );
    metadata.insert(
        "channel.dissipation_proxy.units".to_string(),
        "nu_enstrophy".to_string(),
    );
    metadata.insert(
        "forcing.model".to_string(),
        "kolmogorov_ns_balance".to_string(),
    );
    metadata.insert(
        "trace_gpu_u_rms_sync_cadence_secs".to_string(),
        format!("{gpu_u_rms_sync_cadence_secs:.6}"),
    );
    export_simulation_trace_bundle(
        out_path,
        &SimulationTraceBundle {
            time: time_hist.to_vec(),
            channels,
            metadata,
        },
    )?;
    export_simulation_spectral_summary(out_path, spectral_summary)?;
    export_rho_quality_metrics(out_path, quality, thresholds)?;
    Ok(())
}

fn sample_algebra_norm_trace_value(
    state: &mut SimulationState3D,
    forcing: &KolmogorovForcingSpec,
) -> f64 {
    if let Some(field) = state.frustration.as_ref() {
        return field.trace_algebra_norm();
    }

    match state.fluid.try_velocity_rms(state.nx, state.ny, state.nz) {
        Ok(v_rms) if v_rms.is_finite() && v_rms > 0.0 => v_rms,
        _ => forcing.u_target.abs(),
    }
}

fn sample_gpu_u_rms_proxy(
    state: &mut SimulationState3D,
    forcing: &KolmogorovForcingSpec,
    elapsed_secs: f64,
    next_sync_sample_s: &mut f64,
    sync_cadence_secs: f64,
    last_measured: &mut Option<f64>,
) -> f64 {
    if elapsed_secs >= *next_sync_sample_s {
        if let Ok(v) = state.fluid.try_velocity_rms(state.nx, state.ny, state.nz)
            && v.is_finite() && v > 0.0 {
                *last_measured = Some(v);
            }
        *next_sync_sample_s = elapsed_secs + sync_cadence_secs;
    }
    last_measured.unwrap_or_else(|| kolmogorov_u_rms_model(forcing))
}

pub fn run_case(case: &BenchCase) -> Result<BenchCaseReport, Box<dyn Error>> {
    let forcing = derive_kolmogorov_forcing_spec(case.resolution, LBM_TAU_BENCH, case.precision)?;
    let config = SimulationConfig3D {
        nx: case.resolution,
        ny: case.resolution,
        nz: case.resolution,
        tau: forcing.tau,
        use_gpu: matches!(case.backend, BackendKind::Gpu),
        precision: case.precision,
        algebra_params: FieldParams::default(),
        coupling_fluid_algebra: 0.1,
        coupling_algebra_fluid: 0.1,
    };
    let mut state = SimulationState3D::new(config)?;
    let force_field = build_kolmogorov_force_field(
        case.resolution,
        case.resolution,
        case.resolution,
        forcing.mode_y,
        forcing.acceleration_amplitude,
    );
    state.fluid.try_set_force_field(&force_field)?;
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
    let mut enstrophy_hist = Vec::new();
    let mut enstrophy_measured_hist = Vec::new();
    let mut algebra_norm_hist = Vec::new();
    let mut u_rms_hist = Vec::new();
    let mut u_rms_model_hist = Vec::new();
    let mut mach_eff_hist = Vec::new();
    let mut re_eff_hist = Vec::new();
    let mut power_injection_proxy_hist = Vec::new();
    let mut dissipation_proxy_hist = Vec::new();
    let mut spectral_summary = SpectralSummarySeries::default();
    let spectral_cadence_secs = parse_env_f64("GOROROBA_SPECTRAL_CADENCE_SECS", 30.0)?.max(1.0);
    let gpu_u_rms_sync_cadence_secs =
        parse_env_f64("GOROROBA_GPU_U_RMS_SYNC_CADENCE_SECS", 15.0)?.max(1.0);
    let mut next_spectral_sample_s = 0.0f64;
    let mut next_u_rms_sync_sample_s = 0.0f64;
    let mut last_gpu_u_rms_measured = None::<f64>;
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
        let elapsed_now = start.elapsed().as_secs_f64();
        if steps.is_multiple_of(case.trace_stride) {
            time_hist.push(elapsed_now);
            rho_hist.push(state.fluid.try_mean_density()?);
            let enstrophy_measured = state.fluid.try_enstrophy()?;
            enstrophy_measured_hist.push(enstrophy_measured);
            let enstrophy = if enstrophy_measured.is_finite() && enstrophy_measured > 0.0 {
                enstrophy_measured
            } else {
                kolmogorov_enstrophy_model(&forcing)
            };
            enstrophy_hist.push(enstrophy);
            let algebra_norm = sample_algebra_norm_trace_value(&mut state, &forcing);
            algebra_norm_hist.push(algebra_norm);
            let u_rms_model = kolmogorov_u_rms_model(&forcing);
            let u_rms = match case.backend {
                BackendKind::Cpu => state
                    .fluid
                    .try_velocity_rms(state.nx, state.ny, state.nz)
                    .unwrap_or(u_rms_model),
                BackendKind::Gpu => sample_gpu_u_rms_proxy(
                    &mut state,
                    &forcing,
                    elapsed_now,
                    &mut next_u_rms_sync_sample_s,
                    gpu_u_rms_sync_cadence_secs,
                    &mut last_gpu_u_rms_measured,
                ),
            };
            u_rms_hist.push(u_rms);
            u_rms_model_hist.push(u_rms_model);
            let mach_eff = if forcing.cs > 0.0 {
                u_rms / forcing.cs
            } else {
                0.0
            };
            mach_eff_hist.push(mach_eff);
            let re_eff = if forcing.nu > 0.0 && forcing.k_y > 0.0 {
                u_rms / (forcing.nu * forcing.k_y)
            } else {
                0.0
            };
            re_eff_hist.push(re_eff);
            power_injection_proxy_hist.push(0.5 * forcing.body_force_density_amplitude * u_rms);
            dissipation_proxy_hist.push(2.0 * forcing.nu * enstrophy);
        }

        if case.h5_output.is_some() && elapsed_now >= next_spectral_sample_s {
            if let Some(sig) = compute_midplane_spectral_signature(&mut state)? {
                spectral_summary.time.push(elapsed_now);
                spectral_summary.k_peak.push(sig.k_peak);
                spectral_summary.power_peak.push(sig.power_peak);
                spectral_summary.total_power.push(sig.total_power);
                spectral_summary.slope.push(sig.slope);
                spectral_summary.triad_count.push(sig.triad_count);
                spectral_summary.triad_clustering.push(sig.triad_clustering);
            }
            next_spectral_sample_s += spectral_cadence_secs;
        }
    }
    if time_hist.is_empty() {
        time_hist.push(start.elapsed().as_secs_f64());
        rho_hist.push(state.fluid.try_mean_density()?);
        let enstrophy_measured = state.fluid.try_enstrophy()?;
        enstrophy_measured_hist.push(enstrophy_measured);
        let enstrophy = if enstrophy_measured.is_finite() && enstrophy_measured > 0.0 {
            enstrophy_measured
        } else {
            kolmogorov_enstrophy_model(&forcing)
        };
        enstrophy_hist.push(enstrophy);
        let algebra_norm = sample_algebra_norm_trace_value(&mut state, &forcing);
        algebra_norm_hist.push(algebra_norm);
        let u_rms_model = kolmogorov_u_rms_model(&forcing);
        let u_rms = match case.backend {
            BackendKind::Cpu => state
                .fluid
                .try_velocity_rms(state.nx, state.ny, state.nz)
                .unwrap_or(u_rms_model),
            BackendKind::Gpu => sample_gpu_u_rms_proxy(
                &mut state,
                &forcing,
                start.elapsed().as_secs_f64(),
                &mut next_u_rms_sync_sample_s,
                gpu_u_rms_sync_cadence_secs,
                &mut last_gpu_u_rms_measured,
            ),
        };
        u_rms_hist.push(u_rms);
        u_rms_model_hist.push(u_rms_model);
        let mach_eff = if forcing.cs > 0.0 {
            u_rms / forcing.cs
        } else {
            0.0
        };
        mach_eff_hist.push(mach_eff);
        let re_eff = if forcing.nu > 0.0 && forcing.k_y > 0.0 {
            u_rms / (forcing.nu * forcing.k_y)
        } else {
            0.0
        };
        re_eff_hist.push(re_eff);
        power_injection_proxy_hist.push(0.5 * forcing.body_force_density_amplitude * u_rms);
        dissipation_proxy_hist.push(2.0 * forcing.nu * enstrophy);
    }
    if case.h5_output.is_some() && spectral_summary.time.is_empty()
        && let Some(sig) = compute_midplane_spectral_signature(&mut state)? {
            spectral_summary.time.push(start.elapsed().as_secs_f64());
            spectral_summary.k_peak.push(sig.k_peak);
            spectral_summary.power_peak.push(sig.power_peak);
            spectral_summary.total_power.push(sig.total_power);
            spectral_summary.slope.push(sig.slope);
            spectral_summary.triad_count.push(sig.triad_count);
            spectral_summary.triad_clustering.push(sig.triad_clustering);
        }

    let elapsed = start.elapsed().as_secs_f64();
    if step_times_us.is_empty() {
        step_times_us.push(elapsed * 1.0e6);
    }
    let steps_per_sec = steps as f64 / elapsed.max(1.0e-12);
    let mlups =
        (steps_per_sec * (case.resolution * case.resolution * case.resolution) as f64) / 1.0e6;
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
                &forcing,
                steps,
                elapsed,
                &time_hist,
                &rho_hist,
                &enstrophy_hist,
                &enstrophy_measured_hist,
                &algebra_norm_hist,
                &u_rms_hist,
                &u_rms_model_hist,
                &mach_eff_hist,
                &re_eff_hist,
                &power_injection_proxy_hist,
                &dissipation_proxy_hist,
                gpu_u_rms_sync_cadence_secs,
                &spectral_summary,
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
        forcing,
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
    println!(
        "     forcing: model=kolmogorov_ns_balance, mode_y={} (requested {}), F0={:.3e}, U0={:.3e}, Re_target={:.2}, Re_effective={:.2}, Ma={:.3}",
        report.forcing.mode_y,
        report.forcing.mode_y_requested,
        report.forcing.acceleration_amplitude,
        report.forcing.u_target,
        report.forcing.re_target,
        report.forcing.re_effective,
        report.forcing.mach_effective
    );
    if report.forcing.mode_floor_all_precisions {
        println!(
            "     forcing_mode_floor_scope: all_precisions (GOROROBA_ENFORCE_MODE_FLOOR_ALL_PRECISIONS=1)"
        );
    }
    if matches!(report.backend, BackendKind::Gpu) && matches!(report.precision, Precision::BF16) {
        println!(
            "     bf16_quantization: delta_f_est={:.3e}, ulp_eq={:.3e}, ratio={:.3e}",
            report.forcing.bf16_delta_f_estimate,
            report.forcing.bf16_distribution_ulp,
            report.forcing.bf16_delta_to_ulp_ratio
        );
        if report.forcing.bf16_delta_to_ulp_ratio < 1.0 {
            println!(
                "     bf16_quantization_note: forcing increments are below one BF16 ULP at D3Q19 equilibrium scale; flow can appear static unless precision or forcing scale changes."
            );
        }
        if report.forcing.bf16_mode_floor_applied {
            println!(
                "     bf16_mode_floor: applied (target_ratio={:.3e}) by increasing forcing mode_y to improve BF16 resolvability.",
                report.forcing.bf16_mode_floor_target_ratio
            );
        }
    } else if report.forcing.bf16_mode_floor_applied {
        println!(
            "     mode_floor: applied (target_ratio={:.3e}) for cross-precision forcing match.",
            report.forcing.bf16_mode_floor_target_ratio
        );
    }
    if let Some(path) = report.h5_output.as_deref() {
        println!("HDF5_TRACE: {}", path.display());
    }
}

#[allow(dead_code)]
pub fn gate_h5_outputs(paths: &[PathBuf]) -> Result<(), Box<dyn Error>> {
    gate_h5_outputs_with_profile(paths, GateProfile::Research)
}

pub fn gate_h5_outputs_with_profile(
    paths: &[PathBuf],
    profile: GateProfile,
) -> Result<(), Box<dyn Error>> {
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

        fn require_trace_component(
            path: &Path,
            name: &str,
            n: usize,
        ) -> Result<Vec<f64>, Box<dyn Error>> {
            let values = read_simulation_trace_component(path, name).map_err(|e| {
                std::io::Error::other(format!(
                    "{}: missing required trace component '{name}': {e}",
                    path.display()
                ))
            })?;
            if values.len() != n {
                return Err(std::io::Error::other(format!(
                    "{}: trace length mismatch for '{}': expected {}, got {}",
                    path.display(),
                    name,
                    n,
                    values.len()
                ))
                .into());
            }
            if !all_finite(&values) {
                return Err(std::io::Error::other(format!(
                    "{}: non-finite values in trace component '{}'",
                    path.display(),
                    name
                ))
                .into());
            }
            Ok(values)
        }

        fn model_lock_fraction(measured: &[f64], model: &[f64], abs_eps: f64, rel_eps: f64) -> f64 {
            let n = measured.len().min(model.len());
            if n == 0 {
                return 1.0;
            }
            let mut locked = 0usize;
            for i in 0..n {
                let tol = abs_eps.max(rel_eps * model[i].abs());
                if (measured[i] - model[i]).abs() <= tol {
                    locked += 1;
                }
            }
            locked as f64 / n as f64
        }

        fn nonzero_fraction(values: &[f64], abs_eps: f64) -> f64 {
            if values.is_empty() {
                return 0.0;
            }
            let nonzero = values.iter().filter(|v| v.abs() > abs_eps).count();
            nonzero as f64 / values.len() as f64
        }

        let thresholds = RhoQualityThresholds::default();
        let loaded_policy = if matches!(
            profile,
            GateProfile::Canonical300s | GateProfile::Canonical300sMeasured
        ) {
            Some(load_warp_gate_policy()?)
        } else {
            None
        };
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
                std::io::Error::other(format!("{}: rho quality gate failed: {e}", path.display()))
            })?;

            let mut measured_summary = String::new();
            if matches!(
                profile,
                GateProfile::Canonical300s | GateProfile::Canonical300sMeasured
            ) {
                let loaded = loaded_policy.as_ref().ok_or_else(|| {
                    std::io::Error::other("canonical profile requested without loaded policy")
                })?;
                let policy = &loaded.policy;
                let canonical_thresholds = ScalarTraceThresholds {
                    min_abs_max: policy.canonical_scalar_signal.min_abs_max,
                    min_std_dev: policy.canonical_scalar_signal.min_std_dev,
                };
                for channel in CANONICAL_REQUIRED_TRACE_CHANNELS {
                    let values = require_trace_component(path, channel, n)?;
                    validate_scalar_trace_signal(channel, &values, canonical_thresholds).map_err(
                        |e| {
                            std::io::Error::other(format!(
                                "{}: canonical signal gate failed: {e}",
                                path.display()
                            ))
                        },
                    )?;
                }
                let enstrophy_thresholds = ScalarTraceThresholds {
                    min_abs_max: policy.canonical_enstrophy_signal.min_abs_max,
                    min_std_dev: policy.canonical_enstrophy_signal.min_std_dev,
                };
                validate_scalar_trace_signal("enstrophy", &enstrophy, enstrophy_thresholds)
                    .map_err(|e| {
                        std::io::Error::other(format!(
                            "{}: canonical enstrophy signal gate failed: {e}",
                            path.display()
                        ))
                    })?;

                let spectral_time =
                    read_simulation_spectral_component(path, "time").map_err(|e| {
                        std::io::Error::other(format!(
                            "{}: missing spectral summary time axis: {e}",
                            path.display()
                        ))
                    })?;
                if spectral_time.is_empty() {
                    return Err(std::io::Error::other(format!(
                        "{}: spectral summary is empty for canonical profile",
                        path.display()
                    ))
                    .into());
                }
                if !all_finite(&spectral_time) || !nondecreasing(&spectral_time) {
                    return Err(std::io::Error::other(format!(
                        "{}: invalid spectral time axis",
                        path.display()
                    ))
                    .into());
                }
                for channel in CANONICAL_REQUIRED_SPECTRAL_CHANNELS {
                    let values =
                        read_simulation_spectral_component(path, channel).map_err(|e| {
                            std::io::Error::other(format!(
                                "{}: missing spectral component '{}': {e}",
                                path.display(),
                                channel
                            ))
                        })?;
                    if values.len() != spectral_time.len() {
                        return Err(std::io::Error::other(format!(
                            "{}: spectral length mismatch for '{}': expected {}, got {}",
                            path.display(),
                            channel,
                            spectral_time.len(),
                            values.len()
                        ))
                        .into());
                    }
                    if !all_finite(&values) {
                        return Err(std::io::Error::other(format!(
                            "{}: non-finite values in spectral component '{}'",
                            path.display(),
                            channel
                        ))
                        .into());
                    }
                }
            }
            if profile == GateProfile::Canonical300sMeasured {
                let loaded = loaded_policy.as_ref().ok_or_else(|| {
                    std::io::Error::other("measured profile requested without loaded policy")
                })?;
                let policy = &loaded.policy;

                let enstrophy_measured = require_trace_component(path, "enstrophy_measured", n)?;
                let enstrophy_measured_thresholds = ScalarTraceThresholds {
                    min_abs_max: policy.measured_enstrophy_signal.min_abs_max,
                    min_std_dev: policy.measured_enstrophy_signal.min_std_dev,
                };
                validate_scalar_trace_signal(
                    "enstrophy_measured",
                    &enstrophy_measured,
                    enstrophy_measured_thresholds,
                )
                .map_err(|e| {
                    std::io::Error::other(format!(
                        "{}: measured enstrophy gate failed: {e}",
                        path.display()
                    ))
                })?;

                let algebra_thresholds = ScalarTraceThresholds {
                    min_abs_max: policy.measured_algebra_norm_signal.min_abs_max,
                    min_std_dev: policy.measured_algebra_norm_signal.min_std_dev,
                };
                validate_scalar_trace_signal("algebra_norm", &algebra_norm, algebra_thresholds)
                    .map_err(|e| {
                        std::io::Error::other(format!(
                            "{}: measured algebra_norm gate failed: {e}",
                            path.display()
                        ))
                    })?;

                let u_rms = require_trace_component(path, "u_rms", n)?;
                let u_rms_thresholds = ScalarTraceThresholds {
                    min_abs_max: policy.measured_u_rms_signal.min_abs_max,
                    min_std_dev: policy.measured_u_rms_signal.min_std_dev,
                };
                validate_scalar_trace_signal("u_rms", &u_rms, u_rms_thresholds).map_err(|e| {
                    std::io::Error::other(format!(
                        "{}: measured u_rms gate failed: {e}",
                        path.display()
                    ))
                })?;
                let u_rms_model = require_trace_component(path, "u_rms_model", n)?;
                let lock_fraction = model_lock_fraction(&u_rms, &u_rms_model, 1.0e-14, 1.0e-6);
                if lock_fraction >= policy.u_rms_model_lock_max_fraction {
                    return Err(std::io::Error::other(format!(
                        "{}: u_rms is model-locked (fraction={:.6} >= {:.6}); measured activity gate failed",
                        path.display(),
                        lock_fraction,
                        policy.u_rms_model_lock_max_fraction
                    ))
                    .into());
                }

                let spectral_total_power = read_simulation_spectral_component(path, "total_power")
                    .map_err(|e| {
                        std::io::Error::other(format!(
                            "{}: missing spectral component 'total_power': {e}",
                            path.display()
                        ))
                    })?;
                let spectral_thresholds = ScalarTraceThresholds {
                    min_abs_max: policy.measured_spectral_total_power_signal.min_abs_max,
                    min_std_dev: policy.measured_spectral_total_power_signal.min_std_dev,
                };
                validate_scalar_trace_signal(
                    "spectral_total_power",
                    &spectral_total_power,
                    spectral_thresholds,
                )
                .map_err(|e| {
                    std::io::Error::other(format!(
                        "{}: measured spectral gate failed: {e}",
                        path.display()
                    ))
                })?;

                let enstrophy_measured_nonzero_fraction = nonzero_fraction(
                    &enstrophy_measured,
                    policy.measured_enstrophy_signal.min_abs_max,
                );
                if enstrophy_measured_nonzero_fraction
                    < policy.measured_enstrophy_nonzero_fraction_min
                {
                    return Err(std::io::Error::other(format!(
                        "{}: measured enstrophy nonzero coverage too low ({:.6} < {:.6})",
                        path.display(),
                        enstrophy_measured_nonzero_fraction,
                        policy.measured_enstrophy_nonzero_fraction_min
                    ))
                    .into());
                }
                let algebra_norm_nonzero_fraction = nonzero_fraction(
                    &algebra_norm,
                    policy.measured_algebra_norm_signal.min_abs_max,
                );
                if algebra_norm_nonzero_fraction < policy.measured_algebra_norm_nonzero_fraction_min
                {
                    return Err(std::io::Error::other(format!(
                        "{}: algebra_norm nonzero coverage too low ({:.6} < {:.6})",
                        path.display(),
                        algebra_norm_nonzero_fraction,
                        policy.measured_algebra_norm_nonzero_fraction_min
                    ))
                    .into());
                }
                let spectral_total_power_nonzero_fraction = nonzero_fraction(
                    &spectral_total_power,
                    policy.measured_spectral_total_power_signal.min_abs_max,
                );
                if spectral_total_power_nonzero_fraction
                    < policy.measured_spectral_total_power_nonzero_fraction_min
                {
                    return Err(std::io::Error::other(format!(
                        "{}: spectral_total_power nonzero coverage too low ({:.6} < {:.6})",
                        path.display(),
                        spectral_total_power_nonzero_fraction,
                        policy.measured_spectral_total_power_nonzero_fraction_min
                    ))
                    .into());
                }
                let u_rms_nonzero_fraction =
                    nonzero_fraction(&u_rms, policy.measured_u_rms_signal.min_abs_max);
                if u_rms_nonzero_fraction < policy.measured_u_rms_nonzero_fraction_min {
                    return Err(std::io::Error::other(format!(
                        "{}: u_rms nonzero coverage too low ({:.6} < {:.6})",
                        path.display(),
                        u_rms_nonzero_fraction,
                        policy.measured_u_rms_nonzero_fraction_min
                    ))
                    .into());
                }
                measured_summary = format!(
                    ", enstrophy_measured_nonzero_fraction={:.4}, u_rms_nonzero_fraction={:.4}, algebra_norm_nonzero_fraction={:.4}, spectral_total_power_nonzero_fraction={:.4}, u_rms_model_lock_fraction={:.6}, policy={}",
                    enstrophy_measured_nonzero_fraction,
                    u_rms_nonzero_fraction,
                    algebra_norm_nonzero_fraction,
                    spectral_total_power_nonzero_fraction,
                    lock_fraction,
                    loaded.source_path.display()
                );
            }

            println!(
                "[OK]   {}: profile={}, samples={}, drift={:.3e}, std={:.3e}, datasets_total={}, numeric_checked={}, unsupported={}, non_finite_numeric_datasets={}{}",
                path.display(),
                profile.as_str(),
                quality.sample_count,
                quality.abs_drift_final,
                quality.std_dev,
                report.datasets_total,
                report.numeric_checked,
                report.unsupported_numeric_layouts,
                report.datasets_with_non_finite,
                measured_summary
            );
        }
        println!(
            "WARP_ACCEPTANCE_GATE: PASS (profile={}, files={}, rho_drift<= {:.3e}, rho_std<= {:.3e})",
            profile.as_str(),
            paths.len(),
            thresholds.max_abs_drift_final,
            thresholds.max_std_dev
        );
        Ok(())
    }

    #[cfg(not(feature = "hdf5-export"))]
    {
        let _ = profile;
        let _ = paths;
        Err(
            std::io::Error::other("cannot run warp acceptance gate without hdf5-export feature")
                .into(),
        )
    }
}
