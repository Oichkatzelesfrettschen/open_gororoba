#[cfg(feature = "hdf5-export")]
use data_core::hdf5_export::{
    read_simulation_spectral_component, read_simulation_trace_component,
    scan_hdf5_numeric_datasets, NumericDatasetScanStatus,
};
#[cfg(feature = "hdf5-export")]
use data_core::quality::{
    validate_rho_trace, validate_scalar_trace_signal, RhoQualityThresholds, ScalarTraceThresholds,
};
#[cfg(feature = "hdf5-export")]
use std::collections::BTreeSet;
use std::error::Error;
#[cfg(feature = "hdf5-export")]
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
#[cfg(feature = "hdf5-export")]
struct CliArgs {
    allow_empty: bool,
    profile: GateProfile,
    inputs: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(feature = "hdf5-export")]
enum GateProfile {
    Smoke,
    Research,
    Canonical300s,
    Canonical300sMeasured,
}

#[cfg(feature = "hdf5-export")]
impl GateProfile {
    fn parse(raw: &str) -> Result<Self, Box<dyn Error>> {
        match raw {
            "smoke" => Ok(Self::Smoke),
            "research" => Ok(Self::Research),
            "canonical_300s" | "canonical-300s" => Ok(Self::Canonical300s),
            "canonical_300s_measured" | "canonical-300s-measured" | "canonical_300s_strict"
            | "canonical-300s-strict" => Ok(Self::Canonical300sMeasured),
            other => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "invalid profile '{other}', expected smoke|research|canonical_300s|canonical_300s_measured"
                ),
            )
            .into()),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Smoke => "smoke",
            Self::Research => "research",
            Self::Canonical300s => "canonical_300s",
            Self::Canonical300sMeasured => "canonical_300s_measured",
        }
    }
}

#[cfg(feature = "hdf5-export")]
fn parse_args() -> Result<CliArgs, Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut allow_empty = false;
    let mut profile = GateProfile::Research;
    let mut inputs = Vec::new();
    let mut idx = 0usize;
    while idx < args.len() {
        let arg = &args[idx];
        if arg == "--allow-empty" {
            allow_empty = true;
            idx += 1;
            continue;
        }
        if arg == "--profile" {
            let value = args.get(idx + 1).ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "--profile requires a value",
                )
            })?;
            profile = GateProfile::parse(value)?;
            idx += 2;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--profile=") {
            profile = GateProfile::parse(value)?;
            idx += 1;
            continue;
        } else if arg == "--help" || arg == "-h" {
            println!(
                "Usage: warp-acceptance-gate [--allow-empty] [--profile smoke|research|canonical_300s|canonical_300s_measured] <h5-path-or-glob> [more paths/globs]"
            );
            println!(
                "Runs both simulation-trace checks and recursive numeric HDF5 checks in one fail-closed gate."
            );
            std::process::exit(0);
        } else {
            inputs.push(arg.clone());
            idx += 1;
            continue;
        }
    }
    if inputs.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "expected at least one path or glob pattern",
        )
        .into());
    }
    Ok(CliArgs {
        allow_empty,
        profile,
        inputs,
    })
}

#[cfg(feature = "hdf5-export")]
fn expand_inputs(inputs: &[String], allow_empty: bool) -> Result<Vec<PathBuf>, Box<dyn Error>> {
    let mut paths = BTreeSet::new();
    for input in inputs {
        if input.contains('*') || input.contains('?') || input.contains('[') {
            let mut matched = false;
            for entry in glob::glob(input)? {
                let path = entry?;
                if path.is_file() {
                    matched = true;
                    paths.insert(path);
                }
            }
            if !matched && !allow_empty {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("glob pattern matched no files: {input}"),
                )
                .into());
            }
        } else {
            let path = PathBuf::from(input);
            if !path.exists() {
                if !allow_empty {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::NotFound,
                        format!("file not found: {}", path.display()),
                    )
                    .into());
                }
            } else if path.is_file() {
                paths.insert(path);
            }
        }
    }
    Ok(paths.into_iter().collect())
}

#[cfg(feature = "hdf5-export")]
fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite())
}

#[cfg(feature = "hdf5-export")]
fn nondecreasing(values: &[f64]) -> bool {
    values.windows(2).all(|w| w[1] >= w[0])
}

#[cfg(feature = "hdf5-export")]
fn require_trace_component(path: &Path, name: &str, n: usize) -> Result<Vec<f64>, Box<dyn Error>> {
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

#[cfg(feature = "hdf5-export")]
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

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();

    #[cfg(feature = "hdf5-export")]
    {
        let cli = parse_args()?;
        let paths = expand_inputs(&cli.inputs, cli.allow_empty)?;
        if paths.is_empty() {
            if cli.allow_empty {
                println!("WARP_ACCEPTANCE_GATE: SKIP (no files matched and --allow-empty set)");
                return Ok(());
            }
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "no files to validate after expansion",
            )
            .into());
        }

        let thresholds = RhoQualityThresholds::default();
        let mut file_count = 0usize;
        let mut datasets_total = 0usize;
        let mut numeric_checked = 0usize;
        let mut unsupported_layouts = 0usize;
        let mut non_finite_numeric_datasets = 0usize;

        for path in &paths {
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
            let rho_quality = validate_rho_trace(&rho, thresholds).map_err(|e| {
                std::io::Error::other(format!("{}: rho quality gate failed: {e}", path.display()))
            })?;

            if matches!(
                cli.profile,
                GateProfile::Canonical300s | GateProfile::Canonical300sMeasured
            ) {
                let nonzero_thresholds = ScalarTraceThresholds {
                    // Canonical traces can be near-steady for some physically valid
                    // channels (e.g., power injection proxy in quasi-laminar regimes),
                    // so we require finite + non-zero magnitude and avoid mandatory
                    // variance for those channels.
                    min_abs_max: 1.0e-14,
                    min_std_dev: 0.0,
                };
                let required_channels = [
                    "u_rms",
                    "mach_eff",
                    "re_eff",
                    "power_injection_proxy",
                    "dissipation_proxy",
                ];
                for channel in required_channels {
                    let values = require_trace_component(path, channel, n)?;
                    validate_scalar_trace_signal(channel, &values, nonzero_thresholds).map_err(
                        |e| {
                            std::io::Error::other(format!(
                                "{}: canonical signal gate failed: {e}",
                                path.display()
                            ))
                        },
                    )?;
                }
                let enstrophy_thresholds = ScalarTraceThresholds {
                    min_abs_max: 1.0e-20,
                    min_std_dev: 0.0,
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
                for channel in [
                    "k_peak",
                    "total_power",
                    "slope",
                    "triad_count",
                    "triad_clustering",
                ] {
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
            if cli.profile == GateProfile::Canonical300sMeasured {
                let measured_thresholds = ScalarTraceThresholds {
                    min_abs_max: 1.0e-20,
                    min_std_dev: 0.0,
                };
                let enstrophy_measured = require_trace_component(path, "enstrophy_measured", n)?;
                validate_scalar_trace_signal(
                    "enstrophy_measured",
                    &enstrophy_measured,
                    measured_thresholds,
                )
                .map_err(|e| {
                    std::io::Error::other(format!(
                        "{}: measured enstrophy gate failed: {e}",
                        path.display()
                    ))
                })?;

                let algebra_thresholds = ScalarTraceThresholds {
                    min_abs_max: 1.0e-20,
                    min_std_dev: 0.0,
                };
                validate_scalar_trace_signal("algebra_norm", &algebra_norm, algebra_thresholds)
                    .map_err(|e| {
                        std::io::Error::other(format!(
                            "{}: measured algebra_norm gate failed: {e}",
                            path.display()
                        ))
                    })?;

                let u_rms = require_trace_component(path, "u_rms", n)?;
                let u_rms_model = require_trace_component(path, "u_rms_model", n)?;
                let lock_fraction = model_lock_fraction(&u_rms, &u_rms_model, 1.0e-14, 1.0e-6);
                if lock_fraction >= 0.999 {
                    return Err(std::io::Error::other(format!(
                        "{}: u_rms is model-locked (fraction={:.6}); measured activity gate failed",
                        path.display(),
                        lock_fraction
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
                validate_scalar_trace_signal(
                    "spectral_total_power",
                    &spectral_total_power,
                    ScalarTraceThresholds {
                        min_abs_max: 1.0e-24,
                        min_std_dev: 0.0,
                    },
                )
                .map_err(|e| {
                    std::io::Error::other(format!(
                        "{}: measured spectral gate failed: {e}",
                        path.display()
                    ))
                })?;
            }

            file_count += 1;
            datasets_total += report.datasets_total;
            numeric_checked += report.numeric_checked;
            unsupported_layouts += report.unsupported_numeric_layouts;
            non_finite_numeric_datasets += report.datasets_with_non_finite;
            println!(
                "[OK]   {}: profile={}, n={}, rho_final={:.6}, rho_drift={:.3e}, rho_std={:.3e}, datasets_total={}, numeric_checked={}, unsupported={}, non_finite_numeric_datasets={}",
                path.display(),
                cli.profile.as_str(),
                n,
                rho_quality.final_value,
                rho_quality.abs_drift_final,
                rho_quality.std_dev,
                report.datasets_total,
                report.numeric_checked,
                report.unsupported_numeric_layouts,
                report.datasets_with_non_finite
            );
        }

        println!(
            "WARP_ACCEPTANCE_GATE: PASS (profile={}, files={}, datasets_total={}, numeric_checked={}, unsupported={}, non_finite_numeric_datasets={}, rho_drift<= {:.3e}, rho_std<= {:.3e})",
            cli.profile.as_str(),
            file_count,
            datasets_total,
            numeric_checked,
            unsupported_layouts,
            non_finite_numeric_datasets,
            thresholds.max_abs_drift_final,
            thresholds.max_std_dev
        );
        Ok(())
    }

    #[cfg(not(feature = "hdf5-export"))]
    {
        Err(std::io::Error::other(
            "warp-acceptance-gate requires hdf5-export feature: cargo run -p gororoba_cli --features hdf5-export --bin warp-acceptance-gate -- <paths>",
        )
        .into())
    }
}
