#[cfg(feature = "hdf5-export")]
use data_core::hdf5_export::{
    read_simulation_trace_component, scan_hdf5_numeric_datasets, NumericDatasetScanStatus,
};
#[cfg(feature = "hdf5-export")]
use data_core::quality::{validate_rho_trace, RhoQualityThresholds};
#[cfg(feature = "hdf5-export")]
use std::collections::BTreeSet;
use std::error::Error;
#[cfg(feature = "hdf5-export")]
use std::path::PathBuf;

#[derive(Debug, Clone)]
#[cfg(feature = "hdf5-export")]
struct CliArgs {
    allow_empty: bool,
    inputs: Vec<String>,
}

#[cfg(feature = "hdf5-export")]
fn parse_args() -> Result<CliArgs, Box<dyn Error>> {
    let mut allow_empty = false;
    let mut inputs = Vec::new();
    for arg in std::env::args().skip(1) {
        if arg == "--allow-empty" {
            allow_empty = true;
        } else if arg == "--help" || arg == "-h" {
            println!(
                "Usage: warp-acceptance-gate [--allow-empty] <h5-path-or-glob> [more paths/globs]"
            );
            println!(
                "Runs both simulation-trace checks and recursive numeric HDF5 checks in one fail-closed gate."
            );
            std::process::exit(0);
        } else {
            inputs.push(arg);
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
                std::io::Error::other(format!(
                    "{}: rho quality gate failed: {e}",
                    path.display()
                ))
            })?;

            file_count += 1;
            datasets_total += report.datasets_total;
            numeric_checked += report.numeric_checked;
            unsupported_layouts += report.unsupported_numeric_layouts;
            non_finite_numeric_datasets += report.datasets_with_non_finite;
            println!(
                "[OK]   {}: n={}, rho_final={:.6}, rho_drift={:.3e}, rho_std={:.3e}, datasets_total={}, numeric_checked={}, unsupported={}, non_finite_numeric_datasets={}",
                path.display(),
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
            "WARP_ACCEPTANCE_GATE: PASS (files={}, datasets_total={}, numeric_checked={}, unsupported={}, non_finite_numeric_datasets={}, rho_drift<= {:.3e}, rho_std<= {:.3e})",
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
