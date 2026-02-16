#[cfg(feature = "hdf5-export")]
use data_core::hdf5_export::read_simulation_trace_component;
use data_core::quality::{validate_rho_trace, RhoQualityThresholds};
use std::collections::BTreeSet;
use std::error::Error;
use std::path::PathBuf;

fn expand_inputs(inputs: &[String]) -> Result<Vec<PathBuf>, Box<dyn Error>> {
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
            if !matched {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("glob pattern matched no files: {input}"),
                )
                .into());
            }
        } else {
            let path = PathBuf::from(input);
            if !path.exists() {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::NotFound,
                    format!("file not found: {}", path.display()),
                )
                .into());
            }
            paths.insert(path);
        }
    }
    Ok(paths.into_iter().collect())
}

fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|v| v.is_finite())
}

fn nondecreasing(values: &[f64]) -> bool {
    values.windows(2).all(|w| w[1] >= w[0])
}

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: simulation-trace-gate <h5-path-or-glob> [more paths/globs]");
        eprintln!("Checks /simulation/trace/{{time,rho_mean,enstrophy,algebra_norm}}");
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "expected at least one path or glob pattern",
        )
        .into());
    }

    let paths = expand_inputs(&args)?;
    if paths.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "no files to validate after expansion",
        )
        .into());
    }

    #[cfg(feature = "hdf5-export")]
    {
        let thresholds = RhoQualityThresholds::default();
        let mut ok_count = 0usize;
        for path in &paths {
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
            if !all_finite(&time) || !all_finite(&rho) || !all_finite(&enstrophy) || !all_finite(&algebra_norm) {
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

            ok_count += 1;
            println!(
                "[OK]   {}: n={}, rho_final={:.6}, rho_drift={:.3e}, rho_std={:.3e}, traces_finite=true, time_monotonic=true",
                path.display(),
                n,
                rho_quality.final_value,
                rho_quality.abs_drift_final,
                rho_quality.std_dev
            );
        }
        println!(
            "SIM_TRACE_GATE: PASS ({} files checked; rho thresholds drift<= {:.3e}, std<= {:.3e})",
            ok_count,
            thresholds.max_abs_drift_final,
            thresholds.max_std_dev
        );
        return Ok(());
    }

    #[cfg(not(feature = "hdf5-export"))]
    {
        let _ = paths;
        return Err(std::io::Error::other(
            "simulation-trace-gate requires hdf5-export feature: cargo run -p gororoba_cli --features hdf5-export --bin simulation-trace-gate -- <paths>",
        )
        .into());
    }
}
