#[cfg(feature = "hdf5-export")]
use data_artifacts_core::hdf5_export::read_rho_mean_trace;
#[cfg(feature = "hdf5-export")]
use data_artifacts_core::quality::{RhoQualityThresholds, validate_rho_trace};
use std::{collections::BTreeSet, error::Error, path::PathBuf};

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

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: rho-trace-gate <h5-path-or-glob> [more paths/globs]");
        eprintln!("Example: rho-trace-gate 'data/h5/production/**/*.h5'");
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
            let rho = read_rho_mean_trace(path)?;
            let quality = validate_rho_trace(&rho, thresholds).map_err(|e| {
                std::io::Error::other(format!("rho gate failed for {}: {e}", path.display()))
            })?;
            ok_count += 1;
            println!(
                "[OK]   {}: samples={}, finite={}/{}, final={:.6}, drift={:.3e}, std={:.3e}",
                path.display(),
                quality.sample_count,
                quality.finite_count,
                quality.sample_count,
                quality.final_value,
                quality.abs_drift_final,
                quality.std_dev
            );
        }
        println!(
            "FINITE_GATE: PASS ({} files checked, thresholds: drift<= {:.3e}, std<= {:.3e})",
            ok_count, thresholds.max_abs_drift_final, thresholds.max_std_dev
        );
        Ok(())
    }

    #[cfg(not(feature = "hdf5-export"))]
    {
        let _ = paths;
        Err(std::io::Error::other(
            "rho-trace-gate requires hdf5-export feature: cargo run -p gororoba_cli_data --features hdf5-export --bin rho-trace-gate -- <paths>",
        )
        .into())
    }
}
