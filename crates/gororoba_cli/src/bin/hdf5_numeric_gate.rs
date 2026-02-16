#[cfg(feature = "hdf5-export")]
use data_core::hdf5_export::{scan_hdf5_numeric_datasets, NumericDatasetScanStatus};
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

fn main() -> Result<(), Box<dyn Error>> {
    tracing_subscriber::fmt::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("Usage: hdf5-numeric-gate <h5-path-or-glob> [more paths/globs]");
        eprintln!("Recursively checks all HDF5 datasets and fails on NaN/Inf in numeric datasets.");
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
        let mut checked_files = 0usize;
        for path in &paths {
            let report = scan_hdf5_numeric_datasets(path)?;
            for entry in &report.entries {
                if entry.status == NumericDatasetScanStatus::Checked
                    && (entry.nan_count > 0 || entry.inf_count > 0)
                {
                    return Err(std::io::Error::other(format!(
                        "{}: non-finite numeric dataset at {} (dtype={}, nan={}, inf={})",
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
                        "{}: unsupported numeric dataset layout at {} (dtype={}); fail-closed",
                        path.display(),
                        entry.path,
                        entry.dtype
                    ))
                    .into());
                }
            }
            checked_files += 1;
            println!(
                "[OK]   {}: datasets_total={}, numeric_checked={}, non_numeric_skipped={}, unsupported_numeric_layouts={}, non_finite_numeric_datasets={}",
                path.display(),
                report.datasets_total,
                report.numeric_checked,
                report.non_numeric_skipped,
                report.unsupported_numeric_layouts,
                report.datasets_with_non_finite
            );
        }
        println!(
            "HDF5_NUMERIC_GATE: PASS ({} files checked; recursive numeric NaN/Inf scan)",
            checked_files
        );
        Ok(())
    }

    #[cfg(not(feature = "hdf5-export"))]
    {
        let _ = paths;
        Err(std::io::Error::other(
            "hdf5-numeric-gate requires hdf5-export feature: cargo run -p gororoba_cli --features hdf5-export --bin hdf5-numeric-gate -- <paths>",
        )
        .into())
    }
}
