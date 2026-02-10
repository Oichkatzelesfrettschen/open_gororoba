//! Emit attractor regression metrics for C-590 as CSV.
//!
//! This binary computes frustration ratios for selected CD dimensions and writes
//! deterministic CSV rows with per-dimension runtime.

use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use algebra_core::analysis::boxkites::compute_frustration_ratio;
use clap::Parser;

const SCHEMA_VERSION: &str = "c590_attractor_sweep_v1";
const CSV_HEADER: &str =
    "schema_version,profile_tag,dim,frustration_ratio,delta_to_three_eighths,elapsed_seconds,run_unix_seconds\n";

#[derive(Parser, Debug)]
#[command(name = "c590-attractor-sweep")]
#[command(about = "Compute C-590 attractor ratios and emit CSV rows")]
struct Args {
    /// Comma-separated dimensions to evaluate (powers of two).
    #[arg(long, default_value = "16,32,64,128,256,512,1024")]
    dims: String,

    /// Free-form profile label stored in CSV (for example: debug, release).
    #[arg(long, default_value = "unknown")]
    profile_tag: String,

    /// Output CSV path. If omitted, CSV is printed to stdout.
    #[arg(long)]
    output: Option<PathBuf>,

    /// Append rows into an existing output file (header is written only once).
    #[arg(long, default_value_t = false)]
    append: bool,
}

fn parse_dims(raw: &str) -> Result<Vec<usize>, String> {
    let mut dims = Vec::new();
    let mut seen = HashSet::new();
    for token in raw.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        let dim = token
            .parse::<usize>()
            .map_err(|_| format!("invalid dimension token '{token}'"))?;
        if !dim.is_power_of_two() {
            return Err(format!("dimension {dim} is not a power of two"));
        }
        // Policy: preserve first occurrence order while removing duplicates.
        if seen.insert(dim) {
            dims.push(dim);
        }
    }
    if dims.is_empty() {
        return Err("no valid dimensions parsed from --dims".to_string());
    }

    for &dim in &dims {
        debug_assert!(dim.is_power_of_two());
    }
    Ok(dims)
}

fn build_csv(profile_tag: &str, dims: &[usize], run_unix_seconds: u64) -> String {
    let target = 0.375_f64;

    let mut csv = String::from(CSV_HEADER);

    for dim in dims {
        let started = Instant::now();
        let res = compute_frustration_ratio(*dim);
        let elapsed = started.elapsed().as_secs_f64();
        let delta = res.frustration_ratio - target;

        csv.push_str(&format!(
            "{},{},{},{:.12},{:.12},{:.6},{}\n",
            SCHEMA_VERSION, profile_tag, dim, res.frustration_ratio, delta, elapsed, run_unix_seconds
        ));
    }
    csv
}

fn append_csv(path: &PathBuf, csv: &str) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create output directory {}: {e}", parent.display()))?;
    }

    let body_lines: Vec<&str> = csv.lines().skip(1).collect();
    if path.exists()
        && path
            .metadata()
            .map_err(|e| format!("failed to inspect output file {}: {e}", path.display()))?
            .len()
            > 0
    {
        let mut f = OpenOptions::new()
            .append(true)
            .open(path)
            .map_err(|e| format!("failed to open {} for append: {e}", path.display()))?;
        for line in body_lines {
            f.write_all(line.as_bytes())
                .map_err(|e| format!("failed to append row to {}: {e}", path.display()))?;
            f.write_all(b"\n")
                .map_err(|e| format!("failed to append newline to {}: {e}", path.display()))?;
        }
        Ok(())
    } else {
        fs::write(path, csv).map_err(|e| format!("failed to write {}: {e}", path.display()))
    }
}

fn run(args: Args) -> Result<(), String> {
    let dims = parse_dims(&args.dims)?;
    let run_unix_seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("System clock before UNIX_EPOCH")
        .as_secs();
    let csv = build_csv(&args.profile_tag, &dims, run_unix_seconds);

    if let Some(path) = args.output {
        if args.append {
            append_csv(&path, &csv)?;
        } else {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent).map_err(|e| {
                    format!("failed to create output parent directory {}: {e}", parent.display())
                })?;
            }
            fs::write(&path, csv).map_err(|e| format!("failed to write {}: {e}", path.display()))?;
        }
        eprintln!("Wrote {}", path.display());
    } else {
        print!("{csv}");
    }
    Ok(())
}

fn main() {
    let args = Args::parse();
    if let Err(err) = run(args) {
        eprintln!("Error: {err}");
        std::process::exit(2);
    }
}

#[cfg(test)]
mod tests {
    use super::parse_dims;

    #[test]
    fn parse_dims_rejects_non_power_of_two() {
        let err = parse_dims("16,24,32").expect_err("24 should be rejected");
        assert!(err.contains("not a power of two"));
    }

    #[test]
    fn parse_dims_dedupes_preserves_first_order() {
        let dims = parse_dims("32,16,32,64,16,128").expect("valid dedupe parse");
        assert_eq!(dims, vec![32, 16, 64, 128]);
    }

    #[test]
    fn parse_dims_rejects_empty_input() {
        let err = parse_dims(" , , ").expect_err("empty token stream should fail");
        assert!(err.contains("no valid dimensions"));
    }
}
