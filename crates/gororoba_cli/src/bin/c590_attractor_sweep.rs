//! Emit attractor regression metrics for C-590 as CSV.
//!
//! This binary computes frustration ratios for selected CD dimensions and writes
//! deterministic CSV rows with per-dimension runtime.

use std::fs;
use std::path::PathBuf;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use algebra_core::analysis::boxkites::compute_frustration_ratio;
use clap::Parser;

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
}

fn parse_dims(raw: &str) -> Vec<usize> {
    let dims: Vec<usize> = raw
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();
    assert!(!dims.is_empty(), "No valid dimensions parsed from --dims");
    for &dim in &dims {
        assert!(
            dim.is_power_of_two(),
            "Dimension {} is not a power of two",
            dim
        );
    }
    dims
}

fn main() {
    let args = Args::parse();
    let dims = parse_dims(&args.dims);
    let ts_unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("System clock before UNIX_EPOCH")
        .as_secs();
    let target = 0.375_f64;

    let mut csv = String::from(
        "profile_tag,dim,frustration_ratio,delta_to_three_eighths,elapsed_seconds,run_unix_seconds\n",
    );

    for dim in dims {
        let started = Instant::now();
        let res = compute_frustration_ratio(dim);
        let elapsed = started.elapsed().as_secs_f64();
        let delta = res.frustration_ratio - target;

        csv.push_str(&format!(
            "{},{},{:.12},{:.12},{:.6},{}\n",
            args.profile_tag, dim, res.frustration_ratio, delta, elapsed, ts_unix
        ));
    }

    if let Some(path) = args.output {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("Failed to create output parent directory");
        }
        fs::write(&path, csv).expect("Failed to write CSV output");
        eprintln!("Wrote {}", path.display());
    } else {
        print!("{csv}");
    }
}
