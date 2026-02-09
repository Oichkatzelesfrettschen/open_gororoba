//! ET discrete billiard phase analysis (Experiment B).
//!
//! Computes band-level billiard dynamics summaries and optional per-strut
//! trajectory detail exports.

use std::path::PathBuf;

use algebra_core::experimental::algebraic_dynamics::{
    et_billiard_phase_sweep, experiment_b_billiard_vs_spectroscopy,
};
use clap::Parser;

#[derive(Parser)]
#[command(name = "et-billiard")]
#[command(about = "Sweep ET discrete billiard dynamics and compare with spectroscopy bands")]
struct Args {
    /// Comma-separated CD levels N (sedenions and above).
    #[arg(long, value_delimiter = ',', default_value = "4,5,6")]
    n_levels: Vec<usize>,

    /// Number of trajectory steps per strut simulation.
    #[arg(long, default_value = "10000")]
    n_steps: usize,

    /// RNG seed for reproducibility.
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Output CSV path for band-level summary rows.
    #[arg(long, default_value = "data/csv/et_billiard_phase_structure.csv")]
    output: PathBuf,

    /// Optional output CSV path for per-strut trajectory metrics.
    #[arg(long)]
    details: Option<PathBuf>,
}

fn main() {
    let args = Args::parse();

    if args.n_levels.is_empty() {
        eprintln!("ERROR: --n-levels must not be empty");
        std::process::exit(1);
    }
    if args.n_steps == 0 {
        eprintln!("ERROR: --n-steps must be > 0");
        std::process::exit(1);
    }
    if args.n_levels.iter().any(|n| *n < 4) {
        eprintln!("ERROR: all --n-levels entries must be >= 4");
        std::process::exit(1);
    }

    if let Some(parent) = args.output.parent() {
        if let Err(err) = std::fs::create_dir_all(parent) {
            eprintln!(
                "ERROR: failed to create output directory {}: {err}",
                parent.display()
            );
            std::process::exit(1);
        }
    }

    let mut summary_wtr = match csv::Writer::from_path(&args.output) {
        Ok(writer) => writer,
        Err(err) => {
            eprintln!(
                "ERROR: failed to open output CSV {}: {err}",
                args.output.display()
            );
            std::process::exit(1);
        }
    };

    if let Err(err) = summary_wtr.write_record([
        "n",
        "band_index",
        "behavior",
        "s_lo",
        "s_hi",
        "mean_entropy_rate",
        "std_entropy_rate",
        "mean_free_path",
        "mean_dmz_transition_rate",
        "mean_coverage",
        "mean_fill_ratio",
        "fill_entropy_correlation",
        "fill_mfp_correlation",
        "n_steps",
        "seed",
    ]) {
        eprintln!("ERROR: failed to write summary CSV header: {err}");
        std::process::exit(1);
    }

    let mut details_wtr = match &args.details {
        Some(path) => {
            if let Some(parent) = path.parent() {
                if let Err(err) = std::fs::create_dir_all(parent) {
                    eprintln!(
                        "ERROR: failed to create details directory {}: {err}",
                        parent.display()
                    );
                    std::process::exit(1);
                }
            }
            match csv::Writer::from_path(path) {
                Ok(mut writer) => {
                    if let Err(err) = writer.write_record([
                        "n",
                        "s",
                        "k",
                        "n_steps",
                        "n_distinct_cells",
                        "coverage",
                        "entropy_rate",
                        "n_reflections",
                        "mean_free_path",
                        "dmz_transition_rate",
                        "n_valid_cells",
                        "n_dmz_cells",
                        "fill_ratio",
                        "seed",
                    ]) {
                        eprintln!("ERROR: failed to write details CSV header: {err}");
                        std::process::exit(1);
                    }
                    Some(writer)
                }
                Err(err) => {
                    eprintln!(
                        "ERROR: failed to open details CSV {}: {err}",
                        path.display()
                    );
                    std::process::exit(1);
                }
            }
        }
        None => None,
    };

    for n in &args.n_levels {
        let sweep = et_billiard_phase_sweep(*n, args.n_steps, args.seed);
        let comparison = experiment_b_billiard_vs_spectroscopy(*n, args.n_steps, args.seed);

        for entry in &comparison.entries {
            if let Err(err) = summary_wtr.write_record([
                comparison.n.to_string(),
                entry.band_index.to_string(),
                entry.behavior.clone(),
                entry.s_lo.to_string(),
                entry.s_hi.to_string(),
                format!("{:.10}", entry.mean_entropy_rate),
                format!("{:.10}", entry.std_entropy_rate),
                format!("{:.10}", entry.mean_free_path),
                format!("{:.10}", entry.mean_dmz_transition_rate),
                format!("{:.10}", entry.mean_coverage),
                format!("{:.10}", entry.mean_fill_ratio),
                format!("{:.10}", sweep.fill_entropy_correlation),
                format!("{:.10}", sweep.fill_mfp_correlation),
                args.n_steps.to_string(),
                args.seed.to_string(),
            ]) {
                eprintln!("ERROR: failed to write summary row for N={n}: {err}");
                std::process::exit(1);
            }
        }

        if let Some(writer) = details_wtr.as_mut() {
            for traj in &sweep.trajectories {
                if let Err(err) = writer.write_record([
                    traj.n.to_string(),
                    traj.s.to_string(),
                    traj.k.to_string(),
                    traj.n_steps.to_string(),
                    traj.n_distinct_cells.to_string(),
                    format!("{:.10}", traj.coverage),
                    format!("{:.10}", traj.entropy_rate),
                    traj.n_reflections.to_string(),
                    format!("{:.10}", traj.mean_free_path),
                    format!("{:.10}", traj.dmz_transition_rate),
                    traj.n_valid_cells.to_string(),
                    traj.n_dmz_cells.to_string(),
                    format!("{:.10}", traj.fill_ratio),
                    args.seed.to_string(),
                ]) {
                    eprintln!(
                        "ERROR: failed to write details row for N={}, S={}: {err}",
                        traj.n, traj.s
                    );
                    std::process::exit(1);
                }
            }
        }
    }

    if let Err(err) = summary_wtr.flush() {
        eprintln!("ERROR: failed to flush summary CSV: {err}");
        std::process::exit(1);
    }

    if let Some(writer) = details_wtr.as_mut() {
        if let Err(err) = writer.flush() {
            eprintln!("ERROR: failed to flush details CSV: {err}");
            std::process::exit(1);
        }
    }

    eprintln!("Wrote ET billiard summary to {}", args.output.display());
    if let Some(details_path) = &args.details {
        eprintln!("Wrote ET billiard details to {}", details_path.display());
    }
}
