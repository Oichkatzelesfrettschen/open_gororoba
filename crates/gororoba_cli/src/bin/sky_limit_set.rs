//! Sky-limit-set correspondence analysis (Experiment C).
//!
//! Compares ET skybox invariants and Coxeter-group-derived invariants across
//! selected CD levels.

use std::collections::BTreeSet;
use std::path::PathBuf;

use algebra_core::experimental::algebraic_dynamics::{experiment_c_sky_limit_set, CoxeterType};
use clap::Parser;

#[derive(Parser)]
#[command(name = "sky-limit-set")]
#[command(about = "Compare ET skybox invariants with Coxeter group invariants")]
struct Args {
    /// Comma-separated CD levels N (sedenions and above).
    #[arg(long, value_delimiter = ',', default_value = "4,5,6")]
    n_levels: Vec<usize>,

    /// Comma-separated Coxeter families to include (A,B,D).
    #[arg(long, value_delimiter = ',', default_value = "A,B,D")]
    coxeter_types: Vec<String>,

    /// Output CSV path.
    #[arg(long, default_value = "data/csv/sky_limit_set_comparison.csv")]
    output: PathBuf,
}

fn parse_group_token(token: &str) -> Option<CoxeterType> {
    match token.trim().to_ascii_uppercase().as_str() {
        "A" => Some(CoxeterType::A),
        "B" => Some(CoxeterType::B),
        "D" => Some(CoxeterType::D),
        _ => None,
    }
}

fn group_name(group: CoxeterType) -> &'static str {
    match group {
        CoxeterType::A => "A",
        CoxeterType::B => "B",
        CoxeterType::D => "D",
    }
}

fn main() {
    let args = Args::parse();

    if args.n_levels.is_empty() {
        eprintln!("ERROR: --n-levels must not be empty");
        std::process::exit(1);
    }
    if args.n_levels.iter().any(|n| *n < 4) {
        eprintln!("ERROR: all --n-levels entries must be >= 4");
        std::process::exit(1);
    }

    let mut included = BTreeSet::new();
    for token in &args.coxeter_types {
        let group = match parse_group_token(token) {
            Some(value) => value,
            None => {
                eprintln!("ERROR: unsupported Coxeter type '{token}' (expected A,B,D)");
                std::process::exit(1);
            }
        };
        included.insert(group_name(group));
    }

    if included.is_empty() {
        eprintln!("ERROR: --coxeter-types must not be empty");
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

    let results = experiment_c_sky_limit_set(&args.n_levels);

    let mut wtr = match csv::Writer::from_path(&args.output) {
        Ok(writer) => writer,
        Err(err) => {
            eprintln!(
                "ERROR: failed to open output CSV {}: {err}",
                args.output.display()
            );
            std::process::exit(1);
        }
    };

    if let Err(err) = wtr.write_record([
        "n",
        "box_kite_count",
        "mean_dmz_density",
        "mean_n_empty_components",
        "mean_interior_dmz_density",
        "coxeter_type",
        "coxeter_rank",
        "coxeter_number",
        "n_positive_roots",
        "spectral_radius",
        "cartan_determinant",
        "rank_ratio",
        "density_root_ratio",
        "match_score",
    ]) {
        eprintln!("ERROR: failed to write CSV header: {err}");
        std::process::exit(1);
    }

    for row in &results {
        for cmp in &row.coxeter_comparisons {
            let coxeter_type = group_name(cmp.coxeter.group_type);
            if !included.contains(coxeter_type) {
                continue;
            }

            if let Err(err) = wtr.write_record([
                row.n.to_string(),
                row.box_kite_count.to_string(),
                format!("{:.10}", row.mean_dmz_density),
                format!("{:.10}", row.mean_n_empty_components),
                format!("{:.10}", row.mean_interior_dmz_density),
                coxeter_type.to_string(),
                cmp.coxeter.rank.to_string(),
                cmp.coxeter.coxeter_number.to_string(),
                cmp.coxeter.n_positive_roots.to_string(),
                format!("{:.10}", cmp.spectral_radius),
                format!("{:.10}", cmp.coxeter.cartan_determinant),
                format!("{:.10}", cmp.rank_ratio),
                format!("{:.10}", cmp.density_root_ratio),
                format!("{:.10}", cmp.match_score),
            ]) {
                eprintln!("ERROR: failed to write CSV row for N={}: {err}", row.n);
                std::process::exit(1);
            }
        }
    }

    if let Err(err) = wtr.flush() {
        eprintln!("ERROR: failed to flush CSV: {err}");
        std::process::exit(1);
    }

    eprintln!(
        "Wrote sky-limit-set comparison to {}",
        args.output.display()
    );
}
