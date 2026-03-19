use anyhow::{Context, Result};
use clap::Parser;
use csv::Writer;
use gororoba_cli_data::nanograv_timing::load_release;
use nalgebra::{Cholesky, DMatrix};
use rand::prelude::*;
use rand_distr::Normal;
use stats_core::astrophysics::{angular_separation, hellings_downs};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "nanograv-synthetic-gen",
    about = "Generates a synthetic NANOGrav dataset with pure Hellings-Downs correlation and standard noise"
)]
struct Args {
    #[arg(
        long,
        default_value = "data/external/nanograv_15yr_timing/NANOGrav15yr_PulsarTiming_v2.1.0"
    )]
    root: PathBuf,

    #[arg(long, default_value = "data/csv/nanograv_synthetic_pairwise_hd.csv")]
    pairwise_out: PathBuf,

    #[arg(long, default_value_t = 42)]
    seed: u64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut rng = StdRng::seed_from_u64(args.seed);

    println!("Loading metadata from {:?}...", args.root);
    let release = load_release(&args.root).context("Failed to load real dataset for metadata")?;
    let mut pulsars = Vec::new();

    for (name, data) in &release {
        if let Some(vec) = data.sky_vector() {
            pulsars.push((name.clone(), vec));
        }
    }
    let n = pulsars.len();
    println!("Found {} pulsars with sky coordinates.", n);

    // 1. Construct HD Correlation Matrix
    let mut gamma = DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            if i == j {
                gamma[(i, j)] = 1.0;
            } else {
                let sep = angular_separation(pulsars[i].1, pulsars[j].1);
                gamma[(i, j)] = hellings_downs(sep);
            }
        }
    }

    // 2. Cholesky Decomposition for spatial mixing
    for i in 0..n {
        gamma[(i, i)] += 1e-9;
    }
    let _chol = Cholesky::new(gamma).context("HD matrix is not positive definite")?;

    // 3. Generate Synthetic Pairwise Audit CSV
    println!("Generating synthetic pairwise cross-correlations...");
    let mut writer = Writer::from_path(&args.pairwise_out)?;
    writer.write_record([
        "pulsar_a",
        "pulsar_b",
        "separation_deg",
        "hellings_downs",
        "overlap_bins",
        "avg_residual_pearson",
    ])?;

    let normal = Normal::new(0.0, 0.1).unwrap();

    for i in 0..n {
        for j in (i + 1)..n {
            let sep = angular_separation(pulsars[i].1, pulsars[j].1);
            let hd = hellings_downs(sep);

            // Synthetic Pearson = HD value + Gaussian noise
            let synth_pearson = (hd + normal.sample(&mut rng)).clamp(-1.0, 1.0);

            writer.write_record([
                &pulsars[i].0,
                &pulsars[j].0,
                &format!("{:.12}", sep.to_degrees()),
                &format!("{:.12}", hd),
                &"160".to_string(),
                &format!("{:.12}", synth_pearson),
            ])?;
        }
    }

    println!("Synthetic pairwise audit saved to {:?}", args.pairwise_out);
    Ok(())
}
