use anyhow::{Context, Result, bail};
use clap::Parser;
use csv::Writer;
use gororoba_cli_data::nanograv_timing::load_release;
use nalgebra::{Cholesky, DMatrix, DVector};
use rand::prelude::*;
use rand_distr::{Distribution, StandardNormal};
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

    #[arg(long, default_value_t = 2048)]
    draws: usize,

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
    if args.draws < 2 {
        bail!("--draws must be >= 2");
    }

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
    let chol = Cholesky::new(gamma).context("HD matrix is not positive definite")?;
    let lower = chol.l();

    let mut draws = Vec::with_capacity(args.draws);
    for _ in 0..args.draws {
        let z = DVector::from_iterator(n, (0..n).map(|_| StandardNormal.sample(&mut rng)));
        let sample = &lower * z;
        draws.push(sample.iter().copied().collect::<Vec<_>>());
    }

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
    let overlap_bins = args.draws.to_string();

    for i in 0..n {
        for j in (i + 1)..n {
            let sep = angular_separation(pulsars[i].1, pulsars[j].1);
            let hd = hellings_downs(sep);
            let synth_pearson = sample_pearson(&draws, i, j);

            writer.write_record([
                &pulsars[i].0,
                &pulsars[j].0,
                &format!("{:.12}", sep.to_degrees()),
                &format!("{:.12}", hd),
                &overlap_bins,
                &format!("{:.12}", synth_pearson),
            ])?;
        }
    }

    println!("Synthetic pairwise audit saved to {:?}", args.pairwise_out);
    Ok(())
}

fn sample_pearson(draws: &[Vec<f64>], left: usize, right: usize) -> f64 {
    if draws.len() < 2 {
        return 0.0;
    }
    let mut sum_left = 0.0;
    let mut sum_right = 0.0;
    let mut sum_left_sq = 0.0;
    let mut sum_right_sq = 0.0;
    let mut sum_cross = 0.0;
    for draw in draws {
        let x = draw[left];
        let y = draw[right];
        sum_left += x;
        sum_right += y;
        sum_left_sq += x * x;
        sum_right_sq += y * y;
        sum_cross += x * y;
    }
    let n = draws.len() as f64;
    let cov = sum_cross - (sum_left * sum_right / n);
    let var_left = sum_left_sq - (sum_left * sum_left / n);
    let var_right = sum_right_sq - (sum_right * sum_right / n);
    if var_left <= 0.0 || var_right <= 0.0 {
        0.0
    } else {
        cov / (var_left.sqrt() * var_right.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::sample_pearson;

    #[test]
    fn pearson_from_draws_returns_expected_sign() {
        let draws = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
            vec![4.0, 8.0],
        ];
        let pearson = sample_pearson(&draws, 0, 1);
        assert!((pearson - 1.0).abs() < 1.0e-12);
    }
}
