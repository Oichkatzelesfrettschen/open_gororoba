//! File-level bootstrap CIs for the cross-mission log-sigma survey.
//!
//! Consumes the per-file record JSON written by
//! `heliosphere-selfsim-logstats` and attaches a 95% confidence interval
//! to each mission's mean log-sigma (the log-normal shape parameter of
//! the unnormalized staple-associator distribution). Daily files are
//! independent sampling units drawn across each mission's archive, so an
//! ordinary bootstrap over files is the appropriate resampling design --
//! the within-file autocorrelation is already absorbed into the per-file
//! statistic.
//!
//! The mission-level question the intervals decide: which pairwise
//! log-sigma orderings across heliocentric distance survive sampling
//! uncertainty, given the modest per-mission file counts.
//!
//! The RNG is a fixed-seed ChaCha8 stream keyed on (seed, mission index),
//! so a rerun reproduces every interval bit-for-bit.
//!
//! Usage:
//!   selfsim-logsigma-bootstrap \
//!     --records data/output/agg_selfsim_rust.json \
//!     --out data/output/selfsim_logsigma_bootstrap.json

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Deserialize;

#[derive(Parser, Debug)]
#[command(about = "File-level bootstrap CIs for per-mission selfsim log-sigma")]
struct Args {
    /// Per-file record JSON from heliosphere-selfsim-logstats.
    #[arg(long)]
    records: PathBuf,

    /// Output JSON report path.
    #[arg(long)]
    out: PathBuf,

    /// Number of bootstrap resamples.
    #[arg(long, default_value_t = 10_000)]
    resamples: usize,

    /// RNG seed for the ChaCha8 bootstrap stream.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

#[derive(Deserialize)]
struct FileRecord {
    log_std: f64,
}

#[derive(Deserialize)]
struct MissionAggregate {
    distance_au: f64,
    records: Vec<FileRecord>,
}

/// Percentile of a sorted sample via linear interpolation.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let raw = fs::read_to_string(&args.records)
        .with_context(|| format!("read {}", args.records.display()))?;
    let missions: BTreeMap<String, MissionAggregate> = serde_json::from_str(&raw)?;

    let mut lines: Vec<String> = Vec::new();
    for (mission_index, (name, agg)) in missions.iter().enumerate() {
        let sigmas: Vec<f64> = agg.records.iter().map(|r| r.log_std).collect();
        let n_files = sigmas.len();
        let mean = sigmas.iter().sum::<f64>() / n_files as f64;

        let mut rng =
            ChaCha8Rng::seed_from_u64(args.seed.wrapping_add((mission_index as u64) << 32));
        let mut draws: Vec<f64> = (0..args.resamples)
            .map(|_| {
                let mut acc = 0.0;
                for _ in 0..n_files {
                    acc += sigmas[rng.random_range(0..n_files)];
                }
                acc / n_files as f64
            })
            .collect();
        draws.sort_unstable_by(|a, b| a.partial_cmp(b).expect("finite log_std means"));
        let lo = percentile(&draws, 0.025);
        let hi = percentile(&draws, 0.975);
        eprintln!(
            "{:>10} ({:6.2} AU): log_sigma {:.4} [{:.4}, {:.4}] over {} files",
            name, agg.distance_au, mean, lo, hi, n_files
        );
        lines.push(format!(
            "  \"{}\": {{\"distance_au\": {}, \"n_files\": {}, \"log_sigma_mean\": {:.6}, \"ci95\": [{:.6}, {:.6}]}}",
            name, agg.distance_au, n_files, mean, lo, hi
        ));
    }

    let report = format!(
        "{{\n  \"resamples\": {},\n  \"seed\": {},\n{}\n}}\n",
        args.resamples,
        args.seed,
        lines.join(",\n")
    );
    fs::write(&args.out, &report)?;
    println!("{}", report);
    Ok(())
}
