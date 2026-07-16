//! Block-bootstrap ROC-AUC comparison for the Staples THEMIS benchmark.
//!
//! Consumes the flat score CSV written by `themis-staples-score-export`
//! (columns `assoc,dbdt,rot,bmag,label`) and decides the paper-grade
//! question the point estimates leave open: does the normalized CD
//! staple-associator beat the field-rotation-angle baseline on bulk
//! magnetopause-crossing detection, or do the two detectors tie within
//! sampling uncertainty?
//!
//! Per-sample scores from magnetometer time series are strongly
//! autocorrelated, so an i.i.d. bootstrap understates the variance and
//! produces overconfident intervals. The moving-block bootstrap (Kunsch
//! 1989, Ann. Statist. 17) resamples contiguous blocks whose length
//! matches the daily-file scale (~29k samples per THEMIS-A FGM day at
//! spin cadence), preserving within-day dependence while treating days
//! as approximately exchangeable.
//!
//! AUC is the Mann-Whitney U statistic normalized by n_pos * n_neg,
//! computed from average ranks so tied scores contribute 1/2 -- the
//! standard identity AUC = (R_pos - n_pos(n_pos+1)/2) / (n_pos n_neg).
//!
//! The RNG is a fixed-seed ChaCha8 stream, so a rerun with the same
//! inputs reproduces every interval bit-for-bit.
//!
//! Usage:
//!   staples-roc-block-bootstrap \
//!     --scores data/output/benchmark_scores.csv \
//!     --out data/output/staples_roc_block_bootstrap.json

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use clap::Parser;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

#[derive(Parser, Debug)]
#[command(about = "Block-bootstrap ROC-AUC CIs for staple-associator vs baselines")]
struct Args {
    /// Score CSV from themis-staples-score-export (assoc,dbdt,rot,bmag,label).
    #[arg(long)]
    scores: PathBuf,

    /// Output JSON report path.
    #[arg(long)]
    out: PathBuf,

    /// Contiguous block length in samples; the default matches the
    /// ~29k-sample THEMIS-A daily-file scale that sets the
    /// autocorrelation unit.
    #[arg(long, default_value_t = 29_000)]
    block_len: usize,

    /// Number of bootstrap resamples.
    #[arg(long, default_value_t = 200)]
    resamples: usize,

    /// RNG seed for the ChaCha8 bootstrap stream.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

/// Average-rank Mann-Whitney AUC over (score, label) pairs.
///
/// Sorts an index permutation by score, assigns average ranks across
/// tie runs, and sums positive-class ranks. Returns NaN when either
/// class is empty.
fn rank_auc(scores: &[f32], labels: &[u8]) -> f64 {
    let n = scores.len();
    let n_pos = labels.iter().filter(|&&l| l == 1).count();
    let n_neg = n - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return f64::NAN;
    }
    let mut order: Vec<u32> = (0..n as u32).collect();
    order.sort_unstable_by(|&a, &b| {
        scores[a as usize]
            .partial_cmp(&scores[b as usize])
            .expect("scores are finite by construction of the exporter")
    });

    let mut rank_sum_pos = 0.0_f64;
    let mut i = 0usize;
    while i < n {
        // Tie run [i, j): every member receives the average rank.
        let mut j = i + 1;
        while j < n && scores[order[j] as usize] == scores[order[i] as usize] {
            j += 1;
        }
        // Ranks are 1-based: positions i+1 ..= j.
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for &idx in &order[i..j] {
            if labels[idx as usize] == 1 {
                rank_sum_pos += avg_rank;
            }
        }
        i = j;
    }

    let n_pos_f = n_pos as f64;
    (rank_sum_pos - n_pos_f * (n_pos_f + 1.0) / 2.0) / (n_pos_f * n_neg as f64)
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

struct CiSummary {
    point: f64,
    lo: f64,
    hi: f64,
}

fn ci(point: f64, draws: &mut [f64]) -> CiSummary {
    draws.sort_unstable_by(|a, b| a.partial_cmp(b).expect("bootstrap draws are finite"));
    CiSummary {
        point,
        lo: percentile(draws, 0.025),
        hi: percentile(draws, 0.975),
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Column layout of the exporter: assoc,dbdt,rot,bmag,label.
    let mut assoc: Vec<f32> = Vec::new();
    let mut dbdt: Vec<f32> = Vec::new();
    let mut rot: Vec<f32> = Vec::new();
    let mut labels: Vec<u8> = Vec::new();

    let reader = BufReader::new(File::open(&args.scores)?);
    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        if line_no == 0 {
            continue;
        }
        let mut cols = line.split(',');
        let a: f32 = cols.next().ok_or_else(|| anyhow::anyhow!("row"))?.parse()?;
        let d: f32 = cols.next().ok_or_else(|| anyhow::anyhow!("row"))?.parse()?;
        let r: f32 = cols.next().ok_or_else(|| anyhow::anyhow!("row"))?.parse()?;
        let _bmag = cols.next();
        let l: u8 = cols.next().ok_or_else(|| anyhow::anyhow!("row"))?.parse()?;
        assoc.push(a);
        dbdt.push(d);
        rot.push(r);
        labels.push(l);
    }
    let n = labels.len();
    anyhow::ensure!(n > args.block_len, "need more samples than one block");
    let n_pos = labels.iter().filter(|&&l| l == 1).count();
    eprintln!("loaded {} samples, {} positives", n, n_pos);

    let auc_assoc = rank_auc(&assoc, &labels);
    let auc_rot = rank_auc(&rot, &labels);
    let auc_dbdt = rank_auc(&dbdt, &labels);
    eprintln!(
        "point AUC: assoc={:.4} rot={:.4} dbdt={:.4}",
        auc_assoc, auc_rot, auc_dbdt
    );

    // Moving-block bootstrap: each resample draws ceil(n / block_len)
    // block start positions uniformly, concatenates the blocks, and
    // recomputes both AUCs on the identical resampled index set so the
    // delta draw is a paired comparison.
    let n_blocks = n.div_ceil(args.block_len);
    let max_start = n - args.block_len;

    let draws: Vec<(f64, f64)> = (0..args.resamples)
        .into_par_iter()
        .map(|rep| {
            // A per-resample stream keyed on (seed, rep) keeps the draw
            // sequence independent of the rayon scheduling order.
            let mut rng = ChaCha8Rng::seed_from_u64(args.seed.wrapping_add(rep as u64));
            let mut s_assoc: Vec<f32> = Vec::with_capacity(n_blocks * args.block_len);
            let mut s_rot: Vec<f32> = Vec::with_capacity(n_blocks * args.block_len);
            let mut s_lab: Vec<u8> = Vec::with_capacity(n_blocks * args.block_len);
            for _ in 0..n_blocks {
                let start = rng.random_range(0..=max_start);
                let end = start + args.block_len;
                s_assoc.extend_from_slice(&assoc[start..end]);
                s_rot.extend_from_slice(&rot[start..end]);
                s_lab.extend_from_slice(&labels[start..end]);
            }
            (rank_auc(&s_assoc, &s_lab), rank_auc(&s_rot, &s_lab))
        })
        .collect();

    let mut d_assoc: Vec<f64> = draws.iter().map(|&(a, _)| a).filter(|x| x.is_finite()).collect();
    let mut d_rot: Vec<f64> = draws.iter().map(|&(_, r)| r).filter(|x| x.is_finite()).collect();
    let mut d_delta: Vec<f64> = draws
        .iter()
        .filter(|(a, r)| a.is_finite() && r.is_finite())
        .map(|&(a, r)| a - r)
        .collect();
    let p_delta_le_zero =
        d_delta.iter().filter(|&&d| d <= 0.0).count() as f64 / d_delta.len() as f64;

    let s_assoc = ci(auc_assoc, &mut d_assoc);
    let s_rot = ci(auc_rot, &mut d_rot);
    let s_delta = ci(auc_assoc - auc_rot, &mut d_delta);

    let report = format!(
        "{{\n  \"n_samples\": {},\n  \"n_positives\": {},\n  \"block_len\": {},\n  \"resamples\": {},\n  \"seed\": {},\n  \"auc_assoc\": {{\"point\": {:.6}, \"ci95\": [{:.6}, {:.6}]}},\n  \"auc_rot\": {{\"point\": {:.6}, \"ci95\": [{:.6}, {:.6}]}},\n  \"auc_dbdt_point\": {:.6},\n  \"auc_delta_assoc_minus_rot\": {{\"point\": {:.6}, \"ci95\": [{:.6}, {:.6}]}},\n  \"p_delta_le_zero\": {:.4}\n}}\n",
        n,
        n_pos,
        args.block_len,
        args.resamples,
        args.seed,
        s_assoc.point,
        s_assoc.lo,
        s_assoc.hi,
        s_rot.point,
        s_rot.lo,
        s_rot.hi,
        auc_dbdt,
        s_delta.point,
        s_delta.lo,
        s_delta.hi,
        p_delta_le_zero
    );
    File::create(&args.out)?.write_all(report.as_bytes())?;
    println!("{}", report);
    Ok(())
}
