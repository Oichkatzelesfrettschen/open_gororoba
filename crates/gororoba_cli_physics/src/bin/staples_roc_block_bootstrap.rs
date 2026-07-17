//! Block-bootstrap ROC-AUC comparison for the Staples THEMIS benchmark.
//!
//! Consumes the flat score CSV written by `themis-staples-score-export`
//! (columns `assoc,dbdt,rot,bmag,label`) and decides the paper-grade
//! questions the point estimates leave open: does the normalized CD
//! staple-associator beat the field-rotation-angle baseline on bulk
//! magnetopause-crossing detection, does the edge survive within each
//! sample regime, and is the conclusion stable under the block-length
//! choice?
//!
//! Per-sample scores from magnetometer time series are strongly
//! autocorrelated, so an i.i.d. bootstrap understates the variance and
//! produces overconfident intervals. The moving-block bootstrap (Kunsch
//! 1989, Ann. Statist. 17) resamples contiguous blocks whose length
//! matches the daily-file scale (~29k samples per THEMIS-A FGM day at
//! spin cadence), preserving within-day dependence while treating days
//! as approximately exchangeable. A half/double block-length sweep
//! reports the bulk delta CI at each length as a sensitivity check.
//!
//! Sample-regime strata follow the benchmark's fixed feature definition:
//! positives split at the median |dB/dt| among positives, high-gradient
//! above the median and low-gradient at or below; each stratum scores against
//! the full negative class. The split point is computed once on the
//! full dataset and reused across resamples, since it is a feature
//! definition rather than a resampled statistic. The worst-case AUC
//! (min over strata, per detector, per resample) quantifies the
//! robustness claim with a paired CI of its own.
//!
//! AUC is the Mann-Whitney U statistic normalized by n_pos * n_neg,
//! computed from average ranks so tied scores contribute 1/2 -- the
//! standard identity AUC = (R_pos - n_pos(n_pos+1)/2) / (n_pos n_neg).
//!
//! The RNG is a fixed-seed ChaCha8 stream keyed on (seed, block_len,
//! resample index), so a rerun with the same inputs reproduces every
//! interval bit-for-bit.
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

    /// Comma-separated block lengths for the bulk-delta sensitivity
    /// sweep; the default brackets the primary length by half/double.
    #[arg(long, default_value = "14500,58000")]
    block_len_sweep: String,

    /// Number of bootstrap resamples.
    #[arg(long, default_value_t = 200)]
    resamples: usize,

    /// RNG seed for the ChaCha8 bootstrap stream.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

/// Sample tags: negative, low-gradient positive (|dB/dt| at or below the
/// positive-class median), high-gradient positive (above). Per-time-point strata, not crossing-event classes.
const TAG_NEG: u8 = 0;
const TAG_POS_LOW_GRADIENT: u8 = 1;
const TAG_POS_HIGH_GRADIENT: u8 = 2;

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

/// AUC of one sample-regime stratum against the full negative class: samples
/// carrying the other stratum's tag drop out, the requested tag becomes
/// the positive label.
fn stratum_auc(scores: &[f32], tags: &[u8], stratum_tag: u8) -> f64 {
    let mut s: Vec<f32> = Vec::with_capacity(scores.len());
    let mut l: Vec<u8> = Vec::with_capacity(scores.len());
    for (&sc, &t) in scores.iter().zip(tags) {
        if t == TAG_NEG || t == stratum_tag {
            s.push(sc);
            l.push(u8::from(t == stratum_tag));
        }
    }
    rank_auc(&s, &l)
}

/// One paired resample's statistics for both detectors.
struct BootDraw {
    bulk_assoc: f64,
    bulk_rot: f64,
    low_gradient_assoc: f64,
    low_gradient_rot: f64,
    high_gradient_assoc: f64,
    high_gradient_rot: f64,
}

/// Moving-block bootstrap: each resample draws ceil(n / block_len) block
/// start positions uniformly, concatenates the blocks, and recomputes
/// every statistic on the identical resampled index set so each delta
/// draw is a paired comparison. `with_strata` gates the four stratum
/// AUCs, which triple the per-resample sort cost.
fn bootstrap(
    assoc: &[f32],
    rot: &[f32],
    tags: &[u8],
    block_len: usize,
    resamples: usize,
    seed: u64,
    with_strata: bool,
) -> Vec<BootDraw> {
    let n = tags.len();
    let n_blocks = n.div_ceil(block_len);
    let max_start = n - block_len;
    (0..resamples)
        .into_par_iter()
        .map(|rep| {
            // A per-resample stream keyed on (seed, block_len, rep) keeps
            // the draw sequence independent of rayon scheduling order and
            // distinct across sweep lengths.
            let mut rng = ChaCha8Rng::seed_from_u64(
                seed.wrapping_add(rep as u64)
                    .wrapping_add((block_len as u64) << 32),
            );
            let mut s_assoc: Vec<f32> = Vec::with_capacity(n_blocks * block_len);
            let mut s_rot: Vec<f32> = Vec::with_capacity(n_blocks * block_len);
            let mut s_tag: Vec<u8> = Vec::with_capacity(n_blocks * block_len);
            for _ in 0..n_blocks {
                let start = rng.random_range(0..=max_start);
                let end = start + block_len;
                s_assoc.extend_from_slice(&assoc[start..end]);
                s_rot.extend_from_slice(&rot[start..end]);
                s_tag.extend_from_slice(&tags[start..end]);
            }
            let bulk_labels: Vec<u8> = s_tag.iter().map(|&t| u8::from(t != TAG_NEG)).collect();
            let (ra, rr, ca, cr) = if with_strata {
                (
                    stratum_auc(&s_assoc, &s_tag, TAG_POS_LOW_GRADIENT),
                    stratum_auc(&s_rot, &s_tag, TAG_POS_LOW_GRADIENT),
                    stratum_auc(&s_assoc, &s_tag, TAG_POS_HIGH_GRADIENT),
                    stratum_auc(&s_rot, &s_tag, TAG_POS_HIGH_GRADIENT),
                )
            } else {
                (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
            };
            BootDraw {
                bulk_assoc: rank_auc(&s_assoc, &bulk_labels),
                bulk_rot: rank_auc(&s_rot, &bulk_labels),
                low_gradient_assoc: ra,
                low_gradient_rot: rr,
                high_gradient_assoc: ca,
                high_gradient_rot: cr,
            }
        })
        .collect()
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
    p_le_zero: f64,
}

/// Percentile 95% CI plus the fraction of draws at or below zero (the
/// one-sided bootstrap evidence that the paired delta is positive).
fn ci(point: f64, draws: &[f64]) -> CiSummary {
    let mut finite: Vec<f64> = draws.iter().copied().filter(|x| x.is_finite()).collect();
    finite.sort_unstable_by(|a, b| a.partial_cmp(b).expect("finite draws"));
    let p_le_zero = finite.iter().filter(|&&d| d <= 0.0).count() as f64 / finite.len() as f64;
    CiSummary {
        point,
        lo: percentile(&finite, 0.025),
        hi: percentile(&finite, 0.975),
        p_le_zero,
    }
}

fn json_ci(s: &CiSummary) -> String {
    format!(
        "{{\"point\": {:.6}, \"ci95\": [{:.6}, {:.6}], \"p_le_zero\": {:.4}}}",
        s.point, s.lo, s.hi, s.p_le_zero
    )
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // Column layout of the exporter: assoc,dbdt,rot,bmag,label.
    let mut assoc: Vec<f32> = Vec::new();
    let mut dbdt: Vec<f32> = Vec::new();
    let mut rot: Vec<f32> = Vec::new();
    let mut labels: Vec<u8> = Vec::new();

    let reader = BufReader::new(File::open(&args.scores)?);
    // The exporter's schema gained a leading file_id column; both
    // layouts parse, keyed off the header.
    let mut has_file_id = false;
    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        if line_no == 0 {
            has_file_id = line.starts_with("file_id,");
            continue;
        }
        let mut cols = line.split(',');
        if has_file_id {
            cols.next();
        }
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

    // Fixed stratum definition: median |dB/dt| among positives splits
    // high-gradient (above) from low-gradient (at or below).
    let mut pos_dbdt: Vec<f32> = labels
        .iter()
        .zip(&dbdt)
        .filter(|&(&l, _)| l == 1)
        .map(|(_, &d)| d)
        .collect();
    pos_dbdt.sort_unstable_by(|a, b| a.partial_cmp(b).expect("finite dbdt"));
    let split = pos_dbdt[pos_dbdt.len() / 2];
    let tags: Vec<u8> = labels
        .iter()
        .zip(&dbdt)
        .map(|(&l, &d)| {
            if l == 0 {
                TAG_NEG
            } else if d > split {
                TAG_POS_HIGH_GRADIENT
            } else {
                TAG_POS_LOW_GRADIENT
            }
        })
        .collect();
    let n_comp = tags.iter().filter(|&&t| t == TAG_POS_HIGH_GRADIENT).count();
    eprintln!(
        "stratum split at |dB/dt|={:.6e}: {} high-gradient, {} low-gradient",
        split,
        n_comp,
        n_pos - n_comp
    );

    // Point estimates on the full dataset.
    let auc_assoc = rank_auc(&assoc, &labels);
    let auc_rot = rank_auc(&rot, &labels);
    let auc_dbdt = rank_auc(&dbdt, &labels);
    let pt_rot_assoc = stratum_auc(&assoc, &tags, TAG_POS_LOW_GRADIENT);
    let pt_rot_rot = stratum_auc(&rot, &tags, TAG_POS_LOW_GRADIENT);
    let pt_comp_assoc = stratum_auc(&assoc, &tags, TAG_POS_HIGH_GRADIENT);
    let pt_comp_rot = stratum_auc(&rot, &tags, TAG_POS_HIGH_GRADIENT);
    eprintln!(
        "point AUC: bulk assoc={:.4} rot={:.4} dbdt={:.4}; low-gradient {:.4}/{:.4}; high-gradient {:.4}/{:.4}",
        auc_assoc, auc_rot, auc_dbdt, pt_rot_assoc, pt_rot_rot, pt_comp_assoc, pt_comp_rot
    );

    // Primary run carries the stratum statistics.
    let draws = bootstrap(
        &assoc,
        &rot,
        &tags,
        args.block_len,
        args.resamples,
        args.seed,
        true,
    );

    let collect = |f: &dyn Fn(&BootDraw) -> f64| -> Vec<f64> { draws.iter().map(f).collect() };
    let s_bulk_assoc = ci(auc_assoc, &collect(&|d| d.bulk_assoc));
    let s_bulk_rot = ci(auc_rot, &collect(&|d| d.bulk_rot));
    let s_bulk_delta = ci(
        auc_assoc - auc_rot,
        &collect(&|d| d.bulk_assoc - d.bulk_rot),
    );
    let s_low_gradient_delta = ci(
        pt_rot_assoc - pt_rot_rot,
        &collect(&|d| d.low_gradient_assoc - d.low_gradient_rot),
    );
    let s_high_gradient_delta = ci(
        pt_comp_assoc - pt_comp_rot,
        &collect(&|d| d.high_gradient_assoc - d.high_gradient_rot),
    );
    // Worst-case AUC (min over strata) is the robustness statistic;
    // its paired delta asks whether the associator's floor beats the
    // rotation baseline's floor.
    let s_worst_delta = ci(
        pt_rot_assoc.min(pt_comp_assoc) - pt_rot_rot.min(pt_comp_rot),
        &collect(&|d| {
            d.low_gradient_assoc.min(d.high_gradient_assoc) - d.low_gradient_rot.min(d.high_gradient_rot)
        }),
    );

    // Block-length sensitivity sweep: bulk delta only.
    let mut sweep_json: Vec<String> = Vec::new();
    for tok in args.block_len_sweep.split(',') {
        let bl: usize = tok.trim().parse()?;
        if bl == 0 || bl >= n {
            continue;
        }
        let sw = bootstrap(&assoc, &rot, &tags, bl, args.resamples, args.seed, false);
        let deltas: Vec<f64> = sw.iter().map(|d| d.bulk_assoc - d.bulk_rot).collect();
        let s = ci(auc_assoc - auc_rot, &deltas);
        eprintln!(
            "sweep block_len={}: delta {:.6} [{:.6}, {:.6}]",
            bl, s.point, s.lo, s.hi
        );
        sweep_json.push(format!(
            "    {{\"block_len\": {}, \"delta_assoc_minus_rot\": {}}}",
            bl,
            json_ci(&s)
        ));
    }

    let report = format!(
        "{{\n  \"n_samples\": {},\n  \"n_positives\": {},\n  \"n_high_gradient_positive\": {},\n  \"n_low_gradient_positive\": {},\n  \"stratum_split_dbdt\": {:.6e},\n  \"block_len\": {},\n  \"resamples\": {},\n  \"seed\": {},\n  \"auc_assoc\": {},\n  \"auc_rot\": {},\n  \"auc_dbdt_point\": {:.6},\n  \"auc_delta_assoc_minus_rot\": {},\n  \"stratum_low_gradient_positive\": {{\"auc_assoc_point\": {:.6}, \"auc_rot_point\": {:.6}, \"delta_assoc_minus_rot\": {}}},\n  \"stratum_high_gradient_positive\": {{\"auc_assoc_point\": {:.6}, \"auc_rot_point\": {:.6}, \"delta_assoc_minus_rot\": {}}},\n  \"worst_case_delta_assoc_minus_rot\": {},\n  \"block_len_sweep\": [\n{}\n  ]\n}}\n",
        n,
        n_pos,
        n_comp,
        n_pos - n_comp,
        split,
        args.block_len,
        args.resamples,
        args.seed,
        json_ci(&s_bulk_assoc),
        json_ci(&s_bulk_rot),
        auc_dbdt,
        json_ci(&s_bulk_delta),
        pt_rot_assoc,
        pt_rot_rot,
        json_ci(&s_low_gradient_delta),
        pt_comp_assoc,
        pt_comp_rot,
        json_ci(&s_high_gradient_delta),
        json_ci(&s_worst_delta),
        sweep_json.join(",\n")
    );
    File::create(&args.out)?.write_all(report.as_bytes())?;
    println!("{}", report);
    Ok(())
}
