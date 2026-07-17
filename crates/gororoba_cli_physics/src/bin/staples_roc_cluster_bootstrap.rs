//! File-cluster bootstrap ROC-AUC inference for the Staples THEMIS
//! benchmark, with a simultaneous-dominance certificate.
//!
//! Consumes the score CSV written by `themis-staples-score-export`
//! (columns `file_id,assoc,dbdt,rot,bmag,label`) and treats each daily
//! file as the intact sampling cluster: every resample draws whole
//! files with replacement and keeps every sample inside each selected
//! file. Cluster resampling gives materially better interval coverage
//! than contiguous-block resampling when observations within a day are
//! correlated, because blocks that straddle file boundaries mix days
//! instead of resampling them.
//!
//! Three estimands report separately:
//!   1. pooled time-point AUC (descriptive, all samples weighted equally),
//!   2. mean daily AUC (performance on a typical crossing day; files
//!      containing both classes contribute),
//!   3. cluster-resampled pooled AUC deltas with 95% percentile CIs.
//!
//! Positive samples stratify at the median |dB/dt| among positives into
//! `high_gradient_positive` and `low_gradient_positive` SAMPLE REGIMES.
//! These are per-time-point strata, deliberately named so: one crossing
//! event can contribute samples to both, and event-level "compressive
//! vs rotational crossing" classification needs integrated per-event
//! statistics that this scorer does not compute.
//!
//! The simultaneous margin
//!   M = min(delta_bulk, delta_high_gradient, delta_low_gradient)
//! recomputes inside every paired resample; a lower 95% bound of M
//! above zero certifies strict componentwise AUC dominance as ONE
//! CI-backed proposition instead of three marginal intervals. Tail
//! proportions report as "< 1/(B+1)" when zero nonpositive draws occur,
//! never as an exact zero.
//!
//! AUC per resample uses a rank walk over a per-detector global sort
//! computed once: a resample only reweights samples by their file's
//! draw count, so average-rank Mann-Whitney AUC follows from one O(n)
//! pass per detector accumulating multiplicity-weighted tie groups for
//! all three strata simultaneously. 2,000 resamples cost minutes, not
//! hours.
//!
//! The RNG is a fixed-seed ChaCha8 stream keyed on (seed, resample), so
//! a rerun with the same inputs reproduces every interval bit-for-bit.
//!
//! Usage:
//!   staples-roc-cluster-bootstrap \
//!     --scores data/output/benchmark_scores.csv \
//!     --out data/output/staples_roc_cluster_bootstrap.json

use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use clap::Parser;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

#[derive(Parser, Debug)]
#[command(about = "File-cluster bootstrap ROC-AUC CIs with a simultaneous dominance margin")]
struct Args {
    /// Score CSV from themis-staples-score-export
    /// (file_id,assoc,dbdt,rot,bmag,label).
    #[arg(long)]
    scores: PathBuf,

    /// Output JSON report path.
    #[arg(long)]
    out: PathBuf,

    /// Number of bootstrap resamples.
    #[arg(long, default_value_t = 2_000)]
    resamples: usize,

    /// RNG seed for the ChaCha8 bootstrap stream.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

/// Sample strata: negative, low-gradient positive (|dB/dt| at or below
/// the positive-class median), high-gradient positive (above). These
/// are per-time-point regimes, not crossing-event classes.
const TAG_NEG: u8 = 0;
const TAG_POS_LOW_GRADIENT: u8 = 1;
const TAG_POS_HIGH_GRADIENT: u8 = 2;

/// One detector's precomputed global ordering: sample indices sorted by
/// score ascending. A resample reweights samples without reordering
/// them, so this sort happens once per detector.
struct DetectorOrder {
    order: Vec<u32>,
}

impl DetectorOrder {
    fn new(scores: &[f32]) -> Self {
        let mut order: Vec<u32> = (0..scores.len() as u32).collect();
        order.sort_unstable_by(|&a, &b| {
            scores[a as usize]
                .partial_cmp(&scores[b as usize])
                .expect("scores are finite by construction of the exporter")
        });
        Self { order }
    }
}

/// Multiplicity-weighted average-rank AUC for the three strata in one
/// walk. `weight[s]` is how many times sample s appears in the
/// resample (its file's draw count). Strata share the walk because a
/// tie group's members are interchangeable within each stratum: the
/// bulk stratum sees every sample, each gradient stratum sees the
/// negatives plus its own positives.
fn walk_auc_three_strata(
    scores: &[f32],
    det: &DetectorOrder,
    tags: &[u8],
    weight: &[u32],
) -> (f64, f64, f64) {
    // Per-stratum accumulators: cumulative rank position, positive rank
    // sum, positive count, total count. Index 0 = bulk, 1 = low-gradient
    // stratum, 2 = high-gradient stratum.
    let mut cum = [0u64; 3];
    let mut rank_sum_pos = [0f64; 3];
    let mut n_pos = [0u64; 3];
    let mut n_tot = [0u64; 3];

    let order = &det.order;
    let n = order.len();
    let mut i = 0usize;
    while i < n {
        let v = scores[order[i] as usize];
        // Tie run [i, j): identical scores share an average rank per
        // stratum.
        let mut j = i;
        // Per-stratum multiplicity and positive multiplicity in the run.
        let mut m = [0u64; 3];
        let mut mp = [0u64; 3];
        while j < n && scores[order[j] as usize] == v {
            let idx = order[j] as usize;
            let w = u64::from(weight[idx]);
            if w > 0 {
                let t = tags[idx];
                m[0] += w;
                if t != TAG_NEG {
                    mp[0] += w;
                }
                if t == TAG_NEG || t == TAG_POS_LOW_GRADIENT {
                    m[1] += w;
                    if t == TAG_POS_LOW_GRADIENT {
                        mp[1] += w;
                    }
                }
                if t == TAG_NEG || t == TAG_POS_HIGH_GRADIENT {
                    m[2] += w;
                    if t == TAG_POS_HIGH_GRADIENT {
                        mp[2] += w;
                    }
                }
            }
            j += 1;
        }
        for s in 0..3 {
            if m[s] > 0 {
                // Ranks are 1-based: the run occupies positions
                // cum+1 ..= cum+m, so the average is cum + (m+1)/2.
                let avg_rank = cum[s] as f64 + (m[s] as f64 + 1.0) / 2.0;
                rank_sum_pos[s] += avg_rank * mp[s] as f64;
                n_pos[s] += mp[s];
                n_tot[s] += m[s];
                cum[s] += m[s];
            }
        }
        i = j;
    }

    let auc = |s: usize| -> f64 {
        let p = n_pos[s] as f64;
        let q = (n_tot[s] - n_pos[s]) as f64;
        if p == 0.0 || q == 0.0 {
            return f64::NAN;
        }
        (rank_sum_pos[s] - p * (p + 1.0) / 2.0) / (p * q)
    };
    (auc(0), auc(1), auc(2))
}

/// Plain (unweighted) AUC over an arbitrary subset, for the per-file
/// daily estimand where a fresh sort per file is cheap.
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
            .expect("finite scores")
    });
    let mut rank_sum_pos = 0.0_f64;
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && scores[order[j] as usize] == scores[order[i] as usize] {
            j += 1;
        }
        let avg_rank = (i + 1 + j) as f64 / 2.0;
        for &idx in &order[i..j] {
            if labels[idx as usize] == 1 {
                rank_sum_pos += avg_rank;
            }
        }
        i = j;
    }
    let p = n_pos as f64;
    (rank_sum_pos - p * (p + 1.0) / 2.0) / (p * n_neg as f64)
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
    n_le_zero: usize,
    n_draws: usize,
}

fn ci(point: f64, draws: &[f64]) -> CiSummary {
    let mut finite: Vec<f64> = draws.iter().copied().filter(|x| x.is_finite()).collect();
    finite.sort_unstable_by(|a, b| a.partial_cmp(b).expect("finite draws"));
    CiSummary {
        point,
        lo: percentile(&finite, 0.025),
        hi: percentile(&finite, 0.975),
        n_le_zero: finite.iter().filter(|&&d| d <= 0.0).count(),
        n_draws: finite.len(),
    }
}

/// Tail proportion with Monte Carlo resolution made explicit: zero
/// nonpositive draws reports as "< 1/(B+1)", never exact zero (a
/// randomly drawn tail probability has resolution one draw).
fn json_p(n_le_zero: usize, n_draws: usize) -> String {
    if n_le_zero == 0 {
        format!("\"< {:.6}\"", 1.0 / (n_draws + 1) as f64)
    } else {
        format!("{:.6}", (n_le_zero + 1) as f64 / (n_draws + 1) as f64)
    }
}

fn json_ci(s: &CiSummary) -> String {
    format!(
        "{{\"point\": {:.6}, \"ci95\": [{:.6}, {:.6}], \"p_le_zero\": {}}}",
        s.point,
        s.lo,
        s.hi,
        json_p(s.n_le_zero, s.n_draws)
    )
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let mut file_id: Vec<u32> = Vec::new();
    let mut assoc: Vec<f32> = Vec::new();
    let mut dbdt: Vec<f32> = Vec::new();
    let mut rot: Vec<f32> = Vec::new();
    let mut labels: Vec<u8> = Vec::new();

    let reader = BufReader::new(File::open(&args.scores)?);
    for (line_no, line) in reader.lines().enumerate() {
        let line = line?;
        if line_no == 0 {
            anyhow::ensure!(
                line.starts_with("file_id,"),
                "scores CSV lacks the file_id column; re-export with the current exporter"
            );
            continue;
        }
        let mut cols = line.split(',');
        let f: u32 = cols.next().ok_or_else(|| anyhow::anyhow!("row"))?.parse()?;
        let a: f32 = cols.next().ok_or_else(|| anyhow::anyhow!("row"))?.parse()?;
        let d: f32 = cols.next().ok_or_else(|| anyhow::anyhow!("row"))?.parse()?;
        let r: f32 = cols.next().ok_or_else(|| anyhow::anyhow!("row"))?.parse()?;
        let _bmag = cols.next();
        let l: u8 = cols.next().ok_or_else(|| anyhow::anyhow!("row"))?.parse()?;
        file_id.push(f);
        assoc.push(a);
        dbdt.push(d);
        rot.push(r);
        labels.push(l);
    }
    let n = labels.len();
    let n_files = (*file_id.iter().max().ok_or_else(|| anyhow::anyhow!("empty"))? + 1) as usize;
    let n_pos = labels.iter().filter(|&&l| l == 1).count();
    eprintln!("loaded {} samples, {} positives, {} files", n, n_pos, n_files);

    // Fixed stratum definition: median |dB/dt| among positives.
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

    let det_assoc = DetectorOrder::new(&assoc);
    let det_rot = DetectorOrder::new(&rot);

    // Estimand 1: pooled time-point AUCs (all weights 1). This is the
    // deterministic descriptive result on the committed benchmark.
    let unit = vec![1u32; n];
    let (pt_assoc_bulk, pt_assoc_low, pt_assoc_high) =
        walk_auc_three_strata(&assoc, &det_assoc, &tags, &unit);
    let (pt_rot_bulk, pt_rot_low, pt_rot_high) =
        walk_auc_three_strata(&rot, &det_rot, &tags, &unit);
    let pt_m = (pt_assoc_bulk - pt_rot_bulk)
        .min(pt_assoc_low - pt_rot_low)
        .min(pt_assoc_high - pt_rot_high);
    eprintln!(
        "pooled: bulk {:.4}/{:.4} low {:.4}/{:.4} high {:.4}/{:.4} M={:.6}",
        pt_assoc_bulk, pt_rot_bulk, pt_assoc_low, pt_rot_low, pt_assoc_high, pt_rot_high, pt_m
    );

    // Estimand 2: mean daily AUC over files carrying both classes.
    let mut file_ranges: Vec<(usize, usize)> = vec![(usize::MAX, 0); n_files];
    for (idx, &f) in file_id.iter().enumerate() {
        let e = &mut file_ranges[f as usize];
        e.0 = e.0.min(idx);
        e.1 = e.1.max(idx + 1);
    }
    let daily: Vec<(f64, f64)> = file_ranges
        .par_iter()
        .filter_map(|&(s, e)| {
            if s == usize::MAX {
                return None;
            }
            let a = rank_auc(&assoc[s..e], &labels[s..e]);
            let r = rank_auc(&rot[s..e], &labels[s..e]);
            (a.is_finite() && r.is_finite()).then_some((a, r))
        })
        .collect();
    let n_daily = daily.len();
    let mean_daily_assoc = daily.iter().map(|d| d.0).sum::<f64>() / n_daily as f64;
    let mean_daily_rot = daily.iter().map(|d| d.1).sum::<f64>() / n_daily as f64;
    eprintln!(
        "mean daily AUC over {} two-class files: assoc {:.4} rot {:.4}",
        n_daily, mean_daily_assoc, mean_daily_rot
    );

    // Estimand 3: file-cluster bootstrap. Each resample draws n_files
    // whole files with replacement; sample weights are the draw counts
    // of their files. The paired margin M recomputes inside every
    // resample.
    struct Draw {
        bulk_assoc: f64,
        bulk_rot: f64,
        d_bulk: f64,
        d_low: f64,
        d_high: f64,
        m: f64,
        d_daily_mean: f64,
    }
    let draws: Vec<Draw> = (0..args.resamples)
        .into_par_iter()
        .map(|rep| {
            let mut rng = ChaCha8Rng::seed_from_u64(args.seed.wrapping_add(rep as u64));
            let mut file_count = vec![0u32; n_files];
            for _ in 0..n_files {
                file_count[rng.random_range(0..n_files)] += 1;
            }
            let weight: Vec<u32> = file_id.iter().map(|&f| file_count[f as usize]).collect();
            let (ab, al, ah) = walk_auc_three_strata(&assoc, &det_assoc, &tags, &weight);
            let (rb, rl, rh) = walk_auc_three_strata(&rot, &det_rot, &tags, &weight);
            // Daily-mean delta reuses the per-file AUCs weighted by the
            // same draw counts, matching the cluster design. Two-class
            // file k appears daily[k] times; recover its file index by
            // walking files with both classes in the same order.
            let mut acc = 0.0f64;
            let mut tot = 0u64;
            let mut daily_iter = daily.iter();
            for &(s, e) in &file_ranges {
                if s == usize::MAX {
                    continue;
                }
                let has_pos = labels[s..e].contains(&1);
                let has_neg = labels[s..e].contains(&0);
                if has_pos && has_neg {
                    let &(da, dr) = daily_iter.next().expect("daily list aligned");
                    let w = u64::from(file_count[file_id[s] as usize]);
                    acc += (da - dr) * w as f64;
                    tot += w;
                }
            }
            Draw {
                bulk_assoc: ab,
                bulk_rot: rb,
                d_bulk: ab - rb,
                d_low: al - rl,
                d_high: ah - rh,
                m: (ab - rb).min(al - rl).min(ah - rh),
                d_daily_mean: if tot > 0 { acc / tot as f64 } else { f64::NAN },
            }
        })
        .collect();

    let collect = |f: &dyn Fn(&Draw) -> f64| -> Vec<f64> { draws.iter().map(f).collect() };
    let s_bulk_assoc = ci(pt_assoc_bulk, &collect(&|d| d.bulk_assoc));
    let s_bulk_rot = ci(pt_rot_bulk, &collect(&|d| d.bulk_rot));
    let s_d_bulk = ci(pt_assoc_bulk - pt_rot_bulk, &collect(&|d| d.d_bulk));
    let s_d_low = ci(pt_assoc_low - pt_rot_low, &collect(&|d| d.d_low));
    let s_d_high = ci(pt_assoc_high - pt_rot_high, &collect(&|d| d.d_high));
    let s_m = ci(pt_m, &collect(&|d| d.m));
    let s_d_daily = ci(
        mean_daily_assoc - mean_daily_rot,
        &collect(&|d| d.d_daily_mean),
    );

    let report = format!(
        "{{\n  \"design\": \"file_cluster_bootstrap\",\n  \"n_samples\": {},\n  \"n_positives\": {},\n  \"n_files\": {},\n  \"n_two_class_files\": {},\n  \"stratum_split_dbdt\": {:.6e},\n  \"resamples\": {},\n  \"seed\": {},\n  \"pooled_auc_assoc\": {},\n  \"pooled_auc_rot\": {},\n  \"delta_bulk\": {},\n  \"delta_low_gradient_positive\": {},\n  \"delta_high_gradient_positive\": {},\n  \"simultaneous_margin_m\": {},\n  \"mean_daily_auc\": {{\"assoc\": {:.6}, \"rot\": {:.6}, \"delta\": {}}}\n}}\n",
        n,
        n_pos,
        n_files,
        n_daily,
        split,
        args.resamples,
        args.seed,
        json_ci(&s_bulk_assoc),
        json_ci(&s_bulk_rot),
        json_ci(&s_d_bulk),
        json_ci(&s_d_low),
        json_ci(&s_d_high),
        json_ci(&s_m),
        mean_daily_assoc,
        mean_daily_rot,
        json_ci(&s_d_daily),
    );
    File::create(&args.out)?.write_all(report.as_bytes())?;
    println!("{}", report);
    Ok(())
}
