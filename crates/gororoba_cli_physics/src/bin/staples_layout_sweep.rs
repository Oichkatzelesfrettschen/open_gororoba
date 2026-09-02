//! Embedding-layout sweep for the staple associator (preregistered).
//!
//! The staple vector assigns four lags and three field components (plus
//! the fixed magnitude channel) to the sixteen sedenion basis slots. That
//! assignment is a free choice the pipeline never varied: `permute_channels`
//! probes one component permutation, and no lane has permuted the lags.
//! The sweep scores the canonical CD tensor under every structured layout,
//! all 4! x 3! = 144 (lag permutation, component permutation) pairs with the
//! magnitude channel held in slot 3, on the identical stratified subsample.
//!
//! Preregistered reading: the canonical layout (identity permutations) is
//! declared in advance. Under exchangeability of layouts its rank among the
//! 144 is uniform, so rank r has probability r/144 of arising by selection.
//! An equivalence margin of 0.005 AUC is declared: layouts within that
//! margin of the canonical are read as equivalent, and a canonical layout
//! that beats every other layout by more than the margin marks the
//! lag-to-basis assignment as a fitted parameter of the detector.

use anyhow::{Context, Result};
use cd_kernel::mult_table::CdMultTable;
use clap::Parser;
use gororoba_cli_physics::staple_associator::{STAPLE_DIM, STAPLE_LAGS};
use gororoba_cli_physics::staple_benchmark::{
    LabeledFile, load_labeled_files, permutations, rank_auc, staple_embedding_layout,
    stratified_keep,
};
use gororoba_cli_physics::staple_controls::SparseCubicTensor;
use rayon::prelude::*;
use serde::Serialize;
use std::{fs, path::PathBuf, time::Instant};

#[derive(Parser, Debug)]
#[command(about = "Score the CD associator under all 144 structured staple layouts")]
struct Args {
    #[arg(long)]
    matched_files: PathBuf,
    #[arg(long)]
    catalog: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 0.05)]
    neg_fraction: f64,
    #[arg(long, default_value_t = 2.0)]
    label_pad_minutes: f64,
    #[arg(long, default_value_t = 500)]
    min_samples: usize,
    /// AUC difference below which two layouts are read as equivalent.
    #[arg(long, default_value_t = 0.005)]
    equivalence_margin: f64,
}

#[derive(Serialize, Clone)]
struct LayoutResult {
    lag_perm: [usize; STAPLE_LAGS],
    comp_perm: [usize; 3],
    auc: f64,
    canonical: bool,
}

#[derive(Serialize)]
struct Output {
    preregistration: Preregistration,
    files: usize,
    subsample_size: usize,
    subsample_positives: usize,
    canonical_auc: f64,
    canonical_rank: usize,
    rank_probability_under_exchangeability: f64,
    canonical_percentile: f64,
    best: LayoutResult,
    best_minus_canonical: f64,
    layouts_within_margin_of_canonical: usize,
    layouts_above_canonical_by_more_than_margin: usize,
    distribution: DistributionSummary,
    layouts: Vec<LayoutResult>,
    decision: String,
    elapsed_seconds: f64,
}

#[derive(Serialize)]
struct DistributionSummary {
    n: usize,
    mean: f64,
    std: f64,
    min: f64,
    median: f64,
    max: f64,
}

#[derive(Serialize)]
struct Preregistration {
    hypothesis: &'static str,
    canonical_layout: &'static str,
    falsifier: &'static str,
    equivalence_margin: f64,
    seed: u64,
    neg_fraction: f64,
}

struct Kept {
    file: LabeledFile,
    keep: Vec<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let start = Instant::now();
    let files = load_labeled_files(
        &args.matched_files,
        &args.catalog,
        args.label_pad_minutes,
        args.min_samples,
    )?;
    anyhow::ensure!(!files.is_empty(), "no labeled files");
    let kept: Vec<Kept> = files
        .into_iter()
        .map(|file| {
            let keep = stratified_keep(&file.labels, file.file_id, args.seed, args.neg_fraction);
            Kept { file, keep }
        })
        .collect();
    let labels: Vec<bool> = kept
        .iter()
        .flat_map(|k| k.keep.iter().map(|&i| k.file.labels[i]))
        .collect();
    eprintln!(
        "{} files, subsample {} ({} positives) in {:.0} s",
        kept.len(),
        labels.len(),
        labels.iter().filter(|&&l| l).count(),
        start.elapsed().as_secs_f64()
    );

    let table = CdMultTable::generate(STAPLE_DIM);
    let cd = SparseCubicTensor::from_associator(&table);

    let lag_perms = permutations::<STAPLE_LAGS>();
    let comp_perms = permutations::<3>();
    let mut layouts: Vec<LayoutResult> = Vec::with_capacity(lag_perms.len() * comp_perms.len());
    for &lag_perm in &lag_perms {
        for &comp_perm in &comp_perms {
            let scores: Vec<f64> = kept
                .par_iter()
                .flat_map_iter(|k| {
                    let staples = staple_embedding_layout(&k.file.rows, lag_perm, comp_perm);
                    k.keep
                        .iter()
                        .map(|&i| {
                            cd.normalized_score(&staples[i], &staples[i + 1], &staples[i + 2])
                        })
                        .collect::<Vec<f64>>()
                })
                .collect();
            let auc = rank_auc(&scores, &labels);
            let canonical = lag_perm == [0, 1, 2, 3] && comp_perm == [0, 1, 2];
            layouts.push(LayoutResult {
                lag_perm,
                comp_perm,
                auc,
                canonical,
            });
            eprintln!(
                "layout {:?}/{:?}: auc {:.4}{} ({:.0} s)",
                lag_perm,
                comp_perm,
                auc,
                if canonical { " canonical" } else { "" },
                start.elapsed().as_secs_f64()
            );
        }
    }

    let canonical = layouts
        .iter()
        .find(|l| l.canonical)
        .context("canonical layout scored")?
        .clone();
    let canonical_rank = 1 + layouts.iter().filter(|l| l.auc > canonical.auc).count();
    let n = layouts.len();
    let canonical_percentile =
        layouts.iter().filter(|l| l.auc <= canonical.auc).count() as f64 / n as f64;
    let best = layouts
        .iter()
        .max_by(|a, b| {
            a.auc
                .partial_cmp(&b.auc)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .context("best layout")?
        .clone();
    let within = layouts
        .iter()
        .filter(|l| !l.canonical && (l.auc - canonical.auc).abs() <= args.equivalence_margin)
        .count();
    let above = layouts
        .iter()
        .filter(|l| l.auc - canonical.auc > args.equivalence_margin)
        .count();
    let mut sorted: Vec<f64> = layouts.iter().map(|l| l.auc).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let std = (sorted.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0)).sqrt();
    let distribution = DistributionSummary {
        n,
        mean,
        std,
        min: sorted[0],
        median: sorted[n / 2],
        max: sorted[n - 1],
    };
    let decision = if above == 0 && canonical_rank <= (n as f64 * 0.05).ceil() as usize {
        format!(
            "canonical layout is a fitted parameter: rank {canonical_rank} of {n} (selection probability {:.3}) and no layout beats it by more than {}; the declared parameter count of the detector rises by one",
            canonical_rank as f64 / n as f64,
            args.equivalence_margin
        )
    } else if above == 0 {
        format!(
            "layout-insensitive within the margin: rank {canonical_rank} of {n}, {within} layouts within {} of the canonical AUC and none above it by more; the assignment is not a fitted parameter",
            args.equivalence_margin
        )
    } else {
        format!(
            "canonical layout is not the best: {above} layouts exceed it by more than {} (best {:?}/{:?} at {:.4} against {:.4}); the reported AUC understates what the tensor can reach and the assignment is a free parameter that was never tuned",
            args.equivalence_margin, best.lag_perm, best.comp_perm, best.auc, canonical.auc
        )
    };
    eprintln!("{decision}");

    layouts.sort_by(|a, b| {
        b.auc
            .partial_cmp(&a.auc)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let output = Output {
        preregistration: Preregistration {
            hypothesis: "The canonical staple layout (time-ordered lags, Bx By Bz magnitude) is one arbitrary member of the 144 structured layouts and its AUC is typical of them.",
            canonical_layout: "lag_perm [0,1,2,3], comp_perm [0,1,2], magnitude fixed in channel slot 3",
            falsifier: "The canonical layout ranks in the top 5% of the 144 layouts with no other layout within the equivalence margin above it, which makes the assignment a fitted parameter; or layouts exceed it by more than the margin, which makes the reported AUC a lower bound on what the tensor reaches.",
            equivalence_margin: args.equivalence_margin,
            seed: args.seed,
            neg_fraction: args.neg_fraction,
        },
        files: kept.len(),
        subsample_size: labels.len(),
        subsample_positives: labels.iter().filter(|&&l| l).count(),
        canonical_auc: canonical.auc,
        canonical_rank,
        rank_probability_under_exchangeability: canonical_rank as f64 / n as f64,
        canonical_percentile,
        best_minus_canonical: best.auc - canonical.auc,
        best,
        layouts_within_margin_of_canonical: within,
        layouts_above_canonical_by_more_than_margin: above,
        distribution,
        layouts,
        decision,
        elapsed_seconds: start.elapsed().as_secs_f64(),
    };
    if let Some(parent) = args.out.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&args.out, serde_json::to_string_pretty(&output)?)
        .with_context(|| format!("write {}", args.out.display()))?;
    eprintln!("wrote {}", args.out.display());
    Ok(())
}
