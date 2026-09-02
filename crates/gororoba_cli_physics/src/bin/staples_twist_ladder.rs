//! Random-twist ladder for the staple associator (preregistered).
//!
//! The sedenion associator tensor is the associator of one twisted group
//! algebra of (Z_2)^4: e_i e_j = sigma(i,j) e_{i XOR j} with sigma the
//! Cayley-Dickson sign table. The existing controls sit below it: the
//! sign scramble (C-1632) keeps the CD support but is no algebra, the
//! sparsity-matched and dense random tensors are no algebras either. This
//! ladder adds the rung in between: `n_draws` uniformly random unital
//! twists, each a genuine XOR-graded algebra with its own associator, its
//! own support and unit coefficients, scored by the identical pipeline on
//! the identical sample.
//!
//! Sample: every positive of the E-239 benchmark plus a seeded
//! `neg_fraction` of its negatives, stratified by daily file. Pooled AUC on
//! that subsample estimates the full pooled AUC without bias, and the
//! canonical tensor is also scored on the full sample for reference.
//!
//! Preregistered reading: the CD tensor is twist-specific when its AUC
//! lies above the 97.5th percentile of the random-twist ensemble (tail
//! probability (1 + #draws >= CD) / (n_draws + 1) below 0.025). A CD AUC
//! inside the ensemble means the detector measures XOR-graded cubic
//! structure that any twist supplies, and C-1632 narrows from
//! "sedenion-specific" to "algebra-class-specific".

use anyhow::{Context, Result};
use cd_kernel::mult_table::CdMultTable;
use clap::Parser;
use gororoba_cli_physics::staple_associator::{
    STAPLE_DIM, joint_associator_norms, staple_embedding,
};
use gororoba_cli_physics::staple_benchmark::{
    load_labeled_files, percentile, rank_auc, stratified_keep,
};
use gororoba_cli_physics::staple_controls::{
    SparseCubicTensor, cd_twist, random_unital_twist, twist_sha256,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde::Serialize;
use std::{fs, path::PathBuf, time::Instant};

#[derive(Parser, Debug)]
#[command(
    about = "Random unital twist ensemble against the sedenion associator on the staples benchmark"
)]
struct Args {
    /// CSV with a `path` column listing daily THEMIS-A FGM files.
    #[arg(long)]
    matched_files: PathBuf,
    /// Crossing catalog CSV with a `TIMESTAMP` column (Staples et al. 2020).
    #[arg(long)]
    catalog: PathBuf,
    /// Output JSON.
    #[arg(long)]
    out: PathBuf,
    /// Random unital twists to draw.
    #[arg(long, default_value_t = 999)]
    n_draws: usize,
    /// Exact-support sign scrambles of the CD tensor (the C-1632 control family).
    #[arg(long, default_value_t = 999)]
    n_scrambles: usize,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Fraction of negatives kept per file; every positive is kept.
    #[arg(long, default_value_t = 0.05)]
    neg_fraction: f64,
    #[arg(long, default_value_t = 2.0)]
    label_pad_minutes: f64,
    #[arg(long, default_value_t = 500)]
    min_samples: usize,
}

struct Triple {
    a: [f64; STAPLE_DIM],
    b: [f64; STAPLE_DIM],
    c: [f64; STAPLE_DIM],
}

#[derive(Serialize)]
struct Draw {
    index: usize,
    auc: f64,
    term_count: usize,
    positive_terms: usize,
    negative_terms: usize,
    twist_sha256: String,
}

#[derive(Serialize)]
struct ScrambleDraw {
    index: usize,
    seed: u64,
    auc: f64,
}

#[derive(Serialize)]
struct EnsembleSummary {
    n: usize,
    mean: f64,
    std: f64,
    min: f64,
    p2_5: f64,
    p50: f64,
    p97_5: f64,
    max: f64,
    /// (1 + #{draw AUC >= canonical AUC}) / (n + 1)
    tail_probability_ge_canonical: f64,
    draws_at_or_above_canonical: usize,
}

#[derive(Serialize)]
struct Output {
    preregistration: Preregistration,
    sample: SampleRecord,
    canonical: CanonicalRecord,
    random_twist: EnsembleSummary,
    sign_scramble: EnsembleSummary,
    twist_draws: Vec<Draw>,
    scramble_draws: Vec<ScrambleDraw>,
    decision: String,
    elapsed_seconds: f64,
}

#[derive(Serialize)]
struct Preregistration {
    hypothesis: &'static str,
    control_family: &'static str,
    falsifier: &'static str,
    seed_policy: String,
    n_draws: usize,
    n_scrambles: usize,
    neg_fraction: f64,
    label_pad_minutes: f64,
    min_samples: usize,
}

#[derive(Serialize)]
struct SampleRecord {
    files: usize,
    full_samples: usize,
    full_positives: usize,
    subsample_size: usize,
    subsample_positives: usize,
}

#[derive(Serialize)]
struct CanonicalRecord {
    full_sample_auc: f64,
    subsample_auc: f64,
    term_count: usize,
    positive_terms: usize,
    negative_terms: usize,
    twist_sha256: String,
    from_twist_reproduces_from_associator: bool,
}

fn summarize(aucs: &[f64], canonical: f64) -> EnsembleSummary {
    let mut sorted = aucs.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let std =
        (sorted.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / (n as f64 - 1.0).max(1.0)).sqrt();
    let above = sorted.iter().filter(|&&a| a >= canonical).count();
    EnsembleSummary {
        n,
        mean,
        std,
        min: sorted[0],
        p2_5: percentile(&sorted, 0.025),
        p50: percentile(&sorted, 0.5),
        p97_5: percentile(&sorted, 0.975),
        max: sorted[n - 1],
        tail_probability_ge_canonical: (1 + above) as f64 / (n + 1) as f64,
        draws_at_or_above_canonical: above,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let start = Instant::now();
    eprintln!("loading matched files ...");
    let files = load_labeled_files(
        &args.matched_files,
        &args.catalog,
        args.label_pad_minutes,
        args.min_samples,
    )?;
    anyhow::ensure!(!files.is_empty(), "no labeled files");

    let table = CdMultTable::generate(STAPLE_DIM);
    let cd = SparseCubicTensor::from_associator(&table);
    let sigma0 = cd_twist(&table);
    let cd_from_twist = SparseCubicTensor::from_twist(&sigma0);

    let mut full_scores: Vec<f64> = Vec::new();
    let mut full_labels: Vec<bool> = Vec::new();
    let mut triples: Vec<Triple> = Vec::new();
    let mut sub_labels: Vec<bool> = Vec::new();
    for f in &files {
        let staples = staple_embedding(&f.rows);
        let assoc = joint_associator_norms(&staples, true);
        anyhow::ensure!(
            assoc.len() == f.labels.len(),
            "label alignment in {}",
            f.path
        );
        full_scores.extend_from_slice(&assoc);
        full_labels.extend_from_slice(&f.labels);
        for k in stratified_keep(&f.labels, f.file_id, args.seed, args.neg_fraction) {
            triples.push(Triple {
                a: staples[k],
                b: staples[k + 1],
                c: staples[k + 2],
            });
            sub_labels.push(f.labels[k]);
        }
    }
    eprintln!(
        "{} files, {} samples ({} positives), subsample {} ({} positives) in {:.0} s",
        files.len(),
        full_scores.len(),
        full_labels.iter().filter(|&&l| l).count(),
        triples.len(),
        sub_labels.iter().filter(|&&l| l).count(),
        start.elapsed().as_secs_f64()
    );
    let sample = SampleRecord {
        files: files.len(),
        full_samples: full_scores.len(),
        full_positives: full_labels.iter().filter(|&&l| l).count(),
        subsample_size: triples.len(),
        subsample_positives: sub_labels.iter().filter(|&&l| l).count(),
    };
    drop(files);

    let full_sample_auc = rank_auc(&full_scores, &full_labels);
    drop(full_scores);
    drop(full_labels);

    let score = |t: &SparseCubicTensor| -> Vec<f64> {
        triples
            .par_iter()
            .map(|tr| t.normalized_score(&tr.a, &tr.b, &tr.c))
            .collect()
    };
    let cd_scores = score(&cd);
    let twist_scores = score(&cd_from_twist);
    let reproduces = cd_from_twist.term_count() == cd.term_count()
        && cd_scores
            .iter()
            .zip(&twist_scores)
            .all(|(x, y)| (x - y).abs() <= 1e-12 * x.abs().max(1.0));
    anyhow::ensure!(
        reproduces,
        "from_twist(cd_twist) must reproduce from_associator"
    );
    let canonical_auc = rank_auc(&cd_scores, &sub_labels);
    let (cd_pos, cd_neg) = cd.sign_counts();
    eprintln!(
        "canonical CD AUC: full {:.4}, subsample {:.4} ({} terms)",
        full_sample_auc,
        canonical_auc,
        cd.term_count()
    );

    let mut twist_draws = Vec::with_capacity(args.n_draws);
    for d in 0..args.n_draws {
        let mut rng = ChaCha8Rng::seed_from_u64(
            args.seed
                .wrapping_add((d as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        );
        let sigma = random_unital_twist(&mut rng);
        let t = SparseCubicTensor::from_twist(&sigma);
        let auc = rank_auc(&score(&t), &sub_labels);
        let (pos, neg) = t.sign_counts();
        twist_draws.push(Draw {
            index: d,
            auc,
            term_count: t.term_count(),
            positive_terms: pos,
            negative_terms: neg,
            twist_sha256: twist_sha256(&sigma),
        });
        if (d + 1) % 25 == 0 {
            eprintln!(
                "twist draw {}/{}: auc {:.4}, terms {} ({:.0} s)",
                d + 1,
                args.n_draws,
                auc,
                t.term_count(),
                start.elapsed().as_secs_f64()
            );
        }
    }

    let mut scramble_draws = Vec::with_capacity(args.n_scrambles);
    for d in 0..args.n_scrambles {
        let seed = args.seed + d as u64;
        let t = cd.sign_scrambled(seed);
        let auc = rank_auc(&score(&t), &sub_labels);
        scramble_draws.push(ScrambleDraw {
            index: d,
            seed,
            auc,
        });
        if (d + 1) % 25 == 0 {
            eprintln!(
                "scramble {}/{}: auc {:.4} ({:.0} s)",
                d + 1,
                args.n_scrambles,
                auc,
                start.elapsed().as_secs_f64()
            );
        }
    }

    let random_twist = summarize(
        &twist_draws.iter().map(|d| d.auc).collect::<Vec<_>>(),
        canonical_auc,
    );
    let sign_scramble = summarize(
        &scramble_draws.iter().map(|d| d.auc).collect::<Vec<_>>(),
        canonical_auc,
    );
    let decision = if random_twist.tail_probability_ge_canonical < 0.025 {
        format!(
            "twist-specific: canonical AUC {canonical_auc:.4} exceeds the random-twist 97.5th percentile {:.4} (tail probability {:.4})",
            random_twist.p97_5, random_twist.tail_probability_ge_canonical
        )
    } else {
        format!(
            "algebra-class-specific at most: canonical AUC {canonical_auc:.4} sits inside the random-twist ensemble [{:.4}, {:.4}] (tail probability {:.4}); C-1632 narrows from sedenion-specific to XOR-graded-algebra-specific",
            random_twist.p2_5, random_twist.p97_5, random_twist.tail_probability_ge_canonical
        )
    };
    eprintln!("{decision}");

    let output = Output {
        preregistration: Preregistration {
            hypothesis: "The sedenion associator tensor discriminates THEMIS-A magnetopause crossings better than the associator of a random unital XOR-graded twist on the same basis.",
            control_family: "n_draws uniformly random unital twists sigma: {1..15}^2 -> {+1,-1} with sigma(0,.) = sigma(.,0) = +1, each scored through SparseCubicTensor::from_twist on the identical stratified subsample; plus n_scrambles exact-support sign scrambles of the CD tensor (seeds seed..seed+n_scrambles-1, so seed 42 reproduces C-1632) as the lower rung.",
            falsifier: "Canonical AUC at or below the 97.5th percentile of the random-twist AUC ensemble (tail probability >= 0.025).",
            seed_policy: format!(
                "ChaCha8; twist d uses seed {} + (d+1)*0x9E3779B97F4A7C15 (wrapping); subsample keeps every positive and each negative with probability {} from ChaCha8(seed XOR file_id*0x9E3779B97F4A7C15)",
                args.seed, args.neg_fraction
            ),
            n_draws: args.n_draws,
            n_scrambles: args.n_scrambles,
            neg_fraction: args.neg_fraction,
            label_pad_minutes: args.label_pad_minutes,
            min_samples: args.min_samples,
        },
        sample,
        canonical: CanonicalRecord {
            full_sample_auc,
            subsample_auc: canonical_auc,
            term_count: cd.term_count(),
            positive_terms: cd_pos,
            negative_terms: cd_neg,
            twist_sha256: twist_sha256(&sigma0),
            from_twist_reproduces_from_associator: reproduces,
        },
        random_twist,
        sign_scramble,
        twist_draws,
        scramble_draws,
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
