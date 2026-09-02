//! Preregistered incremental-information test for the CD associator against
//! six-sample geometric baselines on the Staples-labeled THEMIS benchmark.
//!
//! Two logistic models are fitted on identical rows: baseline B over
//! ln(dbdt), ln(rot), ln(cumrot6), ln(maxrot6), ln(pvi6), ln(gram6), and
//! B+A which adds ln(assoc).  Validation is 5-fold grouped on the intact
//! daily-file cluster `file_id`, so a window never straddles the split.
//! The paired file-level bootstrap answers one question: does the
//! associator move ROC-AUC beyond the baselines at matched receptive
//! field, or is it redundant with them.
//!
//! Out-of-fold predictions are held as logits in f32.  An f32 probability
//! saturates to exactly 0 or 1 at |eta| above roughly 17, and ln(p) then
//! sends the whole log loss to -inf; the logit form stays finite and
//! log loss comes from softplus(eta) - y*eta.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write as _;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use serde_json::{Value, json};

/// Feature columns in fitting order.  The baseline occupies the first six
/// slots; `assoc` is appended so model B is a strict prefix of model B+A.
const FEATURE_NAMES: [&str; 7] = [
    "ln_dbdt",
    "ln_rot",
    "ln_cumrot6",
    "ln_maxrot6",
    "ln_pvi6",
    "ln_gram6",
    "ln_assoc",
];

/// CSV column index for each entry of `FEATURE_NAMES`.
/// Header: file_id,assoc,dbdt,rot,bmag,label,cumrot6,maxrot6,pvi6,gram6,scram,chperm
const FEATURE_CSV_COL: [usize; 7] = [2, 3, 6, 7, 8, 9, 1];
const COL_FILE_ID: usize = 0;
const COL_LABEL: usize = 5;

const N_FEATURES: usize = 7;
const N_BASELINE: usize = 6;
const LOG_EPS: f64 = 1e-12;
const RIDGE: f64 = 1e-6;
const MAX_NEWTON_ITERS: usize = 25;
/// Relative deviance tolerance.  The penalized deviance runs near 4e6 on the
/// full benchmark, where an absolute 1e-8 sits about a decade above the f64
/// resolution of the sum and depends on the rayon reduction order, so the
/// stopping rule would never fire and every fold would silently exhaust the
/// iteration cap.
const DEVIANCE_REL_TOL: f64 = 1e-8;
const N_FOLDS: usize = 5;
const N_CALIBRATION_BINS: usize = 10;
/// Odd multiplier from the golden-ratio constant used to decorrelate the
/// per-resample ChaCha8 stream keys.
const RESAMPLE_KEY_ODD: u64 = 0x9E37_79B9_7F4A_7C15;

#[derive(Parser, Debug)]
#[command(
    name = "staples-incremental-information",
    about = "Preregistered grouped-CV incremental-information test for the CD associator"
)]
struct Args {
    /// Benchmark score CSV emitted by themis-staples-score-export.
    #[arg(long, default_value = "data/output/benchmark_scores.csv")]
    input: PathBuf,
    /// Destination JSON report.
    #[arg(long, default_value = "data/output/staples_incremental_information.json")]
    output: PathBuf,
    /// Master seed for the fold shuffle and the bootstrap stream keys.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Paired file-level bootstrap resamples.
    #[arg(long, default_value_t = 1000)]
    n_boot: usize,
    /// Stop after this many data rows; 0 reads the whole file.
    #[arg(long, default_value_t = 0)]
    max_rows: usize,
}

// ---------------------------------------------------------------------------
// numerics
// ---------------------------------------------------------------------------

/// Sigmoid branched on the sign of `eta` so neither exp() overflows.
fn sigmoid(eta: f64) -> f64 {
    if eta >= 0.0 {
        1.0 / (1.0 + (-eta).exp())
    } else {
        let e = eta.exp();
        e / (1.0 + e)
    }
}

/// softplus(eta) = ln(1 + exp(eta)), evaluated through ln_1p on the negative
/// branch so a large positive eta returns eta rather than inf.
fn softplus(eta: f64) -> f64 {
    eta.max(0.0) + (-(eta.abs())).exp().ln_1p()
}

/// Per-sample logistic loss from the logit: softplus(eta) - y*eta.
fn logit_loss(eta: f64, y: f64) -> f64 {
    softplus(eta) - y * eta
}

/// In-place Cholesky factorization of a symmetric positive definite matrix
/// stored row-major, returning the lower triangle.  The ridge makes the
/// Newton Hessian SPD even when a fold's weights underflow on saturated rows.
fn cholesky(a: &mut [f64], p: usize) -> Result<()> {
    for i in 0..p {
        for j in 0..=i {
            let mut s = a[i * p + j];
            for k in 0..j {
                s -= a[i * p + k] * a[j * p + k];
            }
            if i == j {
                if s <= 0.0 {
                    bail!("Hessian lost positive definiteness at pivot {i}");
                }
                a[i * p + i] = s.sqrt();
            } else {
                a[i * p + j] = s / a[j * p + j];
            }
        }
        for j in (i + 1)..p {
            a[i * p + j] = 0.0;
        }
    }
    Ok(())
}

/// Solve `L L^T x = b` for a lower-triangular Cholesky factor.
fn cholesky_solve(l: &[f64], b: &[f64], p: usize) -> Vec<f64> {
    let mut y = vec![0.0f64; p];
    for i in 0..p {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * p + k] * y[k];
        }
        y[i] = s / l[i * p + i];
    }
    let mut x = vec![0.0f64; p];
    for i in (0..p).rev() {
        let mut s = y[i];
        for k in (i + 1)..p {
            s -= l[k * p + i] * x[k];
        }
        x[i] = s / l[i * p + i];
    }
    x
}

// ---------------------------------------------------------------------------
// IRLS
// ---------------------------------------------------------------------------

/// Fitted coefficients on standardized features plus the fit trajectory.
#[derive(Debug, Clone)]
struct FitResult {
    /// `beta[0]` is the intercept; `beta[1..]` follow `cols` order.
    beta: Vec<f64>,
    deviance: Vec<f64>,
    iterations: usize,
    converged: bool,
}

/// Newton-Raphson (IRLS) logistic fit over `rows` of a row-major f32 feature
/// block, using the feature columns named in `cols` after standardization by
/// `(mean, std)`.  The ridge penalty applies to the standardized slopes only;
/// the intercept is unpenalized, which is the textbook convention and keeps
/// the fit invariant under a shift of the prevalence.
#[allow(clippy::too_many_arguments)]
fn fit_irls(
    feats: &[f32],
    stride: usize,
    cols: &[usize],
    rows: &[u32],
    labels: &[u8],
    mean: &[f64],
    std: &[f64],
    ridge: f64,
) -> Result<FitResult> {
    let d = cols.len();
    let p = d + 1;
    let mut beta = vec![0.0f64; p];
    // Unpenalized intercept-only start: the log odds of the observed rate.
    let pos = rows.iter().filter(|&&r| labels[r as usize] == 1).count() as f64;
    let n = rows.len() as f64;
    let rate = (pos / n).clamp(1e-12, 1.0 - 1e-12);
    beta[0] = (rate / (1.0 - rate)).ln();

    let mut trajectory = Vec::with_capacity(MAX_NEWTON_ITERS + 1);
    let mut prev_dev = f64::INFINITY;
    let mut iterations = 0usize;
    let mut converged = false;

    for _ in 0..MAX_NEWTON_ITERS {
        iterations += 1;
        // One pass accumulates the deviance, the gradient and the upper
        // triangle of X^T W X.  Chunked so rayon reduces partial sums.
        let (dev, grad, hess) = rows
            .par_chunks(1 << 16)
            .map(|chunk| {
                let mut dev = 0.0f64;
                let mut grad = vec![0.0f64; p];
                let mut hess = vec![0.0f64; p * p];
                let mut x = vec![0.0f64; p];
                x[0] = 1.0;
                for &r in chunk {
                    let base = r as usize * stride;
                    for (k, &c) in cols.iter().enumerate() {
                        x[k + 1] = (f64::from(feats[base + c]) - mean[k]) / std[k];
                    }
                    let mut eta = 0.0f64;
                    for j in 0..p {
                        eta += beta[j] * x[j];
                    }
                    let y = f64::from(labels[r as usize]);
                    dev += logit_loss(eta, y);
                    let mu = sigmoid(eta);
                    let w = mu * (1.0 - mu);
                    let resid = y - mu;
                    for j in 0..p {
                        grad[j] += resid * x[j];
                        let wxj = w * x[j];
                        for k in j..p {
                            hess[j * p + k] += wxj * x[k];
                        }
                    }
                }
                (dev, grad, hess)
            })
            .reduce(
                || (0.0f64, vec![0.0f64; p], vec![0.0f64; p * p]),
                |mut a, b| {
                    a.0 += b.0;
                    for j in 0..p {
                        a.1[j] += b.1[j];
                    }
                    for j in 0..(p * p) {
                        a.2[j] += b.2[j];
                    }
                    a
                },
            );

        let mut grad = grad;
        let mut hess = hess;
        for j in 1..p {
            grad[j] -= ridge * beta[j];
            hess[j * p + j] += ridge;
        }
        for j in 0..p {
            for k in 0..j {
                hess[j * p + k] = hess[k * p + j];
            }
        }
        let penalty: f64 = ridge * beta[1..].iter().map(|b| b * b).sum::<f64>() * 0.5;
        let pen_dev = 2.0 * (dev + penalty);
        trajectory.push(pen_dev);

        cholesky(&mut hess, p)?;
        let step = cholesky_solve(&hess, &grad, p);
        for j in 0..p {
            beta[j] += step[j];
        }

        if (prev_dev - pen_dev).abs() / prev_dev.abs().max(1.0) < DEVIANCE_REL_TOL {
            converged = true;
            break;
        }
        prev_dev = pen_dev;
    }

    Ok(FitResult {
        beta,
        deviance: trajectory,
        iterations,
        converged,
    })
}

// ---------------------------------------------------------------------------
// metrics
// ---------------------------------------------------------------------------

/// Weighted ROC-AUC over samples pre-sorted by descending score, with tied
/// scores collapsed into one group so a tie contributes half its rectangle.
/// This is the Mann-Whitney statistic with average ranks.
fn weighted_auc(scores: &[f32], labels: &[u8], weights: &[f64]) -> f64 {
    let n = scores.len();
    let mut total_pos = 0.0f64;
    let mut total_neg = 0.0f64;
    for i in 0..n {
        if labels[i] == 1 {
            total_pos += weights[i];
        } else {
            total_neg += weights[i];
        }
    }
    if total_pos <= 0.0 || total_neg <= 0.0 {
        return f64::NAN;
    }
    // Descending order, so the negatives a positive outranks are the ones
    // still ahead in the scan; a tied group splits its rectangle in half.
    let mut auc = 0.0f64;
    let mut neg_seen = 0.0f64;
    let mut i = 0usize;
    while i < n {
        let mut j = i;
        let mut gp = 0.0f64;
        let mut gn = 0.0f64;
        while j < n && scores[j] == scores[i] {
            let w = weights[j];
            if labels[j] == 1 { gp += w } else { gn += w }
            j += 1;
        }
        auc += gp * ((total_neg - neg_seen - gn) + 0.5 * gn);
        neg_seen += gn;
        i = j;
    }
    auc / (total_pos * total_neg)
}

/// Weighted average precision over samples pre-sorted by descending score.
/// One precision/recall point per distinct score, matching the
/// distinct-threshold convention, and the same tie grouping as `weighted_auc`.
fn weighted_average_precision(scores: &[f32], labels: &[u8], weights: &[f64]) -> f64 {
    let n = scores.len();
    let total_pos: f64 = (0..n)
        .filter(|&i| labels[i] == 1)
        .map(|i| weights[i])
        .sum();
    if total_pos <= 0.0 {
        return f64::NAN;
    }
    let mut tp = 0.0f64;
    let mut fp = 0.0f64;
    let mut prev_recall = 0.0f64;
    let mut ap = 0.0f64;
    let mut i = 0usize;
    while i < n {
        let mut j = i;
        while j < n && scores[j] == scores[i] {
            let w = weights[j];
            if labels[j] == 1 { tp += w } else { fp += w }
            j += 1;
        }
        let denom = tp + fp;
        if denom > 0.0 {
            let recall = tp / total_pos;
            ap += (recall - prev_recall) * (tp / denom);
            prev_recall = recall;
        }
        i = j;
    }
    ap
}

/// Unweighted metrics on the out-of-fold logits.
struct PointMetrics {
    roc_auc: f64,
    pr_auc: f64,
    log_loss: f64,
    brier: f64,
    ece: f64,
    calibration: Vec<(f64, f64, usize)>,
}

fn point_metrics(sorted_logit: &[f32], sorted_label: &[u8]) -> PointMetrics {
    let n = sorted_logit.len();
    let ones = vec![1.0f64; n];
    let roc_auc = weighted_auc(sorted_logit, sorted_label, &ones);
    let pr_auc = weighted_average_precision(sorted_logit, sorted_label, &ones);
    let (loss_sum, brier_sum) = (0..n)
        .into_par_iter()
        .map(|i| {
            let eta = f64::from(sorted_logit[i]);
            let y = f64::from(sorted_label[i]);
            let p = sigmoid(eta);
            (logit_loss(eta, y), (p - y) * (p - y))
        })
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    // Equal-count bins walk the already-sorted order, so a bin is a
    // contiguous slice and no second sort is needed.
    let mut calibration = Vec::with_capacity(N_CALIBRATION_BINS);
    let mut ece = 0.0f64;
    for b in 0..N_CALIBRATION_BINS {
        let lo = n * b / N_CALIBRATION_BINS;
        let hi = n * (b + 1) / N_CALIBRATION_BINS;
        if hi <= lo {
            continue;
        }
        let count = hi - lo;
        let mean_pred: f64 =
            sorted_logit[lo..hi].iter().map(|&e| sigmoid(f64::from(e))).sum::<f64>() / count as f64;
        let observed: f64 =
            sorted_label[lo..hi].iter().map(|&y| f64::from(y)).sum::<f64>() / count as f64;
        ece += (count as f64 / n as f64) * (mean_pred - observed).abs();
        calibration.push((mean_pred, observed, count));
    }
    PointMetrics {
        roc_auc,
        pr_auc,
        log_loss: loss_sum / n as f64,
        brier: brier_sum / n as f64,
        ece,
        calibration,
    }
}

/// Nearest-rank percentile of an unsorted sample.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let rank = (q * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

// ---------------------------------------------------------------------------
// data loading
// ---------------------------------------------------------------------------

struct Dataset {
    /// Row-major, `N_FEATURES` f32 per sample: ln(x + 1e-12) of each column.
    feats: Vec<f32>,
    labels: Vec<u8>,
    /// Dense index into `file_ids`.
    file_idx: Vec<u16>,
    file_ids: Vec<u32>,
    /// Count of exact zeros seen before the log transform, per feature.
    zero_counts: [u64; N_FEATURES],
}

/// Per-chunk parse output: flattened features, labels, raw file ids, and the
/// exact-zero tally used to interpret the log transform.
type ChunkParts = (Vec<f32>, Vec<u8>, Vec<u32>, [u64; N_FEATURES]);

fn parse_field(bytes: &[u8]) -> Result<f64> {
    std::str::from_utf8(bytes)?
        .trim()
        .parse::<f64>()
        .context("non-numeric CSV field")
}

fn load(path: &std::path::Path, max_rows: usize) -> Result<Dataset> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    // SAFETY: the benchmark CSV is a read-only artifact; the map is dropped
    // before any writer could touch it in this process.
    let map = unsafe { memmap2::Mmap::map(&file)? };
    let data: &[u8] = &map;
    let body_start = data
        .iter()
        .position(|&b| b == b'\n')
        .context("CSV has no header newline")?
        + 1;

    let chunk_target = 64usize << 20;
    let mut bounds = vec![body_start];
    let mut cursor = body_start;
    while cursor < data.len() {
        let mut next = (cursor + chunk_target).min(data.len());
        while next < data.len() && data[next] != b'\n' {
            next += 1;
        }
        if next < data.len() {
            next += 1;
        }
        bounds.push(next);
        cursor = next;
    }

    eprintln!("parsing {} chunks from {}", bounds.len() - 1, path.display());
    let parts: Vec<ChunkParts> = bounds
        .windows(2)
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|w| -> Result<_> {
            let slice = &data[w[0]..w[1]];
            let mut feats = Vec::new();
            let mut labels = Vec::new();
            let mut files = Vec::new();
            let mut zeros = [0u64; N_FEATURES];
            let mut fields: Vec<&[u8]> = Vec::with_capacity(12);
            for line in slice.split(|&b| b == b'\n') {
                if line.is_empty() {
                    continue;
                }
                fields.clear();
                fields.extend(line.split(|&b| b == b','));
                if fields.len() < 10 {
                    bail!("short CSV row with {} fields", fields.len());
                }
                files.push(parse_field(fields[COL_FILE_ID])? as u32);
                labels.push(parse_field(fields[COL_LABEL])? as u8);
                for (k, &c) in FEATURE_CSV_COL.iter().enumerate() {
                    let v = parse_field(fields[c])?;
                    if v == 0.0 {
                        zeros[k] += 1;
                    }
                    feats.push((v + LOG_EPS).ln() as f32);
                }
            }
            Ok((feats, labels, files, zeros))
        })
        .collect::<Result<Vec<_>>>()?;

    let total: usize = parts.iter().map(|p| p.1.len()).sum();
    let cap = if max_rows > 0 { max_rows.min(total) } else { total };
    let mut feats = Vec::with_capacity(cap * N_FEATURES);
    let mut labels = Vec::with_capacity(cap);
    let mut raw_files = Vec::with_capacity(cap);
    let mut zero_counts = [0u64; N_FEATURES];
    'outer: for (f, l, fi, z) in parts {
        for k in 0..N_FEATURES {
            zero_counts[k] += z[k];
        }
        for i in 0..l.len() {
            if labels.len() == cap {
                break 'outer;
            }
            feats.extend_from_slice(&f[i * N_FEATURES..(i + 1) * N_FEATURES]);
            labels.push(l[i]);
            raw_files.push(fi[i]);
        }
    }

    let mut file_ids: Vec<u32> = raw_files.clone();
    file_ids.sort_unstable();
    file_ids.dedup();
    if file_ids.len() > usize::from(u16::MAX) {
        bail!("more than 65535 file clusters");
    }
    let lookup: HashMap<u32, u16> = file_ids
        .iter()
        .enumerate()
        .map(|(i, &f)| (f, i as u16))
        .collect();
    let file_idx: Vec<u16> = raw_files.iter().map(|f| lookup[f]).collect();

    Ok(Dataset {
        feats,
        labels,
        file_idx,
        file_ids,
        zero_counts,
    })
}

// ---------------------------------------------------------------------------
// fold assignment
// ---------------------------------------------------------------------------

/// Deal the sorted unique file ids round-robin after a single ChaCha8 shuffle,
/// so a fold holds whole daily files and the split is reproducible from
/// `seed` alone.
fn assign_folds(file_ids: &[u32], seed: u64) -> Vec<Vec<u32>> {
    let mut shuffled: Vec<u32> = file_ids.to_vec();
    shuffled.sort_unstable();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // Fisher-Yates, descending, drawing from the ChaCha8 stream.
    for i in (1..shuffled.len()).rev() {
        let j = (rng.random::<u64>() % (i as u64 + 1)) as usize;
        shuffled.swap(i, j);
    }
    let mut folds = vec![Vec::new(); N_FOLDS];
    for (i, f) in shuffled.into_iter().enumerate() {
        folds[i % N_FOLDS].push(f);
    }
    for f in &mut folds {
        f.sort_unstable();
    }
    folds
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------

struct ModelSpec {
    name: &'static str,
    cols: Vec<usize>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    eprintln!(
        "staples incremental information: seed={} n_boot={} input={}",
        args.seed,
        args.n_boot,
        args.input.display()
    );

    let ds = load(&args.input, args.max_rows)?;
    let n = ds.labels.len();
    let n_pos: usize = ds.labels.iter().filter(|&&y| y == 1).count();
    let n_files = ds.file_ids.len();
    eprintln!("loaded {n} rows, {n_pos} positives, {n_files} file clusters");

    let folds = assign_folds(&ds.file_ids, args.seed);
    let mut fold_of_file = vec![usize::MAX; n_files];
    let index_of: HashMap<u32, usize> = ds
        .file_ids
        .iter()
        .enumerate()
        .map(|(i, &f)| (f, i))
        .collect();
    for (k, fold) in folds.iter().enumerate() {
        for f in fold {
            fold_of_file[index_of[f]] = k;
        }
    }
    let fold_of_row: Vec<u8> = ds
        .file_idx
        .iter()
        .map(|&fi| fold_of_file[fi as usize] as u8)
        .collect();

    let models = [
        ModelSpec {
            name: "B",
            cols: (0..N_BASELINE).collect(),
        },
        ModelSpec {
            name: "B+A",
            cols: (0..N_FEATURES).collect(),
        },
    ];

    let mut oof_logits: Vec<Vec<f32>> = vec![vec![0.0f32; n]; models.len()];
    let mut fold_reports: Vec<Value> = Vec::new();
    let mut degenerate_std = 0usize;

    for (k, fold_files) in folds.iter().enumerate() {
        let train: Vec<u32> = (0..n as u32)
            .filter(|&i| fold_of_row[i as usize] as usize != k)
            .collect();
        let test: Vec<u32> = (0..n as u32)
            .filter(|&i| fold_of_row[i as usize] as usize == k)
            .collect();
        eprintln!(
            "fold {k}: {} train rows, {} held-out rows",
            train.len(),
            test.len()
        );

        // Standardization on the training folds only, all seven columns at
        // once; model B simply ignores the assoc entry.
        let mut mean = [0.0f64; N_FEATURES];
        let mut m2 = [0.0f64; N_FEATURES];
        for &r in &train {
            let row = &ds.feats[r as usize * N_FEATURES..(r as usize + 1) * N_FEATURES];
            for (acc, &v) in mean.iter_mut().zip(row) {
                *acc += f64::from(v);
            }
        }
        let nt = train.len() as f64;
        for acc in &mut mean {
            *acc /= nt;
        }
        for &r in &train {
            let row = &ds.feats[r as usize * N_FEATURES..(r as usize + 1) * N_FEATURES];
            for ((acc, &v), &mu) in m2.iter_mut().zip(row).zip(mean.iter()) {
                let d = f64::from(v) - mu;
                *acc += d * d;
            }
        }
        let mut std = [1.0f64; N_FEATURES];
        for ((slot, &sq), _) in std.iter_mut().zip(m2.iter()).zip(0..N_FEATURES) {
            let s = (sq / nt).sqrt();
            if s > 0.0 && s.is_finite() {
                *slot = s;
            } else {
                degenerate_std += 1;
            }
        }

        let mut model_entries = Vec::new();
        for (mi, m) in models.iter().enumerate() {
            let sub_mean: Vec<f64> = m.cols.iter().map(|&c| mean[c]).collect();
            let sub_std: Vec<f64> = m.cols.iter().map(|&c| std[c]).collect();
            let fit = fit_irls(
                &ds.feats,
                N_FEATURES,
                &m.cols,
                &train,
                &ds.labels,
                &sub_mean,
                &sub_std,
                RIDGE,
            )?;
            eprintln!(
                "  model {}: {} Newton iterations, converged={}, deviance {:.6}",
                m.name,
                fit.iterations,
                fit.converged,
                fit.deviance.last().copied().unwrap_or(f64::NAN)
            );
            let beta = fit.beta.clone();
            let logits = &mut oof_logits[mi];
            for &r in &test {
                let base = r as usize * N_FEATURES;
                let mut eta = beta[0];
                for (j, &c) in m.cols.iter().enumerate() {
                    eta += beta[j + 1] * ((f64::from(ds.feats[base + c]) - sub_mean[j]) / sub_std[j]);
                }
                logits[r as usize] = eta as f32;
            }
            model_entries.push(json!({
                "model": m.name,
                "intercept": fit.beta[0],
                "standardized_coefficients": m.cols.iter().enumerate()
                    .map(|(j, &c)| json!({"feature": FEATURE_NAMES[c], "beta": fit.beta[j + 1]}))
                    .collect::<Vec<_>>(),
                "feature_mean": sub_mean,
                "feature_std": sub_std,
                "penalized_deviance_trajectory": fit.deviance,
                "iterations": fit.iterations,
                "converged": fit.converged,
            }));
        }
        fold_reports.push(json!({
            "fold": k,
            "file_ids": fold_files,
            "n_train_rows": train.len(),
            "n_test_rows": test.len(),
            "n_test_positives": test.iter().filter(|&&r| ds.labels[r as usize] == 1).count(),
            "models": model_entries,
        }));
    }

    // Sorted views: one descending sort per model feeds every metric and
    // every bootstrap resample, turning each resample into a sequential scan
    // instead of 23.7M random gathers.
    eprintln!("sorting out-of-fold predictions");
    let zero_counts = ds.zero_counts;
    let labels = ds.labels.clone();
    let file_idx = ds.file_idx.clone();
    drop(ds);

    let mut sorted: Vec<(Vec<f32>, Vec<u8>, Vec<u16>)> = Vec::new();
    for logits in &oof_logits {
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.par_sort_unstable_by(|&a, &b| {
            logits[b as usize]
                .partial_cmp(&logits[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let s_logit: Vec<f32> = order.iter().map(|&i| logits[i as usize]).collect();
        let s_label: Vec<u8> = order.iter().map(|&i| labels[i as usize]).collect();
        let s_file: Vec<u16> = order.iter().map(|&i| file_idx[i as usize]).collect();
        sorted.push((s_logit, s_label, s_file));
    }
    drop(oof_logits);

    let point: Vec<PointMetrics> = sorted.iter().map(|s| point_metrics(&s.0, &s.1)).collect();
    for (m, pm) in models.iter().zip(&point) {
        eprintln!(
            "  {}: ROC-AUC {:.6} PR-AUC {:.6} logloss {:.6} brier {:.6} ECE {:.6}",
            m.name, pm.roc_auc, pm.pr_auc, pm.log_loss, pm.brier, pm.ece
        );
    }

    // Paired file-level bootstrap.  Both models see the same draw counts in
    // a given resample, so the deltas are paired by construction.
    eprintln!("bootstrapping {} resamples over {n_files} files", args.n_boot);
    let draws: Vec<(f64, f64, f64, f64, f64, f64)> = (0..args.n_boot)
        .into_par_iter()
        .map(|b| {
            let key = args.seed ^ RESAMPLE_KEY_ODD.wrapping_mul(b as u64 + 1);
            let mut rng = ChaCha8Rng::seed_from_u64(key);
            let mut counts = vec![0.0f64; n_files];
            for _ in 0..n_files {
                let j = (rng.random::<u64>() % n_files as u64) as usize;
                counts[j] += 1.0;
            }
            let mut out = [0.0f64; 6];
            for (mi, s) in sorted.iter().enumerate() {
                let w: Vec<f64> = s.2.iter().map(|&f| counts[f as usize]).collect();
                out[mi * 3] = weighted_auc(&s.0, &s.1, &w);
                out[mi * 3 + 1] = weighted_average_precision(&s.0, &s.1, &w);
                let (ls, ws) = (0..s.0.len())
                    .map(|i| {
                        (
                            w[i] * logit_loss(f64::from(s.0[i]), f64::from(s.1[i])),
                            w[i],
                        )
                    })
                    .fold((0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
                out[mi * 3 + 2] = ls / ws;
            }
            (out[0], out[1], out[2], out[3], out[4], out[5])
        })
        .collect();

    let delta_summary = |vals: &mut Vec<f64>, higher_is_better: bool| -> Value {
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let adverse = vals
            .iter()
            .filter(|&&d| if higher_is_better { d <= 0.0 } else { d >= 0.0 })
            .count();
        let frac = if adverse == 0 {
            json!(format!("< 1/{}", vals.len() + 1))
        } else {
            json!(adverse as f64 / vals.len() as f64)
        };
        json!({
            "mean": vals.iter().sum::<f64>() / vals.len() as f64,
            "ci95_lo": percentile(vals, 0.025),
            "ci95_hi": percentile(vals, 0.975),
            "adverse_fraction": frac,
            "adverse_count": adverse,
        })
    };

    let mut d_auc: Vec<f64> = draws.iter().map(|d| d.3 - d.0).collect();
    let mut d_pr: Vec<f64> = draws.iter().map(|d| d.4 - d.1).collect();
    let mut d_ll: Vec<f64> = draws.iter().map(|d| d.5 - d.2).collect();
    let auc_json = delta_summary(&mut d_auc, true);
    let pr_json = delta_summary(&mut d_pr, true);
    let ll_json = delta_summary(&mut d_ll, false);

    let lo = auc_json["ci95_lo"].as_f64().unwrap_or(f64::NAN);
    let hi = auc_json["ci95_hi"].as_f64().unwrap_or(f64::NAN);
    let (verdict, decision) = if lo > 0.0 {
        (
            "associator_carries_incremental_information",
            "The 95 percent interval of the paired delta ROC-AUC excludes zero and is positive, so the CD associator carries information beyond the six-sample geometric baselines at matched receptive field.",
        )
    } else if hi < 0.0 {
        (
            "associator_degrades_baseline",
            "The 95 percent interval of the paired delta ROC-AUC excludes zero and is negative, so adding the CD associator lowers discrimination relative to the six-sample geometric baselines.",
        )
    } else {
        (
            "associator_redundant",
            "The 95 percent interval of the paired delta ROC-AUC contains zero, so the CD associator is redundant with the six-sample geometric baselines at matched receptive field.",
        )
    };
    eprintln!("decision: {verdict} (delta ROC-AUC 95% CI [{lo:.6}, {hi:.6}])");

    let prevalence = n_pos as f64 / n as f64;
    let null_ll = -(prevalence * prevalence.ln() + (1.0 - prevalence) * (1.0 - prevalence).ln());
    let metrics_json = |m: &ModelSpec, pm: &PointMetrics| {
        json!({
            "model": m.name,
            "features": m.cols.iter().map(|&c| FEATURE_NAMES[c]).collect::<Vec<_>>(),
            "roc_auc": pm.roc_auc,
            "pr_auc": pm.pr_auc,
            "log_loss": pm.log_loss,
            "brier": pm.brier,
            "expected_calibration_error": pm.ece,
            "calibration_bins": pm.calibration.iter().map(|&(p, o, c)| json!({
                "mean_predicted": p, "observed_rate": o, "count": c
            })).collect::<Vec<_>>(),
        })
    };

    let report = json!({
        "analysis": "staples_incremental_information",
        "input": args.input.display().to_string(),
        "seed": args.seed,
        "n_boot": args.n_boot,
        "n_samples": n,
        "n_positives": n_pos,
        "prevalence": prevalence,
        "n_files": n_files,
        "preregistered_design": {
            "feature_transform": "ln(x + 1e-12)",
            "baseline_features": FEATURE_NAMES[..N_BASELINE].to_vec(),
            "added_feature": FEATURE_NAMES[N_FEATURES - 1],
            "estimator": "logistic regression, Newton-Raphson (IRLS)",
            "ridge": RIDGE,
            "ridge_applies_to": "standardized slopes only; intercept unpenalized",
            "max_newton_iterations": MAX_NEWTON_ITERS,
            "deviance_tolerance": DEVIANCE_REL_TOL,
            "deviance_tolerance_form": "relative: |dev_prev - dev| / max(1, |dev|); the absolute 1e-8 of the protocol is below the f64 resolution of a deviance near 4e6 and would never fire",
            "standardization": "mean and std from training folds only, applied to the held-out fold; a zero training std is replaced by 1.0",
            "degenerate_std_replacements": degenerate_std,
            "cv": "5 folds grouped by file_id; sorted unique ids shuffled once by ChaCha8(seed) Fisher-Yates, then dealt round-robin",
            "tie_convention": "identical predicted logits form one group; the same grouping drives ROC-AUC (average-rank Mann-Whitney) and average precision (one distinct-threshold point per group)",
            "prediction_storage": "out-of-fold logits in f32; log loss from softplus(eta) - y*eta so a saturated probability cannot produce -inf",
            "bootstrap": "paired; each resample draws n_files file clusters with replacement and reweights samples by their file's draw count; both models share the draw counts",
            "bootstrap_key_derivation": "ChaCha8Rng::seed_from_u64(seed XOR (0x9E3779B97F4A7C15 * (b + 1))), b the zero-based resample index; n_files draws of (random_u64 mod n_files)",
            "percentile_convention": "nearest-rank on the sorted resample values, ceil(q * B) clamped to [1, B]",
            "decision_rule": "delta = (B+A) minus B; if the 95 percent percentile interval of delta ROC-AUC excludes zero and is positive the associator carries information beyond the baselines; if it contains zero the associator is redundant with them at matched receptive field",
            "memory_plan": "features held as 7 f32 per row (about 663 MB at 23.7M rows), released after the last fold; each model then keeps a descending-sorted logit/label/file triple of about 166 MB"
        },
        "zero_counts_before_log": FEATURE_NAMES.iter().zip(zero_counts.iter())
            .map(|(name, &c)| json!({"feature": name, "exact_zeros": c}))
            .collect::<Vec<_>>(),
        "null_model_reference": {
            "roc_auc": 0.5,
            "pr_auc": prevalence,
            "log_loss": null_ll,
            "brier": prevalence * (1.0 - prevalence),
            "note": "intercept-only model at the observed prevalence"
        },
        "folds": fold_reports,
        "models": models.iter().zip(&point).map(|(m, pm)| metrics_json(m, pm)).collect::<Vec<_>>(),
        "paired_bootstrap_delta": {
            "definition": "(B+A) minus B on the same resample",
            "roc_auc": auc_json,
            "pr_auc": pr_json,
            "log_loss": ll_json,
        },
        "verdict": verdict,
        "decision": decision,
    });

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut out = File::create(&args.output)
        .with_context(|| format!("create {}", args.output.display()))?;
    out.write_all(serde_json::to_string_pretty(&report)?.as_bytes())?;
    out.write_all(b"\n")?;
    eprintln!("wrote {}", args.output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newton_recovers_known_logistic_coefficients() {
        let n = 20_000usize;
        let truth = [0.9f64, -1.4, 0.6];
        let intercept = -0.3f64;
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let mut feats = Vec::with_capacity(n * 3);
        let mut labels = Vec::with_capacity(n);
        for _ in 0..n {
            let mut eta = intercept;
            for t in truth.iter() {
                // Box-Muller from two uniforms keeps the test free of rand_distr.
                let u1: f64 = rng.random::<f64>().max(1e-12);
                let u2: f64 = rng.random::<f64>();
                let x: f64 = (-2.0f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                feats.push(x as f32);
                eta += t * x;
            }
            let p = sigmoid(eta);
            labels.push(u8::from(rng.random::<f64>() < p));
        }
        let rows: Vec<u32> = (0..n as u32).collect();
        let fit = fit_irls(
            &feats,
            3,
            &[0, 1, 2],
            &rows,
            &labels,
            &[0.0, 0.0, 0.0],
            &[1.0, 1.0, 1.0],
            RIDGE,
        )
        .expect("fit converges");
        assert!((fit.beta[0] - intercept).abs() < 0.08, "{:?}", fit.beta);
        for (k, t) in truth.iter().enumerate() {
            assert!((fit.beta[k + 1] - t).abs() < 0.08, "{:?}", fit.beta);
        }
    }

    #[test]
    fn rank_auc_handles_ties() {
        // Descending order: 3.0 (pos), 2.0 (pos), 2.0 (neg), 1.0 (neg).
        // Pairs: (3,2neg)=1, (3,1)=1, (2pos,2neg)=0.5, (2pos,1)=1 -> 3.5/4.
        let scores = [3.0f32, 2.0, 2.0, 1.0];
        let labels = [1u8, 1, 0, 0];
        let w = [1.0f64; 4];
        assert!((weighted_auc(&scores, &labels, &w) - 0.875).abs() < 1e-12);
    }

    #[test]
    fn average_precision_hand_example() {
        // Descending: pos, neg, pos, neg.  Distinct thresholds give
        // precision 1/1 at recall 1/2 and 2/3 at recall 1.
        let scores = [4.0f32, 3.0, 2.0, 1.0];
        let labels = [1u8, 0, 1, 0];
        let w = [1.0f64; 4];
        let ap = weighted_average_precision(&scores, &labels, &w);
        let expect = 0.5 * 1.0 + 0.5 * (2.0 / 3.0);
        assert!((ap - expect).abs() < 1e-12, "{ap}");
    }

    #[test]
    fn unit_weights_reproduce_unweighted_auc() {
        let scores = [5.0f32, 4.0, 4.0, 3.0, 2.0, 1.0, 1.0];
        let labels = [1u8, 0, 1, 1, 0, 0, 1];
        let ones = vec![1.0f64; scores.len()];
        let twos = vec![2.0f64; scores.len()];
        let a = weighted_auc(&scores, &labels, &ones);
        // A uniform reweighting cancels in the normalization, so the
        // weighted statistic must equal the unweighted one.
        let b = weighted_auc(&scores, &labels, &twos);
        assert!((a - b).abs() < 1e-12);
        assert!(a > 0.0 && a < 1.0);
    }

    #[test]
    fn grouped_folds_are_deterministic_and_disjoint() {
        let ids: Vec<u32> = (0..813).collect();
        let a = assign_folds(&ids, 42);
        let b = assign_folds(&ids, 42);
        assert_eq!(a, b);
        let c = assign_folds(&ids, 43);
        assert_ne!(a, c);
        let mut all: Vec<u32> = a.iter().flatten().copied().collect();
        assert_eq!(all.len(), ids.len());
        all.sort_unstable();
        all.dedup();
        assert_eq!(all, ids);
        for f in &a {
            assert!(f.len() == 162 || f.len() == 163, "{}", f.len());
        }
    }
}
