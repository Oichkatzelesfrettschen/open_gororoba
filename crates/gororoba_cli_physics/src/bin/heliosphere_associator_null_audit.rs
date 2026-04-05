use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use csv::ReaderBuilder;
use data_core::{
    HeliosphereFeatureRow, HeliosphereInvariantSample, compute_invariant_samples,
    magnetic_takens_embed,
};
use rand::{RngExt, seq::SliceRandom};
use rustfft::{FftPlanner, num_complex::Complex64};
use serde::Serialize;
use std::{collections::BTreeMap, f64::consts::PI, fs, path::PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-associator-null-audit",
    about = "Deconfounding audit: separation, nulls, and invariant embeddings"
)]
struct Cli {
    #[arg(long)]
    cube_csv: PathBuf,

    #[arg(long)]
    out: Option<PathBuf>,

    #[arg(long, default_value_t = 10)]
    null_iterations: usize,

    /// Exclude rows from these missions (repeatable, case-sensitive).
    #[arg(long = "exclude-mission")]
    exclude_missions: Vec<String>,

    /// Include ONLY rows from these missions (repeatable, case-sensitive).
    /// When set, --exclude-mission is ignored.
    #[arg(long = "include-mission")]
    include_missions: Vec<String>,

    /// Embedding dimension for magnetic-takens lane (power of 2, >= 16).
    #[arg(long, default_value_t = 16)]
    embedding_dim: usize,

    /// Block size for block-shuffle null (preserves short-range autocorrelation).
    #[arg(long, default_value_t = 10)]
    block_size: usize,

    /// Takens lag in time steps (1 = consecutive hourly).
    #[arg(long, default_value_t = 1)]
    takens_lag: usize,
}

#[derive(Debug, Clone, Serialize)]
struct AuditReport {
    generated_at_utc: String,
    cube_csv: String,
    results: Vec<EmbeddingResult>,
}

#[derive(Debug, Clone, Serialize)]
struct EmbeddingResult {
    embedding_name: String,
    base_summaries: Vec<GroupSummary>,
    null_summaries: BTreeMap<String, Vec<GroupNullSummary>>,
}

#[derive(Debug, Clone, Serialize)]
struct GroupSummary {
    mission: String,
    product: String,
    mean_associator_norm: f64,
    max_associator_norm: f64,
}

#[derive(Debug, Clone, Serialize)]
struct GroupNullSummary {
    mission: String,
    product: String,
    null_mean_of_means: f64,
    null_std_of_means: f64,
    null_median_of_means: f64,
    null_p05: f64,
    null_p95: f64,
}

fn null_stats(means: &[f64]) -> (f64, f64, f64, f64, f64) {
    if means.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let n = means.len() as f64;
    let avg = means.iter().sum::<f64>() / n;
    let var = means.iter().map(|&m| (m - avg).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();

    let mut sorted = means.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let p05_idx = ((sorted.len() as f64 * 0.05).floor() as usize).min(sorted.len() - 1);
    let p95_idx = ((sorted.len() as f64 * 0.95).floor() as usize).min(sorted.len() - 1);
    (avg, std, sorted[p05_idx], median, sorted[p95_idx])
}

fn build_null_summary(mission: &str, product: &str, iteration_means: &[f64]) -> GroupNullSummary {
    let (avg, std, p05, med, p95) = null_stats(iteration_means);
    GroupNullSummary {
        mission: mission.to_string(),
        product: product.to_string(),
        null_mean_of_means: avg,
        null_std_of_means: std,
        null_median_of_means: med,
        null_p05: p05,
        null_p95: p95,
    }
}

fn associator_mean(norms: &[f64]) -> f64 {
    if norms.is_empty() {
        0.0
    } else {
        norms.iter().sum::<f64>() / norms.len() as f64
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut reader = ReaderBuilder::new()
        .from_path(&cli.cube_csv)
        .with_context(|| format!("open {}", cli.cube_csv.display()))?;

    let include: std::collections::HashSet<&str> =
        cli.include_missions.iter().map(|s| s.as_str()).collect();
    let exclude: std::collections::HashSet<&str> =
        cli.exclude_missions.iter().map(|s| s.as_str()).collect();
    if !include.is_empty() {
        println!("Including ONLY missions: {:?}", cli.include_missions);
    } else if !exclude.is_empty() {
        println!("Excluding missions: {:?}", cli.exclude_missions);
    }

    let mut all_rows = Vec::new();
    for result in reader.deserialize::<HeliosphereFeatureRow>() {
        let r = result?;
        if !include.is_empty() && !include.contains(r.mission.as_str()) {
            continue;
        }
        if include.is_empty() && exclude.contains(r.mission.as_str()) {
            continue;
        }
        if r.bx.is_finite()
            && r.by.is_finite()
            && r.bz.is_finite()
            && r.b_mag.is_finite()
            && r.b_mag > 0.0
        {
            all_rows.push(r);
        }
    }

    let mut groups: BTreeMap<(String, String), Vec<HeliosphereFeatureRow>> = BTreeMap::new();
    for row in all_rows.iter() {
        groups
            .entry((row.mission.clone(), row.product.clone()))
            .or_default()
            .push(row.clone());
    }
    for rows in groups.values_mut() {
        rows.sort_by(|a, b| {
            a.year
                .cmp(&b.year)
                .then(a.doy.cmp(&b.doy))
                .then(a.hour.cmp(&b.hour))
        });
    }

    let mut results = Vec::new();

    // 1. Legacy Raw Embedding (always 16D)
    results.push(audit_generic(
        "legacy-raw",
        &groups,
        |rows| rows.iter().map(|r| r.algebra_vector().to_vec()).collect(),
        16,
        cli.null_iterations,
        cli.block_size,
    ));

    // 2. Dynamic Bias-Free Embedding (always 16D)
    results.push(audit_generic(
        "dynamic-bias-free",
        &groups,
        |rows| {
            rows.iter()
                .map(|r| r.algebra_vector_dynamic_bias_free().to_vec())
                .collect()
        },
        16,
        cli.null_iterations,
        cli.block_size,
    ));

    // 3. Invariant Embedding (10 channels padded to 16D)
    let invariant_samples = compute_invariant_samples(&all_rows);
    let mut inv_groups: BTreeMap<(String, String), Vec<HeliosphereInvariantSample>> =
        BTreeMap::new();
    for sample in invariant_samples {
        inv_groups
            .entry((sample.mission.clone(), sample.product.clone()))
            .or_default()
            .push(sample);
    }
    for rows in inv_groups.values_mut() {
        rows.sort_by(|a, b| {
            a.year
                .cmp(&b.year)
                .then(a.doy.cmp(&b.doy))
                .then(a.hour.cmp(&b.hour))
        });
    }
    results.push(audit_generic(
        "invariant-residuals",
        &inv_groups,
        |samples: &[HeliosphereInvariantSample]| {
            samples
                .iter()
                .map(|s| {
                    let mut v = vec![0.0; 16];
                    v[..10].copy_from_slice(&s.channels[..10]);
                    v
                })
                .collect()
        },
        16,
        cli.null_iterations,
        cli.block_size,
    ));

    // 4. Magnetic Takens Embedding (dimension-parameterized)
    let dim = cli.embedding_dim;
    let lag = cli.takens_lag;
    results.push(audit_generic(
        "magnetic-takens",
        &groups,
        |rows: &[HeliosphereFeatureRow]| magnetic_takens_embed(rows, dim, lag).0,
        dim,
        cli.null_iterations,
        cli.block_size,
    ));

    let report = AuditReport {
        generated_at_utc: Utc::now().to_rfc3339(),
        cube_csv: cli.cube_csv.to_string_lossy().to_string(),
        results,
    };

    let json = serde_json::to_string_pretty(&report)?;
    if let Some(out_path) = cli.out {
        fs::write(out_path, json)?;
    } else {
        println!("{json}");
    }

    Ok(())
}

/// Phase-randomize a sequence of D-dimensional vectors.
/// For each channel independently: FFT, randomize phases (preserving conjugate
/// symmetry for real output), IFFT. This preserves the power spectrum and
/// cross-time autocorrelation structure while destroying nonlinear phase coupling.
fn phase_randomize_surrogate(vectors: &[Vec<f64>], dim: usize) -> Vec<Vec<f64>> {
    let n = vectors.len();
    if n < 2 {
        return vectors.to_vec();
    }
    let mut planner = FftPlanner::<f64>::new();
    let fft_fwd = planner.plan_fft_forward(n);
    let fft_inv = planner.plan_fft_inverse(n);
    let mut rng = rand::rng();

    // Generate one set of random phases (shared across channels for cross-channel
    // coherence preservation). Phases are conjugate-symmetric: phi[k] = -phi[n-k].
    let half = n / 2;
    let random_phases: Vec<f64> = (0..=half).map(|_| rng.random::<f64>() * 2.0 * PI).collect();

    let mut result = vec![vec![0.0; dim]; n];

    for ch in 0..dim {
        let mut buf: Vec<Complex64> = vectors.iter().map(|v| Complex64::new(v[ch], 0.0)).collect();

        fft_fwd.process(&mut buf);

        // Randomize phases while preserving conjugate symmetry
        // DC component (k=0) and Nyquist (k=n/2 if n even) keep real values
        for k in 1..half {
            let phase = Complex64::from_polar(1.0, random_phases[k]);
            buf[k] *= phase;
            buf[n - k] *= phase.conj();
        }
        if n.is_multiple_of(2) {
            // Nyquist: keep magnitude, flip sign randomly
            let sign = if random_phases[half] > PI { -1.0 } else { 1.0 };
            buf[half] *= sign;
        }

        fft_inv.process(&mut buf);

        let scale = 1.0 / n as f64;
        for (i, c) in buf.iter().enumerate() {
            result[i][ch] = c.re * scale;
        }
    }

    result
}

/// Multivariate phase-randomized surrogate: INDEPENDENT random phases per channel.
/// This destroys cross-channel phase coupling while preserving per-channel power spectrum.
/// Stronger than the shared-phase version which preserves cross-channel coherence.
fn phase_randomize_surrogate_multivariate(vectors: &[Vec<f64>], dim: usize) -> Vec<Vec<f64>> {
    let n = vectors.len();
    if n < 2 {
        return vectors.to_vec();
    }
    let mut planner = FftPlanner::<f64>::new();
    let fft_fwd = planner.plan_fft_forward(n);
    let fft_inv = planner.plan_fft_inverse(n);
    let mut rng = rand::rng();
    let half = n / 2;

    let mut result = vec![vec![0.0; dim]; n];

    for ch in 0..dim {
        // Independent random phases PER CHANNEL (not shared)
        let channel_phases: Vec<f64> = (0..=half).map(|_| rng.random::<f64>() * 2.0 * PI).collect();

        let mut buf: Vec<Complex64> = vectors.iter().map(|v| Complex64::new(v[ch], 0.0)).collect();

        fft_fwd.process(&mut buf);

        for k in 1..half {
            let phase = Complex64::from_polar(1.0, channel_phases[k]);
            buf[k] *= phase;
            buf[n - k] *= phase.conj();
        }
        if n.is_multiple_of(2) {
            let sign = if channel_phases[half] > PI { -1.0 } else { 1.0 };
            buf[half] *= sign;
        }

        fft_inv.process(&mut buf);

        let scale = 1.0 / n as f64;
        for (i, c) in buf.iter().enumerate() {
            result[i][ch] = c.re * scale;
        }
    }

    result
}

/// Dimension-generic audit: embed, compute base associators, then run
/// five null families:
/// - temporal-shuffle (harsh: destroys all temporal order)
/// - channel-permutation (harsh: destroys algebraic channel assignment)
/// - block-shuffle (moderate: preserves short-range autocorrelation)
/// - phase-randomized (spectral: preserves power spectrum, destroys phase)
fn audit_generic<T, F>(
    name: &str,
    groups: &BTreeMap<(String, String), Vec<T>>,
    embed_fn: F,
    dim: usize,
    null_iters: usize,
    block_size: usize,
) -> EmbeddingResult
where
    T: Clone,
    F: Fn(&[T]) -> Vec<Vec<f64>>,
{
    let mut base_summaries = Vec::new();
    let mut group_vectors: BTreeMap<(String, String), Vec<Vec<f64>>> = BTreeMap::new();

    for ((mission, product), rows) in groups {
        let vectors = embed_fn(rows);
        let associators = cd_kernel::batch_sliding_associator_norms(&vectors, dim);

        if associators.is_empty() {
            continue;
        }

        let mean = associators.iter().sum::<f64>() / associators.len() as f64;
        let max = associators
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        base_summaries.push(GroupSummary {
            mission: mission.clone(),
            product: product.clone(),
            mean_associator_norm: mean,
            max_associator_norm: max,
        });
        group_vectors.insert((mission.clone(), product.clone()), vectors);
    }

    // Temporal Shuffle Null
    let mut shuffle_nulls = Vec::new();
    for ((mission, product), vectors) in &group_vectors {
        let mut iteration_means = Vec::new();
        for _ in 0..null_iters {
            let mut shuffled = vectors.clone();
            shuffled.shuffle(&mut rand::rng());
            let norms = cd_kernel::batch_sliding_associator_norms(&shuffled, dim);
            iteration_means.push(associator_mean(&norms));
        }
        shuffle_nulls.push(build_null_summary(mission, product, &iteration_means));
    }

    // Channel Permutation Null
    let mut permute_nulls = Vec::new();
    for ((mission, product), vectors) in &group_vectors {
        let mut iteration_means = Vec::new();
        for _ in 0..null_iters {
            let mut indices: Vec<usize> = (0..dim).collect();
            indices.shuffle(&mut rand::rng());

            let permuted: Vec<Vec<f64>> = vectors
                .iter()
                .map(|v| indices.iter().map(|&idx| v[idx]).collect())
                .collect();

            let norms = cd_kernel::batch_sliding_associator_norms(&permuted, dim);
            iteration_means.push(associator_mean(&norms));
        }
        permute_nulls.push(build_null_summary(mission, product, &iteration_means));
    }

    // Block-Shuffle Null (moderate: preserves short-range autocorrelation)
    let mut block_shuffle_nulls = Vec::new();
    for ((mission, product), vectors) in &group_vectors {
        let mut iteration_means = Vec::new();
        for _ in 0..null_iters {
            let mut blocks: Vec<&[Vec<f64>]> = vectors.chunks(block_size).collect();
            blocks.shuffle(&mut rand::rng());
            let shuffled: Vec<Vec<f64>> = blocks.into_iter().flatten().cloned().collect();
            let norms = cd_kernel::batch_sliding_associator_norms(&shuffled, dim);
            iteration_means.push(associator_mean(&norms));
        }
        block_shuffle_nulls.push(build_null_summary(mission, product, &iteration_means));
    }

    // Phase-Randomized Surrogate Null (spectral: preserves power spectrum, destroys phase)
    let mut phase_nulls = Vec::new();
    for ((mission, product), vectors) in &group_vectors {
        let mut iteration_means = Vec::new();
        let n = vectors.len();
        if n < 4 {
            phase_nulls.push(build_null_summary(mission, product, &[]));
            continue;
        }
        for _ in 0..null_iters {
            let surrogate = phase_randomize_surrogate(vectors, dim);
            let norms = cd_kernel::batch_sliding_associator_norms(&surrogate, dim);
            iteration_means.push(associator_mean(&norms));
        }
        phase_nulls.push(build_null_summary(mission, product, &iteration_means));
    }

    // Multivariate Phase-Randomized Null (strongest spectral: independent phases per channel)
    let mut mv_phase_nulls = Vec::new();
    for ((mission, product), vectors) in &group_vectors {
        let mut iteration_means = Vec::new();
        let n = vectors.len();
        if n < 4 {
            mv_phase_nulls.push(build_null_summary(mission, product, &[]));
            continue;
        }
        for _ in 0..null_iters {
            let surrogate = phase_randomize_surrogate_multivariate(vectors, dim);
            let norms = cd_kernel::batch_sliding_associator_norms(&surrogate, dim);
            iteration_means.push(associator_mean(&norms));
        }
        mv_phase_nulls.push(build_null_summary(mission, product, &iteration_means));
    }

    let mut null_summaries = BTreeMap::new();
    null_summaries.insert("temporal-shuffle".to_string(), shuffle_nulls);
    null_summaries.insert("channel-permutation".to_string(), permute_nulls);
    null_summaries.insert("block-shuffle".to_string(), block_shuffle_nulls);
    null_summaries.insert("phase-randomized".to_string(), phase_nulls);
    null_summaries.insert("multivariate-phase-randomized".to_string(), mv_phase_nulls);

    EmbeddingResult {
        embedding_name: name.to_string(),
        base_summaries,
        null_summaries,
    }
}
