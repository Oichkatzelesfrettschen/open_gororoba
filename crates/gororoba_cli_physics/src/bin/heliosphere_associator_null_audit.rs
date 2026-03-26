use anyhow::{Context, Result};
use chrono::Utc;
use clap::Parser;
use csv::ReaderBuilder;
use data_core::{
    compute_invariant_samples, HeliosphereFeatureRow, HeliosphereInvariantSample,
};
use rand::seq::SliceRandom;
use rand::thread_rng;
use serde::Serialize;
use std::{collections::BTreeMap, fs, path::PathBuf};

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
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut reader = ReaderBuilder::new()
        .from_path(&cli.cube_csv)
        .with_context(|| format!("open {}", cli.cube_csv.display()))?;

    let mut all_rows = Vec::new();
    for result in reader.deserialize::<HeliosphereFeatureRow>() {
        all_rows.push(result?);
    }

    // Group by mission + product
    let mut groups: BTreeMap<(String, String), Vec<HeliosphereFeatureRow>> = BTreeMap::new();
    for row in all_rows.iter() {
        groups
            .entry((row.mission.clone(), row.product.clone()))
            .or_default()
            .push(row.clone());
    }

    let mut results = Vec::new();

    // 1. Legacy Raw Embedding
    results.push(audit_embedding(
        "legacy-raw",
        &groups,
        |rows| rows.iter().map(|r| r.algebra_vector()).collect(),
        cli.null_iterations,
    ));

    // 2. Dynamic Bias-Free Embedding
    results.push(audit_embedding(
        "dynamic-bias-free",
        &groups,
        |rows| rows.iter().map(|r| r.algebra_vector_dynamic_bias_free()).collect(),
        cli.null_iterations,
    ));

    // 3. Invariant Embedding (Dimensionless residuals)
    let invariant_samples = compute_invariant_samples(&all_rows);
    let mut inv_groups: BTreeMap<(String, String), Vec<HeliosphereInvariantSample>> = BTreeMap::new();
    for sample in invariant_samples {
        inv_groups
            .entry((sample.mission.clone(), sample.product.clone()))
            .or_default()
            .push(sample);
    }

    results.push(audit_invariant_embedding(
        "invariant-residuals",
        &inv_groups,
        cli.null_iterations,
    ));

    // 4. Magnetic Takens Embedding (16D phase space)
    results.push(audit_embedding(
        "magnetic-takens",
        &groups,
        |rows| {
            let mut v16s = Vec::new();
            for window in rows.windows(4) {
                let mut v16 = [0.0; 16];
                let local_mean_b = (window[0].b_mag + window[1].b_mag + window[2].b_mag + window[3].b_mag) / 4.0;
                if local_mean_b > 0.0 {
                    for i in 0..4 {
                        v16[i * 4 + 0] = window[i].bx / local_mean_b;
                        v16[i * 4 + 1] = window[i].by / local_mean_b;
                        v16[i * 4 + 2] = window[i].bz / local_mean_b;
                        v16[i * 4 + 3] = (window[i].b_mag - local_mean_b) / local_mean_b;
                    }
                    v16s.push(v16);
                }
            }
            v16s
        },
        cli.null_iterations,
    ));

    let report = AuditReport {
        generated_at_utc: Utc::now().to_rfc3339(),
        cube_csv: cli.cube_csv.to_string_lossy().to_string(),
        results,
    };

    let toml = serde_json::to_string_pretty(&report)?;
    if let Some(out_path) = cli.out {
        fs::write(out_path, toml)?;
    } else {
        println!("{}", toml);
    }

    Ok(())
}

fn audit_embedding<F>(
    name: &str,
    groups: &BTreeMap<(String, String), Vec<HeliosphereFeatureRow>>,
    embed_fn: F,
    null_iters: usize,
) -> EmbeddingResult
where
    F: Fn(&[HeliosphereFeatureRow]) -> Vec<[f64; 16]>,
{
    let mut base_summaries = Vec::new();
    let mut group_vectors = BTreeMap::new();

    for ((mission, product), rows) in groups {
        let vectors = embed_fn(rows);
        let associators = cd_kernel::batch_sedenion_associator_norms(&vectors);

        if associators.is_empty() { continue; }

        let mean = associators.iter().sum::<f64>() / associators.len() as f64;
        let max = associators.iter().copied().fold(f64::NEG_INFINITY, f64::max);

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
            shuffled.shuffle(&mut thread_rng());
            let associators = cd_kernel::batch_sedenion_associator_norms(&shuffled);
            let m = if associators.is_empty() { 0.0 } else {
                associators.iter().sum::<f64>() / associators.len() as f64
            };
            iteration_means.push(m);
        }
        shuffle_nulls.push(GroupNullSummary {
            mission: mission.clone(),
            product: product.clone(),
            null_mean_of_means: iteration_means.iter().sum::<f64>() / null_iters as f64,
        });
    }

    // Channel Permutation Null
    let mut permute_nulls = Vec::new();
    for ((mission, product), vectors) in &group_vectors {
        let mut iteration_means = Vec::new();
        for _ in 0..null_iters {
            let mut shuffled_vectors = Vec::new();
            let mut indices: Vec<usize> = (0..16).collect();
            indices.shuffle(&mut thread_rng());
            
            for v in vectors {
                let mut v_perm = [0.0; 16];
                for (i, &idx) in indices.iter().enumerate() {
                    v_perm[i] = v[idx];
                }
                shuffled_vectors.push(v_perm);
            }

            let associators = cd_kernel::batch_sedenion_associator_norms(&shuffled_vectors);
            let m = if associators.is_empty() { 0.0 } else {
                associators.iter().sum::<f64>() / associators.len() as f64
            };
            iteration_means.push(m);
        }
        permute_nulls.push(GroupNullSummary {
            mission: mission.clone(),
            product: product.clone(),
            null_mean_of_means: iteration_means.iter().sum::<f64>() / null_iters as f64,
        });
    }

    let mut null_summaries = BTreeMap::new();
    null_summaries.insert("temporal-shuffle".to_string(), shuffle_nulls);
    null_summaries.insert("channel-permutation".to_string(), permute_nulls);

    EmbeddingResult {
        embedding_name: name.to_string(),
        base_summaries,
        null_summaries,
    }
}

fn audit_invariant_embedding(
    name: &str,
    groups: &BTreeMap<(String, String), Vec<HeliosphereInvariantSample>>,
    null_iters: usize,
) -> EmbeddingResult {
    let mut base_summaries = Vec::new();
    let mut group_vectors = BTreeMap::new();

    for ((mission, product), samples) in groups {
        let vectors: Vec<[f64; 16]> = samples.iter().map(|s| {
            let mut v = [0.0; 16];
            for i in 0..10 { v[i] = s.channels[i]; }
            v
        }).collect();

        let associators = cd_kernel::batch_sedenion_associator_norms(&vectors);

        if associators.is_empty() { continue; }

        let mean = associators.iter().sum::<f64>() / associators.len() as f64;
        let max = associators.iter().copied().fold(f64::NEG_INFINITY, f64::max);

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
            shuffled.shuffle(&mut thread_rng());
            let associators = cd_kernel::batch_sedenion_associator_norms(&shuffled);
            let m = if associators.is_empty() { 0.0 } else {
                associators.iter().sum::<f64>() / associators.len() as f64
            };
            iteration_means.push(m);
        }
        shuffle_nulls.push(GroupNullSummary {
            mission: mission.clone(),
            product: product.clone(),
            null_mean_of_means: iteration_means.iter().sum::<f64>() / null_iters as f64,
        });
    }

    // Channel Permutation Null
    let mut permute_nulls = Vec::new();
    for ((mission, product), vectors) in &group_vectors {
        let mut iteration_means = Vec::new();
        for _ in 0..null_iters {
            let mut shuffled_vectors = Vec::new();
            let mut indices: Vec<usize> = (0..16).collect();
            indices.shuffle(&mut thread_rng());
            
            for v in vectors {
                let mut v_perm = [0.0; 16];
                for (i, &idx) in indices.iter().enumerate() {
                    v_perm[i] = v[idx];
                }
                shuffled_vectors.push(v_perm);
            }

            let associators = cd_kernel::batch_sedenion_associator_norms(&shuffled_vectors);
            let m = if associators.is_empty() { 0.0 } else {
                associators.iter().sum::<f64>() / associators.len() as f64
            };
            iteration_means.push(m);
        }
        permute_nulls.push(GroupNullSummary {
            mission: mission.clone(),
            product: product.clone(),
            null_mean_of_means: iteration_means.iter().sum::<f64>() / null_iters as f64,
        });
    }

    let mut null_summaries = BTreeMap::new();
    null_summaries.insert("temporal-shuffle".to_string(), shuffle_nulls);
    null_summaries.insert("channel-permutation".to_string(), permute_nulls);

    EmbeddingResult {
        embedding_name: name.to_string(),
        base_summaries,
        null_summaries,
    }
}

