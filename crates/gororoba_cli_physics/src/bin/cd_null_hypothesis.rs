//! CD Null Hypothesis Test: is the Cayley-Dickson algebraic structure essential,
//! or would ANY 32D nonlinear embedding produce comparable results?
//!
//! Three comparison methods on the SAME MAST data:
//! 1. CD associator: ||(a*b)*c - a*(b*c)|| with pathion multiplication
//! 2. Random trilinear: ||T(a,b,c)|| where T is a random 3-tensor
//! 3. Simple L2 triplet: ||a - 2b + c|| (linear baseline, no multiplication)
//!
//! If CD >> random >> L2, the algebra matters.
//! If CD ~ random >> L2, any nonlinearity suffices.
//! If CD ~ random ~ L2, we're measuring nothing special.

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;
use std::{fs, path::PathBuf, sync::Arc};
use zarrs::array::Array;
use zarrs_filesystem::FilesystemStore;

#[derive(Parser)]
#[command(name = "cd-null-hypothesis")]
struct Cli {
    /// Zarr directory for MAST shot.
    #[arg(long)]
    zarr_dir: PathBuf,

    /// Embedding dimension.
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,

    /// Output JSON.
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/cd_null_hypothesis.json"
    )]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct MethodResult {
    method: String,
    overall_mean: f64,
    overall_std: f64,
    early_mean: f64,
    late_mean: f64,
    temporal_contrast: f64,
}

#[derive(Debug, Serialize)]
struct NullOutput {
    n_samples: usize,
    embedding_dim: usize,
    results: Vec<MethodResult>,
    cd_vs_random_ratio: f64,
    cd_vs_l2_ratio: f64,
    interpretation: String,
}

fn read_zarr_f32(store: &Arc<FilesystemStore>, name: &str) -> Result<Vec<f32>> {
    let array = Array::open(store.clone(), &format!("/{name}"))
        .with_context(|| format!("Failed to open '{name}'"))?;
    let shape = array.shape().to_vec();
    let subset = zarrs::array::ArraySubset::new_with_shape(shape);
    array
        .retrieve_array_subset::<Vec<f32>>(&subset)
        .with_context(|| format!("Failed to read '{name}'"))
}

/// Random trilinear form: T(a,b,c) = sum_ijk R_ijk * a_i * b_j * c_k
/// Simplified: for each output dim, compute random weighted product of triplet
fn random_trilinear_norms(embedded: &[Vec<f64>], dim: usize, seed: u64) -> Vec<f64> {
    use rand::Rng;
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);

    // Generate random coefficients (dim x dim x dim is too large, use rank-1 approx)
    // T(a,b,c)_k = sum_i (r1_ki * a_i) * sum_j (r2_kj * b_j) * sum_l (r3_kl * c_l)
    let r1: Vec<Vec<f64>> = (0..dim)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    let r2: Vec<Vec<f64>> = (0..dim)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();
    let r3: Vec<Vec<f64>> = (0..dim)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect();

    let mut norms = Vec::new();
    for w in 0..embedded.len().saturating_sub(2) {
        let a = &embedded[w];
        let b = &embedded[w + 1];
        let c = &embedded[w + 2];

        let mut t_norm_sq = 0.0;
        for k in 0..dim {
            let mut ra = 0.0;
            let mut rb = 0.0;
            let mut rc = 0.0;
            for i in 0..dim {
                ra += r1[k][i] * a[i];
                rb += r2[k][i] * b[i];
                rc += r3[k][i] * c[i];
            }
            let t_k = ra * rb * rc;
            t_norm_sq += t_k * t_k;
        }
        norms.push(t_norm_sq.sqrt());
    }
    norms
}

/// Simple L2 triplet difference: ||a - 2b + c|| (discrete second derivative)
fn l2_triplet_norms(embedded: &[Vec<f64>], dim: usize) -> Vec<f64> {
    let mut norms = Vec::new();
    for w in 0..embedded.len().saturating_sub(2) {
        let a = &embedded[w];
        let b = &embedded[w + 1];
        let c = &embedded[w + 2];

        let mut norm_sq = 0.0;
        for i in 0..dim {
            let d = a[i] - 2.0 * b[i] + c[i];
            norm_sq += d * d;
        }
        norms.push(norm_sq.sqrt());
    }
    norms
}

fn analyze_norms(norms: &[f64], _n_windows: usize) -> (f64, f64, f64, f64, f64) {
    if norms.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let n = norms.len() as f64;
    let mean = norms.iter().sum::<f64>() / n;
    let var = norms.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    let std = var.sqrt();

    let third = norms.len() / 3;
    let early = norms[..third].iter().sum::<f64>() / third.max(1) as f64;
    let late = norms[norms.len() - third..].iter().sum::<f64>() / third.max(1) as f64;
    let contrast = if early > 1e-15 {
        (early - late).abs() / early
    } else {
        0.0
    };

    (mean, std, early, late, contrast)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("=== CD Null Hypothesis Test ===");

    let store = Arc::new(
        FilesystemStore::new(&cli.zarr_dir)
            .map_err(|e| anyhow::anyhow!("Store: {:?}", e))?,
    );

    let sig_n2_f32 = read_zarr_f32(&store, "n=2_signal")?;
    let sig_nodd_f32 = read_zarr_f32(&store, "n=odd_signal")?;
    let _time_f32 = read_zarr_f32(&store, "time")?;
    let n = sig_n2_f32.len();
    println!("  Loaded {} samples", n);

    let sig_n2: Vec<f64> = sig_n2_f32.iter().map(|&x| x as f64).collect();
    let sig_nodd: Vec<f64> = sig_nodd_f32.iter().map(|&x| x as f64).collect();

    // Build embeddings (same for all methods)
    let stride = 50;
    let channels = 2;
    let steps = cli.embedding_dim / channels;
    let n_embed = if n > steps * stride { (n - steps * stride) / stride } else { 0 };

    let mut embedded: Vec<Vec<f64>> = Vec::with_capacity(n_embed);
    for w in 0..n_embed {
        let mut v = Vec::with_capacity(cli.embedding_dim);
        for s in 0..steps {
            let idx = w * stride + s * stride;
            if idx < n {
                v.push(sig_n2[idx]);
                v.push(sig_nodd[idx]);
            }
        }
        while v.len() < cli.embedding_dim {
            v.push(0.0);
        }
        v.truncate(cli.embedding_dim);
        // Direction normalize
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        embedded.push(v);
    }

    println!("  Built {} embedding vectors", embedded.len());

    // Method 1: CD associator (the real deal)
    let cd_norms =
        cd_kernel::batch_sliding_associator_norms_parallel(&embedded, cli.embedding_dim);
    let (cd_mean, cd_std, cd_early, cd_late, cd_contrast) =
        analyze_norms(&cd_norms, embedded.len());

    // Method 2: Random trilinear form
    let rand_norms = random_trilinear_norms(&embedded, cli.embedding_dim, 42);
    let (rand_mean, rand_std, rand_early, rand_late, rand_contrast) =
        analyze_norms(&rand_norms, embedded.len());

    // Method 3: L2 triplet (linear baseline)
    let l2_norms = l2_triplet_norms(&embedded, cli.embedding_dim);
    let (l2_mean, l2_std, l2_early, l2_late, l2_contrast) =
        analyze_norms(&l2_norms, embedded.len());

    println!("\n  Method         | Mean     | Std      | Early    | Late     | Contrast");
    println!("  ──────────────-+──────────+──────────+──────────+──────────+─────────");
    println!(
        "  CD associator  | {:.6} | {:.6} | {:.6} | {:.6} | {:.4}",
        cd_mean, cd_std, cd_early, cd_late, cd_contrast
    );
    println!(
        "  Random trilin  | {:.6} | {:.6} | {:.6} | {:.6} | {:.4}",
        rand_mean, rand_std, rand_early, rand_late, rand_contrast
    );
    println!(
        "  L2 triplet     | {:.6} | {:.6} | {:.6} | {:.6} | {:.4}",
        l2_mean, l2_std, l2_early, l2_late, l2_contrast
    );

    let cd_vs_rand = if rand_contrast > 1e-15 {
        cd_contrast / rand_contrast
    } else {
        f64::INFINITY
    };
    let cd_vs_l2 = if l2_contrast > 1e-15 {
        cd_contrast / l2_contrast
    } else {
        f64::INFINITY
    };

    let interp = format!(
        "CD contrast={:.4}, Random contrast={:.4}, L2 contrast={:.4}. CD/Random={:.2}x, CD/L2={:.2}x. {}",
        cd_contrast, rand_contrast, l2_contrast, cd_vs_rand, cd_vs_l2,
        if cd_vs_rand > 2.0 { "CD ALGEBRAIC STRUCTURE MATTERS: significantly higher temporal contrast than random trilinear." }
        else if cd_vs_rand > 1.2 { "CD slightly better than random: algebra provides modest advantage." }
        else { "CD ~ random: algebraic structure may be cosmetic. FURTHER INVESTIGATION NEEDED." }
    );
    println!("\n  {}", interp);

    let results = vec![
        MethodResult {
            method: "cd_associator".into(),
            overall_mean: cd_mean,
            overall_std: cd_std,
            early_mean: cd_early,
            late_mean: cd_late,
            temporal_contrast: cd_contrast,
        },
        MethodResult {
            method: "random_trilinear".into(),
            overall_mean: rand_mean,
            overall_std: rand_std,
            early_mean: rand_early,
            late_mean: rand_late,
            temporal_contrast: rand_contrast,
        },
        MethodResult {
            method: "l2_triplet".into(),
            overall_mean: l2_mean,
            overall_std: l2_std,
            early_mean: l2_early,
            late_mean: l2_late,
            temporal_contrast: l2_contrast,
        },
    ];

    let output = NullOutput {
        n_samples: n,
        embedding_dim: cli.embedding_dim,
        results,
        cd_vs_random_ratio: cd_vs_rand,
        cd_vs_l2_ratio: cd_vs_l2,
        interpretation: interp,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&output)?)?;
    println!("  Wrote {}", cli.out_json.display());

    Ok(())
}
