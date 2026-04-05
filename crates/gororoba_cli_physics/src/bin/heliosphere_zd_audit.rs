//! Zero-divisor proximity audit for weak-field CD embeddings.
//!
//! In 32D Cayley-Dickson algebras, zero divisors exist: elements a where
//! there exists b != 0 such that a*b = 0. Near the ZD subspace, |a*b|
//! can become anomalously small, potentially amplifying the associator.
//!
//! This binary checks whether Rosetta cavity embeddings are systematically
//! closer to zero-divisor subspace than outside embeddings.

use anyhow::Result;
use cd_kernel::cayley_dickson::simd::pathion_multiply_flat;
use data_core::catalogs::rosetta::parse_rosetta_amda_mag_csv_minutes;
use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use std::{fs, path::PathBuf};

#[derive(Debug, Serialize)]
struct ZdAuditResult {
    n_cavity_vectors: usize,
    n_outside_vectors: usize,
    n_random_probes: usize,
    cavity_mean_min_product: f64,
    outside_mean_min_product: f64,
    cavity_mean_product: f64,
    outside_mean_product: f64,
    ratio_cavity_to_outside: f64,
    interpretation: String,
}

fn main() -> Result<()> {
    println!("=== Zero-Divisor Proximity Audit ===");

    // Load Rosetta data
    let data_dir = PathBuf::from("data/external/rosetta");
    let mut all_minutes = Vec::new();
    for month in 7..=9 {
        let fname = format!("rosetta_rpcmag_2015_{:02}.csv", month);
        let path = data_dir.join(&fname);
        if path.exists() {
            let content = fs::read_to_string(&path)?;
            let records = parse_rosetta_amda_mag_csv_minutes(&content);
            all_minutes.extend(records);
        }
    }

    println!("  Loaded {} minutes", all_minutes.len());

    // Build 32D Takens embeddings
    let channels = 4usize;
    let steps = 8; // 32D / 4
    let cavity_threshold = 2.0;

    let mut cavity_vecs: Vec<[f64; 32]> = Vec::new();
    let mut outside_vecs: Vec<[f64; 32]> = Vec::new();

    for w in 0..=all_minutes.len().saturating_sub(steps) {
        let sum_b: f64 = (0..steps).map(|s| all_minutes[w + s].b_magnitude).sum();
        let mean_b = sum_b / steps as f64;
        if mean_b <= 0.01 || !mean_b.is_finite() {
            continue;
        }

        let mut v = [0.0f64; 32];
        for s in 0..steps {
            let rec = &all_minutes[w + s];
            v[s * channels] = rec.bx_cseq / mean_b;
            v[s * channels + 1] = rec.by_cseq / mean_b;
            v[s * channels + 2] = rec.bz_cseq / mean_b;
            v[s * channels + 3] = (rec.b_magnitude - mean_b) / mean_b;
        }

        // Classify by center point's |B|
        let center = &all_minutes[w + steps / 2];
        if center.b_magnitude < cavity_threshold {
            cavity_vecs.push(v);
        } else {
            outside_vecs.push(v);
        }
    }

    println!(
        "  Cavity vectors: {}, Outside: {}",
        cavity_vecs.len(),
        outside_vecs.len()
    );

    // For each embedding vector, probe ZD proximity:
    // Compute min(|a*b|) over N random unit vectors b.
    // If ZD proximity is high, min(|a*b|) will be small.
    let n_probes = 100;
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    // Generate random unit vectors for probing
    let random_bs: Vec<[f64; 32]> = (0..n_probes)
        .map(|_| {
            let mut b = [0.0f64; 32];
            for x in &mut b {
                *x = rng.random::<f64>() - 0.5;
            }
            let norm = b.iter().map(|x| x * x).sum::<f64>().sqrt();
            for x in &mut b {
                *x /= norm;
            }
            b
        })
        .collect();

    println!(
        "  Computing ZD proximity ({} probes per vector)...",
        n_probes
    );

    // Subsample for speed
    let max_sample = 500;
    let cavity_sample: Vec<&[f64; 32]> = if cavity_vecs.len() > max_sample {
        let step = cavity_vecs.len() / max_sample;
        cavity_vecs.iter().step_by(step).take(max_sample).collect()
    } else {
        cavity_vecs.iter().collect()
    };
    let outside_sample: Vec<&[f64; 32]> = {
        let step = outside_vecs.len().max(1) / max_sample.min(outside_vecs.len()).max(1);
        outside_vecs
            .iter()
            .step_by(step.max(1))
            .take(max_sample)
            .collect()
    };

    let zd_stats = |vecs: &[&[f64; 32]]| -> (f64, f64) {
        let mut min_products = Vec::new();
        let mut all_products = Vec::new();

        for a in vecs {
            // Normalize a to unit
            let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm_a < 1e-15 {
                continue;
            }
            let mut a_unit = [0.0f64; 32];
            for i in 0..32 {
                a_unit[i] = a[i] / norm_a;
            }

            let mut min_prod = f64::MAX;
            for b in &random_bs {
                let product = pathion_multiply_flat(&a_unit, b);
                let prod_norm = product.iter().map(|x| x * x).sum::<f64>().sqrt();
                if prod_norm < min_prod {
                    min_prod = prod_norm;
                }
                all_products.push(prod_norm);
            }
            min_products.push(min_prod);
        }

        let mean_min = if min_products.is_empty() {
            0.0
        } else {
            min_products.iter().sum::<f64>() / min_products.len() as f64
        };
        let mean_all = if all_products.is_empty() {
            0.0
        } else {
            all_products.iter().sum::<f64>() / all_products.len() as f64
        };
        (mean_min, mean_all)
    };

    println!("  Probing cavity vectors ({})...", cavity_sample.len());
    let (cav_min, cav_mean) = zd_stats(&cavity_sample);

    println!("  Probing outside vectors ({})...", outside_sample.len());
    let (out_min, out_mean) = zd_stats(&outside_sample);

    let ratio = cav_min / out_min.max(1e-15);

    println!("\n=== Results ===");
    println!(
        "  Cavity:  min_product={:.6}, mean_product={:.6}",
        cav_min, cav_mean
    );
    println!(
        "  Outside: min_product={:.6}, mean_product={:.6}",
        out_min, out_mean
    );
    println!("  Ratio (cavity/outside): {:.4}", ratio);

    let interpretation = if ratio < 0.5 {
        "Cavity vectors are CLOSER to zero-divisor subspace (min product significantly smaller). The cavity amplification is partly an algebraic zero-divisor proximity effect."
    } else if ratio < 0.9 {
        "Cavity vectors are slightly closer to ZD subspace but the effect is moderate. The amplification is primarily normalization, not algebraic."
    } else {
        "Cavity and outside vectors have similar ZD proximity. The cavity amplification is NOT a zero-divisor artifact -- it is purely normalization-driven."
    };

    println!("  -> {}", interpretation);

    let result = ZdAuditResult {
        n_cavity_vectors: cavity_sample.len(),
        n_outside_vectors: outside_sample.len(),
        n_random_probes: n_probes,
        cavity_mean_min_product: cav_min,
        outside_mean_min_product: out_min,
        cavity_mean_product: cav_mean,
        outside_mean_product: out_mean,
        ratio_cavity_to_outside: ratio,
        interpretation: interpretation.to_string(),
    };

    let out_path = "data/output/heliosphere/ablations/decomposition_q_zd_audit.json";
    fs::write(out_path, serde_json::to_string_pretty(&result)?)?;
    println!("\nWrote {}", out_path);

    Ok(())
}
