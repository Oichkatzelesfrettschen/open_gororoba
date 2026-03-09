//! CD Ultrametric Scaling Law Experiment
//!
//! Tests whether ultrametric fraction scales systematically with
//! Cayley-Dickson doubling level. For each dimension D = 2^k (k=1..8),
//! generates N random unit elements and measures:
//!
//! 1. Euclidean ultrametric fraction (global, via distance matrix)
//! 2. Local ultrametric index (N-D neighborhood test)
//! 3. Baire ultrametric fraction (trinary quantization)
//!
//! The hypothesis is that higher-dimensional CD algebras exhibit
//! increasing ultrametric structure due to the recursive doubling
//! creating nested hierarchical relationships between basis elements.

use clap::Parser;
use gororoba_algebra::construction::cayley_dickson::cd_multiply;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use stats_core::ultrametric::{
    baire_codebook::codebook_baire_ultrametric_test_nd,
    local::{euclidean_distance_matrix_nd, local_ultrametricity_test_nd},
    ultrametric_fraction_from_matrix,
};

#[derive(Parser)]
#[command(
    name = "cd-ultrametric-scaling",
    about = "Test ultrametric scaling across Cayley-Dickson doubling levels"
)]
struct Args {
    /// Number of random elements per dimension.
    #[arg(long, default_value_t = 100)]
    n_points: usize,

    /// Number of triples for fraction tests.
    #[arg(long, default_value_t = 10_000)]
    n_triples: usize,

    /// Number of permutations for null distributions.
    #[arg(long, default_value_t = 100)]
    n_perms: usize,

    /// Maximum CD dimension (power of 2).
    #[arg(long, default_value_t = 256)]
    max_dim: usize,

    /// Random seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

/// Generate N random unit elements in the D-dimensional CD algebra.
///
/// Each element is normalized to unit norm. Uses the multiplication
/// structure by generating pairs of random elements and multiplying them,
/// producing algebraically-correlated vectors.
fn generate_cd_elements(dim: usize, n: usize, rng: &mut ChaCha8Rng) -> Vec<Vec<f64>> {
    let mut elements = Vec::with_capacity(n);

    for _ in 0..n {
        // Generate two random vectors
        let a: Vec<f64> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let b: Vec<f64> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();

        // Multiply using CD algebra structure
        let product = cd_multiply(&a, &b);

        // Normalize to unit sphere
        let norm: f64 = product.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            let unit: Vec<f64> = product.iter().map(|x| x / norm).collect();
            elements.push(unit);
        }
    }

    elements
}

/// Quantize a floating-point vector to trinary {-1, 0, 1} using thresholds.
///
/// Components > threshold -> 1, components < -threshold -> -1, else 0.
fn quantize_trinary(v: &[f64], threshold: f64) -> Vec<i8> {
    v.iter()
        .map(|&x| {
            if x > threshold {
                1
            } else if x < -threshold {
                -1
            } else {
                0
            }
        })
        .collect()
}

fn main() {
    let args = Args::parse();
    let mut rng = ChaCha8Rng::seed_from_u64(args.seed);

    println!("=== Cayley-Dickson Ultrametric Scaling Law ===");
    println!(
        "N_points={}, N_triples={}, N_perms={}, seed={}",
        args.n_points, args.n_triples, args.n_perms, args.seed
    );
    println!();
    println!(
        "{:<8} {:<12} {:<12} {:<12} {:<12} {:<12}",
        "Dim", "Euclid_frac", "Euclid_null", "Local_mean", "Baire_frac", "Baire_null"
    );
    println!("{}", "-".repeat(72));

    let mut results: Vec<(usize, f64, f64, f64, f64, f64)> = Vec::new();

    let mut dim = 2;
    while dim <= args.max_dim {
        let elements = generate_cd_elements(dim, args.n_points, &mut rng);
        let actual_n = elements.len();

        if actual_n < 3 {
            eprintln!("  dim={}: fewer than 3 valid elements, skipping", dim);
            dim *= 2;
            continue;
        }

        // 1. Global Euclidean ultrametric fraction
        let dist_matrix = euclidean_distance_matrix_nd(&elements);
        let euclid_frac =
            ultrametric_fraction_from_matrix(&dist_matrix, actual_n, args.n_triples, args.seed);

        // Null: shuffle elements (permutation of rows preserves marginal distributions
        // but breaks algebraic correlations)
        let mut null_fracs = Vec::new();
        let mut shuffled = elements.clone();
        for _ in 0..args.n_perms {
            // Column-independent shuffle: shuffle each component independently
            for col in 0..dim {
                let mut vals: Vec<f64> = shuffled.iter().map(|v| v[col]).collect();
                vals.shuffle(&mut rng);
                for (i, &val) in vals.iter().enumerate() {
                    shuffled[i][col] = val;
                }
            }
            let null_dm = euclidean_distance_matrix_nd(&shuffled);
            let nf =
                ultrametric_fraction_from_matrix(&null_dm, actual_n, args.n_triples, args.seed);
            null_fracs.push(nf);
        }
        let euclid_null = null_fracs.iter().sum::<f64>() / args.n_perms as f64;

        // 2. Local ultrametric index
        // Estimate epsilon: use 75th percentile of distances to ensure neighborhoods
        // are populated even in high dimensions where concentration of measure
        // pushes all distances toward the same value.
        let epsilon = {
            let mut sorted_dists = dist_matrix.clone();
            sorted_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted_dists[sorted_dists.len() * 3 / 4] * 0.8
        };

        let local_result = local_ultrametricity_test_nd(
            &elements,
            epsilon,
            args.n_triples.min(500),
            args.n_perms.min(20),
            args.seed,
        );
        let local_mean = local_result.mean_local_index;

        // 3. Baire ultrametric fraction (trinary quantization)
        let threshold = 0.3; // Components > 0.3 -> 1, < -0.3 -> -1, else 0
        let trinary: Vec<Vec<i8>> = elements
            .iter()
            .map(|v| quantize_trinary(v, threshold))
            .collect();

        let baire_result = codebook_baire_ultrametric_test_nd(
            &trinary,
            args.n_triples,
            args.n_perms.min(50),
            args.seed,
        );

        println!(
            "{:<8} {:<12.6} {:<12.6} {:<12.6} {:<12.6} {:<12.6}",
            dim,
            euclid_frac,
            euclid_null,
            local_mean,
            baire_result.ultrametric_fraction,
            baire_result.null_fraction_mean,
        );

        results.push((
            dim,
            euclid_frac,
            euclid_null,
            local_mean,
            baire_result.ultrametric_fraction,
            baire_result.null_fraction_mean,
        ));

        dim *= 2;
    }

    println!();
    println!("=== Scaling Analysis ===");

    // Check if Euclidean ultrametric fraction increases with dimension
    let euclid_excesses: Vec<f64> = results.iter().map(|r| r.1 - r.2).collect();
    let dims_f64: Vec<f64> = results.iter().map(|r| (r.0 as f64).ln()).collect();

    if euclid_excesses.len() >= 3 {
        // Simple linear regression: excess vs log(dim)
        let n = euclid_excesses.len() as f64;
        let mean_x = dims_f64.iter().sum::<f64>() / n;
        let mean_y = euclid_excesses.iter().sum::<f64>() / n;
        let cov: f64 = dims_f64
            .iter()
            .zip(euclid_excesses.iter())
            .map(|(&x, &y)| (x - mean_x) * (y - mean_y))
            .sum::<f64>();
        let var_x: f64 = dims_f64.iter().map(|&x| (x - mean_x).powi(2)).sum();

        if var_x > 1e-10 {
            let slope = cov / var_x;
            let intercept = mean_y - slope * mean_x;

            // R^2
            let ss_res: f64 = dims_f64
                .iter()
                .zip(euclid_excesses.iter())
                .map(|(&x, &y)| (y - (slope * x + intercept)).powi(2))
                .sum();
            let ss_tot: f64 = euclid_excesses.iter().map(|&y| (y - mean_y).powi(2)).sum();
            let r_squared = if ss_tot > 1e-15 {
                1.0 - ss_res / ss_tot
            } else {
                0.0
            };

            println!(
                "Euclidean excess vs log(dim): slope={:.6}, intercept={:.6}, R^2={:.4}",
                slope, intercept, r_squared
            );

            if slope > 0.0 && r_squared > 0.5 {
                println!("PASS: Ultrametric fraction increases with CD doubling level (C-780)");
            } else if slope > 0.0 {
                println!(
                    "INFO: Positive trend (slope={:.4}) but weak fit (R^2={:.4})",
                    slope, r_squared
                );
            } else {
                println!("INFO: No increasing trend (slope={:.4})", slope);
            }
        }
    }

    // Write CSV output
    let csv_path = "data/csv/cd_ultrametric_scaling.csv";
    if let Ok(mut f) = std::fs::File::create(csv_path) {
        use std::io::Write;
        let _ = writeln!(
            f,
            "dim,euclid_frac,euclid_null,euclid_excess,local_mean,baire_frac,baire_null"
        );
        for &(dim, ef, en, lm, bf, bn) in &results {
            let _ = writeln!(
                f,
                "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                dim,
                ef,
                en,
                ef - en,
                lm,
                bf,
                bn
            );
        }
        println!();
        println!("Results written to {}", csv_path);
    }
}
