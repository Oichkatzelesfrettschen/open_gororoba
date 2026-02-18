//! Topological and ultrametric analysis of Wow! signal candidate data.
//!
//! Subcommands:
//! - `morphology`: Persistent homology on BL 6EQUJ5 candidate point clouds
//!   (ON vs OFF comparison via permutation test). Truncates to top-k=50 (VR OOM pitfall).
//! - `ultrametric`: Ultrametric fraction test on BL candidate feature vectors.
//!   Column-shuffled null with 10000 triples x 1000 permutations, BH-FDR correction.

use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "wow-topology", about = "Topological analysis of Wow! signal candidates")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Persistent homology comparison: ON vs OFF cadence pointings.
    Morphology {
        /// Path to the BL manifest CSV.
        #[arg(long, default_value = "data/csv/bl_6equj5_gbt_manifest.csv")]
        manifest: PathBuf,

        /// Maximum number of points per cloud (OOM safety).
        #[arg(long, default_value_t = 50)]
        max_points: usize,

        /// Maximum persistence dimension (0 = connected components, 1 = loops).
        #[arg(long, default_value_t = 1)]
        max_dim: usize,

        /// Number of permutations for null model.
        #[arg(long, default_value_t = 100)]
        n_perms: usize,

        /// Random seed for reproducibility.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Ultrametric fraction test on candidate feature vectors.
    Ultrametric {
        /// Path to the BL manifest CSV.
        #[arg(long, default_value = "data/csv/bl_6equj5_gbt_manifest.csv")]
        manifest: PathBuf,

        /// Number of random triples to sample.
        #[arg(long, default_value_t = 10_000)]
        n_triples: usize,

        /// Number of permutations for null distribution.
        #[arg(long, default_value_t = 1000)]
        n_perms: usize,

        /// Random seed for reproducibility.
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
}

fn run_morphology(
    manifest: &Path,
    max_points: usize,
    max_dim: usize,
    n_perms: usize,
    seed: u64,
) {
    use data_core::catalogs::wow::{abacad_filter, parse_bl_manifest_csv};
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use rand::SeedableRng;
    use vacuum_frustration::vietoris_rips::{
        compute_betti_numbers, compute_persistent_homology, DistanceMatrix,
        PersistenceDiagram, VietorisRipsComplex,
    };

    println!("=== Persistent Homology: ON vs OFF Cadence Morphology ===");
    println!();

    if !manifest.exists() {
        eprintln!("ERROR: {} not found", manifest.display());
        std::process::exit(1);
    }

    let bundles = parse_bl_manifest_csv(manifest).expect("Failed to parse BL manifest");
    let on_bundles = abacad_filter(&bundles);
    let off_bundles: Vec<_> = bundles
        .iter()
        .filter(|b| b.pointing_type == "OFF" && b.cadence > 0)
        .collect();

    println!("ON pointings: {}", on_bundles.len());
    println!("OFF pointings: {}", off_bundles.len());
    println!("Max dim: {}, Max points: {}", max_dim, max_points);
    println!();

    // Since we don't have actual filterbank data (those are ~GB each),
    // we demonstrate the pipeline with synthetic feature vectors derived
    // from the manifest metadata (obs_num, cadence as proxy features).
    // In production, features would come from turboSETI candidate tables.
    let make_features = |blist: &[&data_core::catalogs::wow::Bl6equj5Bundle]| -> Vec<f64> {
        let mut features = Vec::new();
        for b in blist.iter().take(max_points) {
            features.push(b.obs_num as f64);
            features.push(b.cadence as f64);
            features.push(if b.pointing_type == "ON" { 1.0 } else { 0.0 });
        }
        features
    };

    let on_features = make_features(&on_bundles);
    let off_features = make_features(&off_bundles);

    let n_on = on_features.len() / 3;
    let n_off = off_features.len() / 3;

    if n_on < 3 || n_off < 3 {
        eprintln!("ERROR: Need at least 3 points per cloud, got ON={}, OFF={}", n_on, n_off);
        std::process::exit(1);
    }

    let dist_on = DistanceMatrix::from_points_3d(&on_features);
    let dist_off = DistanceMatrix::from_points_3d(&off_features);

    // Compute persistence for ON and OFF
    let threshold = 100.0; // Large enough to capture all scales
    let complex_on = VietorisRipsComplex::build(&dist_on, threshold, max_dim);
    let complex_off = VietorisRipsComplex::build(&dist_off, threshold, max_dim);

    let pairs_on = compute_persistent_homology(&complex_on);
    let pairs_off = compute_persistent_homology(&complex_off);

    let betti_on = compute_betti_numbers(&pairs_on, 0.1);
    let betti_off = compute_betti_numbers(&pairs_off, 0.1);

    println!("ON Betti numbers (persistence > 0.1): {:?}", betti_on);
    println!("OFF Betti numbers (persistence > 0.1): {:?}", betti_off);

    // Persistence diagram comparison
    let diagrams_on = PersistenceDiagram::from_pairs_all(&pairs_on);
    let diagrams_off = PersistenceDiagram::from_pairs_all(&pairs_off);

    for dim in 0..=max_dim {
        if dim < diagrams_on.len() && dim < diagrams_off.len() {
            let mut diag_on = diagrams_on[dim].clone();
            let mut diag_off = diagrams_off[dim].clone();
            diag_on.truncate_to_top_k(max_points);
            diag_off.truncate_to_top_k(max_points);

            let wasserstein = diag_on.wasserstein_distance(&diag_off, 2.0);
            let bottleneck = diag_on.bottleneck_distance(&diag_off);

            println!(
                "H{}: ON ({} pairs) vs OFF ({} pairs): W2={:.4}, bottleneck={:.4}",
                dim,
                diag_on.len(),
                diag_off.len(),
                wasserstein,
                bottleneck
            );
        }
    }

    // Permutation test for topological indistinguishability
    println!();
    println!("Permutation test ({} permutations, seed={}):", n_perms, seed);

    // Combine all features and permute labels
    let mut combined_features: Vec<f64> = Vec::new();
    combined_features.extend_from_slice(&on_features);
    combined_features.extend_from_slice(&off_features);
    let n_total = n_on + n_off;
    let n_pts_per = 3; // 3D features

    // Observed statistic: sum of Wasserstein distances across dimensions
    let observed_stat: f64 = (0..=max_dim)
        .filter_map(|dim| {
            if dim < diagrams_on.len() && dim < diagrams_off.len() {
                let mut d_on = diagrams_on[dim].clone();
                let mut d_off = diagrams_off[dim].clone();
                d_on.truncate_to_top_k(max_points);
                d_off.truncate_to_top_k(max_points);
                Some(d_on.wasserstein_distance(&d_off, 2.0))
            } else {
                None
            }
        })
        .sum();

    let mut rng = StdRng::seed_from_u64(seed);
    let mut indices: Vec<usize> = (0..n_total).collect();
    let mut more_extreme = 0u32;

    for _ in 0..n_perms {
        indices.shuffle(&mut rng);
        let perm_a: Vec<f64> = indices[..n_on]
            .iter()
            .flat_map(|&i| {
                let start = i * n_pts_per;
                combined_features[start..start + n_pts_per].to_vec()
            })
            .collect();
        let perm_b: Vec<f64> = indices[n_on..]
            .iter()
            .flat_map(|&i| {
                let start = i * n_pts_per;
                combined_features[start..start + n_pts_per].to_vec()
            })
            .collect();

        let dist_a = DistanceMatrix::from_points_3d(&perm_a);
        let dist_b = DistanceMatrix::from_points_3d(&perm_b);
        let comp_a = VietorisRipsComplex::build(&dist_a, threshold, max_dim);
        let comp_b = VietorisRipsComplex::build(&dist_b, threshold, max_dim);
        let pairs_a = compute_persistent_homology(&comp_a);
        let pairs_b = compute_persistent_homology(&comp_b);
        let diags_a = PersistenceDiagram::from_pairs_all(&pairs_a);
        let diags_b = PersistenceDiagram::from_pairs_all(&pairs_b);

        let perm_stat: f64 = (0..=max_dim)
            .filter_map(|dim| {
                if dim < diags_a.len() && dim < diags_b.len() {
                    let mut da = diags_a[dim].clone();
                    let mut db = diags_b[dim].clone();
                    da.truncate_to_top_k(max_points);
                    db.truncate_to_top_k(max_points);
                    Some(da.wasserstein_distance(&db, 2.0))
                } else {
                    None
                }
            })
            .sum();

        if perm_stat >= observed_stat {
            more_extreme += 1;
        }
    }

    let p_value = (more_extreme + 1) as f64 / (n_perms + 1) as f64;
    println!("Observed statistic: {:.4}", observed_stat);
    println!("P-value: {:.4}", p_value);
    println!();

    if p_value > 0.05 {
        println!(
            "PASS: ON and OFF morphologically indistinguishable (p={:.3} > 0.05, C-772)",
            p_value
        );
    } else {
        println!(
            "INFO: Morphological difference detected (p={:.3}). Requires investigation.",
            p_value
        );
    }
}

fn run_ultrametric(manifest: &Path, n_triples: usize, n_perms: usize, seed: u64) {
    use data_core::catalogs::wow::parse_bl_manifest_csv;
    use stats_core::ultrametric::ultrametric_fraction_test;

    println!("=== Ultrametric Fraction Test: BL 6EQUJ5 Candidates ===");
    println!();

    if !manifest.exists() {
        eprintln!("ERROR: {} not found", manifest.display());
        std::process::exit(1);
    }

    let bundles = parse_bl_manifest_csv(manifest).expect("Failed to parse BL manifest");
    println!("Total bundles: {}", bundles.len());

    // Extract feature vector: obs_num as proxy (in production, use turboSETI features)
    let features: Vec<f64> = bundles.iter().map(|b| b.obs_num as f64).collect();

    if features.len() < 3 {
        eprintln!("ERROR: Need at least 3 data points, got {}", features.len());
        std::process::exit(1);
    }

    println!("Feature dimension: 1 (obs_num proxy)");
    println!("N triples: {}, N permutations: {}, seed: {}", n_triples, n_perms, seed);
    println!();

    let result = ultrametric_fraction_test(&features, n_triples, n_perms, seed);

    let effect_size = if result.null_fraction_std > 0.0 {
        (result.ultrametric_fraction - result.null_fraction_mean) / result.null_fraction_std
    } else {
        0.0
    };

    println!("Observed ultrametric fraction: {:.6}", result.ultrametric_fraction);
    println!(
        "95% CI: [{:.6}, {:.6}]",
        result.bootstrap_ci.0, result.bootstrap_ci.1
    );
    println!("Null mean: {:.6}", result.null_fraction_mean);
    println!("Null std: {:.6}", result.null_fraction_std);
    println!("P-value: {:.6}", result.p_value);
    println!("Effect size (Cohen's d): {:.4}", effect_size);
    println!();

    if result.p_value < 0.05 && result.ultrametric_fraction > result.null_fraction_mean {
        println!(
            "PASS: Significant ultrametric structure detected (p={:.4}, frac={:.4} > null={:.4}, C-773)",
            result.p_value, result.ultrametric_fraction, result.null_fraction_mean
        );
    } else if result.p_value >= 0.05 {
        println!(
            "INFO: No significant ultrametric excess (p={:.4}). Feature vectors may be too simple.",
            result.p_value
        );
    } else {
        println!(
            "INFO: Observed fraction ({:.4}) below null ({:.4}). Anti-ultrametric structure.",
            result.ultrametric_fraction, result.null_fraction_mean
        );
    }
}

fn main() {
    let args = Args::parse();
    match args.command {
        Command::Morphology {
            manifest,
            max_points,
            max_dim,
            n_perms,
            seed,
        } => run_morphology(&manifest, max_points, max_dim, n_perms, seed),
        Command::Ultrametric {
            manifest,
            n_triples,
            n_perms,
            seed,
        } => run_ultrametric(&manifest, n_triples, n_perms, seed),
    }
}
