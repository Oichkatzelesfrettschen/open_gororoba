//! Topological and ultrametric analysis of Wow! signal candidate data.
//!
//! Subcommands:
//! - `morphology`: Persistent homology on BL 6EQUJ5 candidate point clouds
//!   (ON vs OFF comparison via permutation test). Truncates to top-k=50 (VR OOM pitfall).
//! - `ultrametric`: Ultrametric fraction test on BL candidate feature vectors.
//!   Column-shuffled null with 10000 triples x 1000 permutations, BH-FDR correction.
//!
//! When `--candidates` is provided, uses real (freq, drift_rate, snr) features
//! from bl-doppler-search output instead of synthetic manifest proxies.

use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(
    name = "wow-topology",
    about = "Topological analysis of Wow! signal candidates"
)]
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

        /// Path to consolidated Doppler results CSV (real candidate features).
        #[arg(long)]
        candidates: Option<PathBuf>,

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

        /// Path to consolidated Doppler results CSV (real candidate features).
        #[arg(long)]
        candidates: Option<PathBuf>,

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

/// A candidate hit loaded from the consolidated CSV.
struct CandidateRow {
    pointing_type: String,
    freq_mhz: f64,
    drift_rate_hz_s: f64,
    snr: f64,
}

/// Load candidates from the consolidated doppler results CSV.
///
/// Expected columns: obs_id,pointing_type,cadence,freq_mhz,drift_rate_hz_s,snr,coarse_channel
fn load_candidates(path: &Path) -> Vec<CandidateRow> {
    let content = match std::fs::read_to_string(path) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ERROR: Failed to read {}: {}", path.display(), e);
            std::process::exit(1);
        }
    };

    let mut rows = Vec::new();
    for line in content.lines() {
        if line.starts_with('#') || line.starts_with("obs_id") || line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        if fields.len() < 6 {
            continue;
        }
        let pointing_type = fields[1].trim().to_string();
        let freq_mhz: f64 = fields[3].trim().parse().unwrap_or(0.0);
        let drift_rate_hz_s: f64 = fields[4].trim().parse().unwrap_or(0.0);
        let snr: f64 = fields[5].trim().parse().unwrap_or(0.0);

        rows.push(CandidateRow {
            pointing_type,
            freq_mhz,
            drift_rate_hz_s,
            snr,
        });
    }
    rows
}

fn run_morphology(
    manifest: &Path,
    candidates: &Option<PathBuf>,
    max_points: usize,
    max_dim: usize,
    n_perms: usize,
    seed: u64,
) {
    use data_core::catalogs::wow::{abacad_filter, parse_bl_manifest_csv};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use rand::seq::SliceRandom;
    use vacuum_frustration::vietoris_rips::{
        DistanceMatrix, PersistenceDiagram, VietorisRipsComplex, compute_betti_numbers,
        compute_persistent_homology,
    };

    println!("=== Persistent Homology: ON vs OFF Cadence Morphology ===");
    println!();

    let (on_features, off_features) = if let Some(cand_path) = candidates {
        // Real candidate features from Doppler search
        println!("Using real candidate features from {}", cand_path.display());
        let rows = load_candidates(cand_path);

        let on_feats: Vec<f64> = rows
            .iter()
            .filter(|r| r.pointing_type == "ON")
            .take(max_points)
            .flat_map(|r| vec![r.freq_mhz, r.drift_rate_hz_s, r.snr])
            .collect();
        let off_feats: Vec<f64> = rows
            .iter()
            .filter(|r| r.pointing_type == "OFF")
            .take(max_points)
            .flat_map(|r| vec![r.freq_mhz, r.drift_rate_hz_s, r.snr])
            .collect();

        println!("ON candidates: {}", on_feats.len() / 3);
        println!("OFF candidates: {}", off_feats.len() / 3);
        (on_feats, off_feats)
    } else {
        // Synthetic features from manifest metadata
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

        println!("Using synthetic features from manifest");
        println!("ON pointings: {}", on_bundles.len());
        println!("OFF pointings: {}", off_bundles.len());

        let make_features = |blist: &[&data_core::catalogs::wow::Bl6equj5Bundle]| -> Vec<f64> {
            let mut features = Vec::new();
            for b in blist.iter().take(max_points) {
                features.push(b.obs_num as f64);
                features.push(b.cadence as f64);
                features.push(if b.pointing_type == "ON" { 1.0 } else { 0.0 });
            }
            features
        };

        (make_features(&on_bundles), make_features(&off_bundles))
    };

    println!("Max dim: {}, Max points: {}", max_dim, max_points);
    println!();

    let n_on = on_features.len() / 3;
    let n_off = off_features.len() / 3;

    if n_on < 3 || n_off < 3 {
        eprintln!(
            "ERROR: Need at least 3 points per cloud, got ON={}, OFF={}",
            n_on, n_off
        );
        std::process::exit(1);
    }

    let dist_on = DistanceMatrix::from_points_3d(&on_features);
    let dist_off = DistanceMatrix::from_points_3d(&off_features);

    let threshold = 100.0;
    let complex_on = VietorisRipsComplex::build(&dist_on, threshold, max_dim);
    let complex_off = VietorisRipsComplex::build(&dist_off, threshold, max_dim);

    let pairs_on = compute_persistent_homology(&complex_on);
    let pairs_off = compute_persistent_homology(&complex_off);

    let betti_on = compute_betti_numbers(&pairs_on, 0.1);
    let betti_off = compute_betti_numbers(&pairs_off, 0.1);

    println!("ON Betti numbers (persistence > 0.1): {:?}", betti_on);
    println!("OFF Betti numbers (persistence > 0.1): {:?}", betti_off);

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

    // Permutation test
    println!();
    println!(
        "Permutation test ({} permutations, seed={}):",
        n_perms, seed
    );

    let mut combined_features: Vec<f64> = Vec::new();
    combined_features.extend_from_slice(&on_features);
    combined_features.extend_from_slice(&off_features);
    let n_total = n_on + n_off;
    let n_pts_per = 3;

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

    let source = if candidates.is_some() {
        "real candidates"
    } else {
        "synthetic proxies"
    };
    if p_value > 0.05 {
        println!(
            "PASS: ON and OFF morphologically indistinguishable (p={:.3} > 0.05, C-772, {})",
            p_value, source
        );
    } else {
        println!(
            "INFO: Morphological difference detected (p={:.3}, {}). Requires investigation.",
            p_value, source
        );
    }
}

fn run_ultrametric(
    manifest: &Path,
    candidates: &Option<PathBuf>,
    n_triples: usize,
    n_perms: usize,
    seed: u64,
) {
    use data_core::catalogs::wow::parse_bl_manifest_csv;
    use stats_core::ultrametric::local::{
        euclidean_distance_matrix_nd, local_ultrametricity_test_nd,
    };
    use stats_core::ultrametric::ultrametric_fraction_from_matrix;

    println!("=== Multi-D Ultrametric Analysis: BL 6EQUJ5 Candidates ===");
    println!();

    // Build N-D point vectors (each point is a Vec<f64> of dimension D)
    let (points, dim, source_label) = if let Some(cand_path) = candidates {
        println!("Using real candidate features from {}", cand_path.display());
        let rows = load_candidates(cand_path);
        println!("Total candidates: {}", rows.len());

        let pts: Vec<Vec<f64>> = rows
            .iter()
            .map(|r| vec![r.freq_mhz, r.drift_rate_hz_s, r.snr])
            .collect();
        println!("Feature dimension: 3 (freq_mhz, drift_rate_hz_s, snr)");
        (pts, 3usize, "real candidates")
    } else {
        if !manifest.exists() {
            eprintln!("ERROR: {} not found", manifest.display());
            std::process::exit(1);
        }

        let bundles = parse_bl_manifest_csv(manifest).expect("Failed to parse BL manifest");
        println!("Total bundles: {}", bundles.len());

        // Use (obs_num, cadence, pointing_type_flag) as 3D synthetic features
        let pts: Vec<Vec<f64>> = bundles
            .iter()
            .map(|b| {
                vec![
                    b.obs_num as f64,
                    b.cadence as f64,
                    if b.pointing_type == "ON" { 1.0 } else { 0.0 },
                ]
            })
            .collect();
        println!("Feature dimension: 3 (obs_num, cadence, pointing_flag)");
        (pts, 3usize, "synthetic proxies")
    };

    let n_points = points.len();
    if n_points < 3 {
        eprintln!("ERROR: Need at least 3 data points, got {}", n_points);
        std::process::exit(1);
    }

    println!("N points: {}", n_points);
    println!("Dimension: {}", dim);
    println!(
        "N triples: {}, N permutations: {}, seed: {}",
        n_triples, n_perms, seed
    );
    println!();

    // --- Global N-D ultrametric fraction via distance matrix ---
    println!("--- Global Ultrametric Fraction ({}D Euclidean) ---", dim);
    let dist_matrix = euclidean_distance_matrix_nd(&points);
    let global_frac = ultrametric_fraction_from_matrix(&dist_matrix, n_points, n_triples, seed);

    // Null: shuffle rows and recompute
    let mut null_fracs = Vec::with_capacity(n_perms);
    let mut shuffled = points.clone();
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    use rand::prelude::*;
    for _ in 0..n_perms {
        shuffled.shuffle(&mut rng);
        let null_dm = euclidean_distance_matrix_nd(&shuffled);
        let nf = ultrametric_fraction_from_matrix(&null_dm, n_points, n_triples, seed);
        null_fracs.push(nf);
    }
    let null_mean = null_fracs.iter().sum::<f64>() / n_perms as f64;
    let null_var = null_fracs
        .iter()
        .map(|f| (f - null_mean).powi(2))
        .sum::<f64>()
        / n_perms as f64;
    let null_std = null_var.sqrt();
    let n_ge = null_fracs.iter().filter(|&&f| f >= global_frac).count();
    let p_global = (n_ge as f64 + 1.0) / (n_perms as f64 + 1.0);
    let effect_size = if null_std > 0.0 {
        (global_frac - null_mean) / null_std
    } else {
        0.0
    };

    println!("Observed fraction: {:.6}", global_frac);
    println!("Null mean: {:.6} (+/- {:.6})", null_mean, null_std);
    println!("P-value: {:.6}", p_global);
    println!("Effect size (Cohen's d): {:.4}", effect_size);
    println!();

    // --- Local ultrametricity (N-D neighborhood test) ---
    println!("--- Local Ultrametricity ({}D neighborhoods) ---", dim);

    // Estimate epsilon: use median pairwise distance / 2
    let median_dist = {
        let mut sorted_dists = dist_matrix.clone();
        sorted_dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted_dists[sorted_dists.len() / 2]
    };
    let epsilon = median_dist * 0.5;
    println!(
        "Auto epsilon: {:.4} (median_dist/2 = {:.4}/2)",
        epsilon, median_dist
    );

    let local_result =
        local_ultrametricity_test_nd(&points, epsilon, n_triples.min(500), n_perms.min(50), seed);

    println!(
        "Testable neighborhoods: {}/{}",
        local_result.n_testable, n_points
    );
    println!(
        "Mean local index: {:.6} (median={:.6}, std={:.6})",
        local_result.mean_local_index,
        local_result.median_local_index,
        local_result.std_local_index,
    );
    println!("Null mean: {:.6}", local_result.null_mean_index);
    println!("P-value: {:.6}", local_result.p_value);
    println!("Verdict: {:?}", local_result.verdict);
    println!();

    // --- Summary ---
    println!("--- Summary (C-773, {}) ---", source_label);
    if p_global < 0.05 && global_frac > null_mean {
        println!(
            "PASS: Global {}D ultrametric excess (p={:.4}, frac={:.4} > null={:.4})",
            dim, p_global, global_frac, null_mean,
        );
    } else {
        println!(
            "INFO: No global {}D ultrametric excess (p={:.4}, frac={:.4}, null={:.4})",
            dim, p_global, global_frac, null_mean,
        );
    }

    if local_result.p_value < 0.05 {
        println!(
            "PASS: Local {}D ultrametric signal (p={:.4}, mean_idx={:.4} > null={:.4})",
            dim, local_result.p_value, local_result.mean_local_index, local_result.null_mean_index,
        );
    } else {
        println!(
            "INFO: No local {}D ultrametric signal (p={:.4}, mean_idx={:.4})",
            dim, local_result.p_value, local_result.mean_local_index,
        );
    }
}

fn main() {
    let args = Args::parse();
    match args.command {
        Command::Morphology {
            manifest,
            candidates,
            max_points,
            max_dim,
            n_perms,
            seed,
        } => run_morphology(&manifest, &candidates, max_points, max_dim, n_perms, seed),
        Command::Ultrametric {
            manifest,
            candidates,
            n_triples,
            n_perms,
            seed,
        } => run_ultrametric(&manifest, &candidates, n_triples, n_perms, seed),
    }
}
