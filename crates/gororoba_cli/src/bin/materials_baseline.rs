//! Materials science baseline pipeline.
//!
//! Loads JARVIS and AFLOW datasets, featurizes compounds with Magpie-style
//! composition descriptors, and runs OLS linear regression baselines for
//! formation energy and band gap prediction.
//!
//! Usage:
//!   materials-baseline [--data-dir data/external] [--seed 42]

use clap::Parser;
use data_core::catalogs::aflow;
use data_core::catalogs::jarvis;
use materials_core::baselines;
use materials_core::featurizer;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "materials-baseline", about = "Materials science OLS baselines")]
struct Args {
    /// Directory containing downloaded datasets.
    #[arg(long, default_value = "data/external")]
    data_dir: String,

    /// PRNG seed for train/test split.
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Test set fraction (0..1).
    #[arg(long, default_value_t = 0.2)]
    test_fraction: f64,
}

/// One featurized sample ready for regression.
struct Sample {
    formula: String,
    features: Vec<f64>,
    formation_energy: f64,
    band_gap: f64,
}

/// Featurize JARVIS materials, skipping those with missing targets or unknown elements.
fn featurize_jarvis(materials: &[jarvis::JarvisMaterial]) -> Vec<Sample> {
    let mut samples = Vec::new();
    for mat in materials {
        let fe = match mat.formation_energy_peratom {
            Some(v) if v.is_finite() => v,
            _ => continue,
        };
        let bg = match mat.optb88vdw_bandgap {
            Some(v) if v.is_finite() => v,
            _ => continue,
        };
        let feats = match featurizer::featurize(&mat.formula) {
            Ok(f) => featurizer::feature_vector(&f),
            Err(_) => continue,
        };
        samples.push(Sample {
            formula: mat.formula.clone(),
            features: feats,
            formation_energy: fe,
            band_gap: bg,
        });
    }
    samples
}

/// Featurize AFLOW materials, skipping those with missing targets or unknown elements.
fn featurize_aflow(materials: &[aflow::AflowMaterial]) -> Vec<Sample> {
    let mut samples = Vec::new();
    for mat in materials {
        if !mat.enthalpy_formation_atom.is_finite() || !mat.egap.is_finite() {
            continue;
        }
        let feats = match featurizer::featurize(&mat.compound) {
            Ok(f) => featurizer::feature_vector(&f),
            Err(_) => continue,
        };
        samples.push(Sample {
            formula: mat.compound.clone(),
            features: feats,
            formation_energy: mat.enthalpy_formation_atom,
            band_gap: mat.egap,
        });
    }
    samples
}

/// Run baseline and print results for a given dataset/target combination.
fn run_and_report(
    label: &str,
    target_name: &str,
    features: &[Vec<f64>],
    targets: &[f64],
    test_fraction: f64,
    seed: u64,
) {
    match baselines::run_baseline(features, targets, test_fraction, seed) {
        Ok(result) => {
            println!(
                "  {:<20} {:<20} {:>6} {:>6}   {:>8.4}   {:>8.4}   {:>7.4}",
                label,
                target_name,
                result.n_train,
                result.n_test,
                result.mae,
                result.rmse,
                result.r_squared,
            );
        }
        Err(e) => {
            eprintln!("  {label} / {target_name}: FAILED -- {e}");
        }
    }
}

fn main() {
    let args = Args::parse();
    let data_dir = PathBuf::from(&args.data_dir);

    // -- Load JARVIS --
    let jarvis_path = data_dir.join("jarvis_dft_3d.json");
    let jarvis_samples = if jarvis_path.exists() {
        eprintln!("Loading JARVIS from {} ...", jarvis_path.display());
        match jarvis::parse_jarvis_json(&jarvis_path) {
            Ok(mats) => {
                eprintln!("  {} raw records", mats.len());
                let samples = featurize_jarvis(&mats);
                eprintln!("  {} featurized samples", samples.len());
                samples
            }
            Err(e) => {
                eprintln!("  JARVIS load failed: {e}");
                Vec::new()
            }
        }
    } else {
        eprintln!("JARVIS file not found: {}", jarvis_path.display());
        eprintln!("  Run: cargo run -p gororoba_cli --bin fetch-datasets -- --dataset JARVIS");
        Vec::new()
    };

    // -- Load AFLOW --
    let aflow_path = data_dir.join("aflow_materials.json");
    let aflow_samples = if aflow_path.exists() {
        eprintln!("Loading AFLOW from {} ...", aflow_path.display());
        match aflow::parse_aflow_json(&aflow_path) {
            Ok(mats) => {
                eprintln!("  {} raw records", mats.len());
                let samples = featurize_aflow(&mats);
                eprintln!("  {} featurized samples", samples.len());
                samples
            }
            Err(e) => {
                eprintln!("  AFLOW load failed: {e}");
                Vec::new()
            }
        }
    } else {
        eprintln!("AFLOW file not found: {}", aflow_path.display());
        eprintln!("  Run: cargo run -p gororoba_cli --bin fetch-datasets -- --dataset AFLOW");
        Vec::new()
    };

    // -- Results table --
    println!();
    println!("Materials Science OLS Baselines");
    println!("===============================");
    println!();
    println!(
        "  {:<20} {:<20} {:>6} {:>6}   {:>8}   {:>8}   {:>7}",
        "Dataset", "Target", "Train", "Test", "MAE", "RMSE", "R^2"
    );
    println!("  {}", "-".repeat(85));

    // JARVIS baselines
    if jarvis_samples.len() >= 10 {
        let feats: Vec<Vec<f64>> = jarvis_samples.iter().map(|s| s.features.clone()).collect();

        let fe_targets: Vec<f64> = jarvis_samples.iter().map(|s| s.formation_energy).collect();
        run_and_report(
            "JARVIS",
            "formation_energy",
            &feats,
            &fe_targets,
            args.test_fraction,
            args.seed,
        );

        let bg_targets: Vec<f64> = jarvis_samples.iter().map(|s| s.band_gap).collect();
        run_and_report(
            "JARVIS",
            "band_gap",
            &feats,
            &bg_targets,
            args.test_fraction,
            args.seed,
        );
    } else {
        eprintln!(
            "  JARVIS: too few samples ({}), skipping",
            jarvis_samples.len()
        );
    }

    // AFLOW baselines
    if aflow_samples.len() >= 10 {
        let feats: Vec<Vec<f64>> = aflow_samples.iter().map(|s| s.features.clone()).collect();

        let fe_targets: Vec<f64> = aflow_samples.iter().map(|s| s.formation_energy).collect();
        run_and_report(
            "AFLOW",
            "formation_energy",
            &feats,
            &fe_targets,
            args.test_fraction,
            args.seed,
        );

        let bg_targets: Vec<f64> = aflow_samples.iter().map(|s| s.band_gap).collect();
        run_and_report(
            "AFLOW",
            "band_gap",
            &feats,
            &bg_targets,
            args.test_fraction,
            args.seed,
        );
    } else {
        eprintln!(
            "  AFLOW: too few samples ({}), skipping",
            aflow_samples.len()
        );
    }

    // -- Cross-validation: overlapping formulas --
    if !jarvis_samples.is_empty() && !aflow_samples.is_empty() {
        println!();
        println!("Cross-database comparison (overlapping formulas)");
        println!("------------------------------------------------");

        let jarvis_formulas: std::collections::HashMap<&str, &Sample> = jarvis_samples
            .iter()
            .map(|s| (s.formula.as_str(), s))
            .collect();

        let mut n_overlap = 0;
        let mut fe_diffs = Vec::new();
        let mut bg_diffs = Vec::new();

        for aflow_s in &aflow_samples {
            if let Some(jarvis_s) = jarvis_formulas.get(aflow_s.formula.as_str()) {
                n_overlap += 1;
                fe_diffs.push((jarvis_s.formation_energy - aflow_s.formation_energy).abs());
                bg_diffs.push((jarvis_s.band_gap - aflow_s.band_gap).abs());
            }
        }

        println!("  Overlapping formulas: {n_overlap}");
        if !fe_diffs.is_empty() {
            let mean_fe: f64 = fe_diffs.iter().sum::<f64>() / fe_diffs.len() as f64;
            let mean_bg: f64 = bg_diffs.iter().sum::<f64>() / bg_diffs.len() as f64;
            println!("  Mean |dE_form| between databases: {mean_fe:.4} eV/atom");
            println!("  Mean |dE_gap|  between databases: {mean_bg:.4} eV");
        }
    }

    println!();
}
