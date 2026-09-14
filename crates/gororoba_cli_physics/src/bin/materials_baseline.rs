//! Materials science baseline pipeline.
//!
//! Loads JARVIS and AFLOW datasets, featurizes compounds with Magpie-style
//! composition descriptors, and runs OLS linear regression baselines for
//! formation energy and band gap prediction.
//!
//! Usage:
//!   `materials-baseline [--data-dir data/external] [--seed 42]`

use clap::Parser;
use data_core::catalogs::{aflow, jarvis};
use materials_core::{baselines, featurizer};
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

const BASELINE_FEATURE_INDICES: [usize; 19] = [
    0, 1, 2, 3, // composition
    4, 5, 6, 7, 8, // mass
    24, 25, 26, 27, 28, // valence
    49, 50, 51, 52, 53, // atomic number
];

/// Select columns whose elemental properties exist for every recognized element.
///
/// The observation mask remains authoritative: an absent selected value rejects
/// the sample instead of converting missingness to a numerical sentinel.
fn select_baseline_features(
    masked_features: &featurizer::MaskedFeatureVector,
) -> Option<Vec<f64>> {
    BASELINE_FEATURE_INDICES
        .iter()
        .map(|&feature_index| {
            let value = *masked_features.values.get(feature_index)?;
            let observed = *masked_features.observed.get(feature_index)?;
            (observed && value.is_finite()).then_some(value)
        })
        .collect()
}

fn baseline_features(formula: &str) -> Option<Vec<f64>> {
    let composition_features = featurizer::featurize(formula).ok()?;
    let masked_features = featurizer::feature_vector(&composition_features);
    select_baseline_features(&masked_features)
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
        let feats = match baseline_features(&mat.formula) {
            Some(values) => values,
            None => continue,
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
        let feats = match baseline_features(&mat.compound) {
            Some(values) => values,
            None => continue,
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
        eprintln!("  Run: cargo run -p gororoba_cli_data --bin fetch-datasets -- --dataset JARVIS");
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
        eprintln!("  Run: cargo run -p gororoba_cli_data --bin fetch-datasets -- --dataset AFLOW");
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

#[cfg(test)]
mod tests {
    use super::*;

    fn jarvis_alumina() -> jarvis::JarvisMaterial {
        jarvis::JarvisMaterial {
            jid: "JVASP-test".to_string(),
            formula: "Al2O3".to_string(),
            elements: vec!["Al".to_string(), "O".to_string()],
            nelements: 2,
            energy_per_atom: None,
            formation_energy_peratom: Some(-3.2),
            optb88vdw_bandgap: Some(6.1),
            ehull: None,
            spg_symbol: None,
            spg_number: None,
            density: None,
            volume: None,
        }
    }

    fn aflow_alumina() -> aflow::AflowMaterial {
        aflow::AflowMaterial {
            auid: "aflow:test".to_string(),
            compound: "Al2O3".to_string(),
            species: vec!["Al".to_string(), "O".to_string()],
            nspecies: 2,
            natoms: 5,
            enthalpy_formation_atom: -3.2,
            egap: 6.1,
            density: None,
            volume_atom: None,
            spacegroup: None,
            pearson_symbol: None,
        }
    }

    #[test]
    fn alumina_with_missing_optional_properties_reaches_both_baselines() {
        let masked_features = featurizer::feature_vector(
            &featurizer::featurize("Al2O3").expect("Al2O3 must featurize"),
        );
        assert!(masked_features.into_complete_values().is_none());

        let jarvis_samples = featurize_jarvis(&[jarvis_alumina()]);
        let aflow_samples = featurize_aflow(&[aflow_alumina()]);

        assert_eq!(jarvis_samples.len(), 1);
        assert_eq!(aflow_samples.len(), 1);
        assert_eq!(jarvis_samples[0].features.len(), 19);
        assert_eq!(aflow_samples[0].features.len(), 19);
    }

    #[test]
    fn baseline_subset_rejects_unobserved_selected_values() {
        let mut masked_features = featurizer::feature_vector(
            &featurizer::featurize("Al2O3").expect("Al2O3 must featurize"),
        );
        masked_features.observed[BASELINE_FEATURE_INDICES[0]] = false;
        masked_features.values[BASELINE_FEATURE_INDICES[0]] = 0.0;

        assert!(select_baseline_features(&masked_features).is_none());
    }
}
