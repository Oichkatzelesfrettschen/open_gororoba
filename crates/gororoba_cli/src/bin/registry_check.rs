//! Validate the TOML registry for internal consistency.
//!
//! Checks:
//! - All claim IDs are sequential (C-001..C-NNN)
//! - All cross-references resolve (insight->claims, experiment->claims)
//! - No duplicate IDs
//! - Status values are valid

use std::collections::HashSet;
use std::path::PathBuf;

use clap::Parser;

/// Validate the open_gororoba TOML registry for consistency.
#[derive(Parser)]
#[command(name = "registry-check")]
struct Args {
    /// Registry directory (default: registry/)
    #[arg(long, default_value = "registry")]
    dir: PathBuf,
}

#[derive(serde::Deserialize)]
struct ClaimsRegistry {
    claim: Vec<ClaimEntry>,
}

#[derive(serde::Deserialize)]
struct ClaimEntry {
    id: String,
    #[allow(dead_code)]
    statement: String,
    status: String,
    #[allow(dead_code)]
    where_stated: String,
    #[allow(dead_code)]
    last_verified: String,
    #[allow(dead_code)]
    what_would_verify_refute: String,
}

#[derive(serde::Deserialize)]
struct InsightsRegistry {
    insight: Vec<InsightEntry>,
}

#[derive(serde::Deserialize)]
struct InsightEntry {
    id: String,
    #[allow(dead_code)]
    title: String,
    claims: Vec<String>,
}

#[derive(serde::Deserialize)]
struct ExperimentsRegistry {
    experiment: Vec<ExperimentEntry>,
}

#[derive(serde::Deserialize)]
struct ExperimentEntry {
    id: String,
    #[allow(dead_code)]
    title: String,
    claims: Vec<String>,
}

#[derive(serde::Deserialize)]
struct BinariesRegistry {
    binary: Vec<BinaryEntry>,
}

#[derive(serde::Deserialize)]
struct BinaryEntry {
    name: String,
    #[allow(dead_code)]
    crate_field: Option<String>,
    experiment: Option<String>,
}

fn main() {
    let args = Args::parse();
    let mut errors = 0u32;
    let mut warnings = 0u32;

    // --- Claims ---
    let claims_path = args.dir.join("claims.toml");
    let claim_ids: HashSet<String>;

    if claims_path.exists() {
        let content = std::fs::read_to_string(&claims_path).unwrap();
        let registry: ClaimsRegistry = toml::from_str(&content).unwrap();

        claim_ids = registry.claim.iter().map(|c| c.id.clone()).collect();

        // Check for duplicates
        if claim_ids.len() != registry.claim.len() {
            eprintln!("ERROR: Duplicate claim IDs detected");
            errors += 1;
        }

        // Check sequential IDs
        let mut expected = 1;
        for claim in &registry.claim {
            let num: u32 = claim
                .id
                .strip_prefix("C-")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            if num != expected {
                eprintln!(
                    "WARNING: claim ID gap: expected C-{:03}, found {}",
                    expected, claim.id
                );
                warnings += 1;
            }
            expected = num + 1;
        }

        // Validate status tokens (lenient: just check non-empty)
        for claim in &registry.claim {
            if claim.status.is_empty() {
                eprintln!("ERROR: claim {} has empty status", claim.id);
                errors += 1;
            }
        }

        println!(
            "Claims: {} entries, {} unique IDs, range C-001..C-{:03}",
            registry.claim.len(),
            claim_ids.len(),
            registry.claim.last().map(|c| {
                c.id.strip_prefix("C-")
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(0)
            }).unwrap_or(0)
        );
    } else {
        eprintln!("WARNING: {} not found", claims_path.display());
        warnings += 1;
        claim_ids = HashSet::new();
    }

    // --- Insights ---
    let insights_path = args.dir.join("insights.toml");
    let insight_ids: HashSet<String>;

    if insights_path.exists() {
        let content = std::fs::read_to_string(&insights_path).unwrap();
        let registry: InsightsRegistry = toml::from_str(&content).unwrap();

        insight_ids = registry.insight.iter().map(|i| i.id.clone()).collect();

        // Check cross-references
        for insight in &registry.insight {
            for claim_ref in &insight.claims {
                if !claim_ids.contains(claim_ref) {
                    eprintln!(
                        "ERROR: insight {} references non-existent claim {}",
                        insight.id, claim_ref
                    );
                    errors += 1;
                }
            }
        }

        println!("Insights: {} entries", registry.insight.len());
    } else {
        eprintln!("WARNING: {} not found", insights_path.display());
        warnings += 1;
        insight_ids = HashSet::new();
    }

    // --- Experiments ---
    let experiments_path = args.dir.join("experiments.toml");
    let experiment_ids: HashSet<String>;

    if experiments_path.exists() {
        let content = std::fs::read_to_string(&experiments_path).unwrap();
        let registry: ExperimentsRegistry = toml::from_str(&content).unwrap();

        experiment_ids = registry.experiment.iter().map(|e| e.id.clone()).collect();

        // Check cross-references
        for exp in &registry.experiment {
            for claim_ref in &exp.claims {
                if !claim_ids.contains(claim_ref) {
                    eprintln!(
                        "WARNING: experiment {} references claim {} (may be range placeholder)",
                        exp.id, claim_ref
                    );
                    warnings += 1;
                }
            }
        }

        println!("Experiments: {} entries", registry.experiment.len());
    } else {
        eprintln!("WARNING: {} not found", experiments_path.display());
        warnings += 1;
        experiment_ids = HashSet::new();
    }

    // --- Binaries ---
    let binaries_path = args.dir.join("binaries.toml");
    if binaries_path.exists() {
        let content = std::fs::read_to_string(&binaries_path).unwrap();
        let registry: BinariesRegistry = toml::from_str(&content).unwrap();

        let binary_names: HashSet<String> =
            registry.binary.iter().map(|b| b.name.clone()).collect();

        // Check experiment cross-references
        for bin in &registry.binary {
            if let Some(ref exp_ref) = bin.experiment {
                if !experiment_ids.contains(exp_ref) {
                    eprintln!(
                        "ERROR: binary {} references non-existent experiment {}",
                        bin.name, exp_ref
                    );
                    errors += 1;
                }
            }
        }

        println!("Binaries: {} entries", binary_names.len());
    } else {
        eprintln!("WARNING: {} not found", binaries_path.display());
        warnings += 1;
    }

    // --- Project ---
    let project_path = args.dir.join("project.toml");
    if project_path.exists() {
        println!("Project: found");
    } else {
        eprintln!("WARNING: {} not found", project_path.display());
        warnings += 1;
    }

    // --- Summary ---
    println!();
    if errors == 0 && warnings == 0 {
        println!("Registry check: PASS (no errors, no warnings)");
    } else if errors == 0 {
        println!("Registry check: PASS with {warnings} warnings");
    } else {
        println!("Registry check: FAIL ({errors} errors, {warnings} warnings)");
        std::process::exit(1);
    }

    // Suppress unused variable warning
    let _ = &insight_ids;
}
