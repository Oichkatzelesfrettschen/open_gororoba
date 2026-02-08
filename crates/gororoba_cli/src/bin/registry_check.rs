//! Validate the TOML registry for internal consistency.
//!
//! Checks:
//! - All claim IDs are sequential (C-001..C-NNN)
//! - All insight IDs are sequential (I-001..I-NNN)
//! - All cross-references resolve (insight->claims, experiment->claims)
//! - No duplicate IDs in any registry
//! - Status values are from valid enum set
//! - Claim count matches project.toml
//! - Binary registry matches actual [[bin]] sections in Cargo.toml
//! - Experiment->binary cross-references resolve

use std::collections::{BTreeSet, HashSet};
use std::path::PathBuf;

use clap::Parser;

/// Validate the open_gororoba TOML registry for consistency.
#[derive(Parser)]
#[command(name = "registry-check")]
struct Args {
    /// Registry directory (default: registry/)
    #[arg(long, default_value = "registry")]
    dir: PathBuf,

    /// Path to gororoba_cli Cargo.toml for binary cross-check
    #[arg(long, default_value = "crates/gororoba_cli/Cargo.toml")]
    cargo_toml: PathBuf,
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
    #[allow(dead_code)]
    date: Option<String>,
    status: Option<String>,
    claims: Vec<String>,
    #[allow(dead_code)]
    sprint: Option<u32>,
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
    binary: Option<String>,
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
    description: Option<String>,
    experiment: Option<String>,
}

#[derive(serde::Deserialize)]
struct ProjectRegistry {
    project: ProjectMeta,
}

#[derive(serde::Deserialize)]
struct ProjectMeta {
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    test_count: Option<u32>,
    claim_count: Option<u32>,
    insight_count: Option<u32>,
    binary_count: Option<u32>,
}

const VALID_CLAIM_STATUSES: &[&str] = &[
    "Verified",
    "Refuted",
    "Open",
    "Partial",
    "Pending",
    "Verified (algebraic)",
    "Verified (numerical)",
    "Verified (parsing)",
    "Verified (parsing + dimension check)",
    "Verified (exact match)",
    "Verified (all 4 dims)",
    "Verified (strict inclusions)",
    "Verified (all 3 transitions)",
    "Verified (dims 16, 32)",
    "Verified (dim=64)",
    "Verified (4476 rows across 5 dims)",
    "Verified (statistical)",
    "Verified (convergence)",
    "Verified (literature match)",
    "Verified (unit test)",
    "Refuted (E8 connection absent)",
    "Refuted (CSV wrong)",
    "Superseded",
    "Established",
    "Inconclusive",
    "Theoretical",
    "Closed/Toy",
    "Closed/Negative-Result",
    "Closed/Obstructed",
    "Closed/Analogy",
    "Closed/Research-Program",
    "Closed/Source-Insufficient",
    "Closed/Methodology-Insufficient",
    "Closed/Refuted",
];

const VALID_INSIGHT_STATUSES: &[&str] = &[
    "verified",
    "open",
    "superseded",
    "cross-validation-complete",
    "partial",
];

fn main() {
    let args = Args::parse();
    let mut errors = 0u32;
    let mut warnings = 0u32;

    // --- Claims ---
    let claims_path = args.dir.join("claims.toml");
    let claim_ids: HashSet<String>;
    let claim_count: usize;

    if claims_path.exists() {
        let content = std::fs::read_to_string(&claims_path).unwrap();
        let registry: ClaimsRegistry = toml::from_str(&content).unwrap();

        claim_ids = registry.claim.iter().map(|c| c.id.clone()).collect();
        claim_count = registry.claim.len();

        // Check for duplicates
        if claim_ids.len() != claim_count {
            eprintln!("ERROR: Duplicate claim IDs detected ({} unique vs {} total)",
                claim_ids.len(), claim_count);
            errors += 1;
        }

        // Check sequential IDs and detect gaps
        let mut expected = 1u32;
        let mut gaps = Vec::new();
        for claim in &registry.claim {
            let num: u32 = claim
                .id
                .strip_prefix("C-")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            if num > expected {
                for missing in expected..num {
                    gaps.push(format!("C-{missing:03}"));
                }
            }
            expected = num + 1;
        }
        if !gaps.is_empty() {
            eprintln!("WARNING: {} claim ID gaps: {}", gaps.len(),
                if gaps.len() <= 5 { gaps.join(", ") } else {
                    format!("{}... and {} more", gaps[..5].join(", "), gaps.len() - 5)
                });
            warnings += 1;
        }

        // Validate status tokens
        for claim in &registry.claim {
            if claim.status.is_empty() {
                eprintln!("ERROR: claim {} has empty status", claim.id);
                errors += 1;
            } else {
                // Check against known statuses (case-sensitive prefix match)
                let status_ok = VALID_CLAIM_STATUSES.iter().any(|valid| {
                    claim.status == *valid || claim.status.starts_with(valid)
                });
                if !status_ok {
                    eprintln!("WARNING: claim {} has unusual status: \"{}\"",
                        claim.id, claim.status);
                    warnings += 1;
                }
            }
        }

        println!(
            "Claims: {} entries, {} unique IDs, range C-001..C-{:03}",
            claim_count,
            claim_ids.len(),
            registry.claim.last().map(|c| {
                c.id.strip_prefix("C-")
                    .and_then(|s| s.parse::<u32>().ok())
                    .unwrap_or(0)
            }).unwrap_or(0)
        );
    } else {
        eprintln!("ERROR: {} not found", claims_path.display());
        errors += 1;
        claim_ids = HashSet::new();
        claim_count = 0;
    }

    // --- Insights ---
    let insights_path = args.dir.join("insights.toml");
    let insight_ids: HashSet<String>;
    let insight_count: usize;

    if insights_path.exists() {
        let content = std::fs::read_to_string(&insights_path).unwrap();
        let registry: InsightsRegistry = toml::from_str(&content).unwrap();

        insight_ids = registry.insight.iter().map(|i| i.id.clone()).collect();
        insight_count = registry.insight.len();

        // Check for duplicates
        if insight_ids.len() != insight_count {
            eprintln!("ERROR: Duplicate insight IDs detected");
            errors += 1;
        }

        // Check sequential IDs (I-001..I-NNN)
        let nums: BTreeSet<u32> = registry.insight.iter().filter_map(|i| {
            i.id.strip_prefix("I-").and_then(|s| s.parse().ok())
        }).collect();

        if let (Some(&first), Some(&last)) = (nums.iter().next(), nums.iter().next_back()) {
            let expected_count = (last - first + 1) as usize;
            if nums.len() != expected_count {
                let missing: Vec<String> = (first..=last)
                    .filter(|n| !nums.contains(n))
                    .map(|n| format!("I-{n:03}"))
                    .collect();
                eprintln!("WARNING: {} insight ID gaps: {}", missing.len(), missing.join(", "));
                warnings += 1;
            }
        }

        // Check cross-references and status
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
            if let Some(ref status) = insight.status {
                if !VALID_INSIGHT_STATUSES.contains(&status.as_str()) {
                    eprintln!("WARNING: insight {} has unusual status: \"{}\"",
                        insight.id, status);
                    warnings += 1;
                }
            }
        }

        println!("Insights: {} entries, IDs {:?}..{:?}",
            insight_count,
            nums.iter().next().map(|n| format!("I-{n:03}")).unwrap_or_default(),
            nums.iter().next_back().map(|n| format!("I-{n:03}")).unwrap_or_default(),
        );
    } else {
        eprintln!("ERROR: {} not found", insights_path.display());
        errors += 1;
        insight_ids = HashSet::new();
        insight_count = 0;
    }

    // --- Experiments ---
    let experiments_path = args.dir.join("experiments.toml");
    let experiment_ids: HashSet<String>;
    let binary_names_from_experiments: HashSet<String>;

    if experiments_path.exists() {
        let content = std::fs::read_to_string(&experiments_path).unwrap();
        let registry: ExperimentsRegistry = toml::from_str(&content).unwrap();

        experiment_ids = registry.experiment.iter().map(|e| e.id.clone()).collect();
        binary_names_from_experiments = registry.experiment.iter()
            .filter_map(|e| e.binary.clone())
            .collect();

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
        binary_names_from_experiments = HashSet::new();
    }

    // --- Binaries ---
    let binaries_path = args.dir.join("binaries.toml");
    let registry_binary_names: HashSet<String>;

    if binaries_path.exists() {
        let content = std::fs::read_to_string(&binaries_path).unwrap();
        let registry: BinariesRegistry = toml::from_str(&content).unwrap();

        registry_binary_names = registry.binary.iter().map(|b| b.name.clone()).collect();

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

        // Check experiment->binary cross-references
        for exp_binary in &binary_names_from_experiments {
            if !registry_binary_names.contains(exp_binary) {
                eprintln!(
                    "WARNING: experiment references binary \"{}\" not in binaries.toml",
                    exp_binary
                );
                warnings += 1;
            }
        }

        // Cross-check against Cargo.toml [[bin]] sections
        if args.cargo_toml.exists() {
            let cargo_content = std::fs::read_to_string(&args.cargo_toml).unwrap();
            let cargo_binary_names: HashSet<String> = cargo_content
                .lines()
                .filter(|l| l.starts_with("name = \""))
                .filter_map(|l| {
                    l.strip_prefix("name = \"")
                        .and_then(|s| s.strip_suffix('"'))
                        .map(|s| s.to_string())
                })
                // Exclude the library name (gororoba_cli)
                .filter(|n| n != "gororoba_cli")
                .collect();

            let missing_from_registry: Vec<_> = cargo_binary_names
                .difference(&registry_binary_names)
                .collect();
            let missing_from_cargo: Vec<_> = registry_binary_names
                .difference(&cargo_binary_names)
                .collect();

            if !missing_from_registry.is_empty() {
                eprintln!(
                    "WARNING: {} binaries in Cargo.toml but not in binaries.toml: {}",
                    missing_from_registry.len(),
                    missing_from_registry.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                );
                warnings += 1;
            }
            if !missing_from_cargo.is_empty() {
                eprintln!(
                    "WARNING: {} binaries in binaries.toml but not in Cargo.toml: {}",
                    missing_from_cargo.len(),
                    missing_from_cargo.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                );
                warnings += 1;
            }
        }

        println!("Binaries: {} entries", registry_binary_names.len());
    } else {
        eprintln!("WARNING: {} not found", binaries_path.display());
        warnings += 1;
        registry_binary_names = HashSet::new();
    }

    // --- Project ---
    let project_path = args.dir.join("project.toml");
    if project_path.exists() {
        let content = std::fs::read_to_string(&project_path).unwrap();
        let project: ProjectRegistry = toml::from_str(&content).unwrap();

        // Cross-check counts
        if let Some(expected_claims) = project.project.claim_count {
            if expected_claims as usize != claim_count {
                eprintln!(
                    "WARNING: project.toml claim_count={} but claims.toml has {} entries",
                    expected_claims, claim_count
                );
                warnings += 1;
            }
        }
        if let Some(expected_insights) = project.project.insight_count {
            if expected_insights as usize != insight_count {
                eprintln!(
                    "WARNING: project.toml insight_count={} but insights.toml has {} entries",
                    expected_insights, insight_count
                );
                warnings += 1;
            }
        }
        if let Some(expected_binaries) = project.project.binary_count {
            if expected_binaries as usize != registry_binary_names.len() {
                eprintln!(
                    "WARNING: project.toml binary_count={} but binaries.toml has {} entries",
                    expected_binaries, registry_binary_names.len()
                );
                warnings += 1;
            }
        }

        println!("Project: found (v{})", project.project.name);
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

    // Suppress unused variable warnings
    let _ = &insight_ids;
    let _ = &insight_count;
}
