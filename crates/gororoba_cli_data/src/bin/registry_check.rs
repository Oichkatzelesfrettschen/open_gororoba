//! Validate the TOML registry for internal consistency.
//!
//! Checks:
//! - Claim/insight ID gaps are governed by policy (warn/error/off)
//! - All cross-references resolve (insight->claims, experiment->claims)
//! - No duplicate IDs in any registry
//! - Status values are from valid enum set
//! - Claim count matches project.toml
//! - Binary registry matches actual [[bin]] sections in Cargo.toml
//! - Experiment->binary cross-references resolve

use std::{
    collections::{BTreeSet, HashSet},
    path::{Path, PathBuf},
};

use clap::{Parser, ValueEnum};
use data_core::registry::{ArtifactRegistry, LacunaeRegistry, MonographRegistry};
use gororoba_cli::claims::schema::TOML_CLAIM_STATUSES;
use walkdir::WalkDir;

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

    /// Typed schema policy for selected registry files.
    /// warn: emit warnings but continue (default).
    /// error: fail when typed schema drift is found.
    /// off: skip typed schema drift checks.
    #[arg(long, value_enum, default_value_t = TypedPolicy::Warn)]
    typed_policy: TypedPolicy,

    /// Identity gap policy for claim/insight ID gaps.
    /// warn: emit warnings for drift beyond governed baseline (default).
    /// error: fail when drift beyond governed baseline is found.
    /// off: skip claim/insight governed gap checks.
    #[arg(long, value_enum, default_value_t = IdentityGapPolicy::Warn)]
    identity_gap_policy: IdentityGapPolicy,

    /// Path to governed identity-gap baseline file.
    #[arg(long, default_value = "registry/identity_gap_policy.toml")]
    identity_gap_policy_file: PathBuf,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum TypedPolicy {
    Warn,
    Error,
    Off,
}

impl TypedPolicy {
    fn as_str(self) -> &'static str {
        match self {
            Self::Warn => "warn",
            Self::Error => "error",
            Self::Off => "off",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum IdentityGapPolicy {
    Warn,
    Error,
    Off,
}

impl IdentityGapPolicy {
    fn as_str(self) -> &'static str {
        match self {
            Self::Warn => "warn",
            Self::Error => "error",
            Self::Off => "off",
        }
    }
}

#[derive(serde::Deserialize)]
struct ClaimsRegistry {
    claim: Vec<ClaimEntry>,
}

/// Claim entry struct -- uses #[serde(default)] on all fields except id/status
/// so that new optional fields added by the consolidation pipeline do not break
/// deserialization.
// Constructed by serde::Deserialize; not all fields read by this binary.
#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct ClaimEntry {
    id: String,
    #[serde(default)]
    statement: String,
    status: String,
    #[serde(default)]
    where_stated: String,
    #[serde(default)]
    last_verified: String,
    #[serde(default)]
    what_would_verify_refute: String,
    // New optional fields added by consolidation pipeline
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    confidence: Option<String>,
    #[serde(default)]
    phase: Option<String>,
    #[serde(default)]
    sprint: Option<u32>,
    #[serde(default)]
    dependencies: Option<Vec<String>>,
    #[serde(default)]
    claims: Option<Vec<String>>,
    #[serde(default)]
    insights: Option<Vec<String>>,
    #[serde(default)]
    supporting_evidence: Option<Vec<String>>,
    #[serde(default)]
    verification_method: Option<String>,
    #[serde(default)]
    status_note: Option<String>,
}

#[derive(serde::Deserialize)]
struct InsightsRegistry {
    insight: Vec<InsightEntry>,
}

// Constructed by serde::Deserialize; not all fields read by this binary.
#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct InsightEntry {
    id: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    insight: Option<String>,
    #[serde(default)]
    date: Option<String>,
    status: Option<String>,
    #[serde(default)]
    claims: Vec<String>,
    #[serde(default)]
    related_claims: Vec<String>,
    #[serde(default)]
    sprint: Option<u32>,
    #[serde(default)]
    summary: Option<String>,
    #[serde(default)]
    supporting_evidence: Option<Vec<String>>,
    #[serde(default)]
    confidence: Option<String>,
    #[serde(default)]
    verified_date: Option<String>,
    #[serde(default)]
    phase: Option<String>,
    #[serde(default)]
    experimental_support: Option<Vec<String>>,
}

// Constructed by serde::Deserialize; not all fields read by this binary.
#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct ExperimentsRegistry {
    #[serde(default)]
    experiments: Option<ExperimentsHeader>,
    experiment: Vec<ExperimentEntry>,
}

/// Header section [experiments] in experiments.toml.
// Constructed by serde::Deserialize; not all fields read by this binary.
#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct ExperimentsHeader {
    #[serde(default)]
    updated: Option<String>,
    #[serde(default)]
    authoritative: Option<bool>,
    #[serde(default)]
    experiment_count: Option<u32>,
    #[serde(default)]
    deterministic_count: Option<u32>,
    #[serde(default)]
    gpu_count: Option<u32>,
    #[serde(default)]
    seeded_count: Option<u32>,
    #[serde(default)]
    status_allowlist: Option<Vec<String>>,
}

// Constructed by serde::Deserialize; not all fields read by this binary.
#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct ExperimentEntry {
    id: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    binary: Option<String>,
    #[serde(default)]
    claims: Vec<String>,
    // Extra fields present in the rich schema
    #[serde(default)]
    binary_registered: Option<bool>,
    #[serde(default)]
    binary_experiment_declared: Option<String>,
    #[serde(default)]
    method: Option<String>,
    #[serde(default)]
    input: Option<String>,
    #[serde(default)]
    output: Option<Vec<String>>,
    #[serde(default)]
    run: Option<String>,
    #[serde(default)]
    run_command_sha256: Option<String>,
    #[serde(default)]
    claim_refs: Option<Vec<String>>,
    #[serde(default)]
    deterministic: Option<bool>,
    #[serde(default)]
    gpu: Option<bool>,
    #[serde(default)]
    seed: Option<u64>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    status_token: Option<String>,
    #[serde(default)]
    lineage_id: Option<String>,
    #[serde(default)]
    input_path_refs: Option<Vec<String>>,
    #[serde(default)]
    output_path_refs: Option<Vec<String>>,
    #[serde(default)]
    dataset_refs: Option<Vec<String>>,
    #[serde(default)]
    reproducibility_class: Option<String>,
}

#[derive(serde::Deserialize)]
struct BinariesRegistry {
    binary: Vec<BinaryEntry>,
}

// Constructed by serde::Deserialize; not all fields read by this binary.
#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct BinaryEntry {
    name: String,
    description: Option<String>,
    experiment: Option<String>,
}

#[derive(serde::Deserialize)]
struct ProjectRegistry {
    project: ProjectMeta,
}

// Constructed by serde::Deserialize; not all fields read by this binary.
#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct ProjectMeta {
    name: String,
    test_count: Option<u32>,
    claim_count: Option<u32>,
    insight_count: Option<u32>,
    experiment_count: Option<u32>,
    complete_experiment_count: Option<u32>,
    binary_count: Option<u32>,
}

#[derive(serde::Deserialize)]
struct CargoManifest {
    #[serde(default)]
    bin: Vec<CargoBinEntry>,
}

#[derive(serde::Deserialize)]
struct CargoBinEntry {
    name: String,
}

#[derive(Default, serde::Deserialize)]
struct IdentityGapPolicyFile {
    #[serde(default)]
    claims: IdentityGapPolicyLane,
    #[serde(default)]
    insights: IdentityGapPolicyLane,
}

#[derive(Default, serde::Deserialize)]
struct IdentityGapPolicyLane {
    #[serde(default)]
    governed_gap_ids: Vec<String>,
}

#[derive(Default)]
struct LoadedIdentityGapPolicy {
    claims: BTreeSet<u32>,
    insights: BTreeSet<u32>,
}

/// Valid claim statuses: canonical TOML tokens from schema.rs plus legacy
/// parenthetical variants still found in the registry.
const VALID_CLAIM_STATUSES_LEGACY: &[&str] = &[
    // Legacy parenthetical variants (pre-consolidation)
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
    // Legacy open statuses
    "Open",
    "Pending",
];

const VALID_INSIGHT_STATUSES: &[&str] = &[
    "verified",
    "open",
    "superseded",
    "cross-validation-complete",
    "partial",
];

fn parse_all_registry_toml_files(registry_dir: &Path) -> (usize, Vec<String>) {
    let mut parsed_files = 0usize;
    let mut parse_errors = Vec::new();

    for entry in WalkDir::new(registry_dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file())
    {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("toml") {
            continue;
        }

        parsed_files += 1;
        let content = match std::fs::read_to_string(path) {
            Ok(content) => content,
            Err(err) => {
                parse_errors.push(format!("{} (read error: {err})", path.display()));
                continue;
            }
        };

        if let Err(err) = toml::from_str::<toml::Value>(&content) {
            parse_errors.push(format!("{} ({err})", path.display()));
        }
    }

    (parsed_files, parse_errors)
}

fn parse_typed_registry_file<T: serde::de::DeserializeOwned>(
    path: &Path,
    label: &str,
) -> Option<String> {
    if !path.exists() {
        return Some(format!(
            "{} not found for typed validation ({label})",
            path.display()
        ));
    }

    let content = match std::fs::read_to_string(path) {
        Ok(content) => content,
        Err(err) => {
            return Some(format!(
                "{} read failure for typed validation ({label}): {err}",
                path.display()
            ));
        }
    };

    match toml::from_str::<T>(&content) {
        Ok(_) => None,
        Err(err) => Some(format!(
            "{} failed typed validation ({label}): {err}",
            path.display()
        )),
    }
}

fn load_cargo_bin_names(cargo_toml: &Path) -> Result<HashSet<String>, String> {
    let cargo_content = std::fs::read_to_string(cargo_toml)
        .map_err(|err| format!("{} read failure: {err}", cargo_toml.display()))?;
    let manifest: CargoManifest = toml::from_str(&cargo_content)
        .map_err(|err| format!("{} parse failure: {err}", cargo_toml.display()))?;

    Ok(manifest
        .bin
        .into_iter()
        .filter_map(|bin| {
            let name = bin.name.trim();
            if name.is_empty() {
                None
            } else {
                Some(name.to_string())
            }
        })
        .collect())
}

fn parse_numbered_id(id: &str, prefix: &str) -> Option<u32> {
    id.strip_prefix(prefix)
        .and_then(|suffix| suffix.parse::<u32>().ok())
}

fn collect_missing_ids(ids: &BTreeSet<u32>) -> Vec<u32> {
    let (Some(&first), Some(&last)) = (ids.iter().next(), ids.iter().next_back()) else {
        return Vec::new();
    };
    (first..=last)
        .filter(|value| !ids.contains(value))
        .collect()
}

fn compress_gap_ids(prefix: &str, ids: &[u32]) -> Vec<String> {
    if ids.is_empty() {
        return Vec::new();
    }

    let mut ranges = Vec::new();
    let mut start = ids[0];
    let mut prev = ids[0];

    for &id in &ids[1..] {
        if id == prev + 1 {
            prev = id;
            continue;
        }

        if start == prev {
            ranges.push(format!("{prefix}{start:03}"));
        } else {
            ranges.push(format!("{prefix}{start:03}..{prefix}{prev:03}"));
        }
        start = id;
        prev = id;
    }

    if start == prev {
        ranges.push(format!("{prefix}{start:03}"));
    } else {
        ranges.push(format!("{prefix}{start:03}..{prefix}{prev:03}"));
    }

    ranges
}

fn summarize_gap_ids(prefix: &str, ids: &[u32]) -> String {
    let ranges = compress_gap_ids(prefix, ids);
    if ranges.is_empty() {
        return "none".to_string();
    }
    if ranges.len() <= 5 {
        return ranges.join(", ");
    }
    format!(
        "{}... and {} more",
        ranges[..5].join(", "),
        ranges.len() - 5
    )
}

fn parse_governed_gap_token(token: &str, prefix: &str) -> Result<Vec<u32>, String> {
    let token = token.trim();
    if token.is_empty() {
        return Err("empty token".to_string());
    }

    if let Some((start_raw, end_raw)) = token.split_once("..") {
        let start = parse_numbered_id(start_raw.trim(), prefix)
            .ok_or_else(|| format!("invalid start ID: {start_raw}"))?;
        let end = parse_numbered_id(end_raw.trim(), prefix)
            .ok_or_else(|| format!("invalid end ID: {end_raw}"))?;
        if end < start {
            return Err(format!("range end precedes start: {start_raw}..{end_raw}"));
        }
        Ok((start..=end).collect())
    } else {
        let id = parse_numbered_id(token, prefix).ok_or_else(|| format!("invalid ID: {token}"))?;
        Ok(vec![id])
    }
}

fn expand_governed_gap_tokens(
    tokens: &[String],
    lane_label: &str,
    prefix: &str,
) -> Result<BTreeSet<u32>, String> {
    let mut ids = BTreeSet::new();
    for token in tokens {
        let expanded = parse_governed_gap_token(token, prefix).map_err(|err| {
            format!("{lane_label}.governed_gap_ids entry \"{token}\" parse error: {err}")
        })?;
        ids.extend(expanded);
    }
    Ok(ids)
}

fn load_identity_gap_policy(path: &Path) -> Result<LoadedIdentityGapPolicy, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|err| format!("{} read failure: {err}", path.display()))?;
    let raw: IdentityGapPolicyFile = toml::from_str(&content)
        .map_err(|err| format!("{} parse failure: {err}", path.display()))?;

    let claims = expand_governed_gap_tokens(&raw.claims.governed_gap_ids, "claims", "C-")?;
    let insights = expand_governed_gap_tokens(&raw.insights.governed_gap_ids, "insights", "I-")?;

    Ok(LoadedIdentityGapPolicy { claims, insights })
}

fn evaluate_identity_gap_lane(
    lane_label: &str,
    prefix: &str,
    missing_ids: &[u32],
    governed_ids: &BTreeSet<u32>,
    mode: IdentityGapPolicy,
    warnings: &mut u32,
    errors: &mut u32,
) {
    let missing_set: BTreeSet<u32> = missing_ids.iter().copied().collect();
    let governed_intent: Vec<u32> = missing_ids
        .iter()
        .copied()
        .filter(|id| governed_ids.contains(id))
        .collect();
    let drift_beyond_baseline: Vec<u32> = missing_ids
        .iter()
        .copied()
        .filter(|id| !governed_ids.contains(id))
        .collect();
    let baseline_now_present: Vec<u32> = governed_ids
        .iter()
        .copied()
        .filter(|id| !missing_set.contains(id))
        .collect();

    if governed_intent.is_empty() {
        println!("Identity gap ({lane_label}): governed-intent gaps: none");
    } else {
        println!(
            "Identity gap ({lane_label}): governed-intent gaps: {} ({})",
            governed_intent.len(),
            summarize_gap_ids(prefix, &governed_intent)
        );
    }

    if drift_beyond_baseline.is_empty() {
        println!("Identity gap ({lane_label}): drift beyond baseline: none");
    } else {
        match mode {
            IdentityGapPolicy::Warn => {
                eprintln!(
                    "WARNING: Identity gap ({lane_label}) drift beyond baseline (policy={}): {} ({})",
                    mode.as_str(),
                    drift_beyond_baseline.len(),
                    summarize_gap_ids(prefix, &drift_beyond_baseline)
                );
                *warnings += 1;
            }
            IdentityGapPolicy::Error => {
                eprintln!(
                    "ERROR: Identity gap ({lane_label}) drift beyond baseline (policy={}): {} ({})",
                    mode.as_str(),
                    drift_beyond_baseline.len(),
                    summarize_gap_ids(prefix, &drift_beyond_baseline)
                );
                *errors += 1;
            }
            IdentityGapPolicy::Off => {}
        }
    }

    if !baseline_now_present.is_empty() {
        println!(
            "Identity gap ({lane_label}): baseline entries now present (candidate cleanup): {} ({})",
            baseline_now_present.len(),
            summarize_gap_ids(prefix, &baseline_now_present)
        );
    }
}

fn main() {
    let args = Args::parse();
    let mut errors = 0u32;
    let mut warnings = 0u32;

    // --- Parse all registry TOML files (NA-003) ---
    if !args.dir.exists() {
        eprintln!("ERROR: registry directory {} not found", args.dir.display());
        errors += 1;
    } else {
        let (parsed_files, parse_errors) = parse_all_registry_toml_files(&args.dir);
        if parsed_files == 0 {
            eprintln!("ERROR: no TOML files found under {}", args.dir.display());
            errors += 1;
        } else {
            println!("Registry TOML parse sweep: {parsed_files} files parsed");
        }

        for err in parse_errors {
            eprintln!("ERROR: TOML parse failure: {err}");
            errors += 1;
        }

        match args.typed_policy {
            TypedPolicy::Off => {
                println!(
                    "Typed schema policy: {} (skipping typed schema drift checks)",
                    args.typed_policy.as_str()
                );
            }
            TypedPolicy::Warn | TypedPolicy::Error => {
                let typed_drift = [
                    parse_typed_registry_file::<MonographRegistry>(
                        &args.dir.join("monograph.toml"),
                        "data_core::registry::MonographRegistry",
                    ),
                    parse_typed_registry_file::<LacunaeRegistry>(
                        &args.dir.join("lacunae.toml"),
                        "data_core::registry::LacunaeRegistry",
                    ),
                    parse_typed_registry_file::<ArtifactRegistry>(
                        &args.dir.join("data_artifact_narratives.toml"),
                        "data_core::registry::ArtifactRegistry",
                    ),
                ]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();

                if typed_drift.is_empty() {
                    println!(
                        "Typed schema policy: {} (no typed schema drift detected)",
                        args.typed_policy.as_str()
                    );
                } else {
                    for typed_error in typed_drift {
                        match args.typed_policy {
                            TypedPolicy::Warn => {
                                eprintln!(
                                    "WARNING: typed schema drift (policy={}): {typed_error}",
                                    args.typed_policy.as_str()
                                );
                                warnings += 1;
                            }
                            TypedPolicy::Error => {
                                eprintln!(
                                    "ERROR: typed schema drift (policy={}): {typed_error}",
                                    args.typed_policy.as_str()
                                );
                                errors += 1;
                            }
                            TypedPolicy::Off => {}
                        }
                    }
                }
            }
        }
    }

    let mut identity_gap_policy = LoadedIdentityGapPolicy::default();
    match args.identity_gap_policy {
        IdentityGapPolicy::Off => {
            println!(
                "Identity gap policy: {} (skipping governed claim/insight gap checks)",
                args.identity_gap_policy.as_str()
            );
        }
        IdentityGapPolicy::Warn | IdentityGapPolicy::Error => {
            match load_identity_gap_policy(&args.identity_gap_policy_file) {
                Ok(policy) => {
                    println!(
                        "Identity gap policy: {} (loaded baseline from {})",
                        args.identity_gap_policy.as_str(),
                        args.identity_gap_policy_file.display()
                    );
                    identity_gap_policy = policy;
                }
                Err(err) => match args.identity_gap_policy {
                    IdentityGapPolicy::Warn => {
                        eprintln!(
                            "WARNING: identity gap policy load failure (policy={}): {err}; using empty baseline",
                            args.identity_gap_policy.as_str()
                        );
                        warnings += 1;
                    }
                    IdentityGapPolicy::Error => {
                        eprintln!(
                            "ERROR: identity gap policy load failure (policy={}): {err}; using empty baseline",
                            args.identity_gap_policy.as_str()
                        );
                        errors += 1;
                    }
                    IdentityGapPolicy::Off => {}
                },
            }
        }
    }

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
            eprintln!(
                "ERROR: Duplicate claim IDs detected ({} unique vs {} total)",
                claim_ids.len(),
                claim_count
            );
            errors += 1;
        }

        let mut claim_nums = BTreeSet::new();
        for claim in &registry.claim {
            if let Some(num) = parse_numbered_id(&claim.id, "C-") {
                claim_nums.insert(num);
            } else {
                eprintln!("ERROR: claim ID {} does not match C-NNN format", claim.id);
                errors += 1;
            }
        }

        if args.identity_gap_policy != IdentityGapPolicy::Off {
            let claim_missing = collect_missing_ids(&claim_nums);
            evaluate_identity_gap_lane(
                "claims",
                "C-",
                &claim_missing,
                &identity_gap_policy.claims,
                args.identity_gap_policy,
                &mut warnings,
                &mut errors,
            );
        }

        // Validate status tokens against canonical + legacy sets
        for claim in &registry.claim {
            if claim.status.is_empty() {
                eprintln!("ERROR: claim {} has empty status", claim.id);
                errors += 1;
            } else {
                // Check against canonical TOML statuses (from schema.rs)
                let is_canonical = TOML_CLAIM_STATUSES
                    .iter()
                    .any(|valid| claim.status == *valid);
                // Also accept legacy parenthetical variants
                let is_legacy = VALID_CLAIM_STATUSES_LEGACY
                    .iter()
                    .any(|valid| claim.status == *valid);
                // Also accept case variants (e.g., "verified" -> "Verified")
                let is_case_variant = {
                    let lower = claim.status.to_lowercase();
                    TOML_CLAIM_STATUSES
                        .iter()
                        .any(|valid| valid.to_lowercase() == lower)
                };
                if !is_canonical && !is_legacy && !is_case_variant {
                    eprintln!(
                        "WARNING: claim {} has unusual status: \"{}\"",
                        claim.id, claim.status
                    );
                    warnings += 1;
                }
            }
        }

        println!(
            "Claims: {} entries, {} unique IDs, range C-001..C-{:03}",
            claim_count,
            claim_ids.len(),
            claim_nums.iter().next_back().copied().unwrap_or(0)
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

        let mut nums = BTreeSet::new();
        for insight in &registry.insight {
            if let Some(num) = parse_numbered_id(&insight.id, "I-") {
                nums.insert(num);
            } else {
                eprintln!(
                    "ERROR: insight ID {} does not match I-NNN format",
                    insight.id
                );
                errors += 1;
            }
        }

        if args.identity_gap_policy != IdentityGapPolicy::Off {
            let insight_missing = collect_missing_ids(&nums);
            evaluate_identity_gap_lane(
                "insights",
                "I-",
                &insight_missing,
                &identity_gap_policy.insights,
                args.identity_gap_policy,
                &mut warnings,
                &mut errors,
            );
        }

        // Check cross-references and status
        for insight in &registry.insight {
            // Check both `claims` and `related_claims` fields
            let all_claim_refs: Vec<&String> = insight
                .claims
                .iter()
                .chain(insight.related_claims.iter())
                .collect();
            for claim_ref in all_claim_refs {
                if !claim_ids.contains(claim_ref) {
                    eprintln!(
                        "ERROR: insight {} references non-existent claim {}",
                        insight.id, claim_ref
                    );
                    errors += 1;
                }
            }
            if let Some(ref status) = insight.status
                && !VALID_INSIGHT_STATUSES.contains(&status.as_str())
            {
                eprintln!(
                    "WARNING: insight {} has unusual status: \"{}\"",
                    insight.id, status
                );
                warnings += 1;
            }
        }

        println!(
            "Insights: {} entries, IDs {:?}..{:?}",
            insight_count,
            nums.iter()
                .next()
                .map(|n| format!("I-{n:03}"))
                .unwrap_or_default(),
            nums.iter()
                .next_back()
                .map(|n| format!("I-{n:03}"))
                .unwrap_or_default(),
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
    let experiment_count: usize;
    let complete_experiment_count: usize;
    let binary_names_from_experiments: HashSet<String>;

    if experiments_path.exists() {
        let content = std::fs::read_to_string(&experiments_path).unwrap();
        let registry: ExperimentsRegistry = toml::from_str(&content).unwrap();

        experiment_ids = registry.experiment.iter().map(|e| e.id.clone()).collect();
        experiment_count = registry.experiment.len();
        complete_experiment_count = registry
            .experiment
            .iter()
            .filter(|entry| {
                let status = entry.status.as_deref().unwrap_or("").trim().to_lowercase();
                status == "complete" || status == "completed" || status == "done"
            })
            .count();
        binary_names_from_experiments = registry
            .experiment
            .iter()
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

        println!("Experiments: {} entries", experiment_count);
    } else {
        eprintln!("WARNING: {} not found", experiments_path.display());
        warnings += 1;
        experiment_ids = HashSet::new();
        experiment_count = 0;
        complete_experiment_count = 0;
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
            if let Some(ref exp_ref) = bin.experiment
                && !experiment_ids.contains(exp_ref)
            {
                eprintln!(
                    "ERROR: binary {} references non-existent experiment {}",
                    bin.name, exp_ref
                );
                errors += 1;
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
            match load_cargo_bin_names(&args.cargo_toml) {
                Ok(cargo_binary_names) => {
                    let mut missing_from_registry: Vec<String> = cargo_binary_names
                        .difference(&registry_binary_names)
                        .cloned()
                        .collect();
                    let mut missing_from_cargo: Vec<String> = registry_binary_names
                        .difference(&cargo_binary_names)
                        .cloned()
                        .collect();
                    missing_from_registry.sort();
                    missing_from_cargo.sort();

                    if !missing_from_registry.is_empty() {
                        eprintln!(
                            "ERROR: {} binaries in Cargo.toml but not in binaries.toml: {}",
                            missing_from_registry.len(),
                            missing_from_registry.join(", ")
                        );
                        errors += 1;
                    }
                    if !missing_from_cargo.is_empty() {
                        eprintln!(
                            "ERROR: {} binaries in binaries.toml but not in Cargo.toml: {}",
                            missing_from_cargo.len(),
                            missing_from_cargo.join(", ")
                        );
                        errors += 1;
                    }
                }
                Err(err) => {
                    eprintln!("ERROR: {err}");
                    errors += 1;
                }
            }
        } else {
            eprintln!("ERROR: {} not found", args.cargo_toml.display());
            errors += 1;
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
        if let Some(expected_claims) = project.project.claim_count
            && expected_claims as usize != claim_count
        {
            eprintln!(
                "ERROR: project.toml claim_count={} but claims.toml has {} entries",
                expected_claims, claim_count
            );
            errors += 1;
        }
        if let Some(expected_insights) = project.project.insight_count
            && expected_insights as usize != insight_count
        {
            eprintln!(
                "ERROR: project.toml insight_count={} but insights.toml has {} entries",
                expected_insights, insight_count
            );
            errors += 1;
        }
        if let Some(expected_experiments) = project.project.experiment_count
            && expected_experiments as usize != experiment_count
        {
            eprintln!(
                "ERROR: project.toml experiment_count={} but experiments.toml has {} entries",
                expected_experiments, experiment_count
            );
            errors += 1;
        }
        if let Some(expected_complete) = project.project.complete_experiment_count
            && expected_complete as usize != complete_experiment_count
        {
            eprintln!(
                "ERROR: project.toml complete_experiment_count={} but experiments.toml has {} complete/done entries",
                expected_complete, complete_experiment_count
            );
            errors += 1;
        }
        if let Some(expected_binaries) = project.project.binary_count
            && expected_binaries as usize != registry_binary_names.len()
        {
            eprintln!(
                "ERROR: project.toml binary_count={} but binaries.toml has {} entries",
                expected_binaries,
                registry_binary_names.len()
            );
            errors += 1;
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
