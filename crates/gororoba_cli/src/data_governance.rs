use anyhow::{Context, Result};
use chrono::{DateTime, NaiveDate, Utc};
use csv::StringRecord;
use glob::Pattern;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use walkdir::WalkDir;

pub const DEFAULT_GOVERNANCE_PATH: &str = "registry/data_governance.toml";
pub const DEFAULT_EXPERIMENTS_PATH: &str = "registry/experiments.toml";
pub const DEFAULT_EXTERNAL_PROVENANCE_PATH: &str = "data/external/PROVENANCE.local.json";
pub const DEFAULT_EXTERNAL_SOURCES_PATH: &str = "data/external/SOURCES.toml";
pub const DEFAULT_ARTIFACTS_MANIFEST_PATH: &str = "data/artifacts/ARTIFACTS_MANIFEST.csv";
pub const DEFAULT_EVIDENCE_MANIFEST_PATH: &str = "data/evidence/MANIFEST.toml";
pub const DEFAULT_SEMANTIC_VALIDATORS_PATH: &str = "registry/data_semantic_validators.toml";
pub const DEFAULT_GENERATED_PATTERNS_PATH: &str = "registry/data_generated_patterns.toml";

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DataGovernanceRoot {
    #[serde(default)]
    pub data_governance: DataGovernanceHeader,
    #[serde(default)]
    pub lane: Vec<LanePolicy>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct DataGovernanceHeader {
    #[serde(default)]
    pub updated: String,
    #[serde(default)]
    pub authoritative: bool,
    #[serde(default)]
    pub default_gitignore_required: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LanePolicy {
    pub id: String,
    pub root: String,
    #[serde(rename = "class")]
    pub lane_class: String,
    #[serde(default)]
    pub strict: bool,
    #[serde(default)]
    pub cleanable: bool,
    #[serde(default)]
    pub gitignore_required: bool,
    #[serde(default)]
    pub naming_policy: String,
    #[serde(default)]
    pub required_manifests: Vec<String>,
    #[serde(default)]
    pub gitignore_allowlist: Vec<String>,
    #[serde(default)]
    pub forbidden_globs: Vec<String>,
}

impl LanePolicy {
    pub fn contains_path(&self, path: &str) -> bool {
        path == self.root || path.starts_with(&(self.root.clone() + "/"))
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExternalProvenanceRoot {
    #[serde(default)]
    pub hashes: Vec<ExternalHashEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExternalHashEntry {
    pub path: String,
    #[serde(default)]
    pub sha256: String,
    #[serde(default)]
    pub size_bytes: u64,
    #[serde(default)]
    pub mtime_utc: String,
    #[serde(default)]
    pub source_id: String,
    #[serde(default)]
    pub source_status: String,
    #[serde(default)]
    pub source_access_class: String,
    #[serde(default)]
    pub source_canonical_url: String,
    #[serde(default)]
    pub source_retrieval_method: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ExternalSourcesRoot {
    #[serde(default)]
    pub external_sources: ExternalSourcesHeader,
    #[serde(default)]
    pub source: Vec<ExternalSourceRule>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ExternalSourcesHeader {
    #[serde(default)]
    pub updated: String,
    #[serde(default)]
    pub authoritative: bool,
    #[serde(default)]
    pub policy_version: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExternalSourceRule {
    pub id: String,
    pub path_glob: String,
    pub canonical_url: String,
    #[serde(default)]
    pub mirror_urls: Vec<String>,
    #[serde(default)]
    pub access_class: String,
    #[serde(default)]
    pub status: String,
    #[serde(default)]
    pub retrieval_method: String,
    #[serde(default)]
    pub attempt_deadline_utc: String,
    #[serde(default)]
    pub resolution_deadline_utc: String,
    #[serde(default)]
    pub blocker_note: String,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
    #[serde(default)]
    pub manual_manifest_refs: Vec<String>,
    #[serde(default)]
    pub blocked_action_plan: Vec<String>,
    #[serde(default)]
    pub scientific_validator_refs: Vec<String>,
}

impl ExternalSourceRule {
    pub fn is_blocked(&self) -> bool {
        self.status.eq_ignore_ascii_case("blocked")
    }

    pub fn is_active(&self) -> bool {
        self.status.eq_ignore_ascii_case("active")
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct SemanticValidatorsRoot {
    #[serde(default)]
    pub semantic_validators: SemanticValidatorsHeader,
    #[serde(default)]
    pub validator: Vec<SemanticValidatorEntry>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct GeneratedOriginPatternsRoot {
    #[serde(default)]
    pub generated_origin_patterns: GeneratedOriginPatternsHeader,
    #[serde(default)]
    pub pattern: Vec<GeneratedOriginPatternEntry>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct GeneratedOriginPatternsHeader {
    #[serde(default)]
    pub updated: String,
    #[serde(default)]
    pub authoritative: bool,
    #[serde(default)]
    pub policy_version: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GeneratedOriginPatternEntry {
    pub id: String,
    pub path_glob: String,
    pub origin_ref: String,
    #[serde(default)]
    pub note: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct SemanticValidatorsHeader {
    #[serde(default)]
    pub updated: String,
    #[serde(default)]
    pub authoritative: bool,
    #[serde(default)]
    pub policy_version: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SemanticValidatorEntry {
    pub id: String,
    pub lane_id: String,
    pub name: String,
    pub validator_kind: String,
    #[serde(default = "default_true")]
    pub enabled: bool,
    #[serde(default = "default_hard_fail")]
    pub severity: String,
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct ArtifactManifestEntry {
    pub artifact_path: String,
    pub provenance_required: bool,
}

#[derive(Debug, Clone)]
pub struct EvidenceManifestEntry {
    pub path: String,
    pub sha256: String,
    pub status: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct FileOriginRecord {
    pub path: String,
    pub lane_id: String,
    pub origin_kind: String,
    pub origin_ref: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AuditReport {
    pub total_files: usize,
    pub origin_counts: BTreeMap<String, usize>,
    pub lane_counts: BTreeMap<String, usize>,
    pub strict_unknown_count: usize,
    pub unknown_paths: Vec<String>,
    pub naming_violations: Vec<String>,
    pub records: Vec<FileOriginRecord>,
}

pub struct AuditInputs {
    pub governance: DataGovernanceRoot,
    pub external_hashes: HashMap<String, ExternalHashEntry>,
    pub artifacts_manifest: HashMap<String, ArtifactManifestEntry>,
    pub evidence_manifest: HashMap<String, EvidenceManifestEntry>,
    pub experiment_output_patterns: Vec<String>,
    pub generated_origin_patterns: Vec<GeneratedOriginPatternEntry>,
    pub root: PathBuf,
}

pub fn load_data_governance(path: &Path) -> Result<DataGovernanceRoot> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("read governance registry {}", path.display()))?;
    let parsed: DataGovernanceRoot =
        toml::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
    if parsed.lane.is_empty() {
        anyhow::bail!("no [[lane]] entries in {}", path.display());
    }
    Ok(parsed)
}

pub fn load_external_hashes(path: &Path) -> Result<HashMap<String, ExternalHashEntry>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("read external provenance {}", path.display()))?;
    let parsed: ExternalProvenanceRoot =
        serde_json::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
    let mut out = HashMap::new();
    for entry in parsed.hashes {
        let rel = entry.path.trim_start_matches("./");
        let full = if rel.starts_with("data/external/") {
            rel.to_string()
        } else {
            format!("data/external/{rel}")
        };
        out.insert(full, entry);
    }
    Ok(out)
}

pub fn load_external_sources(path: &Path) -> Result<ExternalSourcesRoot> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let parsed: ExternalSourcesRoot =
        toml::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
    if parsed.source.is_empty() {
        anyhow::bail!("no [[source]] entries in {}", path.display());
    }
    Ok(parsed)
}

pub fn load_semantic_validators(path: &Path) -> Result<SemanticValidatorsRoot> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let parsed: SemanticValidatorsRoot =
        toml::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
    if parsed.validator.is_empty() {
        anyhow::bail!("no [[validator]] entries in {}", path.display());
    }
    Ok(parsed)
}

pub fn load_generated_origin_patterns(path: &Path) -> Result<Vec<GeneratedOriginPatternEntry>> {
    let raw = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let parsed: GeneratedOriginPatternsRoot =
        toml::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
    if parsed.pattern.is_empty() {
        anyhow::bail!("no [[pattern]] entries in {}", path.display());
    }
    Ok(parsed.pattern)
}

pub fn load_artifacts_manifest(path: &Path) -> Result<HashMap<String, ArtifactManifestEntry>> {
    let mut rdr = csv::Reader::from_path(path)
        .with_context(|| format!("read artifacts manifest {}", path.display()))?;
    let mut out = HashMap::new();
    for row in rdr.records() {
        let record = row.with_context(|| format!("parse CSV row in {}", path.display()))?;
        let entry = parse_artifact_row(&record, path)?;
        out.insert(entry.artifact_path.clone(), entry);
    }
    Ok(out)
}

pub fn artifacts_manifest_matches(
    path: &str,
    manifest: &HashMap<String, ArtifactManifestEntry>,
) -> bool {
    if manifest.contains_key(path) {
        return true;
    }
    manifest
        .keys()
        .any(|candidate| artifact_manifest_pattern_matches(candidate, path))
}

fn artifact_manifest_pattern_matches(candidate: &str, path: &str) -> bool {
    if candidate == path {
        return true;
    }
    if candidate.ends_with('/') {
        return path.starts_with(candidate);
    }
    if candidate.contains('*') || candidate.contains('?') || candidate.contains('[') {
        return Pattern::new(candidate).is_ok_and(|pattern| pattern.matches(path));
    }
    false
}

fn parse_artifact_row(
    record: &StringRecord,
    manifest_path: &Path,
) -> Result<ArtifactManifestEntry> {
    let artifact_path = record
        .get(0)
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .ok_or_else(|| anyhow::anyhow!("missing artifact_path in {}", manifest_path.display()))?;
    let raw_prov = record.get(3).map(str::trim).unwrap_or("0");
    let provenance_required = matches!(raw_prov, "1" | "true" | "True" | "TRUE");
    Ok(ArtifactManifestEntry {
        artifact_path: artifact_path.to_string(),
        provenance_required,
    })
}

pub fn load_evidence_manifest(path: &Path) -> Result<HashMap<String, EvidenceManifestEntry>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("read evidence manifest {}", path.display()))?;
    let value: toml::Value =
        toml::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
    let mut out = HashMap::new();
    if let Some(artifacts) = value.get("artifact").and_then(toml::Value::as_array) {
        for item in artifacts {
            let rel = item
                .get("path")
                .and_then(toml::Value::as_str)
                .unwrap_or("")
                .trim();
            if rel.is_empty() {
                continue;
            }
            let full_path = format!("data/evidence/{rel}");
            out.insert(
                full_path.clone(),
                EvidenceManifestEntry {
                    path: full_path,
                    sha256: item
                        .get("sha256")
                        .and_then(toml::Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                    status: item
                        .get("status")
                        .and_then(toml::Value::as_str)
                        .unwrap_or("unknown")
                        .to_string(),
                },
            );
        }
    }
    Ok(out)
}

pub fn load_experiment_output_patterns(path: &Path) -> Result<Vec<String>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("read experiments registry {}", path.display()))?;
    let value: toml::Value =
        toml::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
    let mut out = BTreeSet::new();
    if let Some(items) = value.get("experiment").and_then(toml::Value::as_array) {
        for item in items {
            for key in ["output_path_refs", "output"] {
                if let Some(values) = item.get(key).and_then(toml::Value::as_array) {
                    for v in values {
                        if let Some(s) = v.as_str() {
                            let normalized = s.trim().trim_matches('`').to_string();
                            if normalized.starts_with("data/") && !normalized.contains('<') {
                                out.insert(normalized);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(out.into_iter().collect())
}

pub fn audit_data(inputs: &AuditInputs) -> Result<AuditReport> {
    let mut records = Vec::new();
    let mut origin_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut lane_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut unknown_paths = Vec::new();
    let mut naming_violations = Vec::new();
    let mut strict_unknown_count = 0usize;
    let pattern_cache = compile_patterns(&inputs.experiment_output_patterns);
    let generated_pattern_cache =
        compile_generated_origin_patterns(&inputs.generated_origin_patterns);

    for entry in WalkDir::new(&inputs.root)
        .into_iter()
        .filter_map(std::result::Result::ok)
        .filter(|e| e.file_type().is_file())
    {
        let full = entry.path();
        let rel = to_repo_rel(full)?;
        let lane = lane_for_path(&inputs.governance.lane, &rel);
        let lane_id = lane
            .map(|l| l.id.clone())
            .unwrap_or_else(|| "unclassified".to_string());
        *lane_counts.entry(lane_id.clone()).or_insert(0) += 1;

        if let Some(l) = lane
            && needs_naming_check(l)
            && has_naming_violation(&rel)
            && !is_allowlisted(l, &rel)
        {
            naming_violations.push(rel.clone());
        }

        let (origin_kind, origin_ref) =
            classify_origin(&rel, lane, inputs, &pattern_cache, &generated_pattern_cache);
        *origin_counts.entry(origin_kind.clone()).or_insert(0) += 1;
        if origin_kind == "unknown_origin" {
            unknown_paths.push(rel.clone());
            if lane.is_some_and(|l| l.strict) {
                strict_unknown_count += 1;
            }
        }
        records.push(FileOriginRecord {
            path: rel,
            lane_id,
            origin_kind,
            origin_ref,
        });
    }

    Ok(AuditReport {
        total_files: records.len(),
        origin_counts,
        lane_counts,
        strict_unknown_count,
        unknown_paths,
        naming_violations,
        records,
    })
}

fn classify_origin(
    rel: &str,
    lane: Option<&LanePolicy>,
    inputs: &AuditInputs,
    pattern_cache: &[Pattern],
    generated_pattern_cache: &[CompiledGeneratedOriginPattern],
) -> (String, String) {
    if rel.contains("/legacy/") {
        return (
            "legacy_frozen".to_string(),
            "data/csv/legacy historical lane".to_string(),
        );
    }
    if rel.starts_with("data/external/") {
        if rel == DEFAULT_EXTERNAL_PROVENANCE_PATH || rel == DEFAULT_EXTERNAL_SOURCES_PATH {
            return ("generated_reproducible".to_string(), rel.to_string());
        }
        if inputs.external_hashes.contains_key(rel) {
            return (
                "external_reproducible_fetch".to_string(),
                DEFAULT_EXTERNAL_PROVENANCE_PATH.to_string(),
            );
        }
        return (
            "unknown_origin".to_string(),
            "missing external hash".to_string(),
        );
    }
    if rel.starts_with("data/artifacts/") {
        if rel == DEFAULT_ARTIFACTS_MANIFEST_PATH || rel == "data/artifacts/README.md" {
            return (
                "generated_reproducible".to_string(),
                DEFAULT_ARTIFACTS_MANIFEST_PATH.to_string(),
            );
        }
        if artifacts_manifest_matches(rel, &inputs.artifacts_manifest) {
            return (
                "generated_reproducible".to_string(),
                DEFAULT_ARTIFACTS_MANIFEST_PATH.to_string(),
            );
        }
        if rel.ends_with(".PROVENANCE.json") {
            let target = rel.trim_end_matches(".PROVENANCE.json");
            if artifacts_manifest_matches(target, &inputs.artifacts_manifest) {
                return (
                    "generated_reproducible".to_string(),
                    DEFAULT_ARTIFACTS_MANIFEST_PATH.to_string(),
                );
            }
        }
        if rel == "data/artifacts/PROVENANCE.local.json" {
            return (
                "generated_reproducible".to_string(),
                "data/artifacts/PROVENANCE.local.json".to_string(),
            );
        }
        return (
            "unknown_origin".to_string(),
            "missing artifacts manifest row".to_string(),
        );
    }
    if rel.starts_with("data/evidence/") {
        if rel == DEFAULT_EVIDENCE_MANIFEST_PATH {
            return (
                "generated_reproducible".to_string(),
                DEFAULT_EVIDENCE_MANIFEST_PATH.to_string(),
            );
        }
        if inputs.evidence_manifest.contains_key(rel) {
            return (
                "generated_reproducible".to_string(),
                DEFAULT_EVIDENCE_MANIFEST_PATH.to_string(),
            );
        }
        return (
            "unknown_origin".to_string(),
            "missing evidence manifest row".to_string(),
        );
    }
    if let Some(l) = lane
        && l.lane_class == "manual_import_exception"
    {
        return (
            "legacy_frozen".to_string(),
            format!("lane {} manual import exception", l.id),
        );
    }
    if let Some(origin_ref) = generated_origin_ref(rel, generated_pattern_cache) {
        return ("generated_reproducible".to_string(), origin_ref.to_string());
    }
    if matches_experiment_pattern(rel, pattern_cache) {
        return (
            "generated_reproducible".to_string(),
            DEFAULT_EXPERIMENTS_PATH.to_string(),
        );
    }
    if let Some(l) = lane {
        return (
            "unknown_origin".to_string(),
            format!("lane {} has no matching origin contract", l.id),
        );
    }
    (
        "unknown_origin".to_string(),
        "path not covered by lane contract".to_string(),
    )
}

#[derive(Debug, Clone)]
struct CompiledGeneratedOriginPattern {
    matcher: Pattern,
    origin_ref: String,
}

fn compile_generated_origin_patterns(
    entries: &[GeneratedOriginPatternEntry],
) -> Vec<CompiledGeneratedOriginPattern> {
    let mut out = Vec::new();
    for entry in entries {
        if let Ok(matcher) = Pattern::new(&entry.path_glob) {
            out.push(CompiledGeneratedOriginPattern {
                matcher,
                origin_ref: entry.origin_ref.clone(),
            });
        }
    }
    out
}

fn generated_origin_ref<'a>(
    path: &str,
    patterns: &'a [CompiledGeneratedOriginPattern],
) -> Option<&'a str> {
    for entry in patterns {
        if entry.matcher.matches(path) {
            return Some(entry.origin_ref.as_str());
        }
    }
    None
}

fn compile_patterns(patterns: &[String]) -> Vec<Pattern> {
    patterns
        .iter()
        .filter_map(|p| {
            if p.ends_with('/') {
                let mut prefix = p.clone();
                prefix.push('*');
                Pattern::new(&prefix).ok()
            } else if p.contains('*') || p.contains('?') || p.contains('[') {
                Pattern::new(p).ok()
            } else {
                let mut prefix = p.clone();
                prefix.push('*');
                Pattern::new(&(prefix + "")).ok()
            }
        })
        .collect()
}

fn matches_experiment_pattern(path: &str, patterns: &[Pattern]) -> bool {
    for pattern in patterns {
        if pattern.matches(path) {
            return true;
        }
    }
    false
}

fn lane_for_path<'a>(lanes: &'a [LanePolicy], path: &str) -> Option<&'a LanePolicy> {
    lanes
        .iter()
        .filter(|lane| lane.contains_path(path))
        .max_by_key(|lane| lane.root.len())
}

pub fn to_repo_rel(path: &Path) -> Result<String> {
    let cwd = std::env::current_dir().context("resolve current directory")?;
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    };
    let rel = absolute.strip_prefix(&cwd).with_context(|| {
        format!(
            "path {} not within repo root {}",
            absolute.display(),
            cwd.display()
        )
    })?;
    Ok(rel.to_string_lossy().replace('\\', "/"))
}

fn has_naming_violation(path: &str) -> bool {
    let file_name = path.rsplit('/').next().unwrap_or(path);
    file_name.contains(' ')
        || file_name.bytes().any(|b| b.is_ascii_uppercase())
        || file_name.contains(" 2.")
}

fn needs_naming_check(lane: &LanePolicy) -> bool {
    lane.naming_policy == "lower_snake"
}

fn is_allowlisted(lane: &LanePolicy, path: &str) -> bool {
    lane.gitignore_allowlist.iter().any(|p| p == path)
}

pub fn list_required_manifests(governance: &DataGovernanceRoot) -> BTreeMap<String, Vec<String>> {
    let mut out: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for lane in &governance.lane {
        out.insert(lane.id.clone(), lane.required_manifests.clone());
    }
    out
}

pub fn source_rule_for_path<'a>(
    path: &str,
    sources: &'a ExternalSourcesRoot,
) -> Option<&'a ExternalSourceRule> {
    let mut best: Option<(&ExternalSourceRule, usize)> = None;
    for rule in &sources.source {
        let pattern = match Pattern::new(&rule.path_glob) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if pattern.matches(path) {
            let score = glob_specificity(&rule.path_glob);
            match best {
                Some((_, best_score)) if best_score >= score => {}
                _ => best = Some((rule, score)),
            }
        }
    }
    best.map(|(rule, _)| rule)
}

pub fn external_source_contract_issues(sources: &ExternalSourcesRoot) -> Vec<String> {
    let mut issues = Vec::new();
    let mut seen = BTreeSet::new();
    for rule in &sources.source {
        if rule.id.trim().is_empty() {
            issues.push("source rule with empty id".to_string());
        }
        if !seen.insert(rule.id.clone()) {
            issues.push(format!("duplicate source id {}", rule.id));
        }
        if rule.path_glob.trim().is_empty() {
            issues.push(format!("source {} has empty path_glob", rule.id));
        } else if Pattern::new(&rule.path_glob).is_err() {
            issues.push(format!(
                "source {} has invalid glob {}",
                rule.id, rule.path_glob
            ));
        }
        if rule.canonical_url.trim().is_empty() {
            issues.push(format!("source {} has empty canonical_url", rule.id));
        }
        if rule.status.trim().is_empty() {
            issues.push(format!("source {} has empty status", rule.id));
        }
        if rule.access_class.trim().is_empty() {
            issues.push(format!("source {} has empty access_class", rule.id));
        }
        if rule.is_active() {
            if rule.scientific_validator_refs.is_empty() {
                issues.push(format!(
                    "source {} active without scientific_validator_refs",
                    rule.id
                ));
            }
            if rule
                .scientific_validator_refs
                .iter()
                .any(|v| v == "structure_only")
            {
                issues.push(format!(
                    "source {} active uses deprecated scientific validator ref structure_only",
                    rule.id
                ));
            }
        }
        if rule.is_blocked()
            && rule.access_class.eq_ignore_ascii_case("manual_or_licensed")
            && rule.manual_manifest_refs.is_empty()
        {
            issues.push(format!(
                "source {} blocked manual_or_licensed without manual_manifest_refs",
                rule.id
            ));
        }
        if rule.is_blocked() && rule.blocked_action_plan.is_empty() {
            issues.push(format!(
                "source {} blocked without blocked_action_plan",
                rule.id
            ));
        }
        for plan_ref in &rule.blocked_action_plan {
            if plan_ref.trim().is_empty() {
                issues.push(format!(
                    "source {} has empty blocked_action_plan ref",
                    rule.id
                ));
                continue;
            }
            if !reference_target_exists(plan_ref) {
                issues.push(format!(
                    "source {} references missing blocked_action_plan ref {}",
                    rule.id, plan_ref
                ));
            }
        }
    }
    issues
}

pub fn blocked_source_deadline_issues(
    sources: &ExternalSourcesRoot,
    now: DateTime<Utc>,
) -> Vec<String> {
    let mut issues = Vec::new();
    for rule in &sources.source {
        if !rule.is_blocked() {
            continue;
        }
        if rule.blocker_note.trim().is_empty() {
            issues.push(format!("source {} blocked without blocker_note", rule.id));
        }
        if rule.evidence_refs.is_empty() {
            issues.push(format!("source {} blocked without evidence_refs", rule.id));
        }
        if rule.blocked_action_plan.is_empty() {
            issues.push(format!(
                "source {} blocked without blocked_action_plan",
                rule.id
            ));
        } else {
            for plan_ref in &rule.blocked_action_plan {
                if plan_ref.trim().is_empty() {
                    issues.push(format!(
                        "source {} has empty blocked_action_plan ref",
                        rule.id
                    ));
                    continue;
                }
                if !reference_target_exists(plan_ref) {
                    issues.push(format!(
                        "source {} missing blocked_action_plan ref {}",
                        rule.id, plan_ref
                    ));
                }
            }
        }
        if rule.attempt_deadline_utc.trim().is_empty() {
            issues.push(format!(
                "source {} blocked without attempt_deadline_utc",
                rule.id
            ));
        } else if parse_deadline_utc(&rule.attempt_deadline_utc).is_none() {
            issues.push(format!(
                "source {} invalid attempt_deadline_utc {}",
                rule.id, rule.attempt_deadline_utc
            ));
        }
        if rule.resolution_deadline_utc.trim().is_empty() {
            issues.push(format!(
                "source {} blocked without resolution_deadline_utc",
                rule.id
            ));
            continue;
        }
        let Some(deadline) = parse_deadline_utc(&rule.resolution_deadline_utc) else {
            issues.push(format!(
                "source {} invalid resolution_deadline_utc {}",
                rule.id, rule.resolution_deadline_utc
            ));
            continue;
        };
        if now > deadline {
            issues.push(format!(
                "source {} blocked deadline overdue ({})",
                rule.id, rule.resolution_deadline_utc
            ));
        }
    }
    issues
}

pub fn missing_semantic_lane_validators(
    governance: &DataGovernanceRoot,
    validators: &SemanticValidatorsRoot,
) -> Vec<String> {
    let enabled_lane_ids: BTreeSet<String> = validators
        .validator
        .iter()
        .filter(|v| v.enabled)
        .map(|v| v.lane_id.clone())
        .collect();
    governance
        .lane
        .iter()
        .filter(|lane| !enabled_lane_ids.contains(&lane.id))
        .map(|lane| lane.id.clone())
        .collect()
}

pub fn validators_for_lane<'a>(
    lane_id: &str,
    validators: &'a SemanticValidatorsRoot,
) -> Vec<&'a SemanticValidatorEntry> {
    validators
        .validator
        .iter()
        .filter(|v| v.enabled && v.lane_id == lane_id)
        .collect()
}

pub fn git_ignored_paths(repo_root: &Path, paths: &[String]) -> Result<BTreeSet<String>> {
    if paths.is_empty() {
        return Ok(BTreeSet::new());
    }
    const CHECK_IGNORE_CHUNK: usize = 2_000;
    let mut ignored = BTreeSet::new();
    for chunk in paths.chunks(CHECK_IGNORE_CHUNK) {
        let chunk_ignored = git_ignored_paths_chunk(repo_root, chunk)?;
        ignored.extend(chunk_ignored);
    }
    Ok(ignored)
}

pub fn git_tracked_paths(repo_root: &Path, paths: &[String]) -> Result<BTreeSet<String>> {
    if paths.is_empty() {
        return Ok(BTreeSet::new());
    }
    const LS_FILES_CHUNK: usize = 2_000;
    let mut tracked = BTreeSet::new();
    for chunk in paths.chunks(LS_FILES_CHUNK) {
        let chunk_tracked = git_tracked_paths_chunk(repo_root, chunk)?;
        tracked.extend(chunk_tracked);
    }
    Ok(tracked)
}

fn git_ignored_paths_chunk(repo_root: &Path, paths: &[String]) -> Result<BTreeSet<String>> {
    let mut command = Command::new("git");
    command
        .arg("check-ignore")
        .arg("--stdin")
        .current_dir(repo_root)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().context("spawn git check-ignore")?;
    if let Some(mut stdin) = child.stdin.take() {
        for p in paths {
            writeln!(stdin, "{p}").context("write check-ignore stdin")?;
        }
    }
    let output = child
        .wait_with_output()
        .context("wait for git check-ignore")?;
    let mut ignored = BTreeSet::new();
    let stdout = String::from_utf8(output.stdout).context("decode git check-ignore stdout")?;
    for line in stdout.lines() {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            ignored.insert(trimmed.to_string());
        }
    }
    if !output.status.success() && ignored.is_empty() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("git check-ignore failed: {}", stderr.trim());
    }
    Ok(ignored)
}

fn git_tracked_paths_chunk(repo_root: &Path, paths: &[String]) -> Result<BTreeSet<String>> {
    let output = Command::new("git")
        .arg("ls-files")
        .arg("--")
        .args(paths)
        .current_dir(repo_root)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .context("run git ls-files")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!("git ls-files failed: {}", stderr.trim());
    }

    let mut tracked = BTreeSet::new();
    let stdout = String::from_utf8(output.stdout).context("decode git ls-files stdout")?;
    for line in stdout.lines() {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            tracked.insert(trimmed.to_string());
        }
    }
    Ok(tracked)
}

pub fn collect_files_under(root: &Path) -> Result<Vec<String>> {
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in WalkDir::new(root)
        .into_iter()
        .filter_map(std::result::Result::ok)
        .filter(|e| e.file_type().is_file())
    {
        out.push(to_repo_rel(entry.path())?);
    }
    out.sort();
    Ok(out)
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let digest = Sha256::digest(&bytes);
    Ok(format!("{digest:x}"))
}

fn default_true() -> bool {
    true
}

fn default_hard_fail() -> String {
    "hard_fail".to_string()
}

fn glob_specificity(glob: &str) -> usize {
    glob.chars()
        .filter(|c| !matches!(c, '*' | '?' | '[' | ']'))
        .count()
}

fn reference_target_exists(reference: &str) -> bool {
    let path_text = reference.split('#').next().unwrap_or(reference).trim();
    if path_text.is_empty() {
        return false;
    }
    Path::new(path_text).exists()
}

pub fn parse_deadline_utc(raw: &str) -> Option<DateTime<Utc>> {
    if let Ok(ts) = DateTime::parse_from_rfc3339(raw) {
        return Some(ts.with_timezone(&Utc));
    }
    if let Ok(day) = NaiveDate::parse_from_str(raw, "%Y-%m-%d") {
        let dt = day.and_hms_opt(23, 59, 59)?;
        return Some(DateTime::<Utc>::from_naive_utc_and_offset(dt, Utc));
    }
    None
}
