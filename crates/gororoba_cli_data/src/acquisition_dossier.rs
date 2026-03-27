use anyhow::{Context, Result};
use lit_search::MultiQueryExecutionReport;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct ResearchDossier {
    pub schema_version: u32,
    pub generated_at_utc: String,
    pub project_id: String,
    pub project_api_root: String,
    pub search_queue_path: String,
    pub search_target_id: String,
    pub crosswalk_id: String,
    pub title: String,
    pub window: String,
    pub priority: String,
    pub status: String,
    pub kind: String,
    pub why_now: String,
    pub query_seeds: Vec<String>,
    pub requested_sources: Vec<String>,
    pub preferred_source_families: Vec<String>,
    pub limit_per_query: usize,
    pub year_min: u32,
    pub min_relevance: i64,
    pub report: MultiQueryExecutionReport,
    pub top_hits: Vec<DossierHit>,
    pub stage_suggestions: Vec<StageSuggestion>,
    pub batch_manifest_rel: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DossierHit {
    pub rank: usize,
    pub canonical_id: String,
    pub relevance_score: i64,
    pub source_family: String,
    pub route_class: String,
    pub host_class: String,
    pub title: String,
    pub year: u32,
    pub source: String,
    pub venue: String,
    pub citation_count: u32,
    pub doi: String,
    pub url: String,
    pub pdf_url: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct StageSuggestion {
    pub rank: usize,
    pub suggestion_id: String,
    pub canonical_id: String,
    pub relevance_score: i64,
    pub source: String,
    pub source_family: String,
    pub route_class: String,
    pub host_class: String,
    pub paper_title: String,
    pub action: String,
    pub candidate_url: String,
    pub command: String,
    pub rationale: String,
    pub default_selected: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DossierBatchManifest {
    pub schema_version: u32,
    pub generated_at_utc: String,
    pub project_id: String,
    pub project_api_root: String,
    pub search_queue_path: String,
    pub output_dir: String,
    pub requested_sources: Vec<String>,
    pub preferred_source_families: Vec<String>,
    pub year_min: u32,
    pub limit_per_query: usize,
    pub min_relevance: i64,
    pub windows: Vec<String>,
    pub priorities: Vec<String>,
    pub statuses: Vec<String>,
    pub critical_only: bool,
    pub entries: Vec<DossierBatchEntry>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct DossierBatchEntry {
    pub search_target_id: String,
    pub title: String,
    pub window: String,
    pub priority: String,
    pub kind: String,
    pub dossier_json: String,
    pub dossier_markdown: String,
    pub suggestion_count: usize,
}

pub fn load_research_dossier(path: &Path) -> Result<ResearchDossier> {
    let body = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&body).with_context(|| format!("parse {}", path.display()))
}

pub fn write_research_dossier(path: &Path, dossier: &ResearchDossier) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let body = serde_json::to_string_pretty(dossier)?;
    fs::write(path, format!("{body}\n")).with_context(|| format!("write {}", path.display()))
}

pub fn load_dossier_batch_manifest(path: &Path) -> Result<DossierBatchManifest> {
    let body = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&body).with_context(|| format!("parse {}", path.display()))
}

pub fn write_dossier_batch_manifest(path: &Path, manifest: &DossierBatchManifest) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let body = toml::to_string_pretty(manifest)?;
    fs::write(path, format!("{body}\n")).with_context(|| format!("write {}", path.display()))
}

pub fn resolve_manifest_entry_path(manifest_path: &Path, relative_or_absolute: &str) -> PathBuf {
    let candidate = PathBuf::from(relative_or_absolute);
    if candidate.is_absolute() {
        candidate
    } else {
        manifest_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(candidate)
    }
}
