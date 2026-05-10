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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn research_dossier_json_round_trip() {
        let dossier = ResearchDossier {
            schema_version: 3,
            project_id: "test-project".to_string(),
            search_target_id: "tgt-alpha".to_string(),
            title: "Test Dossier".to_string(),
            limit_per_query: 10,
            year_min: 2020,
            min_relevance: 50,
            top_hits: vec![DossierHit {
                rank: 1,
                canonical_id: "hit-1".to_string(),
                relevance_score: 95,
                title: "A Paper".to_string(),
                year: 2024,
                ..DossierHit::default()
            }],
            ..ResearchDossier::default()
        };
        let json = serde_json::to_string_pretty(&dossier).expect("serialize");
        let parsed: ResearchDossier = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(parsed.schema_version, 3);
        assert_eq!(parsed.project_id, "test-project");
        assert_eq!(parsed.top_hits.len(), 1);
        assert_eq!(parsed.top_hits[0].relevance_score, 95);
    }

    #[test]
    fn dossier_batch_manifest_toml_round_trip() {
        let manifest = DossierBatchManifest {
            schema_version: 1,
            project_id: "batch-test".to_string(),
            year_min: 2015,
            limit_per_query: 5,
            min_relevance: 30,
            entries: vec![DossierBatchEntry {
                search_target_id: "tgt-a".to_string(),
                title: "Entry A".to_string(),
                suggestion_count: 3,
                ..DossierBatchEntry::default()
            }],
            ..DossierBatchManifest::default()
        };
        let toml_str = toml::to_string_pretty(&manifest).expect("serialize");
        let parsed: DossierBatchManifest = toml::from_str(&toml_str).expect("deserialize");
        assert_eq!(parsed.project_id, "batch-test");
        assert_eq!(parsed.entries.len(), 1);
        assert_eq!(parsed.entries[0].suggestion_count, 3);
    }

    #[test]
    fn dossier_file_round_trip_via_tempfile() {
        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        let path = tmp.path();

        let dossier = ResearchDossier {
            schema_version: 2,
            title: "Round-trip".to_string(),
            ..ResearchDossier::default()
        };
        write_research_dossier(path, &dossier).expect("write");
        let loaded = load_research_dossier(path).expect("load");
        assert_eq!(loaded.schema_version, 2);
        assert_eq!(loaded.title, "Round-trip");
    }

    #[test]
    fn batch_manifest_file_round_trip_via_tempfile() {
        let tmp = tempfile::NamedTempFile::new().expect("create temp file");
        let path = tmp.path();

        let manifest = DossierBatchManifest {
            schema_version: 1,
            project_id: "file-test".to_string(),
            ..DossierBatchManifest::default()
        };
        write_dossier_batch_manifest(path, &manifest).expect("write");
        let loaded = load_dossier_batch_manifest(path).expect("load");
        assert_eq!(loaded.project_id, "file-test");
    }

    #[test]
    fn resolve_manifest_entry_path_absolute() {
        let manifest = Path::new("/data/manifests/batch.toml");
        let result = resolve_manifest_entry_path(manifest, "/absolute/path.json");
        assert_eq!(result, PathBuf::from("/absolute/path.json"));
    }

    #[test]
    fn resolve_manifest_entry_path_relative() {
        let manifest = Path::new("/data/manifests/batch.toml");
        let result = resolve_manifest_entry_path(manifest, "entries/dossier.json");
        assert_eq!(
            result,
            PathBuf::from("/data/manifests/entries/dossier.json")
        );
    }
}
