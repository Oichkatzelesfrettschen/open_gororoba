use camino::Utf8PathBuf;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum ArtifactStatus {
    Downloaded,
    Downloadable,
    Blocked,
    CitationOnlyNoLink,
    Unverified,
}

impl ArtifactStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Downloaded => "downloaded",
            Self::Downloadable => "downloadable",
            Self::Blocked => "blocked",
            Self::CitationOnlyNoLink => "citation_only_no_link",
            Self::Unverified => "unverified",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value.trim() {
            "downloaded" => Some(Self::Downloaded),
            "downloadable" => Some(Self::Downloadable),
            "blocked" => Some(Self::Blocked),
            "citation_only_no_link" => Some(Self::CitationOnlyNoLink),
            "unverified" => Some(Self::Unverified),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum MirrorKind {
    Working,
    WorkingPdf,
    Nonworking,
    Unverified,
}

impl MirrorKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Working => "working",
            Self::WorkingPdf => "working_pdf",
            Self::Nonworking => "nonworking",
            Self::Unverified => "unverified",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactRecord {
    pub id: String,
    pub key: String,
    pub title: String,
    pub citation: String,
    pub status: ArtifactStatus,
    pub minimum_requirement_met: bool,
    pub canonical_functional_url: Option<String>,
    pub canonical_download_path: Option<Utf8PathBuf>,
    pub source_refs: Vec<String>,
    pub all_links: Vec<String>,
    pub downloaded_paths: Vec<Utf8PathBuf>,
    pub doi_list: Vec<String>,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocumentRecord {
    pub id: String,
    pub path: Utf8PathBuf,
    pub title: String,
    pub kind: String,
    pub authoring_mode: String,
    pub generated: bool,
    pub status: String,
    pub toml_backing: Option<Utf8PathBuf>,
    pub sha256: Option<String>,
    pub size_bytes: Option<i64>,
    pub line_count: Option<i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LaneAssignment {
    pub artifact_id: String,
    pub lane_name: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MirrorObservationRecord {
    pub artifact_id: String,
    pub url: String,
    pub mirror_kind: MirrorKind,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ArtifactQueryResult {
    pub artifact: ArtifactRecord,
    pub lanes: Vec<String>,
    pub mirror_observations: Vec<MirrorObservationRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DocumentQueryResult {
    pub document: DocumentRecord,
    pub source_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IndexStats {
    pub indexed_at: String,
    pub artifact_count: usize,
    pub document_count: usize,
    pub lane_assignment_count: usize,
    pub mirror_observation_count: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DoctorReport {
    pub generated_at: String,
    pub artifact_count: usize,
    pub document_count: usize,
    pub missing_minimum_count: usize,
    pub blocked_count: usize,
    pub unverified_count: usize,
    pub citation_only_count: usize,
    pub missing_lane_assignment_count: usize,
    pub documents_without_backing_count: usize,
    pub last_indexed_at: Option<String>,
    pub last_exported_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PantheonSeedSummary {
    pub db_path: String,
    pub findings_count: usize,
    pub risk_count: usize,
    pub overflow_task_count: usize,
    pub max_active_overflow: usize,
}
