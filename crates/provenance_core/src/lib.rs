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

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DownloadJobRecord {
    pub id: Option<i64>,
    pub requested_url: String,
    pub transfer_kind: String,
    pub requested_backend: String,
    pub route_scheme: String,
    pub route_host: Option<String>,
    pub route_backends: Vec<String>,
    pub note: Option<String>,
    pub status: String,
    pub final_url: Option<String>,
    pub output_path: Option<String>,
    pub created_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DownloadAttemptRecord {
    pub id: Option<i64>,
    pub job_id: Option<i64>,
    pub backend: String,
    pub succeeded: bool,
    pub failure_class: Option<String>,
    pub http_code: Option<i64>,
    pub content_type: Option<String>,
    pub bytes: i64,
    pub sha256: Option<String>,
    pub is_pdf: bool,
    pub final_url: Option<String>,
    pub note: String,
    pub error_message: Option<String>,
    pub recorded_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DownloadQueryResult {
    pub job: DownloadJobRecord,
    pub attempts: Vec<DownloadAttemptRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DownloadCampaignRecord {
    pub id: Option<i64>,
    pub name: String,
    pub command_kind: String,
    pub input_path: String,
    pub out_ledger_path: Option<String>,
    pub dest_dir: Option<String>,
    pub note: Option<String>,
    pub created_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DownloadCampaignQueryResult {
    pub campaign: DownloadCampaignRecord,
    pub job_count: usize,
    pub success_count: usize,
    pub failure_count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LiteratureVerificationRunRecord {
    pub id: Option<i64>,
    pub input_path: String,
    pub topic: Option<String>,
    pub hypotheses_path: Option<String>,
    pub domains: Vec<String>,
    pub search_queries: Vec<String>,
    pub total_entries: usize,
    pub verified_count: usize,
    pub suspicious_count: usize,
    pub hallucinated_count: usize,
    pub skipped_count: usize,
    pub integrity_score: f64,
    pub novelty_score: Option<f64>,
    pub novelty_assessment: Option<String>,
    pub recommendation: Option<String>,
    pub search_coverage: Option<String>,
    pub total_papers_retrieved: Option<usize>,
    pub created_at: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LiteratureVerificationResultRecord {
    pub id: Option<i64>,
    pub run_id: Option<i64>,
    pub cite_key: String,
    pub title: String,
    pub status: String,
    pub confidence: f64,
    pub method: String,
    pub details: String,
    pub doi: Option<String>,
    pub arxiv_id: Option<String>,
    pub matched_paper_title: Option<String>,
    pub matched_paper_source: Option<String>,
    pub matched_paper_year: Option<i64>,
    pub matched_paper_url: Option<String>,
    pub relevance_score: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LiteratureNoveltySimilarPaperRecord {
    pub id: Option<i64>,
    pub run_id: Option<i64>,
    pub title: String,
    pub paper_id: String,
    pub year: i64,
    pub venue: String,
    pub citation_count: i64,
    pub similarity: f64,
    pub url: String,
    pub cite_key: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LiteratureVerificationQueryResult {
    pub run: LiteratureVerificationRunRecord,
    pub results: Vec<LiteratureVerificationResultRecord>,
    pub similar_papers: Vec<LiteratureNoveltySimilarPaperRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CountSummary {
    pub key: String,
    pub count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct BackendHealthSummary {
    pub backend: String,
    pub success_count: usize,
    pub failure_count: usize,
    pub total_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DownloadLedgerProjectionRow {
    pub id: String,
    pub url: String,
    pub http_code: String,
    pub content_type: String,
    pub bytes: u64,
    pub sha256: String,
    pub is_pdf: String,
    pub note: String,
    pub status: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IndexStats {
    pub indexed_at: String,
    pub artifact_count: usize,
    pub document_count: usize,
    pub lane_assignment_count: usize,
    pub mirror_observation_count: usize,
    pub claim_count: usize,
    pub insight_count: usize,
    pub experiment_count: usize,
    pub binary_count: usize,
    pub theorem_count: usize,
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
    pub download_job_count: usize,
    pub download_attempt_count: usize,
    pub top_failed_download_hosts: Vec<CountSummary>,
    pub top_active_download_hosts: Vec<CountSummary>,
    pub backend_health: Vec<BackendHealthSummary>,
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

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ClaimRecord {
    pub id: String,
    pub statement: String,
    pub status: String,
    pub where_stated: String,
    pub last_verified: String,
    pub formal_proof: Option<String>,
    pub status_note: Option<String>,
    pub compat_toml_text: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct InsightRecord {
    pub id: String,
    pub title: String,
    pub status: String,
    pub claim_refs: Vec<String>,
    pub status_note: Option<String>,
    pub compat_toml_text: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExperimentRecord {
    pub id: String,
    pub title: String,
    pub status: String,
    pub binary: Option<String>,
    pub claim_refs: Vec<String>,
    pub status_note: Option<String>,
    pub compat_toml_text: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct BinaryRecord {
    pub name: String,
    pub crate_name: String,
    pub description: String,
    pub experiment: Option<String>,
    pub source: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct TheoremRecord {
    pub id: String,
    pub title: String,
    pub proof_path: Utf8PathBuf,
    pub status: String,
    pub linked_claim_ids: Vec<String>,
    pub source: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ControlPlaneCounts {
    pub claim_count: usize,
    pub insight_count: usize,
    pub experiment_count: usize,
    pub complete_experiment_count: usize,
    pub binary_count: usize,
    pub theorem_count: usize,
    pub kernel_checked_claim_count: usize,
    pub proof_file_count: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExternalSourceContractsMeta {
    pub updated: String,
    pub authoritative: bool,
    pub policy_version: String,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExternalSourceContractRecord {
    pub id: String,
    pub path_glob: String,
    pub canonical_url: String,
    pub mirror_urls: Vec<String>,
    pub access_class: String,
    pub status: String,
    pub retrieval_method: String,
    pub attempt_deadline_utc: String,
    pub resolution_deadline_utc: String,
    pub blocker_note: String,
    pub evidence_refs: Vec<String>,
    pub manual_manifest_refs: Vec<String>,
    pub blocked_action_plan: Vec<String>,
    pub scientific_validator_refs: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExternalSourceDossiersMeta {
    pub updated: String,
    pub authoritative: bool,
    pub source_markdown_glob: String,
    pub document_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn artifact_status_roundtrip() {
        let variants = [
            "downloaded",
            "downloadable",
            "blocked",
            "citation_only_no_link",
            "unverified",
        ];
        for s in variants {
            let parsed = ArtifactStatus::parse(s).expect("parse should succeed");
            assert_eq!(parsed.as_str(), s);
        }
    }

    #[test]
    fn artifact_status_parse_invalid_returns_none() {
        assert!(ArtifactStatus::parse("nonexistent").is_none());
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExternalSourceDossierRecord {
    pub id: String,
    pub source_markdown: String,
    pub slug: String,
    pub title: String,
    pub status_token: String,
    pub content_kind: String,
    pub authority_level: String,
    pub verification_level: String,
    pub operational_role: String,
    pub source_lineage_summary: String,
    pub truth_surfaces: Vec<String>,
    pub artifact_contract_paths: Vec<String>,
    pub has_full_transcript: bool,
    pub claim_refs: Vec<String>,
    pub url_refs: Vec<String>,
    pub path_refs: Vec<String>,
    pub line_count: usize,
    pub notes: String,
    pub body_markdown: String,
}
