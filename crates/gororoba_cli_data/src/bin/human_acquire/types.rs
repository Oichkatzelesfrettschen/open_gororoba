//! Type definitions for the `human-acquire` binary: clap Cli + Args,
//! session/source/probe/evidence data types, capability summaries, and
//! handoff/queue/promote/consume sub-types. ~750 lines of declarative
//! type definitions split out so the bin root focuses on dispatch +
//! per-command logic. Fields are pub(crate) so the bin root can
//! construct, match, and read across the module boundary.

use clap::{Parser, Subcommand, ValueEnum};
use data_core::download_stack::{DEFAULT_PROBE_BYTES, DownloadBackend, TransferResult};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "human-acquire",
    about = "Thin operator-guided acquisition controller for stubborn external sources"
)]
pub(crate) struct Cli {
    #[arg(long, default_value = ".")]
    pub(crate) repo_root: PathBuf,

    #[arg(long, default_value = "registry/download_host_policies.toml")]
    pub(crate) policy_registry: PathBuf,

    #[arg(long, default_value = "reports/acquisition_sessions")]
    pub(crate) sessions_dir: PathBuf,

    #[command(subcommand)]
    pub(crate) command: Commands,
}

#[derive(Subcommand, Debug)]
pub(crate) enum Commands {
    Stage(StageArgs),
    StageQueue(StageQueueArgs),
    Promote(PromoteArgs),
    ImportDossier(ImportDossierArgs),
    Record(RecordArgs),
    Show(ShowArgs),
    ConsumeHandoff(ConsumeHandoffArgs),
}

#[derive(Parser, Debug)]
pub(crate) struct StageArgs {
    #[arg(long)]
    pub(crate) url: String,

    #[arg(long)]
    pub(crate) source_id: Option<String>,

    #[arg(long)]
    pub(crate) title: Option<String>,

    #[arg(long)]
    pub(crate) site: Option<String>,

    #[arg(long)]
    pub(crate) access_class: Option<String>,

    #[arg(long)]
    pub(crate) dest_rel: Option<PathBuf>,

    #[arg(long)]
    pub(crate) session_id: Option<String>,

    #[arg(long, value_enum, default_value_t = TransportSubstrate::Auto)]
    pub(crate) transport_substrate: TransportSubstrate,

    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    pub(crate) backend: BackendArg,

    #[arg(long, default_value_t = DEFAULT_PROBE_BYTES)]
    pub(crate) probe_bytes: usize,

    #[arg(long, default_value_t = false)]
    pub(crate) probe: bool,

    #[arg(long = "note")]
    pub(crate) note: Vec<String>,
}

#[derive(Parser, Debug)]
pub(crate) struct StageQueueArgs {
    #[arg(long)]
    pub(crate) queue: PathBuf,

    #[arg(long = "id")]
    pub(crate) id_filters: Vec<String>,

    #[arg(long)]
    pub(crate) window: Vec<String>,

    #[arg(long)]
    pub(crate) priority: Vec<String>,

    #[arg(long)]
    pub(crate) status: Vec<String>,

    #[arg(long, default_value_t = false)]
    pub(crate) critical_only: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) fail_on_existing: bool,

    #[arg(long, value_enum, default_value_t = TransportSubstrate::Auto)]
    pub(crate) transport_substrate: TransportSubstrate,

    #[arg(long = "note")]
    pub(crate) note: Vec<String>,
}

#[derive(Parser, Debug)]
pub(crate) struct PromoteArgs {
    #[arg(long)]
    pub(crate) session: PathBuf,

    #[arg(long)]
    pub(crate) url: Vec<String>,

    #[arg(long)]
    pub(crate) url_file: Option<PathBuf>,

    #[arg(long)]
    pub(crate) site: Option<String>,

    #[arg(long)]
    pub(crate) access_class: Option<String>,

    #[arg(long)]
    pub(crate) dest_rel: Option<PathBuf>,

    #[arg(long)]
    pub(crate) browser_trace_rel: Option<PathBuf>,

    #[arg(long)]
    pub(crate) cookie_jar_rel: Option<PathBuf>,

    #[arg(long)]
    pub(crate) storage_state_rel: Option<PathBuf>,

    #[arg(long)]
    pub(crate) profile_root_rel: Option<PathBuf>,

    #[arg(long)]
    pub(crate) effective_url: Option<String>,

    #[arg(long)]
    pub(crate) http_code: Option<u16>,

    #[arg(long, value_enum)]
    pub(crate) transport_substrate: Option<TransportSubstrate>,

    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    pub(crate) backend: BackendArg,

    #[arg(long, default_value_t = DEFAULT_PROBE_BYTES)]
    pub(crate) probe_bytes: usize,

    #[arg(long, default_value_t = false)]
    pub(crate) probe: bool,

    #[arg(long)]
    pub(crate) handoff_out: Option<PathBuf>,

    #[arg(long = "note")]
    pub(crate) note: Vec<String>,
}

#[derive(Parser, Debug)]
pub(crate) struct RecordArgs {
    #[arg(long)]
    pub(crate) session: PathBuf,

    #[arg(long)]
    pub(crate) project_api_root: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    pub(crate) reconcile_project_api: bool,

    #[arg(long, value_enum)]
    pub(crate) status: SessionStatus,

    #[arg(long)]
    pub(crate) artifact_rel: Option<PathBuf>,

    #[arg(long)]
    pub(crate) browser_trace_rel: Option<PathBuf>,

    #[arg(long)]
    pub(crate) cookie_jar_rel: Option<PathBuf>,

    #[arg(long)]
    pub(crate) storage_state_rel: Option<PathBuf>,

    #[arg(long)]
    pub(crate) profile_root_rel: Option<PathBuf>,

    #[arg(long)]
    pub(crate) profile_bundle_rel: Option<PathBuf>,

    #[arg(long)]
    pub(crate) effective_url: Option<String>,

    #[arg(long)]
    pub(crate) http_code: Option<u16>,

    #[arg(long)]
    pub(crate) host_scope: Option<String>,

    #[arg(long, value_enum)]
    pub(crate) transport_substrate: Option<TransportSubstrate>,

    #[arg(long = "note")]
    pub(crate) note: Vec<String>,
}

#[derive(Parser, Debug)]
pub(crate) struct ShowArgs {
    #[arg(long)]
    pub(crate) session: PathBuf,
}

#[derive(Parser, Debug)]
pub(crate) struct ImportDossierArgs {
    #[arg(long)]
    pub(crate) dossier: Vec<PathBuf>,

    #[arg(long)]
    pub(crate) batch_manifest: Vec<PathBuf>,

    #[arg(long)]
    pub(crate) max_rank: Option<usize>,

    #[arg(long = "rank")]
    pub(crate) rank: Vec<usize>,

    #[arg(long, default_value_t = false)]
    pub(crate) all_suggestions: bool,

    #[arg(long = "note")]
    pub(crate) note: Vec<String>,
}

#[derive(Parser, Debug)]
pub(crate) struct ConsumeHandoffArgs {
    #[arg(long)]
    pub(crate) capsule: Vec<PathBuf>,

    #[arg(long)]
    pub(crate) handoff_tsv: Vec<PathBuf>,

    #[arg(long)]
    pub(crate) project_api_root: Option<PathBuf>,

    #[arg(long)]
    pub(crate) project_cache_root: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    pub(crate) reconcile_project_api: bool,

    #[arg(long)]
    pub(crate) output_root: PathBuf,

    #[arg(long)]
    pub(crate) report_out: Option<PathBuf>,

    #[arg(long)]
    pub(crate) search_queue_sync: Option<PathBuf>,

    #[arg(long = "header")]
    pub(crate) header: Vec<String>,

    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    pub(crate) backend: BackendArg,

    #[arg(long, default_value_t = false)]
    pub(crate) skip_existing: bool,

    #[arg(long, default_value_t = false)]
    pub(crate) extract_pdf_sidecar: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub(crate) enum SessionStatus {
    Staged,
    InProgress,
    Downloaded,
    Blocked,
    Deferred,
    Abandoned,
}

impl SessionStatus {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Staged => "staged",
            Self::InProgress => "in_progress",
            Self::Downloaded => "downloaded",
            Self::Blocked => "blocked",
            Self::Deferred => "deferred",
            Self::Abandoned => "abandoned",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TransportSubstrate {
    Auto,
    DownloadStack,
    Silksurf,
}

impl TransportSubstrate {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::DownloadStack => "download_stack",
            Self::Silksurf => "silksurf",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum BackendArg {
    Auto,
    Reqwest,
    Curl,
    Wget,
    Aria2,
    Ureq,
}

impl From<BackendArg> for DownloadBackend {
    fn from(value: BackendArg) -> Self {
        match value {
            BackendArg::Auto => DownloadBackend::Auto,
            BackendArg::Reqwest => DownloadBackend::Reqwest,
            BackendArg::Curl => DownloadBackend::CurlCli,
            BackendArg::Wget => DownloadBackend::WgetCli,
            BackendArg::Aria2 => DownloadBackend::Aria2Cli,
            BackendArg::Ureq => DownloadBackend::Ureq,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct AcquisitionSession {
    pub(crate) format_version: u32,
    pub(crate) session_id: String,
    pub(crate) created_utc: String,
    pub(crate) updated_utc: String,
    pub(crate) status: SessionStatus,
    pub(crate) source: SourceTarget,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) lineage: Option<SessionLineage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) search: Option<SearchContext>,
    pub(crate) controller: ControllerPlan,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) route: Option<RouteSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) probe: Option<ProbeSummary>,
    #[serde(default)]
    pub(crate) evidence: EvidenceState,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) events: Vec<SessionEvent>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SourceTarget {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) source_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) site: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) access_class: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SessionLineage {
    pub(crate) parent_session_id: String,
    pub(crate) parent_manifest: String,
    pub(crate) relation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SearchContext {
    pub(crate) queue_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) project_id: Option<String>,
    pub(crate) search_target_id: String,
    pub(crate) window: String,
    pub(crate) priority: String,
    pub(crate) kind: String,
    pub(crate) status: String,
    pub(crate) why_now: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) preferred_lanes: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) query_seeds: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ControllerPlan {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) policy_registry_rel: Option<String>,
    pub(crate) sessions_dir_rel: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) dest_rel: Option<String>,
    pub(crate) transport_substrate: TransportSubstrate,
    pub(crate) requested_backend: String,
    pub(crate) probe_bytes: usize,
    pub(crate) probe_requested: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct RouteSummary {
    pub(crate) scheme: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) host: Option<String>,
    pub(crate) retry_class: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) policy_name: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) backends: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ProbeSummary {
    pub(crate) attempted_utc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) capabilities: Option<CapabilitySummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) terminal_result: Option<TerminalProbeResult>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) attempts: Vec<ProbeAttemptSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) final_error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct CapabilitySummary {
    pub(crate) scheme: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) host: Option<String>,
    pub(crate) surface: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) content_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) http_code: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) content_length: Option<u64>,
    pub(crate) supports_ranges: bool,
    pub(crate) rsync_reachable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) final_url: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct TerminalProbeResult {
    pub(crate) backend: String,
    pub(crate) kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) final_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) http_code: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) content_type: Option<String>,
    pub(crate) bytes: u64,
    pub(crate) is_pdf: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) sha256: Option<String>,
    pub(crate) note: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct ProbeAttemptSummary {
    pub(crate) backend: String,
    pub(crate) succeeded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) failure_class: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) http_code: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) content_type: Option<String>,
    pub(crate) bytes: u64,
    pub(crate) is_pdf: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) final_url: Option<String>,
    pub(crate) note: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) error_message: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub(crate) struct EvidenceState {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) artifact_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) text_sidecar_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) browser_trace_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cookie_jar_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) storage_state_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) profile_bundle_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) request_capsule_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) host_scope: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) effective_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) http_code: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) sha256: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SessionEvent {
    pub(crate) at_utc: String,
    pub(crate) action: String,
    pub(crate) status: SessionStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) note: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct SearchQueueFile {
    #[serde(default)]
    pub(crate) schema_version: Option<u32>,
    #[serde(default)]
    pub(crate) project_id: Option<String>,
    #[serde(default)]
    pub(crate) last_updated: Option<String>,
    #[serde(default)]
    pub(crate) summary: Option<SearchQueueSummary>,
    #[serde(default)]
    pub(crate) search_target: Vec<SearchQueueTarget>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SearchQueueTarget {
    pub(crate) id: String,
    pub(crate) window: String,
    pub(crate) priority: String,
    pub(crate) kind: String,
    pub(crate) title: String,
    pub(crate) status: String,
    pub(crate) why_now: String,
    #[serde(default)]
    pub(crate) preferred_lanes: Vec<String>,
    #[serde(default)]
    pub(crate) query_seeds: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_attempt_utc: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_attempt_result: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_attempt_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_attempt_output_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_attempt_http_code: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_attempt_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) last_attempt_note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SearchQueueSummary {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) critical_retrieval_blockers: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) century_normalization_tracks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) terminology_audit_tracks: Option<usize>,
}

#[derive(Debug)]
pub(crate) struct HandoffRow {
    pub(crate) session_id: String,
    pub(crate) parent_session_id: String,
    pub(crate) url: String,
    pub(crate) effective_url: String,
    pub(crate) cookie_jar_rel: String,
    pub(crate) storage_state_rel: String,
    pub(crate) browser_trace_rel: String,
    pub(crate) profile_bundle_rel: String,
    pub(crate) request_capsule_rel: String,
    pub(crate) http_code: String,
    pub(crate) transport_substrate: String,
    pub(crate) requested_backend: String,
    pub(crate) dest_rel: String,
    pub(crate) note: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct BrowserProfileBundle {
    pub(crate) schema_version: u32,
    pub(crate) host_scope: String,
    pub(crate) bundle_kind: String,
    pub(crate) bundle_root_rel: String,
    pub(crate) latest_session_id: String,
    pub(crate) latest_manifest_rel: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cookie_jar_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) storage_state_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) browser_trace_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) effective_url: Option<String>,
    pub(crate) updated_utc: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RequestHeaderHint {
    pub(crate) name: String,
    pub(crate) value: String,
    pub(crate) source: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct RequestCapsule {
    pub(crate) schema_version: u32,
    pub(crate) generated_utc: String,
    pub(crate) session_id: String,
    pub(crate) parent_session_id: String,
    pub(crate) url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) effective_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) host_scope: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) source_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) site: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) access_class: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) search_queue_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) search_project_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) search_target_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) search_window: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) search_priority: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) search_kind: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) search_status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) cookie_jar_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) storage_state_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) browser_trace_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) profile_bundle_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) request_capsule_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) http_code: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) policy_registry_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) sessions_dir_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) bundle_root_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) bundle_latest_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) bundle_latest_manifest_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) bundle_updated_utc: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub(crate) header_hints: Vec<RequestHeaderHint>,
    pub(crate) transport_substrate: String,
    pub(crate) requested_backend: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) dest_rel: Option<String>,
    pub(crate) note: String,
}

#[derive(Debug)]
pub(crate) struct ConsumeReportRow {
    pub(crate) session_id: String,
    pub(crate) url: String,
    pub(crate) output_rel: String,
    pub(crate) result: String,
    pub(crate) backend: String,
    pub(crate) http_code: String,
    pub(crate) sha256: String,
    pub(crate) final_url: String,
    pub(crate) header_count: usize,
    pub(crate) note: String,
    pub(crate) error: String,
}

#[derive(Debug, Clone)]
pub(crate) enum ConsumeOutcome {
    Downloaded {
        output_path: PathBuf,
        result: TransferResult,
        header_count: usize,
        text_sidecar_rel: Option<String>,
    },
    Failed {
        output_path: PathBuf,
        error: String,
        header_count: usize,
    },
    SkippedExisting {
        output_path: PathBuf,
        bytes: u64,
        sha256: String,
        header_count: usize,
        text_sidecar_rel: Option<String>,
    },
}

#[derive(Debug, Clone)]
pub(crate) struct QueueSyncUpdate {
    pub(crate) search_target_id: String,
    pub(crate) attempted_utc: String,
    pub(crate) outcome: String,
    pub(crate) session_id: String,
    pub(crate) output_rel: Option<String>,
    pub(crate) http_code: Option<u16>,
    pub(crate) sha256: Option<String>,
    pub(crate) note: String,
}

#[derive(Debug, Clone)]
pub(crate) struct PromoteSpec {
    pub(crate) url: String,
    pub(crate) source_id: Option<String>,
    pub(crate) title: Option<String>,
    pub(crate) site: Option<String>,
    pub(crate) access_class: Option<String>,
    pub(crate) note: Option<String>,
}
