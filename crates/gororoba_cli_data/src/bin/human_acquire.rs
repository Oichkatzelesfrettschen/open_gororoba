use anyhow::{Context, Result, bail};
use chrono::Utc;
use clap::{Parser, Subcommand, ValueEnum};
use data_core::download_stack::{
    DEFAULT_PROBE_BYTES, DownloadBackend, DownloadStack, EndpointCapabilities, TransferKind,
    TransferRequest, TransferResult, TransferTrace, load_host_policy_registry,
};
use gororoba_cli_data::{
    acquisition_dossier::{
        ResearchDossier, StageSuggestion, load_dossier_batch_manifest, load_research_dossier,
        resolve_manifest_entry_path,
    },
    project_api_contract::{
        AcquisitionJournalRow, ProjectApiCrosswalkBinding, append_acquisition_journal_rows,
        journal_multi_value, load_project_api_context, load_project_api_crosswalk,
        project_relative_path, project_relative_path_from_display, resolve_crosswalk_binding,
    },
    source_provenance,
};
use lit_search::pdf::PdfExtractor;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashSet,
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
};
use url::Url;

#[derive(Parser, Debug)]
#[command(
    name = "human-acquire",
    about = "Thin operator-guided acquisition controller for stubborn external sources"
)]
struct Cli {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    #[arg(long, default_value = "registry/download_host_policies.toml")]
    policy_registry: PathBuf,

    #[arg(long, default_value = "reports/acquisition_sessions")]
    sessions_dir: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Stage(StageArgs),
    StageQueue(StageQueueArgs),
    Promote(PromoteArgs),
    ImportDossier(ImportDossierArgs),
    Record(RecordArgs),
    Show(ShowArgs),
    ConsumeHandoff(ConsumeHandoffArgs),
}

#[derive(Parser, Debug)]
struct StageArgs {
    #[arg(long)]
    url: String,

    #[arg(long)]
    source_id: Option<String>,

    #[arg(long)]
    title: Option<String>,

    #[arg(long)]
    site: Option<String>,

    #[arg(long)]
    access_class: Option<String>,

    #[arg(long)]
    dest_rel: Option<PathBuf>,

    #[arg(long)]
    session_id: Option<String>,

    #[arg(long, value_enum, default_value_t = TransportSubstrate::Auto)]
    transport_substrate: TransportSubstrate,

    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    backend: BackendArg,

    #[arg(long, default_value_t = DEFAULT_PROBE_BYTES)]
    probe_bytes: usize,

    #[arg(long, default_value_t = false)]
    probe: bool,

    #[arg(long = "note")]
    note: Vec<String>,
}

#[derive(Parser, Debug)]
struct StageQueueArgs {
    #[arg(long)]
    queue: PathBuf,

    #[arg(long = "id")]
    id_filters: Vec<String>,

    #[arg(long)]
    window: Vec<String>,

    #[arg(long)]
    priority: Vec<String>,

    #[arg(long)]
    status: Vec<String>,

    #[arg(long, default_value_t = false)]
    critical_only: bool,

    #[arg(long, default_value_t = false)]
    fail_on_existing: bool,

    #[arg(long, value_enum, default_value_t = TransportSubstrate::Auto)]
    transport_substrate: TransportSubstrate,

    #[arg(long = "note")]
    note: Vec<String>,
}

#[derive(Parser, Debug)]
struct PromoteArgs {
    #[arg(long)]
    session: PathBuf,

    #[arg(long)]
    url: Vec<String>,

    #[arg(long)]
    url_file: Option<PathBuf>,

    #[arg(long)]
    site: Option<String>,

    #[arg(long)]
    access_class: Option<String>,

    #[arg(long)]
    dest_rel: Option<PathBuf>,

    #[arg(long)]
    browser_trace_rel: Option<PathBuf>,

    #[arg(long)]
    cookie_jar_rel: Option<PathBuf>,

    #[arg(long)]
    storage_state_rel: Option<PathBuf>,

    #[arg(long)]
    profile_root_rel: Option<PathBuf>,

    #[arg(long)]
    effective_url: Option<String>,

    #[arg(long)]
    http_code: Option<u16>,

    #[arg(long, value_enum)]
    transport_substrate: Option<TransportSubstrate>,

    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    backend: BackendArg,

    #[arg(long, default_value_t = DEFAULT_PROBE_BYTES)]
    probe_bytes: usize,

    #[arg(long, default_value_t = false)]
    probe: bool,

    #[arg(long)]
    handoff_out: Option<PathBuf>,

    #[arg(long = "note")]
    note: Vec<String>,
}

#[derive(Parser, Debug)]
struct RecordArgs {
    #[arg(long)]
    session: PathBuf,

    #[arg(long)]
    project_api_root: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    reconcile_project_api: bool,

    #[arg(long, value_enum)]
    status: SessionStatus,

    #[arg(long)]
    artifact_rel: Option<PathBuf>,

    #[arg(long)]
    browser_trace_rel: Option<PathBuf>,

    #[arg(long)]
    cookie_jar_rel: Option<PathBuf>,

    #[arg(long)]
    storage_state_rel: Option<PathBuf>,

    #[arg(long)]
    profile_root_rel: Option<PathBuf>,

    #[arg(long)]
    profile_bundle_rel: Option<PathBuf>,

    #[arg(long)]
    effective_url: Option<String>,

    #[arg(long)]
    http_code: Option<u16>,

    #[arg(long)]
    host_scope: Option<String>,

    #[arg(long, value_enum)]
    transport_substrate: Option<TransportSubstrate>,

    #[arg(long = "note")]
    note: Vec<String>,
}

#[derive(Parser, Debug)]
struct ShowArgs {
    #[arg(long)]
    session: PathBuf,
}

#[derive(Parser, Debug)]
struct ImportDossierArgs {
    #[arg(long)]
    dossier: Vec<PathBuf>,

    #[arg(long)]
    batch_manifest: Vec<PathBuf>,

    #[arg(long)]
    max_rank: Option<usize>,

    #[arg(long = "rank")]
    rank: Vec<usize>,

    #[arg(long, default_value_t = false)]
    all_suggestions: bool,

    #[arg(long = "note")]
    note: Vec<String>,
}

#[derive(Parser, Debug)]
struct ConsumeHandoffArgs {
    #[arg(long)]
    capsule: Vec<PathBuf>,

    #[arg(long)]
    handoff_tsv: Vec<PathBuf>,

    #[arg(long)]
    project_api_root: Option<PathBuf>,

    #[arg(long)]
    project_cache_root: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    reconcile_project_api: bool,

    #[arg(long)]
    output_root: PathBuf,

    #[arg(long)]
    report_out: Option<PathBuf>,

    #[arg(long)]
    search_queue_sync: Option<PathBuf>,

    #[arg(long = "header")]
    header: Vec<String>,

    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    backend: BackendArg,

    #[arg(long, default_value_t = false)]
    skip_existing: bool,

    #[arg(long, default_value_t = false)]
    extract_pdf_sidecar: bool,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum SessionStatus {
    Staged,
    InProgress,
    Downloaded,
    Blocked,
    Deferred,
    Abandoned,
}

impl SessionStatus {
    fn as_str(self) -> &'static str {
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
enum TransportSubstrate {
    Auto,
    DownloadStack,
    Silksurf,
}

impl TransportSubstrate {
    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::DownloadStack => "download_stack",
            Self::Silksurf => "silksurf",
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum BackendArg {
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
struct AcquisitionSession {
    format_version: u32,
    session_id: String,
    created_utc: String,
    updated_utc: String,
    status: SessionStatus,
    source: SourceTarget,
    #[serde(skip_serializing_if = "Option::is_none")]
    lineage: Option<SessionLineage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    search: Option<SearchContext>,
    controller: ControllerPlan,
    #[serde(skip_serializing_if = "Option::is_none")]
    route: Option<RouteSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    probe: Option<ProbeSummary>,
    #[serde(default)]
    evidence: EvidenceState,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    events: Vec<SessionEvent>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SourceTarget {
    #[serde(skip_serializing_if = "Option::is_none")]
    url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    site: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    access_class: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SessionLineage {
    parent_session_id: String,
    parent_manifest: String,
    relation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchContext {
    queue_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    project_id: Option<String>,
    search_target_id: String,
    window: String,
    priority: String,
    kind: String,
    status: String,
    why_now: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    preferred_lanes: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    query_seeds: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ControllerPlan {
    #[serde(skip_serializing_if = "Option::is_none")]
    policy_registry_rel: Option<String>,
    sessions_dir_rel: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    dest_rel: Option<String>,
    transport_substrate: TransportSubstrate,
    requested_backend: String,
    probe_bytes: usize,
    probe_requested: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct RouteSummary {
    scheme: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    host: Option<String>,
    retry_class: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    policy_name: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    backends: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ProbeSummary {
    attempted_utc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    capabilities: Option<CapabilitySummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    terminal_result: Option<TerminalProbeResult>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    attempts: Vec<ProbeAttemptSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    final_error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CapabilitySummary {
    scheme: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    host: Option<String>,
    surface: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    http_code: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content_length: Option<u64>,
    supports_ranges: bool,
    rsync_reachable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    final_url: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct TerminalProbeResult {
    backend: String,
    kind: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    final_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    http_code: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content_type: Option<String>,
    bytes: u64,
    is_pdf: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    sha256: Option<String>,
    note: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ProbeAttemptSummary {
    backend: String,
    succeeded: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    failure_class: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    http_code: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content_type: Option<String>,
    bytes: u64,
    is_pdf: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    final_url: Option<String>,
    note: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_message: Option<String>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct EvidenceState {
    #[serde(skip_serializing_if = "Option::is_none")]
    artifact_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text_sidecar_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    browser_trace_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cookie_jar_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    storage_state_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    profile_bundle_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_capsule_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    host_scope: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    effective_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    http_code: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sha256: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SessionEvent {
    at_utc: String,
    action: String,
    status: SessionStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    note: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct SearchQueueFile {
    #[serde(default)]
    schema_version: Option<u32>,
    #[serde(default)]
    project_id: Option<String>,
    #[serde(default)]
    last_updated: Option<String>,
    #[serde(default)]
    summary: Option<SearchQueueSummary>,
    #[serde(default)]
    search_target: Vec<SearchQueueTarget>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchQueueTarget {
    id: String,
    window: String,
    priority: String,
    kind: String,
    title: String,
    status: String,
    why_now: String,
    #[serde(default)]
    preferred_lanes: Vec<String>,
    #[serde(default)]
    query_seeds: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_attempt_utc: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_attempt_result: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_attempt_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_attempt_output_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_attempt_http_code: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_attempt_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_attempt_note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SearchQueueSummary {
    #[serde(skip_serializing_if = "Option::is_none")]
    critical_retrieval_blockers: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    century_normalization_tracks: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    terminology_audit_tracks: Option<usize>,
}

#[derive(Debug)]
struct HandoffRow {
    session_id: String,
    parent_session_id: String,
    url: String,
    effective_url: String,
    cookie_jar_rel: String,
    storage_state_rel: String,
    browser_trace_rel: String,
    profile_bundle_rel: String,
    request_capsule_rel: String,
    http_code: String,
    transport_substrate: String,
    requested_backend: String,
    dest_rel: String,
    note: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct BrowserProfileBundle {
    schema_version: u32,
    host_scope: String,
    bundle_kind: String,
    bundle_root_rel: String,
    latest_session_id: String,
    latest_manifest_rel: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    cookie_jar_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    storage_state_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    browser_trace_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    effective_url: Option<String>,
    updated_utc: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RequestHeaderHint {
    name: String,
    value: String,
    source: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct RequestCapsule {
    schema_version: u32,
    generated_utc: String,
    session_id: String,
    parent_session_id: String,
    url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    effective_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    host_scope: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    site: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    access_class: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    search_queue_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    search_project_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    search_target_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    search_window: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    search_priority: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    search_status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cookie_jar_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    storage_state_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    browser_trace_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    profile_bundle_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    request_capsule_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    http_code: Option<u16>,
    #[serde(skip_serializing_if = "Option::is_none")]
    policy_registry_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sessions_dir_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bundle_root_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bundle_latest_session_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bundle_latest_manifest_rel: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bundle_updated_utc: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    header_hints: Vec<RequestHeaderHint>,
    transport_substrate: String,
    requested_backend: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    dest_rel: Option<String>,
    note: String,
}

#[derive(Debug)]
struct ConsumeReportRow {
    session_id: String,
    url: String,
    output_rel: String,
    result: String,
    backend: String,
    http_code: String,
    sha256: String,
    final_url: String,
    header_count: usize,
    note: String,
    error: String,
}

#[derive(Debug, Clone)]
enum ConsumeOutcome {
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
struct QueueSyncUpdate {
    search_target_id: String,
    attempted_utc: String,
    outcome: String,
    session_id: String,
    output_rel: Option<String>,
    http_code: Option<u16>,
    sha256: Option<String>,
    note: String,
}

#[derive(Debug, Clone)]
struct PromoteSpec {
    url: String,
    source_id: Option<String>,
    title: Option<String>,
    site: Option<String>,
    access_class: Option<String>,
    note: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let repo_root = resolve_repo_root(&cli.repo_root);
    let policy_registry = repo_path(&repo_root, &cli.policy_registry);
    let sessions_dir = repo_path(&repo_root, &cli.sessions_dir);

    match cli.command {
        Commands::Stage(args) => run_stage(&repo_root, &policy_registry, &sessions_dir, args),
        Commands::StageQueue(args) => run_stage_queue(&repo_root, &sessions_dir, args),
        Commands::Promote(args) => run_promote(&repo_root, &policy_registry, &sessions_dir, args),
        Commands::ImportDossier(args) => {
            run_import_dossier(&repo_root, &policy_registry, &sessions_dir, args)
        }
        Commands::Record(args) => run_record(&repo_root, &sessions_dir, args),
        Commands::Show(args) => run_show(&repo_root, &sessions_dir, args),
        Commands::ConsumeHandoff(args) => run_consume_handoff(&repo_root, &policy_registry, args),
    }
}

fn run_stage(
    repo_root: &Path,
    policy_registry: &Path,
    sessions_dir: &Path,
    args: StageArgs,
) -> Result<()> {
    let session_id = sanitize_token(&args.session_id.unwrap_or_else(|| {
        derive_session_id(
            args.source_id.as_deref(),
            args.title.as_deref(),
            args.url.as_str(),
        )
    }));
    let session_dir = sessions_dir.join(&session_id);
    let manifest_path = session_dir.join("session.toml");
    let checklist_path = session_dir.join("checklist.md");
    if manifest_path.exists() {
        bail!("session already exists at {}", manifest_path.display());
    }

    let now = utc_now();
    let stack = build_stack(policy_registry)?;
    let mut request = TransferRequest::probe(args.url.clone());
    request.backend = args.backend.into();
    request.probe_bytes = args.probe_bytes;
    if !args.note.is_empty() {
        request.note = Some(args.note.join(" | "));
    }

    let trace = if args.probe {
        Some(stack.probe_with_trace(&request))
    } else {
        None
    };
    let route = trace
        .as_ref()
        .map(RouteSummary::from_trace)
        .unwrap_or_else(|| RouteSummary::from_request(&stack, &request));
    let host_scope = derive_host_scope(request.url.as_str(), None);

    let mut session = AcquisitionSession {
        format_version: 1,
        session_id: session_id.clone(),
        created_utc: now.clone(),
        updated_utc: now.clone(),
        status: SessionStatus::Staged,
        source: SourceTarget {
            url: Some(args.url),
            source_id: args.source_id,
            title: args.title,
            site: args.site,
            access_class: args.access_class,
        },
        lineage: None,
        search: None,
        controller: ControllerPlan {
            policy_registry_rel: Some(to_repo_display_path(repo_root, policy_registry)),
            sessions_dir_rel: to_repo_display_path(repo_root, sessions_dir),
            dest_rel: args
                .dest_rel
                .as_ref()
                .map(|path| display_input_path(repo_root, path)),
            transport_substrate: args.transport_substrate,
            requested_backend: request.backend.as_str().to_string(),
            probe_bytes: args.probe_bytes,
            probe_requested: args.probe,
        },
        route: Some(route),
        probe: trace
            .as_ref()
            .map(|trace| ProbeSummary::from_trace(trace, &now)),
        evidence: EvidenceState {
            artifact_rel: None,
            text_sidecar_rel: None,
            browser_trace_rel: None,
            cookie_jar_rel: None,
            storage_state_rel: None,
            profile_bundle_rel: None,
            request_capsule_rel: None,
            host_scope,
            effective_url: None,
            http_code: None,
            bytes: None,
            sha256: None,
        },
        events: vec![SessionEvent {
            at_utc: now,
            action: "stage".to_string(),
            status: SessionStatus::Staged,
            note: join_notes(&args.note),
        }],
    };

    if args.transport_substrate == TransportSubstrate::Silksurf {
        session.events.push(SessionEvent {
            at_utc: utc_now(),
            action: "transport_hint".to_string(),
            status: SessionStatus::Staged,
            note: Some(
                "silksurf is recorded as an optional transport substrate; browser/session control remains outside the thin controller".to_string(),
            ),
        });
        session.updated_utc = utc_now();
    }

    write_session(repo_root, &manifest_path, &session)?;
    write_checklist(repo_root, &checklist_path, &manifest_path, &session)?;
    print_stage_summary(repo_root, &manifest_path, &checklist_path, &session);
    Ok(())
}

fn run_stage_queue(repo_root: &Path, sessions_dir: &Path, args: StageQueueArgs) -> Result<()> {
    let queue_path = repo_path(repo_root, &args.queue);
    let queue = load_search_queue(&queue_path)?;
    let queue_display = to_repo_display_path(repo_root, &queue_path);
    let mut created = 0_usize;
    let mut skipped = 0_usize;

    for target in queue
        .search_target
        .iter()
        .filter(|target| target_matches_filters(target, &args))
    {
        let session_id = sanitize_token(&format!(
            "{}_{}",
            queue.project_id.as_deref().unwrap_or("queue"),
            target.id
        ));
        let manifest_path = sessions_dir.join(&session_id).join("session.toml");
        let checklist_path = sessions_dir.join(&session_id).join("checklist.md");
        if manifest_path.exists() {
            if args.fail_on_existing {
                bail!("session already exists at {}", manifest_path.display());
            }
            skipped += 1;
            println!(
                "skip_existing_session={} manifest={}",
                session_id,
                to_repo_display_path(repo_root, &manifest_path)
            );
            continue;
        }

        let now = utc_now();
        let session = AcquisitionSession {
            format_version: 1,
            session_id: session_id.clone(),
            created_utc: now.clone(),
            updated_utc: now.clone(),
            status: SessionStatus::Staged,
            source: SourceTarget {
                url: None,
                source_id: Some(target.id.clone()),
                title: Some(target.title.clone()),
                site: None,
                access_class: None,
            },
            lineage: None,
            search: Some(SearchContext {
                queue_path: queue_display.clone(),
                project_id: queue.project_id.clone(),
                search_target_id: target.id.clone(),
                window: target.window.clone(),
                priority: target.priority.clone(),
                kind: target.kind.clone(),
                status: target.status.clone(),
                why_now: target.why_now.clone(),
                preferred_lanes: target.preferred_lanes.clone(),
                query_seeds: target.query_seeds.clone(),
            }),
            controller: ControllerPlan {
                policy_registry_rel: None,
                sessions_dir_rel: to_repo_display_path(repo_root, sessions_dir),
                dest_rel: None,
                transport_substrate: args.transport_substrate,
                requested_backend: DownloadBackend::Auto.as_str().to_string(),
                probe_bytes: DEFAULT_PROBE_BYTES,
                probe_requested: false,
            },
            route: None,
            probe: None,
            evidence: EvidenceState::default(),
            events: vec![SessionEvent {
                at_utc: now,
                action: "stage_queue".to_string(),
                status: SessionStatus::Staged,
                note: Some(queue_stage_note(target, &args.note)),
            }],
        };

        write_session(repo_root, &manifest_path, &session)?;
        write_checklist(repo_root, &checklist_path, &manifest_path, &session)?;
        println!(
            "staged_session={} search_target={} manifest={}",
            session_id,
            target.id,
            to_repo_display_path(repo_root, &manifest_path)
        );
        created += 1;
    }

    println!("queue={queue_display}");
    println!("queue_created={created}");
    println!("queue_skipped={skipped}");
    Ok(())
}

fn run_promote(
    repo_root: &Path,
    policy_registry: &Path,
    sessions_dir: &Path,
    args: PromoteArgs,
) -> Result<()> {
    let parent_manifest = resolve_session_manifest(repo_root, sessions_dir, &args.session);
    let urls = collect_promotion_urls(repo_root, &args)?;
    if urls.is_empty() {
        bail!("promotion requires at least one URL");
    }
    let specs = urls
        .into_iter()
        .map(|url| PromoteSpec {
            url,
            source_id: None,
            title: None,
            site: args.site.clone(),
            access_class: args.access_class.clone(),
            note: None,
        })
        .collect::<Vec<_>>();
    promote_specs(
        repo_root,
        policy_registry,
        sessions_dir,
        &parent_manifest,
        specs,
        args.dest_rel.as_ref(),
        args.browser_trace_rel.as_ref(),
        args.cookie_jar_rel.as_ref(),
        args.storage_state_rel.as_ref(),
        args.profile_root_rel.as_ref(),
        args.effective_url.clone(),
        args.http_code,
        args.transport_substrate,
        args.backend,
        args.probe_bytes,
        args.probe,
        args.handoff_out.as_ref(),
        &args.note,
    )
}

fn run_import_dossier(
    repo_root: &Path,
    policy_registry: &Path,
    sessions_dir: &Path,
    args: ImportDossierArgs,
) -> Result<()> {
    let dossier_paths = collect_import_dossier_paths(repo_root, &args)?;
    if dossier_paths.is_empty() {
        bail!("import-dossier requires at least one --dossier or --batch-manifest");
    }

    for dossier_path in dossier_paths {
        let dossier = load_research_dossier(&dossier_path)?;
        let parent_manifest = ensure_parent_session_for_dossier(repo_root, sessions_dir, &dossier)?;
        let selected = select_dossier_suggestions(&dossier, &args)?;
        if selected.is_empty() {
            println!(
                "import_skip_dossier={} reason=no_selected_suggestions",
                dossier_path.display()
            );
            continue;
        }
        let specs = selected
            .into_iter()
            .map(|suggestion| dossier_suggestion_to_promote_spec(&dossier, suggestion, &args.note))
            .collect::<Vec<_>>();
        promote_specs(
            repo_root,
            policy_registry,
            sessions_dir,
            &parent_manifest,
            specs,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            BackendArg::Auto,
            DEFAULT_PROBE_BYTES,
            false,
            None,
            &args.note,
        )?;
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn promote_specs(
    repo_root: &Path,
    policy_registry: &Path,
    sessions_dir: &Path,
    parent_manifest: &Path,
    specs: Vec<PromoteSpec>,
    dest_rel: Option<&PathBuf>,
    browser_trace_rel: Option<&PathBuf>,
    cookie_jar_rel: Option<&PathBuf>,
    storage_state_rel: Option<&PathBuf>,
    profile_root_rel: Option<&PathBuf>,
    effective_url: Option<String>,
    http_code: Option<u16>,
    transport_substrate: Option<TransportSubstrate>,
    backend: BackendArg,
    probe_bytes: usize,
    probe: bool,
    handoff_out: Option<&PathBuf>,
    notes: &[String],
) -> Result<()> {
    let parent_checklist = parent_manifest
        .parent()
        .context("session manifest must have a parent directory")?
        .join("checklist.md");
    let mut parent = load_session(parent_manifest)?;
    let stack = build_stack(policy_registry)?;
    let inherited_substrate = transport_substrate.unwrap_or(parent.controller.transport_substrate);
    let shared_browser_trace = browser_trace_rel
        .as_ref()
        .map(|path| display_input_path(repo_root, path));
    let shared_cookie_jar = cookie_jar_rel
        .as_ref()
        .map(|path| display_input_path(repo_root, path));
    let shared_storage_state = storage_state_rel
        .as_ref()
        .map(|path| display_input_path(repo_root, path));
    let mut child_ids = Vec::new();
    let mut handoff_rows = Vec::new();

    for (index, spec) in specs.into_iter().enumerate() {
        let child_session_id =
            allocate_child_session_id(sessions_dir, &parent.session_id, &spec.url, index);
        let child_dir = sessions_dir.join(&child_session_id);
        let child_manifest = child_dir.join("session.toml");
        let child_checklist = child_dir.join("checklist.md");
        let host_scope = derive_host_scope(spec.url.as_str(), effective_url.as_deref());
        let inherited_bundle = if let Some(profile_root_rel) = profile_root_rel {
            load_profile_bundle(repo_root, profile_root_rel, host_scope.as_deref())?
        } else {
            None
        };
        let resolved_effective_url = effective_url.clone().or_else(|| {
            inherited_bundle
                .as_ref()
                .and_then(|bundle| bundle.effective_url.clone())
        });
        let resolved_browser_trace = shared_browser_trace.clone().or_else(|| {
            inherited_bundle
                .as_ref()
                .and_then(|bundle| bundle.browser_trace_rel.clone())
        });
        let resolved_cookie_jar = shared_cookie_jar.clone().or_else(|| {
            inherited_bundle
                .as_ref()
                .and_then(|bundle| bundle.cookie_jar_rel.clone())
        });
        let resolved_storage_state = shared_storage_state.clone().or_else(|| {
            inherited_bundle
                .as_ref()
                .and_then(|bundle| bundle.storage_state_rel.clone())
        });
        let mut request = TransferRequest::probe(spec.url.clone());
        request.backend = backend.into();
        request.probe_bytes = probe_bytes;
        let note_text = join_optional_notes(&[join_notes(notes), spec.note.clone()]);
        if let Some(note_text) = &note_text {
            request.note = Some(note_text.clone());
        }
        let trace = if probe {
            Some(stack.probe_with_trace(&request))
        } else {
            None
        };
        let route = trace
            .as_ref()
            .map(RouteSummary::from_trace)
            .unwrap_or_else(|| RouteSummary::from_request(&stack, &request));
        let now = utc_now();
        let promote_note = promote_note(&parent.session_id, notes);
        let promote_note = match (inherited_bundle.is_some(), spec.note.as_deref()) {
            (true, Some(spec_note)) => format!(
                "{promote_note} | reused_profile_bundle_for_host_scope={} | {spec_note}",
                host_scope.as_deref().unwrap_or("generic_host")
            ),
            (true, None) => format!(
                "{promote_note} | reused_profile_bundle_for_host_scope={}",
                host_scope.as_deref().unwrap_or("generic_host")
            ),
            (false, Some(spec_note)) => format!("{promote_note} | {spec_note}"),
            (false, None) => promote_note,
        };
        let mut child = AcquisitionSession {
            format_version: 1,
            session_id: child_session_id.clone(),
            created_utc: now.clone(),
            updated_utc: now.clone(),
            status: SessionStatus::Staged,
            source: SourceTarget {
                url: Some(spec.url.clone()),
                source_id: spec
                    .source_id
                    .clone()
                    .or_else(|| parent.source.source_id.clone()),
                title: spec.title.clone().or_else(|| parent.source.title.clone()),
                site: spec.site.clone().or_else(|| parent.source.site.clone()),
                access_class: spec
                    .access_class
                    .clone()
                    .or_else(|| parent.source.access_class.clone()),
            },
            lineage: Some(SessionLineage {
                parent_session_id: parent.session_id.clone(),
                parent_manifest: to_repo_display_path(repo_root, parent_manifest),
                relation: "promoted_child".to_string(),
            }),
            search: parent.search.clone(),
            controller: ControllerPlan {
                policy_registry_rel: Some(to_repo_display_path(repo_root, policy_registry)),
                sessions_dir_rel: to_repo_display_path(repo_root, sessions_dir),
                dest_rel: dest_rel.map(|path| display_input_path(repo_root, path)),
                transport_substrate: inherited_substrate,
                requested_backend: request.backend.as_str().to_string(),
                probe_bytes,
                probe_requested: probe,
            },
            route: Some(route),
            probe: trace
                .as_ref()
                .map(|trace| ProbeSummary::from_trace(trace, &now)),
            evidence: EvidenceState {
                artifact_rel: None,
                text_sidecar_rel: None,
                browser_trace_rel: resolved_browser_trace,
                cookie_jar_rel: resolved_cookie_jar,
                storage_state_rel: resolved_storage_state,
                profile_bundle_rel: None,
                request_capsule_rel: None,
                host_scope,
                effective_url: resolved_effective_url,
                http_code,
                bytes: None,
                sha256: None,
            },
            events: vec![SessionEvent {
                at_utc: now,
                action: "promote_child".to_string(),
                status: SessionStatus::Staged,
                note: Some(promote_note),
            }],
        };

        if let Some(profile_root_rel) = profile_root_rel {
            let bundle_manifest =
                write_profile_bundle(repo_root, profile_root_rel, &child, &child_manifest)?;
            if let Some(bundle_manifest) = bundle_manifest {
                child.evidence.profile_bundle_rel =
                    Some(to_repo_display_path(repo_root, &bundle_manifest));
            }
        }

        if session_is_handoff_ready(&child) {
            let capsule_path = child_dir.join("request_capsule.json");
            child.evidence.request_capsule_rel =
                Some(to_repo_display_path(repo_root, &capsule_path));
            let capsule = RequestCapsule::from_session(repo_root, &parent.session_id, &child)?;
            write_request_capsule(&capsule_path, &capsule)?;
            handoff_rows.push(HandoffRow::from_session(&parent.session_id, &child));
        }

        write_session(repo_root, &child_manifest, &child)?;
        write_checklist(repo_root, &child_checklist, &child_manifest, &child)?;
        println!(
            "child_session={} parent_session={} manifest={}",
            child_session_id,
            parent.session_id,
            to_repo_display_path(repo_root, &child_manifest)
        );
        child_ids.push(child_session_id);
    }

    if !matches!(
        parent.status,
        SessionStatus::Downloaded | SessionStatus::Abandoned
    ) {
        parent.status = SessionStatus::InProgress;
    }
    parent.updated_utc = utc_now();
    parent.events.push(SessionEvent {
        at_utc: parent.updated_utc.clone(),
        action: "promote".to_string(),
        status: parent.status,
        note: Some(format!(
            "spawned_child_sessions={}{}",
            child_ids.join(","),
            render_optional_suffix(&join_notes(notes))
        )),
    });
    write_session(repo_root, parent_manifest, &parent)?;
    write_checklist(repo_root, &parent_checklist, parent_manifest, &parent)?;
    if let Some(handoff_out) = handoff_out {
        let handoff_path = repo_path(repo_root, handoff_out);
        write_handoff_rows(&handoff_path, &handoff_rows)?;
        println!("handoff={}", to_repo_display_path(repo_root, &handoff_path));
        println!("handoff_rows={}", handoff_rows.len());
    }
    print_stage_summary(repo_root, parent_manifest, &parent_checklist, &parent);
    Ok(())
}

fn run_record(repo_root: &Path, sessions_dir: &Path, args: RecordArgs) -> Result<()> {
    let manifest_path = resolve_session_manifest(repo_root, sessions_dir, &args.session);
    let checklist_path = manifest_path
        .parent()
        .context("session manifest must have a parent directory")?
        .join("checklist.md");
    let mut session = load_session(&manifest_path)?;

    if let Some(substrate) = args.transport_substrate {
        session.controller.transport_substrate = substrate;
    }
    if let Some(effective_url) = args.effective_url {
        session.evidence.effective_url = Some(effective_url);
    }
    if let Some(http_code) = args.http_code {
        session.evidence.http_code = Some(http_code);
    }
    if let Some(host_scope) = args.host_scope {
        session.evidence.host_scope = Some(sanitize_token(&host_scope));
    }
    if let Some(browser_trace_rel) = args.browser_trace_rel.as_ref() {
        session.evidence.browser_trace_rel = Some(display_input_path(repo_root, browser_trace_rel));
    }
    if let Some(cookie_jar_rel) = args.cookie_jar_rel.as_ref() {
        session.evidence.cookie_jar_rel = Some(display_input_path(repo_root, cookie_jar_rel));
    }
    if let Some(storage_state_rel) = args.storage_state_rel.as_ref() {
        session.evidence.storage_state_rel = Some(display_input_path(repo_root, storage_state_rel));
    }
    if let Some(profile_root_rel) = args.profile_root_rel.as_ref()
        && let Some(bundle_manifest) =
            write_profile_bundle(repo_root, profile_root_rel, &session, &manifest_path)?
    {
        session.evidence.profile_bundle_rel =
            Some(to_repo_display_path(repo_root, &bundle_manifest));
    }
    if let Some(profile_bundle_rel) = args.profile_bundle_rel.as_ref() {
        session.evidence.profile_bundle_rel =
            Some(display_input_path(repo_root, profile_bundle_rel));
    }
    if let Some(artifact_rel) = args.artifact_rel.as_ref() {
        let artifact_display = display_input_path(repo_root, artifact_rel);
        let artifact_path = repo_path(repo_root, artifact_rel);
        if artifact_path.exists() {
            let digest = hash_file(&artifact_path)?;
            let bytes = fs::metadata(&artifact_path)
                .with_context(|| format!("metadata {}", artifact_path.display()))?
                .len();
            session.evidence.artifact_rel = Some(artifact_display);
            session.evidence.sha256 = Some(digest);
            session.evidence.bytes = Some(bytes);
        } else if matches!(args.status, SessionStatus::Downloaded) {
            bail!(
                "artifact {} does not exist, cannot record downloaded status",
                artifact_path.display()
            );
        } else {
            session.evidence.artifact_rel = Some(artifact_display);
        }
    }

    if session.evidence.host_scope.is_none() {
        session.evidence.host_scope = derive_host_scope(
            session.source.url.as_deref().unwrap_or_default(),
            session.evidence.effective_url.as_deref(),
        );
    }

    session.status = args.status;
    session.updated_utc = utc_now();
    session.events.push(SessionEvent {
        at_utc: session.updated_utc.clone(),
        action: "record".to_string(),
        status: session.status,
        note: join_notes(&args.note),
    });

    if session_is_handoff_ready(&session) {
        let capsule_path = manifest_path
            .parent()
            .context("session manifest must have a parent directory")?
            .join("request_capsule.json");
        session.evidence.request_capsule_rel = Some(to_repo_display_path(repo_root, &capsule_path));
        let parent_id = session
            .lineage
            .as_ref()
            .map(|lineage| lineage.parent_session_id.as_str())
            .unwrap_or(session.session_id.as_str());
        let capsule = RequestCapsule::from_session(repo_root, parent_id, &session)?;
        write_request_capsule(&capsule_path, &capsule)?;
    }

    write_session(repo_root, &manifest_path, &session)?;
    write_checklist(repo_root, &checklist_path, &manifest_path, &session)?;
    if let Some(project_api_root) = &args.project_api_root {
        let project_api = load_project_api_context(project_api_root)?;
        let crosswalk = load_project_api_crosswalk(&project_api.crosswalk_path)?;
        let binding = resolve_crosswalk_binding(
            &crosswalk,
            session
                .search
                .as_ref()
                .map(|search| search.search_target_id.as_str()),
            session.source.source_id.as_deref(),
        );
        let journal_row = journal_row_from_record_session(
            repo_root,
            &project_api.repo_root,
            &manifest_path,
            &session,
            binding,
        );
        let should_reconcile = args.reconcile_project_api && !journal_row.project_artifact_rel.is_empty();
        append_acquisition_journal_rows(&project_api.acquisition_journal_path, &[journal_row])?;
        println!(
            "acquisition_journal={}",
            project_api
                .acquisition_journal_path
                .strip_prefix(&project_api.repo_root)
                .unwrap_or(&project_api.acquisition_journal_path)
                .display()
        );
        if should_reconcile {
            run_cd_cache_reconcile(repo_root, &project_api.repo_root)?;
        }
    }
    print_stage_summary(repo_root, &manifest_path, &checklist_path, &session);
    Ok(())
}

fn run_show(repo_root: &Path, sessions_dir: &Path, args: ShowArgs) -> Result<()> {
    let manifest_path = resolve_session_manifest(repo_root, sessions_dir, &args.session);
    let checklist_path = manifest_path
        .parent()
        .context("session manifest must have a parent directory")?
        .join("checklist.md");
    let session = load_session(&manifest_path)?;
    print_stage_summary(repo_root, &manifest_path, &checklist_path, &session);
    if !session.events.is_empty() {
        println!("events={}", session.events.len());
        for event in &session.events {
            println!(
                "event={} action={} status={} note={}",
                event.at_utc,
                event.action,
                event.status.as_str(),
                event.note.clone().unwrap_or_default()
            );
        }
    }
    Ok(())
}

fn run_consume_handoff(
    repo_root: &Path,
    policy_registry: &Path,
    args: ConsumeHandoffArgs,
) -> Result<()> {
    if args.capsule.is_empty() && args.handoff_tsv.is_empty() {
        bail!("consume-handoff requires at least one --capsule or --handoff-tsv input");
    }

    let stack = build_stack(policy_registry)?;
    let output_root = repo_path(repo_root, &args.output_root);
    fs::create_dir_all(&output_root)
        .with_context(|| format!("create {}", output_root.display()))?;
    let mut jobs = Vec::new();
    for capsule_path in &args.capsule {
        jobs.push(load_request_capsule_path(repo_root, capsule_path)?);
    }
    for handoff_path in &args.handoff_tsv {
        jobs.extend(load_handoff_jobs(repo_root, handoff_path)?);
    }

    let explicit_headers = parse_header_specs(&args.header)?;
    let project_api = if let Some(project_api_root) = &args.project_api_root {
        Some(load_project_api_context(project_api_root)?)
    } else {
        None
    };
    let project_cache_root = args
        .project_cache_root
        .as_ref()
        .map(|path| repo_path(repo_root, path));
    let project_api_crosswalk = if let Some(project_api) = &project_api {
        Some(load_project_api_crosswalk(&project_api.crosswalk_path)?)
    } else {
        None
    };
    let mut report_rows = Vec::new();
    let mut journal_rows = Vec::new();
    let mut succeeded = 0_usize;
    let mut failed = 0_usize;
    let mut skipped = 0_usize;
    let mut queue_updates = Vec::new();

    for capsule in jobs {
        let request_headers = request_headers_for_capsule(repo_root, &capsule, &explicit_headers);
        let output_path = handoff_output_path(&output_root, &capsule);
        if args.skip_existing && output_path.exists() {
            let sha256 = hash_file(&output_path)?;
            let bytes = fs::metadata(&output_path)
                .with_context(|| format!("metadata {}", output_path.display()))?
                .len();
            skipped += 1;
            let outcome = ConsumeOutcome::SkippedExisting {
                output_path: output_path.clone(),
                bytes,
                sha256: sha256.clone(),
                header_count: request_headers.len(),
                text_sidecar_rel: maybe_extract_pdf_sidecar(
                    repo_root,
                    &output_path,
                    args.extract_pdf_sidecar,
                )?,
            };
            report_rows.push(ConsumeReportRow::skipped(
                repo_root,
                &capsule,
                &output_path,
                request_headers.len(),
                &sha256,
            ));
            writeback_session_outcome(repo_root, &capsule, &outcome)?;
            if capsule.search_target_id.is_some() {
                queue_updates.push(build_queue_update(repo_root, &capsule, &outcome));
            }
            if let (Some(project_api), Some(crosswalk)) = (&project_api, &project_api_crosswalk) {
                let binding = resolve_crosswalk_binding(
                    crosswalk,
                    capsule.search_target_id.as_deref(),
                    capsule.source_id.as_deref(),
                );
                journal_rows.push(journal_row_from_consume_outcome(
                    repo_root,
                    &project_api.repo_root,
                    &capsule,
                    &outcome,
                    binding,
                    project_cache_root.as_deref(),
                ));
            }
            println!(
                "skip_existing session_id={} output={}",
                capsule.session_id,
                to_repo_display_path(repo_root, &output_path)
            );
            continue;
        }

        let mut request = TransferRequest::download(capsule.url.clone(), &output_path);
        request.backend = preferred_backend(&capsule, args.backend);
        request.note = Some(format!(
            "human_acquire consume-handoff | parent_session={} | {}",
            capsule.parent_session_id, capsule.note
        ));
        request.headers = request_headers.clone();
        let trace = stack.recover_with_trace(&request);
        match trace.into_result(&request.url) {
            Ok(result) => {
                succeeded += 1;
                let outcome = ConsumeOutcome::Downloaded {
                    output_path: output_path.clone(),
                    result: result.clone(),
                    header_count: request_headers.len(),
                    text_sidecar_rel: maybe_extract_pdf_sidecar(
                        repo_root,
                        &output_path,
                        args.extract_pdf_sidecar,
                    )?,
                };
                println!(
                    "downloaded session_id={} output={} backend={} bytes={} sha256={}",
                    capsule.session_id,
                    to_repo_display_path(repo_root, &output_path),
                    result.backend.as_str(),
                    result.bytes,
                    result.sha256.clone().unwrap_or_default()
                );
                writeback_session_outcome(repo_root, &capsule, &outcome)?;
                if capsule.search_target_id.is_some() {
                    queue_updates.push(build_queue_update(repo_root, &capsule, &outcome));
                }
                if let (Some(project_api), Some(crosswalk)) = (&project_api, &project_api_crosswalk)
                {
                    let binding = resolve_crosswalk_binding(
                        crosswalk,
                        capsule.search_target_id.as_deref(),
                        capsule.source_id.as_deref(),
                    );
                    journal_rows.push(journal_row_from_consume_outcome(
                        repo_root,
                        &project_api.repo_root,
                        &capsule,
                        &outcome,
                        binding,
                        project_cache_root.as_deref(),
                    ));
                }
                report_rows.push(ConsumeReportRow::success(
                    repo_root,
                    &capsule,
                    &output_path,
                    request_headers.len(),
                    &result,
                ));
            }
            Err(error) => {
                failed += 1;
                let outcome = ConsumeOutcome::Failed {
                    output_path: output_path.clone(),
                    error: error.to_string(),
                    header_count: request_headers.len(),
                };
                println!(
                    "download_failed session_id={} output={} error={}",
                    capsule.session_id,
                    to_repo_display_path(repo_root, &output_path),
                    error
                );
                writeback_session_outcome(repo_root, &capsule, &outcome)?;
                if capsule.search_target_id.is_some() {
                    queue_updates.push(build_queue_update(repo_root, &capsule, &outcome));
                }
                if let (Some(project_api), Some(crosswalk)) = (&project_api, &project_api_crosswalk)
                {
                    let binding = resolve_crosswalk_binding(
                        crosswalk,
                        capsule.search_target_id.as_deref(),
                        capsule.source_id.as_deref(),
                    );
                    journal_rows.push(journal_row_from_consume_outcome(
                        repo_root,
                        &project_api.repo_root,
                        &capsule,
                        &outcome,
                        binding,
                        project_cache_root.as_deref(),
                    ));
                }
                report_rows.push(ConsumeReportRow::failure(
                    repo_root,
                    &capsule,
                    &output_path,
                    request_headers.len(),
                    &error.to_string(),
                ));
            }
        }
    }

    if let Some(queue_path) = &args.search_queue_sync {
        sync_search_queue(repo_root, queue_path, &queue_updates)?;
        println!(
            "search_queue_synced={}",
            to_repo_display_path(repo_root, &repo_path(repo_root, queue_path))
        );
    }

    if let Some(project_api) = &project_api
        && !journal_rows.is_empty()
    {
        append_acquisition_journal_rows(&project_api.acquisition_journal_path, &journal_rows)?;
        println!(
            "acquisition_journal_synced={}",
            project_api
                .acquisition_journal_path
                .strip_prefix(&project_api.repo_root)
                .unwrap_or(&project_api.acquisition_journal_path)
                .display()
        );
        if args.reconcile_project_api
            && journal_rows
                .iter()
                .any(|row| !row.project_artifact_rel.is_empty())
        {
            run_cd_cache_reconcile(repo_root, &project_api.repo_root)?;
        }
    }

    if let Some(report_out) = &args.report_out {
        let report_path = repo_path(repo_root, report_out);
        write_consume_report(&report_path, &report_rows)?;
        println!("report={}", to_repo_display_path(repo_root, &report_path));
    }

    println!(
        "consume_summary total={} succeeded={} failed={} skipped={}",
        report_rows.len(),
        succeeded,
        failed,
        skipped
    );

    if failed > 0 {
        bail!("consume-handoff recorded {failed} failed download attempts");
    }
    Ok(())
}

fn journal_parent_session_id(session: &AcquisitionSession) -> String {
    session
        .lineage
        .as_ref()
        .map(|lineage| lineage.parent_session_id.clone())
        .unwrap_or_else(|| session.session_id.clone())
}

fn intentional_project_artifact_rel(
    project_repo_root: &Path,
    project_cache_root: Option<&Path>,
    output_path: &Path,
) -> String {
    let Some(project_cache_root) = project_cache_root else {
        return String::new();
    };
    if output_path.starts_with(project_cache_root) && output_path.starts_with(project_repo_root) {
        project_relative_path(project_repo_root, output_path)
    } else {
        String::new()
    }
}

fn maybe_extract_pdf_sidecar(
    repo_root: &Path,
    output_path: &Path,
    enabled: bool,
) -> Result<Option<String>> {
    if !enabled {
        return Ok(None);
    }
    let is_pdf = output_path
        .extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("pdf"))
        .unwrap_or(false);
    if !is_pdf || !output_path.exists() {
        return Ok(None);
    }
    let markdown = PdfExtractor::extract_to_markdown(output_path)
        .with_context(|| format!("extract markdown from {}", output_path.display()))?;
    let sidecar_path = output_path.with_extension("md");
    fs::write(&sidecar_path, markdown)
        .with_context(|| format!("write {}", sidecar_path.display()))?;
    Ok(Some(to_repo_display_path(repo_root, &sidecar_path)))
}

fn run_cd_cache_reconcile(repo_root: &Path, project_api_root: &Path) -> Result<()> {
    let output = Command::new("cargo")
        .current_dir(repo_root)
        .args([
            "run",
            "-q",
            "-p",
            "gororoba_cli_data",
            "--bin",
            "cd-cache-reconcile",
            "--",
            "--project-api-root",
            project_api_root.to_string_lossy().as_ref(),
        ])
        .output()
        .with_context(|| format!("run cd-cache-reconcile for {}", project_api_root.display()))?;
    if !output.status.success() {
        bail!(
            "cd-cache-reconcile failed for {}\nstdout:\n{}\nstderr:\n{}",
            project_api_root.display(),
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !stdout.trim().is_empty() {
        print!("{stdout}");
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !stderr.trim().is_empty() {
        eprint!("{stderr}");
    }
    Ok(())
}

fn journal_row_from_record_session(
    local_repo_root: &Path,
    project_repo_root: &Path,
    manifest_path: &Path,
    session: &AcquisitionSession,
    binding: Option<&ProjectApiCrosswalkBinding>,
) -> AcquisitionJournalRow {
    let artifact_rel = session.evidence.artifact_rel.clone().unwrap_or_default();
    AcquisitionJournalRow {
        at_utc: session.updated_utc.clone(),
        action: "record".to_string(),
        session_id: session.session_id.clone(),
        parent_session_id: journal_parent_session_id(session),
        source_id: session.source.source_id.clone().unwrap_or_default(),
        search_target_id: session
            .search
            .as_ref()
            .map(|search| search.search_target_id.clone())
            .or_else(|| binding.map(|binding| binding.search_target_id.clone()))
            .unwrap_or_default(),
        status: session.status.as_str().to_string(),
        outcome: format!("recorded_{}", session.status.as_str()),
        url: session.source.url.clone().unwrap_or_default(),
        effective_url: session.evidence.effective_url.clone().unwrap_or_default(),
        artifact_rel: artifact_rel.clone(),
        project_artifact_rel: project_relative_path_from_display(
            local_repo_root,
            project_repo_root,
            &artifact_rel,
        ),
        bytes: session
            .evidence
            .bytes
            .map(|value| value.to_string())
            .unwrap_or_default(),
        sha256: session.evidence.sha256.clone().unwrap_or_default(),
        http_code: session
            .evidence
            .http_code
            .map(|code| code.to_string())
            .unwrap_or_default(),
        request_capsule_rel: session
            .evidence
            .request_capsule_rel
            .clone()
            .unwrap_or_default(),
        browser_trace_rel: session
            .evidence
            .browser_trace_rel
            .clone()
            .unwrap_or_default(),
        cookie_jar_rel: session.evidence.cookie_jar_rel.clone().unwrap_or_default(),
        storage_state_rel: session
            .evidence
            .storage_state_rel
            .clone()
            .unwrap_or_default(),
        profile_bundle_rel: session
            .evidence
            .profile_bundle_rel
            .clone()
            .unwrap_or_default(),
        transport_substrate: session.controller.transport_substrate.as_str().to_string(),
        requested_backend: session.controller.requested_backend.clone(),
        crosswalk_id: binding
            .map(|binding| binding.id.clone())
            .unwrap_or_default(),
        candidate_id: binding
            .and_then(|binding| binding.candidate_id.clone())
            .unwrap_or_default(),
        chronology_row_id: binding
            .and_then(|binding| binding.chronology_row_id.clone())
            .unwrap_or_default(),
        inventory_blocker_id: binding
            .and_then(|binding| binding.inventory_blocker_id.clone())
            .unwrap_or_default(),
        row_ledger_refs: binding
            .map(|binding| journal_multi_value(&binding.row_ledger_refs))
            .unwrap_or_default(),
        gap_ids: binding
            .map(|binding| journal_multi_value(&binding.gap_ids))
            .unwrap_or_default(),
        era_ids: binding
            .map(|binding| journal_multi_value(&binding.era_ids))
            .unwrap_or_default(),
        nomenclature_ids: binding
            .map(|binding| journal_multi_value(&binding.nomenclature_ids))
            .unwrap_or_default(),
        note: if latest_event_note(session).is_empty() {
            format!(
                "record_manifest={}",
                to_repo_display_path(local_repo_root, manifest_path)
            )
        } else {
            format!(
                "record_manifest={} | {}",
                to_repo_display_path(local_repo_root, manifest_path),
                latest_event_note(session)
            )
        },
    }
}

fn journal_row_from_consume_outcome(
    local_repo_root: &Path,
    project_repo_root: &Path,
    capsule: &RequestCapsule,
    outcome: &ConsumeOutcome,
    binding: Option<&ProjectApiCrosswalkBinding>,
    project_cache_root: Option<&Path>,
) -> AcquisitionJournalRow {
    let (artifact_rel, project_artifact_rel, bytes, sha256, http_code, outcome_name) = match outcome
    {
        ConsumeOutcome::Downloaded {
            output_path,
            result,
            ..
        } => (
            to_repo_display_path(local_repo_root, output_path),
            intentional_project_artifact_rel(project_repo_root, project_cache_root, output_path),
            result.bytes.to_string(),
            result.sha256.clone().unwrap_or_default(),
            result
                .http_code
                .map(|code| code.to_string())
                .unwrap_or_default(),
            "downloaded".to_string(),
        ),
        ConsumeOutcome::Failed {
            output_path, error, ..
        } => (
            to_repo_display_path(local_repo_root, output_path),
            String::new(),
            String::new(),
            String::new(),
            capsule
                .http_code
                .map(|code| code.to_string())
                .unwrap_or_default(),
            format!("failed:{error}"),
        ),
        ConsumeOutcome::SkippedExisting {
            output_path,
            bytes,
            sha256,
            ..
        } => (
            to_repo_display_path(local_repo_root, output_path),
            intentional_project_artifact_rel(project_repo_root, project_cache_root, output_path),
            bytes.to_string(),
            sha256.clone(),
            capsule
                .http_code
                .map(|code| code.to_string())
                .unwrap_or_default(),
            "skipped_existing".to_string(),
        ),
    };
    AcquisitionJournalRow {
        at_utc: utc_now(),
        action: "consume_handoff".to_string(),
        session_id: capsule.session_id.clone(),
        parent_session_id: capsule.parent_session_id.clone(),
        source_id: capsule.source_id.clone().unwrap_or_default(),
        search_target_id: capsule
            .search_target_id
            .clone()
            .or_else(|| binding.map(|binding| binding.search_target_id.clone()))
            .unwrap_or_default(),
        status: if matches!(outcome, ConsumeOutcome::Failed { .. }) {
            "blocked".to_string()
        } else {
            "downloaded".to_string()
        },
        outcome: outcome_name,
        url: capsule.url.clone(),
        effective_url: capsule.effective_url.clone().unwrap_or_default(),
        artifact_rel,
        project_artifact_rel,
        bytes,
        sha256,
        http_code,
        request_capsule_rel: capsule.request_capsule_rel.clone().unwrap_or_default(),
        browser_trace_rel: capsule.browser_trace_rel.clone().unwrap_or_default(),
        cookie_jar_rel: capsule.cookie_jar_rel.clone().unwrap_or_default(),
        storage_state_rel: capsule.storage_state_rel.clone().unwrap_or_default(),
        profile_bundle_rel: capsule.profile_bundle_rel.clone().unwrap_or_default(),
        transport_substrate: capsule.transport_substrate.clone(),
        requested_backend: capsule.requested_backend.clone(),
        crosswalk_id: binding
            .map(|binding| binding.id.clone())
            .unwrap_or_default(),
        candidate_id: binding
            .and_then(|binding| binding.candidate_id.clone())
            .unwrap_or_default(),
        chronology_row_id: binding
            .and_then(|binding| binding.chronology_row_id.clone())
            .unwrap_or_default(),
        inventory_blocker_id: binding
            .and_then(|binding| binding.inventory_blocker_id.clone())
            .unwrap_or_default(),
        row_ledger_refs: binding
            .map(|binding| journal_multi_value(&binding.row_ledger_refs))
            .unwrap_or_default(),
        gap_ids: binding
            .map(|binding| journal_multi_value(&binding.gap_ids))
            .unwrap_or_default(),
        era_ids: binding
            .map(|binding| journal_multi_value(&binding.era_ids))
            .unwrap_or_default(),
        nomenclature_ids: binding
            .map(|binding| journal_multi_value(&binding.nomenclature_ids))
            .unwrap_or_default(),
        note: capsule.note.clone(),
    }
}

fn build_stack(policy_registry: &Path) -> Result<DownloadStack> {
    let stack = DownloadStack::default();
    if policy_registry.exists() {
        let policies = load_host_policy_registry(policy_registry)?;
        Ok(stack.with_host_policies(policies))
    } else {
        Ok(stack)
    }
}

fn write_session(
    repo_root: &Path,
    manifest_path: &Path,
    session: &AcquisitionSession,
) -> Result<()> {
    if let Some(parent) = manifest_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let body = toml::to_string_pretty(session)?;
    fs::write(manifest_path, format!("{body}\n"))
        .with_context(|| format!("write {}", to_repo_display_path(repo_root, manifest_path)))
}

fn write_checklist(
    repo_root: &Path,
    checklist_path: &Path,
    manifest_path: &Path,
    session: &AcquisitionSession,
) -> Result<()> {
    if let Some(parent) = checklist_path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let body = render_checklist(repo_root, manifest_path, session);
    fs::write(checklist_path, body)
        .with_context(|| format!("write {}", to_repo_display_path(repo_root, checklist_path)))
}

fn load_session(path: &Path) -> Result<AcquisitionSession> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parse {}", path.display()))
}

fn resolve_session_manifest(repo_root: &Path, sessions_dir: &Path, input: &Path) -> PathBuf {
    let direct = repo_path(repo_root, input);
    if direct.is_dir() {
        return direct.join("session.toml");
    }
    if direct.exists() || direct.extension().is_some() {
        return direct;
    }
    sessions_dir.join(input).join("session.toml")
}

fn render_checklist(
    repo_root: &Path,
    manifest_path: &Path,
    session: &AcquisitionSession,
) -> String {
    let mut out = String::new();
    out.push_str("# Human Acquire Session\n\n");
    out.push_str(&format!("- Session ID: `{}`\n", session.session_id));
    out.push_str(&format!("- Status: `{}`\n", session.status.as_str()));
    out.push_str(&format!("- Target: `{}`\n", session_target_label(session)));
    if let Some(url) = &session.source.url {
        out.push_str(&format!("- URL: `{url}`\n"));
    }
    if let Some(title) = &session.source.title {
        out.push_str(&format!("- Title: `{title}`\n"));
    }
    if let Some(source_id) = &session.source.source_id {
        out.push_str(&format!("- Source ID: `{source_id}`\n"));
    }
    if let Some(site) = &session.source.site {
        out.push_str(&format!("- Site: `{site}`\n"));
    }
    if let Some(access_class) = &session.source.access_class {
        out.push_str(&format!("- Access Class: `{access_class}`\n"));
    }
    out.push_str(&format!(
        "- Transport Substrate: `{}`\n",
        session.controller.transport_substrate.as_str()
    ));
    out.push_str(&format!(
        "- Requested Backend: `{}`\n",
        session.controller.requested_backend
    ));
    if let Some(dest_rel) = &session.controller.dest_rel {
        out.push_str(&format!("- Destination: `{dest_rel}`\n"));
    }
    out.push_str(&format!(
        "- Manifest: `{}`\n",
        to_repo_display_path(repo_root, manifest_path)
    ));
    if let Some(lineage) = &session.lineage {
        out.push('\n');
        out.push_str("## Lineage\n\n");
        out.push_str(&format!(
            "- Parent Session ID: `{}`\n",
            lineage.parent_session_id
        ));
        out.push_str(&format!(
            "- Parent Manifest: `{}`\n",
            lineage.parent_manifest
        ));
        out.push_str(&format!("- Relation: `{}`\n", lineage.relation));
    }
    if let Some(search) = &session.search {
        out.push('\n');
        out.push_str("## Search Context\n\n");
        out.push_str(&format!("- Queue: `{}`\n", search.queue_path));
        out.push_str(&format!(
            "- Search Target ID: `{}`\n",
            search.search_target_id
        ));
        out.push_str(&format!("- Window: `{}`\n", search.window));
        out.push_str(&format!("- Priority: `{}`\n", search.priority));
        out.push_str(&format!("- Kind: `{}`\n", search.kind));
        out.push_str(&format!("- Queue Status: `{}`\n", search.status));
        out.push_str(&format!("- Why Now: `{}`\n", search.why_now));
        if !search.preferred_lanes.is_empty() {
            out.push_str("- Preferred Lanes:\n");
            for lane in &search.preferred_lanes {
                out.push_str(&format!("  - `{lane}`\n"));
            }
        }
        if !search.query_seeds.is_empty() {
            out.push_str("- Query Seeds:\n");
            for seed in &search.query_seeds {
                out.push_str(&format!("  - `{seed}`\n"));
            }
        }
    }
    if let Some(route) = &session.route {
        out.push('\n');
        out.push_str("## Route\n\n");
        out.push_str(&format!("- Scheme: `{}`\n", route.scheme));
        if let Some(host) = &route.host {
            out.push_str(&format!("- Host: `{host}`\n"));
        }
        out.push_str(&format!("- Retry Class: `{}`\n", route.retry_class));
        if let Some(policy_name) = &route.policy_name {
            out.push_str(&format!("- Policy: `{policy_name}`\n"));
        }
        if !route.backends.is_empty() {
            out.push_str(&format!(
                "- Candidate Backends: `{}`\n",
                route.backends.join(", ")
            ));
        }
    }
    if let Some(probe) = &session.probe {
        out.push('\n');
        out.push_str("## Probe\n\n");
        out.push_str(&format!("- Attempted UTC: `{}`\n", probe.attempted_utc));
        if let Some(capabilities) = &probe.capabilities {
            out.push_str(&format!("- Surface: `{}`\n", capabilities.surface));
            if let Some(code) = capabilities.http_code {
                out.push_str(&format!("- HTTP Code: `{code}`\n"));
            }
            if let Some(content_type) = &capabilities.content_type {
                out.push_str(&format!("- Content Type: `{content_type}`\n"));
            }
            if let Some(final_url) = &capabilities.final_url {
                out.push_str(&format!("- Final URL: `{final_url}`\n"));
            }
        }
        if let Some(final_error) = &probe.final_error {
            out.push_str(&format!("- Final Error: `{final_error}`\n"));
        }
    }
    render_evidence_section(&mut out, session);
    out.push('\n');
    out.push_str("## Operator Loop\n\n");
    if session.source.url.is_some() {
        out.push_str(
            "1. Decide whether the direct transport lane is good enough or whether a headed browser is needed.\n",
        );
        out.push_str(
            "2. If a live browser is needed, let the human clear the page normally and keep traces, cookies, and storage state together.\n",
        );
        out.push_str(
            "3. Save the acquired artifact into the intended destination path when possible.\n",
        );
        out.push_str(
            "4. Record the result with `human-acquire record --session <id-or-path> --status <status>` so the session stays authoritative.\n",
        );
    } else {
        out.push_str(
            "1. Resolve the first concrete landing page, holder lane, or mirror candidate from the preferred lanes and query seeds.\n",
        );
        out.push_str(
            "2. Once a live retrieval lane exists, either promote this dossier into URL child sessions or continue attaching search evidence here.\n",
        );
        out.push_str(
            "3. Keep browser traces, storage state, and correspondence attached here so blocker state stays structured.\n",
        );
        out.push_str(
            "4. Record progress with `human-acquire record --session <id-or-path> --status <status>` so unresolved blockers do not collapse back into prose.\n",
        );
    }
    if session.controller.transport_substrate == TransportSubstrate::Silksurf {
        out.push_str(
            "5. Treat silksurf as an optional transport substrate after the human clears the page; it is not required for the session contract itself.\n",
        );
    }
    if !session.events.is_empty() {
        out.push('\n');
        out.push_str("## Event Log\n\n");
        for event in &session.events {
            out.push_str(&format!(
                "- `{}` `{}` `{}` {}\n",
                event.at_utc,
                event.action,
                event.status.as_str(),
                event.note.clone().unwrap_or_default()
            ));
        }
    }
    out
}

fn render_evidence_section(out: &mut String, session: &AcquisitionSession) {
    if session.evidence.artifact_rel.is_none()
        && session.evidence.text_sidecar_rel.is_none()
        && session.evidence.browser_trace_rel.is_none()
        && session.evidence.cookie_jar_rel.is_none()
        && session.evidence.storage_state_rel.is_none()
        && session.evidence.profile_bundle_rel.is_none()
        && session.evidence.request_capsule_rel.is_none()
        && session.evidence.host_scope.is_none()
        && session.evidence.effective_url.is_none()
        && session.evidence.http_code.is_none()
    {
        return;
    }
    out.push('\n');
    out.push_str("## Evidence\n\n");
    if let Some(host_scope) = &session.evidence.host_scope {
        out.push_str(&format!("- Host Scope: `{host_scope}`\n"));
    }
    if let Some(browser_trace_rel) = &session.evidence.browser_trace_rel {
        out.push_str(&format!("- Browser Trace: `{browser_trace_rel}`\n"));
    }
    if let Some(cookie_jar_rel) = &session.evidence.cookie_jar_rel {
        out.push_str(&format!("- Cookie Jar: `{cookie_jar_rel}`\n"));
    }
    if let Some(storage_state_rel) = &session.evidence.storage_state_rel {
        out.push_str(&format!("- Storage State: `{storage_state_rel}`\n"));
    }
    if let Some(profile_bundle_rel) = &session.evidence.profile_bundle_rel {
        out.push_str(&format!("- Profile Bundle: `{profile_bundle_rel}`\n"));
    }
    if let Some(request_capsule_rel) = &session.evidence.request_capsule_rel {
        out.push_str(&format!("- Request Capsule: `{request_capsule_rel}`\n"));
    }
    if let Some(effective_url) = &session.evidence.effective_url {
        out.push_str(&format!("- Effective URL: `{effective_url}`\n"));
    }
    if let Some(http_code) = session.evidence.http_code {
        out.push_str(&format!("- HTTP Code: `{http_code}`\n"));
    }
    if let Some(artifact_rel) = &session.evidence.artifact_rel {
        out.push_str(&format!("- Artifact: `{artifact_rel}`\n"));
    }
    if let Some(text_sidecar_rel) = &session.evidence.text_sidecar_rel {
        out.push_str(&format!("- Text Sidecar: `{text_sidecar_rel}`\n"));
    }
}

fn print_stage_summary(
    repo_root: &Path,
    manifest_path: &Path,
    checklist_path: &Path,
    session: &AcquisitionSession,
) {
    println!("session_id={}", session.session_id);
    println!("status={}", session.status.as_str());
    println!("target={}", session_target_label(session));
    println!("url={}", session.source.url.clone().unwrap_or_default());
    if let Some(lineage) = &session.lineage {
        println!("parent_session_id={}", lineage.parent_session_id);
        println!("parent_manifest={}", lineage.parent_manifest);
        println!("lineage_relation={}", lineage.relation);
    }
    println!(
        "transport_substrate={}",
        session.controller.transport_substrate.as_str()
    );
    println!(
        "requested_backend={}",
        session.controller.requested_backend.as_str()
    );
    if let Some(search) = &session.search {
        println!("search_target_id={}", search.search_target_id);
        println!("search_window={}", search.window);
        println!("search_priority={}", search.priority);
        println!("search_status={}", search.status);
    }
    if let Some(route) = &session.route {
        println!("scheme={}", route.scheme);
        println!("host={}", route.host.clone().unwrap_or_default());
        println!("retry_class={}", route.retry_class);
        println!("policy={}", route.policy_name.clone().unwrap_or_default());
        println!("backends={}", route.backends.join(","));
    }
    println!(
        "manifest={}",
        to_repo_display_path(repo_root, manifest_path)
    );
    println!(
        "checklist={}",
        to_repo_display_path(repo_root, checklist_path)
    );
    if let Some(dest_rel) = &session.controller.dest_rel {
        println!("dest={dest_rel}");
    }
    if let Some(host_scope) = &session.evidence.host_scope {
        println!("host_scope={host_scope}");
    }
    if let Some(browser_trace_rel) = &session.evidence.browser_trace_rel {
        println!("browser_trace={browser_trace_rel}");
    }
    if let Some(cookie_jar_rel) = &session.evidence.cookie_jar_rel {
        println!("cookie_jar={cookie_jar_rel}");
    }
    if let Some(storage_state_rel) = &session.evidence.storage_state_rel {
        println!("storage_state={storage_state_rel}");
    }
    if let Some(profile_bundle_rel) = &session.evidence.profile_bundle_rel {
        println!("profile_bundle={profile_bundle_rel}");
    }
    if let Some(request_capsule_rel) = &session.evidence.request_capsule_rel {
        println!("request_capsule={request_capsule_rel}");
    }
    if let Some(effective_url) = &session.evidence.effective_url {
        println!("effective_url={effective_url}");
    }
    if let Some(http_code) = session.evidence.http_code {
        println!("http_code={http_code}");
    }
    if let Some(artifact_rel) = &session.evidence.artifact_rel {
        println!("artifact={artifact_rel}");
    }
    if let Some(text_sidecar_rel) = &session.evidence.text_sidecar_rel {
        println!("text_sidecar={text_sidecar_rel}");
    }
}

fn resolve_repo_root(path: &Path) -> PathBuf {
    if path == Path::new(".") {
        source_provenance::default_repo_root()
    } else {
        path.to_path_buf()
    }
}

fn repo_path(repo_root: &Path, maybe_relative: &Path) -> PathBuf {
    if maybe_relative.is_absolute() {
        maybe_relative.to_path_buf()
    } else {
        repo_root.join(maybe_relative)
    }
}

fn to_repo_display_path(repo_root: &Path, path: &Path) -> String {
    path.strip_prefix(repo_root)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn display_input_path(repo_root: &Path, input: &Path) -> String {
    let resolved = repo_path(repo_root, input);
    to_repo_display_path(repo_root, &resolved)
}

fn session_target_label(session: &AcquisitionSession) -> String {
    session
        .source
        .url
        .clone()
        .or_else(|| session.source.title.clone())
        .or_else(|| session.source.source_id.clone())
        .or_else(|| {
            session
                .search
                .as_ref()
                .map(|search| search.search_target_id.clone())
        })
        .unwrap_or_else(|| session.session_id.clone())
}

fn derive_session_id(source_id: Option<&str>, title: Option<&str>, url: &str) -> String {
    let stamp = Utc::now().format("%Y%m%dT%H%M%SZ").to_string();
    let host = Url::parse(url)
        .ok()
        .and_then(|parsed| parsed.host_str().map(sanitize_token))
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "site".to_string());
    let seed = source_id
        .map(sanitize_token)
        .filter(|value| !value.is_empty())
        .or_else(|| title.map(sanitize_token).filter(|value| !value.is_empty()))
        .unwrap_or_else(|| sanitize_token(url));
    let suffix = if seed.is_empty() {
        "artifact".to_string()
    } else {
        seed
    };
    format!("{stamp}_{host}_{suffix}")
}

fn sanitize_token(value: &str) -> String {
    let mut out = String::new();
    let mut prev_underscore = false;
    for ch in value.chars() {
        let mapped = match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' => {
                prev_underscore = false;
                ch.to_ascii_lowercase()
            }
            _ => {
                if prev_underscore {
                    continue;
                }
                prev_underscore = true;
                '_'
            }
        };
        out.push(mapped);
    }
    out.trim_matches('_').to_string()
}

fn join_notes(notes: &[String]) -> Option<String> {
    if notes.is_empty() {
        None
    } else {
        Some(notes.join(" | "))
    }
}

fn join_optional_notes(notes: &[Option<String>]) -> Option<String> {
    let joined = notes
        .iter()
        .filter_map(|note| note.as_ref())
        .map(|note| note.trim())
        .filter(|note| !note.is_empty())
        .collect::<Vec<_>>();
    if joined.is_empty() {
        None
    } else {
        Some(joined.join(" | "))
    }
}

fn collect_promotion_urls(repo_root: &Path, args: &PromoteArgs) -> Result<Vec<String>> {
    let mut urls = Vec::new();
    for url in &args.url {
        let trimmed = url.trim();
        if !trimmed.is_empty() {
            urls.push(trimmed.to_string());
        }
    }
    if let Some(url_file) = &args.url_file {
        let path = repo_path(repo_root, url_file);
        let raw = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        for line in raw.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            urls.push(trimmed.to_string());
        }
    }
    Ok(urls)
}

fn collect_import_dossier_paths(
    repo_root: &Path,
    args: &ImportDossierArgs,
) -> Result<Vec<PathBuf>> {
    let mut seen = HashSet::new();
    let mut paths = Vec::new();
    for dossier in &args.dossier {
        let resolved = repo_path(repo_root, dossier);
        if seen.insert(resolved.clone()) {
            paths.push(resolved);
        }
    }
    for manifest in &args.batch_manifest {
        let manifest_path = repo_path(repo_root, manifest);
        let batch = load_dossier_batch_manifest(&manifest_path)?;
        for entry in batch.entries {
            let resolved = resolve_manifest_entry_path(&manifest_path, &entry.dossier_json);
            if seen.insert(resolved.clone()) {
                paths.push(resolved);
            }
        }
    }
    Ok(paths)
}

fn ensure_parent_session_for_dossier(
    repo_root: &Path,
    sessions_dir: &Path,
    dossier: &ResearchDossier,
) -> Result<PathBuf> {
    let session_id = sanitize_token(&format!(
        "{}_{}",
        dossier.project_id, dossier.search_target_id
    ));
    let manifest_path = sessions_dir.join(&session_id).join("session.toml");
    if manifest_path.exists() {
        return Ok(manifest_path);
    }

    let queue_path = PathBuf::from(&dossier.search_queue_path);
    let queue = load_search_queue(&queue_path)?;
    let target = queue
        .search_target
        .iter()
        .find(|target| target.id == dossier.search_target_id)
        .with_context(|| {
            format!(
                "search target {} missing from {}",
                dossier.search_target_id, dossier.search_queue_path
            )
        })?;
    let checklist_path = manifest_path
        .parent()
        .context("session manifest must have a parent directory")?
        .join("checklist.md");
    let now = utc_now();
    let session = AcquisitionSession {
        format_version: 1,
        session_id: session_id.clone(),
        created_utc: now.clone(),
        updated_utc: now.clone(),
        status: SessionStatus::Staged,
        source: SourceTarget {
            url: None,
            source_id: Some(target.id.clone()),
            title: Some(target.title.clone()),
            site: None,
            access_class: None,
        },
        lineage: None,
        search: Some(SearchContext {
            queue_path: dossier.search_queue_path.clone(),
            project_id: Some(dossier.project_id.clone()),
            search_target_id: target.id.clone(),
            window: target.window.clone(),
            priority: target.priority.clone(),
            kind: target.kind.clone(),
            status: target.status.clone(),
            why_now: target.why_now.clone(),
            preferred_lanes: target.preferred_lanes.clone(),
            query_seeds: target.query_seeds.clone(),
        }),
        controller: ControllerPlan {
            policy_registry_rel: None,
            sessions_dir_rel: to_repo_display_path(repo_root, sessions_dir),
            dest_rel: None,
            transport_substrate: TransportSubstrate::Auto,
            requested_backend: DownloadBackend::Auto.as_str().to_string(),
            probe_bytes: DEFAULT_PROBE_BYTES,
            probe_requested: false,
        },
        route: None,
        probe: None,
        evidence: EvidenceState::default(),
        events: vec![SessionEvent {
            at_utc: now,
            action: "import_dossier_stage_parent".to_string(),
            status: SessionStatus::Staged,
            note: Some(format!(
                "staged from dossier {}",
                to_repo_display_path(
                    repo_root,
                    &repo_path(repo_root, Path::new(&dossier.search_queue_path))
                )
            )),
        }],
    };
    write_session(repo_root, &manifest_path, &session)?;
    write_checklist(repo_root, &checklist_path, &manifest_path, &session)?;
    Ok(manifest_path)
}

fn select_dossier_suggestions<'a>(
    dossier: &'a ResearchDossier,
    args: &ImportDossierArgs,
) -> Result<Vec<&'a StageSuggestion>> {
    let mut selected = dossier
        .stage_suggestions
        .iter()
        .filter(|suggestion| {
            if let Some(max_rank) = args.max_rank {
                suggestion.rank <= max_rank
            } else {
                true
            }
        })
        .collect::<Vec<_>>();

    if !args.rank.is_empty() {
        let rank_set = args.rank.iter().copied().collect::<HashSet<_>>();
        selected.retain(|suggestion| rank_set.contains(&suggestion.rank));
    } else if !args.all_suggestions {
        selected.retain(|suggestion| suggestion.default_selected);
        if selected.is_empty()
            && let Some(first) = dossier.stage_suggestions.first()
        {
            selected.push(first);
        }
    }

    let available_ranks = dossier
        .stage_suggestions
        .iter()
        .map(|suggestion| suggestion.rank.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    for rank in &args.rank {
        if !dossier
            .stage_suggestions
            .iter()
            .any(|suggestion| suggestion.rank == *rank)
        {
            bail!(
                "dossier {} does not contain rank {}; available ranks: {}",
                dossier.search_target_id,
                rank,
                available_ranks
            );
        }
    }
    Ok(selected)
}

fn dossier_suggestion_to_promote_spec(
    dossier: &ResearchDossier,
    suggestion: &StageSuggestion,
    notes: &[String],
) -> PromoteSpec {
    PromoteSpec {
        url: suggestion.candidate_url.clone(),
        source_id: Some(suggestion.source.clone()),
        title: Some(suggestion.paper_title.clone()),
        site: Some(suggestion.host_class.clone()),
        access_class: Some(suggestion.route_class.clone()),
        note: join_optional_notes(&[
            Some(format!(
                "dossier_search_target_id={}",
                dossier.search_target_id
            )),
            Some(format!(
                "dossier_suggestion_id={}",
                suggestion.suggestion_id
            )),
            Some(format!("canonical_id={}", suggestion.canonical_id)),
            Some(format!("source_family={}", suggestion.source_family)),
            Some(format!("route_class={}", suggestion.route_class)),
            join_notes(notes),
        ]),
    }
}

fn load_search_queue(path: &Path) -> Result<SearchQueueFile> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parse {}", path.display()))
}

fn target_matches_filters(target: &SearchQueueTarget, args: &StageQueueArgs) -> bool {
    if args.critical_only && target.priority != "critical" {
        return false;
    }
    if !args.id_filters.is_empty() && !args.id_filters.iter().any(|id| id == &target.id) {
        return false;
    }
    if !args.window.is_empty() && !args.window.iter().any(|window| window == &target.window) {
        return false;
    }
    if !args.priority.is_empty()
        && !args
            .priority
            .iter()
            .any(|priority| priority == &target.priority)
    {
        return false;
    }
    if !args.status.is_empty() && !args.status.iter().any(|status| status == &target.status) {
        return false;
    }
    true
}

fn queue_stage_note(target: &SearchQueueTarget, notes: &[String]) -> String {
    let mut parts = vec![format!("queued from {}", target.id)];
    if !notes.is_empty() {
        parts.push(notes.join(" | "));
    }
    parts.join(" | ")
}

fn promote_note(parent_session_id: &str, notes: &[String]) -> String {
    let mut parts = vec![format!("promoted from {parent_session_id}")];
    if !notes.is_empty() {
        parts.push(notes.join(" | "));
    }
    parts.join(" | ")
}

fn render_optional_suffix(value: &Option<String>) -> String {
    value
        .as_ref()
        .map(|value| format!(" | {value}"))
        .unwrap_or_default()
}

fn allocate_child_session_id(
    sessions_dir: &Path,
    parent_session_id: &str,
    url: &str,
    ordinal: usize,
) -> String {
    let base = derive_child_session_id(parent_session_id, url, ordinal);
    let mut candidate = base.clone();
    let mut counter = 2_usize;
    while sessions_dir.join(&candidate).join("session.toml").exists() {
        candidate = format!("{base}_{counter}");
        counter += 1;
    }
    candidate
}

fn derive_child_session_id(parent_session_id: &str, url: &str, ordinal: usize) -> String {
    let parsed = Url::parse(url).ok();
    let host = parsed
        .as_ref()
        .and_then(|parsed| parsed.host_str())
        .map(sanitize_token)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "site".to_string());
    let leaf = parsed
        .as_ref()
        .and_then(|parsed| parsed.path_segments())
        .and_then(|mut segments| segments.rfind(|segment| !segment.is_empty()))
        .map(sanitize_token)
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| format!("child_{:02}", ordinal + 1));
    sanitize_token(&format!("{parent_session_id}_{host}_{leaf}"))
}

fn derive_host_scope(url: &str, effective_url: Option<&str>) -> Option<String> {
    effective_url
        .and_then(|value| Url::parse(value).ok())
        .or_else(|| Url::parse(url).ok())
        .and_then(|parsed| parsed.host_str().map(sanitize_token))
        .filter(|value| !value.is_empty())
}

fn utc_now() -> String {
    Utc::now().to_rfc3339()
}

fn hash_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 8192];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn session_is_handoff_ready(session: &AcquisitionSession) -> bool {
    session.source.url.is_some()
        && (session.evidence.cookie_jar_rel.is_some()
            || session.evidence.storage_state_rel.is_some()
            || session.evidence.browser_trace_rel.is_some()
            || session.evidence.effective_url.is_some()
            || session.evidence.http_code.is_some())
}

fn write_handoff_rows(path: &Path, rows: &[HandoffRow]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let mut body = String::from(HandoffRow::header());
    body.push('\n');
    for row in rows {
        body.push_str(&row.to_tsv_line());
        body.push('\n');
    }
    fs::write(path, body).with_context(|| format!("write {}", path.display()))
}

fn load_profile_bundle(
    repo_root: &Path,
    profile_root_rel: &Path,
    host_scope: Option<&str>,
) -> Result<Option<BrowserProfileBundle>> {
    let Some(host_scope) = host_scope else {
        return Ok(None);
    };
    let bundle_manifest = repo_path(repo_root, profile_root_rel)
        .join(host_scope)
        .join("bundle.toml");
    if !bundle_manifest.exists() {
        return Ok(None);
    }
    let body = fs::read_to_string(&bundle_manifest)
        .with_context(|| format!("read {}", bundle_manifest.display()))?;
    let bundle =
        toml::from_str(&body).with_context(|| format!("parse {}", bundle_manifest.display()))?;
    Ok(Some(bundle))
}

fn write_profile_bundle(
    repo_root: &Path,
    profile_root_rel: &Path,
    session: &AcquisitionSession,
    child_manifest: &Path,
) -> Result<Option<PathBuf>> {
    let Some(host_scope) = session.evidence.host_scope.clone().or_else(|| {
        derive_host_scope(
            session.source.url.as_deref().unwrap_or_default(),
            session.evidence.effective_url.as_deref(),
        )
    }) else {
        return Ok(None);
    };
    let bundle_root = repo_path(repo_root, profile_root_rel).join(&host_scope);
    fs::create_dir_all(&bundle_root)
        .with_context(|| format!("create {}", bundle_root.display()))?;
    let bundle_manifest = bundle_root.join("bundle.toml");
    let bundle = BrowserProfileBundle {
        schema_version: 1,
        host_scope,
        bundle_kind: "headed_browser_profile".to_string(),
        bundle_root_rel: to_repo_display_path(repo_root, &bundle_root),
        latest_session_id: session.session_id.clone(),
        latest_manifest_rel: to_repo_display_path(repo_root, child_manifest),
        cookie_jar_rel: session.evidence.cookie_jar_rel.clone(),
        storage_state_rel: session.evidence.storage_state_rel.clone(),
        browser_trace_rel: session.evidence.browser_trace_rel.clone(),
        effective_url: session.evidence.effective_url.clone(),
        updated_utc: utc_now(),
    };
    let body = toml::to_string_pretty(&bundle)?;
    fs::write(&bundle_manifest, format!("{body}\n"))
        .with_context(|| format!("write {}", bundle_manifest.display()))?;
    Ok(Some(bundle_manifest))
}

fn write_request_capsule(path: &Path, capsule: &RequestCapsule) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let body = serde_json::to_string_pretty(capsule)?;
    fs::write(path, format!("{body}\n")).with_context(|| format!("write {}", path.display()))
}

fn load_request_capsule_path(repo_root: &Path, path: &Path) -> Result<RequestCapsule> {
    let resolved = repo_path(repo_root, path);
    let body =
        fs::read_to_string(&resolved).with_context(|| format!("read {}", resolved.display()))?;
    serde_json::from_str(&body).with_context(|| format!("parse {}", resolved.display()))
}

fn load_handoff_jobs(repo_root: &Path, handoff_path: &Path) -> Result<Vec<RequestCapsule>> {
    let resolved = repo_path(repo_root, handoff_path);
    let body =
        fs::read_to_string(&resolved).with_context(|| format!("read {}", resolved.display()))?;
    let mut lines = body.lines();
    let header = lines
        .next()
        .context("handoff TSV missing header row")?
        .split('\t')
        .collect::<Vec<_>>();
    let mut jobs = Vec::new();
    for line in lines.filter(|line| !line.trim().is_empty()) {
        let row = HandoffRow::from_tsv_record(&header, line)?;
        if !row.request_capsule_rel.is_empty() {
            let capsule_path = PathBuf::from(&row.request_capsule_rel);
            if repo_path(repo_root, &capsule_path).exists() {
                jobs.push(load_request_capsule_path(repo_root, &capsule_path)?);
                continue;
            }
        }
        jobs.push(RequestCapsule::from_handoff_row(repo_root, &row));
    }
    Ok(jobs)
}

fn parse_header_specs(headers: &[String]) -> Result<Vec<(String, String)>> {
    headers
        .iter()
        .map(|header| {
            let (name, value) = header
                .split_once(':')
                .context("header must be in 'Name: Value' form")?;
            let name = name.trim();
            let value = value.trim();
            if name.is_empty() || value.is_empty() {
                bail!("header must be in 'Name: Value' form");
            }
            Ok((name.to_string(), value.to_string()))
        })
        .collect()
}

fn request_headers_for_capsule(
    repo_root: &Path,
    capsule: &RequestCapsule,
    explicit_headers: &[(String, String)],
) -> Vec<(String, String)> {
    let mut headers = capsule
        .header_hints
        .iter()
        .map(|hint| (hint.name.clone(), hint.value.clone()))
        .collect::<Vec<_>>();
    if !headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("Referer"))
        && let Some(effective_url) = &capsule.effective_url
    {
        headers.push(("Referer".to_string(), effective_url.clone()));
    }
    if !headers
        .iter()
        .any(|(name, _)| name.eq_ignore_ascii_case("Cookie"))
        && let Some(cookie_header) = cookie_header_from_capsule(repo_root, capsule)
    {
        headers.push(("Cookie".to_string(), cookie_header));
    }
    for (name, value) in explicit_headers {
        headers.retain(|(existing, _)| !existing.eq_ignore_ascii_case(name));
        headers.push((name.clone(), value.clone()));
    }
    headers
}

fn handoff_output_path(output_root: &Path, capsule: &RequestCapsule) -> PathBuf {
    if let Some(dest_rel) = &capsule.dest_rel {
        let dest_path = PathBuf::from(dest_rel);
        if dest_path.is_absolute() {
            return dest_path;
        }
        return output_root.join(dest_path);
    }
    let leaf = Url::parse(&capsule.url)
        .ok()
        .and_then(|parsed| {
            parsed
                .path_segments()
                .and_then(|mut segments| segments.rfind(|segment| !segment.is_empty()))
                .map(str::to_string)
        })
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "artifact.bin".to_string());
    output_root.join(&capsule.session_id).join(leaf)
}

fn preferred_backend(capsule: &RequestCapsule, fallback: BackendArg) -> DownloadBackend {
    if fallback != BackendArg::Auto {
        return fallback.into();
    }
    DownloadBackend::parse(&capsule.requested_backend).unwrap_or(DownloadBackend::Auto)
}

fn load_profile_bundle_from_rel(
    repo_root: &Path,
    bundle_rel: Option<&str>,
) -> Option<BrowserProfileBundle> {
    let bundle_rel = bundle_rel?;
    let bundle_path = repo_path(repo_root, Path::new(bundle_rel));
    let body = fs::read_to_string(&bundle_path).ok()?;
    toml::from_str(&body).ok()
}

fn build_header_hints(
    repo_root: &Path,
    url: &str,
    effective_url: Option<&str>,
    cookie_jar_rel: Option<&str>,
    browser_trace_rel: Option<&str>,
    storage_state_rel: Option<&str>,
) -> Vec<RequestHeaderHint> {
    let mut hints = Vec::new();
    hints.extend(trace_header_hints(
        repo_root,
        browser_trace_rel,
        url,
        effective_url,
    ));
    hints.extend(storage_state_header_hints(
        repo_root,
        storage_state_rel,
        url,
        effective_url,
    ));
    if let Some(effective_url) = effective_url {
        hints.push(RequestHeaderHint {
            name: "Referer".to_string(),
            value: effective_url.to_string(),
            source: "effective_url".to_string(),
        });
        if let Some(origin) = origin_from_url(effective_url) {
            hints.push(RequestHeaderHint {
                name: "Origin".to_string(),
                value: origin,
                source: "effective_url".to_string(),
            });
        }
    }
    if let Some(cookie_header) =
        cookie_header_from_jar(repo_root, cookie_jar_rel, url, effective_url)
    {
        hints.push(RequestHeaderHint {
            name: "Cookie".to_string(),
            value: cookie_header,
            source: "cookie_jar".to_string(),
        });
    }
    dedupe_header_hints(hints)
}

fn cookie_header_from_capsule(repo_root: &Path, capsule: &RequestCapsule) -> Option<String> {
    cookie_header_from_jar(
        repo_root,
        capsule.cookie_jar_rel.as_deref(),
        &capsule.url,
        capsule.effective_url.as_deref(),
    )
}

fn cookie_header_from_jar(
    repo_root: &Path,
    cookie_jar_rel: Option<&str>,
    url: &str,
    effective_url: Option<&str>,
) -> Option<String> {
    let cookie_jar_rel = cookie_jar_rel?;
    let host = request_host(url, effective_url)?;
    let cookie_path = repo_path(repo_root, Path::new(cookie_jar_rel));
    let body = fs::read_to_string(cookie_path).ok()?;
    parse_json_cookie_header(&body, &host).or_else(|| parse_netscape_cookie_header(&body, &host))
}

fn request_host(url: &str, effective_url: Option<&str>) -> Option<String> {
    effective_url
        .and_then(|value| Url::parse(value).ok())
        .or_else(|| Url::parse(url).ok())
        .and_then(|parsed| parsed.host_str().map(|host| host.to_ascii_lowercase()))
}

fn origin_from_url(url: &str) -> Option<String> {
    let parsed = Url::parse(url).ok()?;
    let host = parsed.host_str()?;
    match parsed.port() {
        Some(port) => Some(format!("{}://{}:{port}", parsed.scheme(), host)),
        None => Some(format!("{}://{}", parsed.scheme(), host)),
    }
}

fn parse_json_cookie_header(body: &str, host: &str) -> Option<String> {
    let value: serde_json::Value = serde_json::from_str(body).ok()?;
    let cookies = match value {
        serde_json::Value::Array(entries) => entries,
        serde_json::Value::Object(map) => map
            .get("cookies")
            .and_then(|cookies| cookies.as_array().cloned())
            .unwrap_or_default(),
        _ => return None,
    };
    let mut seen = HashSet::new();
    let mut parts = Vec::new();
    for cookie in cookies {
        let object = match cookie {
            serde_json::Value::Object(object) => object,
            _ => continue,
        };
        let Some(name) = object.get("name").and_then(|value| value.as_str()) else {
            continue;
        };
        let Some(value) = object.get("value").and_then(|value| value.as_str()) else {
            continue;
        };
        let domain = object
            .get("domain")
            .and_then(|value| value.as_str())
            .unwrap_or(host);
        if !cookie_domain_matches(host, domain) {
            continue;
        }
        if seen.insert(name.to_string()) {
            parts.push(format!("{name}={value}"));
        }
    }
    (!parts.is_empty()).then(|| parts.join("; "))
}

fn parse_netscape_cookie_header(body: &str, host: &str) -> Option<String> {
    let mut seen = HashSet::new();
    let mut parts = Vec::new();
    for line in body.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let fields = trimmed.split('\t').collect::<Vec<_>>();
        if fields.len() < 7 {
            continue;
        }
        let domain = fields[0];
        let name = fields[5];
        let value = fields[6];
        if name.is_empty() || !cookie_domain_matches(host, domain) {
            continue;
        }
        if seen.insert(name.to_string()) {
            parts.push(format!("{name}={value}"));
        }
    }
    (!parts.is_empty()).then(|| parts.join("; "))
}

fn cookie_domain_matches(host: &str, domain: &str) -> bool {
    let normalized = domain.trim().trim_start_matches('.').to_ascii_lowercase();
    host == normalized || host.ends_with(&format!(".{normalized}"))
}

fn trace_header_hints(
    repo_root: &Path,
    browser_trace_rel: Option<&str>,
    url: &str,
    effective_url: Option<&str>,
) -> Vec<RequestHeaderHint> {
    let Some(browser_trace_rel) = browser_trace_rel else {
        return Vec::new();
    };
    let trace_path = repo_path(repo_root, Path::new(browser_trace_rel));
    let Ok(body) = fs::read_to_string(trace_path) else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&body) else {
        return Vec::new();
    };
    let Some(entries) = value
        .get("log")
        .and_then(|log| log.get("entries"))
        .and_then(|entries| entries.as_array())
    else {
        return Vec::new();
    };
    let wanted_host = request_host(url, effective_url);
    let wanted_url = effective_url.unwrap_or(url);
    let mut selected = Vec::new();
    for entry in entries {
        let Some(request) = entry.get("request") else {
            continue;
        };
        let Some(request_url) = request.get("url").and_then(|value| value.as_str()) else {
            continue;
        };
        let same_host = wanted_host
            .as_ref()
            .zip(Url::parse(request_url).ok())
            .and_then(|(host, parsed)| parsed.host_str().map(|candidate| host == candidate))
            .unwrap_or(false);
        if request_url != wanted_url && !same_host {
            continue;
        }
        let Some(headers) = request.get("headers").and_then(|value| value.as_array()) else {
            continue;
        };
        selected = headers
            .iter()
            .filter_map(|header| {
                let object = header.as_object()?;
                let name = object.get("name")?.as_str()?.trim();
                let value = object.get("value")?.as_str()?.trim();
                if name.is_empty() || value.is_empty() || !trace_header_allowed(name) {
                    return None;
                }
                Some(RequestHeaderHint {
                    name: name.to_string(),
                    value: value.to_string(),
                    source: "browser_trace".to_string(),
                })
            })
            .collect::<Vec<_>>();
    }
    selected
}

fn trace_header_allowed(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "origin" | "accept-language" | "x-requested-with"
    ) || lower.contains("csrf")
        || lower.contains("xsrf")
}

fn storage_state_header_hints(
    repo_root: &Path,
    storage_state_rel: Option<&str>,
    url: &str,
    effective_url: Option<&str>,
) -> Vec<RequestHeaderHint> {
    let Some(storage_state_rel) = storage_state_rel else {
        return Vec::new();
    };
    let storage_path = repo_path(repo_root, Path::new(storage_state_rel));
    let Ok(body) = fs::read_to_string(storage_path) else {
        return Vec::new();
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&body) else {
        return Vec::new();
    };
    let Some(origins) = value.get("origins").and_then(|value| value.as_array()) else {
        return Vec::new();
    };
    let wanted_host = request_host(url, effective_url);
    let mut hints = Vec::new();
    for origin_entry in origins {
        let Some(origin_url) = origin_entry.get("origin").and_then(|value| value.as_str()) else {
            continue;
        };
        let same_host = wanted_host
            .as_ref()
            .zip(Url::parse(origin_url).ok())
            .and_then(|(host, parsed)| parsed.host_str().map(|candidate| host == candidate))
            .unwrap_or(false);
        if !same_host {
            continue;
        }
        let Some(local_storage) = origin_entry
            .get("localStorage")
            .and_then(|value| value.as_array())
        else {
            continue;
        };
        for storage_entry in local_storage {
            let Some(object) = storage_entry.as_object() else {
                continue;
            };
            let Some(name) = object.get("name").and_then(|value| value.as_str()) else {
                continue;
            };
            let Some(value) = object.get("value").and_then(|value| value.as_str()) else {
                continue;
            };
            let lower = name.to_ascii_lowercase();
            let header_name = if lower.contains("xsrf") {
                Some("X-XSRF-Token")
            } else if lower.contains("csrf") {
                Some("X-CSRF-Token")
            } else {
                None
            };
            if let Some(header_name) = header_name {
                hints.push(RequestHeaderHint {
                    name: header_name.to_string(),
                    value: value.to_string(),
                    source: format!("storage_state:{name}"),
                });
            }
        }
    }
    hints
}

fn dedupe_header_hints(hints: Vec<RequestHeaderHint>) -> Vec<RequestHeaderHint> {
    let mut seen = HashSet::new();
    let mut deduped = Vec::new();
    for hint in hints {
        let key = hint.name.to_ascii_lowercase();
        if seen.insert(key) {
            deduped.push(hint);
        }
    }
    deduped
}

fn write_consume_report(path: &Path, rows: &[ConsumeReportRow]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let mut body = String::from(ConsumeReportRow::header());
    body.push('\n');
    for row in rows {
        body.push_str(&row.to_tsv_line());
        body.push('\n');
    }
    fs::write(path, body).with_context(|| format!("write {}", path.display()))
}

fn session_manifest_path_for_capsule(
    repo_root: &Path,
    capsule: &RequestCapsule,
) -> Option<PathBuf> {
    if let Some(request_capsule_rel) = &capsule.request_capsule_rel {
        let request_capsule_path = repo_path(repo_root, Path::new(request_capsule_rel));
        let session_manifest = request_capsule_path.parent()?.join("session.toml");
        if session_manifest.exists() {
            return Some(session_manifest);
        }
    }
    if let Some(sessions_dir_rel) = &capsule.sessions_dir_rel {
        let session_manifest = repo_path(repo_root, Path::new(sessions_dir_rel))
            .join(&capsule.session_id)
            .join("session.toml");
        if session_manifest.exists() {
            return Some(session_manifest);
        }
    }
    if capsule.bundle_latest_session_id.as_deref() == Some(capsule.session_id.as_str())
        && let Some(bundle_latest_manifest_rel) = &capsule.bundle_latest_manifest_rel
    {
        let session_manifest = repo_path(repo_root, Path::new(bundle_latest_manifest_rel));
        if session_manifest.exists() {
            return Some(session_manifest);
        }
    }
    None
}

fn writeback_session_outcome(
    repo_root: &Path,
    capsule: &RequestCapsule,
    outcome: &ConsumeOutcome,
) -> Result<()> {
    let Some(manifest_path) = session_manifest_path_for_capsule(repo_root, capsule) else {
        return Ok(());
    };
    let checklist_path = manifest_path
        .parent()
        .context("session manifest must have a parent directory")?
        .join("checklist.md");
    let mut session = load_session(&manifest_path)?;
    let now = utc_now();
    match outcome {
        ConsumeOutcome::Downloaded {
            output_path,
            result,
            header_count,
            text_sidecar_rel,
        } => {
            session.status = SessionStatus::Downloaded;
            session.evidence.artifact_rel = Some(to_repo_display_path(repo_root, output_path));
            session.evidence.text_sidecar_rel = text_sidecar_rel.clone();
            session.evidence.bytes = Some(result.bytes);
            session.evidence.sha256 = result.sha256.clone();
            if let Some(http_code) = result.http_code {
                session.evidence.http_code = Some(http_code);
            }
            if let Some(final_url) = &result.final_url {
                session.evidence.effective_url = Some(final_url.clone());
            }
            session.events.push(SessionEvent {
                at_utc: now.clone(),
                action: "consume_handoff".to_string(),
                status: SessionStatus::Downloaded,
                note: Some(format!(
                    "downloaded via consume-handoff backend={} output={} bytes={} sha256={} headers={}",
                    result.backend.as_str(),
                    to_repo_display_path(repo_root, output_path),
                    result.bytes,
                    result.sha256.clone().unwrap_or_default(),
                    header_count
                )),
            });
        }
        ConsumeOutcome::SkippedExisting {
            output_path,
            bytes,
            sha256,
            header_count,
            text_sidecar_rel,
        } => {
            session.status = SessionStatus::Downloaded;
            session.evidence.artifact_rel = Some(to_repo_display_path(repo_root, output_path));
            session.evidence.text_sidecar_rel = text_sidecar_rel.clone();
            session.evidence.bytes = Some(*bytes);
            session.evidence.sha256 = Some(sha256.clone());
            session.events.push(SessionEvent {
                at_utc: now.clone(),
                action: "consume_handoff".to_string(),
                status: SessionStatus::Downloaded,
                note: Some(format!(
                    "skipped existing artifact during consume-handoff output={} bytes={} sha256={} headers={}",
                    to_repo_display_path(repo_root, output_path),
                    bytes,
                    sha256,
                    header_count
                )),
            });
        }
        ConsumeOutcome::Failed {
            output_path,
            error,
            header_count,
        } => {
            if !matches!(session.status, SessionStatus::Downloaded) {
                session.status = SessionStatus::Blocked;
            }
            session.events.push(SessionEvent {
                at_utc: now.clone(),
                action: "consume_handoff".to_string(),
                status: session.status,
                note: Some(format!(
                    "consume-handoff failed output={} headers={} error={}",
                    to_repo_display_path(repo_root, output_path),
                    header_count,
                    error
                )),
            });
        }
    }
    session.updated_utc = now;
    write_session(repo_root, &manifest_path, &session)?;
    write_checklist(repo_root, &checklist_path, &manifest_path, &session)?;
    Ok(())
}

fn build_queue_update(
    repo_root: &Path,
    capsule: &RequestCapsule,
    outcome: &ConsumeOutcome,
) -> QueueSyncUpdate {
    match outcome {
        ConsumeOutcome::Downloaded {
            output_path,
            result,
            header_count,
            ..
        } => QueueSyncUpdate {
            search_target_id: capsule.search_target_id.clone().unwrap_or_default(),
            attempted_utc: utc_now(),
            outcome: "downloaded".to_string(),
            session_id: capsule.session_id.clone(),
            output_rel: Some(to_repo_display_path(repo_root, output_path)),
            http_code: result.http_code,
            sha256: result.sha256.clone(),
            note: format!(
                "consume-handoff downloaded backend={} headers={} {}",
                result.backend.as_str(),
                header_count,
                capsule.note
            ),
        },
        ConsumeOutcome::SkippedExisting {
            output_path,
            sha256,
            header_count,
            ..
        } => QueueSyncUpdate {
            search_target_id: capsule.search_target_id.clone().unwrap_or_default(),
            attempted_utc: utc_now(),
            outcome: "skipped_existing".to_string(),
            session_id: capsule.session_id.clone(),
            output_rel: Some(to_repo_display_path(repo_root, output_path)),
            http_code: capsule.http_code,
            sha256: Some(sha256.clone()),
            note: format!(
                "consume-handoff found existing artifact headers={} {}",
                header_count, capsule.note
            ),
        },
        ConsumeOutcome::Failed {
            output_path,
            error,
            header_count,
        } => QueueSyncUpdate {
            search_target_id: capsule.search_target_id.clone().unwrap_or_default(),
            attempted_utc: utc_now(),
            outcome: "failed".to_string(),
            session_id: capsule.session_id.clone(),
            output_rel: Some(to_repo_display_path(repo_root, output_path)),
            http_code: capsule.http_code,
            sha256: None,
            note: format!(
                "consume-handoff failed headers={} error={} {}",
                header_count, error, capsule.note
            ),
        },
    }
}

fn sync_search_queue(
    repo_root: &Path,
    queue_path: &Path,
    updates: &[QueueSyncUpdate],
) -> Result<()> {
    if updates.is_empty() {
        return Ok(());
    }
    let queue_path = repo_path(repo_root, queue_path);
    let mut queue = load_search_queue(&queue_path)?;
    for update in updates {
        if update.search_target_id.is_empty() {
            continue;
        }
        if let Some(target) = queue
            .search_target
            .iter_mut()
            .find(|target| target.id == update.search_target_id)
        {
            if matches!(update.outcome.as_str(), "downloaded" | "skipped_existing") {
                target.status = "downloaded_locally_via_consume_handoff".to_string();
            }
            target.last_attempt_utc = Some(update.attempted_utc.clone());
            target.last_attempt_result = Some(update.outcome.clone());
            target.last_attempt_session_id = Some(update.session_id.clone());
            target.last_attempt_output_rel = update.output_rel.clone();
            target.last_attempt_http_code = update.http_code;
            target.last_attempt_sha256 = update.sha256.clone();
            target.last_attempt_note = Some(update.note.clone());
        }
    }
    queue.last_updated = Some(Utc::now().date_naive().to_string());
    let body = toml::to_string_pretty(&queue)?;
    fs::write(&queue_path, format!("{body}\n"))
        .with_context(|| format!("write {}", queue_path.display()))
}

impl RouteSummary {
    fn from_request(stack: &DownloadStack, request: &TransferRequest) -> Self {
        let route = stack.route(request, TransferKind::Probe);
        Self::from_route(route)
    }

    fn from_trace(trace: &TransferTrace) -> Self {
        Self::from_route(trace.route.clone())
    }

    fn from_route(route: data_core::download_stack::DownloadRoute) -> Self {
        Self {
            scheme: route.scheme,
            host: route.host,
            retry_class: route.retry_class.as_str().to_string(),
            policy_name: route.policy_name,
            backends: route
                .backends
                .into_iter()
                .map(|backend| backend.as_str().to_string())
                .collect(),
        }
    }
}

impl ProbeSummary {
    fn from_trace(trace: &TransferTrace, attempted_utc: &str) -> Self {
        Self {
            attempted_utc: attempted_utc.to_string(),
            capabilities: trace
                .capabilities
                .as_ref()
                .map(CapabilitySummary::from_capabilities),
            terminal_result: trace
                .terminal_result
                .as_ref()
                .map(TerminalProbeResult::from_result),
            attempts: trace
                .attempts
                .iter()
                .map(ProbeAttemptSummary::from_attempt)
                .collect(),
            final_error: trace.final_error.clone(),
        }
    }
}

impl CapabilitySummary {
    fn from_capabilities(capabilities: &EndpointCapabilities) -> Self {
        Self {
            scheme: capabilities.scheme.clone(),
            host: capabilities.host.clone(),
            surface: capabilities.surface.as_str().to_string(),
            content_type: capabilities.content_type.clone(),
            http_code: capabilities.http_code,
            content_length: capabilities.content_length,
            supports_ranges: capabilities.supports_ranges,
            rsync_reachable: capabilities.rsync_reachable,
            final_url: capabilities.final_url.clone(),
        }
    }
}

impl TerminalProbeResult {
    fn from_result(result: &data_core::download_stack::TransferResult) -> Self {
        Self {
            backend: result.backend.as_str().to_string(),
            kind: match result.kind {
                TransferKind::Probe => "probe",
                TransferKind::Download => "download",
            }
            .to_string(),
            final_url: result.final_url.clone(),
            http_code: result.http_code,
            content_type: result.content_type.clone(),
            bytes: result.bytes,
            is_pdf: result.is_pdf,
            sha256: result.sha256.clone(),
            note: result.note.clone(),
        }
    }
}

impl ProbeAttemptSummary {
    fn from_attempt(attempt: &data_core::download_stack::TransferAttempt) -> Self {
        Self {
            backend: attempt.backend.as_str().to_string(),
            succeeded: attempt.succeeded,
            failure_class: attempt.failure_class.clone(),
            http_code: attempt.http_code,
            content_type: attempt.content_type.clone(),
            bytes: attempt.bytes,
            is_pdf: attempt.is_pdf,
            final_url: attempt.final_url.clone(),
            note: attempt.note.clone(),
            error_message: attempt.error_message.clone(),
        }
    }
}

impl HandoffRow {
    fn from_session(parent_session_id: &str, session: &AcquisitionSession) -> Self {
        Self {
            session_id: session.session_id.clone(),
            parent_session_id: parent_session_id.to_string(),
            url: session.source.url.clone().unwrap_or_default(),
            effective_url: session.evidence.effective_url.clone().unwrap_or_default(),
            cookie_jar_rel: session.evidence.cookie_jar_rel.clone().unwrap_or_default(),
            storage_state_rel: session
                .evidence
                .storage_state_rel
                .clone()
                .unwrap_or_default(),
            browser_trace_rel: session
                .evidence
                .browser_trace_rel
                .clone()
                .unwrap_or_default(),
            profile_bundle_rel: session
                .evidence
                .profile_bundle_rel
                .clone()
                .unwrap_or_default(),
            request_capsule_rel: session
                .evidence
                .request_capsule_rel
                .clone()
                .unwrap_or_default(),
            http_code: session
                .evidence
                .http_code
                .map(|code| code.to_string())
                .unwrap_or_default(),
            transport_substrate: session.controller.transport_substrate.as_str().to_string(),
            requested_backend: session.controller.requested_backend.clone(),
            dest_rel: session.controller.dest_rel.clone().unwrap_or_default(),
            note: latest_event_note(session),
        }
    }

    fn header() -> &'static str {
        "session_id\tparent_session_id\turl\teffective_url\tcookie_jar_rel\tstorage_state_rel\tbrowser_trace_rel\tprofile_bundle_rel\trequest_capsule_rel\thttp_code\ttransport_substrate\trequested_backend\tdest_rel\tnote"
    }

    fn from_tsv_record(header: &[&str], line: &str) -> Result<Self> {
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != header.len() {
            bail!(
                "handoff TSV field count mismatch: expected {} fields, found {}",
                header.len(),
                fields.len()
            );
        }
        let value = |name: &str| -> Result<String> {
            let index = header
                .iter()
                .position(|column| *column == name)
                .with_context(|| format!("handoff TSV missing column '{name}'"))?;
            Ok(fields[index].to_string())
        };
        Ok(Self {
            session_id: value("session_id")?,
            parent_session_id: value("parent_session_id")?,
            url: value("url")?,
            effective_url: value("effective_url")?,
            cookie_jar_rel: value("cookie_jar_rel")?,
            storage_state_rel: value("storage_state_rel")?,
            browser_trace_rel: value("browser_trace_rel")?,
            profile_bundle_rel: value("profile_bundle_rel")?,
            request_capsule_rel: value("request_capsule_rel")?,
            http_code: value("http_code")?,
            transport_substrate: value("transport_substrate")?,
            requested_backend: value("requested_backend")?,
            dest_rel: value("dest_rel")?,
            note: value("note")?,
        })
    }

    fn to_tsv_line(&self) -> String {
        [
            &self.session_id,
            &self.parent_session_id,
            &self.url,
            &self.effective_url,
            &self.cookie_jar_rel,
            &self.storage_state_rel,
            &self.browser_trace_rel,
            &self.profile_bundle_rel,
            &self.request_capsule_rel,
            &self.http_code,
            &self.transport_substrate,
            &self.requested_backend,
            &self.dest_rel,
            &self.note,
        ]
        .iter()
        .map(|value| escape_tsv_field(value))
        .collect::<Vec<_>>()
        .join("\t")
    }
}

impl RequestCapsule {
    fn from_session(
        repo_root: &Path,
        parent_session_id: &str,
        session: &AcquisitionSession,
    ) -> Result<Self> {
        let bundle =
            load_profile_bundle_from_rel(repo_root, session.evidence.profile_bundle_rel.as_deref());
        Ok(Self {
            schema_version: 1,
            generated_utc: utc_now(),
            session_id: session.session_id.clone(),
            parent_session_id: parent_session_id.to_string(),
            url: session.source.url.clone().unwrap_or_default(),
            effective_url: session.evidence.effective_url.clone(),
            host_scope: session.evidence.host_scope.clone(),
            source_id: session.source.source_id.clone(),
            title: session.source.title.clone(),
            site: session.source.site.clone(),
            access_class: session.source.access_class.clone(),
            search_queue_path: session
                .search
                .as_ref()
                .map(|search| search.queue_path.clone()),
            search_project_id: session
                .search
                .as_ref()
                .and_then(|search| search.project_id.clone()),
            search_target_id: session
                .search
                .as_ref()
                .map(|search| search.search_target_id.clone()),
            search_window: session.search.as_ref().map(|search| search.window.clone()),
            search_priority: session
                .search
                .as_ref()
                .map(|search| search.priority.clone()),
            search_status: session.search.as_ref().map(|search| search.status.clone()),
            cookie_jar_rel: session.evidence.cookie_jar_rel.clone(),
            storage_state_rel: session.evidence.storage_state_rel.clone(),
            browser_trace_rel: session.evidence.browser_trace_rel.clone(),
            profile_bundle_rel: session.evidence.profile_bundle_rel.clone(),
            request_capsule_rel: session.evidence.request_capsule_rel.clone(),
            http_code: session.evidence.http_code,
            policy_registry_rel: session.controller.policy_registry_rel.clone(),
            sessions_dir_rel: Some(session.controller.sessions_dir_rel.clone()),
            bundle_root_rel: bundle.as_ref().map(|bundle| bundle.bundle_root_rel.clone()),
            bundle_latest_session_id: bundle
                .as_ref()
                .map(|bundle| bundle.latest_session_id.clone()),
            bundle_latest_manifest_rel: bundle
                .as_ref()
                .map(|bundle| bundle.latest_manifest_rel.clone()),
            bundle_updated_utc: bundle.as_ref().map(|bundle| bundle.updated_utc.clone()),
            header_hints: build_header_hints(
                repo_root,
                session.source.url.as_deref().unwrap_or_default(),
                session.evidence.effective_url.as_deref(),
                session.evidence.cookie_jar_rel.as_deref(),
                session.evidence.browser_trace_rel.as_deref(),
                session.evidence.storage_state_rel.as_deref(),
            ),
            transport_substrate: session.controller.transport_substrate.as_str().to_string(),
            requested_backend: session.controller.requested_backend.clone(),
            dest_rel: session.controller.dest_rel.clone(),
            note: latest_event_note(session),
        })
    }

    fn from_handoff_row(repo_root: &Path, row: &HandoffRow) -> Self {
        let bundle = load_profile_bundle_from_rel(
            repo_root,
            (!row.profile_bundle_rel.is_empty()).then_some(row.profile_bundle_rel.as_str()),
        );
        Self {
            schema_version: 1,
            generated_utc: utc_now(),
            session_id: row.session_id.clone(),
            parent_session_id: row.parent_session_id.clone(),
            url: row.url.clone(),
            effective_url: (!row.effective_url.is_empty()).then_some(row.effective_url.clone()),
            host_scope: derive_host_scope(
                &row.url,
                (!row.effective_url.is_empty()).then_some(row.effective_url.as_str()),
            ),
            source_id: None,
            title: None,
            site: None,
            access_class: None,
            search_queue_path: None,
            search_project_id: None,
            search_target_id: None,
            search_window: None,
            search_priority: None,
            search_status: None,
            cookie_jar_rel: (!row.cookie_jar_rel.is_empty()).then_some(row.cookie_jar_rel.clone()),
            storage_state_rel: (!row.storage_state_rel.is_empty())
                .then_some(row.storage_state_rel.clone()),
            browser_trace_rel: (!row.browser_trace_rel.is_empty())
                .then_some(row.browser_trace_rel.clone()),
            profile_bundle_rel: (!row.profile_bundle_rel.is_empty())
                .then_some(row.profile_bundle_rel.clone()),
            request_capsule_rel: (!row.request_capsule_rel.is_empty())
                .then_some(row.request_capsule_rel.clone()),
            http_code: row.http_code.parse::<u16>().ok(),
            policy_registry_rel: None,
            sessions_dir_rel: None,
            bundle_root_rel: bundle.as_ref().map(|bundle| bundle.bundle_root_rel.clone()),
            bundle_latest_session_id: bundle
                .as_ref()
                .map(|bundle| bundle.latest_session_id.clone()),
            bundle_latest_manifest_rel: bundle
                .as_ref()
                .map(|bundle| bundle.latest_manifest_rel.clone()),
            bundle_updated_utc: bundle.as_ref().map(|bundle| bundle.updated_utc.clone()),
            header_hints: build_header_hints(
                repo_root,
                &row.url,
                (!row.effective_url.is_empty()).then_some(row.effective_url.as_str()),
                (!row.cookie_jar_rel.is_empty()).then_some(row.cookie_jar_rel.as_str()),
                (!row.browser_trace_rel.is_empty()).then_some(row.browser_trace_rel.as_str()),
                (!row.storage_state_rel.is_empty()).then_some(row.storage_state_rel.as_str()),
            ),
            transport_substrate: row.transport_substrate.clone(),
            requested_backend: row.requested_backend.clone(),
            dest_rel: (!row.dest_rel.is_empty()).then_some(row.dest_rel.clone()),
            note: row.note.clone(),
        }
    }
}

impl ConsumeReportRow {
    fn success(
        repo_root: &Path,
        capsule: &RequestCapsule,
        output_path: &Path,
        header_count: usize,
        result: &TransferResult,
    ) -> Self {
        Self {
            session_id: capsule.session_id.clone(),
            url: capsule.url.clone(),
            output_rel: to_repo_display_path(repo_root, output_path),
            result: "downloaded".to_string(),
            backend: result.backend.as_str().to_string(),
            http_code: result
                .http_code
                .map(|code| code.to_string())
                .unwrap_or_default(),
            sha256: result.sha256.clone().unwrap_or_default(),
            final_url: result.final_url.clone().unwrap_or_default(),
            header_count,
            note: capsule.note.clone(),
            error: String::new(),
        }
    }

    fn failure(
        repo_root: &Path,
        capsule: &RequestCapsule,
        output_path: &Path,
        header_count: usize,
        error: &str,
    ) -> Self {
        Self {
            session_id: capsule.session_id.clone(),
            url: capsule.url.clone(),
            output_rel: to_repo_display_path(repo_root, output_path),
            result: "failed".to_string(),
            backend: String::new(),
            http_code: capsule
                .http_code
                .map(|code| code.to_string())
                .unwrap_or_default(),
            sha256: String::new(),
            final_url: capsule.effective_url.clone().unwrap_or_default(),
            header_count,
            note: capsule.note.clone(),
            error: error.to_string(),
        }
    }

    fn skipped(
        repo_root: &Path,
        capsule: &RequestCapsule,
        output_path: &Path,
        header_count: usize,
        sha256: &str,
    ) -> Self {
        Self {
            session_id: capsule.session_id.clone(),
            url: capsule.url.clone(),
            output_rel: to_repo_display_path(repo_root, output_path),
            result: "skipped_existing".to_string(),
            backend: String::new(),
            http_code: capsule
                .http_code
                .map(|code| code.to_string())
                .unwrap_or_default(),
            sha256: sha256.to_string(),
            final_url: capsule.effective_url.clone().unwrap_or_default(),
            header_count,
            note: capsule.note.clone(),
            error: String::new(),
        }
    }

    fn header() -> &'static str {
        "session_id\turl\toutput_rel\tresult\tbackend\thttp_code\tsha256\tfinal_url\theader_count\tnote\terror"
    }

    fn to_tsv_line(&self) -> String {
        [
            &self.session_id,
            &self.url,
            &self.output_rel,
            &self.result,
            &self.backend,
            &self.http_code,
            &self.sha256,
            &self.final_url,
            &self.header_count.to_string(),
            &self.note,
            &self.error,
        ]
        .iter()
        .map(|value| escape_tsv_field(value))
        .collect::<Vec<_>>()
        .join("\t")
    }
}

fn latest_event_note(session: &AcquisitionSession) -> String {
    session
        .events
        .last()
        .and_then(|event| event.note.clone())
        .unwrap_or_default()
}

fn escape_tsv_field(value: &str) -> String {
    value.replace(['\t', '\n'], " ")
}

#[cfg(test)]
mod tests {
    use super::{TransportSubstrate, derive_host_scope, escape_tsv_field, sanitize_token};

    #[test]
    fn sanitize_token_collapses_noise() {
        assert_eq!(sanitize_token("Dickson 1906 / AMS"), "dickson_1906_ams");
    }

    #[test]
    fn transport_substrate_strings_match_contract() {
        assert_eq!(TransportSubstrate::Silksurf.as_str(), "silksurf");
        assert_eq!(TransportSubstrate::DownloadStack.as_str(), "download_stack");
    }

    #[test]
    fn escape_tsv_field_strips_tabs_and_newlines() {
        assert_eq!(escape_tsv_field("a\tb\nc"), "a b c");
    }

    #[test]
    fn derive_host_scope_prefers_effective_url() {
        assert_eq!(
            derive_host_scope("https://example.com/foo", Some("https://www.ams.org/bar")),
            Some("www_ams_org".to_string())
        );
    }
}
