use axum::extract::{Path as AxumPath, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse};
use axum::routing::{get, post};
use axum::{Json, Router};
use clap::Parser;
use gororoba_engine::{
    Thesis1Pipeline, Thesis2Pipeline, Thesis3Pipeline, Thesis4Pipeline, ThesisEvidence,
    ThesisPipeline,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;
use tokio::time::{timeout, Duration};
use stats_core::helpers::{mean, median, std_dev};

const INDEX_HTML: &str = include_str!("../../../../apps/gororoba_studio/ui/index.html");
const APP_JS: &str = include_str!("../../../../apps/gororoba_studio/ui/app.js");
const STYLES_CSS: &str = include_str!("../../../../apps/gororoba_studio/ui/styles.css");

#[derive(Debug, Parser)]
#[command(
    name = "gororoba-studio",
    about = "Interactive web studio for thesis pipelines, live evidence runs, and benchmark snapshots."
)]
struct Args {
    #[arg(long, default_value = "127.0.0.1")]
    host: String,
    #[arg(long, default_value_t = 8088)]
    port: u16,
}

#[derive(Debug, Clone)]
struct AppState {
    pipelines: Arc<Vec<PipelineDescriptor>>,
    history: Arc<Mutex<Vec<RunResponse>>>,
    run_counter: Arc<AtomicU64>,
    catalog_source: String,
    catalog_warnings: Arc<Vec<String>>,
    registry_path: String,
    orchestration_surface: Arc<OrchestrationSurfaceResponse>,
}

#[derive(Debug, Clone, Serialize)]
struct PipelineDescriptor {
    id: String,
    title: String,
    hypothesis: String,
    primary_metric: String,
    quick_profile: String,
    full_profile: String,
    experiment_id: String,
    lineage_id: String,
    registry_binary: String,
    artifact_paths: Vec<String>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum RunProfile {
    #[default]
    Quick,
    Full,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct RunRequest {
    profile: Option<RunProfile>,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    api_version: &'static str,
    service: &'static str,
    status: &'static str,
    unix_seconds: u64,
}

#[derive(Debug, Clone, Serialize)]
struct RunResponse {
    api_version: &'static str,
    run_id: u64,
    unix_seconds: u64,
    experiment_id: String,
    source_experiment_id: Option<String>,
    source_lineage_id: Option<String>,
    artifact_links: Vec<String>,
    profile: RunProfile,
    duration_ms: u128,
    thesis_id: usize,
    label: String,
    metric_value: f64,
    threshold: f64,
    passes_gate: bool,
    config_snapshot: Value,
    messages: Vec<String>,
}

#[derive(Debug, Serialize)]
struct SuiteResponse {
    api_version: &'static str,
    profile: RunProfile,
    total_duration_ms: u128,
    pass_count: usize,
    fail_count: usize,
    success_rate: f64,
    results: Vec<RunResponse>,
    failures: Vec<RunFailure>,
}

#[derive(Debug, Clone, Serialize)]
struct RunFailure {
    api_version: &'static str,
    experiment_id: String,
    profile: RunProfile,
    error: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct BenchmarkRequest {
    profile: Option<RunProfile>,
    iterations: Option<usize>,
}

#[derive(Debug, Serialize)]
struct BenchmarkResponse {
    api_version: &'static str,
    experiment_id: String,
    profile: RunProfile,
    iterations_requested: usize,
    iterations_completed: usize,
    pass_count: usize,
    fail_count: usize,
    mean_duration_ms: f64,
    median_duration_ms: f64,
    min_duration_ms: u128,
    max_duration_ms: u128,
    mean_metric_value: f64,
    metric_stddev: f64,
    runs: Vec<RunResponse>,
    failures: Vec<RunFailure>,
}

#[derive(Debug, Clone, Default, Deserialize)]
struct ReproducibilityRequest {
    profile: Option<RunProfile>,
    iterations: Option<usize>,
    tolerance: Option<f64>,
}

#[derive(Debug, Serialize)]
struct ReproducibilityResponse {
    api_version: &'static str,
    experiment_id: String,
    profile: RunProfile,
    iterations_requested: usize,
    iterations_completed: usize,
    tolerance: f64,
    baseline_metric_value: f64,
    max_metric_delta: f64,
    gate_consistent: bool,
    stable: bool,
    runs: Vec<RunResponse>,
    failures: Vec<RunFailure>,
}

#[derive(Debug, Serialize)]
struct VersionResponse {
    api_version: &'static str,
    service: &'static str,
    package_version: &'static str,
    catalog_source: String,
    pipeline_count: usize,
    catalog_warnings: Vec<String>,
    registry_path: String,
    orchestration_lane_count: usize,
    orchestration_queue_item_count: usize,
    orchestration_warning_count: usize,
}

#[derive(Debug, Clone, Serialize)]
struct OrchestrationLaneHealth {
    lane_id: String,
    lane_label: String,
    source_kind: String,
    source_path: String,
    status: String,
    summary: String,
    evidence_links: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct OrchestrationQueueItem {
    queue_item_id: String,
    queue_kind: String,
    source_path: String,
    status: String,
    summary: String,
    evidence_links: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct PromotionQueueSummary {
    source_paths: Vec<String>,
    item_count: usize,
    ready_count: usize,
    pending_count: usize,
    blocked_count: usize,
    status_counts: BTreeMap<String, usize>,
    items: Vec<OrchestrationQueueItem>,
}

#[derive(Debug, Clone, Serialize)]
struct OrchestrationSurfaceResponse {
    api_version: &'static str,
    lane_health_feed: Vec<OrchestrationLaneHealth>,
    promotion_queue_summary: PromotionQueueSummary,
    warnings: Vec<String>,
}

#[derive(Debug, Serialize)]
struct ApiErrorResponse {
    api_version: &'static str,
    error_code: &'static str,
    message: String,
    known_ids: Vec<String>,
    details: Value,
}

#[derive(Debug, Deserialize)]
struct StudioPipelineCatalog {
    #[serde(default)]
    pipeline: Vec<StudioPipelineCatalogEntry>,
}

#[derive(Debug, Clone, Deserialize)]
struct StudioPipelineCatalogEntry {
    id: Option<String>,
    title: Option<String>,
    hypothesis: Option<String>,
    primary_metric: Option<String>,
    quick_profile: Option<String>,
    full_profile: Option<String>,
    experiment_id: Option<String>,
    lineage_id: Option<String>,
    registry_binary: Option<String>,
    #[serde(default)]
    artifact_paths: Vec<String>,
}

const RUN_TIMEOUT_SECONDS: u64 = 60;
const MAX_HISTORY_ITEMS: usize = 200;
const API_VERSION: &str = "studio.v1";
const ORCH_DEFAULT_GATE_RUNTIME_PATH: &str = "reports/gate_validation_runtime_2026_02_14.toml";
const ORCH_DEFAULT_TRANCHE_LEDGER_PATH: &str = "reports/tranche5_execution_ledger_2026_02_14.toml";
const ORCH_DEFAULT_INTAKE_LEDGER_PATH: &str =
    "reports/research_intake_execution_ledger_2026_02_14.toml";
const ORCH_DEFAULT_PROMOTION_REGISTRY_PATH: &str = "registry/hypercomplex_taxonomy_promotion.toml";
const ORCH_DEFAULT_RECONCILIATION_PATH: &str =
    "reports/research_intake_reconciliation_2026_02_14.toml";
const ORCH_DEFAULT_BINARY_RECONCILIATION_PATH: &str =
    "reports/binary_project_drift_reconciliation_2026_02_14.toml";

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |dur| dur.as_secs())
}

fn locate_studio_pipeline_catalog() -> PathBuf {
    if let Ok(path) = std::env::var("GOROROBA_STUDIO_PIPELINE_CATALOG") {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../registry/studio_pipeline_catalog.toml")
        .canonicalize()
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../registry/studio_pipeline_catalog.toml")
        })
}

#[derive(Debug, Clone)]
struct OrchestrationFeedPaths {
    gate_runtime: PathBuf,
    tranche_ledger: PathBuf,
    intake_ledger: PathBuf,
    promotion_registry: PathBuf,
    reconciliation_report: PathBuf,
    binary_reconciliation_report: PathBuf,
}

fn workspace_root_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../.."))
}

fn resolve_orchestration_path(env_key: &str, default_relative_path: &str) -> PathBuf {
    if let Ok(path) = std::env::var(env_key) {
        return PathBuf::from(path);
    }
    workspace_root_dir().join(default_relative_path)
}

fn default_orchestration_feed_paths() -> OrchestrationFeedPaths {
    OrchestrationFeedPaths {
        gate_runtime: resolve_orchestration_path(
            "GOROROBA_STUDIO_GATE_RUNTIME_FEED",
            ORCH_DEFAULT_GATE_RUNTIME_PATH,
        ),
        tranche_ledger: resolve_orchestration_path(
            "GOROROBA_STUDIO_TRANCHE_LEDGER_FEED",
            ORCH_DEFAULT_TRANCHE_LEDGER_PATH,
        ),
        intake_ledger: resolve_orchestration_path(
            "GOROROBA_STUDIO_INTAKE_LEDGER_FEED",
            ORCH_DEFAULT_INTAKE_LEDGER_PATH,
        ),
        promotion_registry: resolve_orchestration_path(
            "GOROROBA_STUDIO_PROMOTION_REGISTRY_FEED",
            ORCH_DEFAULT_PROMOTION_REGISTRY_PATH,
        ),
        reconciliation_report: resolve_orchestration_path(
            "GOROROBA_STUDIO_RECONCILIATION_FEED",
            ORCH_DEFAULT_RECONCILIATION_PATH,
        ),
        binary_reconciliation_report: resolve_orchestration_path(
            "GOROROBA_STUDIO_BINARY_RECONCILIATION_FEED",
            ORCH_DEFAULT_BINARY_RECONCILIATION_PATH,
        ),
    }
}

fn display_path_from_workspace(path: &Path) -> String {
    let root = workspace_root_dir();
    path.strip_prefix(&root)
        .map(|trimmed| trimmed.to_string_lossy().replace('\\', "/"))
        .unwrap_or_else(|_| path.display().to_string())
}

fn collect_unique_links<I>(links: I) -> Vec<String>
where
    I: IntoIterator<Item = String>,
{
    let mut unique = BTreeSet::new();
    for link in links {
        let trimmed = link.trim();
        if !trimmed.is_empty() {
            unique.insert(trimmed.to_string());
        }
    }
    unique.into_iter().collect()
}

fn read_toml_document(path: &Path) -> Result<toml::Value, String> {
    let raw = fs::read_to_string(path)
        .map_err(|err| format!("failed to read {}: {err}", path.display()))?;
    toml::from_str::<toml::Value>(&raw)
        .map_err(|err| format!("failed to parse {}: {err}", path.display()))
}

fn table_string(value: &toml::Value, table_key: &str, field_key: &str) -> Option<String> {
    value
        .get(table_key)?
        .get(field_key)?
        .as_str()
        .map(|item| item.to_string())
}

fn table_i64(value: &toml::Value, table_key: &str, field_key: &str) -> Option<i64> {
    value.get(table_key)?.get(field_key)?.as_integer()
}

fn table_bool(value: &toml::Value, table_key: &str, field_key: &str) -> Option<bool> {
    value.get(table_key)?.get(field_key)?.as_bool()
}

fn table_array_strings(value: &toml::Value, table_key: &str, field_key: &str) -> Vec<String> {
    value
        .get(table_key)
        .and_then(|table| table.get(field_key))
        .and_then(|items| items.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str())
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(|item| item.to_string())
                .collect()
        })
        .unwrap_or_default()
}

fn array_strings(value: &toml::Value, field_key: &str) -> Vec<String> {
    value
        .get(field_key)
        .and_then(|items| items.as_array())
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str())
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(|item| item.to_string())
                .collect()
        })
        .unwrap_or_default()
}

fn lane_health_fallback(
    lane_id: &str,
    lane_label: &str,
    source_kind: &str,
    source_path: &Path,
    reason: &str,
) -> OrchestrationLaneHealth {
    let source_path = display_path_from_workspace(source_path);
    OrchestrationLaneHealth {
        lane_id: lane_id.to_string(),
        lane_label: lane_label.to_string(),
        source_kind: source_kind.to_string(),
        source_path: source_path.clone(),
        status: "missing".to_string(),
        summary: reason.to_string(),
        evidence_links: vec![source_path],
    }
}

fn queue_item_fallback(
    queue_item_id: &str,
    queue_kind: &str,
    source_path: &Path,
    reason: &str,
) -> OrchestrationQueueItem {
    let source_path = display_path_from_workspace(source_path);
    OrchestrationQueueItem {
        queue_item_id: queue_item_id.to_string(),
        queue_kind: queue_kind.to_string(),
        source_path: source_path.clone(),
        status: "missing".to_string(),
        summary: reason.to_string(),
        evidence_links: vec![source_path],
    }
}

fn parse_gate_runtime_lane(path: &Path) -> Result<OrchestrationLaneHealth, String> {
    let doc = read_toml_document(path)?;
    let source_path = display_path_from_workspace(path);
    let governance_gate = table_string(&doc, "final_state", "governance_gate")
        .unwrap_or_else(|| "unknown".to_string());
    let registry_gate = table_string(&doc, "final_state", "registry_acceptance_gate")
        .unwrap_or_else(|| "unknown".to_string());
    let open_blockers = table_i64(&doc, "final_state", "open_blockers").unwrap_or(-1);
    let report_status =
        table_string(&doc, "report", "status").unwrap_or_else(|| "unknown".to_string());
    let status = if governance_gate == "pass" && registry_gate == "pass" && open_blockers == 0 {
        "healthy"
    } else if report_status == "completed" {
        "degraded"
    } else {
        "blocked"
    };

    let mut evidence_links = vec![source_path.clone()];
    if let Some(items) = doc
        .get("resolved_blocker")
        .and_then(|value| value.as_array())
    {
        for item in items {
            evidence_links.extend(array_strings(item, "evidence"));
        }
    }

    Ok(OrchestrationLaneHealth {
        lane_id: "runtime-gates".to_string(),
        lane_label: "Runtime governance and acceptance gates".to_string(),
        source_kind: "gate_artifact".to_string(),
        source_path,
        status: status.to_string(),
        summary: format!(
            "report_status={report_status}, governance_gate={governance_gate}, registry_acceptance_gate={registry_gate}, open_blockers={open_blockers}"
        ),
        evidence_links: collect_unique_links(evidence_links),
    })
}

fn parse_tranche_ledger_lane(path: &Path) -> Result<OrchestrationLaneHealth, String> {
    let doc = read_toml_document(path)?;
    let source_path = display_path_from_workspace(path);
    let ledger_status =
        table_string(&doc, "ledger", "status").unwrap_or_else(|| "unknown".to_string());
    let completed_count = table_i64(&doc, "ledger", "completed_count")
        .unwrap_or(0)
        .max(0) as usize;
    let in_progress_count = table_i64(&doc, "ledger", "in_progress_count")
        .unwrap_or(0)
        .max(0) as usize;
    let queued_count = table_i64(&doc, "ledger", "queued_count")
        .unwrap_or(0)
        .max(0) as usize;
    let consistency_status =
        table_string(&doc, "consistency", "status").unwrap_or_else(|| "unknown".to_string());
    let status = if consistency_status == "pass" && queued_count == 0 {
        if in_progress_count == 0 {
            "healthy"
        } else {
            "active"
        }
    } else if ledger_status == "active" {
        "active"
    } else {
        "degraded"
    };

    let mut evidence_links = vec![source_path.clone()];
    evidence_links.extend(table_array_strings(&doc, "ledger", "source_ledgers"));
    for table_name in ["completed", "in_progress", "queued"] {
        if let Some(items) = doc.get(table_name).and_then(|value| value.as_array()) {
            for item in items {
                evidence_links.extend(array_strings(item, "evidence_refs"));
            }
        }
    }
    if let Some(items) = doc.get("ownership").and_then(|value| value.as_array()) {
        for item in items {
            evidence_links.extend(array_strings(item, "artifacts"));
        }
    }

    Ok(OrchestrationLaneHealth {
        lane_id: "tracker-synchronization".to_string(),
        lane_label: "Execution tracker synchronization lane".to_string(),
        source_kind: "ledger_artifact".to_string(),
        source_path,
        status: status.to_string(),
        summary: format!(
            "ledger_status={ledger_status}, completed={completed_count}, in_progress={in_progress_count}, queued={queued_count}, consistency={consistency_status}"
        ),
        evidence_links: collect_unique_links(evidence_links),
    })
}

fn parse_intake_ledger_lane(path: &Path) -> Result<OrchestrationLaneHealth, String> {
    let doc = read_toml_document(path)?;
    let source_path = display_path_from_workspace(path);
    let ledger_status =
        table_string(&doc, "ledger", "status").unwrap_or_else(|| "unknown".to_string());
    let entry_count = doc
        .get("entry")
        .and_then(|value| value.as_array())
        .map_or(0, |items| items.len());
    let success_count = doc
        .get("entry")
        .and_then(|value| value.as_array())
        .map(|items| {
            items
                .iter()
                .filter(|item| {
                    item.get("status").and_then(|status| status.as_str()) == Some("success")
                })
                .count()
        })
        .unwrap_or(0);
    let failure_count = entry_count.saturating_sub(success_count);
    let status = if entry_count == 0 {
        "degraded"
    } else if failure_count == 0 {
        "healthy"
    } else {
        "degraded"
    };

    Ok(OrchestrationLaneHealth {
        lane_id: "research-intake-execution".to_string(),
        lane_label: "Research intake execution lane".to_string(),
        source_kind: "ledger_artifact".to_string(),
        source_path: source_path.clone(),
        status: status.to_string(),
        summary: format!(
            "ledger_status={ledger_status}, entries={entry_count}, success={success_count}, failure={failure_count}"
        ),
        evidence_links: vec![source_path],
    })
}

fn parse_promotion_registry_queue(path: &Path) -> Result<Vec<OrchestrationQueueItem>, String> {
    let doc = read_toml_document(path)?;
    let source_path = display_path_from_workspace(path);
    let mut items = Vec::new();

    if let Some(promotions) = doc.get("promotion").and_then(|value| value.as_array()) {
        for (index, promotion) in promotions.iter().enumerate() {
            let id = promotion
                .get("id")
                .and_then(|value| value.as_str())
                .map_or_else(
                    || format!("promotion-{:03}", index + 1),
                    |value| value.to_string(),
                );
            let candidate_kind = promotion
                .get("candidate_kind")
                .and_then(|value| value.as_str())
                .unwrap_or("promotion_candidate");
            let statement = promotion
                .get("statement")
                .and_then(|value| value.as_str())
                .unwrap_or("promotion candidate");
            let evidence_status = promotion
                .get("evidence_status")
                .and_then(|value| value.as_str())
                .unwrap_or("unknown");
            let claim_ready = promotion
                .get("claim_lane_ready")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            let task_ready = promotion
                .get("task_lane_ready")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            let ticket_ready = promotion
                .get("ticket_lane_ready")
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            let status = if claim_ready && (task_ready || ticket_ready) {
                "ready"
            } else if claim_ready {
                "claim_ready"
            } else {
                "blocked"
            };

            let mut evidence_links = vec![source_path.clone()];
            if let Some(anchor) = promotion
                .get("evidence_anchor")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                evidence_links.push(anchor.to_string());
            }
            if let Some(where_stated) = promotion
                .get("where_stated")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                evidence_links.push(where_stated.to_string());
            }

            items.push(OrchestrationQueueItem {
                queue_item_id: id,
                queue_kind: format!("promotion::{candidate_kind}"),
                source_path: source_path.clone(),
                status: status.to_string(),
                summary: format!("{statement} (evidence_status={evidence_status})"),
                evidence_links: collect_unique_links(evidence_links),
            });
        }
    }

    Ok(items)
}

fn parse_reconciliation_queue(path: &Path) -> Result<Vec<OrchestrationQueueItem>, String> {
    let doc = read_toml_document(path)?;
    let source_path = display_path_from_workspace(path);
    let mut base_links = vec![source_path.clone()];
    if let Some(intake_path) = table_string(&doc, "report", "intake_path") {
        base_links.push(intake_path);
    }
    if let Some(mapping_path) = table_string(&doc, "report", "sensational_mapping_path") {
        base_links.push(mapping_path);
    }
    if let Some(validation_path) = table_string(&doc, "report", "sensational_validation_path") {
        base_links.push(validation_path);
    }
    let base_links = collect_unique_links(base_links);

    let mut items = Vec::new();
    if let Some(resolved) = doc.get("resolved_sync").and_then(|value| value.as_array()) {
        for (index, entry) in resolved.iter().enumerate() {
            let id = entry
                .get("id")
                .and_then(|value| value.as_str())
                .map_or_else(
                    || format!("resolved-{:03}", index + 1),
                    |value| value.to_string(),
                );
            let action = entry
                .get("action")
                .and_then(|value| value.as_str())
                .unwrap_or("reconciled");
            let mut evidence_links = base_links.clone();
            if let Some(value) = entry
                .get("value")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                evidence_links.push(value.to_string());
            }
            items.push(OrchestrationQueueItem {
                queue_item_id: id,
                queue_kind: "reconciliation::resolved_sync".to_string(),
                source_path: source_path.clone(),
                status: "reconciled".to_string(),
                summary: action.to_string(),
                evidence_links: collect_unique_links(evidence_links),
            });
        }
    }
    if let Some(unresolved) = doc.get("unresolved").and_then(|value| value.as_array()) {
        for (index, entry) in unresolved.iter().enumerate() {
            let id = entry
                .get("id")
                .and_then(|value| value.as_str())
                .map_or_else(
                    || format!("unresolved-{:03}", index + 1),
                    |value| value.to_string(),
                );
            let reason = entry
                .get("reason")
                .and_then(|value| value.as_str())
                .unwrap_or("missing reconciliation evidence");
            let mut evidence_links = base_links.clone();
            if let Some(candidate_url) = entry
                .get("candidate_url")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                evidence_links.push(candidate_url.to_string());
            }
            items.push(OrchestrationQueueItem {
                queue_item_id: id,
                queue_kind: "reconciliation::unresolved".to_string(),
                source_path: source_path.clone(),
                status: "needs_evidence".to_string(),
                summary: reason.to_string(),
                evidence_links: collect_unique_links(evidence_links),
            });
        }
    }

    Ok(items)
}

fn parse_binary_reconciliation_queue(path: &Path) -> Result<Vec<OrchestrationQueueItem>, String> {
    let doc = read_toml_document(path)?;
    let source_path = display_path_from_workspace(path);
    let mut items = Vec::new();

    if let Some(resolutions) = doc
        .get("binary_resolution")
        .and_then(|value| value.as_array())
    {
        for (index, resolution) in resolutions.iter().enumerate() {
            let name = resolution
                .get("name")
                .and_then(|value| value.as_str())
                .map_or_else(
                    || format!("binary-{:03}", index + 1),
                    |value| value.to_string(),
                );
            let raw_status = resolution
                .get("status")
                .and_then(|value| value.as_str())
                .unwrap_or("pending");
            let normalized_status =
                if raw_status.contains("added") || raw_status.contains("resolved") {
                    "reconciled"
                } else {
                    raw_status
                };
            let mut evidence_links = vec![source_path.clone()];
            if let Some(bin_path) = resolution
                .get("path")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                evidence_links.push(bin_path.to_string());
            }
            items.push(OrchestrationQueueItem {
                queue_item_id: name.clone(),
                queue_kind: "reconciliation::binary_drift".to_string(),
                source_path: source_path.clone(),
                status: normalized_status.to_string(),
                summary: format!("{name} ({raw_status})"),
                evidence_links: collect_unique_links(evidence_links),
            });
        }
    }

    if items.is_empty() {
        let resolved = table_bool(&doc, "reconciliation", "resolved_binary_drift").unwrap_or(false);
        let status = if resolved { "reconciled" } else { "pending" };
        items.push(OrchestrationQueueItem {
            queue_item_id: "binary-drift".to_string(),
            queue_kind: "reconciliation::binary_drift".to_string(),
            source_path: source_path.clone(),
            status: status.to_string(),
            summary: "Binary/project drift reconciliation status".to_string(),
            evidence_links: vec![source_path],
        });
    }

    Ok(items)
}

fn load_lane_health_feed(
    paths: &OrchestrationFeedPaths,
    warnings: &mut Vec<String>,
) -> Vec<OrchestrationLaneHealth> {
    let mut lanes = Vec::new();
    match parse_gate_runtime_lane(&paths.gate_runtime) {
        Ok(item) => lanes.push(item),
        Err(err) => {
            warnings.push(format!("lane feed unavailable (runtime-gates): {err}"));
            lanes.push(lane_health_fallback(
                "runtime-gates",
                "Runtime governance and acceptance gates",
                "gate_artifact",
                &paths.gate_runtime,
                "Runtime gate feed missing; fallback lane snapshot emitted.",
            ));
        }
    }
    match parse_tranche_ledger_lane(&paths.tranche_ledger) {
        Ok(item) => lanes.push(item),
        Err(err) => {
            warnings.push(format!(
                "lane feed unavailable (tracker-synchronization): {err}"
            ));
            lanes.push(lane_health_fallback(
                "tracker-synchronization",
                "Execution tracker synchronization lane",
                "ledger_artifact",
                &paths.tranche_ledger,
                "Tranche ledger feed missing; fallback lane snapshot emitted.",
            ));
        }
    }
    match parse_intake_ledger_lane(&paths.intake_ledger) {
        Ok(item) => lanes.push(item),
        Err(err) => {
            warnings.push(format!(
                "lane feed unavailable (research-intake-execution): {err}"
            ));
            lanes.push(lane_health_fallback(
                "research-intake-execution",
                "Research intake execution lane",
                "ledger_artifact",
                &paths.intake_ledger,
                "Research intake ledger missing; fallback lane snapshot emitted.",
            ));
        }
    }
    lanes
}

fn load_promotion_queue_summary(
    paths: &OrchestrationFeedPaths,
    warnings: &mut Vec<String>,
) -> PromotionQueueSummary {
    let mut items = Vec::new();

    match parse_promotion_registry_queue(&paths.promotion_registry) {
        Ok(mut parsed) => items.append(&mut parsed),
        Err(err) => {
            warnings.push(format!("promotion queue feed unavailable: {err}"));
            items.push(queue_item_fallback(
                "promotion-registry",
                "promotion::registry",
                &paths.promotion_registry,
                "Promotion registry feed missing; fallback queue item emitted.",
            ));
        }
    }

    match parse_reconciliation_queue(&paths.reconciliation_report) {
        Ok(mut parsed) => items.append(&mut parsed),
        Err(err) => {
            warnings.push(format!("reconciliation queue feed unavailable: {err}"));
            items.push(queue_item_fallback(
                "research-intake-reconciliation",
                "reconciliation::research_intake",
                &paths.reconciliation_report,
                "Research intake reconciliation feed missing; fallback queue item emitted.",
            ));
        }
    }

    match parse_binary_reconciliation_queue(&paths.binary_reconciliation_report) {
        Ok(mut parsed) => items.append(&mut parsed),
        Err(err) => {
            warnings.push(format!(
                "binary reconciliation queue feed unavailable: {err}"
            ));
            items.push(queue_item_fallback(
                "binary-project-drift-reconciliation",
                "reconciliation::binary_drift",
                &paths.binary_reconciliation_report,
                "Binary drift reconciliation feed missing; fallback queue item emitted.",
            ));
        }
    }

    items.sort_by(|left, right| {
        left.queue_item_id
            .cmp(&right.queue_item_id)
            .then(left.queue_kind.cmp(&right.queue_kind))
    });

    let mut status_counts = BTreeMap::new();
    for item in &items {
        *status_counts.entry(item.status.clone()).or_insert(0) += 1;
    }
    let ready_count = items
        .iter()
        .filter(|item| matches!(item.status.as_str(), "ready" | "reconciled" | "resolved"))
        .count();
    let blocked_count = items
        .iter()
        .filter(|item| matches!(item.status.as_str(), "blocked" | "missing"))
        .count();
    let pending_count = items.len().saturating_sub(ready_count + blocked_count);
    let source_paths = collect_unique_links(vec![
        display_path_from_workspace(&paths.promotion_registry),
        display_path_from_workspace(&paths.reconciliation_report),
        display_path_from_workspace(&paths.binary_reconciliation_report),
    ]);

    PromotionQueueSummary {
        source_paths,
        item_count: items.len(),
        ready_count,
        pending_count,
        blocked_count,
        status_counts,
        items,
    }
}

fn load_orchestration_surface(paths: &OrchestrationFeedPaths) -> OrchestrationSurfaceResponse {
    let mut warnings = Vec::new();
    let lane_health_feed = load_lane_health_feed(paths, &mut warnings);
    let promotion_queue_summary = load_promotion_queue_summary(paths, &mut warnings);
    OrchestrationSurfaceResponse {
        api_version: API_VERSION,
        lane_health_feed,
        promotion_queue_summary,
        warnings,
    }
}

fn merge_artifact_paths(primary: &[String], secondary: &[String]) -> Vec<String> {
    let mut unique = BTreeSet::new();
    for path in primary.iter().chain(secondary.iter()) {
        if !path.trim().is_empty() {
            unique.insert(path.clone());
        }
    }
    unique.into_iter().collect()
}

fn default_pipeline_catalog() -> Vec<PipelineDescriptor> {
    vec![
        PipelineDescriptor {
            id: "thesis-1".to_string(),
            title: "T1 Viscous Vacuum Correlation".to_string(),
            hypothesis: "Frustration density and viscosity should remain strongly coupled in spatial slices.".to_string(),
            primary_metric: "Spearman correlation".to_string(),
            quick_profile: "8^3 field, subregions=2".to_string(),
            full_profile: "16^3 field, subregions=2".to_string(),
            experiment_id: String::new(),
            lineage_id: String::new(),
            registry_binary: String::new(),
            artifact_paths: Vec::new(),
        },
        PipelineDescriptor {
            id: "thesis-2".to_string(),
            title: "T2 Non-Newtonian Thickening".to_string(),
            hypothesis: "Associator-coupled viscosity should increase with strain-rate under non-Newtonian conditions.".to_string(),
            primary_metric: "Viscosity ratio (high/low)".to_string(),
            quick_profile: "alpha=0.4, n=1.25".to_string(),
            full_profile: "alpha=0.5, n=1.5".to_string(),
            experiment_id: String::new(),
            lineage_id: String::new(),
            registry_binary: String::new(),
            artifact_paths: Vec::new(),
        },
        PipelineDescriptor {
            id: "thesis-3".to_string(),
            title: "T3 Plateau to Epoch Alignment".to_string(),
            hypothesis: "Loss plateaus should align with interpretable epoch markers while preserving hubble consistency.".to_string(),
            primary_metric: "Plateau count".to_string(),
            quick_profile: "32 epochs".to_string(),
            full_profile: "96 epochs".to_string(),
            experiment_id: String::new(),
            lineage_id: String::new(),
            registry_binary: String::new(),
            artifact_paths: Vec::new(),
        },
        PipelineDescriptor {
            id: "thesis-4".to_string(),
            title: "T4 Latency Scaling Law".to_string(),
            hypothesis: "Collision return-time statistics should exhibit inverse-square style scaling.".to_string(),
            primary_metric: "Inverse-square R^2".to_string(),
            quick_profile: "500 steps".to_string(),
            full_profile: "2000 steps".to_string(),
            experiment_id: String::new(),
            lineage_id: String::new(),
            registry_binary: String::new(),
            artifact_paths: Vec::new(),
        },
    ]
}

fn has_text(value: Option<&str>) -> bool {
    value.is_some_and(|item| !item.trim().is_empty())
}

fn trimmed_owned(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(|item| item.to_string())
}

fn catalog_missing_required_fields(entry: &StudioPipelineCatalogEntry) -> Vec<&'static str> {
    let mut missing = Vec::new();
    if !has_text(entry.title.as_deref()) {
        missing.push("title");
    }
    if !has_text(entry.hypothesis.as_deref()) {
        missing.push("hypothesis");
    }
    if !has_text(entry.primary_metric.as_deref()) {
        missing.push("primary_metric");
    }
    if !has_text(entry.quick_profile.as_deref()) {
        missing.push("quick_profile");
    }
    if !has_text(entry.full_profile.as_deref()) {
        missing.push("full_profile");
    }
    missing
}

fn apply_catalog_entry(
    pipeline: &mut PipelineDescriptor,
    entry: &StudioPipelineCatalogEntry,
    warnings: &mut Vec<String>,
) {
    let missing = catalog_missing_required_fields(entry);
    if !missing.is_empty() {
        warnings.push(format!(
            "{} missing required catalog fields: {}",
            pipeline.id,
            missing.join(", ")
        ));
    }

    if let Some(value) = trimmed_owned(entry.title.as_deref()) {
        pipeline.title = value;
    }
    if let Some(value) = trimmed_owned(entry.hypothesis.as_deref()) {
        pipeline.hypothesis = value;
    }
    if let Some(value) = trimmed_owned(entry.primary_metric.as_deref()) {
        pipeline.primary_metric = value;
    }
    if let Some(value) = trimmed_owned(entry.quick_profile.as_deref()) {
        pipeline.quick_profile = value;
    }
    if let Some(value) = trimmed_owned(entry.full_profile.as_deref()) {
        pipeline.full_profile = value;
    }
    if let Some(value) = trimmed_owned(entry.experiment_id.as_deref()) {
        pipeline.experiment_id = value;
    }
    if let Some(value) = trimmed_owned(entry.lineage_id.as_deref()) {
        pipeline.lineage_id = value;
    }
    if let Some(value) = trimmed_owned(entry.registry_binary.as_deref()) {
        pipeline.registry_binary = value;
    }

    let artifact_paths: Vec<String> = entry
        .artifact_paths
        .iter()
        .filter_map(|path| trimmed_owned(Some(path.as_str())))
        .collect();
    if !artifact_paths.is_empty() {
        pipeline.artifact_paths = merge_artifact_paths(&artifact_paths, &[]);
    }
}

fn registry_backed_catalog(
    registry_path: &PathBuf,
) -> Result<(Vec<PipelineDescriptor>, Vec<String>), String> {
    let raw = fs::read_to_string(registry_path)
        .map_err(|err| format!("failed to read {}: {err}", registry_path.display()))?;
    let parsed: StudioPipelineCatalog = toml::from_str(&raw)
        .map_err(|err| format!("failed to parse studio pipeline catalog: {err}"))?;
    let mut catalog = default_pipeline_catalog();
    let mut warnings = Vec::new();
    let mut entries_by_id = BTreeMap::new();

    for (index, entry) in parsed.pipeline.into_iter().enumerate() {
        if let Some(id) = trimmed_owned(entry.id.as_deref()) {
            if entries_by_id.insert(id.clone(), entry).is_some() {
                warnings.push(format!("duplicate catalog entry for {id}; last entry wins"));
            }
        } else {
            warnings.push(format!(
                "catalog entry #{} missing required field: id",
                index + 1
            ));
        }
    }

    let known_ids: BTreeSet<String> = catalog.iter().map(|pipeline| pipeline.id.clone()).collect();
    for pipeline in &mut catalog {
        if let Some(entry) = entries_by_id.get(&pipeline.id) {
            apply_catalog_entry(pipeline, entry, &mut warnings);
        } else {
            warnings.push(format!(
                "{} missing in studio catalog; using fallback catalog descriptor",
                pipeline.id
            ));
        }
    }

    for id in entries_by_id.keys() {
        if !known_ids.contains(id) {
            warnings.push(format!(
                "unknown pipeline id in studio catalog ignored: {id}"
            ));
        }
    }

    Ok((catalog, warnings))
}

fn default_state() -> AppState {
    let registry_path = locate_studio_pipeline_catalog();
    let registry_path_text = registry_path.display().to_string();
    let (pipelines, catalog_source, warnings) = match registry_backed_catalog(&registry_path) {
        Ok((catalog, warnings)) => (catalog, "registry".to_string(), warnings),
        Err(err) => (
            default_pipeline_catalog(),
            "fallback".to_string(),
            vec![format!("registry catalog load failed: {err}")],
        ),
    };
    let orchestration_paths = default_orchestration_feed_paths();
    let orchestration_surface = load_orchestration_surface(&orchestration_paths);
    AppState {
        pipelines: Arc::new(pipelines),
        history: Arc::new(Mutex::new(Vec::new())),
        run_counter: Arc::new(AtomicU64::new(1)),
        catalog_source,
        catalog_warnings: Arc::new(warnings),
        registry_path: registry_path_text,
        orchestration_surface: Arc::new(orchestration_surface),
    }
}

fn bounded_iterations(value: Option<usize>, default_value: usize, min: usize, max: usize) -> usize {
    let raw = value.unwrap_or(default_value);
    raw.clamp(min, max)
}


fn execute_thesis(experiment_id: &str, profile: RunProfile) -> Result<RunResponse, String> {
    let start = Instant::now();
    let (evidence, parameters): (ThesisEvidence, Value) = match (experiment_id, profile) {
        ("thesis-1", RunProfile::Quick) => {
            let pipeline = Thesis1Pipeline {
                grid_size: 8,
                lambda: 1.0,
                n_sub: 2,
                p_threshold: 0.05,
            };
            (
                pipeline.execute(),
                json!({
                    "grid_size": pipeline.grid_size,
                    "lambda": pipeline.lambda,
                    "n_sub": pipeline.n_sub,
                    "p_threshold": pipeline.p_threshold,
                }),
            )
        }
        ("thesis-1", RunProfile::Full) => {
            let pipeline = Thesis1Pipeline::default();
            (
                pipeline.execute(),
                json!({
                    "grid_size": pipeline.grid_size,
                    "lambda": pipeline.lambda,
                    "n_sub": pipeline.n_sub,
                    "p_threshold": pipeline.p_threshold,
                }),
            )
        }
        ("thesis-2", RunProfile::Quick) => {
            let pipeline = Thesis2Pipeline {
                alpha: 0.4,
                beta: 1.0,
                power_index: 1.25,
                viscosity_ratio_threshold: 1.02,
            };
            (
                pipeline.execute(),
                json!({
                    "alpha": pipeline.alpha,
                    "beta": pipeline.beta,
                    "power_index": pipeline.power_index,
                    "viscosity_ratio_threshold": pipeline.viscosity_ratio_threshold,
                }),
            )
        }
        ("thesis-2", RunProfile::Full) => {
            let pipeline = Thesis2Pipeline::default();
            (
                pipeline.execute(),
                json!({
                    "alpha": pipeline.alpha,
                    "beta": pipeline.beta,
                    "power_index": pipeline.power_index,
                    "viscosity_ratio_threshold": pipeline.viscosity_ratio_threshold,
                }),
            )
        }
        ("thesis-3", RunProfile::Quick) => {
            let pipeline = Thesis3Pipeline {
                epochs: 32,
                curvature_threshold: 1e-4,
                min_plateau_length: 3,
                optimization_steps: 100,
                ..Thesis3Pipeline::default()
            };
            (
                pipeline.execute(),
                json!({
                    "epochs": pipeline.epochs,
                    "curvature_threshold": pipeline.curvature_threshold,
                    "min_plateau_length": pipeline.min_plateau_length,
                }),
            )
        }
        ("thesis-3", RunProfile::Full) => {
            let pipeline = Thesis3Pipeline {
                epochs: 96,
                curvature_threshold: 1e-4,
                min_plateau_length: 3,
                ..Thesis3Pipeline::default()
            };
            (
                pipeline.execute(),
                json!({
                    "epochs": pipeline.epochs,
                    "curvature_threshold": pipeline.curvature_threshold,
                    "min_plateau_length": pipeline.min_plateau_length,
                }),
            )
        }
        ("thesis-4", RunProfile::Quick) => {
            let pipeline = Thesis4Pipeline {
                n_steps: 500,
                dim: 16,
                seed: 42,
                n_shells: 20,
                r2_threshold: 0.7,
            };
            (
                pipeline.execute(),
                json!({
                    "n_steps": pipeline.n_steps,
                    "dim": pipeline.dim,
                    "seed": pipeline.seed,
                    "r2_threshold": pipeline.r2_threshold,
                }),
            )
        }
        ("thesis-4", RunProfile::Full) => {
            let pipeline = Thesis4Pipeline::default();
            (
                pipeline.execute(),
                json!({
                    "n_steps": pipeline.n_steps,
                    "dim": pipeline.dim,
                    "seed": pipeline.seed,
                    "r2_threshold": pipeline.r2_threshold,
                }),
            )
        }
        _ => return Err(format!("unknown experiment id: {experiment_id}")),
    };

    Ok(RunResponse {
        api_version: API_VERSION,
        run_id: 0,
        unix_seconds: now_unix_seconds(),
        experiment_id: experiment_id.to_string(),
        source_experiment_id: None,
        source_lineage_id: None,
        artifact_links: Vec::new(),
        profile,
        duration_ms: start.elapsed().as_millis(),
        thesis_id: evidence.thesis_id,
        label: evidence.label,
        metric_value: evidence.metric_value,
        threshold: evidence.threshold,
        passes_gate: evidence.passes_gate,
        config_snapshot: json!({
            "profile": profile,
            "parameters": parameters,
        }),
        messages: evidence.messages,
    })
}

fn known_pipeline_ids(state: &AppState) -> Vec<String> {
    state
        .pipelines
        .iter()
        .map(|entry| entry.id.clone())
        .collect()
}

fn is_known_pipeline(state: &AppState, experiment_id: &str) -> bool {
    state
        .pipelines
        .iter()
        .any(|entry| entry.id == experiment_id)
}

fn api_error(
    status: StatusCode,
    error_code: &'static str,
    message: impl Into<String>,
    known_ids: Vec<String>,
    details: Value,
) -> axum::response::Response {
    (
        status,
        Json(ApiErrorResponse {
            api_version: API_VERSION,
            error_code,
            message: message.into(),
            known_ids,
            details,
        }),
    )
        .into_response()
}

async fn execute_thesis_async(
    experiment_id: String,
    profile: RunProfile,
) -> Result<RunResponse, String> {
    let id_for_timeout = experiment_id.clone();
    let handle = tokio::task::spawn_blocking(move || execute_thesis(&experiment_id, profile));
    let joined = timeout(Duration::from_secs(RUN_TIMEOUT_SECONDS), handle)
        .await
        .map_err(|_| format!("execution timed out for {id_for_timeout}"))?;
    joined.map_err(|err| format!("execution task failed for {id_for_timeout}: {err}"))?
}

async fn push_history(state: &AppState, mut run: RunResponse) -> RunResponse {
    run.run_id = state.run_counter.fetch_add(1, Ordering::Relaxed);
    run.unix_seconds = now_unix_seconds();
    if let Some(catalog) = state
        .pipelines
        .iter()
        .find(|pipeline| pipeline.id == run.experiment_id)
    {
        run.artifact_links = catalog.artifact_paths.clone();
        run.source_experiment_id = if catalog.experiment_id.is_empty() {
            None
        } else {
            Some(catalog.experiment_id.clone())
        };
        run.source_lineage_id = if catalog.lineage_id.is_empty() {
            None
        } else {
            Some(catalog.lineage_id.clone())
        };
        run.config_snapshot["catalog_source"] = json!(state.catalog_source.clone());
        run.config_snapshot["registry_binary"] = json!(catalog.registry_binary.clone());
    }
    let mut history = state.history.lock().await;
    history.insert(0, run.clone());
    history.truncate(MAX_HISTORY_ITEMS);
    run
}

async fn run_batch(
    state: &AppState,
    ids: &[String],
    profile: RunProfile,
) -> (Vec<RunResponse>, Vec<RunFailure>) {
    let mut successes = Vec::with_capacity(ids.len());
    let mut failures = Vec::new();
    for id in ids {
        match execute_thesis_async(id.clone(), profile).await {
            Ok(run) => {
                let recorded = push_history(state, run).await;
                successes.push(recorded);
            }
            Err(error) => failures.push(RunFailure {
                api_version: API_VERSION,
                experiment_id: id.clone(),
                profile,
                error,
            }),
        }
    }
    (successes, failures)
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn app_js() -> impl IntoResponse {
    (
        [(
            axum::http::header::CONTENT_TYPE,
            "text/javascript; charset=utf-8",
        )],
        APP_JS,
    )
}

async fn styles_css() -> impl IntoResponse {
    (
        [(axum::http::header::CONTENT_TYPE, "text/css; charset=utf-8")],
        STYLES_CSS,
    )
}

async fn health() -> Json<HealthResponse> {
    let unix_seconds = now_unix_seconds();
    Json(HealthResponse {
        api_version: API_VERSION,
        service: "gororoba-studio",
        status: "ok",
        unix_seconds,
    })
}

async fn version(State(state): State<AppState>) -> Json<VersionResponse> {
    Json(VersionResponse {
        api_version: API_VERSION,
        service: "gororoba-studio",
        package_version: env!("CARGO_PKG_VERSION"),
        catalog_source: state.catalog_source.clone(),
        pipeline_count: state.pipelines.len(),
        catalog_warnings: (*state.catalog_warnings).clone(),
        registry_path: state.registry_path.clone(),
        orchestration_lane_count: state.orchestration_surface.lane_health_feed.len(),
        orchestration_queue_item_count: state
            .orchestration_surface
            .promotion_queue_summary
            .item_count,
        orchestration_warning_count: state.orchestration_surface.warnings.len(),
    })
}

async fn orchestration(State(state): State<AppState>) -> Json<OrchestrationSurfaceResponse> {
    Json((*state.orchestration_surface).clone())
}

async fn list_pipelines(State(state): State<AppState>) -> Json<Vec<PipelineDescriptor>> {
    Json((*state.pipelines).clone())
}

async fn list_history(State(state): State<AppState>) -> Json<Vec<RunResponse>> {
    let history = state.history.lock().await;
    Json(history.clone())
}

async fn run_experiment(
    State(state): State<AppState>,
    AxumPath(experiment_id): AxumPath<String>,
    payload: Option<Json<RunRequest>>,
) -> impl IntoResponse {
    let profile = payload
        .map(|p| p.0.profile.unwrap_or_default())
        .unwrap_or_default();
    match execute_thesis_async(experiment_id.clone(), profile).await {
        Ok(result) => {
            let recorded = push_history(&state, result).await;
            (StatusCode::OK, Json(recorded)).into_response()
        }
        Err(err) => {
            if is_known_pipeline(&state, &experiment_id) {
                api_error(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    "execution_failed",
                    err,
                    known_pipeline_ids(&state),
                    json!({"experiment_id": experiment_id}),
                )
            } else {
                api_error(
                    StatusCode::NOT_FOUND,
                    "unknown_pipeline",
                    err,
                    known_pipeline_ids(&state),
                    json!({"experiment_id": experiment_id}),
                )
            }
        }
    }
}

async fn run_suite(
    State(state): State<AppState>,
    payload: Option<Json<RunRequest>>,
) -> impl IntoResponse {
    let profile = payload
        .map(|p| p.0.profile.unwrap_or_default())
        .unwrap_or_default();
    let start = Instant::now();
    let ids = known_pipeline_ids(&state);
    let (results, failures) = run_batch(&state, &ids, profile).await;

    let pass_count = results.iter().filter(|r| r.passes_gate).count();
    let fail_count = results.len().saturating_sub(pass_count);
    let success_rate = if results.is_empty() {
        0.0
    } else {
        pass_count as f64 / results.len() as f64
    };
    let response = SuiteResponse {
        api_version: API_VERSION,
        profile,
        total_duration_ms: start.elapsed().as_millis(),
        pass_count,
        fail_count,
        success_rate,
        results,
        failures,
    };
    let status = if response.failures.is_empty() {
        StatusCode::OK
    } else {
        StatusCode::MULTI_STATUS
    };
    (status, Json(response)).into_response()
}

async fn benchmark_experiment(
    State(state): State<AppState>,
    AxumPath(experiment_id): AxumPath<String>,
    payload: Option<Json<BenchmarkRequest>>,
) -> impl IntoResponse {
    if !is_known_pipeline(&state, &experiment_id) {
        return api_error(
            StatusCode::NOT_FOUND,
            "unknown_pipeline",
            format!("unknown experiment id: {experiment_id}"),
            known_pipeline_ids(&state),
            json!({"experiment_id": experiment_id}),
        );
    }

    let request = payload.map(|p| p.0).unwrap_or_default();
    let profile = request.profile.unwrap_or_default();
    let iterations = bounded_iterations(request.iterations, 5, 1, 25);
    let ids: Vec<String> = (0..iterations).map(|_| experiment_id.clone()).collect();
    let (runs, failures) = run_batch(&state, &ids, profile).await;

    if runs.is_empty() {
        return api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "benchmark_no_successful_runs",
            "benchmark produced no successful runs",
            known_pipeline_ids(&state),
            json!({ "failures": failures }),
        );
    }

    let mut duration_values: Vec<f64> = runs.iter().map(|run| run.duration_ms as f64).collect();
    let mean_duration_ms = mean(&duration_values);
    let median_duration_ms = median(&mut duration_values);
    let min_duration_ms = runs
        .iter()
        .map(|run| run.duration_ms)
        .min()
        .unwrap_or_default();
    let max_duration_ms = runs
        .iter()
        .map(|run| run.duration_ms)
        .max()
        .unwrap_or_default();

    let metric_values: Vec<f64> = runs.iter().map(|run| run.metric_value).collect();
    let mean_metric_value = mean(&metric_values);
    let metric_stddev = std_dev(&metric_values, mean_metric_value);
    let pass_count = runs.iter().filter(|run| run.passes_gate).count();
    let fail_count = runs.len().saturating_sub(pass_count);

    let response = BenchmarkResponse {
        api_version: API_VERSION,
        experiment_id,
        profile,
        iterations_requested: iterations,
        iterations_completed: runs.len(),
        pass_count,
        fail_count,
        mean_duration_ms,
        median_duration_ms,
        min_duration_ms,
        max_duration_ms,
        mean_metric_value,
        metric_stddev,
        runs,
        failures,
    };
    (StatusCode::OK, Json(response)).into_response()
}

async fn reproducibility_experiment(
    State(state): State<AppState>,
    AxumPath(experiment_id): AxumPath<String>,
    payload: Option<Json<ReproducibilityRequest>>,
) -> impl IntoResponse {
    if !is_known_pipeline(&state, &experiment_id) {
        return api_error(
            StatusCode::NOT_FOUND,
            "unknown_pipeline",
            format!("unknown experiment id: {experiment_id}"),
            known_pipeline_ids(&state),
            json!({"experiment_id": experiment_id}),
        );
    }

    let request = payload.map(|p| p.0).unwrap_or_default();
    let profile = request.profile.unwrap_or_default();
    let iterations = bounded_iterations(request.iterations, 3, 2, 20);
    let tolerance = request.tolerance.unwrap_or(1e-9).max(0.0);
    let ids: Vec<String> = (0..iterations).map(|_| experiment_id.clone()).collect();
    let (runs, failures) = run_batch(&state, &ids, profile).await;

    if runs.is_empty() {
        return api_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "reproducibility_no_successful_runs",
            "reproducibility check produced no successful runs",
            known_pipeline_ids(&state),
            json!({ "failures": failures }),
        );
    }

    let baseline_metric_value = runs[0].metric_value;
    let gate_consistent = runs
        .iter()
        .all(|run| run.passes_gate == runs[0].passes_gate);
    let max_metric_delta = runs
        .iter()
        .map(|run| (run.metric_value - baseline_metric_value).abs())
        .fold(0.0, f64::max);
    let stable = failures.is_empty() && gate_consistent && max_metric_delta <= tolerance;

    let response = ReproducibilityResponse {
        api_version: API_VERSION,
        experiment_id,
        profile,
        iterations_requested: iterations,
        iterations_completed: runs.len(),
        tolerance,
        baseline_metric_value,
        max_metric_delta,
        gate_consistent,
        stable,
        runs,
        failures,
    };
    (StatusCode::OK, Json(response)).into_response()
}

fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/assets/app.js", get(app_js))
        .route("/assets/styles.css", get(styles_css))
        .route("/api/health", get(health))
        .route("/api/version", get(version))
        .route("/api/orchestration", get(orchestration))
        .route("/api/pipelines", get(list_pipelines))
        .route("/api/history", get(list_history))
        .route("/api/run/{experiment_id}", post(run_experiment))
        .route("/api/run-suite", post(run_suite))
        .route("/api/benchmark/{experiment_id}", post(benchmark_experiment))
        .route(
            "/api/reproducibility/{experiment_id}",
            post(reproducibility_experiment),
        )
        .with_state(state)
}

#[tokio::main]
async fn main() -> Result<(), String> {
    let args = Args::parse();
    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|err| format!("failed to bind {addr}: {err}"))?;

    let state = default_state();
    let app = build_router(state);

    println!("gororoba-studio listening on http://{addr}");
    axum::serve(listener, app)
        .await
        .map_err(|err| format!("server exited with error: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::response::Response;
    use serde_json::Value as JsonValue;
    use std::path::Path;

    fn write_temp_catalog(contents: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        let path = std::env::temp_dir().join(format!(
            "gororoba_studio_catalog_{}_{}.toml",
            std::process::id(),
            nanos
        ));
        fs::write(&path, contents).expect("temporary catalog write should succeed");
        path
    }

    fn remove_temp_catalog(path: &Path) {
        let _ = fs::remove_file(path);
    }

    fn make_temp_fixture_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |duration| duration.as_nanos());
        let path = std::env::temp_dir().join(format!(
            "gororoba_studio_{}_{}_{}",
            prefix,
            std::process::id(),
            nanos
        ));
        fs::create_dir_all(&path).expect("fixture directory should be created");
        path
    }

    fn write_fixture_file(root: &Path, name: &str, contents: &str) -> PathBuf {
        let path = root.join(name);
        fs::write(&path, contents).expect("fixture file write should succeed");
        path
    }

    fn remove_temp_fixture_dir(path: &Path) {
        let _ = fs::remove_dir_all(path);
    }

    async fn response_json(response: Response) -> JsonValue {
        let bytes = to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("response body should be readable");
        serde_json::from_slice(&bytes).expect("response body should be valid JSON")
    }

    #[test]
    fn pipeline_catalog_lists_four_entries() {
        let items = default_pipeline_catalog();
        assert_eq!(items.len(), 4);
        assert!(items.iter().all(|item| item.id.starts_with("thesis-")));
    }

    #[test]
    fn registry_catalog_prefers_canonical_metadata() {
        let path = write_temp_catalog(
            r#"
[[pipeline]]
id = "thesis-1"
title = "Catalog T1"
hypothesis = "Catalog hypothesis T1"
primary_metric = "Catalog metric T1"
quick_profile = "Catalog quick T1"
full_profile = "Catalog full T1"
experiment_id = "E-CAT-1"
lineage_id = "XL-CAT-1"
registry_binary = "catalog-binary-1"
artifact_paths = ["data/catalog/t1.toml"]

[[pipeline]]
id = "thesis-2"
title = "Catalog T2"
hypothesis = "Catalog hypothesis T2"
primary_metric = "Catalog metric T2"
quick_profile = "Catalog quick T2"
full_profile = "Catalog full T2"
experiment_id = "E-CAT-2"
lineage_id = "XL-CAT-2"
registry_binary = "catalog-binary-2"
artifact_paths = ["data/catalog/t2.toml"]

[[pipeline]]
id = "thesis-3"
title = "Catalog T3"
hypothesis = "Catalog hypothesis T3"
primary_metric = "Catalog metric T3"
quick_profile = "Catalog quick T3"
full_profile = "Catalog full T3"
experiment_id = "E-CAT-3"
lineage_id = "XL-CAT-3"
registry_binary = "catalog-binary-3"
artifact_paths = ["data/catalog/t3.toml"]

[[pipeline]]
id = "thesis-4"
title = "Catalog T4"
hypothesis = "Catalog hypothesis T4"
primary_metric = "Catalog metric T4"
quick_profile = "Catalog quick T4"
full_profile = "Catalog full T4"
experiment_id = "E-CAT-4"
lineage_id = "XL-CAT-4"
registry_binary = "catalog-binary-4"
artifact_paths = ["data/catalog/t4.toml"]
"#,
        );

        let (catalog, warnings) =
            registry_backed_catalog(&path).expect("catalog should parse and load");
        remove_temp_catalog(&path);

        assert!(warnings.is_empty());
        let thesis_one = catalog
            .iter()
            .find(|pipeline| pipeline.id == "thesis-1")
            .expect("thesis-1 should be present");
        assert_eq!(thesis_one.title, "Catalog T1");
        assert_eq!(thesis_one.experiment_id, "E-CAT-1");
        assert_eq!(thesis_one.registry_binary, "catalog-binary-1");
        assert_eq!(thesis_one.artifact_paths, vec!["data/catalog/t1.toml"]);
    }

    #[test]
    fn registry_catalog_warns_when_required_fields_are_missing() {
        let path = write_temp_catalog(
            r#"
[[pipeline]]
id = "thesis-1"
hypothesis = "Only hypothesis is provided"
primary_metric = "Spearman correlation"
quick_profile = "Quick profile"
full_profile = "Full profile"
"#,
        );

        let (catalog, warnings) =
            registry_backed_catalog(&path).expect("catalog should parse and load");
        remove_temp_catalog(&path);

        assert!(warnings
            .iter()
            .any(|warning| warning.contains("thesis-1 missing required catalog fields: title")));
        let thesis_one = catalog
            .iter()
            .find(|pipeline| pipeline.id == "thesis-1")
            .expect("thesis-1 should be present");
        assert_eq!(thesis_one.title, "T1 Viscous Vacuum Correlation");
    }

    #[test]
    fn unknown_pipeline_id_is_rejected() {
        let result = execute_thesis("unknown", RunProfile::Quick);
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn health_contract_exposes_api_version() {
        let payload = health().await.0;
        assert_eq!(payload.api_version, API_VERSION);
        assert_eq!(payload.service, "gororoba-studio");
        assert_eq!(payload.status, "ok");
        assert!(payload.unix_seconds > 0);
    }

    #[tokio::test]
    async fn version_contract_exposes_catalog_metadata() {
        let state = default_state();
        let payload = version(State(state.clone())).await.0;
        assert_eq!(payload.api_version, API_VERSION);
        assert_eq!(payload.package_version, env!("CARGO_PKG_VERSION"));
        assert_eq!(payload.pipeline_count, state.pipelines.len());
        assert!(!payload.registry_path.is_empty());
        assert_eq!(
            payload.orchestration_lane_count,
            state.orchestration_surface.lane_health_feed.len()
        );
        assert_eq!(
            payload.orchestration_queue_item_count,
            state
                .orchestration_surface
                .promotion_queue_summary
                .item_count
        );
        assert_eq!(
            payload.orchestration_warning_count,
            state.orchestration_surface.warnings.len()
        );
    }

    #[test]
    fn orchestration_surface_uses_deterministic_fallback_for_missing_feeds() {
        let root = make_temp_fixture_dir("missing_feeds");
        let paths = OrchestrationFeedPaths {
            gate_runtime: root.join("missing_gate_runtime.toml"),
            tranche_ledger: root.join("missing_tranche_ledger.toml"),
            intake_ledger: root.join("missing_intake_ledger.toml"),
            promotion_registry: root.join("missing_promotion_registry.toml"),
            reconciliation_report: root.join("missing_reconciliation.toml"),
            binary_reconciliation_report: root.join("missing_binary_reconciliation.toml"),
        };

        let payload = load_orchestration_surface(&paths);
        remove_temp_fixture_dir(&root);

        assert_eq!(payload.api_version, API_VERSION);
        assert_eq!(payload.lane_health_feed.len(), 3);
        assert!(payload
            .lane_health_feed
            .iter()
            .all(|lane| lane.status == "missing"));
        assert_eq!(payload.promotion_queue_summary.item_count, 3);
        assert!(payload
            .promotion_queue_summary
            .items
            .iter()
            .all(|item| item.status == "missing"));
        assert!(payload.warnings.len() >= 6);
    }

    #[test]
    fn orchestration_surface_parses_lane_and_queue_evidence_links() {
        let root = make_temp_fixture_dir("feed_parse");
        let gate_runtime = write_fixture_file(
            &root,
            "gate_runtime.toml",
            r#"
[report]
status = "completed"

[[resolved_blocker]]
evidence = ["reports/gate_fix.toml"]

[final_state]
governance_gate = "pass"
registry_acceptance_gate = "pass"
open_blockers = 0
"#,
        );
        let tranche_ledger = write_fixture_file(
            &root,
            "tranche_ledger.toml",
            r#"
[ledger]
status = "active"
completed_count = 2
in_progress_count = 1
queued_count = 0
source_ledgers = ["reports/tranche4_execution_ledger_2026_02_14.toml"]

[[completed]]
evidence_refs = ["reports/tranche_done.toml"]

[[in_progress]]
evidence_refs = ["reports/tranche_in_progress.toml"]

[[ownership]]
artifacts = ["reports/ownership_trace.toml"]

[consistency]
status = "pass"
"#,
        );
        let intake_ledger = write_fixture_file(
            &root,
            "intake_ledger.toml",
            r#"
[ledger]
status = "active"

[[entry]]
status = "success"

[[entry]]
status = "success"
"#,
        );
        let promotion_registry = write_fixture_file(
            &root,
            "promotion_registry.toml",
            r#"
[[promotion]]
id = "PROM-1"
candidate_kind = "taxonomy_claim"
statement = "Promote taxonomy claim"
evidence_anchor = "reports/promotion_evidence.toml"
where_stated = "reports/promotion_source.toml"
evidence_status = "supported"
claim_lane_ready = true
task_lane_ready = true
ticket_lane_ready = false
"#,
        );
        let reconciliation_report = write_fixture_file(
            &root,
            "reconciliation.toml",
            r#"
[report]
intake_path = "registry/research_intake_2026_02_14.toml"
sensational_mapping_path = "reports/sensational_primary_source_mapping_2026_02_14.toml"
sensational_validation_path = "reports/sensational_primary_validation_2026_02_14.toml"

[[resolved_sync]]
id = "RI-100"
action = "Canonicalized DOI mapping"
value = "https://doi.org/10.1111/example"

[[unresolved]]
id = "RI-101"
reason = "Missing direct citation evidence."
candidate_url = "https://arxiv.org/abs/2501.00001"
"#,
        );
        let binary_reconciliation = write_fixture_file(
            &root,
            "binary_reconciliation.toml",
            r#"
[[binary_resolution]]
name = "registry-event-tracker"
status = "added_to_cargo"
path = "crates/gororoba_cli/src/bin/registry_event_tracker.rs"
"#,
        );

        let paths = OrchestrationFeedPaths {
            gate_runtime,
            tranche_ledger,
            intake_ledger,
            promotion_registry,
            reconciliation_report,
            binary_reconciliation_report: binary_reconciliation,
        };
        let payload = load_orchestration_surface(&paths);
        remove_temp_fixture_dir(&root);

        assert!(payload.warnings.is_empty());
        let runtime_lane = payload
            .lane_health_feed
            .iter()
            .find(|lane| lane.lane_id == "runtime-gates")
            .expect("runtime lane should exist");
        assert_eq!(runtime_lane.status, "healthy");
        assert!(runtime_lane
            .evidence_links
            .contains(&"reports/gate_fix.toml".to_string()));

        let queue_item = payload
            .promotion_queue_summary
            .items
            .iter()
            .find(|item| item.queue_item_id == "PROM-1")
            .expect("promotion queue item should exist");
        assert_eq!(queue_item.status, "ready");
        assert!(queue_item
            .evidence_links
            .contains(&"reports/promotion_evidence.toml".to_string()));
        assert_eq!(payload.promotion_queue_summary.item_count, 4);
        assert_eq!(
            payload.promotion_queue_summary.status_counts.get("ready"),
            Some(&1)
        );
        assert_eq!(
            payload
                .promotion_queue_summary
                .status_counts
                .get("reconciled"),
            Some(&2)
        );
        assert_eq!(
            payload
                .promotion_queue_summary
                .status_counts
                .get("needs_evidence"),
            Some(&1)
        );
    }

    #[tokio::test]
    async fn orchestration_contract_exposes_lane_health_and_queue_summary() {
        let state = default_state();
        let payload = orchestration(State(state)).await.0;
        assert_eq!(payload.api_version, API_VERSION);
        assert_eq!(payload.lane_health_feed.len(), 3);
        assert_eq!(
            payload.promotion_queue_summary.item_count,
            payload.promotion_queue_summary.items.len()
        );
        assert!(payload
            .lane_health_feed
            .iter()
            .all(|lane| !lane.evidence_links.is_empty()));
        assert!(payload
            .promotion_queue_summary
            .items
            .iter()
            .all(|item| !item.evidence_links.is_empty()));
    }

    #[test]
    fn thesis_two_quick_profile_runs() {
        let result = execute_thesis("thesis-2", RunProfile::Quick)
            .expect("thesis-2 quick profile should execute");
        assert_eq!(result.experiment_id, "thesis-2");
        assert!(result.metric_value.is_finite());
        assert!(result.duration_ms < 300_000);
        assert_eq!(result.api_version, API_VERSION);
        assert_eq!(
            result.config_snapshot["profile"],
            JsonValue::String("quick".to_string())
        );
        assert!(result.config_snapshot["parameters"]
            .as_object()
            .is_some_and(|obj| !obj.is_empty()));
    }

    #[test]
    fn reproducibility_metric_is_stable_for_quick_profile() {
        let first =
            execute_thesis("thesis-4", RunProfile::Quick).expect("first run should succeed");
        let second =
            execute_thesis("thesis-4", RunProfile::Quick).expect("second run should succeed");
        let delta = (first.metric_value - second.metric_value).abs();
        assert!(delta <= 1e-12);
    }

    #[tokio::test]
    async fn run_batch_collects_partial_failures() {
        let state = default_state();
        let ids = vec!["thesis-1".to_string(), "unknown".to_string()];
        let (runs, failures) = run_batch(&state, &ids, RunProfile::Quick).await;
        assert_eq!(runs.len(), 1);
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].experiment_id, "unknown");
        assert_eq!(failures[0].api_version, API_VERSION);
    }

    #[tokio::test]
    async fn run_endpoint_and_history_contract() {
        let state = default_state();
        let before = list_history(State(state.clone())).await.0;
        assert!(before.is_empty());

        let response = run_experiment(
            State(state.clone()),
            AxumPath("thesis-1".to_string()),
            Some(Json(RunRequest {
                profile: Some(RunProfile::Quick),
            })),
        )
        .await
        .into_response();
        assert_eq!(response.status(), StatusCode::OK);
        let payload = response_json(response).await;
        assert_eq!(
            payload["api_version"],
            JsonValue::String(API_VERSION.to_string())
        );
        assert_eq!(
            payload["experiment_id"],
            JsonValue::String("thesis-1".to_string())
        );
        assert!(payload["run_id"].as_u64().unwrap_or_default() >= 1);
        assert!(payload["artifact_links"].is_array());

        let after = list_history(State(state)).await.0;
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].experiment_id, "thesis-1");
    }

    #[tokio::test]
    async fn unknown_pipeline_uses_standard_error_shape() {
        let state = default_state();
        let response = run_experiment(
            State(state),
            AxumPath("nope".to_string()),
            Some(Json(RunRequest {
                profile: Some(RunProfile::Quick),
            })),
        )
        .await
        .into_response();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let payload = response_json(response).await;
        assert_eq!(
            payload["api_version"],
            JsonValue::String(API_VERSION.to_string())
        );
        assert_eq!(
            payload["error_code"],
            JsonValue::String("unknown_pipeline".to_string())
        );
        assert!(payload["known_ids"].is_array());
    }

    #[tokio::test]
    async fn suite_benchmark_and_repro_contracts_hold() {
        let state = default_state();
        let suite_response = run_suite(
            State(state.clone()),
            Some(Json(RunRequest {
                profile: Some(RunProfile::Quick),
            })),
        )
        .await
        .into_response();
        assert!(
            suite_response.status() == StatusCode::OK
                || suite_response.status() == StatusCode::MULTI_STATUS
        );
        let suite_payload = response_json(suite_response).await;
        assert_eq!(
            suite_payload["api_version"],
            JsonValue::String(API_VERSION.to_string())
        );
        let result_count = suite_payload["results"]
            .as_array()
            .map_or(0, |items| items.len());
        let pass_count = suite_payload["pass_count"].as_u64().unwrap_or_default() as usize;
        let fail_count = suite_payload["fail_count"].as_u64().unwrap_or_default() as usize;
        assert_eq!(result_count, pass_count + fail_count);

        let benchmark_response = benchmark_experiment(
            State(state.clone()),
            AxumPath("thesis-2".to_string()),
            Some(Json(BenchmarkRequest {
                profile: Some(RunProfile::Quick),
                iterations: Some(3),
            })),
        )
        .await
        .into_response();
        assert_eq!(benchmark_response.status(), StatusCode::OK);
        let benchmark_payload = response_json(benchmark_response).await;
        assert_eq!(
            benchmark_payload["api_version"],
            JsonValue::String(API_VERSION.to_string())
        );
        assert_eq!(benchmark_payload["iterations_requested"].as_u64(), Some(3));
        let benchmark_runs = benchmark_payload["runs"]
            .as_array()
            .map_or(0, |items| items.len());
        assert!(benchmark_runs >= 1);

        let repro_response = reproducibility_experiment(
            State(state),
            AxumPath("thesis-4".to_string()),
            Some(Json(ReproducibilityRequest {
                profile: Some(RunProfile::Quick),
                iterations: Some(3),
                tolerance: Some(1e-9),
            })),
        )
        .await
        .into_response();
        assert_eq!(repro_response.status(), StatusCode::OK);
        let repro_payload = response_json(repro_response).await;
        assert_eq!(
            repro_payload["api_version"],
            JsonValue::String(API_VERSION.to_string())
        );
        assert!(repro_payload["stable"].is_boolean());
    }

    #[tokio::test]
    async fn index_and_assets_contracts_are_served() {
        let html = index().await;
        assert!(html.0.contains("Gororoba Studio"));

        let js_response = app_js().await.into_response();
        assert_eq!(js_response.status(), StatusCode::OK);
        let js_body = to_bytes(js_response.into_body(), usize::MAX)
            .await
            .expect("js body should be readable");
        let js_text = String::from_utf8(js_body.to_vec()).expect("js should be utf8");
        assert!(js_text.contains("runBenchmarkBtn"));

        let css_response = styles_css().await.into_response();
        assert_eq!(css_response.status(), StatusCode::OK);
        let css_body = to_bytes(css_response.into_body(), usize::MAX)
            .await
            .expect("css body should be readable");
        let css_text = String::from_utf8(css_body.to_vec()).expect("css should be utf8");
        assert!(css_text.contains(".pipeline-status"));
    }
}
