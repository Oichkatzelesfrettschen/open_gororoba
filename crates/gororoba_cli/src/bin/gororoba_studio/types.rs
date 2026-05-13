//! Type definitions for the `gororoba-studio` interactive web
//! studio: clap Args + AppState + ~20 request/response DTOs for the
//! axum HTTP API + studio pipeline catalog types + UI asset
//! include_str! constants + per-endpoint timeout/version constants.
//!
//! Fields and constants are pub(crate). Uses #[path] indirection
//! because the binary has an explicit Cargo.toml path.

use clap::Parser;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use tokio::sync::Mutex;
use serde_json::Value;

pub(crate) const INDEX_HTML: &str = include_str!("../../../../../apps/gororoba_studio/ui/index.html");
pub(crate) const APP_JS: &str = include_str!("../../../../../apps/gororoba_studio/ui/app.js");
pub(crate) const STYLES_CSS: &str = include_str!("../../../../../apps/gororoba_studio/ui/styles.css");

#[derive(Debug, Parser)]
#[command(
    name = "gororoba-studio",
    about = "Interactive web studio for thesis pipelines, live evidence runs, and benchmark snapshots."
)]
pub(crate) struct Args {
    #[arg(long, default_value = "127.0.0.1")]
    pub(crate) host: String,
    #[arg(long, default_value_t = 8088)]
    pub(crate) port: u16,
}

#[derive(Debug, Clone)]
pub(crate) struct AppState {
    pub(crate) pipelines: Arc<Vec<PipelineDescriptor>>,
    pub(crate) history: Arc<Mutex<Vec<RunResponse>>>,
    pub(crate) run_counter: Arc<AtomicU64>,
    pub(crate) catalog_source: String,
    pub(crate) catalog_warnings: Arc<Vec<String>>,
    pub(crate) registry_path: String,
    pub(crate) orchestration_surface: Arc<OrchestrationSurfaceResponse>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct PipelineDescriptor {
    pub(crate) id: String,
    pub(crate) title: String,
    pub(crate) hypothesis: String,
    pub(crate) primary_metric: String,
    pub(crate) quick_profile: String,
    pub(crate) full_profile: String,
    pub(crate) experiment_id: String,
    pub(crate) lineage_id: String,
    pub(crate) registry_binary: String,
    pub(crate) artifact_paths: Vec<String>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum RunProfile {
    #[default]
    Quick,
    Full,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct RunRequest {
    pub(crate) profile: Option<RunProfile>,
}

#[derive(Debug, Serialize)]
pub(crate) struct HealthResponse {
    pub(crate) api_version: &'static str,
    pub(crate) service: &'static str,
    pub(crate) status: &'static str,
    pub(crate) unix_seconds: u64,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RunResponse {
    pub(crate) api_version: &'static str,
    pub(crate) run_id: u64,
    pub(crate) unix_seconds: u64,
    pub(crate) experiment_id: String,
    pub(crate) source_experiment_id: Option<String>,
    pub(crate) source_lineage_id: Option<String>,
    pub(crate) artifact_links: Vec<String>,
    pub(crate) profile: RunProfile,
    pub(crate) duration_ms: u128,
    pub(crate) thesis_id: usize,
    pub(crate) label: String,
    pub(crate) metric_value: f64,
    pub(crate) threshold: f64,
    pub(crate) passes_gate: bool,
    pub(crate) config_snapshot: Value,
    pub(crate) messages: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct SuiteResponse {
    pub(crate) api_version: &'static str,
    pub(crate) profile: RunProfile,
    pub(crate) total_duration_ms: u128,
    pub(crate) pass_count: usize,
    pub(crate) fail_count: usize,
    pub(crate) success_rate: f64,
    pub(crate) results: Vec<RunResponse>,
    pub(crate) failures: Vec<RunFailure>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RunFailure {
    pub(crate) api_version: &'static str,
    pub(crate) experiment_id: String,
    pub(crate) profile: RunProfile,
    pub(crate) error: String,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct BenchmarkRequest {
    pub(crate) profile: Option<RunProfile>,
    pub(crate) iterations: Option<usize>,
}

#[derive(Debug, Serialize)]
pub(crate) struct BenchmarkResponse {
    pub(crate) api_version: &'static str,
    pub(crate) experiment_id: String,
    pub(crate) profile: RunProfile,
    pub(crate) iterations_requested: usize,
    pub(crate) iterations_completed: usize,
    pub(crate) pass_count: usize,
    pub(crate) fail_count: usize,
    pub(crate) mean_duration_ms: f64,
    pub(crate) median_duration_ms: f64,
    pub(crate) min_duration_ms: u128,
    pub(crate) max_duration_ms: u128,
    pub(crate) mean_metric_value: f64,
    pub(crate) metric_stddev: f64,
    pub(crate) runs: Vec<RunResponse>,
    pub(crate) failures: Vec<RunFailure>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub(crate) struct ReproducibilityRequest {
    pub(crate) profile: Option<RunProfile>,
    pub(crate) iterations: Option<usize>,
    pub(crate) tolerance: Option<f64>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ReproducibilityResponse {
    pub(crate) api_version: &'static str,
    pub(crate) experiment_id: String,
    pub(crate) profile: RunProfile,
    pub(crate) iterations_requested: usize,
    pub(crate) iterations_completed: usize,
    pub(crate) tolerance: f64,
    pub(crate) baseline_metric_value: f64,
    pub(crate) max_metric_delta: f64,
    pub(crate) gate_consistent: bool,
    pub(crate) stable: bool,
    pub(crate) runs: Vec<RunResponse>,
    pub(crate) failures: Vec<RunFailure>,
}

#[derive(Debug, Serialize)]
pub(crate) struct VersionResponse {
    pub(crate) api_version: &'static str,
    pub(crate) service: &'static str,
    pub(crate) package_version: &'static str,
    pub(crate) catalog_source: String,
    pub(crate) pipeline_count: usize,
    pub(crate) catalog_warnings: Vec<String>,
    pub(crate) registry_path: String,
    pub(crate) orchestration_lane_count: usize,
    pub(crate) orchestration_queue_item_count: usize,
    pub(crate) orchestration_warning_count: usize,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct OrchestrationLaneHealth {
    pub(crate) lane_id: String,
    pub(crate) lane_label: String,
    pub(crate) source_kind: String,
    pub(crate) source_path: String,
    pub(crate) status: String,
    pub(crate) summary: String,
    pub(crate) evidence_links: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct OrchestrationQueueItem {
    pub(crate) queue_item_id: String,
    pub(crate) queue_kind: String,
    pub(crate) source_path: String,
    pub(crate) status: String,
    pub(crate) summary: String,
    pub(crate) evidence_links: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct PromotionQueueSummary {
    pub(crate) source_paths: Vec<String>,
    pub(crate) item_count: usize,
    pub(crate) ready_count: usize,
    pub(crate) pending_count: usize,
    pub(crate) blocked_count: usize,
    pub(crate) status_counts: BTreeMap<String, usize>,
    pub(crate) items: Vec<OrchestrationQueueItem>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct OrchestrationSurfaceResponse {
    pub(crate) api_version: &'static str,
    pub(crate) lane_health_feed: Vec<OrchestrationLaneHealth>,
    pub(crate) promotion_queue_summary: PromotionQueueSummary,
    pub(crate) warnings: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(crate) struct ApiErrorResponse {
    pub(crate) api_version: &'static str,
    pub(crate) error_code: &'static str,
    pub(crate) message: String,
    pub(crate) known_ids: Vec<String>,
    pub(crate) details: Value,
}

#[derive(Debug, Deserialize)]
pub(crate) struct StudioPipelineCatalog {
    #[serde(default)]
    pub(crate) pipeline: Vec<StudioPipelineCatalogEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct StudioPipelineCatalogEntry {
    pub(crate) id: Option<String>,
    pub(crate) title: Option<String>,
    pub(crate) hypothesis: Option<String>,
    pub(crate) primary_metric: Option<String>,
    pub(crate) quick_profile: Option<String>,
    pub(crate) full_profile: Option<String>,
    pub(crate) experiment_id: Option<String>,
    pub(crate) lineage_id: Option<String>,
    pub(crate) registry_binary: Option<String>,
    #[serde(default)]
    pub(crate) artifact_paths: Vec<String>,
}

pub(crate) const RUN_TIMEOUT_SECONDS: u64 = 60;
pub(crate) const MAX_HISTORY_ITEMS: usize = 200;
pub(crate) const API_VERSION: &str = "studio.v1";
pub(crate) const ORCH_DEFAULT_GATE_RUNTIME_PATH: &str = "reports/gate_validation_runtime_2026_02_14.toml";
pub(crate) const ORCH_DEFAULT_TRANCHE_LEDGER_PATH: &str = "reports/tranche5_execution_ledger_2026_02_14.toml";
pub(crate) const ORCH_DEFAULT_INTAKE_LEDGER_PATH: &str =
    "reports/research_intake_execution_ledger_2026_02_14.toml";
pub(crate) const ORCH_DEFAULT_PROMOTION_REGISTRY_PATH: &str = "registry/hypercomplex_taxonomy_promotion.toml";
pub(crate) const ORCH_DEFAULT_RECONCILIATION_PATH: &str =
    "reports/research_intake_reconciliation_2026_02_14.toml";
pub(crate) const ORCH_DEFAULT_BINARY_RECONCILIATION_PATH: &str =
    "reports/binary_project_drift_reconciliation_2026_02_14.toml";
