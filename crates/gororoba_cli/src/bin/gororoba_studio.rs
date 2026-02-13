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
use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};
use tokio::sync::Mutex;
use tokio::time::{timeout, Duration};

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
struct ExperimentsRegistry {
    #[serde(default)]
    experiment: Vec<RegistryExperiment>,
}

#[derive(Debug, Clone, Deserialize)]
struct RegistryExperiment {
    id: String,
    title: String,
    binary: String,
    #[serde(default)]
    method: String,
    #[serde(default)]
    lineage_id: String,
    #[serde(default)]
    output_path_refs: Vec<String>,
    #[serde(default)]
    output: Vec<String>,
}

const RUN_TIMEOUT_SECONDS: u64 = 60;
const MAX_HISTORY_ITEMS: usize = 200;
const API_VERSION: &str = "studio.v1";

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |dur| dur.as_secs())
}

fn locate_experiments_registry() -> PathBuf {
    if let Ok(path) = std::env::var("GOROROBA_EXPERIMENTS_REGISTRY") {
        return PathBuf::from(path);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../registry/experiments.toml")
        .canonicalize()
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../registry/experiments.toml")
        })
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

fn pick_registry_experiment<'a>(
    experiments: &'a [RegistryExperiment],
    thesis_id: &str,
) -> Option<&'a RegistryExperiment> {
    let lower = |text: &str| text.to_ascii_lowercase();
    match thesis_id {
        "thesis-1" => experiments
            .iter()
            .find(|exp| lower(&exp.title).contains("thesis 1")),
        "thesis-2" => experiments.iter().find(|exp| {
            let title = lower(&exp.title);
            title.contains("thesis 2") && title.contains("3d associator-coupled")
        }),
        "thesis-3" => experiments
            .iter()
            .find(|exp| lower(&exp.title).contains("thesis 3")),
        "thesis-4" => experiments
            .iter()
            .find(|exp| lower(&exp.title).contains("t1 x t4"))
            .or_else(|| {
                experiments
                    .iter()
                    .find(|exp| lower(&exp.method).contains("t4 (shell return-time power law)"))
            }),
        _ => None,
    }
}

fn registry_backed_catalog(
    registry_path: &PathBuf,
) -> Result<(Vec<PipelineDescriptor>, Vec<String>), String> {
    let raw = fs::read_to_string(registry_path)
        .map_err(|err| format!("failed to read {}: {err}", registry_path.display()))?;
    let parsed: ExperimentsRegistry = toml::from_str(&raw)
        .map_err(|err| format!("failed to parse experiments registry: {err}"))?;
    let mut catalog = default_pipeline_catalog();
    let mut warnings = Vec::new();

    for pipeline in &mut catalog {
        if let Some(exp) = pick_registry_experiment(&parsed.experiment, &pipeline.id) {
            pipeline.title = exp.title.clone();
            pipeline.experiment_id = exp.id.clone();
            pipeline.lineage_id = exp.lineage_id.clone();
            pipeline.registry_binary = exp.binary.clone();
            pipeline.artifact_paths = merge_artifact_paths(&exp.output_path_refs, &exp.output);
            if pipeline.artifact_paths.is_empty() {
                warnings.push(format!(
                    "{} mapped to {} but no artifact paths were declared",
                    pipeline.id, exp.id
                ));
            }
        } else {
            warnings.push(format!(
                "{} missing in registry mapping; using fallback catalog descriptor",
                pipeline.id
            ));
        }
    }

    Ok((catalog, warnings))
}

fn default_state() -> AppState {
    let registry_path = locate_experiments_registry();
    let registry_path_text = registry_path.display().to_string();
    let (pipelines, catalog_source, warnings) = match registry_backed_catalog(&registry_path) {
        Ok((catalog, warnings)) => (catalog, "registry".to_string(), warnings),
        Err(err) => (
            default_pipeline_catalog(),
            "fallback".to_string(),
            vec![format!("registry catalog load failed: {err}")],
        ),
    };
    AppState {
        pipelines: Arc::new(pipelines),
        history: Arc::new(Mutex::new(Vec::new())),
        run_counter: Arc::new(AtomicU64::new(1)),
        catalog_source,
        catalog_warnings: Arc::new(warnings),
        registry_path: registry_path_text,
    }
}

fn bounded_iterations(value: Option<usize>, default_value: usize, min: usize, max: usize) -> usize {
    let raw = value.unwrap_or(default_value);
    raw.clamp(min, max)
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) * 0.5
    } else {
        values[mid]
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn stddev(values: &[f64], mean_value: f64) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean_value;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
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
    })
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
    let metric_stddev = stddev(&metric_values, mean_metric_value);
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
    }

    #[test]
    fn thesis_two_quick_profile_runs() {
        let result = execute_thesis("thesis-2", RunProfile::Quick)
            .expect("thesis-2 quick profile should execute");
        assert_eq!(result.experiment_id, "thesis-2");
        assert!(result.metric_value.is_finite());
        assert!(result.duration_ms <= u128::MAX);
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
