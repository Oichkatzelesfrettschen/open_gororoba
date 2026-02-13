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
}

#[derive(Debug, Clone, Serialize)]
struct PipelineDescriptor {
    id: &'static str,
    title: &'static str,
    hypothesis: &'static str,
    primary_metric: &'static str,
    quick_profile: &'static str,
    full_profile: &'static str,
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
    service: &'static str,
    status: &'static str,
    unix_seconds: u64,
}

#[derive(Debug, Clone, Serialize)]
struct RunResponse {
    run_id: u64,
    unix_seconds: u64,
    experiment_id: String,
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

const RUN_TIMEOUT_SECONDS: u64 = 60;
const MAX_HISTORY_ITEMS: usize = 200;

fn now_unix_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |dur| dur.as_secs())
}

fn default_state() -> AppState {
    AppState {
        pipelines: Arc::new(pipeline_catalog()),
        history: Arc::new(Mutex::new(Vec::new())),
        run_counter: Arc::new(AtomicU64::new(1)),
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

fn pipeline_catalog() -> Vec<PipelineDescriptor> {
    vec![
        PipelineDescriptor {
            id: "thesis-1",
            title: "T1 Viscous Vacuum Correlation",
            hypothesis: "Frustration density and viscosity should remain strongly coupled in spatial slices.",
            primary_metric: "Spearman correlation",
            quick_profile: "8^3 field, subregions=2",
            full_profile: "16^3 field, subregions=2",
        },
        PipelineDescriptor {
            id: "thesis-2",
            title: "T2 Non-Newtonian Thickening",
            hypothesis: "Associator-coupled viscosity should increase with strain-rate under non-Newtonian conditions.",
            primary_metric: "Viscosity ratio (high/low)",
            quick_profile: "alpha=0.4, n=1.25",
            full_profile: "alpha=0.5, n=1.5",
        },
        PipelineDescriptor {
            id: "thesis-3",
            title: "T3 Plateau to Epoch Alignment",
            hypothesis: "Loss plateaus should align with interpretable epoch markers while preserving hubble consistency.",
            primary_metric: "Plateau count",
            quick_profile: "32 epochs",
            full_profile: "96 epochs",
        },
        PipelineDescriptor {
            id: "thesis-4",
            title: "T4 Latency Scaling Law",
            hypothesis: "Collision return-time statistics should exhibit inverse-square style scaling.",
            primary_metric: "Inverse-square R^2",
            quick_profile: "500 steps",
            full_profile: "2000 steps",
        },
    ]
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
        run_id: 0,
        unix_seconds: now_unix_seconds(),
        experiment_id: experiment_id.to_string(),
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
        .map(|entry| entry.id.to_string())
        .collect()
}

fn is_known_pipeline(state: &AppState, experiment_id: &str) -> bool {
    state
        .pipelines
        .iter()
        .any(|entry| entry.id == experiment_id)
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
        service: "gororoba-studio",
        status: "ok",
        unix_seconds,
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
        Err(err) => (
            if is_known_pipeline(&state, &experiment_id) {
                StatusCode::INTERNAL_SERVER_ERROR
            } else {
                StatusCode::NOT_FOUND
            },
            Json(json!({
                "error": err,
                "known_ids": known_pipeline_ids(&state)
            })),
        )
            .into_response(),
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
        return (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": format!("unknown experiment id: {experiment_id}"),
                "known_ids": known_pipeline_ids(&state),
            })),
        )
            .into_response();
    }

    let request = payload.map(|p| p.0).unwrap_or_default();
    let profile = request.profile.unwrap_or_default();
    let iterations = bounded_iterations(request.iterations, 5, 1, 25);
    let ids: Vec<String> = (0..iterations).map(|_| experiment_id.clone()).collect();
    let (runs, failures) = run_batch(&state, &ids, profile).await;

    if runs.is_empty() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": "benchmark produced no successful runs",
                "failures": failures,
            })),
        )
            .into_response();
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
        return (
            StatusCode::NOT_FOUND,
            Json(json!({
                "error": format!("unknown experiment id: {experiment_id}"),
                "known_ids": known_pipeline_ids(&state),
            })),
        )
            .into_response();
    }

    let request = payload.map(|p| p.0).unwrap_or_default();
    let profile = request.profile.unwrap_or_default();
    let iterations = bounded_iterations(request.iterations, 3, 2, 20);
    let tolerance = request.tolerance.unwrap_or(1e-9).max(0.0);
    let ids: Vec<String> = (0..iterations).map(|_| experiment_id.clone()).collect();
    let (runs, failures) = run_batch(&state, &ids, profile).await;

    if runs.is_empty() {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({
                "error": "reproducibility check produced no successful runs",
                "failures": failures,
            })),
        )
            .into_response();
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

#[tokio::main]
async fn main() -> Result<(), String> {
    let args = Args::parse();
    let addr = format!("{}:{}", args.host, args.port);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|err| format!("failed to bind {addr}: {err}"))?;

    let state = default_state();
    let app = Router::new()
        .route("/", get(index))
        .route("/assets/app.js", get(app_js))
        .route("/assets/styles.css", get(styles_css))
        .route("/api/health", get(health))
        .route("/api/pipelines", get(list_pipelines))
        .route("/api/history", get(list_history))
        .route("/api/run/{experiment_id}", post(run_experiment))
        .route("/api/run-suite", post(run_suite))
        .route("/api/benchmark/{experiment_id}", post(benchmark_experiment))
        .route(
            "/api/reproducibility/{experiment_id}",
            post(reproducibility_experiment),
        )
        .with_state(state);

    println!("gororoba-studio listening on http://{addr}");
    axum::serve(listener, app)
        .await
        .map_err(|err| format!("server exited with error: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value as JsonValue;

    #[test]
    fn pipeline_catalog_lists_four_entries() {
        let items = pipeline_catalog();
        assert_eq!(items.len(), 4);
        assert!(items.iter().all(|item| item.id.starts_with("thesis-")));
    }

    #[test]
    fn unknown_pipeline_id_is_rejected() {
        let result = execute_thesis("unknown", RunProfile::Quick);
        assert!(result.is_err());
    }

    #[test]
    fn thesis_two_quick_profile_runs() {
        let result = execute_thesis("thesis-2", RunProfile::Quick)
            .expect("thesis-2 quick profile should execute");
        assert_eq!(result.experiment_id, "thesis-2");
        assert!(result.metric_value.is_finite());
        assert!(result.duration_ms <= u128::MAX);
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
    }
}
