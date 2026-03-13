use anyhow::{Context, Result, bail};
use chrono::Utc;
use clap::{Parser, Subcommand, ValueEnum};
use csv::ReaderBuilder;
use data_core::download_stack::{
    DEFAULT_PROBE_BYTES, DownloadBackend, DownloadLedgerRow, DownloadStack, TransferRequest,
    TransferTrace, load_host_policy_registry,
};
use provenance_core::{DownloadAttemptRecord, DownloadCampaignRecord, DownloadJobRecord};
use provenance_store::ProvenanceStore;
use std::{
    collections::BTreeSet,
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(
    name = "download-recovery",
    about = "Universalized Rust download and recovery CLI with standardized ledger output"
)]
struct Cli {
    #[arg(long, default_value = "registry/download_host_policies.toml")]
    policy_registry: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Route(RouteArgs),
    Probe(ProbeArgs),
    ProbeBatch(ProbeBatchArgs),
    ProbeOverrides(ProbeOverridesArgs),
    Recover(RecoverArgs),
    RecoverBatch(RecoverBatchArgs),
    ExportLedger(ExportLedgerArgs),
}

#[derive(Parser, Debug)]
struct RouteArgs {
    #[arg(long)]
    url: String,

    #[arg(long, value_enum, default_value_t = TransferKindArg::Probe)]
    kind: TransferKindArg,

    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    backend: BackendArg,
}

#[derive(Parser, Debug)]
struct ProbeArgs {
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    db: PathBuf,

    #[arg(long)]
    url: String,

    #[arg(long)]
    id: Option<String>,

    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    backend: BackendArg,

    #[arg(long, default_value_t = DEFAULT_PROBE_BYTES)]
    probe_bytes: usize,

    #[arg(long)]
    note: Option<String>,

    #[arg(long)]
    out_ledger: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct ProbeBatchArgs {
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    db: PathBuf,

    #[arg(long)]
    input: PathBuf,

    #[arg(long)]
    out_ledger: PathBuf,

    #[arg(long)]
    campaign_name: Option<String>,

    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    backend: BackendArg,

    #[arg(long, default_value_t = DEFAULT_PROBE_BYTES)]
    probe_bytes: usize,
}

#[derive(Parser, Debug)]
struct ProbeOverridesArgs {
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    db: PathBuf,

    #[arg(
        long,
        default_value = "registry/manual_mirror_overrides_2026_02_15.toml"
    )]
    overrides: PathBuf,

    #[arg(long)]
    out_ledger: PathBuf,

    #[arg(long)]
    campaign_name: Option<String>,

    #[arg(long)]
    override_id: Vec<String>,

    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    backend: BackendArg,

    #[arg(long, default_value_t = DEFAULT_PROBE_BYTES)]
    probe_bytes: usize,
}

#[derive(Parser, Debug)]
struct RecoverArgs {
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    db: PathBuf,

    #[arg(long)]
    url: String,

    #[arg(long)]
    dest: PathBuf,

    #[arg(long)]
    id: Option<String>,

    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    backend: BackendArg,

    #[arg(long, default_value_t = DEFAULT_PROBE_BYTES)]
    probe_bytes: usize,

    #[arg(long)]
    note: Option<String>,

    #[arg(long)]
    out_ledger: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct RecoverBatchArgs {
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    db: PathBuf,

    #[arg(long)]
    input: PathBuf,

    #[arg(long)]
    dest_dir: PathBuf,

    #[arg(long)]
    out_ledger: PathBuf,

    #[arg(long)]
    campaign_name: Option<String>,

    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    backend: BackendArg,

    #[arg(long, default_value_t = DEFAULT_PROBE_BYTES)]
    probe_bytes: usize,
}

#[derive(Parser, Debug)]
struct ExportLedgerArgs {
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    db: PathBuf,

    #[arg(long)]
    out_ledger: PathBuf,

    #[arg(long, default_value_t = 200)]
    limit: usize,

    #[arg(long)]
    needle: Option<String>,

    #[arg(long)]
    host: Option<String>,

    #[arg(long)]
    status: Option<String>,

    #[arg(long)]
    backend: Option<String>,
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

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum TransferKindArg {
    Probe,
    Download,
}

#[derive(Debug, serde::Deserialize)]
struct BatchRow {
    url: String,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    note: Option<String>,
    #[serde(default)]
    output_rel: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct MirrorOverridesRegistry {
    #[serde(default, rename = "mirror_override")]
    entries: Vec<MirrorOverrideEntry>,
}

#[derive(Debug, serde::Deserialize)]
struct MirrorOverrideEntry {
    id: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    urls: Vec<String>,
    #[serde(default)]
    notes: Option<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let stack = build_stack(&cli.policy_registry)?;
    match cli.command {
        Commands::Route(args) => run_route(&stack, args),
        Commands::Probe(args) => run_probe(&stack, args),
        Commands::ProbeBatch(args) => run_probe_batch(&stack, args),
        Commands::ProbeOverrides(args) => run_probe_overrides(&stack, args),
        Commands::Recover(args) => run_recover(&stack, args),
        Commands::RecoverBatch(args) => run_recover_batch(&stack, args),
        Commands::ExportLedger(args) => run_export_ledger(args),
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

fn run_route(stack: &DownloadStack, args: RouteArgs) -> Result<()> {
    let mut request = TransferRequest::probe(args.url);
    request.backend = args.backend.into();
    let kind = match args.kind {
        TransferKindArg::Probe => data_core::download_stack::TransferKind::Probe,
        TransferKindArg::Download => data_core::download_stack::TransferKind::Download,
    };
    let route = stack.route(&request, kind);
    println!("scheme={}", route.scheme);
    println!("host={}", route.host.unwrap_or_default());
    println!("retry_class={}", route.retry_class.as_str());
    println!("policy={}", route.policy_name.unwrap_or_default());
    println!(
        "backends={}",
        route
            .backends
            .into_iter()
            .map(|backend| backend.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    Ok(())
}

fn run_probe(stack: &DownloadStack, args: ProbeArgs) -> Result<()> {
    let mut store = ProvenanceStore::open(&args.db)?;
    let mut request = TransferRequest::probe(args.url.clone());
    request.backend = args.backend.into();
    request.probe_bytes = args.probe_bytes;
    request.note = args.note.clone();
    let trace = stack.probe_with_trace(&request);
    record_trace(&mut store, &request, &trace)?;
    let result = trace.clone().into_result(&request.url)?;
    let row = result.to_ledger_row(args.id.unwrap_or_else(|| derive_id(&args.url, 0)));
    emit_rows(&[row], args.out_ledger.as_deref())?;
    Ok(())
}

fn run_probe_batch(stack: &DownloadStack, args: ProbeBatchArgs) -> Result<()> {
    let mut store = ProvenanceStore::open(&args.db)?;
    let rows = load_batch_rows(&args.input)?;
    let campaign_id = create_batch_campaign(
        &mut store,
        args.campaign_name.as_deref(),
        "probe_batch",
        &args.input,
        Some(&args.out_ledger),
        None,
        None,
    )?;
    let mut outputs = Vec::with_capacity(rows.len());
    for (index, row) in rows.into_iter().enumerate() {
        let mut request = TransferRequest::probe(row.url.clone());
        request.backend = args.backend.into();
        request.probe_bytes = args.probe_bytes;
        request.note = row.note.clone();
        let trace = stack.probe_with_trace(&request);
        let job_id = record_trace(&mut store, &request, &trace)?;
        if let Some(campaign_id) = campaign_id {
            store.link_download_job_to_campaign(campaign_id, job_id)?;
        }
        let result = trace
            .clone()
            .into_result(&request.url)
            .with_context(|| format!("probe failed for {}", row.url))?;
        outputs
            .push(result.to_ledger_row(row.id.unwrap_or_else(|| derive_id(&row.url, index + 1))));
    }
    emit_rows(&outputs, Some(&args.out_ledger))?;
    Ok(())
}

fn run_probe_overrides(stack: &DownloadStack, args: ProbeOverridesArgs) -> Result<()> {
    let mut store = ProvenanceStore::open(&args.db)?;
    let entries = load_manual_override_candidates(&args.overrides, &args.override_id)?;
    let campaign_id = create_batch_campaign(
        &mut store,
        args.campaign_name.as_deref(),
        "probe_overrides",
        &args.overrides,
        Some(&args.out_ledger),
        None,
        Some("manual mirror override PDF candidate probe"),
    )?;
    let mut outputs = Vec::with_capacity(entries.len());
    for (index, row) in entries.into_iter().enumerate() {
        let mut request = TransferRequest::probe(row.url.clone());
        request.backend = args.backend.into();
        request.probe_bytes = args.probe_bytes;
        request.note = row.note.clone();
        let trace = stack.probe_with_trace(&request);
        let job_id = record_trace(&mut store, &request, &trace)?;
        if let Some(campaign_id) = campaign_id {
            store.link_download_job_to_campaign(campaign_id, job_id)?;
        }
        outputs.push(trace_to_ledger_row(
            &request,
            &trace,
            row.id.unwrap_or_else(|| derive_id(&row.url, index + 1)),
        ));
    }
    emit_rows(&outputs, Some(&args.out_ledger))?;
    Ok(())
}

fn run_recover(stack: &DownloadStack, args: RecoverArgs) -> Result<()> {
    let mut store = ProvenanceStore::open(&args.db)?;
    let mut request = TransferRequest::download(args.url.clone(), args.dest.clone());
    request.backend = args.backend.into();
    request.probe_bytes = args.probe_bytes;
    request.note = args.note.clone();
    let trace = stack.recover_with_trace(&request);
    record_trace(&mut store, &request, &trace)?;
    let result = trace.clone().into_result(&request.url)?;
    let row = result.to_ledger_row(args.id.unwrap_or_else(|| derive_id(&args.url, 0)));
    emit_rows(&[row], args.out_ledger.as_deref())?;
    Ok(())
}

fn run_recover_batch(stack: &DownloadStack, args: RecoverBatchArgs) -> Result<()> {
    let mut store = ProvenanceStore::open(&args.db)?;
    let rows = load_batch_rows(&args.input)?;
    let campaign_id = create_batch_campaign(
        &mut store,
        args.campaign_name.as_deref(),
        "recover_batch",
        &args.input,
        Some(&args.out_ledger),
        Some(&args.dest_dir),
        None,
    )?;
    let mut outputs = Vec::with_capacity(rows.len());
    for (index, row) in rows.into_iter().enumerate() {
        let destination = match row.output_rel {
            Some(rel) if !rel.trim().is_empty() => args.dest_dir.join(rel),
            _ => args
                .dest_dir
                .join(default_recovery_filename(&row.url, index + 1)?),
        };
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut request = TransferRequest::download(row.url.clone(), destination);
        request.backend = args.backend.into();
        request.probe_bytes = args.probe_bytes;
        request.note = row.note.clone();
        let trace = stack.recover_with_trace(&request);
        let job_id = record_trace(&mut store, &request, &trace)?;
        if let Some(campaign_id) = campaign_id {
            store.link_download_job_to_campaign(campaign_id, job_id)?;
        }
        let result = trace
            .clone()
            .into_result(&request.url)
            .with_context(|| format!("recovery failed for {}", row.url))?;
        outputs
            .push(result.to_ledger_row(row.id.unwrap_or_else(|| derive_id(&row.url, index + 1))));
    }
    emit_rows(&outputs, Some(&args.out_ledger))?;
    Ok(())
}

fn run_export_ledger(args: ExportLedgerArgs) -> Result<()> {
    let store = ProvenanceStore::open(&args.db)?;
    let projected = store.project_download_history_rows(
        args.limit,
        args.needle.as_deref(),
        args.host.as_deref(),
        args.status.as_deref(),
        args.backend.as_deref(),
    )?;
    let rows = projected
        .into_iter()
        .map(|row| DownloadLedgerRow {
            id: row.id,
            url: row.url,
            http_code: row.http_code,
            content_type: row.content_type,
            bytes: row.bytes,
            sha256: row.sha256,
            is_pdf: row.is_pdf,
            note: format!("status={}; {}", row.status, row.note),
        })
        .collect::<Vec<_>>();
    emit_rows(&rows, Some(&args.out_ledger))?;
    Ok(())
}

fn create_batch_campaign(
    store: &mut ProvenanceStore,
    campaign_name: Option<&str>,
    command_kind: &str,
    input_path: &Path,
    out_ledger_path: Option<&Path>,
    dest_dir: Option<&Path>,
    note: Option<&str>,
) -> Result<Option<i64>> {
    let default_name = input_path
        .file_stem()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .unwrap_or_else(|| command_kind.to_string());
    let campaign = DownloadCampaignRecord {
        id: None,
        name: campaign_name.unwrap_or(&default_name).to_string(),
        command_kind: command_kind.to_string(),
        input_path: input_path.to_string_lossy().to_string(),
        out_ledger_path: out_ledger_path.map(|path| path.to_string_lossy().to_string()),
        dest_dir: dest_dir.map(|path| path.to_string_lossy().to_string()),
        note: note.map(str::to_string),
        created_at: Utc::now().to_rfc3339(),
    };
    store.create_download_campaign(&campaign).map(Some)
}

fn record_trace(
    store: &mut ProvenanceStore,
    request: &TransferRequest,
    trace: &TransferTrace,
) -> Result<i64> {
    let timestamp = Utc::now().to_rfc3339();
    let job = DownloadJobRecord {
        id: None,
        requested_url: request.url.clone(),
        transfer_kind: match trace.route.kind {
            data_core::download_stack::TransferKind::Probe => "probe".to_string(),
            data_core::download_stack::TransferKind::Download => "download".to_string(),
        },
        requested_backend: request.backend.as_str().to_string(),
        route_scheme: trace.route.scheme.clone(),
        route_host: trace.route.host.clone(),
        route_backends: trace
            .route
            .backends
            .iter()
            .map(|backend| backend.as_str().to_string())
            .collect(),
        note: request.note.clone(),
        status: if trace.terminal_result.is_some() {
            "succeeded".to_string()
        } else {
            "failed".to_string()
        },
        final_url: trace
            .terminal_result
            .as_ref()
            .and_then(|result| result.final_url.clone()),
        output_path: trace
            .terminal_result
            .as_ref()
            .and_then(|result| result.output_path.as_ref())
            .map(|path| path.to_string_lossy().to_string()),
        created_at: timestamp.clone(),
    };
    let attempts = trace
        .attempts
        .iter()
        .map(|attempt| DownloadAttemptRecord {
            id: None,
            job_id: None,
            backend: attempt.backend.as_str().to_string(),
            succeeded: attempt.succeeded,
            failure_class: attempt.failure_class.clone(),
            http_code: attempt.http_code.map(i64::from),
            content_type: attempt.content_type.clone(),
            bytes: i64::try_from(attempt.bytes).unwrap_or(i64::MAX),
            sha256: attempt.sha256.clone(),
            is_pdf: attempt.is_pdf,
            final_url: attempt.final_url.clone(),
            note: attempt.note.clone(),
            error_message: attempt.error_message.clone(),
            recorded_at: timestamp.clone(),
        })
        .collect::<Vec<_>>();
    store.record_download_trace(&job, &attempts)
}

fn trace_to_ledger_row(
    request: &TransferRequest,
    trace: &TransferTrace,
    id: String,
) -> DownloadLedgerRow {
    if let Some(result) = &trace.terminal_result {
        return result.to_ledger_row(id);
    }
    let attempt = trace.attempts.last();
    let failure_suffix = attempt
        .and_then(|attempt| attempt.failure_class.as_deref())
        .map(|value| format!("failure_class={value}"))
        .unwrap_or_else(|| "failure_class=unknown".to_string());
    let error_suffix = attempt
        .and_then(|attempt| attempt.error_message.as_deref())
        .map(|value| format!("error={value}"))
        .unwrap_or_else(|| {
            trace
                .final_error
                .as_deref()
                .map(|value| format!("error={value}"))
                .unwrap_or_else(|| "error=unknown".to_string())
        });
    let note = match attempt {
        Some(attempt) => format!("{}; {failure_suffix}; {error_suffix}", attempt.note),
        None => format!("standardized probe exhausted; {failure_suffix}; {error_suffix}"),
    };
    DownloadLedgerRow {
        id,
        url: attempt
            .and_then(|attempt| attempt.final_url.clone())
            .unwrap_or_else(|| request.url.clone()),
        http_code: attempt
            .and_then(|attempt| attempt.http_code)
            .map(|value| value.to_string())
            .unwrap_or_default(),
        content_type: attempt
            .and_then(|attempt| attempt.content_type.clone())
            .unwrap_or_default(),
        bytes: attempt.map(|attempt| attempt.bytes).unwrap_or(0),
        sha256: attempt
            .and_then(|attempt| attempt.sha256.clone())
            .unwrap_or_default(),
        is_pdf: if attempt.map(|attempt| attempt.is_pdf).unwrap_or(false) {
            "yes"
        } else {
            "no"
        }
        .to_string(),
        note,
    }
}

fn load_manual_override_candidates(path: &Path, filters: &[String]) -> Result<Vec<BatchRow>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read manual override registry {}", path.display()))?;
    let registry: MirrorOverridesRegistry = toml::from_str(&text).with_context(|| {
        format!(
            "failed to parse manual override registry {}",
            path.display()
        )
    })?;
    let filter_set = filters
        .iter()
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .map(str::to_string)
        .collect::<BTreeSet<_>>();
    let mut seen = BTreeSet::new();
    let mut rows = Vec::new();
    for entry in registry.entries {
        if !filter_set.is_empty() && !filter_set.contains(entry.id.as_str()) {
            continue;
        }
        for url in entry.urls {
            let normalized = normalize_candidate_url(&url);
            if normalized.is_empty() || !is_direct_pdf_candidate(&normalized) {
                continue;
            }
            if !seen.insert(normalized.clone()) {
                continue;
            }
            let note = match (&entry.title, &entry.notes) {
                (Some(title), Some(notes))
                    if !title.trim().is_empty() && !notes.trim().is_empty() =>
                {
                    Some(format!(
                        "manual override {}: {}; {}",
                        entry.id,
                        title.trim(),
                        notes.trim()
                    ))
                }
                (Some(title), _) if !title.trim().is_empty() => {
                    Some(format!("manual override {}: {}", entry.id, title.trim()))
                }
                (_, Some(notes)) if !notes.trim().is_empty() => {
                    Some(format!("manual override {}: {}", entry.id, notes.trim()))
                }
                _ => Some(format!("manual override {}", entry.id)),
            };
            rows.push(BatchRow {
                url: normalized.clone(),
                id: Some(format!(
                    "{}_{}",
                    entry.id,
                    derive_id(&normalized, rows.len() + 1)
                )),
                note,
                output_rel: None,
            });
        }
    }
    if rows.is_empty() {
        bail!(
            "manual override registry {} did not yield any direct PDF candidates",
            path.display()
        );
    }
    Ok(rows)
}

fn normalize_candidate_url(url: &str) -> String {
    url.trim()
        .trim_matches('`')
        .trim_matches('"')
        .trim_matches('\'')
        .trim_end_matches(['.', ',', ';', ':'])
        .to_string()
}

fn is_direct_pdf_candidate(url: &str) -> bool {
    let lower = url.to_ascii_lowercase();
    if !(lower.starts_with("http://") || lower.starts_with("https://")) {
        return false;
    }
    lower.ends_with(".pdf")
        || lower.ends_with("/pdf")
        || lower.ends_with("/pdf/")
        || lower.contains("/pdf?")
        || lower.contains("/pdf/")
        || lower.contains("/doi/pdf/")
        || lower.contains("/doi/epdf/")
        || lower.contains("tandfonline.com/doi/pdf/")
        || lower.contains("tandfonline.com/doi/epdf/")
        || (lower.contains("journals.aps.org/") && lower.contains("/pdf/"))
        || (lower.contains("royalsocietypublishing.org") && lower.contains("/doi/pdf/"))
        || lower.contains("arxiv.org/pdf/")
        || (lower.contains("/bitstreams/") && lower.ends_with("/download"))
        || lower.ends_with("/download")
        || lower.contains("/download?")
        || lower.contains("openreview.net/pdf")
}

fn load_batch_rows(path: &Path) -> Result<Vec<BatchRow>> {
    let delimiter = if path.extension().and_then(|ext| ext.to_str()) == Some("csv") {
        b','
    } else {
        b'\t'
    };
    let mut reader = ReaderBuilder::new()
        .delimiter(delimiter)
        .flexible(true)
        .from_path(path)
        .with_context(|| format!("failed to open batch input {}", path.display()))?;
    let mut rows = Vec::new();
    for record in reader.deserialize() {
        let row: BatchRow =
            record.with_context(|| format!("failed to parse batch row from {}", path.display()))?;
        rows.push(row);
    }
    if rows.is_empty() {
        bail!("batch input {} did not contain any rows", path.display());
    }
    Ok(rows)
}

fn emit_rows(rows: &[DownloadLedgerRow], out_path: Option<&Path>) -> Result<()> {
    if let Some(path) = out_path {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut text = String::new();
        text.push_str(DownloadLedgerRow::header());
        text.push('\n');
        for row in rows {
            text.push_str(&row.to_tsv_line());
            text.push('\n');
        }
        fs::write(path, text)
            .with_context(|| format!("failed to write ledger {}", path.display()))?;
    }
    for row in rows {
        println!("{}", row.to_tsv_line());
    }
    Ok(())
}

fn derive_id(url: &str, ordinal: usize) -> String {
    let mut seed = url
        .replace("https://", "")
        .replace("http://", "")
        .replace("ftp://", "")
        .replace("ftps://", "")
        .replace("sftp://", "")
        .chars()
        .map(|ch| match ch {
            'a'..='z' | 'A'..='Z' | '0'..='9' => ch.to_ascii_lowercase(),
            _ => '_',
        })
        .collect::<String>();
    while seed.contains("__") {
        seed = seed.replace("__", "_");
    }
    seed = seed.trim_matches('_').to_string();
    if seed.is_empty() {
        format!("download_{ordinal:04}")
    } else {
        seed
    }
}

fn default_recovery_filename(url: &str, ordinal: usize) -> Result<String> {
    let parsed = url::Url::parse(url).with_context(|| format!("invalid URL {url}"))?;
    if let Some(name) = parsed
        .path_segments()
        .and_then(|mut segments| segments.rfind(|segment| !segment.is_empty()))
    {
        let sanitized = name.replace('/', "_");
        if !sanitized.is_empty() {
            return Ok(sanitized);
        }
    }
    Ok(format!("download_{ordinal:04}.bin"))
}
