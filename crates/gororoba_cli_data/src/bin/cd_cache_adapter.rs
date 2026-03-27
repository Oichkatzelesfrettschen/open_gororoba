use anyhow::{Context, Result, bail};
use chrono::Utc;
use clap::{Args, Parser, Subcommand, ValueEnum};
use gororoba_cli_data::{
    acquisition_dossier::{
        DossierBatchEntry, DossierBatchManifest, DossierHit, ResearchDossier, StageSuggestion,
        load_dossier_batch_manifest, write_dossier_batch_manifest, write_research_dossier,
    },
    project_api_contract::{
        ProjectApiContext, ProjectApiCrosswalkBinding, ProjectApiCrosswalkFile,
        load_project_api_context, load_project_api_crosswalk,
    },
};
use lit_search::{
    MultiQueryExecutionOutcome, Paper, SearchEngine, canonicalize_doi, normalize_source_name,
    search::{
        SOURCE_FAMILY_NAMES, SourceTier, normalize_source_family_name, source_names_for_family,
    },
    sources::ApiKeys,
};
use serde::Deserialize;
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    process::Command,
};
use url::Url;

#[derive(Parser, Debug)]
#[command(
    name = "cd-cache-adapter",
    about = "Project-specific CayleyDickson cache adapter that emits actionable search, promote, and reconcile steps without mutating the cache directly"
)]
struct Cli {
    #[arg(long)]
    project_api_root: PathBuf,

    #[arg(long, default_value = ".")]
    gororoba_repo_root: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Status(StatusArgs),
    EmitActions(EmitActionsArgs),
    ResearchDossier(ResearchDossierArgs),
    StageFromDossier(StageFromDossierArgs),
}

#[derive(Args, Debug, Clone, Default)]
struct QueueFilters {
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
}

#[derive(Args, Debug)]
struct StatusArgs {
    #[command(flatten)]
    filters: QueueFilters,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum ActionFormat {
    Tsv,
    Markdown,
}

#[derive(Args, Debug)]
struct EmitActionsArgs {
    #[command(flatten)]
    filters: QueueFilters,

    #[arg(long, value_enum, default_value_t = ActionFormat::Tsv)]
    format: ActionFormat,

    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct ResearchDossierArgs {
    #[command(flatten)]
    filters: QueueFilters,

    #[arg(long, default_value_t = 5)]
    limit_per_query: usize,

    #[arg(long, default_value_t = 0)]
    year_min: u32,

    #[arg(long, default_value = "open")]
    tier: String,

    #[arg(long = "source")]
    source: Vec<String>,

    #[arg(long = "source-family")]
    source_family: Vec<String>,

    #[arg(long = "preferred-source-family")]
    preferred_source_family: Vec<String>,

    #[arg(long, default_value = "docs/reports/cd_cache_dossiers")]
    output_dir: PathBuf,

    #[arg(long, default_value_t = 10)]
    max_hits: usize,

    #[arg(long)]
    min_relevance: Option<i64>,

    #[arg(long)]
    batch_manifest_out: Option<PathBuf>,
}

#[derive(Args, Debug)]
struct StageFromDossierArgs {
    #[arg(long)]
    dossier: Vec<PathBuf>,

    #[arg(long)]
    batch_manifest: Vec<PathBuf>,

    #[arg(long)]
    sessions_dir: Option<PathBuf>,

    #[arg(long)]
    max_rank: Option<usize>,

    #[arg(long = "rank")]
    rank: Vec<usize>,

    #[arg(long, default_value_t = false)]
    all_suggestions: bool,

    #[arg(long = "note")]
    note: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct SearchQueueFile {
    #[serde(default)]
    project_id: Option<String>,
    #[serde(default)]
    summary: Option<SearchQueueSummary>,
    #[serde(default)]
    search_target: Vec<SearchQueueTarget>,
}

#[derive(Debug, Deserialize)]
struct SearchQueueSummary {
    #[serde(default)]
    critical_retrieval_blockers: Option<usize>,
    #[serde(default)]
    century_normalization_tracks: Option<usize>,
    #[serde(default)]
    terminology_audit_tracks: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
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
}

#[derive(Debug)]
struct ActionRow {
    search_target_id: String,
    title: String,
    crosswalk_id: String,
    window: String,
    priority: String,
    status: String,
    kind: String,
    primary_action: String,
    search_command: String,
    promote_command: String,
    reconcile_command: String,
    preferred_lane_preview: String,
    query_seed_preview: String,
}

#[derive(Debug, Clone)]
struct DossierBuildConfig<'a> {
    project_id: &'a str,
    project_api_root: &'a Path,
    search_queue_path: &'a Path,
    limit_per_query: usize,
    year_min: u32,
    max_hits: usize,
    min_relevance: i64,
    preferred_source_families: Vec<String>,
}

#[derive(Debug)]
struct PendingDossier {
    json_path: PathBuf,
    markdown_path: PathBuf,
    dossier: ResearchDossier,
}

#[derive(Debug, Clone)]
struct RankedPaper<'a> {
    paper: &'a Paper,
    canonical_id: String,
    relevance_score: i64,
    source_family: String,
    route_class: String,
    host_class: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let project_api = load_project_api_context(&cli.project_api_root)?;
    let crosswalk = load_project_api_crosswalk(&project_api.crosswalk_path)?;
    let queue = load_search_queue(&project_api.search_queue_path)?;
    let gororoba_repo_root = resolve_repo_root(&cli.gororoba_repo_root);

    match cli.command {
        Commands::Status(args) => run_status(&project_api, &crosswalk, &queue, &args.filters),
        Commands::EmitActions(args) => {
            run_emit_actions(&project_api, &crosswalk, &queue, &gororoba_repo_root, args)
        }
        Commands::ResearchDossier(args) => {
            run_research_dossier(&project_api, &crosswalk, &queue, &gororoba_repo_root, args).await
        }
        Commands::StageFromDossier(args) => run_stage_from_dossier(&gororoba_repo_root, args),
    }
}

fn run_status(
    project_api: &ProjectApiContext,
    crosswalk: &ProjectApiCrosswalkFile,
    queue: &SearchQueueFile,
    filters: &QueueFilters,
) -> Result<()> {
    let targets = filtered_targets(queue, filters);
    let mut priority_counts: BTreeMap<String, usize> = BTreeMap::new();
    let mut status_counts: BTreeMap<String, usize> = BTreeMap::new();
    for target in &targets {
        *priority_counts.entry(target.priority.clone()).or_default() += 1;
        *status_counts.entry(target.status.clone()).or_default() += 1;
    }

    println!("project_api_root={}", project_api.project_api_dir.display());
    println!("search_queue={}", project_api.search_queue_path.display());
    println!(
        "project_id={}",
        queue.project_id.as_deref().unwrap_or("cayley-dickson")
    );
    if let Some(summary) = &queue.summary {
        println!(
            "summary_critical_retrieval_blockers={}",
            summary.critical_retrieval_blockers.unwrap_or_default()
        );
        println!(
            "summary_century_normalization_tracks={}",
            summary.century_normalization_tracks.unwrap_or_default()
        );
        println!(
            "summary_terminology_audit_tracks={}",
            summary.terminology_audit_tracks.unwrap_or_default()
        );
    }
    println!("filtered_targets={}", targets.len());

    for (priority, count) in priority_counts {
        println!("priority_count[{priority}]={count}");
    }
    for (status, count) in status_counts {
        println!("status_count[{status}]={count}");
    }

    for target in targets {
        let binding = binding_for_target(crosswalk, target);
        println!(
            "target={} title={} kind={} crosswalk={} action={} lane={} why_now={}",
            target.id,
            target.title,
            target.kind,
            binding
                .map(|binding| binding.id.as_str())
                .unwrap_or("unbound"),
            primary_action(target),
            target
                .preferred_lanes
                .first()
                .map(String::as_str)
                .unwrap_or(""),
            target.why_now
        );
    }

    Ok(())
}

fn run_emit_actions(
    project_api: &ProjectApiContext,
    crosswalk: &ProjectApiCrosswalkFile,
    queue: &SearchQueueFile,
    gororoba_repo_root: &Path,
    args: EmitActionsArgs,
) -> Result<()> {
    let mut rows = Vec::new();
    let reconcile_command = format!(
        "cargo run -q -p gororoba_cli_data --bin cd-cache-reconcile -- --project-api-root \"{}\"",
        project_api.repo_root.display()
    );

    for target in filtered_targets(queue, &args.filters) {
        let binding = binding_for_target(crosswalk, target);
        let project_id = queue.project_id.as_deref().unwrap_or("cayley-dickson");
        rows.push(ActionRow {
            search_target_id: target.id.clone(),
            title: target.title.clone(),
            crosswalk_id: binding
                .map(|binding| binding.id.clone())
                .unwrap_or_else(|| "unbound".to_string()),
            window: target.window.clone(),
            priority: target.priority.clone(),
            status: target.status.clone(),
            kind: target.kind.clone(),
            primary_action: primary_action(target).to_string(),
            search_command: format!(
                "cargo run -q -p gororoba_cli_data --bin human-acquire -- --repo-root \"{}\" stage-queue --queue \"{}\" --id \"{}\"",
                gororoba_repo_root.display(),
                project_api.search_queue_path.display(),
                target.id
            ),
            promote_command: promote_command(gororoba_repo_root, project_id, target),
            reconcile_command: reconcile_command.clone(),
            preferred_lane_preview: target
                .preferred_lanes
                .first()
                .cloned()
                .unwrap_or_default(),
            query_seed_preview: target.query_seeds.first().cloned().unwrap_or_default(),
        });
    }

    let rendered = match args.format {
        ActionFormat::Tsv => render_tsv(&rows),
        ActionFormat::Markdown => render_markdown(&rows),
    };

    if let Some(output) = args.output {
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
        fs::write(&output, rendered).with_context(|| format!("write {}", output.display()))?;
        println!("wrote={}", output.display());
    } else {
        print!("{rendered}");
    }
    Ok(())
}

async fn run_research_dossier(
    project_api: &ProjectApiContext,
    crosswalk: &ProjectApiCrosswalkFile,
    queue: &SearchQueueFile,
    gororoba_repo_root: &Path,
    args: ResearchDossierArgs,
) -> Result<()> {
    let output_dir = resolve_output_dir(gororoba_repo_root, &args.output_dir);
    fs::create_dir_all(&output_dir).with_context(|| format!("create {}", output_dir.display()))?;

    let tier = parse_source_tier(&args.tier);
    let sources = resolve_requested_sources(&args.source, &args.source_family)?;
    let engine = SearchEngine::new(ApiKeys::from_env(), tier);
    let project_id = queue.project_id.as_deref().unwrap_or("cayley-dickson");
    let targets = filtered_targets(queue, &args.filters);
    let mut pending = Vec::new();

    for target in targets {
        let binding = binding_for_target(crosswalk, target);
        let queries = deduplicate_query_seeds(target);
        let preferred_source_families = if args.preferred_source_family.is_empty() {
            default_preferred_source_families(target)
        } else {
            normalize_source_family_list(&args.preferred_source_family)
        };
        let min_relevance = args
            .min_relevance
            .unwrap_or_else(|| default_min_relevance(target));
        let outcome = engine
            .search_queries_parallel_with_sources(
                &queries,
                args.limit_per_query,
                args.year_min,
                &sources,
            )
            .await;
        let dossier = build_dossier(
            DossierBuildConfig {
                project_id,
                project_api_root: &project_api.repo_root,
                search_queue_path: &project_api.search_queue_path,
                limit_per_query: args.limit_per_query,
                year_min: args.year_min,
                max_hits: args.max_hits,
                min_relevance,
                preferred_source_families,
            },
            target,
            binding,
            &queries,
            outcome,
        );
        let stem = format!(
            "{}_{}",
            sanitize_token(&dossier.search_target_id),
            Utc::now().format("%Y%m%d")
        );
        pending.push(PendingDossier {
            json_path: output_dir.join(format!("{stem}.json")),
            markdown_path: output_dir.join(format!("{stem}.md")),
            dossier,
        });
    }

    let batch_manifest_path = args
        .batch_manifest_out
        .as_ref()
        .map(|path| resolve_output_dir(gororoba_repo_root, path))
        .unwrap_or_else(|| default_batch_manifest_path(&output_dir, &args.filters));
    let batch_manifest_rel = repo_display_path(gororoba_repo_root, &batch_manifest_path);

    for pending_dossier in &mut pending {
        pending_dossier.dossier.batch_manifest_rel = Some(batch_manifest_rel.clone());
        hydrate_stage_suggestion_commands(
            &mut pending_dossier.dossier,
            gororoba_repo_root,
            project_api,
            &pending_dossier.json_path,
        );
        write_research_dossier(&pending_dossier.json_path, &pending_dossier.dossier)?;
        fs::write(
            &pending_dossier.markdown_path,
            render_research_markdown(&pending_dossier.dossier),
        )
        .with_context(|| format!("write {}", pending_dossier.markdown_path.display()))?;
        println!(
            "wrote_json={} wrote_markdown={} search_target={} hits={} suggestions={}",
            pending_dossier.json_path.display(),
            pending_dossier.markdown_path.display(),
            pending_dossier.dossier.search_target_id,
            pending_dossier.dossier.top_hits.len(),
            pending_dossier.dossier.stage_suggestions.len()
        );
    }

    let batch_manifest = DossierBatchManifest {
        schema_version: 1,
        generated_at_utc: Utc::now().to_rfc3339(),
        project_id: project_id.to_string(),
        project_api_root: project_api.repo_root.display().to_string(),
        search_queue_path: project_api.search_queue_path.display().to_string(),
        output_dir: repo_display_path(gororoba_repo_root, &output_dir),
        requested_sources: sources,
        preferred_source_families: pending
            .first()
            .map(|item| item.dossier.preferred_source_families.clone())
            .unwrap_or_default(),
        year_min: args.year_min,
        limit_per_query: args.limit_per_query,
        min_relevance: pending
            .first()
            .map(|item| item.dossier.min_relevance)
            .unwrap_or_default(),
        windows: args.filters.window.clone(),
        priorities: args.filters.priority.clone(),
        statuses: args.filters.status.clone(),
        critical_only: args.filters.critical_only,
        entries: pending
            .iter()
            .map(|item| DossierBatchEntry {
                search_target_id: item.dossier.search_target_id.clone(),
                title: item.dossier.title.clone(),
                window: item.dossier.window.clone(),
                priority: item.dossier.priority.clone(),
                kind: item.dossier.kind.clone(),
                dossier_json: repo_display_path(gororoba_repo_root, &item.json_path),
                dossier_markdown: repo_display_path(gororoba_repo_root, &item.markdown_path),
                suggestion_count: item.dossier.stage_suggestions.len(),
            })
            .collect(),
    };
    write_dossier_batch_manifest(&batch_manifest_path, &batch_manifest)?;
    println!("batch_manifest={}", batch_manifest_path.display());

    Ok(())
}

fn run_stage_from_dossier(gororoba_repo_root: &Path, args: StageFromDossierArgs) -> Result<()> {
    let dossier_paths = collect_dossier_paths(gororoba_repo_root, &args)?;
    if dossier_paths.is_empty() {
        bail!("stage-from-dossier requires at least one --dossier or --batch-manifest");
    }

    let mut staged = 0_usize;
    for dossier_path in dossier_paths {
        let mut command = Command::new("cargo");
        command.current_dir(gororoba_repo_root);
        command.args([
            "run",
            "-q",
            "-p",
            "gororoba_cli_data",
            "--bin",
            "human-acquire",
            "--",
            "--repo-root",
            gororoba_repo_root.to_string_lossy().as_ref(),
        ]);
        if let Some(sessions_dir) = &args.sessions_dir {
            command.args(["--sessions-dir", sessions_dir.to_string_lossy().as_ref()]);
        }
        command.args([
            "import-dossier",
            "--dossier",
            dossier_path.to_string_lossy().as_ref(),
        ]);
        if args.all_suggestions {
            command.arg("--all-suggestions");
        }
        if let Some(max_rank) = args.max_rank {
            command.args(["--max-rank", &max_rank.to_string()]);
        }
        for rank in &args.rank {
            command.args(["--rank", &rank.to_string()]);
        }
        for note in &args.note {
            command.args(["--note", note]);
        }
        let output = command.output().with_context(|| {
            format!(
                "run human-acquire import-dossier for {}",
                dossier_path.display()
            )
        })?;
        if !output.status.success() {
            bail!(
                "human-acquire import-dossier failed for {}\nstdout:\n{}\nstderr:\n{}",
                dossier_path.display(),
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
        staged += 1;
    }
    println!("staged_dossiers={staged}");
    Ok(())
}

fn build_dossier(
    config: DossierBuildConfig<'_>,
    target: &SearchQueueTarget,
    binding: Option<&ProjectApiCrosswalkBinding>,
    queries: &[String],
    outcome: MultiQueryExecutionOutcome,
) -> ResearchDossier {
    let ranked_candidates = collapse_ranked_candidates(rank_papers_for_target(
        target,
        &outcome.papers,
        &config.preferred_source_families,
    ));
    let top_hits = ranked_candidates
        .iter()
        .take(config.max_hits)
        .enumerate()
        .map(|(index, ranked)| DossierHit {
            rank: index + 1,
            canonical_id: ranked.canonical_id.clone(),
            relevance_score: ranked.relevance_score,
            source_family: ranked.source_family.clone(),
            route_class: ranked.route_class.clone(),
            host_class: ranked.host_class.clone(),
            title: ranked.paper.title.clone(),
            year: ranked.paper.year,
            source: ranked.paper.source.clone(),
            venue: ranked.paper.venue.clone(),
            citation_count: ranked.paper.citation_count,
            doi: ranked.paper.doi.clone(),
            url: ranked.paper.url.clone(),
            pdf_url: ranked.paper.pdf_url.clone(),
        })
        .collect::<Vec<_>>();

    let preferred_set = config
        .preferred_source_families
        .iter()
        .cloned()
        .collect::<HashSet<_>>();
    let preferred_hits_available = ranked_candidates
        .iter()
        .any(|ranked| preferred_set.contains(&ranked.source_family));
    let stage_suggestions = ranked_candidates
        .iter()
        .take(config.max_hits)
        .enumerate()
        .filter_map(|(index, ranked)| {
            build_stage_suggestion(
                target,
                ranked,
                index + 1,
                config.min_relevance,
                preferred_hits_available,
                &preferred_set,
            )
        })
        .collect::<Vec<_>>();

    ResearchDossier {
        schema_version: 1,
        generated_at_utc: Utc::now().to_rfc3339(),
        project_id: config.project_id.to_string(),
        project_api_root: config.project_api_root.display().to_string(),
        search_queue_path: config.search_queue_path.display().to_string(),
        search_target_id: target.id.clone(),
        crosswalk_id: binding
            .map(|binding| binding.id.clone())
            .unwrap_or_else(|| "unbound".to_string()),
        title: target.title.clone(),
        window: target.window.clone(),
        priority: target.priority.clone(),
        status: target.status.clone(),
        kind: target.kind.clone(),
        why_now: target.why_now.clone(),
        query_seeds: queries.to_vec(),
        requested_sources: outcome.report.requested_sources.clone(),
        preferred_source_families: config.preferred_source_families,
        limit_per_query: config.limit_per_query,
        year_min: config.year_min,
        min_relevance: config.min_relevance,
        report: outcome.report,
        top_hits,
        stage_suggestions,
        batch_manifest_rel: None,
    }
}

fn build_stage_suggestion(
    target: &SearchQueueTarget,
    ranked: &RankedPaper<'_>,
    rank: usize,
    min_relevance: i64,
    preferred_hits_available: bool,
    preferred_source_families: &HashSet<String>,
) -> Option<StageSuggestion> {
    let candidate_url = preferred_candidate_url(ranked.paper);
    if candidate_url.is_empty()
        || ranked.relevance_score < min_relevance
        || !candidate_passes_stage_filters(target, ranked.paper)
    {
        return None;
    }
    if preferred_hits_available && !preferred_source_families.contains(&ranked.source_family) {
        return None;
    }

    let action = match ranked.route_class.as_str() {
        "direct_pdf" => "stage_direct_pdf_child",
        "doi_resolver" => "stage_doi_resolver_child",
        "openalex_landing" | "crossref_landing" | "metadata_landing" => {
            "stage_metadata_landing_child"
        }
        _ => "stage_browser_landing_child",
    };

    Some(StageSuggestion {
        rank,
        suggestion_id: sanitize_token(&format!("{}_{}", target.id, ranked.canonical_id)),
        canonical_id: ranked.canonical_id.clone(),
        relevance_score: ranked.relevance_score,
        source: ranked.paper.source.clone(),
        source_family: ranked.source_family.clone(),
        route_class: ranked.route_class.clone(),
        host_class: ranked.host_class.clone(),
        paper_title: ranked.paper.title.clone(),
        action: action.to_string(),
        candidate_url,
        command: String::new(),
        rationale: format!(
            "Candidate from {} [{}] with relevance score {} and {} citations for queue target {}.",
            ranked.paper.source,
            ranked.source_family,
            ranked.relevance_score,
            ranked.paper.citation_count,
            target.id
        ),
        default_selected: candidate_is_default_selected(target, ranked, rank),
    })
}

fn hydrate_stage_suggestion_commands(
    dossier: &mut ResearchDossier,
    gororoba_repo_root: &Path,
    project_api: &ProjectApiContext,
    dossier_json_path: &Path,
) {
    let dossier_path = repo_display_path(gororoba_repo_root, dossier_json_path);
    for suggestion in &mut dossier.stage_suggestions {
        suggestion.command = format!(
            "cargo run -q -p gororoba_cli_data --bin cd-cache-adapter -- --project-api-root '{}' --gororoba-repo-root '{}' stage-from-dossier --dossier '{}' --rank '{}'",
            project_api.repo_root.display(),
            gororoba_repo_root.display(),
            dossier_path,
            suggestion.rank
        );
    }
}

fn render_research_markdown(dossier: &ResearchDossier) -> String {
    let mut body = String::new();
    body.push_str(&format!("# {}\n\n", dossier.title));
    body.push_str(&format!(
        "- search_target_id: `{}`\n- crosswalk_id: `{}`\n- window: `{}`\n- priority: `{}`\n- status: `{}`\n- kind: `{}`\n- generated_at_utc: `{}`\n- min_relevance: `{}`\n- preferred_source_families: `{}`\n",
        dossier.search_target_id,
        dossier.crosswalk_id,
        dossier.window,
        dossier.priority,
        dossier.status,
        dossier.kind,
        dossier.generated_at_utc,
        dossier.min_relevance,
        dossier.preferred_source_families.join(", ")
    ));
    if let Some(batch_manifest_rel) = &dossier.batch_manifest_rel {
        body.push_str(&format!("- batch_manifest: `{}`\n", batch_manifest_rel));
    }
    body.push('\n');
    body.push_str(&format!("{}\n\n", dossier.why_now));
    body.push_str("## Query Seeds\n");
    for query in &dossier.query_seeds {
        body.push_str(&format!("- `{}`\n", query));
    }
    body.push('\n');
    body.push_str("## Search Summary\n");
    body.push_str(&format!(
        "- requested_sources: `{}`\n- limit_per_query: `{}`\n- year_min: `{}`\n- raw_results: `{}`\n- deduplicated_results: `{}`\n\n",
        dossier.requested_sources.join(", "),
        dossier.limit_per_query,
        dossier.year_min,
        dossier.report.raw_result_count,
        dossier.report.deduplicated_result_count,
    ));
    for query_report in &dossier.report.query_reports {
        body.push_str(&format!(
            "### Query `{}`\n- raw_results: `{}`\n- deduplicated_results: `{}`\n- cache_hits: `{}`\n",
            query_report.query,
            query_report.raw_result_count,
            query_report.deduplicated_result_count,
            query_report.cache_hit_count
        ));
        for source_report in &query_report.source_reports {
            let mut detail = format!(
                "- {} -> {} result(s)",
                source_report.source, source_report.result_count
            );
            if source_report.cache_hit {
                detail.push_str(" [cache_hit]");
            }
            if !source_report.skipped_reason.is_empty() {
                detail.push_str(&format!(" [skipped={}]", source_report.skipped_reason));
            }
            if !source_report.error.is_empty() {
                detail.push_str(&format!(" [error={}]", source_report.error));
            }
            body.push_str(&detail);
            body.push('\n');
        }
        body.push('\n');
    }

    body.push_str("## Top Hits\n");
    body.push_str("| rank | relevance | canonical_id | source_family | route_class | year | source | citations | title | url |\n");
    body.push_str("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for hit in &dossier.top_hits {
        body.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            hit.rank,
            hit.relevance_score,
            escape_markdown_cell(&hit.canonical_id),
            escape_markdown_cell(&hit.source_family),
            escape_markdown_cell(&hit.route_class),
            hit.year,
            escape_markdown_cell(&hit.source),
            hit.citation_count,
            escape_markdown_cell(&hit.title),
            escape_markdown_cell(&preferred_link(hit)),
        ));
    }
    body.push('\n');

    body.push_str("## Stage Suggestions\n");
    if dossier.stage_suggestions.is_empty() {
        body.push_str("- No stageable URLs cleared the dossier filters.\n");
    } else {
        for suggestion in &dossier.stage_suggestions {
            body.push_str(&format!(
                "- rank {}: `{}`\n  canonical_id: `{}`\n  source_family: `{}`\n  route_class: `{}`\n  default_selected: `{}`\n  url: `{}`\n  command: `{}`\n",
                suggestion.rank,
                suggestion.paper_title,
                suggestion.canonical_id,
                suggestion.source_family,
                suggestion.route_class,
                suggestion.default_selected,
                suggestion.candidate_url,
                suggestion.command,
            ));
        }
    }
    body
}

fn load_search_queue(path: &Path) -> Result<SearchQueueFile> {
    let raw = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parse {}", path.display()))
}

fn filtered_targets<'a>(
    queue: &'a SearchQueueFile,
    filters: &QueueFilters,
) -> Vec<&'a SearchQueueTarget> {
    queue
        .search_target
        .iter()
        .filter(|target| matches_filters(target, filters))
        .collect()
}

fn matches_filters(target: &SearchQueueTarget, filters: &QueueFilters) -> bool {
    if filters.critical_only && target.priority != "critical" {
        return false;
    }
    if !filters.id_filters.is_empty() && !filters.id_filters.iter().any(|value| value == &target.id)
    {
        return false;
    }
    if !filters.window.is_empty() && !filters.window.iter().any(|value| value == &target.window) {
        return false;
    }
    if !filters.priority.is_empty()
        && !filters
            .priority
            .iter()
            .any(|value| value == &target.priority)
    {
        return false;
    }
    if !filters.status.is_empty() && !filters.status.iter().any(|value| value == &target.status) {
        return false;
    }
    true
}

fn binding_for_target<'a>(
    crosswalk: &'a ProjectApiCrosswalkFile,
    target: &SearchQueueTarget,
) -> Option<&'a ProjectApiCrosswalkBinding> {
    crosswalk
        .binding
        .iter()
        .find(|binding| binding.search_target_id == target.id || binding.id == target.id)
}

fn primary_action(target: &SearchQueueTarget) -> &'static str {
    match target.kind.as_str() {
        "browser_confirmed_free_pdf_lane" => "stage_queue_then_browser_promote",
        "metadata_confirmed_pdf_path_resolution" => "stage_queue_then_pdf_path_resolution",
        "holder_workflow_followup" | "article_or_chapter_delivery_followup" => {
            "stage_queue_then_holder_followup"
        }
        "terminology_and_alias_audit" => "stage_queue_then_taxonomy_extract",
        "historical_density_and_normalization" | "row_level_primary_witness_audit" => {
            "stage_queue_then_century_normalization"
        }
        _ => "stage_queue_then_research_followup",
    }
}

fn promote_command(
    gororoba_repo_root: &Path,
    project_id: &str,
    target: &SearchQueueTarget,
) -> String {
    if !matches!(
        target.kind.as_str(),
        "browser_confirmed_free_pdf_lane" | "metadata_confirmed_pdf_path_resolution"
    ) {
        return String::new();
    }

    let session_id = sanitize_token(&format!("{project_id}_{}", target.id));
    let session_dir = gororoba_repo_root
        .join("reports/acquisition_sessions")
        .join(&session_id);
    format!(
        "cargo run -q -p gororoba_cli_data --bin human-acquire -- --repo-root \"{}\" promote --session \"{}\" --url \"<resolved_url>\" --handoff-out \"{}\"",
        gororoba_repo_root.display(),
        session_dir.join("session.toml").display(),
        session_dir.join("handoff.tsv").display()
    )
}

fn render_tsv(rows: &[ActionRow]) -> String {
    let mut body = String::from(
        "search_target_id\ttitle\tcrosswalk_id\twindow\tpriority\tstatus\tkind\tprimary_action\tsearch_command\tpromote_command\treconcile_command\tpreferred_lane_preview\tquery_seed_preview\n",
    );
    for row in rows {
        let line = [
            &row.search_target_id,
            &row.title,
            &row.crosswalk_id,
            &row.window,
            &row.priority,
            &row.status,
            &row.kind,
            &row.primary_action,
            &row.search_command,
            &row.promote_command,
            &row.reconcile_command,
            &row.preferred_lane_preview,
            &row.query_seed_preview,
        ]
        .iter()
        .map(|value| escape_tsv_field(value))
        .collect::<Vec<_>>()
        .join("\t");
        body.push_str(&line);
        body.push('\n');
    }
    body
}

fn render_markdown(rows: &[ActionRow]) -> String {
    let mut body = String::from(
        "| search_target_id | title | window | priority | status | primary_action | crosswalk_id | preferred_lane_preview | query_seed_preview |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n",
    );
    for row in rows {
        body.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} |\n",
            escape_markdown_cell(&row.search_target_id),
            escape_markdown_cell(&row.title),
            escape_markdown_cell(&row.window),
            escape_markdown_cell(&row.priority),
            escape_markdown_cell(&row.status),
            escape_markdown_cell(&row.primary_action),
            escape_markdown_cell(&row.crosswalk_id),
            escape_markdown_cell(&row.preferred_lane_preview),
            escape_markdown_cell(&row.query_seed_preview),
        ));
    }
    body
}

fn deduplicate_query_seeds(target: &SearchQueueTarget) -> Vec<String> {
    let mut queries = Vec::new();
    for query in &target.query_seeds {
        if !queries
            .iter()
            .any(|existing: &String| existing.eq_ignore_ascii_case(query))
        {
            queries.push(query.clone());
        }
    }
    if queries.is_empty() {
        queries.push(target.title.clone());
    }
    queries
}

fn rank_papers_for_target<'a>(
    target: &SearchQueueTarget,
    papers: &'a [Paper],
    preferred_source_families: &[String],
) -> Vec<RankedPaper<'a>> {
    let expected_year = expected_year_from_target(target);
    let normalized_target_title = normalize_text_for_match(&target.title);
    let cleaned_queries = deduplicate_query_seeds(target)
        .into_iter()
        .map(|query| normalize_text_for_match(&query))
        .collect::<Vec<_>>();
    let target_tokens = collect_match_tokens(&target.title);
    let query_tokens = target
        .query_seeds
        .iter()
        .flat_map(|query| collect_match_tokens(query))
        .collect::<Vec<_>>();
    let preferred_set = preferred_source_families.iter().collect::<HashSet<_>>();

    let mut ranked = papers
        .iter()
        .map(|paper| {
            let normalized_paper_title = normalize_text_for_match(&paper.title);
            let mut score = 0_i64;
            if !normalized_target_title.is_empty()
                && normalized_paper_title.contains(&normalized_target_title)
            {
                score += 120;
            }
            for query in &cleaned_queries {
                if !query.is_empty() && normalized_paper_title.contains(query) {
                    score += 90;
                }
            }
            for token in &target_tokens {
                if normalized_paper_title.contains(token) {
                    score += 12;
                }
            }
            for token in &query_tokens {
                if normalized_paper_title.contains(token) {
                    score += 6;
                }
            }
            let source_family = map_source_family(&paper.source);
            if preferred_set.contains(&source_family) {
                score += 24;
            }
            let route_class = classify_route_class(paper);
            if route_class == "direct_pdf" {
                score += 20;
            } else if route_class == "doi_resolver" {
                score += 8;
            }
            if let Some(expected_year) = expected_year
                && paper.year > 0
            {
                let delta = paper.year.abs_diff(expected_year);
                if delta == 0 {
                    score += 40;
                } else if delta <= 1 {
                    score += 24;
                } else if delta <= 5 {
                    score += 12;
                } else if delta >= 25 {
                    score -= 120;
                } else if delta >= 10 {
                    score -= 50;
                }
            }
            if normalized_paper_title.contains("cayley")
                || normalized_paper_title.contains("dickson")
            {
                score += 4;
            }
            RankedPaper {
                paper,
                canonical_id: canonical_candidate_id(paper),
                relevance_score: score,
                source_family,
                route_class,
                host_class: classify_host_class(&preferred_candidate_url(paper)),
            }
        })
        .collect::<Vec<_>>();

    ranked.sort_by(|left, right| {
        right
            .relevance_score
            .cmp(&left.relevance_score)
            .then_with(|| right.paper.citation_count.cmp(&left.paper.citation_count))
            .then_with(|| right.paper.year.cmp(&left.paper.year))
    });
    ranked
}

fn collapse_ranked_candidates<'a>(ranked: Vec<RankedPaper<'a>>) -> Vec<RankedPaper<'a>> {
    let mut best_by_id: HashMap<String, RankedPaper<'a>> = HashMap::new();
    for candidate in ranked {
        match best_by_id.get(&candidate.canonical_id) {
            Some(existing)
                if existing.relevance_score > candidate.relevance_score
                    || (existing.relevance_score == candidate.relevance_score
                        && existing.paper.citation_count >= candidate.paper.citation_count) => {}
            _ => {
                best_by_id.insert(candidate.canonical_id.clone(), candidate);
            }
        }
    }
    let mut collapsed = best_by_id.into_values().collect::<Vec<_>>();
    collapsed.sort_by(|left, right| {
        right
            .relevance_score
            .cmp(&left.relevance_score)
            .then_with(|| right.paper.citation_count.cmp(&left.paper.citation_count))
            .then_with(|| right.paper.year.cmp(&left.paper.year))
    });
    collapsed
}

fn normalize_text_for_match(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch.is_ascii_whitespace() {
                ch.to_ascii_lowercase()
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn collect_match_tokens(value: &str) -> Vec<String> {
    normalize_text_for_match(value)
        .split_whitespace()
        .filter(|token| token.len() >= 4)
        .map(ToOwned::to_owned)
        .collect()
}

fn parse_source_tier(value: &str) -> SourceTier {
    match value.to_ascii_lowercase().as_str() {
        "core" => SourceTier::Core,
        "all" => SourceTier::All,
        _ => SourceTier::Open,
    }
}

fn resolve_repo_root(input: &Path) -> PathBuf {
    if input.is_absolute() {
        input.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(input)
    }
}

fn resolve_output_dir(gororoba_repo_root: &Path, output_dir: &Path) -> PathBuf {
    if output_dir.is_absolute() {
        output_dir.to_path_buf()
    } else {
        gororoba_repo_root.join(output_dir)
    }
}

fn repo_display_path(repo_root: &Path, path: &Path) -> String {
    path.strip_prefix(repo_root)
        .unwrap_or(path)
        .display()
        .to_string()
}

fn resolve_requested_sources(
    explicit_sources: &[String],
    source_families: &[String],
) -> Result<Vec<String>> {
    let mut sources = Vec::new();
    for source in explicit_sources {
        let normalized = normalize_source_name(source);
        if !sources
            .iter()
            .any(|existing: &String| existing == &normalized)
        {
            sources.push(normalized);
        }
    }
    for family in normalize_source_family_list(source_families) {
        let Some(family_sources) = source_names_for_family(&family) else {
            bail!(
                "unknown source family '{}'; valid families: {}",
                family,
                SOURCE_FAMILY_NAMES.join(", ")
            );
        };
        for source in family_sources {
            let normalized = normalize_source_name(source);
            if !sources.iter().any(|existing| existing == &normalized) {
                sources.push(normalized);
            }
        }
    }
    Ok(sources)
}

fn normalize_source_family_list(families: &[String]) -> Vec<String> {
    let mut normalized = Vec::new();
    for family in families {
        let normalized_family = normalize_source_family_name(family);
        if !normalized
            .iter()
            .any(|existing: &String| existing == &normalized_family)
        {
            normalized.push(normalized_family);
        }
    }
    normalized
}

fn default_preferred_source_families(target: &SearchQueueTarget) -> Vec<String> {
    match target.kind.as_str() {
        "browser_confirmed_free_pdf_lane" => vec![
            "resolver".to_string(),
            "citation_graph".to_string(),
            "archive".to_string(),
        ],
        "metadata_confirmed_pdf_path_resolution" => vec![
            "archive".to_string(),
            "resolver".to_string(),
            "citation_graph".to_string(),
        ],
        "holder_workflow_followup" | "article_or_chapter_delivery_followup" => vec![
            "citation_graph".to_string(),
            "resolver".to_string(),
            "catalog".to_string(),
        ],
        "terminology_and_alias_audit" => vec![
            "archive".to_string(),
            "citation_graph".to_string(),
            "open".to_string(),
        ],
        _ => vec![
            "archive".to_string(),
            "citation_graph".to_string(),
            "resolver".to_string(),
        ],
    }
}

fn default_min_relevance(target: &SearchQueueTarget) -> i64 {
    match target.kind.as_str() {
        "browser_confirmed_free_pdf_lane" => 24,
        "metadata_confirmed_pdf_path_resolution" => 22,
        "holder_workflow_followup" | "article_or_chapter_delivery_followup" => 18,
        "terminology_and_alias_audit" => 12,
        _ => 14,
    }
}

fn expected_year_from_target(target: &SearchQueueTarget) -> Option<u32> {
    for token in target
        .id
        .split('_')
        .chain(target.title.split_whitespace())
        .chain(
            target
                .query_seeds
                .iter()
                .flat_map(|query| query.split_whitespace()),
        )
    {
        let digits = token
            .chars()
            .filter(|ch| ch.is_ascii_digit())
            .collect::<String>();
        if digits.len() == 4
            && let Ok(year) = digits.parse::<u32>()
            && (1500..=2026).contains(&year)
        {
            return Some(year);
        }
    }
    None
}

fn candidate_passes_stage_filters(target: &SearchQueueTarget, paper: &Paper) -> bool {
    let title_overlap = title_overlap_score(target, paper);
    let identifier_match = candidate_matches_target_identifier(target, paper);
    let minimum_overlap = minimum_title_overlap(target);
    if title_overlap < minimum_overlap && !identifier_match {
        return false;
    }
    let Some(expected_year) = expected_year_from_target(target) else {
        return true;
    };
    if paper.year == 0 {
        return !is_exact_retrieval_kind(target);
    }
    let delta = paper.year.abs_diff(expected_year);
    match target.kind.as_str() {
        "browser_confirmed_free_pdf_lane" | "metadata_confirmed_pdf_path_resolution" => delta <= 10,
        "holder_workflow_followup" | "article_or_chapter_delivery_followup" => delta <= 35,
        _ => true,
    }
}

fn candidate_is_default_selected(
    target: &SearchQueueTarget,
    ranked: &RankedPaper<'_>,
    rank: usize,
) -> bool {
    let expected_year = expected_year_from_target(target);
    let year_matches = expected_year
        .map(|expected| ranked.paper.year > 0 && ranked.paper.year.abs_diff(expected) <= 5)
        .unwrap_or(false);
    match target.kind.as_str() {
        "browser_confirmed_free_pdf_lane" | "metadata_confirmed_pdf_path_resolution" => {
            rank == 1 && year_matches
        }
        "holder_workflow_followup" | "article_or_chapter_delivery_followup" => {
            rank == 1 && candidate_passes_stage_filters(target, ranked.paper)
        }
        _ => rank == 1,
    }
}

fn is_exact_retrieval_kind(target: &SearchQueueTarget) -> bool {
    matches!(
        target.kind.as_str(),
        "browser_confirmed_free_pdf_lane" | "metadata_confirmed_pdf_path_resolution"
    )
}

fn minimum_title_overlap(target: &SearchQueueTarget) -> usize {
    match target.kind.as_str() {
        "browser_confirmed_free_pdf_lane" | "metadata_confirmed_pdf_path_resolution" => 2,
        "holder_workflow_followup" | "article_or_chapter_delivery_followup" => 2,
        _ => 1,
    }
}

fn title_overlap_score(target: &SearchQueueTarget, paper: &Paper) -> usize {
    let paper_tokens = collect_match_tokens(&paper.title)
        .into_iter()
        .collect::<HashSet<_>>();
    if paper_tokens.is_empty() {
        return 0;
    }
    collect_target_match_tokens(target)
        .into_iter()
        .filter(|token| paper_tokens.contains(token))
        .count()
}

fn collect_target_match_tokens(target: &SearchQueueTarget) -> HashSet<String> {
    target
        .query_seeds
        .iter()
        .chain(std::iter::once(&target.title))
        .flat_map(|value| collect_match_tokens(value))
        .filter(|token| !is_generic_match_token(token))
        .collect()
}

fn is_generic_match_token(token: &str) -> bool {
    matches!(
        token,
        "exact"
            | "original"
            | "utrecht"
            | "body"
            | "stable"
            | "direct"
            | "follow"
            | "followup"
            | "workflow"
            | "delivery"
            | "article"
            | "chapter"
            | "lane"
            | "with"
            | "their"
            | "always"
    )
}

fn candidate_matches_target_identifier(target: &SearchQueueTarget, paper: &Paper) -> bool {
    let haystack = [
        paper.doi.as_str(),
        paper.url.as_str(),
        paper.pdf_url.as_str(),
        paper.paper_id.as_str(),
        paper.title.as_str(),
    ]
    .join(" ")
    .to_ascii_lowercase();
    if haystack.is_empty() {
        return false;
    }
    collect_target_identifiers(target)
        .into_iter()
        .any(|identifier| haystack.contains(&identifier))
}

fn collect_target_identifiers(target: &SearchQueueTarget) -> HashSet<String> {
    let mut identifiers = HashSet::new();
    for value in target
        .query_seeds
        .iter()
        .chain(std::iter::once(&target.id))
        .chain(std::iter::once(&target.title))
    {
        let lower = value.to_ascii_lowercase();
        if let Some(doi) = extract_target_doi(&lower) {
            identifiers.insert(doi);
        }
        for token in lower.split(|ch: char| {
            ch.is_whitespace() || matches!(ch, '"' | '\'' | ',' | ';' | '(' | ')')
        }) {
            let cleaned = token.trim_matches(|ch: char| ch == '.' || ch == ':');
            if looks_like_identifier(cleaned) {
                identifiers.insert(cleaned.to_string());
            }
        }
    }
    identifiers
}

fn extract_target_doi(value: &str) -> Option<String> {
    value
        .find("10.")
        .map(|index| &value[index..])
        .map(|tail| {
            tail.chars()
                .take_while(|ch| !ch.is_whitespace() && *ch != '"' && *ch != '\'' && *ch != ')')
                .collect::<String>()
        })
        .filter(|candidate| candidate.contains('/'))
}

fn looks_like_identifier(value: &str) -> bool {
    if value.len() >= 8 && value.chars().all(|ch| ch.is_ascii_digit()) {
        return true;
    }
    value.len() >= 12
        && value.chars().any(|ch| ch.is_ascii_digit())
        && value.contains('-')
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '-' | '.' | '/'))
}

fn map_source_family(source: &str) -> String {
    match normalize_source_name(source).as_str() {
        "openalex" | "semantic_scholar" => "citation_graph".to_string(),
        "crossref" | "datacite" | "unpaywall" => "resolver".to_string(),
        "arxiv" | "jstage" | "inspirehep" | "hal" | "dblp" => "archive".to_string(),
        "europepmc" | "scielo" | "core" | "cinii" | "ads" | "lens" | "google_scholar" => {
            "catalog".to_string()
        }
        other => other.to_string(),
    }
}

fn canonical_candidate_id(paper: &Paper) -> String {
    let canonical_doi = canonicalize_doi(&paper.doi);
    if !canonical_doi.is_empty() {
        return format!("doi:{canonical_doi}");
    }
    for candidate in [&paper.url, &paper.pdf_url, &paper.paper_id] {
        if let Some(openalex_id) = openalex_id(candidate) {
            return format!("openalex:{openalex_id}");
        }
    }
    if !paper.arxiv_id.is_empty() {
        return format!("arxiv:{}", paper.arxiv_id.to_ascii_lowercase());
    }
    format!("title:{}", sanitize_token(&paper.title))
}

fn openalex_id(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with('W') && trimmed[1..].chars().all(|ch| ch.is_ascii_digit()) {
        return Some(trimmed.to_string());
    }
    let parsed = Url::parse(trimmed).ok()?;
    if !parsed.host_str()?.contains("openalex.org") {
        return None;
    }
    parsed
        .path_segments()?
        .find(|segment| segment.starts_with('W'))
        .map(ToOwned::to_owned)
}

fn preferred_candidate_url(paper: &Paper) -> String {
    if !paper.pdf_url.trim().is_empty() {
        paper.pdf_url.clone()
    } else {
        paper.url.clone()
    }
}

fn classify_route_class(paper: &Paper) -> String {
    let candidate_url = preferred_candidate_url(paper);
    if !paper.pdf_url.trim().is_empty() || candidate_url.to_ascii_lowercase().ends_with(".pdf") {
        return "direct_pdf".to_string();
    }
    let Some(host) = Url::parse(&candidate_url)
        .ok()
        .and_then(|parsed| parsed.host_str().map(ToOwned::to_owned))
    else {
        return "generic_landing".to_string();
    };
    if host == "doi.org" || host == "dx.doi.org" {
        "doi_resolver".to_string()
    } else if host.contains("openalex.org") {
        "openalex_landing".to_string()
    } else if host.contains("crossref") {
        "crossref_landing".to_string()
    } else if host.contains("ams.org") || host.contains("springer") || host.contains("celebratio") {
        "metadata_landing".to_string()
    } else {
        "generic_landing".to_string()
    }
}

fn classify_host_class(url: &str) -> String {
    Url::parse(url)
        .ok()
        .and_then(|parsed| parsed.host_str().map(sanitize_token))
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown_host".to_string())
}

fn preferred_link(hit: &DossierHit) -> String {
    if !hit.pdf_url.is_empty() {
        hit.pdf_url.clone()
    } else {
        hit.url.clone()
    }
}

fn default_batch_manifest_path(output_dir: &Path, filters: &QueueFilters) -> PathBuf {
    let batch_label = if !filters.window.is_empty() {
        sanitize_token(&filters.window.join("_"))
    } else if filters.critical_only {
        "critical".to_string()
    } else {
        "mixed".to_string()
    };
    output_dir.join(format!(
        "batch_{}_{}.toml",
        batch_label,
        Utc::now().format("%Y%m%dT%H%M%SZ")
    ))
}

fn collect_dossier_paths(
    gororoba_repo_root: &Path,
    args: &StageFromDossierArgs,
) -> Result<Vec<PathBuf>> {
    let mut seen = HashSet::new();
    let mut paths = Vec::new();
    for dossier in &args.dossier {
        let path = resolve_output_dir(gororoba_repo_root, dossier);
        if seen.insert(path.clone()) {
            paths.push(path);
        }
    }
    for manifest_path in &args.batch_manifest {
        let resolved_manifest = resolve_output_dir(gororoba_repo_root, manifest_path);
        let manifest = load_dossier_batch_manifest(&resolved_manifest)?;
        for entry in manifest.entries {
            let dossier_path =
                resolve_output_dir(gororoba_repo_root, Path::new(&entry.dossier_json));
            if seen.insert(dossier_path.clone()) {
                paths.push(dossier_path);
            }
        }
    }
    Ok(paths)
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

fn escape_tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

fn escape_markdown_cell(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}
