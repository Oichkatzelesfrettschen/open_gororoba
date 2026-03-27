use anyhow::{Context, Result};
use chrono::Utc;
use clap::{Args, Parser, Subcommand, ValueEnum};
use gororoba_cli_data::project_api_contract::{
    ProjectApiContext, ProjectApiCrosswalkBinding, ProjectApiCrosswalkFile,
    load_project_api_context, load_project_api_crosswalk,
};
use lit_search::{
    MultiQueryExecutionOutcome, Paper, SearchEngine, normalize_source_name, search::SourceTier,
    sources::ApiKeys,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

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

    #[arg(long, default_value = "docs/reports/cd_cache_dossiers")]
    output_dir: PathBuf,

    #[arg(long, default_value_t = 10)]
    max_hits: usize,
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

#[derive(Debug, Clone, Deserialize, Serialize)]
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

#[derive(Debug, Serialize)]
struct ResearchDossier {
    generated_at_utc: String,
    project_id: String,
    search_target_id: String,
    crosswalk_id: String,
    title: String,
    window: String,
    priority: String,
    status: String,
    kind: String,
    why_now: String,
    query_seeds: Vec<String>,
    requested_sources: Vec<String>,
    limit_per_query: usize,
    year_min: u32,
    report: lit_search::MultiQueryExecutionReport,
    top_hits: Vec<DossierHit>,
    stage_suggestions: Vec<StageSuggestion>,
}

#[derive(Debug, Serialize)]
struct DossierHit {
    rank: usize,
    relevance_score: i64,
    title: String,
    year: u32,
    source: String,
    venue: String,
    citation_count: u32,
    doi: String,
    url: String,
    pdf_url: String,
}

#[derive(Debug, Serialize)]
struct StageSuggestion {
    rank: usize,
    relevance_score: i64,
    paper_title: String,
    action: String,
    candidate_url: String,
    command: String,
    rationale: String,
}

struct DossierBuildConfig<'a> {
    project_id: &'a str,
    gororoba_repo_root: &'a Path,
    limit_per_query: usize,
    year_min: u32,
    max_hits: usize,
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
            query_seed_preview: target
                .query_seeds
                .first()
                .cloned()
                .unwrap_or_default(),
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
    _project_api: &ProjectApiContext,
    crosswalk: &ProjectApiCrosswalkFile,
    queue: &SearchQueueFile,
    gororoba_repo_root: &Path,
    args: ResearchDossierArgs,
) -> Result<()> {
    let output_dir = resolve_output_dir(gororoba_repo_root, &args.output_dir);
    fs::create_dir_all(&output_dir).with_context(|| format!("create {}", output_dir.display()))?;

    let tier = parse_source_tier(&args.tier);
    let sources = args
        .source
        .iter()
        .map(|source| normalize_source_name(source))
        .collect::<Vec<_>>();
    let engine = SearchEngine::new(ApiKeys::from_env(), tier);
    let project_id = queue.project_id.as_deref().unwrap_or("cayley-dickson");

    for target in filtered_targets(queue, &args.filters) {
        let binding = binding_for_target(crosswalk, target);
        let queries = deduplicate_query_seeds(target);
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
                gororoba_repo_root,
                limit_per_query: args.limit_per_query,
                year_min: args.year_min,
                max_hits: args.max_hits,
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
        let json_path = output_dir.join(format!("{stem}.json"));
        let markdown_path = output_dir.join(format!("{stem}.md"));
        fs::write(&json_path, serde_json::to_string_pretty(&dossier)?)
            .with_context(|| format!("write {}", json_path.display()))?;
        fs::write(&markdown_path, render_research_markdown(&dossier))
            .with_context(|| format!("write {}", markdown_path.display()))?;
        println!(
            "wrote_json={} wrote_markdown={} search_target={} hits={}",
            json_path.display(),
            markdown_path.display(),
            dossier.search_target_id,
            dossier.top_hits.len()
        );
    }

    Ok(())
}

fn build_dossier(
    config: DossierBuildConfig<'_>,
    target: &SearchQueueTarget,
    binding: Option<&ProjectApiCrosswalkBinding>,
    queries: &[String],
    outcome: MultiQueryExecutionOutcome,
) -> ResearchDossier {
    let ranked_candidates = rank_papers_for_target(target, &outcome.papers);
    let top_hits = ranked_candidates
        .iter()
        .take(config.max_hits)
        .enumerate()
        .map(|(index, (paper, relevance_score))| DossierHit {
            rank: index + 1,
            relevance_score: *relevance_score,
            title: paper.title.clone(),
            year: paper.year,
            source: paper.source.clone(),
            venue: paper.venue.clone(),
            citation_count: paper.citation_count,
            doi: paper.doi.clone(),
            url: paper.url.clone(),
            pdf_url: paper.pdf_url.clone(),
        })
        .collect::<Vec<_>>();

    let stage_suggestions = ranked_candidates
        .iter()
        .take(config.max_hits)
        .enumerate()
        .filter_map(|(index, (paper, relevance_score))| {
            build_stage_suggestion(
                config.gororoba_repo_root,
                target,
                paper,
                *relevance_score,
                index + 1,
            )
        })
        .collect::<Vec<_>>();

    ResearchDossier {
        generated_at_utc: Utc::now().to_rfc3339(),
        project_id: config.project_id.to_string(),
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
        limit_per_query: config.limit_per_query,
        year_min: config.year_min,
        report: outcome.report,
        top_hits,
        stage_suggestions,
    }
}

fn build_stage_suggestion(
    gororoba_repo_root: &Path,
    target: &SearchQueueTarget,
    paper: &Paper,
    relevance_score: i64,
    rank: usize,
) -> Option<StageSuggestion> {
    let candidate_url = if !paper.pdf_url.is_empty() {
        paper.pdf_url.clone()
    } else {
        paper.url.clone()
    };
    if candidate_url.is_empty() || relevance_score < 20 {
        return None;
    }

    let action = if !paper.pdf_url.is_empty() {
        "stage_direct_pdf"
    } else {
        "stage_landing_page"
    };
    let command = format!(
        "cargo run -q -p gororoba_cli_data --bin human-acquire -- --repo-root {} stage --url {} --source-id {} --title {} --note {}",
        sh_quote(&gororoba_repo_root.display().to_string()),
        sh_quote(&candidate_url),
        sh_quote(&paper.source),
        sh_quote(&paper.title),
        sh_quote(&format!("search_target_id={}", target.id)),
    );
    Some(StageSuggestion {
        rank,
        relevance_score,
        paper_title: paper.title.clone(),
        action: action.to_string(),
        candidate_url,
        command,
        rationale: format!(
            "Top dossier hit from {} with relevance score {} and {} citations for queue target {}.",
            paper.source, relevance_score, paper.citation_count, target.id
        ),
    })
}

fn render_research_markdown(dossier: &ResearchDossier) -> String {
    let mut body = String::new();
    body.push_str(&format!("# {}\n\n", dossier.title));
    body.push_str(&format!(
        "- search_target_id: `{}`\n- crosswalk_id: `{}`\n- window: `{}`\n- priority: `{}`\n- status: `{}`\n- kind: `{}`\n- generated_at_utc: `{}`\n\n",
        dossier.search_target_id,
        dossier.crosswalk_id,
        dossier.window,
        dossier.priority,
        dossier.status,
        dossier.kind,
        dossier.generated_at_utc
    ));
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
    body.push_str("| rank | relevance | year | source | citations | title | url | pdf |\n");
    body.push_str("| --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for hit in &dossier.top_hits {
        body.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
            hit.rank,
            hit.relevance_score,
            hit.year,
            escape_markdown_cell(&hit.source),
            hit.citation_count,
            escape_markdown_cell(&hit.title),
            escape_markdown_cell(&hit.url),
            escape_markdown_cell(&hit.pdf_url),
        ));
    }
    body.push('\n');

    body.push_str("## Stage Suggestions\n");
    if dossier.stage_suggestions.is_empty() {
        body.push_str("- No stageable URLs were present in the top hits.\n");
    } else {
        for suggestion in &dossier.stage_suggestions {
            body.push_str(&format!(
                "- rank {}: `{}`\n  relevance: `{}`\n  action: `{}`\n  url: `{}`\n  command: `{}`\n",
                suggestion.rank,
                suggestion.paper_title,
                suggestion.relevance_score,
                suggestion.action,
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
) -> Vec<(&'a Paper, i64)> {
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
            if normalized_paper_title.contains("cayley")
                || normalized_paper_title.contains("dickson")
            {
                score += 4;
            }
            (paper, score)
        })
        .collect::<Vec<_>>();

    ranked.sort_by(|(left_paper, left_score), (right_paper, right_score)| {
        right_score
            .cmp(left_score)
            .then_with(|| right_paper.citation_count.cmp(&left_paper.citation_count))
            .then_with(|| right_paper.year.cmp(&left_paper.year))
    });
    ranked
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

fn sh_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn escape_tsv_field(value: &str) -> String {
    value.replace(['\t', '\n', '\r'], " ")
}

fn escape_markdown_cell(value: &str) -> String {
    value.replace('|', "\\|").replace('\n', " ")
}
