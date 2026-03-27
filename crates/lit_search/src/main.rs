//! CLI for academic literature search.
//!
//! Usage:
//!   lit-search "sedenion zero divisor" --limit 10 --tier all
//!   lit-search --query-file queries.txt --parallel-queries --source openalex --source crossref
//!   lit-search crawl "https://arxiv.org/abs/2301.00001"
//!   lit-search extract-pdf ./paper.pdf

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use lit_search::{
    ALL_SOURCE_NAMES, DatasetDiff, DatasetManifest, DatasetRelease, MultiQueryExecutionReport,
    SOURCE_FAMILY_NAMES, SearchEngine, SearchExecutionReport, SemanticScholarDatasetsClient,
    crawler::WebCrawler, critique::build_critique_prompt, download, evolution::EvolutionStore,
    normalize_source_name, pdf::PdfExtractor, search::SourceTier, source_names_for_family,
    sources::ApiKeys,
};
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "lit-search",
    about = "Academic Research & Literature Intelligence CLI"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Default query if no subcommand is used
    query: Option<String>,

    /// Limit results (default 10)
    #[arg(long, default_value_t = 10)]
    limit: usize,

    /// Tier: open, core, or all (default all)
    #[arg(long, default_value = "all")]
    tier: String,

    /// Minimum publication year
    #[arg(long, default_value_t = 0)]
    year_min: u32,

    /// Download PDFs to directory
    #[arg(long)]
    download: Option<PathBuf>,

    /// Search by DOI instead of query
    #[arg(long)]
    doi: Option<String>,

    /// Domains to search
    #[arg(long)]
    domain: Vec<String>,

    /// Expand domains automatically
    #[arg(long)]
    expand_domains: bool,

    /// Explicit sources to use instead of the default tier set
    #[arg(long = "source")]
    source: Vec<String>,

    /// Expand a named source family into concrete sources
    #[arg(long = "source-family")]
    source_family: Vec<String>,

    /// Read multiple queries from a newline-delimited file
    #[arg(long)]
    query_file: Option<PathBuf>,

    /// Run multi-query searches in parallel
    #[arg(long)]
    parallel_queries: bool,

    /// Print per-source execution stats after the results
    #[arg(long)]
    show_source_stats: bool,

    /// Write the execution report to JSON
    #[arg(long)]
    report_json_out: Option<PathBuf>,

    /// Print the available sources and source families, then exit
    #[arg(long)]
    list_sources: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Search for academic papers
    Search {
        query: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    /// Crawl a web page and convert to markdown
    Crawl {
        url: String,
        #[arg(long, default_value_t = 30)]
        timeout: u64,
    },
    /// Extract text and math from a scientific PDF
    ExtractPdf { path: PathBuf },
    /// Critique a paper draft using academic personas
    Critique {
        #[arg(long)]
        draft: PathBuf,
        #[arg(long)]
        evidence: Option<PathBuf>,
        #[arg(long)]
        persona: String,
    },
    /// Generate prompt overlay from evolution lessons
    Evolution {
        #[arg(long)]
        stage: String,
        #[arg(long, default_value = ".evolution")]
        store: PathBuf,
        #[arg(long)]
        skills: Option<PathBuf>,
    },
    /// Inspect Semantic Scholar dataset releases, manifests, and diffs
    S2Datasets {
        #[command(subcommand)]
        command: DatasetCommands,
    },
}

#[derive(Subcommand, Debug)]
enum DatasetCommands {
    /// List available release IDs
    Releases {
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Show one release and its dataset summaries
    Release {
        release_id: Option<String>,
        #[arg(long, default_value_t = false)]
        latest: bool,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Show one dataset manifest for a release
    Dataset {
        dataset_name: String,
        release_id: Option<String>,
        #[arg(long, default_value_t = false)]
        latest: bool,
        #[arg(long)]
        file_limit: Option<usize>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
    /// Show incremental diff manifests between two releases
    Diff {
        start_release_id: String,
        end_release_id: String,
        dataset_name: String,
        #[arg(long)]
        file_limit: Option<usize>,
        #[arg(long)]
        json_out: Option<PathBuf>,
    },
}

struct RunSearchArgs {
    query: String,
    limit: usize,
    tier: String,
    year_min: u32,
    download_dir: Option<PathBuf>,
    doi: Option<String>,
    domains: Vec<String>,
    expand_domains: bool,
    sources: Vec<String>,
    source_families: Vec<String>,
    query_file: Option<PathBuf>,
    parallel_queries: bool,
    show_source_stats: bool,
    report_json_out: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum SearchReportEnvelope {
    Single { report: SearchExecutionReport },
    Multi { report: MultiQueryExecutionReport },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let cli = Cli::parse();

    if cli.list_sources {
        print_available_sources();
        return Ok(());
    }

    if let Some(cmd) = cli.command {
        match cmd {
            Commands::Search { query, limit } => {
                run_search(RunSearchArgs {
                    query,
                    limit,
                    tier: cli.tier,
                    year_min: cli.year_min,
                    download_dir: cli.download,
                    doi: cli.doi,
                    domains: cli.domain,
                    expand_domains: cli.expand_domains,
                    sources: cli.source,
                    source_families: cli.source_family,
                    query_file: cli.query_file,
                    parallel_queries: cli.parallel_queries,
                    show_source_stats: cli.show_source_stats,
                    report_json_out: cli.report_json_out,
                })
                .await?;
            }
            Commands::Crawl { url, timeout } => {
                let crawler = WebCrawler::new(timeout, 100_000);
                println!("Crawling {}...", url);
                let res = crawler.crawl(&url).await;
                if res.success {
                    println!("\n# {}\n", res.title);
                    println!("{}", res.markdown);
                } else {
                    eprintln!("Crawl failed: {:?}", res.error);
                }
            }
            Commands::ExtractPdf { path } => {
                println!("Extracting PDF: {}...", path.display());
                let md = PdfExtractor::extract_to_markdown(&path)?;
                println!("{}", md);
            }
            Commands::Critique {
                draft,
                evidence,
                persona,
            } => {
                let draft_text =
                    std::fs::read_to_string(draft).context("Failed to read draft file")?;
                let evidence_text = if let Some(e) = evidence {
                    std::fs::read_to_string(e).unwrap_or_default()
                } else {
                    String::new()
                };
                let (system, user) =
                    build_critique_prompt(&draft_text, &evidence_text, "", &persona);
                println!("--- PERSONA: {} ---", persona.to_uppercase());
                println!("--- SYSTEM PROMPT ---\n{}\n", system);
                println!("--- USER PROMPT ---\n{}\n", user);
            }
            Commands::Evolution {
                stage,
                store,
                skills,
            } => {
                let store_inst = EvolutionStore::new(store)?;
                let overlay = store_inst.build_overlay(&stage, 5, skills.as_deref());
                println!("{}", overlay);
            }
            Commands::S2Datasets { command } => {
                run_s2_datasets(command).await?;
            }
        }
    } else if let Some(query) = cli.query {
        run_search(RunSearchArgs {
            query,
            limit: cli.limit,
            tier: cli.tier,
            year_min: cli.year_min,
            download_dir: cli.download,
            doi: cli.doi,
            domains: cli.domain,
            expand_domains: cli.expand_domains,
            sources: cli.source,
            source_families: cli.source_family,
            query_file: cli.query_file,
            parallel_queries: cli.parallel_queries,
            show_source_stats: cli.show_source_stats,
            report_json_out: cli.report_json_out,
        })
        .await?;
    } else {
        eprintln!("Error: No query or command provided. Use --help for usage.");
        std::process::exit(1);
    }

    Ok(())
}

async fn run_s2_datasets(command: DatasetCommands) -> Result<()> {
    let keys = ApiKeys::from_env();
    let client = SemanticScholarDatasetsClient::new(keys.s2_api_key.clone());

    match command {
        DatasetCommands::Releases { limit, json_out } => {
            let releases = client.list_releases().await?;
            let total = releases.len();
            let start = total.saturating_sub(limit);
            let visible = releases[start..].to_vec();
            println!("Semantic Scholar dataset releases: {} total", total);
            for release in &visible {
                println!("  {release}");
            }
            write_optional_json(&json_out, &visible)?;
        }
        DatasetCommands::Release {
            release_id,
            latest,
            json_out,
        } => {
            let release_id = resolve_release_id(&client, release_id, latest).await?;
            let release = client.fetch_release(&release_id).await?;
            print_release(&release);
            write_optional_json(&json_out, &release)?;
        }
        DatasetCommands::Dataset {
            dataset_name,
            release_id,
            latest,
            file_limit,
            json_out,
        } => {
            let release_id = resolve_release_id(&client, release_id, latest).await?;
            let manifest = client
                .fetch_dataset_manifest(&release_id, &dataset_name)
                .await?;
            print_manifest(&release_id, &manifest, file_limit);
            write_optional_json(&json_out, &manifest)?;
        }
        DatasetCommands::Diff {
            start_release_id,
            end_release_id,
            dataset_name,
            file_limit,
            json_out,
        } => {
            let diff = client
                .fetch_dataset_diff(&start_release_id, &end_release_id, &dataset_name)
                .await?;
            print_diff(&diff, file_limit);
            write_optional_json(&json_out, &diff)?;
        }
    }

    Ok(())
}

async fn run_search(args: RunSearchArgs) -> Result<()> {
    let tier = match args.tier.to_lowercase().as_str() {
        "core" => SourceTier::Core,
        "open" => SourceTier::Open,
        _ => SourceTier::All,
    };

    let keys = ApiKeys::from_env();
    let engine = SearchEngine::new(keys, tier);
    let mut sources = args
        .sources
        .iter()
        .map(|source| normalize_source_name(source))
        .collect::<Vec<_>>();
    for family in &args.source_families {
        let Some(family_sources) = source_names_for_family(family) else {
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
    let queries = if let Some(query_file) = &args.query_file {
        load_query_file(query_file)?
    } else {
        vec![args.query.clone()]
    };

    let (results, report) = if let Some(doi) = args.doi {
        let outcome = engine
            .search_with_sources(&format!("DOI:{doi}"), 5, 0, &sources)
            .await;
        (
            outcome.papers,
            SearchReportEnvelope::Single {
                report: outcome.report,
            },
        )
    } else if args.expand_domains || !args.domains.is_empty() {
        let outcome = engine
            .search_topic_with_sources(
                &args.query,
                &args.domains,
                args.limit,
                args.year_min,
                &sources,
            )
            .await;
        (
            outcome.papers,
            SearchReportEnvelope::Multi {
                report: outcome.report,
            },
        )
    } else if queries.len() > 1 {
        let outcome = if args.parallel_queries {
            engine
                .search_queries_parallel_with_sources(&queries, args.limit, args.year_min, &sources)
                .await
        } else {
            engine
                .search_queries_with_sources(&queries, args.limit, args.year_min, &sources)
                .await
        };
        (
            outcome.papers,
            SearchReportEnvelope::Multi {
                report: outcome.report,
            },
        )
    } else {
        let outcome = engine
            .search_with_sources(&args.query, args.limit, args.year_min, &sources)
            .await;
        (
            outcome.papers,
            SearchReportEnvelope::Single {
                report: outcome.report,
            },
        )
    };

    println!("Found {} results:\n", results.len());
    for (idx, paper) in results.iter().enumerate() {
        println!("{}. {} ({})", idx + 1, paper.title, paper.year);
        if !paper.doi.is_empty() {
            println!("   DOI: {}", paper.doi);
        }
        if !paper.arxiv_id.is_empty() {
            println!("   arXiv: {}", paper.arxiv_id);
        }
        if !paper.pdf_url.is_empty() {
            println!("   PDF: {}", paper.pdf_url);
        }
        println!(
            "   Citations: {} | Source: {}",
            paper.citation_count, paper.source
        );
        println!();
    }

    if args.show_source_stats {
        print_execution_report(&report);
    }

    if let Some(dir) = args.download_dir {
        let with_pdf: usize = results
            .iter()
            .filter(|paper| !paper.pdf_url.is_empty())
            .count();
        println!("Downloading {with_pdf} PDFs to {}...\n", dir.display());
        let dl_results = download::download_pdfs(&results, &dir, engine.client()).await;
        for result in &dl_results {
            if let Some(path) = &result.path {
                println!("  OK: {} -> {}", result.title, path.display());
            } else if let Some(err) = &result.error {
                println!("  FAIL: {} -- {}", result.title, err);
            }
        }
    }

    if let Some(report_json_out) = args.report_json_out {
        if let Some(parent) = report_json_out.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create {}", parent.display()))?;
        }
        std::fs::write(&report_json_out, serde_json::to_string_pretty(&report)?)
            .with_context(|| format!("Failed to write {}", report_json_out.display()))?;
    }

    if std::env::var("LIT_SEARCH_JSON").is_ok() {
        println!("{}", serde_json::to_string_pretty(&results).unwrap());
    }

    Ok(())
}

fn print_available_sources() {
    println!("searchable_sources={}", ALL_SOURCE_NAMES.join(", "));
    for family in SOURCE_FAMILY_NAMES {
        let entries = source_names_for_family(family)
            .map(|sources| sources.join(", "))
            .unwrap_or_default();
        println!("source_family[{family}]={entries}");
    }
}

async fn resolve_release_id(
    client: &SemanticScholarDatasetsClient,
    release_id: Option<String>,
    latest: bool,
) -> Result<String> {
    match (release_id, latest) {
        (Some(id), false) => Ok(id),
        (None, true) => client.latest_release_id().await.map_err(Into::into),
        (Some(_), true) => bail!("provide either a release_id or --latest, not both"),
        (None, false) => bail!("provide a release_id or pass --latest"),
    }
}

fn print_release(release: &DatasetRelease) {
    println!("release_id={}", release.release_id);
    println!("datasets={}", release.datasets.len());
    for dataset in &release.datasets {
        println!("- {}", dataset.name);
        if !dataset.description.is_empty() {
            println!("  {}", single_line(&dataset.description));
        }
    }
}

fn print_manifest(release_id: &str, manifest: &DatasetManifest, file_limit: Option<usize>) {
    println!("release_id={release_id}");
    println!("dataset={}", manifest.name);
    if !manifest.description.is_empty() {
        println!("description={}", single_line(&manifest.description));
    }
    println!("files={}", manifest.files.len());
    let visible = file_limit.unwrap_or(10).min(manifest.files.len());
    for file in manifest.files.iter().take(visible) {
        println!("  file={file}");
    }
    if visible < manifest.files.len() {
        println!("  ... {} more files", manifest.files.len() - visible);
    }
}

fn print_diff(diff: &DatasetDiff, file_limit: Option<usize>) {
    println!("dataset={}", diff.dataset);
    println!("start_release={}", diff.start_release);
    println!("end_release={}", diff.end_release);
    println!("diff_segments={}", diff.diffs.len());
    for segment in &diff.diffs {
        println!(
            "- {} -> {} | update_files={} delete_files={}",
            segment.from_release,
            segment.to_release,
            segment.update_files.len(),
            segment.delete_files.len()
        );
        let visible = file_limit.unwrap_or(5).min(segment.update_files.len());
        for file in segment.update_files.iter().take(visible) {
            println!("  update={file}");
        }
        if visible < segment.update_files.len() {
            println!(
                "  ... {} more update files",
                segment.update_files.len() - visible
            );
        }
    }
}

fn single_line(text: &str) -> String {
    text.lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("")
        .trim()
        .to_string()
}

fn write_optional_json<T: Serialize>(path: &Option<PathBuf>, value: &T) -> Result<()> {
    if let Some(path) = path {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("Failed to create {}", parent.display()))?;
        }
        std::fs::write(path, serde_json::to_string_pretty(value)?)
            .with_context(|| format!("Failed to write {}", path.display()))?;
    }
    Ok(())
}

fn load_query_file(path: &Path) -> Result<Vec<String>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let queries = raw
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    if queries.is_empty() {
        bail!(
            "query file {} did not contain any non-empty lines",
            path.display()
        );
    }
    Ok(queries)
}

fn print_execution_report(report: &SearchReportEnvelope) {
    match report {
        SearchReportEnvelope::Single { report } => {
            println!("\nExecution report:");
            print_single_report(report);
        }
        SearchReportEnvelope::Multi { report } => {
            println!("\nExecution report:");
            println!(
                "  queries={} raw_results={} deduplicated={}",
                report.queries.len(),
                report.raw_result_count,
                report.deduplicated_result_count
            );
            for query_report in &report.query_reports {
                print_single_report(query_report);
            }
        }
    }
}

fn print_single_report(report: &SearchExecutionReport) {
    println!(
        "  query={} raw_results={} deduplicated={} cache_hits={}",
        report.query,
        report.raw_result_count,
        report.deduplicated_result_count,
        report.cache_hit_count
    );
    for source_report in &report.source_reports {
        let mut suffix = String::new();
        if source_report.cache_hit {
            suffix.push_str(" cache_hit");
        }
        if !source_report.skipped_reason.is_empty() {
            suffix.push_str(&format!(" skipped={}", source_report.skipped_reason));
        }
        if !source_report.error.is_empty() {
            suffix.push_str(&format!(" error={}", source_report.error));
        }
        println!(
            "    {} -> {}{}",
            source_report.source, source_report.result_count, suffix
        );
    }
}
