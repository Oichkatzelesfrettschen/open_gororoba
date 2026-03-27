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
    MultiQueryExecutionReport, SearchEngine, SearchExecutionReport, crawler::WebCrawler,
    critique::build_critique_prompt, download, evolution::EvolutionStore, normalize_source_name,
    pdf::PdfExtractor, search::SourceTier, sources::ApiKeys,
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

async fn run_search(args: RunSearchArgs) -> Result<()> {
    let tier = match args.tier.to_lowercase().as_str() {
        "core" => SourceTier::Core,
        "open" => SourceTier::Open,
        _ => SourceTier::All,
    };

    let keys = ApiKeys::from_env();
    let engine = SearchEngine::new(keys, tier);
    let sources = args
        .sources
        .iter()
        .map(|source| normalize_source_name(source))
        .collect::<Vec<_>>();
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
