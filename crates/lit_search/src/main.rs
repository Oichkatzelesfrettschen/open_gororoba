//! CLI for academic literature search.
//!
//! Usage:
//!   lit-search "sedenion zero divisor" --limit 10 --tier all
//!   lit-search crawl "https://arxiv.org/abs/2301.00001"
//!   lit-search extract-pdf ./paper.pdf

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use lit_search::{
    crawler::WebCrawler,
    critique::build_critique_prompt,
    download,
    evolution::EvolutionStore,
    pdf::PdfExtractor,
    search::SourceTier,
    sources::ApiKeys,
    SearchEngine,
};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "lit-search", about = "Academic Research & Literature Intelligence CLI")]
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
    ExtractPdf {
        path: PathBuf,
    },
    /// Critique a paper draft using academic personas
    Critique {
        #[arg(long)]
        draft: PathBuf,
        #[arg(long)]
        evidence: Option<PathBuf>,
        #[arg(long)]
        persona: String, // board, balanced, tank, bros
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
            Commands::Critique { draft, evidence, persona } => {
                let draft_text = std::fs::read_to_string(draft).context("Failed to read draft file")?;
                let evidence_text = if let Some(e) = evidence {
                    std::fs::read_to_string(e).unwrap_or_default()
                } else {
                    String::new()
                };
                let (system, user) = build_critique_prompt(&draft_text, &evidence_text, "", &persona);
                println!("--- PERSONA: {} ---", persona.to_uppercase());
                println!("--- SYSTEM PROMPT ---\n{}\n", system);
                println!("--- USER PROMPT ---\n{}\n", user);
            }
            Commands::Evolution { stage, store, skills } => {
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

    let results = if let Some(d) = args.doi {
        engine.search_by_doi(&d).await
    } else if args.expand_domains || !args.domains.is_empty() {
        engine.search_topic(&args.query, &args.domains, args.limit, args.year_min).await
    } else {
        engine.search(&args.query, args.limit, args.year_min).await
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

    if let Some(dir) = args.download_dir {
        let with_pdf: usize = results.iter().filter(|p| !p.pdf_url.is_empty()).count();
        println!("Downloading {with_pdf} PDFs to {}...\n", dir.display());
        let dl_results = download::download_pdfs(&results, &dir, engine.client()).await;
        for r in &dl_results {
            if let Some(path) = &r.path {
                println!("  OK: {} -> {}", r.title, path.display());
            } else if let Some(err) = &r.error {
                println!("  FAIL: {} -- {}", r.title, err);
            }
        }
    }

    if std::env::var("LIT_SEARCH_JSON").is_ok() {
        println!("{}", serde_json::to_string_pretty(&results).unwrap());
    }

    Ok(())
}
