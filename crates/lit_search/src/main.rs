//! CLI for academic literature search.
//!
//! Usage:
//!   lit-search "sedenion zero divisor" --limit 10 --tier all
//!   lit-search --doi "10.1016/S0096-3003(99)00140-X"
//!   lit-search "cayley dickson" --download ./papers/

use lit_search::{SearchEngine, download, search::SourceTier, sources::ApiKeys};
use std::path::PathBuf;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!(
            "Usage: lit-search <query> [--limit N] [--tier core|open|all] [--doi DOI] [--year-min YEAR] [--download DIR]"
        );
        std::process::exit(1);
    }

    let mut query = String::new();
    let mut limit = 10_usize;
    let mut tier = SourceTier::All;
    let mut year_min = 0_u32;
    let mut doi_mode = false;
    let mut download_dir: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--limit" => {
                i += 1;
                limit = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(10);
            }
            "--tier" => {
                i += 1;
                tier = match args.get(i).map(|s| s.as_str()) {
                    Some("core") => SourceTier::Core,
                    Some("open") => SourceTier::Open,
                    _ => SourceTier::All,
                };
            }
            "--year-min" => {
                i += 1;
                year_min = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            }
            "--doi" => {
                doi_mode = true;
                i += 1;
                query = args.get(i).cloned().unwrap_or_default();
            }
            "--download" => {
                i += 1;
                download_dir = args.get(i).map(PathBuf::from);
            }
            s if !s.starts_with("--") => {
                query = s.to_string();
            }
            _ => {}
        }
        i += 1;
    }

    let keys = ApiKeys::from_env();
    let engine = SearchEngine::new(keys, tier);

    let results = if doi_mode {
        engine.search_by_doi(&query).await
    } else {
        engine.search(&query, limit, year_min).await
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

    // Download PDFs if requested
    if let Some(dir) = download_dir {
        let with_pdf: usize = results.iter().filter(|p| !p.pdf_url.is_empty()).count();
        println!("Downloading {with_pdf} PDFs to {}...\n", dir.display());

        let dl_results = download::download_pdfs(&results, &dir, engine.client()).await;

        let success = dl_results.iter().filter(|r| r.path.is_some()).count();
        let failed = dl_results.iter().filter(|r| r.error.is_some()).count();

        for r in &dl_results {
            if let Some(path) = &r.path {
                println!("  OK: {} -> {}", r.title, path.display());
            } else if let Some(err) = &r.error {
                println!("  FAIL: {} -- {}", r.title, err);
            }
        }

        println!("\nDownload summary: {success} succeeded, {failed} failed");
    }

    // JSON output for piping
    if std::env::var("LIT_SEARCH_JSON").is_ok() {
        println!("{}", serde_json::to_string_pretty(&results).unwrap());
    }
}
