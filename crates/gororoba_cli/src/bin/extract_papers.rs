//! Batch PDF extraction: reads papers/MANIFEST.toml, extracts each paper
//! to structured TOML + equations.tex + table CSVs using the docpipe crate.

use std::path::{Path, PathBuf};

use clap::Parser;

/// Extract all papers listed in MANIFEST.toml to structured TOML output.
#[derive(Parser)]
#[command(name = "extract-papers")]
struct Args {
    /// Path to MANIFEST.toml (default: papers/MANIFEST.toml)
    #[arg(long, default_value = "papers/MANIFEST.toml")]
    manifest: PathBuf,

    /// Output directory for extracted papers (default: papers/extracted)
    #[arg(long, default_value = "papers/extracted")]
    output: PathBuf,

    /// Only extract a single paper by ID
    #[arg(long)]
    only: Option<String>,

    /// Skip papers that already have paper.toml in output dir
    #[arg(long, default_value_t = false)]
    skip_existing: bool,
}

/// Minimal MANIFEST.toml structure for paper entries.
#[derive(serde::Deserialize)]
struct Manifest {
    paper: Vec<ManifestEntry>,
}

#[derive(serde::Deserialize)]
struct ManifestEntry {
    id: String,
    title: String,
    authors: Vec<String>,
    year: Option<u32>,     // Some entries might not have year
    arxiv: Option<String>, // Empty string or missing
    local_pdf: String,
    #[allow(dead_code)]
    status: String,
}

fn main() {
    let args = Args::parse();

    let manifest_text = match std::fs::read_to_string(&args.manifest) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("ERROR: cannot read {}: {e}", args.manifest.display());
            std::process::exit(1);
        }
    };

    let manifest: Manifest = match toml::from_str(&manifest_text) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("ERROR: cannot parse MANIFEST.toml: {e}");
            std::process::exit(1);
        }
    };

    let papers: Vec<&ManifestEntry> = if let Some(ref only_id) = args.only {
        manifest
            .paper
            .iter()
            .filter(|p| p.id == *only_id)
            .collect()
    } else {
        manifest.paper.iter().collect()
    };

    if papers.is_empty() {
        eprintln!("No papers to extract.");
        std::process::exit(1);
    }

    println!("Extracting {} papers...", papers.len());

    let mut success = 0;
    let mut failed = 0;

    for entry in &papers {
        let output_dir = args.output.join(&entry.id);
        let toml_path = output_dir.join("paper.toml");

        if args.skip_existing && toml_path.exists() {
            println!("  SKIP  {} (already extracted)", entry.id);
            success += 1;
            continue;
        }

        let pdf_path = Path::new(&entry.local_pdf);
        if !pdf_path.exists() {
            eprintln!("  MISS  {} (PDF not found: {})", entry.id, entry.local_pdf);
            failed += 1;
            continue;
        }

        // Build metadata override from MANIFEST.toml
        let arxiv = entry
            .arxiv
            .as_deref()
            .filter(|s| !s.is_empty())
            .map(String::from);

        let metadata = docpipe::PaperMetadata {
            title: entry.title.clone(),
            authors: entry.authors.clone(),
            arxiv,
            year: entry.year,
            abstract_text: None, // will be extracted from PDF
        };

        match docpipe::toml_out::extract_and_write(pdf_path, &output_dir, Some(metadata)) {
            Ok(paper) => {
                let n_sections = paper.sections.len();
                let n_equations = paper.equations.len();
                let n_tables = paper.tables.len();
                let n_figures = paper.figures.len();
                let text_kb = paper.full_text.len() / 1024;
                println!(
                    "  OK    {} ({text_kb}KB, {n_sections} sections, \
                     {n_equations} eqs, {n_tables} tables, {n_figures} figs)",
                    entry.id
                );
                success += 1;
            }
            Err(e) => {
                eprintln!("  FAIL  {} ({e})", entry.id);
                failed += 1;
            }
        }
    }

    println!("\nDone: {success} extracted, {failed} failed, {} total", papers.len());

    if failed > 0 {
        std::process::exit(1);
    }
}
