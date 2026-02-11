//! Batch PDF extraction: reads papers/MANIFEST.toml, extracts each paper
//! to structured TOML + equations.tex + table CSVs using the docpipe crate.

use std::path::{Path, PathBuf};

use clap::Parser;
use docpipe::equation_catalog::{
    build_catalog, convert_historical_csv_path, index_module_files, write_catalog_toml,
    EquationCatalogInputRow, EquationSourceStream,
};

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

    /// Build an equation catalog TOML artifact as part of extraction.
    #[arg(long, default_value_t = false)]
    build_equation_catalog: bool,

    /// Catalog TOML output path (relative paths are resolved under `--output`).
    #[arg(long, default_value = "equation_catalog.toml")]
    equation_catalog_output: PathBuf,

    /// Optional historical text-stream catalog CSV to merge.
    #[arg(long)]
    equation_catalog_text_csv: Option<PathBuf>,

    /// Optional historical PDF text-stream catalog CSV to merge.
    #[arg(long)]
    equation_catalog_pdf_csv: Option<PathBuf>,

    /// Optional historical PDF OCR-stream catalog CSV to merge.
    #[arg(long)]
    equation_catalog_ocr_csv: Option<PathBuf>,

    /// Directory of equation module `.tex` files for parity reporting.
    #[arg(long, default_value = "synthesis/modules/equations")]
    equation_modules_dir: PathBuf,
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

fn infer_framework(entry: &ManifestEntry) -> String {
    let joined = format!(
        "{} {}",
        entry.id.to_ascii_lowercase(),
        entry.title.to_ascii_lowercase()
    );
    if joined.contains("aether") {
        "Aether".to_string()
    } else if joined.contains("genesis") {
        "Genesis".to_string()
    } else if joined.contains("pais") {
        "Pais".to_string()
    } else if joined.contains("tourmaline") {
        "Tourmaline".to_string()
    } else if joined.contains("superforce") {
        "Superforce".to_string()
    } else if joined.contains("unified") {
        "Unified".to_string()
    } else {
        "Literature".to_string()
    }
}

fn catalog_rows_from_paper(
    entry: &ManifestEntry,
    paper: &docpipe::ExtractedPaper,
) -> Vec<EquationCatalogInputRow> {
    let framework = infer_framework(entry);
    paper
        .equations
        .iter()
        .enumerate()
        .map(|(idx, equation)| {
            let mut row = EquationCatalogInputRow::new(
                equation.latex.clone(),
                framework.clone(),
                EquationSourceStream::PdfText,
            );
            row.source_doc = entry.local_pdf.clone();
            row.source_line = format!("equation:{:03}", idx + 1);
            row.description = format!("Extracted by docpipe from paper {}", entry.id);
            row.verification_status = "Theoretical".to_string();
            row
        })
        .collect()
}

fn load_existing_extracted_paper(path: &Path) -> Option<docpipe::ExtractedPaper> {
    let toml_text = std::fs::read_to_string(path).ok()?;
    toml::from_str(&toml_text).ok()
}

fn append_historical_rows(
    target: &mut Vec<EquationCatalogInputRow>,
    csv_path: Option<&PathBuf>,
    stream: EquationSourceStream,
    had_errors: &mut bool,
    label: &str,
) {
    let Some(path) = csv_path else {
        return;
    };

    match convert_historical_csv_path(path, stream) {
        Ok(mut rows) => {
            let count = rows.len();
            target.append(&mut rows);
            println!("Loaded {count} rows from {label}: {}", path.display());
        }
        Err(err) => {
            eprintln!(
                "ERROR: failed loading {label} CSV {}: {err}",
                path.display()
            );
            *had_errors = true;
        }
    }
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
        manifest.paper.iter().filter(|p| p.id == *only_id).collect()
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
    let mut catalog_had_errors = false;
    let mut catalog_inputs: Vec<EquationCatalogInputRow> = Vec::new();

    for entry in &papers {
        let output_dir = args.output.join(&entry.id);
        let toml_path = output_dir.join("paper.toml");

        if args.skip_existing && toml_path.exists() {
            println!("  SKIP  {} (already extracted)", entry.id);
            if args.build_equation_catalog {
                if let Some(existing_paper) = load_existing_extracted_paper(&toml_path) {
                    catalog_inputs.extend(catalog_rows_from_paper(entry, &existing_paper));
                } else {
                    eprintln!(
                        "  WARN  {} (could not parse existing {})",
                        entry.id,
                        toml_path.display()
                    );
                    catalog_had_errors = true;
                }
            }
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
                if args.build_equation_catalog {
                    catalog_inputs.extend(catalog_rows_from_paper(entry, &paper));
                }
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

    if args.build_equation_catalog {
        append_historical_rows(
            &mut catalog_inputs,
            args.equation_catalog_text_csv.as_ref(),
            EquationSourceStream::Text,
            &mut catalog_had_errors,
            "text stream",
        );
        append_historical_rows(
            &mut catalog_inputs,
            args.equation_catalog_pdf_csv.as_ref(),
            EquationSourceStream::PdfText,
            &mut catalog_had_errors,
            "pdf text stream",
        );
        append_historical_rows(
            &mut catalog_inputs,
            args.equation_catalog_ocr_csv.as_ref(),
            EquationSourceStream::PdfOcr,
            &mut catalog_had_errors,
            "pdf ocr stream",
        );

        let known_modules = match index_module_files(&args.equation_modules_dir) {
            Ok(modules) => modules,
            Err(err) => {
                eprintln!(
                    "ERROR: failed indexing module directory {}: {err}",
                    args.equation_modules_dir.display()
                );
                catalog_had_errors = true;
                Vec::new()
            }
        };

        let catalog = build_catalog(catalog_inputs, &known_modules);
        let catalog_output = if args.equation_catalog_output.is_absolute() {
            args.equation_catalog_output.clone()
        } else {
            args.output.join(&args.equation_catalog_output)
        };
        match write_catalog_toml(&catalog_output, &catalog) {
            Ok(()) => {
                println!(
                    "Equation catalog written: {} (rows: {}, missing module links: {})",
                    catalog_output.display(),
                    catalog.rows.len(),
                    catalog.gap_report.total_missing_module_links
                );
            }
            Err(err) => {
                eprintln!(
                    "ERROR: failed writing equation catalog {}: {err}",
                    catalog_output.display()
                );
                catalog_had_errors = true;
            }
        }
    }

    println!(
        "\nDone: {success} extracted, {failed} failed, {} total",
        papers.len()
    );

    if failed > 0 || catalog_had_errors {
        std::process::exit(1);
    }
}
