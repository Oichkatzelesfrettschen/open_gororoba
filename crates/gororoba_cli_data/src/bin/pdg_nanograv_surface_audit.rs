//! Surface audit for the PDG neutrino and NANOGrav 15-year source lanes.
//!
//! This binary does not fetch from the network. It records the current official
//! source surfaces, checks local repo coverage, and repairs the checked-in
//! NANOGrav free-spectrum CSV when it has drifted to the all-NaN placeholder.
//!
//! The report is Rust-first: current parser/provider/audit surfaces are treated
//! as authoritative, while any old Python fetch/test paths are only migration
//! debt markers.

use clap::Parser;
use csv::ReaderBuilder;
use data_core::{
    catalogs::nanograv::{bestfit, write_free_spectrum_csv},
    parse_nanograv_free_spectrum,
};
use std::{
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
    process,
};

const PDG_2024_HIGHLIGHTS_URL: &str = "https://pdg.lbl.gov/2024/reviews/rpp2024-rev-highlights.pdf";
const NANOGRAV_15YR_BACKGROUND_URL: &str = "https://nanograv.org/15yr/SMBHB";
const NANOGRAV_ZENODO_URL: &str =
    "https://zenodo.org/api/records/10344086/files/NANOGrav15yr_KDE-FreeSpectra_v1.1.0.zip/content";

#[derive(Parser, Debug)]
#[command(name = "pdg-nanograv-surface-audit")]
#[command(about = "Audit local PDG/NANOGrav source surfaces and repair NANOGrav CSV drift")]
struct Args {
    /// Output TOML report path.
    #[arg(long, default_value = "reports/pdg_nanograv_surface_audit.toml")]
    output: PathBuf,

    /// Rewrite the checked-in NANOGrav CSV from the embedded best-fit table when invalid.
    #[arg(long, default_value_t = true)]
    repair_nanograv_csv: bool,
}

#[derive(Debug, Clone)]
struct NanogravCsvStatus {
    row_count: usize,
    invalid_value_count: usize,
    interval_violation_count: usize,
    matches_bestfit_count: usize,
    fully_matches_bestfit: bool,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../")
}

fn repo_path(relative: &str) -> PathBuf {
    repo_root().join(relative)
}

fn parse_nanograv_or_exit(path: &Path) -> Vec<data_core::catalogs::nanograv::FreeSpectrumPoint> {
    parse_nanograv_free_spectrum(path).unwrap_or_else(|err| {
        eprintln!("ERROR: failed to parse {}: {err}", path.display());
        process::exit(1);
    })
}

fn summarize_nanograv_csv(path: &Path) -> NanogravCsvStatus {
    let rows = parse_nanograv_or_exit(path);
    let mut invalid_value_count = 0usize;
    let mut interval_violation_count = 0usize;
    let mut matches_bestfit_count = 0usize;

    for (row, expected) in rows.iter().zip(bestfit::HD_FREE_SPECTRUM.iter()) {
        let values = [row.log10_rho, row.log10_rho_lo, row.log10_rho_hi];
        invalid_value_count += values.iter().filter(|value| !value.is_finite()).count();

        if row.log10_rho_lo.is_finite()
            && row.log10_rho.is_finite()
            && row.log10_rho_hi.is_finite()
            && !(row.log10_rho_lo <= row.log10_rho && row.log10_rho <= row.log10_rho_hi)
        {
            interval_violation_count += 1;
        }

        let frequency_match = (row.frequency - expected.frequency).abs() < 1e-18;
        let median_match = (row.log10_rho - expected.log10_rho).abs() < 1e-6;
        let lo_match = (row.log10_rho_lo - expected.log10_rho_lo).abs() < 1e-6;
        let hi_match = (row.log10_rho_hi - expected.log10_rho_hi).abs() < 1e-6;
        if frequency_match && median_match && lo_match && hi_match {
            matches_bestfit_count += 1;
        }
    }

    NanogravCsvStatus {
        row_count: rows.len(),
        invalid_value_count,
        interval_violation_count,
        matches_bestfit_count,
        fully_matches_bestfit: rows.len() == bestfit::N_BINS && matches_bestfit_count == rows.len(),
    }
}

fn csv_shape_ok(path: &Path) -> Result<usize, String> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|err| format!("open {}: {err}", path.display()))?;
    let mut rows = 0usize;
    for result in reader.records() {
        result.map_err(|err| format!("parse {}: {err}", path.display()))?;
        rows += 1;
    }
    Ok(rows)
}

fn render_bool(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

fn write_or_exit(path: &Path, text: &str) {
    if let Some(parent) = path.parent()
        && let Err(err) = fs::create_dir_all(parent)
    {
        eprintln!("ERROR: failed to create {}: {err}", parent.display());
        process::exit(1);
    }
    if let Err(err) = fs::write(path, text) {
        eprintln!("ERROR: failed to write {}: {err}", path.display());
        process::exit(1);
    }
}

fn main() {
    let args = Args::parse();

    let nanograv_csv_path = repo_path("data/external/nanograv_15yr_freespectrum.csv");
    let nanograv_paper_path = repo_path(
        "data/papers/documents_extracted/arxiv-2306-16213-agazie-et-al-2023-nanograv-15yr-gwb/paper.toml",
    );
    let nanograv_table_path = repo_path(
        "data/papers/documents_extracted/arxiv-2306-16213-agazie-et-al-2023-nanograv-15yr-gwb/table_1.csv",
    );
    let c070_doc_path = repo_path("docs/external_sources/C070_NANOGRAV_SPECTRUM_MATCH_SOURCES.md");
    let legacy_neutrino_fetch_path = repo_path("src/scripts/data/fetch_neutrino_params.py");
    let legacy_neutrino_test_path = repo_path("tests/test_neutrino_params.py");
    let rust_pmns_surface_path = repo_path("crates/stats_core/src/lib.rs");
    let rust_particle_audit_path =
        repo_path("crates/gororoba_cli_data/src/bin/particle_numerology_audit.rs");
    let rust_nanograv_surface_path = repo_path("crates/data_core/src/catalogs/nanograv.rs");

    let before = summarize_nanograv_csv(&nanograv_csv_path);
    let mut repaired = false;
    if args.repair_nanograv_csv && !before.fully_matches_bestfit {
        write_free_spectrum_csv(&bestfit::HD_FREE_SPECTRUM, &nanograv_csv_path).unwrap_or_else(
            |err| {
                eprintln!(
                    "ERROR: failed to rewrite {} from embedded best-fit table: {err}",
                    nanograv_csv_path.display()
                );
                process::exit(1);
            },
        );
        repaired = true;
    }
    let after = summarize_nanograv_csv(&nanograv_csv_path);

    let extracted_table_rows = match csv_shape_ok(&nanograv_table_path) {
        Ok(rows) => rows.to_string(),
        Err(err) => format!("\"{err}\""),
    };

    let mut out = String::new();
    let _ = writeln!(out, "[metadata]");
    let _ = writeln!(
        out,
        "title = \"PDG neutrino and NANOGrav surface audit with local NANOGrav CSV repair\""
    );
    let _ = writeln!(
        out,
        "repair_nanograv_csv = {}",
        render_bool(args.repair_nanograv_csv)
    );
    let _ = writeln!(out, "repaired_nanograv_csv = {}", render_bool(repaired));
    let _ = writeln!(out);

    let _ = writeln!(out, "[official_sources]");
    let _ = writeln!(out, "pdg_2024_highlights = \"{PDG_2024_HIGHLIGHTS_URL}\"");
    let _ = writeln!(
        out,
        "nanograv_15yr_background = \"{NANOGRAV_15YR_BACKGROUND_URL}\""
    );
    let _ = writeln!(out, "nanograv_kde_zip = \"{NANOGRAV_ZENODO_URL}\"");
    let _ = writeln!(out);

    let _ = writeln!(out, "[nanograv.local_surfaces]");
    let _ = writeln!(
        out,
        "csv_path = \"{}\"",
        nanograv_csv_path
            .strip_prefix(repo_root())
            .unwrap_or(&nanograv_csv_path)
            .display()
    );
    let _ = writeln!(
        out,
        "paper_extract_path = \"{}\"",
        nanograv_paper_path
            .strip_prefix(repo_root())
            .unwrap_or(&nanograv_paper_path)
            .display()
    );
    let _ = writeln!(
        out,
        "table_extract_path = \"{}\"",
        nanograv_table_path
            .strip_prefix(repo_root())
            .unwrap_or(&nanograv_table_path)
            .display()
    );
    let _ = writeln!(
        out,
        "c070_source_doc_present = {}",
        render_bool(c070_doc_path.exists())
    );
    let _ = writeln!(
        out,
        "paper_extract_present = {}",
        render_bool(nanograv_paper_path.exists())
    );
    let _ = writeln!(
        out,
        "table_extract_present = {}",
        render_bool(nanograv_table_path.exists())
    );
    let _ = writeln!(out, "table_extract_csv_status = {}", extracted_table_rows);
    let _ = writeln!(out);

    let _ = writeln!(out, "[nanograv.csv_before]");
    let _ = writeln!(out, "row_count = {}", before.row_count);
    let _ = writeln!(out, "invalid_value_count = {}", before.invalid_value_count);
    let _ = writeln!(
        out,
        "interval_violation_count = {}",
        before.interval_violation_count
    );
    let _ = writeln!(
        out,
        "matches_bestfit_count = {}",
        before.matches_bestfit_count
    );
    let _ = writeln!(
        out,
        "fully_matches_bestfit = {}",
        render_bool(before.fully_matches_bestfit)
    );
    let _ = writeln!(out);

    let _ = writeln!(out, "[nanograv.csv_after]");
    let _ = writeln!(out, "row_count = {}", after.row_count);
    let _ = writeln!(out, "invalid_value_count = {}", after.invalid_value_count);
    let _ = writeln!(
        out,
        "interval_violation_count = {}",
        after.interval_violation_count
    );
    let _ = writeln!(
        out,
        "matches_bestfit_count = {}",
        after.matches_bestfit_count
    );
    let _ = writeln!(
        out,
        "fully_matches_bestfit = {}",
        render_bool(after.fully_matches_bestfit)
    );
    let _ = writeln!(out);

    let _ = writeln!(out, "[pdg_neutrino.local_surfaces]");
    let _ = writeln!(
        out,
        "legacy_fetch_script_present = {}",
        render_bool(legacy_neutrino_fetch_path.exists())
    );
    let _ = writeln!(
        out,
        "legacy_test_present = {}",
        render_bool(legacy_neutrino_test_path.exists())
    );
    let _ = writeln!(
        out,
        "rust_pmns_surface_path = \"crates/stats_core/src/lib.rs\""
    );
    let _ = writeln!(
        out,
        "rust_pmns_surface_present = {}",
        render_bool(rust_pmns_surface_path.exists())
    );
    let _ = writeln!(
        out,
        "rust_particle_audit_path = \"crates/gororoba_cli_data/src/bin/particle_numerology_audit.rs\""
    );
    let _ = writeln!(
        out,
        "rust_particle_audit_present = {}",
        render_bool(rust_particle_audit_path.exists())
    );
    let _ = writeln!(
        out,
        "rust_nanograv_surface_path = \"crates/data_core/src/catalogs/nanograv.rs\""
    );
    let _ = writeln!(
        out,
        "rust_nanograv_surface_present = {}",
        render_bool(rust_nanograv_surface_path.exists())
    );
    let _ = writeln!(out, "claim_surface = \"registry/claims.toml\"");

    write_or_exit(&args.output, &out);
}
