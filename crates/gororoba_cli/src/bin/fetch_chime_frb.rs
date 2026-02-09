//! fetch-chime-frb: Download CHIME/FRB catalog CSV files.
//!
//! Downloads the CHIME/FRB Catalog 1 (536 events, Amiri+ 2021) and/or
//! Catalog 2 (4539 events, CHIME/FRB Collaboration 2025) from the
//! official CHIME/FRB Open Data portal.
//!
//! The CHIME website is a JavaScript SPA, so this fetcher uses the
//! known API endpoints for CSV export. If those fail, the user should
//! download manually from https://www.chime-frb.ca/catalog and place
//! the CSV in data/external/.
//!
//! Usage:
//!   fetch-chime-frb --catalog both --output-dir data/external/

use clap::Parser;
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "fetch-chime-frb")]
#[command(about = "Download CHIME/FRB catalog CSV data for ultrametric analysis")]
struct Args {
    /// Which catalog to download: 1, 2, or both
    #[arg(long, default_value = "both")]
    catalog: String,

    /// Output directory for CSV files
    #[arg(long, default_value = "data/external")]
    output_dir: PathBuf,

    /// Skip download if file already exists
    #[arg(long)]
    skip_existing: bool,
}

/// Known CHIME/FRB catalog download URLs.
///
/// These point to the official CHIME/FRB data portal's CSV export endpoints.
/// The CHIME website is JavaScript-based, so these URLs may require a browser
/// session or cookie. If they fail, fallback to manual download instructions.
const CATALOG1_URLS: &[&str] = &[
    // Primary: Google Cloud Storage (stable direct download)
    "https://storage.googleapis.com/chimefrb-dev.appspot.com/catalog1/chimefrbcat1.csv",
    // Fallback: CHIME/FRB website CSV export (requires JS, unlikely to work)
    "https://www.chime-frb.ca/catalog/csv",
];

const CATALOG2_URLS: &[&str] = &[
    // Primary: Google Cloud Storage (stable direct download)
    "https://storage.googleapis.com/chimefrb-dev.appspot.com/catalog2/chimefrbcat2.csv",
    // Fallback: CHIME/FRB website
    "https://www.chime-frb.ca/catalog2/csv",
];

fn main() {
    let args = Args::parse();

    // Ensure output directory exists
    fs::create_dir_all(&args.output_dir).expect("Failed to create output directory");

    let download_cat1 = args.catalog == "1" || args.catalog == "both";
    let download_cat2 = args.catalog == "2" || args.catalog == "both";

    if download_cat1 {
        let path = args.output_dir.join("chime_frb_cat1.csv");
        if args.skip_existing && path.exists() {
            eprintln!("Catalog 1 already exists at {}, skipping", path.display());
            print_checksum(&path);
        } else {
            download_catalog("Catalog 1", CATALOG1_URLS, &path);
        }
    }

    if download_cat2 {
        let path = args.output_dir.join("chime_frb_cat2.csv");
        if args.skip_existing && path.exists() {
            eprintln!("Catalog 2 already exists at {}, skipping", path.display());
            print_checksum(&path);
        } else {
            download_catalog("Catalog 2", CATALOG2_URLS, &path);
        }
    }
}

fn download_catalog(name: &str, urls: &[&str], output_path: &Path) {
    eprintln!("Downloading CHIME/FRB {}...", name);

    for url in urls {
        eprintln!("  Trying: {}", url);

        match attempt_download(url) {
            Ok(data) => {
                // Validate: check it looks like a CSV with expected columns
                if validate_csv_data(&data) {
                    fs::write(output_path, &data).expect("Failed to write CSV");
                    eprintln!("  Saved to: {}", output_path.display());
                    print_checksum(output_path);

                    // Count rows
                    let n_rows = data.lines().count().saturating_sub(1);
                    eprintln!("  Events: {}", n_rows);
                    return;
                } else {
                    eprintln!("  Response is not valid CSV data (possibly HTML/JS page)");
                }
            }
            Err(e) => {
                eprintln!("  Failed: {}", e);
            }
        }
    }

    // All URLs failed -- print manual download instructions
    eprintln!();
    eprintln!("ERROR: Could not download {} automatically.", name);
    eprintln!("The CHIME/FRB website requires JavaScript, which prevents direct HTTP downloads.");
    eprintln!();
    eprintln!("Manual download instructions:");
    eprintln!("  1. Visit https://www.chime-frb.ca/catalog in a browser");
    eprintln!("  2. Click 'Download CSV' or 'Export' button");
    eprintln!("  3. Save the file to: {}", output_path.display());
    eprintln!();
    eprintln!("Alternatively, use the cfod Python package:");
    eprintln!("  pip install cfod");
    eprintln!(
        "  python -c \"from cfod.catalog1 import catalog; catalog.as_dataframe().to_csv('{}')\"",
        output_path.display()
    );
    eprintln!();
    std::process::exit(1);
}

fn attempt_download(url: &str) -> Result<String, String> {
    let response = ureq::get(url)
        .header("Accept", "text/csv,text/plain,*/*")
        .header("User-Agent", "gororoba-fetch/0.1 (research)")
        .call()
        .map_err(|e| format!("Request failed: {}", e))?;

    let status = response.status();
    if status != 200 {
        return Err(format!("HTTP {}", status));
    }

    let body = response
        .into_body()
        .read_to_string()
        .map_err(|e| format!("Read failed: {}", e))?;

    Ok(body)
}

/// Check if the response looks like valid CHIME FRB CSV data.
fn validate_csv_data(data: &str) -> bool {
    let first_line = match data.lines().next() {
        Some(line) => line.to_lowercase(),
        None => return false,
    };

    // Check for expected column headers (case-insensitive)
    // CHIME Cat 1 has columns like: tns_name, bonsai_dm, dm_exc_ne2001, etc.
    // We accept the data if it has at least one DM-related column.
    let dm_indicators = ["dm", "dispersion", "bonsai_dm", "dm_fitb", "dm_exc"];
    let has_dm_column = dm_indicators
        .iter()
        .any(|&indicator| first_line.contains(indicator));

    // Reject if it looks like HTML
    let is_html = first_line.contains("<!doctype")
        || first_line.contains("<html")
        || first_line.contains("<script");

    has_dm_column && !is_html
}

fn print_checksum(path: &Path) {
    if let Ok(data) = fs::read(path) {
        let mut hasher = Sha256::new();
        hasher.update(&data);
        let hash = hasher.finalize();
        eprintln!("  SHA256: {:x}", hash);
    }
}
