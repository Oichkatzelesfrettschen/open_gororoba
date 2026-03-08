use anyhow::{Context, Result};
use clap::{ArgAction, Parser};
use data_core::fetcher::download_to_string;
use serde_json::Value;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(
    name = "hepdata-refresh",
    about = "Re-download HEPData CSV tables using table DOI metadata embedded in local CSV headers"
)]
struct Args {
    #[arg(long, default_value = "data/external")]
    root: PathBuf,
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "alice_pbpb_raa,cms_oo_raa"
    )]
    dirs: Vec<String>,
    #[arg(
        long,
        default_value_t = false,
        action = ArgAction::Set,
        num_args = 0..=1,
        default_missing_value = "true"
    )]
    skip_existing: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut files = Vec::new();
    for dir in &args.dirs {
        let lane_root = args.root.join(dir);
        collect_csv_files(&lane_root, &mut files)
            .with_context(|| format!("collect CSV files under {}", lane_root.display()))?;
    }
    files.sort();

    if files.is_empty() {
        anyhow::bail!("no CSV files found for requested directories");
    }

    let mut refreshed = 0usize;
    let mut skipped = 0usize;
    for file in &files {
        if args.skip_existing && file.exists() {
            eprintln!("SKIP {}", file.display());
            skipped += 1;
            continue;
        }

        let doi = extract_table_doi(file)
            .with_context(|| format!("extract table DOI from {}", file.display()))?;
        let csv_url =
            resolve_csv_url_for_doi(&doi).with_context(|| format!("resolve CSV URL for {doi}"))?;
        let body =
            download_to_string(&csv_url).with_context(|| format!("download CSV from {csv_url}"))?;
        fs::write(file, body + "\n")
            .with_context(|| format!("write refreshed CSV {}", file.display()))?;
        eprintln!("OK {} <= {}", file.display(), csv_url);
        refreshed += 1;
    }

    eprintln!(
        "HEPDATA_REFRESH complete: refreshed={}, skipped={}, total={}",
        refreshed,
        skipped,
        files.len()
    );
    Ok(())
}

fn collect_csv_files(root: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
    if !root.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(root).with_context(|| format!("read_dir {}", root.display()))? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_csv_files(&path, out)?;
        } else if path
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.eq_ignore_ascii_case("csv"))
            .unwrap_or(false)
        {
            out.push(path);
        }
    }
    Ok(())
}

fn extract_table_doi(path: &Path) -> Result<String> {
    let text = fs::read_to_string(path)?;
    for line in text.lines().take(32) {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("#: table_doi:") {
            let doi = rest.trim();
            if doi.is_empty() {
                anyhow::bail!("empty table DOI in header");
            }
            return Ok(doi.to_string());
        }
    }
    anyhow::bail!("missing '#: table_doi:' header");
}

fn resolve_csv_url_for_doi(doi: &str) -> Result<String> {
    let url = format!("https://doi.org/{doi}");
    let body = download_to_string(&url).with_context(|| format!("resolve DOI {doi}"))?;
    let json: Value =
        serde_json::from_str(&body).with_context(|| format!("parse DOI JSON {doi}"))?;
    let distributions = json
        .get("distribution")
        .and_then(Value::as_array)
        .with_context(|| format!("DOI response missing distribution array: {doi}"))?;
    for item in distributions {
        let encoding = item
            .get("encodingFormat")
            .and_then(Value::as_str)
            .unwrap_or("");
        let desc = item
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_ascii_lowercase();
        let content_url = item.get("contentUrl").and_then(Value::as_str).unwrap_or("");
        if content_url.starts_with("https://")
            && (encoding.eq_ignore_ascii_case("text/csv") || desc.contains("csv"))
        {
            return Ok(content_url.to_string());
        }
    }
    anyhow::bail!("DOI distribution did not include a CSV contentUrl for {doi}");
}
