use anyhow::{Context, Result};
use clap::Parser;
use gororoba_cli::data_governance::{
    DEFAULT_EXTERNAL_SOURCES_PATH, load_external_sources, sha256_file, source_rule_for_path,
};
use serde::Serialize;
use std::path::PathBuf;
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(
    name = "record-external-hashes",
    about = "Record SHA-256 hashes for data/external into PROVENANCE.local.json"
)]
struct Args {
    #[arg(long, default_value = "data/external")]
    root: PathBuf,
    #[arg(long, default_value = "data/external/PROVENANCE.local.json")]
    output: PathBuf,
    #[arg(long, default_value = DEFAULT_EXTERNAL_SOURCES_PATH)]
    sources: PathBuf,
}

#[derive(Debug, Serialize)]
struct HashRow {
    path: String,
    size_bytes: u64,
    mtime_utc: String,
    sha256: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_access_class: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_canonical_url: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_retrieval_method: Option<String>,
}

#[derive(Debug, Serialize)]
struct ProvenanceDoc {
    generated_at_utc: String,
    root: String,
    hash_backend: String,
    hashes: Vec<HashRow>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if !args.root.exists() {
        anyhow::bail!("external root does not exist: {}", args.root.display());
    }
    let sources = load_external_sources(&args.sources)?;
    let mut rows = Vec::new();
    for entry in WalkDir::new(&args.root)
        .into_iter()
        .filter_map(std::result::Result::ok)
        .filter(|e| e.file_type().is_file())
    {
        let path = entry.path();
        let rel = path
            .strip_prefix(&args.root)
            .with_context(|| format!("strip prefix for {}", path.display()))?
            .to_string_lossy()
            .replace('\\', "/");
        let metadata = path
            .metadata()
            .with_context(|| format!("metadata {}", path.display()))?;
        let mtime = metadata
            .modified()
            .ok()
            .map(|t| {
                chrono::DateTime::<chrono::Utc>::from(t)
                    .to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
            })
            .unwrap_or_else(|| "unknown".to_string());
        let sha = sha256_file(path)?;
        let repo_rel = if rel.starts_with("data/external/") {
            rel.clone()
        } else {
            format!("data/external/{rel}")
        };
        let source = source_rule_for_path(&repo_rel, &sources);
        rows.push(HashRow {
            path: rel,
            size_bytes: metadata.len(),
            mtime_utc: mtime,
            sha256: sha,
            source_id: source.map(|s| s.id.clone()),
            source_status: source.map(|s| s.status.clone()),
            source_access_class: source.map(|s| s.access_class.clone()),
            source_canonical_url: source.map(|s| s.canonical_url.clone()),
            source_retrieval_method: source.map(|s| s.retrieval_method.clone()),
        });
    }
    rows.sort_by(|a, b| a.path.cmp(&b.path));
    let payload = ProvenanceDoc {
        generated_at_utc: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
        root: args.root.to_string_lossy().to_string(),
        hash_backend: "rust".to_string(),
        hashes: rows,
    };
    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create output directory {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(&payload).context("serialize provenance JSON")?;
    std::fs::write(&args.output, json + "\n")
        .with_context(|| format!("write {}", args.output.display()))?;
    println!(
        "WROTE {} with {} file hash rows",
        args.output.display(),
        payload.hashes.len()
    );
    Ok(())
}
