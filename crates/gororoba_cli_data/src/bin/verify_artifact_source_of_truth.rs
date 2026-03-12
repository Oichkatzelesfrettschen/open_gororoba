use anyhow::Result;
use clap::Parser;
use gororoba_cli_data::source_provenance;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "verify-artifact-source-of-truth",
    about = "Verify the canonical artifact source-of-truth registry in Rust"
)]
struct Cli {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    #[arg(long, default_value = "registry/artifact_source_of_truth.toml")]
    registry: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let repo_root = if cli.repo_root == Path::new(".") {
        source_provenance::default_repo_root()
    } else {
        cli.repo_root
    };
    let stats = source_provenance::verify_artifact_source_of_truth(&repo_root, &cli.registry)?;
    println!(
        "OK: artifact source-of-truth verified. artifacts={} downloaded={} downloadable={} blocked={} citation_only={} unverified={} missing_minimum={}",
        stats.artifact_count,
        stats.downloaded_count,
        stats.downloadable_count,
        stats.blocked_count,
        stats.citation_only_count,
        stats.unverified_count,
        stats.missing_minimum_count
    );
    Ok(())
}
