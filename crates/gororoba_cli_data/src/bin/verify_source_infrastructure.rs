use anyhow::Result;
use clap::Parser;
use gororoba_cli_data::source_provenance;
use std::path::{Path, PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "verify-source-infrastructure",
    about = "Verify source-lane projections against the master artifact registry in Rust"
)]
struct Cli {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    #[arg(long, default_value = "registry/source_infrastructure.toml")]
    infrastructure: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let repo_root = if cli.repo_root == Path::new(".") {
        source_provenance::default_repo_root()
    } else {
        cli.repo_root
    };
    let stats = source_provenance::verify_source_infrastructure(&repo_root, &cli.infrastructure)?;
    println!(
        "OK: source infrastructure verified. artifacts={} lanes={}",
        stats.total_artifact_count,
        stats.lane_counts.len()
    );
    Ok(())
}
