use anyhow::Result;
use clap::Parser;
use gororoba_cli_data::source_provenance;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "artifact-source-of-truth",
    about = "Build the canonical artifact source-of-truth registry in Rust"
)]
struct Cli {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    #[arg(long, default_value = "registry/artifact_source_of_truth.toml")]
    out_registry: PathBuf,

    #[arg(
        long,
        default_value = "reports/artifact_source_of_truth_reconciliation_2026_02_15.toml"
    )]
    out_report: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let repo_root = if cli.repo_root == PathBuf::from(".") {
        source_provenance::default_repo_root()
    } else {
        cli.repo_root
    };
    let stats = source_provenance::build_artifact_source_of_truth(
        &repo_root,
        &cli.out_registry,
        &cli.out_report,
    )?;
    println!(
        "Wrote artifact source-of-truth registry: {} artifacts={}",
        cli.out_registry.display(),
        stats.artifact_count
    );
    println!("Wrote reconciliation report: {}", cli.out_report.display());
    Ok(())
}
