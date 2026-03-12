use anyhow::Result;
use clap::Parser;
use gororoba_cli_data::source_provenance;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "source-truth-infrastructure",
    about = "Project deterministic source-lane registries from the master artifact registry in Rust"
)]
struct Cli {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,

    #[arg(long, default_value = "registry/artifact_source_of_truth.toml")]
    source: PathBuf,

    #[arg(long, default_value = "registry/source_infrastructure.toml")]
    out_infrastructure: PathBuf,

    #[arg(long, default_value = "registry/source_lanes")]
    lane_dir: PathBuf,

    #[arg(
        long,
        default_value = "reports/source_infrastructure_reconciliation_2026_02_15.toml"
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
    let stats = source_provenance::build_source_truth_infrastructure(
        &repo_root,
        &cli.source,
        &cli.out_infrastructure,
        &cli.lane_dir,
        &cli.out_report,
    )?;
    println!(
        "Wrote source infrastructure manifest: {}",
        cli.out_infrastructure.display()
    );
    for (lane, count) in &stats.lane_counts {
        println!("Wrote lane: registry/source_lanes/{lane}.toml artifacts={count}");
    }
    println!(
        "Wrote source infrastructure reconciliation report: {}",
        cli.out_report.display()
    );
    Ok(())
}
