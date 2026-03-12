use anyhow::{Context, Result};
use clap::Parser;
use materials_core::write_c053_summary;
use std::path::PathBuf;

const DEFAULT_OUTPUT: &str = "data/csv/c053_pathion_tmm_summary.csv";

#[derive(Parser, Debug)]
#[command(
    name = "c053-pathion-metamaterial-mapping",
    about = "Write the deterministic C-053 diagonal-only pathion toy mapping CSV"
)]
struct Args {
    #[arg(long, default_value = DEFAULT_OUTPUT)]
    output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();
    write_c053_summary(&args.output).map_err(anyhow::Error::msg)?;
    println!(
        "{}",
        args.output
            .canonicalize()
            .with_context(|| format!("canonicalizing {}", args.output.display()))?
            .display()
    );
    Ok(())
}
