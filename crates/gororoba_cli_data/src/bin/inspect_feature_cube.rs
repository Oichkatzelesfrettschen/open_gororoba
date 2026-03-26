use anyhow::{Context, Result};
use clap::Parser;
use polars::prelude::*;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "inspect-feature-cube", about = "Inspect the heliosphere feature cube (Rust replacement for inspect_cube.py)")]
struct Cli {
    #[arg(long, default_value = "data/output/heliosphere/full_feature_cube.csv")]
    input: PathBuf,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(cli.input))?
        .finish()
        .context("Failed to read feature cube CSV")?;

    let cols = vec![
        "density_cm3",
        "speed_kms",
        "temperature_k",
        "b_mag",
        "crs_flux",
        "spectral_peak",
    ];

    println!("--- Counts by (mission, product) ---");
    let counts = df
        .clone()
        .group_by(["mission", "product"])?
        .select(&cols)
        .count()?
        .sort(["mission", "product"], Default::default())?;
    println!("{}", counts);

    println!("\n--- Mean values by (mission, product) ---");
    let means = df
        .group_by(["mission", "product"])?
        .select(&cols)
        .mean()?
        .sort(["mission", "product"], Default::default())?;
    println!("{}", means);

    Ok(())
}
