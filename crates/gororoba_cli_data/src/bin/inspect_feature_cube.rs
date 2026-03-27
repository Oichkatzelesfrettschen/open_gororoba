use anyhow::{Context, Result};
use clap::Parser;
use polars::prelude::*;
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "inspect-feature-cube",
    about = "Inspect the heliosphere feature cube (Rust replacement for inspect_cube.py)"
)]
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
        .select(cols.iter().copied())
        .count()?
        .sort(["mission", "product"], Default::default())?;
    println!("{}", counts);

    println!("\n--- Mean values by (mission, product) ---");
    let mission = df
        .column("mission")?
        .str()
        .context("mission column is not string-typed")?;
    let product = df
        .column("product")?
        .str()
        .context("product column is not string-typed")?;

    let numeric_columns = cols
        .iter()
        .map(|name| {
            let col = df.column(name)?;
            let casted = col
                .cast(&DataType::Float64)
                .with_context(|| format!("failed to cast {} to Float64", name))?;
            let ca = casted
                .f64()
                .with_context(|| format!("{} is not numeric after cast", name))?;
            Ok::<_, anyhow::Error>(((*name).to_string(), ca.into_iter().collect::<Vec<_>>()))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut groups: BTreeMap<(String, String), (usize, Vec<f64>)> = BTreeMap::new();
    for row_idx in 0..df.height() {
        let Some(mission_value) = mission.get(row_idx) else {
            continue;
        };
        let Some(product_value) = product.get(row_idx) else {
            continue;
        };
        let entry = groups
            .entry((mission_value.to_string(), product_value.to_string()))
            .or_insert_with(|| (0usize, vec![0.0; cols.len()]));
        entry.0 += 1;
        for (col_idx, (_, values)) in numeric_columns.iter().enumerate() {
            if let Some(value) = values[row_idx] {
                entry.1[col_idx] += value;
            }
        }
    }

    println!("{:<20} {:<24} {}", "mission", "product", cols.join(" | "));
    for ((mission_value, product_value), (count, sums)) in groups {
        let means = sums
            .into_iter()
            .map(|sum| format!("{:.6}", sum / count as f64))
            .collect::<Vec<_>>()
            .join(" | ");
        println!("{:<20} {:<24} {}", mission_value, product_value, means);
    }

    Ok(())
}
