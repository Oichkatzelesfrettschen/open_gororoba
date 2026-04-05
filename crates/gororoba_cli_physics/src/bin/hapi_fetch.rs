//! Universal CDAWeb HAPI client: catalog, info, download any space physics dataset.
//!
//! HAPI (Heliophysics API) is a REST standard providing uniform access to ALL
//! CDAWeb datasets. This binary replaces ad-hoc wget/curl with reproducible
//! Rust-native downloads.
//!
//! Usage:
//!   hapi-fetch catalog --search voyager    # List matching datasets
//!   hapi-fetch info --dataset VOYAGER1_48S_MAG  # Show parameters + time range
//!   hapi-fetch download --dataset VOYAGER1_48S_MAG --params "F1,B1,B2,B3" \
//!     --start 1979-01-01 --end 1979-12-31 --out data.csv
//!   hapi-fetch download --dataset AC_H0_MFI --params "Magnitude,BGSM" \
//!     --start 2020-01-01 --end 2020-12-31 --yearly --out-dir data/ace/

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

const HAPI_BASE: &str = "https://cdaweb.gsfc.nasa.gov/hapi";

#[derive(Parser)]
#[command(name = "hapi-fetch", about = "Universal CDAWeb HAPI client")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// List datasets matching a search term
    Catalog {
        #[arg(long)]
        search: String,
    },
    /// Show dataset parameters and time range
    Info {
        #[arg(long)]
        dataset: String,
    },
    /// Download data as CSV
    Download {
        #[arg(long)]
        dataset: String,
        /// Comma-separated parameter names (in order listed by `info`)
        #[arg(long)]
        params: String,
        #[arg(long)]
        start: String,
        #[arg(long)]
        end: String,
        /// Output CSV (single file mode)
        #[arg(long)]
        out: Option<PathBuf>,
        /// Output directory (yearly chunk mode)
        #[arg(long)]
        out_dir: Option<PathBuf>,
        /// Split into yearly files
        #[arg(long)]
        yearly: bool,
    },
}

fn hapi_get(url: &str) -> Result<String> {
    let body = ureq::get(url)
        .call()
        .map_err(|e| anyhow::anyhow!("HAPI request failed: {}", e))?
        .into_body()
        .with_config()
        .limit(500 * 1024 * 1024) // 500 MB limit for large datasets
        .read_to_string()
        .map_err(|e| anyhow::anyhow!("HAPI read failed: {}", e))?;
    Ok(body)
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.cmd {
        Cmd::Catalog { search } => {
            println!("=== HAPI Catalog Search: '{}' ===", search);
            let body = hapi_get(&format!("{}/catalog", HAPI_BASE))?;
            let parsed: serde_json::Value = serde_json::from_str(&body)?;
            let catalog = parsed["catalog"]
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("No catalog array"))?;

            let search_lower = search.to_lowercase();
            let mut matches = 0;
            for entry in catalog {
                let id = entry["id"].as_str().unwrap_or("");
                if id.to_lowercase().contains(&search_lower) {
                    println!("  {}", id);
                    matches += 1;
                }
            }
            println!("  {} datasets matching '{}'", matches, search);
        }

        Cmd::Info { dataset } => {
            println!("=== HAPI Info: {} ===", dataset);
            let body = hapi_get(&format!("{}/info?id={}", HAPI_BASE, dataset))?;
            let parsed: serde_json::Value = serde_json::from_str(&body)?;

            if let Some(start) = parsed["startDate"].as_str() {
                println!(
                    "  Time range: {} to {}",
                    start,
                    parsed["stopDate"].as_str().unwrap_or("?")
                );
            }

            if let Some(params) = parsed["parameters"].as_array() {
                println!("  Parameters ({}):", params.len());
                for (i, p) in params.iter().enumerate() {
                    let name = p["name"].as_str().unwrap_or("?");
                    let units = p["units"].as_str().unwrap_or("");
                    let desc = p["description"].as_str().unwrap_or("");
                    let desc_short = if desc.len() > 60 { &desc[..60] } else { desc };
                    println!("    [{:>2}] {:30} {:10} {}", i, name, units, desc_short);
                }
            }
        }

        Cmd::Download {
            dataset,
            params,
            start,
            end,
            out,
            out_dir,
            yearly,
        } => {
            if yearly {
                let dir = out_dir.unwrap_or_else(|| PathBuf::from("."));
                std::fs::create_dir_all(&dir)?;

                let start_year: u16 = start[..4].parse()?;
                let end_year: u16 = end[..4].parse()?;

                for year in start_year..=end_year {
                    let y_start = format!("{}-01-01T00:00:00Z", year);
                    let y_end = format!("{}-12-31T23:59:59Z", year);
                    let out_file = dir.join(format!("{}_{}.csv", dataset.to_lowercase(), year));

                    print!("  {} {}: ", dataset, year);
                    let url = format!(
                        "{}/data?id={}&time.min={}&time.max={}&parameters={}&format=csv",
                        HAPI_BASE, dataset, y_start, y_end, params
                    );

                    match hapi_get(&url) {
                        Ok(body) => {
                            let lines: Vec<&str> = body.lines().collect();
                            // Check for HAPI error
                            if body.contains("\"status\"") && body.contains("error") {
                                println!("HAPI error");
                                continue;
                            }
                            std::fs::write(&out_file, &body)?;
                            println!("{} records -> {}", lines.len(), out_file.display());
                        }
                        Err(e) => println!("FAILED: {}", e),
                    }
                }
            } else {
                let out_file = out.unwrap_or_else(|| PathBuf::from("hapi_output.csv"));
                println!("=== HAPI Download: {} ===", dataset);
                println!("  Params: {}", params);
                println!("  Range: {} to {}", start, end);

                let url = format!(
                    "{}/data?id={}&time.min={}&time.max={}&parameters={}&format=csv",
                    HAPI_BASE,
                    dataset,
                    if start.contains('T') {
                        start.clone()
                    } else {
                        format!("{}T00:00:00Z", start)
                    },
                    if end.contains('T') {
                        end.clone()
                    } else {
                        format!("{}T23:59:59Z", end)
                    },
                    params
                );

                let body = hapi_get(&url)?;
                let lines: Vec<&str> = body.lines().collect();
                std::fs::write(&out_file, &body)?;
                println!("  {} records -> {}", lines.len(), out_file.display());
            }
        }
    }

    Ok(())
}
