//! Pure Rust MAST Zarr downloader: fetch Mirnov diagnostic data from FAIR-MAST
//! public S3 bucket using ureq (sync HTTP). Downloads Zarr v2 chunks to local
//! filesystem for analysis by mast-mirnov-cd.
//!
//! No authentication required. Data source: s3://mast/ at s3.echo.stfc.ac.uk

use anyhow::{Context, Result};
use clap::Parser;
use rayon::prelude::*;
use std::{fs, path::PathBuf};

#[derive(Parser)]
#[command(name = "mast-fetch")]
struct Cli {
    /// MAST shot numbers (comma-separated).
    #[arg(long)]
    shots: String,

    /// Diagnostic to download (default: ama = Mirnov Bv analysis).
    #[arg(long, default_value = "ama")]
    diagnostic: String,

    /// Signals to download (comma-separated, URL-encoded = as %3D).
    #[arg(long, default_value = "time,n%3D2_signal,n%3Dodd_signal")]
    signals: String,

    /// Output base directory.
    #[arg(long, default_value = "data/external/mast")]
    out_dir: PathBuf,
}

const S3_BASE: &str = "https://s3.echo.stfc.ac.uk/mast/level1/shots";

fn download_chunk(url: &str, path: &PathBuf) -> Result<usize> {
    let resp = ureq::get(url)
        .call()
        .with_context(|| format!("GET {url}"))?;
    let buf = resp
        .into_body()
        .read_to_vec()
        .with_context(|| format!("read body from {url}"))?;
    fs::write(path, &buf)?;
    Ok(buf.len())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let shots: Vec<u32> = cli
        .shots
        .split(',')
        .map(|s| s.trim().parse::<u32>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .context("Invalid shot numbers")?;

    let signals: Vec<String> = cli.signals.split(',').map(|s| s.to_string()).collect();

    println!("=== MAST Zarr Fetcher (ureq, pure Rust) ===");
    println!("  Shots: {:?}", shots);
    println!("  Diagnostic: {}", cli.diagnostic);
    println!("  Signals: {:?}", signals);

    for &shot in &shots {
        let shot_base = format!("{S3_BASE}/{shot}.zarr/{}", cli.diagnostic);
        let local_base = cli
            .out_dir
            .join(format!("shot{shot}.zarr/{}", cli.diagnostic));
        fs::create_dir_all(&local_base)?;

        // Download metadata
        for meta in [".zattrs", ".zgroup"] {
            let url = format!("{shot_base}/{meta}");
            let path = local_base.join(meta);
            match download_chunk(&url, &path) {
                Ok(n) => println!("  shot {shot}: {meta} ({n} bytes)"),
                Err(e) => println!("  shot {shot}: {meta} FAILED: {e}"),
            }
        }

        // Download each signal's metadata + chunks in parallel
        for sig_encoded in &signals {
            let sig_local = sig_encoded.replace("%3D", "=");
            let sig_dir = local_base.join(&sig_local);
            fs::create_dir_all(&sig_dir)?;

            // .zarray metadata
            let zarray_url = format!("{shot_base}/{sig_encoded}/.zarray");
            let zarray_path = sig_dir.join(".zarray");
            match download_chunk(&zarray_url, &zarray_path) {
                Ok(_) => {}
                Err(e) => {
                    println!("  shot {shot}/{sig_local}: .zarray FAILED: {e} (skipping)");
                    continue;
                }
            }

            // Read .zarray to determine chunk count
            let zarray_str = fs::read_to_string(&zarray_path)?;
            let n_chunks = if let Some(shape_start) = zarray_str.find("\"shape\"") {
                // Parse shape and chunks from JSON
                let shape_val = zarray_str[shape_start..]
                    .split('[')
                    .nth(1)
                    .and_then(|s| s.split(']').next())
                    .and_then(|s| s.trim().parse::<u64>().ok())
                    .unwrap_or(0);
                let chunk_val = if let Some(cs) = zarray_str.find("\"chunks\"") {
                    zarray_str[cs..]
                        .split('[')
                        .nth(1)
                        .and_then(|s| s.split(']').next())
                        .and_then(|s| s.trim().parse::<u64>().ok())
                        .unwrap_or(1)
                } else {
                    1
                };
                if chunk_val > 0 {
                    shape_val.div_ceil(chunk_val) as usize
                } else {
                    1
                }
            } else {
                8 // default guess
            };

            // Download chunks in parallel with rayon
            let chunks: Vec<(usize, String, PathBuf)> = (0..n_chunks)
                .map(|i| {
                    let url = format!("{shot_base}/{sig_encoded}/{i}");
                    let path = sig_dir.join(format!("{i}"));
                    (i, url, path)
                })
                .collect();

            let results: Vec<(usize, Result<usize>)> = chunks
                .par_iter()
                .map(|(i, url, path)| (*i, download_chunk(url, path)))
                .collect();

            let mut total_bytes = 0usize;
            let mut ok_count = 0usize;
            for (_, result) in &results {
                if let Ok(n) = result {
                    total_bytes += n;
                    ok_count += 1;
                }
            }
            println!(
                "  shot {shot}/{sig_local}: {ok_count}/{n_chunks} chunks, {:.1} KB total",
                total_bytes as f64 / 1024.0
            );
        }
    }

    println!("\n  Fetch complete. Data at {}", cli.out_dir.display());
    Ok(())
}
