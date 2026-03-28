//! Dump per-minute associator norms as CSV for beta scatter analysis.
//! Output: doy,hour,minute,associator_norm

use chrono::{Datelike, NaiveDate};
use clap::Parser;
use data_core::catalogs::themis::parse_themis_fgm_hapi_csv_minutes;
use std::fs;

#[derive(Parser)]
struct Cli {
    #[arg(long)]
    start_date: String,
    #[arg(long)]
    end_date: String,
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,
    #[arg(long, default_value_t = 200.0)]
    max_bmag: f64,
    #[arg(long, default_value = "data/external")]
    data_dir: String,
    /// Normalization: current (default) or direction (unit vectors, zero magnitude channel).
    #[arg(long, default_value = "current")]
    normalization: String,
}

fn main() {
    let cli = Cli::parse();
    let start = NaiveDate::parse_from_str(&cli.start_date, "%Y-%m-%d").unwrap();
    let end = NaiveDate::parse_from_str(&cli.end_date, "%Y-%m-%d").unwrap();

    let mut bx = Vec::new();
    let mut by = Vec::new();
    let mut bz = Vec::new();
    let mut keys: Vec<(u16, u8, u8)> = Vec::new();

    for offset in 0..=(end - start).num_days() {
        let date = start + chrono::Duration::days(offset);
        let path = format!(
            "{}/themis/tha_fgm_{:04}_{:03}.csv",
            cli.data_dir,
            date.year(),
            date.ordinal()
        );
        if let Ok(content) = fs::read_to_string(&path) {
            for r in parse_themis_fgm_hapi_csv_minutes(&content, "THA") {
                if r.b_magnitude <= cli.max_bmag {
                    bx.push(r.bx_gse);
                    by.push(r.by_gse);
                    bz.push(r.bz_gse);
                    keys.push((r.doy, r.hour, r.minute));
                }
            }
        }
    }

    let n = bx.len();
    let channels = 4usize;
    let steps = cli.embedding_dim / channels;

    let mut embedded: Vec<Vec<f64>> = Vec::new();
    let mut embed_keys: Vec<(u16, u8, u8)> = Vec::new();

    for w in 0..=n.saturating_sub(steps) {
        let sum_b: f64 = (0..steps)
            .map(|s| (bx[w + s].powi(2) + by[w + s].powi(2) + bz[w + s].powi(2)).sqrt())
            .sum();
        let mean_b = sum_b / steps as f64;
        if mean_b <= 0.01 || !mean_b.is_finite() {
            continue;
        }

        let mut v = vec![0.0; cli.embedding_dim];
        let mut skip = false;
        for s in 0..steps {
            let i = w + s;
            let bmag = (bx[i].powi(2) + by[i].powi(2) + bz[i].powi(2)).sqrt();
            if cli.normalization == "direction" {
                if bmag < 1e-12 { skip = true; break; }
                v[s * channels] = bx[i] / bmag;
                v[s * channels + 1] = by[i] / bmag;
                v[s * channels + 2] = bz[i] / bmag;
                v[s * channels + 3] = 0.0;
            } else {
                v[s * channels] = bx[i] / mean_b;
                v[s * channels + 1] = by[i] / mean_b;
                v[s * channels + 2] = bz[i] / mean_b;
                v[s * channels + 3] = (bmag - mean_b) / mean_b;
            }
        }
        if skip { continue; }
        embedded.push(v);
        embed_keys.push(keys[w + steps - 1]);
    }

    eprintln!("Embedded {} vectors, computing associator...", embedded.len());

    let norms =
        cd_kernel::batch_sliding_associator_norms_parallel(&embedded, cli.embedding_dim);

    eprintln!("Dumping {} norms", norms.len());
    println!("doy,hour,minute,associator_norm");
    for (k, &norm) in norms.iter().enumerate() {
        let (doy, h, m) = embed_keys[k + 2];
        println!("{},{},{},{:.6}", doy, h, m, norm);
    }
}
