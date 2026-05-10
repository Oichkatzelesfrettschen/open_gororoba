//! Project 32D Takens embeddings to top 3 principal components.
//!
//! Outputs CSV: PC1, PC2, PC3, associator_norm, bmag, doy, hour, minute
//! for downstream visualization.

use chrono::{Datelike, NaiveDate};
use clap::Parser;
use data_core::catalogs::themis::parse_themis_fgm_hapi_csv_minutes;
use stats_core::helpers::svd_right_singular_vectors;
use std::fs;

#[derive(Parser)]
#[command(name = "heliosphere-embed-pca")]
struct Cli {
    #[arg(long)]
    start_date: String,
    #[arg(long)]
    end_date: String,
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,
    #[arg(long, default_value_t = 200.0)]
    max_bmag: f64,
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/pca_projection.csv"
    )]
    out_csv: String,
    #[arg(long, default_value = "data/external")]
    data_dir: String,
}

fn main() {
    let cli = Cli::parse();
    let start = NaiveDate::parse_from_str(&cli.start_date, "%Y-%m-%d").unwrap();
    let end = NaiveDate::parse_from_str(&cli.end_date, "%Y-%m-%d").unwrap();

    // Load data
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
    eprintln!("Loaded {} minutes", n);

    let channels = 4usize;
    let steps = cli.embedding_dim / channels;
    let dim = cli.embedding_dim;

    // Build embeddings
    let mut embedded: Vec<Vec<f64>> = Vec::new();
    let mut embed_keys: Vec<(u16, u8, u8)> = Vec::new();
    let mut embed_bmag: Vec<f64> = Vec::new();

    for w in 0..=n.saturating_sub(steps) {
        let sum_b: f64 = (0..steps)
            .map(|s| (bx[w + s].powi(2) + by[w + s].powi(2) + bz[w + s].powi(2)).sqrt())
            .sum();
        let mean_b = sum_b / steps as f64;
        if mean_b <= 0.01 || !mean_b.is_finite() {
            continue;
        }

        let mut v = vec![0.0; dim];
        for s in 0..steps {
            let i = w + s;
            let bmag = (bx[i].powi(2) + by[i].powi(2) + bz[i].powi(2)).sqrt();
            v[s * channels] = bx[i] / mean_b;
            v[s * channels + 1] = by[i] / mean_b;
            v[s * channels + 2] = bz[i] / mean_b;
            v[s * channels + 3] = (bmag - mean_b) / mean_b;
        }
        embedded.push(v);
        embed_keys.push(keys[w + steps - 1]);
        embed_bmag.push(mean_b);
    }

    let n_emb = embedded.len();
    eprintln!("Embedded {} vectors ({}D)", n_emb, dim);

    if n_emb < 10 {
        eprintln!("Too few vectors for PCA");
        return;
    }

    // Compute associator norms
    let norms = cd_kernel::batch_sliding_associator_norms_parallel(&embedded, dim);
    eprintln!("Computed {} norms", norms.len());

    // Build centered matrix as plain Vec<Vec<f64>> for SVD
    let mean_vec: Vec<f64> = (0..dim)
        .map(|d| embedded.iter().map(|v| v[d]).sum::<f64>() / n_emb as f64)
        .collect();

    let centered: Vec<Vec<f64>> = embedded[..n_emb]
        .iter()
        .map(|row| (0..dim).map(|j| row[j] - mean_vec[j]).collect())
        .collect();

    eprintln!("Computing SVD...");
    let (svals, v_t_rows) = svd_right_singular_vectors(&centered);

    // Project to top 3 PCs
    eprintln!("Projecting to 3 PCs...");

    // Explained variance
    let total_var: f64 = svals.iter().map(|s| s * s).sum();
    for (k, sv) in svals.iter().take(3).enumerate() {
        let var = sv.powi(2) / total_var * 100.0;
        eprintln!("  PC{}: {:.1}% variance", k + 1, var);
    }

    // Output CSV
    let mut out = String::from("pc1,pc2,pc3,associator_norm,bmag,doy,hour,minute\n");

    for i in 0..n_emb {
        // Project: row_i * V[:, 0:3]
        let mut pcs = [0.0f64; 3];
        for (k, pc) in pcs.iter_mut().enumerate().take(3.min(v_t_rows.len())) {
            *pc = centered[i]
                .iter()
                .zip(v_t_rows[k].iter())
                .map(|(a, b)| a * b)
                .sum();
        }

        // Associator norm (offset by 2 for triple)
        let a_norm = norms.get(i.saturating_sub(2)).copied().unwrap_or(0.0);

        let (doy, h, m) = embed_keys[i];
        out.push_str(&format!(
            "{:.6},{:.6},{:.6},{:.6},{:.2},{},{},{}\n",
            pcs[0], pcs[1], pcs[2], a_norm, embed_bmag[i], doy, h, m
        ));
    }

    fs::create_dir_all(
        std::path::Path::new(&cli.out_csv)
            .parent()
            .unwrap_or(std::path::Path::new(".")),
    )
    .ok();
    fs::write(&cli.out_csv, out).unwrap();
    eprintln!("Wrote {}", cli.out_csv);
}
