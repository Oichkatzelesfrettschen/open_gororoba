//! Q2: Transverse vs compressive mode content correlation with associator.
//!
//! For sliding windows of magnetic field data, compute BOTH:
//! (a) The 32D CD associator norm
//! (b) The perpendicular spectral fraction (transverse mode content)
//!
//! Then correlate: do high-associator intervals correspond to transverse-
//! dominant or compressive-dominant spectral content?

use anyhow::{Context, Result};
use chrono::{Datelike, NaiveDate};
use clap::Args;
use data_core::catalogs::{
    maven_mag::parse_maven_mag_hapi_csv_minutes, mms::parse_mms_fgm_hapi_csv_minutes,
    themis::parse_themis_fgm_hapi_csv_minutes,
};
use serde::Serialize;
use spectral_core::coherence::field_aligned_spectral_fractions;
use std::{fs, path::PathBuf};

#[derive(Args, Debug)]
pub struct Cli {
    #[arg(long)]
    mission: String,
    #[arg(long)]
    start_date: String,
    #[arg(long)]
    end_date: String,
    #[arg(long, default_value_t = 32)]
    embedding_dim: usize,
    #[arg(long, default_value_t = 100.0)]
    max_bmag: f64,
    /// Window size (minutes) for spectral fraction calculation.
    #[arg(long, default_value_t = 60)]
    spectral_window: usize,
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/decomposition_q2.json"
    )]
    out_json: PathBuf,
    #[arg(long, default_value = "data/external")]
    data_dir: PathBuf,
}

#[derive(Debug, Serialize)]
struct Q2Result {
    mission: String,
    n_windows: usize,
    correlation_assoc_vs_perp_frac: f64,
    high_assoc_mean_perp_frac: f64,
    low_assoc_mean_perp_frac: f64,
    overall_mean_perp_frac: f64,
    interpretation: String,
}

pub fn run(cli: Cli) -> Result<()> {

    let start = NaiveDate::parse_from_str(&cli.start_date, "%Y-%m-%d")
        .with_context(|| format!("bad start: {}", cli.start_date))?;
    let end = NaiveDate::parse_from_str(&cli.end_date, "%Y-%m-%d")
        .with_context(|| format!("bad end: {}", cli.end_date))?;

    println!(
        "=== Q2: Transverse/Compressive Decomposition: {} ===",
        cli.mission
    );

    let (bx, by, bz) = load_components(&cli, start, end)?;
    let n = bx.len();
    println!("  Loaded {} minutes", n);

    if n < cli.spectral_window * 2 {
        anyhow::bail!(
            "Too few records ({}) for window size {}",
            n,
            cli.spectral_window
        );
    }

    // Compute associator in sliding windows matching the spectral window
    let channels = 4usize;
    let steps = cli.embedding_dim / channels;
    let sw = cli.spectral_window;

    // For each non-overlapping window, compute both associator mean and perp fraction
    let n_windows = n / sw;
    let mut window_assoc: Vec<f64> = Vec::new();
    let mut window_perp: Vec<f64> = Vec::new();

    println!("  Computing {} windows of {} min each...", n_windows, sw);

    for wi in 0..n_windows {
        let start_idx = wi * sw;
        let end_idx = (start_idx + sw).min(n);
        let wlen = end_idx - start_idx;

        if wlen < steps + 2 {
            continue;
        }

        // Associator for this window
        let mut embedded: Vec<Vec<f64>> = Vec::new();
        for w in start_idx..=(end_idx.saturating_sub(steps)) {
            let sum_b: f64 = (0..steps)
                .map(|s| {
                    let i = w + s;
                    if i >= n {
                        return 0.0;
                    }
                    (bx[i].powi(2) + by[i].powi(2) + bz[i].powi(2)).sqrt()
                })
                .sum();
            let mean_b = sum_b / steps as f64;
            if mean_b <= 0.01 || !mean_b.is_finite() {
                continue;
            }

            let mut v = vec![0.0; cli.embedding_dim];
            for s in 0..steps {
                let i = w + s;
                if i >= n {
                    break;
                }
                let bmag = (bx[i].powi(2) + by[i].powi(2) + bz[i].powi(2)).sqrt();
                v[s * channels] = bx[i] / mean_b;
                v[s * channels + 1] = by[i] / mean_b;
                v[s * channels + 2] = bz[i] / mean_b;
                v[s * channels + 3] = (bmag - mean_b) / mean_b;
            }
            embedded.push(v);
        }

        if embedded.len() < 3 {
            continue;
        }

        let norms =
            cd_kernel::batch_sliding_associator_norms_parallel(&embedded, cli.embedding_dim);
        if norms.is_empty() {
            continue;
        }
        let assoc_mean = norms.iter().sum::<f64>() / norms.len() as f64;

        // Perpendicular spectral fraction for this window
        let win_bx = &bx[start_idx..end_idx];
        let win_by = &by[start_idx..end_idx];
        let win_bz = &bz[start_idx..end_idx];

        // Local mean field direction
        let mean_bx: f64 = win_bx.iter().sum::<f64>() / wlen as f64;
        let mean_by: f64 = win_by.iter().sum::<f64>() / wlen as f64;
        let mean_bz: f64 = win_bz.iter().sum::<f64>() / wlen as f64;
        let mean_bmag = (mean_bx.powi(2) + mean_by.powi(2) + mean_bz.powi(2)).sqrt();

        if mean_bmag < 0.01 {
            continue;
        }

        let b_hat = [
            mean_bx / mean_bmag,
            mean_by / mean_bmag,
            mean_bz / mean_bmag,
        ];

        let nperseg = (wlen / 4).max(8);
        let noverlap = nperseg / 2;

        let (_freqs, _par_frac, perp_frac) =
            field_aligned_spectral_fractions(win_bx, win_by, win_bz, b_hat, nperseg, noverlap);

        if perp_frac.is_empty() {
            continue;
        }

        // Band-average the perpendicular fraction
        let mean_perp: f64 = perp_frac.iter().sum::<f64>() / perp_frac.len() as f64;

        window_assoc.push(assoc_mean);
        window_perp.push(mean_perp);
    }

    println!("  Computed {} valid windows", window_assoc.len());

    if window_assoc.len() < 5 {
        anyhow::bail!("Too few valid windows ({})", window_assoc.len());
    }

    // Spearman rank correlation
    let corr = spearman_rank_correlation(&window_assoc, &window_perp);

    // Split by associator median: high vs low
    let mut sorted_assoc = window_assoc.clone();
    sorted_assoc.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_assoc = sorted_assoc[sorted_assoc.len() / 2];

    let high_perp: Vec<f64> = window_assoc
        .iter()
        .zip(window_perp.iter())
        .filter(|&(&a, _)| a > median_assoc)
        .map(|(&_, &p)| p)
        .collect();
    let low_perp: Vec<f64> = window_assoc
        .iter()
        .zip(window_perp.iter())
        .filter(|&(&a, _)| a <= median_assoc)
        .map(|(&_, &p)| p)
        .collect();

    let high_mean = if high_perp.is_empty() {
        0.0
    } else {
        high_perp.iter().sum::<f64>() / high_perp.len() as f64
    };
    let low_mean = if low_perp.is_empty() {
        0.0
    } else {
        low_perp.iter().sum::<f64>() / low_perp.len() as f64
    };
    let overall_mean = window_perp.iter().sum::<f64>() / window_perp.len() as f64;

    let interpretation = if corr > 0.2 {
        format!(
            "Positive correlation (r={corr:.3}): high associator correlates with transverse-dominant spectral content"
        )
    } else if corr < -0.2 {
        format!(
            "Negative correlation (r={corr:.3}): high associator correlates with compressive-dominant spectral content"
        )
    } else {
        format!(
            "Weak/no correlation (r={corr:.3}): associator is independent of transverse/compressive mode mix"
        )
    };

    println!("\n=== Results ===");
    println!("  Spearman correlation (assoc vs perp_frac): {:.4}", corr);
    println!("  High-assoc mean perp fraction: {:.4}", high_mean);
    println!("  Low-assoc mean perp fraction:  {:.4}", low_mean);
    println!("  Overall mean perp fraction:    {:.4}", overall_mean);
    println!("  -> {}", interpretation);

    let result = Q2Result {
        mission: cli.mission.clone(),
        n_windows: window_assoc.len(),
        correlation_assoc_vs_perp_frac: corr,
        high_assoc_mean_perp_frac: high_mean,
        low_assoc_mean_perp_frac: low_mean,
        overall_mean_perp_frac: overall_mean,
        interpretation,
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&result)?)?;
    println!("\nWrote {}", cli.out_json.display());

    Ok(())
}

fn spearman_rank_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 3 {
        return 0.0;
    }

    let rank = |vals: &[f64]| -> Vec<f64> {
        let mut indexed: Vec<(usize, f64)> = vals.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut ranks = vec![0.0; n];
        for (rank, &(orig_idx, _)) in indexed.iter().enumerate() {
            ranks[orig_idx] = rank as f64;
        }
        ranks
    };

    let rx = rank(x);
    let ry = rank(y);

    let mean_rx = rx.iter().sum::<f64>() / n as f64;
    let mean_ry = ry.iter().sum::<f64>() / n as f64;

    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for i in 0..n {
        let dx = rx[i] - mean_rx;
        let dy = ry[i] - mean_ry;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    let denom = (var_x * var_y).sqrt();
    if denom < 1e-15 { 0.0 } else { cov / denom }
}

fn load_components(
    cli: &Cli,
    start: NaiveDate,
    end: NaiveDate,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mut bx_all = Vec::new();
    let mut by_all = Vec::new();
    let mut bz_all = Vec::new();
    let mission = cli.mission.to_lowercase();

    for offset in 0..=(end - start).num_days() {
        let date = start + chrono::Duration::days(offset);
        if mission.starts_with("themis") {
            let probe = if mission.contains("a") { "tha" } else { "thb" };
            let path = cli.data_dir.join("themis").join(format!(
                "{}_fgm_{:04}_{:03}.csv",
                probe,
                date.year(),
                date.ordinal()
            ));
            if path.exists() {
                let content = fs::read_to_string(&path)?;
                for r in parse_themis_fgm_hapi_csv_minutes(&content, &probe.to_uppercase()) {
                    if r.b_magnitude <= cli.max_bmag {
                        bx_all.push(r.bx_gse);
                        by_all.push(r.by_gse);
                        bz_all.push(r.bz_gse);
                    }
                }
            }
        } else if mission == "maven" {
            let path = cli.data_dir.join("maven").join(format!(
                "maven_mag_{:04}_{:03}.csv",
                date.year(),
                date.ordinal()
            ));
            if path.exists() {
                let content = fs::read_to_string(&path)?;
                for r in parse_maven_mag_hapi_csv_minutes(&content) {
                    if r.b_magnitude <= cli.max_bmag {
                        bx_all.push(r.bx_ss);
                        by_all.push(r.by_ss);
                        bz_all.push(r.bz_ss);
                    }
                }
            }
        } else if mission == "mms" {
            let doy = date.ordinal() as u16;
            let path = cli
                .data_dir
                .join("mms")
                .join(format!("mms1_fgm_srvy_l2_2024_{doy}_{doy}.csv"));
            if path.exists() {
                let content = fs::read_to_string(&path)?;
                for r in parse_mms_fgm_hapi_csv_minutes(&content) {
                    if r.b_magnitude <= cli.max_bmag {
                        bx_all.push(r.bx_gse);
                        by_all.push(r.by_gse);
                        bz_all.push(r.bz_gse);
                    }
                }
            }
        }
    }
    Ok((bx_all, by_all, bz_all))
}
