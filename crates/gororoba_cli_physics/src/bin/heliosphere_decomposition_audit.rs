//! Decomposition audit: phase vs spectral, transverse vs compressive.
//!
//! Runs the 32D CD associator on real data and two surrogate families:
//! (a) shared-phase-randomized (preserves cross-channel coherence)
//! (b) independent-phase-randomized (destroys all phase coupling)
//!
//! If real >> shared >> indep: signal is cross-channel phase organization
//! If real ~ shared >> indep: signal is spectral-only
//! If shared ~ indep: no cross-channel structure at all

use anyhow::{Context, Result};
use chrono::{Datelike, NaiveDate};
use clap::Parser;
use data_core::{
    catalogs::mms::parse_mms_fgm_hapi_csv_minutes,
    catalogs::themis::parse_themis_fgm_hapi_csv_minutes,
    catalogs::maven_mag::parse_maven_mag_hapi_csv_minutes,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use spectral_core::{phase_randomize_mv_independent, phase_randomize_mv_shared};
use std::{fs, path::PathBuf};

#[derive(Parser, Debug)]
#[command(
    name = "heliosphere-decomposition-audit",
    about = "Phase vs spectral decomposition of CD associator signal"
)]
struct Cli {
    /// Mission: themis-a, maven, mms
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

    #[arg(long, default_value_t = 20)]
    n_surrogates: usize,

    #[arg(long, default_value = "data/output/heliosphere/ablations/decomposition_audit.json")]
    out_json: PathBuf,

    #[arg(long, default_value = "data/external")]
    data_dir: PathBuf,
}

#[derive(Debug, Serialize)]
struct DecompResult {
    mission: String,
    n_vectors: usize,
    n_associator_norms: usize,
    real_mean: f64,
    real_std: f64,
    shared_phase_mean: f64,
    indep_phase_mean: f64,
    ratio_real_to_shared: f64,
    ratio_real_to_indep: f64,
    ratio_shared_to_indep: f64,
    interpretation: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let start = NaiveDate::parse_from_str(&cli.start_date, "%Y-%m-%d")
        .with_context(|| format!("bad start: {}", cli.start_date))?;
    let end = NaiveDate::parse_from_str(&cli.end_date, "%Y-%m-%d")
        .with_context(|| format!("bad end: {}", cli.end_date))?;

    println!("=== Decomposition Audit: {} ===", cli.mission);

    // Load minute-level data
    let (bx_series, by_series, bz_series) = load_components(&cli, start, end)?;
    let n = bx_series.len();
    println!("  Loaded {} minutes", n);

    if n < 20 {
        anyhow::bail!("Too few records ({})", n);
    }

    // Build Takens embedding
    let channels = 4usize;
    let steps = cli.embedding_dim / channels;

    let mut embedded: Vec<Vec<f64>> = Vec::new();
    for w in 0..=n.saturating_sub(steps) {
        let sum_b: f64 = (0..steps)
            .map(|s| {
                let i = w + s;
                (bx_series[i].powi(2) + by_series[i].powi(2) + bz_series[i].powi(2)).sqrt()
            })
            .sum();
        let mean_b = sum_b / steps as f64;
        if mean_b <= 0.01 || !mean_b.is_finite() { continue; }

        let mut v = vec![0.0; cli.embedding_dim];
        for s in 0..steps {
            let i = w + s;
            let bmag = (bx_series[i].powi(2) + by_series[i].powi(2) + bz_series[i].powi(2)).sqrt();
            v[s * channels] = bx_series[i] / mean_b;
            v[s * channels + 1] = by_series[i] / mean_b;
            v[s * channels + 2] = bz_series[i] / mean_b;
            v[s * channels + 3] = (bmag - mean_b) / mean_b;
        }
        embedded.push(v);
    }

    println!("  Embedded {} vectors ({}D)", embedded.len(), cli.embedding_dim);

    // Real associator
    let real_norms = cd_kernel::batch_sliding_associator_norms_parallel(
        &embedded, cli.embedding_dim,
    );
    let real_mean = real_norms.iter().sum::<f64>() / real_norms.len() as f64;
    let real_var = real_norms.iter().map(|&a| (a - real_mean).powi(2)).sum::<f64>()
        / real_norms.len() as f64;
    let real_std = real_var.sqrt();

    println!("  Real: mean={:.4}, std={:.4}", real_mean, real_std);

    // Surrogates: extract channel time series from embedded vectors
    let dim = cli.embedding_dim;
    let ch_series: Vec<Vec<f64>> = (0..dim)
        .map(|d| embedded.iter().map(|v| v[d]).collect())
        .collect();

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut shared_means = Vec::new();
    let mut indep_means = Vec::new();

    for i in 0..cli.n_surrogates {
        // Shared-phase: preserves cross-channel coherence
        let surr_shared = phase_randomize_mv_shared(&ch_series, &mut rng);
        let emb_shared: Vec<Vec<f64>> = (0..embedded.len())
            .map(|t| surr_shared.iter().map(|ch| ch[t]).collect())
            .collect();
        let norms_s = cd_kernel::batch_sliding_associator_norms_parallel(&emb_shared, dim);
        let mean_s = norms_s.iter().sum::<f64>() / norms_s.len() as f64;
        shared_means.push(mean_s);

        // Independent-phase: destroys all cross-channel coupling
        let surr_indep = phase_randomize_mv_independent(&ch_series, &mut rng);
        let emb_indep: Vec<Vec<f64>> = (0..embedded.len())
            .map(|t| surr_indep.iter().map(|ch| ch[t]).collect())
            .collect();
        let norms_i = cd_kernel::batch_sliding_associator_norms_parallel(&emb_indep, dim);
        let mean_i = norms_i.iter().sum::<f64>() / norms_i.len() as f64;
        indep_means.push(mean_i);

        if (i + 1) % 5 == 0 {
            println!("  Surrogate {}/{}: shared={:.4}, indep={:.4}", i + 1, cli.n_surrogates, mean_s, mean_i);
        }
    }

    let shared_mean = shared_means.iter().sum::<f64>() / shared_means.len() as f64;
    let indep_mean = indep_means.iter().sum::<f64>() / indep_means.len() as f64;

    let r_to_s = real_mean / shared_mean.max(1e-15);
    let r_to_i = real_mean / indep_mean.max(1e-15);
    let s_to_i = shared_mean / indep_mean.max(1e-15);

    let interpretation = if r_to_s > 1.2 && s_to_i > 1.2 {
        "Cross-channel phase organization (real > shared > indep)"
    } else if r_to_s <= 1.2 && r_to_i > 1.2 {
        "Spectral-only (phase coupling irrelevant, but spectrum matters)"
    } else if r_to_i <= 1.2 {
        "No signal above surrogates (associator is noise)"
    } else {
        "Mixed / marginal"
    };

    println!("\n=== Results ===");
    println!("  Real:    {:.4}", real_mean);
    println!("  Shared:  {:.4} (real/shared = {:.3})", shared_mean, r_to_s);
    println!("  Indep:   {:.4} (real/indep  = {:.3})", indep_mean, r_to_i);
    println!("  Shared/Indep = {:.3}", s_to_i);
    println!("  -> {}", interpretation);

    let result = DecompResult {
        mission: cli.mission.clone(),
        n_vectors: embedded.len(),
        n_associator_norms: real_norms.len(),
        real_mean, real_std,
        shared_phase_mean: shared_mean,
        indep_phase_mean: indep_mean,
        ratio_real_to_shared: r_to_s,
        ratio_real_to_indep: r_to_i,
        ratio_shared_to_indep: s_to_i,
        interpretation: interpretation.to_string(),
    };

    if let Some(parent) = cli.out_json.parent() { fs::create_dir_all(parent)?; }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&result)?)?;
    println!("\nWrote {}", cli.out_json.display());

    Ok(())
}

fn load_components(
    cli: &Cli,
    start: NaiveDate,
    end: NaiveDate,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    let mut bx = Vec::new();
    let mut by = Vec::new();
    let mut bz = Vec::new();

    let mission = cli.mission.to_lowercase();

    for offset in 0..=(end - start).num_days() {
        let date = start + chrono::Duration::days(offset);

        if mission.starts_with("themis") {
            let probe = if mission.contains("a") { "tha" } else { "thb" };
            let path = cli.data_dir.join("themis").join(
                format!("{}_fgm_{:04}_{:03}.csv", probe, date.year(), date.ordinal()),
            );
            if path.exists() {
                let content = fs::read_to_string(&path)?;
                let p_upper = probe.to_uppercase();
                for r in parse_themis_fgm_hapi_csv_minutes(&content, &p_upper) {
                    if r.b_magnitude <= cli.max_bmag {
                        bx.push(r.bx_gse); by.push(r.by_gse); bz.push(r.bz_gse);
                    }
                }
            }
        } else if mission == "maven" {
            let path = cli.data_dir.join("maven").join(
                format!("maven_mag_{:04}_{:03}.csv", date.year(), date.ordinal()),
            );
            if path.exists() {
                let content = fs::read_to_string(&path)?;
                for r in parse_maven_mag_hapi_csv_minutes(&content) {
                    if r.b_magnitude <= cli.max_bmag {
                        bx.push(r.bx_ss); by.push(r.by_ss); bz.push(r.bz_ss);
                    }
                }
            }
        } else if mission == "mms" {
            let doy = date.ordinal() as u16;
            let path = cli.data_dir.join("mms").join(
                format!("mms1_fgm_srvy_l2_2024_{doy}_{doy}.csv"),
            );
            if path.exists() {
                let content = fs::read_to_string(&path)?;
                for r in parse_mms_fgm_hapi_csv_minutes(&content) {
                    if r.b_magnitude <= cli.max_bmag {
                        bx.push(r.bx_gse); by.push(r.by_gse); bz.push(r.bz_gse);
                    }
                }
            }
        }
    }

    Ok((bx, by, bz))
}
