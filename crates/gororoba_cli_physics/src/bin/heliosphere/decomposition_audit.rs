//! Decomposition audit: phase vs spectral, transverse vs compressive.
//!
//! Runs the 32D CD associator on real data and two surrogate families:
//! (a) shared-phase-randomized (preserves cross-channel coherence)
//! (b) independent-phase-randomized (destroys all phase coupling)
//!
//! Surrogates act on the three physical field components and the Takens embedding is
//! rebuilt from each surrogate afterward. Randomizing the 32 delay coordinates directly
//! breaks the lag-copy identity between neighboring vectors, so the surrogate would then
//! measure destroyed delay structure rather than destroyed cross-channel phase.
//!
//! If real >> shared >> indep: signal is cross-channel phase organization
//! If real ~ shared >> indep: signal is spectral-only
//! If shared ~ indep: no cross-channel structure at all

use anyhow::{Context, Result};
use chrono::{Datelike, NaiveDate};
use clap::Args;
use data_core::catalogs::{
    maven_mag::parse_maven_mag_hapi_csv_minutes, mms::parse_mms_fgm_hapi_csv_minutes,
    themis::parse_themis_fgm_hapi_csv_minutes,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use spectral_core::{phase_randomize_mv_independent, phase_randomize_mv_shared};
use std::{fs, path::PathBuf};

#[derive(Args, Debug)]
pub struct Cli {
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

    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/decomposition_audit.json"
    )]
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

pub fn run(cli: Cli) -> Result<()> {
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

    let embedded = embed_takens(&bx_series, &by_series, &bz_series, cli.embedding_dim);

    println!(
        "  Embedded {} vectors ({}D)",
        embedded.len(),
        cli.embedding_dim
    );

    // Real associator
    let real_norms =
        cd_kernel::batch_sliding_associator_norms_parallel(&embedded, cli.embedding_dim);
    let real_mean = real_norms.iter().sum::<f64>() / real_norms.len() as f64;
    let real_var = real_norms
        .iter()
        .map(|&a| (a - real_mean).powi(2))
        .sum::<f64>()
        / real_norms.len() as f64;
    let real_std = real_var.sqrt();

    println!("  Real: mean={:.4}, std={:.4}", real_mean, real_std);

    // Surrogates act on the physical Bx, By, Bz series; each surrogate is re-embedded.
    let dim = cli.embedding_dim;
    let ch_series: Vec<Vec<f64>> = vec![bx_series.clone(), by_series.clone(), bz_series.clone()];

    let mut rng = ChaCha8Rng::seed_from_u64(42);

    let mut shared_means = Vec::new();
    let mut indep_means = Vec::new();

    for i in 0..cli.n_surrogates {
        // Shared-phase: preserves cross-channel coherence
        let surr_shared = phase_randomize_mv_shared(&ch_series, &mut rng);
        let emb_shared = embed_takens(&surr_shared[0], &surr_shared[1], &surr_shared[2], dim);
        let norms_s = cd_kernel::batch_sliding_associator_norms_parallel(&emb_shared, dim);
        let mean_s = norms_s.iter().sum::<f64>() / norms_s.len() as f64;
        shared_means.push(mean_s);

        // Independent-phase: destroys all cross-channel coupling
        let surr_indep = phase_randomize_mv_independent(&ch_series, &mut rng);
        let emb_indep = embed_takens(&surr_indep[0], &surr_indep[1], &surr_indep[2], dim);
        let norms_i = cd_kernel::batch_sliding_associator_norms_parallel(&emb_indep, dim);
        let mean_i = norms_i.iter().sum::<f64>() / norms_i.len() as f64;
        indep_means.push(mean_i);

        if (i + 1) % 5 == 0 {
            println!(
                "  Surrogate {}/{}: shared={:.4}, indep={:.4}",
                i + 1,
                cli.n_surrogates,
                mean_s,
                mean_i
            );
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
    println!(
        "  Shared:  {:.4} (real/shared = {:.3})",
        shared_mean, r_to_s
    );
    println!("  Indep:   {:.4} (real/indep  = {:.3})", indep_mean, r_to_i);
    println!("  Shared/Indep = {:.3}", s_to_i);
    println!("  -> {}", interpretation);

    let result = DecompResult {
        mission: cli.mission.clone(),
        n_vectors: embedded.len(),
        n_associator_norms: real_norms.len(),
        real_mean,
        real_std,
        shared_phase_mean: shared_mean,
        indep_phase_mean: indep_mean,
        ratio_real_to_shared: r_to_s,
        ratio_real_to_indep: r_to_i,
        ratio_shared_to_indep: s_to_i,
        interpretation: interpretation.to_string(),
    };

    if let Some(parent) = cli.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&cli.out_json, serde_json::to_string_pretty(&result)?)?;
    println!("\nWrote {}", cli.out_json.display());

    Ok(())
}

/// Takens delay embedding of a Bx, By, Bz minute series into `embedding_dim / 4` steps of
/// four channels (Bx, By, Bz, |B| contrast), each window normalized by its mean |B|.
/// Windows whose mean |B| is at or below 0.01 nT or non-finite are dropped.
fn embed_takens(bx: &[f64], by: &[f64], bz: &[f64], embedding_dim: usize) -> Vec<Vec<f64>> {
    let channels = 4usize;
    let steps = embedding_dim / channels;
    let n = bx.len().min(by.len()).min(bz.len());
    if steps == 0 || n < steps {
        return Vec::new();
    }
    let bmag = |i: usize| (bx[i].powi(2) + by[i].powi(2) + bz[i].powi(2)).sqrt();

    let mut embedded: Vec<Vec<f64>> = Vec::new();
    for w in 0..=n - steps {
        let mean_b = (0..steps).map(|s| bmag(w + s)).sum::<f64>() / steps as f64;
        if mean_b <= 0.01 || !mean_b.is_finite() {
            continue;
        }

        let mut v = vec![0.0; embedding_dim];
        for s in 0..steps {
            let i = w + s;
            v[s * channels] = bx[i] / mean_b;
            v[s * channels + 1] = by[i] / mean_b;
            v[s * channels + 2] = bz[i] / mean_b;
            v[s * channels + 3] = (bmag(i) - mean_b) / mean_b;
        }
        embedded.push(v);
    }
    embedded
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
            let path = cli.data_dir.join("themis").join(format!(
                "{}_fgm_{:04}_{:03}.csv",
                probe,
                date.year(),
                date.ordinal()
            ));
            if path.exists() {
                let content = fs::read_to_string(&path)?;
                let p_upper = probe.to_uppercase();
                for r in parse_themis_fgm_hapi_csv_minutes(&content, &p_upper) {
                    if r.b_magnitude <= cli.max_bmag {
                        bx.push(r.bx_gse);
                        by.push(r.by_gse);
                        bz.push(r.bz_gse);
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
                        bx.push(r.bx_ss);
                        by.push(r.by_ss);
                        bz.push(r.bz_ss);
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
                        bx.push(r.bx_gse);
                        by.push(r.by_gse);
                        bz.push(r.bz_gse);
                    }
                }
            }
        }
    }

    Ok((bx, by, bz))
}

#[cfg(test)]
mod tests {
    use super::embed_takens;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use spectral_core::phase_randomize_mv_independent;

    /// A rotating field of constant magnitude keeps every window mean equal, so the
    /// delay coordinates of consecutive vectors must be exact shifted copies.
    fn rotating_field(n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let bx: Vec<f64> = (0..n).map(|i| 5.0 * (0.1 * i as f64).cos()).collect();
        let by: Vec<f64> = (0..n).map(|i| 5.0 * (0.1 * i as f64).sin()).collect();
        let bz = vec![0.0; n];
        (bx, by, bz)
    }

    #[test]
    fn embedding_preserves_lag_copy_identity() {
        let (bx, by, bz) = rotating_field(64);
        let emb = embed_takens(&bx, &by, &bz, 32);
        assert_eq!(emb.len(), 64 - 8 + 1);
        for w in 0..emb.len() - 1 {
            for s in 1..8 {
                for c in 0..4 {
                    let later = emb[w + 1][(s - 1) * 4 + c];
                    let earlier = emb[w][s * 4 + c];
                    assert!(
                        (later - earlier).abs() < 1e-9,
                        "lag copy broken at w={w} s={s} c={c}: {later} vs {earlier}"
                    );
                }
            }
        }
    }

    #[test]
    fn surrogate_then_embed_keeps_lag_copy_identity() {
        // Randomizing the physical channels first leaves the re-embedded vectors with the
        // same shifted-copy relation as the real embedding; randomizing the 32 delay
        // coordinates after embedding would break it.
        let (bx, by, bz) = rotating_field(128);
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let surr = phase_randomize_mv_independent(&[bx, by, bz], &mut rng);
        let emb = embed_takens(&surr[0], &surr[1], &surr[2], 32);
        assert!(emb.len() > 2);
        for w in 0..emb.len() - 1 {
            let scale = emb[w][0] / surr[0][w];
            let scale_next = emb[w + 1][0] / surr[0][w + 1];
            for s in 1..8 {
                for c in 0..3 {
                    // Divide out each window's own mean |B| before comparing raw copies.
                    let earlier = emb[w][s * 4 + c] / scale;
                    let later = emb[w + 1][(s - 1) * 4 + c] / scale_next;
                    assert!(
                        (later - earlier).abs() < 1e-9,
                        "lag copy broken at w={w} s={s} c={c}"
                    );
                }
            }
        }
    }
}
