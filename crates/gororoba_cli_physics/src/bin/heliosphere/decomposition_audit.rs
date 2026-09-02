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

/// Mean |B| at or below this value marks a window as unusable: the
/// embedding divides by the window mean, so a near-zero field turns a
/// measurement gap into an unbounded vector.
const NORMALIZATION_FLOOR_NT: f64 = 0.01;

/// Surrogate draws whose accepted windows fall under the floor are redrawn
/// this many times before the audit fails. Redrawing keeps every surrogate
/// on the real support; dropping windows would change the sample count.
const MAX_SURROGATE_RESAMPLES: usize = 20;

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
    support: SupportRecord,
}

/// The temporal support every series in the audit is evaluated on. The
/// real series fixes the accepted window indices; each surrogate is
/// embedded on exactly those indices, so the three means are ratios over
/// one sample and not over three.
#[derive(Debug, Serialize)]
struct SupportRecord {
    n_minutes: usize,
    window_steps: usize,
    windows_total: usize,
    support_count: usize,
    windows_dropped: usize,
    support_sha256: String,
    normalization_floor_nt: f64,
    seed: u64,
    n_surrogates: usize,
    shared_surrogate_counts: Vec<usize>,
    indep_surrogate_counts: Vec<usize>,
    shared_resamples: Vec<usize>,
    indep_resamples: Vec<usize>,
}

/// Window indices accepted by the real series.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SupportMask {
    steps: usize,
    windows_total: usize,
    accepted: Vec<usize>,
}

impl SupportMask {
    fn sha256_hex(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update((self.steps as u64).to_le_bytes());
        hasher.update((self.windows_total as u64).to_le_bytes());
        for &w in &self.accepted {
            hasher.update((w as u64).to_le_bytes());
        }
        hasher
            .finalize()
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect()
    }
}

/// A surrogate draw embedded on the real support, with the number of
/// redraws it took to land every accepted window above the floor.
struct SurrogateEmbedding {
    vectors: Vec<Vec<f64>>,
    resamples: usize,
}

fn embed_surrogate_on_support<F>(
    mut draw: F,
    mask: &SupportMask,
    dim: usize,
    label: &str,
) -> Result<SurrogateEmbedding>
where
    F: FnMut() -> Vec<Vec<f64>>,
{
    for resamples in 0..=MAX_SURROGATE_RESAMPLES {
        let surr = draw();
        if let Some(vectors) = embed_takens_on_support(&surr[0], &surr[1], &surr[2], dim, mask) {
            return Ok(SurrogateEmbedding { vectors, resamples });
        }
    }
    anyhow::bail!(
        "{label} surrogate fell under the {NORMALIZATION_FLOOR_NT} nT floor on accepted support \
         in {MAX_SURROGATE_RESAMPLES} consecutive redraws; the audit refuses to change the \
         sample count silently"
    )
}

fn mean_of(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
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

    let (embedded, mask) = embed_takens(&bx_series, &by_series, &bz_series, cli.embedding_dim);

    println!(
        "  Embedded {} vectors ({}D) on {} of {} windows",
        embedded.len(),
        cli.embedding_dim,
        mask.accepted.len(),
        mask.windows_total
    );
    if embedded.is_empty() {
        anyhow::bail!("no window clears the {NORMALIZATION_FLOOR_NT} nT floor");
    }

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

    let seed = 42u64;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut shared_means = Vec::new();
    let mut indep_means = Vec::new();
    let mut shared_counts = Vec::new();
    let mut indep_counts = Vec::new();
    let mut shared_resamples = Vec::new();
    let mut indep_resamples = Vec::new();

    for i in 0..cli.n_surrogates {
        // Shared-phase: preserves cross-channel coherence
        let shared = embed_surrogate_on_support(
            || phase_randomize_mv_shared(&ch_series, &mut rng),
            &mask,
            dim,
            "shared-phase",
        )?;
        let norms_s = cd_kernel::batch_sliding_associator_norms_parallel(&shared.vectors, dim);
        let mean_s = mean_of(&norms_s);
        shared_means.push(mean_s);
        shared_counts.push(shared.vectors.len());
        shared_resamples.push(shared.resamples);

        // Independent-phase: destroys all cross-channel coupling
        let indep = embed_surrogate_on_support(
            || phase_randomize_mv_independent(&ch_series, &mut rng),
            &mask,
            dim,
            "independent-phase",
        )?;
        let norms_i = cd_kernel::batch_sliding_associator_norms_parallel(&indep.vectors, dim);
        let mean_i = mean_of(&norms_i);
        indep_means.push(mean_i);
        indep_counts.push(indep.vectors.len());
        indep_resamples.push(indep.resamples);

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
        support: SupportRecord {
            n_minutes: n,
            window_steps: mask.steps,
            windows_total: mask.windows_total,
            support_count: mask.accepted.len(),
            windows_dropped: mask.windows_total - mask.accepted.len(),
            support_sha256: mask.sha256_hex(),
            normalization_floor_nt: NORMALIZATION_FLOOR_NT,
            seed,
            n_surrogates: cli.n_surrogates,
            shared_surrogate_counts: shared_counts,
            indep_surrogate_counts: indep_counts,
            shared_resamples,
            indep_resamples,
        },
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
/// Windows whose mean |B| is at or below `NORMALIZATION_FLOOR_NT` or non-finite are
/// dropped; the returned mask lists the accepted window starts.
fn embed_takens(
    bx: &[f64],
    by: &[f64],
    bz: &[f64],
    embedding_dim: usize,
) -> (Vec<Vec<f64>>, SupportMask) {
    let steps = embedding_dim / 4;
    let n = bx.len().min(by.len()).min(bz.len());
    if steps == 0 || n < steps {
        return (
            Vec::new(),
            SupportMask {
                steps,
                windows_total: 0,
                accepted: Vec::new(),
            },
        );
    }
    let windows_total = n - steps + 1;
    let mut embedded = Vec::new();
    let mut accepted = Vec::new();
    for w in 0..windows_total {
        if let Some(v) = embed_window(bx, by, bz, w, steps) {
            embedded.push(v);
            accepted.push(w);
        }
    }
    (
        embedded,
        SupportMask {
            steps,
            windows_total,
            accepted,
        },
    )
}

/// Embed a surrogate series on the real support. Every accepted window is
/// kept, so the result has exactly `mask.accepted.len()` vectors; a window
/// that falls under the floor in the surrogate returns `None` and the
/// caller redraws.
fn embed_takens_on_support(
    bx: &[f64],
    by: &[f64],
    bz: &[f64],
    embedding_dim: usize,
    mask: &SupportMask,
) -> Option<Vec<Vec<f64>>> {
    let steps = embedding_dim / 4;
    let n = bx.len().min(by.len()).min(bz.len());
    if steps != mask.steps || n < steps || n - steps + 1 != mask.windows_total {
        return None;
    }
    mask.accepted
        .iter()
        .map(|&w| embed_window(bx, by, bz, w, steps))
        .collect()
}

fn embed_window(bx: &[f64], by: &[f64], bz: &[f64], w: usize, steps: usize) -> Option<Vec<f64>> {
    let channels = 4usize;
    let bmag = |i: usize| (bx[i].powi(2) + by[i].powi(2) + bz[i].powi(2)).sqrt();
    let mean_b = (0..steps).map(|s| bmag(w + s)).sum::<f64>() / steps as f64;
    if mean_b <= NORMALIZATION_FLOOR_NT || !mean_b.is_finite() {
        return None;
    }
    let mut v = vec![0.0; steps * channels];
    for s in 0..steps {
        let i = w + s;
        v[s * channels] = bx[i] / mean_b;
        v[s * channels + 1] = by[i] / mean_b;
        v[s * channels + 2] = bz[i] / mean_b;
        v[s * channels + 3] = (bmag(i) - mean_b) / mean_b;
    }
    Some(v)
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
    use super::{
        MAX_SURROGATE_RESAMPLES, SupportMask, embed_surrogate_on_support, embed_takens,
        embed_takens_on_support,
    };
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use spectral_core::{phase_randomize_mv_independent, phase_randomize_mv_shared};

    /// A rotating field with a measurement gap: 40 minutes of zero field in the middle.
    /// The real embedding drops every window touching the gap; a phase-randomized
    /// surrogate spreads the gap's energy over the whole record and keeps those windows,
    /// which is the support divergence the shared mask removes.
    fn gapped_field(n: usize, gap_start: usize, gap_len: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let (mut bx, mut by, bz) = rotating_field(n);
        for i in gap_start..gap_start + gap_len {
            bx[i] = 0.0;
            by[i] = 0.0;
        }
        (bx, by, bz)
    }

    #[test]
    fn low_field_gap_drops_windows_from_the_real_support() {
        let (bx, by, bz) = gapped_field(256, 100, 40);
        let (emb, mask) = embed_takens(&bx, &by, &bz, 32);
        assert_eq!(mask.windows_total, 256 - 8 + 1);
        // A window is rejected only when its mean |B| is under the floor; with 8 steps
        // that is every window lying entirely inside the 40-minute gap.
        assert_eq!(mask.windows_total - mask.accepted.len(), 40 - 8 + 1);
        assert_eq!(emb.len(), mask.accepted.len());
        assert!(!mask.accepted.contains(&110));
        assert!(mask.accepted.contains(&99));
    }

    #[test]
    fn unmasked_surrogates_diverge_from_the_real_support_and_masked_ones_match_it() {
        let (bx, by, bz) = gapped_field(256, 100, 40);
        let (_, real_mask) = embed_takens(&bx, &by, &bz, 32);
        let channels = vec![bx, by, bz];
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        for _ in 0..4 {
            let shared = phase_randomize_mv_shared(&channels, &mut rng);
            let (_, own_mask) = embed_takens(&shared[0], &shared[1], &shared[2], 32);
            assert_ne!(
                own_mask.accepted, real_mask.accepted,
                "the surrogate's own support must differ, or the fix has nothing to fix"
            );
            let on_support =
                embed_takens_on_support(&shared[0], &shared[1], &shared[2], 32, &real_mask)
                    .expect("phase randomization keeps the surrogate above the floor");
            assert_eq!(on_support.len(), real_mask.accepted.len());

            let indep = phase_randomize_mv_independent(&channels, &mut rng);
            let on_support =
                embed_takens_on_support(&indep[0], &indep[1], &indep[2], 32, &real_mask)
                    .expect("phase randomization keeps the surrogate above the floor");
            assert_eq!(on_support.len(), real_mask.accepted.len());
        }
    }

    #[test]
    fn surrogate_embedding_reports_resamples_and_refuses_after_the_cap() {
        let (bx, by, bz) = rotating_field(64);
        let (_, mask) = embed_takens(&bx, &by, &bz, 32);
        let zero = vec![vec![0.0; 64]; 3];
        let good = vec![bx.clone(), by.clone(), bz.clone()];
        // The first two draws sit under the floor, the third clears it.
        let mut draws = vec![good, zero.clone(), zero.clone()];
        let ok = embed_surrogate_on_support(|| draws.pop().unwrap(), &mask, 32, "test")
            .expect("third draw clears the floor");
        assert_eq!(ok.resamples, 2);
        assert_eq!(ok.vectors.len(), mask.accepted.len());

        let mut draws_made = 0usize;
        let err = embed_surrogate_on_support(
            || {
                draws_made += 1;
                zero.clone()
            },
            &mask,
            32,
            "test",
        );
        assert!(err.is_err());
        assert_eq!(draws_made, MAX_SURROGATE_RESAMPLES + 1);
    }

    #[test]
    fn support_mask_rejects_a_series_of_a_different_length() {
        let (bx, by, bz) = rotating_field(64);
        let (_, mask) = embed_takens(&bx, &by, &bz, 32);
        let short: Vec<f64> = bx[..60].to_vec();
        assert!(embed_takens_on_support(&short, &by[..60], &bz[..60], 32, &mask).is_none());
        let other = SupportMask {
            steps: 4,
            windows_total: mask.windows_total,
            accepted: mask.accepted.clone(),
        };
        assert!(embed_takens_on_support(&bx, &by, &bz, 32, &other).is_none());
    }

    #[test]
    fn support_hash_tracks_the_accepted_indices() {
        let (bx, by, bz) = gapped_field(256, 100, 40);
        let (_, mask) = embed_takens(&bx, &by, &bz, 32);
        let (bx2, by2, bz2) = gapped_field(256, 120, 40);
        let (_, mask2) = embed_takens(&bx2, &by2, &bz2, 32);
        assert_eq!(mask.accepted.len(), mask2.accepted.len());
        assert_ne!(mask.sha256_hex(), mask2.sha256_hex());
        assert_eq!(mask.sha256_hex(), mask.clone().sha256_hex());
    }

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
        let (emb, _) = embed_takens(&bx, &by, &bz, 32);
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
        let (emb, _) = embed_takens(&surr[0], &surr[1], &surr[2], 32);
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
