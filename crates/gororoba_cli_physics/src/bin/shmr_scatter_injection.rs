//! Contrarian D1: SHMR scatter phase-smearing injection test.
//!
//! Challenges the alpha_zd upper limit by showing that Moster+2013 SHMR scatter
//! (~0.15 dex, Behroozi+2019) propagates to ~25% r_s uncertainty, which
//! phase-smears high-k Fourier modes (k >= 4) by more than pi/2 radians.
//!
//! Method:
//! 1. Inject ZD signal at alpha_zd=0.01 (10x above claimed limit) into raw data
//! 2. Run stacking + Fourier analysis WITHOUT r_s perturbation (baseline recovery)
//! 3. Run stacking + Fourier analysis WITH 0.15 dex lognormal r_s perturbation
//! 4. Compare per-mode recovery SNR: do high-k modes lose sensitivity?
//!
//! Detection: Recovery SNR drops > 50% for modes 5-7 with scatter
//! Failure: Recovery SNR changes < 20% across all modes
//!
//! Uses 20 Monte Carlo realizations of the scatter to get statistics.

use clap::Parser;
use cosmology_core::{
    harmonic_stacking::{
        CdDimensionParams, NormalizedPoint, NormalizedResiduals, StackingConfig,
        fourier_at_wavenumbers, stack_residuals,
    },
    nfw_utils::{nfw_enclosed_mass_from_params, nfw_params_from_mass},
};
use data_core::catalogs::manga::{parse_manga_dapall_csv, parse_manga_rotcurves};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, LogNormal};
use std::{f64::consts::PI, path::PathBuf};

const G_KPC_KMS2: f64 = 4.302e-6;

#[derive(Parser)]
#[command(name = "shmr-scatter-injection")]
#[command(about = "D1: SHMR scatter phase-smearing test for injection recovery")]
struct Cli {
    #[arg(
        long,
        default_value = "data/external/manga/rotcurves/manga_rotcurves_all.csv"
    )]
    rotcurves: PathBuf,

    #[arg(long, default_value = "data/external/manga/dapall_selection.csv")]
    dapall: PathBuf,

    #[arg(long, default_value_t = 16)]
    cd_dim: usize,

    #[arg(long, default_value_t = 8)]
    min_points: usize,

    #[arg(long, default_value_t = 0.5)]
    x_min: f64,

    #[arg(long, default_value_t = 10.0)]
    x_max: f64,

    /// Injected alpha_zd value.
    #[arg(long, default_value_t = 0.01)]
    alpha_zd: f64,

    /// SHMR scatter in dex (Behroozi+2019: ~0.15).
    #[arg(long, default_value_t = 0.15)]
    shmr_scatter_dex: f64,

    /// Number of Monte Carlo realizations.
    #[arg(long, default_value_t = 20)]
    n_mc: usize,

    #[arg(long, default_value = "data/results/e183/shmr_scatter_injection.csv")]
    out_csv: PathBuf,
}

struct GalaxyData {
    plateifu: String,
    r_s_kpc: f64,
    points_raw: Vec<RawPoint>, // Physical coordinates (r_kpc, v_obs, v_err)
}

struct RawPoint {
    r_kpc: f64,
    v_obs_km_s: f64,
    v_err_km_s: f64,
    v_nfw_km_s: f64, // NFW prediction at this radius
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    eprintln!("Loading MaNGA rotation curves...");
    let rotcurves = parse_manga_rotcurves(&cli.rotcurves).map_err(|e| anyhow::anyhow!(e))?;
    let dapall = parse_manga_dapall_csv(&cli.dapall).map_err(|e| anyhow::anyhow!(e))?;
    let dapall_map: std::collections::HashMap<String, _> = dapall
        .into_iter()
        .map(|e| (e.plateifu.clone(), e))
        .collect();

    // Build galaxies with raw physical coordinates
    let mut galaxies: Vec<GalaxyData> = Vec::new();
    let mut skipped = 0usize;

    for galaxy in &rotcurves {
        let meta = match dapall_map.get(&galaxy.plateifu) {
            Some(m) => m,
            None => {
                skipped += 1;
                continue;
            }
        };

        let log_m200 = meta.estimated_log_m200();
        if !log_m200.is_finite() {
            skipped += 1;
            continue;
        }
        let m200 = 10.0_f64.powf(log_m200);
        let nfw = nfw_params_from_mass(m200, meta.z);
        let r_s = nfw.r_s_kpc;
        if r_s <= 0.0 {
            skipped += 1;
            continue;
        }

        let points: Vec<RawPoint> = galaxy
            .points
            .iter()
            .filter_map(|pt| {
                if pt.r_kpc <= 0.0 {
                    return None;
                }
                let m_enc = nfw_enclosed_mass_from_params(pt.r_kpc, &nfw);
                let v_sq = G_KPC_KMS2 * m_enc / pt.r_kpc;
                let v_nfw = v_sq.max(0.0).sqrt();
                if v_nfw < 1.0 {
                    return None;
                }
                if !pt.v_obs_km_s.is_finite() || !pt.v_err_km_s.is_finite() {
                    return None;
                }
                Some(RawPoint {
                    r_kpc: pt.r_kpc,
                    v_obs_km_s: pt.v_obs_km_s,
                    v_err_km_s: pt.v_err_km_s,
                    v_nfw_km_s: v_nfw,
                })
            })
            .collect();

        if points.len() >= cli.min_points {
            galaxies.push(GalaxyData {
                plateifu: galaxy.plateifu.clone(),
                r_s_kpc: r_s,
                points_raw: points,
            });
        } else {
            skipped += 1;
        }
    }

    let n_gal = galaxies.len();
    eprintln!("Loaded {} galaxies ({} skipped)", n_gal, skipped);

    let cd_params = CdDimensionParams::new(cli.cd_dim);
    let n_modes = cd_params.n_modes;
    let wavenumbers: Vec<f64> = (1..=n_modes)
        .map(|n| 2.0 * PI * n as f64 / n_modes as f64)
        .collect();

    let config = StackingConfig {
        x_min: cli.x_min,
        x_max: cli.x_max,
        n_grid: 200,
        min_galaxies_per_bin: 3,
        inverse_variance_weighting: true,
        cd_params: cd_params.clone(),
        exclude_psf_flagged: false,
    };

    // === BASELINE: Inject signal with EXACT r_s ===
    eprintln!("\n--- Baseline: Injection with exact r_s ---");
    let baseline_power = run_injection(
        &galaxies,
        &wavenumbers,
        &config,
        cli.alpha_zd,
        |g| g.r_s_kpc, // Use exact r_s
    );

    eprintln!("Per-mode Fourier power (exact r_s):");
    for (i, (&k, &p)) in wavenumbers.iter().zip(baseline_power.iter()).enumerate() {
        eprintln!("  mode {}: k={:.4}, power={:.6e}", i + 1, k, p);
    }

    // === SCATTER: Inject signal with perturbed r_s ===
    eprintln!(
        "\n--- Scatter: Injection with {:.2} dex SHMR scatter ---",
        cli.shmr_scatter_dex
    );

    // Log-normal scatter: if scatter = 0.15 dex, then sigma_ln = 0.15 * ln(10) = 0.345
    let sigma_ln = cli.shmr_scatter_dex * 10.0_f64.ln();

    let mut scatter_powers: Vec<Vec<f64>> = Vec::new();

    for mc in 0..cli.n_mc {
        let mut rng = ChaCha8Rng::seed_from_u64(42 + mc as u64);
        let ln_dist = LogNormal::new(0.0, sigma_ln).unwrap();

        let power = run_injection(&galaxies, &wavenumbers, &config, cli.alpha_zd, |g| {
            let factor: f64 = ln_dist.sample(&mut rng);
            g.r_s_kpc * factor
        });

        scatter_powers.push(power);
    }

    // Compute per-mode statistics across MC realizations
    eprintln!("\nPer-mode recovery comparison (exact vs scattered):");
    eprintln!(
        "{:>5}  {:>8}  {:>12}  {:>12}  {:>10}  {:>10}",
        "mode", "k", "exact_power", "scatter_med", "ratio", "drop_pct"
    );

    let mut results: Vec<(usize, f64, f64, f64, f64, f64)> = Vec::new();
    let mut n_modes_dropped = 0usize;

    for mode_idx in 0..n_modes {
        let exact_p = baseline_power[mode_idx];

        let mut scatter_mode: Vec<f64> = scatter_powers.iter().map(|p| p[mode_idx]).collect();
        scatter_mode.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_p = scatter_mode[cli.n_mc / 2];
        let p25 = scatter_mode[cli.n_mc / 4];
        let p75 = scatter_mode[3 * cli.n_mc / 4];

        let ratio = if exact_p > 1e-20 {
            median_p / exact_p
        } else {
            f64::NAN
        };
        let drop_pct = (1.0 - ratio) * 100.0;

        if drop_pct > 50.0 {
            n_modes_dropped += 1;
        }

        eprintln!(
            "{:>5}  {:>8.4}  {:>12.6e}  {:>12.6e}  {:>10.4}  {:>10.1}%",
            mode_idx + 1,
            wavenumbers[mode_idx],
            exact_p,
            median_p,
            ratio,
            drop_pct
        );

        results.push((
            mode_idx + 1,
            wavenumbers[mode_idx],
            exact_p,
            median_p,
            ratio,
            drop_pct,
        ));

        let _ = (p25, p75);
    }

    // Phase error analysis
    eprintln!("\n--- Theoretical phase error at x=1.0 ---");
    for (i, &k) in wavenumbers.iter().enumerate() {
        let delta_phi = k * 1.0 * (10.0_f64.powf(cli.shmr_scatter_dex) - 1.0);
        eprintln!(
            "  mode {}: k={:.4}, delta_phi={:.3} rad ({:.1} deg)",
            i + 1,
            k,
            delta_phi,
            delta_phi.to_degrees()
        );
    }

    // Verdict
    eprintln!("\n--- Contrarian D1 Verdict ---");
    let modes_above_50pct_drop: Vec<usize> =
        results.iter().filter(|r| r.5 > 50.0).map(|r| r.0).collect();

    if modes_above_50pct_drop.len() >= 3 {
        eprintln!(
            "DETECTION: SHMR scatter destroys {} of 7 modes (>50% power drop)",
            modes_above_50pct_drop.len()
        );
        eprintln!("  Affected modes: {:?}", modes_above_50pct_drop);
        eprintln!(
            "  The alpha_zd upper limit is INVALID for modes {}-7",
            modes_above_50pct_drop[0]
        );
        eprintln!(
            "  Effective sensitivity uses only {} modes, weakening the limit by ~{:.0}%",
            7 - modes_above_50pct_drop.len(),
            (1.0 - ((7 - modes_above_50pct_drop.len()) as f64 / 7.0).sqrt()) * 100.0
        );
    } else if n_modes_dropped == 0 {
        eprintln!("FAILURE: SHMR scatter does not significantly affect any mode");
        eprintln!("  All modes retain >50% power -- the null result is robust to r_s uncertainty");
    } else {
        eprintln!(
            "INCONCLUSIVE: {} modes affected, fewer than 3",
            n_modes_dropped
        );
    }

    // Write CSV
    let mut wtr = csv::Writer::from_path(&cli.out_csv)?;
    wtr.write_record([
        "mode",
        "k",
        "exact_power",
        "scatter_median_power",
        "ratio",
        "drop_pct",
        "n_mc",
    ])?;
    for r in &results {
        wtr.write_record(&[
            format!("{}", r.0),
            format!("{:.6}", r.1),
            format!("{:.8e}", r.2),
            format!("{:.8e}", r.3),
            format!("{:.6}", r.4),
            format!("{:.2}", r.5),
            format!("{}", cli.n_mc),
        ])?;
    }
    wtr.write_record(&[
        "SUMMARY".to_string(),
        format!("alpha_zd={}", cli.alpha_zd),
        format!("scatter_dex={}", cli.shmr_scatter_dex),
        format!("n_modes_dropped={}", n_modes_dropped),
        format!("n_galaxies={}", n_gal),
        format!("n_mc={}", cli.n_mc),
        "".to_string(),
    ])?;
    wtr.flush()?;

    eprintln!("\nResults written to {:?}", cli.out_csv);
    Ok(())
}

/// Run injection + stacking + Fourier analysis with a given r_s mapping function.
fn run_injection<F>(
    galaxies: &[GalaxyData],
    wavenumbers: &[f64],
    config: &StackingConfig,
    alpha_zd: f64,
    mut r_s_fn: F,
) -> Vec<f64>
where
    F: FnMut(&GalaxyData) -> f64,
{
    let mut normalized: Vec<NormalizedResiduals> = Vec::new();

    for g in galaxies {
        let r_s = r_s_fn(g);
        if r_s <= 0.0 {
            continue;
        }

        let points: Vec<NormalizedPoint> = g
            .points_raw
            .iter()
            .filter_map(|pt| {
                let x = pt.r_kpc / r_s;
                let v_nfw = pt.v_nfw_km_s;
                if v_nfw < 1.0 {
                    return None;
                }

                // Inject ZD signal: delta_v += alpha_zd * sum_n (0.5/n) * cos(k_n * x) * exp(-x)
                let mut injected_delta = (pt.v_obs_km_s - v_nfw) / v_nfw;
                if alpha_zd > 0.0 {
                    for (mode_idx, &k) in wavenumbers.iter().enumerate() {
                        let n = (mode_idx + 1) as f64;
                        let w = 0.5 / n;
                        injected_delta += alpha_zd * w * (k * x).cos() * (-x).exp();
                    }
                }

                let delta_v_err = pt.v_err_km_s / v_nfw;
                if !injected_delta.is_finite() || !delta_v_err.is_finite() {
                    return None;
                }

                Some(NormalizedPoint {
                    x,
                    delta_v: injected_delta,
                    delta_v_err,
                })
            })
            .collect();

        if points.len() >= 8 {
            normalized.push(NormalizedResiduals {
                name: g.plateifu.clone(),
                r_s_kpc: r_s,
                points,
            });
        }
    }

    let result = stack_residuals(&normalized, config);
    let (power, _phase) = fourier_at_wavenumbers(
        &result.x_grid,
        &result.delta_stack,
        &result.n_contributing,
        config.min_galaxies_per_bin,
        wavenumbers,
    );
    power
}
