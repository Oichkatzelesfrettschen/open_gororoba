//! Mass-binned phase coherence analysis for MaNGA harmonic stacking.
//!
//! Tests the hypothesis that ZD forcing phase depends on halo mass M_200,
//! so the full-sample stack averages over incoherent phases (destroying a
//! sub-noise signal). Splitting into narrow mass bins acts as an NMR-like
//! "spin echo" that recovers phase coherence.
//!
//! For each of 10 equal-N mass bins, stacks rotation curves independently
//! and computes the complex DFT at 7 CD-ZD wavenumbers. Then measures:
//! 1. Per-bin SNR (does any bin exceed 2.0?)
//! 2. Spearman rank correlation of phase vs mass-bin index (monotonic trend?)
//! 3. Per-mode Rayleigh R across the 10 bin phases (phase clustering?)
//!
//! Detection: Spearman |rho| > 0.7 for >= 3 modes, or any bin SNR > 2.0
//! Rejection: max |rho| < 0.5 AND max bin SNR < 1.5
//!
//! Reference: Hypothesis H1, plan imperative-skipping-kahn.md

use clap::Parser;
use cosmology_core::{
    harmonic_stacking::{
        CdDimensionParams, NormalizedPoint, NormalizedResiduals, StackingConfig,
        fourier_at_wavenumbers, stack_residuals,
    },
    nfw_utils::{nfw_enclosed_mass_from_params, nfw_params_from_mass},
};
use data_core::catalogs::manga::{parse_manga_dapall_csv, parse_manga_rotcurves};
use std::f64::consts::PI;
use std::path::PathBuf;

const G_KPC_KMS2: f64 = 4.302e-6;

#[derive(Parser)]
#[command(name = "mass-binned-phase")]
#[command(about = "Mass-binned phase coherence test for ZD harmonic stacking")]
struct Cli {
    /// Path to MaNGA rotation curves CSV.
    #[arg(long, default_value = "data/external/manga/rotcurves/manga_rotcurves_all.csv")]
    rotcurves: PathBuf,

    /// Path to DAPall selection CSV.
    #[arg(long, default_value = "data/external/manga/dapall_selection.csv")]
    dapall: PathBuf,

    /// Number of mass bins.
    #[arg(long, default_value_t = 10)]
    n_bins: usize,

    /// CD dimension.
    #[arg(long, default_value_t = 16)]
    cd_dim: usize,

    /// Minimum rotation curve points per galaxy.
    #[arg(long, default_value_t = 8)]
    min_points: usize,

    /// Minimum x = r/r_s.
    #[arg(long, default_value_t = 0.5)]
    x_min: f64,

    /// Maximum x = r/r_s.
    #[arg(long, default_value_t = 10.0)]
    x_max: f64,

    /// Output CSV path.
    #[arg(long, default_value = "mass_binned_phase.csv")]
    out_csv: PathBuf,
}

/// Galaxy with its log(M200) for sorting.
struct GalaxyWithMass {
    log_m200: f64,
    residuals: NormalizedResiduals,
}

/// Spearman rank correlation between two slices of equal length.
fn spearman_rho(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len());
    let n = x.len();
    if n < 3 {
        return 0.0;
    }

    fn ranks(v: &[f64]) -> Vec<f64> {
        let mut indexed: Vec<(usize, f64)> = v.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut r = vec![0.0_f64; v.len()];
        let mut i = 0;
        while i < indexed.len() {
            let mut j = i;
            while j < indexed.len() && indexed[j].1 == indexed[i].1 {
                j += 1;
            }
            let avg_rank = (i + j + 1) as f64 / 2.0;
            for item in indexed.iter().take(j).skip(i) {
                r[item.0] = avg_rank;
            }
            i = j;
        }
        r
    }

    let rx = ranks(x);
    let ry = ranks(y);

    let mean_rx = rx.iter().sum::<f64>() / n as f64;
    let mean_ry = ry.iter().sum::<f64>() / n as f64;

    let mut num = 0.0_f64;
    let mut dx2 = 0.0_f64;
    let mut dy2 = 0.0_f64;
    for i in 0..n {
        let drx = rx[i] - mean_rx;
        let dry = ry[i] - mean_ry;
        num += drx * dry;
        dx2 += drx * drx;
        dy2 += dry * dry;
    }

    if dx2 == 0.0 || dy2 == 0.0 {
        return 0.0;
    }
    num / (dx2 * dy2).sqrt()
}

/// Rayleigh test statistic R for a set of angles (radians).
/// R = 1 means perfect phase clustering, R ~ 0 means uniform.
fn rayleigh_r(phases: &[f64]) -> f64 {
    let n = phases.len() as f64;
    if n < 2.0 {
        return 0.0;
    }
    let sum_cos: f64 = phases.iter().map(|p| p.cos()).sum();
    let sum_sin: f64 = phases.iter().map(|p| p.sin()).sum();
    ((sum_cos * sum_cos + sum_sin * sum_sin) / (n * n)).sqrt()
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    eprintln!("Loading MaNGA rotation curves from {:?}...", cli.rotcurves);
    let rotcurves = parse_manga_rotcurves(&cli.rotcurves).map_err(|e| anyhow::anyhow!(e))?;
    eprintln!("Loaded {} galaxy rotation curves", rotcurves.len());

    eprintln!("Loading DAPall metadata from {:?}...", cli.dapall);
    let dapall = parse_manga_dapall_csv(&cli.dapall).map_err(|e| anyhow::anyhow!(e))?;

    let dapall_map: std::collections::HashMap<String, _> = dapall
        .into_iter()
        .map(|e| (e.plateifu.clone(), e))
        .collect();

    let mut galaxies: Vec<GalaxyWithMass> = Vec::new();
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
        let z = meta.z;
        let nfw = nfw_params_from_mass(m200, z);
        let r_s = nfw.r_s_kpc;

        let points: Vec<NormalizedPoint> = galaxy
            .points
            .iter()
            .filter_map(|pt| {
                if pt.r_kpc <= 0.0 || r_s <= 0.0 {
                    return None;
                }
                let x = pt.r_kpc / r_s;
                let m_enc = nfw_enclosed_mass_from_params(pt.r_kpc, &nfw);
                let v_sq = G_KPC_KMS2 * m_enc / pt.r_kpc;
                let v_nfw = v_sq.max(0.0).sqrt();
                if v_nfw < 1.0 {
                    return None;
                }
                let delta_v = (pt.v_obs_km_s - v_nfw) / v_nfw;
                let delta_v_err = pt.v_err_km_s / v_nfw;
                if !delta_v.is_finite() || !delta_v_err.is_finite() {
                    return None;
                }
                Some(NormalizedPoint {
                    x,
                    delta_v,
                    delta_v_err,
                })
            })
            .collect();

        if points.len() >= cli.min_points {
            galaxies.push(GalaxyWithMass {
                log_m200,
                residuals: NormalizedResiduals {
                    name: galaxy.plateifu.clone(),
                    r_s_kpc: r_s,
                    points,
                },
            });
        } else {
            skipped += 1;
        }
    }

    eprintln!(
        "Normalized {} galaxies ({} skipped)",
        galaxies.len(),
        skipped
    );

    // Sort by log(M200)
    galaxies.sort_by(|a, b| a.log_m200.partial_cmp(&b.log_m200).unwrap());

    let n_total = galaxies.len();
    let bin_size = n_total / cli.n_bins;
    let remainder = n_total % cli.n_bins;

    eprintln!(
        "Splitting into {} mass bins (~{} galaxies each)",
        cli.n_bins, bin_size
    );

    let cd_params = CdDimensionParams::new(cli.cd_dim);
    let n_modes = cd_params.n_modes;

    // CD-ZD wavenumbers
    let wavenumbers: Vec<f64> = (1..=n_modes)
        .map(|n| 2.0 * PI * n as f64 / n_modes as f64)
        .collect();

    // Per-bin stacking results
    struct BinResult {
        bin_index: usize,
        n_galaxies: usize,
        log_m200_median: f64,
        log_m200_min: f64,
        log_m200_max: f64,
        power: Vec<f64>,
        phase: Vec<f64>,
        rms_residual: f64,
        snr: f64,
    }

    let mut bin_results: Vec<BinResult> = Vec::new();
    let mut offset = 0_usize;

    for bin_idx in 0..cli.n_bins {
        let this_bin_size = bin_size + if bin_idx < remainder { 1 } else { 0 };
        let bin_galaxies: Vec<&NormalizedResiduals> = galaxies[offset..offset + this_bin_size]
            .iter()
            .map(|g| &g.residuals)
            .collect();
        let bin_masses: Vec<f64> = galaxies[offset..offset + this_bin_size]
            .iter()
            .map(|g| g.log_m200)
            .collect();

        let median_mass = bin_masses[bin_masses.len() / 2];
        let min_mass = bin_masses[0];
        let max_mass = bin_masses[bin_masses.len() - 1];

        // Build owned Vec for stack_residuals
        let owned: Vec<NormalizedResiduals> =
            bin_galaxies.iter().map(|g| (*g).clone()).collect();

        let config = StackingConfig {
            x_min: cli.x_min,
            x_max: cli.x_max,
            n_grid: 200,
            min_galaxies_per_bin: 3,
            inverse_variance_weighting: true,
            cd_params: cd_params.clone(),
            exclude_psf_flagged: false,
        };

        let result = stack_residuals(&owned, &config);

        // Compute DFT at CD-ZD wavenumbers on this bin's stacked profile
        let (power, phase) = fourier_at_wavenumbers(
            &result.x_grid,
            &result.delta_stack,
            &result.n_contributing,
            config.min_galaxies_per_bin,
            &wavenumbers,
        );

        let max_power = power.iter().copied().fold(0.0_f64, f64::max);
        let snr = if result.rms_residual > 0.0 {
            max_power.sqrt() / result.rms_residual
        } else {
            0.0
        };

        eprintln!(
            "  Bin {}: N={}, log(M200)=[{:.2}, {:.2}], median={:.2}, SNR={:.4}, RMS={:.6}",
            bin_idx, this_bin_size, min_mass, max_mass, median_mass, snr, result.rms_residual
        );

        bin_results.push(BinResult {
            bin_index: bin_idx,
            n_galaxies: this_bin_size,
            log_m200_median: median_mass,
            log_m200_min: min_mass,
            log_m200_max: max_mass,
            power,
            phase,
            rms_residual: result.rms_residual,
            snr,
        });

        offset += this_bin_size;
    }

    // Spearman rank correlation of phase vs bin index for each mode
    let bin_indices: Vec<f64> = (0..cli.n_bins).map(|i| i as f64).collect();

    eprintln!("\n--- Phase trend analysis ---");
    eprintln!(
        "{:>5}  {:>8}  {:>10}  {:>10}  {:>10}",
        "mode", "k", "spearman", "rayleigh_r", "max_bin_snr"
    );

    let mut mode_results: Vec<(usize, f64, f64, f64)> = Vec::new();

    for (mode_idx, &k) in wavenumbers.iter().enumerate() {
        let phases: Vec<f64> = bin_results.iter().map(|b| b.phase[mode_idx]).collect();
        let rho = spearman_rho(&bin_indices, &phases);
        let r = rayleigh_r(&phases);

        eprintln!(
            "{:>5}  {:>8.4}  {:>10.4}  {:>10.4}  {:>10.4}",
            mode_idx + 1,
            k,
            rho,
            r,
            bin_results
                .iter()
                .map(|b| {
                    if b.rms_residual > 0.0 {
                        b.power[mode_idx].sqrt() / b.rms_residual
                    } else {
                        0.0
                    }
                })
                .fold(0.0_f64, f64::max)
        );

        mode_results.push((mode_idx + 1, rho, r, k));
    }

    // Summary statistics
    let max_spearman = mode_results
        .iter()
        .map(|m| m.1.abs())
        .fold(0.0_f64, f64::max);
    let max_bin_snr = bin_results
        .iter()
        .map(|b| b.snr)
        .fold(0.0_f64, f64::max);
    let modes_above_07: usize = mode_results.iter().filter(|m| m.1.abs() > 0.7).count();

    eprintln!("\n--- Detection criteria ---");
    eprintln!("Max |Spearman rho|: {:.4}", max_spearman);
    eprintln!("Modes with |rho| > 0.7: {}", modes_above_07);
    eprintln!("Max per-bin SNR: {:.4}", max_bin_snr);

    if modes_above_07 >= 3 || max_bin_snr > 2.0 {
        eprintln!("RESULT: DETECTION -- mass-dependent phase coherence found");
    } else if max_spearman < 0.5 && max_bin_snr < 1.5 {
        eprintln!("RESULT: REJECTED -- no mass-dependent phase coherence");
    } else {
        eprintln!("RESULT: INCONCLUSIVE -- between detection and rejection thresholds");
    }

    // Write CSV
    let mut wtr = csv::Writer::from_path(&cli.out_csv)?;
    wtr.write_record([
        "mode",
        "k",
        "bin_index",
        "n_galaxies",
        "log_m200_median",
        "log_m200_min",
        "log_m200_max",
        "re",
        "im",
        "power",
        "phase",
        "rms",
        "bin_snr",
        "mode_snr",
    ])?;

    for bin in &bin_results {
        for (mode_idx, &k) in wavenumbers.iter().enumerate() {
            let p = bin.power[mode_idx];
            let ph = bin.phase[mode_idx];
            let re = p.sqrt() * ph.cos();
            let im = p.sqrt() * ph.sin();
            let mode_snr = if bin.rms_residual > 0.0 {
                p.sqrt() / bin.rms_residual
            } else {
                0.0
            };
            wtr.write_record(&[
                format!("{}", mode_idx + 1),
                format!("{:.6}", k),
                format!("{}", bin.bin_index),
                format!("{}", bin.n_galaxies),
                format!("{:.4}", bin.log_m200_median),
                format!("{:.4}", bin.log_m200_min),
                format!("{:.4}", bin.log_m200_max),
                format!("{:.8e}", re),
                format!("{:.8e}", im),
                format!("{:.8e}", p),
                format!("{:.6}", ph),
                format!("{:.8e}", bin.rms_residual),
                format!("{:.6}", bin.snr),
                format!("{:.6}", mode_snr),
            ])?;
        }
    }
    wtr.flush()?;

    // Write summary CSV
    let summary_path = cli.out_csv.with_extension("summary.csv");
    let mut swtr = csv::Writer::from_path(&summary_path)?;
    swtr.write_record(["mode", "k", "spearman_rho", "rayleigh_r"])?;
    for (mode, rho, r, k) in &mode_results {
        swtr.write_record(&[
            format!("{mode}"),
            format!("{k:.6}"),
            format!("{rho:.6}"),
            format!("{r:.6}"),
        ])?;
    }
    swtr.flush()?;

    eprintln!("Results written to {:?}", cli.out_csv);
    eprintln!("Summary written to {:?}", summary_path);

    Ok(())
}
