//! Rust-native LoTSS analysis suite.
//!
//! Replaces the legacy Python utilities that lived under `src/lotss_analysis/`
//! with clap-driven Rust entrypoints:
//! - `flux-distribution`: DR1/DR2/DR3 N(S) summary and power-law fits.
//! - `kinematic-bisection`: Radio-loud vs radio-quiet MaNGA inner-core RMS test.
//! - `ultrametric`: DR1 spatial p-ultrametricity against a shuffled null.

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand};
use data_core::catalogs::lotss::{LoTSSRelease, load_from_fits};
use rand::{
    RngExt, SeedableRng,
    seq::{SliceRandom, index::sample},
};
use rand_chacha::ChaCha8Rng;
use serde::Serialize;
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};
use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};

const N_SAMPLE_DEFAULT: usize = 5_000;
const N_BOOTSTRAP_DEFAULT: usize = 100;
const N_TRIPLES_DEFAULT: usize = 50_000;
const ULTRA_TOLERANCE: f64 = 1.0e-6;
#[derive(Parser)]
#[command(name = "lotss-analysis")]
#[command(about = "Rust-native LoTSS analysis workflows")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compare LoTSS DR1/DR2/DR3 source-count distributions.
    FluxDistribution {
        #[arg(long)]
        dr1: Option<PathBuf>,
        #[arg(long)]
        dr2: Option<PathBuf>,
        #[arg(long)]
        dr3: Option<PathBuf>,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long, default_value_t = 40)]
        bins: usize,
    },
    /// Split MaNGA kinematics by LoTSS detection state.
    KinematicBisection {
        #[arg(long, default_value = "data/external/manga/manga_lotss_xmatch_dr3.csv")]
        xmatch: PathBuf,
        #[arg(long, default_value = "data/external/manga/manga_rotcurves_all.csv")]
        rotcurves: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
        #[arg(long, default_value_t = 0.5)]
        inner_lo: f64,
        #[arg(long, default_value_t = 1.25)]
        inner_hi: f64,
        #[arg(long, default_value_t = 1_000)]
        n_bootstrap: usize,
    },
    /// Test LoTSS DR1 sky positions for p-ultrametricity.
    Ultrametric {
        #[arg(long, default_value = "data/external/radio_surveys/lotss_dr1.fits")]
        lotss: PathBuf,
        #[arg(long, default_value_t = N_SAMPLE_DEFAULT)]
        n_sample: usize,
        #[arg(long, default_value_t = N_BOOTSTRAP_DEFAULT)]
        n_bootstrap: usize,
        #[arg(long, default_value_t = N_TRIPLES_DEFAULT)]
        n_triples: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, Serialize)]
struct FluxReleaseReport {
    release: String,
    n_sources: usize,
    area_sq_deg: f64,
    flux_min_mjy: f64,
    flux_max_mjy: f64,
    flux_median_mjy: f64,
    power_law_gamma: f64,
    power_law_gamma_err: f64,
    fit_intercept: f64,
    fit_intercept_err: f64,
    n_fit_bins: usize,
}

#[derive(Debug, Clone, Serialize)]
struct FluxDistributionReport {
    generated_at_utc: String,
    bins: usize,
    releases: Vec<FluxReleaseReport>,
}

#[derive(Debug, Clone, Serialize)]
struct KinematicBisectionReport {
    generated_at_utc: String,
    experiment_id: String,
    xmatch_path: String,
    rotcurves_path: String,
    sample: BisectionSampleReport,
    inner_core_rms: BisectionInnerRmsReport,
    mann_whitney_u: MannWhitneyReport,
    pearson_flux_rms: PearsonReport,
}

#[derive(Debug, Clone, Serialize)]
struct BisectionSampleReport {
    n_manga_total: usize,
    n_radio_loud: usize,
    n_radio_quiet: usize,
    detection_fraction: f64,
    inner_lo: f64,
    inner_hi: f64,
}

#[derive(Debug, Clone, Serialize)]
struct BisectionInnerRmsReport {
    rms_radio_loud_median: f64,
    rms_radio_quiet_median: f64,
    rms_ratio_loud_over_quiet: f64,
    rms_ratio_ci_lo: f64,
    rms_ratio_ci_hi: f64,
    n_bootstrap: usize,
}

#[derive(Debug, Clone, Serialize)]
struct MannWhitneyReport {
    statistic: f64,
    p_value: f64,
    effect_direction: String,
}

#[derive(Debug, Clone, Serialize)]
struct PearsonReport {
    r: f64,
    p_value: f64,
    n_pairs: usize,
}

#[derive(Debug, Clone, Serialize)]
struct UltrametricReport {
    generated_at_utc: String,
    experiment_id: String,
    sample: UltrametricSampleReport,
    results: UltrametricResultsReport,
}

#[derive(Debug, Clone, Serialize)]
struct UltrametricSampleReport {
    n_sample: usize,
    n_triples_per_estimate: usize,
    n_bootstrap: usize,
}

#[derive(Debug, Clone, Serialize)]
struct UltrametricResultsReport {
    p_ultra_observed: f64,
    p_ultra_null_mean: f64,
    p_ultra_null_std: f64,
    p_ultra_ci_lo: f64,
    p_ultra_ci_hi: f64,
    z_score: f64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::FluxDistribution {
            dr1,
            dr2,
            dr3,
            output,
            bins,
        } => cmd_flux_distribution(
            dr1.as_deref(),
            dr2.as_deref(),
            dr3.as_deref(),
            output.as_deref(),
            bins,
        ),
        Command::KinematicBisection {
            xmatch,
            rotcurves,
            output,
            inner_lo,
            inner_hi,
            n_bootstrap,
        } => cmd_kinematic_bisection(
            &xmatch,
            &rotcurves,
            output.as_deref(),
            inner_lo,
            inner_hi,
            n_bootstrap,
        ),
        Command::Ultrametric {
            lotss,
            n_sample,
            n_bootstrap,
            n_triples,
            output,
        } => cmd_ultrametric(&lotss, output.as_deref(), n_sample, n_bootstrap, n_triples),
    }
}

fn cmd_flux_distribution(
    dr1: Option<&Path>,
    dr2: Option<&Path>,
    dr3: Option<&Path>,
    output: Option<&Path>,
    bins: usize,
) -> Result<()> {
    let mut releases = Vec::new();
    if let Some(path) = dr1 {
        releases.push(build_flux_release_report(
            "DR1",
            path,
            LoTSSRelease::DR1,
            424.0,
            bins,
        )?);
    }
    if let Some(path) = dr2 {
        releases.push(build_flux_release_report(
            "DR2",
            path,
            LoTSSRelease::DR2,
            5_720.0,
            bins,
        )?);
    }
    if let Some(path) = dr3 {
        releases.push(build_flux_release_report(
            "DR3",
            path,
            LoTSSRelease::DR3,
            36_000.0,
            bins,
        )?);
    }
    if releases.is_empty() {
        bail!("Provide at least one of --dr1, --dr2, or --dr3");
    }
    let report = FluxDistributionReport {
        generated_at_utc: chrono::Utc::now().to_rfc3339(),
        bins,
        releases,
    };
    let out_path = output
        .map(PathBuf::from)
        .unwrap_or_else(default_flux_distribution_report_path);
    write_toml_report(&out_path, &report)?;
    println!("Report written to {}", out_path.display());
    Ok(())
}

fn build_flux_release_report(
    label: &str,
    path: &Path,
    release: LoTSSRelease,
    area_sq_deg: f64,
    bins: usize,
) -> Result<FluxReleaseReport> {
    println!("Loading {} fluxes from {}...", label, path.display());
    let mut flux = collect_positive_flux_values(path, release)?;
    if flux.is_empty() {
        bail!("No positive flux values found in {}", path.display());
    }
    flux.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let flux_min = flux[0];
    let flux_max = *flux.last().unwrap_or(&flux_min);
    let flux_median = percentile_sorted(&flux, 50.0);
    let (s_mid, dn_ds_norm) = differential_source_count(&flux, area_sq_deg, bins)?;
    let (gamma, gamma_err, intercept, intercept_err) = fit_power_law(&s_mid, &dn_ds_norm);
    Ok(FluxReleaseReport {
        release: label.to_string(),
        n_sources: flux.len(),
        area_sq_deg,
        flux_min_mjy: flux_min,
        flux_max_mjy: flux_max,
        flux_median_mjy: flux_median,
        power_law_gamma: gamma,
        power_law_gamma_err: gamma_err,
        fit_intercept: intercept,
        fit_intercept_err: intercept_err,
        n_fit_bins: s_mid.len(),
    })
}

fn collect_positive_flux_values(path: &Path, release: LoTSSRelease) -> Result<Vec<f64>> {
    let _ = release;
    read_positive_total_flux_column(path)
}

fn differential_source_count(
    flux_mjy: &[f64],
    area_sq_deg: f64,
    bins: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if flux_mjy.len() < 10 || bins < 4 {
        bail!(
            "Insufficient flux samples ({}) or bins ({bins})",
            flux_mjy.len()
        );
    }
    let omega_sr = area_sq_deg / ((180.0 / std::f64::consts::PI).powi(2));
    let log_min = flux_mjy
        .iter()
        .copied()
        .filter(|v| *v > 0.0)
        .map(f64::log10)
        .fold(f64::INFINITY, f64::min);
    let log_max = flux_mjy
        .iter()
        .copied()
        .filter(|v| *v > 0.0)
        .map(f64::log10)
        .fold(f64::NEG_INFINITY, f64::max);
    let step = (log_max - log_min) / bins as f64;
    let mut counts = vec![0usize; bins];
    for &value in flux_mjy {
        let mut idx = ((value.log10() - log_min) / step).floor() as isize;
        if idx < 0 {
            idx = 0;
        }
        if idx as usize >= bins {
            idx = bins as isize - 1;
        }
        counts[idx as usize] += 1;
    }
    let mut s_mid = Vec::new();
    let mut dn_ds_norm = Vec::new();
    for (idx, count) in counts.into_iter().enumerate() {
        if count < 5 {
            continue;
        }
        let left = 10f64.powf(log_min + idx as f64 * step);
        let right = 10f64.powf(log_min + (idx + 1) as f64 * step);
        let delta_s = right - left;
        let mid = 0.5 * (left + right);
        let dn_ds = count as f64 / (omega_sr * delta_s);
        s_mid.push(mid);
        dn_ds_norm.push(dn_ds * mid.powf(2.5));
    }
    Ok((s_mid, dn_ds_norm))
}

fn fit_power_law(s_mid: &[f64], dn_ds_norm: &[f64]) -> (f64, f64, f64, f64) {
    if s_mid.len() < 3 || s_mid.len() != dn_ds_norm.len() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let xs: Vec<f64> = s_mid.iter().copied().map(f64::log10).collect();
    let ys: Vec<f64> = dn_ds_norm
        .iter()
        .copied()
        .map(|value| (value.abs() + 1.0e-20).log10())
        .collect();
    let n = xs.len() as f64;
    let x_mean = xs.iter().sum::<f64>() / n;
    let y_mean = ys.iter().sum::<f64>() / n;
    let sxx = xs.iter().map(|x| (x - x_mean).powi(2)).sum::<f64>();
    if sxx <= 0.0 {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let sxy = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| (x - x_mean) * (y - y_mean))
        .sum::<f64>();
    let slope = sxy / sxx;
    let intercept = y_mean - slope * x_mean;
    if xs.len() < 3 {
        return (slope, f64::NAN, intercept, f64::NAN);
    }
    let sse = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| {
            let y_hat = intercept + slope * *x;
            (y - y_hat).powi(2)
        })
        .sum::<f64>();
    let sigma2 = sse / (n - 2.0);
    let slope_err = (sigma2 / sxx).sqrt();
    let intercept_err = (sigma2 * (1.0 / n + x_mean.powi(2) / sxx)).sqrt();
    (slope, slope_err, intercept, intercept_err)
}

fn cmd_kinematic_bisection(
    xmatch: &Path,
    rotcurves: &Path,
    output: Option<&Path>,
    inner_lo: f64,
    inner_hi: f64,
    n_bootstrap: usize,
) -> Result<()> {
    let xmatch_map = load_xmatch(xmatch)?;
    let rms_map = load_inner_rms(rotcurves, inner_lo, inner_hi)?;
    let mut loud_rms = Vec::new();
    let mut quiet_rms = Vec::new();
    let mut loud_flux = Vec::new();
    for (plateifu, info) in &xmatch_map {
        let Some(rms) = rms_map.get(plateifu) else {
            continue;
        };
        if info.detected {
            loud_rms.push(*rms);
            loud_flux.push(info.flux_mjy);
        } else {
            quiet_rms.push(*rms);
        }
    }
    if loud_rms.len() < 2 || quiet_rms.len() < 2 {
        bail!("Insufficient matched galaxies for statistical tests");
    }
    loud_rms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    quiet_rms.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let detection_fraction = loud_rms.len() as f64 / xmatch_map.len() as f64;
    let loud_median = percentile_sorted(&loud_rms, 50.0);
    let quiet_median = percentile_sorted(&quiet_rms, 50.0);
    let (mw_stat, mw_p) = mann_whitney_u_test(&loud_rms, &quiet_rms)?;
    let effect_dir = if loud_median > quiet_median {
        "loud > quiet"
    } else {
        "loud <= quiet"
    };
    let (ratio, ratio_lo, ratio_hi) = bootstrap_rms_ratio(&loud_rms, &quiet_rms, n_bootstrap, 42);
    let valid_pairs: Vec<(f64, f64)> = loud_flux
        .iter()
        .copied()
        .zip(loud_rms.iter().copied())
        .filter(|(flux, rms)| flux.is_finite() && rms.is_finite())
        .collect();
    let (pearson_r, pearson_p, pearson_n) = if valid_pairs.len() >= 3 {
        let xs: Vec<f64> = valid_pairs.iter().map(|(x, _)| *x).collect();
        let ys: Vec<f64> = valid_pairs.iter().map(|(_, y)| *y).collect();
        let r = pearson_r(&xs, &ys);
        let p = pearson_p_value(r, xs.len())?;
        (r, p, xs.len())
    } else {
        (f64::NAN, f64::NAN, 0)
    };
    let report = KinematicBisectionReport {
        generated_at_utc: chrono::Utc::now().to_rfc3339(),
        experiment_id: "E-198".to_string(),
        xmatch_path: xmatch.display().to_string(),
        rotcurves_path: rotcurves.display().to_string(),
        sample: BisectionSampleReport {
            n_manga_total: xmatch_map.len(),
            n_radio_loud: loud_rms.len(),
            n_radio_quiet: quiet_rms.len(),
            detection_fraction,
            inner_lo,
            inner_hi,
        },
        inner_core_rms: BisectionInnerRmsReport {
            rms_radio_loud_median: loud_median,
            rms_radio_quiet_median: quiet_median,
            rms_ratio_loud_over_quiet: ratio,
            rms_ratio_ci_lo: ratio_lo,
            rms_ratio_ci_hi: ratio_hi,
            n_bootstrap,
        },
        mann_whitney_u: MannWhitneyReport {
            statistic: mw_stat,
            p_value: mw_p,
            effect_direction: effect_dir.to_string(),
        },
        pearson_flux_rms: PearsonReport {
            r: pearson_r,
            p_value: pearson_p,
            n_pairs: pearson_n,
        },
    };
    let out_path = output
        .map(PathBuf::from)
        .unwrap_or_else(default_kinematic_bisection_report_path);
    write_toml_report(&out_path, &report)?;
    println!("Report written to {}", out_path.display());
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct XmatchInfo {
    detected: bool,
    flux_mjy: f64,
}

fn load_xmatch(path: &Path) -> Result<HashMap<String, XmatchInfo>> {
    let mut rdr =
        csv::Reader::from_path(path).with_context(|| format!("Cannot open {}", path.display()))?;
    let mut result = HashMap::new();
    for row in rdr.deserialize::<HashMap<String, String>>() {
        let row = row?;
        let plateifu = row
            .get("plateifu")
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
            .ok_or_else(|| anyhow!("Missing plateifu in {}", path.display()))?;
        let detected = row
            .get("lotss_detected")
            .map(|value| parse_truthy(value))
            .unwrap_or(false);
        let flux_mjy = row
            .get("lotss_flux_mjy")
            .and_then(|value| value.trim().parse::<f64>().ok())
            .unwrap_or(f64::NAN);
        result.insert(plateifu, XmatchInfo { detected, flux_mjy });
    }
    Ok(result)
}

fn load_inner_rms(path: &Path, inner_lo: f64, inner_hi: f64) -> Result<HashMap<String, f64>> {
    let mut rdr =
        csv::Reader::from_path(path).with_context(|| format!("Cannot open {}", path.display()))?;
    let headers = rdr.headers()?.clone();
    let plateifu_idx = headers
        .iter()
        .position(|header| header == "plateifu" || header == "name")
        .ok_or_else(|| anyhow!("No plateifu/name column in {}", path.display()))?;
    let x_idx = headers
        .iter()
        .position(|header| header == "x")
        .ok_or_else(|| {
            anyhow!(
                "No x column in {}. kinematic-bisection expects a normalized residual CSV \
                 with x and delta_v columns, not the raw MaNGA rotation-curve master CSV.",
                path.display()
            )
        })?;
    let delta_v_idx = headers
        .iter()
        .position(|header| header == "delta_v")
        .ok_or_else(|| {
            anyhow!(
                "No delta_v column in {}. kinematic-bisection expects a normalized residual CSV \
                 with x and delta_v columns, not the raw MaNGA rotation-curve master CSV.",
                path.display()
            )
        })?;
    let mut raw: HashMap<String, Vec<f64>> = HashMap::new();
    for record in rdr.records() {
        let record = record?;
        let Some(plateifu) = record.get(plateifu_idx).map(str::trim) else {
            continue;
        };
        if plateifu.is_empty() {
            continue;
        }
        let x = record
            .get(x_idx)
            .and_then(|value| value.trim().parse::<f64>().ok());
        let delta_v = record
            .get(delta_v_idx)
            .and_then(|value| value.trim().parse::<f64>().ok());
        let (Some(x), Some(delta_v)) = (x, delta_v) else {
            continue;
        };
        if inner_lo <= x && x <= inner_hi {
            raw.entry(plateifu.to_string()).or_default().push(delta_v);
        }
    }
    Ok(raw
        .into_iter()
        .filter_map(|(plateifu, values)| {
            if values.len() < 3 {
                return None;
            }
            let mean_sq =
                values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64;
            Some((plateifu, mean_sq.sqrt()))
        })
        .collect())
}

fn bootstrap_rms_ratio(
    loud_rms: &[f64],
    quiet_rms: &[f64],
    n_bootstrap: usize,
    seed: u64,
) -> (f64, f64, f64) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let observed = percentile_sorted(loud_rms, 50.0) / percentile_sorted(quiet_rms, 50.0);
    let mut ratios = Vec::with_capacity(n_bootstrap);
    for _ in 0..n_bootstrap {
        let loud_sample = bootstrap_sample(loud_rms, &mut rng);
        let quiet_sample = bootstrap_sample(quiet_rms, &mut rng);
        let denom = percentile_sorted(&quiet_sample, 50.0);
        if denom > 0.0 {
            ratios.push(percentile_sorted(&loud_sample, 50.0) / denom);
        }
    }
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    (
        observed,
        percentile_sorted(&ratios, 2.5),
        percentile_sorted(&ratios, 97.5),
    )
}

fn bootstrap_sample(values: &[f64], rng: &mut ChaCha8Rng) -> Vec<f64> {
    (0..values.len())
        .map(|_| {
            let idx = rng.random_range(0..values.len());
            values[idx]
        })
        .collect()
}

fn mann_whitney_u_test(sample_a: &[f64], sample_b: &[f64]) -> Result<(f64, f64)> {
    let n1 = sample_a.len();
    let n2 = sample_b.len();
    let total = n1 + n2;
    let mut combined: Vec<(f64, usize)> = sample_a
        .iter()
        .copied()
        .map(|value| (value, 0))
        .chain(sample_b.iter().copied().map(|value| (value, 1)))
        .collect();
    combined.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; total];
    let mut tie_counts = Vec::new();
    let mut idx = 0usize;
    while idx < combined.len() {
        let start = idx;
        let value = combined[idx].0;
        while idx < combined.len() && combined[idx].0 == value {
            idx += 1;
        }
        let mean_rank = (start + 1 + idx) as f64 / 2.0;
        for rank_slot in ranks.iter_mut().take(idx).skip(start) {
            *rank_slot = mean_rank;
        }
        tie_counts.push(idx - start);
    }
    let rank_sum_a = combined
        .iter()
        .zip(ranks.iter())
        .filter(|((_, group), _)| *group == 0)
        .map(|(_, rank)| *rank)
        .sum::<f64>();
    let u1 = rank_sum_a - (n1 * (n1 + 1) / 2) as f64;
    let mean_u = (n1 * n2) as f64 / 2.0;
    let tie_term = tie_counts
        .iter()
        .map(|&count| {
            let c = count as f64;
            c.powi(3) - c
        })
        .sum::<f64>();
    let total_f = total as f64;
    let variance =
        (n1 * n2) as f64 / 12.0 * ((total + 1) as f64 - tie_term / (total_f * (total_f - 1.0)));
    let sigma = variance.sqrt();
    let normal = Normal::new(0.0, 1.0)?;
    let continuity = if u1 > mean_u {
        0.5
    } else if u1 < mean_u {
        -0.5
    } else {
        0.0
    };
    let z = if sigma > 0.0 {
        (u1 - mean_u - continuity) / sigma
    } else {
        0.0
    };
    let p_value = 2.0 * (1.0 - normal.cdf(z.abs()));
    Ok((u1, p_value.clamp(0.0, 1.0)))
}

fn pearson_r(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let x_mean = xs.iter().sum::<f64>() / n;
    let y_mean = ys.iter().sum::<f64>() / n;
    let num = xs
        .iter()
        .zip(ys.iter())
        .map(|(x, y)| (x - x_mean) * (y - y_mean))
        .sum::<f64>();
    let den_x = xs.iter().map(|x| (x - x_mean).powi(2)).sum::<f64>().sqrt();
    let den_y = ys.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>().sqrt();
    if den_x == 0.0 || den_y == 0.0 {
        f64::NAN
    } else {
        num / (den_x * den_y)
    }
}

fn pearson_p_value(r: f64, n: usize) -> Result<f64> {
    if !r.is_finite() || n < 3 || r.abs() >= 1.0 {
        return Ok(f64::NAN);
    }
    let dof = (n - 2) as f64;
    let t = r.abs() * (dof / (1.0 - r * r)).sqrt();
    let student = StudentsT::new(0.0, 1.0, dof)?;
    Ok(2.0 * (1.0 - student.cdf(t)))
}

fn cmd_ultrametric(
    lotss: &Path,
    output: Option<&Path>,
    n_sample: usize,
    n_bootstrap: usize,
    n_triples: usize,
) -> Result<()> {
    println!("Loading LoTSS DR1 from {}...", lotss.display());
    let sources = load_from_fits(lotss, LoTSSRelease::DR1).map_err(|e| anyhow!(e.to_string()))?;
    if sources.len() < 3 {
        bail!("Need at least 3 sources for ultrametric analysis");
    }
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let sample_size = n_sample.min(sources.len());
    let sampled_indices = sample(&mut rng, sources.len(), sample_size).into_vec();
    let sample_ra: Vec<f64> = sampled_indices
        .iter()
        .map(|&idx| sources[idx].ra_deg)
        .collect();
    let sample_dec: Vec<f64> = sampled_indices
        .iter()
        .map(|&idx| sources[idx].dec_deg)
        .collect();
    let coords: Vec<(f64, f64)> = sample_ra
        .iter()
        .copied()
        .zip(sample_dec.iter().copied())
        .map(|(ra, dec)| projected_coords(ra, dec))
        .collect();
    let observed = p_ultra_vectorized(&coords, n_triples, &mut rng);
    let mut null_values = Vec::with_capacity(n_bootstrap);
    let mut ra_vals = sample_ra.clone();
    let mut dec_vals = sample_dec.clone();
    for _ in 0..n_bootstrap {
        ra_vals.shuffle(&mut rng);
        dec_vals.shuffle(&mut rng);
        let shuffled: Vec<(f64, f64)> = ra_vals
            .iter()
            .copied()
            .zip(dec_vals.iter().copied())
            .map(|(ra, dec)| projected_coords(ra, dec))
            .collect();
        null_values.push(p_ultra_vectorized(&shuffled, n_triples, &mut rng));
    }
    let null_mean = null_values.iter().sum::<f64>() / null_values.len() as f64;
    let null_std = {
        let var = null_values
            .iter()
            .map(|value| (value - null_mean).powi(2))
            .sum::<f64>()
            / null_values.len() as f64;
        var.sqrt()
    };
    let mut sorted_null = null_values.clone();
    sorted_null.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let report = UltrametricReport {
        generated_at_utc: chrono::Utc::now().to_rfc3339(),
        experiment_id: "E-199".to_string(),
        sample: UltrametricSampleReport {
            n_sample: coords.len(),
            n_triples_per_estimate: n_triples,
            n_bootstrap,
        },
        results: UltrametricResultsReport {
            p_ultra_observed: observed,
            p_ultra_null_mean: null_mean,
            p_ultra_null_std: null_std,
            p_ultra_ci_lo: percentile_sorted(&sorted_null, 2.5),
            p_ultra_ci_hi: percentile_sorted(&sorted_null, 97.5),
            z_score: if null_std > 0.0 {
                (observed - null_mean) / null_std
            } else {
                f64::NAN
            },
        },
    };
    let out_path = output
        .map(PathBuf::from)
        .unwrap_or_else(default_ultrametric_report_path);
    write_toml_report(&out_path, &report)?;
    println!("Report written to {}", out_path.display());
    Ok(())
}

fn projected_coords(ra_deg: f64, dec_deg: f64) -> (f64, f64) {
    (ra_deg * dec_deg.to_radians().cos(), dec_deg)
}

fn p_ultra_vectorized(points: &[(f64, f64)], n_triples: usize, rng: &mut ChaCha8Rng) -> f64 {
    if points.len() < 3 {
        return f64::NAN;
    }
    let mut ultra = 0usize;
    let mut valid = 0usize;
    for _ in 0..n_triples {
        let i = rng.random_range(0..points.len());
        let mut j = rng.random_range(0..points.len() - 1);
        if j >= i {
            j += 1;
        }
        let mut k = rng.random_range(0..points.len() - 2);
        if k >= i.min(j) {
            k += 1;
        }
        if k >= i.max(j) {
            k += 1;
        }
        let d_ab = euclidean_2d(points[i], points[j]);
        let d_bc = euclidean_2d(points[j], points[k]);
        let d_ac = euclidean_2d(points[i], points[k]);
        let mut dists = [d_ab, d_bc, d_ac];
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if dists[1].is_finite() && dists[2].is_finite() {
            if dists[2] <= dists[1] * (1.0 + ULTRA_TOLERANCE) {
                ultra += 1;
            }
            valid += 1;
        }
    }
    ultra as f64 / valid.max(1) as f64
}

fn euclidean_2d(a: (f64, f64), b: (f64, f64)) -> f64 {
    let dx = a.0 - b.0;
    let dy = a.1 - b.1;
    (dx * dx + dy * dy).sqrt()
}

fn percentile_sorted(values: &[f64], percentile: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let pct = percentile.clamp(0.0, 100.0) / 100.0;
    let rank = pct * (values.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        values[lower]
    } else {
        let weight = rank - lower as f64;
        values[lower] * (1.0 - weight) + values[upper] * weight
    }
}

fn parse_truthy(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "y"
    )
}

fn write_toml_report<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let body = toml::to_string_pretty(value)?;
    fs::write(path, body)?;
    Ok(())
}

fn read_positive_total_flux_column(path: &Path) -> Result<Vec<f64>> {
    use fitsio::{FitsFile, hdu::HduInfo};

    let mut fits = FitsFile::open(path).with_context(|| format!("FITS open {}", path.display()))?;
    let num_hdus = {
        let mut count = 0usize;
        for _ in fits.iter() {
            count += 1;
        }
        count
    };
    let mut table_idx = None;
    let mut total_flux_name = None;
    for idx in 1..num_hdus {
        let hdu = fits.hdu(idx)?;
        if let HduInfo::TableInfo {
            column_descriptions,
            ..
        } = hdu.info
        {
            table_idx = Some(idx);
            total_flux_name = column_descriptions
                .iter()
                .find(|desc| desc.name.eq_ignore_ascii_case("Total_flux"))
                .map(|desc| desc.name.clone());
            if total_flux_name.is_some() {
                break;
            }
        }
    }
    let table_idx =
        table_idx.ok_or_else(|| anyhow!("No BINTABLE HDU found in {}", path.display()))?;
    let total_flux_name = total_flux_name
        .ok_or_else(|| anyhow!("No Total_flux column found in {}", path.display()))?;
    let hdu = fits.hdu(table_idx)?;
    let flux: Vec<f32> = hdu.read_col(&mut fits, &total_flux_name)?;
    Ok(flux
        .into_iter()
        .map(|value| value as f64)
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect())
}

fn default_flux_distribution_report_path() -> PathBuf {
    PathBuf::from("reports").join(format!(
        "lotss_flux_distribution_{}.toml",
        chrono::Utc::now().date_naive()
    ))
}

fn default_kinematic_bisection_report_path() -> PathBuf {
    PathBuf::from("reports").join(format!(
        "lotss_manga_bisection_{}.toml",
        chrono::Utc::now().date_naive()
    ))
}

fn default_ultrametric_report_path() -> PathBuf {
    PathBuf::from("reports").join(format!(
        "lotss_ultrametric_dr1_{}.toml",
        chrono::Utc::now().date_naive()
    ))
}
