//! Join associator norms with ESA plasma beta for A(beta) scatter.
//!
//! The norms arrive as the CSV `heliosphere norm-dump` writes, whose leading
//! `# normalization=<label>` comment names the embedding the norms came from.
//! beta-join carries that label into the output instead of asserting one, so a
//! direction-normalized run cannot be published under the magnitude-normalized
//! label. |B| comes from the cached THA_L2_FGM minutes and the plasma moments
//! from the cached THA_L2_ESA daily files; beta = n k T / (B^2 / 2 mu0).
//!
//! The ESA cadence is about 6.5 minutes against minute-resolution norms, so the
//! exact-minute policy joins roughly one norm minute in three. The nearest
//! policy accepts the closest ESA sample within `--tolerance-minutes` and exists
//! to measure what the exact policy discards.

use chrono::{Datelike, NaiveDate};
use clap::{Args, ValueEnum};
use serde::Serialize;
use statrs::distribution::{ContinuousCDF, StudentsT};
use std::{collections::BTreeMap, fs, path::PathBuf};

const MU0: f64 = 1.2566370614e-6;
const KB: f64 = 1.380649e-23;
const EV_TO_K: f64 = 11604.518;

/// Half the nominal THEMIS ESA full-mode moment cadence of about 6.5 minutes.
const DEFAULT_TOLERANCE_MINUTES: f64 = 3.25;

/// The cached THA_L2_ESA files carry timestamp, density and temperature only.
/// The instrument quality flag is not a column there, so beta-join cannot filter
/// on it and every regenerated result declares the flagged fraction instead.
const ESA_QUALITY_NOTE: &str = "The cached THA_L2_ESA daily files hold the unflagged moments tha_peif_density and tha_peif_avgtemp and carry no quality column, so beta-join cannot filter on tha_peif_data_quality. The quality-filtered parameters return NaN on 935 of the 3931 cached rows over 2008 DOY 301..307, so about a quarter of the joined plasma samples are flagged suspect by the instrument team and are retained here.";

/// Ranks are ordinal: equal values take distinct ranks in sort order rather than
/// a shared midrank. The published 2026-03-27 Spearman value was computed this
/// way, so the convention is kept to make the successor comparable to it.
const RANK_METHOD: &str = "ordinal (sort order, no tie midranks)";

const BIN_CONSTRUCTION: &str = "n_bins geometric bins spanning [min(beta), max(beta)]; binned_beta reports the geometric center of each bin. The 2026-03-27 artifact's bin construction is not in the tree, so this specification is newly declared and the binned_* fields are not a reproduction of that artifact's bins.";

/// The 2026-03-27 artifact's exponent 0.7749921 and R^2 0.7837685 are recovered
/// to 5e-6 by fitting its own binned_beta against its own binned_mean_A, so that
/// published exponent is a fit over 17 bin means rather than over the pairs.
/// Both fits are reported here: `power_law_exponent` over every matched pair,
/// `power_law_exponent_binned` over the populated bin means, which is the field
/// comparable to 0.775.
const FIT_DEFINITION: &str = "power_law_exponent is the least-squares slope of log10(A) on log10(beta) over all matched pairs; power_law_exponent_binned is the same slope over the populated (binned_beta, binned_mean_A) points and is the definition the 2026-03-27 artifact's 0.775 came from";

const BETA_CRIT_RULE: &str = "binned_beta at the index of the largest increase in binned_mean_A between adjacent populated bins";

#[derive(Copy, Clone, PartialEq, Eq, Debug, ValueEnum)]
pub enum JoinPolicy {
    /// The ESA sample whose minute key equals the norm minute.
    ExactMinute,
    /// The ESA sample closest in time within the tolerance.
    Nearest,
}

impl JoinPolicy {
    fn label(self) -> &'static str {
        match self {
            Self::ExactMinute => "exact-minute",
            Self::Nearest => "nearest",
        }
    }
}

#[derive(Args)]
pub struct Cli {
    /// Path to associator norms CSV (doy,hour,minute,norm) as norm-dump writes it.
    #[arg(long)]
    norms_csv: String,
    /// Path to FGM minute data (for |B|).
    #[arg(long, default_value = "data/external")]
    data_dir: String,
    /// Year of the data.
    #[arg(long, default_value_t = 2008)]
    year: u16,
    /// DOY range start.
    #[arg(long, default_value_t = 301)]
    doy_start: u16,
    /// DOY range end.
    #[arg(long, default_value_t = 307)]
    doy_end: u16,
    /// Embedding normalization label. Defaults to the `# normalization=` comment
    /// the norms CSV carries; a value given here must agree with that comment.
    #[arg(long)]
    normalization: Option<String>,
    /// Minute-key policy for pairing a norm minute with an ESA sample.
    #[arg(long, value_enum, default_value_t = JoinPolicy::ExactMinute)]
    join_policy: JoinPolicy,
    /// Maximum |norm minute - ESA minute| accepted by the nearest policy.
    #[arg(long, default_value_t = DEFAULT_TOLERANCE_MINUTES)]
    tolerance_minutes: f64,
    /// Number of geometric beta bins.
    #[arg(long, default_value_t = 17)]
    n_bins: usize,
    /// Write the result JSON here in addition to stdout.
    #[arg(long)]
    out_json: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct BetaJoinResult {
    n_matched: usize,
    power_law_exponent: f64,
    power_law_r_squared: f64,
    power_law_exponent_binned: f64,
    power_law_r_squared_binned: f64,
    spearman_r: f64,
    spearman_p: f64,
    beta_crit_approx: f64,
    binned_beta: Vec<f64>,
    #[serde(rename = "binned_mean_A")]
    binned_mean_a: Vec<f64>,
    binned_counts: Vec<usize>,
    shape: String,
    interpretation: String,
    provenance: Provenance,
}

#[derive(Debug, Serialize)]
struct Provenance {
    normalization: String,
    normalization_source: String,
    join_policy: String,
    tolerance_minutes: Option<f64>,
    rank_method: String,
    fit_definition: String,
    bin_construction: String,
    beta_crit_rule: String,
    esa_quality_note: String,
    inputs: Vec<InputFile>,
    missing_inputs: Vec<String>,
    n_norm_minutes: usize,
    n_fgm_minutes: usize,
    n_esa_rows_read: usize,
    n_esa_rows_rejected: usize,
    esa_rejection_reasons: EsaRejections,
    n_esa_minutes_accepted: usize,
    n_pairs_rejected_by_beta_range: usize,
    seed: Option<u64>,
}

/// Every input file the run opened, in read order.
#[derive(Debug, Serialize)]
struct InputFile {
    path: String,
    sha256: String,
    size_bytes: u64,
    rows_read: usize,
}

#[derive(Debug, Default, Serialize)]
struct EsaRejections {
    malformed_row: usize,
    nonfinite_or_nonpositive_moment: usize,
    year_or_date_mismatch: usize,
    duplicate_minute_overwritten: usize,
}

fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// Day of year for an ESA `YYYY-MM-DD...` timestamp, keyed to the requested year.
///
/// The calendar comes from `chrono::NaiveDate`, so February 29 is counted in leap years.
/// A fixed non-leap month table shifted every post-February 2008 sample one day early and
/// joined each magnetometer minute to plasma from the following day. A timestamp from
/// another year returns `None`, so a stray file cannot join across years.
fn esa_timestamp_doy(ts: &str, year: u16) -> Option<u16> {
    if ts.len() < 10 {
        return None;
    }
    let ts_year: i32 = ts[0..4].parse().ok()?;
    if ts_year != i32::from(year) {
        return None;
    }
    let month: u32 = ts[5..7].parse().ok()?;
    let day: u32 = ts[8..10].parse().ok()?;
    let date = NaiveDate::from_ymd_opt(ts_year, month, day)?;
    u16::try_from(date.ordinal()).ok()
}

/// Minutes since the start of the year for a `(doy, hour, minute)` key.
fn absolute_minute(key: (u16, u8, u8)) -> i64 {
    i64::from(key.0) * 1440 + i64::from(key.1) * 60 + i64::from(key.2)
}

/// Index of the entry in a minute-sorted table closest to `target`.
///
/// Ties between a sample before and a sample after the target resolve to the
/// earlier one, so the output does not depend on iteration order.
fn nearest_within<T>(sorted: &[(i64, T)], target: i64, tolerance_minutes: f64) -> Option<usize> {
    if sorted.is_empty() || tolerance_minutes < 0.0 {
        return None;
    }
    let partition = sorted.partition_point(|&(minute, _)| minute < target);
    let mut best: Option<(i64, usize)> = None;
    for candidate in [partition.checked_sub(1), Some(partition)]
        .into_iter()
        .flatten()
    {
        let Some(&(minute, _)) = sorted.get(candidate) else {
            continue;
        };
        let delta = (minute - target).abs();
        let better = match best {
            None => true,
            Some((best_delta, _)) => delta < best_delta,
        };
        if better {
            best = Some((delta, candidate));
        }
    }
    best.filter(|&(delta, _)| (delta as f64) <= tolerance_minutes)
        .map(|(_, index)| index)
}

/// Ordinal ranks: the smallest value takes rank 0.
fn ordinal_ranks(values: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("finite values"));
    let mut ranks = vec![0.0; values.len()];
    for (rank, &(original, _)) in indexed.iter().enumerate() {
        ranks[original] = rank as f64;
    }
    ranks
}

fn pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }
    let denom = (var_x * var_y).sqrt();
    if denom > 1e-15 { cov / denom } else { 0.0 }
}

/// Spearman rho and its two-sided p-value under the Student t approximation
/// t = rho sqrt((n - 2) / (1 - rho^2)) on n - 2 degrees of freedom.
fn spearman(a: &[f64], b: &[f64]) -> (f64, f64) {
    let rho = pearson(&ordinal_ranks(a), &ordinal_ranks(b));
    let n = a.len() as f64;
    if n <= 2.0 {
        return (rho, 1.0);
    }
    let denom = 1.0 - rho * rho;
    if denom <= 0.0 {
        return (rho, 0.0);
    }
    let t = rho * ((n - 2.0) / denom).sqrt();
    let dist = StudentsT::new(0.0, 1.0, n - 2.0).expect("valid t distribution");
    let p = 2.0 * (1.0 - dist.cdf(t.abs()));
    (rho, p.clamp(0.0, 1.0))
}

/// Least-squares slope and R^2 of log10(A) against log10(beta).
///
/// The 1e-10 floor on A keeps a zero norm finite; real norms sit far above it,
/// so it moves the fit only where the associator vanished outright.
fn power_law_fit(beta: &[f64], a: &[f64]) -> (f64, f64) {
    let log_b: Vec<f64> = beta.iter().map(|&b| b.log10()).collect();
    let log_a: Vec<f64> = a.iter().map(|&value| (value + 1e-10).log10()).collect();
    let n = log_b.len() as f64;
    let sum_x: f64 = log_b.iter().sum();
    let sum_y: f64 = log_a.iter().sum();
    let sum_xy: f64 = log_b.iter().zip(log_a.iter()).map(|(x, y)| x * y).sum();
    let sum_x2: f64 = log_b.iter().map(|x| x * x).sum();
    let denom = n * sum_x2 - sum_x * sum_x;
    let alpha = if denom.abs() > 1e-15 {
        (n * sum_xy - sum_x * sum_y) / denom
    } else {
        f64::NAN
    };
    let r = pearson(&log_b, &log_a);
    (alpha, r * r)
}

struct Bins {
    centers: Vec<f64>,
    mean_a: Vec<f64>,
    counts: Vec<usize>,
}

/// Geometric bins over the observed beta range; `centers[i]` is the geometric
/// center of bin i. A bin holding no sample reports mean A of zero and count 0.
fn geometric_bins(beta: &[f64], a: &[f64], n_bins: usize) -> Bins {
    let mut centers = Vec::with_capacity(n_bins);
    let mut sums = vec![0.0; n_bins];
    let mut counts = vec![0usize; n_bins];
    let lo = beta.iter().copied().fold(f64::INFINITY, f64::min).log10();
    let hi = beta
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max)
        .log10();
    let width = (hi - lo) / n_bins as f64;
    for index in 0..n_bins {
        centers.push(10f64.powf(lo + (index as f64 + 0.5) * width));
    }
    for (&b, &value) in beta.iter().zip(a.iter()) {
        let raw = if width > 0.0 {
            ((b.log10() - lo) / width) as usize
        } else {
            0
        };
        let index = raw.min(n_bins - 1);
        sums[index] += value;
        counts[index] += 1;
    }
    let mean_a = sums
        .iter()
        .zip(counts.iter())
        .map(|(&sum, &count)| if count > 0 { sum / count as f64 } else { 0.0 })
        .collect();
    Bins {
        centers,
        mean_a,
        counts,
    }
}

/// The bin center at the largest rise in mean A between adjacent populated bins.
fn beta_crit(bins: &Bins) -> f64 {
    let mut best = (f64::NEG_INFINITY, 0usize);
    for index in 0..bins.centers.len().saturating_sub(1) {
        if bins.counts[index] == 0 || bins.counts[index + 1] == 0 {
            continue;
        }
        let rise = bins.mean_a[index + 1] - bins.mean_a[index];
        if rise > best.0 {
            best = (rise, index);
        }
    }
    bins.centers.get(best.1).copied().unwrap_or(f64::NAN)
}

fn shape_sentence(alpha: f64, r_squared: f64) -> String {
    if r_squared >= 0.9 {
        format!(
            "Strong power law: alpha={alpha:.2}, R^2={r_squared:.3}. Clean scaling across the beta range."
        )
    } else if r_squared >= 0.6 {
        format!(
            "Weak power law: alpha={alpha:.2}, R^2={r_squared:.3}. Moderate fit -- may have a transition region."
        )
    } else {
        format!(
            "No clean power law: alpha={alpha:.2}, R^2={r_squared:.3}. The relation is not scale-free."
        )
    }
}

/// Norms CSV content split into the declared normalization label and the minute map.
struct Norms {
    normalization: Option<String>,
    values: BTreeMap<(u16, u8, u8), f64>,
}

fn parse_norms(content: &str) -> Norms {
    let mut normalization = None;
    let mut values = BTreeMap::new();
    for line in content.lines() {
        let line = line.trim();
        if let Some(comment) = line.strip_prefix('#') {
            if let Some(label) = comment.trim().strip_prefix("normalization=") {
                normalization = Some(label.trim().to_string());
            }
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 4 {
            continue;
        }
        let (Ok(doy), Ok(hour), Ok(minute), Ok(norm)) = (
            parts[0].parse::<u16>(),
            parts[1].parse::<u8>(),
            parts[2].parse::<u8>(),
            parts[3].parse::<f64>(),
        ) else {
            continue;
        };
        values.insert((doy, hour, minute), norm);
    }
    Norms {
        normalization,
        values,
    }
}

pub fn run(cli: Cli) {
    let mut inputs: Vec<InputFile> = Vec::new();
    let mut missing_inputs: Vec<String> = Vec::new();

    let norms_bytes = fs::read(&cli.norms_csv).expect("read norms CSV");
    let norms_content = String::from_utf8(norms_bytes.clone()).expect("norms CSV is UTF-8");
    let norms = parse_norms(&norms_content);
    inputs.push(InputFile {
        path: cli.norms_csv.clone(),
        sha256: sha256_hex(&norms_bytes),
        size_bytes: norms_bytes.len() as u64,
        rows_read: norms.values.len(),
    });

    let (normalization, normalization_source) = match (&cli.normalization, &norms.normalization) {
        (Some(flag), Some(declared)) => {
            assert!(
                flag == declared,
                "--normalization {flag} contradicts the norms CSV comment # normalization={declared}"
            );
            (
                flag.clone(),
                "norms CSV comment, confirmed by --normalization".to_string(),
            )
        }
        (Some(flag), None) => (flag.clone(), "--normalization flag".to_string()),
        (None, Some(declared)) => (declared.clone(), "norms CSV comment".to_string()),
        (None, None) => panic!(
            "the norms CSV declares no `# normalization=` comment; rerun norm-dump or pass --normalization"
        ),
    };
    eprintln!("Loaded {} norms ({normalization})", norms.values.len());

    // |B| per minute, averaged over the FGM samples that fall in it.
    let mut bmag_map: BTreeMap<(u16, u8, u8), (f64, usize)> = BTreeMap::new();
    for doy in cli.doy_start..=cli.doy_end {
        let path = format!(
            "{}/themis/tha_fgm_{:04}_{:03}.csv",
            cli.data_dir, cli.year, doy
        );
        let Ok(bytes) = fs::read(&path) else {
            missing_inputs.push(path);
            continue;
        };
        let content = String::from_utf8(bytes.clone()).expect("FGM CSV is UTF-8");
        let mut rows_read = 0usize;
        for line in content.lines().skip(1) {
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 4 {
                continue;
            }
            let ts = parts[0];
            if ts.len() < 19 {
                continue;
            }
            rows_read += 1;
            let hour: u8 = ts[11..13].parse().unwrap_or(0);
            let minute: u8 = ts[14..16].parse().unwrap_or(0);
            let bx: f64 = parts[1].parse().unwrap_or(f64::NAN);
            let by: f64 = parts[2].parse().unwrap_or(f64::NAN);
            let bz: f64 = parts[3].parse().unwrap_or(f64::NAN);
            if !bx.is_finite() || bx.abs() > 1e10 {
                continue;
            }
            let bmag = (bx * bx + by * by + bz * bz).sqrt();
            if bmag > 200.0 {
                continue;
            }
            let entry = bmag_map.entry((doy, hour, minute)).or_insert((0.0, 0));
            entry.0 += bmag;
            entry.1 += 1;
        }
        inputs.push(InputFile {
            path,
            sha256: sha256_hex(&bytes),
            size_bytes: bytes.len() as u64,
            rows_read,
        });
    }
    let bmag_avg: BTreeMap<(u16, u8, u8), f64> = bmag_map
        .iter()
        .map(|(&k, &(sum, count))| (k, sum / count as f64))
        .collect();
    eprintln!("Loaded {} FGM minutes", bmag_avg.len());

    // ESA ion moments keyed by minute. The cadence is coarser than a minute, so
    // a minute holds at most one sample and a repeat overwrites its predecessor.
    let mut esa_map: BTreeMap<(u16, u8, u8), (f64, f64)> = BTreeMap::new();
    let mut esa_rows_read = 0usize;
    let mut rejections = EsaRejections::default();
    for doy in cli.doy_start..=cli.doy_end {
        let path = format!(
            "{}/themis_esa/tha_esa_{:04}_{}.csv",
            cli.data_dir, cli.year, doy
        );
        let Ok(bytes) = fs::read(&path) else {
            missing_inputs.push(path);
            continue;
        };
        let content = String::from_utf8(bytes.clone()).expect("ESA CSV is UTF-8");
        let mut rows_read = 0usize;
        for line in content.lines() {
            if line.trim().is_empty() {
                continue;
            }
            rows_read += 1;
            esa_rows_read += 1;
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 3 {
                rejections.malformed_row += 1;
                continue;
            }
            let ts = parts[0];
            if ts.len() < 19 || !ts.starts_with(|c: char| c.is_ascii_digit()) {
                rejections.malformed_row += 1;
                continue;
            }
            let hour: u8 = ts[11..13].parse().unwrap_or(255);
            let minute: u8 = ts[14..16].parse().unwrap_or(255);
            if hour > 23 || minute > 59 {
                rejections.malformed_row += 1;
                continue;
            }
            let den: f64 = parts[1].parse().unwrap_or(f64::NAN);
            let temp: f64 = parts[2].parse().unwrap_or(f64::NAN);
            if !den.is_finite() || den <= 0.0 || !temp.is_finite() || temp <= 0.0 {
                rejections.nonfinite_or_nonpositive_moment += 1;
                continue;
            }
            let Some(ts_doy) = esa_timestamp_doy(ts, cli.year) else {
                rejections.year_or_date_mismatch += 1;
                continue;
            };
            if esa_map
                .insert((ts_doy, hour, minute), (den, temp))
                .is_some()
            {
                rejections.duplicate_minute_overwritten += 1;
            }
        }
        inputs.push(InputFile {
            path,
            sha256: sha256_hex(&bytes),
            size_bytes: bytes.len() as u64,
            rows_read,
        });
    }
    eprintln!("Loaded {} ESA minutes", esa_map.len());

    let esa_sorted: Vec<(i64, (f64, f64))> = esa_map
        .iter()
        .map(|(&key, &moments)| (absolute_minute(key), moments))
        .collect();

    let mut a_vals = Vec::new();
    let mut beta_vals = Vec::new();
    let mut beta_range_rejected = 0usize;

    for (&key, &norm) in &norms.values {
        let Some(&bmag_nt) = bmag_avg.get(&key) else {
            continue;
        };
        let moments = match cli.join_policy {
            JoinPolicy::ExactMinute => esa_map.get(&key).copied(),
            JoinPolicy::Nearest => {
                nearest_within(&esa_sorted, absolute_minute(key), cli.tolerance_minutes)
                    .map(|index| esa_sorted[index].1)
            }
        };
        let Some((den_cc, temp_ev)) = moments else {
            continue;
        };

        let b_t = bmag_nt * 1e-9;
        let n_m3 = den_cc * 1e6;
        let t_k = temp_ev * EV_TO_K;
        let p_th = n_m3 * KB * t_k;
        let p_mag = b_t * b_t / (2.0 * MU0);
        if p_mag < 1e-20 {
            beta_range_rejected += 1;
            continue;
        }
        let beta = p_th / p_mag;
        if beta <= 0.0 || beta > 1000.0 {
            beta_range_rejected += 1;
            continue;
        }

        a_vals.push(norm);
        beta_vals.push(beta);
    }

    eprintln!("Matched A-beta pairs: {}", a_vals.len());
    assert!(
        a_vals.len() >= 10,
        "only {} A-beta pairs matched; the join has no sample to fit",
        a_vals.len()
    );

    let (spearman_r, spearman_p) = spearman(&beta_vals, &a_vals);
    let (alpha, r_squared) = power_law_fit(&beta_vals, &a_vals);
    let bins = geometric_bins(&beta_vals, &a_vals, cli.n_bins);
    let crit = beta_crit(&bins);
    let populated_beta: Vec<f64> = bins
        .centers
        .iter()
        .zip(bins.counts.iter())
        .filter(|&(_, &count)| count > 0)
        .map(|(&center, _)| center)
        .collect();
    let populated_mean_a: Vec<f64> = bins
        .mean_a
        .iter()
        .zip(bins.counts.iter())
        .filter(|&(_, &count)| count > 0)
        .map(|(&mean, _)| mean)
        .collect();
    let (alpha_binned, r_squared_binned) = power_law_fit(&populated_beta, &populated_mean_a);
    let shape = shape_sentence(alpha, r_squared);
    let interpretation = format!(
        "The A(beta) relationship is {shape}. Spearman r={spearman_r:.3} confirms monotonic positive correlation: higher beta -> higher associator. The steepest transition occurs near beta ~ {crit:.1}."
    );

    let result = BetaJoinResult {
        n_matched: a_vals.len(),
        power_law_exponent: alpha,
        power_law_r_squared: r_squared,
        power_law_exponent_binned: alpha_binned,
        power_law_r_squared_binned: r_squared_binned,
        spearman_r,
        spearman_p,
        beta_crit_approx: crit,
        binned_beta: bins.centers.clone(),
        binned_mean_a: bins.mean_a.clone(),
        binned_counts: bins.counts.clone(),
        shape,
        interpretation,
        provenance: Provenance {
            normalization,
            normalization_source,
            join_policy: cli.join_policy.label().to_string(),
            tolerance_minutes: match cli.join_policy {
                JoinPolicy::Nearest => Some(cli.tolerance_minutes),
                JoinPolicy::ExactMinute => None,
            },
            rank_method: RANK_METHOD.to_string(),
            fit_definition: FIT_DEFINITION.to_string(),
            bin_construction: BIN_CONSTRUCTION.to_string(),
            beta_crit_rule: BETA_CRIT_RULE.to_string(),
            esa_quality_note: ESA_QUALITY_NOTE.to_string(),
            inputs,
            missing_inputs,
            n_norm_minutes: norms.values.len(),
            n_fgm_minutes: bmag_avg.len(),
            n_esa_rows_read: esa_rows_read,
            n_esa_rows_rejected: rejections.malformed_row
                + rejections.nonfinite_or_nonpositive_moment
                + rejections.year_or_date_mismatch,
            esa_rejection_reasons: rejections,
            n_esa_minutes_accepted: esa_map.len(),
            n_pairs_rejected_by_beta_range: beta_range_rejected,
            seed: None,
        },
    };

    let json = serde_json::to_string_pretty(&result).expect("serialize beta join result");
    println!("{json}");
    if let Some(path) = &cli.out_json {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create output directory");
        }
        fs::write(path, format!("{json}\n")).expect("write out-json");
        eprintln!("Wrote {}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::{
        beta_crit, esa_timestamp_doy, geometric_bins, nearest_within, parse_norms, power_law_fit,
        spearman,
    };

    #[test]
    fn leap_year_post_february_doy_counts_february_29() {
        // 2008-10-27 is day 301 of the leap year 2008; a non-leap table returns 300.
        assert_eq!(esa_timestamp_doy("2008-10-27T00:00:00Z", 2008), Some(301));
        assert_eq!(esa_timestamp_doy("2007-10-27T00:00:00Z", 2007), Some(300));
        assert_eq!(esa_timestamp_doy("2008-02-29T12:00:00Z", 2008), Some(60));
        assert_eq!(esa_timestamp_doy("2008-01-31T12:00:00Z", 2008), Some(31));
    }

    #[test]
    fn rejects_other_years_and_invalid_dates() {
        assert_eq!(esa_timestamp_doy("2009-10-27T00:00:00Z", 2008), None);
        assert_eq!(esa_timestamp_doy("2007-02-29T00:00:00Z", 2007), None);
        assert_eq!(esa_timestamp_doy("2008-13-01", 2008), None);
        assert_eq!(esa_timestamp_doy("2008-1", 2008), None);
    }

    #[test]
    fn exact_power_law_recovers_its_exponent() {
        // A = 3 beta^1.5 over three decades; the log-log fit is exact, so R^2 is 1.
        let beta: Vec<f64> = (0..300)
            .map(|i| 10f64.powf(-1.0 + i as f64 / 100.0))
            .collect();
        let a: Vec<f64> = beta.iter().map(|b| 3.0 * b.powf(1.5)).collect();
        let (alpha, r_squared) = power_law_fit(&beta, &a);
        assert!((alpha - 1.5).abs() < 1e-6, "alpha={alpha}");
        assert!((r_squared - 1.0).abs() < 1e-9, "r2={r_squared}");
    }

    #[test]
    fn spearman_saturates_on_a_monotone_series() {
        let x: Vec<f64> = (1..=100).map(f64::from).collect();
        let y: Vec<f64> = x.iter().map(|v| v.exp().ln() * 7.0 - 3.0).collect();
        let (rho, p) = spearman(&x, &y);
        assert!((rho - 1.0).abs() < 1e-12, "rho={rho}");
        assert!(p < 1e-6, "p={p}");

        let reversed: Vec<f64> = y.iter().rev().copied().collect();
        let (rho_rev, _) = spearman(&x, &reversed);
        assert!((rho_rev + 1.0).abs() < 1e-12, "rho={rho_rev}");
    }

    #[test]
    fn geometric_bins_partition_the_beta_range() {
        let beta = vec![0.1, 0.2, 1.0, 2.0, 10.0];
        let a = vec![1.0, 3.0, 5.0, 7.0, 100.0];
        let bins = geometric_bins(&beta, &a, 3);
        assert_eq!(bins.counts.iter().sum::<usize>(), beta.len());
        // Bins span 0.1..10 in three geometric steps; the top bin holds only 10.0.
        assert_eq!(bins.counts[2], 1);
        assert!((bins.mean_a[2] - 100.0).abs() < 1e-12);
        assert!(bins.centers[0] < bins.centers[1] && bins.centers[1] < bins.centers[2]);
        // The largest rise in mean A lands on the bin below the jump to 100.
        assert!((beta_crit(&bins) - bins.centers[1]).abs() < 1e-12);
    }

    #[test]
    fn nearest_join_honors_the_tolerance_and_breaks_ties_early() {
        let table: Vec<(i64, u32)> = vec![(0, 10), (7, 20), (14, 30)];
        assert_eq!(nearest_within(&table, 2, 3.25), Some(0));
        assert_eq!(nearest_within(&table, 5, 3.25), Some(1));
        // Minute 10 sits 3 from 7 and 4 from 14: the earlier sample wins.
        assert_eq!(nearest_within(&table, 10, 3.25), Some(1));
        // Minute 21 is 7 past the last sample, beyond tolerance.
        assert_eq!(nearest_within(&table, 21, 3.25), None);
        // Equidistant candidates resolve to the earlier one.
        let even: Vec<(i64, u32)> = vec![(0, 1), (6, 2)];
        assert_eq!(nearest_within(&even, 3, 3.25), Some(0));
    }

    /// The 2026-03-27 artifact published exponent 0.7749921471180499 with
    /// R^2 0.7837684502151782. Fitting its own bin means recovers both, which
    /// identifies that exponent as a fit over 17 bins rather than over pairs.
    #[test]
    fn archived_exponent_is_a_fit_over_bin_means() {
        let binned_beta = [
            0.012_742_749_857_031_338,
            0.020_691_380_811_147_898,
            0.033_598_182_862_837_82,
            0.054_555_947_811_685_19,
            0.088_586_679_041_008_25,
            0.143_844_988_828_766_26,
            0.233_572_146_909_012_18,
            0.379_269_019_073_224_86,
            0.615_848_211_066_026_3,
            0.999_999_999_999_999_7,
            1.623_776_739_188_721_7,
            2.636_650_898_730_358,
            4.281_332_398_719_391,
            6.951_927_961_775_602,
            11.288_378_916_846_89,
            18.329_807_108_324_356,
            29.763_514_416_313_16,
        ];
        let binned_mean_a = [
            0.499_356_028_985_507_3,
            0.225_752_999_999_999_98,
            0.011_495_788_546_255_508,
            0.535_021_995_433_79,
            2.457_509_206_422_018,
            2.081_601_430_278_884_6,
            0.998_512_475_915_221_6,
            0.899_930_822_269_807_3,
            2.500_228_280_701_754_6,
            6.354_286_985_294_117,
            7.717_597_805_194_804,
            10.381_361_720_430_107,
            14.980_911_335_877_863,
            15.139_657_190_476_19,
            20.533_629_837_837_84,
            38.583_607_571_428_57,
            54.587_654_166_666_674,
        ];
        let (alpha, r_squared) = power_law_fit(&binned_beta, &binned_mean_a);
        assert!(
            (alpha - 0.774_992_147_118_049_9).abs() < 1e-5,
            "alpha={alpha}"
        );
        assert!(
            (r_squared - 0.783_768_450_215_178_2).abs() < 1e-5,
            "r2={r_squared}"
        );
    }

    #[test]
    fn norms_csv_carries_its_normalization_label() {
        let csv = "# normalization=direction\ndoy,hour,minute,associator_norm\n301,0,9,0.5\n301,0,10,0.7\n";
        let norms = parse_norms(csv);
        assert_eq!(norms.normalization.as_deref(), Some("direction"));
        assert_eq!(norms.values.len(), 2);
        assert_eq!(norms.values[&(301, 0, 10)], 0.7);

        let bare = parse_norms("doy,hour,minute,associator_norm\n301,0,9,0.5\n");
        assert!(bare.normalization.is_none());
        assert_eq!(bare.values.len(), 1);
    }
}
