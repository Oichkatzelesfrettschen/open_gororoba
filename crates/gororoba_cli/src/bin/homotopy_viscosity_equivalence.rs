use std::fs;
use std::path::{Path, PathBuf};

use clap::Parser;
use csv::StringRecord;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(
    name = "homotopy-viscosity-equivalence",
    about = "Compare two dynamics curves via lagged correlation and DTW."
)]
struct Args {
    #[arg(long)]
    loss_csv: PathBuf,

    #[arg(long)]
    viscosity_csv: PathBuf,

    #[arg(long, default_value = "loss")]
    loss_column: String,

    #[arg(long, default_value = "value")]
    viscosity_column: String,

    #[arg(long, default_value_t = 64)]
    max_lag: usize,

    #[arg(long, default_value_t = 500)]
    permutations: usize,

    #[arg(long, default_value_t = 42)]
    seed: u64,

    #[arg(
        long,
        default_value = "data/equivalence/homotopy_viscosity_equivalence.toml"
    )]
    output: PathBuf,
}

#[derive(Debug, Serialize)]
struct EquivalenceMetadata {
    loss_csv: String,
    viscosity_csv: String,
    loss_column: String,
    viscosity_column: String,
    max_lag: usize,
    permutations: usize,
    seed: u64,
}

#[derive(Debug, Serialize)]
struct CurveSummary {
    raw_length: usize,
    aligned_length: usize,
    mean: f64,
    std_dev: f64,
    min: f64,
    max: f64,
}

#[derive(Debug, Serialize)]
struct EquivalenceMetrics {
    best_lag: isize,
    best_lagged_correlation: f64,
    dtw_distance: f64,
    normalized_dtw_distance: f64,
    permutation_pvalue: f64,
    null_mean_abs_correlation: f64,
}

#[derive(Debug, Serialize)]
struct EquivalenceDecision {
    accepted_as_isomorphic: bool,
    correlation_threshold: f64,
    pvalue_threshold: f64,
    dtw_threshold: f64,
    reason: String,
}

#[derive(Debug, Serialize)]
struct EquivalenceReport {
    hypothesis_lane: String,
    experiment_id: String,
    metadata: EquivalenceMetadata,
    loss_curve: CurveSummary,
    viscosity_curve: CurveSummary,
    metrics: EquivalenceMetrics,
    decision: EquivalenceDecision,
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn std_dev(values: &[f64], mu: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let var = values
        .iter()
        .map(|v| {
            let d = v - mu;
            d * d
        })
        .sum::<f64>()
        / (values.len() as f64 - 1.0);
    var.sqrt()
}

fn zscore(values: &[f64]) -> Vec<f64> {
    let mu = mean(values);
    let sd = std_dev(values, mu);
    if sd <= 0.0 {
        return vec![0.0; values.len()];
    }
    values.iter().map(|v| (v - mu) / sd).collect()
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mu_a = mean(a);
    let mu_b = mean(b);
    let mut num = 0.0;
    let mut den_a = 0.0;
    let mut den_b = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let dx = x - mu_a;
        let dy = y - mu_b;
        num += dx * dy;
        den_a += dx * dx;
        den_b += dy * dy;
    }
    if den_a <= 0.0 || den_b <= 0.0 {
        return 0.0;
    }
    num / (den_a.sqrt() * den_b.sqrt())
}

fn overlap_for_lag<'a>(a: &'a [f64], b: &'a [f64], lag: isize) -> (&'a [f64], &'a [f64]) {
    if a.is_empty() || b.is_empty() {
        return (&[], &[]);
    }
    if lag >= 0 {
        let lag_u = lag as usize;
        if lag_u >= b.len() {
            return (&[], &[]);
        }
        let n = a.len().min(b.len() - lag_u);
        (&a[..n], &b[lag_u..lag_u + n])
    } else {
        let lag_u = (-lag) as usize;
        if lag_u >= a.len() {
            return (&[], &[]);
        }
        let n = (a.len() - lag_u).min(b.len());
        (&a[lag_u..lag_u + n], &b[..n])
    }
}

fn best_lagged_correlation(a: &[f64], b: &[f64], max_lag: usize) -> (isize, f64) {
    let mut best_lag = 0isize;
    let mut best_corr = f64::NEG_INFINITY;
    for lag in -(max_lag as isize)..=(max_lag as isize) {
        let (x, y) = overlap_for_lag(a, b, lag);
        if x.len() < 3 {
            continue;
        }
        let c = pearson(x, y);
        if c > best_corr {
            best_corr = c;
            best_lag = lag;
        }
    }
    if !best_corr.is_finite() {
        (0, 0.0)
    } else {
        (best_lag, best_corr)
    }
}

fn dtw_distance(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let n = a.len();
    let m = b.len();
    let mut dp = vec![vec![f64::INFINITY; m + 1]; n + 1];
    dp[0][0] = 0.0;
    for i in 1..=n {
        for j in 1..=m {
            let cost = (a[i - 1] - b[j - 1]).abs();
            let prev = dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
            dp[i][j] = cost + prev;
        }
    }
    dp[n][m]
}

fn resample_to_len(values: &[f64], out_len: usize) -> Vec<f64> {
    if values.is_empty() || out_len == 0 {
        return Vec::new();
    }
    if values.len() == out_len {
        return values.to_vec();
    }
    if values.len() == 1 {
        return vec![values[0]; out_len];
    }
    let in_last = (values.len() - 1) as f64;
    let out_last = (out_len - 1) as f64;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = (i as f64) * in_last / out_last;
        let left = pos.floor() as usize;
        let right = pos.ceil() as usize;
        if left == right {
            out.push(values[left]);
            continue;
        }
        let alpha = pos - left as f64;
        out.push(values[left] * (1.0 - alpha) + values[right] * alpha);
    }
    out
}

fn parse_numeric_cell(cell: &str) -> Option<f64> {
    let trimmed = cell.trim();
    if trimmed.is_empty() {
        return None;
    }
    trimmed.parse::<f64>().ok()
}

fn column_index(headers: &StringRecord, wanted: &str) -> Option<usize> {
    headers.iter().position(|h| h.eq_ignore_ascii_case(wanted))
}

fn read_curve(path: &Path, preferred_column: &str) -> Result<Vec<f64>, String> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| format!("failed to open CSV {}: {e}", path.display()))?;

    let headers = rdr
        .headers()
        .map_err(|e| format!("failed to read headers {}: {e}", path.display()))?
        .clone();

    let preferred_idx = column_index(&headers, preferred_column);
    let mut out = Vec::new();

    for row in rdr.records() {
        let rec = row.map_err(|e| format!("failed reading row {}: {e}", path.display()))?;
        let value = if let Some(idx) = preferred_idx {
            rec.get(idx).and_then(parse_numeric_cell)
        } else {
            rec.iter().find_map(parse_numeric_cell)
        };
        if let Some(v) = value {
            out.push(v);
        }
    }

    if out.is_empty() {
        return Err(format!(
            "no numeric values extracted from {} (column hint: {})",
            path.display(),
            preferred_column
        ));
    }
    Ok(out)
}

fn min_max(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let mut min_v = f64::INFINITY;
    let mut max_v = f64::NEG_INFINITY;
    for &v in values {
        min_v = min_v.min(v);
        max_v = max_v.max(v);
    }
    (min_v, max_v)
}

fn permutation_pvalue(
    a: &[f64],
    b: &[f64],
    observed_abs_corr: f64,
    permutations: usize,
    seed: u64,
) -> (f64, f64) {
    if permutations == 0 {
        return (1.0, 0.0);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut exceed = 0usize;
    let mut null_sum = 0.0;
    let mut shuffled = b.to_vec();
    for _ in 0..permutations {
        shuffled.shuffle(&mut rng);
        let c = pearson(a, &shuffled).abs();
        null_sum += c;
        if c >= observed_abs_corr {
            exceed += 1;
        }
    }
    let p = (exceed as f64 + 1.0) / (permutations as f64 + 1.0);
    let null_mean = null_sum / permutations as f64;
    (p, null_mean)
}

fn build_report(args: &Args) -> Result<EquivalenceReport, String> {
    let loss_raw = read_curve(&args.loss_csv, &args.loss_column)?;
    let visc_raw = read_curve(&args.viscosity_csv, &args.viscosity_column)?;

    let aligned_len = loss_raw.len().min(visc_raw.len()).max(16);
    let loss_aligned = resample_to_len(&loss_raw, aligned_len);
    let visc_aligned = resample_to_len(&visc_raw, aligned_len);

    let loss_z = zscore(&loss_aligned);
    let visc_z = zscore(&visc_aligned);

    let (best_lag, best_corr) = best_lagged_correlation(&loss_z, &visc_z, args.max_lag);
    let dtw = dtw_distance(&loss_z, &visc_z);
    let dtw_norm = if aligned_len == 0 {
        0.0
    } else {
        dtw / aligned_len as f64
    };

    let (pvalue, null_mean_abs) = permutation_pvalue(
        &loss_z,
        &visc_z,
        best_corr.abs(),
        args.permutations,
        args.seed,
    );

    let corr_threshold = 0.70;
    let pvalue_threshold = 0.05;
    let dtw_threshold = 0.80;
    let accepted = best_corr.abs() >= corr_threshold
        && pvalue <= pvalue_threshold
        && dtw_norm <= dtw_threshold;

    let decision_reason = if accepted {
        "Accepted: strong lagged correlation with significant permutation p-value and low normalized DTW."
    } else {
        "Rejected: one or more criteria failed (correlation, p-value, or normalized DTW)."
    };

    let (loss_min, loss_max) = min_max(&loss_aligned);
    let (visc_min, visc_max) = min_max(&visc_aligned);
    let loss_mean = mean(&loss_aligned);
    let visc_mean = mean(&visc_aligned);
    let loss_sd = std_dev(&loss_aligned, loss_mean);
    let visc_sd = std_dev(&visc_aligned, visc_mean);

    Ok(EquivalenceReport {
        hypothesis_lane: "runtime_type_inference_hott".to_string(),
        experiment_id: "EXP-HVE-001".to_string(),
        metadata: EquivalenceMetadata {
            loss_csv: args.loss_csv.display().to_string(),
            viscosity_csv: args.viscosity_csv.display().to_string(),
            loss_column: args.loss_column.clone(),
            viscosity_column: args.viscosity_column.clone(),
            max_lag: args.max_lag,
            permutations: args.permutations,
            seed: args.seed,
        },
        loss_curve: CurveSummary {
            raw_length: loss_raw.len(),
            aligned_length: aligned_len,
            mean: loss_mean,
            std_dev: loss_sd,
            min: loss_min,
            max: loss_max,
        },
        viscosity_curve: CurveSummary {
            raw_length: visc_raw.len(),
            aligned_length: aligned_len,
            mean: visc_mean,
            std_dev: visc_sd,
            min: visc_min,
            max: visc_max,
        },
        metrics: EquivalenceMetrics {
            best_lag,
            best_lagged_correlation: best_corr,
            dtw_distance: dtw,
            normalized_dtw_distance: dtw_norm,
            permutation_pvalue: pvalue,
            null_mean_abs_correlation: null_mean_abs,
        },
        decision: EquivalenceDecision {
            accepted_as_isomorphic: accepted,
            correlation_threshold: corr_threshold,
            pvalue_threshold,
            dtw_threshold,
            reason: decision_reason.to_string(),
        },
    })
}

fn main() -> Result<(), String> {
    let args = Args::parse();
    let report = build_report(&args)?;
    let text =
        toml::to_string_pretty(&report).map_err(|e| format!("failed to serialize TOML: {e}"))?;
    if let Some(parent) = args.output.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "failed to create output directory {}: {e}",
                parent.display()
            )
        })?;
    }
    fs::write(&args.output, text)
        .map_err(|e| format!("failed to write output {}: {e}", args.output.display()))?;
    println!("Wrote {}", args.output.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{dtw_distance, pearson, resample_to_len, zscore};

    #[test]
    fn test_pearson_identity() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        assert!((pearson(&a, &b) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_dtw_zero_for_equal_series() {
        let a = [0.1, 0.2, 0.3, 0.4];
        let d = dtw_distance(&a, &a);
        assert!(d.abs() < 1e-12);
    }

    #[test]
    fn test_resample_len() {
        let a = [0.0, 10.0];
        let r = resample_to_len(&a, 5);
        assert_eq!(r.len(), 5);
        assert!((r[0] - 0.0).abs() < 1e-12);
        assert!((r[4] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_zscore_constant_series() {
        let a = [2.0, 2.0, 2.0];
        let z = zscore(&a);
        assert_eq!(z, vec![0.0, 0.0, 0.0]);
    }
}
