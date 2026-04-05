//! Systematic CD anomaly scanner: find statistically unusual events across datasets.
//!
//! Applies Tukey fence outlier detection (IQR * k) to CD time series from
//! multiple datasets, identifying anomalous windows that may represent
//! previously uncategorized physical events.
//!
//! Scans:
//! - Voyager MAG lifetime CD profiles
//! - SDO SHARP flare CD
//! - MESSENGER Mercury orbit CD
//! - ACE L1 CD
//! - Any JSON with "associator_norms" or "cd_values" arrays
//!
//! Usage:
//!   cd-anomaly-scan --data-dir data/output/heliosphere/ablations/ --k 2.0

use anyhow::{Context, Result};
use clap::Parser;
use serde::Serialize;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "cd-anomaly-scan")]
struct Cli {
    /// Directory containing CD result JSONs and CSVs
    #[arg(long, default_value = "data/output/heliosphere/ablations")]
    data_dir: PathBuf,

    /// Tukey fence multiplier (1.5 = mild, 2.0 = moderate, 3.0 = extreme)
    #[arg(long, default_value_t = 2.0)]
    k: f64,

    /// Minimum number of data points for a file to be analyzed
    #[arg(long, default_value_t = 10)]
    min_points: usize,

    /// Output JSON
    #[arg(
        long,
        default_value = "data/output/heliosphere/ablations/anomaly_scan.json"
    )]
    out_json: PathBuf,
}

#[derive(Debug, Serialize)]
struct Anomaly {
    dataset: String,
    index: usize,
    value: f64,
    upper_fence: f64,
    sigma_above_median: f64,
    context: String,
}

#[derive(Debug, Serialize)]
struct DatasetSummary {
    filename: String,
    n_points: usize,
    median: f64,
    iqr: f64,
    upper_fence: f64,
    n_anomalies: usize,
    anomaly_fraction: f64,
    top_anomalies: Vec<Anomaly>,
}

#[derive(Debug, Serialize)]
struct ScanResult {
    k_factor: f64,
    n_datasets: usize,
    total_anomalies: usize,
    datasets: Vec<DatasetSummary>,
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p * (sorted.len() - 1) as f64) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Extract CD values from a JSON file by looking for common field names.
fn extract_cd_from_json(path: &std::path::Path) -> Result<Vec<f64>> {
    let content = std::fs::read_to_string(path)?;
    let parsed: serde_json::Value = serde_json::from_str(&content)?;

    // Try multiple known field names
    let candidates = [
        "associator_norms",
        "cd_values",
        "norms",
        "associator",
        "values",
        "a_values",
        "cd_norms",
    ];

    // Top-level array
    for key in &candidates {
        if let Some(arr) = parsed[key].as_array() {
            let vals: Vec<f64> = arr
                .iter()
                .filter_map(|v| v.as_f64())
                .filter(|v| v.is_finite())
                .collect();
            if !vals.is_empty() {
                return Ok(vals);
            }
        }
    }

    // Nested in "results" or "data"
    for container in ["results", "data", "segments"] {
        if let Some(arr) = parsed[container].as_array() {
            let mut vals = Vec::new();
            for item in arr {
                for key in [
                    "cd_median",
                    "cd_mean",
                    "mean_associator",
                    "median_associator",
                    "a_median",
                    "a_mean",
                    "associator",
                ] {
                    if let Some(v) = item[key].as_f64()
                        && v.is_finite()
                    {
                        vals.push(v);
                        break;
                    }
                }
            }
            if !vals.is_empty() {
                return Ok(vals);
            }
        }
    }

    // Single-level fields for per-bin results
    let mut vals = Vec::new();
    if let Some(obj) = parsed.as_object() {
        for (key, val) in obj {
            if (key.contains("cd") || key.contains("associator") || key.contains("_a_"))
                && val.is_number()
                && let Some(v) = val.as_f64()
                && v.is_finite()
                && v > 0.0
            {
                vals.push(v);
            }
        }
    }
    if !vals.is_empty() {
        return Ok(vals);
    }

    anyhow::bail!("No CD values found in {}", path.display())
}

/// Extract CD values from a CSV file by looking for columns with "cd", "associator", or "a_" prefix.
fn extract_cd_from_csv(path: &std::path::Path) -> Result<Vec<f64>> {
    let content = std::fs::read_to_string(path)?;
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        anyhow::bail!("Empty CSV");
    }

    // Find header and CD column
    let header = lines[0];
    let cols: Vec<&str> = header.split(',').collect();
    let cd_col = cols.iter().position(|c| {
        let lower = c.to_lowercase();
        lower.contains("associator")
            || lower.contains("cd_")
            || lower == "a"
            || lower.contains("a_32d")
            || lower.contains("a_median")
    });

    if let Some(col_idx) = cd_col {
        let mut vals = Vec::new();
        for line in &lines[1..] {
            let fields: Vec<&str> = line.split(',').collect();
            if let Some(field) = fields.get(col_idx)
                && let Ok(v) = field.trim().parse::<f64>()
                && v.is_finite()
                && v > 0.0
            {
                vals.push(v);
            }
        }
        if !vals.is_empty() {
            return Ok(vals);
        }
    }

    // Fallback: try last numeric column
    let mut vals = Vec::new();
    for line in &lines[1..] {
        let fields: Vec<&str> = line.split(',').collect();
        if let Some(field) = fields.last()
            && let Ok(v) = field.trim().parse::<f64>()
            && v.is_finite()
            && v > 0.0
        {
            vals.push(v);
        }
    }
    if !vals.is_empty() {
        return Ok(vals);
    }

    anyhow::bail!("No CD column found in {}", path.display())
}

fn analyze_dataset(filename: &str, values: &[f64], k: f64) -> DatasetSummary {
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let q1 = percentile_sorted(&sorted, 0.25);
    let median = percentile_sorted(&sorted, 0.5);
    let q3 = percentile_sorted(&sorted, 0.75);
    let iqr = q3 - q1;
    let upper_fence = q3 + k * iqr;

    // Standard deviation for sigma scoring
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
    let std_dev = variance.sqrt().max(1e-12);

    let mut anomalies: Vec<Anomaly> = Vec::new();
    for (i, &v) in values.iter().enumerate() {
        if v > upper_fence {
            anomalies.push(Anomaly {
                dataset: filename.to_string(),
                index: i,
                value: v,
                upper_fence,
                sigma_above_median: (v - median) / std_dev,
                context: format!("idx={}, val={:.4}, fence={:.4}", i, v, upper_fence),
            });
        }
    }

    // Sort by value descending, keep top 10
    anomalies.sort_by(|a, b| {
        b.value
            .partial_cmp(&a.value)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n_anomalies = anomalies.len();
    anomalies.truncate(10);

    DatasetSummary {
        filename: filename.to_string(),
        n_points: values.len(),
        median,
        iqr,
        upper_fence,
        n_anomalies,
        anomaly_fraction: n_anomalies as f64 / values.len() as f64,
        top_anomalies: anomalies,
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    println!("=== CD Anomaly Scanner ===");
    println!("  Data dir: {}", cli.data_dir.display());
    println!(
        "  Tukey k: {:.1} (upper fence = Q3 + {:.1} * IQR)",
        cli.k, cli.k
    );
    println!("  Min points: {}", cli.min_points);

    let mut datasets = Vec::new();
    let mut total_anomalies = 0;

    // Scan all JSON and CSV files
    let entries: Vec<_> = std::fs::read_dir(&cli.data_dir)
        .with_context(|| format!("Reading {}", cli.data_dir.display()))?
        .filter_map(|e| e.ok())
        .collect();

    let mut files: Vec<PathBuf> = entries
        .iter()
        .map(|e| e.path())
        .filter(|p| {
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            ext == "json" || ext == "csv"
        })
        .collect();
    files.sort();

    println!("  Found {} data files\n", files.len());

    for path in &files {
        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        let values = if path.extension().and_then(|e| e.to_str()) == Some("json") {
            extract_cd_from_json(path)
        } else {
            extract_cd_from_csv(path)
        };

        match values {
            Ok(vals) if vals.len() >= cli.min_points => {
                let summary = analyze_dataset(filename, &vals, cli.k);
                if summary.n_anomalies > 0 {
                    println!(
                        "  {:50} {:>6} pts, {:>4} anomalies ({:.1}%), median={:.3}, max_anomaly={:.3}",
                        filename,
                        summary.n_points,
                        summary.n_anomalies,
                        summary.anomaly_fraction * 100.0,
                        summary.median,
                        summary
                            .top_anomalies
                            .first()
                            .map(|a| a.value)
                            .unwrap_or(0.0)
                    );
                }
                total_anomalies += summary.n_anomalies;
                datasets.push(summary);
            }
            Ok(vals) => {
                println!(
                    "  {:50} {:>6} pts (below min_points={})",
                    filename,
                    vals.len(),
                    cli.min_points
                );
            }
            Err(_) => {
                // Silently skip files with no CD data
            }
        }
    }

    println!("\n=== Scan Summary ===");
    println!("  Datasets analyzed: {}", datasets.len());
    println!("  Total anomalies: {}", total_anomalies);

    // Top anomalies across all datasets
    let mut all_anomalies: Vec<&Anomaly> = datasets
        .iter()
        .flat_map(|d| d.top_anomalies.iter())
        .collect();
    all_anomalies.sort_by(|a, b| {
        b.sigma_above_median
            .partial_cmp(&a.sigma_above_median)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    if !all_anomalies.is_empty() {
        println!("\n  Top 20 anomalies by sigma:");
        for (i, a) in all_anomalies.iter().take(20).enumerate() {
            println!(
                "    {:>2}. {:40} idx={:>5} val={:>10.3} ({:.1} sigma)",
                i + 1,
                a.dataset,
                a.index,
                a.value,
                a.sigma_above_median
            );
        }
    }

    let result = ScanResult {
        k_factor: cli.k,
        n_datasets: datasets.len(),
        total_anomalies,
        datasets,
    };
    let json = serde_json::to_string_pretty(&result)?;
    if let Some(parent) = cli.out_json.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&cli.out_json, &json)?;
    println!("\nResults -> {}", cli.out_json.display());

    Ok(())
}
