//! Cross-backend compute parity report generator.
//!
//! Reads CUDA, Vulkan, and CPU benchmark CSVs (unified schema with `backend`
//! column) and produces a Markdown comparison table showing:
//! - MLUPS per backend for each (variant, precision, grid_size) triple
//! - CUDA/Vulkan and CUDA/CPU speedup ratios
//! - Feature parity matrix (which backends support which features)
//!
//! The report auto-detects the `backend` column (new schema) vs legacy format
//! (no backend column) for backwards compatibility.
//!
//! # Example
//!
//! ```bash
//! cargo run --release -p gororoba_cli_physics --bin parity-report \
//!     --no-default-features -- \
//!     --cuda-csv data/benchmarks/cuda_kernel_baseline.csv \
//!     --cpu-csv data/benchmarks/cpu_lbm_baseline.csv \
//!     --output data/benchmarks/parity_report.md
//! ```

use anyhow::Result;
use clap::Parser;
use std::{collections::BTreeMap, io::Write as _};

#[derive(Parser, Debug)]
#[command(name = "parity-report", version)]
struct Config {
    /// CUDA benchmark CSV.
    #[arg(long, default_value = "data/benchmarks/cuda_kernel_baseline.csv")]
    cuda_csv: String,

    /// Vulkan benchmark CSV.
    #[arg(long, default_value = "data/benchmarks/vulkan_kernel_baseline.csv")]
    vulkan_csv: String,

    /// CPU benchmark CSV.
    #[arg(long, default_value = "data/benchmarks/cpu_lbm_baseline.csv")]
    cpu_csv: String,

    /// Output Markdown path.
    #[arg(long, default_value = "data/benchmarks/parity_report.md")]
    output: String,
}

#[derive(Debug, Clone)]
struct Row {
    workload: String,
    precision: String,
    variant: String,
    grid_size: usize,
    mlups: Option<f64>,
}

fn parse_csv(path: &str, label: &str) -> Vec<Row> {
    let mut rows = Vec::new();
    let Ok(content) = std::fs::read_to_string(path) else {
        eprintln!("Warning: cannot read {path}, skipping {label}");
        return rows;
    };
    let header = content.lines().next().unwrap_or("");
    // Detect whether the CSV has a "backend" column (new schema) or not (legacy).
    let has_backend = header.starts_with("backend,");
    let offset: usize = if has_backend { 1 } else { 0 };
    for line in content.lines().skip(1) {
        let fields: Vec<&str> = line.splitn(15, ',').collect();
        if fields.len() < offset + 12 {
            continue;
        }
        rows.push(Row {
            workload: fields[offset].to_string(),
            precision: fields[offset + 1].to_string(),
            variant: fields[offset + 2].to_string(),
            grid_size: fields[offset + 3].parse().unwrap_or(0),
            mlups: fields[offset + 8].parse().ok(),
        });
    }
    rows
}

type Key = (String, String, usize); // (variant, precision, grid)

fn build_lookup(rows: &[Row]) -> BTreeMap<Key, &Row> {
    let mut map = BTreeMap::new();
    for r in rows {
        let key = (r.variant.clone(), r.precision.clone(), r.grid_size);
        map.entry(key).or_insert(r);
    }
    map
}

fn main() -> Result<()> {
    let cfg = Config::parse();

    let cuda_rows = parse_csv(&cfg.cuda_csv, "CUDA");
    let vulkan_rows = parse_csv(&cfg.vulkan_csv, "Vulkan");
    let cpu_rows = parse_csv(&cfg.cpu_csv, "CPU");

    let cuda_map = build_lookup(&cuda_rows);
    let vulkan_map = build_lookup(&vulkan_rows);
    let cpu_map = build_lookup(&cpu_rows);

    // Collect all keys
    let mut all_keys: Vec<Key> = Vec::new();
    for k in cuda_map
        .keys()
        .chain(vulkan_map.keys())
        .chain(cpu_map.keys())
    {
        if !all_keys.contains(k) {
            all_keys.push(k.clone());
        }
    }
    all_keys.sort();

    if let Some(parent) = std::path::Path::new(&cfg.output).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = std::fs::File::create(&cfg.output)?;

    writeln!(f, "# Compute Parity Report")?;
    writeln!(f)?;
    writeln!(
        f,
        "| Variant | Precision | Grid | CUDA MLUPS | Vulkan MLUPS | CPU MLUPS | CUDA/Vulkan | CUDA/CPU |"
    )?;
    writeln!(
        f,
        "|---------|-----------|------|------------|--------------|----------|-------------|----------|"
    )?;

    for key in &all_keys {
        let cuda = cuda_map.get(key).and_then(|r| r.mlups);
        let vulkan = vulkan_map.get(key).and_then(|r| r.mlups);
        let cpu = cpu_map.get(key).and_then(|r| r.mlups);

        let fmt = |v: Option<f64>| v.map_or("--".to_string(), |x| format!("{x:.1}"));
        let ratio = |a: Option<f64>, b: Option<f64>| match (a, b) {
            (Some(x), Some(y)) if y > 0.0 => format!("{:.1}x", x / y),
            _ => "--".to_string(),
        };

        writeln!(
            f,
            "| {} | {} | {} | {} | {} | {} | {} | {} |",
            key.0,
            key.1,
            key.2,
            fmt(cuda),
            fmt(vulkan),
            fmt(cpu),
            ratio(cuda, vulkan),
            ratio(cuda, cpu),
        )?;
    }

    writeln!(f)?;

    // Feature parity matrix
    writeln!(f, "## Feature Parity Matrix")?;
    writeln!(f)?;
    writeln!(f, "| Feature | CUDA | Vulkan | CPU |")?;
    writeln!(f, "|---------|------|--------|-----|")?;

    let features = [
        "BGK",
        "MRT",
        "Smagorinsky",
        "Tiled",
        "A-A streaming",
        "FP16",
        "BF16",
        "FP32",
        "FP64",
        "Box-counting",
    ];

    let has = |rows: &[Row], feat: &str| -> &'static str {
        let feat_lower = feat.to_lowercase();
        let found = rows.iter().any(|r| {
            r.variant.to_lowercase().contains(&feat_lower)
                || r.precision.to_lowercase().contains(&feat_lower)
                || r.workload.to_lowercase().contains(&feat_lower)
        });
        if found { "Y" } else { "--" }
    };

    for feat in features {
        writeln!(
            f,
            "| {} | {} | {} | {} |",
            feat,
            has(&cuda_rows, feat),
            has(&vulkan_rows, feat),
            has(&cpu_rows, feat),
        )?;
    }

    println!("Parity report written to: {}", cfg.output);
    println!(
        "Rows: CUDA={}, Vulkan={}, CPU={}",
        cuda_rows.len(),
        vulkan_rows.len(),
        cpu_rows.len()
    );
    Ok(())
}
