//! ZD Spectral Dimension Census
//!
//! Computes the graph spectral dimension d_s(t) of cross-assessor zero-divisor
//! graphs across Cayley-Dickson dimensions. Tests whether the sequence of
//! plateau values flows from ~2 (UV) to ~4 (IR), recovering the Calcagni
//! spectral dimension flow from pure algebraic structure.
//!
//! # Usage
//!
//! zd-spectral-dimension --min-dim 16 --max-dim 256 --t-points 300

use algebra_analysis::spectral_dimension::{dimensional_flow_analysis, spectral_gap_census};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "zd-spectral-dimension")]
#[command(about = "Compute spectral dimension of zero-divisor graphs across CD dimensions")]
struct Cli {
    /// Minimum CD dimension (must be power of 2, >= 16).
    #[arg(long, default_value = "16")]
    min_dim: usize,

    /// Maximum CD dimension (must be power of 2).
    #[arg(long, default_value = "256")]
    max_dim: usize,

    /// Number of diffusion-time sample points.
    #[arg(long, default_value = "300")]
    t_points: usize,

    /// Minimum diffusion time.
    #[arg(long, default_value = "0.01")]
    t_min: f64,

    /// Maximum diffusion time.
    #[arg(long, default_value = "50.0")]
    t_max: f64,

    /// Output directory for CSV files.
    #[arg(long, default_value = "data/csv")]
    output_dir: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Build dimension list: powers of 2 from min_dim to max_dim
    let mut dims = Vec::new();
    let mut d = cli.min_dim;
    while d <= cli.max_dim {
        if d >= 16 && d.is_power_of_two() {
            dims.push(d);
        }
        d *= 2;
    }

    if dims.is_empty() {
        eprintln!(
            "No valid dimensions in range [{}, {}]",
            cli.min_dim, cli.max_dim
        );
        std::process::exit(1);
    }

    eprintln!("ZD Spectral Dimension Census");
    eprintln!("Dimensions: {:?}", dims);
    eprintln!(
        "t range: [{}, {}], {} points",
        cli.t_min, cli.t_max, cli.t_points
    );
    eprintln!();

    let result = dimensional_flow_analysis(&dims, cli.t_min, cli.t_max, cli.t_points);

    // Print summary table
    eprintln!(
        "{:<8} {:>6} {:>8} {:>12} {:>10}",
        "CD_dim", "comps", "nodes", "d_s_plateau", "R^2"
    );
    eprintln!("{}", "-".repeat(50));

    for dim_result in &result.dimensions {
        let plateau_str = match &dim_result.plateau {
            Some(p) => format!("{:.4}", p.d_s),
            None => "N/A".to_string(),
        };
        let r2_str = match &dim_result.plateau {
            Some(p) => format!("{:.4}", p.r_squared),
            None => "N/A".to_string(),
        };
        eprintln!(
            "{:<8} {:>6} {:>8} {:>12} {:>10}",
            dim_result.cd_dim,
            dim_result.n_components,
            dim_result.nodes_per_component,
            plateau_str,
            r2_str,
        );
    }

    // Write curve CSV
    std::fs::create_dir_all(&cli.output_dir)?;

    let curve_path = cli.output_dir.join("zd_spectral_dimension_curves.csv");
    let mut wtr = csv::Writer::from_path(&curve_path)?;
    wtr.write_record(["cd_dim", "ln_t", "d_s"])?;
    for dim_result in &result.dimensions {
        for &(ln_t, d_s) in &dim_result.curve {
            wtr.write_record([
                dim_result.cd_dim.to_string(),
                format!("{ln_t:.6}"),
                format!("{d_s:.6}"),
            ])?;
        }
    }
    wtr.flush()?;
    eprintln!("\nCurves written to {}", curve_path.display());

    // Write plateau summary CSV
    let plateau_path = cli.output_dir.join("zd_spectral_dimension_plateaus.csv");
    let mut wtr = csv::Writer::from_path(&plateau_path)?;
    wtr.write_record([
        "cd_dim",
        "n_components",
        "nodes_per_component",
        "d_s_plateau",
        "ln_t_min",
        "ln_t_max",
        "r_squared",
    ])?;
    for dim_result in &result.dimensions {
        if let Some(p) = &dim_result.plateau {
            wtr.write_record([
                dim_result.cd_dim.to_string(),
                dim_result.n_components.to_string(),
                dim_result.nodes_per_component.to_string(),
                format!("{:.6}", p.d_s),
                format!("{:.6}", p.ln_t_min),
                format!("{:.6}", p.ln_t_max),
                format!("{:.6}", p.r_squared),
            ])?;
        }
    }
    wtr.flush()?;
    eprintln!("Plateaus written to {}", plateau_path.display());

    // Write eigenvalue summary
    let eigen_path = cli.output_dir.join("zd_laplacian_eigenvalues.csv");
    let mut wtr = csv::Writer::from_path(&eigen_path)?;
    wtr.write_record(["cd_dim", "eigenvalue_index", "eigenvalue"])?;
    for dim_result in &result.dimensions {
        for (i, &ev) in dim_result.eigenvalues.iter().enumerate() {
            wtr.write_record([
                dim_result.cd_dim.to_string(),
                i.to_string(),
                format!("{ev:.10}"),
            ])?;
        }
    }
    wtr.flush()?;
    eprintln!("Eigenvalues written to {}", eigen_path.display());

    // Spectral gap census
    eprintln!();
    eprintln!("Spectral Gap Census");
    eprintln!(
        "{:<8} {:>6} {:>8} {:>10} {:>10} {:>10} {:>9}",
        "CD_dim", "comps", "nodes", "lambda_2", "lambda_max", "gap_ratio", "expander"
    );
    eprintln!("{}", "-".repeat(67));

    let gap_results = spectral_gap_census(&dims);
    for r in &gap_results {
        eprintln!(
            "{:<8} {:>6} {:>8} {:>10.4} {:>10.4} {:>10.4} {:>9}",
            r.cd_dim,
            r.n_components,
            r.nodes_per_component,
            r.lambda_2,
            r.lambda_max,
            r.gap_ratio,
            if r.is_expander { "YES" } else { "NO" }
        );
    }

    let gap_path = cli.output_dir.join("zd_spectral_gap.csv");
    let mut wtr = csv::Writer::from_path(&gap_path)?;
    wtr.write_record([
        "cd_dim",
        "n_components",
        "nodes_per_component",
        "lambda_2",
        "lambda_max",
        "gap_ratio",
        "is_expander",
    ])?;
    for r in &gap_results {
        wtr.write_record([
            r.cd_dim.to_string(),
            r.n_components.to_string(),
            r.nodes_per_component.to_string(),
            format!("{:.10}", r.lambda_2),
            format!("{:.10}", r.lambda_max),
            format!("{:.10}", r.gap_ratio),
            r.is_expander.to_string(),
        ])?;
    }
    wtr.flush()?;
    eprintln!("Spectral gaps written to {}", gap_path.display());

    Ok(())
}
