//! ZD Quantum Chaos Census
//!
//! Nearest-Neighbor Spacing Distribution (NNSD) analysis of zero-divisor graph
//! Laplacian eigenvalue spectra across Cayley-Dickson dimensions. Diagnoses
//! quantum chaos vs integrability via the Brody parameter.
//!
//! # Usage
//!
//! zd-quantum-chaos --min-dim 64 --max-dim 256 --by-motif

use algebra_analysis::quantum_chaos::{
    export_motif_classes, nnsd_by_motif_class, nnsd_census, poisson_spacing, wigner_surmise_goe,
};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "zd-quantum-chaos")]
#[command(about = "NNSD analysis of zero-divisor graph Laplacian spectra")]
struct Cli {
    /// Minimum CD dimension (must be power of 2, >= 16).
    #[arg(long, default_value = "64")]
    min_dim: usize,

    /// Maximum CD dimension (must be power of 2).
    #[arg(long, default_value = "256")]
    max_dim: usize,

    /// Also run per-motif-class analysis at each feasible dimension.
    #[arg(long)]
    by_motif: bool,

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

    eprintln!("ZD Quantum Chaos Census (NNSD)");
    eprintln!("Dimensions: {:?}", dims);
    eprintln!();

    // Run census
    let results = nnsd_census(&dims);

    if results.is_empty() {
        eprintln!("No dimensions had enough eigenvalues (need >= 30).");
        eprintln!("Try --min-dim 64 or higher.");
        return Ok(());
    }

    // Summary table to stderr
    eprintln!(
        "{:<8} {:>8} {:>12} {:>10} {:>10} {:>10} {:>10} {:>8} {:>12}",
        "CD_dim",
        "n_eigs",
        "mean_spacing",
        "var",
        "SSE_pois",
        "SSE_goe",
        "KS_pois",
        "KS_goe",
        "brody_q"
    );
    eprintln!("{}", "-".repeat(98));

    for r in &results {
        eprintln!(
            "{:<8} {:>8} {:>12.6} {:>10.6} {:>10.6} {:>10.6} {:>10.6} {:>8.6} {:>12}",
            r.cd_dim,
            r.n_eigenvalues,
            r.mean_spacing,
            r.spacing_variance,
            r.poisson_sse,
            r.goe_sse,
            r.ks_poisson,
            r.ks_goe,
            format!("{:.4} ({})", r.brody_q, r.verdict),
        );
    }

    // Write CSV files
    std::fs::create_dir_all(&cli.output_dir)?;

    // 1. Summary CSV
    let summary_path = cli.output_dir.join("zd_nnsd.csv");
    let mut wtr = csv::Writer::from_path(&summary_path)?;
    wtr.write_record([
        "cd_dim",
        "n_eigenvalues",
        "mean_spacing",
        "spacing_variance",
        "poisson_sse",
        "goe_sse",
        "ks_poisson",
        "ks_goe",
        "brody_q",
        "verdict",
    ])?;
    for r in &results {
        wtr.write_record([
            r.cd_dim.to_string(),
            r.n_eigenvalues.to_string(),
            format!("{:.10}", r.mean_spacing),
            format!("{:.10}", r.spacing_variance),
            format!("{:.10}", r.poisson_sse),
            format!("{:.10}", r.goe_sse),
            format!("{:.10}", r.ks_poisson),
            format!("{:.10}", r.ks_goe),
            format!("{:.10}", r.brody_q),
            r.verdict.to_string(),
        ])?;
    }
    wtr.flush()?;
    eprintln!("\nSummary written to {}", summary_path.display());

    // 2. Histogram CSV
    let hist_path = cli.output_dir.join("zd_nnsd_histogram.csv");
    let mut wtr = csv::Writer::from_path(&hist_path)?;
    wtr.write_record(["cd_dim", "bin_center", "p_empirical", "p_poisson", "p_goe"])?;
    for r in &results {
        for (center, p_emp) in r.bin_centers.iter().zip(r.histogram.iter()) {
            wtr.write_record([
                r.cd_dim.to_string(),
                format!("{center:.6}"),
                format!("{p_emp:.10}"),
                format!("{:.10}", poisson_spacing(*center)),
                format!("{:.10}", wigner_surmise_goe(*center)),
            ])?;
        }
    }
    wtr.flush()?;
    eprintln!("Histograms written to {}", hist_path.display());

    // 3. Motif-class CSV (if --by-motif)
    if cli.by_motif {
        let motif_path = cli.output_dir.join("zd_nnsd_motif.csv");
        let mut wtr = csv::Writer::from_path(&motif_path)?;
        wtr.write_record([
            "cd_dim",
            "motif_edge_count",
            "n_components",
            "n_eigenvalues",
            "brody_q",
            "verdict",
        ])?;

        for &dim in &dims {
            eprintln!("\nMotif-class analysis for dim={dim}...");
            let motif_results = nnsd_by_motif_class(dim);

            for m in &motif_results {
                let (n_eigs, q, v) = match &m.nnsd {
                    Some(r) => (
                        r.n_eigenvalues.to_string(),
                        format!("{:.10}", r.brody_q),
                        r.verdict.to_string(),
                    ),
                    None => ("N/A".to_string(), "N/A".to_string(), "N/A".to_string()),
                };
                wtr.write_record([
                    m.cd_dim.to_string(),
                    m.motif_edge_count.to_string(),
                    m.n_components_in_class.to_string(),
                    n_eigs,
                    q,
                    v,
                ])?;
                eprintln!(
                    "  edges={:>4}  comps={:>4}  eigs={:>4}  brody_q={}  verdict={}",
                    m.motif_edge_count,
                    m.n_components_in_class,
                    match &m.nnsd {
                        Some(r) => r.n_eigenvalues.to_string(),
                        None => "N/A".to_string(),
                    },
                    match &m.nnsd {
                        Some(r) => format!("{:.4}", r.brody_q),
                        None => "N/A".to_string(),
                    },
                    match &m.nnsd {
                        Some(r) => r.verdict.to_string(),
                        None => "N/A".to_string(),
                    },
                );
            }
        }
        wtr.flush()?;
        eprintln!("Motif-class results written to {}", motif_path.display());

        // 4. Export .dot files and invariants CSV for each motif class
        let dot_dir = cli.output_dir.join("motif_dot");
        std::fs::create_dir_all(&dot_dir)?;

        let inv_path = cli.output_dir.join("zd_motif_invariants.csv");
        let mut inv_wtr = csv::Writer::from_path(&inv_path)?;
        inv_wtr.write_record([
            "cd_dim",
            "n_edges",
            "n_vertices",
            "n_components",
            "is_regular",
            "degree",
            "diameter",
            "girth",
            "triangles",
            "k2_parts",
            "spectrum",
        ])?;

        for &dim in &dims {
            let classes = export_motif_classes(dim);
            eprintln!(
                "\nExporting {} motif .dot files for dim={dim}...",
                classes.len()
            );

            for (edge_count, dot_string, inv) in &classes {
                // Write .dot file
                let dot_file = dot_dir.join(format!("motif_dim{dim}_edges{edge_count}.dot"));
                std::fs::write(&dot_file, dot_string)?;
                eprintln!(
                    "  {} (V={} E={} diam={} girth={} tri={} k2={}{})",
                    dot_file.display(),
                    inv.n_vertices,
                    inv.n_edges,
                    inv.diameter,
                    if inv.girth == usize::MAX {
                        "inf".to_string()
                    } else {
                        inv.girth.to_string()
                    },
                    inv.triangle_count,
                    inv.k2_parts,
                    if inv.is_regular {
                        format!(" {}-regular", inv.regularity_degree)
                    } else {
                        String::new()
                    },
                );

                // Write invariants row
                let spec_str: Vec<String> =
                    inv.spectrum.iter().map(|&v| format!("{v:.4}")).collect();
                inv_wtr.write_record([
                    inv.cd_dim.to_string(),
                    inv.n_edges.to_string(),
                    inv.n_vertices.to_string(),
                    inv.n_components_in_class.to_string(),
                    inv.is_regular.to_string(),
                    inv.regularity_degree.to_string(),
                    inv.diameter.to_string(),
                    if inv.girth == usize::MAX {
                        "inf".to_string()
                    } else {
                        inv.girth.to_string()
                    },
                    inv.triangle_count.to_string(),
                    inv.k2_parts.to_string(),
                    format!("[{}]", spec_str.join(",")),
                ])?;
            }
        }
        inv_wtr.flush()?;
        eprintln!("Motif invariants written to {}", inv_path.display());
        eprintln!("Dot files written to {}", dot_dir.display());
    }

    Ok(())
}
