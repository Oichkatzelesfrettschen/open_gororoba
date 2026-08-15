//! zd-crystal-bands: ZD adjacency crystal band analysis across CD dimensions.
//!
//! Interprets the ZD motif graph as a tight-binding Hamiltonian and extracts
//! band structure (eigenvalue spectrum), flat band fraction, and degeneracy
//! tables at each Cayley-Dickson dimension.
//!
//! Usage:
//!   zd-crystal-bands --dims 16,32,64                   # standard sweep
//!   zd-crystal-bands --dims 16 --reggiani               # factorization check
//!   zd-crystal-bands --dims 16,32 --details --tol 0.01  # per-component CSVs

use algebra_analysis::crystal_bands::{
    DimSpectrumSummary, dim_spectrum_summary, verify_reggiani_factorization,
};
use std::{fs, path::Path, time::Instant};

#[derive(clap::Args)]
pub struct Args {
    /// Comma-separated dimensions (powers of 2, >= 16)
    #[arg(short, long, value_delimiter = ',')]
    dims: Vec<usize>,

    /// Run Reggiani factorization check (D=16 only)
    #[arg(long)]
    reggiani: bool,

    /// Write per-component degeneracy CSVs
    #[arg(long)]
    details: bool,

    /// Eigenvalue grouping tolerance
    #[arg(long, default_value = "0.01")]
    tol: f64,

    /// Output directory for CSV files
    #[arg(short, long, default_value = "data/csv")]
    output_dir: String,
}

fn write_summary_csv(summaries: &[DimSpectrumSummary], path: &Path) -> std::io::Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record([
        "dim",
        "n_components",
        "total_nodes",
        "total_eigenvalues",
        "flat_band_fraction",
        "is_integer_spectrum",
        "distinct_eigenvalue_count",
    ])?;

    for s in summaries {
        wtr.write_record(&[
            s.dim.to_string(),
            s.n_components.to_string(),
            s.total_nodes.to_string(),
            s.total_eigenvalues.to_string(),
            format!("{:.6}", s.flat_band_fraction),
            s.is_integer_spectrum.to_string(),
            s.distinct_eigenvalue_count.to_string(),
        ])?;
    }
    wtr.flush()
}

fn write_degeneracy_csv(summary: &DimSpectrumSummary, path: &Path) -> std::io::Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record(["eigenvalue", "degeneracy"])?;
    for (ev, deg) in &summary.degeneracies {
        wtr.write_record(&[format!("{:.6}", ev), deg.to_string()])?;
    }
    wtr.flush()
}

fn write_class_csv(summary: &DimSpectrumSummary, path: &Path) -> std::io::Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record([
        "node_count",
        "edge_count",
        "multiplicity",
        "flat_band_fraction",
        "spectrum",
    ])?;
    for cs in &summary.class_spectra {
        let spec_str: Vec<String> = cs.spectrum.iter().map(|v| format!("{:.4}", v)).collect();
        wtr.write_record(&[
            cs.node_count.to_string(),
            cs.edge_count.to_string(),
            cs.multiplicity.to_string(),
            format!("{:.6}", cs.flat_band_fraction),
            spec_str.join(";"),
        ])?;
    }
    wtr.flush()
}

fn write_reggiani_csv(
    result: &algebra_analysis::crystal_bands::ReggianiFacResult,
    path: &Path,
) -> std::io::Result<()> {
    let mut wtr = csv::Writer::from_path(path)?;
    wtr.write_record(["eigenvalue", "full_degeneracy", "reduced_degeneracy"])?;
    for ((ev, fd), (_, rd)) in result
        .full_degeneracies
        .iter()
        .zip(result.per_component_degeneracies.iter())
    {
        wtr.write_record(&[format!("{:.6}", ev), fd.to_string(), rd.to_string()])?;
    }
    wtr.flush()
}

pub fn run(args: Args) {
    if args.dims.is_empty() {
        eprintln!("Error: no dimensions specified (use --dims 16,32,64)");
        std::process::exit(1);
    }

    for &dim in &args.dims {
        if dim < 16 || !dim.is_power_of_two() {
            eprintln!("Error: dimension {} is not a power of 2 >= 16", dim);
            std::process::exit(1);
        }
    }

    let out_dir = Path::new(&args.output_dir);
    fs::create_dir_all(out_dir).expect("failed to create output directory");

    let mut summaries = Vec::new();

    for &dim in &args.dims {
        let t0 = Instant::now();
        let summary = dim_spectrum_summary(dim, args.tol);
        let elapsed = t0.elapsed();

        eprintln!(
            "D={}: {} components, {} nodes, {} evals, fbf={:.4}, integer={}, {:.2}s",
            dim,
            summary.n_components,
            summary.total_nodes,
            summary.total_eigenvalues,
            summary.flat_band_fraction,
            summary.is_integer_spectrum,
            elapsed.as_secs_f64()
        );

        // Degeneracy table to stderr
        for (ev, deg) in &summary.degeneracies {
            eprintln!("  eigenvalue {:.4}: degeneracy {}", ev, deg);
        }

        // Per-dimension degeneracy CSV
        let deg_path = out_dir.join(format!("zd_crystal_bands_degeneracies_{}d.csv", dim));
        write_degeneracy_csv(&summary, &deg_path).expect("failed to write degeneracy CSV");
        eprintln!("  -> {}", deg_path.display());

        if args.details {
            let class_path = out_dir.join(format!("zd_crystal_bands_classes_{}d.csv", dim));
            write_class_csv(&summary, &class_path).expect("failed to write class CSV");
            eprintln!("  -> {}", class_path.display());
        }

        summaries.push(summary);
    }

    // Summary CSV
    let summary_path = out_dir.join("zd_crystal_bands_summary.csv");
    write_summary_csv(&summaries, &summary_path).expect("failed to write summary CSV");
    eprintln!("Summary -> {}", summary_path.display());

    // Reggiani factorization check
    if args.reggiani {
        let t0 = Instant::now();
        let result = verify_reggiani_factorization(args.tol);
        let elapsed = t0.elapsed();

        eprintln!(
            "\nReggiani factorization: factorizes={}, n_components={}, {:.2}s",
            result.factorizes,
            result.n_components,
            elapsed.as_secs_f64()
        );
        eprintln!("  Full degeneracies: {:?}", result.full_degeneracies);
        eprintln!(
            "  Reduced degeneracies: {:?}",
            result.per_component_degeneracies
        );

        let reg_path = out_dir.join("zd_crystal_bands_reggiani.csv");
        write_reggiani_csv(&result, &reg_path).expect("failed to write Reggiani CSV");
        eprintln!("  -> {}", reg_path.display());
    }
}
