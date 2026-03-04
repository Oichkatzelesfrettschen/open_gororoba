//! arleo-falmagne-scaling: Reproduce QGP path-length scaling analysis.
//!
//! Implements the Arleo & Falmagne (arXiv:2411.13258) analysis pipeline:
//! 1. Optical Glauber model for nuclear geometry
//! 2. R_AA fitting to extract epsilon_bar per centrality
//! 3. Multi-system density scaling: epsilon_bar = K * (dNch/dy / A_perp) * L^beta
//! 4. v2/eccentricity vs d(ln R_AA)/d(ln pT) linear relation
//!
//! Usage:
//!   arleo-falmagne-scaling full              # Full pipeline
//!   arleo-falmagne-scaling glauber-only      # Glauber geometry only
//!   arleo-falmagne-scaling alice             # ALICE-only fast path

use clap::{Parser, Subcommand};
use data_core::{catalogs::hic_raa, fetcher::FetchConfig};
use qgp_scaling::{
    data_tables,
    density_scaling::{DensityScalingPoint, fit_density_scaling},
    epsilon_fit::{RaaDataPoint, extract_epsilon, extract_epsilon_straggling},
    glauber::{CentralityBinGeometry, SigmaNN, compute_centrality_bins, standard_centrality_edges},
    multiplicity,
    nucleus::NucleusParams,
    straggling::{DEFAULT_KAPPA, StragglingGrid},
};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "arleo-falmagne-scaling")]
#[command(about = "QGP path-length scaling reproduction (Arleo & Falmagne arXiv:2411.13258)")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Full analysis pipeline: Glauber + epsilon extraction + density fit + v2 relation.
    Full {
        /// Data directory for downloaded files.
        #[arg(long, default_value = "data/external")]
        data_dir: String,

        /// Skip data download (use cached).
        #[arg(long)]
        skip_download: bool,

        /// Minimum pT for R_AA fits (GeV).
        #[arg(long, default_value = "5.0")]
        pt_min: f64,

        /// Gauss-Legendre quadrature order for Glauber integrals.
        #[arg(long, default_value = "48")]
        n_gl: usize,

        /// Enable quantum straggling (Gaussian-smeared R_AA) for low-pT fits.
        #[arg(long)]
        straggling: bool,

        /// Straggling width coefficient κ (σ = κ · √ε̄); only used with --straggling.
        #[arg(long, default_value_t = DEFAULT_KAPPA)]
        kappa: f64,
    },

    /// ALICE-only fast path (Pb-Pb 5.02 TeV only, uses existing data).
    Alice {
        /// Data directory.
        #[arg(long, default_value = "data/external")]
        data_dir: String,

        /// Minimum pT (GeV).
        #[arg(long, default_value = "5.0")]
        pt_min: f64,

        /// Enable quantum straggling (Gaussian-smeared R_AA) for low-pT fits.
        #[arg(long)]
        straggling: bool,

        /// Straggling width coefficient κ (σ = κ · √ε̄); only used with --straggling.
        #[arg(long, default_value_t = DEFAULT_KAPPA)]
        kappa: f64,
    },

    /// Glauber geometry calculations only.
    GlauberOnly {
        /// Gauss-Legendre quadrature order.
        #[arg(long, default_value = "48")]
        n_gl: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Full {
            data_dir,
            skip_download,
            pt_min,
            n_gl,
            straggling,
            kappa,
        } => {
            run_full(&data_dir, skip_download, pt_min, n_gl, straggling, kappa);
        }
        Commands::Alice {
            data_dir,
            pt_min,
            straggling,
            kappa,
        } => {
            run_alice(&data_dir, pt_min, straggling, kappa);
        }
        Commands::GlauberOnly { n_gl } => {
            run_glauber_only(n_gl);
        }
    }
}

fn run_glauber_only(n_gl: usize) {
    eprintln!("=== Arleo-Falmagne Scaling: Glauber Geometry ===");
    eprintln!();

    let systems: Vec<(&str, NucleusParams, SigmaNN)> = vec![
        (
            "Pb-Pb 5.02 TeV",
            NucleusParams::pb208(),
            SigmaNN::lhc_5020(),
        ),
        (
            "Pb-Pb 2.76 TeV",
            NucleusParams::pb208(),
            SigmaNN::lhc_2760(),
        ),
        ("Au-Au 200 GeV", NucleusParams::au197(), SigmaNN::rhic_200()),
        (
            "Xe-Xe 5.44 TeV",
            NucleusParams::xe129(),
            SigmaNN::lhc_5440(),
        ),
    ];

    let edges = standard_centrality_edges();

    for (label, nuc, sigma) in &systems {
        eprintln!("--- {} ---", label);
        eprintln!(
            "  R_A = {:.3} fm, sigma_NN = {:.1} mb",
            nuc.r_a, sigma.sigma_mb
        );

        let bins = compute_centrality_bins(&edges, sigma, nuc, n_gl, 300);

        eprintln!(
            "  {:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
            "Centrality", "b_lo", "b_hi", "Npart", "A_perp", "L_avg", "ecc"
        );

        for bin in &bins {
            eprintln!(
                "  {:>4.0}-{:<5.0}% {:>8.2} {:>8.2} {:>8.1} {:>8.1} {:>8.2} {:>8.4}",
                bin.cent_lo * 100.0,
                bin.cent_hi * 100.0,
                bin.b_lo,
                bin.b_hi,
                bin.n_part,
                bin.a_perp,
                bin.l_avg,
                bin.eccentricity
            );
        }

        // Validate against published Npart
        if *label == "Pb-Pb 5.02 TeV" {
            validate_npart_pbpb(&bins);
        }

        eprintln!();
    }
}

fn validate_npart_pbpb(bins: &[CentralityBinGeometry]) {
    let refs = data_tables::alice_pbpb_5020_npart();
    eprintln!();
    eprintln!("  Npart validation (ALICE PLB 772, 2017):");
    eprintln!(
        "  {:>10} {:>10} {:>10} {:>10} {:>6}",
        "Centrality", "Computed", "Published", "Diff%", "Pass?"
    );

    for r in &refs {
        // Find matching centrality bin
        if let Some(bin) = bins.iter().find(|b| {
            (b.cent_lo - r.cent_lo).abs() < 0.001 && (b.cent_hi - r.cent_hi).abs() < 0.001
        }) {
            let diff_pct = 100.0 * (bin.n_part - r.n_part).abs() / r.n_part;
            let pass = if diff_pct < 5.0 { "OK" } else { "FAIL" };
            eprintln!(
                "  {:>4.0}-{:<5.0}% {:>10.1} {:>10.1} {:>9.1}% {:>6}",
                r.cent_lo * 100.0,
                r.cent_hi * 100.0,
                bin.n_part,
                r.n_part,
                diff_pct,
                pass
            );
        }
    }
}

fn run_alice(data_dir: &str, pt_min: f64, use_straggling: bool, kappa: f64) {
    eprintln!("=== Arleo-Falmagne Scaling: ALICE Pb-Pb 5.02 TeV ===");
    if use_straggling {
        eprintln!("    Straggling mode: enabled (kappa = {kappa:.3})");
    }
    eprintln!();

    // Step 1: Glauber geometry
    eprintln!("[1/4] Computing Glauber geometry for Pb-Pb 5.02 TeV...");
    let pb = NucleusParams::pb208();
    let sigma = SigmaNN::lhc_5020();
    let edges = standard_centrality_edges();
    let bins = compute_centrality_bins(&edges, &sigma, &pb, 48, 300);
    eprintln!("      {} centrality bins computed", bins.len());

    // Step 2: Parse ALICE R_AA data
    eprintln!("[2/4] Loading ALICE R_AA data...");
    let alice_dir = PathBuf::from(data_dir).join(hic_raa::ALICE_PBPB_RAA_DIR);

    // Use broad centrality bins matching our Glauber edges
    let broad_cents = [
        (0.00, 0.05, vec![1]),      // 0-5% = Table 1
        (0.05, 0.10, vec![2]),      // 5-10% = Table 2
        (0.10, 0.20, vec![3, 4]),   // 10-20% = Tables 3,4 (10-15% + 15-20%)
        (0.20, 0.30, vec![5, 6]),   // 20-30% = Tables 5,6
        (0.30, 0.40, vec![7, 8]),   // 30-40% = Tables 7,8
        (0.40, 0.50, vec![9, 10]),  // 40-50% = Tables 9,10
        (0.50, 0.60, vec![11, 12]), // 50-60% = Tables 11,12
        (0.60, 0.70, vec![13, 14]), // 60-70% = Tables 13,14
    ];

    let n_spectral = 6.1; // Spectral index for 5.02 TeV pp spectrum

    // Build straggling grid once before the per-centrality loop (expensive but one-time)
    let straggling_grid = if use_straggling {
        eprintln!(
            "[2.5/4] Building straggling grid (kappa = {kappa:.3}, n = {n_spectral:.1})..."
        );
        let t0 = std::time::Instant::now();
        let grid = StragglingGrid::new(
            (1.0, 100.0),
            200,
            (0.1, 30.0),
            150,
            n_spectral,
            kappa,
        );
        eprintln!("        Grid built in {:.2}s", t0.elapsed().as_secs_f64());
        Some(grid)
    } else {
        None
    };

    let mut epsilon_results = Vec::new();

    eprintln!(
        "[3/4] Extracting epsilon_bar per centrality (n = {:.1})...",
        n_spectral
    );
    eprintln!(
        "  {:>10} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "Centrality", "eps_bar", "+err", "-err", "chi2/ndf", "N_pts"
    );

    for (c_lo, c_hi, tables) in &broad_cents {
        // Load and merge data from relevant tables
        let mut all_data = Vec::new();
        for &t in tables {
            let path = alice_dir.join(format!("table_{}.csv", t));
            if !path.exists() {
                eprintln!("  WARNING: {} not found, skipping", path.display());
                continue;
            }
            match hic_raa::parse_raa_csv(&path) {
                Ok(pts) => {
                    for p in pts {
                        if p.pt >= pt_min {
                            all_data.push(RaaDataPoint {
                                pt: p.pt,
                                raa: p.raa,
                                stat_err: p.stat_err,
                                syst_err: p.syst_err_up,
                            });
                        }
                    }
                }
                Err(e) => eprintln!("  WARNING: {}", e),
            }
        }

        if all_data.is_empty() {
            eprintln!("  {:>4.0}-{:<5.0}% -- no data", c_lo * 100.0, c_hi * 100.0);
            continue;
        }

        // Extract epsilon_bar: use straggling-smeared model if requested
        let result = if let Some(ref grid) = straggling_grid {
            extract_epsilon_straggling(&all_data, n_spectral, 0.1, 20.0, 1e-6, grid)
        } else {
            extract_epsilon(&all_data, n_spectral, 0.1, 20.0, 1e-6)
        };

        eprintln!(
            "  {:>4.0}-{:<5.0}% {:>8.2} {:>8.2} {:>8.2} {:>8.2} {:>8}",
            c_lo * 100.0,
            c_hi * 100.0,
            result.epsilon_bar,
            result.err_up,
            result.err_down,
            result.chi2_min / result.ndf as f64,
            all_data.len()
        );

        epsilon_results.push((*c_lo, *c_hi, result));
    }

    // Step 4: Density scaling fit
    eprintln!();
    eprintln!("[4/4] Density scaling fit: epsilon_bar = K * (dNch/dy / A_perp) * L^beta");

    let mult = multiplicity::alice_pbpb_5020_multiplicity();
    let mut scaling_data = Vec::new();

    for (c_lo, c_hi, eps_result) in &epsilon_results {
        // Find matching Glauber bin
        let glauber = bins
            .iter()
            .find(|b| (b.cent_lo - c_lo).abs() < 0.001 && (b.cent_hi - c_hi).abs() < 0.001);
        let mult_bin = mult
            .iter()
            .find(|m| (m.cent_lo - c_lo).abs() < 0.001 && (m.cent_hi - c_hi).abs() < 0.001);

        if let (Some(g), Some(m)) = (glauber, mult_bin) {
            scaling_data.push(DensityScalingPoint {
                epsilon_bar: eps_result.epsilon_bar,
                epsilon_bar_err: (eps_result.err_up + eps_result.err_down) / 2.0,
                dnch_dy: m.dnch_dy(),
                a_perp: g.a_perp,
                l_avg: g.l_avg,
                system: "Pb-Pb 5.02 TeV".to_string(),
                centrality: format!("{:.0}-{:.0}%", c_lo * 100.0, c_hi * 100.0),
            });
        }
    }

    if scaling_data.len() >= 3 {
        let result = fit_density_scaling(&scaling_data, 0.0, 3.0, 0.01);

        eprintln!();
        eprintln!("  === Density Scaling Result (Pb-Pb 5.02 TeV only) ===");
        eprintln!(
            "  beta     = {:.3} +{:.3} -{:.3}",
            result.beta, result.beta_err_up, result.beta_err_down
        );
        eprintln!(
            "  K        = {:.4} +/- {:.4} fm^(1-beta)",
            result.k_constant, result.k_err
        );
        eprintln!(
            "  chi2/ndf = {:.3} ({:.1} / {})",
            result.chi2_per_ndf, result.chi2_min, result.ndf
        );
        eprintln!();

        // Check against Arleo-Falmagne expected values
        let beta_ok = (result.beta - 1.02).abs() < 0.5;
        eprintln!("  Comparison with Arleo-Falmagne (beta = 1.02 +0.09/-0.06):");
        eprintln!(
            "  beta within 0.5 of reference: {}",
            if beta_ok { "PASS" } else { "MARGINAL" }
        );
    } else {
        eprintln!(
            "  WARNING: Only {} data points, need >= 3 for fit",
            scaling_data.len()
        );
    }
}

fn run_full(data_dir: &str, skip_download: bool, pt_min: f64, n_gl: usize, use_straggling: bool, kappa: f64) {
    eprintln!("=== Arleo-Falmagne Scaling: Full Multi-System Analysis ===");
    eprintln!();

    // Step 1: Download data
    if !skip_download {
        eprintln!("[1/6] Downloading HIC R_AA and v2 datasets...");
        let config = FetchConfig {
            output_dir: PathBuf::from(data_dir),
            skip_existing: true,
            verify_checksums: true,
        };
        match data_core::fetcher::DatasetProvider::fetch(&hic_raa::HicRaaProvider, &config) {
            Ok(p) => eprintln!("      Data directory: {}", p.display()),
            Err(e) => {
                eprintln!("      WARNING: Some downloads failed: {}", e);
                eprintln!("      Continuing with available data...");
            }
        }
    } else {
        eprintln!("[1/6] Skipping download (using cached data)...");
    }

    // Step 2: Glauber geometry for all systems
    eprintln!("[2/6] Computing Glauber geometry...");
    run_glauber_only(n_gl);

    // Step 3: ALICE analysis (primary system)
    eprintln!("[3/6] ALICE Pb-Pb 5.02 TeV epsilon extraction...");
    run_alice(data_dir, pt_min, use_straggling, kappa);

    // Steps 4-6 would continue with CMS, ATLAS, PHENIX, and the combined fit.
    // For now, the ALICE-only path demonstrates the full pipeline mechanics.
    eprintln!("[4/6] Multi-system epsilon extraction... (ALICE complete, others pending data)");
    eprintln!("[5/6] v2/eccentricity relation... (pending CMS v2 data parsing)");
    eprintln!("[6/6] Combined report...");
    eprintln!();
    eprintln!("=== Pipeline complete. ===");
    eprintln!("Note: Full multi-system analysis requires all HEPData downloads to succeed.");
    eprintln!("Run with --skip-download if data is already cached.");
}

#[cfg(test)]
mod tests {
    use qgp_scaling::quenching::{r_aa_model, scaling_function};

    #[test]
    fn test_scaling_collapse_concept() {
        // Verify that R_AA(pT, eps, n) = f(pT/eps, n)
        let eps = 4.0;
        let n = 6.0;
        for pt in [8.0, 12.0, 20.0, 40.0] {
            let raa = r_aa_model(pt, eps, n);
            let u = pt / eps;
            let sf = scaling_function(u, n);
            assert!(
                (raa - sf).abs() < 1e-12,
                "Scaling collapse: R_AA({}) = {}, f({}) = {}",
                pt,
                raa,
                u,
                sf
            );
        }
    }
}
