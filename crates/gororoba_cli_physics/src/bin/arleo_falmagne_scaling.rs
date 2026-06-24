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
use data_core::{
    catalogs::{hic_raa, hic_raa_fetch},
    fetcher::FetchConfig,
};
use qgp_scaling::{
    competing_models::{self, MeasuredRaaPoint, arleo_falmagne_raa, compare_models},
    data_tables::{self, eccentricity_event_by_event},
    density_scaling::{DensityScalingPoint, fit_density_scaling, fit_density_scaling_multi_k},
    epsilon_fit::{RaaDataPoint, extract_epsilon, extract_epsilon_straggling},
    glauber::{CentralityBinGeometry, SigmaNN, compute_centrality_bins, standard_centrality_edges},
    multiplicity,
    nucleus::NucleusParams,
    quenching::log_derivative_raa,
    straggling::{DEFAULT_KAPPA, StragglingGrid},
    v2_relation::{V2SlopePoint, fit_v2_relation},
};
use std::path::PathBuf;

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

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

        /// Straggling width coefficient kappa (sigma = kappa * sqrtepsilon_bar); only used with --straggling.
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

        /// Straggling width coefficient kappa (sigma = kappa * sqrtepsilon_bar); only used with --straggling.
        #[arg(long, default_value_t = DEFAULT_KAPPA)]
        kappa: f64,
    },

    /// Glauber geometry calculations only.
    GlauberOnly {
        /// Gauss-Legendre quadrature order.
        #[arg(long, default_value = "48")]
        n_gl: usize,
    },

    /// pT cutoff sweep: scan beta vs pT_min to find perturbative breakdown.
    PtSweep {
        /// Data directory.
        #[arg(long, default_value = "data/external")]
        data_dir: String,

        /// Gauss-Legendre quadrature order.
        #[arg(long, default_value = "48")]
        n_gl: usize,
    },

    /// Compare eccentricity models: optical Glauber vs MC Glauber (event-by-event).
    EccCompare {
        /// Data directory.
        #[arg(long, default_value = "data/external")]
        data_dir: String,

        /// Minimum pT for R_AA fits (GeV).
        #[arg(long, default_value = "5.0")]
        pt_min: f64,

        /// Gauss-Legendre quadrature order.
        #[arg(long, default_value = "48")]
        n_gl: usize,
    },

    /// Cross-validate beta: density scaling vs azimuthal (v2) extraction.
    CrossValidate {
        /// Data directory.
        #[arg(long, default_value = "data/external")]
        data_dir: String,

        /// Minimum pT for R_AA fits (GeV).
        #[arg(long, default_value = "5.0")]
        pt_min: f64,

        /// Gauss-Legendre quadrature order.
        #[arg(long, default_value = "48")]
        n_gl: usize,
    },

    /// BIC comparison: Arleo-Falmagne vs CUJET3.0 vs fractional Langevin.
    BicCompare {
        /// Data directory.
        #[arg(long, default_value = "data/external")]
        data_dir: String,

        /// Minimum pT for R_AA fits (GeV).
        #[arg(long, default_value = "5.0")]
        pt_min: f64,
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
        Commands::PtSweep { data_dir, n_gl } => {
            run_pt_sweep(&data_dir, n_gl);
        }
        Commands::EccCompare {
            data_dir,
            pt_min,
            n_gl,
        } => {
            run_ecc_compare(&data_dir, pt_min, n_gl);
        }
        Commands::CrossValidate {
            data_dir,
            pt_min,
            n_gl,
        } => {
            run_cross_validate(&data_dir, pt_min, n_gl);
        }
        Commands::BicCompare { data_dir, pt_min } => {
            run_bic_compare(&data_dir, pt_min);
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

fn alice_straggling_pt_range(
    centrality_data: &[(f64, f64, Vec<RaaDataPoint>)],
) -> Option<(f64, f64)> {
    let mut pt_min = f64::INFINITY;
    let mut pt_max = f64::NEG_INFINITY;

    for (_, _, data_points) in centrality_data {
        for data_point in data_points {
            if data_point.pt.is_finite() && data_point.pt > 0.0 {
                pt_min = pt_min.min(data_point.pt);
                pt_max = pt_max.max(data_point.pt);
            }
        }
    }

    if !pt_min.is_finite() {
        return None;
    }

    if pt_max > pt_min {
        return Some((pt_min, pt_max));
    }

    let lower = (pt_min * 0.95).max(f64::MIN_POSITIVE);
    let upper = (pt_min * 1.05).max(lower + f64::EPSILON);
    Some((lower, upper))
}

fn run_alice(data_dir: &str, pt_min: f64, use_straggling: bool, kappa: f64) {
    eprintln!("=== Arleo-Falmagne Scaling: ALICE Pb-Pb 5.02 TeV ===");
    if use_straggling {
        eprintln!("    Straggling mode: enabled (kappa = {kappa:.3})");
    }
    eprintln!();

    // Step 1: Glauber geometry (optical model for A_perp, L_avg)
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

    let mut alice_centrality_data = Vec::new();

    for (c_lo, c_hi, tables) in &broad_cents {
        let mut all_data = Vec::new();
        for &table_index in tables {
            let path = alice_dir.join(format!("table_{}.csv", table_index));
            if !path.exists() {
                eprintln!("  WARNING: {} not found, skipping", path.display());
                continue;
            }
            match hic_raa::parse_raa_csv(&path) {
                Ok(points) => {
                    for point in points {
                        if point.pt >= pt_min {
                            all_data.push(RaaDataPoint {
                                pt: point.pt,
                                raa: point.raa,
                                stat_err: point.stat_err,
                                syst_err: point.syst_err_up,
                            });
                        }
                    }
                }
                Err(error) => eprintln!("  WARNING: {}", error),
            }
        }
        alice_centrality_data.push((*c_lo, *c_hi, all_data));
    }

    let straggling_grid = if use_straggling {
        let Some(pt_range) = alice_straggling_pt_range(&alice_centrality_data) else {
            eprintln!(
                "  ERROR: no positive finite pT samples remain for straggling grid construction"
            );
            return;
        };

        eprintln!(
            "[2.5/4] Building straggling grid (pT = {:.3}-{:.3} GeV, kappa = {kappa:.3}, n = {n_spectral:.1})...",
            pt_range.0, pt_range.1
        );
        let started_at = std::time::Instant::now();
        let grid = StragglingGrid::new(pt_range, 200, (0.1, 30.0), 150, n_spectral, kappa);
        eprintln!(
            "        Grid built in {:.2}s",
            started_at.elapsed().as_secs_f64()
        );
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

    for (c_lo, c_hi, all_data) in &alice_centrality_data {
        if all_data.is_empty() {
            eprintln!("  {:>4.0}-{:<5.0}% -- no data", c_lo * 100.0, c_hi * 100.0);
            continue;
        }

        // Extract epsilon_bar: use straggling-smeared model if requested
        let result = if use_straggling {
            let Some(grid) = straggling_grid.as_ref() else {
                eprintln!("  ERROR: straggling grid unavailable during epsilon extraction");
                return;
            };
            extract_epsilon_straggling(all_data, 0.1, 20.0, 1e-6, grid)
        } else {
            extract_epsilon(all_data, n_spectral, 0.1, 20.0, 1e-6)
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

fn run_pt_sweep(data_dir: &str, n_gl: usize) {
    eprintln!("=== pT Cutoff Sweep: beta vs pT_min ===");
    eprintln!();

    let pb = NucleusParams::pb208();
    let sigma = SigmaNN::lhc_5020();
    let edges = standard_centrality_edges();
    let bins = compute_centrality_bins(&edges, &sigma, &pb, n_gl, 300);
    let mult = multiplicity::alice_pbpb_5020_multiplicity();
    let alice_dir = PathBuf::from(data_dir).join(hic_raa::ALICE_PBPB_RAA_DIR);

    let n_spectral = 6.1;
    let broad_cents: Vec<(f64, f64, Vec<usize>)> = vec![
        (0.00, 0.05, vec![1]),
        (0.05, 0.10, vec![2]),
        (0.10, 0.20, vec![3, 4]),
        (0.20, 0.30, vec![5, 6]),
        (0.30, 0.40, vec![7, 8]),
        (0.40, 0.50, vec![9, 10]),
        (0.50, 0.60, vec![11, 12]),
        (0.60, 0.70, vec![13, 14]),
    ];

    // Pre-load all R_AA data to avoid repeated IO
    let mut all_raa: Vec<(f64, f64, Vec<RaaDataPoint>)> = Vec::new();
    for (c_lo, c_hi, tables) in &broad_cents {
        let mut data = Vec::new();
        for &t in tables {
            let path = alice_dir.join(format!("table_{}.csv", t));
            if let Ok(pts) = hic_raa::parse_raa_csv(&path) {
                for p in pts {
                    data.push(RaaDataPoint {
                        pt: p.pt,
                        raa: p.raa,
                        stat_err: p.stat_err,
                        syst_err: p.syst_err_up,
                    });
                }
            }
        }
        all_raa.push((*c_lo, *c_hi, data));
    }

    let pt_mins = [3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0];

    eprintln!(
        "  {:>8} {:>8} {:>8} {:>10} {:>6}",
        "pT_min", "beta", "beta_err", "chi2/ndf", "N_pts"
    );

    for &pt_min in &pt_mins {
        let mut scaling_data = Vec::new();

        for (c_lo, c_hi, raa_data) in &all_raa {
            let filtered: Vec<RaaDataPoint> = raa_data
                .iter()
                .filter(|p| p.pt >= pt_min)
                .cloned()
                .collect();
            if filtered.is_empty() {
                continue;
            }

            let eps = extract_epsilon(&filtered, n_spectral, 0.1, 20.0, 1e-6);
            let g = bins
                .iter()
                .find(|b| (b.cent_lo - c_lo).abs() < 0.01 && (b.cent_hi - c_hi).abs() < 0.01);
            let m = mult
                .iter()
                .find(|m| (m.cent_lo - c_lo).abs() < 0.01 && (m.cent_hi - c_hi).abs() < 0.01);
            if let (Some(g), Some(m)) = (g, m) {
                scaling_data.push(DensityScalingPoint {
                    epsilon_bar: eps.epsilon_bar,
                    epsilon_bar_err: (eps.err_up + eps.err_down) / 2.0,
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
            eprintln!(
                "  {:>7.1} {:>8.3} {:>8.3} {:>10.3} {:>6}",
                pt_min,
                result.beta,
                result.beta_err_up,
                result.chi2_per_ndf,
                scaling_data.len()
            );
        } else {
            eprintln!(
                "  {:>7.1}    -- insufficient data ({} bins)",
                pt_min,
                scaling_data.len()
            );
        }
    }

    eprintln!();
    eprintln!("  Arleo-Falmagne reference: beta = 1.02 +0.09/-0.06 (pT > 5 GeV)");
    eprintln!("  Perturbative breakdown expected below ~5-7 GeV.");
}

/// Compare eccentricity models: optical Glauber vs MC Glauber (event-by-event).
///
/// Runs the v2/eccentricity relation fit twice with different eccentricity sources
/// and reports the beta shift. If |delta_beta| < 0.2, eccentricity assumption is robust.
fn run_ecc_compare(data_dir: &str, pt_min: f64, n_gl: usize) {
    eprintln!("=== Eccentricity Model Comparison: Optical vs Event-by-Event ===");
    eprintln!();

    let pb = NucleusParams::pb208();
    let sigma = SigmaNN::lhc_5020();
    let edges = standard_centrality_edges();
    let bins_optical = compute_centrality_bins(&edges, &sigma, &pb, n_gl, 300);
    let _bins_mc = data_tables::alice_pbpb_5020_mc_glauber();

    let alice_dir = PathBuf::from(data_dir).join(hic_raa::ALICE_PBPB_RAA_DIR);
    let v2_dir = PathBuf::from(data_dir).join("hic_raa");
    let cms_v2_tables = hic_raa::cms_pbpb_5020_v2_tables();

    let v2_cents: Vec<(f64, f64, usize)> = vec![
        (0.00, 0.05, 1),
        (0.05, 0.10, 2),
        (0.10, 0.20, 3),
        (0.20, 0.30, 4),
        (0.30, 0.40, 5),
        (0.40, 0.50, 6),
    ];

    // Helper: build v2 slope data for a given eccentricity source
    let build_v2_data =
        |ecc_source: &str, get_ecc: &dyn Fn(f64, f64) -> f64| -> Vec<V2SlopePoint> {
            let mut v2_slope_data = Vec::new();

            for &(c_lo, c_hi, v2_table_idx) in &v2_cents {
                if v2_table_idx > cms_v2_tables.len() {
                    continue;
                }
                let v2_path = v2_dir.join(cms_v2_tables[v2_table_idx - 1].filename);
                let v2_data = match hic_raa::parse_v2_csv(&v2_path) {
                    Ok(d) => d,
                    Err(_) => continue,
                };

                let raa_table = (c_lo * 20.0) as usize + 1;
                let raa_path = alice_dir.join(format!("table_{}.csv", raa_table));
                let raa_pts = match hic_raa::parse_raa_csv(&raa_path) {
                    Ok(d) => d,
                    Err(_) => continue,
                };

                let raa_filtered: Vec<_> = raa_pts.iter().filter(|p| p.pt >= pt_min).collect();
                if raa_filtered.len() < 3 {
                    continue;
                }

                let raa_pt: Vec<f64> = raa_filtered.iter().map(|p| p.pt).collect();
                let raa_vals: Vec<f64> = raa_filtered.iter().map(|p| p.raa).collect();
                let slopes = log_derivative_raa(&raa_pt, &raa_vals);

                let ecc = get_ecc(c_lo, c_hi);
                if ecc < 0.01 {
                    continue;
                }

                for (i, slope) in slopes.iter().enumerate() {
                    if slope.is_nan() || i >= raa_pt.len() {
                        continue;
                    }
                    let target_pt = raa_pt[i];
                    if let Some(v2p) = v2_data
                        .iter()
                        .min_by(|a, b| {
                            (a.pt - target_pt)
                                .abs()
                                .partial_cmp(&(b.pt - target_pt).abs())
                                .unwrap()
                        })
                        .filter(|v2p| (v2p.pt - target_pt).abs() < 2.0)
                    {
                        v2_slope_data.push(V2SlopePoint {
                            pt: target_pt,
                            v2_over_ecc: v2p.v2 / ecc,
                            v2_over_ecc_err: v2p.stat_err / ecc,
                            dln_raa_dln_pt: *slope,
                            slope_err: 0.01,
                            centrality: format!(
                                "{} {:.0}-{:.0}%",
                                ecc_source,
                                c_lo * 100.0,
                                c_hi * 100.0
                            ),
                        });
                    }
                }
            }
            v2_slope_data
        };

    // Model 1: Optical Glauber eccentricity (computed from thickness functions)
    eprintln!("[1/2] Fitting with optical Glauber eccentricity...");
    let data_optical = build_v2_data("optical", &|c_lo, c_hi| {
        bins_optical
            .iter()
            .find(|b| (b.cent_lo - c_lo).abs() < 0.01 && (b.cent_hi - c_hi).abs() < 0.01)
            .map(|b| b.eccentricity)
            .unwrap_or(0.0)
    });

    // Model 2: MC Glauber event-by-event epsilon_2{2}
    eprintln!("[2/2] Fitting with MC Glauber epsilon_2{{2}}...");
    let data_mc = build_v2_data("mc-glauber", &|c_lo, c_hi| {
        eccentricity_event_by_event(c_lo, c_hi, "pbpb").unwrap_or(0.0)
    });

    if data_optical.len() < 3 || data_mc.len() < 3 {
        eprintln!(
            "  ERROR: insufficient data for comparison ({} optical, {} MC)",
            data_optical.len(),
            data_mc.len()
        );
        eprintln!(
            "  Make sure ALICE R_AA + CMS v2 data is downloaded to {}",
            data_dir
        );
        return;
    }

    let result_optical = fit_v2_relation(&data_optical, true);
    let result_mc = fit_v2_relation(&data_mc, true);

    let delta_beta = (result_optical.beta - result_mc.beta).abs();
    let verdict = if delta_beta < 0.2 {
        "ROBUST"
    } else {
        "SENSITIVE"
    };

    eprintln!();
    eprintln!("  +----------------------------------------------------------+");
    eprintln!("  | Eccentricity Model Comparison                             |");
    eprintln!("  +----------------------------------------------------------+");
    eprintln!(
        "  | Optical Glauber: beta = {:.4} +/- {:.4} (R^2 = {:.4}) |",
        result_optical.beta, result_optical.beta_err, result_optical.r_squared,
    );
    eprintln!(
        "  | MC eps_2{{2}}:      beta = {:.4} +/- {:.4} (R^2 = {:.4}) |",
        result_mc.beta, result_mc.beta_err, result_mc.r_squared,
    );
    eprintln!(
        "  | delta_beta = {:.4}                                      |",
        delta_beta,
    );
    eprintln!(
        "  | verdict: {} (threshold 0.2)                       |",
        verdict,
    );
    eprintln!("  +----------------------------------------------------------+");
    eprintln!();

    // Also check with 5.36 TeV MC Glauber if available
    let bins_536 = data_tables::alice_pbpb_5360_mc_glauber();
    if !bins_536.is_empty() {
        eprintln!(
            "  [info] Pb-Pb 5.36 TeV MC Glauber table loaded ({} bins)",
            bins_536.len()
        );
        eprintln!(
            "  [info] 0-5%: Npart={:.1}, A_perp={:.1} fm^2, ecc={:.3}",
            bins_536[0].n_part, bins_536[0].a_perp, bins_536[0].eccentricity
        );
        eprintln!("  [info] R_AA at 5.36 TeV not yet published -- comparison deferred.");
    }
}

/// Cross-validate the path-length exponent beta from two independent methods:
///   1. Density scaling: epsilon_bar = K * (dNch/dy / A_perp) * L^beta
///   2. Azimuthal: v2/eccentricity = (beta/2) * d(ln R_AA)/d(ln pT)
///
/// Computes z-score = |beta_density - beta_azimuthal| / sqrt(err_d^2 + err_a^2).
/// If z < 2, universality holds; if z >= 2, universality breaks.
fn run_cross_validate(data_dir: &str, pt_min: f64, n_gl: usize) {
    eprintln!("=== Beta Cross-Validation: Density vs Azimuthal ===");
    eprintln!();

    let n_spectral = 6.1;
    let pb = NucleusParams::pb208();
    let sigma = SigmaNN::lhc_5020();
    let edges = standard_centrality_edges();
    let bins = compute_centrality_bins(&edges, &sigma, &pb, n_gl, 300);

    // --- Branch 1: Density scaling beta ---
    eprintln!("[1/3] Extracting beta_density from multi-system density scaling...");
    let mut multi_system_data = Vec::new();

    // Pb-Pb epsilon extraction
    let alice_dir = PathBuf::from(data_dir).join(hic_raa::ALICE_PBPB_RAA_DIR);
    let pbpb_mult = multiplicity::alice_pbpb_5020_multiplicity();
    let broad_cents: Vec<(f64, f64, Vec<usize>)> = vec![
        (0.00, 0.05, vec![1]),
        (0.05, 0.10, vec![2]),
        (0.10, 0.20, vec![3, 4]),
        (0.20, 0.30, vec![5, 6]),
        (0.30, 0.40, vec![7, 8]),
        (0.40, 0.50, vec![9, 10]),
        (0.50, 0.60, vec![11, 12]),
        (0.60, 0.70, vec![13, 14]),
    ];

    for (c_lo, c_hi, tables) in &broad_cents {
        let mut all_data = Vec::new();
        for &t in tables {
            let path = alice_dir.join(format!("table_{}.csv", t));
            if let Ok(pts) = hic_raa::parse_raa_csv(&path) {
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
        }
        if all_data.is_empty() {
            continue;
        }
        let eps = extract_epsilon(&all_data, n_spectral, 0.1, 20.0, 1e-6);
        let g = bins
            .iter()
            .find(|b| (b.cent_lo - c_lo).abs() < 0.01 && (b.cent_hi - c_hi).abs() < 0.01);
        let m = pbpb_mult
            .iter()
            .find(|m| (m.cent_lo - c_lo).abs() < 0.01 && (m.cent_hi - c_hi).abs() < 0.01);
        if let (Some(g), Some(m)) = (g, m) {
            multi_system_data.push(DensityScalingPoint {
                epsilon_bar: eps.epsilon_bar,
                epsilon_bar_err: (eps.err_up + eps.err_down) / 2.0,
                dnch_dy: m.dnch_dy(),
                a_perp: g.a_perp,
                l_avg: g.l_avg,
                system: "ALICE Pb-Pb 5.02 TeV".to_string(),
                centrality: format!("{:.0}-{:.0}%", c_lo * 100.0, c_hi * 100.0),
            });
        }
    }

    // Xe-Xe epsilon extraction
    let xe = NucleusParams::xe129();
    let sigma_xe = SigmaNN::lhc_5440();
    let bins_xe = compute_centrality_bins(&edges, &sigma_xe, &xe, n_gl, 300);
    let xexe_mult = multiplicity::alice_xexe_5440_multiplicity();
    let xexe_dir = PathBuf::from(data_dir).join(hic_raa::ALICE_XEXE_RAA_DIR);
    let xexe_cents: Vec<(f64, f64, usize)> = vec![
        (0.00, 0.05, 1),
        (0.05, 0.10, 2),
        (0.10, 0.20, 3),
        (0.20, 0.30, 4),
        (0.30, 0.40, 5),
        (0.40, 0.50, 6),
        (0.50, 0.60, 7),
        (0.60, 0.70, 8),
    ];
    for (c_lo, c_hi, table_idx) in &xexe_cents {
        let path = xexe_dir.join(format!("alice_xexe_raa_table{}.csv", table_idx));
        let raa_data = match hic_raa::parse_raa_csv(&path) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let fit_data: Vec<RaaDataPoint> = raa_data
            .iter()
            .filter(|p| p.pt >= pt_min)
            .map(|p| RaaDataPoint {
                pt: p.pt,
                raa: p.raa,
                stat_err: p.stat_err,
                syst_err: p.syst_err_up,
            })
            .collect();
        if fit_data.is_empty() {
            continue;
        }
        let eps = extract_epsilon(&fit_data, n_spectral, 0.1, 20.0, 1e-6);
        let g = bins_xe
            .iter()
            .find(|b| (b.cent_lo - c_lo).abs() < 0.01 && (b.cent_hi - c_hi).abs() < 0.01);
        let m = xexe_mult
            .iter()
            .find(|m| (m.cent_lo - c_lo).abs() < 0.01 && (m.cent_hi - c_hi).abs() < 0.01);
        if let (Some(g), Some(m)) = (g, m) {
            multi_system_data.push(DensityScalingPoint {
                epsilon_bar: eps.epsilon_bar,
                epsilon_bar_err: (eps.err_up + eps.err_down) / 2.0,
                dnch_dy: m.dnch_dy(),
                a_perp: g.a_perp,
                l_avg: g.l_avg,
                system: "ALICE Xe-Xe 5.44 TeV".to_string(),
                centrality: format!("{:.0}-{:.0}%", c_lo * 100.0, c_hi * 100.0),
            });
        }
    }

    if multi_system_data.len() < 4 {
        eprintln!(
            "  ERROR: insufficient multi-system data ({} points, need >= 4)",
            multi_system_data.len()
        );
        eprintln!("  Make sure ALICE R_AA data is downloaded to {}", data_dir);
        return;
    }

    let density_result = fit_density_scaling_multi_k(&multi_system_data, 0.0, 3.0, 0.01);
    let beta_density = density_result.beta;
    let beta_density_err = (density_result.beta_err_up + density_result.beta_err_down) / 2.0;
    eprintln!(
        "  beta_density = {:.4} +{:.4} -{:.4} (chi2/ndf = {:.3})",
        beta_density,
        density_result.beta_err_up,
        density_result.beta_err_down,
        density_result.chi2_per_ndf,
    );

    // --- Branch 2: Azimuthal beta from v2/eccentricity ---
    eprintln!();
    eprintln!("[2/3] Extracting beta_azimuthal from v2/eccentricity relation...");

    let v2_dir = PathBuf::from(data_dir).join("hic_raa");
    let cms_v2_tables = hic_raa::cms_pbpb_5020_v2_tables();
    let mut v2_slope_data = Vec::new();

    let v2_cents: Vec<(f64, f64, usize)> = vec![
        (0.00, 0.05, 1),
        (0.05, 0.10, 2),
        (0.10, 0.20, 3),
        (0.20, 0.30, 4),
        (0.30, 0.40, 5),
        (0.40, 0.50, 6),
    ];

    for (c_lo, c_hi, v2_table_idx) in &v2_cents {
        if *v2_table_idx > cms_v2_tables.len() {
            continue;
        }
        let v2_path = v2_dir.join(cms_v2_tables[v2_table_idx - 1].filename);
        let v2_data = match hic_raa::parse_v2_csv(&v2_path) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let raa_table = (*c_lo * 20.0) as usize + 1;
        let raa_path = alice_dir.join(format!("table_{}.csv", raa_table));
        let raa_pts = match hic_raa::parse_raa_csv(&raa_path) {
            Ok(d) => d,
            Err(_) => continue,
        };

        let raa_filtered: Vec<_> = raa_pts.iter().filter(|p| p.pt >= pt_min).collect();
        if raa_filtered.len() < 3 {
            continue;
        }

        let raa_pt: Vec<f64> = raa_filtered.iter().map(|p| p.pt).collect();
        let raa_vals: Vec<f64> = raa_filtered.iter().map(|p| p.raa).collect();
        let slopes = log_derivative_raa(&raa_pt, &raa_vals);

        let ecc = bins
            .iter()
            .find(|b| (b.cent_lo - c_lo).abs() < 0.01 && (b.cent_hi - c_hi).abs() < 0.01)
            .map(|b| b.eccentricity)
            .unwrap_or(0.2);

        for (i, slope) in slopes.iter().enumerate() {
            if slope.is_nan() || i >= raa_pt.len() {
                continue;
            }
            let target_pt = raa_pt[i];
            if let Some(v2p) = v2_data
                .iter()
                .min_by(|a, b| {
                    (a.pt - target_pt)
                        .abs()
                        .partial_cmp(&(b.pt - target_pt).abs())
                        .unwrap()
                })
                .filter(|v2p| (v2p.pt - target_pt).abs() < 2.0 && ecc > 0.01)
            {
                v2_slope_data.push(V2SlopePoint {
                    pt: target_pt,
                    v2_over_ecc: v2p.v2 / ecc,
                    v2_over_ecc_err: v2p.stat_err / ecc,
                    dln_raa_dln_pt: *slope,
                    slope_err: 0.01,
                    centrality: format!("{:.0}-{:.0}%", c_lo * 100.0, c_hi * 100.0),
                });
            }
        }
    }

    if v2_slope_data.len() < 3 {
        eprintln!(
            "  ERROR: insufficient v2 data ({} points, need >= 3)",
            v2_slope_data.len()
        );
        eprintln!("  Make sure CMS v2 data is downloaded to {}", data_dir);
        return;
    }

    let v2_result = fit_v2_relation(&v2_slope_data, true);
    let beta_azimuthal = v2_result.beta;
    let beta_azimuthal_err = v2_result.beta_err;
    eprintln!(
        "  beta_azimuthal = {:.4} +/- {:.4} (R^2 = {:.4}, chi2/ndf = {:.3})",
        beta_azimuthal,
        beta_azimuthal_err,
        v2_result.r_squared,
        v2_result.chi2 / v2_result.ndf as f64,
    );

    // --- Branch 3: Z-score universality test ---
    eprintln!();
    eprintln!("[3/3] Beta universality z-score test...");
    eprintln!();

    let delta_beta = (beta_density - beta_azimuthal).abs();
    let combined_err =
        (beta_density_err * beta_density_err + beta_azimuthal_err * beta_azimuthal_err).sqrt();
    let z_score = if combined_err > 0.0 {
        delta_beta / combined_err
    } else {
        f64::INFINITY
    };

    let verdict = if z_score < 2.0 {
        "UNIVERSALITY HOLDS"
    } else {
        "UNIVERSALITY BREAKS"
    };

    eprintln!("  +--------------------------------------------------+");
    eprintln!("  | Beta Cross-Validation Summary                     |");
    eprintln!("  +--------------------------------------------------+");
    eprintln!(
        "  | beta_density    = {:.4} +/- {:.4}                 |",
        beta_density, beta_density_err,
    );
    eprintln!(
        "  | beta_azimuthal  = {:.4} +/- {:.4}                 |",
        beta_azimuthal, beta_azimuthal_err,
    );
    eprintln!(
        "  | delta_beta      = {:.4}                           |",
        delta_beta,
    );
    eprintln!(
        "  | combined_err    = {:.4}                           |",
        combined_err,
    );
    eprintln!(
        "  | z-score         = {:.3}                            |",
        z_score,
    );
    eprintln!(
        "  | verdict: {} (z {} 2)     |",
        verdict,
        if z_score < 2.0 { "<" } else { ">=" },
    );
    eprintln!("  +--------------------------------------------------+");
}

fn run_full(
    data_dir: &str,
    skip_download: bool,
    pt_min: f64,
    n_gl: usize,
    use_straggling: bool,
    kappa: f64,
) {
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
        match data_core::fetcher::DatasetProvider::fetch(&hic_raa_fetch::HicRaaProvider, &config) {
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

    // Step 4: CMS v2 + v2/eccentricity relation
    eprintln!("[4/6] v2/eccentricity vs d(ln R_AA)/d(ln pT) relation...");
    let v2_dir = PathBuf::from(data_dir).join("hic_raa");
    let alice_dir_full = PathBuf::from(data_dir).join(hic_raa::ALICE_PBPB_RAA_DIR);
    let pb = NucleusParams::pb208();
    let sigma_full = SigmaNN::lhc_5020();
    let edges_full = standard_centrality_edges();
    let bins_full = compute_centrality_bins(&edges_full, &sigma_full, &pb, n_gl, 300);

    let cms_v2_tables = hic_raa::cms_pbpb_5020_v2_tables();
    let mut v2_slope_data = Vec::new();

    // Broad centrality bins for v2 correlation: use 0-5%, 5-10%, 10-20%, 20-30%, 30-40%, 40-50%
    let v2_cents: Vec<(f64, f64, usize)> = vec![
        (0.00, 0.05, 1),
        (0.05, 0.10, 2),
        (0.10, 0.20, 3),
        (0.20, 0.30, 4),
        (0.30, 0.40, 5),
        (0.40, 0.50, 6),
    ];

    for (c_lo, c_hi, v2_table_idx) in &v2_cents {
        // Load v2 data
        if *v2_table_idx > cms_v2_tables.len() {
            continue;
        }
        let v2_path = v2_dir.join(cms_v2_tables[v2_table_idx - 1].filename);
        let v2_data = match hic_raa::parse_v2_csv(&v2_path) {
            Ok(d) => d,
            Err(_) => continue,
        };

        // Load R_AA for same centrality to compute d(ln R_AA)/d(ln pT)
        let raa_table = (*c_lo * 20.0) as usize + 1; // Approximate ALICE table index
        let raa_path = alice_dir_full.join(format!("table_{}.csv", raa_table));
        let raa_pts = match hic_raa::parse_raa_csv(&raa_path) {
            Ok(d) => d,
            Err(_) => continue,
        };

        // Filter to pT >= pt_min
        let raa_filtered: Vec<_> = raa_pts.iter().filter(|p| p.pt >= pt_min).collect();
        if raa_filtered.len() < 3 {
            continue;
        }

        let raa_pt: Vec<f64> = raa_filtered.iter().map(|p| p.pt).collect();
        let raa_vals: Vec<f64> = raa_filtered.iter().map(|p| p.raa).collect();
        let slopes = log_derivative_raa(&raa_pt, &raa_vals);

        // Get eccentricity from Glauber
        let ecc = bins_full
            .iter()
            .find(|b| (b.cent_lo - c_lo).abs() < 0.01 && (b.cent_hi - c_hi).abs() < 0.01)
            .map(|b| b.eccentricity)
            .unwrap_or(0.2);

        // Match v2 data to R_AA pT bins
        for (i, slope) in slopes.iter().enumerate() {
            if slope.is_nan() || i >= raa_pt.len() {
                continue;
            }
            // Find closest v2 point
            let target_pt = raa_pt[i];
            if let Some(v2p) = v2_data
                .iter()
                .min_by(|a, b| {
                    (a.pt - target_pt)
                        .abs()
                        .partial_cmp(&(b.pt - target_pt).abs())
                        .unwrap()
                })
                .filter(|v2p| (v2p.pt - target_pt).abs() < 2.0 && ecc > 0.01)
            {
                v2_slope_data.push(V2SlopePoint {
                    pt: target_pt,
                    v2_over_ecc: v2p.v2 / ecc,
                    v2_over_ecc_err: v2p.stat_err / ecc,
                    dln_raa_dln_pt: *slope,
                    slope_err: 0.01,
                    centrality: format!("{:.0}-{:.0}%", c_lo * 100.0, c_hi * 100.0),
                });
            }
        }
    }

    if v2_slope_data.len() >= 3 {
        let v2_result = fit_v2_relation(&v2_slope_data, true);
        eprintln!(
            "  v2 relation: beta = {:.3} +/- {:.3} (R^2 = {:.4}, chi2/ndf = {:.3})",
            v2_result.beta,
            v2_result.beta_err,
            v2_result.r_squared,
            v2_result.chi2 / v2_result.ndf as f64,
        );
    } else {
        eprintln!(
            "  v2 relation: insufficient matched data ({} points, need >= 3)",
            v2_slope_data.len(),
        );
    }

    // Step 5: Multi-system density scaling with per-system K (Pb-Pb + Xe-Xe)
    eprintln!("[5/6] Multi-system density scaling (shared beta, per-system K)...");
    let mut multi_system_data = Vec::new();
    let n_spectral_full = 6.1;

    // 5a: Re-extract ALICE Pb-Pb epsilon_bar for the multi-system pool
    let alice_dir_ms = PathBuf::from(data_dir).join(hic_raa::ALICE_PBPB_RAA_DIR);
    let pbpb_mult = multiplicity::alice_pbpb_5020_multiplicity();
    let broad_cents_ms: Vec<(f64, f64, Vec<usize>)> = vec![
        (0.00, 0.05, vec![1]),
        (0.05, 0.10, vec![2]),
        (0.10, 0.20, vec![3, 4]),
        (0.20, 0.30, vec![5, 6]),
        (0.30, 0.40, vec![7, 8]),
        (0.40, 0.50, vec![9, 10]),
        (0.50, 0.60, vec![11, 12]),
        (0.60, 0.70, vec![13, 14]),
    ];

    for (c_lo, c_hi, tables) in &broad_cents_ms {
        let mut all_data = Vec::new();
        for &t in tables {
            let path = alice_dir_ms.join(format!("table_{}.csv", t));
            if let Ok(pts) = hic_raa::parse_raa_csv(&path) {
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
        }
        if all_data.is_empty() {
            continue;
        }

        let eps = extract_epsilon(&all_data, n_spectral_full, 0.1, 20.0, 1e-6);
        let g = bins_full
            .iter()
            .find(|b| (b.cent_lo - c_lo).abs() < 0.01 && (b.cent_hi - c_hi).abs() < 0.01);
        let m = pbpb_mult
            .iter()
            .find(|m| (m.cent_lo - c_lo).abs() < 0.01 && (m.cent_hi - c_hi).abs() < 0.01);
        if let (Some(g), Some(m)) = (g, m) {
            multi_system_data.push(DensityScalingPoint {
                epsilon_bar: eps.epsilon_bar,
                epsilon_bar_err: (eps.err_up + eps.err_down) / 2.0,
                dnch_dy: m.dnch_dy(),
                a_perp: g.a_perp,
                l_avg: g.l_avg,
                system: "ALICE Pb-Pb 5.02 TeV".to_string(),
                centrality: format!("{:.0}-{:.0}%", c_lo * 100.0, c_hi * 100.0),
            });
        }
    }
    let n_pbpb = multi_system_data.len();
    eprintln!("  Pb-Pb: {} centrality bins loaded", n_pbpb);

    // 5b: ALICE Xe-Xe 5.44 TeV epsilon extraction
    let xe = NucleusParams::xe129();
    let sigma_xe = SigmaNN::lhc_5440();
    let bins_xe = compute_centrality_bins(&edges_full, &sigma_xe, &xe, n_gl, 300);
    let xexe_mult = multiplicity::alice_xexe_5440_multiplicity();
    let xexe_dir = PathBuf::from(data_dir).join(hic_raa::ALICE_XEXE_RAA_DIR);

    let xexe_cents: Vec<(f64, f64, usize)> = vec![
        (0.00, 0.05, 1),
        (0.05, 0.10, 2),
        (0.10, 0.20, 3),
        (0.20, 0.30, 4),
        (0.30, 0.40, 5),
        (0.40, 0.50, 6),
        (0.50, 0.60, 7),
        (0.60, 0.70, 8),
    ];

    for (c_lo, c_hi, table_idx) in &xexe_cents {
        let path = xexe_dir.join(format!("alice_xexe_raa_table{}.csv", table_idx));
        let raa_data = match hic_raa::parse_raa_csv(&path) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let fit_data: Vec<RaaDataPoint> = raa_data
            .iter()
            .filter(|p| p.pt >= pt_min)
            .map(|p| RaaDataPoint {
                pt: p.pt,
                raa: p.raa,
                stat_err: p.stat_err,
                syst_err: p.syst_err_up,
            })
            .collect();
        if fit_data.is_empty() {
            continue;
        }

        let eps = extract_epsilon(&fit_data, n_spectral_full, 0.1, 20.0, 1e-6);
        let g = bins_xe
            .iter()
            .find(|b| (b.cent_lo - c_lo).abs() < 0.01 && (b.cent_hi - c_hi).abs() < 0.01);
        let m = xexe_mult
            .iter()
            .find(|m| (m.cent_lo - c_lo).abs() < 0.01 && (m.cent_hi - c_hi).abs() < 0.01);
        if let (Some(g), Some(m)) = (g, m) {
            multi_system_data.push(DensityScalingPoint {
                epsilon_bar: eps.epsilon_bar,
                epsilon_bar_err: (eps.err_up + eps.err_down) / 2.0,
                dnch_dy: m.dnch_dy(),
                a_perp: g.a_perp,
                l_avg: g.l_avg,
                system: "ALICE Xe-Xe 5.44 TeV".to_string(),
                centrality: format!("{:.0}-{:.0}%", c_lo * 100.0, c_hi * 100.0),
            });
        }
    }
    let n_xexe = multi_system_data.len() - n_pbpb;
    eprintln!("  Xe-Xe: {} centrality bins loaded", n_xexe);

    // 5c: Multi-system fit with per-system K and shared beta
    if multi_system_data.len() >= 4 {
        let multi_result = fit_density_scaling_multi_k(&multi_system_data, 0.0, 3.0, 0.01);
        eprintln!();
        eprintln!("  === Multi-System Density Scaling (shared beta, per-system K) ===");
        eprintln!(
            "  beta     = {:.3} +{:.3} -{:.3}",
            multi_result.beta, multi_result.beta_err_up, multi_result.beta_err_down,
        );
        for (sys, k, ke) in &multi_result.k_per_system {
            eprintln!("  K({})  = {:.4} +/- {:.4}", sys, k, ke);
        }
        eprintln!(
            "  chi2/ndf = {:.3} ({:.1} / {})",
            multi_result.chi2_per_ndf, multi_result.chi2_min, multi_result.ndf,
        );
        eprintln!();
        let beta_ok = (multi_result.beta - 1.02).abs() < 0.3;
        eprintln!("  Comparison with Arleo-Falmagne (beta = 1.02 +0.09/-0.06):");
        eprintln!(
            "  beta within 0.3 of reference: {}",
            if beta_ok { "PASS" } else { "MARGINAL" }
        );
    } else {
        // Fallback to single-K fit if insufficient multi-system data
        eprintln!(
            "  Multi-system: insufficient data ({} points, need >= 4)",
            multi_system_data.len()
        );
        if multi_system_data.len() >= 3 {
            let single_result = fit_density_scaling(&multi_system_data, 0.0, 3.0, 0.01);
            eprintln!(
                "  Single-K fallback: beta = {:.3}, K = {:.4}, chi2/ndf = {:.3}",
                single_result.beta, single_result.k_constant, single_result.chi2_per_ndf
            );
        }
    }

    // Step 6: Summary
    eprintln!();
    eprintln!("[6/6] Pipeline complete.");
    eprintln!("  ALICE single-system: see above (step 3)");
    if v2_slope_data.len() >= 3 {
        eprintln!("  v2 relation: fitted ({} points)", v2_slope_data.len());
    }
    eprintln!(
        "  Multi-system: {} Pb-Pb + {} Xe-Xe = {} total points",
        n_pbpb,
        n_xexe,
        multi_system_data.len()
    );
    eprintln!(
        "Note: Full multi-system analysis requires Xe-Xe data in {}/{}.",
        data_dir,
        hic_raa::ALICE_XEXE_RAA_DIR
    );
}

fn run_bic_compare(data_dir: &str, pt_min: f64) {
    eprintln!("=== BIC Model Comparison: Arleo-Falmagne vs CUJET3.0 vs Frac. Langevin ===");
    eprintln!();

    // Load ALICE Pb-Pb R_AA data (0-5% centrality for comparison with CUJET3.0)
    let alice_dir = PathBuf::from(data_dir).join(hic_raa::ALICE_PBPB_RAA_DIR);
    let path_0_5 = alice_dir.join("table_1.csv"); // 0-5% centrality

    let measured_data: Vec<MeasuredRaaPoint> = match hic_raa::parse_raa_csv(&path_0_5) {
        Ok(pts) => pts
            .iter()
            .filter(|p| p.pt >= pt_min)
            .map(|p| {
                let total_err = (p.stat_err * p.stat_err + p.syst_err_up * p.syst_err_up).sqrt();
                MeasuredRaaPoint {
                    pt: p.pt,
                    raa: p.raa,
                    total_err,
                }
            })
            .collect(),
        Err(e) => {
            eprintln!("ERROR: Could not load ALICE 0-5% R_AA: {}", e);
            eprintln!("  Expected: {}", path_0_5.display());
            eprintln!("  Run 'arleo-falmagne-scaling full' first to download data.");
            return;
        }
    };

    if measured_data.is_empty() {
        eprintln!("ERROR: No ALICE data points above pT > {:.1} GeV", pt_min);
        return;
    }

    eprintln!(
        "  ALICE Pb-Pb 5.02 TeV, 0-5%: {} points (pT > {:.1} GeV)",
        measured_data.len(),
        pt_min
    );
    eprintln!();

    // Build model curves
    // 1. CUJET3.0 at 2.76 TeV, 0-5% (closest available centrality)
    let cujet3 = competing_models::cujet3_pbpb_2760_0_5();

    // 2. Fractional Langevin at 5.02 TeV, 0-10%
    let langevin = competing_models::langevin_pbpb_5020_0_10();

    // 3. Arleo-Falmagne: compute R_AA from extracted epsilon_bar
    //    We need to extract epsilon_bar first, then build the curve
    let n_spectral = 6.1;
    let fit_data: Vec<RaaDataPoint> = measured_data
        .iter()
        .map(|p| RaaDataPoint {
            pt: p.pt,
            raa: p.raa,
            stat_err: p.total_err * 0.7, // approximate split
            syst_err: p.total_err * 0.7,
        })
        .collect();

    let eps_result = extract_epsilon(&fit_data, n_spectral, 0.1, 20.0, 1e-6);
    let pt_points: Vec<f64> = measured_data.iter().map(|p| p.pt).collect();
    let af_curve = arleo_falmagne_raa(eps_result.epsilon_bar, n_spectral, &pt_points);

    eprintln!(
        "  Arleo-Falmagne: epsilon_bar = {:.3} GeV (extracted from 0-5%)",
        eps_result.epsilon_bar
    );
    eprintln!(
        "  CUJET3.0: {} (pT range {:.0}-{:.0} GeV)",
        cujet3.name,
        cujet3.pt_range().0,
        cujet3.pt_range().1
    );
    eprintln!(
        "  Frac. Langevin: {} (pT range {:.0}-{:.0} GeV)",
        langevin.name,
        langevin.pt_range().0,
        langevin.pt_range().1
    );
    eprintln!();

    // Compute BIC for all models
    let results = compare_models(&[af_curve, cujet3, langevin], &measured_data);

    // Display results table
    eprintln!(
        "  {:>30} {:>5} {:>8} {:>6} {:>10} {:>8} {:>10}",
        "Model", "k", "chi2", "n", "chi2/n", "excl", "BIC"
    );
    eprintln!("  {}", "-".repeat(87));

    let best_bic = results[0].bic;
    for r in &results {
        let delta_bic = r.bic - best_bic;
        let chi2_per_n = if r.n_points > 0 {
            r.chi2 / r.n_points as f64
        } else {
            f64::INFINITY
        };
        eprintln!(
            "  {:>30} {:>5} {:>8.1} {:>6} {:>10.3} {:>8} {:>10.1}  (dBIC={:.1})",
            r.model_name,
            r.n_params,
            r.chi2,
            r.n_points,
            chi2_per_n,
            r.n_excluded,
            r.bic,
            delta_bic
        );
    }

    eprintln!();

    // Interpretation
    let best = &results[0];
    if results.len() >= 2 {
        let delta = results[1].bic - best.bic;
        let strength = if delta > 10.0 {
            "VERY STRONG"
        } else if delta > 6.0 {
            "STRONG"
        } else if delta > 2.0 {
            "POSITIVE"
        } else {
            "WEAK"
        };
        eprintln!("  Best model: {} (BIC = {:.1})", best.model_name, best.bic);
        eprintln!(
            "  Evidence strength vs second-best: {} (delta-BIC = {:.1})",
            strength, delta
        );
        eprintln!();
        eprintln!("  Kass-Raftery scale: 0-2 weak, 2-6 positive, 6-10 strong, >10 very strong");
    }

    // Caveats
    eprintln!();
    eprintln!("  CAVEATS:");
    eprintln!("  - CUJET3.0 predictions are at 2.76 TeV (not 5.02 TeV)");
    eprintln!("  - Frac. Langevin is for D mesons (heavy quarks), not light hadrons");
    eprintln!("  - Digitized values have ~5% reading uncertainty");
    eprintln!("  - Arleo-Falmagne is self-consistent (fitted to same data)");
}

#[cfg(test)]
mod tests {
    use super::{RaaDataPoint, alice_straggling_pt_range};
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

    #[test]
    fn test_alice_straggling_pt_range_uses_filtered_samples() {
        let centrality_data = vec![
            (
                0.0,
                0.05,
                vec![
                    RaaDataPoint {
                        pt: 0.35,
                        raa: 0.25,
                        stat_err: 0.01,
                        syst_err: 0.02,
                    },
                    RaaDataPoint {
                        pt: 12.0,
                        raa: 0.55,
                        stat_err: 0.01,
                        syst_err: 0.02,
                    },
                ],
            ),
            (
                0.05,
                0.10,
                vec![RaaDataPoint {
                    pt: 125.0,
                    raa: 0.75,
                    stat_err: 0.02,
                    syst_err: 0.03,
                }],
            ),
        ];

        let range = alice_straggling_pt_range(&centrality_data).expect("range from data");
        assert!(
            (range.0 - 0.35).abs() < 1e-12,
            "grid pt_min should follow filtered data, got {}",
            range.0
        );
        assert!(
            (range.1 - 125.0).abs() < 1e-12,
            "grid pt_max should follow filtered data, got {}",
            range.1
        );
    }
}
