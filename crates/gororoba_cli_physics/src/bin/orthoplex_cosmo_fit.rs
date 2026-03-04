//! orthoplex-cosmo-fit: Fit orthoplex dark energy model to real observational data.
//!
//! Downloads Pantheon+ SN Ia and uses DESI DR1 BAO measurements to fit the
//! orthoplex dark energy model (diffusion on K_{2,2,...,2} complete multipartite
//! graph). Compares against Lambda-CDM. Outputs w(z) table to CSV.
//!
//! Usage:
//!   orthoplex-cosmo-fit                         # Full pipeline
//!   orthoplex-cosmo-fit --skip-download         # Use cached data
//!   orthoplex-cosmo-fit --k 63 --z-max 3.0      # K_{2,2,...,2} with k=63 parts
//!   orthoplex-cosmo-fit --csv orthoplex_w.csv   # Custom output path
//!   orthoplex-cosmo-fit --json                  # JSON output

use clap::Parser;
use cosmology_core::{
    RealBaoData, compare_orthoplex, desi_to_real_bao, filter_pantheon_data, w_of_z_table,
    w_orthoplex,
};
use data_core::{
    catalogs::{
        desi_bao::desi_dr1_bao,
        pantheon::{PantheonProvider, parse_pantheon_dat},
    },
    fetcher::{DatasetProvider, FetchConfig},
};
use std::{io::Write, path::PathBuf};

#[derive(Parser)]
#[command(name = "orthoplex-cosmo-fit")]
#[command(about = "Fit orthoplex dark energy to Pantheon+ SN Ia + DESI DR1 BAO data")]
struct Args {
    /// Data directory for downloaded files.
    #[arg(long, default_value = "data/external")]
    data_dir: String,

    /// Skip downloading (use cached data only).
    #[arg(long)]
    skip_download: bool,

    /// Minimum CMB-frame redshift for SN sample.
    #[arg(long, default_value = "0.01")]
    z_min: f64,

    /// Include Cepheid calibrators in SN sample.
    #[arg(long)]
    include_calibrators: bool,

    /// Number of parts k in K_{2,2,...,2} graph (default: k=63 for dim-256 CD algebra).
    #[arg(long, default_value = "63")]
    k: usize,

    /// Maximum redshift for w(z) CSV output.
    #[arg(long, default_value = "3.0")]
    z_max: f64,

    /// Number of points in w(z) CSV output.
    #[arg(long, default_value = "301")]
    n_points: usize,

    /// Output path for w(z) CSV file.
    #[arg(long, default_value = "orthoplex_w_of_z.csv")]
    csv: String,

    /// Output as JSON (fit results only, no CSV).
    #[arg(long)]
    json: bool,
}

fn main() {
    let args = Args::parse();
    let config = FetchConfig {
        output_dir: PathBuf::from(&args.data_dir),
        skip_existing: true,
        verify_checksums: true,
    };

    // -----------------------------------------------------------------------
    // Step 1: Acquire Pantheon+ SN Ia data
    // -----------------------------------------------------------------------
    eprintln!("=== Orthoplex Dark Energy Fit ===");
    eprintln!(
        "    Graph topology: K_{{2,2,...,2}} with k={} parts (n={} vertices)",
        args.k,
        2 * args.k
    );
    eprintln!();

    let pantheon_path = if args.skip_download {
        let p = config.output_dir.join("PantheonPlusSH0ES.dat");
        if !p.exists() {
            eprintln!("ERROR: Pantheon+ data not found at {}", p.display());
            eprintln!("Run without --skip-download to fetch data first.");
            std::process::exit(1);
        }
        p
    } else {
        eprintln!("[1/5] Downloading Pantheon+ SH0ES data...");
        match PantheonProvider.fetch(&config) {
            Ok(p) => {
                eprintln!("      OK: {}", p.display());
                p
            }
            Err(e) => {
                eprintln!("ERROR: Failed to download Pantheon+ data: {e}");
                std::process::exit(1);
            }
        }
    };

    // -----------------------------------------------------------------------
    // Step 2: Parse and filter Pantheon+ data
    // -----------------------------------------------------------------------
    eprintln!("[2/5] Parsing Pantheon+ SN Ia data...");
    let sne = match parse_pantheon_dat(&pantheon_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR: Failed to parse Pantheon+ data: {e}");
            std::process::exit(1);
        }
    };
    eprintln!("      Parsed {} raw supernovae", sne.len());

    let z_cmb: Vec<f64> = sne.iter().map(|s| s.z_cmb).collect();
    let mu: Vec<f64> = sne.iter().map(|s| s.mu).collect();
    let mu_err: Vec<f64> = sne.iter().map(|s| s.mu_err).collect();
    let is_cal: Vec<bool> = sne.iter().map(|s| s.is_calibrator).collect();

    let sn_data = filter_pantheon_data(
        &z_cmb,
        &mu,
        &mu_err,
        &is_cal,
        args.z_min,
        args.include_calibrators,
    );
    eprintln!(
        "      After filtering (z_min={}, excl. calibrators={}): {} SNe",
        args.z_min, !args.include_calibrators, sn_data.n_sne,
    );

    if sn_data.n_sne == 0 {
        eprintln!("ERROR: No SNe passed filtering.");
        std::process::exit(1);
    }

    // -----------------------------------------------------------------------
    // Step 3: Load DESI DR1 BAO measurements
    // -----------------------------------------------------------------------
    eprintln!("[3/5] Loading DESI DR1 BAO measurements...");
    let desi = desi_dr1_bao();
    let bao_data: RealBaoData = desi_to_real_bao(
        &desi.iter().map(|b| b.z_eff).collect::<Vec<f64>>(),
        &desi.iter().map(|b| b.is_isotropic).collect::<Vec<bool>>(),
        &desi.iter().map(|b| b.dm_over_rd).collect::<Vec<f64>>(),
        &desi.iter().map(|b| b.dm_over_rd_err).collect::<Vec<f64>>(),
        &desi.iter().map(|b| b.dh_over_rd).collect::<Vec<f64>>(),
        &desi.iter().map(|b| b.dh_over_rd_err).collect::<Vec<f64>>(),
        &desi.iter().map(|b| b.rho).collect::<Vec<f64>>(),
        &desi
            .iter()
            .map(|b| b.tracer.clone())
            .collect::<Vec<String>>(),
    );
    let n_bao_data = cosmology_core::bao_data_point_count(&bao_data);
    eprintln!(
        "      Loaded {} BAO bins ({} data points)",
        bao_data.z_eff.len(),
        n_bao_data
    );

    // -----------------------------------------------------------------------
    // Step 4: Fit models and compare
    // -----------------------------------------------------------------------
    eprintln!("[4/5] Fitting Lambda-CDM and orthoplex models...");
    eprintln!();

    let comparison = compare_orthoplex(&sn_data, &bao_data, args.k);

    if args.json {
        print_json(&comparison, sn_data.n_sne, bao_data.z_eff.len());
    } else {
        print_report(&comparison, sn_data.n_sne, bao_data.z_eff.len());
    }

    // -----------------------------------------------------------------------
    // Step 5: Write w(z) CSV
    // -----------------------------------------------------------------------
    eprintln!("[5/5] Writing w(z) table to {}...", args.csv);
    let orth = &comparison.orthoplex;
    let table = w_of_z_table(
        orth.k,
        orth.alpha,
        orth.beta,
        orth.t_0,
        args.z_max,
        args.n_points,
    );

    let mut file = match std::fs::File::create(&args.csv) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("ERROR: Cannot create {}: {e}", args.csv);
            std::process::exit(1);
        }
    };

    writeln!(file, "z,w,d_s,t,beta_ds").unwrap();
    for &(z, w, ds, t, beta_ds) in &table {
        writeln!(file, "{z:.6},{w:.8},{ds:.8},{t:.8},{beta_ds:.8}").unwrap();
    }

    eprintln!("      Wrote {} rows to {}", table.len(), args.csv);
    eprintln!();
    eprintln!("Done.");
}

fn print_report(c: &cosmology_core::OrthoplexComparison, n_sne: usize, n_bao: usize) {
    let n_data_total = c.lcdm.n_data;
    let dof_lcdm = n_data_total as f64 - c.lcdm.n_params as f64;
    let dof_orthoplex = c.orthoplex.n_data as f64 - c.orthoplex.n_params as f64;

    println!("================================================================");
    println!("    ORTHOPLEX DARK ENERGY FIT RESULTS");
    println!("================================================================");
    println!();
    println!("Data summary:");
    println!("  Pantheon+ SN Ia:     {} supernovae", n_sne);
    println!(
        "  DESI DR1 BAO:        {} bins ({} data pts)",
        n_bao,
        n_data_total - n_sne
    );
    println!("  Total data points:   {}", n_data_total);
    println!();
    println!("----------------------------------------------------------------");
    println!("  Lambda-CDM (2 params: Omega_m, H_0)");
    println!("----------------------------------------------------------------");
    println!("  Omega_m       = {:.4}", c.lcdm.omega_m);
    println!("  H_0           = {:.2} km/s/Mpc", c.lcdm.h0);
    println!("  chi2_total    = {:.2}", c.lcdm.chi2_total);
    println!("    chi2_SN     = {:.2}", c.lcdm.chi2_sn);
    println!("    chi2_BAO    = {:.2}", c.lcdm.chi2_bao);
    println!(
        "  chi2/dof      = {:.3} ({:.0}/{:.0})",
        c.lcdm.chi2_total / dof_lcdm,
        c.lcdm.chi2_total,
        dof_lcdm
    );
    println!("  AIC           = {:.2}", c.lcdm.aic);
    println!("  BIC           = {:.2}", c.lcdm.bic);
    println!();
    println!("----------------------------------------------------------------");
    println!(
        "  Orthoplex K_{{2,2,...,2}} k={} (5 params: Omega_m, H_0, alpha, beta, t_0)",
        c.orthoplex.k
    );
    println!("----------------------------------------------------------------");
    println!("  Omega_m       = {:.4}", c.orthoplex.omega_m);
    println!("  H_0           = {:.2} km/s/Mpc", c.orthoplex.h0);
    println!("  alpha         = {:.4}", c.orthoplex.alpha);
    println!("  beta          = {:.6}", c.orthoplex.beta);
    println!("  t_0           = {:.4}", c.orthoplex.t_0);
    println!("  w(z=0)        = {:.6}", c.orthoplex.w_0);
    println!("  w(z=2)        = {:.6}", c.orthoplex.w_high_z);
    println!("  chi2_total    = {:.2}", c.orthoplex.chi2_total);
    println!("    chi2_SN     = {:.2}", c.orthoplex.chi2_sn);
    println!("    chi2_BAO    = {:.2}", c.orthoplex.chi2_bao);
    println!(
        "  chi2/dof      = {:.3} ({:.0}/{:.0})",
        c.orthoplex.chi2_total / dof_orthoplex,
        c.orthoplex.chi2_total,
        dof_orthoplex
    );
    println!("  AIC           = {:.2}", c.orthoplex.aic);
    println!("  BIC           = {:.2}", c.orthoplex.bic);
    println!();

    // w(z) at key redshifts
    println!("  w(z) at key redshifts:");
    for z in [0.0, 0.3, 0.5, 1.0, 2.0] {
        let w = w_orthoplex(
            z,
            c.orthoplex.k,
            c.orthoplex.alpha,
            c.orthoplex.beta,
            c.orthoplex.t_0,
        );
        println!("    w(z={z:.1})      = {w:.6}");
    }

    println!();
    println!("================================================================");
    println!("  MODEL COMPARISON");
    println!("================================================================");
    println!("  Delta AIC  = {:.2} (orthoplex - LCDM)", c.delta_aic);
    println!("  Delta BIC  = {:.2} (orthoplex - LCDM)", c.delta_bic);
    println!();

    if c.delta_bic > 10.0 {
        println!("  Verdict: Very strong evidence for Lambda-CDM over orthoplex.");
    } else if c.delta_bic > 6.0 {
        println!("  Verdict: Strong evidence for Lambda-CDM over orthoplex.");
    } else if c.delta_bic > 2.0 {
        println!("  Verdict: Positive evidence for Lambda-CDM over orthoplex.");
    } else if c.delta_bic > -2.0 {
        println!("  Verdict: No significant difference between models.");
    } else if c.delta_bic > -6.0 {
        println!("  Verdict: Positive evidence for orthoplex over Lambda-CDM.");
    } else {
        println!("  Verdict: Strong evidence for orthoplex over Lambda-CDM.");
    }

    println!();
    println!("  (Kass & Raftery 1995 BIC interpretation scale)");
    println!("================================================================");
}

fn print_json(c: &cosmology_core::OrthoplexComparison, n_sne: usize, n_bao: usize) {
    println!("{{");
    println!("  \"data\": {{");
    println!("    \"n_sne\": {},", n_sne);
    println!("    \"n_bao_bins\": {},", n_bao);
    println!("    \"n_data_total\": {}", c.lcdm.n_data);
    println!("  }},");
    println!("  \"lcdm\": {{");
    println!("    \"omega_m\": {:.6},", c.lcdm.omega_m);
    println!("    \"h0\": {:.4},", c.lcdm.h0);
    println!("    \"chi2_total\": {:.4},", c.lcdm.chi2_total);
    println!("    \"chi2_sn\": {:.4},", c.lcdm.chi2_sn);
    println!("    \"chi2_bao\": {:.4},", c.lcdm.chi2_bao);
    println!("    \"aic\": {:.4},", c.lcdm.aic);
    println!("    \"bic\": {:.4}", c.lcdm.bic);
    println!("  }},");
    println!("  \"orthoplex\": {{");
    println!("    \"k\": {},", c.orthoplex.k);
    println!("    \"omega_m\": {:.6},", c.orthoplex.omega_m);
    println!("    \"h0\": {:.4},", c.orthoplex.h0);
    println!("    \"alpha\": {:.4},", c.orthoplex.alpha);
    println!("    \"beta\": {:.6},", c.orthoplex.beta);
    println!("    \"t_0\": {:.4},", c.orthoplex.t_0);
    println!("    \"w_0\": {:.6},", c.orthoplex.w_0);
    println!("    \"w_high_z\": {:.6},", c.orthoplex.w_high_z);
    println!("    \"chi2_total\": {:.4},", c.orthoplex.chi2_total);
    println!("    \"chi2_sn\": {:.4},", c.orthoplex.chi2_sn);
    println!("    \"chi2_bao\": {:.4},", c.orthoplex.chi2_bao);
    println!("    \"aic\": {:.4},", c.orthoplex.aic);
    println!("    \"bic\": {:.4}", c.orthoplex.bic);
    println!("  }},");
    println!("  \"comparison\": {{");
    println!("    \"delta_aic\": {:.4},", c.delta_aic);
    println!("    \"delta_bic\": {:.4}", c.delta_bic);
    println!("  }}");
    println!("}}");
}
