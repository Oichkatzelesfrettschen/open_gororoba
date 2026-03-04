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
    OrthoplexComparison, RealBaoData, compare_orthoplex_all, compare_orthoplex_fixed_beta,
    cosmic_chronometer_data, desi_to_real_bao, filter_pantheon_data, growth_rate_data,
    w_of_z_table, w_orthoplex,
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

    /// Run only the 4-param fixed-beta (beta=1.0) fit.
    #[arg(long)]
    fixed_beta: bool,
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
    // Step 4: Load cosmic chronometer H(z) + f*sigma8 growth rate data
    // -----------------------------------------------------------------------
    let cc_data = cosmic_chronometer_data();
    let fsig_data = growth_rate_data();
    eprintln!(
        "[4/6] Loaded {} CC H(z) + {} f*sigma8 growth rate measurements",
        cc_data.len(),
        fsig_data.len(),
    );

    // -----------------------------------------------------------------------
    // Step 5: Fit models and compare
    // -----------------------------------------------------------------------
    if args.fixed_beta {
        eprintln!("[5/6] Fitting Lambda-CDM and orthoplex (fixed beta=1.0, 4 params)...");
    } else {
        eprintln!("[5/6] Fitting Lambda-CDM and orthoplex (free + fixed beta)...");
    }
    eprintln!();

    let comparison = if args.fixed_beta {
        compare_orthoplex_fixed_beta(&sn_data, &bao_data, &cc_data, &fsig_data, args.k)
    } else {
        compare_orthoplex_all(&sn_data, &bao_data, &cc_data, &fsig_data, args.k)
    };

    if args.json {
        print_json(&comparison, sn_data.n_sne, bao_data.z_eff.len());
    } else {
        print_report(&comparison, sn_data.n_sne, bao_data.z_eff.len());
    }

    // -----------------------------------------------------------------------
    // Step 6: Write w(z) CSV
    // -----------------------------------------------------------------------
    eprintln!("[6/6] Writing w(z) table to {}...", args.csv);
    // Use fixed-beta result if available and better, otherwise free-beta.
    let orth = if let Some(ref fb) = comparison.orthoplex_fixed_beta {
        if fb.chi2_total <= comparison.orthoplex.chi2_total + 1.0 { fb } else { &comparison.orthoplex }
    } else {
        &comparison.orthoplex
    };
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

fn print_report(c: &OrthoplexComparison, n_sne: usize, n_bao: usize) {
    let n_data_total = c.lcdm.n_data;
    let n_cc = cosmology_core::cosmic_chronometer_data().len();
    let dof_lcdm = n_data_total as f64 - c.lcdm.n_params as f64;
    let dof_orthoplex = c.orthoplex.n_data as f64 - c.orthoplex.n_params as f64;

    println!("================================================================");
    println!("    ORTHOPLEX DARK ENERGY FIT RESULTS");
    println!("================================================================");
    println!();
    let n_bao_pts = n_data_total - n_sne - 1 - n_cc;
    println!("Data summary:");
    println!("  Pantheon+ SN Ia:     {} supernovae", n_sne);
    println!(
        "  DESI DR1 BAO:        {} bins ({} data pts)",
        n_bao, n_bao_pts
    );
    println!("  CMB shift parameter: 1 data point (R = 1.7502 +/- 0.0046)");
    println!("  Cosmic chronometers: {} H(z) measurements", n_cc);
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
    println!("    chi2_CMB    = {:.2}", c.lcdm.chi2_cmb);
    println!("    chi2_CC     = {:.2}", c.lcdm.chi2_cc);
    println!("    chi2_fsig8  = {:.2}", c.lcdm.chi2_fsig);
    println!(
        "  chi2/dof      = {:.3} ({:.0}/{:.0})",
        c.lcdm.chi2_total / dof_lcdm,
        c.lcdm.chi2_total,
        dof_lcdm
    );
    println!("  AIC           = {:.2}", c.lcdm.aic);
    println!("  BIC           = {:.2}", c.lcdm.bic);

    // Print free-beta section only if it was fitted (not fixed-beta-only mode)
    if !c.orthoplex.beta_fixed {
        println!();
        println!("----------------------------------------------------------------");
        println!(
            "  Orthoplex K_{{2,2,...,2}} k={} (5 params: Omega_m, H_0, alpha, beta, t_0)",
            c.orthoplex.k
        );
        println!("----------------------------------------------------------------");
        print_orthoplex_result(&c.orthoplex, dof_orthoplex);
    }

    // Print fixed-beta section
    if let Some(ref fb) = c.orthoplex_fixed_beta {
        let dof_fb = fb.n_data as f64 - fb.n_params as f64;
        println!();
        println!("----------------------------------------------------------------");
        println!(
            "  Orthoplex K_{{2,2,...,2}} k={} (4 params: Omega_m, H_0, alpha, t_0; beta=1.0 FIXED)",
            fb.k
        );
        println!("----------------------------------------------------------------");
        print_orthoplex_result(fb, dof_fb);
    } else if c.orthoplex.beta_fixed {
        // Fixed-beta-only mode: result is in c.orthoplex
        println!();
        println!("----------------------------------------------------------------");
        println!(
            "  Orthoplex K_{{2,2,...,2}} k={} (4 params: Omega_m, H_0, alpha, t_0; beta=1.0 FIXED)",
            c.orthoplex.k
        );
        println!("----------------------------------------------------------------");
        print_orthoplex_result(&c.orthoplex, dof_orthoplex);
    }

    println!();
    println!("================================================================");
    println!("  MODEL COMPARISON");
    println!("================================================================");

    if !c.orthoplex.beta_fixed {
        println!("  Free beta (5 params):");
        println!("    Delta AIC  = {:.2} (orthoplex - LCDM)", c.delta_aic);
        println!("    Delta BIC  = {:.2} (orthoplex - LCDM)", c.delta_bic);
        print_bic_verdict("    ", c.delta_bic);
    }

    if let Some(delta_bic_fb) = c.delta_bic_fixed_beta {
        let delta_aic_fb = c.delta_aic_fixed_beta.unwrap_or(0.0);
        println!();
        println!("  Fixed beta=1.0 (4 params):");
        println!("    Delta AIC  = {:.2} (orthoplex - LCDM)", delta_aic_fb);
        println!("    Delta BIC  = {:.2} (orthoplex - LCDM)", delta_bic_fb);
        print_bic_verdict("    ", delta_bic_fb);
    } else if c.orthoplex.beta_fixed {
        println!("  Fixed beta=1.0 (4 params):");
        println!("    Delta AIC  = {:.2} (orthoplex - LCDM)", c.delta_aic);
        println!("    Delta BIC  = {:.2} (orthoplex - LCDM)", c.delta_bic);
        print_bic_verdict("    ", c.delta_bic);
    }

    println!();
    println!("  (Kass & Raftery 1995 BIC interpretation scale)");
    println!("================================================================");
}

fn print_orthoplex_result(orth: &cosmology_core::OrthoplexFitResult, dof: f64) {
    println!("  Omega_m       = {:.4}", orth.omega_m);
    println!("  H_0           = {:.2} km/s/Mpc", orth.h0);
    println!("  alpha         = {:.4}", orth.alpha);
    println!("  beta          = {:.6}{}", orth.beta, if orth.beta_fixed { " (FIXED)" } else { "" });
    println!("  t_0           = {:.6}", orth.t_0);
    println!("  w(z=0)        = {:.6}", orth.w_0);
    println!("  w(z=2)        = {:.6}", orth.w_high_z);
    println!("  chi2_total    = {:.2}", orth.chi2_total);
    println!("    chi2_SN     = {:.2}", orth.chi2_sn);
    println!("    chi2_BAO    = {:.2}", orth.chi2_bao);
    println!("    chi2_CMB    = {:.2}", orth.chi2_cmb);
    println!("    chi2_CC     = {:.2}", orth.chi2_cc);
    println!("    chi2_fsig8  = {:.2}", orth.chi2_fsig);
    println!(
        "  chi2/dof      = {:.3} ({:.0}/{:.0})",
        orth.chi2_total / dof,
        orth.chi2_total,
        dof
    );
    println!("  AIC           = {:.2}", orth.aic);
    println!("  BIC           = {:.2}", orth.bic);

    println!();
    println!("  w(z) at key redshifts:");
    for z in [0.0, 0.3, 0.5, 1.0, 2.0] {
        let w = w_orthoplex(z, orth.k, orth.alpha, orth.beta, orth.t_0);
        println!("    w(z={z:.1})      = {w:.6}");
    }
}

fn print_bic_verdict(prefix: &str, delta_bic: f64) {
    if delta_bic > 10.0 {
        println!("{prefix}Verdict: Very strong evidence for Lambda-CDM over orthoplex.");
    } else if delta_bic > 6.0 {
        println!("{prefix}Verdict: Strong evidence for Lambda-CDM over orthoplex.");
    } else if delta_bic > 2.0 {
        println!("{prefix}Verdict: Positive evidence for Lambda-CDM over orthoplex.");
    } else if delta_bic > -2.0 {
        println!("{prefix}Verdict: No significant difference between models.");
    } else if delta_bic > -6.0 {
        println!("{prefix}Verdict: Positive evidence for orthoplex over Lambda-CDM.");
    } else {
        println!("{prefix}Verdict: Strong evidence for orthoplex over Lambda-CDM.");
    }
}

fn print_json(c: &OrthoplexComparison, n_sne: usize, n_bao: usize) {
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
    println!("    \"chi2_cmb\": {:.4},", c.lcdm.chi2_cmb);
    println!("    \"chi2_cc\": {:.4},", c.lcdm.chi2_cc);
    println!("    \"aic\": {:.4},", c.lcdm.aic);
    println!("    \"bic\": {:.4}", c.lcdm.bic);
    println!("  }},");
    println!("  \"orthoplex\": {{");
    println!("    \"k\": {},", c.orthoplex.k);
    println!("    \"beta_fixed\": {},", c.orthoplex.beta_fixed);
    println!("    \"omega_m\": {:.6},", c.orthoplex.omega_m);
    println!("    \"h0\": {:.4},", c.orthoplex.h0);
    println!("    \"alpha\": {:.4},", c.orthoplex.alpha);
    println!("    \"beta\": {:.6},", c.orthoplex.beta);
    println!("    \"t_0\": {:.6},", c.orthoplex.t_0);
    println!("    \"w_0\": {:.6},", c.orthoplex.w_0);
    println!("    \"w_high_z\": {:.6},", c.orthoplex.w_high_z);
    println!("    \"chi2_total\": {:.4},", c.orthoplex.chi2_total);
    println!("    \"chi2_sn\": {:.4},", c.orthoplex.chi2_sn);
    println!("    \"chi2_bao\": {:.4},", c.orthoplex.chi2_bao);
    println!("    \"chi2_cmb\": {:.4},", c.orthoplex.chi2_cmb);
    println!("    \"chi2_cc\": {:.4},", c.orthoplex.chi2_cc);
    println!("    \"aic\": {:.4},", c.orthoplex.aic);
    println!("    \"bic\": {:.4}", c.orthoplex.bic);
    println!("  }},");
    if let Some(ref fb) = c.orthoplex_fixed_beta {
        println!("  \"orthoplex_fixed_beta\": {{");
        println!("    \"k\": {},", fb.k);
        println!("    \"omega_m\": {:.6},", fb.omega_m);
        println!("    \"h0\": {:.4},", fb.h0);
        println!("    \"alpha\": {:.4},", fb.alpha);
        println!("    \"beta\": {:.6},", fb.beta);
        println!("    \"t_0\": {:.6},", fb.t_0);
        println!("    \"w_0\": {:.6},", fb.w_0);
        println!("    \"w_high_z\": {:.6},", fb.w_high_z);
        println!("    \"chi2_total\": {:.4},", fb.chi2_total);
        println!("    \"aic\": {:.4},", fb.aic);
        println!("    \"bic\": {:.4}", fb.bic);
        println!("  }},");
    }
    println!("  \"comparison\": {{");
    println!("    \"delta_aic\": {:.4},", c.delta_aic);
    print!("    \"delta_bic\": {:.4}", c.delta_bic);
    if let (Some(da), Some(db)) = (c.delta_aic_fixed_beta, c.delta_bic_fixed_beta) {
        println!(",");
        println!("    \"delta_aic_fixed_beta\": {:.4},", da);
        println!("    \"delta_bic_fixed_beta\": {:.4}", db);
    } else {
        println!();
    }
    println!("  }}");
    println!("}}");
}
