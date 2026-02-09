//! real-cosmo-fit: Fit Lambda-CDM and bounce cosmology to real observational data.
//!
//! Downloads Pantheon+ SN Ia and uses DESI DR1 BAO measurements to perform
//! a joint chi-square fit. Reports best-fit parameters, chi2/dof, AIC, BIC,
//! and model comparison (delta_BIC).
//!
//! Usage:
//!   real-cosmo-fit                    # Full pipeline with data download
//!   real-cosmo-fit --skip-download    # Use cached data only
//!   real-cosmo-fit --z-min 0.01       # Set minimum redshift cut
//!   real-cosmo-fit --json             # JSON output

use clap::Parser;
use cosmology_core::{compare_models, desi_to_real_bao, filter_pantheon_data, RealBaoData};
use data_core::catalogs::desi_bao::desi_dr1_bao;
use data_core::catalogs::pantheon::{parse_pantheon_dat, PantheonProvider};
use data_core::fetcher::{DatasetProvider, FetchConfig};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "real-cosmo-fit")]
#[command(about = "Fit Lambda-CDM and bounce cosmology to Pantheon+ SN Ia + DESI DR1 BAO data")]
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

    /// Output as JSON.
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
    eprintln!("=== Real Cosmology Fit ===");
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
        eprintln!("[1/4] Downloading Pantheon+ SH0ES data...");
        match PantheonProvider.fetch(&config) {
            Ok(p) => {
                eprintln!("      OK: {}", p.display());
                p
            }
            Err(e) => {
                eprintln!("ERROR: Failed to download Pantheon+ data: {}", e);
                std::process::exit(1);
            }
        }
    };

    // -----------------------------------------------------------------------
    // Step 2: Parse and filter Pantheon+ data
    // -----------------------------------------------------------------------
    eprintln!("[2/4] Parsing Pantheon+ SN Ia data...");
    let sne = match parse_pantheon_dat(&pantheon_path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR: Failed to parse Pantheon+ data: {}", e);
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
        eprintln!("ERROR: No SNe passed filtering. Check z_min or data quality.");
        std::process::exit(1);
    }

    // -----------------------------------------------------------------------
    // Step 3: Load DESI DR1 BAO measurements
    // -----------------------------------------------------------------------
    eprintln!("[3/4] Loading DESI DR1 BAO measurements (7 bins: 5 anisotropic + 2 isotropic)...");
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
    for b in &desi {
        if b.is_isotropic {
            eprintln!(
                "        z={:.3} ({:>10}): DV/rd={:.2}+/-{:.2} [isotropic]",
                b.z_eff, b.tracer, b.dm_over_rd, b.dm_over_rd_err
            );
        } else {
            eprintln!(
                "        z={:.3} ({:>10}): DM/rd={:.2}+/-{:.2}, DH/rd={:.2}+/-{:.2}, rho={:.3}",
                b.z_eff,
                b.tracer,
                b.dm_over_rd,
                b.dm_over_rd_err,
                b.dh_over_rd,
                b.dh_over_rd_err,
                b.rho
            );
        }
    }

    // -----------------------------------------------------------------------
    // Step 4: Fit models and compare
    // -----------------------------------------------------------------------
    eprintln!("[4/4] Fitting Lambda-CDM and bounce models...");
    eprintln!();

    let comparison = compare_models(&sn_data, &bao_data);

    if args.json {
        print_json(&comparison, sn_data.n_sne, bao_data.z_eff.len());
    } else {
        print_report(&comparison, sn_data.n_sne, bao_data.z_eff.len());
    }
}

fn print_report(c: &cosmology_core::ModelComparison, n_sne: usize, n_bao: usize) {
    let n_data_total = c.lcdm.n_data;
    let dof_lcdm = n_data_total as f64 - c.lcdm.n_params as f64;
    let dof_bounce = n_data_total as f64 - c.bounce.n_params as f64;

    println!("================================================================");
    println!("    REAL OBSERVATIONAL COSMOLOGY FIT RESULTS");
    println!("================================================================");
    println!();
    println!("Data summary:");
    println!("  Pantheon+ SN Ia:     {} supernovae", n_sne);
    println!(
        "  DESI DR1 BAO:        {} bins (5 anisotropic + 2 isotropic = {} data pts)",
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
    println!("  Bounce Cosmology (3 params: Omega_m, H_0, q_corr)");
    println!("----------------------------------------------------------------");
    println!("  Omega_m       = {:.4}", c.bounce.omega_m);
    println!("  H_0           = {:.2} km/s/Mpc", c.bounce.h0);
    println!("  q_corr        = {:.2e}", c.bounce.q_corr);
    println!("  chi2_total    = {:.2}", c.bounce.chi2_total);
    println!("    chi2_SN     = {:.2}", c.bounce.chi2_sn);
    println!("    chi2_BAO    = {:.2}", c.bounce.chi2_bao);
    println!(
        "  chi2/dof      = {:.3} ({:.0}/{:.0})",
        c.bounce.chi2_total / dof_bounce,
        c.bounce.chi2_total,
        dof_bounce
    );
    println!("  AIC           = {:.2}", c.bounce.aic);
    println!("  BIC           = {:.2}", c.bounce.bic);
    println!("  n_s (bounce)  = {:.4}", c.n_s_bounce);
    println!();
    println!("================================================================");
    println!("  MODEL COMPARISON");
    println!("================================================================");
    println!("  Delta AIC  = {:.2} (bounce - LCDM)", c.delta_aic);
    println!("  Delta BIC  = {:.2} (bounce - LCDM)", c.delta_bic);
    println!();

    if c.delta_bic > 10.0 {
        println!("  Verdict: Very strong evidence for Lambda-CDM over bounce.");
    } else if c.delta_bic > 6.0 {
        println!("  Verdict: Strong evidence for Lambda-CDM over bounce.");
    } else if c.delta_bic > 2.0 {
        println!("  Verdict: Positive evidence for Lambda-CDM over bounce.");
    } else if c.delta_bic > -2.0 {
        println!("  Verdict: No significant difference between models.");
    } else {
        println!("  Verdict: Evidence favors bounce model.");
    }

    println!();
    println!("  (Kass & Raftery 1995 BIC interpretation scale)");
    println!("================================================================");
}

fn print_json(c: &cosmology_core::ModelComparison, n_sne: usize, n_bao: usize) {
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
    println!("  \"bounce\": {{");
    println!("    \"omega_m\": {:.6},", c.bounce.omega_m);
    println!("    \"h0\": {:.4},", c.bounce.h0);
    println!("    \"q_corr\": {:.6e},", c.bounce.q_corr);
    println!("    \"chi2_total\": {:.4},", c.bounce.chi2_total);
    println!("    \"chi2_sn\": {:.4},", c.bounce.chi2_sn);
    println!("    \"chi2_bao\": {:.4},", c.bounce.chi2_bao);
    println!("    \"aic\": {:.4},", c.bounce.aic);
    println!("    \"bic\": {:.4},", c.bounce.bic);
    println!("    \"n_s\": {:.6}", c.n_s_bounce);
    println!("  }},");
    println!("  \"comparison\": {{");
    println!("    \"delta_aic\": {:.4},", c.delta_aic);
    println!("    \"delta_bic\": {:.4}", c.delta_bic);
    println!("  }}");
    println!("}}");
}
