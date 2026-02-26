//! sfwm-figure-repro: Reproduce all 4 figures from Son & Chekhova (2026).
//!
//! Generates 8 CSV files covering:
//!   - Fig 2(a-d): SFWM phenomenology (power scaling, polarization, coincidences, g(2))
//!   - Fig 3(a-b): SPDC comparison (linear scaling, g(2))
//!   - Fig 4(a-b): Phase-matching analysis (Maker fringes, rate dominance)
//!
//! # Reference
//! Son, C. & Chekhova, M. (2026), "Spontaneous four-wave mixing in a thin layer
//! with second-order nonlinearity", arXiv:2601.23137v1

use anyhow::Result;
use clap::Parser;
use std::fs;
use std::io::Write;

use materials_core::{SellmeierParams, fused_silica_sellmeier, linbo3_ordinary_sellmeier};
use optics_core::{
    SfwmMaterialParams, g2_sfwm_model, g2_spdc_model, maker_fringe_sweep, photon_number_sfwm,
    photon_number_spdc, polarization_dependence, rate_sweep_with_dk,
};

#[derive(Parser)]
#[command(name = "sfwm-figure-repro")]
#[command(about = "Reproduce Son & Chekhova (2026) SFWM figures")]
struct Args {
    /// Output directory for CSV files
    #[arg(long, default_value = "data/sfwm_repro")]
    output_dir: String,

    /// Number of sweep points
    #[arg(long, default_value = "500")]
    n_points: usize,

    /// Maximum pump power [W]
    #[arg(long, default_value = "0.5")]
    max_power_w: f64,

    /// Maximum crystal thickness [um]
    #[arg(long, default_value = "100.0")]
    max_thickness_um: f64,
}

/// Build SfwmMaterialParams from Sellmeier refractive indices.
fn build_params_from_sellmeier(
    sellmeier_o: &SellmeierParams,
    _sellmeier_sio2: &SellmeierParams,
) -> SfwmMaterialParams {
    SfwmMaterialParams {
        chi2_shg: 27.0e-12,
        chi2_spdc: 27.0e-12,
        chi3_sfwm: 2.4e-21,
        n_pump: sellmeier_o.refractive_index(1.030),
        n_signal: sellmeier_o.refractive_index(0.770),
        n_idler: sellmeier_o.refractive_index(1.550),
        n_sh: sellmeier_o.refractive_index(0.515),
        lambda_pump: 1.030,
        lambda_signal: 0.770,
        lambda_idler: 1.550,
        thickness: 10.0,
    }
}

fn write_csv(path: &str, header: &str, rows: &[String]) -> Result<()> {
    let mut f = fs::File::create(path)?;
    writeln!(f, "{}", header)?;
    for row in rows {
        writeln!(f, "{}", row)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    fs::create_dir_all(&args.output_dir)?;

    let sellmeier_o = linbo3_ordinary_sellmeier();
    let sellmeier_sio2 = fused_silica_sellmeier();
    let params = build_params_from_sellmeier(&sellmeier_o, &sellmeier_sio2);

    // Dimensionless scaling coefficients for photon number models.
    // These set the absolute scale of the curves; the power-law exponent
    // (2 for SFWM, 1 for SPDC) is the testable prediction.
    let alpha_sfwm = 1e3; // N_vis ~ alpha * P^2
    let alpha_ir = 1e3; // N_ir includes both P and P^2 terms
    let alpha_cc = 1e3; // coincidence counts
    let alpha_spdc = 1e4; // SPDC linear scaling

    // Power sweep points
    let powers: Vec<f64> = (1..=args.n_points)
        .map(|i| i as f64 * args.max_power_w / args.n_points as f64)
        .collect();

    // Thickness sweep points
    let thicknesses: Vec<f64> = (1..=args.n_points)
        .map(|i| i as f64 * args.max_thickness_um / args.n_points as f64)
        .collect();

    // ---- Fig 2(a): SFWM power scaling ----
    // N_vis ~ P^2 (direct SFWM), N_ir ~ P + P^2 (SFWM + stimulated Raman)
    let fig2a: Vec<String> = powers
        .iter()
        .map(|&p| {
            let n_vis = photon_number_sfwm(p, alpha_sfwm);
            let n_ir = photon_number_spdc(p, alpha_ir * 0.1) + photon_number_sfwm(p, alpha_ir);
            format!("{:.6e},{:.6e},{:.6e}", p, n_vis, n_ir)
        })
        .collect();
    write_csv(
        &format!("{}/fig2a.csv", args.output_dir),
        "power_w,n_vis,n_ir",
        &fig2a,
    )?;

    // ---- Fig 2(b): Polarization dependence ----
    let n_theta = 180;
    let fig2b: Vec<String> = (0..=n_theta)
        .map(|i| {
            let theta_deg = i as f64 * 360.0 / n_theta as f64;
            let theta_rad = theta_deg.to_radians();
            let rate = polarization_dependence(theta_rad);
            format!("{:.2},{:.6e},{:.6e}", theta_deg, rate, rate)
        })
        .collect();
    write_csv(
        &format!("{}/fig2b.csv", args.output_dir),
        "theta_deg,rate_vis,rate_ir",
        &fig2b,
    )?;

    // ---- Fig 2(c): SFWM coincidence counts ----
    // N_cc ~ P^2 (two-photon correlation from SFWM)
    let fig2c: Vec<String> = powers
        .iter()
        .map(|&p| {
            let n_cc = photon_number_sfwm(p, alpha_cc);
            format!("{:.6e},{:.6e}", p, n_cc)
        })
        .collect();
    write_csv(
        &format!("{}/fig2c.csv", args.output_dir),
        "power_w,n_cc",
        &fig2c,
    )?;

    // ---- Fig 2(d): SFWM g(2) ----
    // g(2) = 1 + a/P^2, with a chosen so g(2)(P_min) >> 2
    let a_g2_sfwm = 1e-4;
    let fig2d: Vec<String> = powers
        .iter()
        .map(|&p| {
            let g2 = g2_sfwm_model(p, a_g2_sfwm);
            format!("{:.6e},{:.6e}", p, g2)
        })
        .collect();
    write_csv(
        &format!("{}/fig2d.csv", args.output_dir),
        "power_w,g2_sfwm",
        &fig2d,
    )?;

    // ---- Fig 3(a): SPDC coincidence counts ----
    // N_cc ~ P (linear pump dependence)
    let fig3a: Vec<String> = powers
        .iter()
        .map(|&p| {
            let n_cc = photon_number_spdc(p, alpha_spdc);
            format!("{:.6e},{:.6e}", p, n_cc)
        })
        .collect();
    write_csv(
        &format!("{}/fig3a.csv", args.output_dir),
        "power_w,n_cc_spdc",
        &fig3a,
    )?;

    // ---- Fig 3(b): SPDC g(2) ----
    // g(2) = 1 + a/P
    let a_g2_spdc = 1e-2;
    let fig3b: Vec<String> = powers
        .iter()
        .map(|&p| {
            let g2 = g2_spdc_model(p, a_g2_spdc);
            format!("{:.6e},{:.6e}", p, g2)
        })
        .collect();
    write_csv(
        &format!("{}/fig3b.csv", args.output_dir),
        "power_w,g2_spdc",
        &fig3b,
    )?;

    // ---- Fig 4(a): Maker fringes |F|^2 ----
    // Use paper-calibrated dk for the key theoretical figure
    let paper_params = SfwmMaterialParams::linbo3_paper_calibrated();
    let fringes = maker_fringe_sweep(&paper_params, &thicknesses);
    let fig4a: Vec<String> = fringes
        .iter()
        .map(|&(l, fs, fh, fp)| format!("{:.4},{:.6e},{:.6e},{:.6e}", l, fs, fh, fp))
        .collect();
    write_csv(
        &format!("{}/fig4a.csv", args.output_dir),
        "thickness_um,f_sfwm_sq,f_shg_sq,f_spdc_sq",
        &fig4a,
    )?;

    // ---- Fig 4(b): Rate sweep ----
    let wm = paper_params.paper_wavevector_mismatches();
    let rates = rate_sweep_with_dk(&paper_params, &wm, &thicknesses);
    let fig4b: Vec<String> = rates
        .iter()
        .map(|&(l, rd, rc)| format!("{:.4},{:.6e},{:.6e}", l, rd, rc))
        .collect();
    write_csv(
        &format!("{}/fig4b.csv", args.output_dir),
        "thickness_um,r_direct,r_cascaded",
        &fig4b,
    )?;

    // Summary
    let wm_sellmeier = params.wavevector_mismatches();
    let wm_paper = paper_params.paper_wavevector_mismatches();
    println!("Son & Chekhova (2026) SFWM figure reproduction complete.");
    println!(
        "Output: {}/fig{{2a,2b,2c,2d,3a,3b,4a,4b}}.csv",
        args.output_dir
    );
    println!();
    println!("Sellmeier-derived refractive indices:");
    println!("  n_pump(1030 nm) = {:.4}", params.n_pump);
    println!("  n_signal(770 nm) = {:.4}", params.n_signal);
    println!("  n_idler(1550 nm) = {:.4}", params.n_idler);
    println!("  n_sh(515 nm)    = {:.4}", params.n_sh);
    println!();
    println!("Wavevector mismatches (Sellmeier):");
    println!(
        "  dk_SFWM = {:.4} 1/um  (L_coh = {:.1} um)",
        wm_sellmeier.dk_sfwm,
        std::f64::consts::PI / wm_sellmeier.dk_sfwm.abs()
    );
    println!(
        "  dk_SHG  = {:.4} 1/um  (L_coh = {:.1} um)",
        wm_sellmeier.dk_shg,
        std::f64::consts::PI / wm_sellmeier.dk_shg.abs()
    );
    println!(
        "  dk_SPDC = {:.4} 1/um  (L_coh = {:.1} um)",
        wm_sellmeier.dk_spdc,
        std::f64::consts::PI / wm_sellmeier.dk_spdc.abs()
    );
    println!();
    println!("Wavevector mismatches (paper calibrated):");
    println!(
        "  dk_SFWM = {:.4} 1/um  (L_coh = 33.3 um)",
        wm_paper.dk_sfwm
    );
    println!("  dk_SHG  = {:.4} 1/um  (L_coh = 3.1 um)", wm_paper.dk_shg);
    println!("  dk_SPDC = {:.4} 1/um  (L_coh = 3.4 um)", wm_paper.dk_spdc);

    Ok(())
}
