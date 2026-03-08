//! Dark Matter Halo NFW Fitting Pipeline
//!
//! Generates a synthetic DM halo density field with NFW-like profile,
//! extracts the radial profile, fits NFW parameters, and compares
//! with the Dutton-Maccio (2014) concentration-mass relation.
//!
//! This binary does NOT require GPU -- it uses a CPU-generated density
//! field to validate the NFW fitting pipeline.
//!
//! # Usage
//! ```text
//! cargo run --bin dark-halo-nfw-fit -- --grid 64 --r-s 10.0 --rho-s 50.0
//! ```

use cosmology_core::halo_profile::{
    dutton_maccio_concentration, extract_radial_profile, fit_nfw_profile, nfw_density,
};

fn main() {
    let grid: usize = std::env::args()
        .position(|a| a == "--grid")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    let r_s_true: f64 = std::env::args()
        .position(|a| a == "--r-s")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(10.0);

    let rho_s_true: f64 = std::env::args()
        .position(|a| a == "--rho-s")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(50.0);

    eprintln!("=== Dark Matter Halo NFW Fitting Pipeline ===");
    eprintln!(
        "Grid: {}^3, true r_s = {}, true rho_s = {}",
        grid, r_s_true, rho_s_true
    );

    // Generate synthetic NFW density field
    let n = grid;
    let center = [n as f64 / 2.0; 3];
    let mut rho_field = vec![0.0f32; n * n * n];

    for iz in 0..n {
        for iy in 0..n {
            for ix in 0..n {
                let dx = ix as f64 - center[0];
                let dy = iy as f64 - center[1];
                let dz = iz as f64 - center[2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();

                let rho = if r > 0.5 {
                    nfw_density(r, r_s_true, rho_s_true)
                } else {
                    // Avoid singularity at center
                    nfw_density(0.5, r_s_true, rho_s_true)
                };

                rho_field[ix + n * (iy + n * iz)] = rho as f32;
            }
        }
    }

    // Extract radial profile
    let r_max = n as f64 / 2.0 - 1.0;
    let n_bins = 30;
    let profile = extract_radial_profile(&rho_field, n, n, n, center, n_bins, r_max);

    eprintln!("\nRadial profile ({} bins):", n_bins);
    println!("r,rho_measured,rho_true,count");
    for bin in &profile {
        if bin.count > 0 {
            let rho_true = nfw_density(bin.r, r_s_true, rho_s_true);
            println!("{:.4},{:.6},{:.6},{}", bin.r, bin.rho, rho_true, bin.count);
        }
    }

    // Fit NFW profile
    let fit = fit_nfw_profile(&profile);

    eprintln!("\n=== NFW Fit Results ===");
    eprintln!(
        "  r_s: {:.4} (true: {:.4}, error: {:.2}%)",
        fit.r_s,
        r_s_true,
        100.0 * (fit.r_s - r_s_true).abs() / r_s_true
    );
    eprintln!(
        "  rho_s: {:.4} (true: {:.4}, error: {:.2}%)",
        fit.rho_s,
        rho_s_true,
        100.0 * (fit.rho_s - rho_s_true).abs() / rho_s_true
    );
    eprintln!("  chi2: {:.6}", fit.chi2);

    // Compare with Dutton-Maccio c-M relation
    eprintln!("\n=== Dutton-Maccio c-M Comparison ===");
    let masses = [1e10, 1e11, 1e12, 1e13, 1e14, 1e15];
    println!("\nM200_solar,c_DM14");
    for &m in &masses {
        let c = dutton_maccio_concentration(m);
        println!("{:.2e},{:.3}", m, c);
        eprintln!("  M200 = {:.2e} Msun -> c_DM14 = {:.3}", m, c);
    }

    // Verdict
    let r_s_error = (fit.r_s - r_s_true).abs() / r_s_true;
    let rho_s_error = (fit.rho_s - rho_s_true).abs() / rho_s_true;
    let pass = r_s_error < 0.15 && rho_s_error < 0.15;

    eprintln!("\n=== Verdict ===");
    eprintln!("NFW fit recovery: {}", if pass { "PASS" } else { "FAIL" });
    eprintln!("  r_s error: {:.2}%", r_s_error * 100.0);
    eprintln!("  rho_s error: {:.2}%", rho_s_error * 100.0);
}
