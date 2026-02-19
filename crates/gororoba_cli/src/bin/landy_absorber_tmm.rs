//! CLI binary: Landy 2008 Perfect Metamaterial Absorber TMM Reproduction.
//!
//! Sweeps 9-14 GHz, computes absorption spectrum via TMM Lorentz effective-medium
//! model, and reports peak absorption, FWHM, and impedance mismatch.
//!
//! Usage:
//!   cargo run --release --bin landy-absorber-tmm

use materials_core::landy_absorber::{
    absorption_spectrum, effective_n, impedance, landy_2008_params, LandyParams,
};
use std::f64::consts::PI;

fn main() {
    println!("=== Landy 2008 Perfect Metamaterial Absorber ===");
    println!("    PRL 100, 207402 (2008)");
    println!("    TMM Lorentz effective-medium reproduction");
    println!();

    let params = landy_2008_params();
    let f0_ghz = params.omega_0e / (2.0 * PI) / 1e9;

    println!("Parameters:");
    println!("  omega_0  = {:.4e} rad/s ({:.3} GHz)", params.omega_0e, f0_ghz);
    println!("  S_e = S_m = {:.3}", params.strength_e);
    println!("  gamma/omega_0 = {:.2}%", params.gamma_e / params.omega_0e * 100.0);
    println!("  thickness = {:.2} mm (lambda/{})", params.thickness * 1e3,
             (2.998e8 / (f0_ghz * 1e9) / params.thickness) as u32);
    println!();

    // Sweep 9-14 GHz, 500 points
    let n_pts = 500;
    let freqs: Vec<f64> = (0..n_pts)
        .map(|i| (9.0 + 5.0 * i as f64 / (n_pts - 1) as f64) * 1e9)
        .collect();

    let spectrum = absorption_spectrum(&params, &freqs);

    // Find peak absorption
    let (peak_f, peak_r, peak_t, peak_a) = spectrum
        .iter()
        .max_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
        .copied()
        .unwrap();

    println!("--- Single-layer results ---");
    println!("  Peak absorption:  A = {:.4} ({:.1}%)", peak_a, peak_a * 100.0);
    println!("  Peak frequency:   f = {:.4} GHz", peak_f / 1e9);
    println!("  At peak: R = {:.4}, T = {:.6}", peak_r, peak_t);

    // Impedance at peak
    let omega_peak = 2.0 * PI * peak_f;
    let z = impedance(omega_peak, &params);
    println!("  Impedance:        |Z/Z_0 - 1| = {:.2e}", (z - num_complex::Complex64::new(1.0, 0.0)).norm());

    // Effective n at peak
    let n_eff = effective_n(omega_peak, &params);
    println!("  Effective index:  n_eff = {:.4} + {:.4}i", n_eff.re, n_eff.im);
    println!();

    // FWHM
    let half_max = peak_a / 2.0;
    let above_half: Vec<f64> = spectrum
        .iter()
        .filter(|(_, _, _, a)| *a >= half_max)
        .map(|(f, _, _, _)| *f)
        .collect();

    if above_half.len() >= 2 {
        let fwhm_hz = above_half.last().unwrap() - above_half.first().unwrap();
        let fwhm_pct = fwhm_hz / peak_f * 100.0;
        println!("  FWHM:             {:.2} GHz ({:.1}%)", fwhm_hz / 1e9, fwhm_pct);
    }
    println!();

    // Two-layer stack
    println!("--- Two-layer stack ---");
    let double_params = LandyParams {
        thickness: 2.0 * params.thickness,
        ..params
    };
    let spectrum_2 = absorption_spectrum(&double_params, &freqs);
    let (f2, r2, t2, a2) = spectrum_2
        .iter()
        .max_by(|a, b| a.3.partial_cmp(&b.3).unwrap())
        .copied()
        .unwrap();
    println!("  Peak absorption:  A = {:.6} ({:.2}%)", a2, a2 * 100.0);
    println!("  Peak frequency:   f = {:.4} GHz", f2 / 1e9);
    println!("  At peak: R = {:.6}, T = {:.8}", r2, t2);
    println!();

    // Verification summary
    println!("=== Verification ===");
    let pass_peak = peak_a >= 0.96;
    let pass_freq = (peak_f - 11.48e9).abs() < 0.5e9;
    let pass_fwhm = if above_half.len() >= 2 {
        let fwhm_rel = (above_half.last().unwrap() - above_half.first().unwrap()) / peak_f;
        fwhm_rel > 0.02 && fwhm_rel < 0.10
    } else {
        false
    };
    let pass_multilayer = a2 > 0.999;

    println!("  [{}] Peak A >= 0.96 (got {:.4})", if pass_peak { "PASS" } else { "FAIL" }, peak_a);
    println!("  [{}] Peak near 11.48 GHz (got {:.3} GHz)", if pass_freq { "PASS" } else { "FAIL" }, peak_f / 1e9);
    println!("  [{}] FWHM in [2%, 10%]", if pass_fwhm { "PASS" } else { "FAIL" });
    println!("  [{}] Two-layer A > 0.999 (got {:.6})", if pass_multilayer { "PASS" } else { "FAIL" }, a2);

    let all_pass = pass_peak && pass_freq && pass_fwhm && pass_multilayer;
    println!();
    if all_pass {
        println!("RESULT: Landy 2008 TMM reproduction VERIFIED.");
    } else {
        println!("WARNING: Some checks failed. Calibration may need adjustment.");
    }
}
