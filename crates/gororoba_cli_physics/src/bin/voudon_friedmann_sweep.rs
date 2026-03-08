//! voudon-friedmann-sweep: Simulate cosmological evolution with Voudon pressure.
//!
//! Evaluates the 'Smoothing Effect' of 256D algebraic pressure on the
//! expansion of the universe and calculates the Voudon Smoothing Scale.

use cosmology_core::{FlatLCDM, VoudonFriedmann};
use algebra_core::construction::deep_space::compute_voudon_imbalance_density;

fn main() -> anyhow::Result<()> {
    println!("=== Phase 2: Voudon-Friedmann Cosmological Integration ===");

    // 1. Get Algebraic input from 256D Voudon algebra
    let phi = compute_voudon_imbalance_density();
    println!("  Input Voudon Imbalance Density: {:.8}", phi);

    // 2. Initialize Modified Cosmology
    let base = FlatLCDM::planck2018();
    // Experimental coupling for macroscopic smoothing
    let alpha = 1e-4; 
    let model = VoudonFriedmann::new(base, phi, alpha);

    println!("\n[ Cosmological Profile ]");
    println!("  H0:            {:.2} km/s/Mpc", model.base.h0);
    println!("  Voudon Alpha:  {:.2e}", model.alpha_voudon);
    println!("  Smoothing Scale: {:.2} Mpc", model.smoothing_scale_mpc());

    // 3. Expansion Sweep
    println!("\n[ Redshift Sweep: E(z) comparison ]");
    println!("  {:<10} | {:<15} | {:<15} | {:<10}", "z", "E_LCDM(z)", "E_Voudon(z)", "Delta (%)");
    println!("{:-<10}-|-{:-<15}-|-{:-<15}-|-{:-<10}", "", "", "", "");

    for z_idx in 0..11 {
        let z = z_idx as f64 * 0.5;
        let e_lcdm = model.base.e_z(z);
        let e_voudon = model.e_z(z);
        let delta = (e_voudon - e_lcdm) / e_lcdm * 100.0;
        println!("  {:<10.1} | {:<15.6} | {:<15.6} | {:<10.4}%", z, e_lcdm, e_voudon, delta);
    }

    println!("\n=== Final Analysis ===");
    if model.smoothing_scale_mpc() > 10.0 {
        println!("  VERDICT: PASS (Voudon Pressure induces macroscopic smoothing at galaxy-cluster scales)");
    } else {
        println!("  VERDICT: WEAK (Algebraic Pressure insufficient for cosmological smoothing)");
    }

    Ok(())
}
