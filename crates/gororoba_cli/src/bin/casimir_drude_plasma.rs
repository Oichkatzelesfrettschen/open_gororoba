//! Casimir force: Drude vs plasma model discrepancy for metallic plates.
//!
//! Demonstrates the Drude-plasma controversy in Casimir physics:
//! the Drude model predicts r_TE(xi=0) = 0 while the plasma model gives
//! a finite TE contribution at the n=0 Matsubara term.  The difference is
//! ~1-2% of the total force at room temperature -- experimentally relevant
//! because modern Casimir experiments reach sub-1% precision.
//!
//! Sweeps separation from 100 nm to 5 um for Au/Au, Ag/Ag, Cu/Cu plates.
//!
//! References:
//!   Klimchitskaya, Mohideen, Mostepanenko, Rev. Mod. Phys. 81, 1827 (2009)
//!   Lambrecht, Maia Neto, Reynaud, New J. Phys. 8, 243 (2006)
//!
//! Example usage:
//! ```bash
//! cargo run --release --bin casimir-drude-plasma
//! ```

use clap::Parser;
use materials_core::{
    casimir_drude_plasma_discrepancy,
    gold_drude_lorentz, silver_drude_lorentz, copper_drude_lorentz,
    DrudeLorentzParams,
};
use std::io::Write;

#[derive(Parser)]
#[command(name = "casimir-drude-plasma")]
#[command(about = "Casimir Drude vs plasma model discrepancy for Au, Ag, Cu plates")]
struct Args {
    /// Number of Matsubara frequency terms
    #[arg(long, default_value_t = 500)]
    n_matsubara: usize,

    /// Number of Gauss-Legendre quadrature points
    #[arg(long, default_value_t = 48)]
    n_gauss: usize,

    /// Temperature in Kelvin
    #[arg(long, default_value_t = 300.0)]
    temperature_k: f64,

    /// Number of separation points (log-spaced 100 nm to 5 um)
    #[arg(long, default_value_t = 12)]
    n_points: usize,
}

/// Material descriptor: name, model, plasma frequency in eV.
struct MetalSpec {
    name: &'static str,
    omega_p_ev: f64,
    model: DrudeLorentzParams,
}

fn main() {
    let args = Args::parse();

    let metals: Vec<MetalSpec> = vec![
        MetalSpec { name: "Au", omega_p_ev: 9.0,  model: gold_drude_lorentz() },
        MetalSpec { name: "Ag", omega_p_ev: 9.01, model: silver_drude_lorentz() },
        MetalSpec { name: "Cu", omega_p_ev: 8.71, model: copper_drude_lorentz() },
    ];

    // Log-spaced separations: 100 nm to 5 um
    let d_min = 100e-9_f64;
    let d_max = 5e-6_f64;
    let n = args.n_points;
    let separations: Vec<f64> = (0..n)
        .map(|i| d_min * (d_max / d_min).powf(i as f64 / (n - 1).max(1) as f64))
        .collect();

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    writeln!(out, "# Casimir Drude vs plasma model (Sprint 45)").unwrap();
    writeln!(out, "# T={:.1} K, N_Matsubara={}, N_GL={}", args.temperature_k, args.n_matsubara, args.n_gauss).unwrap();
    writeln!(out, "# {:>8} {:>12} {:>14} {:>14} {:>12}",
        "metal", "d_nm", "E_Drude", "E_Plasma", "disc_%").unwrap();

    for spec in &metals {
        for &d in &separations {
            let (e_drude, e_plasma, disc_pct) = casimir_drude_plasma_discrepancy(
                &spec.model, spec.omega_p_ev, d,
                args.temperature_k, args.n_matsubara, args.n_gauss,
            );
            writeln!(
                out,
                "  {:>8} {:>12.1} {:>14.4e} {:>14.4e} {:>12.4}",
                spec.name, d * 1e9, e_drude, e_plasma, disc_pct
            ).unwrap();
        }
        writeln!(out).unwrap();
    }

    // Summary: discrepancy at key separations
    let key_seps = [100e-9, 200e-9, 500e-9, 1000e-9, 2000e-9];
    writeln!(out, "# Summary: discrepancy % at selected separations").unwrap();
    writeln!(out, "# {:>8} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "metal", "d=100nm", "d=200nm", "d=500nm", "d=1um", "d=2um").unwrap();
    for spec in &metals {
        let discs: Vec<String> = key_seps.iter().map(|&d| {
            let (_, _, pct) = casimir_drude_plasma_discrepancy(
                &spec.model, spec.omega_p_ev, d,
                args.temperature_k, args.n_matsubara, args.n_gauss,
            );
            format!("{:.3}%", pct)
        }).collect();
        writeln!(out, "  {:>8} {:>10} {:>10} {:>10} {:>10} {:>10}",
            spec.name, discs[0], discs[1], discs[2], discs[3], discs[4]).unwrap();
    }

    eprintln!("Drude-plasma controversy explanation:");
    eprintln!("  Drude: r_TE(xi=0) = 0  (TE mode cannot couple at zero frequency)");
    eprintln!("  Plasma: r_TE_plasma(k_perp) = (k_perp - sqrt(k_perp^2 + wp^2/c^2)) / (...)");
    eprintln!("  The difference is in the n=0 Matsubara term only.");
    eprintln!("  Plasma is more attractive (larger |E|) -> larger discrepancy at small d");
    eprintln!("  because the n=0 term dominates at large separations (thermal regime).");
    eprintln!("Refs: Klimchitskaya et al. Rev. Mod. Phys. 81, 1827 (2009)");
}
