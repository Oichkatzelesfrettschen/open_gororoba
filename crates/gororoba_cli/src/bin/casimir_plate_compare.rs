//! Casimir force comparison between dielectric plate pairs (Si, Ge, SiO2).
//!
//! Sweeps plate separation from 100 nm to 10 um and computes the Casimir
//! energy (J/m^2), force (N/m^2), and eta = E/E_ideal for three symmetric
//! material pairs: Si/Si, Ge/Ge, SiO2/SiO2.
//!
//! Uses the correct Lifshitz formula with Gauss-Legendre quadrature (Sprint 45).
//! Two previously identified bugs in casimir_energy_density are fixed:
//!   1. s_i = sqrt(p^2 + eps_i - 1), not sqrt(eps_i*p^2 + eps_i - 1)
//!   2. Integration measure is p dp (not p^2 dp)
//!
//! Example usage:
//! ```bash
//! cargo run --release --bin casimir-plate-compare
//! cargo run --release --bin casimir-plate-compare -- --n-matsubara 1000 --n-gauss 64
//! ```

use clap::Parser;
use materials_core::{
    casimir_energy_ideal, casimir_lifshitz_energy, casimir_lifshitz_force,
    germanium_optical, silica_casimir_optical, silicon_optical,
};
use std::f64::consts::PI;
use std::io::Write;

#[derive(Parser)]
#[command(name = "casimir-plate-compare")]
#[command(about = "Casimir force comparison: Si, Ge, SiO2 symmetric plate pairs")]
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

    /// Number of separation points (log-spaced from 100 nm to 10 um)
    #[arg(long, default_value_t = 16)]
    n_points: usize,
}

fn main() {
    let args = Args::parse();
    let si = silicon_optical();
    let ge = germanium_optical();
    let sio2 = silica_casimir_optical();
    let materials: &[(&str, &materials_core::DrudeLorentzParams)] =
        &[("Si", &si), ("Ge", &ge), ("SiO2", &sio2)];

    // Log-spaced separations: 100 nm to 10 um
    let d_min = 100e-9_f64;
    let d_max = 10e-6_f64;
    let n = args.n_points;
    let separations: Vec<f64> = (0..n)
        .map(|i| d_min * (d_max / d_min).powf(i as f64 / (n - 1).max(1) as f64))
        .collect();

    let stdout = std::io::stdout();
    let mut out = stdout.lock();

    writeln!(out, "# Casimir plate comparison (correct Lifshitz formula, Sprint 45)").unwrap();
    writeln!(out, "# T={:.1} K, N_Matsubara={}, N_GL={}", args.temperature_k, args.n_matsubara, args.n_gauss).unwrap();
    writeln!(out, "# {:>12} {:>8} {:>14} {:>14} {:>8}", "d_nm", "pair", "E_J_m2", "F_N_m2", "eta").unwrap();

    for (name, mat) in materials {
        for &d in &separations {
            let e = casimir_lifshitz_energy(
                mat, mat, d, args.temperature_k, args.n_matsubara, args.n_gauss,
            );
            let f = casimir_lifshitz_force(
                mat, mat, d, args.temperature_k, args.n_matsubara, args.n_gauss,
            );
            let e_ideal = casimir_energy_ideal(d);
            let eta = e / e_ideal;
            writeln!(
                out,
                "  {:>12.1} {:>8} {:>14.4e} {:>14.4e} {:>8.4}",
                d * 1e9,
                name,
                e,
                f,
                eta
            )
            .unwrap();
        }
        writeln!(out).unwrap();
    }

    // Summary: ordering check at d=100nm
    let d_test = 100e-9_f64;
    writeln!(out, "# Ordering at d={:.0} nm (larger |F| = stronger Casimir force):", d_test * 1e9).unwrap();
    let mut ranked: Vec<(String, f64)> = materials
        .iter()
        .map(|(name, mat)| {
            let f = casimir_lifshitz_force(mat, mat, d_test, args.temperature_k, args.n_matsubara, args.n_gauss);
            (name.to_string(), f.abs())
        })
        .collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (rank, (name, f_abs)) in ranked.iter().enumerate() {
        writeln!(out, "#  {}. {} |F| = {:.4e} N/m^2", rank + 1, name, f_abs).unwrap();
    }

    // Compute eta range
    writeln!(out).unwrap();
    writeln!(out, "# eta at d=100nm and d=1um:").unwrap();
    for (name, mat) in materials {
        let eta_100 = {
            let e = casimir_lifshitz_energy(mat, mat, 100e-9, args.temperature_k, args.n_matsubara, args.n_gauss);
            let e_id = casimir_energy_ideal(100e-9);
            e / e_id
        };
        let eta_1000 = {
            let e = casimir_lifshitz_energy(mat, mat, 1000e-9, args.temperature_k, args.n_matsubara, args.n_gauss);
            let e_id = casimir_energy_ideal(1000e-9);
            e / e_id
        };
        writeln!(out, "#  {} eta(100nm)={:.4} eta(1um)={:.4}", name, eta_100, eta_1000).unwrap();
    }

    eprintln!("Lifshitz formula using dimensionless variable p = kappa*c/xi_n.");
    eprintln!("s_i = sqrt(p^2 + eps_i - 1)  [corrected from old sqrt(eps_i*p^2+eps_i-1)]");
    eprintln!("Integration measure: p dp  [corrected from old p^2 dp]");
    eprintln!("n=0 quasi-static: Drude convention, r_TE=0, r_TM=(eps_s-1)/(eps_s+1).");
    eprintln!("Physics note: 2*PI*kT/hbar = {:.3e} rad/s at {:.0} K (thermal coherence).",
        2.0 * PI * 8.617_333_262e-5 * args.temperature_k * 1.519_267_447e15,
        args.temperature_k);
}
