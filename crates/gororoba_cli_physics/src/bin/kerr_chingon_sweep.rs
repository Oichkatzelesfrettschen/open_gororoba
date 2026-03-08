//! Kerr-Newman Chingon Frame Dragging Sweep
//!
//! Compares standard Kerr-Newman frame dragging with AVT-modulated
//! Chingon frame dragging across a range of radii, spin parameters,
//! and charge values.
//!
//! Output: CSV with (a, Q, r, theta, omega_KN, omega_chingon, relative_correction, tau)

use std::f64::consts::FRAC_PI_2;

use gr_core::{
    chingon_frame_dragging::{ChingonAvtSpectrum, sweep_frame_dragging},
    kerr_newman,
};

fn main() {
    let alpha_chingon: f64 = std::env::args()
        .position(|a| a == "--alpha")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1e-6);

    let n_radial: usize = std::env::args()
        .position(|a| a == "--n-radial")
        .and_then(|i| std::env::args().nth(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    eprintln!("=== Kerr-Newman Chingon Frame Dragging Sweep ===");
    eprintln!("alpha_chingon: {:.2e}", alpha_chingon);
    eprintln!("Radial points: {}", n_radial);

    // Compute 64D Chingon AVT spectrum
    eprintln!("Computing 64D Chingon AVT spectrum...");
    let spectrum = ChingonAvtSpectrum::compute();
    eprintln!(
        "  {} violations, spectral_norm = {:.4}, {} eigenvalues",
        spectrum.violation_count,
        spectrum.spectral_norm,
        spectrum.eigenvalues.len()
    );

    // Print top 5 eigenvalues
    let top_n = spectrum.eigenvalues.len().min(5);
    let top_evs: Vec<f64> = spectrum
        .eigenvalues
        .iter()
        .rev()
        .take(top_n)
        .copied()
        .collect();
    eprintln!("  Top {} eigenvalues: {:?}", top_n, top_evs);

    let m = 1.0; // Mass in geometrized units
    let theta = FRAC_PI_2; // Equatorial plane

    // Spin and charge parameter grid
    let spins = [0.0, 0.3, 0.5, 0.7, 0.9, 0.99];
    let charges = [0.0, 0.1, 0.3];

    // CSV header
    println!("a,Q,r,r_over_M,theta,omega_KN,omega_chingon,relative_correction,tau,r_plus");

    for &a in &spins {
        for &q in &charges {
            // Check sub-extremal condition
            if a * a + q * q > m * m {
                continue;
            }

            let r_plus = kerr_newman::outer_horizon(m, a, q);
            let results = sweep_frame_dragging(&spectrum, m, a, q, alpha_chingon, theta, n_radial);

            for comp in &results {
                println!(
                    "{},{},{:.6},{:.4},{:.4},{:.10e},{:.10e},{:.10e},{:.10e},{:.6}",
                    a,
                    q,
                    comp.r,
                    comp.r / m,
                    comp.theta,
                    comp.omega_kn,
                    comp.omega_chingon,
                    comp.relative_correction,
                    comp.tau,
                    r_plus
                );
            }
        }
    }

    // Summary statistics
    eprintln!("\n=== Summary ===");
    for &a in &[0.5, 0.9, 0.99] {
        let q = 0.0;
        let r_plus = kerr_newman::outer_horizon(m, a, q);
        let results = sweep_frame_dragging(&spectrum, m, a, q, alpha_chingon, theta, n_radial);

        if let Some(near_horizon) = results.first() {
            eprintln!(
                "  a={:.2}: r_+={:.4}, near-horizon correction = {:.6e}",
                a, r_plus, near_horizon.relative_correction
            );
        }
    }
}
