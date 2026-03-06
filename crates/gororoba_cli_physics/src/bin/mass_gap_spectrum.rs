//! mass-gap-spectrum: Map 32D Pathion eigenvalue spectrum to physical masses;
//! compare with GW190521 QNM ringdown.
//!
//! Uses the left-multiplication eigenvalue spectrum from cd_kernel and maps
//! eigenvalues to solar masses via the Barbero-Immirzi parameter from LQG.

use pathion_ellip::pathion_eigenvalues::PathionEigenvalueSpectrum;
use std::f64::consts::PI;

/// Map a Laplacian eigenvalue to a mass via LQG area spectrum.
///
/// M = M_planck * sqrt(lambda / (8 * pi * gamma))
fn eigenvalue_to_mass_solar(lambda: f64, gamma: f64) -> f64 {
    let m_planck = 2.176e-8; // kg
    let m_solar = 1.989e30; // kg
    let m = m_planck * (lambda / (8.0 * PI * gamma)).sqrt();
    m / m_solar
}

/// Kerr QNM frequency for (l=2, m=2) fundamental mode.
///
/// Fit from Berti, Cardoso & Starinets (2009), Class. Quantum Grav. 26, 163001.
/// omega = (f1 + f2 * (1 - a_f)^f3) / M_final
fn qnm_frequency_kerr(m_final_solar: f64, a_final: f64) -> f64 {
    let f1 = 1.5251;
    let f2 = -1.1568;
    let f3 = 0.1292;
    // Convert M_solar to geometric units (G M / c^3 in seconds)
    let gm_c3 = m_final_solar * 1.989e30 * 6.674e-11 / (2.998e8_f64).powi(3);
    let omega_dimless = f1 + f2 * (1.0 - a_final).powf(f3);
    omega_dimless / gm_c3
}

fn main() {
    println!("=== Mass-Gap Spectrum: Pathion Eigenvalue -> Physical Mass ===");
    println!();

    // Compute eigenvalue spectrum
    println!("Computing 32D Pathion ZD graph eigenvalues...");
    let spec = PathionEigenvalueSpectrum::compute();
    println!(
        "  {} eigenvalues, {} components",
        spec.eigenvalues.len(),
        spec.n_components
    );
    println!();

    // GW190521 parameters
    let m_final = 142.0_f64; // solar masses
    let a_final = 0.72_f64; // dimensionless spin
    let m1 = 85.0_f64;
    let m2 = 66.0_f64;

    let omega_qnm = qnm_frequency_kerr(m_final, a_final);
    println!("GW190521 Reference:");
    println!("  M1 = {} Msun, M2 = {} Msun", m1, m2);
    println!("  M_final = {} Msun, a_final = {}", m_final, a_final);
    println!("  Radiated energy: {:.0} Msun", m1 + m2 - m_final);
    println!("  QNM (l=2,m=2) omega = {:.4e} rad/s", omega_qnm);
    println!();

    // Sweep Immirzi gamma
    let gammas = [0.1, 0.2, 0.274, 0.3, 0.4];

    println!("eigenvalue,mass_solar,gamma,qnm_freq,residual_vs_gw190521");
    for &gamma in &gammas {
        for &ev in &spec.eigenvalues {
            if ev < 1e-10 {
                continue;
            }
            let mass = eigenvalue_to_mass_solar(ev, gamma);
            let omega = qnm_frequency_kerr(mass, 0.7);
            let residual = (omega - omega_qnm) / omega_qnm;
            println!("{:.6},{:.6e},{:.3},{:.4e},{:.6}", ev, mass, gamma, omega, residual);
        }
    }
}
