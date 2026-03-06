//! pathion-shadow-cli: Compute 32D Pathion-modulated Kerr shadow boundaries
//! and deflection angles via exact Carlson RF+RJ integrals and quartic root-finding.

use pathion_ellip::diagonalizer::PathionDiagonalizer;
use pathion_ellip::pathion_shadow::pathion_deflection;
use pathion_ellip::shadow_boundary::pathion_shadow_boundary;

fn main() -> anyhow::Result<()> {
    println!("=== Pathion Shadow: 32D Carlson RF+RJ Deflection + Shadow Boundary ===");
    println!();

    let mass = 1.0;
    let mut spin = [0.0; 32];
    spin[0] = 0.99; // a/M = 0.99
    // Add small Pathion fluctuations in higher dimensions
    for (i, s) in spin.iter_mut().enumerate().skip(1) {
        *s = 0.01 * (i as f64).sin();
    }

    // --- Part 1: Point deflection ---
    let mut xi = [0.0; 32];
    xi[0] = 6.0;
    let mut eta = [0.0; 32];
    eta[0] = 20.0;

    println!("Point deflection: mass={mass}, a={}, xi={}, eta={}", spin[0], xi[0], eta[0]);
    let result = pathion_deflection(mass, &spin, &xi, &eta)?;

    let diag = PathionDiagonalizer::new();
    let planes = diag.project(&result);

    println!();
    println!("{:<8},{:<20},{:<20}", "plane", "real_deflection", "imag_rotation");
    for (i, p) in planes.iter().enumerate() {
        println!("{:<8},{:<20.8},{:<20.8}", i, p.re, p.im);
    }

    let avg_defl = result.iter().sum::<f64>() / 32.0;
    let variance = result.iter().map(|&x| (x - avg_defl).powi(2)).sum::<f64>() / 32.0;

    println!();
    println!("Mean deflection: {avg_defl:.8}");
    println!("Pathion variance: {variance:.8e}");
    if variance > 1e-10 {
        println!("Non-zero Pathion variance: shadow splinters into 15+1 sub-rings.");
    }

    // --- Part 2: Shadow boundary ---
    println!();
    println!("=== Shadow Boundary (equatorial observer) ===");
    println!();

    let boundary = pathion_shadow_boundary(mass, spin[0], 100);
    println!("alpha,beta,xi,eta");
    for pt in &boundary {
        println!("{:.6},{:.6},{:.6},{:.6}", pt.alpha, pt.beta, pt.xi_0, pt.eta_0);
    }

    println!();
    println!("Shadow boundary: {} points", boundary.len());

    // Compute shadow centroid and size
    if !boundary.is_empty() {
        let n = boundary.len() as f64;
        let cx = boundary.iter().map(|p| p.alpha).sum::<f64>() / n;
        let cy = boundary.iter().map(|p| p.beta).sum::<f64>() / n;
        let max_r = boundary
            .iter()
            .map(|p| ((p.alpha - cx).powi(2) + (p.beta - cy).powi(2)).sqrt())
            .fold(0.0_f64, f64::max);
        println!("Centroid: ({cx:.4}, {cy:.4})");
        println!("Max radius from centroid: {max_r:.4} M");
    }

    Ok(())
}
