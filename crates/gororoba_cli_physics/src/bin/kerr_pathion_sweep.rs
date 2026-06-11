//! kerr-pathion-sweep: Sweep spin parameter `a=[0.9..0.9999]` comparing
//! control (no sink) vs eigenvalue-modulated Pathion sink stabilization.
//!
//! Produces CSV output: a_star, max_rho_ctrl, max_rho_sink, accum_energy, stability_index
//! where stability_index = log10(max_rho_ctrl / max_rho_sink).

use pathion_ellip::pathion_eigenvalues::PathionEigenvalueSpectrum;

fn main() {
    println!("=== Kerr Pathion Eigenvalue Sweep ===");
    println!();

    // Compute Pathion eigenvalue spectrum
    println!("Computing 32D Pathion ZD graph eigenvalues...");
    let spec = PathionEigenvalueSpectrum::compute();
    println!(
        "  {} eigenvalues, {} connected components",
        spec.eigenvalues.len(),
        spec.n_components
    );
    println!("  Eigenvalues (f32 for GPU): {:?}", spec.eigenvalues_16);
    println!();

    // Spin parameter sweep
    let spins = [0.9, 0.95, 0.99, 0.999, 0.9999];
    let mass = 1500.0_f64;

    // CSV header
    println!("a_star,max_rho_ctrl,max_rho_sink,accum_energy,stability_index");

    for &a_frac in &spins {
        let a = a_frac * mass;

        // CPU-side simulation (no GPU required for this sweep)
        // Simplified LBM-like density evolution with Pathion sink
        let n = 32_usize; // Small grid for CPU sweep
        let steps = 500_u32;
        let dt = 0.01_f32;
        let coupling = 1.0_f32;
        let damping = 50.0_f32;

        let max_rho_ctrl = simulate_cpu(n, mass as f32, a as f32, steps, dt, &[0.0; 16]);
        let max_rho_sink = simulate_cpu(n, mass as f32, a as f32, steps, dt, &spec.eigenvalues_16);

        let accum = max_rho_sink * coupling * damping * dt * steps as f32;
        let stability_index = if max_rho_sink > 0.0 && max_rho_ctrl > 0.0 {
            (max_rho_ctrl / max_rho_sink).log10()
        } else {
            0.0
        };

        println!(
            "{:.4},{:.6},{:.6},{:.4e},{:.4}",
            a_frac, max_rho_ctrl, max_rho_sink, accum, stability_index
        );
    }
}

/// CPU-side simplified simulation of density evolution under Kerr-like forcing
/// with eigenvalue-modulated Pathion sink.
fn simulate_cpu(
    n: usize,
    mass: f32,
    spin: f32,
    steps: u32,
    dt: f32,
    eigenvalues: &[f32; 16],
) -> f32 {
    let total = n * n * n;
    let center = n as f32 / 2.0;
    let r_plus = mass + (mass * mass - spin * spin).max(0.0).sqrt();

    // Initialize density field
    let mut rho = vec![1.0f32; total];
    let mut max_rho = 1.0f32;

    for _ in 0..steps {
        let mut new_rho = rho.clone();

        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let idx = x + n * (y + n * z);
                    let dx = x as f32 - center;
                    let dy = y as f32 - center;
                    let dz = z as f32 - center;
                    let r = (dx * dx + dy * dy + dz * dz).sqrt().max(0.5);

                    // Inward forcing (Kerr-like)
                    let forcing = (50.0 * mass) / (r * r);

                    // Profile
                    let profile = if r > r_plus + 2.0 {
                        (-(r - r_plus - 2.0) / 5.0).exp()
                    } else {
                        1.0
                    };

                    // Eigenvalue-modulated sink
                    let theta = dy.atan2(dx);
                    let mut sink_strength = 0.0f32;
                    for (k, &ev) in eigenvalues.iter().enumerate().take(16) {
                        let phase = (k as f32 * theta).cos();
                        sink_strength += ev * phase * phase;
                    }

                    let tau = rho[idx];
                    let damping = sink_strength * tau * tau * profile;

                    new_rho[idx] = rho[idx] + (forcing - damping) * dt;
                    new_rho[idx] = new_rho[idx].max(0.0); // Prevent negative density
                }
            }
        }

        rho = new_rho;
        max_rho = rho.iter().copied().fold(0.0f32, f32::max);

        // Early exit on explosion
        if max_rho.is_nan() || max_rho > 1e6 {
            return max_rho;
        }
    }

    max_rho
}
