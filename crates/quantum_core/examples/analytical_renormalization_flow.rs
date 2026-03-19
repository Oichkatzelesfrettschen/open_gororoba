use std::fs::File;
use std::io::Write;
use std::path::Path;

use verified_core::axiomatic_gates::VACUUM_PHI;

// Calculates the Exact Discrete Topological Associator Flux Volume
fn flux_volume(dim: usize) -> f64 {
    if dim >= 16 {
        let d = dim as f64;
        (d / 8.0) * std::f64::consts::SQRT_2 + (d / 2.0)
    } else {
        0.0
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌌 Initializing Analytical Renormalization Flow (Pure Rust)...");
    println!("   Solving Exact S'(\\phi) = 0 via Newton-Raphson Root Finding.");
    
    let out_dir = Path::new("data/results");
    std::fs::create_dir_all(out_dir)?;
    let out_path = out_dir.join("analytical_renormalization_flow.csv");
    let mut file = File::create(&out_path)?;
    writeln!(file, "dim,temperature,k_factor,optimal_phi,shift")?;

    let dimensions: Vec<usize> = vec![
        16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536,
        1_048_576, 16_777_216, 268_435_456, 1_073_741_824, 4_294_967_296,
        68_719_476_736, 1_099_511_627_776
    ];

    let temperatures: Vec<f64> = vec![
        0.0, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 
        10000.0, 100000.0, 1e6, 1e9, 1e12
    ];
    let base_avt_coupling = 5.0;

    for &dim in &dimensions {
        println!("===================================================");
        println!("🔥 Analytical Sweep for Manifold Dimension: {}", dim);
        
        let v_d = flux_volume(dim);
        
        for &t in &temperatures {
            let effective_coupling = base_avt_coupling / (1.0 + t);
            let k = effective_coupling * v_d;
            
            // We want to find the root of:
            // f(phi) = K(0.375 - phi) - ln((1-phi)/phi) = 0
            // This occurs when Topological Friction balances Cosmological Entropy
            let mut phi_opt = 0.0;
            
            // f'(phi) = -K + 1/(phi(1-phi))
            // f'(phi) = 0 when phi(1-phi) = 1/K => phi = (1 - sqrt(1 - 4/K)) / 2
            if k >= 4.0 {
                let phi_max = (1.0 - (1.0 - 4.0 / k).sqrt()) / 2.0;
                let f_max = k * (VACUUM_PHI - phi_max) - ((1.0 - phi_max) / phi_max).ln();
                
                // If f_max > 0, then roots exist. The physical root is the larger one in (phi_max, 0.375)
                if f_max > 0.0 {
                    let mut phi = 0.374; // Start near the attractor to find the right root
                    for _ in 0..100 {
                        let f = k * (VACUUM_PHI - phi) - ((1.0 - phi) / phi).ln();
                        let f_prime = -k + 1.0 / (phi * (1.0 - phi));
                        let delta = f / f_prime;
                        phi -= delta;
                        
                        // Prevent jumping out of bounds
                        if phi <= phi_max { phi = phi_max + 1e-6; }
                        if phi >= VACUUM_PHI { phi = VACUUM_PHI - 1e-6; }
                        
                        if delta.abs() < 1e-14 {
                            break;
                        }
                    }
                    // Final cleanup
                    if phi > VACUUM_PHI { phi = VACUUM_PHI; }
                    phi_opt = phi;
                }
            }

            let shift = VACUUM_PHI - phi_opt;
            
            if (shift - 0.375).abs() < 1e-6 {
                println!("  Temp: {:9.3e} | K: {:10.2} | Phi: {:.6} | Shift: {:.6} (Total Melt)", 
                         t, k, phi_opt, shift);
            } else if shift > 0.0001 {
                println!("  Temp: {:9.3e} | K: {:10.2} | Phi: {:.6} | Shift: {:.6} (Unlocking Flow)", 
                         t, k, phi_opt, shift);
            } else {
                println!("  Temp: {:9.3e} | K: {:10.2} | Phi: {:.6} | Shift: {:.6} [Rigid / Bound]", 
                         t, k, phi_opt, shift);
            }

            writeln!(file, "{},{},{:.6},{:.12},{:.12}", dim, t, k, phi_opt, shift)?;
        }
    }

    println!("✅ Exact analytical renormalization data generated at {}", out_path.display());
    Ok(())
}
