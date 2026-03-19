use std::fs::File;
use std::io::Write;
use std::path::Path;

use verified_core::unified_action::{
    compute_local_action,
    AlgebraicCosmologicalConstant,
    TopologicalFrictionLagrangian,
    ActionComponent,
};
use verified_core::axiomatic_gates::VACUUM_PHI;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌌 Initializing Recursive Dimensional Action Sweep...");
    
    let steps = 10000;
    let out_dir = Path::new("data/results");
    std::fs::create_dir_all(out_dir)?;
    
    let out_path = out_dir.join("recursive_dimensional_sweep.csv");
    let mut file = File::create(&out_path)?;
    
    writeln!(file, "dim,phi,l_grav,l_avt,action")?;

    let ricci_scalar = 0.0;     
    let matter_density = 0.0;   
    let scale_factor = 1.0;
    let avt_coupling = 5.0;     

    // Sweep from dimension 2 to 2^24 (16.7M)
    let max_power = 24;
    
    for p in 1..=max_power {
        let dim = 1 << p;
        let mut max_action = f64::NEG_INFINITY;
        let mut optimal_phi = 0.0;
        
        let cc = AlgebraicCosmologicalConstant { scale_factor };
        let avt = TopologicalFrictionLagrangian {
            coupling_strength: avt_coupling,
            manifold_dimension: dim,
        };

        for i in 0..=steps {
            let phi = i as f64 / steps as f64;
            
            let l_grav = ricci_scalar * scale_factor + cc.lagrangian_density(phi);
            let l_avt = avt.lagrangian_density(phi);
            
            let action = compute_local_action(phi, ricci_scalar, matter_density, scale_factor, avt_coupling, dim);

            if action > max_action {
                max_action = action;
                optimal_phi = phi;
            }
            
            if p <= 8 || (p > 8 && i % 100 == 0) {
                writeln!(file, "{},{},{},{},{}", dim, phi, l_grav, l_avt, action)?;
            }
        }
        
        println!("Dim: 2^{:<2} ({:9}) | Optimal Phi: {:.5} | Shift: {:.5} | Asymptotic Ratio: {:.5}", 
                 p, dim, optimal_phi, VACUUM_PHI - optimal_phi, optimal_phi / VACUUM_PHI);
    }

    println!("✅ Sweep complete. Data written to {}", out_path.display());
    Ok(())
}
