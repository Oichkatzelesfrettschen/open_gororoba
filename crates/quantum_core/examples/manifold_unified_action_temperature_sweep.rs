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

// Simulating thermal noise effects on the algebraic vacuum attractor
// By injecting an effective "temperature" (T) that broadens the topological friction well
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌌 Initializing Thermal Unified Action Sweep...");
    
    let steps = 1000;
    let out_dir = Path::new("data/results");
    std::fs::create_dir_all(out_dir)?;
    
    let out_path = out_dir.join("thermal_action_sweep.csv");
    let mut file = File::create(&out_path)?;
    
    writeln!(file, "temperature,phi,l_grav,l_avt,action")?;

    let ricci_scalar = 0.0;     
    let matter_density = 0.0;   
    let scale_factor = 1.0;
    let base_avt_coupling = 5.0;     
    let dim = 512;

    let temperatures: Vec<f64> = vec![0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0];
    
    for &t in &temperatures {
        let mut max_action = f64::NEG_INFINITY;
        let mut optimal_phi = 0.0;
        
        let cc = AlgebraicCosmologicalConstant { scale_factor };
        
        // Effective coupling decreases with temperature (thermal disruption of topological rigidity)
        let effective_coupling = base_avt_coupling / (1.0 + t);
        
        let avt = TopologicalFrictionLagrangian {
            coupling_strength: effective_coupling,
            manifold_dimension: dim,
        };

        for i in 0..=steps {
            let phi = i as f64 / steps as f64;
            
            let l_grav = ricci_scalar * scale_factor + cc.lagrangian_density(phi);
            let l_avt = avt.lagrangian_density(phi);
            
            let action = compute_local_action(phi, ricci_scalar, matter_density, scale_factor, effective_coupling, dim);

            if action > max_action {
                max_action = action;
                optimal_phi = phi;
            }

            writeln!(file, "{},{},{},{},{}", t, phi, l_grav, l_avt, action)?;
        }
        
        println!("Temp: {:5.2} | Optimal Phi: {:.4} | Shift: {:.4}", 
                 t, optimal_phi, VACUUM_PHI - optimal_phi);
    }

    println!("✅ Sweep complete. Data written to {}", out_path.display());
    Ok(())
}
