//! # Hyperdimensional Thermal Phase Diagram Sweep
//!
//! This executable explores the structural integrity of the algebraic vacuum state
//! under thermal disruption. It generates a 2D phase diagram by sweeping across
//! both algebraic dimension and effective temperature, revealing how topological
//! stiffness scales.
//!
//! ## Physics
//!
//! The simulation models the effect of thermal energy on the "Topological Friction"
//! term in the Unified Action. The effective coupling constant `K` of the friction
//! term is modeled as being inversely proportional to the temperature `T`:
//! `K_eff = K_base / (1 + T)`
//!
//! This simulates how thermal noise can disrupt the rigid algebraic structure,
//! "melting" the potential well that locks `phi` to the `3/8` attractor.
//!
//! ## Simulation
//!
//! The sweep calculates the optimal `phi` that maximizes the action for each
//! `(Dimension, Temperature)` pair. The results demonstrate that:
//! -   Low-dimensional manifolds (e.g., 16D) are fragile. Their topological
//!     structure "melts" at very low temperatures, causing `phi` to drift away
//!     from the `3/8` attractor.
//! -   High-dimensional manifolds (e.g., `2^30`D) are incredibly robust. They
//!     exhibit extreme "topological stiffness," remaining locked at `phi = 3/8`
//!     even at extremely high temperatures characteristic of the early universe.
//!
//! ## Output
//!
//! A CSV file `thermal_hyperdimensional_phase_diagram.csv` is generated, mapping
//! out the full phase space of the algebraic vacuum's response to thermal energy.

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
    println!("🌌 Initializing Hyperdimensional Thermal Phase Diagram Sweep...");
    println!("   Probing Topo-Thermal Unlocking across dimensions up to D = 10^9");
    
    let steps = 5000;
    let out_dir = Path::new("data/results");
    std::fs::create_dir_all(out_dir)?;
    
    let out_path = out_dir.join("thermal_hyperdimensional_phase_diagram.csv");
    let mut file = File::create(&out_path)?;
    
    writeln!(file, "dim,temperature,phi,l_grav,l_avt,action")?;

    let ricci_scalar = 0.0;     
    let matter_density = 0.0;   
    let scale_factor = 1.0;
    let base_avt_coupling = 5.0;     

    let dimensions: Vec<usize> = vec![
        16, 
        256, 
        4096, 
        65536, 
        1_048_576, 
        16_777_216,
        1_073_741_824 // 2^30 (~ 1 Billion)
    ];

    // Simulating energies from vacuum baseline up to early-universe GUT scales
    let temperatures: Vec<f64> = vec![0.001, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0];
    
    for &dim in &dimensions {
        println!("===================================================");
        println!("🔥 Sweeping Manifold Dimension: {}", dim);
        
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

                writeln!(file, "{},{},{},{},{},{}", dim, t, phi, l_grav, l_avt, action)?;
            }
            
            let shift = VACUUM_PHI - optimal_phi;
            if shift > 0.001 {
                println!("  Temp: {:9.3} | Optimal Phi: {:.4} | Shift: {:.4} (Unlocking Phase)", 
                         t, optimal_phi, shift);
            } else {
                println!("  Temp: {:9.3} | Optimal Phi: {:.4} | Shift: {:.4} [Rigid]", 
                         t, optimal_phi, shift);
            }
        }
    }

    println!("✅ Phase Diagram complete. Data written to {}", out_path.display());
    Ok(())
}
