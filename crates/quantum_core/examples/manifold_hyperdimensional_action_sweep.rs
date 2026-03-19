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
    println!("🌌 Initializing Hyperdimensional Unified Action Sweep...");
    
    let steps = 1000;
    let out_dir = Path::new("data/results");
    std::fs::create_dir_all(out_dir)?;
    
    let out_path = out_dir.join("hyperdimensional_action_sweep.csv");
    let mut file = File::create(&out_path)?;
    
    writeln!(file, "dim,phi,l_grav,l_avt,action")?;

    let ricci_scalar = 0.0;     
    let matter_density = 0.0;   
    let scale_factor = 1.0;
    let avt_coupling = 5.0;     

    let dimensions: Vec<usize> = vec![
        16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144,
        1_048_576, // 2^20
        16_777_216, // 2^24
        268_435_456, // 2^28
        1_073_741_824, // 2^30
        4_294_967_296, // 2^32
        68_719_476_736, // 2^36
        1_099_511_627_776, // 2^40
        281_474_976_710_656, // 2^48
        // Using large explicit usize values
        1_125_899_906_842_624, // 2^50
        1_152_921_504_606_846_976, // 2^60
    ];
    
    for &dim in &dimensions {
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

            writeln!(file, "{},{},{},{},{}", dim, phi, l_grav, l_avt, action)?;
        }
        
        println!("Dim: {:7} | Optimal Phi: {:.4} | Shift: {:.4}", dim, optimal_phi, VACUUM_PHI - optimal_phi);
    }

    println!("✅ Sweep complete. Data written to {}", out_path.display());
    Ok(())
}