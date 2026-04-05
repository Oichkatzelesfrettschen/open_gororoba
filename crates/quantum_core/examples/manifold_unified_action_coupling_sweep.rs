use std::{fs::File, io::Write, path::Path};

use verified_core::{
    axiomatic_gates::VACUUM_PHI,
    unified_action::{
        ActionComponent, AlgebraicCosmologicalConstant, TopologicalFrictionLagrangian,
        compute_local_action,
    },
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("<EMOJI+1F30C> Initializing Topological Coupling Action Sweep...");

    let steps = 1000;
    let out_dir = Path::new("data/results");
    std::fs::create_dir_all(out_dir)?;

    let out_path = out_dir.join("coupling_action_sweep.csv");
    let mut file = File::create(&out_path)?;

    writeln!(file, "coupling,phi,l_grav,l_avt,action")?;

    let ricci_scalar = 0.0;
    let matter_density = 0.0;
    let scale_factor = 1.0;
    let dim = 512;

    // Sweep coupling strength logarithmically
    let couplings: Vec<f64> = vec![
        0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0, 1000.0,
    ];

    for &coupling in &couplings {
        let mut max_action = f64::NEG_INFINITY;
        let mut optimal_phi = 0.0;

        let cc = AlgebraicCosmologicalConstant { scale_factor };

        let avt = TopologicalFrictionLagrangian {
            coupling_strength: coupling,
            manifold_dimension: dim,
        };

        for i in 0..=steps {
            let phi = i as f64 / steps as f64;

            let l_grav = ricci_scalar * scale_factor + cc.lagrangian_density(phi);
            let l_avt = avt.lagrangian_density(phi);

            let action = compute_local_action(
                phi,
                ricci_scalar,
                matter_density,
                scale_factor,
                coupling,
                dim,
            );

            if action > max_action {
                max_action = action;
                optimal_phi = phi;
            }

            writeln!(file, "{},{},{},{},{}", coupling, phi, l_grav, l_avt, action)?;
        }

        println!(
            "Coupling: {:8.3} | Optimal Phi: {:.4} | Shift: {:.4}",
            coupling,
            optimal_phi,
            VACUUM_PHI - optimal_phi
        );
    }

    println!(
        "<EMOJI+2705> Sweep complete. Data written to {}",
        out_path.display()
    );
    Ok(())
}
