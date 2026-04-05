use std::{fs::File, io::Write, path::Path};

use verified_core::{
    axiomatic_gates::VACUUM_PHI,
    unified_action::{
        ActionComponent, AlgebraicCosmologicalConstant, TopologicalFrictionLagrangian,
        compute_local_action,
    },
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("<EMOJI+1F30C> Initializing Bott Periodicity Mod 8 Action Sweep...");

    let steps = 1000;
    let out_dir = Path::new("data/results");
    std::fs::create_dir_all(out_dir)?;

    let out_path = out_dir.join("bott_periodicity_sweep.csv");
    let mut file = File::create(&out_path)?;

    writeln!(file, "dim,phi,l_grav,l_avt,action")?;

    let ricci_scalar = 0.0;
    let matter_density = 0.0;
    let scale_factor = 1.0;
    let avt_coupling = 5.0;

    let dimensions = vec![
        8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128,
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

            let action = compute_local_action(
                phi,
                ricci_scalar,
                matter_density,
                scale_factor,
                avt_coupling,
                dim,
            );

            if action > max_action {
                max_action = action;
                optimal_phi = phi;
            }

            writeln!(file, "{},{},{},{},{}", dim, phi, l_grav, l_avt, action)?;
        }

        println!(
            "Dim: {:4} (k = {}) | Optimal Phi: {:.4} | Shift: {:.4}",
            dim,
            dim / 8,
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
