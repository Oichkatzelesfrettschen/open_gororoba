use std::{fs::File, io::Write, path::Path};

use verified_core::{
    axiomatic_gates::VACUUM_PHI,
    unified_action::{ActionComponent, AlgebraicCosmologicalConstant},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("<EMOJI+1F30C> Initializing Non-Integer/Fractal Dimensional Action Sweep...");

    let steps = 1000;
    let out_dir = Path::new("data/results");
    std::fs::create_dir_all(out_dir)?;

    let out_path = out_dir.join("fractal_dimensional_sweep.csv");
    let mut file = File::create(&out_path)?;

    writeln!(file, "dim,phi,l_grav,l_avt,action")?;

    let ricci_scalar = 0.0;
    let matter_density = 0.0;
    let scale_factor = 1.0;
    let avt_coupling = 5.0;

    // Sweep continuous dimensions from 1 to 10
    // We need to modify TopologicalFrictionLagrangian to accept f64 dimension
    // But since it accepts usize right now, we will simulate it directly here for the proof of concept

    // Instead of using TopologicalFrictionLagrangian, we'll manually compute the continuous action
    let dimensions: Vec<f64> = (10..=100).map(|x| x as f64 / 10.0).collect();

    for &dim in &dimensions {
        let mut max_action = f64::NEG_INFINITY;
        let mut optimal_phi = 0.0;

        let cc = AlgebraicCosmologicalConstant { scale_factor };

        for i in 0..=steps {
            let phi = i as f64 / steps as f64;

            let l_grav = ricci_scalar * scale_factor + cc.lagrangian_density(phi);

            // Continuous version of topological friction
            let deviation = phi - VACUUM_PHI;
            let l_avt = -avt_coupling * deviation.powi(2) * dim.ln();

            let action = l_grav + matter_density + l_avt;

            if action > max_action {
                max_action = action;
                optimal_phi = phi;
            }

            writeln!(file, "{},{},{},{},{}", dim, phi, l_grav, l_avt, action)?;
        }

        println!(
            "Dim: {:5.1} | Optimal Phi: {:.4} | Shift: {:.4}",
            dim,
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
