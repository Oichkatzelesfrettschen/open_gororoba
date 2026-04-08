//! Bridge analysis: Linking Algebraic Phase Transitions to Fluid Defect Floors.
//!
//! This executable tests the hypothesis that the "Topological Defect Floor"
//! observed in LBM simulations is fundamentally driven by the
//! non-associativity phase transition of the underlying CD-16 (Sedenion) manifold.

use algebra_analysis::phase_transition::PhaseTransitionAnalyzer;
use nalgebra::DVector;
use std::{
    fs::File,
    io::{BufRead, BufReader},
};
use verified_core::coupler_manifold::{CouplerJacobian, CouplerPoint};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("--- Step 1: Algebraic Baseline (CD-8 vs CD-16) ---");
    let mut alg_points = Vec::new();

    for &dim in &[8, 16, 32, 64] {
        let analyzer = PhaseTransitionAnalyzer::new(dim);
        let density = analyzer.calculate_defect_density(5000, 42);
        // Manually construct a 1-observable CouplerPoint to match fluid count
        let p = CouplerPoint {
            g: DVector::from_vec(vec![dim as f64]),
            o: DVector::from_vec(vec![density + 1e-12]),
        };
        println!("Dim (g) = {}, Defect Density (O) = {:.4}", dim, p.o[0]);
        alg_points.push(p);
    }

    let j_alg = CouplerJacobian::estimate_from_delta(&alg_points[0], &alg_points[1])?;
    println!("Algebraic Jacobian J (8 -> 16): {:.4}", j_alg.j_mat[(0, 0)]);

    println!("\n--- Step 2: Fluid Defect Baseline (LBM Topo Voids) ---");
    let path = "../../data/topological_voids.csv";
    let thresholds = vec![0.45, 0.50, 0.55];
    let mut fluid_points = Vec::new();

    for &thr in &thresholds {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut count = 0;
        for line in reader.lines().skip(1) {
            let l = line?;
            let parts: Vec<&str> = l.split(',').collect();
            if parts.len() < 5 {
                continue;
            }
            if let Ok(imbalance) = parts[4].parse::<f64>() && imbalance > thr {
                count += 1;
            }
        }
        println!("Imbalance > {:.2} (g): Count O = {}", thr, count);
        fluid_points.push(CouplerPoint {
            g: DVector::from_vec(vec![thr]),
            o: DVector::from_vec(vec![count as f64 + 1e-6]),
        });
    }

    let j_fluid = CouplerJacobian::estimate_from_delta(&fluid_points[0], &fluid_points[1])?;
    println!(
        "Fluid Jacobian J ({:.2} -> {:.2}): {:.4}",
        thresholds[0],
        thresholds[1],
        j_fluid.j_mat[(0, 0)]
    );

    println!("\n--- Step 3: Two-Sector Mixture Modeling (Fluid Floor) ---");

    // We assume the lowest count (at thr=0.55) is an estimate of the 'burst floor' (O_floor)
    // O_obs = O_smooth(g) + O_floor
    let o_floor = fluid_points[2].o[0];
    let o_smooth_0 = fluid_points[0].o[0] - o_floor;
    let o_smooth_1 = fluid_points[1].o[0] - o_floor;

    let p_smooth_0 = CouplerPoint {
        g: DVector::from_vec(vec![thresholds[0]]),
        o: DVector::from_vec(vec![o_smooth_0]),
    };
    let p_smooth_1 = CouplerPoint {
        g: DVector::from_vec(vec![thresholds[1]]),
        o: DVector::from_vec(vec![o_smooth_1]),
    };

    if let Ok(jac_smooth) = CouplerJacobian::estimate_from_delta(&p_smooth_0, &p_smooth_1) {
        println!(
            "Floor-Subtracted Fluid Jacobian J: {:.4}",
            jac_smooth.j_mat[(0, 0)]
        );
        println!("(Original was {:.4})", j_fluid.j_mat[(0, 0)]);
    }

    println!("\n--- Step 4: The Connection ---");
    // We treat 'imbalance' in the fluid as a proxy for 'algebraic defect density'.
    // If the fluid is 'saturated' by the Sedenion floor, then the Jacobians should
    // align or show a predictable delta.

    let j_cross = j_alg.detect_confound(&j_fluid, 1.0);
    if j_cross.is_empty() {
        println!(
            "!! NOVEL INSIGHT: The fluid defect scaling is statistically consistent with the pure algebraic phase transition !!"
        );
        println!(
            "This suggests the 'Defect Floor' in the fluid is a direct projection of the Sedenion non-associativity bound."
        );
    } else {
        println!(
            "Mismatched Scaling: Delta J = {:.2}",
            (j_alg.j_mat[(0, 0)] - j_fluid.j_mat[(0, 0)]).abs()
        );
        println!(
            "The fluid adds extra complexity beyond the static algebra (likely due to dynamical dissipation)."
        );
    }

    Ok(())
}
