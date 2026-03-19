//! Mock experiment for the Coupler-Manifold Monograph.
//!
//! This script simulates:
//! 1. QEC Distance Sweeps with "Sector Tagging" (Smooth vs. Burst).
//! 2. MIPT Two-Path Tuning (Density vs. Depth) to detect Confounding.
//! 3. Rare-event Cosmic Ray Bursts.
//!
//! Run with: `cargo run --example coupler_manifold_mock`

use nalgebra::{DMatrix, DVector};
use quantum_core::coupler_manifold::{
    CouplerPoint, CouplerJacobian, IdentifiabilityAudit,
    qec::TwoSectorMixture, mipt::effective_measurement_rate
};

fn main() {
    println!("--- Step 1: Decoupling Smooth vs. Burst Sectors in QEC ---");
    
    // Define a theoretical model for distance-7 surface code
    // Smooth sector: p/p_thr = 0.5, J = (d+1)/2 = 4.0
    // Burst sector: rare events occurring once per hour (~10^9 cycles)
    let model = TwoSectorMixture {
        smooth_epsilon: 1.43e-3, // 0.143% from Google 2023
        burst_amplitude: 0.1,    // 10% error during burst
        burst_rate_per_cycle: 1e-9,
    };

    println!("Average Error: {:.2e}", model.average_error());
    println!("Error (No Burst): {:.2e}", model.conditional_error(false));
    println!("Error (With Burst): {:.2e}", model.conditional_error(true));

    // Simulate a sweep of p/p_thr near 0.5
    let p_vals = vec![0.48, 0.49, 0.50, 0.51, 0.52];
    let d = 7.0;
    let _p_thr = 1.0; // Normalized

    println!("\nSweeping p/p_thr for d=7:");
    for p in p_vals {
        // Simple power law for smooth sector: eps ~ (p/p_thr)^((d+1)/2)
        let p_ratio: f64 = p / 0.5;
        let eps_smooth = 1.43e-3 * p_ratio.powf((d + 1.0) / 2.0);
        
        // Observed error is mixture
        let eps_obs = eps_smooth + (model.burst_rate_per_cycle * model.burst_amplitude);
        
        println!("p={:.2}: eps_smooth={:.2e}, eps_obs={:.2e}", p, eps_smooth, eps_obs);
    }

    println!("\n--- Step 2: Proving MIPT Confounding in Google/IBM Sweeps ---");
    
    // Path 1: Vary p by layout (M) at fixed Depth (T=5)
    // Path 2: Vary p by Depth (T) at fixed Layout (M=10)
    let l = 40.0;
    
    // Point A (Base): M=10, T=5
    let p_a = effective_measurement_rate(10.0, l, 5.0);
    let o_a = DVector::from_vec(vec![0.8]); // Teleportation fidelity proxy
    let base = CouplerPoint {
        g: DVector::from_vec(vec![p_a, 5.0]), // [p, T]
        o: o_a.clone(),
    };

    // Point B (Path 1: Density): M=11, T=5
    let p_b = effective_measurement_rate(11.0, l, 5.0);
    let o_b = DVector::from_vec(vec![0.82]); 
    let density_pert = CouplerPoint {
        g: DVector::from_vec(vec![p_b, 5.0]),
        o: o_b,
    };

    // Point C (Path 2: Depth): M=10, T=4.5 (Decrease depth to increase effective p)
    let p_c = effective_measurement_rate(10.0, l, 4.5);
    // Depth decrease also REDUCES noise, so fidelity should go UP more than expected 
    // from just p change. This is the confounding!
    let o_c = DVector::from_vec(vec![0.85]); 
    let depth_pert = CouplerPoint {
        g: DVector::from_vec(vec![p_c, 4.5]),
        o: o_c,
    };

    let j_density = CouplerJacobian::estimate_from_delta(&base, &density_pert).unwrap();
    let j_depth = CouplerJacobian::estimate_from_delta(&base, &depth_pert).unwrap();

    println!("Jacobian (Density Path) [0,0] (d ln O / d ln p): {:.2}", j_density.j_mat[(0,0)]);
    println!("Jacobian (Depth Path)   [0,0] (d ln O / d ln p): {:.2}", j_depth.j_mat[(0,0)]);
    println!("Mismatch {:.2} vs {:.2} indicates confounding from T-dependent noise.", 
             j_density.j_mat[(0,0)], j_depth.j_mat[(0,0)]);

    // Fisher Audit
    let sigma_y_inv = DMatrix::identity(1, 1) * 100.0; // High precision
    let fisher = j_depth.fisher_information(&sigma_y_inv);
    let audit = IdentifiabilityAudit::perform(&fisher).unwrap();
    println!("Identifiability Audit (Depth Path): Identifiable={}, Cond={:.2e}", 
             audit.is_identifiable, audit.condition_number);

    println!("\n--- Step 3: Mapping to Non-Euclidean Holography (Bruhat-Tits) ---");
    use quantum_core::coupler_manifold::tree_geometry::BruhatTitsTree;
    
    let tree = BruhatTitsTree::new(2);
    let d02 = tree.p_adic_norm(0, 2);
    let d04 = tree.p_adic_norm(0, 4);
    let d24 = tree.p_adic_norm(2, 4);

    println!("p-adic distances (p=2):");
    println!("d(0, 2) = {:.2}", d02);
    println!("d(0, 4) = {:.2}", d04);
    println!("d(2, 4) = {:.2}", d24);
    println!("Strong Triangle Inequality: d(0,4) <= max(d(0,2), d(2,4)) -> {:.2} <= {:.2}", 
             d04, d02.max(d24));
}
