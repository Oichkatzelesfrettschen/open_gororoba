//! cmb-axis-crucible: Test 256D Voudon Global Bias vs Planck 2018 Axis of Evil.
//!
//! Validates whether the Voudon Global Bias vector aligns with the observed
//! anomalous multipole alignment in the Cosmic Microwave Background.

use cosmology_core::VoudonCmbAnalyzer;
use gororoba_algebra::construction::deep_space::compute_voudon_imbalance_density;
use nalgebra::Vector3;

fn main() -> anyhow::Result<()> {
    println!("=== Phase 3: CMB Axis of Evil Crucible (256D Voudon Alignment) ===");

    // 1. Get Algebraic input from 256D Voudon algebra
    let phi = compute_voudon_imbalance_density();
    println!("  Voudon Imbalance Density (Phi): {:.8}", phi);

    // 2. Initialize CMB Analyzer
    let analyzer = VoudonCmbAnalyzer::new(phi);
    let axis = analyzer.project_axis();

    println!("\n[ Voudon Global Bias Projection ]");
    println!("  Vector: [{:.6}, {:.6}, {:.6}]", axis.x, axis.y, axis.z);

    // 3. Observed 'Axis of Evil' coordinates (Approximate)
    // The axis is roughly (l, b) = (260, 60) in galactic coordinates.
    // Projected to J2000 Cartesian unit vector:
    let axis_obs = Vector3::new(-0.15, 0.85, 0.50).normalize();
    println!(
        "  Observed CMB Axis (Planck): [{:.6}, {:.6}, {:.6}]",
        axis_obs.x, axis_obs.y, axis_obs.z
    );

    // 4. Alignment Test
    let alignment = axis.dot(&axis_obs).abs();
    let angle_deg = alignment.acos().to_degrees();

    println!("\n=== Final Alignment Verdict ===");
    println!("  Cosine Alignment: {:.6}", alignment);
    println!("  Angular Deviation: {:.2} degrees", angle_deg);

    if alignment > 0.9 {
        println!(
            "  VERDICT: PASS (Strong alignment detected. 256D Voudon bias explains Axis of Evil)"
        );
    } else if alignment > 0.5 {
        println!("  VERDICT: TENTATIVE (Partial correlation. Refine Voudon-CMB coupling model)");
    } else {
        println!(
            "  VERDICT: FAIL (No significant alignment. Axis of Evil is likely non-algebraic)"
        );
    }

    Ok(())
}
