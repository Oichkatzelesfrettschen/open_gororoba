//! cmb-axis-crucible: Test 256D Voudon Global Bias vs Planck 2018 Axis of Evil.
//!
//! Validates whether the Voudon Global Bias vector aligns with the observed
//! anomalous multipole alignment in the Cosmic Microwave Background.

use cosmology_core::VoudonCmbAnalyzer;
use gororoba_algebra::construction::deep_space::compute_voudon_imbalance_density;

fn main() -> anyhow::Result<()> {
    println!("=== Phase 3: CMB Axis of Evil Crucible (256D Voudon Alignment) ===");

    // 1. Get Algebraic input from 256D Voudon algebra
    let phi = compute_voudon_imbalance_density();
    println!("  Voudon Imbalance Density (Phi): {:.8}", phi);

    // 2. Initialize CMB Analyzer
    let analyzer = VoudonCmbAnalyzer::new(phi);
    let axis = analyzer.project_axis();

    println!("\n[ Voudon Global Bias Projection ]");
    println!("  Vector: [{:.6}, {:.6}, {:.6}]", axis[0], axis[1], axis[2]);

    // 3. Observed 'Axis of Evil' coordinates (Approximate)
    // The axis is roughly (l, b) = (260, 60) in galactic coordinates.
    // Projected to J2000 Cartesian unit vector:
    let raw = [-0.15_f64, 0.85, 0.50];
    let obs_norm = (raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]).sqrt();
    let axis_obs = [raw[0] / obs_norm, raw[1] / obs_norm, raw[2] / obs_norm];
    println!(
        "  Observed CMB Axis (Planck): [{:.6}, {:.6}, {:.6}]",
        axis_obs[0], axis_obs[1], axis_obs[2]
    );

    // 4. Alignment Test
    let alignment =
        (axis[0] * axis_obs[0] + axis[1] * axis_obs[1] + axis[2] * axis_obs[2]).abs();
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
