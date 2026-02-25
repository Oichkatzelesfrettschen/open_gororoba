//! CLI binary for C-011 Bypass Model Comparison.
//!
//! Compares baseline (obstructed) gravastar EoS against two associative
//! bypass strategies: quaternion projection and quaternion restriction.
//!
//! Usage:
//!   cargo run --release --bin gravastar-bypass-compare [-- --sweep]
//!
//! Options:
//!   --sweep   Run gamma stability sweep (slower, more data)

use cosmology_core::bypass_models::{
    BypassConfig, BypassModel, bypass_stability_sweep, compare_bypass_models,
};

fn main() {
    let do_sweep = std::env::args().any(|a| a == "--sweep");

    println!("=== C-011 Bypass Model Comparison ===");
    println!();

    let config = BypassConfig {
        r1: 5.0,
        m_target: 10.0,
        compactness: 0.7,
        gamma: 2.0,
        aniso_lambda: 0.0,
        associator_norm: 1.0,
    };

    println!("Configuration:");
    println!(
        "  R1 = {}, M_target = {}, C = {}",
        config.r1, config.m_target, config.compactness
    );
    println!(
        "  gamma = {}, lambda = {}",
        config.gamma, config.aniso_lambda
    );
    println!("  associator_norm = {}", config.associator_norm);
    println!();

    // Single-point comparison
    let comp = compare_bypass_models(&config);
    println!("{}", comp.summary_table());
    println!();

    // Report boundary deviations
    println!("Boundary deviations (relative to baseline):");
    for (model, dev) in comp.boundary_deviations() {
        let name = match model {
            BypassModel::Baseline => "Baseline",
            BypassModel::QuaternionProjection => "A: Projection",
            BypassModel::QuaternionRestriction => "B: Restriction",
        };
        match dev {
            Some((md, cd)) => println!("  {}: mass_dev={:.4}, comp_dev={:.4}", name, md, cd),
            None => println!("  {}: FAILED", name),
        }
    }
    println!();

    if comp.all_valid() {
        println!("STATUS: All three models produce valid gravastar solutions.");
        println!("        Bypass models are associative (obstruction=0) at the cost of");
        println!("        reduced effective stiffness (K_eff < K_baseline).");
    } else {
        println!("WARNING: Not all models produced valid solutions.");
    }

    // Stability sweep
    if do_sweep {
        println!();
        println!("--- Gamma Stability Sweep ---");
        let sweep = bypass_stability_sweep(&config, 1.2, 3.5, 12);

        println!(
            "{:<8} {:<25} {:>10} {:>12} {:>8}",
            "gamma", "Model", "Mass", "Compactness", "Causal"
        );
        println!("{}", "-".repeat(65));

        for (gi, gamma) in sweep.gammas.iter().enumerate() {
            for (model, results) in &sweep.model_results {
                let name = match model {
                    BypassModel::Baseline => "Baseline",
                    BypassModel::QuaternionProjection => "A: Projection",
                    BypassModel::QuaternionRestriction => "B: Restriction",
                };
                match &results[gi] {
                    Some(obs) => println!(
                        "{:<8.2} {:<25} {:>10.4} {:>12.6} {:>8}",
                        gamma,
                        name,
                        obs.mass,
                        obs.compactness,
                        if obs.is_causal { "yes" } else { "no" }
                    ),
                    None => println!(
                        "{:<8.2} {:<25} {:>10} {:>12} {:>8}",
                        gamma, name, "FAILED", "-", "-"
                    ),
                }
            }
        }
    }
}
