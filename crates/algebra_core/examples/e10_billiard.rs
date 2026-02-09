//! E10 Cosmological Billiard Simulation (Upgraded)
//!
//! This executable simulates the chaotic dynamics of a "particle" (the Universe)
//! bouncing inside the fundamental Weyl chamber of the E10 Kac-Moody algebra.
//! Uses the library's HyperbolicBilliard with proper geodesic flow on H^9.

use algebra_core::experimental::billiard_stats::{self, NullModel};
use algebra_core::lie::kac_moody::E10RootSystem;
use algebra_core::physics::billiard_sim::{BilliardConfig, HyperbolicBilliard, LorentzVec};
use std::collections::HashSet;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let seed: u64 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(42);
    let n_steps: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);

    println!("=== E10 Cosmological Billiard Simulation ===");
    println!("RNG seed: {}, target bounces: {}", seed, n_steps);

    let e10 = E10RootSystem::new();
    let mut billiard = HyperbolicBilliard::from_e10(&e10, seed);

    println!("Initial constraints: {:?}", billiard.diagnostics());
    println!("Simulating bounces...");

    let mut sequence = Vec::new();
    for i in 0..n_steps {
        match billiard.step() {
            Some(res) => {
                sequence.push(res.wall_idx);
                if i % 100 == 0 && i > 0 {
                    println!(
                        "  Bounce {}: wall {}, diag error = {:.2e}",
                        i, res.wall_idx, res.diagnostics.pos_norm_error
                    );
                }
            }
            None => {
                println!("Particle escaped at step {}!", i);
                break;
            }
        }
    }

    println!("\n=== Results ===");
    println!("Actual bounces completed: {}", sequence.len());
    println!(
        "Max constraint errors: pos={:.2e} tang={:.2e} vel={:.2e}",
        billiard.max_pos_error, billiard.max_tangency_error, billiard.max_vel_error
    );

    // Hit frequencies
    let mut wall_hits = vec![0u64; 10];
    for &w in &sequence {
        wall_hits[w] += 1;
    }
    for (i, &h) in wall_hits.iter().enumerate() {
        println!(
            "  Wall {}: {:4} hits ({:.1}%)",
            i,
            h,
            100.0 * h as f64 / sequence.len() as f64
        );
    }

    // Locality and Fano Analysis
    let analysis = billiard_stats::fano_structure_analysis_from_sequence(&sequence, 1000, seed);
    println!("\n{}", billiard_stats::fano_analysis_report(&analysis));

    println!("\nDone.");
}
