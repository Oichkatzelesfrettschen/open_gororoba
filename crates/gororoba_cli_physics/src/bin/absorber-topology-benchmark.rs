//! CLI binary for the C-010 Absorber Topology Benchmark.
//!
//! Compares mode-isolation performance of ZD clique topology vs local
//! alternatives (ring, chain, complete graph). Outputs a summary table
//! and per-topology metrics.
//!
//! Usage:
//!   `cargo run --release --bin absorber-topology-benchmark [-- --full]`
//!
//! Options:
//!   --full    Run with 7 x K6 (42 nodes) instead of 2 x K3 (6 nodes)

use optics_core::{
    absorber_benchmark::{reduced_c010_suite, run_benchmark, standard_c010_suite},
    tcmt::KerrCavity,
};

fn main() {
    let full = std::env::args().any(|a| a == "--full");

    let kappa = 1.0e10; // Coupling rate (rad/s)
    let cavity = KerrCavity::new(
        1.0e15,  // omega_0 ~ 1 PHz (optical)
        1.0e6,   // Q_intrinsic
        5.0e5,   // Q_external
        1.5,     // n_linear (glass-like)
        2.0e-18, // n2 (silica-like)
        1.0e-18, // v_eff (1 um^3)
    );
    let freq_spacing = 1.0e12; // 1 THz between channels
    let drive_amp = 1.0e-3;
    let dt = 1.0e-12; // 1 ps timestep
    let n_steps = 5000;

    let (suite, driven) = if full {
        println!("=== C-010 Absorber Topology Benchmark (FULL: 7 x K6) ===");
        // Drive first clique (nodes 0..5) in the 7xK6 topology
        let driven: Vec<usize> = (0..6).collect();
        (standard_c010_suite(kappa), driven)
    } else {
        println!("=== C-010 Absorber Topology Benchmark (REDUCED: 2 x K3) ===");
        // Drive first clique (nodes 0..2) in the 2xK3 topology
        let driven: Vec<usize> = (0..3).collect();
        (reduced_c010_suite(kappa), driven)
    };

    println!("Driven nodes: {:?}", driven);
    println!("Steps: {}, dt: {:.0e} s", n_steps, dt);
    println!();

    let result = run_benchmark(
        &suite,
        &cavity,
        freq_spacing,
        &driven,
        drive_amp,
        dt,
        n_steps,
    );

    println!("{}", result.summary_table());
    println!();

    if result.zd_dominates() {
        println!("RESULT: ZD clique topology DOMINATES all local alternatives.");
        println!("        Confirms C-010 negative result: local topologies cannot");
        println!("        reproduce the mode isolation of disconnected ZD cliques.");
    } else {
        println!("WARNING: ZD topology did NOT dominate. Check parameters.");
    }

    println!();
    println!("Per-topology details:");
    for r in &result.results {
        println!(
            "  {}: {} components, isolation={:.6}, xtalk={:.1} dB, gap={:.4}",
            r.topology_name, r.n_components, r.isolation_ratio, r.crosstalk_db, r.spectral_gap
        );
    }
}
