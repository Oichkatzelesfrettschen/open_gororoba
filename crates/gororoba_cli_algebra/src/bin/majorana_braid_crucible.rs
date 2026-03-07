//! Majorana braid crucible: experiment runner for CD tower braiding.
//!
//! Subcommands:
//! - braid: Execute standard 4-Majorana braiding at specified dim
//! - fusion: Box-kite fusion channel statistics
//! - protection: Missing-edge associator analysis
//! - aligned-chsh: Braid-aligned CHSH test
//! - complex-time: Wick rotation sweep
//! - scaling: Friction vs dimension scaling

use algebra_experimental::majorana_braiding;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "majorana-braid-crucible")]
#[command(about = "Majorana braiding operators mapped to the Cayley-Dickson tower")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Execute standard 4-Majorana braiding at specified dimension
    Braid {
        #[arg(long, default_value_t = 16)]
        dim: usize,
    },
    /// Box-kite fusion channel statistics
    Fusion {
        #[arg(long, default_value_t = 16)]
        dim: usize,
    },
    /// Missing-edge associator analysis
    Protection {
        #[arg(long, default_value_t = 16)]
        dim: usize,
    },
    /// Braid-aligned CHSH test
    AlignedChsh {
        #[arg(long, default_value_t = 16)]
        dim: usize,
        #[arg(long, default_value_t = 10000)]
        n_samples: usize,
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Wick rotation sweep
    ComplexTime {
        #[arg(long, default_value_t = 16)]
        dim: usize,
        #[arg(long, default_value_t = 20)]
        theta_steps: usize,
    },
    /// Friction vs dimension scaling
    Scaling {
        #[arg(long, default_value_t = 512)]
        max_dim: usize,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::Braid { dim } => {
            println!("=== Majorana Braid Experiment (dim={dim}) ===\n");

            let circuit = vec![(0, 1), (1, 2), (2, 3), (0, 1)];
            let result = majorana_braiding::braiding_experiment(&circuit, dim, 42);

            println!("Braid sequence: {:?}", result.braid_sequence);
            println!("Clifford parity: {}", result.clifford_parity);
            println!("CD parity:       {}", result.cd_parity);
            println!("Total friction:  {:.6}", result.total_friction);
            println!("Fidelity:        {:.6}", result.fidelity);
            println!("\nFriction per braid:");
            for (i, f) in result.friction_per_braid.iter().enumerate() {
                println!("  sigma_{}: {:.6}", i, f);
            }

            // Clifford control group
            println!("\n--- Clifford Control Group ---");
            for (i, j) in &[(0usize, 1usize), (1, 2), (2, 3)] {
                let u_err = majorana_braiding::check_braid_unitarity(*i, *j);
                let p_err = majorana_braiding::check_braid_preserves_parity(*i, *j);
                println!(
                    "  U_{}{}: unitarity_err={:.2e}, [P,U]_norm={:.2e}",
                    i, j, u_err, p_err
                );
            }
        }
        Commands::Fusion { dim } => {
            println!("=== Box-Kite Fusion Channels (dim={dim}) ===\n");

            let fusion = majorana_braiding::boxkite_fusion_channels(dim);
            for (kite_idx, n_channels, signs) in &fusion {
                println!(
                    "Box-kite {}: {} channels, signs = {:?}",
                    kite_idx, n_channels, signs
                );
            }

            let all_two = fusion.iter().all(|(_, n, _)| *n == 2);
            println!(
                "\nIsing anyon fusion (all 2-channel): {}",
                if all_two { "CONFIRMED" } else { "FALSIFIED" }
            );
        }
        Commands::Protection { dim } => {
            println!("=== Missing-Edge Associator Analysis (dim={dim}) ===\n");

            let sign_table =
                algebra_experimental::bell_inequality::SignTableCache::new(dim);
            let results =
                majorana_braiding::test_missing_edge_protection(dim, &sign_table);

            for (i, j, max_assoc) in &results {
                let status = if *max_assoc < 1e-12 {
                    "PROTECTED"
                } else {
                    "EXPOSED"
                };
                println!(
                    "  ({:3}, {:3}): max|[e_i, e_k, e_j]| = {:.6}  [{}]",
                    i, j, max_assoc, status
                );
            }

            let n_protected = results.iter().filter(|(_, _, m)| *m < 1e-12).count();
            println!(
                "\nProtected: {}/{} missing edges",
                n_protected,
                results.len()
            );
        }
        Commands::AlignedChsh { dim, n_samples, seed } => {
            println!(
                "=== Braid-Aligned CHSH Test (dim={dim}, n={n_samples}) ===\n"
            );

            let result = majorana_braiding::chsh_braid_aligned(dim, n_samples, seed);

            println!("S = {:.6} +/- {:.6}", result.s_value, result.s_error);
            println!("Violates classical: {}", result.violates_classical);
            println!("Correlators: {:?}", result.correlators);
            println!("ZD pairs: {}", result.n_zd_pairs);
            println!("Usable channels: {}", result.n_usable_channels);
            println!("Optimal probes: {}", result.n_optimal_probes);
        }
        Commands::ComplexTime { dim, theta_steps } => {
            println!(
                "=== Complex-Time Braid Sweep (dim={dim}, steps={theta_steps}) ===\n"
            );

            let modes = majorana_braiding::map_majoranas_to_cd(4, dim);
            let sign_table =
                algebra_experimental::bell_inequality::SignTableCache::new(dim);

            println!("{:>8} {:>12} {:>12}", "theta", "friction", "max_assoc");
            for step in 0..=theta_steps {
                let theta =
                    std::f64::consts::FRAC_PI_2 * step as f64 / theta_steps as f64;
                let result = majorana_braiding::complex_time_braid(
                    &modes[0],
                    &modes[1],
                    theta,
                    &sign_table,
                );
                println!(
                    "{:8.4} {:12.6} {:12.6}",
                    theta, result.topological_friction, result.max_associator_norm
                );
            }
        }
        Commands::Scaling { max_dim } => {
            println!("=== Friction vs Dimension Scaling ===\n");

            let results = majorana_braiding::friction_dimensional_scaling(max_dim);

            println!("{:>6} {:>14} {:>14}", "dim", "friction", "max_assoc");
            for (dim, friction, max_assoc) in &results {
                println!("{:6} {:14.6} {:14.6}", dim, friction, max_assoc);
            }
        }
    }
}
