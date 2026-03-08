//! Phase transition crucible: defect density and friction landscape analysis.
//!
//! Subcommands:
//! - density-scan: Defect density vs dimension
//! - friction-scan: Topological friction vs dimension
//! - order-param: Order parameter vs dimension with derived critical density

use algebra_analysis::phase_transition::{
    PhaseTransitionAnalyzer, defect_saturation, missing_edge_fraction, zd_edge_density,
};
use algebra_experimental::majorana_braiding;
use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "phase-transition-crucible")]
#[command(about = "Phase transition landscape analysis for Cayley-Dickson algebras")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Defect density vs dimension scan
    DensityScan {
        /// Minimum dimension (must be power of 2, >= 4)
        #[arg(long, default_value_t = 4)]
        dim_min: usize,
        /// Maximum dimension (must be power of 2)
        #[arg(long, default_value_t = 512)]
        dim_max: usize,
        /// Number of Monte Carlo samples per dimension
        #[arg(long, default_value_t = 100_000)]
        n_samples: usize,
        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// Topological friction vs dimension scaling
    FrictionScan {
        /// Maximum dimension (must be power of 2)
        #[arg(long, default_value_t = 512)]
        dim_max: usize,
    },
    /// Order parameter vs dimension with derived critical density
    OrderParam {
        /// Minimum dimension (must be power of 2, >= 4)
        #[arg(long, default_value_t = 4)]
        dim_min: usize,
        /// Maximum dimension (must be power of 2)
        #[arg(long, default_value_t = 512)]
        dim_max: usize,
        /// Number of Monte Carlo samples per dimension
        #[arg(long, default_value_t = 100_000)]
        n_samples: usize,
        /// Random seed
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli.command {
        Commands::DensityScan {
            dim_min,
            dim_max,
            n_samples,
            seed,
        } => {
            println!("# Phase Transition Density Scan");
            println!(
                "# dim, defect_density, theoretical_saturation, edge_density, missing_fraction"
            );

            let mut dim = dim_min.max(4).next_power_of_two();
            while dim <= dim_max {
                let analyzer = PhaseTransitionAnalyzer::new(dim);
                let density = if dim >= 16 {
                    analyzer.calculate_defect_density(n_samples, seed)
                } else {
                    0.0
                };
                let saturation = defect_saturation(dim);
                let edge_d = zd_edge_density(dim);
                let missing_f = missing_edge_fraction(dim);

                println!("{dim}, {density:.6}, {saturation:.6}, {edge_d:.6}, {missing_f:.6}");
                dim *= 2;
            }
        }
        Commands::FrictionScan { dim_max } => {
            println!("# Friction Dimensional Scaling");
            println!("# dim, total_friction, max_associator_norm");

            let results = majorana_braiding::friction_dimensional_scaling(dim_max);
            for (dim, friction, max_norm) in &results {
                println!("{dim}, {friction:.6}, {max_norm:.6}");
            }
        }
        Commands::OrderParam {
            dim_min,
            dim_max,
            n_samples,
            seed,
        } => {
            println!("# Order Parameter vs Dimension");
            println!("# dim, defect_density, order_parameter, phase, critical_density");

            let mut dim = dim_min.max(4).next_power_of_two();
            while dim <= dim_max {
                let analyzer = PhaseTransitionAnalyzer::new(dim);
                let density = if dim >= 16 {
                    analyzer.calculate_defect_density(n_samples, seed)
                } else {
                    0.0
                };
                let order = analyzer.calculate_order_parameter(density);
                let phase = analyzer.evaluate_phase(density);

                println!(
                    "{dim}, {density:.6}, {order:.6}, {phase:?}, {:.6}",
                    analyzer.critical_density
                );
                dim *= 2;
            }
        }
    }
}
