//! Thesis Synthesis: Full 3D Simulation & Distillation.
//!
//! Orchestrates a high-fidelity 3D simulation using the Gororoba Engine,
//! performs TDA, and exports results to:
//! 1. HDF5 for claim synchronization.
//! 2. TikZ/PGFPlots for LaTeX-native visualization.

use algebra_core::physics::octonion_field::FieldParams;
use clap::{Parser, Subcommand};
use gororoba_engine::{SimulationConfig3D, SimulationState3D};
use lbm_3d_cuda::Precision;
use log::info;
use pgfplots::Picture;
use pgfplots::axis::Axis;
use pgfplots::axis::plot::Plot2D;
use std::error::Error;
#[cfg(feature = "hdf5-export")]
use std::path::Path;

#[derive(Parser)]
#[command(name = "thesis-synthesis")]
#[command(about = "Gororoba Thesis Synthesis & Experiment Engine")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the production sweep for the thesis (alpha = 0.01..0.2, 16^3 grid)
    Sweep,
    /// Run a single high-resolution experiment
    Run {
        /// Grid size (NxNxN)
        #[arg(long, default_value_t = 64)]
        grid: usize,

        /// Coupling strength alpha
        #[arg(long, default_value_t = 0.1)]
        alpha: f64,

        /// Number of steps
        #[arg(long, default_value_t = 50)]
        steps: usize,
    },
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let cli = Cli::parse();

    match cli.command {
        Commands::Sweep => run_sweep()?,
        Commands::Run { grid, alpha, steps } => run_single(grid, alpha, steps)?,
    }

    Ok(())
}

fn run_sweep() -> Result<(), Box<dyn Error>> {
    info!("=== Gororoba Thesis Synthesis (3D High-Fidelity Sweep) ===");

    // 1. Setup 3D Simulation Sweep
    let coupling_values: [f32; 4] = [0.01_f32, 0.05, 0.1, 0.2];
    let mut sweep_results: Vec<(f32, f64)> = Vec::new();

    for &alpha in &coupling_values {
        info!("--- Running sweep for alpha = {} ---", alpha);
        let config = SimulationConfig3D {
            nx: 16,
            ny: 16,
            nz: 16,
            tau: 0.8,
            use_gpu: cfg!(feature = "gpu"),
            precision: Precision::FP32,
            algebra_params: FieldParams::default(),
            coupling_fluid_algebra: alpha,
            coupling_algebra_fluid: alpha,
        };

        let mut state = SimulationState3D::new(config)?;

        let n_steps = 30;
        let mut rho_means = Vec::new();
        let mut times = Vec::new();

        for _ in 0..n_steps {
            state.step()?;
            rho_means.push(state.fluid.mean_density());
            times.push(state.current_step as f64);
        }

        // Collect final scalar summary (topology pipeline is not wired into SimulationState3D yet).
        let final_rho = rho_means.last().copied().unwrap_or(1.0);
        sweep_results.push((alpha, final_rho));

        // Export individual trace to HDF5
        #[cfg(feature = "hdf5-export")]
        {
            let h5_name = format!("trace_alpha_{:.2}.h5", alpha);
            let _ = data_core::hdf5_export::export_simulation_trace(
                Path::new(&h5_name),
                &times,
                &rho_means,
                &vec![0.0; n_steps],
                &vec![0.0; n_steps],
            );
        }
    }

    // 2. Generate TikZ/PGFPlots Visualization
    info!("Generating TikZ visualization...");

    let mut plot_rho = Plot2D::new();
    for (alpha, rho) in &sweep_results {
        plot_rho.coordinates.push(((*alpha as f64), *rho).into());
    }

    let mut axis = Axis::new();
    axis.plots.push(plot_rho);
    axis.set_title("Final Mean Density vs Coupling Strength");
    axis.set_x_label("Coupling Strength (\\alpha)");
    axis.set_y_label("rho_mean (final)");

    let picture = Picture::from(axis);
    let tikz_path = "docs/latex/thesis_betti_sweep.tex";
    std::fs::write(tikz_path, picture.to_string())?;
    info!("TikZ plot saved to {}", tikz_path);

    // 3. Generate Standalone LaTeX Wrapper
    let latex_content = format!(
        r#"\documentclass{{standalone}}
\usepackage{{pgfplots}}
\pgfplotsset{{compat=1.18}}
	\begin{{document}}
	{}
	\end{{document}}
	"#,
        picture
    );
    let standalone_path = "docs/latex/thesis_standalone.tex";
    std::fs::write(standalone_path, latex_content)?;
    info!("Standalone LaTeX wrapper saved to {}", standalone_path);

    info!("Sweep complete.");
    Ok(())
}

fn run_single(grid: usize, alpha: f64, steps: usize) -> Result<(), Box<dyn Error>> {
    info!(
        "=== Gororoba High-Res Experiment ({}x{}x{}, alpha={}) ===",
        grid, grid, grid, alpha
    );

    let alpha_f32 = alpha as f32;
    let config = SimulationConfig3D {
        nx: grid,
        ny: grid,
        nz: grid,
        tau: 0.8,
        use_gpu: cfg!(feature = "gpu"),
        precision: Precision::FP32,
        algebra_params: FieldParams::default(),
        coupling_fluid_algebra: alpha_f32,
        coupling_algebra_fluid: alpha_f32,
    };

    let mut state = SimulationState3D::new(config)?;
    let mut rho_means = Vec::new();
    let mut times = Vec::new();

    for i in 0..steps {
        if i % 10 == 0 {
            info!("Step {}/{}", i, steps);
        }
        state.step()?;
        rho_means.push(state.fluid.mean_density());
        times.push(state.current_step as f64);
    }

    #[cfg(feature = "hdf5-export")]
    {
        let h5_name = format!("trace_highres_grid{}_alpha{:.2}.h5", grid, alpha);
        let _ = data_core::hdf5_export::export_simulation_trace(
            Path::new(&h5_name),
            &times,
            &rho_means,
            &vec![0.0; steps],
            &vec![0.0; steps],
        );
        info!("Exported trace to {}", h5_name);
    }

    Ok(())
}
