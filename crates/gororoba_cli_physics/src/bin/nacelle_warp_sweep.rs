//! Nacelle warp bubble parameter sweep.
//!
//! Sweeps the number of nacelles (n=1..16) and computes total negative energy,
//! Pfenning-Ford ratio, peak negative energy density, York time, and
//! interior-flat residual for each configuration.
//!
//! In `--paper-mode`, restricts to n=2,3,4 + Alcubierre baseline (matching
//! White et al. 2025 Fig. 5).

use clap::Parser;
use gr_core::warp_metric::{
    NacelleWarpParams, nacelle_energy_density, nacelle_york_time, nacelle_cross_section_energy,
};
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "nacelle-warp-sweep")]
#[command(about = "Sweep nacelle count and compare warp bubble energy budgets")]
struct Args {
    /// Minimum nacelle count
    #[arg(long, default_value_t = 1)]
    n_min: usize,

    /// Maximum nacelle count
    #[arg(long, default_value_t = 16)]
    n_max: usize,

    /// Number of integration grid points per axis
    #[arg(long, default_value_t = 80)]
    n_points: usize,

    /// Paper mode: only n=2,3,4 + Alcubierre baseline (White 2025)
    #[arg(long)]
    paper_mode: bool,

    /// Warp bubble velocity (units of c)
    #[arg(long, default_value_t = 0.1)]
    velocity: f64,

    /// Warp bubble radius
    #[arg(long, default_value_t = 100.0)]
    radius: f64,

    /// Output CSV file path (if omitted, prints to stdout)
    #[arg(long)]
    output: Option<PathBuf>,

    /// Print ASCII cross-sections
    #[arg(long)]
    ascii: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let nacelle_counts: Vec<usize> = if args.paper_mode {
        vec![0, 2, 3, 4] // 0 = Alcubierre baseline
    } else {
        let mut v = vec![0]; // Alcubierre baseline
        v.extend(args.n_min..=args.n_max);
        v
    };

    let mut csv = String::new();
    let _ = writeln!(
        csv,
        "n_nacelles,total_energy,peak_negative_density,york_time_max,interior_residual"
    );

    for &n in &nacelle_counts {
        let params = if n == 0 {
            let sigma = 1.0 / (0.05 * args.radius);
            NacelleWarpParams::alcubierre(args.velocity, args.radius, sigma)
        } else {
            let mut p = NacelleWarpParams::white_2025(args.velocity, n);
            p.r_bubble = args.radius;
            p.sigma = 1.0 / (0.05 * args.radius);
            p
        };

        let half = args.radius * 2.0;
        let n_grid = args.n_points;
        let dx = 2.0 * half / n_grid as f64;

        // Compute energy density on a 2D cross-section (y-z plane at x=0)
        let mut total_energy = 0.0;
        let mut peak_negative = 0.0f64;
        let mut york_max = 0.0f64;

        for iy in 0..n_grid {
            let y = -half + (iy as f64 + 0.5) * dx;
            for iz in 0..n_grid {
                let z = -half + (iz as f64 + 0.5) * dx;
                let rho = nacelle_energy_density(0.0, y, z, &params);
                total_energy += rho * dx * dx;
                if rho < peak_negative {
                    peak_negative = rho;
                }
                let theta = nacelle_york_time(0.0, y, z, &params);
                if theta.abs() > york_max.abs() {
                    york_max = theta;
                }
            }
        }

        // Interior residual: check K at center (should be ~0 for interior-flat)
        let interior_residual = nacelle_york_time(0.0, 0.0, 0.0, &params).abs();

        let label = if n == 0 {
            "Alcubierre".to_string()
        } else {
            format!("{n}")
        };

        let _ = writeln!(
            csv,
            "{label},{total_energy:.6e},{peak_negative:.6e},{york_max:.6e},{interior_residual:.6e}"
        );

        if args.ascii {
            let grid = nacelle_cross_section_energy(&params, half, 40);
            eprintln!("--- n={label} ---");
            eprintln!("{}", grid);
        }
    }

    match args.output {
        Some(path) => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&path, &csv)?;
            eprintln!(
                "wrote {} configurations to {}",
                nacelle_counts.len(),
                path.display()
            );
        }
        None => {
            print!("{csv}");
        }
    }

    Ok(())
}
