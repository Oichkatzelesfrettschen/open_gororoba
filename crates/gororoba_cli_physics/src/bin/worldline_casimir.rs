//! Worldline Monte Carlo Casimir energy computation.
//!
//! Computes Casimir vacuum energy density using the v-loop algorithm
//! from White et al. (2021). Validates against the analytical parallel
//! plate result E/A = -pi^2/(720*a^3).
//!
//! Supports multiple boundary geometries: parallel plates, solid sphere,
//! infinite cylinder, sphere-in-cylinder, pillar-in-cavity.

use casimir_core::{
    InfiniteCylinder, ParallelPlates, PillarInCavity, SolidSphere, SphereInCylinder,
    WorldlineCasimirConfig, casimir_energy_at_point, casimir_energy_profile,
    casimir_parallel_plates_exact, energy_cross_section,
};
use clap::Parser;
use std::{error::Error, fmt::Write as _, fs, path::PathBuf};

#[derive(Debug, Parser)]
#[command(name = "worldline-casimir")]
#[command(about = "Worldline Monte Carlo Casimir energy computation")]
struct Args {
    /// Geometry type
    #[arg(long, default_value = "plates")]
    geometry: String,

    /// Plate separation or characteristic length scale (micrometers)
    #[arg(long, default_value_t = 1.0)]
    separation: f64,

    /// Number of Monte Carlo loops per proper-time sample
    #[arg(long, default_value_t = 10000)]
    n_loops: usize,

    /// Number of points per loop
    #[arg(long, default_value_t = 500)]
    n_loop_points: usize,

    /// Number of profile sample points along the line
    #[arg(long, default_value_t = 20)]
    n_profile: usize,

    /// Random seed
    #[arg(long, default_value_t = 42)]
    seed: u64,

    /// Output CSV file path (if omitted, prints to stdout)
    #[arg(long)]
    output: Option<PathBuf>,

    /// Print ASCII cross-section (for 2D geometries)
    #[arg(long)]
    ascii: bool,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let config = WorldlineCasimirConfig {
        n_loop_points: args.n_loop_points,
        n_loops: args.n_loops,
        t_min: 0.001 * args.separation * args.separation,
        t_max: 10.0 * args.separation * args.separation,
        n_t_points: 16,
        seed: args.seed,
    };

    let a = args.separation;

    let mut csv = String::new();

    match args.geometry.as_str() {
        "plates" => {
            let geom = ParallelPlates { separation: a };
            let exact = casimir_parallel_plates_exact(a);

            let _ = writeln!(csv, "position,energy,exact,relative_error");

            let start = [0.0, 0.0, 0.1 * a];
            let end = [0.0, 0.0, 0.9 * a];
            let profile = casimir_energy_profile(&geom, start, end, args.n_profile, &config);

            for (t, e) in &profile {
                let z = 0.1 * a + t * 0.8 * a;
                let rel_err = if exact.abs() > 1e-30 {
                    (e - exact) / exact
                } else {
                    0.0
                };
                let _ = writeln!(csv, "{z:.6e},{e:.6e},{exact:.6e},{rel_err:.4e}");
            }

            eprintln!("parallel plates: a={a}, exact E/A={exact:.6e}");
        }
        "sphere" => {
            let geom = SolidSphere {
                center: [0.0, 0.0, 0.0],
                radius: a,
            };
            let _ = writeln!(csv, "position,energy");

            let start = [a * 1.1, 0.0, 0.0];
            let end = [a * 3.0, 0.0, 0.0];
            let profile = casimir_energy_profile(&geom, start, end, args.n_profile, &config);

            for (t, e) in &profile {
                let r = a * 1.1 + t * a * 1.9;
                let _ = writeln!(csv, "{r:.6e},{e:.6e}");
            }
        }
        "cylinder" => {
            let geom = InfiniteCylinder { radius: a };
            let _ = writeln!(csv, "position,energy");

            let start = [0.0, 0.0, 0.0];
            let end = [a * 0.9, 0.0, 0.0];
            let profile = casimir_energy_profile(&geom, start, end, args.n_profile, &config);

            for (t, e) in &profile {
                let r = t * a * 0.9;
                let _ = writeln!(csv, "{r:.6e},{e:.6e}");
            }
        }
        "pillar" => {
            let geom = PillarInCavity {
                plate_separation: a,
                pillar_radius: a * 0.1,
            };
            let _ = writeln!(csv, "position,energy");

            let start = [0.0, 0.0, 0.1 * a];
            let end = [0.0, 0.0, 0.9 * a];
            let profile = casimir_energy_profile(&geom, start, end, args.n_profile, &config);

            for (t, e) in &profile {
                let z = 0.1 * a + t * 0.8 * a;
                let _ = writeln!(csv, "{z:.6e},{e:.6e}");
            }

            if args.ascii {
                let grid = energy_cross_section(
                    |y, z| {
                        let result = casimir_energy_at_point(&geom, [0.0, y, z], &config);
                        result.energy
                    },
                    a,
                    20,
                    40,
                );
                eprintln!("{}", grid.render());
            }
        }
        "sphere-cylinder" => {
            let geom = SphereInCylinder {
                sphere_radius: a * 0.3,
                cylinder_radius: a,
            };
            let _ = writeln!(csv, "position,energy");

            let start = [0.0, 0.0, 0.0];
            let end = [a * 0.9, 0.0, 0.0];
            let profile = casimir_energy_profile(&geom, start, end, args.n_profile, &config);

            for (t, e) in &profile {
                let r = t * a * 0.9;
                let _ = writeln!(csv, "{r:.6e},{e:.6e}");
            }
        }
        other => {
            return Err(format!(
                "unknown geometry: {other}. Use: plates, sphere, cylinder, pillar, sphere-cylinder"
            )
            .into());
        }
    }

    match args.output {
        Some(path) => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&path, &csv)?;
            eprintln!("wrote profile to {}", path.display());
        }
        None => {
            print!("{csv}");
        }
    }

    Ok(())
}
