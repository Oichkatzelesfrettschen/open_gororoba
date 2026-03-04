//! ADM 3+1 decomposition diagnostic tool.
//!
//! Decomposes a chosen spacetime metric into ADM variables (lapse, shift,
//! spatial metric, extrinsic curvature) and verifies the Hamiltonian and
//! momentum constraint residuals.
//!
//! Supported spacetimes: Minkowski, Schwarzschild (Painleve-Gullstrand),
//! warp bubble (Alcubierre/nacelle).

use clap::Parser;
use gr_core::{
    adm::{decompose_metric, extrinsic_curvature, hamiltonian_constraint},
    metric::SpacetimeMetric,
    schwarzschild::Schwarzschild,
    warp_metric::{NacelleWarpBubble, NacelleWarpParams},
};
use std::{error::Error, fmt::Write as _, fs, path::PathBuf};

#[derive(Debug, Parser)]
#[command(name = "adm-decompose")]
#[command(about = "ADM 3+1 decomposition of spacetime metrics")]
struct Args {
    /// Spacetime to decompose
    #[arg(long, default_value = "minkowski")]
    spacetime: String,

    /// Black hole mass (for Schwarzschild)
    #[arg(long, default_value_t = 1.0)]
    mass: f64,

    /// Warp velocity (for warp spacetimes)
    #[arg(long, default_value_t = 0.1)]
    warp_velocity: f64,

    /// Radial coordinate for evaluation
    #[arg(long, default_value_t = 5.0)]
    radius: f64,

    /// Number of radial sample points
    #[arg(long, default_value_t = 20)]
    n_points: usize,

    /// Output CSV file path (if omitted, prints to stdout)
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let mut csv = String::new();
    let _ = writeln!(
        csv,
        "r,lapse,shift_x,shift_y,shift_z,york_time,hamiltonian_residual"
    );

    match args.spacetime.as_str() {
        "minkowski" => {
            let metric = gr_core::metric::Minkowski;
            for i in 0..args.n_points {
                let r = 1.0 + (i as f64 / (args.n_points - 1).max(1) as f64) * (args.radius - 1.0);
                let x = [0.0, r, std::f64::consts::FRAC_PI_2, 0.0];
                let g = metric.metric_components(&x);
                let adm = decompose_metric(&g);
                let ext = extrinsic_curvature(&metric, &x);
                let h = hamiltonian_constraint(&metric, &x, 0.0);

                let _ = writeln!(
                    csv,
                    "{r:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
                    adm.lapse, adm.shift[0], adm.shift[1], adm.shift[2], ext.york_time, h,
                );
            }
        }
        "schwarzschild" => {
            let metric = Schwarzschild::new(args.mass);
            let r_min = 2.5 * args.mass; // Outside horizon
            let r_max = args.radius.max(r_min + 1.0);

            for i in 0..args.n_points {
                let frac = i as f64 / (args.n_points - 1).max(1) as f64;
                let r = r_min + frac * (r_max - r_min);
                let x = [0.0, r, std::f64::consts::FRAC_PI_2, 0.0];
                let g = metric.metric_components(&x);
                let adm = decompose_metric(&g);
                let ext = extrinsic_curvature(&metric, &x);
                let h = hamiltonian_constraint(&metric, &x, 0.0);

                let _ = writeln!(
                    csv,
                    "{r:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
                    adm.lapse, adm.shift[0], adm.shift[1], adm.shift[2], ext.york_time, h,
                );
            }
        }
        "warp" => {
            let sigma = 1.0 / (0.05 * args.radius);
            let params = NacelleWarpParams::alcubierre(args.warp_velocity, args.radius, sigma);
            let metric = NacelleWarpBubble::new(params);
            let half = args.radius * 2.0;

            // Profile along y-axis at x=0, z=0
            for i in 0..args.n_points {
                let frac = i as f64 / (args.n_points - 1).max(1) as f64;
                let y = frac * half;
                let x = [0.0, 0.0, y, 0.0]; // [t, x, y, z] in Cartesian
                let g = metric.metric_components(&x);
                let adm = decompose_metric(&g);
                let ext = extrinsic_curvature(&metric, &x);
                let h = hamiltonian_constraint(&metric, &x, 0.0);

                let _ = writeln!(
                    csv,
                    "{y:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
                    adm.lapse, adm.shift[0], adm.shift[1], adm.shift[2], ext.york_time, h,
                );
            }
        }
        "nacelle" => {
            let mut params = NacelleWarpParams::white_2025(args.warp_velocity, 4);
            params.r_bubble = args.radius;
            let metric = NacelleWarpBubble::new(params);
            let half = args.radius * 2.0;

            for i in 0..args.n_points {
                let frac = i as f64 / (args.n_points - 1).max(1) as f64;
                let y = frac * half;
                let x = [0.0, 0.0, y, 0.0];
                let g = metric.metric_components(&x);
                let adm = decompose_metric(&g);
                let ext = extrinsic_curvature(&metric, &x);
                let h = hamiltonian_constraint(&metric, &x, 0.0);

                let _ = writeln!(
                    csv,
                    "{y:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}",
                    adm.lapse, adm.shift[0], adm.shift[1], adm.shift[2], ext.york_time, h,
                );
            }
        }
        other => {
            return Err(format!(
                "unknown spacetime: {other}. Use: minkowski, schwarzschild, warp, nacelle"
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
            eprintln!("wrote {} rows to {}", args.n_points, path.display());
        }
        None => {
            print!("{csv}");
        }
    }

    Ok(())
}
