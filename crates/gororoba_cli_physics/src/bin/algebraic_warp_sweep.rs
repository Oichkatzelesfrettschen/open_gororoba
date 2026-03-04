//! Algebraic warp sweep: Cayley-Dickson coupling to nacelle warp geometry.
//!
//! Non-paper-canonical binary that combines nacelle warp geometry with
//! Cayley-Dickson algebraic coupling. Compares standard GR energy budgets
//! with algebraic corrections from sedenion imbalance.
//!
//! Modes:
//! - `paper`: n=2,3,4 + Alcubierre (standard GR only, Phase 2)
//! - `algebraic`: n=1..16 with imbalance coupling
//! - `tower`: CD dimension tower (1,2,4,8,16) mapped to nacelle counts

use clap::Parser;
use gr_core::{
    adm_algebra_bridge::{
        algebraic_york_time_correction, cd_dimension_nacelle_map, sedenion_stress_energy,
    },
    warp_metric::{NacelleWarpParams, nacelle_energy_density, nacelle_york_time},
};
use std::{error::Error, fmt::Write as _, fs, path::PathBuf};

#[derive(Debug, Parser)]
#[command(name = "algebraic-warp-sweep")]
#[command(about = "Nacelle warp sweep with Cayley-Dickson algebraic coupling")]
struct Args {
    /// Sweep mode: paper, algebraic, or tower
    #[arg(long, default_value = "tower")]
    mode: String,

    /// Dimensionless coupling strength
    #[arg(long, default_value_t = 0.01)]
    coupling_alpha: f64,

    /// Viscosity coupling nu_0 for sedenion stress-energy
    #[arg(long, default_value_t = 1.0)]
    nu_0: f64,

    /// Exponential coupling beta for sedenion stress-energy
    #[arg(long, default_value_t = 1.0)]
    beta_coupling: f64,

    /// Warp velocity
    #[arg(long, default_value_t = 0.1)]
    velocity: f64,

    /// Bubble radius
    #[arg(long, default_value_t = 100.0)]
    radius: f64,

    /// Integration grid points per axis
    #[arg(long, default_value_t = 60)]
    n_points: usize,

    /// Imbalance density (uniform background)
    #[arg(long, default_value_t = 0.375)]
    imbalance: f64,

    /// Output CSV file path
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let configs: Vec<(String, usize, Option<usize>)> = match args.mode.as_str() {
        "paper" => vec![
            ("Alcubierre".into(), 0, None),
            ("n=2".into(), 2, None),
            ("n=3".into(), 3, None),
            ("n=4".into(), 4, None),
        ],
        "algebraic" => {
            let mut v = vec![("Alcubierre".into(), 0, None)];
            for n in 1..=16 {
                v.push((format!("n={n}"), n, None));
            }
            v
        }
        "tower" => {
            // Map CD dimensions to nacelle counts
            let cd_dims = [1, 2, 4, 8, 16];
            cd_dims
                .iter()
                .filter_map(|&dim| {
                    cd_dimension_nacelle_map(dim).map(|n| (format!("CD{dim}/n={n}"), n, Some(dim)))
                })
                .collect()
        }
        other => {
            return Err(format!("unknown mode: {other}. Use: paper, algebraic, tower").into());
        }
    };

    let mut csv = String::new();
    let _ = writeln!(
        csv,
        "label,n_nacelles,cd_dim,total_energy_gr,total_energy_algebraic,imbalance,york_time_correction"
    );

    let half = args.radius * 2.0;
    let n_grid = args.n_points;
    let dx = 2.0 * half / n_grid as f64;

    for (label, n, cd_dim) in &configs {
        let params = if *n == 0 {
            let sigma = 1.0 / (0.05 * args.radius);
            NacelleWarpParams::alcubierre(args.velocity, args.radius, sigma)
        } else {
            let mut p = NacelleWarpParams::white_2025(args.velocity, *n);
            p.r_bubble = args.radius;
            p
        };

        let mut total_gr = 0.0;
        let mut total_algebraic = 0.0;
        let mut york_correction_sum = 0.0;
        let mut count = 0;

        for iy in 0..n_grid {
            let y = -half + (iy as f64 + 0.5) * dx;
            for iz in 0..n_grid {
                let z = -half + (iz as f64 + 0.5) * dx;

                let rho_gr = nacelle_energy_density(0.0, y, z, &params);
                let theta_gr = nacelle_york_time(0.0, y, z, &params);

                total_gr += rho_gr * dx * dx;

                // Algebraic correction
                let k_trace_sq = theta_gr * theta_gr;
                let rho_alg = sedenion_stress_energy(
                    args.imbalance,
                    k_trace_sq,
                    args.nu_0,
                    args.beta_coupling,
                );
                let delta_theta =
                    algebraic_york_time_correction(args.imbalance, theta_gr, args.coupling_alpha);

                total_algebraic += (rho_gr + rho_alg) * dx * dx;
                york_correction_sum += delta_theta;
                count += 1;
            }
        }

        let avg_york_correction = if count > 0 {
            york_correction_sum / count as f64
        } else {
            0.0
        };

        let cd_dim_str = cd_dim.map_or("-".to_string(), |d| d.to_string());

        let _ = writeln!(
            csv,
            "{label},{n},{cd_dim_str},{total_gr:.6e},{total_algebraic:.6e},{:.6e},{avg_york_correction:.6e}",
            args.imbalance,
        );
    }

    match args.output {
        Some(path) => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&path, &csv)?;
            eprintln!(
                "wrote {} configurations to {}",
                configs.len(),
                path.display()
            );
        }
        None => {
            print!("{csv}");
        }
    }

    Ok(())
}
