//! gravastar-sweep: Parameter sweep for gravastar TOV solver.
//!
//! Usage: gravastar-sweep --gamma-min 1.0 --gamma-max 2.5 --output sweep.csv

use clap::Parser;
use cosmology_core::{solve_gravastar, GravastarConfig, PolytropicEos, AnisotropicParams};

#[derive(Parser)]
#[command(name = "gravastar-sweep")]
#[command(about = "Gravastar TOV parameter sweep")]
struct Args {
    /// Minimum polytropic index
    #[arg(long, default_value = "1.0")]
    gamma_min: f64,

    /// Maximum polytropic index
    #[arg(long, default_value = "2.5")]
    gamma_max: f64,

    /// Number of gamma values
    #[arg(long, default_value = "16")]
    n_gamma: usize,

    /// Minimum target mass (solar masses)
    #[arg(long, default_value = "5.0")]
    m_min: f64,

    /// Maximum target mass (solar masses)
    #[arg(long, default_value = "80.0")]
    m_max: f64,

    /// Number of mass values
    #[arg(long, default_value = "16")]
    n_mass: usize,

    /// Vacuum radius fraction
    #[arg(long, default_value = "0.3")]
    r_v_frac: f64,

    /// Shell outer radius fraction
    #[arg(long, default_value = "0.9")]
    r1_frac: f64,

    /// Vacuum energy density
    #[arg(long, default_value = "1e-3")]
    rho_v: f64,

    /// Shell density
    #[arg(long, default_value = "1e-2")]
    rho_shell: f64,

    /// Polytropic constant K
    #[arg(long, default_value = "1.0")]
    k_poly: f64,

    /// Output CSV file
    #[arg(short, long)]
    output: Option<String>,
}

fn main() {
    let args = Args::parse();

    eprintln!("Gravastar sweep: gamma in [{}, {}], M in [{}, {}]",
        args.gamma_min, args.gamma_max, args.m_min, args.m_max);

    let mut results = Vec::new();

    for i in 0..args.n_gamma {
        let gamma = args.gamma_min + (args.gamma_max - args.gamma_min) * (i as f64) / (args.n_gamma - 1) as f64;

        for j in 0..args.n_mass {
            let m_target = args.m_min + (args.m_max - args.m_min) * (j as f64) / (args.n_mass - 1) as f64;

            // Compute radii from mass (rough scaling: r ~ 2.95 * M for neutron stars)
            let r_s = 2.95 * m_target;  // Schwarzschild radius scale
            let _r_v = args.r_v_frac * r_s;
            let r1 = args.r1_frac * r_s;

            let config = GravastarConfig {
                r1,
                m_target,
                compactness_target: 0.7,
                eos: PolytropicEos::new(args.k_poly, gamma),
                aniso: AnisotropicParams::isotropic(),
                dr: 1e-4,
                p_floor: 1e-12,
            };

            match solve_gravastar(&config) {
                Some(sol) => {
                    results.push((gamma, m_target, sol.mass, sol.r2, sol.compactness, sol.is_stable));
                }
                None => {
                    results.push((gamma, m_target, f64::NAN, f64::NAN, f64::NAN, false));
                }
            }
        }
        eprint!(".");
    }
    eprintln!(" done");

    let n_stable = results.iter().filter(|r| r.5).count();
    let n_valid = results.iter().filter(|r| !r.2.is_nan()).count();
    eprintln!("Results: {}/{} valid, {}/{} stable", n_valid, results.len(), n_stable, results.len());

    if let Some(path) = args.output {
        let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");
        wtr.write_record(["gamma", "m_target", "m_final", "r_final", "compactness", "stable"]).unwrap();
        for (gamma, m_target, m_final, r_final, compactness, stable) in &results {
            wtr.write_record(&[
                gamma.to_string(),
                m_target.to_string(),
                m_final.to_string(),
                r_final.to_string(),
                compactness.to_string(),
                stable.to_string(),
            ]).unwrap();
        }
        wtr.flush().unwrap();
        println!("Wrote {} results to {}", results.len(), path);
    } else {
        println!("gamma,m_target,m_final,r_final,compactness,stable");
        for (gamma, m_target, m_final, r_final, compactness, stable) in &results {
            println!("{},{},{},{},{},{}", gamma, m_target, m_final, r_final, compactness, stable);
        }
    }
}
