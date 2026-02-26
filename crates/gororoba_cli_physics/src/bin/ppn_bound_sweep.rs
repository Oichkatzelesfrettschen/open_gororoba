use clap::Parser;
use gr_core::ppn_constraints::{
    check_all_ppn_constraints, kappa_eff_bd, nordtvedt_parameter_bd, ppn_gamma_bd,
};
use std::error::Error;
use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "ppn-bound-sweep")]
#[command(about = "Sweep Brans-Dicke omega and report PPN constraint satisfaction")]
struct Args {
    /// Minimum omega value for sweep
    #[arg(long, default_value_t = 10.0)]
    omega_min: f64,

    /// Maximum omega value for sweep
    #[arg(long, default_value_t = 1e7)]
    omega_max: f64,

    /// Number of logarithmically-spaced sample points
    #[arg(long, default_value_t = 100)]
    n_points: usize,

    /// Output CSV file path (if omitted, prints to stdout)
    #[arg(long)]
    output: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    assert!(
        args.omega_min > 0.0 && args.omega_max > args.omega_min,
        "omega_min must be positive and less than omega_max"
    );
    assert!(args.n_points >= 2, "n_points must be at least 2");

    let log_min = args.omega_min.ln();
    let log_max = args.omega_max.ln();
    let n = args.n_points;

    let mut csv = String::new();
    let _ = writeln!(
        csv,
        "omega,gamma,gamma_deviation,nordtvedt_eta,kappa_eff,cassini_pass,llr_pass,pulsar_pass,gw_speed_pass,n_satisfied"
    );

    for i in 0..n {
        let frac = i as f64 / (n - 1).max(1) as f64;
        let omega = (log_min + frac * (log_max - log_min)).exp();

        let gamma = ppn_gamma_bd(omega);
        let gamma_dev = (gamma - 1.0).abs();
        let eta_n = nordtvedt_parameter_bd(omega);
        let kappa = kappa_eff_bd(1.0, omega);
        let report = check_all_ppn_constraints(omega);

        let _ = writeln!(
            csv,
            "{omega:.6e},{gamma:.12e},{gamma_dev:.6e},{eta_n:.6e},{kappa:.6e},{},{},{},{},{}",
            report.cassini as u8,
            report.llr as u8,
            report.pulsar as u8,
            report.gw_speed as u8,
            report.n_satisfied,
        );
    }

    match args.output {
        Some(path) => {
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&path, &csv)?;
            eprintln!("wrote {} rows to {}", n, path.display());
        }
        None => {
            print!("{csv}");
        }
    }

    Ok(())
}
