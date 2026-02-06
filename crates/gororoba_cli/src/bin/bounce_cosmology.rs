//! bounce-cosmology: Quantum bounce cosmology simulation and fitting.
//!
//! Usage: bounce-cosmology simulate --q-corr 0.001 --output bounce.csv
//!        bounce-cosmology fit --omega-m 0.315 --h0 67.4

use clap::{Parser, Subcommand};
use cosmology_core::{
    BounceParams, simulate_bounce,
    luminosity_distance, distance_modulus, cmb_shift_parameter,
    bao_sound_horizon, spectral_index_bounce,
    hubble_e_lcdm, hubble_e_bounce, Z_STAR,
};

#[derive(Parser)]
#[command(name = "bounce-cosmology")]
#[command(about = "Quantum bounce cosmology simulation and observational fitting")]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Simulate bounce cosmology evolution
    Simulate {
        /// Matter density parameter
        #[arg(long, default_value = "0.3")]
        omega_m: f64,

        /// Cosmological constant density parameter
        #[arg(long, default_value = "0.7")]
        omega_l: f64,

        /// Quantum correction strength
        #[arg(long, default_value = "0.001")]
        q_corr: f64,

        /// End time (dimensionless)
        #[arg(long, default_value = "10.0")]
        t_end: f64,

        /// Number of time steps
        #[arg(long, default_value = "1000")]
        steps: usize,

        /// Initial scale factor
        #[arg(long, default_value = "0.5")]
        a0: f64,

        /// Output CSV file
        #[arg(short, long)]
        output: Option<String>,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Compute observables (luminosity distance, CMB shift, etc.)
    Observables {
        /// Matter density parameter
        #[arg(long, default_value = "0.315")]
        omega_m: f64,

        /// Hubble constant (km/s/Mpc)
        #[arg(long, default_value = "67.4")]
        h0: f64,

        /// Quantum correction (0 for LCDM)
        #[arg(long, default_value = "0.0")]
        q_corr: f64,

        /// Redshifts to compute d_L for (comma-separated)
        #[arg(long, default_value = "0.1,0.5,1.0,2.0")]
        redshifts: String,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Compute Hubble parameter E(z) = H(z)/H_0
    Hubble {
        /// Redshift range end
        #[arg(long, default_value = "3.0")]
        z_max: f64,

        /// Number of points
        #[arg(long, default_value = "100")]
        n_points: usize,

        /// Matter density parameter
        #[arg(long, default_value = "0.315")]
        omega_m: f64,

        /// Quantum correction
        #[arg(long, default_value = "0.0")]
        q_corr: f64,

        /// Output CSV file
        #[arg(short, long)]
        output: Option<String>,
    },
}

fn main() {
    let args = Args::parse();

    match args.command {
        Commands::Simulate { omega_m, omega_l, q_corr, t_end, steps, a0, output, json } => {
            run_simulate(omega_m, omega_l, q_corr, t_end, steps, a0, output, json);
        }
        Commands::Observables { omega_m, h0, q_corr, redshifts, json } => {
            run_observables(omega_m, h0, q_corr, &redshifts, json);
        }
        Commands::Hubble { z_max, n_points, omega_m, q_corr, output } => {
            run_hubble(z_max, n_points, omega_m, q_corr, output);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn run_simulate(
    omega_m: f64,
    omega_l: f64,
    q_corr: f64,
    t_end: f64,
    steps: usize,
    a0: f64,
    output: Option<String>,
    json: bool,
) {
    eprintln!("Bounce cosmology: omega_m={}, omega_l={}, q_corr={}", omega_m, omega_l, q_corr);

    let params = BounceParams { omega_m, omega_l, q_corr };
    let result = simulate_bounce(&params, t_end, steps, a0);

    let a_min: f64 = result.a.iter().cloned().fold(f64::INFINITY, f64::min);
    let a_max: f64 = result.a.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let h_max: f64 = result.h.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    eprintln!("a range: [{:.4}, {:.4}], H_max: {:.4}", a_min, a_max, h_max);

    if json {
        println!("{{");
        println!("  \"omega_m\": {},", omega_m);
        println!("  \"omega_l\": {},", omega_l);
        println!("  \"q_corr\": {},", q_corr);
        println!("  \"a0\": {},", a0);
        println!("  \"a_min\": {},", a_min);
        println!("  \"a_max\": {},", a_max);
        println!("  \"h_max\": {},", h_max);
        println!("  \"steps\": {}", steps);
        println!("}}");
    } else if let Some(path) = output {
        let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");
        wtr.write_record(["time", "a", "H", "q"]).unwrap();
        for i in 0..result.time.len() {
            wtr.write_record(&[
                result.time[i].to_string(),
                result.a[i].to_string(),
                result.h[i].to_string(),
                result.q[i].to_string(),
            ]).unwrap();
        }
        wtr.flush().unwrap();
        println!("Wrote {} points to {}", result.time.len(), path);
    } else {
        println!("time,a,H,q");
        for i in (0..result.time.len()).step_by(result.time.len() / 20) {
            println!("{},{},{},{}", result.time[i], result.a[i], result.h[i], result.q[i]);
        }
    }
}

fn run_observables(omega_m: f64, h0: f64, q_corr: f64, redshifts: &str, json: bool) {
    let z_values: Vec<f64> = redshifts
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    eprintln!("Cosmological observables: omega_m={}, h0={}, q_corr={}", omega_m, h0, q_corr);

    // Compute observables
    let r_d = bao_sound_horizon(omega_m, h0);
    let r_shift = cmb_shift_parameter(omega_m, q_corr, Z_STAR);
    let n_s = spectral_index_bounce(q_corr, omega_m);

    if json {
        println!("{{");
        println!("  \"omega_m\": {},", omega_m);
        println!("  \"h0\": {},", h0);
        println!("  \"q_corr\": {},", q_corr);
        println!("  \"bao_sound_horizon_mpc\": {},", r_d);
        println!("  \"cmb_shift_parameter\": {},", r_shift);
        println!("  \"spectral_index_ns\": {},", n_s);
        println!("  \"distances\": [");
        for (i, &z) in z_values.iter().enumerate() {
            let d_l = luminosity_distance(z, omega_m, h0, q_corr);
            let mu = distance_modulus(z, omega_m, h0, q_corr);
            let comma = if i < z_values.len() - 1 { "," } else { "" };
            println!("    {{\"z\": {}, \"d_L_mpc\": {}, \"mu\": {}}}{}", z, d_l, mu, comma);
        }
        println!("  ]");
        println!("}}");
    } else {
        println!("BAO sound horizon: {:.2} Mpc", r_d);
        println!("CMB shift parameter R: {:.4}", r_shift);
        println!("Spectral index n_s: {:.4}", n_s);
        println!();
        println!("z,d_L[Mpc],mu[mag]");
        for &z in &z_values {
            let d_l = luminosity_distance(z, omega_m, h0, q_corr);
            let mu = distance_modulus(z, omega_m, h0, q_corr);
            println!("{},{:.2},{:.4}", z, d_l, mu);
        }
    }
}

fn run_hubble(z_max: f64, n_points: usize, omega_m: f64, q_corr: f64, output: Option<String>) {
    eprintln!("Hubble parameter E(z): omega_m={}, q_corr={}", omega_m, q_corr);

    let z_values: Vec<f64> = (0..n_points)
        .map(|i| z_max * i as f64 / (n_points - 1) as f64)
        .collect();

    let e_values: Vec<f64> = z_values
        .iter()
        .map(|&z| {
            if q_corr == 0.0 {
                hubble_e_lcdm(z, omega_m)
            } else {
                hubble_e_bounce(z, omega_m, q_corr)
            }
        })
        .collect();

    if let Some(path) = output {
        let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");
        wtr.write_record(["z", "E"]).unwrap();
        for i in 0..n_points {
            wtr.write_record(&[z_values[i].to_string(), e_values[i].to_string()]).unwrap();
        }
        wtr.flush().unwrap();
        println!("Wrote {} points to {}", n_points, path);
    } else {
        println!("z,E");
        for i in (0..n_points).step_by(n_points / 10) {
            println!("{:.3},{:.4}", z_values[i], e_values[i]);
        }
    }
}
