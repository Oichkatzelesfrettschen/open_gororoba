//! kerr-shadow: Compute and output Kerr black hole shadow boundaries.
//!
//! Usage: kerr-shadow --spin 0.9 --output shadow.csv

use clap::Parser;
use gr_core::shadow_boundary;

#[derive(Parser)]
#[command(name = "kerr-shadow")]
#[command(about = "Compute Kerr black hole shadow boundary (Bardeen curve)")]
struct Args {
    /// Spin parameter (0 <= a < 1)
    #[arg(short, long)]
    spin: f64,

    /// Number of boundary points
    #[arg(short, long, default_value = "500")]
    n_points: usize,

    /// Observer inclination in degrees (90 = equatorial)
    #[arg(short, long, default_value = "90.0")]
    inclination: f64,

    /// Output CSV file
    #[arg(short, long)]
    output: Option<String>,

    /// Output as JSON instead of CSV
    #[arg(long)]
    json: bool,
}

fn main() {
    let args = Args::parse();

    if args.spin < 0.0 || args.spin >= 1.0 {
        eprintln!("Error: spin must be in [0, 1)");
        std::process::exit(1);
    }

    let theta_o = args.inclination.to_radians();
    eprintln!("Computing Kerr shadow: a = {}, theta_o = {:.1} deg, n = {}",
        args.spin, args.inclination, args.n_points);

    let (alpha, beta) = shadow_boundary(args.spin, args.n_points, theta_o);

    // Compute shadow properties
    let alpha_min = alpha.iter().cloned().fold(f64::INFINITY, f64::min);
    let alpha_max = alpha.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let beta_max = beta.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    eprintln!("Shadow extent: alpha in [{:.3}, {:.3}], beta_max = {:.3}",
        alpha_min, alpha_max, beta_max);

    if args.json {
        println!("{{");
        println!("  \"spin\": {},", args.spin);
        println!("  \"n_points\": {},", alpha.len());
        println!("  \"alpha_min\": {},", alpha_min);
        println!("  \"alpha_max\": {},", alpha_max);
        println!("  \"beta_max\": {},", beta_max);
        println!("  \"alpha\": {:?},", alpha);
        println!("  \"beta\": {:?}", beta);
        println!("}}");
    } else if let Some(path) = args.output {
        let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");
        wtr.write_record(["alpha", "beta"]).unwrap();
        for (a, b) in alpha.iter().zip(beta.iter()) {
            wtr.write_record(&[a.to_string(), b.to_string()]).unwrap();
        }
        wtr.flush().unwrap();
        println!("Wrote {} points to {}", alpha.len(), path);
    } else {
        println!("alpha,beta");
        for (a, b) in alpha.iter().zip(beta.iter()) {
            println!("{},{}", a, b);
        }
    }
}
