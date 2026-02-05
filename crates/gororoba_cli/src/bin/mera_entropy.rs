//! mera-entropy: MERA tensor network entropy calculations.
//!
//! Usage: mera-entropy --l-max 64 --chi 4 --output entropy.csv

use clap::Parser;
use quantum_core::{mera_entropy_estimate, bekenstein_bound_bits};

#[derive(Parser)]
#[command(name = "mera-entropy")]
#[command(about = "MERA entropy scaling analysis")]
struct Args {
    /// Maximum subsystem size
    #[arg(long, default_value = "64")]
    l_max: usize,

    /// Bond dimension chi
    #[arg(short, long, default_value = "4")]
    chi: usize,

    /// Compute Bekenstein bound comparison
    #[arg(long)]
    bekenstein: bool,

    /// Absorber radius in nm (for Bekenstein)
    #[arg(long, default_value = "100.0")]
    radius: f64,

    /// Energy in eV (for Bekenstein)
    #[arg(long, default_value = "1.0")]
    energy: f64,

    /// Output CSV file
    #[arg(short, long)]
    output: Option<String>,

    /// Output as JSON
    #[arg(long)]
    json: bool,
}

fn main() {
    let args = Args::parse();

    eprintln!("MERA entropy scaling: L_max = {}, chi = {}", args.l_max, args.chi);

    // Compute entropy for powers of 2 up to l_max
    let mut results: Vec<(usize, f64)> = Vec::new();
    let mut l = 1;
    while l <= args.l_max {
        let s = mera_entropy_estimate(l, args.chi, 42);
        results.push((l, s));
        l *= 2;
    }

    // Fit log scaling: S ~ c * log(L)
    // Using simple linear regression on (log(L), S)
    let log_l: Vec<f64> = results.iter().map(|(l, _)| (*l as f64).ln()).collect();
    let s_vals: Vec<f64> = results.iter().map(|(_, s)| *s).collect();

    let n = log_l.len() as f64;
    let sum_x: f64 = log_l.iter().sum();
    let sum_y: f64 = s_vals.iter().sum();
    let sum_xx: f64 = log_l.iter().map(|x| x * x).sum();
    let sum_xy: f64 = log_l.iter().zip(s_vals.iter()).map(|(x, y)| x * y).sum();

    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;

    eprintln!("Log scaling fit: S = {:.4} * log(L) + {:.4}", slope, intercept);
    eprintln!("Central charge estimate: c ~ {:.2} (from c/3 * log(L))", 3.0 * slope);

    if args.bekenstein {
        let bek_bound = bekenstein_bound_bits(args.radius, args.energy);
        eprintln!("Bekenstein bound (R={} nm, E={} eV): S_max = {:.2e} bits",
            args.radius, args.energy, bek_bound);
    }

    if args.json {
        println!("{{");
        println!("  \"chi\": {},", args.chi);
        println!("  \"slope\": {},", slope);
        println!("  \"central_charge_estimate\": {},", 3.0 * slope);
        println!("  \"data\": [");
        for (i, (l, s)) in results.iter().enumerate() {
            let comma = if i < results.len() - 1 { "," } else { "" };
            println!("    {{\"L\": {}, \"S\": {}}}{}", l, s, comma);
        }
        println!("  ]");
        println!("}}");
    } else if let Some(path) = args.output {
        let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");
        wtr.write_record(["L", "S", "log_L", "S_fit"]).unwrap();
        for (l, s) in &results {
            let log_l = (*l as f64).ln();
            let s_fit = slope * log_l + intercept;
            wtr.write_record(&[
                l.to_string(),
                s.to_string(),
                log_l.to_string(),
                s_fit.to_string(),
            ]).unwrap();
        }
        wtr.flush().unwrap();
        println!("Wrote {} points to {}", results.len(), path);
        println!("Fit: S = {:.4} * log(L) + {:.4}", slope, intercept);
    } else {
        println!("L,S,log_L");
        for (l, s) in &results {
            println!("{},{},{:.4}", l, s, (*l as f64).ln());
        }
        println!("# Fit: S = {:.4} * log(L) + {:.4}", slope, intercept);
    }
}
