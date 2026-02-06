//! zd-search: Find zero-divisor pairs in Cayley-Dickson algebras.
//!
//! Usage: zd-search --dim 16 --output zd_pairs.csv

use clap::Parser;
use algebra_core::{find_zero_divisors, find_box_kites, analyze_box_kite_symmetry};

#[derive(Parser)]
#[command(name = "zd-search")]
#[command(about = "Search for zero-divisor pairs in Cayley-Dickson algebras")]
struct Args {
    /// Algebra dimension (must be power of 2, >= 16)
    #[arg(short, long, default_value = "16")]
    dim: usize,

    /// Absolute tolerance for zero detection
    #[arg(short, long, default_value = "1e-10")]
    atol: f64,

    /// Output CSV file
    #[arg(short, long)]
    output: Option<String>,

    /// Also analyze box-kite structure
    #[arg(long)]
    box_kites: bool,

    /// Output as JSON instead of CSV
    #[arg(long)]
    json: bool,
}

fn main() {
    let args = Args::parse();

    if !args.dim.is_power_of_two() || args.dim < 16 {
        eprintln!("Error: dimension must be power of 2 and >= 16");
        std::process::exit(1);
    }

    eprintln!("Searching for zero-divisors in dimension {}...", args.dim);
    let zds = find_zero_divisors(args.dim, args.atol);
    eprintln!("Found {} zero-divisor pairs", zds.len());

    // Statistics
    let min_norm = zds.iter().map(|z| z.4).fold(f64::INFINITY, f64::min);
    let max_norm = zds.iter().map(|z| z.4).fold(0.0, f64::max);
    let exact_count = zds.iter().filter(|z| z.4 < 1e-14).count();

    if args.box_kites {
        eprintln!("Analyzing box-kite structure...");
        let bks = find_box_kites(args.dim, args.atol);
        let sym = analyze_box_kite_symmetry(args.dim, args.atol);
        eprintln!("Found {} box-kites, {} assessors", bks.len(), sym.n_assessors);
    }

    if args.json {
        println!("{{");
        println!("  \"dimension\": {},", args.dim);
        println!("  \"n_pairs\": {},", zds.len());
        println!("  \"exact_zeros\": {},", exact_count);
        println!("  \"min_norm\": {},", min_norm);
        println!("  \"max_norm\": {},", max_norm);
        println!("  \"pairs\": [");
        for (i, (a, b, c, d, norm)) in zds.iter().enumerate() {
            let comma = if i < zds.len() - 1 { "," } else { "" };
            println!("    [{}, {}, {}, {}, {}]{}", a, b, c, d, norm, comma);
        }
        println!("  ]");
        println!("}}");
    } else if let Some(path) = args.output {
        let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");
        wtr.write_record(["i", "j", "k", "l", "norm"]).unwrap();
        for (i, j, k, l, norm) in &zds {
            wtr.write_record(&[
                i.to_string(),
                j.to_string(),
                k.to_string(),
                l.to_string(),
                norm.to_string(),
            ]).unwrap();
        }
        wtr.flush().unwrap();
        println!("Wrote {} pairs to {}", zds.len(), path);
        println!("Exact zeros: {}, min_norm: {:.2e}, max_norm: {:.2e}",
            exact_count, min_norm, max_norm);
    } else {
        println!("i,j,k,l,norm");
        for (i, j, k, l, norm) in &zds {
            println!("{},{},{},{},{:.6e}", i, j, k, l, norm);
        }
    }
}
