//! harper-chern: Harper-Hofstadter model and Chern number calculator.
//!
//! Computes energy spectra and topological Chern numbers for the
//! Harper-Hofstadter model of electrons in a 2D lattice with magnetic flux.
//!
//! Usage:
//!   harper-chern spectrum --p 1 --q 4           # Energy spectrum for alpha=1/4
//!   harper-chern butterfly --q-max 10           # Hofstadter butterfly data
//!   harper-chern chern --p 1 --q 2 --n-grid 21  # Chern numbers via FHS method

use clap::{Parser, Subcommand};
use quantum_core::harper_chern::{
    fhs_chern_numbers, reduced_fractions, verify_chern_sum_zero, verify_diophantine,
};

#[derive(Parser)]
#[command(name = "harper-chern")]
#[command(about = "Harper-Hofstadter model and Chern number calculator")]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Compute energy spectrum at Gamma point for flux alpha = p/q
    Spectrum {
        /// Numerator of flux alpha = p/q
        #[arg(long)]
        p: u32,

        /// Denominator of flux alpha = p/q
        #[arg(long)]
        q: u32,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Generate Hofstadter butterfly spectrum data
    Butterfly {
        /// Maximum denominator q for reduced fractions
        #[arg(long, default_value = "10")]
        q_max: u32,

        /// Output CSV file
        #[arg(short, long)]
        output: Option<String>,

        /// Include Chern numbers (slower)
        #[arg(long)]
        chern: bool,

        /// Grid size for Chern calculation
        #[arg(long, default_value = "17")]
        n_grid: usize,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Compute Chern numbers via FHS method
    Chern {
        /// Numerator of flux alpha = p/q
        #[arg(long)]
        p: u32,

        /// Denominator of flux alpha = p/q
        #[arg(long)]
        q: u32,

        /// Brillouin zone grid size
        #[arg(long, default_value = "21")]
        n_grid: usize,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// List reduced fractions up to q_max
    Fractions {
        /// Maximum denominator
        #[arg(long, default_value = "10")]
        q_max: u32,
    },
}

fn main() {
    let args = Args::parse();

    match args.command {
        Commands::Spectrum { p, q, json } => run_spectrum(p, q, json),
        Commands::Butterfly {
            q_max,
            output,
            chern,
            n_grid,
            json,
        } => run_butterfly(q_max, output, chern, n_grid, json),
        Commands::Chern { p, q, n_grid, json } => run_chern(p, q, n_grid, json),
        Commands::Fractions { q_max } => run_fractions(q_max),
    }
}

fn run_spectrum(p: u32, q: u32, json: bool) {
    let alpha = p as f64 / q as f64;

    // Extract eigenvalues at Gamma via FHS (minimal grid since we only need Gamma point)
    let result = fhs_chern_numbers(p, q, 3);
    let energies = &result.energies_gamma;

    if json {
        println!("{{");
        println!("  \"p\": {},", p);
        println!("  \"q\": {},", q);
        println!("  \"alpha\": {},", alpha);
        println!("  \"n_bands\": {},", q);
        println!(
            "  \"energies\": [{}]",
            energies
                .iter()
                .map(|e| format!("{:.8}", e))
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!("}}");
    } else {
        println!("Harper-Hofstadter Spectrum");
        println!("==========================");
        println!();
        println!("Flux: alpha = {}/{} = {:.6}", p, q, alpha);
        println!("Bands: {}", q);
        println!();
        println!("Eigenvalues at Gamma point (kx=ky=0):");
        for (i, e) in energies.iter().enumerate() {
            println!("  E_{} = {:+.6}", i, e);
        }
        println!();

        // Band gaps
        println!("Band gaps:");
        for i in 1..energies.len() {
            let gap = energies[i] - energies[i - 1];
            println!("  Gap {} (E_{} - E_{}): {:.6}", i, i, i - 1, gap);
        }
    }
}

fn run_butterfly(
    q_max: u32,
    output: Option<String>,
    include_chern: bool,
    n_grid: usize,
    json: bool,
) {
    let fracs = reduced_fractions(q_max);

    if json {
        println!("{{");
        println!("  \"q_max\": {},", q_max);
        println!("  \"n_fractions\": {},", fracs.len());
        println!("  \"include_chern\": {},", include_chern);
        println!("  \"data\": [");

        for (idx, &(p, q)) in fracs.iter().enumerate() {
            let result = fhs_chern_numbers(p, q, n_grid);
            let alpha = p as f64 / q as f64;

            print!(
                "    {{\"p\": {}, \"q\": {}, \"alpha\": {:.8}, \"energies\": [{}]",
                p,
                q,
                alpha,
                result
                    .energies_gamma
                    .iter()
                    .map(|e| format!("{:.8}", e))
                    .collect::<Vec<_>>()
                    .join(", ")
            );

            if include_chern {
                print!(
                    ", \"chern\": [{}]",
                    result
                        .band_cherns
                        .iter()
                        .map(|c| format!("{}", c))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            if idx < fracs.len() - 1 {
                println!("}},");
            } else {
                println!("}}");
            }
        }
        println!("  ]");
        println!("}}");
    } else {
        println!("Hofstadter Butterfly Spectrum");
        println!("============================");
        println!();
        println!("Fractions: {} (q <= {})", fracs.len(), q_max);
        if include_chern {
            println!("Chern calculation: grid = {}", n_grid);
        }
        println!();

        if include_chern {
            println!(
                "{:>5}  {:>5}  {:>10}  {:>8}  Chern numbers",
                "p", "q", "alpha", "sum(C)"
            );
            println!(
                "{:-<5}  {:-<5}  {:-<10}  {:-<8}  {:-<30}",
                "", "", "", "", ""
            );
        } else {
            println!("{:>5}  {:>5}  {:>10}  Energies", "p", "q", "alpha");
            println!("{:-<5}  {:-<5}  {:-<10}  {:-<40}", "", "", "", "");
        }

        for &(p, q) in &fracs {
            let result = fhs_chern_numbers(p, q, n_grid);
            let alpha = p as f64 / q as f64;

            if include_chern {
                let sum: i32 = result.band_cherns.iter().sum();
                println!(
                    "{:5}  {:5}  {:10.6}  {:8}  {:?}",
                    p, q, alpha, sum, result.band_cherns
                );
            } else {
                let e_str: String = result
                    .energies_gamma
                    .iter()
                    .map(|e| format!("{:+.3}", e))
                    .collect::<Vec<_>>()
                    .join(", ");
                println!("{:5}  {:5}  {:10.6}  [{}]", p, q, alpha, e_str);
            }
        }
    }

    if let Some(path) = output {
        let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");

        if include_chern {
            wtr.write_record(["p", "q", "alpha", "band", "energy", "chern"])
                .unwrap();
            for &(p, q) in &fracs {
                let result = fhs_chern_numbers(p, q, n_grid);
                let alpha = p as f64 / q as f64;
                for (band, (&e, &c)) in result
                    .energies_gamma
                    .iter()
                    .zip(result.band_cherns.iter())
                    .enumerate()
                {
                    wtr.write_record(&[
                        p.to_string(),
                        q.to_string(),
                        alpha.to_string(),
                        band.to_string(),
                        e.to_string(),
                        c.to_string(),
                    ])
                    .unwrap();
                }
            }
        } else {
            wtr.write_record(["p", "q", "alpha", "band", "energy"])
                .unwrap();
            for &(p, q) in &fracs {
                let result = fhs_chern_numbers(p, q, 3);
                let alpha = p as f64 / q as f64;
                for (band, &e) in result.energies_gamma.iter().enumerate() {
                    wtr.write_record(&[
                        p.to_string(),
                        q.to_string(),
                        alpha.to_string(),
                        band.to_string(),
                        e.to_string(),
                    ])
                    .unwrap();
                }
            }
        }
        wtr.flush().unwrap();
        eprintln!("Wrote butterfly data to {}", path);
    }
}

fn run_chern(p: u32, q: u32, n_grid: usize, json: bool) {
    let alpha = p as f64 / q as f64;
    let result = fhs_chern_numbers(p, q, n_grid);

    let sum_zero = verify_chern_sum_zero(&result);
    let diophantine = verify_diophantine(&result);

    if json {
        println!("{{");
        println!("  \"p\": {},", p);
        println!("  \"q\": {},", q);
        println!("  \"alpha\": {},", alpha);
        println!("  \"n_grid\": {},", n_grid);
        println!(
            "  \"band_cherns\": [{}],",
            result
                .band_cherns
                .iter()
                .map(|c| format!("{}", c))
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!(
            "  \"gap_cherns\": [{}],",
            result
                .gap_cherns
                .iter()
                .map(|c| format!("{}", c))
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!("  \"sum_zero\": {},", sum_zero);
        println!(
            "  \"diophantine\": [{}],",
            diophantine
                .iter()
                .map(|b| if *b { "true" } else { "false" })
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!(
            "  \"energies\": [{}]",
            result
                .energies_gamma
                .iter()
                .map(|e| format!("{:.8}", e))
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!("}}");
    } else {
        println!("FHS Chern Number Calculation");
        println!("============================");
        println!();
        println!("Flux: alpha = {}/{} = {:.6}", p, q, alpha);
        println!("Grid: {} x {} k-points", n_grid, n_grid);
        println!();
        println!("Band Chern numbers:");
        for (i, c) in result.band_cherns.iter().enumerate() {
            println!("  C_{} = {:+}", i, c);
        }
        println!();
        println!("Gap Chern numbers (Hall conductance):");
        for (i, c) in result.gap_cherns.iter().enumerate() {
            println!("  sigma_{} = {} e^2/h", i + 1, c);
        }
        println!();
        println!("Verification:");
        println!(
            "  Sum of band Cherns = {} (should be 0): {}",
            result.band_cherns.iter().sum::<i32>(),
            if sum_zero { "PASS" } else { "FAIL" }
        );

        println!("  Diophantine equation: {:?}", diophantine);
        println!();
        println!("Energies at Gamma:");
        for (i, e) in result.energies_gamma.iter().enumerate() {
            println!("  E_{} = {:+.6}", i, e);
        }
        println!();
        println!("Note: Known issue - FHS may return 0 due to Berry phase");
        println!("      cancellation. See module docs for debugging status.");
    }
}

fn run_fractions(q_max: u32) {
    let fracs = reduced_fractions(q_max);

    println!("Reduced fractions p/q with q <= {}", q_max);
    println!("=====================================");
    println!();
    println!("Count: {} fractions", fracs.len());
    println!();

    for (i, &(p, q)) in fracs.iter().enumerate() {
        let alpha = p as f64 / q as f64;
        println!("{:3}. {:2}/{:2} = {:.6}", i + 1, p, q, alpha);
    }
}
