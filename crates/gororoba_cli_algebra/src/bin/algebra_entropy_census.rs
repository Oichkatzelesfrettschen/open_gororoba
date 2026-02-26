//! Reframed Thesis 2: Cayley-Dickson Associator Entropy Phase Census.
//!
//! Sweeps CD dimensions 2, 4, 8, 16, 32, 64, 128, 256 and computes the
//! Shannon entropy of the associator-norm distribution at each.
//!
//! Key prediction: a sharp entropy jump at dim=16 (octonion -> sedenion)
//! due to loss of the alternative property introducing zero divisors.
//!
//! PASS: H(16) > 0 with delta_H(8->16) > 10% of H_max.
//! FAIL: H is smooth across dim=16 or jump < 10%.

use algebra_core::analysis::entropy_census::{
    associator_entropy, phase_transition_analysis, zero_divisor_density,
};
use clap::Parser;
use std::fs;
use std::path::Path;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Reframed T2: CD associator entropy phase census"
)]
struct Args {
    /// Number of random triples per dimension
    #[arg(long, default_value_t = 10000)]
    samples: usize,
    /// Number of histogram bins
    #[arg(long, default_value_t = 100)]
    bins: usize,
    /// Maximum dimension (power of 2)
    #[arg(long, default_value_t = 256)]
    max_dim: usize,
    /// RNG seed
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Output directory
    #[arg(long, default_value = "data/entropy_census")]
    output_dir: String,
}

fn main() {
    let args = Args::parse();

    println!("CD Associator Entropy Phase Census");
    println!(
        "  samples={} bins={} max_dim={}",
        args.samples, args.bins, args.max_dim
    );
    println!();

    // Build dimension list: powers of 2 from 2 to max_dim
    let mut dims = Vec::new();
    let mut d = 2;
    while d <= args.max_dim {
        dims.push(d);
        d *= 2;
    }

    let results = phase_transition_analysis(&dims, args.samples, args.bins, args.seed);

    // Find H_max for normalization
    let h_max = results
        .iter()
        .map(|(_, h, _, _)| *h)
        .fold(0.0_f64, f64::max);

    println!(
        "  {:>6}  {:>10}  {:>10}  {:>10}  {:>10}",
        "dim", "H", "delta_H", "H/H_max", "ZD_dens"
    );
    println!(
        "  {:->6}  {:->10}  {:->10}  {:->10}  {:->10}",
        "", "", "", "", ""
    );

    for (dim, h, zd, delta_h) in &results {
        let h_ratio = if h_max > 0.0 { h / h_max } else { 0.0 };
        let zd_str = if zd.is_nan() {
            "N/A".to_string()
        } else {
            format!("{:.6}", zd)
        };
        println!(
            "  {:>6}  {:>10.4}  {:>10.4}  {:>10.4}  {:>10}",
            dim, h, delta_h, h_ratio, zd_str
        );
    }

    // Phase transition analysis
    println!();

    // Find the 8->16 transition
    let h_8 = results
        .iter()
        .find(|(d, _, _, _)| *d == 8)
        .map(|(_, h, _, _)| *h)
        .unwrap_or(0.0);
    let h_16 = results
        .iter()
        .find(|(d, _, _, _)| *d == 16)
        .map(|(_, h, _, _)| *h)
        .unwrap_or(0.0);
    let jump_magnitude = h_16 - h_8;
    let jump_fraction = if h_max > 0.0 {
        jump_magnitude / h_max
    } else {
        0.0
    };

    println!("  Phase transition analysis:");
    println!("    H(8)  = {:.4}", h_8);
    println!("    H(16) = {:.4}", h_16);
    println!(
        "    Jump   = {:.4} ({:.1}% of H_max)",
        jump_magnitude,
        jump_fraction * 100.0
    );

    // PASS criterion: H(16) > 0 and jump > 10% of H_max
    let pass = h_16 > 0.0 && jump_fraction > 0.10;
    let verdict = if pass { "PASS" } else { "FAIL" };

    println!();
    println!(
        "  Verdict: {} (jump {:.1}% of H_max, threshold 10%)",
        verdict,
        jump_fraction * 100.0
    );

    // Also compute individual entropy for the exact magnitude
    let h_exact_16 = associator_entropy(16, args.samples, args.bins, args.seed.wrapping_add(16));
    let h_ln_bins = (args.bins as f64).ln(); // Maximum possible entropy for uniform distribution
    let h_ratio_exact = h_exact_16 / h_ln_bins;
    println!(
        "  H(16) / ln(bins) = {:.4} ({:.1}% of maximum entropy)",
        h_ratio_exact,
        h_ratio_exact * 100.0
    );

    // ZD density at dim=16
    let zd16 = zero_divisor_density(16);
    println!("  ZD density(16) = {:.6}", zd16);

    // Write results
    let dir = Path::new(&args.output_dir);
    fs::create_dir_all(dir).expect("failed to create output dir");
    let toml_path = dir.join("results.toml");

    let mut toml = format!(
        "# CD Associator Entropy Phase Census\n# samples={} bins={}\n# verdict = {}\n\n",
        args.samples, args.bins, verdict
    );
    toml.push_str(&format!(
        "[phase_transition]\nh_8 = {:.6}\nh_16 = {:.6}\njump = {:.6}\njump_fraction = {:.6}\nh_max = {:.6}\npass = {}\n\n",
        h_8, h_16, jump_magnitude, jump_fraction, h_max, pass
    ));

    for (dim, h, zd, delta_h) in &results {
        toml.push_str(&format!(
            "[[dimension]]\ndim = {}\nentropy = {:.6}\ndelta_h = {:.6}\nzd_density = {}\n\n",
            dim,
            h,
            delta_h,
            if zd.is_nan() {
                "\"N/A\"".to_string()
            } else {
                format!("{:.6}", zd)
            }
        ));
    }

    fs::write(&toml_path, toml).expect("failed to write results.toml");
    println!("  Results written to {}", toml_path.display());

    if !pass {
        std::process::exit(1);
    }
}
