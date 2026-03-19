use anyhow::{Context, Result};
use clap::Parser;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

use stats_core::mutual_information::{entropy_ksg_2d, ksg_mutual_information_2d};

const ALGEBRA_NAMES: [&str; 4] = ["CD-ZD", "G2", "J3O", "sl2"];
const RE_COLS: [usize; 4] = [1, 3, 5, 7];

#[derive(Parser, Debug)]
#[command(author, version, about = "Cross-algebra mutual information analysis of per-galaxy DFT phases.")]
struct Args {
    /// Path to cross_algebra_correlation.galaxies.csv
    input: PathBuf,

    /// Number of nearest neighbors for KSG estimator
    #[arg(short, long, default_value_t = 5)]
    k: usize,
}

fn load_galaxy_phases(csv_path: &PathBuf) -> Result<HashMap<String, Vec<f64>>> {
    let mut phases: HashMap<String, Vec<f64>> = ALGEBRA_NAMES
        .iter()
        .map(|&name| (name.to_string(), Vec::new()))
        .collect();

    let file = File::open(csv_path).context("Failed to open CSV file")?;
    let mut rdr = csv::ReaderBuilder::new().has_headers(true).from_reader(file);

    for result in rdr.records() {
        let record = result?;
        for (i, name) in ALGEBRA_NAMES.iter().enumerate() {
            let re: f64 = record[RE_COLS[i]].parse()?;
            let im: f64 = record[RE_COLS[i] + 1].parse()?;
            let phi = im.atan2(re);
            phases.get_mut(*name).unwrap().push(phi);
        }
    }

    Ok(phases)
}

fn embed_phase_on_circle(phi: &[f64]) -> Vec<[f64; 2]> {
    phi.iter().map(|&p| [p.cos(), p.sin()]).collect()
}

fn main() -> Result<()> {
    let args = Args::parse();

    eprintln!("=== H4: Cross-Algebra Mutual Information ===");
    eprintln!("Loading from {}...", args.input.display());

    let phases = load_galaxy_phases(&args.input)?;
    let n_gal = phases.values().next().unwrap().len();
    eprintln!("Loaded {} galaxies, k={} neighbors", n_gal, args.k);

    let mut embedded = HashMap::new();
    for (name, phi) in &phases {
        embedded.insert(name.clone(), embed_phase_on_circle(phi));
    }

    println!("algebra_a,algebra_b,mi_nats,h_a,h_b,nmi,pearson_r");

    let mut results = Vec::new();

    for (i, name_a) in ALGEBRA_NAMES.iter().enumerate() {
        for (j, name_b) in ALGEBRA_NAMES.iter().enumerate() {
            if j <= i {
                continue;
            }

            let x = &embedded[*name_a];
            let y = &embedded[*name_b];

            let mi = ksg_mutual_information_2d(x, y, args.k);
            let h_a = entropy_ksg_2d(x, args.k);
            let h_b = entropy_ksg_2d(y, args.k);

            let min_h = h_a.abs().min(h_b.abs());
            let nmi = if min_h > 1e-10 { mi / min_h } else { 0.0 };

            // Pearson correlation
            let phi_a = &phases[*name_a];
            let phi_b = &phases[*name_b];
            let mean_a = phi_a.iter().sum::<f64>() / n_gal as f64;
            let mean_b = phi_b.iter().sum::<f64>() / n_gal as f64;

            let mut num = 0.0;
            let mut den_a = 0.0;
            let mut den_b = 0.0;
            for (pa, pb) in phi_a.iter().zip(phi_b.iter()) {
                let da = pa - mean_a;
                let db = pb - mean_b;
                num += da * db;
                den_a += da * da;
                den_b += db * db;
            }
            let r = num / (den_a * den_b).sqrt();

            println!(
                "{},{},{:.8},{:.6},{:.6},{:.6},{:.6}",
                name_a, name_b, mi, h_a, h_b, nmi, r
            );

            results.push((name_a.to_string(), name_b.to_string(), mi, nmi, r));
            eprintln!(
                "  {} vs {}: MI={:.6} nats, NMI={:.6}, r={:.6}",
                name_a, name_b, mi, nmi, r
            );
        }
    }

    eprintln!();
    eprintln!("=== Verdict ===");

    let max_nmi = results
        .iter()
        .map(|(_, _, _, nmi, _)| *nmi)
        .fold(0.0, f64::max);
    let min_nmi = results
        .iter()
        .map(|(_, _, _, nmi, _)| *nmi)
        .fold(f64::INFINITY, f64::min);

    if max_nmi < 0.05 {
        eprintln!("  All NMI < 0.05: trivial independence.");
        eprintln!("  FALSIFICATION: no non-linear coupling detected -> REJECT.");
    } else if max_nmi > 0.15 {
        eprintln!(
            "  Max NMI = {:.4} > 0.15: unexpected cross-algebra coupling!",
            max_nmi
        );
        for (a, b, _, nmi, _) in &results {
            if *nmi > 0.15 {
                eprintln!("    {} vs {}: NMI = {:.4}", a, b, nmi);
            }
        }
    } else {
        eprintln!(
            "  NMI range [{:.4}, {:.4}]: weak coupling, inconclusive.",
            min_nmi, max_nmi
        );
    }

    for (name_a, name_b, _, nmi, r) in &results {
        let r2 = r * r;
        if *nmi > 0.05 && *nmi > 2.0 * r2 {
            eprintln!(
                "  Non-linear coupling: {} vs {}: NMI={:.4} >> r^2={:.4}",
                name_a, name_b, nmi, r2
            );
        }
    }

    Ok(())
}
