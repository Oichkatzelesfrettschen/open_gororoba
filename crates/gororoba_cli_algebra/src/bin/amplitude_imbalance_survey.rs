use algebra_core::physics::amplitudes::enumerate_chambers;
use clap::Parser;
use std::collections::HashMap;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Number of gluons
    #[arg(long, default_value_t = 6)]
    n: usize,
}

fn main() {
    let args = Args::parse();
    let n = args.n;

    println!("Surveying Amplitude Chambers for n={} gluons...", n);
    let stats = enumerate_chambers(n);
    println!("Total potential chambers: {}", stats.len());

    let mut imbalance_map: HashMap<String, Vec<i8>> = HashMap::new();

    let mut nonzero_count = 0;
    let mut min_imbalance = 1.0;
    let mut max_imbalance = 0.0;

    for s in &stats {
        let f_str = format!("{:.4}", s.imbalance);
        imbalance_map.entry(f_str).or_default().push(s.amplitude);

        if s.amplitude != 0 {
            nonzero_count += 1;
        }
        if s.imbalance < min_imbalance {
            min_imbalance = s.imbalance;
        }
        if s.imbalance > max_imbalance {
            max_imbalance = s.imbalance;
        }
    }

    println!("\nImbalance Distribution:");
    let mut sorted_keys: Vec<_> = imbalance_map.keys().collect();
    sorted_keys.sort();

    for key in sorted_keys {
        let amps = &imbalance_map[key];
        let n_total = amps.len();
        let n_nonzero = amps.iter().filter(|&&a| a != 0).count();
        println!(
            " - Imbalance {}: {} chambers ({} nonzero)",
            key, n_total, n_nonzero
        );
    }

    println!("\nSummary:");
    println!(
        " - Non-zero Amplitudes: {} ({:.2}%)",
        nonzero_count,
        100.0 * nonzero_count as f64 / stats.len() as f64
    );
    println!(
        " - Imbalance Range: [{:.4}, {:.4}]",
        min_imbalance, max_imbalance
    );

    // Hypothesis Test
    let zero_frust_chambers = stats
        .iter()
        .filter(|s| s.imbalance < 1e-9)
        .collect::<Vec<_>>();
    let all_zero_frust_nonzero = zero_frust_chambers.iter().all(|s| s.amplitude != 0);

    println!("\nHypothesis Test: 'Nonzero region R1 corresponds to imbalance minima'");
    println!(
        " - Zero imbalance chambers: {}",
        zero_frust_chambers.len()
    );
    println!(
        " - All zero-imbalance chambers are non-zero amplitude: {}",
        all_zero_frust_nonzero
    );

    if all_zero_frust_nonzero && nonzero_count == zero_frust_chambers.len() {
        println!(" ==> HYPOTHESIS STRONGLY SUPPORTED: R1 is exactly the zero-imbalance locus.");
    } else if all_zero_frust_nonzero {
        println!(
            " ==> HYPOTHESIS PARTIALLY SUPPORTED: All zero-imbalance chambers are non-zero, but some non-zero amplitudes have higher imbalance?"
        );
        // Actually, if imbalance is fraction of zeros, and any zero factor makes amplitude zero,
        // then non-zero amplitude MUST have zero imbalance.
    }
}
