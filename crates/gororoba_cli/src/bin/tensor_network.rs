//! tensor-network: Classical tensor network simulator.
//!
//! Simulates quantum circuits using classical tensor contraction.
//! Demonstrates entanglement entropy via SVD bipartition.
//!
//! Usage:
//!   tensor-network bell          # Prepare and analyze Bell state
//!   tensor-network ghz --n 5     # Prepare GHZ state with n qubits
//!   tensor-network evolve --n 4 --steps 50  # Random circuit evolution

use clap::{Parser, Subcommand};
use quantum_core::tensor_network_classical::{
    prepare_bell_state, prepare_ghz_state,
    simulate_random_circuit, bell_state_entropy, ghz_state_entropy,
};

#[derive(Parser)]
#[command(name = "tensor-network")]
#[command(about = "Classical tensor network simulator")]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Prepare and analyze Bell state (2 qubits)
    Bell {
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Prepare and analyze GHZ state (n qubits)
    Ghz {
        /// Number of qubits
        #[arg(long, default_value = "3")]
        n: usize,

        /// Bipartition point
        #[arg(long)]
        k: Option<usize>,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Random circuit evolution
    Evolve {
        /// Number of qubits
        #[arg(long, default_value = "4")]
        n: usize,

        /// Number of evolution steps
        #[arg(long, default_value = "50")]
        steps: usize,

        /// Random seed
        #[arg(long, default_value = "42")]
        seed: u64,

        /// Output CSV file
        #[arg(short, long)]
        output: Option<String>,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Analyze entropy scaling with system size
    Scaling {
        /// Minimum qubits
        #[arg(long, default_value = "2")]
        n_min: usize,

        /// Maximum qubits
        #[arg(long, default_value = "8")]
        n_max: usize,

        /// Output CSV file
        #[arg(short, long)]
        output: Option<String>,
    },
}

fn main() {
    let args = Args::parse();

    match args.command {
        Commands::Bell { json } => run_bell(json),
        Commands::Ghz { n, k, json } => run_ghz(n, k, json),
        Commands::Evolve { n, steps, seed, output, json } => run_evolve(n, steps, seed, output, json),
        Commands::Scaling { n_min, n_max, output } => run_scaling(n_min, n_max, output),
    }
}

fn run_bell(json: bool) {
    let state = prepare_bell_state();
    let entropy = bell_state_entropy();
    let ln2 = 2.0_f64.ln();

    if json {
        println!("{{");
        println!("  \"state\": \"Bell\",");
        println!("  \"n_qubits\": 2,");
        println!("  \"entropy\": {},", entropy);
        println!("  \"entropy_expected\": {},", ln2);
        println!("  \"probabilities\": {{");
        println!("    \"00\": {},", state.probability(0));
        println!("    \"01\": {},", state.probability(1));
        println!("    \"10\": {},", state.probability(2));
        println!("    \"11\": {}", state.probability(3));
        println!("  }}");
        println!("}}");
    } else {
        println!("Bell State: (|00> + |11>) / sqrt(2)");
        println!("=================================");
        println!();
        println!("Probabilities:");
        println!("  P(|00>) = {:.6}", state.probability(0));
        println!("  P(|01>) = {:.6}", state.probability(1));
        println!("  P(|10>) = {:.6}", state.probability(2));
        println!("  P(|11>) = {:.6}", state.probability(3));
        println!();
        println!("Entanglement Entropy:");
        println!("  S = {:.6}", entropy);
        println!("  ln(2) = {:.6}", ln2);
        println!("  Maximally entangled: {}", (entropy - ln2).abs() < 1e-6);
    }
}

fn run_ghz(n: usize, k: Option<usize>, json: bool) {
    let state = prepare_ghz_state(n);
    let k = k.unwrap_or(n / 2).max(1).min(n - 1);
    let entropy = ghz_state_entropy(n, k);
    let ln2 = 2.0_f64.ln();

    if json {
        println!("{{");
        println!("  \"state\": \"GHZ\",");
        println!("  \"n_qubits\": {},", n);
        println!("  \"bipartition_k\": {},", k);
        println!("  \"entropy\": {},", entropy);
        println!("  \"entropy_expected\": {},", ln2);
        println!("  \"p_all_zeros\": {},", state.probability(0));
        println!("  \"p_all_ones\": {}", state.probability((1 << n) - 1));
        println!("}}");
    } else {
        println!("GHZ State ({} qubits): (|0...0> + |1...1>) / sqrt(2)", n);
        println!("================================================");
        println!();
        println!("Probabilities:");
        let zeros: String = "0".repeat(n);
        let ones: String = "1".repeat(n);
        println!("  P(|{}>) = {:.6}", zeros, state.probability(0));
        println!("  P(|{}>) = {:.6}", ones, state.probability((1 << n) - 1));
        println!();
        println!("Entanglement Entropy (k={}):", k);
        println!("  S = {:.6}", entropy);
        println!("  ln(2) = {:.6}", ln2);
        println!("  GHZ property: any bipartition gives ln(2)");
    }
}

fn run_evolve(n: usize, steps: usize, seed: u64, output: Option<String>, json: bool) {
    let result = simulate_random_circuit(n, steps, seed);

    let avg_entropy: f64 = result.entropies.iter().sum::<f64>() / result.entropies.len() as f64;
    let max_entropy = result.entropies.iter().cloned().fold(0.0_f64, f64::max);
    let final_entropy = *result.entropies.last().unwrap_or(&0.0);

    if json {
        println!("{{");
        println!("  \"n_qubits\": {},", n);
        println!("  \"n_steps\": {},", steps);
        println!("  \"seed\": {},", seed);
        println!("  \"gate_count\": {},", result.gate_count);
        println!("  \"avg_entropy\": {},", avg_entropy);
        println!("  \"max_entropy\": {},", max_entropy);
        println!("  \"final_entropy\": {}", final_entropy);
        println!("}}");
    } else {
        println!("Random Circuit Evolution");
        println!("========================");
        println!("  N = {} qubits", n);
        println!("  Steps = {}", steps);
        println!("  Gates = {} (Hadamard + CNOT per step)", result.gate_count);
        println!("  Seed = {}", seed);
        println!();
        println!("Entropy Statistics:");
        println!("  Average: {:.6}", avg_entropy);
        println!("  Maximum: {:.6}", max_entropy);
        println!("  Final:   {:.6}", final_entropy);
        println!();

        // Show first/last few entropy values
        println!("Entropy trajectory (first 5, last 5):");
        for (i, e) in result.entropies.iter().enumerate().take(5) {
            println!("  Step {:3}: {:.6}", i + 1, e);
        }
        if steps > 10 {
            println!("  ...");
        }
        for (i, e) in result.entropies.iter().enumerate().skip(steps.saturating_sub(5)) {
            println!("  Step {:3}: {:.6}", i + 1, e);
        }
    }

    if let Some(path) = output {
        let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");
        wtr.write_record(["step", "entropy"]).unwrap();
        for (i, e) in result.entropies.iter().enumerate() {
            wtr.write_record(&[(i + 1).to_string(), e.to_string()]).unwrap();
        }
        wtr.flush().unwrap();
        eprintln!("Wrote {} entropy values to {}", result.entropies.len(), path);
    }
}

fn run_scaling(n_min: usize, n_max: usize, output: Option<String>) {
    println!("Entropy Scaling with System Size");
    println!("=================================");
    println!();

    let ln2 = 2.0_f64.ln();

    println!("n_qubits  GHZ_entropy  Bell_entropy  Random_max");
    println!("--------  -----------  ------------  ----------");

    let mut records = Vec::new();

    for n in n_min..=n_max {
        let ghz_s = if n >= 2 { ghz_state_entropy(n, n / 2) } else { 0.0 };
        let bell_s = if n >= 2 { bell_state_entropy() } else { 0.0 };
        let random_result = simulate_random_circuit(n, 20, 42);
        let random_max = random_result.entropies.iter().cloned().fold(0.0_f64, f64::max);

        println!("{:8}  {:11.6}  {:12.6}  {:10.6}", n, ghz_s, bell_s, random_max);
        records.push((n, ghz_s, bell_s, random_max));
    }

    println!();
    println!("Notes:");
    println!("  - GHZ entropy = ln(2) = {:.6} for any bipartition", ln2);
    println!("  - Bell entropy = ln(2) (maximal for 2 qubits)");
    println!("  - Random circuit can exceed GHZ entropy (volume law)");

    if let Some(path) = output {
        let mut wtr = csv::Writer::from_path(&path).expect("Failed to create CSV");
        wtr.write_record(["n_qubits", "ghz_entropy", "bell_entropy", "random_max"]).unwrap();
        for (n, g, b, r) in &records {
            wtr.write_record(&[
                n.to_string(),
                g.to_string(),
                b.to_string(),
                r.to_string(),
            ]).unwrap();
        }
        wtr.flush().unwrap();
        eprintln!("Wrote {} records to {}", records.len(), path);
    }
}
