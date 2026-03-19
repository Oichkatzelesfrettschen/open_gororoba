use anyhow::Result;
use clap::Parser;
use quantum_core::grover::{grover_search, GroverConfig, optimal_iterations};
use rand::prelude::*;
use std::collections::HashSet;

#[derive(Parser, Debug)]
#[command(author, version, about = "Quantum vs Classical Search Benchmark")]
struct Args {
    #[arg(short, long, default_value = "64,256,1024,4096")]
    sizes: String,

    #[arg(short, long, default_value = "0.01,0.05,0.1,0.25")]
    fractions: String,
    
    #[arg(long, default_value_t = 42)]
    seed: u64,
}

struct BenchmarkSummary {
    n_states: usize,
    n_marked: usize,
    classical_calls: usize,
    quantum_calls: usize,
    speedup_factor: f64,
    theoretical_speedup: f64,
    speedup_ratio: f64,
}

fn classical_random_search(n_states: usize, marked_set: &HashSet<usize>, seed: u64) -> usize {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut calls = 0;
    let mut visited = HashSet::new();

    while visited.len() < n_states {
        let idx = rng.gen_range(0..n_states);
        if !visited.contains(&idx) {
            visited.insert(idx);
            calls += 1;
            if marked_set.contains(&idx) {
                break;
            }
        }
    }
    calls
}

fn run_benchmark_suite(sizes: &[usize], fractions: &[f64], seed: u64) -> Vec<BenchmarkSummary> {
    let mut results = Vec::new();

    for &n_states in sizes {
        let n_qubits = (n_states as f64).log2().ceil() as u32;
        let actual_n_states = 1usize << n_qubits;
        
        let mut rng = StdRng::seed_from_u64(seed);

        for &frac in fractions {
            let n_marked = 1.max((actual_n_states as f64 * frac) as usize);

            let mut indices: Vec<usize> = (0..actual_n_states).collect();
            indices.shuffle(&mut rng);
            let marked_indices: Vec<usize> = indices.into_iter().take(n_marked).collect();
            let marked_set: HashSet<usize> = marked_indices.iter().copied().collect();

            let classical_calls = classical_random_search(actual_n_states, &marked_set, seed);

            let q_iters = optimal_iterations(actual_n_states, n_marked);
            let is_marked = |x: usize| marked_set.contains(&x);
            let q_result = grover_search(
                n_qubits as usize,
                is_marked,
                n_marked,
                GroverConfig { iterations: Some(q_iters), top_k: 1 }
            );

            let quantum_calls = q_result.iterations;
            let theoretical_speedup = (actual_n_states as f64 / n_marked as f64).sqrt();
            let speedup = classical_calls as f64 / quantum_calls.max(1) as f64;
            let speedup_ratio = speedup / theoretical_speedup;

            results.push(BenchmarkSummary {
                n_states: actual_n_states,
                n_marked,
                classical_calls,
                quantum_calls,
                speedup_factor: speedup,
                theoretical_speedup,
                speedup_ratio,
            });
        }
    }
    results
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    let sizes: Vec<usize> = args.sizes.split(',')
        .map(|s| s.trim().parse::<usize>().expect("Invalid size"))
        .collect();
        
    let fractions: Vec<f64> = args.fractions.split(',')
        .map(|s| s.trim().parse::<f64>().expect("Invalid fraction"))
        .collect();

    println!("Quantum vs Classical Search Benchmark");
    println!("============================================================");
    println!();

    let results = run_benchmark_suite(&sizes, &fractions, args.seed);

    println!("| N_states | N_marked | Classical | Quantum | Speedup | Theory | Ratio |");
    println!("|----------|----------|-----------|---------|---------|--------|-------|");

    for r in results {
        println!(
            "| {:8} | {:8} | {:9} | {:7} | {:7.2} | {:6.2} | {:5.2} |",
            r.n_states,
            r.n_marked,
            r.classical_calls,
            r.quantum_calls,
            r.speedup_factor,
            r.theoretical_speedup,
            r.speedup_ratio
        );
    }

    println!();
    println!("Speedup ratio near 1.0 indicates algorithm matches theory.");

    Ok(())
}
