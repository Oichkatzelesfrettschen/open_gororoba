//! Emanation Table survey: compute DMZ counts and regime structure for N=4..7.
//!
//! Usage: cargo run --release -p algebra_core --example et_survey

use algebra_core::{create_strutted_et, et_regimes};

fn main() {
    for n in 4..=7 {
        let g = 1usize << (n - 1);
        let dim = 1usize << n;
        let name = match n {
            4 => "Sedenions",
            5 => "Pathions",
            6 => "Chingons",
            7 => "Routons",
            _ => "?",
        };
        println!("=== N={} ({}, dim={}) ===", n, name, dim);
        for s in 1..g {
            let et = create_strutted_et(n, s);
            println!(
                "  S={:3}: K={:3}, possible={:5}, dmz={:5}, fill={:.3}",
                s, et.tone_row.k, et.total_possible, et.dmz_count,
                et.dmz_count as f64 / et.total_possible as f64
            );
        }
        let regimes = et_regimes(n);
        let mut sorted: Vec<_> = regimes.iter().collect();
        sorted.sort_by_key(|&(dmz, _)| *dmz);
        println!(
            "  {} regimes: {:?}",
            sorted.len(),
            sorted.iter().map(|(dmz, struts)| (*dmz, struts.len())).collect::<Vec<_>>()
        );
        println!();
    }
}
