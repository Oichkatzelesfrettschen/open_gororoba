//! CLI binary for the Prefix-Chain Theorem (T-010, T-011, C-453).
//!
//! Verifies the full theorem from first principles using predicate-based
//! lattice enumeration (no CSV data files required).
//!
//! Usage:
//!   cargo run --release --bin prefix-chain-theorem
//!   cargo run --release --bin prefix-chain-theorem -- --skeleton
//!   cargo run --release --bin prefix-chain-theorem -- --forbidden

use algebra_core::analysis::prefix_chain_theorem::{
    OctonionSkeletonMap, count_forbidden_by_family, format_theorem_summary,
    verify_prefix_chain_theorem,
};

fn main() {
    let show_skeleton = std::env::args().any(|a| a == "--skeleton");
    let show_forbidden = std::env::args().any(|a| a == "--forbidden");

    println!("Verifying Prefix-Chain Theorem (C-453)...");
    println!("(enumerating 3^8 = 6561 candidate vectors per level)\n");

    let result = verify_prefix_chain_theorem();
    println!("{}", format_theorem_summary(&result));

    if result.theorem_verified() {
        println!("OVERALL: THEOREM VERIFIED");
    } else {
        println!("OVERALL: THEOREM VERIFICATION FAILED");
        std::process::exit(1);
    }

    if show_forbidden {
        println!("\n--- Forbidden Prefix Family Breakdown ---");
        let families = count_forbidden_by_family();
        for (family, count) in &families {
            println!("  {:?}: {} points", family, count);
        }
        let total: usize = families.iter().map(|(_, c)| c).sum();
        println!("  Total forbidden: {}", total);
    }

    if show_skeleton {
        println!("\n--- Octonion Skeleton Map (T-011) ---");
        let skeleton = OctonionSkeletonMap::canonical();

        println!("\nHurwitz dimensions: {:?}", skeleton.hurwitz_dimensions);
        println!("Real parity: {}", skeleton.real_parity_constraint);

        println!("\nImaginary octonion unit directions:");
        for (i, dir) in skeleton.imaginary_directions.iter().enumerate() {
            let in_sbase = algebra_core::analysis::codebook::is_in_base_universe(dir);
            println!("  e_{}: {:?}  [in S_base: {}]", i + 1, dir, in_sbase);
        }

        println!("\nFano plane triples (e_i * e_j = e_k):");
        for (i, j, k) in &skeleton.fano_triples {
            println!("  e_{} * e_{} = e_{}", i, j, k);
        }

        println!("\nFano lattice consistency:");
        let checks = skeleton.verify_fano_lattice_consistency();
        for (i, j, k, ok) in &checks {
            println!(
                "  ({},{},{}) -> {}",
                i,
                j,
                k,
                if *ok { "PASS" } else { "FAIL" }
            );
        }

        println!(
            "\nAll directions in S_base: {}",
            if skeleton.verify_directions_in_sbase() {
                "PASS"
            } else {
                "FAIL"
            }
        );
    }
}
