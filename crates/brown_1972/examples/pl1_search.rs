use brown_1972::pl1_emulator::Pl1Emulator;

fn main() {
    println!("===== BROWN 1972 PL/1 EMULATOR - ZERO DIVISOR SEARCH =====\n");

    // Test dimensions 4, 8, 16
    for dim in [4, 8, 16] {
        println!("Searching dimension {}", dim);
        let mut emulator = Pl1Emulator::new(dim);
        emulator.search_basis_pairs();

        println!("  Found {} pairs", emulator.pairs.len());
        if emulator.pairs.is_empty() {
            println!("  No zero divisor pairs found.");
        } else {
            for (idx, pair) in emulator.pairs.iter().enumerate() {
                println!("  [{}] {}", idx + 1, pair.display());
                if pair.verify() {
                    println!("       ✓ Verified: A*B = 0");
                } else {
                    println!("       ✗ Failed verification!");
                }
            }
        }
        println!();
    }

    // Detailed sedenion search with report
    println!("===== DETAILED SEDENION SEARCH =====\n");
    let mut sedenion = Pl1Emulator::new(16);
    println!("Starting systematic Major Theorem search...");
    sedenion.search_basis_pairs();
    println!("Search completed.\n");

    println!("Analyzing pair structure...");
    sedenion.analyze_pairs();
    println!("Analysis completed.\n");

    println!("{}", sedenion.generate_report());

    println!("\n===== DETAILED STATISTICS =====");
    println!("Raw pairs found:        {}", sedenion.stats.pair_count);
    println!("Unique (non-scalar):    {}", sedenion.stats.unique_pair_count);
    println!("Estimated elements:     {}", sedenion.stats.element_count);
    println!("Average norm:           {:.6}", sedenion.stats.avg_norm);
    println!("Has zero divisors:      {}", sedenion.stats.has_zero_divisors);

    // Calculate deduplication ratio
    if sedenion.stats.pair_count > 0 {
        let ratio = sedenion.stats.unique_pair_count as f64 / sedenion.stats.pair_count as f64;
        println!("Dedup ratio:            {:.2}% (unique/raw)", ratio * 100.0);
    }

    // Print detailed analysis
    sedenion.print_pair_analysis();
}
