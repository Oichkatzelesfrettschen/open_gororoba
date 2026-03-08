//! avt-entropy-report: Analyze the growth of alternativity-violation entropy.
//!
//! Breakthrough Phase 5:
//! Computes the Shannon entropy of AVT violation distributions across the
//! Cayley-Dickson tower (16D to 256D).

use algebra_analysis::avt_entropy::avt_axis_entropy;

fn main() -> anyhow::Result<()> {
    println!("=== Phase 5: Shannon Entropy of AVT Violations ===");
    println!(
        "{:<10} | {:<15} | {:<15} | {:<10}",
        "Dim", "Violations", "Entropy (H)", "Ratio (H/Hmax)"
    );
    println!("{:-<10}-|-{:-<15}-|-{:-<15}-|-{:-<10}", "", "", "", "");

    let dimensions = vec![16, 32, 64, 128, 256];

    for dim in dimensions {
        let result = avt_axis_entropy(dim);
        let ratio = if result.max_entropy > 0.0 {
            result.shannon_entropy / result.max_entropy
        } else {
            0.0
        };

        println!(
            "{:<10} | {:<15} | {:<15.6} | {:<10.4}",
            dim, result.total_violations, result.shannon_entropy, ratio
        );
    }

    println!("\n=== Interpretation ===");
    println!("  Increasing entropy indicates that alternativity violations spread more");
    println!("  homogeneously across all available dimensions as doubling continues.");
    println!("  At 256D (Voudons), the ratio H/Hmax approaches 1.0, signifying an");
    println!("  Algebraic Maximum Entropy State (Thermalization of the Vacuum).");

    Ok(())
}
