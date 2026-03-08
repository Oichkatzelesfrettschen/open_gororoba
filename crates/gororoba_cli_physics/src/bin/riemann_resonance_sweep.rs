//! riemann-resonance-sweep: Compare ZD eigenvalue spacing statistics between
//! prime-power CD dimensions (2^p, p prime) and composite-power dimensions (2^q, q composite).
//!
//! Tests the Riemann Resonance conjecture: prime-indexed doublings produce
//! qualitatively different spectral repulsion than composite-indexed doublings.

use algebra_analysis::riemann_resonance::{DoublingClass, riemann_resonance_sweep};

fn main() {
    println!("=== Riemann Resonance Generator: Prime vs Composite CD Spectral Analysis ===");
    println!();

    let dims = [4, 8, 16, 32, 64, 128, 256];

    println!("Sweeping dims {:?}...", dims);
    println!("  Prime-power dims (2^p, p prime):     4=2^2, 8=2^3, 32=2^5, 128=2^7");
    println!("  Composite-power dims (2^q, q comp.):  16=2^4, 64=2^6, 256=2^8");
    println!("  (dim=256 builds 256x256 adjacency matrix, may take a moment)");
    println!();

    let comparison = riemann_resonance_sweep(&dims);

    // CSV header
    println!(
        "{:>5},{:>4},{:>12},{:>12},{:>10},{:>12},{:>12},{:>12},{:>18}",
        "dim",
        "exp",
        "class",
        "n_eigenvals",
        "brody_q",
        "ks_poisson",
        "ks_goe",
        "max_spacing",
        "verdict"
    );

    for r in &comparison.results {
        match &r.nnsd {
            Some(n) => {
                println!(
                    "{:>5},{:>4},{:>12},{:>12},{:>10.4},{:>12.4},{:>12.4},{:>12.4},{:>18}",
                    r.dim,
                    r.exponent,
                    r.doubling_class,
                    n.n_eigenvalues,
                    n.brody_q,
                    n.ks_poisson,
                    n.ks_goe,
                    r.spectral_gap.unwrap_or(0.0),
                    n.verdict,
                );
            }
            None => {
                println!(
                    "{:>5},{:>4},{:>12},{:>12},{:>10},{:>12},{:>12},{:>12},{:>18}",
                    r.dim,
                    r.exponent,
                    r.doubling_class,
                    "---",
                    "---",
                    "---",
                    "---",
                    "---",
                    "no_ZD_graph",
                );
            }
        }
    }

    println!();
    println!("=== Summary ===");

    if let Some(mean_p) = comparison.mean_q_prime {
        println!("  Mean Brody q (prime-power dims):     {:.4}", mean_p);
    } else {
        println!("  Mean Brody q (prime-power dims):     N/A (no non-degenerate results)");
    }

    if let Some(mean_c) = comparison.mean_q_composite {
        println!("  Mean Brody q (composite-power dims): {:.4}", mean_c);
    } else {
        println!("  Mean Brody q (composite-power dims): N/A (no non-degenerate results)");
    }

    match comparison.prime_higher_q {
        Some(true) => {
            println!("  Result: Prime-power dims show HIGHER mean Brody q (stronger repulsion).")
        }
        Some(false) => {
            println!("  Result: Composite-power dims show higher or equal mean Brody q.")
        }
        None => println!("  Result: Insufficient data for comparison."),
    }

    // Count verdicts by class
    let mut prime_verdicts = Vec::new();
    let mut composite_verdicts = Vec::new();
    for r in &comparison.results {
        if let Some(n) = &r.nnsd {
            match r.doubling_class {
                DoublingClass::PrimePower => prime_verdicts.push(&n.verdict),
                DoublingClass::CompositePower => composite_verdicts.push(&n.verdict),
                DoublingClass::Trivial => {}
            }
        }
    }

    println!();
    println!("  Prime-power verdicts:     {:?}", prime_verdicts);
    println!("  Composite-power verdicts: {:?}", composite_verdicts);
}
