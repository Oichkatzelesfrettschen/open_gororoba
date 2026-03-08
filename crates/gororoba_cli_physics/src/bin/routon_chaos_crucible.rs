//! routon-chaos-crucible: Compute NNSD (nearest-neighbor spacing distribution)
//! for ZD graph Laplacian eigenvalue spectra at dims 16, 32, 64, 128.
//!
//! Tests the Poisson -> Wigner-Dyson (quantum chaos) transition as CD dimension
//! increases. Uses the proven quantum_chaos::nnsd_census pipeline.

use algebra_analysis::quantum_chaos::{SpacingVerdict, nnsd_census};

fn main() {
    println!("=== Routon 128D Wigner-Dyson Quantum Chaos Crucible ===");
    println!();

    // Dimensions to analyze: quaternion, octonion, sedenion, pathion, routon
    let dims = [16, 32, 64, 128];

    println!("Computing NNSD census for dims {:?}...", dims);
    println!("  (dim=128 builds 128x128 adjacency matrix, may take a few seconds)");
    println!();

    let results = nnsd_census(&dims);

    // CSV header
    println!(
        "{:>5},{:>12},{:>10},{:>12},{:>12},{:>18}",
        "dim", "n_eigenvalues", "brody_q", "ks_poisson", "ks_goe", "verdict"
    );

    let mut prev_q = 0.0;
    let mut monotone = true;

    for r in &results {
        println!(
            "{:>5},{:>12},{:>10.4},{:>12.4},{:>12.4},{:>18}",
            r.cd_dim, r.n_eigenvalues, r.brody_q, r.ks_poisson, r.ks_goe, r.verdict,
        );

        if r.brody_q < prev_q {
            monotone = false;
        }
        prev_q = r.brody_q;
    }

    println!();
    println!("=== Verdict ===");

    // Check if dim=128 shows at least Intermediate level repulsion
    if let Some(routon_result) = results.iter().find(|r| r.cd_dim == 128) {
        match routon_result.verdict {
            SpacingVerdict::GoeWigner => {
                println!(
                    "  dim=128: GOE Wigner-Dyson (q={:.3}). Full quantum chaos.",
                    routon_result.brody_q
                );
            }
            SpacingVerdict::Intermediate => {
                println!(
                    "  dim=128: Intermediate (q={:.3}). Partial level repulsion.",
                    routon_result.brody_q
                );
            }
            SpacingVerdict::Poisson => {
                println!(
                    "  dim=128: Poisson (q={:.3}). Integrable -- no chaos onset.",
                    routon_result.brody_q
                );
            }
            SpacingVerdict::DegenerateIntegrable => {
                println!("  dim=128: DegenerateIntegrable. Massive degeneracies dominate.");
            }
        }
    }

    if monotone {
        println!("  Brody q is monotonically increasing with dimension.");
    } else {
        println!("  Brody q is NOT monotonically increasing (non-trivial structure).");
    }
}
