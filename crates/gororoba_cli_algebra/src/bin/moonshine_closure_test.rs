//! Moonshine Closure Test: representation closure, superpartner pairing,
//! and Cayley-Dickson tower embedding analysis.
//!
//! Exercises three algebra_experimental modules:
//! - representation_closure: greedy decomposition of j-coefficients into Monster irreps
//! - superpartner_pairing: boson/fermion parity and Witten index
//! - moonshine_embedding: Monster irrep divisibility into CD tower dimensions

use algebra_experimental::{
    moonshine::{
        J_COEFFICIENTS, J_COEFFICIENTS_VALID, MONSTER_REP_DIMENSIONS, MONSTER_REPS_VALID,
        j_constant_term_e8_relation, mckay_e8_observation, verify_moonshine_c1,
        verify_moonshine_c2,
    },
    moonshine_embedding::{
        CD_TOWER, CD_VIOLATION_COUNTS, analyze_embeddings, divisibility_census,
        violation_count_best,
    },
    representation_closure::{analyze_closure, verify_known_decompositions},
    superpartner_pairing::{RepParity, analyze_pairing, assign_parity, witten_index_partial},
};

fn main() {
    println!("=== Moonshine Closure Test ===\n");

    // ------ Phase 1: Moonshine verification ------
    println!("--- Phase 1: Moonshine Verification ---");
    let c1_ok = verify_moonshine_c1();
    let c2_ok = verify_moonshine_c2();
    println!(
        "  c_1 = 196884 = 1 + 196883: {}",
        if c1_ok { "OK" } else { "FAIL" }
    );
    println!(
        "  c_2 = 21493760 = 1 + 196883 + 21296876: {}",
        if c2_ok { "OK" } else { "FAIL" }
    );

    let (j744, e8_dim, factor) = j_constant_term_e8_relation();
    println!("  j-constant = {j744} = {factor} * {e8_dim}  (E8 relation)");
    println!("  McKay E8: {}", mckay_e8_observation());
    println!();

    // ------ Phase 2: Representation closure ------
    println!("--- Phase 2: Representation Closure ---");
    let valid_reps: Vec<u64> = MONSTER_REP_DIMENSIONS[..MONSTER_REPS_VALID]
        .iter()
        .copied()
        .filter(|&d| d > 0)
        .collect();
    println!("  Known Monster irreps ({MONSTER_REPS_VALID} valid):");
    for (i, &dim) in valid_reps.iter().enumerate() {
        let parity = assign_parity(dim);
        let p_char = match parity {
            RepParity::Bosonic => 'B',
            RepParity::Fermionic => 'F',
        };
        println!("    rho_{i}: dim = {dim:>12} [{p_char}]");
    }

    let greedy_ok = verify_known_decompositions();
    println!(
        "\n  Greedy decomposition verification: {}",
        if greedy_ok { "OK" } else { "FAIL" }
    );

    let closure = analyze_closure(J_COEFFICIENTS_VALID);
    println!(
        "  Closure analysis ({} coefficients):",
        closure.coefficients_analyzed
    );
    println!("    Used irreps: {:?}", closure.used_rep_indices);
    println!("    Appears closed: {}", closure.appears_closed);

    for (i, decomp) in closure.decompositions.iter().enumerate() {
        if decomp.is_empty() {
            continue;
        }
        let terms: Vec<String> = decomp
            .iter()
            .map(|&(idx, mult)| {
                if mult == 1 {
                    format!("rho_{idx}")
                } else {
                    format!("{mult}*rho_{idx}")
                }
            })
            .collect();
        let c_n = J_COEFFICIENTS[i];
        println!("    c_{} = {} = {}", i + 1, c_n, terms.join(" + "));
    }
    println!();

    // ------ Phase 3: Superpartner pairing ------
    println!("--- Phase 3: Superpartner Pairing ---");
    let pairing = analyze_pairing();
    println!("  Representations analyzed: {}", pairing.num_reps);
    println!(
        "  Bosonic: {}, Fermionic: {}",
        pairing.num_bosonic, pairing.num_fermionic
    );
    println!("  Parity balanced: {}", pairing.parity_balanced);
    println!("  Superpartner pairs found: {}", pairing.pairs.len());
    for pair in &pairing.pairs {
        println!(
            "    B(rho_{}, dim={}) <-> F(rho_{}, dim={})  ratio={:.4}",
            pair.boson_idx, pair.boson_dim, pair.fermion_idx, pair.fermion_dim, pair.ratio
        );
    }

    // Witten index
    println!("\n  Partial Witten indices:");
    for n in [2, 3, 5, J_COEFFICIENTS_VALID] {
        let w = witten_index_partial(n);
        println!("    W({n}) = {w}");
    }
    println!();

    // ------ Phase 4: CD tower embedding ------
    println!("--- Phase 4: CD Tower Embedding ---");
    println!("  CD tower dimensions:");
    for &(dim, name) in &CD_TOWER {
        println!("    {name:>12}: dim = {dim}");
    }
    println!("  Known AVT violation counts (exact enumeration):");
    for &(dim, count) in &CD_VIOLATION_COUNTS {
        println!("    dim {dim:>4}: {count:>10} violations (exact)");
    }

    // Violation counts for higher dimensions (exact where feasible, sampled otherwise)
    println!("  Higher-dim AVT violation counts:");
    for &dim in &[512, 1024, 2048, 4096] {
        let n_samples = if dim <= 1024 { 100_000 } else { 50_000 };
        let (count, is_exact) = violation_count_best(dim, n_samples, 42);
        let label = if is_exact { "exact" } else { "sampled" };
        println!("    dim {dim:>4}: {count:>14} violations ({label})");
    }

    let embedding = analyze_embeddings();
    println!("\n  Embedding analysis:");
    println!("    Total (rep, cd_dim) pairs: {}", embedding.results.len());
    println!("    Exact embeddings: {}", embedding.num_exact);
    println!("    Resonant CD dimensions: {:?}", embedding.resonant_dims);

    // Show exact embeddings
    let exact: Vec<_> = embedding.results.iter().filter(|r| r.exact).collect();
    if !exact.is_empty() {
        println!("\n  Exact embeddings (rep_dim divides cd_dim or vice versa):");
        for r in &exact {
            println!(
                "    rho_{} (dim={}) into {} (dim={}): {} copies, ratio={:.4}",
                r.rep_index, r.rep_dim, r.cd_name, r.cd_dim, r.copies, r.size_ratio
            );
        }
    }

    // Divisibility census
    let census = divisibility_census();
    println!("\n  j-coefficient divisibility by CD dimension:");
    for &(cd_dim, cd_name, div_count, total) in &census {
        let pct = if total > 0 {
            100.0 * div_count as f64 / total as f64
        } else {
            0.0
        };
        println!("    {cd_name:>12} (dim {cd_dim:>4}): {div_count}/{total} ({pct:.1}%)");
    }

    // ------ Summary ------
    println!("\n=== Summary ===");
    let all_ok = c1_ok && c2_ok && greedy_ok;
    if all_ok {
        println!("All verification checks PASSED.");
    } else {
        println!("Some verification checks FAILED.");
        std::process::exit(1);
    }
}
