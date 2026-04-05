//! Topological Sieve Basis Scan
//!
//! Systematically scans complex-structure choices and basis orientations to find
//! configurations where exactly 8 of 24 SU(5) generators become TopologicalNull.
//! If such a sieve exists, it structurally forbids proton-decaying X/Y leptoquarks.
//!
//! The scan covers:
//! 1. All 15 imaginary basis elements as candidate complex structures (cheap)
//! 2. Haar-distributed random orthogonal bases via Gram-Schmidt (expensive)

use crate::{
    cayley_dickson_structs::Sedenion,
    neutrino_sector::{GeneratorType, classify_generator},
    quantum_state::QuantumState,
    su_n_generators::construct_su5_generators_algebraic,
};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

/// Result of scanning one complex-structure choice on a given basis.
#[derive(Debug, Clone)]
pub struct SieveResult {
    /// Which imaginary basis element (1..15) was used as the complex structure.
    pub complex_structure_index: usize,
    /// How many of the 24 SU(5) generators are TopologicalNull.
    pub null_count: usize,
    /// Classification of each non-null (surviving) generator.
    pub surviving_types: Vec<GeneratorType>,
    /// Indices (0..23) of generators that are TopologicalNull.
    pub null_indices: Vec<usize>,
}

/// Decomposition of surviving generators into SM gauge groups.
#[derive(Debug, Clone)]
pub struct SmDecomposition {
    pub su3_count: usize,
    pub su2_count: usize,
    pub u1_count: usize,
    pub dark_count: usize,
    pub leptoquark_count: usize,
    /// True if surviving generators decompose as 8+3+1 SM + extras.
    pub matches_sm: bool,
}

/// Scan all 15 imaginary basis elements as candidate complex structures.
///
/// For the standard sedenion basis, constructs SU(5) generators with each
/// imaginary unit e_1..e_15 as the complex structure, and counts how many
/// become TopologicalNull.
pub fn scan_complex_structures(basis: &[Sedenion; 16]) -> Vec<SieveResult> {
    let mut results = Vec::with_capacity(15);

    for cs_idx in 1..16 {
        let complex_structure = basis[cs_idx];
        let su5_gens = construct_su5_generators_algebraic(basis, &complex_structure);

        let mut null_count = 0;
        let mut null_indices = Vec::new();
        let mut surviving_types = Vec::new();

        for (i, generator) in su5_gens.iter().enumerate() {
            if *generator == QuantumState::TopologicalNull {
                null_count += 1;
                null_indices.push(i);
            } else {
                surviving_types.push(classify_generator(i));
            }
        }

        results.push(SieveResult {
            complex_structure_index: cs_idx,
            null_count,
            surviving_types,
            null_indices,
        });
    }

    results
}

/// Generate a Haar-distributed random orthogonal 16x16 basis.
///
/// Uses the Mezzadri technique: fill a 16x16 matrix with i.i.d. N(0,1),
/// compute QR decomposition via Gram-Schmidt, adjust signs so R has positive
/// diagonal (ensuring uniform Haar measure on O(16)).
fn random_orthogonal_basis(rng: &mut ChaCha8Rng) -> [Sedenion; 16] {
    let n = 16;
    let normal = StandardNormal;

    // Generate random matrix
    let mut mat = vec![vec![0.0; n]; n];
    for row in &mut mat {
        for val in row.iter_mut() {
            *val = normal.sample(rng);
        }
    }

    // Gram-Schmidt orthogonalization
    let mut q = vec![vec![0.0; n]; n];
    for i in 0..n {
        let mut v = mat[i].clone();

        // Subtract projections onto previous vectors
        for q_j in q.iter().take(i) {
            let dot: f64 = v.iter().zip(q_j.iter()).map(|(a, b)| a * b).sum();
            for k in 0..n {
                v[k] -= dot * q_j[k];
            }
        }

        // Normalize
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for val in &mut v {
                *val /= norm;
            }
        }
        q[i] = v;
    }

    // Convert columns to Sedenion basis elements
    let mut basis = [Sedenion::default(); 16];
    for i in 0..n {
        let mut components = [0.0; 16];
        components[..n].copy_from_slice(&q[i][..n]);
        basis[i] = Sedenion::from_slice(&components);
    }

    basis
}

/// Scan random orthogonal bases for sieve configurations.
///
/// Generates `n_bases` Haar-distributed random orthogonal bases and tests each
/// imaginary element (indices 1..15) as the complex structure. Returns all
/// results with null_count > 0, sorted by null_count descending.
pub fn scan_random_bases(n_bases: usize, seed: u64) -> Vec<SieveResult> {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut hits = Vec::new();

    for _ in 0..n_bases {
        let basis = random_orthogonal_basis(&mut rng);
        let results = scan_complex_structures(&basis);
        for r in results {
            if r.null_count > 0 {
                hits.push(r);
            }
        }
    }

    hits.sort_by_key(|r| std::cmp::Reverse(r.null_count));
    hits
}

/// Verify that a sieve result with null_count=8 decomposes into SM gauge groups.
///
/// If 8 generators are null, the remaining 16 should decompose as:
///   SU(3): 8 generators
///   SU(2): 3 generators
///   U(1):  1 generator
///   Dark:  4 generators (or leptoquark, depending on classification)
pub fn verify_sm_decomposition(result: &SieveResult) -> SmDecomposition {
    let mut su3_count = 0;
    let mut su2_count = 0;
    let mut u1_count = 0;
    let mut dark_count = 0;
    let mut leptoquark_count = 0;

    for gen_type in &result.surviving_types {
        match gen_type {
            GeneratorType::SU3 => su3_count += 1,
            GeneratorType::SU2 => su2_count += 1,
            GeneratorType::U1 => u1_count += 1,
            GeneratorType::Dark => dark_count += 1,
            GeneratorType::Leptoquark => leptoquark_count += 1,
        }
    }

    let matches_sm = su3_count == 8 && su2_count == 3 && u1_count == 1;

    SmDecomposition {
        su3_count,
        su2_count,
        u1_count,
        dark_count,
        leptoquark_count,
        matches_sm,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn standard_basis() -> [Sedenion; 16] {
        let mut basis = [Sedenion::default(); 16];
        for i in 0..16 {
            let mut components = [0.0; 16];
            components[i] = 1.0;
            basis[i] = Sedenion::from_slice(&components);
        }
        basis
    }

    #[test]
    fn test_scan_covers_all_complex_structures() {
        let basis = standard_basis();
        let results = scan_complex_structures(&basis);
        assert_eq!(results.len(), 15, "must scan all 15 imaginary elements");

        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.complex_structure_index, i + 1);
        }
    }

    #[test]
    fn test_standard_basis_sieve_scan() {
        let basis = standard_basis();
        let results = scan_complex_structures(&basis);

        println!("--- SIEVE SCAN: STANDARD BASIS ---");
        let mut found_sieve = false;
        for r in &results {
            println!(
                "  e_{}: null_count={}, surviving={}",
                r.complex_structure_index,
                r.null_count,
                r.surviving_types.len()
            );
            if r.null_count == 8 {
                found_sieve = true;
                let decomp = verify_sm_decomposition(r);
                println!(
                    "    SIEVE FOUND! SU3={}, SU2={}, U1={}, Dark={}, LQ={}, matches_SM={}",
                    decomp.su3_count,
                    decomp.su2_count,
                    decomp.u1_count,
                    decomp.dark_count,
                    decomp.leptoquark_count,
                    decomp.matches_sm
                );
            }
        }

        if !found_sieve {
            println!("  No sieve (null_count=8) found for standard basis.");
            println!("  This is expected: e_15 complex structure gives null_count=0.");
        }
    }

    #[test]
    fn test_random_basis_sieve_scan() {
        // Scan 1000 random bases (15 complex structures each = 15000 checks)
        let hits = scan_random_bases(1000, 42);

        println!("--- SIEVE SCAN: 1000 RANDOM BASES ---");
        println!("  Total hits (null_count > 0): {}", hits.len());

        if hits.is_empty() {
            println!("  No topological nulls found in any random basis.");
            println!("  This suggests the sieve may require a specific algebraic structure.");
        } else {
            let max_null = hits[0].null_count;
            println!("  Maximum null_count: {}", max_null);

            for r in hits.iter().take(5) {
                println!(
                    "    e_{}: null_count={}, surviving_types={:?}",
                    r.complex_structure_index, r.null_count, r.surviving_types
                );

                if r.null_count == 8 {
                    let decomp = verify_sm_decomposition(r);
                    println!(
                        "    SIEVE FOUND! SU3={}, SU2={}, U1={}, Dark={}, matches_SM={}",
                        decomp.su3_count,
                        decomp.su2_count,
                        decomp.u1_count,
                        decomp.dark_count,
                        decomp.matches_sm
                    );
                }
            }
        }
    }
}
