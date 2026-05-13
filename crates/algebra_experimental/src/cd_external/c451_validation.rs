//! C-451 cross-validation at dim=128.
//!
//! `CrossValidation128Result` aggregates the XOR partner law and
//! parity-clique structure at the 128D ZD-adjacency graph, using
//! bucket-optimized adjacency. `verify_c451_128d` is the entry
//! point.
//!
//! Re-exported from `cd_external` via `pub use` so external paths
//! algebra_experimental::cd_external::{CrossValidation128Result,
//! verify_c451_128d} remain stable.

use std::collections::HashMap;

use super::{
    ParityCliqueResult, XorPartnerResult, cross_assessors, diagonal_zero_products_exact,
    verify_xor_partner_law,
};

/// Result of C-451 cross-validation at dim=128.
#[derive(Debug)]
pub struct CrossValidation128Result {
    /// XOR partner law verification at dim=128.
    pub xor_partner: XorPartnerResult,
    /// Parity-clique verification at dim=128.
    pub parity_clique: ParityCliqueResult,
    /// Number of ZD-adjacent pairs found (using bucketed computation).
    pub n_zd_edges: usize,
    /// Number of XOR buckets.
    pub n_buckets: usize,
    /// Total pairs within buckets checked.
    pub n_pairs_checked: usize,
    /// Overall summary.
    pub summary: String,
}

/// Verify C-451: 128D ZD adjacency cross-validation.
///
/// Computes:
/// 1. XOR partner law (partner(i) = i XOR 8 at dim=128, mask=128/16=8)
/// 2. Parity-clique structure via bucket-optimized adjacency
/// 3. Cross-validation statistics
pub fn verify_c451_128d() -> CrossValidation128Result {
    use algebra_analysis::zd_graphs::xor_key;

    let dim = 128;

    // 1. XOR partner law (fast)
    let xor_partner = verify_xor_partner_law(dim);

    // 2. Bucket-optimized adjacency + parity analysis
    let pairs = cross_assessors(dim);
    let n = pairs.len();

    // Build XOR buckets
    let mut buckets: HashMap<usize, Vec<usize>> = HashMap::new();
    for (idx, &(lo, hi)) in pairs.iter().enumerate() {
        let key = xor_key(lo, hi);
        buckets.entry(key).or_default().push(idx);
    }

    let n_buckets = buckets.len();
    let mut n_pairs_checked = 0usize;
    let mut n_zd_edges = 0usize;

    // Build adjacency within buckets
    let mut adj = vec![vec![false; n]; n];
    for bucket in buckets.values() {
        for bi in 0..bucket.len() {
            for bj in (bi + 1)..bucket.len() {
                n_pairs_checked += 1;
                let i = bucket[bi];
                let j = bucket[bj];
                let solutions = diagonal_zero_products_exact(dim, pairs[i], pairs[j]);
                if !solutions.is_empty() {
                    adj[i][j] = true;
                    adj[j][i] = true;
                    n_zd_edges += 1;
                }
            }
        }
    }

    // Parity analysis from adjacency
    let even_indices: Vec<usize> = (0..n).filter(|&i| pairs[i].0.is_multiple_of(2)).collect();
    let odd_indices: Vec<usize> = (0..n).filter(|&i| !pairs[i].0.is_multiple_of(2)).collect();
    let n_even = even_indices.len();
    let n_odd = odd_indices.len();

    let mut n_even_edges = 0usize;
    let mut n_odd_edges = 0usize;
    let mut n_cross_edges = 0usize;

    for i in 0..n {
        for j in (i + 1)..n {
            if adj[i][j] {
                let i_even = pairs[i].0.is_multiple_of(2);
                let j_even = pairs[j].0.is_multiple_of(2);
                if i_even && j_even {
                    n_even_edges += 1;
                } else if !i_even && !j_even {
                    n_odd_edges += 1;
                } else {
                    n_cross_edges += 1;
                }
            }
        }
    }

    let total_edges = n_even_edges + n_odd_edges + n_cross_edges;
    let expected_even = n_even * (n_even.saturating_sub(1)) / 2;
    let expected_odd = n_odd * (n_odd.saturating_sub(1)) / 2;
    let expected_clique_edges = expected_even + expected_odd;
    let is_parity_biclique = n_cross_edges == 0
        && n_even_edges == expected_even
        && n_odd_edges == expected_odd
        && total_edges == expected_clique_edges;

    let parity_clique = ParityCliqueResult {
        dim,
        n_vertices: n,
        n_edges: total_edges,
        n_even,
        n_odd,
        n_even_edges,
        n_odd_edges,
        n_cross_edges,
        expected_clique_edges,
        is_parity_biclique,
    };

    let summary = format!(
        "C-451 dim=128: {} cross-pairs, {} XOR buckets, {} pairs checked, {} ZD edges | \
         XOR partner: {} ({}/{} valid) | \
         Parity: even={} odd={} cross={} biclique={}",
        n,
        n_buckets,
        n_pairs_checked,
        n_zd_edges,
        if xor_partner.universal {
            "UNIVERSAL"
        } else {
            "PARTIAL"
        },
        xor_partner.n_valid,
        xor_partner.n_checked,
        n_even_edges,
        n_odd_edges,
        n_cross_edges,
        is_parity_biclique,
    );

    CrossValidation128Result {
        xor_partner,
        parity_clique,
        n_zd_edges,
        n_buckets,
        n_pairs_checked,
        summary,
    }
}
