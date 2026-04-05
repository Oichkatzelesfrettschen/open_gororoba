//! Cayley-Dickson signed graph imbalance and associativity analysis.
//!
//! Reimplements sedenion_imbalance.py, count_triples.py, and count_triples_full.py.
//! Uses cd_kernel::cayley_dickson::signs::cd_basis_mul_sign_iter for the sign function.

use cd_kernel::cayley_dickson::cd_basis_mul_sign_iter;

/// Result of the Harary-Zaslavsky imbalance computation on a CD signed graph.
#[derive(Debug, Clone)]
pub struct ImbalanceResult {
    pub dim: usize,
    pub total_edges: usize,
    pub min_flips: usize,
    pub density: f64,
}

/// Compute the Harary-Zaslavsky imbalance of the CD signed graph at dimension `dim`.
///
/// A signed graph is balanced iff there exists a potential x_i in {-1, +1} such that
/// sigma_{ij} = x_i * x_j for all edges. The imbalance is the minimum number of edge
/// sign flips needed to reach a balanced state, equivalent to MAX-CUT on the signed graph.
///
/// Exhaustively searches all 2^dim potentials. Only practical for dim <= 16.
pub fn cd_graph_imbalance(dim: usize) -> ImbalanceResult {
    assert!(
        dim.is_power_of_two() && dim >= 2,
        "dim must be a power of 2 >= 2"
    );
    assert!(dim <= 16, "exhaustive search only practical for dim <= 16");

    let edges: Vec<(usize, usize, i32)> = (0..dim)
        .flat_map(|i| ((i + 1)..dim).map(move |j| (i, j, cd_basis_mul_sign_iter(dim, i, j))))
        .collect();

    let total_edges = edges.len();
    let mut best_agreement = 0usize;

    for bits in 0..(1u32 << dim) {
        let mut agreement = 0usize;
        for &(i, j, sign) in &edges {
            let xi = if (bits >> i) & 1 == 1 { 1i32 } else { -1 };
            let xj = if (bits >> j) & 1 == 1 { 1i32 } else { -1 };
            if sign == xi * xj {
                agreement += 1;
            }
        }
        if agreement > best_agreement {
            best_agreement = agreement;
        }
    }

    let min_flips = total_edges - best_agreement;
    ImbalanceResult {
        dim,
        total_edges,
        min_flips,
        density: min_flips as f64 / total_edges as f64,
    }
}

/// Result of associativity analysis on CD triples.
#[derive(Debug, Clone)]
pub struct AssociativityResult {
    pub dim: usize,
    pub total_triples: usize,
    pub associative: usize,
    pub non_associative: usize,
}

/// Check whether (e_i * e_j) * e_k == e_i * (e_j * e_k) in CD algebra of given dimension.
pub fn is_associative_triple(dim: usize, i: usize, j: usize, k: usize) -> bool {
    let s1 = cd_basis_mul_sign_iter(dim, i, j);
    let s2 = cd_basis_mul_sign_iter(dim, j, k);
    let s3 = cd_basis_mul_sign_iter(dim, i ^ j, k);
    let s4 = cd_basis_mul_sign_iter(dim, i, j ^ k);
    s1 * s3 == s2 * s4
}

/// Count associative vs non-associative XOR-canonical triples (i < j, k = i^j > j).
///
/// Reimplements count_triples.py.
pub fn count_canonical_triples(dim: usize) -> AssociativityResult {
    assert!(dim.is_power_of_two() && dim >= 2);
    let mut associative = 0usize;
    let mut non_associative = 0usize;

    for i in 1..dim {
        for j in (i + 1)..dim {
            let k = i ^ j;
            if k > j {
                if is_associative_triple(dim, i, j, k) {
                    associative += 1;
                } else {
                    non_associative += 1;
                }
            }
        }
    }

    AssociativityResult {
        dim,
        total_triples: associative + non_associative,
        associative,
        non_associative,
    }
}

/// Count associative vs non-associative over ALL ordered triples of distinct non-zero units.
///
/// Reimplements count_triples_full.py.
pub fn count_all_ordered_triples(dim: usize) -> AssociativityResult {
    assert!(dim.is_power_of_two() && dim >= 2);
    let mut associative = 0usize;
    let mut non_associative = 0usize;

    for i in 1..dim {
        for j in 1..dim {
            if j == i {
                continue;
            }
            for k in 1..dim {
                if k == i || k == j {
                    continue;
                }
                if is_associative_triple(dim, i, j, k) {
                    associative += 1;
                } else {
                    non_associative += 1;
                }
            }
        }
    }

    AssociativityResult {
        dim,
        total_triples: associative + non_associative,
        associative,
        non_associative,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sedenion_imbalance() {
        let result = cd_graph_imbalance(16);
        assert_eq!(result.total_edges, 120);
        assert!(result.min_flips > 0);
        assert!(result.density > 0.0 && result.density < 1.0);
    }

    #[test]
    fn test_quaternion_fully_associative_canonical() {
        // Quaternions are associative (dim=4), so all triples should be associative.
        let result = count_canonical_triples(4);
        assert_eq!(result.non_associative, 0);
    }

    #[test]
    fn test_octonion_canonical_triples() {
        // Octonions (dim=8): XOR-canonical triples are a strict subset.
        let result = count_canonical_triples(8);
        assert!(result.total_triples > 0);
        // All ordered triples at dim=8 must show non-associativity.
        let full = count_all_ordered_triples(8);
        assert!(full.non_associative > 0);
    }

    #[test]
    fn test_sedenion_canonical_triples() {
        let result = count_canonical_triples(16);
        assert!(result.total_triples > 0);
    }

    #[test]
    fn test_sedenion_all_ordered_triples() {
        let result = count_all_ordered_triples(16);
        assert!(result.total_triples > 0);
        assert!(result.non_associative > 0);
    }

    #[test]
    fn test_is_associative_identity() {
        // (e_0 * e_i) * e_j == e_0 * (e_i * e_j) always, since e_0 is the identity.
        for dim in [4, 8, 16] {
            for i in 1..dim {
                for j in 1..dim {
                    if i != j {
                        assert!(is_associative_triple(dim, 0, i, j));
                    }
                }
            }
        }
    }
}
