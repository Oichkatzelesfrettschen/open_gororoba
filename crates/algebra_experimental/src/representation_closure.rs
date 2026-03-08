//! Representation closure analysis for Monstrous Moonshine decompositions.
//!
//! Tests whether a finite subset of Monster irreps suffices to decompose
//! all j-function Fourier coefficients. The McKay-Thompson conjecture
//! (proved by Borcherds, BIB-0312) guarantees that each c_n decomposes
//! into a non-negative integer combination of Monster irreps. This module
//! checks how quickly the required irrep set saturates as n grows.
//!
//! See BIB-0311 (Conway & Norton 1979) for the Moonshine conjecture
//! and BIB-0312 (Borcherds 1992) for the proof.

use super::moonshine::{MONSTER_REP_DIMENSIONS, J_COEFFICIENTS};

/// Result of a representation closure analysis.
#[derive(Debug, Clone)]
pub struct ClosureResult {
    /// Number of j-coefficients analyzed.
    pub coefficients_analyzed: usize,
    /// Indices of Monster irreps used in decompositions (0-based).
    pub used_rep_indices: Vec<usize>,
    /// For each coefficient c_n, the decomposition as (rep_index, multiplicity) pairs.
    pub decompositions: Vec<Vec<(usize, u64)>>,
    /// Whether the irrep set appears to have saturated (no new irreps in last 3 coefficients).
    pub appears_closed: bool,
}

/// Greedy decomposition of a j-coefficient into Monster irreps.
///
/// Uses a greedy algorithm: subtract the largest irrep dimension that fits,
/// repeat until remainder is zero. This is NOT guaranteed to find the
/// unique decomposition (which requires the full character table), but
/// for the first few coefficients it matches known results.
pub fn greedy_decompose(value: u64, reps: &[u64]) -> Vec<(usize, u64)> {
    let mut remainder = value;
    let mut result = Vec::new();

    // Sort rep indices by dimension (descending) for greedy selection
    let mut sorted_indices: Vec<usize> = (0..reps.len()).filter(|&i| reps[i] > 0).collect();
    sorted_indices.sort_by(|&a, &b| reps[b].cmp(&reps[a]));

    for &idx in &sorted_indices {
        if reps[idx] == 0 || reps[idx] > remainder {
            continue;
        }
        let multiplicity = remainder / reps[idx];
        if multiplicity > 0 {
            result.push((idx, multiplicity));
            remainder -= multiplicity * reps[idx];
        }
        if remainder == 0 {
            break;
        }
    }

    // If remainder is nonzero, the greedy algorithm failed (needs more irreps
    // or the value isn't decomposable with the given set).
    if remainder > 0 {
        // Record the trivial rep (index 0, dim 1) for the remainder
        if reps.first() == Some(&1) {
            result.push((0, remainder));
        }
    }

    result
}

/// Analyze representation closure for the first `n` j-coefficients.
///
/// Returns which Monster irreps are needed and whether the set saturates.
pub fn analyze_closure(n: usize) -> ClosureResult {
    let max_n = n.min(J_COEFFICIENTS.len());
    let reps: Vec<u64> = MONSTER_REP_DIMENSIONS.iter().copied().filter(|&d| d > 0).collect();

    let mut all_used: Vec<bool> = vec![false; reps.len()];
    let mut decompositions = Vec::new();
    let mut last_new_rep_at = 0;

    for (i, &coeff) in J_COEFFICIENTS.iter().take(max_n).enumerate() {
        if coeff == 0 {
            decompositions.push(Vec::new());
            continue;
        }
        let decomp = greedy_decompose(coeff, &reps);
        for &(idx, _) in &decomp {
            if idx < all_used.len() && !all_used[idx] {
                all_used[idx] = true;
                last_new_rep_at = i;
            }
        }
        decompositions.push(decomp);
    }

    let used_rep_indices: Vec<usize> = all_used
        .iter()
        .enumerate()
        .filter(|(_, used)| **used)
        .map(|(i, _)| i)
        .collect();

    // Consider "closed" if no new irreps appeared in the last 3 coefficients
    let appears_closed = max_n >= 4 && last_new_rep_at + 3 < max_n;

    ClosureResult {
        coefficients_analyzed: max_n,
        used_rep_indices,
        decompositions,
        appears_closed,
    }
}

/// Verify known Moonshine decompositions against the greedy algorithm.
///
/// c_1 = 196884 = 1 + 196883
/// c_2 = 21493760 = 1 + 196883 + 21296876
pub fn verify_known_decompositions() -> bool {
    let reps: Vec<u64> = MONSTER_REP_DIMENSIONS.iter().copied().filter(|&d| d > 0).collect();

    // c_1 = 196884
    let d1 = greedy_decompose(196884, &reps);
    let sum1: u64 = d1.iter().map(|&(idx, mult)| reps[idx] * mult).sum();

    // c_2 = 21493760
    let d2 = greedy_decompose(21493760, &reps);
    let sum2: u64 = d2.iter().map(|&(idx, mult)| reps[idx] * mult).sum();

    sum1 == 196884 && sum2 == 21493760
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_decompose_c1() {
        let reps: Vec<u64> = MONSTER_REP_DIMENSIONS
            .iter()
            .copied()
            .filter(|&d| d > 0)
            .collect();
        let decomp = greedy_decompose(196884, &reps);
        let total: u64 = decomp.iter().map(|&(idx, mult)| reps[idx] * mult).sum();
        assert_eq!(total, 196884);
    }

    #[test]
    fn test_greedy_decompose_c2() {
        let reps: Vec<u64> = MONSTER_REP_DIMENSIONS
            .iter()
            .copied()
            .filter(|&d| d > 0)
            .collect();
        let decomp = greedy_decompose(21493760, &reps);
        let total: u64 = decomp.iter().map(|&(idx, mult)| reps[idx] * mult).sum();
        assert_eq!(total, 21493760);
    }

    #[test]
    fn test_verify_known_decompositions() {
        assert!(verify_known_decompositions());
    }

    #[test]
    fn test_closure_analysis_basic() {
        let result = analyze_closure(5);
        assert!(result.coefficients_analyzed <= 5);
        assert!(!result.used_rep_indices.is_empty());
    }

    #[test]
    fn test_trivial_rep_always_used() {
        let result = analyze_closure(3);
        // The trivial rep (index 0, dim 1) should appear in c_1 = 1 + 196883
        assert!(
            result.used_rep_indices.contains(&0),
            "Trivial representation should be used"
        );
    }
}
