//! Generalized cross-assessor enumeration and integer-exact diagonal
//! zero-product detection for arbitrary Cayley-Dickson dimensions.
//!
//! `CrossPair = (low, high)` generalizes the 42-Assessor structure of
//! dim=16 sedenions to higher CD dimensions:
//! - dim=16: low in 1..7, high in 8..15 (56 raw cross-assessors).
//! - dim=32: low in 1..15, high in 16..31 (more cross-assessors).
//! - dim=2^k: low in 1..2^(k-1), high in 2^(k-1)..2^k.
//!
//! The exclusion rules that produce the 42 primitive assessors apply
//! only at dim=16; the generalized API enumerates all raw pairs.
//! `diagonal_zero_products_exact` performs integer-exact zero detection
//! using `cd_basis_mul_sign_iter`, accumulating XOR-indexed coefficient
//! contributions in a fixed-size stack array (no HashMap allocation).

use cd_kernel::cayley_dickson::cd_basis_mul_sign_iter;

/// A generalized cross-assessor pair (low, high) for any power-of-2 dimension.
/// For dim=16: low in 1..7, high in 8..15.
/// For dim=32: low in 1..15, high in 16..31. Etc.
pub type CrossPair = (usize, usize);

/// Generate all cross-assessors for a given Cayley-Dickson dimension.
///
/// Cross-assessors are pairs (i, j) with i in [1, dim/2) and j in [dim/2, dim).
/// For dim=16, these are the 7*8 = 56 raw pairs from which the 42 primitive
/// assessors are drawn (after excluding behind-the-8-ball pairs).
/// For the generalized census, no such exclusion is applied.
pub fn cross_assessors(dim: usize) -> Vec<CrossPair> {
    assert!(
        dim >= 4 && dim.is_power_of_two(),
        "dim must be a power of two >= 4, got {dim}"
    );
    let half = dim / 2;
    let mut result = Vec::with_capacity((half - 1) * half);
    for i in 1..half {
        for j in half..dim {
            result.push((i, j));
        }
    }
    result
}

/// Integer-exact diagonal zero-product detection.
///
/// Given cross-assessor pairs a = (i, j) and b = (k, l), returns all
/// sign pairs (s, t) in {+1, -1}^2 such that
/// `(e_i + s*e_j) * (e_k + t*e_l) = 0`.
///
/// Uses `cd_basis_mul_sign_iter` (iterative) for integer-exact computation
/// with minimal function-call overhead. The coefficient accumulator uses a
/// fixed-size array indexed by the 4 XOR target slots instead of a HashMap.
pub fn diagonal_zero_products_exact(dim: usize, a: CrossPair, b: CrossPair) -> Vec<(i8, i8)> {
    let (i, j) = a;
    let (k, l) = b;

    let idx_ik = i ^ k;
    let idx_il = i ^ l;
    let idx_jk = j ^ k;
    let idx_jl = j ^ l;

    let s_ik = cd_basis_mul_sign_iter(dim, i, k);
    let s_il = cd_basis_mul_sign_iter(dim, i, l);
    let s_jk = cd_basis_mul_sign_iter(dim, j, k);
    let s_jl = cd_basis_mul_sign_iter(dim, j, l);

    // Fixed-size accumulator: at most 4 distinct XOR target indices.
    // Collect unique indices and accumulate coefficients in a stack array
    // instead of heap-allocating a HashMap per (s,t) combination.
    let indices = [idx_ik, idx_il, idx_jk, idx_jl];
    let mut solutions = Vec::new();
    for s in [1i32, -1] {
        for t in [1i32, -1] {
            let contributions = [s_ik, t * s_il, s * s_jk, s * t * s_jl];
            // Accumulate by matching index -- O(4*4) = O(16) comparisons,
            // no allocation. For each unique index among the 4 slots, sum
            // all contributions that target it.
            let all_zero = (0..4).all(|slot| {
                let target = indices[slot];
                // Only check each unique index once (first occurrence)
                if indices[..slot].contains(&target) {
                    return true; // Already checked via earlier slot
                }
                let sum: i32 = (0..4)
                    .filter(|&c| indices[c] == target)
                    .map(|c| contributions[c])
                    .sum();
                sum == 0
            });
            if all_zero {
                solutions.push((s as i8, t as i8));
            }
        }
    }
    solutions
}
