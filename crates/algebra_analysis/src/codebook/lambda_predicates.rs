//! Lambda lattice filtration predicates: Lambda_256 <= Lambda_512 <= Lambda_1024 <= Lambda_2048 <= base.
//!
//! Implements the "predicate cut" filtration from the codebook analysis:
//! - Base: Trinary vectors with even sum and even weight.
//! - 2048D: Base minus 139 forbidden prefixes.
//! - 1024D: 2048D intersected with {l_0 = -1} minus 70 prefixes.
//! - 512D: 1024D minus 6 forbidden trie-cut regions.
//! - 256D: 512D minus 6 forbidden regions.
//!
//! Each `is_in_lambda_*` function returns true iff the input vector
//! satisfies that filtration tier's predicate. The `*_minus_k` variants
//! add the additional constraint `l_0 = -k` used by the octonion parity
//! constraint family. `verify_octonion_parity_constraints` and
//! `enumerate_lambda_4096` exercise the predicate family directly.

use super::enumerate_lattice_by_predicate;

/// A vector in the 8D integer lattice (typically {-1, 0, 1}).
pub type LatticeVector = [i8; 8];

/// Check if a vector is in the "Base Universe" (Trinary, Even Sum, Even Weight).
pub fn is_in_base_universe(v: &LatticeVector) -> bool {
    // 1. Trinary
    if v.iter().any(|&x| !(-1..=1).contains(&x)) {
        return false;
    }
    // 2. Even coordinate sum
    let sum: i32 = v.iter().map(|&x| x as i32).sum();
    if sum % 2 != 0 {
        return false;
    }
    // 3. Even Hamming weight (nonzero count)
    let weight = v.iter().filter(|&&x| x != 0).count();
    if weight % 2 != 0 {
        return false;
    }
    // 4. l_0 != +1 (from analysis of 2048D set)
    if v[0] == 1 {
        return false;
    }
    true
}

/// Check if a vector passes k of the 3 Lambda_2048 exclusion rules
/// (applied cumulatively in canonical order, starting from S_base).
///
/// `k=0` is S_base, `k=3` is Lambda_2048. Intermediate values
/// give sub-filtration levels for studying the S_base -> Lambda_2048 transition.
///
/// The 3 rules, in order (all affect l_0=0 subtree):
///   1. (0, 1, 1) prefix
///   2. (0, 1, 0, 1, 1) prefix
///   3. (0, 1, 0, 1, 0, 1) prefix
pub fn is_in_sbase_minus_k(v: &LatticeVector, k: usize) -> bool {
    assert!(k <= 3, "k must be in [0, 3]");
    if !is_in_base_universe(v) {
        return false;
    }
    // Rule 1: exclude (0, 1, 1) prefix
    if k >= 1 && v[0] == 0 && v[1] == 1 && v[2] == 1 {
        return false;
    }
    // Rule 2: exclude (0, 1, 0, 1, 1) prefix
    if k >= 2 && v[0] == 0 && v[1] == 1 && v[2] == 0 && v[3] == 1 && v[4] == 1 {
        return false;
    }
    // Rule 3: exclude (0, 1, 0, 1, 0, 1) prefix
    if k >= 3 && v[0] == 0 && v[1] == 1 && v[2] == 0 && v[3] == 1 && v[4] == 0 && v[5] == 1 {
        return false;
    }
    true
}

/// Check if a vector is in Lambda_2048 (Base minus 139 forbidden prefixes).
pub fn is_in_lambda_2048(v: &LatticeVector) -> bool {
    if !is_in_base_universe(v) {
        return false;
    }

    // Forbidden prefixes for 2048D
    // (l_0, l_1, l_2) = (0, 1, 1)
    if v[0] == 0 && v[1] == 1 && v[2] == 1 {
        return false;
    }
    // (l_0..l_4) = (0, 1, 0, 1, 1)
    if v[0] == 0 && v[1] == 1 && v[2] == 0 && v[3] == 1 && v[4] == 1 {
        return false;
    }
    // (l_0..l_5) = (0, 1, 0, 1, 0, 1)
    if v[0] == 0 && v[1] == 1 && v[2] == 0 && v[3] == 1 && v[4] == 0 && v[5] == 1 {
        return false;
    }

    true
}

/// Check if a vector is in Lambda_4096 (superset of Lambda_2048).
///
/// At dim=4096, the base universe constraints (trinary, even sum, even weight,
/// l_0 != +1) are sufficient -- no additional prefix exclusions are needed.
/// This means Lambda_4096 = S_base (the full base universe).
///
/// The 4 octonion parity constraints that define the base universe:
/// 1. Trinary: all coordinates in {-1, 0, 1}
/// 2. Even coordinate sum: sum(v) mod 2 = 0
/// 3. Even Hamming weight: |{i : v_i != 0}| mod 2 = 0
/// 4. l_0 constraint: `v[0] != +1`
///
/// These are structurally forced by the octonion algebra (C-589).
pub fn is_in_lambda_4096(v: &LatticeVector) -> bool {
    is_in_base_universe(v)
}

/// Enumerate all Lambda_4096 lattice vectors.
///
/// Returns all base universe vectors (superset of Lambda_2048).
pub fn enumerate_lambda_4096() -> Vec<LatticeVector> {
    enumerate_lattice_by_predicate(is_in_lambda_4096)
}

/// Verify the 4 octonion parity constraints hold for a given set of carriers.
///
/// Returns (n_total, n_trinary, n_even_sum, n_even_weight, n_l0_constraint, all_pass).
pub fn verify_octonion_parity_constraints(
    vectors: &[LatticeVector],
) -> (usize, usize, usize, usize, usize, bool) {
    let mut n_trinary = 0usize;
    let mut n_even_sum = 0usize;
    let mut n_even_weight = 0usize;
    let mut n_l0 = 0usize;

    for v in vectors {
        let trinary = v.iter().all(|&x| (-1..=1).contains(&x));
        if trinary {
            n_trinary += 1;
        }

        let sum: i32 = v.iter().map(|&x| x as i32).sum();
        if sum % 2 == 0 {
            n_even_sum += 1;
        }

        let weight = v.iter().filter(|&&x| x != 0).count();
        if weight % 2 == 0 {
            n_even_weight += 1;
        }

        if v[0] != 1 {
            n_l0 += 1;
        }
    }

    let n = vectors.len();
    let all_pass = n_trinary == n && n_even_sum == n && n_even_weight == n && n_l0 == n;
    (n, n_trinary, n_even_sum, n_even_weight, n_l0, all_pass)
}

/// Check if a vector is in Lambda_1024 (Lambda_2048 with l_0 = -1 minus 70 points).
pub fn is_in_lambda_1024(v: &LatticeVector) -> bool {
    if !is_in_lambda_2048(v) {
        return false;
    }

    // Slice condition
    if v[0] != -1 {
        return false;
    }

    // Additional exclusions for 1024D
    // (-1, 1, 1, 1)
    if v[1] == 1 && v[2] == 1 && v[3] == 1 {
        return false;
    }
    // (-1, 1, 1, 0, 0)
    if v[1] == 1 && v[2] == 1 && v[3] == 0 && v[4] == 0 {
        return false;
    }
    // (-1, 1, 1, 0, 1)
    if v[1] == 1 && v[2] == 1 && v[3] == 0 && v[4] == 1 {
        return false;
    }
    // (-1, 1, 1, 0, -1, 1, {0,1}, {1,0}): excludes exactly 2 points:
    //   [-1,1,1,0,-1,1,0,1] and [-1,1,1,0,-1,1,1,0].
    // These share prefix (-1,1,1,0,-1,1) with positive-valued completions.
    // The other 2 completions (-1,0) and (0,-1) remain in Lambda_1024.
    // Closes the 1026->1024 discrepancy against CSV ground truth.
    // See legacy_crossval::test_lattice_csv_vs_predicate_1024d for provenance.
    if v[1] == 1
        && v[2] == 1
        && v[3] == 0
        && v[4] == -1
        && v[5] == 1
        && ((v[6] == 0 && v[7] == 1) || (v[6] == 1 && v[7] == 0))
    {
        return false;
    }

    true
}

/// Check if a vector is in Lambda_512 (Lambda_1024 minus 6 regions).
pub fn is_in_lambda_512(v: &LatticeVector) -> bool {
    if !is_in_lambda_1024(v) {
        return false;
    }

    // Forbidden regions (l_0 is always -1 here)
    // 1. l_1 = 1
    if v[1] == 1 {
        return false;
    }
    // 2. l_1=0, l_2=1
    if v[1] == 0 && v[2] == 1 {
        return false;
    }
    // 3. l_1=0, l_2=0, l_3=0
    if v[1] == 0 && v[2] == 0 && v[3] == 0 {
        return false;
    }
    // 4. l_1=0, l_2=0, l_3=1
    if v[1] == 0 && v[2] == 0 && v[3] == 1 {
        return false;
    }
    // 5. l_1=0, l_2=0, l_3=-1, l_4=1
    if v[1] == 0 && v[2] == 0 && v[3] == -1 && v[4] == 1 {
        return false;
    }
    // 6. l_1=0, l_2=0, l_3=-1, l_4=0, l_5=1, l_6=1
    if v[1] == 0 && v[2] == 0 && v[3] == -1 && v[4] == 0 && v[5] == 1 && v[6] == 1 {
        return false;
    }

    true
}

/// Check if a vector passes k of the 4 Lambda_1024 exclusion rules
/// (applied cumulatively in canonical order, starting from Lambda_2048).
///
/// `k=0` is Lambda_2048, `k=4` is Lambda_1024. Intermediate values
/// give sub-filtration levels for studying the ultrametricity gradient
/// across the Lambda_2048 -> Lambda_1024 transition.
///
/// The 4 rules, in order:
///   1. l_0 != -1 (slice to l_0=-1; removes l_0=0 vectors -- biggest single cut)
///   2. l_1=1, l_2=1, l_3=1
///   3. l_1=1, l_2=1, l_3=0, l_4=0
///   4. l_1=1, l_2=1, l_3=0, l_4=1
///   5. l_1=1, l_2=1, l_3=0, l_4=-1, l_5=1 (closes 1026->1024 discrepancy)
pub fn is_in_lambda_2048_minus_k(v: &LatticeVector, k: usize) -> bool {
    assert!(k <= 5, "k must be in [0, 5]");
    if !is_in_lambda_2048(v) {
        return false;
    }
    // Rule 1: slice to l_0 = -1
    if k >= 1 && v[0] != -1 {
        return false;
    }
    // Rule 2: exclude (-1, 1, 1, 1)
    if k >= 2 && v[1] == 1 && v[2] == 1 && v[3] == 1 {
        return false;
    }
    // Rule 3: exclude (-1, 1, 1, 0, 0)
    if k >= 3 && v[1] == 1 && v[2] == 1 && v[3] == 0 && v[4] == 0 {
        return false;
    }
    // Rule 4: exclude (-1, 1, 1, 0, 1)
    if k >= 4 && v[1] == 1 && v[2] == 1 && v[3] == 0 && v[4] == 1 {
        return false;
    }
    // Rule 5: exclude (-1, 1, 1, 0, -1, 1, {0,1}, {1,0}) -- the 2 CSV-absent points
    if k >= 5
        && v[1] == 1
        && v[2] == 1
        && v[3] == 0
        && v[4] == -1
        && v[5] == 1
        && ((v[6] == 0 && v[7] == 1) || (v[6] == 1 && v[7] == 0))
    {
        return false;
    }
    true
}

/// Check if a vector passes k of the 6 Lambda_512 exclusion rules
/// (applied cumulatively in canonical order).
///
/// `k=0` is Lambda_1024, `k=6` is Lambda_512. Intermediate values
/// give sub-filtration levels for studying the ultrametricity gradient.
///
/// The 6 rules, in order:
///   1. l_1 = 1
///   2. l_1=0, l_2=1
///   3. l_1=0, l_2=0, l_3=0
///   4. l_1=0, l_2=0, l_3=1
///   5. l_1=0, l_2=0, l_3=-1, l_4=1
///   6. l_1=0, l_2=0, l_3=-1, l_4=0, l_5=1, l_6=1
pub fn is_in_lambda_1024_minus_k(v: &LatticeVector, k: usize) -> bool {
    assert!(k <= 6, "k must be in [0, 6]");
    if !is_in_lambda_1024(v) {
        return false;
    }
    // Apply rules 1..k cumulatively
    if k >= 1 && v[1] == 1 {
        return false;
    }
    if k >= 2 && v[1] == 0 && v[2] == 1 {
        return false;
    }
    if k >= 3 && v[1] == 0 && v[2] == 0 && v[3] == 0 {
        return false;
    }
    if k >= 4 && v[1] == 0 && v[2] == 0 && v[3] == 1 {
        return false;
    }
    if k >= 5 && v[1] == 0 && v[2] == 0 && v[3] == -1 && v[4] == 1 {
        return false;
    }
    if k >= 6 && v[1] == 0 && v[2] == 0 && v[3] == -1 && v[4] == 0 && v[5] == 1 && v[6] == 1 {
        return false;
    }
    true
}

/// Check if a vector passes k of the 6 Lambda_256 exclusion rules
/// (applied cumulatively in canonical order, starting from Lambda_512).
///
/// `k=0` is Lambda_512, `k=6` is Lambda_256. Intermediate values
/// give sub-filtration levels for the Lambda_512 -> Lambda_256 transition.
///
/// The 6 rules, in order:
///   1. l_1 = 0 (removes all l_1=0 vectors; survivors have l_1=-1)
///   2. l_2=1, l_3=1
///   3. l_2=1, l_3=0
///   4. l_2=1, l_3=-1, l_4=1
///   5. l_2=1, l_3=-1, l_4=0
///   6. l_2=1, l_3=-1, l_4=-1, l_5=1, l_6=1, l_7=1 (singleton)
pub fn is_in_lambda_512_minus_k(v: &LatticeVector, k: usize) -> bool {
    assert!(k <= 6, "k must be in [0, 6]");
    if !is_in_lambda_512(v) {
        return false;
    }
    if k >= 1 && v[1] == 0 {
        return false;
    }
    if k >= 2 && v[2] == 1 && v[3] == 1 {
        return false;
    }
    if k >= 3 && v[2] == 1 && v[3] == 0 {
        return false;
    }
    if k >= 4 && v[2] == 1 && v[3] == -1 && v[4] == 1 {
        return false;
    }
    if k >= 5 && v[2] == 1 && v[3] == -1 && v[4] == 0 {
        return false;
    }
    if k >= 6 && v[2] == 1 && v[3] == -1 && v[4] == -1 && v[5] == 1 && v[6] == 1 && v[7] == 1 {
        return false;
    }
    true
}

/// Check if a vector is in Lambda_256 (Lambda_512 minus 6 regions).
pub fn is_in_lambda_256(v: &LatticeVector) -> bool {
    if !is_in_lambda_512(v) {
        return false;
    }

    // Forbidden regions (l_0 = -1)
    // 1. l_1 = 0 (implies l_1 must be -1 for success, since l_1 != 1 from 512 rule)
    if v[1] == 0 {
        return false;
    }

    // For the remaining, l_1 = -1 is established.
    // 2. (-1, -1, 1, 1)
    if v[2] == 1 && v[3] == 1 {
        return false;
    }
    // 3. (-1, -1, 1, 0)
    if v[2] == 1 && v[3] == 0 {
        return false;
    }
    // 4. (-1, -1, 1, -1, 1)
    if v[2] == 1 && v[3] == -1 && v[4] == 1 {
        return false;
    }
    // 5. (-1, -1, 1, -1, 0)
    if v[2] == 1 && v[3] == -1 && v[4] == 0 {
        return false;
    }
    // 6. Singleton (-1, -1, 1, -1, -1, 1, 1, 1)
    if v[2] == 1 && v[3] == -1 && v[4] == -1 && v[5] == 1 && v[6] == 1 && v[7] == 1 {
        return false;
    }

    true
}
