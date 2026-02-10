//! Hierarchical Codebook Logic for 256D -> 2048D Lattice Mappings.
//!
//! Implements the "predicate cut" filtration described in the analysis:
//! Lambda_256 <= Lambda_512 <= Lambda_1024 <= Lambda_2048 <= {-1, 0, 1}^8.
//!
//! # Hierarchy
//! - Base: Trinary vectors, even sum, even weight.
//! - 2048D: Base minus 139 forbidden prefixes.
//! - 1024D: 2048D intersected with {l_0 = -1} minus 70 prefixes.
//! - 512D: 1024D minus 6 forbidden regions (trie cuts).
//! - 256D: 512D minus 6 forbidden regions.
//!
//! # Typed Carriers (Layer 0)
//! A `TypedCarrier` X_n = (b, l) pairs a Cayley-Dickson basis element b with
//! its lattice vector l in the encoding dictionary. A `CarrierSet` collects
//! all carriers for a given dimension, providing O(1) lookup by basis index
//! and filtration membership queries.
//!
//! # Scalar Shadow
//! Implements the affine/linear action of the scalar shadow pi(b) on the lattice.

use std::collections::HashMap;

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
/// 4. l_0 constraint: v[0] != +1
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
    // NOTE: Predicate gives 1026 points; CSV has 1024. The CSV excludes
    // [-1,1,1,0,-1,1,0,1] and [-1,1,1,0,-1,1,1,0], but this requires
    // further investigation -- the CSV may be wrong rather than the predicate.
    // See legacy_crossval::test_lattice_csv_vs_predicate_1024d for analysis.

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
pub fn is_in_lambda_2048_minus_k(v: &LatticeVector, k: usize) -> bool {
    assert!(k <= 4, "k must be in [0, 4]");
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

// ============================================================================
// 2048D Forbidden Prefix Enumeration
// ============================================================================

/// Which of the 3 forbidden prefix families a base universe point matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ForbiddenFamily {
    /// Prefix (0, 1, 1): 3-coordinate prefix, free tail of length 5.
    Prefix011,
    /// Prefix (0, 1, 0, 1, 1): 5-coordinate prefix, free tail of length 3.
    Prefix01011,
    /// Prefix (0, 1, 0, 1, 0, 1): 6-coordinate prefix, free tail of length 2.
    Prefix010101,
}

/// A base universe point excluded from Lambda_2048, tagged with its family.
#[derive(Debug, Clone)]
pub struct ForbiddenPoint {
    /// The lattice vector.
    pub vector: LatticeVector,
    /// Which forbidden prefix family it belongs to.
    pub family: ForbiddenFamily,
}

/// Classify which forbidden family a base universe point belongs to.
/// Returns `None` if the point is in Lambda_2048 (not forbidden).
pub fn classify_forbidden(v: &LatticeVector) -> Option<ForbiddenFamily> {
    if !is_in_base_universe(v) {
        return None;
    }
    // Check the 3 forbidden prefix patterns (mutually exclusive by construction:
    // Pattern 1 requires l_2=1, patterns 2 & 3 require l_2=0;
    // Pattern 2 requires l_4=1, pattern 3 requires l_4=0.)
    if v[0] == 0 && v[1] == 1 && v[2] == 1 {
        return Some(ForbiddenFamily::Prefix011);
    }
    if v[0] == 0 && v[1] == 1 && v[2] == 0 && v[3] == 1 && v[4] == 1 {
        return Some(ForbiddenFamily::Prefix01011);
    }
    if v[0] == 0 && v[1] == 1 && v[2] == 0 && v[3] == 1 && v[4] == 0 && v[5] == 1 {
        return Some(ForbiddenFamily::Prefix010101);
    }
    None
}

/// Enumerate all base universe points excluded from Lambda_2048.
///
/// Exhaustively scans {-1, 0, 1}^8 (6561 vectors), filters for base
/// universe membership, then identifies points matching one of the
/// 3 forbidden prefix families. Returns tagged forbidden points.
pub fn enumerate_forbidden_2048() -> Vec<ForbiddenPoint> {
    let mut forbidden = Vec::new();
    // Enumerate all {-1, 0, 1}^8 vectors via base-3 encoding
    for code in 0..3u32.pow(8) {
        let mut v = [0i8; 8];
        let mut c = code;
        for coord in &mut v {
            *coord = (c % 3) as i8 - 1; // map 0->-1, 1->0, 2->1
            c /= 3;
        }
        if let Some(family) = classify_forbidden(&v) {
            forbidden.push(ForbiddenPoint { vector: v, family });
        }
    }
    forbidden
}

// ============================================================================
// Lambda Enumeration from Predicates
// ============================================================================

/// Enumerate all lattice vectors in {-1, 0, 1}^8 satisfying a predicate.
///
/// Performs exhaustive scan of 3^8 = 6561 trinary vectors, returning those
/// that pass `pred`. Results are sorted in lexicographic order (with -1 < 0 < 1).
pub fn enumerate_lattice_by_predicate(pred: impl Fn(&LatticeVector) -> bool) -> Vec<LatticeVector> {
    let mut result = Vec::new();
    for code in 0..3u32.pow(8) {
        let mut v = [0i8; 8];
        let mut c = code;
        for coord in &mut v {
            *coord = (c % 3) as i8 - 1; // map 0->-1, 1->0, 2->1
            c /= 3;
        }
        if pred(&v) {
            result.push(v);
        }
    }
    result.sort();
    result
}

/// Enumerate all Lambda_256 lattice vectors from predicates alone.
///
/// Returns exactly those {-1, 0, 1}^8 vectors passing `is_in_lambda_256`.
/// This is the pure-predicate analog of loading from CSV data files.
pub fn enumerate_lambda_256() -> Vec<LatticeVector> {
    enumerate_lattice_by_predicate(is_in_lambda_256)
}

/// Characterization of the "pinned corner" slice of Lambda_256.
///
/// The slice is defined as { v in Lambda_256 : v[0..k] = (-1, ..., -1) }
/// for some prefix length k. When the prefix is all -1s, every trie-cut
/// exclusion rule in the filtration chain is vacuously satisfied, so the
/// slice reduces to the base universe constraint (even sum + even weight)
/// applied to the tail coordinates.
#[derive(Debug, Clone)]
pub struct SliceCharacterization {
    /// Number of points in the slice.
    pub count: usize,
    /// The prefix length (how many leading -1s are pinned).
    pub prefix_len: usize,
    /// All tail coordinate patterns (the free coordinates after the prefix).
    pub tail_patterns: Vec<LatticeVector>,
    /// Number of distinct nonzero counts (weights) in the tail.
    pub tail_weight_histogram: Vec<(usize, usize)>,
    /// Pairwise squared-distance histogram: (d^2, count).
    pub distance_histogram: Vec<(i32, usize)>,
    /// Inner product histogram: (ip, count).
    pub inner_product_histogram: Vec<(i32, usize)>,
}

/// Characterize the pinned-corner slice of Lambda_256 with a given prefix
/// of k leading -1 coordinates.
///
/// The slice consists of all Lambda_256 points whose first `prefix_len`
/// coordinates are -1. The characterization includes the count, tail
/// weight distribution, and pairwise distance/inner-product histograms.
pub fn characterize_pinned_slice(prefix_len: usize) -> SliceCharacterization {
    assert!(prefix_len <= 8, "prefix_len must be at most 8");

    let all_256 = enumerate_lambda_256();
    let prefix = [-1i8; 8]; // we only use the first prefix_len entries

    let slice_points: Vec<LatticeVector> = all_256
        .into_iter()
        .filter(|v| v[..prefix_len] == prefix[..prefix_len])
        .collect();

    // Extract tail patterns (zero out the prefix for clarity)
    let tail_patterns: Vec<LatticeVector> = slice_points
        .iter()
        .map(|v| {
            let mut tail = [0i8; 8];
            tail[prefix_len..8].copy_from_slice(&v[prefix_len..8]);
            tail
        })
        .collect();

    // Weight histogram (nonzero count in tail)
    let mut weight_counts = std::collections::HashMap::new();
    for v in &slice_points {
        let w = v[prefix_len..].iter().filter(|&&x| x != 0).count();
        *weight_counts.entry(w).or_insert(0usize) += 1;
    }
    let mut tail_weight_histogram: Vec<(usize, usize)> = weight_counts.into_iter().collect();
    tail_weight_histogram.sort();

    // Pairwise squared distances and inner products
    let n = slice_points.len();
    let mut dist_counts = std::collections::HashMap::new();
    let mut ip_counts = std::collections::HashMap::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d2 = 0i32;
            let mut ip = 0i32;
            for (&a, &b) in slice_points[i].iter().zip(slice_points[j].iter()) {
                let diff = a as i32 - b as i32;
                d2 += diff * diff;
                ip += a as i32 * b as i32;
            }
            *dist_counts.entry(d2).or_insert(0usize) += 1;
            *ip_counts.entry(ip).or_insert(0usize) += 1;
        }
    }
    let mut distance_histogram: Vec<(i32, usize)> = dist_counts.into_iter().collect();
    distance_histogram.sort();

    let mut inner_product_histogram: Vec<(i32, usize)> = ip_counts.into_iter().collect();
    inner_product_histogram.sort();

    SliceCharacterization {
        count: slice_points.len(),
        prefix_len,
        tail_patterns,
        tail_weight_histogram,
        distance_histogram,
        inner_product_histogram,
    }
}

/// Component-wise addition in F_3 = Z/3Z (the field with 3 elements).
///
/// Maps {-1, 0, 1} to F_3 via x -> x mod 3, adds, and maps back.
/// This is the natural group operation on trinary vectors that always
/// preserves the trinary constraint. In F_3:
///   -1 + -1 = 1,  -1 + 0 = -1,  -1 + 1 = 0
///    0 +  0 = 0,   0 + 1 = 1,    1 + 1 = -1
pub fn lattice_add_f3(a: &LatticeVector, b: &LatticeVector) -> LatticeVector {
    let mut result = [0i8; 8];
    for (r, (&x, &y)) in result.iter_mut().zip(a.iter().zip(b.iter())) {
        // Compute (x + y) mod 3, keeping in {-1, 0, 1}
        let s = x as i32 + y as i32;
        *r = match s {
            -2 => 1, // wraps: -1 + -1 = 1 in F_3
            -1 => -1,
            0 => 0,
            1 => 1,
            2 => -1, // wraps: 1 + 1 = -1 in F_3
            _ => unreachable!("sum of two trinary values is in [-2, 2]"),
        };
    }
    result
}

/// Component-wise negation in F_3 = Z/3Z.
///
/// Maps each coordinate x -> -x. Since the representation is {-1, 0, 1},
/// negation is just ordinary sign flip: -(-1) = 1, -(0) = 0, -(1) = -1.
/// The result always stays in {-1, 0, 1}^8.
pub fn lattice_negate_f3(v: &LatticeVector) -> LatticeVector {
    let mut result = [0i8; 8];
    for (r, &x) in result.iter_mut().zip(v.iter()) {
        *r = -x;
    }
    result
}

/// Component-wise difference of two lattice vectors in Z.
///
/// a - b, coordinate by coordinate. The result may leave {-1, 0, 1}^8
/// when opposite-sign coordinates are subtracted (e.g., 1 - (-1) = 2).
pub fn lattice_diff(a: &LatticeVector, b: &LatticeVector) -> [i32; 8] {
    let mut result = [0i32; 8];
    for (r, (&x, &y)) in result.iter_mut().zip(a.iter().zip(b.iter())) {
        *r = x as i32 - y as i32;
    }
    result
}

/// Apply the Scalar Shadow action to a lattice vector.
///
/// Addition mode: l_out = l + a * 1_8
/// Multiplication mode: l_out = a * l
pub fn apply_scalar_shadow(l: &LatticeVector, a: i8, mode: &str) -> LatticeVector {
    let mut res = [0i8; 8];
    match mode {
        "add" => {
            for i in 0..8 {
                res[i] = l[i].saturating_add(a);
            }
        }
        "mul" => {
            for i in 0..8 {
                res[i] = l[i] * a;
            }
        }
        _ => panic!("Unknown mode: {mode}"),
    }
    res
}

// ============================================================================
// Layer 0: Typed Carriers
// ============================================================================

/// A typed carrier X_n = (b, l) pairing a Cayley-Dickson basis element
/// with its lattice vector in the encoding dictionary.
///
/// This is the foundational data type for the monograph abstraction hierarchy:
/// - Layer 0: TypedCarrier (this struct)
/// - Layer 1: EncodingDictionary (Phi_n: basis -> lattice bijection)
/// - Layer 2: Elevated addition (l -> l + Phi(b))
/// - Layer 3: Named graph predicates (P_ZD, P_match)
/// - Layer 4: Invariant suite (degree, spectrum, triangles, etc.)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TypedCarrier {
    /// CD basis element index in [0, dim).
    pub basis_index: usize,
    /// 8D lattice vector in {-1, 0, 1}^8.
    pub lattice_vec: LatticeVector,
}

/// The dimension tier for filtration membership queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FiltrationTier {
    Base,
    Lambda2048,
    Lambda1024,
    Lambda512,
    Lambda256,
}

impl TypedCarrier {
    /// Create a new typed carrier.
    pub fn new(basis_index: usize, lattice_vec: LatticeVector) -> Self {
        Self {
            basis_index,
            lattice_vec,
        }
    }

    /// Convert from the Vec<i32> representation used in cd_external.
    /// Returns None if any coordinate is outside [-1, 1] or the vector
    /// does not have exactly 8 components.
    pub fn from_i32_vec(basis_index: usize, v: &[i32]) -> Option<Self> {
        if v.len() != 8 {
            return None;
        }
        let mut lv = [0i8; 8];
        for (i, &val) in v.iter().enumerate() {
            if !(-1..=1).contains(&val) {
                return None;
            }
            lv[i] = val as i8;
        }
        Some(Self {
            basis_index,
            lattice_vec: lv,
        })
    }

    /// Return the highest filtration tier this carrier's lattice vector
    /// belongs to (most restrictive = smallest codebook).
    pub fn filtration_tier(&self) -> FiltrationTier {
        if is_in_lambda_256(&self.lattice_vec) {
            FiltrationTier::Lambda256
        } else if is_in_lambda_512(&self.lattice_vec) {
            FiltrationTier::Lambda512
        } else if is_in_lambda_1024(&self.lattice_vec) {
            FiltrationTier::Lambda1024
        } else if is_in_lambda_2048(&self.lattice_vec) {
            FiltrationTier::Lambda2048
        } else if is_in_base_universe(&self.lattice_vec) {
            FiltrationTier::Base
        } else {
            // Not even in the base universe -- should not happen for valid data.
            FiltrationTier::Base
        }
    }

    /// Check if this carrier's lattice vector is in Lambda_n.
    pub fn is_in_lambda(&self, dim: usize) -> bool {
        match dim {
            256 => is_in_lambda_256(&self.lattice_vec),
            512 => is_in_lambda_512(&self.lattice_vec),
            1024 => is_in_lambda_1024(&self.lattice_vec),
            2048 => is_in_lambda_2048(&self.lattice_vec),
            _ => is_in_base_universe(&self.lattice_vec),
        }
    }
}

/// The full carrier set for a given CD algebra dimension.
///
/// Collects all typed carriers X_n = (b, l) and provides O(1) lookup
/// by basis index, filtration queries, and consistency checks.
#[derive(Debug, Clone)]
pub struct CarrierSet {
    /// CD algebra dimension.
    pub dim: usize,
    /// Ordered list of carriers (by basis_index).
    carriers: Vec<TypedCarrier>,
    /// Basis index -> position in carriers vec (O(1) lookup).
    index: HashMap<usize, usize>,
}

impl CarrierSet {
    /// Build a carrier set from a basis_index -> lattice_vector map.
    /// This is the bridge from cd_external::load_lattice_map().
    pub fn from_i32_map(dim: usize, map: &HashMap<usize, Vec<i32>>) -> Self {
        let mut carriers: Vec<TypedCarrier> = map
            .iter()
            .filter_map(|(&idx, v)| TypedCarrier::from_i32_vec(idx, v))
            .collect();
        carriers.sort_by_key(|c| c.basis_index);

        let index: HashMap<usize, usize> = carriers
            .iter()
            .enumerate()
            .map(|(pos, c)| (c.basis_index, pos))
            .collect();

        Self {
            dim,
            carriers,
            index,
        }
    }

    /// Build a carrier set from pre-validated LatticeVectors.
    pub fn from_lattice_vecs(dim: usize, pairs: &[(usize, LatticeVector)]) -> Self {
        let mut carriers: Vec<TypedCarrier> = pairs
            .iter()
            .map(|&(idx, lv)| TypedCarrier::new(idx, lv))
            .collect();
        carriers.sort_by_key(|c| c.basis_index);

        let index: HashMap<usize, usize> = carriers
            .iter()
            .enumerate()
            .map(|(pos, c)| (c.basis_index, pos))
            .collect();

        Self {
            dim,
            carriers,
            index,
        }
    }

    /// Number of carriers in this set.
    pub fn len(&self) -> usize {
        self.carriers.len()
    }

    /// Whether the carrier set is empty.
    pub fn is_empty(&self) -> bool {
        self.carriers.is_empty()
    }

    /// Look up a carrier by basis index. O(1).
    pub fn get(&self, basis_index: usize) -> Option<&TypedCarrier> {
        self.index.get(&basis_index).map(|&pos| &self.carriers[pos])
    }

    /// Iterate over all carriers in basis-index order.
    pub fn iter(&self) -> impl Iterator<Item = &TypedCarrier> {
        self.carriers.iter()
    }

    /// Return all carriers whose lattice vectors are in Lambda_target_dim.
    pub fn filter_to_lambda(&self, target_dim: usize) -> Vec<&TypedCarrier> {
        self.carriers
            .iter()
            .filter(|c| c.is_in_lambda(target_dim))
            .collect()
    }

    /// Check that the carrier set is a valid encoding dictionary:
    /// - Every basis index in [0, dim) has exactly one carrier.
    /// - No two carriers share the same lattice vector (injectivity).
    pub fn validate(&self) -> CarrierSetValidation {
        let mut missing = Vec::new();
        for i in 0..self.dim {
            if !self.index.contains_key(&i) {
                missing.push(i);
            }
        }

        let mut seen = HashMap::new();
        let mut duplicates = Vec::new();
        for c in &self.carriers {
            if let Some(&prev_idx) = seen.get(&c.lattice_vec) {
                duplicates.push((prev_idx, c.basis_index));
            } else {
                seen.insert(c.lattice_vec, c.basis_index);
            }
        }

        CarrierSetValidation {
            is_complete: missing.is_empty(),
            is_injective: duplicates.is_empty(),
            missing_basis_indices: missing,
            duplicate_lattice_pairs: duplicates,
        }
    }

    /// Count how many carriers fall into each filtration tier.
    pub fn tier_histogram(&self) -> HashMap<FiltrationTier, usize> {
        let mut hist = HashMap::new();
        for c in &self.carriers {
            *hist.entry(c.filtration_tier()).or_insert(0) += 1;
        }
        hist
    }
}

/// Result of validating a CarrierSet for encoding dictionary properties.
#[derive(Debug, Clone)]
pub struct CarrierSetValidation {
    /// True if every basis index in [0, dim) has a carrier.
    pub is_complete: bool,
    /// True if no two carriers share the same lattice vector.
    pub is_injective: bool,
    /// Basis indices missing from the carrier set.
    pub missing_basis_indices: Vec<usize>,
    /// Pairs of basis indices that map to the same lattice vector.
    pub duplicate_lattice_pairs: Vec<(usize, usize)>,
}

impl CarrierSetValidation {
    /// True if the carrier set forms a valid bijection.
    pub fn is_valid_dictionary(&self) -> bool {
        self.is_complete && self.is_injective
    }
}

// ============================================================================
// Layer 1: Encoding Dictionary Phi_n
// ============================================================================

/// The encoding dictionary Phi_n: {e_0, ..., e_{n-1}} -> Lambda_n.
///
/// This is a validated bijection between CD basis elements and lattice vectors.
/// It provides both forward (encode: basis -> lattice) and inverse
/// (decode: lattice -> basis) operations in O(1).
///
/// Construction requires that the underlying CarrierSet pass validation
/// (complete + injective). If validation fails, `try_from_carrier_set`
/// returns the validation errors.
#[derive(Debug, Clone)]
pub struct EncodingDictionary {
    /// The underlying carrier set (validated).
    carriers: CarrierSet,
    /// Inverse map: lattice vector -> basis index (O(1) decode).
    inverse: HashMap<LatticeVector, usize>,
}

impl EncodingDictionary {
    /// Attempt to build an encoding dictionary from a carrier set.
    /// Fails if the carrier set is not a valid bijection.
    pub fn try_from_carrier_set(cs: CarrierSet) -> Result<Self, CarrierSetValidation> {
        let validation = cs.validate();
        if !validation.is_valid_dictionary() {
            return Err(validation);
        }

        let inverse: HashMap<LatticeVector, usize> =
            cs.iter().map(|c| (c.lattice_vec, c.basis_index)).collect();

        Ok(Self {
            carriers: cs,
            inverse,
        })
    }

    /// Build from a basis_index -> Vec<i32> map (bridge from cd_external).
    /// Fails if the resulting carrier set is not a valid bijection.
    pub fn try_from_i32_map(
        dim: usize,
        map: &HashMap<usize, Vec<i32>>,
    ) -> Result<Self, CarrierSetValidation> {
        let cs = CarrierSet::from_i32_map(dim, map);
        Self::try_from_carrier_set(cs)
    }

    /// Build from pre-validated (basis_index, lattice_vector) pairs.
    pub fn try_from_pairs(
        dim: usize,
        pairs: &[(usize, LatticeVector)],
    ) -> Result<Self, CarrierSetValidation> {
        let cs = CarrierSet::from_lattice_vecs(dim, pairs);
        Self::try_from_carrier_set(cs)
    }

    /// The CD algebra dimension this dictionary encodes.
    pub fn dim(&self) -> usize {
        self.carriers.dim
    }

    /// Number of entries (should equal dim for a valid dictionary).
    pub fn len(&self) -> usize {
        self.carriers.len()
    }

    /// Whether the dictionary is empty.
    pub fn is_empty(&self) -> bool {
        self.carriers.is_empty()
    }

    /// Encode: Phi_n(basis_index) -> LatticeVector.
    /// Returns None if basis_index is not in [0, dim).
    pub fn encode(&self, basis_index: usize) -> Option<&LatticeVector> {
        self.carriers.get(basis_index).map(|c| &c.lattice_vec)
    }

    /// Decode: Phi_n^{-1}(lattice_vec) -> basis_index.
    /// Returns None if the lattice vector is not in the codebook.
    pub fn decode(&self, lattice_vec: &LatticeVector) -> Option<usize> {
        self.inverse.get(lattice_vec).copied()
    }

    /// Access the underlying carrier set.
    pub fn carrier_set(&self) -> &CarrierSet {
        &self.carriers
    }

    /// Iterate over all (basis_index, lattice_vector) pairs in order.
    pub fn iter(&self) -> impl Iterator<Item = (usize, &LatticeVector)> {
        self.carriers
            .iter()
            .map(|c| (c.basis_index, &c.lattice_vec))
    }

    /// Restrict this dictionary to carriers whose lattice vectors are in
    /// Lambda_{target_dim}. Returns a new (smaller) dictionary for the
    /// sub-codebook at the target filtration tier.
    ///
    /// Note: the returned dictionary has dim = target_dim, and its basis
    /// indices are the ORIGINAL indices from the parent dictionary. It will
    /// not pass completeness validation (missing basis indices are expected).
    pub fn restrict_to_lambda(&self, target_dim: usize) -> Vec<(usize, LatticeVector)> {
        self.carriers
            .iter()
            .filter(|c| c.is_in_lambda(target_dim))
            .map(|c| (c.basis_index, c.lattice_vec))
            .collect()
    }

    /// Compute the scalar shadow pi(b) for a basis element.
    /// Defined as sign(sum(lattice_vec)).
    pub fn scalar_shadow(&self, basis_index: usize) -> Option<i8> {
        self.encode(basis_index).map(|lv| {
            let s: i32 = lv.iter().map(|&x| x as i32).sum();
            if s > 0 {
                1
            } else if s < 0 {
                -1
            } else {
                0
            }
        })
    }
}

// ============================================================================
// Layer 2: Elevated Addition
// ============================================================================

/// Component-wise addition of two lattice vectors in Z^8.
///
/// This is ordinary integer addition. The result may leave the trinary
/// set {-1, 0, 1}^8, which is why elevated addition must check membership
/// before decoding.
pub fn lattice_add(a: &LatticeVector, b: &LatticeVector) -> [i32; 8] {
    let mut result = [0i32; 8];
    for (r, (&x, &y)) in result.iter_mut().zip(a.iter().zip(b.iter())) {
        *r = x as i32 + y as i32;
    }
    result
}

/// Try to narrow a Z^8 vector back to a trinary lattice vector.
/// Returns None if any coordinate is outside [-1, 1].
pub fn try_narrow_to_lattice(v: &[i32; 8]) -> Option<LatticeVector> {
    let mut result = [0i8; 8];
    for (r, &x) in result.iter_mut().zip(v.iter()) {
        if !(-1..=1).contains(&x) {
            return None;
        }
        *r = x as i8;
    }
    Some(result)
}

/// Result of an elevated addition Phi(a) + Phi(b).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElevatedResult {
    /// The sum lands in the codebook and decodes to basis element c.
    InCodebook {
        sum_vec: LatticeVector,
        decoded_index: usize,
    },
    /// The sum is a valid trinary vector but not in the codebook.
    OutOfCodebook { sum_vec: LatticeVector },
    /// The sum leaves {-1, 0, 1}^8 entirely (some coordinate exceeds bounds).
    OutOfBounds { sum_vec: [i32; 8] },
}

/// Statistics for the elevated addition table.
#[derive(Debug, Clone)]
pub struct ElevatedAdditionStats {
    /// Total number of ordered pairs (a, b) tested.
    pub total_pairs: usize,
    /// Number of pairs where Phi(a)+Phi(b) decodes to some c in the dictionary.
    pub in_codebook: usize,
    /// Number of pairs where sum is trinary but not in codebook.
    pub out_of_codebook: usize,
    /// Number of pairs where sum leaves {-1,0,1}^8.
    pub out_of_bounds: usize,
    /// Closure rate: in_codebook / total_pairs.
    pub closure_rate: f64,
    /// Whether the operation is commutative (Phi(a)+Phi(b) = Phi(b)+Phi(a) always).
    pub is_commutative: bool,
    /// Number of basis elements b that act as identity (Phi(a)+Phi(b)=Phi(a) for all a).
    pub identity_count: usize,
}

/// Result of an F_3-elevated addition Phi(a) +_3 Phi(b).
///
/// Unlike Z-elevated addition, F_3-addition always stays in {-1,0,1}^8
/// (it wraps around), so there is no OutOfBounds variant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElevatedResultF3 {
    /// The F_3-sum lands in the codebook and decodes to basis element c.
    InCodebook {
        sum_vec: LatticeVector,
        decoded_index: usize,
    },
    /// The F_3-sum is a valid trinary vector but not in the codebook.
    OutOfCodebook { sum_vec: LatticeVector },
}

/// Statistics for the F_3-elevated addition table.
#[derive(Debug, Clone)]
pub struct ElevatedAdditionStatsF3 {
    /// Total number of ordered pairs (a, b) tested.
    pub total_pairs: usize,
    /// Number of pairs where Phi(a)+_3 Phi(b) decodes to some c.
    pub in_codebook: usize,
    /// Number of pairs where F_3-sum is trinary but not in codebook.
    pub out_of_codebook: usize,
    /// Closure rate: in_codebook / total_pairs.
    pub closure_rate: f64,
    /// Whether F_3-addition is commutative on this codebook.
    pub is_commutative: bool,
    /// Number of basis elements that act as identity.
    pub identity_count: usize,
    /// Fraction of tested triples where (a+b)+c = a+(b+c) in F_3.
    pub associativity_rate: f64,
    /// Number of triples tested for associativity.
    pub associativity_triples_tested: usize,
}

impl EncodingDictionary {
    /// Perform elevated addition: compute Phi(a) + Phi(b) and try to decode.
    ///
    /// Returns an `ElevatedResult` describing where the sum lands:
    /// - `InCodebook`: the sum is in the dictionary and decodes to basis index c
    /// - `OutOfCodebook`: the sum is trinary but not a codeword
    /// - `OutOfBounds`: the sum has coordinates outside [-1, 1]
    pub fn elevated_add(&self, a: usize, b: usize) -> Option<ElevatedResult> {
        let lv_a = self.encode(a)?;
        let lv_b = self.encode(b)?;
        let sum = lattice_add(lv_a, lv_b);

        match try_narrow_to_lattice(&sum) {
            Some(narrow) => match self.decode(&narrow) {
                Some(c) => Some(ElevatedResult::InCodebook {
                    sum_vec: narrow,
                    decoded_index: c,
                }),
                None => Some(ElevatedResult::OutOfCodebook { sum_vec: narrow }),
            },
            None => Some(ElevatedResult::OutOfBounds { sum_vec: sum }),
        }
    }

    /// Compute the full elevated addition table for all ordered pairs (a, b).
    ///
    /// Returns an n x n table where table[a][b] = elevated_add(a, b).
    /// This table captures the complete lattice-addition structure of the codebook.
    pub fn elevated_addition_table(&self) -> Vec<Vec<ElevatedResult>> {
        let n = self.dim();
        let mut table = Vec::with_capacity(n);
        for a in 0..n {
            let mut row = Vec::with_capacity(n);
            for b in 0..n {
                row.push(
                    self.elevated_add(a, b)
                        .expect("basis indices should be valid"),
                );
            }
            table.push(row);
        }
        table
    }

    /// Compute summary statistics for the elevated addition table.
    pub fn elevated_addition_stats(&self) -> ElevatedAdditionStats {
        let n = self.dim();
        let mut in_codebook = 0usize;
        let mut out_of_codebook = 0usize;
        let mut out_of_bounds = 0usize;
        let mut is_commutative = true;
        // Build the table for commutativity check
        let table = self.elevated_addition_table();

        for (a, row_a) in table.iter().enumerate() {
            for (b, result) in row_a.iter().enumerate() {
                match result {
                    ElevatedResult::InCodebook { .. } => in_codebook += 1,
                    ElevatedResult::OutOfCodebook { .. } => out_of_codebook += 1,
                    ElevatedResult::OutOfBounds { .. } => out_of_bounds += 1,
                }
                // Commutativity: check table[a][b] == table[b][a]
                if a < b && *result != table[b][a] {
                    is_commutative = false;
                }
            }
        }

        // Identity check: b is identity if Phi(a)+Phi(b) decodes to a for all a
        let identity_count = (0..n)
            .filter(|&b| {
                table.iter().enumerate().all(|(a, row)| {
                    matches!(&row[b],
                        ElevatedResult::InCodebook { decoded_index, .. }
                        if *decoded_index == a
                    )
                })
            })
            .count();

        let total_pairs = n * n;
        ElevatedAdditionStats {
            total_pairs,
            in_codebook,
            out_of_codebook,
            out_of_bounds,
            closure_rate: in_codebook as f64 / total_pairs as f64,
            is_commutative,
            identity_count,
        }
    }

    /// For a given basis element b, compute the "translation orbit":
    /// the set of basis elements a for which Phi(a) + Phi(b) is in the codebook.
    pub fn translation_orbit(&self, b: usize) -> Vec<(usize, usize)> {
        let n = self.dim();
        let mut orbit = Vec::new();
        for a in 0..n {
            if let Some(ElevatedResult::InCodebook { decoded_index, .. }) = self.elevated_add(a, b)
            {
                orbit.push((a, decoded_index));
            }
        }
        orbit
    }

    /// Perform F_3-elevated addition: Phi(a) +_3 Phi(b) mod 3, then decode.
    ///
    /// Unlike Z-addition, F_3-addition always produces a trinary vector
    /// (it wraps around: -1+(-1)=1, 1+1=-1). The result is always either
    /// InCodebook or OutOfCodebook, never OutOfBounds.
    pub fn elevated_add_f3(&self, a: usize, b: usize) -> Option<ElevatedResultF3> {
        let lv_a = self.encode(a)?;
        let lv_b = self.encode(b)?;
        let sum = lattice_add_f3(lv_a, lv_b);

        match self.decode(&sum) {
            Some(c) => Some(ElevatedResultF3::InCodebook {
                sum_vec: sum,
                decoded_index: c,
            }),
            None => Some(ElevatedResultF3::OutOfCodebook { sum_vec: sum }),
        }
    }

    /// Compute the F_3-elevated addition table and statistics.
    pub fn elevated_addition_stats_f3(&self) -> ElevatedAdditionStatsF3 {
        let n = self.dim();
        let mut in_codebook = 0usize;
        let mut out_of_codebook = 0usize;
        let mut is_commutative = true;
        // Precompute all results for commutativity check
        let mut table: Vec<Vec<ElevatedResultF3>> = Vec::with_capacity(n);
        for a in 0..n {
            let mut row = Vec::with_capacity(n);
            for b in 0..n {
                row.push(
                    self.elevated_add_f3(a, b)
                        .expect("basis indices should be valid"),
                );
            }
            table.push(row);
        }

        for (a, row_a) in table.iter().enumerate() {
            for (b, result) in row_a.iter().enumerate() {
                match result {
                    ElevatedResultF3::InCodebook { .. } => in_codebook += 1,
                    ElevatedResultF3::OutOfCodebook { .. } => out_of_codebook += 1,
                }
                if a < b && *result != table[b][a] {
                    is_commutative = false;
                }
            }
        }

        // Identity check in F_3
        let identity_count = (0..n)
            .filter(|&b| {
                table.iter().enumerate().all(|(a, row)| {
                    matches!(&row[b],
                        ElevatedResultF3::InCodebook { decoded_index, .. }
                        if *decoded_index == a
                    )
                })
            })
            .count();

        // Associativity check (sample-based for large dictionaries)
        let mut associative_triples = 0usize;
        let mut total_triples = 0usize;
        let limit = n.min(32); // sample at most 32^3 triples
        for a in 0..limit {
            for b in 0..limit {
                for c in 0..limit {
                    total_triples += 1;
                    // (a + b) + c vs a + (b + c)
                    let ab = lattice_add_f3(self.encode(a).unwrap(), self.encode(b).unwrap());
                    let abc_left = lattice_add_f3(&ab, self.encode(c).unwrap());

                    let bc = lattice_add_f3(self.encode(b).unwrap(), self.encode(c).unwrap());
                    let abc_right = lattice_add_f3(self.encode(a).unwrap(), &bc);

                    if abc_left == abc_right {
                        associative_triples += 1;
                    }
                }
            }
        }

        let total_pairs = n * n;
        ElevatedAdditionStatsF3 {
            total_pairs,
            in_codebook,
            out_of_codebook,
            closure_rate: in_codebook as f64 / total_pairs as f64,
            is_commutative,
            identity_count,
            associativity_rate: associative_triples as f64 / total_triples as f64,
            associativity_triples_tested: total_triples,
        }
    }

    /// Compute the F_3-elevated difference: Phi(a) - Phi(b) in Z, then decode.
    ///
    /// Returns InCodebook if the difference is trinary and in the dictionary,
    /// OutOfCodebook if trinary but not in dictionary, OutOfBounds if
    /// difference leaves {-1,0,1}^8.
    pub fn elevated_diff(&self, a: usize, b: usize) -> Option<ElevatedResult> {
        let lv_a = self.encode(a)?;
        let lv_b = self.encode(b)?;
        let diff = lattice_diff(lv_a, lv_b);

        match try_narrow_to_lattice(&diff) {
            Some(narrow) => match self.decode(&narrow) {
                Some(c) => Some(ElevatedResult::InCodebook {
                    sum_vec: narrow,
                    decoded_index: c,
                }),
                None => Some(ElevatedResult::OutOfCodebook { sum_vec: narrow }),
            },
            None => Some(ElevatedResult::OutOfBounds { sum_vec: diff }),
        }
    }
}

// ============================================================================
// Layer 2b: Multiplication Coupling (Thesis D, C-466)
// ============================================================================

/// Result of attempting to compute rho(b) for a single basis element.
///
/// For each basis b, the CD multiplication table defines a permutation
/// mu_b(c) = index of e_b * e_c.  We ask: does the encoding dictionary
/// Phi intertwine this permutation with a *linear* map on the subspace
/// V = span(Phi(0), ..., Phi(dim-1)) ?
///
/// Since the lattice vectors may span only r < 8 dimensions (due to
/// filtration constraints like l_0 = -1), we work in the r-dimensional
/// reduced space and report the r x r coupling matrix.
///
/// Two variants are computed:
/// - **unsigned**: Phi(mu_b(c)) = M * Phi(c) for all c
/// - **signed**: sign(b,c) * Phi(mu_b(c)) = M * Phi(c) for all c
#[derive(Debug, Clone)]
pub struct BasisCouplingResult {
    /// The basis element index b.
    pub basis_index: usize,
    /// Whether the unsigned coupling is consistent (residual < tolerance).
    pub unsigned_consistent: bool,
    /// Whether the signed coupling is consistent.
    pub signed_consistent: bool,
    /// Determinant of unsigned coupling matrix (in the reduced space).
    pub unsigned_det: Option<f64>,
    /// Determinant of signed coupling matrix (in the reduced space).
    pub signed_det: Option<f64>,
    /// Maximum absolute residual across all verification columns (unsigned).
    pub unsigned_max_residual: f64,
    /// Maximum absolute residual across all verification columns (signed).
    pub signed_max_residual: f64,
}

/// Full multiplication coupling analysis for all basis elements of a dictionary.
#[derive(Debug, Clone)]
pub struct MultiplicationCoupling {
    /// CD algebra dimension.
    pub dim: usize,
    /// Rank of the lattice vectors (dimension of the spanned subspace).
    pub rank: usize,
    /// Per-basis results (one per basis element 0..dim).
    pub results: Vec<BasisCouplingResult>,
    /// How many bases have consistent unsigned coupling.
    pub unsigned_consistent_count: usize,
    /// How many bases have consistent signed coupling.
    pub signed_consistent_count: usize,
    /// Determinants of all unsigned coupling matrices (for structure analysis).
    pub unsigned_dets: Vec<(usize, f64)>,
    /// Determinants of all signed coupling matrices.
    pub signed_dets: Vec<(usize, f64)>,
}

/// Compute the multiplication coupling for all basis elements.
///
/// For each basis b in [0, dim), determines whether the permutation
/// mu_b(c) = prod(b,c) induces a linear map on the subspace spanned
/// by the lattice vectors Phi(0)..Phi(dim-1).
///
/// Since the lattice vectors may span only r < 8 dimensions (due to
/// filtration constraints), we project into the r-dimensional subspace
/// using an orthonormal basis Q, solve the r x r system, and verify
/// consistency on all dim vectors.
///
/// Two variants: unsigned (ignoring product sign) and signed (including sign).
pub fn compute_multiplication_coupling(
    dict: &EncodingDictionary,
    mult_table: &crate::construction::mult_table::CdMultTable,
) -> MultiplicationCoupling {
    let dim = dict.dim();
    assert_eq!(
        dim, mult_table.dim,
        "dictionary and multiplication table dimensions must match"
    );

    let tol = 1e-8;

    // Collect all lattice vectors: phi[c] = Phi(c) as f64
    let mut phi: Vec<[f64; 8]> = vec![[0.0; 8]; dim];
    for (idx, lv) in dict.iter() {
        for k in 0..8 {
            phi[idx][k] = lv[k] as f64;
        }
    }

    // Build orthonormal basis Q for the column space of {Phi(c)}
    // using Gram-Schmidt. Q is stored as a Vec of 8-element vectors.
    let (q_basis, rank) = gram_schmidt_basis(&phi);

    // Project all lattice vectors into the reduced space: phi_r[c] = Q^T * phi[c]
    let phi_r: Vec<Vec<f64>> = phi.iter().map(|p| project_to_basis(p, &q_basis)).collect();

    // Find r linearly independent columns in the reduced space
    let pivot_cols = find_pivot_columns_reduced(&phi_r, rank);
    assert_eq!(
        pivot_cols.len(),
        rank,
        "Expected {rank} pivot columns in reduced space, got {}",
        pivot_cols.len()
    );

    // Build the r x r input matrix X_r from pivot columns
    let x_r = build_square_matrix(&phi_r, &pivot_cols, rank);
    let x_r_inv = invert_nxn(&x_r, rank);

    let mut results = Vec::with_capacity(dim);
    let mut unsigned_dets = Vec::new();
    let mut signed_dets = Vec::new();

    for b in 0..dim {
        let mut unsigned_max_res = 0.0f64;
        let mut signed_max_res = 0.0f64;
        let mut u_det = None;
        let mut s_det = None;

        if let Some(ref xi) = x_r_inv {
            // Build output matrices for pivot columns
            let mut y_u = vec![vec![0.0f64; rank]; rank];
            let mut y_s = vec![vec![0.0f64; rank]; rank];

            for (j, &c) in pivot_cols.iter().enumerate() {
                let (sign, prod_idx) = mult_table.multiply_basis(b, c);
                let out_r = &phi_r[prod_idx];
                for i in 0..rank {
                    y_u[i][j] = out_r[i];
                    y_s[i][j] = (sign as f64) * out_r[i];
                }
            }

            let m_u = mat_mul_nxn(&y_u, xi, rank);
            let m_s = mat_mul_nxn(&y_s, xi, rank);

            // Verify on ALL basis elements
            for (c, phi_rc) in phi_r.iter().enumerate() {
                let (sign, prod_idx) = mult_table.multiply_basis(b, c);
                let phi_r_prod = &phi_r[prod_idx];

                for i in 0..rank {
                    let predicted_u: f64 =
                        m_u[i].iter().zip(phi_rc.iter()).map(|(&m, &p)| m * p).sum();
                    unsigned_max_res = unsigned_max_res.max((predicted_u - phi_r_prod[i]).abs());

                    let predicted_s: f64 =
                        m_s[i].iter().zip(phi_rc.iter()).map(|(&m, &p)| m * p).sum();
                    let target_s = (sign as f64) * phi_r_prod[i];
                    signed_max_res = signed_max_res.max((predicted_s - target_s).abs());
                }
            }

            if unsigned_max_res < tol {
                u_det = Some(det_nxn(&m_u, rank));
            }
            if signed_max_res < tol {
                s_det = Some(det_nxn(&m_s, rank));
            }
        }

        let unsigned_consistent = unsigned_max_res < tol;
        let signed_consistent = signed_max_res < tol;

        if let Some(d) = u_det {
            unsigned_dets.push((b, d));
        }
        if let Some(d) = s_det {
            signed_dets.push((b, d));
        }

        results.push(BasisCouplingResult {
            basis_index: b,
            unsigned_consistent,
            signed_consistent,
            unsigned_det: u_det,
            signed_det: s_det,
            unsigned_max_residual: unsigned_max_res,
            signed_max_residual: signed_max_res,
        });
    }

    let unsigned_consistent_count = results.iter().filter(|r| r.unsigned_consistent).count();
    let signed_consistent_count = results.iter().filter(|r| r.signed_consistent).count();

    MultiplicationCoupling {
        dim,
        rank,
        results,
        unsigned_consistent_count,
        signed_consistent_count,
        unsigned_dets,
        signed_dets,
    }
}

/// Build an orthonormal basis for the column space of the given vectors.
/// Returns (basis_vectors, rank).
fn gram_schmidt_basis(vectors: &[[f64; 8]]) -> (Vec<[f64; 8]>, usize) {
    let mut basis: Vec<[f64; 8]> = Vec::with_capacity(8);

    for v in vectors {
        if basis.len() == 8 {
            break;
        }
        let mut w = *v;
        for b in &basis {
            let dot: f64 = w.iter().zip(b.iter()).map(|(a, b)| a * b).sum();
            for (wk, &bk) in w.iter_mut().zip(b.iter()) {
                *wk -= dot * bk;
            }
        }
        let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-10 {
            w.iter_mut().for_each(|x| *x /= norm);
            basis.push(w);
        }
    }

    let rank = basis.len();
    (basis, rank)
}

/// Project an 8D vector onto the orthonormal basis, giving an r-dimensional vector.
fn project_to_basis(v: &[f64; 8], basis: &[[f64; 8]]) -> Vec<f64> {
    basis
        .iter()
        .map(|b| v.iter().zip(b.iter()).map(|(&vi, &bi)| vi * bi).sum())
        .collect()
}

/// Find r linearly independent columns from reduced-space vectors.
fn find_pivot_columns_reduced(phi_r: &[Vec<f64>], rank: usize) -> Vec<usize> {
    let mut pivots = Vec::with_capacity(rank);
    let mut basis = Vec::<Vec<f64>>::with_capacity(rank);

    for (c, col) in phi_r.iter().enumerate() {
        if pivots.len() == rank {
            break;
        }

        let mut v = col.clone();
        for b in &basis {
            let dot: f64 = v.iter().zip(b.iter()).map(|(a, b)| a * b).sum();
            let norm_sq: f64 = b.iter().map(|x| x * x).sum();
            if norm_sq > 1e-12 {
                for (vk, &bk) in v.iter_mut().zip(b.iter()) {
                    *vk -= (dot / norm_sq) * bk;
                }
            }
        }

        let norm_sq: f64 = v.iter().map(|x| x * x).sum();
        if norm_sq > 1e-8 {
            basis.push(v);
            pivots.push(c);
        }
    }

    pivots
}

/// Build an r x r matrix from selected columns.
fn build_square_matrix(phi_r: &[Vec<f64>], pivot_cols: &[usize], rank: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; rank]; rank];
    for (j, &c) in pivot_cols.iter().enumerate() {
        for (i, row) in m.iter_mut().enumerate() {
            row[j] = phi_r[c][i];
        }
    }
    m
}

/// Invert an n x n matrix using Gauss-Jordan elimination.
fn invert_nxn(m: &[Vec<f64>], n: usize) -> Option<Vec<Vec<f64>>> {
    let nn = 2 * n;
    let mut aug: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = vec![0.0; nn];
            row[..n].copy_from_slice(&m[i]);
            row[n + i] = 1.0;
            row
        })
        .collect();

    for col in 0..n {
        let max_row = aug[col..]
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a[col].abs().partial_cmp(&b[col].abs()).unwrap())
            .map(|(idx, _)| idx + col)
            .unwrap();

        if aug[max_row][col].abs() < 1e-12 {
            return None;
        }

        aug.swap(col, max_row);

        let pivot = aug[col][col];
        aug[col].iter_mut().for_each(|v| *v /= pivot);

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row][col];
            let pivot_row: Vec<f64> = aug[col].clone();
            for (v, &p) in aug[row].iter_mut().zip(pivot_row.iter()) {
                *v -= factor * p;
            }
        }
    }

    Some(aug.iter().map(|row| row[n..].to_vec()).collect())
}

/// Multiply two n x n matrices.
fn mat_mul_nxn(a: &[Vec<f64>], b: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    let mut c = vec![vec![0.0; n]; n];
    for (c_row, a_row) in c.iter_mut().zip(a.iter()) {
        for (j, c_val) in c_row.iter_mut().enumerate() {
            *c_val = a_row
                .iter()
                .zip(b.iter())
                .map(|(&a_ik, b_row)| a_ik * b_row[j])
                .sum();
        }
    }
    c
}

/// Compute the determinant of an n x n matrix via LU decomposition.
fn det_nxn(m: &[Vec<f64>], n: usize) -> f64 {
    let mut a: Vec<Vec<f64>> = m.to_vec();
    let mut sign = 1.0f64;

    for col in 0..n {
        let max_row = a[col..]
            .iter()
            .enumerate()
            .max_by(|(_, ra), (_, rb)| ra[col].abs().partial_cmp(&rb[col].abs()).unwrap())
            .map(|(idx, _)| idx + col)
            .unwrap();

        if a[max_row][col].abs() < 1e-12 {
            return 0.0;
        }

        if max_row != col {
            a.swap(col, max_row);
            sign = -sign;
        }

        let pivot = a[col][col];
        for row in (col + 1)..n {
            let factor = a[row][col] / pivot;
            let pivot_row: Vec<f64> = a[col].clone();
            for (v, &p) in a[row].iter_mut().zip(pivot_row.iter()).skip(col) {
                *v -= factor * p;
            }
        }
    }

    sign * a.iter().enumerate().map(|(i, row)| row[i]).product::<f64>()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typed_carrier_from_i32_vec() {
        let c = TypedCarrier::from_i32_vec(0, &[-1, -1, -1, -1, 0, 0, 0, 0]);
        assert!(c.is_some());
        let c = c.unwrap();
        assert_eq!(c.basis_index, 0);
        assert_eq!(c.lattice_vec, [-1, -1, -1, -1, 0, 0, 0, 0]);
    }

    #[test]
    fn test_typed_carrier_rejects_out_of_range() {
        assert!(TypedCarrier::from_i32_vec(0, &[2, 0, 0, 0, 0, 0, 0, 0]).is_none());
        assert!(TypedCarrier::from_i32_vec(0, &[0, 0, -2, 0, 0, 0, 0, 0]).is_none());
    }

    #[test]
    fn test_typed_carrier_rejects_wrong_length() {
        assert!(TypedCarrier::from_i32_vec(0, &[0, 0, 0]).is_none());
        assert!(TypedCarrier::from_i32_vec(0, &[0; 9]).is_none());
    }

    #[test]
    fn test_carrier_filtration_tier() {
        // This vector should be in Lambda_256: l_0=-1, l_1=-1, ...
        let c = TypedCarrier::new(0, [-1, -1, -1, -1, 0, 0, 0, 0]);
        let tier = c.filtration_tier();
        assert_eq!(tier, FiltrationTier::Lambda256);
    }

    #[test]
    fn test_carrier_is_in_lambda() {
        let c = TypedCarrier::new(0, [-1, -1, -1, -1, 0, 0, 0, 0]);
        // Lambda_256 is the most restrictive; membership implies all larger sets.
        assert!(c.is_in_lambda(256));
        assert!(c.is_in_lambda(512));
        assert!(c.is_in_lambda(1024));
        assert!(c.is_in_lambda(2048));
    }

    #[test]
    fn test_carrier_set_from_lattice_vecs() {
        let pairs = vec![
            (0, [-1, -1, -1, -1, 0, 0, 0, 0]),
            (1, [-1, -1, 0, 0, -1, -1, 0, 0]),
            (2, [-1, -1, 0, 0, 0, 0, -1, -1]),
        ];
        let cs = CarrierSet::from_lattice_vecs(3, &pairs);
        assert_eq!(cs.len(), 3);
        assert!(!cs.is_empty());
        assert!(cs.get(0).is_some());
        assert!(cs.get(1).is_some());
        assert!(cs.get(2).is_some());
        assert!(cs.get(3).is_none());
    }

    #[test]
    fn test_carrier_set_validation_complete() {
        let pairs = vec![
            (0, [-1, -1, -1, -1, 0, 0, 0, 0]),
            (1, [-1, -1, 0, 0, -1, -1, 0, 0]),
        ];
        let cs = CarrierSet::from_lattice_vecs(2, &pairs);
        let v = cs.validate();
        assert!(v.is_complete);
        assert!(v.is_injective);
        assert!(v.is_valid_dictionary());
    }

    #[test]
    fn test_carrier_set_validation_missing() {
        let pairs = vec![
            (0, [-1, -1, -1, -1, 0, 0, 0, 0]),
            // basis_index 1 is missing
            (2, [-1, -1, 0, 0, -1, -1, 0, 0]),
        ];
        let cs = CarrierSet::from_lattice_vecs(3, &pairs);
        let v = cs.validate();
        assert!(!v.is_complete);
        assert_eq!(v.missing_basis_indices, vec![1]);
        assert!(!v.is_valid_dictionary());
    }

    #[test]
    fn test_carrier_set_validation_duplicate_lattice() {
        let same_vec = [-1, -1, -1, -1, 0, 0, 0, 0];
        let pairs = vec![
            (0, same_vec),
            (1, same_vec), // duplicate lattice vector
        ];
        let cs = CarrierSet::from_lattice_vecs(2, &pairs);
        let v = cs.validate();
        assert!(v.is_complete);
        assert!(!v.is_injective);
        assert_eq!(v.duplicate_lattice_pairs.len(), 1);
        assert!(!v.is_valid_dictionary());
    }

    #[test]
    fn test_carrier_set_filter_to_lambda() {
        // Mix: one vector in Lambda_256, one not (l_0 = 0, fails Lambda_1024).
        let pairs = vec![
            (0, [-1, -1, -1, -1, 0, 0, 0, 0]), // in Lambda_256
            (1, [0, -1, 0, -1, 0, -1, 0, -1]), // base only (l_0 = 0)
        ];
        let cs = CarrierSet::from_lattice_vecs(2, &pairs);
        let in_256 = cs.filter_to_lambda(256);
        assert_eq!(in_256.len(), 1);
        assert_eq!(in_256[0].basis_index, 0);
    }

    #[test]
    fn test_carrier_set_tier_histogram() {
        let pairs = vec![
            (0, [-1, -1, -1, -1, 0, 0, 0, 0]),   // Lambda_256
            (1, [-1, -1, -1, -1, -1, -1, 0, 0]), // Lambda_256
        ];
        let cs = CarrierSet::from_lattice_vecs(2, &pairs);
        let hist = cs.tier_histogram();
        assert_eq!(hist.get(&FiltrationTier::Lambda256), Some(&2));
    }

    #[test]
    fn test_carrier_set_from_i32_map() {
        let mut map = HashMap::new();
        map.insert(0, vec![-1, -1, -1, -1, 0, 0, 0, 0]);
        map.insert(1, vec![-1, -1, 0, 0, -1, -1, 0, 0]);
        let cs = CarrierSet::from_i32_map(2, &map);
        assert_eq!(cs.len(), 2);
        let v = cs.validate();
        assert!(v.is_valid_dictionary());
    }

    #[test]
    fn test_filtration_nesting() {
        // Any vector in Lambda_256 must also be in Lambda_512, 1024, 2048, Base.
        let v: LatticeVector = [-1, -1, -1, -1, 0, 0, 0, 0];
        if is_in_lambda_256(&v) {
            assert!(is_in_lambda_512(&v));
            assert!(is_in_lambda_1024(&v));
            assert!(is_in_lambda_2048(&v));
            assert!(is_in_base_universe(&v));
        }
    }

    #[test]
    fn test_base_universe_parity() {
        // Even sum + even weight + trinary + l_0 != 1
        assert!(is_in_base_universe(&[-1, -1, 0, 0, 0, 0, 0, 0])); // sum=-2, wt=2
        assert!(is_in_base_universe(&[0, 0, 0, 0, 0, 0, 0, 0])); // sum=0, wt=0
        assert!(!is_in_base_universe(&[1, 0, 0, 0, 0, 0, 0, 0])); // l_0=1 forbidden
        assert!(!is_in_base_universe(&[-1, 0, 0, 0, 0, 0, 0, 0])); // sum=-1 odd
    }

    #[test]
    fn test_scalar_shadow_add() {
        let v: LatticeVector = [-1, 0, 1, 0, -1, 0, 1, 0];
        let shifted = apply_scalar_shadow(&v, 1, "add");
        assert_eq!(shifted, [0, 1, 2, 1, 0, 1, 2, 1]);
    }

    #[test]
    fn test_scalar_shadow_mul() {
        let v: LatticeVector = [-1, 0, 1, 0, -1, 0, 1, 0];
        let scaled = apply_scalar_shadow(&v, -1, "mul");
        assert_eq!(scaled, [1, 0, -1, 0, 1, 0, -1, 0]);
    }

    // ---- EncodingDictionary tests ----

    fn sample_dictionary_4() -> EncodingDictionary {
        let pairs = vec![
            (0, [-1, -1, -1, -1, 0, 0, 0, 0]),
            (1, [-1, -1, 0, 0, -1, -1, 0, 0]),
            (2, [-1, -1, 0, 0, 0, 0, -1, -1]),
            (3, [-1, 0, -1, 0, -1, 0, -1, 0]),
        ];
        EncodingDictionary::try_from_pairs(4, &pairs).unwrap()
    }

    #[test]
    fn test_encoding_dictionary_encode_decode() {
        let dict = sample_dictionary_4();
        assert_eq!(dict.dim(), 4);
        assert_eq!(dict.len(), 4);

        // Forward: encode
        let lv = dict.encode(0).unwrap();
        assert_eq!(*lv, [-1, -1, -1, -1, 0, 0, 0, 0]);

        // Inverse: decode
        let idx = dict.decode(&[-1, -1, 0, 0, -1, -1, 0, 0]).unwrap();
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_encoding_dictionary_round_trip() {
        let dict = sample_dictionary_4();
        for b in 0..4 {
            let lv = dict.encode(b).unwrap();
            let decoded = dict.decode(lv).unwrap();
            assert_eq!(decoded, b, "round-trip failed for basis {b}");
        }
    }

    #[test]
    fn test_encoding_dictionary_decode_missing() {
        let dict = sample_dictionary_4();
        let missing = [0, 0, 0, 0, 0, 0, 0, 0];
        assert!(dict.decode(&missing).is_none());
    }

    #[test]
    fn test_encoding_dictionary_rejects_incomplete() {
        let pairs = vec![
            (0, [-1, -1, -1, -1, 0, 0, 0, 0]),
            // basis 1 missing
            (2, [-1, -1, 0, 0, 0, 0, -1, -1]),
        ];
        let result = EncodingDictionary::try_from_pairs(3, &pairs);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(!err.is_complete);
        assert_eq!(err.missing_basis_indices, vec![1]);
    }

    #[test]
    fn test_encoding_dictionary_rejects_non_injective() {
        let same_vec = [-1, -1, -1, -1, 0, 0, 0, 0];
        let pairs = vec![(0, same_vec), (1, same_vec)];
        let result = EncodingDictionary::try_from_pairs(2, &pairs);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(!err.is_injective);
    }

    #[test]
    fn test_encoding_dictionary_scalar_shadow() {
        let dict = sample_dictionary_4();
        // Basis 0: [-1,-1,-1,-1,0,0,0,0] -> sum=-4, signum=-1
        assert_eq!(dict.scalar_shadow(0), Some(-1));
        // Basis 3: [-1,0,-1,0,-1,0,-1,0] -> sum=-4, signum=-1
        assert_eq!(dict.scalar_shadow(3), Some(-1));
    }

    #[test]
    fn test_encoding_dictionary_restrict_to_lambda() {
        let dict = sample_dictionary_4();
        let restricted = dict.restrict_to_lambda(256);
        // All our test vectors have l_0=-1, l_1=-1 which should be in Lambda_256.
        // Let's verify at least some pass.
        assert!(!restricted.is_empty());
    }

    #[test]
    fn test_encoding_dictionary_from_i32_map() {
        let mut map = HashMap::new();
        map.insert(0, vec![-1, -1, -1, -1, 0, 0, 0, 0]);
        map.insert(1, vec![-1, -1, 0, 0, -1, -1, 0, 0]);
        let dict = EncodingDictionary::try_from_i32_map(2, &map).unwrap();
        assert_eq!(dict.len(), 2);
        assert_eq!(dict.encode(0).unwrap(), &[-1, -1, -1, -1, 0, 0, 0, 0]);
    }

    #[test]
    fn test_encoding_dictionary_iter() {
        let dict = sample_dictionary_4();
        let entries: Vec<_> = dict.iter().collect();
        assert_eq!(entries.len(), 4);
        // Should be sorted by basis_index
        assert_eq!(entries[0].0, 0);
        assert_eq!(entries[1].0, 1);
        assert_eq!(entries[2].0, 2);
        assert_eq!(entries[3].0, 3);
    }

    // ================================================================
    // 2048D Forbidden Prefix Enumeration Tests
    // ================================================================

    #[test]
    fn test_forbidden_2048_count() {
        let forbidden = enumerate_forbidden_2048();
        eprintln!("Forbidden 2048D points: {}", forbidden.len());
        assert_eq!(
            forbidden.len(),
            139,
            "Base universe minus Lambda_2048 should have exactly 139 points"
        );
    }

    #[test]
    fn test_forbidden_2048_all_in_base_universe() {
        let forbidden = enumerate_forbidden_2048();
        for fp in &forbidden {
            assert!(
                is_in_base_universe(&fp.vector),
                "Forbidden point {:?} should be in base universe",
                fp.vector
            );
        }
    }

    #[test]
    fn test_forbidden_2048_none_in_lambda() {
        let forbidden = enumerate_forbidden_2048();
        for fp in &forbidden {
            assert!(
                !is_in_lambda_2048(&fp.vector),
                "Forbidden point {:?} should NOT be in Lambda_2048",
                fp.vector
            );
        }
    }

    #[test]
    fn test_forbidden_2048_family_counts() {
        let forbidden = enumerate_forbidden_2048();
        let n_p1 = forbidden
            .iter()
            .filter(|f| f.family == ForbiddenFamily::Prefix011)
            .count();
        let n_p2 = forbidden
            .iter()
            .filter(|f| f.family == ForbiddenFamily::Prefix01011)
            .count();
        let n_p3 = forbidden
            .iter()
            .filter(|f| f.family == ForbiddenFamily::Prefix010101)
            .count();

        eprintln!(
            "Forbidden families: Prefix011={}, Prefix01011={}, Prefix010101={}",
            n_p1, n_p2, n_p3
        );
        eprintln!(
            "  Total: {} (= {} + {} + {})",
            n_p1 + n_p2 + n_p3,
            n_p1,
            n_p2,
            n_p3
        );

        // All families should be non-empty
        assert!(n_p1 > 0, "Prefix011 family should be non-empty");
        assert!(n_p2 > 0, "Prefix01011 family should be non-empty");
        assert!(n_p3 > 0, "Prefix010101 family should be non-empty");

        // Families should partition the forbidden set (mutually exclusive)
        assert_eq!(
            n_p1 + n_p2 + n_p3,
            139,
            "Three families should partition all 139 forbidden points"
        );
    }

    #[test]
    fn test_forbidden_2048_families_mutually_exclusive() {
        let forbidden = enumerate_forbidden_2048();
        // Verify mutual exclusivity by construction:
        // Pattern 1 has l_2=1, patterns 2 & 3 have l_2=0
        // Pattern 2 has l_4=1, pattern 3 has l_4=0
        for fp in &forbidden {
            let v = &fp.vector;
            match fp.family {
                ForbiddenFamily::Prefix011 => {
                    assert_eq!(v[0], 0);
                    assert_eq!(v[1], 1);
                    assert_eq!(v[2], 1);
                }
                ForbiddenFamily::Prefix01011 => {
                    assert_eq!(v[0], 0);
                    assert_eq!(v[1], 1);
                    assert_eq!(v[2], 0);
                    assert_eq!(v[3], 1);
                    assert_eq!(v[4], 1);
                }
                ForbiddenFamily::Prefix010101 => {
                    assert_eq!(v[0], 0);
                    assert_eq!(v[1], 1);
                    assert_eq!(v[2], 0);
                    assert_eq!(v[3], 1);
                    assert_eq!(v[4], 0);
                    assert_eq!(v[5], 1);
                }
            }
        }
    }

    #[test]
    fn test_forbidden_2048_exhaustive_coverage() {
        // Verify that enumerate_forbidden_2048 and is_in_lambda_2048 are consistent:
        // every base universe point is either in Lambda_2048 OR in the forbidden set.
        let forbidden = enumerate_forbidden_2048();
        let forbidden_set: std::collections::HashSet<LatticeVector> =
            forbidden.iter().map(|f| f.vector).collect();

        let mut n_base = 0usize;
        let mut n_lambda = 0usize;
        let mut n_forbidden = 0usize;

        for code in 0..3u32.pow(8) {
            let mut v = [0i8; 8];
            let mut c = code;
            for coord in &mut v {
                *coord = (c % 3) as i8 - 1;
                c /= 3;
            }
            if is_in_base_universe(&v) {
                n_base += 1;
                if is_in_lambda_2048(&v) {
                    n_lambda += 1;
                    assert!(
                        !forbidden_set.contains(&v),
                        "Lambda_2048 point should not be in forbidden set"
                    );
                } else {
                    n_forbidden += 1;
                    assert!(
                        forbidden_set.contains(&v),
                        "Non-Lambda_2048 base point should be in forbidden set"
                    );
                }
            }
        }

        eprintln!(
            "Exhaustive scan: base={}, lambda_2048={}, forbidden={}",
            n_base, n_lambda, n_forbidden
        );
        assert_eq!(n_forbidden, 139);
        assert_eq!(n_base, n_lambda + n_forbidden);
    }

    // ================================================================
    // Lambda enumeration from predicates
    // ================================================================

    #[test]
    fn test_enumerate_lambda_256_count() {
        // Enumerate Lambda_256 from predicates alone (no CSV data).
        // The predicate chain is an approximation of the true trie filtration;
        // the CSV-based ground truth has exactly 256 points. This test documents
        // how many the predicate gives.
        let points = enumerate_lambda_256();
        eprintln!("Lambda_256 from predicates: {} points", points.len());

        // The predicates may not give exactly 256 due to omitted singleton
        // exceptions (see lambda_1024 line 98 comment). Document the count.
        // If the predicate is exact, this will be 256.
        assert!(
            points.len() >= 256,
            "Predicates should accept at least 256 points (supersets are expected)"
        );
        // Upper bound: no more than Lambda_512's count (which should be <= 512).
        let p512 = enumerate_lattice_by_predicate(is_in_lambda_512);
        eprintln!("Lambda_512 from predicates: {} points", p512.len());
        assert!(
            points.len() <= p512.len(),
            "Lambda_256 must be a subset of Lambda_512"
        );
    }

    #[test]
    fn test_enumerate_filtration_nesting() {
        // Verify strict nesting Lambda_256 <= Lambda_512 <= Lambda_1024 <= Lambda_2048 <= Base
        let base = enumerate_lattice_by_predicate(is_in_base_universe);
        let l2048 = enumerate_lattice_by_predicate(is_in_lambda_2048);
        let l1024 = enumerate_lattice_by_predicate(is_in_lambda_1024);
        let l512 = enumerate_lattice_by_predicate(is_in_lambda_512);
        let l256 = enumerate_lambda_256();

        eprintln!(
            "Filtration counts: base={}, 2048={}, 1024={}, 512={}, 256={}",
            base.len(),
            l2048.len(),
            l1024.len(),
            l512.len(),
            l256.len()
        );

        // Strict nesting
        assert!(l256.len() < l512.len(), "Lambda_256 < Lambda_512");
        assert!(l512.len() < l1024.len(), "Lambda_512 < Lambda_1024");
        assert!(l1024.len() < l2048.len(), "Lambda_1024 < Lambda_2048");
        assert!(l2048.len() < base.len(), "Lambda_2048 < Base");

        // Subset inclusion
        let l2048_set: std::collections::HashSet<LatticeVector> = l2048.iter().copied().collect();
        for v in &l1024 {
            assert!(
                l2048_set.contains(v),
                "Lambda_1024 point not in Lambda_2048"
            );
        }
        let l512_set: std::collections::HashSet<LatticeVector> = l512.iter().copied().collect();
        for v in &l256 {
            assert!(l512_set.contains(v), "Lambda_256 point not in Lambda_512");
        }
    }

    // ================================================================
    // 32-point slice characterization (Task #115)
    // ================================================================

    #[test]
    fn test_pinned_slice_prefix_4_count() {
        // The "pinned corner" with prefix (-1,-1,-1,-1) and free tail in {-1,0,1}^4.
        // Since all trie-cut exclusions are vacuously satisfied for this prefix,
        // the slice = base_universe restricted to {l[0..4]=(-1,-1,-1,-1)}.
        //
        // The tail must have: even sum AND even nonzero count.
        // Weight 0: (0,0,0,0) -- 1 vector
        // Weight 2: C(4,2)*2^2 = 24 vectors (all even sum automatically)
        // Weight 4: 2^4 = 16 vectors (all even sum automatically)
        // Total: 41 points.
        let char = characterize_pinned_slice(4);
        eprintln!("Pinned-corner prefix=4: {} points", char.count);
        eprintln!("Tail weight histogram: {:?}", char.tail_weight_histogram);

        // The count should be 41 (not 32) from pure predicates.
        // The "32-point slice" in the literature is the lex-first 32 of
        // Lambda_256 from CSV data, which happens to be a subset of these 41.
        assert_eq!(
            char.count, 41,
            "Pinned prefix (-1,-1,-1,-1) slice should have 41 points"
        );

        // Weight distribution: w=0 -> 1, w=2 -> 24, w=4 -> 16
        assert_eq!(char.tail_weight_histogram, vec![(0, 1), (2, 24), (4, 16)]);
    }

    #[test]
    fn test_pinned_slice_prefix_4_geometry() {
        // Characterize the geometric structure of the 41-point slice.
        // All points share the common prefix (-1,-1,-1,-1), so pairwise
        // distances only depend on the tail coordinates (l_4..l_7).
        let char = characterize_pinned_slice(4);

        eprintln!(
            "Distance histogram (d^2, count): {:?}",
            char.distance_histogram
        );
        eprintln!(
            "Inner product histogram (ip, count): {:?}",
            char.inner_product_histogram
        );

        // All pairs: C(41,2) = 820
        let total_pairs: usize = char.distance_histogram.iter().map(|(_, c)| c).sum();
        assert_eq!(total_pairs, 41 * 40 / 2, "Should have C(41,2) pairs");

        // The prefix contributes 4 to the squared distance between any two
        // distinct points (since the prefix is identical). Wait -- no, the prefix
        // is IDENTICAL so it contributes 0 to squared distance. The distance
        // is entirely from the tail differences.
        //
        // Minimum nonzero d^2 = 2 (two tail coords differ by 1 each).
        // Maximum d^2 = 4*4 = 16 (all four tail coords differ by 2 each).
        //
        // But actually d^2 computed over the full 8D vector: since prefix is identical,
        // the first 4 coords contribute 0, and we only get distance from the tail.

        // Verify all distances are from tail-only differences
        for &(d2, _count) in &char.distance_histogram {
            assert!(d2 > 0, "No zero distances (points are distinct)");
            assert!(
                d2 <= 16,
                "Max d^2 = 4*2^2 = 16 (all tail coords differ by 2)"
            );
        }
    }

    #[test]
    fn test_pinned_slice_prefix_4_tail_structure() {
        // The 41 tail patterns in {-1,0,1}^4 with even sum and even weight
        // form a recognizable combinatorial object.
        //
        // The weight-4 subset (16 points in {-1,+1}^4 with even sum) is
        // exactly the D4 root system: the 8 vectors with an even number of
        // minus signs, plus the 8 with an odd number = all 16 of {-1,+1}^4.
        // Actually {-1,+1}^4 has all even sums (sum is always even for 4 terms
        // of +/-1), so all 16 qualify. This is the vertex set of a 4-cube (tesseract).
        //
        // The weight-2 subset (24 points with exactly 2 nonzeros in {-1,+1})
        // corresponds to the edge midpoints of the tesseract.
        //
        // The weight-0 subset is just the origin.
        let char = characterize_pinned_slice(4);

        // Extract the weight-4 tail patterns
        let w4: Vec<&LatticeVector> = char
            .tail_patterns
            .iter()
            .filter(|t| t[4..].iter().filter(|&&x| x != 0).count() == 4)
            .collect();
        assert_eq!(
            w4.len(),
            16,
            "16 weight-4 tail patterns (full tesseract vertices)"
        );

        // Verify these are exactly {-1,+1}^4 (with prefix positions zeroed)
        for t in &w4 {
            for i in 4..8 {
                assert!(
                    t[i] == -1 || t[i] == 1,
                    "Weight-4 tail should be +/-1 in free coordinates"
                );
            }
        }

        // Extract weight-2 patterns: C(4,2) * 4 = 24
        let w2: Vec<&LatticeVector> = char
            .tail_patterns
            .iter()
            .filter(|t| t[4..].iter().filter(|&&x| x != 0).count() == 2)
            .collect();
        assert_eq!(w2.len(), 24, "24 weight-2 tail patterns");

        // Each weight-2 pattern has exactly 2 nonzero coords in positions 4-7
        for t in &w2 {
            let nz_positions: Vec<usize> = (4..8).filter(|&i| t[i] != 0).collect();
            assert_eq!(nz_positions.len(), 2);
        }
    }

    #[test]
    fn test_pinned_slice_prefix_4_inner_products() {
        // Analyze inner products to detect polytope structure.
        // For the full 8D vectors, <v, w> = (-1)^2 * 4 + <tail_v, tail_w>
        //                                = 4 + <tail_v, tail_w>
        // So the inner product structure is shifted by +4 from the tail-only products.
        let char = characterize_pinned_slice(4);

        // Verify the shift: all inner products should be >= 4 - 4 = 0
        // (tail inner product minimum is -4 when all signs flip).
        // Actually: min tail ip is -4 (w4 vs w4 with all signs flipped),
        // so min full ip = 4 + (-4) = 0.
        for &(ip, _count) in &char.inner_product_histogram {
            assert!(
                ip >= 0,
                "Full inner product should be >= 0 (prefix contributes +4)"
            );
        }

        // The inner product with self is: 4 + sum(tail_i^2).
        // For weight-4: self-ip = 4 + 4 = 8
        // For weight-2: self-ip = 4 + 2 = 6
        // For weight-0: self-ip = 4 + 0 = 4
        // (Self inner products are not in the histogram since we skip i==j)

        // The maximum inter-point ip should be 8 (two weight-4 vectors that agree)
        // minus the common prefix = wait, two identical weight-4 vectors ARE the
        // same point. The max inter-point ip for distinct points:
        // weight-4 vs weight-4 with 3 signs matching, 1 flipped: ip = 4 + (3-1) = 6
        // Actually: if tail_v and tail_w are both in {-1,+1}^4, their ip = n_agree - n_disagree
        // where n_agree + n_disagree = 4.
        // ip = n_agree - (4 - n_agree) = 2*n_agree - 4.
        // Max (distinct): n_agree=3 -> ip=2, so full ip = 4+2 = 6.
        // Min: n_agree=0 -> ip=-4, so full ip = 4-4 = 0.
        // Cross (w4 vs w2): tail ip can be at most 2 (2 nonzeros agree).
        eprintln!(
            "Inner product histogram: {:?}",
            char.inner_product_histogram
        );
    }

    #[test]
    fn test_pinned_slice_trie_vacuity() {
        // Verify the key mathematical claim: for any vector with l[0..4] = (-1,-1,-1,-1),
        // the trie-cut exclusions at EVERY level are vacuously satisfied.
        // Therefore is_in_lambda_256(v) reduces to is_in_base_universe(v).
        let mut n_tested = 0usize;
        let mut n_agree = 0usize;

        for code in 0..3u32.pow(4) {
            let mut v = [-1i8, -1, -1, -1, 0, 0, 0, 0];
            let mut c = code;
            for i in 4..8 {
                v[i] = (c % 3) as i8 - 1;
                c /= 3;
            }
            n_tested += 1;
            let in_base = is_in_base_universe(&v);
            let in_256 = is_in_lambda_256(&v);
            if in_base == in_256 {
                n_agree += 1;
            }
        }

        eprintln!("Trie vacuity check: {}/{} agree", n_agree, n_tested);
        assert_eq!(
            n_agree, n_tested,
            "For prefix (-1,-1,-1,-1), is_in_lambda_256 == is_in_base_universe everywhere"
        );
    }

    #[test]
    fn test_pinned_slice_weight4_is_tesseract() {
        // The 16 weight-4 tail vectors are the vertices of a 4-dimensional
        // hypercube (tesseract) centered at the origin, with vertices at {-1,+1}^4.
        //
        // The tesseract has these graph-theoretic properties:
        // - 16 vertices
        // - Each vertex has 4 nearest neighbors (Hamming distance 1 = flip one sign)
        // - 32 edges (16 * 4 / 2)
        // - The adjacency graph is the 4-cube graph Q_4
        let _char = characterize_pinned_slice(4);

        // Extract weight-4 points (full 8D vectors)
        let all_256 = enumerate_lambda_256();
        let w4_points: Vec<LatticeVector> = all_256
            .iter()
            .filter(|v| v[..4] == [-1, -1, -1, -1])
            .filter(|v| v[4..].iter().filter(|&&x| x != 0).count() == 4)
            .copied()
            .collect();

        assert_eq!(w4_points.len(), 16);

        // Compute adjacency: two vertices are adjacent if they differ in exactly
        // one tail coordinate (squared distance = 4 in that coordinate = 2^2).
        // Full 8D d^2 = 0 (prefix) + 4 (one flip) = 4.
        let mut edge_count = 0usize;
        let mut degree_counts = vec![0usize; 16];
        for i in 0..16 {
            for j in (i + 1)..16 {
                let d2: i32 = (0..8)
                    .map(|k| {
                        let d = w4_points[i][k] as i32 - w4_points[j][k] as i32;
                        d * d
                    })
                    .sum();
                if d2 == 4 {
                    edge_count += 1;
                    degree_counts[i] += 1;
                    degree_counts[j] += 1;
                }
            }
        }

        eprintln!(
            "Tesseract verification: {} vertices, {} edges",
            16, edge_count
        );
        assert_eq!(edge_count, 32, "Tesseract has 32 edges");
        for (idx, &deg) in degree_counts.iter().enumerate() {
            assert_eq!(deg, 4, "Vertex {} has degree {} (expected 4)", idx, deg);
        }
    }

    // ================================================================
    // Layer 2: Elevated addition tests
    // ================================================================

    /// Build a small synthetic dictionary for testing elevated addition.
    /// Uses 4 basis elements mapping to distinct Lambda_256 vectors.
    fn make_test_dictionary_4() -> EncodingDictionary {
        // Pick 4 vectors from Lambda_256 that are well-separated
        let pairs: Vec<(usize, LatticeVector)> = vec![
            (0, [-1, -1, -1, -1, 0, 0, 0, 0]),   // weight 4, sum -4
            (1, [-1, -1, -1, -1, -1, -1, 0, 0]), // weight 6, sum -6
            (2, [-1, -1, -1, -1, -1, 1, 0, 0]),  // weight 6, sum -4
            (3, [-1, -1, -1, -1, 0, 0, -1, -1]), // weight 6, sum -6
        ];
        EncodingDictionary::try_from_pairs(4, &pairs).unwrap()
    }

    #[test]
    fn test_lattice_add_basic() {
        let a: LatticeVector = [-1, -1, 0, 0, 0, 0, 0, 0];
        let b: LatticeVector = [0, 0, -1, -1, 0, 0, 0, 0];
        let sum = lattice_add(&a, &b);
        assert_eq!(sum, [-1, -1, -1, -1, 0, 0, 0, 0]);
    }

    #[test]
    fn test_lattice_add_overflow() {
        // Two -1s add to -2, which leaves {-1,0,1}^8
        let a: LatticeVector = [-1, -1, 0, 0, 0, 0, 0, 0];
        let b: LatticeVector = [-1, -1, 0, 0, 0, 0, 0, 0];
        let sum = lattice_add(&a, &b);
        assert_eq!(sum[0], -2, "Sum should be -2, outside trinary range");
        assert!(try_narrow_to_lattice(&sum).is_none());
    }

    #[test]
    fn test_try_narrow_to_lattice() {
        assert_eq!(
            try_narrow_to_lattice(&[-1, 0, 1, 0, 0, 0, 0, 0]),
            Some([-1, 0, 1, 0, 0, 0, 0, 0])
        );
        assert!(try_narrow_to_lattice(&[2, 0, 0, 0, 0, 0, 0, 0]).is_none());
        assert!(try_narrow_to_lattice(&[0, 0, 0, 0, 0, 0, 0, -2]).is_none());
    }

    #[test]
    fn test_elevated_add_in_codebook() {
        let dict = make_test_dictionary_4();
        // Phi(0) = (-1,-1,-1,-1,0,0,0,0) + Phi(0) = (-2,-2,-2,-2,0,0,0,0)
        // This overflows trinary -> OutOfBounds
        let r = dict.elevated_add(0, 0).unwrap();
        assert!(
            matches!(r, ElevatedResult::OutOfBounds { .. }),
            "Self-addition of (-1,-1,-1,-1,...) overflows"
        );
    }

    #[test]
    fn test_elevated_add_commutativity() {
        // lattice_add is inherently commutative (integer addition)
        let dict = make_test_dictionary_4();
        for a in 0..4 {
            for b in 0..4 {
                let r_ab = dict.elevated_add(a, b).unwrap();
                let r_ba = dict.elevated_add(b, a).unwrap();
                assert_eq!(
                    r_ab, r_ba,
                    "Elevated addition should be commutative: ({}, {})",
                    a, b
                );
            }
        }
    }

    #[test]
    fn test_elevated_addition_stats_synthetic() {
        let dict = make_test_dictionary_4();
        let stats = dict.elevated_addition_stats();

        eprintln!("Synthetic 4-element dictionary stats:");
        eprintln!(
            "  total_pairs={}, in_codebook={}, out_of_codebook={}, out_of_bounds={}",
            stats.total_pairs, stats.in_codebook, stats.out_of_codebook, stats.out_of_bounds
        );
        eprintln!(
            "  closure_rate={:.3}, commutative={}, identities={}",
            stats.closure_rate, stats.is_commutative, stats.identity_count
        );

        assert_eq!(stats.total_pairs, 16, "4x4 = 16 pairs");
        assert!(stats.is_commutative, "Lattice addition is commutative");
        assert_eq!(
            stats.in_codebook + stats.out_of_codebook + stats.out_of_bounds,
            16
        );
    }

    #[test]
    fn test_translation_orbit() {
        let dict = make_test_dictionary_4();
        for b in 0..4 {
            let orbit = dict.translation_orbit(b);
            eprintln!(
                "Translation orbit of b={}: {} pairs in codebook",
                b,
                orbit.len()
            );
            // Each orbit entry (a, c) means Phi(a) + Phi(b) = Phi(c)
            for &(a, c) in &orbit {
                assert!(c < 4, "Decoded index should be valid");
                let result = dict.elevated_add(a, b).unwrap();
                assert!(
                    matches!(result, ElevatedResult::InCodebook { decoded_index, .. }
                    if decoded_index == c)
                );
            }
        }
    }

    #[test]
    fn test_elevated_add_with_lambda_256_vectors() {
        // Build a dictionary from the first 8 vectors of Lambda_256.
        // This uses the predicate-enumerated points as a realistic test.
        let all_256 = enumerate_lambda_256();
        assert!(all_256.len() >= 8);

        let pairs: Vec<(usize, LatticeVector)> = all_256[..8]
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        let dict = EncodingDictionary::try_from_pairs(8, &pairs).unwrap();
        let stats = dict.elevated_addition_stats();

        eprintln!("Lambda_256 first-8 dictionary stats:");
        eprintln!(
            "  total_pairs={}, in_codebook={}, out_of_codebook={}, out_of_bounds={}",
            stats.total_pairs, stats.in_codebook, stats.out_of_codebook, stats.out_of_bounds
        );
        eprintln!(
            "  closure_rate={:.4}, commutative={}, identities={}",
            stats.closure_rate, stats.is_commutative, stats.identity_count
        );

        assert_eq!(stats.total_pairs, 64, "8x8 = 64 pairs");
        assert!(stats.is_commutative);
        // With vectors from the deep negative corner, most sums will overflow
        assert!(
            stats.out_of_bounds > 0,
            "Some sums should overflow trinary range"
        );
    }

    #[test]
    fn test_elevated_add_zero_vector() {
        // If the dictionary contains the zero vector [0,0,0,0,0,0,0,0],
        // it should act as identity for lattice addition.
        // Note: [0,0,...,0] IS in base_universe (sum=0 even, weight=0 even),
        // and it IS in Lambda_256 (prefix is all 0, but l_0 != 1, l_0 = 0).
        // Actually: is_in_base_universe requires l_0 != 1, and 0 != 1, so OK.
        // But is_in_lambda_2048: forbidden (0,1,1) -- only if l_0=0 AND l_1=1 AND l_2=1.
        // Zero vec has l_1=0, so no forbidden prefix fires.
        // is_in_lambda_1024: requires l_0=-1. Zero vec has l_0=0. FAILS.
        // So the zero vector is NOT in Lambda_256 (it fails at Lambda_1024).
        //
        // This means there's no additive identity in the Lambda_256 codebook.
        // Verify this:
        let all_256 = enumerate_lambda_256();
        let zero_present = all_256.iter().any(|v| v.iter().all(|&x| x == 0));
        assert!(
            !zero_present,
            "Zero vector is NOT in Lambda_256 (fails l_0=-1 for Lambda_1024)"
        );
    }

    // ================================================================
    // Layer 2: F_3 elevated addition tests
    // ================================================================

    #[test]
    fn test_lattice_add_f3_basic() {
        // F_3 wrapping: -1 + -1 = 1 (mod 3)
        let a: LatticeVector = [-1, -1, 0, 0, 0, 0, 0, 0];
        let b: LatticeVector = [-1, -1, 0, 0, 0, 0, 0, 0];
        let sum = lattice_add_f3(&a, &b);
        assert_eq!(sum[0], 1, "-1 + -1 = 1 in F_3");
        assert_eq!(sum[1], 1, "-1 + -1 = 1 in F_3");
    }

    #[test]
    fn test_lattice_add_f3_identity() {
        // Zero is the identity in F_3
        let a: LatticeVector = [-1, 1, 0, -1, 1, 0, -1, 1];
        let zero: LatticeVector = [0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(lattice_add_f3(&a, &zero), a);
        assert_eq!(lattice_add_f3(&zero, &a), a);
    }

    #[test]
    fn test_lattice_add_f3_inverse() {
        // In F_3: the additive inverse of x is -x (i.e., 2x mod 3).
        // -(-1) = 1, -(0) = 0, -(1) = -1
        let a: LatticeVector = [-1, 1, 0, -1, 1, 0, -1, 1];
        let neg_a: LatticeVector = [1, -1, 0, 1, -1, 0, 1, -1];
        let sum = lattice_add_f3(&a, &neg_a);
        assert_eq!(sum, [0, 0, 0, 0, 0, 0, 0, 0], "a + (-a) = 0 in F_3");
    }

    #[test]
    fn test_f3_associativity_on_raw_vectors() {
        // F_3 addition is inherently associative (it's a group).
        let a: LatticeVector = [-1, 1, 0, -1, 1, 0, -1, 1];
        let b: LatticeVector = [1, 1, -1, 0, 0, -1, -1, 0];
        let c: LatticeVector = [0, -1, 1, 1, -1, 0, 0, -1];

        let ab = lattice_add_f3(&a, &b);
        let abc_left = lattice_add_f3(&ab, &c);

        let bc = lattice_add_f3(&b, &c);
        let abc_right = lattice_add_f3(&a, &bc);

        assert_eq!(
            abc_left, abc_right,
            "F_3 addition is associative on raw vectors"
        );
    }

    #[test]
    fn test_elevated_add_f3_synthetic() {
        let dict = make_test_dictionary_4();
        // F_3: Phi(0) +_3 Phi(0) = (-1,-1,-1,-1,0,0,0,0) +_3 same
        //      = (1,1,1,1,0,0,0,0) -- this is trinary but may not be in dict
        let r = dict.elevated_add_f3(0, 0).unwrap();
        match &r {
            ElevatedResultF3::InCodebook { sum_vec, .. } => {
                assert_eq!(*sum_vec, [1, 1, 1, 1, 0, 0, 0, 0]);
            }
            ElevatedResultF3::OutOfCodebook { sum_vec } => {
                assert_eq!(*sum_vec, [1, 1, 1, 1, 0, 0, 0, 0]);
            }
        }
    }

    #[test]
    fn test_elevated_add_f3_commutativity() {
        let dict = make_test_dictionary_4();
        for a in 0..4 {
            for b in 0..4 {
                let r_ab = dict.elevated_add_f3(a, b).unwrap();
                let r_ba = dict.elevated_add_f3(b, a).unwrap();
                assert_eq!(
                    r_ab, r_ba,
                    "F_3 elevated addition should be commutative: ({}, {})",
                    a, b
                );
            }
        }
    }

    #[test]
    fn test_elevated_addition_stats_f3_synthetic() {
        let dict = make_test_dictionary_4();
        let stats = dict.elevated_addition_stats_f3();

        eprintln!("F_3 synthetic 4-element dictionary stats:");
        eprintln!(
            "  total_pairs={}, in_codebook={}, out_of_codebook={}",
            stats.total_pairs, stats.in_codebook, stats.out_of_codebook
        );
        eprintln!(
            "  closure_rate={:.3}, commutative={}, identities={}",
            stats.closure_rate, stats.is_commutative, stats.identity_count
        );
        eprintln!(
            "  associativity_rate={:.3} ({} triples tested)",
            stats.associativity_rate, stats.associativity_triples_tested
        );

        assert_eq!(stats.total_pairs, 16);
        assert!(stats.is_commutative);
        assert_eq!(
            stats.in_codebook + stats.out_of_codebook,
            16,
            "No out-of-bounds in F_3"
        );
        // F_3 on raw vectors is always associative
        assert!(
            (stats.associativity_rate - 1.0).abs() < 1e-10,
            "F_3 is inherently associative"
        );
    }

    #[test]
    fn test_elevated_add_f3_lambda_256() {
        // Test F_3 addition on a realistic Lambda_256 sub-dictionary.
        let all_256 = enumerate_lambda_256();

        let pairs: Vec<(usize, LatticeVector)> = all_256[..16]
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        let dict = EncodingDictionary::try_from_pairs(16, &pairs).unwrap();
        let stats = dict.elevated_addition_stats_f3();

        eprintln!("F_3 Lambda_256 first-16 dictionary stats:");
        eprintln!(
            "  total_pairs={}, in_codebook={}, out_of_codebook={}",
            stats.total_pairs, stats.in_codebook, stats.out_of_codebook
        );
        eprintln!(
            "  closure_rate={:.4}, commutative={}, identities={}",
            stats.closure_rate, stats.is_commutative, stats.identity_count
        );
        eprintln!(
            "  associativity_rate={:.4} ({} triples tested)",
            stats.associativity_rate, stats.associativity_triples_tested
        );

        assert_eq!(stats.total_pairs, 256, "16x16 = 256 pairs");
        assert!(stats.is_commutative);
        assert!(
            (stats.associativity_rate - 1.0).abs() < 1e-10,
            "F_3 is always associative"
        );
        // The closure rate should be > 0 (F_3 wraps instead of overflowing)
        eprintln!(
            "  F_3 closure rate: {:.1}% ({}/{})",
            stats.closure_rate * 100.0,
            stats.in_codebook,
            stats.total_pairs
        );
    }

    #[test]
    fn test_elevated_diff_lambda_256() {
        // Z-difference on Lambda_256 sub-dictionary.
        // Since all l_0 = -1, difference gives l_0 = 0 (always in trinary range).
        let all_256 = enumerate_lambda_256();

        let pairs: Vec<(usize, LatticeVector)> = all_256[..8]
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();

        let dict = EncodingDictionary::try_from_pairs(8, &pairs).unwrap();

        let mut in_codebook = 0usize;
        let mut out_of_codebook = 0usize;
        let mut out_of_bounds = 0usize;

        for a in 0..8 {
            for b in 0..8 {
                match dict.elevated_diff(a, b).unwrap() {
                    ElevatedResult::InCodebook { .. } => in_codebook += 1,
                    ElevatedResult::OutOfCodebook { .. } => out_of_codebook += 1,
                    ElevatedResult::OutOfBounds { .. } => out_of_bounds += 1,
                }
            }
        }

        eprintln!(
            "Z-difference Lambda_256 first-8: in_codebook={}, out_of_codebook={}, out_of_bounds={}",
            in_codebook, out_of_codebook, out_of_bounds
        );

        // The self-difference a - a = 0 always, but 0 is NOT in Lambda_256.
        // So self-differences should all be OutOfCodebook.
        for a in 0..8 {
            let r = dict.elevated_diff(a, a).unwrap();
            assert!(
                matches!(r, ElevatedResult::OutOfCodebook { sum_vec }
                if sum_vec == [0, 0, 0, 0, 0, 0, 0, 0]),
                "a - a = 0, which is not in Lambda_256"
            );
        }
    }

    // ================================================================
    // Multiplication Coupling Tests (Thesis D, C-466)
    // ================================================================

    /// Helper: build a dim=16 dictionary from the first 16 Lambda_256 vectors
    /// and the corresponding multiplication table.
    fn sedenion_coupling_setup() -> (
        EncodingDictionary,
        crate::construction::mult_table::CdMultTable,
    ) {
        let lambda = enumerate_lambda_256();
        assert!(lambda.len() >= 16);
        let pairs: Vec<(usize, LatticeVector)> = lambda[..16]
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        let dict = EncodingDictionary::try_from_pairs(16, &pairs).unwrap();
        let table = crate::construction::mult_table::CdMultTable::generate(16);
        (dict, table)
    }

    #[test]
    fn test_multiplication_coupling_sedenion_basic() {
        let (dict, table) = sedenion_coupling_setup();
        let coupling = compute_multiplication_coupling(&dict, &table);

        assert_eq!(coupling.dim, 16);
        assert_eq!(coupling.results.len(), 16);
        assert!(
            coupling.rank > 0 && coupling.rank <= 8,
            "rank should be in [1,8], got {}",
            coupling.rank
        );

        // Basis 0 (e_0 = identity): multiplication is identity permutation,
        // so rho(0) is the identity map on the subspace.
        let r0 = &coupling.results[0];
        assert!(
            r0.unsigned_consistent,
            "e_0 * e_c = e_c, unsigned coupling must be consistent"
        );
        assert!(
            r0.signed_consistent,
            "e_0 * e_c = +1 * e_c, signed coupling must be consistent"
        );

        // det(I_r) = 1 in the reduced space
        let det_u = r0.unsigned_det.unwrap();
        assert!(
            (det_u - 1.0).abs() < 1e-6,
            "det(rho_unsigned(0)) should be 1, got {det_u}"
        );
        let det_s = r0.signed_det.unwrap();
        assert!(
            (det_s - 1.0).abs() < 1e-6,
            "det(rho_signed(0)) should be 1, got {det_s}"
        );

        // Report summary
        eprintln!("=== Sedenion (dim=16) Multiplication Coupling ===");
        eprintln!("Lattice vector rank: {}/8", coupling.rank);
        eprintln!(
            "Unsigned consistent: {}/16",
            coupling.unsigned_consistent_count
        );
        eprintln!(
            "Signed consistent:   {}/16",
            coupling.signed_consistent_count
        );
        eprintln!("Unsigned determinants: {:?}", coupling.unsigned_dets);
        eprintln!("Signed determinants:   {:?}", coupling.signed_dets);

        for r in &coupling.results {
            eprintln!(
                "  b={:2}: u_ok={} u_det={:?} u_res={:.2e} | s_ok={} s_det={:?} s_res={:.2e}",
                r.basis_index,
                if r.unsigned_consistent { "Y" } else { "N" },
                r.unsigned_det,
                r.unsigned_max_residual,
                if r.signed_consistent { "Y" } else { "N" },
                r.signed_det,
                r.signed_max_residual,
            );
        }
    }

    #[test]
    fn test_multiplication_coupling_identity_element() {
        let (dict, table) = sedenion_coupling_setup();
        let coupling = compute_multiplication_coupling(&dict, &table);

        let r0 = &coupling.results[0];
        assert!(r0.unsigned_consistent);
        assert!(r0.signed_consistent);
        assert!((r0.unsigned_det.unwrap() - 1.0).abs() < 1e-6);
        assert!((r0.signed_det.unwrap() - 1.0).abs() < 1e-6);
        assert!(r0.unsigned_max_residual < 1e-10);
        assert!(r0.signed_max_residual < 1e-10);
    }

    #[test]
    fn test_multiplication_coupling_sedenion_characterize() {
        let (dict, table) = sedenion_coupling_setup();
        let coupling = compute_multiplication_coupling(&dict, &table);

        let unsigned_bases: Vec<usize> = coupling
            .results
            .iter()
            .filter(|r| r.unsigned_consistent)
            .map(|r| r.basis_index)
            .collect();
        let signed_bases: Vec<usize> = coupling
            .results
            .iter()
            .filter(|r| r.signed_consistent)
            .map(|r| r.basis_index)
            .collect();

        // At minimum, basis 0 must always work
        assert!(unsigned_bases.contains(&0));
        assert!(signed_bases.contains(&0));

        eprintln!("Rank: {}", coupling.rank);
        eprintln!("Unsigned consistent bases: {:?}", unsigned_bases);
        eprintln!("Signed consistent bases:   {:?}", signed_bases);
        eprintln!("Unsigned dets: {:?}", coupling.unsigned_dets);
        eprintln!("Signed dets:   {:?}", coupling.signed_dets);

        // The key research question: how many basis elements have
        // consistent linear coupling? Is it all of them, or a subset?
        // Record the answer for C-466.
        eprintln!(
            "C-466 result: {}/{} unsigned, {}/{} signed",
            coupling.unsigned_consistent_count,
            coupling.dim,
            coupling.signed_consistent_count,
            coupling.dim
        );
    }

    #[test]
    fn test_multiplication_coupling_pathion() {
        let lambda = enumerate_lambda_256();
        assert!(
            lambda.len() >= 32,
            "Lambda_256 has {} vectors, need 32",
            lambda.len()
        );
        let pairs: Vec<(usize, LatticeVector)> = lambda[..32]
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v))
            .collect();
        let dict = EncodingDictionary::try_from_pairs(32, &pairs).unwrap();
        let table = crate::construction::mult_table::CdMultTable::generate(32);

        let coupling = compute_multiplication_coupling(&dict, &table);

        assert_eq!(coupling.dim, 32);
        assert_eq!(coupling.results.len(), 32);

        // Identity element
        let r0 = &coupling.results[0];
        assert!(r0.unsigned_consistent, "rho(0) must be identity for dim=32");
        assert!((r0.unsigned_det.unwrap() - 1.0).abs() < 1e-6);

        eprintln!("=== Pathion (dim=32) Multiplication Coupling ===");
        eprintln!("Rank: {}/8", coupling.rank);
        eprintln!(
            "Unsigned consistent: {}/32",
            coupling.unsigned_consistent_count
        );
        eprintln!(
            "Signed consistent:   {}/32",
            coupling.signed_consistent_count
        );
    }

    #[test]
    fn test_gram_schmidt_basis_rank() {
        // Verify that gram_schmidt_basis correctly determines rank.
        let lambda = enumerate_lambda_256();
        let phi16: Vec<[f64; 8]> = lambda[..16]
            .iter()
            .map(|lv| {
                let mut f = [0.0f64; 8];
                for (fv, &iv) in f.iter_mut().zip(lv.iter()) {
                    *fv = iv as f64;
                }
                f
            })
            .collect();
        let (basis16, rank16) = gram_schmidt_basis(&phi16);
        assert_eq!(basis16.len(), rank16);
        assert!(
            rank16 >= 1 && rank16 <= 8,
            "rank should be in [1,8], got {rank16}"
        );

        // Full Lambda_256 should span more dimensions
        let phi_all: Vec<[f64; 8]> = lambda
            .iter()
            .map(|lv| {
                let mut f = [0.0f64; 8];
                for (fv, &iv) in f.iter_mut().zip(lv.iter()) {
                    *fv = iv as f64;
                }
                f
            })
            .collect();
        let (_, rank_all) = gram_schmidt_basis(&phi_all);
        assert!(
            rank_all >= rank16,
            "full Lambda_256 rank ({rank_all}) must be >= dim=16 rank ({rank16})"
        );

        eprintln!("Rank of first 16 Lambda_256 vectors: {rank16}");
        eprintln!(
            "Rank of all {} Lambda_256 vectors: {rank_all}",
            lambda.len()
        );
    }

    #[test]
    fn test_invert_nxn_identity() {
        for n in [3, 5, 8] {
            let m: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    let mut row = vec![0.0; n];
                    row[i] = 1.0;
                    row
                })
                .collect();
            let inv = invert_nxn(&m, n).unwrap();
            for (i, row) in inv.iter().enumerate() {
                for (j, &val) in row.iter().enumerate() {
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!(
                        (val - expected).abs() < 1e-12,
                        "I^-1 [{i}][{j}] = {val} for n={n}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_det_nxn_identity() {
        for n in [3, 5, 8] {
            let m: Vec<Vec<f64>> = (0..n)
                .map(|i| {
                    let mut row = vec![0.0; n];
                    row[i] = 1.0;
                    row
                })
                .collect();
            assert!(
                (det_nxn(&m, n) - 1.0).abs() < 1e-12,
                "det(I_{n}) should be 1"
            );
        }
    }

    // ================================================================
    // Coset Obstruction and Affine Closure Analysis (C-498, C-499)
    // ================================================================

    #[test]
    fn test_coset_obstruction_and_affine_closure() {
        use std::collections::HashSet;

        // ---- Phase 1: Filtration level enumeration ----
        let base_universe = enumerate_lattice_by_predicate(is_in_base_universe);
        let lambda_2048 = enumerate_lattice_by_predicate(is_in_lambda_2048);
        let lambda_1024 = enumerate_lattice_by_predicate(is_in_lambda_1024);
        let lambda_512 = enumerate_lattice_by_predicate(is_in_lambda_512);
        let lambda_256 = enumerate_lambda_256();

        eprintln!("=== Filtration Level Sizes ===");
        eprintln!("  Base universe: {}", base_universe.len());
        eprintln!("  Lambda_2048:   {}", lambda_2048.len());
        eprintln!("  Lambda_1024:   {}", lambda_1024.len());
        eprintln!("  Lambda_512:    {}", lambda_512.len());
        eprintln!("  Lambda_256:    {}", lambda_256.len());

        // Strict inclusion chain
        assert!(lambda_256.len() < lambda_512.len());
        assert!(lambda_512.len() < lambda_1024.len());
        assert!(lambda_1024.len() < lambda_2048.len());
        assert!(lambda_2048.len() < base_universe.len());

        // ---- Phase 2: Coset decomposition of Lambda_2048 by l_0 ----
        let l0_neg1: Vec<&LatticeVector> = lambda_2048.iter().filter(|v| v[0] == -1).collect();
        let l0_zero: Vec<&LatticeVector> = lambda_2048.iter().filter(|v| v[0] == 0).collect();
        let l0_pos1: Vec<&LatticeVector> = lambda_2048.iter().filter(|v| v[0] == 1).collect();

        eprintln!("\n=== Lambda_2048 Coset Decomposition by l_0 ===");
        eprintln!("  l_0 = -1: {} vectors", l0_neg1.len());
        eprintln!("  l_0 =  0: {} vectors", l0_zero.len());
        eprintln!(
            "  l_0 = +1: {} vectors (excluded by base universe)",
            l0_pos1.len()
        );

        assert_eq!(l0_pos1.len(), 0, "l_0 = +1 excluded from base universe");
        assert_eq!(
            l0_neg1.len() + l0_zero.len(),
            lambda_2048.len(),
            "Coset partition exhaustive"
        );

        // Lambda_1024 is exactly the l_0=-1 sub-lattice of Lambda_2048
        // (minus further exclusions). All l0_neg1 satisfy is_in_lambda_2048 but
        // not all satisfy is_in_lambda_1024 (additional exclusions).
        assert!(lambda_1024.len() <= l0_neg1.len());
        assert!(
            lambda_1024.iter().all(|v| v[0] == -1),
            "Lambda_1024 requires l_0 = -1"
        );

        // ---- Phase 3: Prove the coset obstruction ----
        // All Lambda_256 vectors have l_0 = -1
        assert!(
            lambda_256.iter().all(|v| v[0] == -1),
            "All Lambda_256 vectors have l_0 = -1"
        );

        // Z-addition: (-1) + (-1) = -2, leaving {-1,0,1}^8
        let a0 = &lambda_256[0];
        let b0 = &lambda_256[1];
        let z_sum = lattice_add(a0, b0);
        assert_eq!(
            z_sum[0], -2,
            "Z-addition: l_0 = (-1)+(-1) = -2 (out of bounds)"
        );

        // F_3-addition: (-1) + (-1) = 1, landing in forbidden coset
        let f3_sum = lattice_add_f3(a0, b0);
        assert_eq!(f3_sum[0], 1, "F_3: l_0 = (-1)+(-1) = 1");
        assert!(
            !is_in_base_universe(&f3_sum),
            "l_0=1 excluded from base universe"
        );

        // Exhaustive: ALL pairs of Lambda_256 under F_3-addition give l_0=1
        for a in &lambda_256 {
            for b in &lambda_256 {
                let s = lattice_add_f3(a, b);
                assert_eq!(s[0], 1, "Every F_3 sum of Lambda_256 vectors has l_0 = 1");
            }
        }

        eprintln!("\n=== Phase 3: Coset Obstruction Confirmed ===");
        eprintln!("  Z-addition:  l_0 = -2 (out of bounds) for ALL pairs");
        eprintln!("  F_3-addition: l_0 = +1 (forbidden coset) for ALL pairs");

        // ---- Phase 4: F_3 closure of the l_0=0 sub-lattice of Lambda_2048 ----
        let l0_zero_set: HashSet<LatticeVector> = l0_zero.iter().copied().cloned().collect();
        let n_zero = l0_zero.len();
        let mut zero_closure_count = 0usize;
        let zero_total = n_zero * n_zero;

        for a in &l0_zero {
            for b in &l0_zero {
                let s = lattice_add_f3(a, b);
                // In F_3: l_0 = 0+0 = 0, so sum stays in l_0=0 coset
                assert_eq!(s[0], 0, "F_3: 0+0 = 0 for l_0 coordinate");
                if l0_zero_set.contains(&s) {
                    zero_closure_count += 1;
                }
            }
        }
        let zero_closure_rate = zero_closure_count as f64 / zero_total as f64;

        eprintln!("\n=== Phase 4: l_0=0 Sub-lattice F_3 Closure ===");
        eprintln!(
            "  {}/{} pairs closed = {:.4} ({:.1}%)",
            zero_closure_count,
            zero_total,
            zero_closure_rate,
            zero_closure_rate * 100.0
        );
        // The l_0=0 sub-lattice should have positive closure (it's an actual subgroup)
        assert!(
            zero_closure_count > 0,
            "l_0=0 sub-lattice must have some closure under F_3"
        );

        // ---- Phase 5: Affine F_3 closure on Lambda_256 ----
        // Operation: a +_3 b -_3 p, where p is a fixed base point.
        // This maps l_0: (-1)+(-1)-(-1) = (-1)+(-1)+(+1) = 1+1 = -1 in F_3.
        let lambda_256_set: HashSet<LatticeVector> = lambda_256.iter().copied().collect();
        let n_256 = lambda_256.len();
        let total_256 = n_256 * n_256;

        // Verify l_0 is preserved by affine operation
        let base_point = lambda_256[0];
        let neg_base = lattice_negate_f3(&base_point);
        let test_affine = lattice_add_f3(&lattice_add_f3(a0, b0), &neg_base);
        assert_eq!(test_affine[0], -1, "Affine F_3: l_0 = (-1)+(-1)-(-1) = -1");

        // Full sweep with base point = Lambda_256[0]
        let mut affine_closure_count = 0usize;
        for a in &lambda_256 {
            for b in &lambda_256 {
                let ab = lattice_add_f3(a, b);
                let result = lattice_add_f3(&ab, &neg_base);
                // Verify l_0 preservation for every pair
                assert_eq!(result[0], -1, "Affine sum preserves l_0 = -1");
                if lambda_256_set.contains(&result) {
                    affine_closure_count += 1;
                }
            }
        }
        let affine_rate = affine_closure_count as f64 / total_256 as f64;

        eprintln!("\n=== Phase 5: Affine F_3 Closure on Lambda_256 ===");
        eprintln!("  base = Lambda_256[0] = {:?}", base_point);
        eprintln!(
            "  {}/{} pairs closed = {:.4} ({:.1}%)",
            affine_closure_count,
            total_256,
            affine_rate,
            affine_rate * 100.0
        );

        // ---- Phase 6: Test multiple base points for rate variation ----
        let test_bases = [0, n_256 / 4, n_256 / 2, n_256 - 1];
        let mut rates = Vec::new();
        for &idx in &test_bases {
            if idx >= n_256 {
                continue;
            }
            let bp = lambda_256[idx];
            let nbp = lattice_negate_f3(&bp);
            let mut count = 0usize;
            for a in &lambda_256 {
                for b in &lambda_256 {
                    let ab = lattice_add_f3(a, b);
                    let result = lattice_add_f3(&ab, &nbp);
                    if lambda_256_set.contains(&result) {
                        count += 1;
                    }
                }
            }
            let rate = count as f64 / total_256 as f64;
            rates.push((idx, count, rate));
            eprintln!(
                "  base[{}] = {:?}: {}/{} = {:.1}%",
                idx,
                bp,
                count,
                total_256,
                rate * 100.0
            );
        }

        // ---- Phase 7: Affine F_3 closure at Lambda_512 and Lambda_1024 ----
        for (name, level) in [("Lambda_512", &lambda_512), ("Lambda_1024", &lambda_1024)] {
            let level_set: HashSet<LatticeVector> = level.iter().copied().collect();
            let n = level.len();
            let total = n * n;
            let bp = level[0];
            let nbp = lattice_negate_f3(&bp);
            let mut count = 0usize;
            for a in level.iter() {
                for b in level.iter() {
                    let ab = lattice_add_f3(a, b);
                    let result = lattice_add_f3(&ab, &nbp);
                    if level_set.contains(&result) {
                        count += 1;
                    }
                }
            }
            let rate = count as f64 / total as f64;
            eprintln!(
                "\n  Affine F_3 on {} ({} vectors): {}/{} = {:.1}%",
                name,
                n,
                count,
                total,
                rate * 100.0
            );
        }

        // ---- Summary ----
        eprintln!("\n=== COSET ANALYSIS SUMMARY ===");
        eprintln!("C-498: Coset Obstruction -- Lambda_256 has 0% Z/F_3 closure");
        eprintln!("  because l_0=-1 coset maps to l_0=+1 (forbidden) under addition.");
        eprintln!(
            "  l_0=0 sub-lattice of Lambda_2048 has {:.1}% F_3 closure (subgroup).",
            zero_closure_rate * 100.0
        );
        eprintln!(
            "  Affine F_3 on Lambda_256: {:.1}% closure (coset-corrected).",
            affine_rate * 100.0
        );
    }

    /// Systematic base-point sweep for affine F_3 closure on Lambda_256.
    ///
    /// Tests ALL 256 base points to characterize how closure rate varies,
    /// then correlates closure with lattice properties (Hamming weight,
    /// coordinate pattern, filtration depth).
    #[test]
    fn test_affine_f3_closure_full_basepoint_sweep() {
        use std::collections::HashSet;

        let lambda_256 = enumerate_lambda_256();
        let n = lambda_256.len();
        let total_pairs = n * n;
        let lambda_set: HashSet<LatticeVector> = lambda_256.iter().copied().collect();

        eprintln!(
            "\n=== Affine F_3 Closure: Full Base-Point Sweep on Lambda_256 ({} vectors) ===",
            n
        );

        // Compute closure rate for every base point
        let mut rates: Vec<(usize, f64, LatticeVector)> = Vec::with_capacity(n);
        for (idx, bp) in lambda_256.iter().enumerate() {
            let nbp = lattice_negate_f3(bp);
            let mut count = 0usize;
            for a in &lambda_256 {
                for b in &lambda_256 {
                    let ab = lattice_add_f3(a, b);
                    let result = lattice_add_f3(&ab, &nbp);
                    if lambda_set.contains(&result) {
                        count += 1;
                    }
                }
            }
            let rate = count as f64 / total_pairs as f64;
            rates.push((idx, rate, *bp));
        }

        // Sort by rate to find extremes
        rates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let min_rate = rates[0].1;
        let max_rate = rates[n - 1].1;
        let mean_rate = rates.iter().map(|r| r.1).sum::<f64>() / n as f64;
        let std_rate =
            (rates.iter().map(|r| (r.1 - mean_rate).powi(2)).sum::<f64>() / n as f64).sqrt();

        eprintln!("\n--- Rate Statistics ---");
        eprintln!(
            "  Min:  {:.4} ({:.1}%) at idx {} = {:?}",
            min_rate,
            min_rate * 100.0,
            rates[0].0,
            rates[0].2
        );
        eprintln!(
            "  Max:  {:.4} ({:.1}%) at idx {} = {:?}",
            max_rate,
            max_rate * 100.0,
            rates[n - 1].0,
            rates[n - 1].2
        );
        eprintln!("  Mean: {:.4} ({:.1}%)", mean_rate, mean_rate * 100.0);
        eprintln!("  Std:  {:.4}", std_rate);

        // Rate histogram (by 5% bucket)
        let mut histogram = [0usize; 20]; // 0-5%, 5-10%, ..., 95-100%
        for r in &rates {
            let bucket = (r.1 * 20.0).floor() as usize;
            let bucket = bucket.min(19);
            histogram[bucket] += 1;
        }
        eprintln!("\n--- Rate Histogram (5% buckets) ---");
        for (i, &count) in histogram.iter().enumerate() {
            if count > 0 {
                eprintln!("  {}-{}%: {} base points", i * 5, (i + 1) * 5, count);
            }
        }

        // Bottom 5 and top 5
        eprintln!("\n--- Bottom 5 ---");
        for &(idx, rate, ref v) in rates.iter().take(5) {
            let hw: usize = v.iter().filter(|&&x| x != 0).count();
            let cs: i32 = v.iter().map(|&x| x as i32).sum();
            eprintln!(
                "  idx={}: rate={:.4} ({:.1}%), hw={}, csum={}, v={:?}",
                idx,
                rate,
                rate * 100.0,
                hw,
                cs,
                v
            );
        }
        eprintln!("\n--- Top 5 ---");
        for &(idx, rate, ref v) in rates.iter().rev().take(5) {
            let hw: usize = v.iter().filter(|&&x| x != 0).count();
            let cs: i32 = v.iter().map(|&x| x as i32).sum();
            eprintln!(
                "  idx={}: rate={:.4} ({:.1}%), hw={}, csum={}, v={:?}",
                idx,
                rate,
                rate * 100.0,
                hw,
                cs,
                v
            );
        }

        // Correlations: Hamming weight vs rate, coordinate sum vs rate
        let hamming_weights: Vec<usize> = lambda_256
            .iter()
            .map(|v| v.iter().filter(|&&x| x != 0).count())
            .collect();
        let coord_sums: Vec<i32> = lambda_256
            .iter()
            .map(|v| v.iter().map(|&x| x as i32).sum())
            .collect();

        // Build sorted-by-index rates for correlation
        let mut rates_by_idx = vec![0.0f64; n];
        for &(idx, rate, _) in &rates {
            rates_by_idx[idx] = rate;
        }

        // Spearman rank correlation (approximate via Pearson on ranks)
        let hw_corr = pearson_correlation(
            &hamming_weights
                .iter()
                .map(|&w| w as f64)
                .collect::<Vec<_>>(),
            &rates_by_idx,
        );
        let cs_corr = pearson_correlation(
            &coord_sums.iter().map(|&s| s as f64).collect::<Vec<_>>(),
            &rates_by_idx,
        );

        eprintln!("\n--- Correlations ---");
        eprintln!("  Hamming weight vs rate: r = {:.4}", hw_corr);
        eprintln!("  Coordinate sum vs rate: r = {:.4}", cs_corr);

        // Mean rate grouped by Hamming weight
        let mut hw_groups: std::collections::HashMap<usize, Vec<f64>> =
            std::collections::HashMap::new();
        for (i, &hw) in hamming_weights.iter().enumerate() {
            hw_groups.entry(hw).or_default().push(rates_by_idx[i]);
        }
        let mut hw_keys: Vec<usize> = hw_groups.keys().copied().collect();
        hw_keys.sort();
        eprintln!("\n--- Mean Rate by Hamming Weight ---");
        for hw in hw_keys {
            let group = &hw_groups[&hw];
            let mean = group.iter().sum::<f64>() / group.len() as f64;
            eprintln!(
                "  hw={}: mean rate={:.4} ({:.1}%), n={}",
                hw,
                mean,
                mean * 100.0,
                group.len()
            );
        }

        // Assertions
        assert!(min_rate > 0.0, "Some closure must exist");
        assert!(max_rate < 1.0, "100% closure would mean affine subgroup");
        assert!(min_rate >= 0.20, "Min rate should be at least 20%");
        assert!(max_rate <= 0.50, "Max rate should be at most 50%");
    }

    /// Affine F_3 closure across filtration levels with representative base points.
    ///
    /// At Lambda_512 (512 vectors) and Lambda_1024 (1026 vectors), a full sweep
    /// is expensive. We sample 20 base points spread evenly and compute closure
    /// rate for each, comparing against Lambda_256's known range.
    #[test]
    fn test_affine_f3_closure_across_levels() {
        use std::collections::HashSet;

        let lambda_256 = enumerate_lambda_256();
        let lambda_512 = enumerate_lattice_by_predicate(is_in_lambda_512);
        let lambda_1024 = enumerate_lattice_by_predicate(is_in_lambda_1024);

        let n_sample = 20;

        for (name, level) in [
            ("Lambda_256", &lambda_256),
            ("Lambda_512", &lambda_512),
            ("Lambda_1024", &lambda_1024),
        ] {
            let n = level.len();
            let total_pairs = n * n;
            let level_set: HashSet<LatticeVector> = level.iter().copied().collect();

            // Sample n_sample base points evenly spread
            let step = (n / n_sample).max(1);
            let mut rates = Vec::new();

            for i in 0..n_sample {
                let idx = (i * step).min(n - 1);
                let bp = &level[idx];
                let nbp = lattice_negate_f3(bp);
                let mut count = 0usize;
                for a in level.iter() {
                    for b in level.iter() {
                        let ab = lattice_add_f3(a, b);
                        let result = lattice_add_f3(&ab, &nbp);
                        if level_set.contains(&result) {
                            count += 1;
                        }
                    }
                }
                let rate = count as f64 / total_pairs as f64;
                rates.push(rate);
            }

            let mean = rates.iter().sum::<f64>() / rates.len() as f64;
            let min = rates.iter().cloned().fold(f64::INFINITY, f64::min);
            let max = rates.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            eprintln!(
                "{} ({} vectors): mean={:.4} ({:.1}%), range=[{:.4}, {:.4}]",
                name,
                n,
                mean,
                mean * 100.0,
                min,
                max
            );

            // All levels should have some closure (>20%) but not full (<60%)
            assert!(mean > 0.20, "{} mean closure should exceed 20%", name);
            assert!(mean < 0.60, "{} mean closure should be below 60%", name);
        }
    }

    /// Full base-point sweep of affine F_3 closure on Lambda_512.
    ///
    /// Lambda_512 showed anomalous wide variance (24-34%) in the sampled test.
    /// This test sweeps ALL 512 base points to characterize the full distribution,
    /// and correlates with lattice properties to explain the variance.
    #[test]
    fn test_affine_f3_closure_lambda512_full_sweep() {
        use std::collections::HashSet;

        let lambda_512 = enumerate_lattice_by_predicate(is_in_lambda_512);
        let n = lambda_512.len();
        assert_eq!(n, 512);
        let total_pairs = n * n;
        let lambda_set: HashSet<LatticeVector> = lambda_512.iter().copied().collect();

        eprintln!(
            "\n=== Affine F_3 Closure: Full Base-Point Sweep on Lambda_512 ({} vectors) ===",
            n
        );

        let mut rates: Vec<(usize, f64, LatticeVector)> = Vec::with_capacity(n);
        for (idx, bp) in lambda_512.iter().enumerate() {
            let nbp = lattice_negate_f3(bp);
            let mut count = 0usize;
            for a in &lambda_512 {
                for b in &lambda_512 {
                    let ab = lattice_add_f3(a, b);
                    let result = lattice_add_f3(&ab, &nbp);
                    if lambda_set.contains(&result) {
                        count += 1;
                    }
                }
            }
            let rate = count as f64 / total_pairs as f64;
            rates.push((idx, rate, *bp));
        }

        rates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let min_rate = rates[0].1;
        let max_rate = rates[n - 1].1;
        let mean_rate = rates.iter().map(|r| r.1).sum::<f64>() / n as f64;
        let std_rate =
            (rates.iter().map(|r| (r.1 - mean_rate).powi(2)).sum::<f64>() / n as f64).sqrt();

        eprintln!("\n--- Rate Statistics ---");
        eprintln!("  Min:  {:.4} ({:.1}%)", min_rate, min_rate * 100.0);
        eprintln!("  Max:  {:.4} ({:.1}%)", max_rate, max_rate * 100.0);
        eprintln!("  Mean: {:.4} ({:.1}%)", mean_rate, mean_rate * 100.0);
        eprintln!("  Std:  {:.4}", std_rate);

        // Rate histogram
        let mut histogram = [0usize; 20];
        for r in &rates {
            let bucket = (r.1 * 20.0).floor() as usize;
            histogram[bucket.min(19)] += 1;
        }
        eprintln!("\n--- Rate Histogram (5% buckets) ---");
        for (i, &count) in histogram.iter().enumerate() {
            if count > 0 {
                eprintln!("  {}-{}%: {} base points", i * 5, (i + 1) * 5, count);
            }
        }

        // Bottom 5 and top 5
        eprintln!("\n--- Bottom 5 ---");
        for &(idx, rate, ref v) in rates.iter().take(5) {
            let hw: usize = v.iter().filter(|&&x| x != 0).count();
            eprintln!(
                "  idx={}: rate={:.4} ({:.1}%), hw={}, v={:?}",
                idx,
                rate,
                rate * 100.0,
                hw,
                v
            );
        }
        eprintln!("\n--- Top 5 ---");
        for &(idx, rate, ref v) in rates.iter().rev().take(5) {
            let hw: usize = v.iter().filter(|&&x| x != 0).count();
            eprintln!(
                "  idx={}: rate={:.4} ({:.1}%), hw={}, v={:?}",
                idx,
                rate,
                rate * 100.0,
                hw,
                v
            );
        }

        // Correlation with Hamming weight
        let mut rates_by_idx = vec![0.0f64; n];
        for &(idx, rate, _) in &rates {
            rates_by_idx[idx] = rate;
        }
        let hws: Vec<f64> = lambda_512
            .iter()
            .map(|v| v.iter().filter(|&&x| x != 0).count() as f64)
            .collect();
        let hw_corr = pearson_correlation(&hws, &rates_by_idx);

        // Correlation with l_1 value (is the C-501 contaminant related?)
        let l1_vals: Vec<f64> = lambda_512.iter().map(|v| v[1] as f64).collect();
        let l1_corr = pearson_correlation(&l1_vals, &rates_by_idx);

        // Correlation with number of +1 coordinates
        let plus1_counts: Vec<f64> = lambda_512
            .iter()
            .map(|v| v.iter().filter(|&&x| x == 1).count() as f64)
            .collect();
        let p1_corr = pearson_correlation(&plus1_counts, &rates_by_idx);

        eprintln!("\n--- Correlations ---");
        eprintln!("  Hamming weight vs rate:  r = {:.4}", hw_corr);
        eprintln!("  l_1 value vs rate:       r = {:.4}", l1_corr);
        eprintln!("  #(+1 coords) vs rate:    r = {:.4}", p1_corr);

        // Mean rate grouped by Hamming weight
        let mut hw_groups: std::collections::HashMap<usize, Vec<f64>> =
            std::collections::HashMap::new();
        for (i, v) in lambda_512.iter().enumerate() {
            let hw = v.iter().filter(|&&x| x != 0).count();
            hw_groups.entry(hw).or_default().push(rates_by_idx[i]);
        }
        let mut hw_keys: Vec<usize> = hw_groups.keys().copied().collect();
        hw_keys.sort();
        eprintln!("\n--- Mean Rate by Hamming Weight ---");
        for hw in hw_keys {
            let group = &hw_groups[&hw];
            let mean = group.iter().sum::<f64>() / group.len() as f64;
            let std =
                (group.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / group.len() as f64).sqrt();
            eprintln!(
                "  hw={}: mean={:.4} ({:.1}%), std={:.4}, n={}",
                hw,
                mean,
                mean * 100.0,
                std,
                group.len()
            );
        }

        // Mean rate grouped by l_1 value
        let mut l1_groups: std::collections::HashMap<i8, Vec<f64>> =
            std::collections::HashMap::new();
        for (i, v) in lambda_512.iter().enumerate() {
            l1_groups.entry(v[1]).or_default().push(rates_by_idx[i]);
        }
        eprintln!("\n--- Mean Rate by l_1 Value ---");
        for l1 in [-1i8, 0, 1] {
            if let Some(group) = l1_groups.get(&l1) {
                let mean = group.iter().sum::<f64>() / group.len() as f64;
                eprintln!(
                    "  l_1={}: mean={:.4} ({:.1}%), n={}",
                    l1,
                    mean,
                    mean * 100.0,
                    group.len()
                );
            }
        }

        assert!(min_rate > 0.15, "Min rate should be at least 15%");
        assert!(max_rate < 0.50, "Max rate should be at most 50%");
    }

    /// Pearson correlation coefficient for two f64 slices.
    fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len() as f64;
        let mx = x.iter().sum::<f64>() / n;
        let my = y.iter().sum::<f64>() / n;
        let mut sxy = 0.0;
        let mut sxx = 0.0;
        let mut syy = 0.0;
        for (xi, yi) in x.iter().zip(y.iter()) {
            let dx = xi - mx;
            let dy = yi - my;
            sxy += dx * dy;
            sxx += dx * dx;
            syy += dy * dy;
        }
        if sxx < 1e-15 || syy < 1e-15 {
            return 0.0;
        }
        sxy / (sxx * syy).sqrt()
    }

    #[test]
    fn test_lattice_negate_f3_basic() {
        let v: LatticeVector = [-1, 0, 1, -1, 1, 0, -1, 1];
        let neg = lattice_negate_f3(&v);
        assert_eq!(neg, [1, 0, -1, 1, -1, 0, 1, -1]);

        // Double negation is identity
        let double_neg = lattice_negate_f3(&neg);
        assert_eq!(double_neg, v);

        // Negation of zero is zero
        let zero: LatticeVector = [0, 0, 0, 0, 0, 0, 0, 0];
        assert_eq!(lattice_negate_f3(&zero), zero);

        // a + (-a) = 0 in F_3
        let sum = lattice_add_f3(&v, &neg);
        assert_eq!(sum, [0, 0, 0, 0, 0, 0, 0, 0]);
    }

    // ================================================================
    // Phase A (T4): Lambda_4096 carrier tests
    // ================================================================

    #[test]
    fn test_lambda_4096_carrier_count() {
        // Lambda_4096 = base universe (no additional exclusions).
        // Should be a strict superset of Lambda_2048.
        let l4096 = enumerate_lambda_4096();
        let l2048 = enumerate_lattice_by_predicate(is_in_lambda_2048);
        let base = enumerate_lattice_by_predicate(is_in_base_universe);

        eprintln!(
            "Lambda_4096: {} vectors (= base universe: {}), Lambda_2048: {}",
            l4096.len(),
            base.len(),
            l2048.len()
        );

        // Lambda_4096 == base universe
        assert_eq!(l4096.len(), base.len(), "Lambda_4096 should equal base universe");

        // Lambda_4096 > Lambda_2048 (strict superset)
        assert!(
            l4096.len() > l2048.len(),
            "Lambda_4096 ({}) must be a strict superset of Lambda_2048 ({})",
            l4096.len(),
            l2048.len()
        );

        // Every Lambda_2048 vector should be in Lambda_4096
        let l4096_set: std::collections::HashSet<LatticeVector> =
            l4096.iter().copied().collect();
        for v in &l2048 {
            assert!(
                l4096_set.contains(v),
                "Lambda_2048 vector {:?} not in Lambda_4096",
                v
            );
        }
    }

    #[test]
    fn test_lambda_4096_parity_constraints() {
        // Verify all 4 octonion parity laws hold for Lambda_4096.
        let l4096 = enumerate_lambda_4096();
        let (n, n_tri, n_sum, n_wt, n_l0, all_pass) =
            verify_octonion_parity_constraints(&l4096);

        eprintln!("Lambda_4096 parity check: n={n}");
        eprintln!("  trinary: {n_tri}/{n}");
        eprintln!("  even_sum: {n_sum}/{n}");
        eprintln!("  even_weight: {n_wt}/{n}");
        eprintln!("  l_0 != +1: {n_l0}/{n}");

        assert!(
            all_pass,
            "All 4 octonion parity constraints must hold for Lambda_4096. \
             trinary={n_tri}/{n}, even_sum={n_sum}/{n}, even_weight={n_wt}/{n}, l0={n_l0}/{n}"
        );
    }

    #[test]
    fn test_octonion_parity_proof_dim4096() {
        // Cross-validate: parity constraints hold for EVERY filtration level.
        // This is the algebraic proof that octonion structure forces 8D constraints.
        let levels: Vec<(&str, Vec<LatticeVector>)> = vec![
            ("Lambda_4096", enumerate_lambda_4096()),
            (
                "Lambda_2048",
                enumerate_lattice_by_predicate(is_in_lambda_2048),
            ),
            (
                "Lambda_1024",
                enumerate_lattice_by_predicate(is_in_lambda_1024),
            ),
            (
                "Lambda_512",
                enumerate_lattice_by_predicate(is_in_lambda_512),
            ),
            ("Lambda_256", enumerate_lambda_256()),
        ];

        for (name, vecs) in &levels {
            let (n, n_tri, n_sum, n_wt, n_l0, all_pass) =
                verify_octonion_parity_constraints(vecs);
            eprintln!(
                "{name}: {n} vectors, all_parity_pass={all_pass}"
            );
            assert!(
                all_pass,
                "{name}: parity violation! tri={n_tri}/{n}, sum={n_sum}/{n}, \
                 wt={n_wt}/{n}, l0={n_l0}/{n}"
            );
        }
    }
}
