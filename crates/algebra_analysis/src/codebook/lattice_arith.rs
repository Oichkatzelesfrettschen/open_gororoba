//! Lattice arithmetic and pinned-corner slice characterization on
//! {-1, 0, 1}^8.
//!
//! Includes:
//! - `SliceCharacterization` + `characterize_pinned_slice`: closed-form
//!   counting and parity analysis for the "all-(-1)s prefix" slice of
//!   Lambda_256 used by the codebook proofs.
//! - F_3 lattice operations: `lattice_add_f3`, `lattice_negate_f3`,
//!   `lattice_diff` (over Z, returning [i32; 8]).
//! - `apply_scalar_shadow`: affine/linear action of the scalar shadow
//!   pi(b) on a lattice vector with mode dispatch ("affine" or "linear").

use super::{LatticeVector, enumerate_lambda_256};

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
