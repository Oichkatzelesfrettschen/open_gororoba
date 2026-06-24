//! Cariow-style factorization: implementation and count inventory for CD products.
//!
//! # See also
//!
//! - `super::trigintaduonion` -- dim=32 standard multiply and schedule placeholder
//! - `super::sedenion` -- dim=16 triad-based multiplication (baseline for count analysis)
//! - `algebra_experimental::novel_algorithms::deep_factorization` -- heuristic block search
//!
//! # Background
//!
//! Standard Cayley-Dickson doubling: `(A,B)*(C,D) = (AC - DB*, DA + BC*)`
//! requires 4 sub-multiplications at each level. This gives:
//!   S(n) = 4 * S(n/2),  S(2) = 4  =>  S(n) = n^2
//!
//! The Cariow (2012, 2013) family of algorithms reduces real multiplication
//! counts by factorizing the fixed bilinear form for a specific dimension.
//! That is different from a generic Karatsuba rewrite of the recursive
//! Cayley-Dickson formula: conjugation and non-commutative operand order prevent
//! a dimension-independent 3-for-4 subalgebra multiply.
//!
//! # Published results
//!
//! Cariow (2012): octonion (8D) -- 26 mults vs 64 standard (evid C).
//! Cariow (2013): sedenion (16D) -- 122 mults vs 256 standard, implemented in
//! `cariow2013_sedenion_mul` and Rocq-proved by
//! C1636_Cariow2013SedenionSchedule::C1636_cariow2013_sedenion_mul_eq.
//! Repo code (trigintaduonion.rs): dim=32 -- records a 498-multiplication target.
//! The checked implementation still evaluates the standard four-sedenion split.
//! The 498 number is a target claim, not a verified implementation count.
//!
//! # Conditional count extrapolation for dim=64
//!
//! If C(n) is the Cariow multiplication count and C(32) = 498 (claimed), then the
//! top-level application to dim=64 gives:
//!   C(64) <= 3 * C(32) = 1494
//! vs S(64) = 4096 standard: speedup <= 4096/1494 ~ 2.74x.
//!
//! This extrapolation is conditional on a real dim=32 schedule. The actual
//! Cariow-64 count requires a full analysis of the 64D sign table. Until such
//! analysis is complete, 1494 is only a claim boundary, not an implementation
//! budget.
//!
//! # Evidentiary classification
//!
//! - `standard_mult_count`: T (exact formula n^2, follows from CD doubling)
//! - `cariow_pure_3x_bound`: C (optimistic invalid-generic recurrence marker)
//! - `cariow2013_sedenion_mul`: T (Rocq-proved equality to the 16D CD product)
//! - `cariow_repo_bound`: mixed (16D implemented; higher dimensions are targets)

pub const CARIOW2013_SEDENION_HADAMARD_MULTS: usize = 16;
pub const CARIOW2013_SEDENION_SPARSE_CORRECTION_MULTS: usize = 106;
pub const CARIOW2013_SEDENION_TOTAL_MULTS: usize =
    CARIOW2013_SEDENION_HADAMARD_MULTS + CARIOW2013_SEDENION_SPARSE_CORRECTION_MULTS;

const CARIOW2013_BHAT_TERMS: &[(usize, usize, usize)] = &[
    (0, 0, 0),
    (1, 3, 2),
    (2, 1, 3),
    (3, 2, 1),
    (1, 5, 4),
    (1, 6, 7),
    (2, 6, 4),
    (2, 7, 5),
    (3, 5, 6),
    (3, 7, 4),
    (1, 9, 8),
    (1, 10, 11),
    (2, 10, 8),
    (2, 11, 9),
    (3, 9, 10),
    (3, 11, 8),
    (1, 12, 13),
    (1, 15, 14),
    (2, 12, 14),
    (2, 13, 15),
    (3, 12, 15),
    (3, 14, 13),
    (4, 1, 5),
    (4, 2, 6),
    (4, 3, 7),
    (5, 2, 7),
    (6, 3, 5),
    (7, 1, 6),
    (5, 4, 1),
    (5, 6, 3),
    (6, 4, 2),
    (6, 7, 1),
    (7, 4, 3),
    (7, 5, 2),
    (5, 9, 12),
    (5, 11, 14),
    (6, 9, 15),
    (6, 10, 12),
    (7, 10, 13),
    (7, 11, 12),
    (4, 12, 8),
    (4, 13, 9),
    (4, 14, 10),
    (4, 15, 11),
    (5, 13, 8),
    (5, 15, 10),
    (6, 13, 11),
    (6, 14, 8),
    (7, 14, 9),
    (7, 15, 8),
    (8, 1, 9),
    (8, 2, 10),
    (8, 3, 11),
    (9, 2, 11),
    (10, 3, 9),
    (11, 1, 10),
    (8, 4, 12),
    (8, 5, 13),
    (8, 6, 14),
    (8, 7, 15),
    (9, 4, 13),
    (9, 7, 14),
    (10, 4, 14),
    (10, 5, 15),
    (11, 4, 15),
    (11, 6, 13),
    (9, 8, 1),
    (9, 10, 3),
    (10, 8, 2),
    (10, 11, 1),
    (11, 8, 3),
    (11, 9, 2),
    (9, 12, 5),
    (9, 15, 6),
    (10, 12, 6),
    (10, 13, 7),
    (11, 12, 7),
    (11, 14, 5),
    (13, 1, 12),
    (13, 3, 14),
    (14, 1, 15),
    (14, 2, 12),
    (15, 2, 13),
    (15, 3, 12),
    (12, 5, 9),
    (12, 6, 10),
    (12, 7, 11),
    (13, 7, 10),
    (14, 5, 11),
    (15, 6, 9),
    (12, 8, 4),
    (13, 8, 5),
    (13, 9, 4),
    (13, 11, 6),
    (14, 8, 6),
    (14, 9, 7),
    (14, 10, 4),
    (15, 8, 7),
    (15, 10, 5),
    (15, 11, 4),
    (12, 13, 1),
    (12, 14, 2),
    (12, 15, 3),
    (13, 15, 2),
    (14, 13, 3),
    (15, 14, 1),
];

fn apply_hadamard_stage_16(values: &mut [f64; 16], stage: usize) {
    let half_width = 8 >> stage;
    let width = half_width * 2;

    for block_start in (0..16).step_by(width) {
        for offset in 0..half_width {
            let lo_idx = block_start + offset;
            let hi_idx = lo_idx + half_width;
            let lo = values[lo_idx];
            let hi = values[hi_idx];
            values[lo_idx] = lo + hi;
            values[hi_idx] = lo - hi;
        }
    }
}

fn cariow2013_hadamard_diagonal(b: &[f64; 16]) -> [f64; 16] {
    let mut diagonal = *b;

    for stage in 0..4 {
        apply_hadamard_stage_16(&mut diagonal, stage);
    }
    for value in &mut diagonal {
        *value /= 16.0;
    }

    diagonal
}

fn cariow2013_sparse_correction(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    let mut correction = [0.0; 16];

    for &(row, col, b_idx) in CARIOW2013_BHAT_TERMS {
        correction[row] += 2.0 * b[b_idx] * a[col];
    }

    correction
}

/// Multiply two sedenions through the Cariow 2013 122-multiply schedule.
///
/// Cariow's construction writes the row-flipped product vector as
/// `Y = Bcheck X - 2 * Bhat X`, diagonalizes the block-symmetric Toeplitz
/// matrix `Bcheck` with four 16-point Hadamard stages, and evaluates the sparse
/// correction directly. Multiplications by powers of two are excluded from the
/// published multiplier count.
pub fn cariow2013_sedenion_mul(a: &[f64; 16], b: &[f64; 16]) -> [f64; 16] {
    let diagonal = cariow2013_hadamard_diagonal(b);
    let mut toeplitz_part = *a;

    for stage in 0..4 {
        apply_hadamard_stage_16(&mut toeplitz_part, stage);
    }
    for i in 0..16 {
        toeplitz_part[i] *= diagonal[i];
    }
    for stage in (0..4).rev() {
        apply_hadamard_stage_16(&mut toeplitz_part, stage);
    }

    let correction = cariow2013_sparse_correction(a, b);
    let mut result = [0.0; 16];
    for i in 0..16 {
        result[i] = toeplitz_part[i] - correction[i];
    }
    result[0] = -result[0];
    result
}

/// Exact multiplication count for standard CD doubling multiplication.
///
/// S(n) = n^2 for any power-of-2 dimension n >= 2.
/// Derived from the recursion S(2n) = 4*S(n), S(2) = 4.
///
/// Evidentiary class: T (theorem, follows directly from CD doubling formula).
pub fn standard_mult_count(dim: usize) -> usize {
    assert!(
        dim.is_power_of_two() && dim >= 2,
        "dim must be a power of 2 >= 2"
    );
    dim * dim
}

/// Optimistic 3x recurrence marker.
///
/// C(2n) <= 3 * C(n)  (with additions-only pre/post-computation)
///
/// Base case: C(2) = 4 (complex multiplication, no sub-structure to exploit).
/// C(4)  <= 12 vs S(4)  = 16   (upper bound; actual may be 8 by Karatsuba)
/// C(8)  <= 36 vs S(8)  = 64   (upper bound; Cariow 2012 claims 26)
/// C(16) <= 108 vs S(16) = 256  (optimism marker; Cariow 2013 reports 122)
/// C(32) <= 324 vs S(32) = 1024 (invalid as a generic CD implementation count)
/// C(64) <= 972 vs S(64) = 4096 (invalid as a generic CD implementation count)
///
/// NOTE: The 498 target for dim=32 is greater than 324. That falsifies this pure
/// recurrence as a generic implementation bound for Cayley-Dickson algebras at
/// this level. Keep the function as an optimism marker for count tables; use
/// `cariow_repo_bound` for implemented, published, or local target counts.
///
/// Evidentiary class: C (invalid-generic recurrence marker, not an implementation count).
pub fn cariow_pure_3x_bound(dim: usize) -> usize {
    assert!(
        dim.is_power_of_two() && dim >= 2,
        "dim must be a power of 2 >= 2"
    );
    if dim == 2 {
        return 4;
    }
    3 * cariow_pure_3x_bound(dim / 2)
}

/// Published, implemented, or local Cariow target count.
///
/// For dim=64: C(64) <= 3 * C(32) = 3 * 498 = 1494, if a top-level Cariow
/// reduction applies and each sub-problem uses a real dim=32 schedule.
///
/// This target is conditional on C(32) = 498 being achievable. The
/// `trigintaduonion::mul_optimized` path does not achieve 498; it uses 4 sedenion
/// multiplications = 4 * 256 = 1024 mults (the standard count).
///
/// Evidentiary class: dim=16 is implemented and Rocq-proved.  Other nontrivial
/// dimensions are publication records or conditional targets until their
/// schedules are implemented and checked.
pub fn cariow_repo_bound(dim: usize) -> Option<usize> {
    match dim {
        2 => Some(4),
        4 => Some(8),  // Karatsuba quaternion (standard result)
        8 => Some(26), // Cariow 2012 (published claim, evid C)
        16 => Some(CARIOW2013_SEDENION_TOTAL_MULTS), // Cariow 2013, implemented
        32 => Some(498), // Recorded dim=32 target (evid C, not implemented)
        64 => Some(3 * 498), // = 1494, conditional target (evid C)
        _ => None,     // Unknown for other dimensions
    }
}

/// Speedup ratio: standard / cariow_repo_bound.
///
/// Only defined where `cariow_repo_bound` is known.
/// Evidentiary class: C (same as cariow_repo_bound).
pub fn cariow_speedup(dim: usize) -> Option<f64> {
    cariow_repo_bound(dim).map(|c| standard_mult_count(dim) as f64 / c as f64)
}

/// Summary record of multiplication count analysis at a given dimension.
#[derive(Clone, Copy, Debug)]
pub struct MultCountRecord {
    pub dim: usize,
    /// Standard CD multiplication count (n^2). Evid T.
    pub standard: usize,
    /// Pure 3x recursive bound C(n) = 3*C(n/2). Evid C.
    pub pure_3x_bound: usize,
    /// Published, implemented, or local target count.
    /// None if not known.
    pub repo_bound: Option<usize>,
    /// Speedup = standard / repo_bound. None if repo_bound is None.
    pub speedup: Option<f64>,
}

impl MultCountRecord {
    pub fn compute(dim: usize) -> Self {
        Self {
            dim,
            standard: standard_mult_count(dim),
            pure_3x_bound: cariow_pure_3x_bound(dim),
            repo_bound: cariow_repo_bound(dim),
            speedup: cariow_speedup(dim),
        }
    }
}

/// Print a table of multiplication counts for all known dimensions.
pub fn print_mult_count_table() {
    println!("=== Cariow-Style CD Multiplication Count Analysis ===");
    println!("  Evid T: standard exact; dim=16 Cariow schedule proved. Other targets are C.");
    println!();
    println!(
        "{:>6}  {:>10}  {:>12}  {:>12}  {:>8}",
        "dim", "standard", "pure_3x", "target", "speedup"
    );
    println!("{}", "-".repeat(56));
    for &d in &[2usize, 4, 8, 16, 32, 64] {
        let r = MultCountRecord::compute(d);
        let repo_str = r.repo_bound.map_or("?".to_string(), |v| v.to_string());
        let spd_str = r.speedup.map_or("?".to_string(), |v| format!("{:.2}x", v));
        println!(
            "{:>6}  {:>10}  {:>12}  {:>12}  {:>8}",
            r.dim, r.standard, r.pure_3x_bound, repo_str, spd_str
        );
    }
    println!();
    println!("NOTE: dim=16 uses the implemented Cariow 2013 schedule.");
    println!("NOTE: target for dim=64 is 3*498=1494 (conditional, unverified).");
    println!("NOTE: pure_3x is an optimism marker, not a generic CD implementation count.");
    println!("NOTE: C(32)=498 in trigintaduonion::mul_optimized is not implemented");
    println!("  (the checked-in path falls back to 4*sedenion_multiply = 1024 mults).");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cayley_dickson::sedenion::sedenion_multiply_explicit;

    fn assert_close_16(actual: [f64; 16], expected: [f64; 16]) {
        for i in 0..16 {
            assert!(
                (actual[i] - expected[i]).abs() < 1e-10,
                "mismatch at index {i}: actual {}, expected {}",
                actual[i],
                expected[i]
            );
        }
    }

    /// Standard CD multiplication count is exactly n^2.
    ///
    /// Follows directly from the CD doubling recursion S(2n) = 4*S(n), S(2) = 4.
    /// Evid T (theorem).
    #[test]
    fn test_standard_mult_count_exact() {
        assert_eq!(standard_mult_count(2), 4);
        assert_eq!(standard_mult_count(4), 16);
        assert_eq!(standard_mult_count(8), 64);
        assert_eq!(standard_mult_count(16), 256);
        assert_eq!(standard_mult_count(32), 1024);
        assert_eq!(standard_mult_count(64), 4096);
    }

    #[test]
    fn test_cariow2013_sedenion_multiplier_count() {
        assert_eq!(
            CARIOW2013_BHAT_TERMS.len(),
            CARIOW2013_SEDENION_SPARSE_CORRECTION_MULTS
        );
        assert_eq!(CARIOW2013_SEDENION_TOTAL_MULTS, 122);
        assert_eq!(cariow_repo_bound(16), Some(122));
    }

    #[test]
    fn test_cariow2013_sedenion_mul_matches_basis_products() {
        for left_idx in 0..16 {
            for right_idx in 0..16 {
                let mut left = [0.0; 16];
                let mut right = [0.0; 16];
                left[left_idx] = 1.0;
                right[right_idx] = 1.0;

                assert_close_16(
                    cariow2013_sedenion_mul(&left, &right),
                    sedenion_multiply_explicit(&left, &right),
                );
            }
        }
    }

    #[test]
    fn test_cariow2013_sedenion_mul_matches_deterministic_vectors() {
        for seed in 0..16 {
            let mut left = [0.0; 16];
            let mut right = [0.0; 16];
            for i in 0..16 {
                let left_raw = ((seed + 3 * i + 5) % 11) as f64 - 5.0;
                let right_raw = ((2 * seed + 5 * i + 7) % 13) as f64 - 6.0;
                left[i] = left_raw / 7.0;
                right[i] = right_raw / 11.0;
            }

            assert_close_16(
                cariow2013_sedenion_mul(&left, &right),
                sedenion_multiply_explicit(&left, &right),
            );
        }
    }

    /// Pure 3x recurrence marker is strictly less than standard for dim >= 4.
    #[test]
    fn test_pure_3x_bound_below_standard() {
        for &d in &[4usize, 8, 16, 32, 64] {
            let s = standard_mult_count(d);
            let b = cariow_pure_3x_bound(d);
            assert!(
                b < s,
                "pure_3x({}) = {} should be < standard({}) = {}",
                d,
                b,
                d,
                s
            );
        }
    }

    /// Target count for dim=32 matches the recorded trigintaduonion target (498).
    ///
    /// Evid C (conjecture; implementation does not yet achieve this).
    #[test]
    fn test_repo_bound_dim32_matches_claim() {
        assert_eq!(
            cariow_repo_bound(32),
            Some(498),
            "Recorded dim=32 target is 498 multiplications"
        );
    }

    /// Conditional target for dim=64 is 3*498 = 1494.
    ///
    /// This is a target from applying one level of Cariow reduction at the top,
    /// using the claimed C(32)=498 for each of 3 sub-problems.
    /// Evid C.
    #[test]
    fn test_repo_bound_dim64_target() {
        let bound = cariow_repo_bound(64).unwrap();
        assert_eq!(
            bound, 1494,
            "C(64) <= 3*C(32) = 3*498 = 1494 (conditional target)"
        );
        assert!(
            bound < standard_mult_count(64),
            "Target {} should be < standard {} for dim=64",
            bound,
            4096
        );
        let speedup = cariow_speedup(64).unwrap();
        assert!(
            speedup > 2.5 && speedup < 3.0,
            "Expected speedup ~2.74x for dim=64, got {:.2}x",
            speedup
        );
    }

    /// Target speedup is always > 1 at all recorded dimensions.
    ///
    /// The speedup is not monotone: dim=32 (2.06x) is slightly below dim=16
    /// (2.10x) because the recorded 498 for dim=32 exceeds the pure
    /// 3*C(16)=366 target extrapolation. The dim=64 target gives 2.74x.
    /// Evid C.
    #[test]
    fn test_speedup_always_above_one() {
        for &d in &[4usize, 8, 16, 32, 64] {
            let spd = cariow_speedup(d).unwrap();
            assert!(
                spd > 1.0,
                "speedup at dim={} should be > 1.0, got {:.2}",
                d,
                spd
            );
        }
        // Mixed status: dim=16 is implemented; dim=32 and dim=64 are targets.
        let spd16 = cariow_speedup(16).unwrap();
        let spd32 = cariow_speedup(32).unwrap();
        let spd64 = cariow_speedup(64).unwrap();
        assert!(spd16 > 2.0, "dim=16: 256/122 = 2.10x, got {:.2}", spd16);
        assert!(spd32 > 2.0, "dim=32: 1024/498 = 2.06x, got {:.2}", spd32);
        assert!(spd64 > 2.5, "dim=64: 4096/1494 = 2.74x, got {:.2}", spd64);
    }

    #[test]
    fn test_print_mult_count_table() {
        print_mult_count_table();
    }
}
