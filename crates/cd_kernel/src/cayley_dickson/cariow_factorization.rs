//! Cariow-style factorization: theoretical multiplication counts for CD algebras.
//!
//! # See also
//!
//! - `super::trigintaduonion` -- dim=32 implementation (mul_standard vs mul_optimized)
//! - `super::sedenion` -- dim=16 triad-based multiplication (baseline for count analysis)
//! - `algebra_experimental::novel_algorithms::deep_factorization` -- heuristic block search
//!
//! # Background
//!
//! Standard Cayley-Dickson doubling: `(A,B)*(C,D) = (AC - DB*, DA + BC*)`
//! requires 4 sub-multiplications at each level. This gives:
//!   S(n) = 4 * S(n/2),  S(2) = 4  =>  S(n) = n^2
//!
//! The Cariow (2012, 2013) family of algorithms reduces the top-level factor from
//! 4 to 3 using pre-computed linear combinations of the inputs, similar to the
//! Karatsuba trick for polynomial multiplication. Non-commutativity of sedenions
//! and higher algebras complicates the direct application of Karatsuba, but
//! Cariow's construction exploits the specific sign-table structure of CD algebras.
//!
//! # Published results
//!
//! Cariow (2012): octonion (8D) -- 26 mults vs 64 standard (evid C; unverified
//! against this repo's sedenion_multiply_explicit baseline).
//! Cariow (2013): sedenion (16D) -- 84 mults vs 256 standard (evid C).
//! Repo code (trigintaduonion.rs): dim=32 -- claims 498 mults vs 1024 standard.
//! NOTE: the trigintaduonion mul_optimized implementation does NOT yet achieve 498;
//! it falls back to 4 sedenion multiplications. The 498 is a TARGET from the paper,
//! not a verified implementation count. Evidentiary class: C (conjecture).
//!
//! # Recursive upper bound for dim=64
//!
//! If C(n) is the Cariow multiplication count and C(32) = 498 (claimed), then the
//! top-level application to dim=64 gives:
//!   C(64) <= 3 * C(32) = 1494
//! vs S(64) = 4096 standard: speedup <= 4096/1494 ~ 2.74x.
//!
//! This is a STRUCTURAL UPPER BOUND. The actual Cariow-64 minimum requires
//! a full analysis of the 64D sign table (as in Cariow 2013 for 16D). Until
//! such analysis is complete, 1494 is the conservative upper bound.
//!
//! # Evidentiary classification
//!
//! - `standard_mult_count`: T (exact formula n^2, follows from CD doubling)
//! - `cariow_upper_bound`: C (conjecture; depends on unverified C(32)=498 claim)
//! - `cariow_known`: E (known published values, verified against literature)

/// Exact multiplication count for standard CD doubling multiplication.
///
/// S(n) = n^2 for any power-of-2 dimension n >= 2.
/// Derived from the recursion S(2n) = 4*S(n), S(2) = 4.
///
/// Evidentiary class: T (theorem, follows directly from CD doubling formula).
pub fn standard_mult_count(dim: usize) -> usize {
    assert!(dim.is_power_of_two() && dim >= 2, "dim must be a power of 2 >= 2");
    dim * dim
}

/// Cariow upper bound: at most 3 recursive sub-multiplications at the top level.
///
/// C(2n) <= 3 * C(n)  (with additions-only pre/post-computation)
///
/// Base case: C(2) = 4 (complex multiplication, no sub-structure to exploit).
/// C(4)  <= 12 vs S(4)  = 16   (upper bound; actual may be 8 by Karatsuba)
/// C(8)  <= 36 vs S(8)  = 64   (upper bound; Cariow 2012 claims 26)
/// C(16) <= 108 vs S(16) = 256  (upper bound; Cariow 2013 claims 84)
/// C(32) <= 324 vs S(32) = 1024 (upper bound; repo claims 498 -- better than this)
/// C(64) <= 972 vs S(64) = 4096 (structural upper bound from pure top-level reduction)
///
/// NOTE: C(32) = 498 > 324.  The repo's 498 claim uses a DIFFERENT recursion than
/// pure 3*C(n/2).  The 498 includes additional multiplications for the cross-terms
/// that cannot be eliminated by the top-level Karatsuba-like step alone.
/// This means the "3*C(n)" bound is an OVER-OPTIMISTIC lower bound at this level;
/// the practical bound uses the repo's recursive fallback.  See `cariow_repo_bound`.
///
/// Evidentiary class: C (structural upper bound, not a proven implementation count).
pub fn cariow_pure_3x_bound(dim: usize) -> usize {
    assert!(dim.is_power_of_two() && dim >= 2, "dim must be a power of 2 >= 2");
    if dim == 2 {
        return 4;
    }
    3 * cariow_pure_3x_bound(dim / 2)
}

/// Cariow repo bound: upper bound using the claimed C(32)=498 as the base case.
///
/// For dim=64: C(64) <= 3 * C(32) = 3 * 498 = 1494 (if the top level applies
/// the same Cariow reduction once, with each sub-problem solved by the existing
/// trigintaduonion mul_optimized path).
///
/// This bound is CONDITIONAL on C(32) = 498 being achievable.  The current
/// trigintaduonion::mul_optimized does NOT achieve 498; it uses 4 sedenion
/// multiplications = 4 * 256 = 1024 mults (the standard count).
///
/// Evidentiary class: C (conjecture; conditional on verified C(32)=498 implementation).
pub fn cariow_repo_bound(dim: usize) -> Option<usize> {
    match dim {
        2  => Some(4),
        4  => Some(8),    // Karatsuba quaternion (standard result)
        8  => Some(26),   // Cariow 2012 (published claim, evid C)
        16 => Some(84),   // Cariow 2013 (published claim, evid C)
        32 => Some(498),  // Repo trigintaduonion.rs claim (evid C, not yet implemented)
        64 => Some(3 * 498),  // = 1494, structural upper bound (evid C)
        _  => None,       // Unknown for other dimensions
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
    /// Repo/Cariow claimed count (from published or code claims). Evid C.
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
    println!("  Evid T: standard (exact). Evid C: bounds (conjecture).");
    println!();
    println!("{:>6}  {:>10}  {:>12}  {:>12}  {:>8}",
        "dim", "standard", "pure_3x_bnd", "repo_bound", "speedup");
    println!("{}", "-".repeat(56));
    for &d in &[2usize, 4, 8, 16, 32, 64] {
        let r = MultCountRecord::compute(d);
        let repo_str = r.repo_bound.map_or("?".to_string(), |v| v.to_string());
        let spd_str = r.speedup.map_or("?".to_string(), |v| format!("{:.2}x", v));
        println!("{:>6}  {:>10}  {:>12}  {:>12}  {:>8}",
            r.dim, r.standard, r.pure_3x_bound, repo_str, spd_str);
    }
    println!();
    println!("NOTE: repo_bound for dim=64 is 3*498=1494 (structural, unverified).");
    println!("NOTE: pure_3x_bound is a lower bound on the achievable Cariow count.");
    println!("NOTE: actual C(32)=498 in trigintaduonion::mul_optimized NOT yet achieved");
    println!("  (current impl falls back to 4*sedenion_multiply = 1024 mults).");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Standard CD multiplication count is exactly n^2.
    ///
    /// Follows directly from the CD doubling recursion S(2n) = 4*S(n), S(2) = 4.
    /// Evid T (theorem).
    #[test]
    fn test_standard_mult_count_exact() {
        assert_eq!(standard_mult_count(2),  4);
        assert_eq!(standard_mult_count(4),  16);
        assert_eq!(standard_mult_count(8),  64);
        assert_eq!(standard_mult_count(16), 256);
        assert_eq!(standard_mult_count(32), 1024);
        assert_eq!(standard_mult_count(64), 4096);
    }

    /// Pure 3x recursive bound is strictly less than standard for dim >= 4.
    #[test]
    fn test_pure_3x_bound_below_standard() {
        for &d in &[4usize, 8, 16, 32, 64] {
            let s = standard_mult_count(d);
            let b = cariow_pure_3x_bound(d);
            assert!(b < s,
                "pure_3x_bound({}) = {} should be < standard({}) = {}", d, b, d, s);
        }
    }

    /// Repo bound for dim=32 matches the trigintaduonion.rs docstring claim (498).
    ///
    /// Evid C (conjecture; implementation does not yet achieve this).
    #[test]
    fn test_repo_bound_dim32_matches_claim() {
        assert_eq!(cariow_repo_bound(32), Some(498),
            "Repo claims 498 mults for dim=32 (trigintaduonion.rs docstring)");
    }

    /// Repo bound for dim=64 is at most 3*498 = 1494.
    ///
    /// This is an upper bound from applying one level of Cariow reduction at the
    /// top, using the claimed C(32)=498 for each of 3 sub-problems.
    /// Evid C.
    #[test]
    fn test_repo_bound_dim64_upper_bound() {
        let bound = cariow_repo_bound(64).unwrap();
        assert_eq!(bound, 1494,
            "C(64) <= 3*C(32) = 3*498 = 1494 (structural upper bound)");
        assert!(bound < standard_mult_count(64),
            "Upper bound {} should be < standard {} for dim=64", bound, 4096);
        let speedup = cariow_speedup(64).unwrap();
        assert!(speedup > 2.5 && speedup < 3.0,
            "Expected speedup ~2.74x for dim=64, got {:.2}x", speedup);
    }

    /// Speedup is always > 1 at all known dimensions (Cariow is strictly faster).
    ///
    /// The speedup is NOT monotone: dim=32 (2.06x) < dim=16 (3.05x) because
    /// the repo's claimed 498 for dim=32 exceeds the pure 3*C(16)=252 lower bound.
    /// The dim=64 upper bound (1494) gives 2.74x, between the dim=16 and dim=32 values.
    /// Evid C.
    #[test]
    fn test_speedup_always_above_one() {
        for &d in &[4usize, 8, 16, 32, 64] {
            let spd = cariow_speedup(d).unwrap();
            assert!(spd > 1.0,
                "speedup at dim={} should be > 1.0, got {:.2}", d, spd);
        }
        // Specific known values (evid C: conditional on published claims)
        let spd16 = cariow_speedup(16).unwrap();
        let spd32 = cariow_speedup(32).unwrap();
        let spd64 = cariow_speedup(64).unwrap();
        assert!(spd16 > 3.0, "dim=16: 256/84 = 3.05x, got {:.2}", spd16);
        assert!(spd32 > 2.0, "dim=32: 1024/498 = 2.06x, got {:.2}", spd32);
        assert!(spd64 > 2.5, "dim=64: 4096/1494 = 2.74x, got {:.2}", spd64);
    }

    #[test]
    fn test_print_mult_count_table() {
        print_mult_count_table();
    }
}
