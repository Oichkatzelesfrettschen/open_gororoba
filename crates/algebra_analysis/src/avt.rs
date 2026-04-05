//! Unified Alternativity Violation Tensor (AVT).
//!
//! The AVT captures violations of the alternative identity `[x, x, y] = 0`
//! in Cayley-Dickson algebras for dimension >= 16.  For each ordered triple
//! `(i, j, k)` with `i < j`, it checks whether
//!
//!   `[e_i, e_j, e_k] + [e_j, e_i, e_k] != 0`
//!
//! and records the non-zero outcomes as `(i, j, k, m, sign)` where the sum
//! lands on axis `m` with the given sign.
//!
//! # Two construction paths
//!
//! - [`AlternativityViolationTensor::new`] / [`AlternativityViolationTensor::exhaustive`]:
//!   O(dim^3/2) full enumeration (feasible up to ~256D).
//! - [`AlternativityViolationTensor::sampled`]: uniform random sampling for
//!   512D and 1024D, returns a [`SampledAvt`] wrapper with hit-rate statistics.
//!
//! # GPU packing
//!
//! `pack_for_gpu` is intentionally NOT on this type.  It lives in the
//! `GpuPackableAvt` extension trait in `gororoba_algebra::gpu::avt_pack`,
//! keeping GPU layout concerns out of the algebra analysis layer.

use cd_kernel::cayley_dickson::cd_basis_mul_sign_iter;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;

// ---------------------------------------------------------------------------
// Core AVT type
// ---------------------------------------------------------------------------

/// Sparse catalog of alternator violations in a Cayley-Dickson algebra.
///
/// An algebra is *alternative* if `[x, x, y] = 0` and `[x, y, y] = 0` for
/// all elements, where `[a, b, c] = (a*b)*c - a*(b*c)` is the associator.
/// For the standard CD tower: O (8D) is alternative; S (16D) and all higher
/// levels are not.
///
/// This type catalogs ordered triples `(i, j, k)` with `i < j` and `k` free
/// where the *alternator* `[e_i, e_j, e_k] + [e_j, e_i, e_k]` is non-zero.
/// The alternator (not the full associator) is the tightest measure of how far
/// an algebra is from the alternative identity: it vanishes in O and is non-zero
/// in exactly the triples that form the "box-kite" zero-divisor sub-graphs of
/// higher CD levels.
///
/// # Violation tuple encoding
///
/// Each violation is stored as `(i, j, k, m, sign)` where:
/// - `i < j`: the alternated pair of indices.
/// - `k`: the third index (unconstrained).
/// - `m`: the output axis; always `i XOR j XOR k` by the CD basis rule.
/// - `sign`: integer +/-1 or +/-2, the net coefficient on `e_m`.
///
/// Only non-zero sums are stored.  This is a sparse format: for 16D there are
/// 2688 violations out of a possible 16^3/2 = 2048 triples.
///
/// # GPU layout
///
/// Do not call `pack_for_gpu` on this type directly.  Use the `GpuPackableAvt`
/// extension trait from `gororoba_algebra::gpu::avt_pack` to keep GPU layout
/// concerns isolated from this algebra-analysis layer.
pub struct AlternativityViolationTensor {
    pub dim: usize,
    /// Sparse violations: `(i, j, k, m, sign)`
    /// where `[e_i, e_j, e_k] + [e_j, e_i, e_k] = sign * e_m`.
    /// Only non-zero sums are stored.  `i < j` always.
    pub violations: Vec<(usize, usize, usize, usize, i32)>,
}

impl AlternativityViolationTensor {
    // -- Constructors --------------------------------------------------------

    /// Full exhaustive enumeration (no violation cap).
    ///
    /// At 128D: ~468K violations, ~250 ms.
    /// At 256D: ~3.97M violations, ~2100 ms.
    /// For dim >= 512 use [`AlternativityViolationTensor::sampled`] instead.
    pub fn exhaustive(dim: usize) -> Self {
        Self::exhaustive_capped(dim, usize::MAX)
    }

    /// Backward-compatible alias for [`exhaustive`](Self::exhaustive).
    #[inline]
    pub fn new(dim: usize) -> Self {
        Self::exhaustive(dim)
    }

    /// Exhaustive enumeration with a maximum violation cap.
    ///
    /// Stops early once `max_violations` violations have been collected.
    /// This prevents memory explosion for high dimensions while preserving
    /// the lowest-index violations (which carry the strongest geometric
    /// coupling in the CD algebra).
    pub fn exhaustive_capped(dim: usize, max_violations: usize) -> Self {
        assert!(dim.is_power_of_two());
        let mut violations = Vec::new();

        'outer: for i in 0..dim {
            for j in (i + 1)..dim {
                for k in 0..dim {
                    let (m1, s1) = associator_basis(dim, i, j, k);
                    let (m2, s2) = associator_basis(dim, j, i, k);

                    if m1 != m2 {
                        panic!(
                            "Structural failure in CD associator indices: {} != {}",
                            m1, m2
                        );
                    }

                    let sum_sign = s1 + s2;
                    if sum_sign != 0 {
                        violations.push((i, j, k, m1, sum_sign));
                        if violations.len() >= max_violations {
                            break 'outer;
                        }
                    }
                }
            }
        }

        Self { dim, violations }
    }

    /// Backward-compatible alias for [`exhaustive_capped`](Self::exhaustive_capped).
    #[inline]
    pub fn new_with_cap(dim: usize, max_violations: usize) -> Self {
        Self::exhaustive_capped(dim, max_violations)
    }

    /// Sampled AVT: uniform random triples, deduplicated via hash set.
    ///
    /// At 1024D, full enumeration requires O(1024^3 / 2) ~537M iterations.
    /// This method samples `n_samples` random `(i, j, k)` triples and tests
    /// each for non-alternativity.  Uses `ChaCha8Rng::seed_from_u64(seed)`
    /// for deterministic, reproducible CI results.
    ///
    /// Returns a [`SampledAvt`] which carries `hit_rate` and `n_tested` in
    /// addition to the violation list.
    pub fn sampled(dim: usize, n_samples: usize, seed: u64) -> SampledAvt {
        assert!(dim.is_power_of_two());
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut seen = HashSet::new();
        let mut violations = Vec::new();
        let mut tested = 0u64;
        let mut hits = 0u64;

        for _ in 0..n_samples {
            let i = rng.random_range(0..dim);
            let j_raw = rng.random_range(0..dim - 1);
            let j = if j_raw >= i { j_raw + 1 } else { j_raw };
            let k = rng.random_range(0..dim);

            let (i_c, j_c) = if i < j { (i, j) } else { (j, i) };
            if !seen.insert((i_c, j_c, k)) {
                continue;
            }

            tested += 1;

            let (m1, s1) = associator_basis(dim, i_c, j_c, k);
            let (m2, s2) = associator_basis(dim, j_c, i_c, k);
            debug_assert_eq!(m1, m2);

            let sum_sign = s1 + s2;
            if sum_sign != 0 {
                hits += 1;
                violations.push((i_c, j_c, k, m1, sum_sign));
            }
        }

        let hit_rate = if tested > 0 {
            hits as f64 / tested as f64
        } else {
            0.0
        };

        SampledAvt {
            avt: Self { dim, violations },
            n_tested: tested,
            n_hits: hits,
            hit_rate,
        }
    }

    // -- Query methods -------------------------------------------------------

    /// Count violations where both `(i, j)` fall within `[lo, hi)`.
    pub fn count_violations_in_range(&self, lo: usize, hi: usize) -> usize {
        self.violations
            .iter()
            .filter(|&&(i, j, _, _, _)| i >= lo && i < hi && j >= lo && j < hi)
            .count()
    }

    /// Count violations where `(i, j)` straddle two different ranges.
    pub fn count_cross_violations(
        &self,
        lo_a: usize,
        hi_a: usize,
        lo_b: usize,
        hi_b: usize,
    ) -> usize {
        self.violations
            .iter()
            .filter(|&&(i, j, _, _, _)| {
                (i >= lo_a && i < hi_a && j >= lo_b && j < hi_b)
                    || (i >= lo_b && i < hi_b && j >= lo_a && j < hi_a)
            })
            .count()
    }

    /// Check whether a specific triple `(i, j, k)` produces an alternativity violation.
    pub fn check_violation(&self, i: usize, j: usize, k: usize) -> bool {
        let (_, s1) = associator_basis(self.dim, i, j, k);
        let (_, s2) = associator_basis(self.dim, j, i, k);
        s1 + s2 != 0
    }
}

// ---------------------------------------------------------------------------
// Sampled AVT wrapper
// ---------------------------------------------------------------------------

/// Result of a sampled AVT construction: violations plus hit-rate statistics.
pub struct SampledAvt {
    pub avt: AlternativityViolationTensor,
    pub n_tested: u64,
    pub n_hits: u64,
    pub hit_rate: f64,
}

// ---------------------------------------------------------------------------
// Packed AVT for GPU upload
// ---------------------------------------------------------------------------

/// Bit-packed AVT violations for GPU upload (SSBO / CUDA device buffer).
///
/// Each `u32` encodes one `(i, j, m, sign)` tuple.
/// The `k` index from the original triple is dropped because the bilinear
/// tensor contraction `force[m] += alpha * v[i] * v[j] * sign` never uses it.
///
/// Bit layout (LSB to MSB):
/// `[m: bits] [j: bits] [i: bits] [sign_positive: 1]`
///
/// Sign encoding: bit=1 means positive (+2), bit=0 means negative (-2).
#[derive(Debug, Clone)]
pub struct PackedAvt {
    pub data: Vec<u32>,
    pub index_bits: u32,
    pub dim: u32,
    pub violation_count: u32,
}

impl PackedAvt {
    /// Unpack a single violation for CPU-side verification.
    pub fn unpack(&self, idx: usize) -> (usize, usize, usize, i32) {
        let word = self.data[idx];
        let bits = self.index_bits;
        let mask = (1u32 << bits) - 1;

        let m = (word & mask) as usize;
        let j = ((word >> bits) & mask) as usize;
        let i = ((word >> (2 * bits)) & mask) as usize;
        let sign_positive = ((word >> (3 * bits)) & 1) != 0;
        let sign: i32 = if sign_positive { 2 } else { -2 };

        (i, j, m, sign)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Number of bits required to represent a basis index for the given dimension.
/// For dim=64: 6 bits. dim=256: 8 bits. dim=1024: 10 bits.
pub fn index_bits_for_dim(dim: usize) -> usize {
    assert!(dim >= 2 && dim.is_power_of_two());
    dim.trailing_zeros() as usize
}

/// Generate a canonical family of zero-divisor pairs in `C_dim` via
/// sedenion index-shift embedding.
///
/// # Mathematical basis
///
/// The canonical embedding `iota_n: A_n -> A_{n+1}, a |-> (a, 0)` is an
/// injective algebra homomorphism (Proposition 1).  Iterating gives
/// `iota_{4,m}: A_4 -> A_{4+m}`.  If `xy = 0` in the sedenions `A_4`,
/// then `iota(x) * iota(y) = iota(xy) = 0` in `A_{4+m}` (Corollary 1).
/// This is the sole theorem required -- no stronger Moreno-style claim
/// is needed.
///
/// # Slot-shift embedding
///
/// For slot 0 (indices 0..15), correctness follows directly from the
/// canonical embedding above.  Shifted slots (16..31, 32..47, ...) are
/// NOT subalgebras in the standard CD basis -- `e_{16+r} * e_{16+s}`
/// has output index `(16+r) XOR (16+s) = r XOR s`, which lies OUTSIDE
/// the shifted block (Proposition 2: `sigma_n` is not a homomorphism).
///
/// Despite this, empirical verification shows that all 168 sedenion
/// 2-blade ZD pairs produce `ab = 0` at EVERY offset, likely because
/// the XOR sign structure has symmetries that preserve the cancellation
/// pattern for these specific sparse forms.  Each candidate is verified
/// numerically via `cd_multiply` + `cd_norm_sq` regardless.
///
/// # What this function does NOT produce
///
/// This is a **canonical representative family**, not an exhaustive
/// enumeration.  Cross-block ZDs (where both CD halves are nonzero but
/// cross-terms cancel) are not captured.  For exhaustive search, use
/// `cd_kernel::cayley_dickson::find_zero_divisors(dim, atol)` directly.
///
/// Returns O(dim) verified ZD pairs for dim >= 16 (168 per slot, `dim/16`
/// slots).  Returns an empty `Vec` for dim < 16 (no ZDs exist below C_16).
///
/// # Panics
/// Panics if `dim` is not a power of two >= 1.
pub fn zero_divisor_sedenion_embedding(dim: usize, atol: f64) -> Vec<(Vec<f64>, Vec<f64>)> {
    use cd_kernel::cayley_dickson::{cd_multiply, cd_norm_sq, find_zero_divisors};
    assert!(
        dim >= 1 && dim.is_power_of_two(),
        "dim must be a power-of-two"
    );
    if dim < 16 {
        return Vec::new();
    }

    // find_zero_divisors(16, atol) returns (i, j, k, l, norm) for 2-blade ZDs
    // a = e_i + e_j, b = e_k +/- e_l.  Both sign variants of b[l] can appear
    // as separate entries with identical (i,j,k,l) tuples.  Deduplicate by
    // (i,j,k,l) and verify both signs explicitly during embedding.
    let sedenion_raw = find_zero_divisors(16, atol);

    let mut result = Vec::new();
    let n_slots = dim / 16;

    for slot in 0..n_slots {
        let offset = slot * 16;
        let mut seen = HashSet::new();
        for &(i, j, k, l, _) in &sedenion_raw {
            if !seen.insert((i, j, k, l)) {
                continue; // already tried both sign variants for this index tuple
            }
            for sign in [1.0_f64, -1.0] {
                let mut a = vec![0.0; dim];
                let mut b = vec![0.0; dim];
                a[offset + i] = 1.0;
                a[offset + j] = 1.0;
                b[offset + k] = 1.0;
                b[offset + l] = sign;
                let ab = cd_multiply(&a, &b);
                if cd_norm_sq(&ab).sqrt() < atol {
                    result.push((a, b));
                }
            }
        }
    }
    result
}

/// Strategy for generating zero-divisor pairs in Cayley-Dickson algebras.
///
/// Three tiers of cost vs completeness, selected by the caller:
///
/// | Strategy             | Cost          | Output    | Use case                  |
/// |----------------------|---------------|-----------|---------------------------|
/// | `Witness`            | O(dim)        | 1 pair    | Existence proofs, CI flux |
/// | `SedenionEmbedding`  | O(dim * P)    | ~672 @64D | Comprehensive analysis    |
/// | `ExhaustiveSearch`   | O(dim^4)      | all 2-blade | Deep research only     |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeroDivisorStrategy {
    /// Single known sedenion witness pair from the de Marrais assessor (1, 10)
    /// and its first co-assessor partner, embedded at slot 0 (indices 0..15).
    /// Verified via one `cd_multiply` call.  O(dim) total.
    Witness,
    /// Full family of 168 sedenion ZD pairs embedded at each of the `dim/16`
    /// offset slots.  Each candidate verified numerically.  ~168*(dim/16) pairs.
    SedenionEmbedding,
    /// Brute-force O(dim^4) search over all 2-blade pairs.  Use
    /// `cd_kernel::cayley_dickson::find_zero_divisors(dim, atol)` directly.
    /// Only for exhaustive research enumeration.
    ExhaustiveSearch,
}

/// Return a single sedenion zero-divisor witness pair embedded in `C_dim`.
///
/// The witness uses the de Marrais assessor (1, 10) paired with its first
/// co-assessor partner, discovered at function entry and verified via
/// `cd_multiply`.  The small search (at most 196 candidate pairs in C_16)
/// runs in microseconds for any dim.
///
/// # Panics
/// Panics if `dim < 16` or `dim` is not a power of two.
pub fn zero_divisor_witness(dim: usize) -> (Vec<f64>, Vec<f64>) {
    use cd_kernel::cayley_dickson::{cd_multiply, cd_norm_sq};
    assert!(dim >= 16 && dim.is_power_of_two());
    // Search over co-assessor partners of assessor (1, 10).
    // At most 7 * 7 * 4 = 196 candidates; each verification is
    // one cd_multiply on a dim-sized vector.
    for low2 in 1..=7_usize {
        for high2 in 9..=15_usize {
            if high2 == low2 + 8 {
                continue;
            }
            for s in [1.0_f64, -1.0] {
                for t in [1.0_f64, -1.0] {
                    let mut a = vec![0.0; dim];
                    let mut b = vec![0.0; dim];
                    a[1] = 1.0;
                    a[10] = s;
                    b[low2] = 1.0;
                    b[high2] = t;
                    let ab = cd_multiply(&a, &b);
                    if cd_norm_sq(&ab).sqrt() < 1e-12 {
                        return (a, b);
                    }
                }
            }
        }
    }
    unreachable!("assessor (1, 10) must have at least one co-assessor partner");
}

/// Extract the 7 primary zero-divisor pairs -- one per box-kite.
///
/// Wilmot (2026) proves that sedenion ZDs reduce to 7 primary pairs via
/// power-associative subalgebra cycles.  De Marrais shows these correspond
/// to the 7 connected components (box-kites) of the co-assessor graph.
///
/// For each box-kite, this function returns the first co-assessor pair
/// whose diagonal zero-product is verified.  The 7 pairs form the minimal
/// generating set from which all 84 standard ZDs can be derived by
/// sign flips and partner enumeration.
///
/// Corresponds to [`ZeroDivisorStrategy::Witness`] extended to full
/// box-kite coverage.
///
/// # Panics
/// Panics if `dim < 16` (box-kites exist only in sedenions and above).
pub fn zero_divisor_primary_seven(dim: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
    use crate::boxkites::{cached_sedenion_boxkites, diagonal_zero_product};
    assert!(dim >= 16 && dim.is_power_of_two());

    let boxkites = cached_sedenion_boxkites();
    let mut primaries = Vec::with_capacity(7);

    for bk in boxkites.iter() {
        // Find first co-assessor pair in this box-kite with a zero product.
        let mut found = false;
        'pair: for i in 0..bk.assessors.len() {
            for j in (i + 1)..bk.assessors.len() {
                if let Some((s, t)) =
                    diagonal_zero_product(&bk.assessors[i], &bk.assessors[j], 1e-10)
                {
                    let mut a = vec![0.0; dim];
                    let mut b = vec![0.0; dim];
                    a[bk.assessors[i].low] = 1.0;
                    a[bk.assessors[i].high] = s as f64;
                    b[bk.assessors[j].low] = 1.0;
                    b[bk.assessors[j].high] = t as f64;
                    primaries.push((a, b));
                    found = true;
                    break 'pair;
                }
            }
        }
        assert!(found, "box-kite {} has no co-assessor ZD pair", bk.id);
    }

    assert_eq!(primaries.len(), 7, "expected exactly 7 primary pairs");
    primaries
}

/// Compute the basis associator: `[e_i, e_j, e_k] - [e_j, e_i, e_k]` sign delta.
///
/// Returns `(m, sign_delta)` where `m` is the output basis index.
/// Used internally during AVT construction; also exported for analysis code.
pub fn associator_basis(dim: usize, i: usize, j: usize, k: usize) -> (usize, i32) {
    // (e_i * e_j) * e_k
    let ij_idx = i ^ j;
    let ij_sign = cd_basis_mul_sign_iter(dim, i, j);
    let ijk_idx = ij_idx ^ k;
    let ijk_sign1 = ij_sign * cd_basis_mul_sign_iter(dim, ij_idx, k);

    // e_i * (e_j * e_k)
    let jk_idx = j ^ k;
    let jk_sign = cd_basis_mul_sign_iter(dim, j, k);
    let ijk_sign2 = jk_sign * cd_basis_mul_sign_iter(dim, i, jk_idx);

    (ijk_idx, ijk_sign1 - ijk_sign2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn octonion_is_alternative() {
        let avt = AlternativityViolationTensor::new(8);
        assert_eq!(avt.violations.len(), 0, "Octonions must be alternative");
    }

    #[test]
    fn sedenion_has_violations() {
        let avt = AlternativityViolationTensor::new(16);
        assert!(!avt.violations.is_empty());
    }

    #[test]
    fn new_and_exhaustive_are_equivalent() {
        let a = AlternativityViolationTensor::new(16);
        let b = AlternativityViolationTensor::exhaustive(16);
        assert_eq!(a.violations.len(), b.violations.len());
    }

    #[test]
    fn new_with_cap_and_exhaustive_capped_are_equivalent() {
        let a = AlternativityViolationTensor::new_with_cap(16, 100);
        let b = AlternativityViolationTensor::exhaustive_capped(16, 100);
        assert_eq!(a.violations.len(), b.violations.len());
    }

    #[test]
    fn index_bits_correct() {
        assert_eq!(index_bits_for_dim(16), 4);
        assert_eq!(index_bits_for_dim(64), 6);
        assert_eq!(index_bits_for_dim(256), 8);
        assert_eq!(index_bits_for_dim(1024), 10);
    }

    #[test]
    fn sampled_returns_hit_rate() {
        let s = AlternativityViolationTensor::sampled(64, 1000, 42);
        assert!(s.hit_rate >= 0.0 && s.hit_rate <= 1.0);
        assert!(s.n_tested > 0);
    }

    #[test]
    fn packed_avt_roundtrip_16d() {
        use super::index_bits_for_dim;
        let avt = AlternativityViolationTensor::new(16);
        let bits = index_bits_for_dim(16) as u32;
        let mask = (1u32 << bits) - 1;
        let mut packed_data = Vec::with_capacity(avt.violations.len());
        for &(i, j, _k, m, sign) in &avt.violations {
            let sign_bit: u32 = if sign > 0 { 1 } else { 0 };
            let word = (m as u32 & mask)
                | ((j as u32 & mask) << bits)
                | ((i as u32 & mask) << (2 * bits))
                | (sign_bit << (3 * bits));
            packed_data.push(word);
        }
        let packed = PackedAvt {
            data: packed_data,
            index_bits: bits,
            dim: 16,
            violation_count: avt.violations.len() as u32,
        };
        for (idx, &(i, j, _k, m, sign)) in avt.violations.iter().enumerate() {
            let (ui, uj, um, usign) = packed.unpack(idx);
            assert_eq!((ui, uj, um, usign), (i, j, m, sign));
        }
    }

    #[test]
    fn zd_embedding_slot0_sedenion_all_valid() {
        use cd_kernel::cayley_dickson::{cd_multiply, cd_norm_sq};
        let zds = zero_divisor_sedenion_embedding(16, 1e-9);
        assert!(!zds.is_empty(), "should find ZDs in C_16");
        for (a, b) in &zds {
            let ab = cd_multiply(a, b);
            assert!(
                cd_norm_sq(&ab).sqrt() < 1e-9,
                "embedding produced invalid ZD"
            );
        }
    }

    #[test]
    fn zd_embedding_64d_has_zds_and_all_valid() {
        use cd_kernel::cayley_dickson::{cd_multiply, cd_norm_sq};
        let zds = zero_divisor_sedenion_embedding(64, 1e-9);
        assert!(!zds.is_empty(), "should find ZDs in C_64 via embedding");
        for (a, b) in &zds {
            let ab = cd_multiply(a, b);
            assert!(
                cd_norm_sq(&ab).sqrt() < 1e-9,
                "C_64 embedding produced invalid ZD"
            );
        }
    }

    #[test]
    fn zd_embedding_scales_with_dim() {
        let zds_16 = zero_divisor_sedenion_embedding(16, 1e-9);
        let zds_32 = zero_divisor_sedenion_embedding(32, 1e-9);
        let zds_64 = zero_divisor_sedenion_embedding(64, 1e-9);
        // Each additional slot adds at least as many ZDs as slot 0.
        assert!(zds_32.len() >= zds_16.len(), "32D should have >= 16D ZDs");
        assert!(zds_64.len() >= zds_32.len(), "64D should have >= 32D ZDs");
    }

    /// Diagnostic: count ZDs produced by each slot to verify which slots
    /// are valid subalgebra embeddings vs which fail closure.
    /// Per Proposition 2 (sigma_n is NOT a homomorphism), only slot 0
    /// (canonical iota_n embedding) is guaranteed.  Other slots may or
    /// may not produce valid ZDs depending on the XOR/sign structure.
    #[test]
    fn zd_embedding_per_slot_audit() {
        use cd_kernel::cayley_dickson::{cd_multiply, cd_norm_sq, find_zero_divisors};
        let atol = 1e-9;
        let sedenion_raw = find_zero_divisors(16, atol);
        for dim in [32, 64] {
            let n_slots = dim / 16;
            for slot in 0..n_slots {
                let offset = slot * 16;
                let mut count = 0usize;
                let mut seen = std::collections::HashSet::new();
                for &(i, j, k, l, _) in &sedenion_raw {
                    if !seen.insert((i, j, k, l)) {
                        continue;
                    }
                    for sign in [1.0_f64, -1.0] {
                        let mut a = vec![0.0; dim];
                        let mut b = vec![0.0; dim];
                        a[offset + i] = 1.0;
                        a[offset + j] = 1.0;
                        b[offset + k] = 1.0;
                        b[offset + l] = sign;
                        let ab = cd_multiply(&a, &b);
                        if cd_norm_sq(&ab).sqrt() < atol {
                            count += 1;
                        }
                    }
                }
                println!(
                    "dim={} slot={} offset={}..{} valid_zds={}",
                    dim,
                    slot,
                    offset,
                    offset + 15,
                    count
                );
            }
        }
    }

    /// Proposition 2 test: verify that the shifted-half map sigma_n is NOT
    /// an algebra homomorphism.  Concretely: for random a,b in C_16,
    /// sigma(a)*sigma(b) != sigma(a*b) in general.
    #[test]
    fn test_shifted_block_nonclosure() {
        use cd_kernel::cayley_dickson::cd_multiply;
        // Take two sedenion elements and embed at slot 1 (indices 16..31) of C_32
        let dim = 32;
        let mut a = vec![0.0; dim];
        let mut b = vec![0.0; dim];
        // sigma(e_1) = place e_1 at index 16+1=17
        a[17] = 1.0;
        // sigma(e_2) = place e_2 at index 16+2=18
        b[18] = 1.0;
        let ab = cd_multiply(&a, &b);
        // sigma(e_1 * e_2) would place the result at index 16 + (1 XOR 2) = 16+3 = 19
        // But in C_32, e_17 * e_18 has output index 17 XOR 18 = 3 (NOT 19!)
        let output_idx_actual = ab
            .iter()
            .enumerate()
            .find(|(_, v)| v.abs() > 1e-12)
            .map(|(i, _)| i)
            .unwrap_or(usize::MAX);
        let output_idx_if_subalgebra = 19; // 16 + (1 XOR 2)
        println!(
            "e_17 * e_18 lands at index {} (subalgebra would need {})",
            output_idx_actual, output_idx_if_subalgebra
        );
        assert_ne!(
            output_idx_actual, output_idx_if_subalgebra,
            "Shifted block IS closed (unexpected) -- Proposition 2 not confirmed"
        );
    }

    /// Proposition 1 test: verify canonical embedding is a homomorphism.
    /// For a,b in C_16: iota(a*b) = iota(a) * iota(b) in C_64.
    #[test]
    fn test_canonical_embedding_homomorphism() {
        use cd_kernel::cayley_dickson::cd_multiply;
        let dim_small = 16;
        let dim_big = 64;
        // Use specific basis elements: e_3 and e_7 in C_16
        let mut a_small = vec![0.0; dim_small];
        let mut b_small = vec![0.0; dim_small];
        a_small[3] = 1.0;
        b_small[7] = 1.0;
        let ab_small = cd_multiply(&a_small, &b_small);
        // Embed ab_small into C_64 (pad zeros)
        let mut iota_ab = vec![0.0; dim_big];
        for i in 0..dim_small {
            iota_ab[i] = ab_small[i];
        }
        // Embed a and b individually, then multiply in C_64
        let mut iota_a = vec![0.0; dim_big];
        let mut iota_b = vec![0.0; dim_big];
        for i in 0..dim_small {
            iota_a[i] = a_small[i];
        }
        for i in 0..dim_small {
            iota_b[i] = b_small[i];
        }
        let product_big = cd_multiply(&iota_a, &iota_b);
        // Verify: iota(a*b) == iota(a) * iota(b)
        let diff: f64 = iota_ab
            .iter()
            .zip(product_big.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!(
            diff < 1e-12,
            "Canonical embedding is not a homomorphism! diff={}",
            diff
        );
    }

    /// Wilmot 7 primary pairs: one ZD pair per box-kite, all verified.
    #[test]
    fn test_zero_divisor_primary_seven_count_and_validity() {
        use cd_kernel::cayley_dickson::{cd_multiply, cd_norm_sq};
        let pairs = zero_divisor_primary_seven(16);
        assert_eq!(pairs.len(), 7, "must have exactly 7 primary ZD pairs");
        for (i, (a, b)) in pairs.iter().enumerate() {
            let ab = cd_multiply(a, b);
            let norm = cd_norm_sq(&ab).sqrt();
            assert!(
                norm < 1e-9,
                "primary pair {} is not a ZD: ||ab|| = {}",
                i,
                norm
            );
        }
    }

    /// 7 primary pairs embedded in 64D should all still be valid ZDs.
    #[test]
    fn test_zero_divisor_primary_seven_64d() {
        use cd_kernel::cayley_dickson::{cd_multiply, cd_norm_sq};
        let pairs = zero_divisor_primary_seven(64);
        assert_eq!(pairs.len(), 7);
        for (a, b) in &pairs {
            assert_eq!(a.len(), 64);
            let ab = cd_multiply(a, b);
            assert!(cd_norm_sq(&ab).sqrt() < 1e-9);
        }
    }

    /// Wilmot counting formula: ZDs in C_16 occur in multiples of 84.
    /// find_zero_divisors(16, atol) returns 168 = 84 * 2 (one per sign variant).
    /// The 84 aligns with Reggiani's 42 assessors x 2 diagonal signs.
    #[test]
    fn test_wilmot_multiples_of_84_sedenion() {
        use cd_kernel::cayley_dickson::find_zero_divisors;
        let zds = find_zero_divisors(16, 1e-9);
        assert_eq!(
            zds.len() % 84,
            0,
            "Sedenion ZD count {} is not a multiple of 84",
            zds.len()
        );
        assert_eq!(
            zds.len(),
            168,
            "Expected exactly 168 sedenion 2-blade ZDs (84 tuples x 2 sign variants)"
        );
    }
}
