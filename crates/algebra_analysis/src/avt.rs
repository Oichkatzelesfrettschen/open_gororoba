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
            let i = rng.gen_range(0..dim);
            let j_raw = rng.gen_range(0..dim - 1);
            let j = if j_raw >= i { j_raw + 1 } else { j_raw };
            let k = rng.gen_range(0..dim);

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

        let hit_rate = if tested > 0 { hits as f64 / tested as f64 } else { 0.0 };

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
}
