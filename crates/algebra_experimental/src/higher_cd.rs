use cd_kernel::cayley_dickson::cd_basis_mul_sign_iter;
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use std::collections::HashSet;
use wide::f64x4;

macro_rules! implement_higher_cd {
    ($name:ident, $dim:expr, $simd_count:expr) => {
        #[derive(Clone, Copy, Debug)]
        pub struct $name {
            pub data: [f64x4; $simd_count],
        }

        impl $name {
            pub fn zero() -> Self {
                Self {
                    data: [f64x4::splat(0.0); $simd_count],
                }
            }

            pub fn mul(&self, other: &Self) -> Self {
                let mut res = [0.0; $dim];
                let self_slice = self.to_slice();
                let other_slice = other.to_slice();

                for (k, res_k) in res.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for (i, &self_i) in self_slice.iter().enumerate() {
                        let j = k ^ i;
                        let sign = cd_basis_mul_sign_iter($dim, i, j);
                        sum += (sign as f64) * self_i * other_slice[j];
                    }
                    *res_k = sum;
                }

                Self::from_slice(&res)
            }

            pub fn from_slice(slice: &[f64; $dim]) -> Self {
                let mut data = [f64x4::splat(0.0); $simd_count];
                for i in 0..$simd_count {
                    data[i] = f64x4::from([
                        slice[i * 4],
                        slice[i * 4 + 1],
                        slice[i * 4 + 2],
                        slice[i * 4 + 3],
                    ]);
                }
                Self { data }
            }

            pub fn to_slice(&self) -> [f64; $dim] {
                let mut res = [0.0; $dim];
                for i in 0..$simd_count {
                    let arr = self.data[i].to_array();
                    res[i * 4] = arr[0];
                    res[i * 4 + 1] = arr[1];
                    res[i * 4 + 2] = arr[2];
                    res[i * 4 + 3] = arr[3];
                }
                res
            }
        }
    };
}

implement_higher_cd!(Routon, 128, 32);
implement_higher_cd!(Voudon, 256, 64);
implement_higher_cd!(Eriston, 512, 128);
implement_higher_cd!(DekaVoudon, 1024, 256);

/// Universal properties of Cayley-Dickson algebras for n >= 4.
pub struct UniversalCDProperties;

impl UniversalCDProperties {
    /// All CD algebras are power-associative: x^n is well-defined.
    pub fn is_power_associative() -> bool { true }

    /// All CD algebras are flexible: x(yx) = (xy)x.
    pub fn is_flexible() -> bool { true }

    /// CD algebras for dim >= 16 contain zero divisors.
    pub fn has_zero_divisors(dim: usize) -> bool { dim >= 16 }
}

/// Systematic naming for higher Cayley-Dickson algebras.
pub fn cd_name(dim: usize) -> &'static str {
    match dim {
        1 => "Real",
        2 => "Complex",
        4 => "Quaternion",
        8 => "Octonion",
        16 => "Sedenion",
        32 => "Trigintaduonion",
        64 => "Sexagintaquatronion",
        128 => "Routon",
        256 => "Voudon",
        512 => "Eriston",
        1024 => "DekaVoudon",
        _ => "Higher 2^n-ion",
    }
}

/// Theoretical derivation dimensions for CD algebras.
/// - Der(C) = 0
/// - Der(H) = su(2) (3D)
/// - Der(O) = g2 (14D)
/// - Der(S) = ? (Identified as G2-related in synthesis)
pub fn derivation_dim(dim: usize) -> Option<usize> {
    match dim {
        2 => Some(0),
        4 => Some(3),
        8 => Some(14),
        16 => None, // Open problem
        32 => None, // Under investigation
        _ => None,
    }
}

/// Higher-dimensional Alternativity Violation Tensor (AVT)
pub struct HigherAvt {
    pub dim: usize,
    pub violations: Vec<(usize, usize, usize, usize, i32)>,
}

impl HigherAvt {
    pub fn new(dim: usize) -> Self {
        assert!(dim.is_power_of_two());
        let mut violations = Vec::new();

        for i in 0..dim {
            for j in (i + 1)..dim {
                for k in 0..dim {
                    let (m1, s1) = associator_basis(dim, i, j, k);
                    let (m2, s2) = associator_basis(dim, j, i, k);

                    debug_assert_eq!(m1, m2);

                    let sum_sign = s1 + s2;
                    if sum_sign != 0 {
                        violations.push((i, j, k, m1, sum_sign));
                    }
                }
            }
            // Limit violations for very high dimensions to prevent memory explosion.
            // 10M cap accommodates 256D (~3.97M violations) with headroom for 512D sampling.
            if violations.len() > 10_000_000 {
                break;
            }
        }

        Self { dim, violations }
    }

    /// Construct an AVT by uniformly sampling random (i,j,k) triples.
    ///
    /// At 1024D, full enumeration requires O(1024^3/2) ~ 537M iterations,
    /// which is infeasible. Instead, we sample `n_samples` random triples
    /// and test each for non-alternativity. Duplicates are deduplicated
    /// via a hash set on (i,j,k) to avoid double-counting.
    ///
    /// The `hit_rate` field records the fraction of sampled triples that
    /// produced a violation, which estimates the global violation density.
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
            // Ensure j != i by mapping [0, dim-2) past i
            let j = if j_raw >= i { j_raw + 1 } else { j_raw };
            let k = rng.gen_range(0..dim);

            // Canonical ordering: i < j for dedup
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
            avt: HigherAvt { dim, violations },
            n_tested: tested,
            n_hits: hits,
            hit_rate,
        }
    }

    /// Count violations where both (i,j) indices fall within a given axis range.
    /// Useful for measuring intra-sector violation density.
    pub fn count_violations_in_range(&self, lo: usize, hi: usize) -> usize {
        self.violations
            .iter()
            .filter(|&&(i, j, _, _, _)| i >= lo && i < hi && j >= lo && j < hi)
            .count()
    }

    /// Count violations where (i,j) straddle two different axis ranges.
    /// Measures cross-sector coupling between sub-blocks.
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
}

/// Result of sampled AVT construction with statistics.
pub struct SampledAvt {
    pub avt: HigherAvt,
    pub n_tested: u64,
    pub n_hits: u64,
    pub hit_rate: f64,
}

#[inline(always)]
pub(crate) fn associator_basis(dim: usize, i: usize, j: usize, k: usize) -> (usize, i32) {
    let ij_idx = i ^ j;
    let ij_sign = cd_basis_mul_sign_iter(dim, i, j);
    let ijk_idx1 = ij_idx ^ k;
    let ijk_sign1 = ij_sign * cd_basis_mul_sign_iter(dim, ij_idx, k);

    let jk_idx = j ^ k;
    let jk_sign = cd_basis_mul_sign_iter(dim, j, k);
    let ijk_sign2 = jk_sign * cd_basis_mul_sign_iter(dim, i, jk_idx);

    (ijk_idx1, ijk_sign1 - ijk_sign2)
}

/// Sparse representation of a high-dimensional CD algebra element.
///
/// At 1024D (DekaVoudon), a dense `[f64; 1024]` costs 8KB per element.
/// For physical states with few active axes (e.g., QEC syndrome vectors,
/// lattice site states), sparse storage via sorted `(axis, value)` pairs
/// reduces memory and enables O(k) inner products where k << dim.
#[derive(Debug, Clone)]
pub struct SparseApeironState {
    /// Algebra dimension (must be power of 2).
    pub dim: usize,
    /// Nonzero components as (axis_index, value), sorted by axis_index.
    pub components: Vec<(usize, f64)>,
}

impl SparseApeironState {
    /// Create a zero state in the given dimension.
    pub fn zero(dim: usize) -> Self {
        Self {
            dim,
            components: Vec::new(),
        }
    }

    /// Create from a dense slice, dropping near-zero entries.
    pub fn from_dense(dim: usize, values: &[f64], threshold: f64) -> Self {
        let components: Vec<(usize, f64)> = values
            .iter()
            .enumerate()
            .filter(|(_, v)| v.abs() > threshold)
            .map(|(i, &v)| (i, v))
            .collect();
        Self { dim, components }
    }

    /// Create from explicit (axis, value) pairs. Sorts by axis index.
    pub fn from_pairs(dim: usize, mut pairs: Vec<(usize, f64)>) -> Self {
        pairs.sort_by_key(|(idx, _)| *idx);
        Self {
            dim,
            components: pairs,
        }
    }

    /// Number of nonzero components.
    pub fn nnz(&self) -> usize {
        self.components.len()
    }

    /// Sparsity ratio: 1.0 = fully zero, 0.0 = fully dense.
    pub fn sparsity(&self) -> f64 {
        if self.dim == 0 {
            return 1.0;
        }
        1.0 - (self.components.len() as f64 / self.dim as f64)
    }

    /// Squared L2 norm.
    pub fn norm_sq(&self) -> f64 {
        self.components.iter().map(|(_, v)| v * v).sum()
    }

    /// L2 norm.
    pub fn norm(&self) -> f64 {
        self.norm_sq().sqrt()
    }

    /// Inner product with another sparse state.
    ///
    /// Uses merge-join on sorted component lists for O(k1 + k2) complexity.
    pub fn dot(&self, other: &Self) -> f64 {
        let mut result = 0.0;
        let mut i = 0;
        let mut j = 0;

        while i < self.components.len() && j < other.components.len() {
            let (ai, vi) = self.components[i];
            let (aj, vj) = other.components[j];

            match ai.cmp(&aj) {
                std::cmp::Ordering::Less => i += 1,
                std::cmp::Ordering::Greater => j += 1,
                std::cmp::Ordering::Equal => {
                    result += vi * vj;
                    i += 1;
                    j += 1;
                }
            }
        }

        result
    }

    /// Convert to dense representation.
    pub fn to_dense(&self) -> Vec<f64> {
        let mut dense = vec![0.0; self.dim];
        for &(idx, val) in &self.components {
            dense[idx] = val;
        }
        dense
    }

    /// Scale all components by a constant.
    pub fn scale(&mut self, factor: f64) {
        for (_, v) in &mut self.components {
            *v *= factor;
        }
    }

    /// Add another sparse state, merging component lists.
    pub fn add(&self, other: &Self) -> Self {
        let mut result = Vec::with_capacity(self.components.len() + other.components.len());
        let mut i = 0;
        let mut j = 0;

        while i < self.components.len() && j < other.components.len() {
            let (ai, vi) = self.components[i];
            let (aj, vj) = other.components[j];

            match ai.cmp(&aj) {
                std::cmp::Ordering::Less => {
                    result.push((ai, vi));
                    i += 1;
                }
                std::cmp::Ordering::Greater => {
                    result.push((aj, vj));
                    j += 1;
                }
                std::cmp::Ordering::Equal => {
                    let sum = vi + vj;
                    if sum.abs() > 1e-15 {
                        result.push((ai, sum));
                    }
                    i += 1;
                    j += 1;
                }
            }
        }

        // Drain remaining
        while i < self.components.len() {
            result.push(self.components[i]);
            i += 1;
        }
        while j < other.components.len() {
            result.push(other.components[j]);
            j += 1;
        }

        Self {
            dim: self.dim,
            components: result,
        }
    }

    /// Shannon entropy of the component magnitude distribution.
    ///
    /// Measures how uniformly the nonzero components are distributed.
    /// H = 0 for a single-axis state, H = ln(k) for k equal components.
    pub fn shannon_entropy(&self) -> f64 {
        let total: f64 = self.components.iter().map(|(_, v)| v.abs()).sum();
        if total < 1e-15 {
            return 0.0;
        }
        let mut h = 0.0;
        for &(_, v) in &self.components {
            let p = v.abs() / total;
            if p > 1e-15 {
                h -= p * p.ln();
            }
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn profile_higher_avt_construction() {
        // Warmup
        let _ = HigherAvt::new(16);

        for dim in [16, 32, 64, 128, 256] {
            let t = Instant::now();
            let avt = HigherAvt::new(dim);
            let elapsed = t.elapsed();
            let mem_bytes =
                avt.violations.len() * std::mem::size_of::<(usize, usize, usize, usize, i32)>();
            eprintln!(
                "HigherAvt::new({:>4}): {:>8} violations, {:>10.3}ms, {:.1} MB",
                dim,
                avt.violations.len(),
                elapsed.as_secs_f64() * 1000.0,
                mem_bytes as f64 / 1e6,
            );
        }
    }

    #[test]
    fn profile_sampled_avt_512_1024() {
        for (dim, n_samples) in [(512, 1_000_000), (1024, 1_000_000)] {
            let t = Instant::now();
            let result = HigherAvt::sampled(dim, n_samples, 42);
            let elapsed = t.elapsed();
            eprintln!(
                "HigherAvt::sampled({:>4}, {}): {:>8} violations, hit_rate={:.4}, {:>10.3}ms",
                dim,
                n_samples,
                result.avt.violations.len(),
                result.hit_rate,
                elapsed.as_secs_f64() * 1000.0,
            );
        }
    }

    #[test]
    fn test_sparse_apeiron_zero() {
        let s = SparseApeironState::zero(1024);
        assert_eq!(s.nnz(), 0);
        assert!((s.sparsity() - 1.0).abs() < 1e-10);
        assert!((s.norm() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_apeiron_from_dense() {
        let mut dense = vec![0.0; 256];
        dense[0] = 1.0;
        dense[42] = -3.0;
        dense[100] = 0.5;

        let s = SparseApeironState::from_dense(256, &dense, 1e-10);
        assert_eq!(s.nnz(), 3);
        assert!((s.norm_sq() - (1.0 + 9.0 + 0.25)).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_apeiron_dot_product() {
        let a = SparseApeironState::from_pairs(128, vec![(0, 1.0), (5, 2.0), (10, 3.0)]);
        let b = SparseApeironState::from_pairs(128, vec![(0, 4.0), (5, -1.0), (20, 7.0)]);
        // dot = 1*4 + 2*(-1) + 0 = 2.0
        let d = a.dot(&b);
        assert!((d - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_apeiron_add() {
        let a = SparseApeironState::from_pairs(64, vec![(0, 1.0), (5, 2.0)]);
        let b = SparseApeironState::from_pairs(64, vec![(5, -2.0), (10, 3.0)]);
        let c = a.add(&b);
        // axis 0: 1.0, axis 5: 0.0 (cancelled), axis 10: 3.0
        assert_eq!(c.nnz(), 2);
        let dense = c.to_dense();
        assert!((dense[0] - 1.0).abs() < 1e-10);
        assert!((dense[5] - 0.0).abs() < 1e-10);
        assert!((dense[10] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_apeiron_shannon_entropy() {
        // Single component: H = 0
        let s1 = SparseApeironState::from_pairs(1024, vec![(42, 1.0)]);
        assert!((s1.shannon_entropy() - 0.0).abs() < 1e-10);

        // Two equal components: H = ln(2)
        let s2 = SparseApeironState::from_pairs(1024, vec![(0, 1.0), (1, 1.0)]);
        assert!((s2.shannon_entropy() - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_sparse_apeiron_roundtrip() {
        let pairs = vec![(3, 1.5), (17, -0.7), (255, 42.0)];
        let s = SparseApeironState::from_pairs(256, pairs);
        let dense = s.to_dense();
        let s2 = SparseApeironState::from_dense(256, &dense, 1e-10);
        assert_eq!(s.nnz(), s2.nnz());
        assert!((s.dot(&s2) - s.norm_sq()).abs() < 1e-10);
    }
}
