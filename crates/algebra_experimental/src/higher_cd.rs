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
            let mem_bytes = avt.violations.len() * std::mem::size_of::<(usize, usize, usize, usize, i32)>();
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
}

fn associator_basis(dim: usize, i: usize, j: usize, k: usize) -> (usize, i32) {
    let ij_idx = i ^ j;
    let ij_sign = cd_basis_mul_sign_iter(dim, i, j);
    let ijk_idx1 = ij_idx ^ k;
    let ijk_sign1 = ij_sign * cd_basis_mul_sign_iter(dim, ij_idx, k);

    let jk_idx = j ^ k;
    let jk_sign = cd_basis_mul_sign_iter(dim, j, k);
    let ijk_sign2 = jk_sign * cd_basis_mul_sign_iter(dim, i, jk_idx);

    (ijk_idx1, ijk_sign1 - ijk_sign2)
}
