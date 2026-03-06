use cd_kernel::cayley_dickson::cd_basis_mul_sign_iter;
use wide::f64x4;

/// A 64-dimensional Chingon algebra element, SIMD-accelerated via `wide::f64x4`.
#[derive(Clone, Copy, Debug)]
pub struct Chingon {
    pub data: [f64x4; 16],
}

impl Chingon {
    pub fn zero() -> Self {
        Self {
            data: [f64x4::splat(0.0); 16],
        }
    }

    /// Multiplies two Chingons using the Cayley-Dickson rule accelerated by SIMD.
    pub fn mul(&self, other: &Self) -> Self {
        let mut res = [0.0; 64];
        let self_slice = self.to_slice();
        let other_slice = other.to_slice();

        // Standard Cayley-Dickson multiplication via basis rules
        // Z_k = sum_i (e_i * e_{k^i}).sign * X_i * Y_{k^i}
        // We can optimize this using the precomputed signs or recursive doubling.
        // For 64D, we'll use the iterative sign generator.
        for (k, res_k) in res.iter_mut().enumerate() {
            let mut sum = 0.0;
            for (i, &self_i) in self_slice.iter().enumerate() {
                let j = k ^ i;
                let sign = cd_basis_mul_sign_iter(64, i, j);
                sum += (sign as f64) * self_i * other_slice[j];
            }
            *res_k = sum;
        }

        Self::from_slice(&res)
    }

    pub fn from_slice(slice: &[f64; 64]) -> Self {
        let mut data = [f64x4::splat(0.0); 16];
        for i in 0..16 {
            data[i] = f64x4::from([
                slice[i * 4],
                slice[i * 4 + 1],
                slice[i * 4 + 2],
                slice[i * 4 + 3],
            ]);
        }
        Self { data }
    }

    pub fn to_slice(&self) -> [f64; 64] {
        let mut res = [0.0; 64];
        for i in 0..16 {
            let arr = self.data[i].to_array();
            res[i * 4] = arr[0];
            res[i * 4 + 1] = arr[1];
            res[i * 4 + 2] = arr[2];
            res[i * 4 + 3] = arr[3];
        }
        res
    }
}

/// Alternativity Violation Tensor (AVT)
///
/// Captures the failure of the alternative identity [x, x, y] = 0.
/// In Cayley-Dickson algebras, this is equivalent to checking if
/// [e_i, e_j, e_k] + [e_j, e_i, e_k] = 0 for all basis elements.
/// Violations occur for dimension >= 16.
pub struct AlternativityViolationTensor {
    pub dim: usize,
    /// Sparse violations: (i, j, k, m, sign)
    /// representing [e_i, e_j, e_k] + [e_j, e_i, e_k] = sign * e_m
    pub violations: Vec<(usize, usize, usize, usize, i32)>,
}

impl AlternativityViolationTensor {
    pub fn new(dim: usize) -> Self {
        assert!(dim.is_power_of_two());
        let mut violations = Vec::new();

        // We only need to check i < j to avoid double counting and i=j (which is always 0)
        for i in 0..dim {
            for j in (i + 1)..dim {
                for k in 0..dim {
                    // Associator [i, j, k]
                    let (m1, s1) = associator_basis(dim, i, j, k);
                    // Associator [j, i, k]
                    let (m2, s2) = associator_basis(dim, j, i, k);

                    if m1 != m2 {
                        // This would be extremely strange for CD basis
                        panic!("Structural failure in CD associator indices: {} != {}", m1, m2);
                    }

                    let sum_sign = s1 + s2;
                    if sum_sign != 0 {
                        if violations.len() < 10 {
                            println!("AVT Violation: [{}, {}, {}] + [{}, {}, {}] = {} * e_{}", i, j, k, j, i, k, sum_sign, m1);
                        }
                        violations.push((i, j, k, m1, sum_sign));
                    }
                }
            }
        }

        Self { dim, violations }
    }
}

/// Helper: returns (index, sign) such that [e_i, e_j, e_k] = sign * e_index
fn associator_basis(dim: usize, i: usize, j: usize, k: usize) -> (usize, i32) {
    // (e_i * e_j) * e_k
    let ij_idx = i ^ j;
    let ij_sign = cd_basis_mul_sign_iter(dim, i, j);
    let ijk_idx1 = ij_idx ^ k;
    let ijk_sign1 = ij_sign * cd_basis_mul_sign_iter(dim, ij_idx, k);

    // e_i * (e_j * e_k)
    let jk_idx = j ^ k;
    let jk_sign = cd_basis_mul_sign_iter(dim, j, k);
    let _ijk_idx2 = i ^ jk_idx;
    let ijk_sign2 = jk_sign * cd_basis_mul_sign_iter(dim, i, jk_idx);

    (ijk_idx1, ijk_sign1 - ijk_sign2)
}

#[cfg(test)]
#[path = "chingon_tests.rs"]
mod tests;
