use cd_kernel::cayley_dickson::cd_basis_mul_sign_iter;
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
            // Limit violations for very high dimensions to prevent memory explosion
            if violations.len() > 1_000_000 {
                break;
            }
        }

        Self { dim, violations }
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
