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

    /// Bit-pack violations for GPU upload, dropping unused `k` index.
    ///
    /// The AVT contraction loop uses only `(i, j, m, sign)` -- the `k` index
    /// is the third element of the associator triple used during CONSTRUCTION
    /// but is never referenced during the bilinear tensor contraction:
    ///   `force_64d[m] += alpha * v[i] * v[j] * sign`
    ///
    /// # Packing layout (uint32)
    ///
    /// The bit width per index is `ceil(log2(dim))`:
    ///   - 64D:   6 bits/index -> i(6) + j(6) + m(6) + sign(1) = 19 bits
    ///   - 256D:  8 bits/index -> i(8) + j(8) + m(8) + sign(1) = 25 bits
    ///   - 512D:  9 bits/index -> i(9) + j(9) + m(9) + sign(1) = 28 bits
    ///   - 1024D: 10 bits/index -> i(10) + j(10) + m(10) + sign(1) = 31 bits
    ///
    /// Bit layout (LSB to MSB):
    ///   `[m: bits] [j: bits] [i: bits] [sign_positive: 1]`
    ///
    /// Sign encoding: bit=1 means positive sign, bit=0 means negative.
    /// The magnitude is always 2 (since sum_sign = s1+s2 where each is +/-1,
    /// and only non-zero sums are stored). The GPU reconstructs:
    ///   `float sign_val = (packed >> sign_shift) & 1 ? 2.0f : -2.0f;`
    ///
    /// # Returns
    ///
    /// `(packed_data, index_bits)` where `index_bits` is needed by the GPU
    /// to unpack the fields.
    pub fn pack_for_gpu(&self) -> PackedAvt {
        let bits = index_bits_for_dim(self.dim);
        assert!(3 * bits < 32, "dim {} requires {}*3+1={} bits, exceeds u32",
                self.dim, bits, 3 * bits + 1);

        let mask = (1u32 << bits) - 1;
        let mut packed = Vec::with_capacity(self.violations.len());

        for &(i, j, _k, m, sign) in &self.violations {
            let sign_bit: u32 = if sign > 0 { 1 } else { 0 };
            let word = (m as u32 & mask)
                | ((j as u32 & mask) << bits)
                | ((i as u32 & mask) << (2 * bits))
                | (sign_bit << (3 * bits));
            packed.push(word);
        }

        PackedAvt {
            data: packed,
            index_bits: bits as u32,
            dim: self.dim as u32,
            violation_count: self.violations.len() as u32,
        }
    }
}

/// Bit-packed AVT violations for GPU upload.
///
/// Each `u32` in `data` encodes one `(i, j, m, sign)` tuple.
/// The `k` index is omitted because the tensor contraction never uses it.
#[derive(Debug, Clone)]
pub struct PackedAvt {
    /// Bit-packed violation words.
    pub data: Vec<u32>,
    /// Number of bits per index field (6 for 64D, 8 for 256D, 10 for 1024D).
    pub index_bits: u32,
    /// Algebra dimension.
    pub dim: u32,
    /// Number of violations (== data.len()).
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

/// Compute the number of bits needed to represent indices for a given dimension.
pub fn index_bits_for_dim(dim: usize) -> usize {
    // dim is always a power of 2; we need ceil(log2(dim)) bits.
    // For dim=64: log2(64)=6, for dim=256: 8, for dim=1024: 10.
    assert!(dim >= 2 && dim.is_power_of_two());
    dim.trailing_zeros() as usize
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
