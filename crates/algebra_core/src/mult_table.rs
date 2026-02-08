//! Cayley-Dickson multiplication table generator.
//!
//! Pre-computes the full multiplication table `e_i * e_j = sign * e_k` for
//! any power-of-2 dimension, enabling O(1) basis multiplication lookups and
//! O(dim^2) full element multiplication.
//!
//! The table is verified by a SHA-256 checksum over the (index, sign) data,
//! ensuring deterministic reproduction across platforms.
//!
//! # Literature
//!
//! - Schafer, R. D. (1966): An Introduction to Nonassociative Algebras
//! - Baez, J. C. (2002): The Octonions, Bull. AMS 39(2)
//! - de Marrais, R. (2000): Box-kite structure of sedenion zero-divisors

use sha2::{Digest, Sha256};

use crate::cayley_dickson::cd_multiply;

/// Pre-computed Cayley-Dickson multiplication table.
///
/// For dimension `dim`, the table stores `dim * dim` entries.
/// Each entry `(sign, index)` satisfies `e_i * e_j = sign * e_{index}`.
///
/// All Cayley-Dickson basis products yield exactly one nonzero component
/// with coefficient +1 or -1.  This is a structural property of the
/// doubling construction.
#[derive(Clone, Debug)]
pub struct CdMultTable {
    /// Algebra dimension (must be a power of 2).
    pub dim: usize,
    /// Flat array of result indices: `index[i * dim + j]` = k where `e_i * e_j = +/- e_k`.
    pub index: Vec<usize>,
    /// Flat array of signs: `sign[i * dim + j]` = +1 or -1.
    pub sign: Vec<i8>,
    /// SHA-256 hex digest of the (index, sign) data for verification.
    pub checksum: String,
}

impl CdMultTable {
    /// Generate the multiplication table for dimension `dim`.
    ///
    /// Calls `cd_multiply()` on each basis pair to populate the table.
    /// Panics if `dim` is not a power of 2 or is 0.
    pub fn generate(dim: usize) -> Self {
        assert!(dim > 0 && dim.is_power_of_two(), "dim must be a power of 2");

        let n = dim * dim;
        let mut index = vec![0_usize; n];
        let mut sign = vec![0_i8; n];

        for i in 0..dim {
            for j in 0..dim {
                // Build basis vectors e_i and e_j
                let mut ei = vec![0.0_f64; dim];
                let mut ej = vec![0.0_f64; dim];
                ei[i] = 1.0;
                ej[j] = 1.0;

                let product = cd_multiply(&ei, &ej);

                // Find the single nonzero component
                let mut found_k = 0;
                let mut found_sign = 0_i8;
                for (k, &val) in product.iter().enumerate() {
                    if val.abs() > 0.5 {
                        found_k = k;
                        found_sign = if val > 0.0 { 1 } else { -1 };
                        break;
                    }
                }

                let flat = i * dim + j;
                index[flat] = found_k;
                sign[flat] = found_sign;
            }
        }

        let checksum = Self::compute_checksum(&index, &sign);

        Self {
            dim,
            index,
            sign,
            checksum,
        }
    }

    /// Look up the basis product `e_i * e_j = sign * e_k` in O(1).
    ///
    /// Returns `(sign, result_index)`.
    ///
    /// # Panics
    /// Panics if `i >= dim` or `j >= dim`.
    #[inline]
    pub fn multiply_basis(&self, i: usize, j: usize) -> (i8, usize) {
        debug_assert!(i < self.dim && j < self.dim);
        let flat = i * self.dim + j;
        (self.sign[flat], self.index[flat])
    }

    /// Multiply two general elements using the pre-computed table in O(dim^2).
    ///
    /// `a` and `b` must have length `dim`.
    pub fn multiply_via_table(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        debug_assert_eq!(a.len(), self.dim);
        debug_assert_eq!(b.len(), self.dim);

        let mut result = vec![0.0_f64; self.dim];

        for (i, &ai) in a.iter().enumerate() {
            if ai == 0.0 {
                continue;
            }
            for (j, &bj) in b.iter().enumerate() {
                if bj == 0.0 {
                    continue;
                }
                let (s, k) = self.multiply_basis(i, j);
                result[k] += ai * bj * f64::from(s);
            }
        }

        result
    }

    /// Verify the table by recomputing the SHA-256 checksum.
    ///
    /// Returns `true` if the stored checksum matches.
    pub fn verify(&self) -> bool {
        let recomputed = Self::compute_checksum(&self.index, &self.sign);
        recomputed == self.checksum
    }

    /// Compute SHA-256 checksum over (index, sign) data.
    fn compute_checksum(index: &[usize], sign: &[i8]) -> String {
        let mut hasher = Sha256::new();

        // Hash indices as little-endian bytes
        for &idx in index {
            hasher.update(idx.to_le_bytes());
        }

        // Hash signs as single bytes
        for &s in sign {
            hasher.update([s as u8]);
        }

        let hash = hasher.finalize();
        // Format as hex string
        hash.iter().map(|b| format!("{:02x}", b)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::m3::OctonionTable;

    #[test]
    fn test_dim8_matches_octonion_table() {
        let table = CdMultTable::generate(8);
        let oct = OctonionTable::new();

        for i in 0..8 {
            for j in 0..8 {
                let (oct_sign, oct_idx) = oct.mul_basis(i, j);
                let (cd_sign, cd_idx) = table.multiply_basis(i, j);
                assert_eq!(
                    (cd_sign as i32, cd_idx),
                    (oct_sign, oct_idx),
                    "Mismatch at e_{} * e_{}: CD=({}, {}), Oct=({}, {})",
                    i, j, cd_sign, cd_idx, oct_sign, oct_idx
                );
            }
        }
    }

    #[test]
    fn test_dim16_all_entries_unit() {
        let table = CdMultTable::generate(16);

        for i in 0..16 {
            for j in 0..16 {
                let (s, k) = table.multiply_basis(i, j);
                assert!(
                    s == 1 || s == -1,
                    "Sign at ({}, {}) is {}, expected +/-1",
                    i, j, s
                );
                assert!(
                    k < 16,
                    "Index at ({}, {}) is {}, expected < 16",
                    i, j, k
                );
            }
        }
    }

    #[test]
    fn test_multiply_via_table_matches_cd_multiply() {
        let table = CdMultTable::generate(16);

        // Test with specific elements
        let a = vec![1.0, 2.0, -1.5, 0.5, 0.0, 3.0, -2.0, 1.0,
                     0.5, -0.5, 1.0, 0.0, 2.0, -1.0, 0.0, 0.5];
        let b = vec![0.5, -1.0, 2.0, 0.0, 1.5, -0.5, 0.0, 1.0,
                     -1.0, 0.5, 0.0, 2.0, -0.5, 1.0, -1.5, 0.0];

        let result_table = table.multiply_via_table(&a, &b);
        let result_cd = cd_multiply(&a, &b);

        for k in 0..16 {
            assert!(
                (result_table[k] - result_cd[k]).abs() < 1e-14,
                "Component {} differs: table={}, cd={}",
                k, result_table[k], result_cd[k]
            );
        }
    }

    #[test]
    fn test_checksum_deterministic() {
        let table1 = CdMultTable::generate(8);
        let table2 = CdMultTable::generate(8);
        assert_eq!(
            table1.checksum, table2.checksum,
            "Checksum should be deterministic"
        );
    }

    #[test]
    fn test_verify_succeeds() {
        let table = CdMultTable::generate(16);
        assert!(table.verify(), "Verification should succeed for freshly generated table");
    }

    #[test]
    fn test_dim32_generates_without_panic() {
        let table = CdMultTable::generate(32);
        assert_eq!(table.dim, 32);
        assert_eq!(table.index.len(), 32 * 32);
        assert_eq!(table.sign.len(), 32 * 32);
        assert!(table.verify());

        // Spot check: e_0 * e_k = e_k for all k
        for k in 0..32 {
            let (s, idx) = table.multiply_basis(0, k);
            assert_eq!((s, idx), (1, k), "e_0 * e_{} should be e_{}", k, k);
        }

        // Spot check: e_k * e_k = -e_0 for k >= 1
        for k in 1..32 {
            let (s, idx) = table.multiply_basis(k, k);
            assert_eq!((s, idx), (-1, 0), "e_{} * e_{} should be -e_0", k, k);
        }
    }
}
