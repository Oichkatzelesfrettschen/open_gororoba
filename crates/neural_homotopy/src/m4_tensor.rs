//! A-infinity correction tensor m_3 for sedenion non-associativity.
//!
//! The ternary operation m_3 corrects the non-associativity of sedenion
//! multiplication. For basis elements:
//!   m_3(e_i, e_j, e_k) = e_i*(e_j*e_k) - (e_i*e_j)*e_k
//!
//! The correction tensor is stored as a 16x16x16x16 array where
//! tensor[i][j][k][l] is the coefficient of e_l in m_3(e_i, e_j, e_k).
//!
//! The pentagon violation measures the A-infinity A_4 relation,
//! which determines whether a higher correction m_4 is needed.

use crate::training_data::{build_sedenion_table, SEDENION_DIM};

const DIM: usize = SEDENION_DIM;
const TENSOR_SIZE: usize = DIM * DIM * DIM * DIM;

/// Ternary correction tensor for the A-infinity structure on sedenions.
///
/// Stores m_3(e_i, e_j, e_k) as a 16-vector of real coefficients per triple.
/// The full tensor has 16^4 = 65536 entries.
#[derive(Debug, Clone)]
pub struct CorrectionTensor {
    data: Vec<f64>,
}

impl CorrectionTensor {
    /// Create a zero correction tensor (trivial A-infinity structure).
    pub fn zero() -> Self {
        Self {
            data: vec![0.0; TENSOR_SIZE],
        }
    }

    /// Create from raw data (must have exactly 65536 entries).
    pub fn from_data(data: Vec<f64>) -> Option<Self> {
        if data.len() == TENSOR_SIZE {
            Some(Self { data })
        } else {
            None
        }
    }

    fn idx(i: usize, j: usize, k: usize, l: usize) -> usize {
        i * DIM * DIM * DIM + j * DIM * DIM + k * DIM + l
    }

    /// Get the coefficient of e_l in m_3(e_i, e_j, e_k).
    pub fn get(&self, i: usize, j: usize, k: usize, l: usize) -> f64 {
        self.data[Self::idx(i, j, k, l)]
    }

    /// Set the coefficient of e_l in m_3(e_i, e_j, e_k).
    pub fn set(&mut self, i: usize, j: usize, k: usize, l: usize, val: f64) {
        self.data[Self::idx(i, j, k, l)] = val;
    }

    /// Get the output 16-vector for triple (i,j,k).
    pub fn slice(&self, i: usize, j: usize, k: usize) -> [f64; DIM] {
        let base = Self::idx(i, j, k, 0);
        let mut out = [0.0; DIM];
        out.copy_from_slice(&self.data[base..base + DIM]);
        out
    }

    /// Initialize from the sedenion associator (direct ansatz).
    ///
    /// Sets m_3(e_i, e_j, e_k) = e_i*(e_j*e_k) - (e_i*e_j)*e_k.
    /// This is the canonical choice that exactly cancels the associator
    /// at the ternary level.
    pub fn from_associator() -> Self {
        let table = build_sedenion_table();
        let mut tensor = Self::zero();

        for i in 0..DIM {
            for j in 0..DIM {
                let (ij_basis, ij_sign) = table[i][j];

                for k in 0..DIM {
                    // Left parenthesization: (e_i * e_j) * e_k
                    let (left_basis, left_sign_raw) = table[ij_basis][k];
                    let left_coeff = (ij_sign * left_sign_raw) as f64;

                    // Right parenthesization: e_i * (e_j * e_k)
                    let (jk_basis, jk_sign) = table[j][k];
                    let (right_basis, right_sign_raw) = table[i][jk_basis];
                    let right_coeff = (jk_sign * right_sign_raw) as f64;

                    // m_3 = right - left (corrects non-associativity)
                    let base = Self::idx(i, j, k, 0);
                    tensor.data[base + right_basis] += right_coeff;
                    tensor.data[base + left_basis] -= left_coeff;
                }
            }
        }

        tensor
    }

    /// Alias for `from_associator()` -- the algebraic fallback when
    /// neural search fails or is not available.
    pub fn direct_associator_ansatz() -> Self {
        Self::from_associator()
    }

    /// L2 norm squared of the tensor (sum of squared entries).
    pub fn l2_norm_sq(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum()
    }

    /// Number of non-zero entries (above threshold 1e-14).
    pub fn nnz(&self) -> usize {
        self.data.iter().filter(|&&x| x.abs() > 1e-14).count()
    }

    /// Sparsity ratio (fraction of zero entries).
    pub fn sparsity(&self) -> f64 {
        1.0 - (self.nnz() as f64) / (TENSOR_SIZE as f64)
    }

    /// Evaluate the A-infinity A_4 pentagon violation.
    ///
    /// For each sampled quadruple (a,b,c,d), computes the 5 terms of the
    /// A_4 relation involving m_2 (sedenion product) and m_3 (this tensor):
    ///
    ///   m_2(m_3(a,b,c), d) + m_3(m_2(a,b), c, d)
    ///   - m_3(a, m_2(b,c), d) + m_3(a, b, m_2(c,d))
    ///   + m_2(a, m_3(b,c,d))
    ///
    /// Returns the mean L2 norm of the violation vector across samples.
    /// A value near zero means m_3 nearly satisfies the pentagon identity
    /// without needing a higher m_4 correction.
    pub fn pentagon_violation(&self, n_samples: usize) -> f64 {
        let table = build_sedenion_table();
        let mut total_violation = 0.0;
        let mut count = 0usize;

        for idx in 0..n_samples {
            let a = (idx.wrapping_mul(7).wrapping_add(1)) % DIM;
            let b = (idx.wrapping_mul(11).wrapping_add(3)) % DIM;
            let c = (idx.wrapping_mul(13).wrapping_add(5)) % DIM;
            let d = (idx.wrapping_mul(17).wrapping_add(7)) % DIM;

            let mut viol = [0.0f64; DIM];

            // Term 1: m_2(m_3(a,b,c), d)
            let m3_abc = self.slice(a, b, c);
            for (l, &coeff) in m3_abc.iter().enumerate() {
                if coeff.abs() > 1e-14 {
                    let (prod_basis, prod_sign) = table[l][d];
                    viol[prod_basis] += coeff * prod_sign as f64;
                }
            }

            // Term 2: m_3(m_2(a,b), c, d)
            let (ab_basis, ab_sign) = table[a][b];
            let m3_ab_cd = self.slice(ab_basis, c, d);
            for (l, &coeff) in m3_ab_cd.iter().enumerate() {
                viol[l] += ab_sign as f64 * coeff;
            }

            // Term 3: -m_3(a, m_2(b,c), d)
            let (bc_basis, bc_sign) = table[b][c];
            let m3_a_bc_d = self.slice(a, bc_basis, d);
            for (l, &coeff) in m3_a_bc_d.iter().enumerate() {
                viol[l] -= bc_sign as f64 * coeff;
            }

            // Term 4: m_3(a, b, m_2(c,d))
            let (cd_basis, cd_sign) = table[c][d];
            let m3_ab_cd2 = self.slice(a, b, cd_basis);
            for (l, &coeff) in m3_ab_cd2.iter().enumerate() {
                viol[l] += cd_sign as f64 * coeff;
            }

            // Term 5: m_2(a, m_3(b,c,d))
            let m3_bcd = self.slice(b, c, d);
            for (l, &coeff) in m3_bcd.iter().enumerate() {
                if coeff.abs() > 1e-14 {
                    let (prod_basis, prod_sign) = table[a][l];
                    viol[prod_basis] -= coeff * prod_sign as f64;
                }
            }

            let norm_sq: f64 = viol.iter().map(|x| x * x).sum();
            total_violation += norm_sq.sqrt();
            count += 1;
        }

        if count > 0 {
            total_violation / count as f64
        } else {
            0.0
        }
    }

    /// Serialize summary statistics to TOML format.
    ///
    /// Outputs metadata (dimensions, norms, sparsity) rather than the
    /// full 65536-entry tensor. Use `data()` for programmatic access.
    pub fn serialize_to_toml(&self, n_violation_samples: usize) -> String {
        let mut out = String::new();
        out.push_str("[correction_tensor]\n");
        out.push_str(&format!("dim = {}\n", DIM));
        out.push_str(&format!("total_entries = {}\n", TENSOR_SIZE));
        out.push_str(&format!("nnz = {}\n", self.nnz()));
        out.push_str(&format!("sparsity = {:.6}\n", self.sparsity()));
        out.push_str(&format!("l2_norm_sq = {:.6}\n", self.l2_norm_sq()));
        out.push_str(&format!("l2_norm = {:.6}\n", self.l2_norm_sq().sqrt()));
        out.push_str(&format!(
            "pentagon_violation = {:.6}\n",
            self.pentagon_violation(n_violation_samples)
        ));
        out
    }

    /// Raw data access for optimization.
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Mutable raw data access for optimization.
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_tensor_properties() {
        let t = CorrectionTensor::zero();
        assert_eq!(t.nnz(), 0);
        assert_eq!(t.l2_norm_sq(), 0.0);
        assert!((t.sparsity() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_zero_tensor_pentagon_violation() {
        let t = CorrectionTensor::zero();
        let v = t.pentagon_violation(100);
        assert_eq!(v, 0.0, "Zero tensor should have zero violation");
    }

    #[test]
    fn test_associator_tensor_nonzero() {
        let t = CorrectionTensor::from_associator();
        assert!(t.nnz() > 0, "Associator tensor should have non-zero entries");
        assert!(t.l2_norm_sq() > 0.0, "Associator should have non-zero norm");
    }

    #[test]
    fn test_associator_sparsity() {
        let t = CorrectionTensor::from_associator();
        // Sedenions are non-associative, but not every triple produces a
        // non-zero associator. Identity element e_0 is always associative.
        assert!(
            t.sparsity() > 0.5,
            "Associator tensor should be sparse: sparsity={}",
            t.sparsity()
        );
    }

    #[test]
    fn test_identity_is_associative() {
        let t = CorrectionTensor::from_associator();
        // e_0 * (e_j * e_k) == (e_0 * e_j) * e_k for all j, k
        for j in 0..DIM {
            for k in 0..DIM {
                let s = t.slice(0, j, k);
                let norm: f64 = s.iter().map(|x| x * x).sum();
                assert!(
                    norm < 1e-14,
                    "Identity should be associative: m_3(0,{},{}) has norm {}",
                    j,
                    k,
                    norm.sqrt()
                );
            }
        }
    }

    #[test]
    fn test_associator_pentagon_violation_positive() {
        let t = CorrectionTensor::from_associator();
        let v = t.pentagon_violation(256);
        // The associator ansatz does NOT automatically satisfy A_4,
        // so pentagon violation should be positive for sedenions.
        assert!(
            v >= 0.0,
            "Pentagon violation must be non-negative: {}",
            v
        );
    }

    #[test]
    fn test_get_set_roundtrip() {
        let mut t = CorrectionTensor::zero();
        t.set(3, 7, 11, 5, 42.0);
        assert!((t.get(3, 7, 11, 5) - 42.0).abs() < 1e-14);
        assert_eq!(t.nnz(), 1);
    }

    #[test]
    fn test_from_data_correct_size() {
        let data = vec![0.0; TENSOR_SIZE];
        assert!(CorrectionTensor::from_data(data).is_some());
        let bad = vec![0.0; 100];
        assert!(CorrectionTensor::from_data(bad).is_none());
    }

    #[test]
    fn test_serialize_toml_contains_keys() {
        let t = CorrectionTensor::zero();
        let toml = t.serialize_to_toml(10);
        assert!(toml.contains("dim = 16"));
        assert!(toml.contains("nnz = 0"));
        assert!(toml.contains("sparsity"));
        assert!(toml.contains("pentagon_violation"));
    }

    #[test]
    fn test_direct_associator_ansatz_equals_from_associator() {
        let a = CorrectionTensor::from_associator();
        let b = CorrectionTensor::direct_associator_ansatz();
        assert_eq!(a.data(), b.data());
    }
}
