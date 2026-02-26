use algebra_core::physics::clifford::pauli_matrices;
use nalgebra::Vector3;
use num_complex::Complex64;

/// A right-handed orthonormal basis in R^3, defined by three CD basis indices.
/// Default is (1, 2, 3), corresponding to e1, e2, e3.
#[derive(Debug, Clone, Copy)]
pub struct AlgebraicTriad {
    pub indices: [usize; 3],
}

impl Default for AlgebraicTriad {
    fn default() -> Self {
        Self { indices: [1, 2, 3] }
    }
}

impl AlgebraicTriad {
    pub fn new(i: usize, j: usize, k: usize) -> Self {
        Self { indices: [i, j, k] }
    }

    /// Returns the Pauli matrices corresponding to the triad axes.
    /// This assumes the indices map to the standard Pauli X, Y, Z.
    /// In a more general CD context, this would involve the sign table.
    /// For now, we assume the canonical (1, 2, 3) maps to (sigma_1, sigma_2, sigma_3).
    pub fn to_pauli_axes(&self) -> [nalgebra::Matrix2<Complex64>; 3] {
        let (s1, s2, s3) = pauli_matrices();

        let to_m2 =
            |m: algebra_core::physics::clifford::GammaMatrix| -> nalgebra::Matrix2<Complex64> {
                nalgebra::Matrix2::from_iterator(m.into_iter().cloned())
            };

        let s1 = to_m2(s1);
        let s2 = to_m2(s2);
        let s3 = to_m2(s3);

        let get_sigma = |idx: usize| match idx {
            1 => s1,
            2 => s2,
            3 => s3,
            _ => nalgebra::Matrix2::zeros(),
        };

        [
            get_sigma(self.indices[0]),
            get_sigma(self.indices[1]),
            get_sigma(self.indices[2]),
        ]
    }

    /// Returns the axes as standard unit vectors in R^3.
    pub fn to_axes_vec(&self) -> [Vector3<f64>; 3] {
        [Vector3::x(), Vector3::y(), Vector3::z()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_indices() {
        let triad = AlgebraicTriad::default();
        assert_eq!(triad.indices, [1, 2, 3]);
    }

    #[test]
    fn test_custom_indices() {
        let triad = AlgebraicTriad::new(3, 5, 7);
        assert_eq!(triad.indices, [3, 5, 7]);
    }

    #[test]
    fn test_to_axes_vec_orthonormal() {
        let triad = AlgebraicTriad::default();
        let axes = triad.to_axes_vec();
        // Verify orthonormality
        for ax in &axes {
            assert!((ax.norm() - 1.0).abs() < 1e-14);
        }
        assert!(axes[0].dot(&axes[1]).abs() < 1e-14);
        assert!(axes[0].dot(&axes[2]).abs() < 1e-14);
        assert!(axes[1].dot(&axes[2]).abs() < 1e-14);
    }

    #[test]
    fn test_pauli_axes_hermitian() {
        let triad = AlgebraicTriad::default();
        let paulis = triad.to_pauli_axes();
        // Each Pauli matrix should be Hermitian: M = M^dagger
        for (idx, p) in paulis.iter().enumerate() {
            for i in 0..2 {
                for j in 0..2 {
                    let diff = (p[(i, j)] - p[(j, i)].conj()).norm();
                    assert!(diff < 1e-14,
                        "Pauli {idx} not Hermitian at [{i},{j}]");
                }
            }
        }
    }
}
