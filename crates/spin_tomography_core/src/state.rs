use gororoba_algebra::{kron2, physics::clifford::pauli_matrices};
use nalgebra::{Matrix3, Matrix4, OMatrix, U8, Vector3};
use num_complex::Complex64;

type Matrix8<T> = OMatrix<T, U8, U8>;

/// Represents a two-qubit density matrix state.
#[derive(Debug, Clone)]
pub struct TwoQubitState {
    pub rho: Matrix4<Complex64>,
}

impl TwoQubitState {
    pub fn new(rho: Matrix4<Complex64>) -> Self {
        Self { rho }
    }

    /// Constructs the density matrix from plain [f64;3] Bloch vectors and a [3x3] correlation matrix.
    /// Convenience wrapper for callers that do not import nalgebra.
    pub fn from_ab_t_arr(a: [f64; 3], b: [f64; 3], t: [[f64; 3]; 3]) -> Self {
        Self::from_ab_t(
            &Vector3::from(a),
            &Vector3::from(b),
            &Matrix3::new(
                t[0][0], t[0][1], t[0][2],
                t[1][0], t[1][1], t[1][2],
                t[2][0], t[2][1], t[2][2],
            ),
        )
    }

    /// Constructs the density matrix from the Bloch parameters (a, b, T).
    /// rho = 1/4 * (I + a.sigma x I + I x b.sigma + sum T_ij sigma_i x sigma_j)
    pub fn from_ab_t(a: &Vector3<f64>, b: &Vector3<f64>, t: &Matrix3<f64>) -> Self {
        let (d1, d2, d3) = pauli_matrices();
        let sigmas = [d1, d2, d3];

        let eye2 = nalgebra::Matrix2::<Complex64>::identity();

        let mut rho = Matrix4::<Complex64>::zeros();

        // Identity term
        rho += Matrix4::identity();

        // Single spin a (system 1) -> a . sigma x I
        for i in 0..3 {
            if a[i] != 0.0 {
                let term = kron2(&sigmas[i], &eye2) * Complex64::from(a[i]);
                rho += term;
            }
        }

        // Single spin b (system 2) -> I x b . sigma
        for j in 0..3 {
            if b[j] != 0.0 {
                let term = kron2(&eye2, &sigmas[j]) * Complex64::from(b[j]);
                rho += term;
            }
        }

        // Correlated part -> sum t_ij sigma_i x sigma_j
        for i in 0..3 {
            for j in 0..3 {
                if t[(i, j)] != 0.0 {
                    let term = kron2(&sigmas[i], &sigmas[j]) * Complex64::from(t[(i, j)]);
                    rho += term;
                }
            }
        }

        // Normalize
        Self {
            rho: rho * Complex64::from(0.25),
        }
    }

    /// Computes the partial transpose with respect to system B (second qubit).
    pub fn partial_transpose(&self) -> Self {
        let mut pt = Matrix4::<Complex64>::zeros();

        for row in 0..4 {
            for col in 0..4 {
                let i1 = row / 2;
                let j1 = row % 2;
                let i2 = col / 2;
                let j2 = col % 2;

                // Transpose indices on subsystem B: swap j1 and j2
                let val = self.rho[(row, col)];

                let dest_row = 2 * i1 + j2;
                let dest_col = 2 * i2 + j1;

                pt[(dest_row, dest_col)] = val;
            }
        }

        Self { rho: pt }
    }

    /// Computes the negativity of the state.
    /// N(rho) = sum of absolute values of negative eigenvalues of rho^TB
    pub fn negativity(&self) -> f64 {
        let pt_rho = self.partial_transpose().rho;

        // Map 4x4 Complex to 8x8 Real to use SymmetricEigen
        // M = [[A, -B], [B, A]] where pt_rho = A + iB
        // pt_rho is Hermitian => M is Symmetric.

        let mut m = Matrix8::<f64>::zeros();

        for r in 0..4 {
            for c in 0..4 {
                let val = pt_rho[(r, c)];
                m[(r, c)] = val.re;
                m[(r + 4, c + 4)] = val.re;
                m[(r + 4, c)] = val.im;
                m[(r, c + 4)] = -val.im;
            }
        }

        let eigen = m.symmetric_eigen();

        let mut sum_neg = 0.0;
        // Each eigenvalue appears twice
        for &val in eigen.eigenvalues.iter() {
            if val < -1e-12 {
                sum_neg += val.abs();
            }
        }
        sum_neg / 2.0
    }

    pub fn is_entangled(&self) -> bool {
        self.negativity() > 1e-10
    }

    /// Computes the Jordan product of two states: (A*B + B*A) / 2.
    /// Result is Hermitian if both inputs are Hermitian.
    pub fn jordan_product(&self, other: &Self) -> Self {
        let res = (self.rho * other.rho + other.rho * self.rho) * Complex64::from(0.5);
        Self { rho: res }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maximally_mixed_state() {
        // I/4: a=0, b=0, T=0
        let a = Vector3::zeros();
        let b = Vector3::zeros();
        let t = Matrix3::<f64>::zeros();
        let state = TwoQubitState::from_ab_t(&a, &b, &t);
        let tr = state.rho.trace().re;
        assert!((tr - 1.0).abs() < 1e-14, "Trace should be 1.0, got {tr}");
        // Diagonal should be 0.25 each
        for i in 0..4 {
            assert!((state.rho[(i, i)].re - 0.25).abs() < 1e-14);
        }
    }

    #[test]
    fn test_product_state_not_entangled() {
        // Pure product state |00><00| needs a=(0,0,1), b=(0,0,1), T=diag(0,0,1)
        // because rho = 1/4(I + a.sigma x I + I x b.sigma + T_{ij} sigma_i x sigma_j)
        // For |00><00|: T_{33} = 1 (sigma_z x sigma_z correlation)
        let a = Vector3::new(0.0, 0.0, 1.0);
        let b = Vector3::new(0.0, 0.0, 1.0);
        let mut t = Matrix3::<f64>::zeros();
        t[(2, 2)] = 1.0;
        let state = TwoQubitState::from_ab_t(&a, &b, &t);
        assert!(
            !state.is_entangled(),
            "Product state |00><00| should not be entangled, negativity={}",
            state.negativity()
        );
    }

    #[test]
    fn test_singlet_state_is_entangled() {
        // Anti-correlated singlet: T = diag(-1, -1, -1)
        let a = Vector3::zeros();
        let b = Vector3::zeros();
        let t = -Matrix3::<f64>::identity();
        let state = TwoQubitState::from_ab_t(&a, &b, &t);
        assert!(
            state.is_entangled(),
            "Singlet state should be entangled, negativity={}",
            state.negativity()
        );
    }

    #[test]
    fn test_partial_transpose_double_is_identity() {
        let a = Vector3::new(0.1, 0.2, 0.3);
        let b = Vector3::new(-0.1, 0.0, 0.2);
        let t = Matrix3::<f64>::zeros();
        let state = TwoQubitState::from_ab_t(&a, &b, &t);
        let double_pt = state.partial_transpose().partial_transpose();
        for i in 0..4 {
            for j in 0..4 {
                let diff = (state.rho[(i, j)] - double_pt.rho[(i, j)]).norm();
                assert!(
                    diff < 1e-14,
                    "Double partial transpose should be identity: [{i},{j}] diff={diff}"
                );
            }
        }
    }

    #[test]
    fn test_negativity_nonnegative() {
        let a = Vector3::new(0.5, 0.0, 0.0);
        let b = Vector3::new(0.0, 0.5, 0.0);
        let t = Matrix3::<f64>::identity() * 0.3;
        let state = TwoQubitState::from_ab_t(&a, &b, &t);
        assert!(state.negativity() >= 0.0, "Negativity must be non-negative");
    }

    #[test]
    fn test_jordan_product_self_gives_rho_squared() {
        let a = Vector3::zeros();
        let b = Vector3::zeros();
        let t = Matrix3::<f64>::zeros();
        let state = TwoQubitState::from_ab_t(&a, &b, &t);
        let jp = state.jordan_product(&state);
        // For rho = I/4: jordan_product(rho, rho) = rho^2 = I/16
        let expected_diag = 0.0625;
        for i in 0..4 {
            assert!((jp.rho[(i, i)].re - expected_diag).abs() < 1e-14);
        }
    }
}
