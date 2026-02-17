use algebra_core::physics::clifford::pauli_matrices;
use nalgebra::{Matrix3, Matrix4, OMatrix, Vector3, U8};
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

    /// Constructs the density matrix from the Bloch parameters (a, b, T).
    /// rho = 1/4 * (I + a.sigma x I + I x b.sigma + sum T_ij sigma_i x sigma_j)
    pub fn from_ab_t(a: &Vector3<f64>, b: &Vector3<f64>, t: &Matrix3<f64>) -> Self {
        let (d1, d2, d3) = pauli_matrices();

        let to_m2 =
            |m: algebra_core::physics::clifford::GammaMatrix| -> nalgebra::Matrix2<Complex64> {
                nalgebra::Matrix2::from_iterator(m.into_iter().cloned())
            };

        let sigmas = [to_m2(d1), to_m2(d2), to_m2(d3)];

        let eye2 = nalgebra::Matrix2::<Complex64>::identity();

        let mut rho = Matrix4::<Complex64>::zeros();

        // Identity term
        rho += Matrix4::identity();

        // Single spin a (system 1) -> a . sigma x I
        for i in 0..3 {
            if a[i] != 0.0 {
                let term = kron(&sigmas[i], &eye2) * Complex64::from(a[i]);
                rho += term;
            }
        }

        // Single spin b (system 2) -> I x b . sigma
        for j in 0..3 {
            if b[j] != 0.0 {
                let term = kron(&eye2, &sigmas[j]) * Complex64::from(b[j]);
                rho += term;
            }
        }

        // Correlation term T_ij sigma_i x sigma_j
        for i in 0..3 {
            for j in 0..3 {
                if t[(i, j)] != 0.0 {
                    let term = kron(&sigmas[i], &sigmas[j]) * Complex64::from(t[(i, j)]);
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

// Helper for Kronecker product of 2x2 matrices
fn kron(a: &nalgebra::Matrix2<Complex64>, b: &nalgebra::Matrix2<Complex64>) -> Matrix4<Complex64> {
    let mut m = Matrix4::zeros();
    for i in 0..2 {
        for j in 0..2 {
            let val_a = a[(i, j)];
            for k in 0..2 {
                for l in 0..2 {
                    let val_b = b[(k, l)];
                    m[(2 * i + k, 2 * j + l)] = val_a * val_b;
                }
            }
        }
    }
    m
}
