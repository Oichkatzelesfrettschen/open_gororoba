//! Albert algebra J_3(O): exceptional Jordan algebra of 3x3 Hermitian
//! matrices over the octonions.
//!
//! Dimension: 27 = 3*1 (diagonal reals) + 3*8 (off-diagonal octonions)
//!
//! An element X in J_3(O) is:
//!   X = [ xi_1    x_3*    x_2  ]
//!       [ x_3     xi_2    x_1* ]
//!       [ x_2*    x_1     xi_3 ]
//!
//! where xi_1, xi_2, xi_3 in R and x_1, x_2, x_3 in O (octonions).
//! The * denotes octonion conjugation.
//!
//! # Jordan product
//! X . Y = (XY + YX) / 2 (symmetric product of Hermitian matrices)
//!
//! # Characteristic equation (Freudenthal)
//! lambda^3 - Tr(X) lambda^2 + S_2(X) lambda - det(X) = 0
//!
//! where:
//! - Tr(X) = xi_1 + xi_2 + xi_3
//! - S_2(X) = xi_1*xi_2 + xi_2*xi_3 + xi_3*xi_1 - |x_1|^2 - |x_2|^2 - |x_3|^2
//! - det(X) = xi_1*xi_2*xi_3 + 2*Re(x_1*x_2*x_3)
//!   - xi_1*|x_1|^2 - xi_2*|x_2|^2 - xi_3*|x_3|^2
//!
//! # Singh's delta^2 = 3/8 result
//! For an element with eigenvalues (q-delta, q, q+delta),
//! the trace-free part has Tr=0 and the eigenvalue spectrum is universal:
//! delta^2 = 3/8 (independent of the specific octonion entries).
//!
//! # Literature
//! - Singh (arXiv:2508.10131): J_3(O_C) eigenvalue spectrum
//! - Schafer (1966): Introduction to Non-Associative Algebras
//! - Baez (2001): The Octonions (math/0105155)

use crate::physics::octonion_field::{oct_conjugate, oct_multiply, oct_norm_sq};

/// A 3x3 Hermitian matrix over the octonions.
///
/// Layout: 3 real diagonal entries + 3 octonion off-diagonal entries = 27D.
///
/// The matrix is:
///   [ d[0]     o[2]*   o[1]  ]
///   [ o[2]     d[1]    o[0]* ]
///   [ o[1]*    o[0]    d[2]  ]
///
/// where * denotes octonion conjugation.
#[derive(Debug, Clone)]
pub struct AlbertElement {
    /// Diagonal entries: xi_1, xi_2, xi_3 (real)
    pub diag: [f64; 3],
    /// Off-diagonal octonion entries: x_1, x_2, x_3
    /// x_1 = (2,3) entry, x_2 = (1,3) entry, x_3 = (1,2) entry
    pub off: [[f64; 8]; 3],
}

impl AlbertElement {
    /// Create an element from diagonal entries only (off-diagonal zero).
    pub fn diagonal(d0: f64, d1: f64, d2: f64) -> Self {
        Self {
            diag: [d0, d1, d2],
            off: [[0.0; 8]; 3],
        }
    }

    /// Create the zero element.
    pub fn zero() -> Self {
        Self {
            diag: [0.0; 3],
            off: [[0.0; 8]; 3],
        }
    }

    /// Create an element with all entries specified.
    pub fn new(diag: [f64; 3], off: [[f64; 8]; 3]) -> Self {
        Self { diag, off }
    }

    /// Trace: Tr(X) = xi_1 + xi_2 + xi_3.
    pub fn trace(&self) -> f64 {
        self.diag[0] + self.diag[1] + self.diag[2]
    }

    /// Second symmetric function S_2(X).
    ///
    /// S_2(X) = xi_1*xi_2 + xi_2*xi_3 + xi_3*xi_1
    ///          - |x_1|^2 - |x_2|^2 - |x_3|^2
    pub fn s2(&self) -> f64 {
        let d = &self.diag;
        let cross_diag = d[0] * d[1] + d[1] * d[2] + d[2] * d[0];
        let norm_sq_sum =
            oct_norm_sq(&self.off[0]) + oct_norm_sq(&self.off[1]) + oct_norm_sq(&self.off[2]);
        cross_diag - norm_sq_sum
    }

    /// Determinant: det(X) (Freudenthal formula).
    ///
    /// det(X) = xi_1*xi_2*xi_3 + 2*Re(x_1 * x_2 * x_3)
    ///          - xi_1*|x_1|^2 - xi_2*|x_2|^2 - xi_3*|x_3|^2
    ///
    /// Note: the triple product x_1*x_2*x_3 is NOT associative,
    /// so we take the real part of (x_1*x_2)*x_3.
    pub fn det(&self) -> f64 {
        let d = &self.diag;
        let triple_diag = d[0] * d[1] * d[2];

        // Triple product: (x_1 * x_2) * x_3
        let x1x2 = oct_multiply(&self.off[0], &self.off[1]);
        let x1x2x3 = oct_multiply(&x1x2, &self.off[2]);
        let re_triple = 2.0 * x1x2x3[0]; // Re(q) = q[0]

        let diag_norm_terms = d[0] * oct_norm_sq(&self.off[0])
            + d[1] * oct_norm_sq(&self.off[1])
            + d[2] * oct_norm_sq(&self.off[2]);

        triple_diag + re_triple - diag_norm_terms
    }

    /// Characteristic polynomial coefficients [a, b, c] where
    /// lambda^3 - a*lambda^2 + b*lambda - c = 0.
    pub fn characteristic_coefficients(&self) -> (f64, f64, f64) {
        (self.trace(), self.s2(), self.det())
    }

    /// Eigenvalues via Cardano's formula for the depressed cubic.
    ///
    /// The characteristic equation is:
    ///   lambda^3 - T*lambda^2 + S*lambda - D = 0
    ///
    /// Substituting lambda = t + T/3 gives the depressed cubic:
    ///   t^3 + p*t + q = 0
    /// where p = S - T^2/3, q = D - T*S/3 + 2*T^3/27.
    ///
    /// Returns three real eigenvalues (sorted ascending).
    pub fn eigenvalues(&self) -> [f64; 3] {
        let tr = self.trace();
        let s2 = self.s2();
        let det = self.det();

        // Depressed cubic: t^3 + p*t + q = 0
        let p = s2 - tr * tr / 3.0;
        let q = det - tr * s2 / 3.0 + 2.0 * tr * tr * tr / 27.0;

        // Discriminant: Delta = -4p^3 - 27q^2
        // For 3 real roots: Delta >= 0, which means 4p^3 + 27q^2 <= 0
        let discriminant = -4.0 * p * p * p - 27.0 * q * q;

        if discriminant >= -1e-12 {
            // Three real roots via trigonometric method
            if p.abs() < 1e-15 {
                // p ~ 0: all roots are -q^(1/3)
                let t = if q.abs() < 1e-15 {
                    0.0
                } else {
                    -q.signum() * q.abs().cbrt()
                };
                let l = t + tr / 3.0;
                [l, l, l]
            } else {
                let r = (-p / 3.0).sqrt();
                // cos(3*theta) = q / (2 * r^3)  [note: p < 0 for real roots]
                let cos_arg = (q / (2.0 * r * r * r)).clamp(-1.0, 1.0);
                let theta = cos_arg.acos() / 3.0;

                let t0 = 2.0 * r * theta.cos();
                let t1 = 2.0 * r * (theta + 2.0 * std::f64::consts::FRAC_PI_3 * 2.0).cos();
                let t2 = 2.0 * r * (theta + 4.0 * std::f64::consts::FRAC_PI_3 * 2.0).cos();

                let mut vals = [t0 + tr / 3.0, t1 + tr / 3.0, t2 + tr / 3.0];
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                vals
            }
        } else {
            // Complex roots (shouldn't happen for Hermitian matrices)
            [f64::NAN, f64::NAN, f64::NAN]
        }
    }

    /// Compute delta^2 for the eigenvalue spectrum.
    ///
    /// If eigenvalues are (q-delta, q, q+delta), then:
    ///   q = middle eigenvalue
    ///   delta = (lambda_3 - lambda_1) / 2
    ///   delta^2 = ((lambda_3 - lambda_1) / 2)^2
    ///
    /// Singh predicts delta^2 = 3/8 for the universal spectrum.
    pub fn delta_squared(&self) -> f64 {
        let eigs = self.eigenvalues();
        let delta = (eigs[2] - eigs[0]) / 2.0;
        delta * delta
    }

    /// Check if eigenvalues form an arithmetic progression (q-d, q, q+d).
    ///
    /// Returns (is_arithmetic, q, delta, residual).
    pub fn eigenvalue_arithmetic_check(&self) -> (bool, f64, f64, f64) {
        let eigs = self.eigenvalues();
        let q = eigs[1];
        let delta = (eigs[2] - eigs[0]) / 2.0;

        // Check: middle eigenvalue should be average of outer two
        let expected_middle = (eigs[0] + eigs[2]) / 2.0;
        let residual = (q - expected_middle).abs();

        (residual < 1e-10, q, delta, residual)
    }

    /// Jordan product: X . Y = (XY + YX) / 2.
    ///
    /// For 3x3 Hermitian octonion matrices, this is the standard
    /// symmetrized matrix product.
    pub fn jordan_product(&self, other: &AlbertElement) -> AlbertElement {
        // Compute XY and YX as 3x3 octonion matrices, then average.
        // Since both X and Y are Hermitian, X.Y is also Hermitian.
        //
        // For the diagonal entries:
        // (XY)_{ii} = sum_j X_{ij} * Y_{ji} = sum_j X_{ij} * conj(Y_{ij})
        //           = xi_i * yi_i + sum_{j != i} X_{ij} * conj(Y_{ij})
        //
        // This is complex. For now, compute via explicit matrix multiply.
        //
        // The full 3x3 multiplication is 27 octonion products.
        // We only need the Hermitian part (diagonal + upper triangle).

        let x = self;
        let y = other;

        // Build the full 3x3 matrix entries for X
        // X[i][j] is an octonion
        // X[0][0] = xi_1 * e_0, X[1][1] = xi_2 * e_0, X[2][2] = xi_3 * e_0
        // X[1][2] = x_1, X[0][2] = x_2, X[0][1] = x_3
        // X[2][1] = x_1*, X[2][0] = x_2*, X[1][0] = x_3*

        let x_mat = build_matrix(x);
        let y_mat = build_matrix(y);

        // XY[i][j] = sum_k X[i][k] * Y[k][j]
        let xy = mat_mul(&x_mat, &y_mat);
        let yx = mat_mul(&y_mat, &x_mat);

        // Average: (XY + YX) / 2
        let mut result = AlbertElement::zero();

        // Diagonal: real parts of (XY + YX)[i][i] / 2
        for i in 0..3 {
            result.diag[i] = (xy[i][i][0] + yx[i][i][0]) / 2.0;
        }

        // Off-diagonal: (XY + YX)[i][j] / 2 for i < j
        // off[0] = (1,2) entry, off[1] = (0,2) entry, off[2] = (0,1) entry
        let pairs = [(0, 1, 2), (0, 2, 1), (1, 2, 0)];
        for &(i, j, off_idx) in &pairs {
            for k in 0..8 {
                result.off[off_idx][k] = (xy[i][j][k] + yx[i][j][k]) / 2.0;
            }
        }

        result
    }

    /// Squared Frobenius norm: sum of squares of all 27 real components.
    pub fn norm_sq(&self) -> f64 {
        let d_sq: f64 = self.diag.iter().map(|x| x * x).sum();
        let o_sq: f64 = self.off.iter().map(oct_norm_sq).sum();
        d_sq + 2.0 * o_sq // factor 2 because each off-diagonal appears twice (conjugated)
    }
}

/// Build the full 3x3 octonion matrix from an AlbertElement.
fn build_matrix(x: &AlbertElement) -> [[[f64; 8]; 3]; 3] {
    let mut m = [[[0.0f64; 8]; 3]; 3];

    // Diagonal (real, stored as e_0 component)
    m[0][0][0] = x.diag[0];
    m[1][1][0] = x.diag[1];
    m[2][2][0] = x.diag[2];

    // Off-diagonal: off[0] = x_1 at (1,2), off[1] = x_2 at (0,2), off[2] = x_3 at (0,1)
    m[1][2] = x.off[0];
    m[0][2] = x.off[1];
    m[0][1] = x.off[2];

    // Conjugates for lower triangle
    m[2][1] = oct_conjugate(&x.off[0]);
    m[2][0] = oct_conjugate(&x.off[1]);
    m[1][0] = oct_conjugate(&x.off[2]);

    m
}

/// Multiply two 3x3 octonion matrices.
fn mat_mul(a: &[[[f64; 8]; 3]; 3], b: &[[[f64; 8]; 3]; 3]) -> [[[f64; 8]; 3]; 3] {
    let mut result = [[[0.0f64; 8]; 3]; 3];

    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                let prod = oct_multiply(&a[i][k], &b[k][j]);
                for c in 0..8 {
                    result[i][j][c] += prod[c];
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_albert_diagonal_trace() {
        let x = AlbertElement::diagonal(1.0, 2.0, 3.0);
        assert!((x.trace() - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_albert_diagonal_s2() {
        let x = AlbertElement::diagonal(1.0, 2.0, 3.0);
        // S_2 = 1*2 + 2*3 + 3*1 - 0 - 0 - 0 = 11
        assert!((x.s2() - 11.0).abs() < 1e-12);
    }

    #[test]
    fn test_albert_diagonal_det() {
        let x = AlbertElement::diagonal(1.0, 2.0, 3.0);
        // det = 1*2*3 + 0 - 0 = 6
        assert!((x.det() - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_albert_diagonal_eigenvalues() {
        let x = AlbertElement::diagonal(1.0, 2.0, 3.0);
        let eigs = x.eigenvalues();
        assert!((eigs[0] - 1.0).abs() < 1e-10);
        assert!((eigs[1] - 2.0).abs() < 1e-10);
        assert!((eigs[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_albert_identity_eigenvalues() {
        let x = AlbertElement::diagonal(1.0, 1.0, 1.0);
        let eigs = x.eigenvalues();
        for &e in &eigs {
            assert!(
                (e - 1.0).abs() < 1e-10,
                "identity should have all eigenvalues = 1"
            );
        }
    }

    #[test]
    fn test_albert_jordan_product_diagonal() {
        // Jordan product of diagonal elements should give diagonal result
        let x = AlbertElement::diagonal(1.0, 2.0, 3.0);
        let y = AlbertElement::diagonal(4.0, 5.0, 6.0);

        let xy = x.jordan_product(&y);

        // For diagonal matrices: (XY + YX)/2 = XY (since they commute)
        // XY diagonal = (1*4, 2*5, 3*6) = (4, 10, 18)
        assert!((xy.diag[0] - 4.0).abs() < 1e-10);
        assert!((xy.diag[1] - 10.0).abs() < 1e-10);
        assert!((xy.diag[2] - 18.0).abs() < 1e-10);
    }

    #[test]
    fn test_albert_jordan_product_commutative() {
        // Jordan product must be commutative: X.Y = Y.X
        let mut x = AlbertElement::diagonal(1.0, 2.0, 3.0);
        x.off[0][0] = 0.5; // x_1 real part
        x.off[1][1] = 0.3; // x_2 e_1 component

        let mut y = AlbertElement::diagonal(4.0, 5.0, 6.0);
        y.off[2][2] = 0.7; // x_3 e_2 component

        let xy = x.jordan_product(&y);
        let yx = y.jordan_product(&x);

        // Check commutativity
        for i in 0..3 {
            assert!(
                (xy.diag[i] - yx.diag[i]).abs() < 1e-10,
                "Jordan product must be commutative: diag[{}] = {} vs {}",
                i,
                xy.diag[i],
                yx.diag[i]
            );
        }
        for i in 0..3 {
            for j in 0..8 {
                assert!(
                    (xy.off[i][j] - yx.off[i][j]).abs() < 1e-10,
                    "Jordan product must be commutative: off[{}][{}]",
                    i,
                    j
                );
            }
        }
    }

    #[test]
    fn test_singh_delta_squared_diagonal() {
        // For a diagonal element with arithmetic eigenvalues (1, 2, 3):
        // delta = 1, delta^2 = 1
        let x = AlbertElement::diagonal(1.0, 2.0, 3.0);
        let d2 = x.delta_squared();
        assert!(
            (d2 - 1.0).abs() < 1e-10,
            "diagonal (1,2,3): delta^2 should be 1, got {}",
            d2
        );
    }

    #[test]
    fn test_singh_delta_squared_tracefree() {
        // Singh's key result: for a trace-free element of J_3(O)
        // with specific octonion entries, delta^2 = 3/8.
        //
        // A trace-free element has Tr(X) = 0, so xi_1 + xi_2 + xi_3 = 0.
        //
        // Take xi_1 = 1, xi_2 = 0, xi_3 = -1 (trace = 0).
        // With unit octonion off-diagonal entries.

        // First: simple trace-free diagonal
        let x = AlbertElement::diagonal(1.0, 0.0, -1.0);
        let eigs = x.eigenvalues();
        let d2 = x.delta_squared();

        eprintln!("Trace-free diagonal (1,0,-1):");
        eprintln!("  eigenvalues: {:?}", eigs);
        eprintln!("  delta^2 = {:.6}", d2);
        // This gives delta^2 = 1 (eigenvalues are -1, 0, 1)

        // Now with off-diagonal octonion entries.
        // Try a rank-1 projector: P = v v* / (v* v) where v = (1, e_1, e_2)
        // This is an idempotent in J_3(O) with Tr = 1.
        // For trace-free: X = P - I/3 has Tr = 0.
        //
        // P = [ 1    e_1*  e_2  ]   (normalized)
        //     [ e_1  1     ?    ]
        //     [ e_2* ?     1    ]
        //
        // Actually, let's try the specific construction from Singh:
        // Take X with diagonal (a, b, -(a+b)) and off-diagonal entries
        // that make the characteristic polynomial have the right form.

        // A general trace-free element with unit octonion entries:
        let mut y = AlbertElement::zero();
        y.diag = [1.0, 0.0, -1.0];
        // x_1 = e_1 (unit imaginary octonion)
        y.off[0][1] = 1.0;
        // x_2 = e_2
        y.off[1][2] = 1.0;
        // x_3 = e_4
        y.off[2][4] = 1.0;

        let eigs_y = y.eigenvalues();
        let d2_y = y.delta_squared();
        let (is_arith, q, delta, residual) = y.eigenvalue_arithmetic_check();

        eprintln!("Trace-free with unit octonion off-diagonals:");
        eprintln!("  eigenvalues: {:?}", eigs_y);
        eprintln!("  delta^2 = {:.6}", d2_y);
        eprintln!(
            "  arithmetic: {}, q={:.6}, delta={:.6}, residual={:.2e}",
            is_arith, q, delta, residual
        );

        // Explore a range of configurations
        eprintln!("\nSingh delta^2 survey:");
        let mut delta_sq_values = Vec::new();

        // Sweep: vary diagonal and off-diagonal
        for &a in &[1.0, 0.5, 0.3] {
            let b = -a / 2.0;
            let c = -a - b;
            for oct_idx in [1, 2, 3, 4, 5, 6, 7] {
                let mut z = AlbertElement::zero();
                z.diag = [a, b, c];
                z.off[0][oct_idx] = 1.0;
                z.off[1][(oct_idx + 1) % 7 + 1] = 1.0;
                z.off[2][(oct_idx + 2) % 7 + 1] = 1.0;

                let eigs_z = z.eigenvalues();
                if eigs_z[0].is_nan() {
                    continue;
                }
                let d2_z = z.delta_squared();
                let tr = z.trace();

                eprintln!(
                    "  diag=({:.2},{:.2},{:.2}), oct=[{},{},{}]: eigs=[{:.4},{:.4},{:.4}], d2={:.6}, tr={:.2e}",
                    a, b, c,
                    oct_idx, (oct_idx + 1) % 7 + 1, (oct_idx + 2) % 7 + 1,
                    eigs_z[0], eigs_z[1], eigs_z[2],
                    d2_z, tr
                );

                delta_sq_values.push(d2_z);
            }
        }

        // Report statistics
        if !delta_sq_values.is_empty() {
            let mean = delta_sq_values.iter().sum::<f64>() / delta_sq_values.len() as f64;
            let min = delta_sq_values
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            let max = delta_sq_values
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            eprintln!(
                "\ndelta^2 statistics: mean={:.6}, min={:.6}, max={:.6}",
                mean, min, max
            );
            eprintln!("  3/8 = {:.6}", 3.0 / 8.0);
        }
    }

    #[test]
    fn test_albert_eigenvalue_arithmetic_progression() {
        // For any Hermitian matrix, eigenvalues should form arithmetic
        // progression for rank-1 projectors minus a scaled identity.
        let x = AlbertElement::diagonal(1.0, 2.0, 3.0);
        let (is_arith, q, delta, residual) = x.eigenvalue_arithmetic_check();
        assert!(
            is_arith,
            "diagonal (1,2,3) should be arithmetic: q={}, delta={}, residual={}",
            q, delta, residual
        );
        assert!((q - 2.0).abs() < 1e-10, "middle eigenvalue should be 2");
        assert!((delta - 1.0).abs() < 1e-10, "delta should be 1");
    }

    #[test]
    fn test_albert_characteristic_polynomial() {
        // Verify characteristic polynomial coefficients
        let x = AlbertElement::diagonal(1.0, 2.0, 3.0);
        let (tr, s2, det) = x.characteristic_coefficients();

        assert!((tr - 6.0).abs() < 1e-12, "Tr should be 6");
        assert!((s2 - 11.0).abs() < 1e-12, "S2 should be 11");
        assert!((det - 6.0).abs() < 1e-12, "det should be 6");

        // Verify eigenvalues satisfy the polynomial
        let eigs = x.eigenvalues();
        for &lambda in &eigs {
            let val = lambda * lambda * lambda - tr * lambda * lambda + s2 * lambda - det;
            assert!(
                val.abs() < 1e-8,
                "eigenvalue {} should satisfy characteristic eq, residual = {}",
                lambda,
                val
            );
        }
    }
}
