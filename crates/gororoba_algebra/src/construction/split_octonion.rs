//! Split-Octonion algebra: 8-dimensional composition algebra with signature (4,4).
//!
//! Unlike the standard octonions, split-octonions have zero divisors and
//! satisfy a different norm composition property. They form an important bridge
//! to string theory and M-theory, particularly in relation to the exceptional
//! group embeddings and supersymmetric formulations requiring specific Lorentzian
//! signatures.
//!
//! Implementation follows the mnemonic matrix rule from Gazeau et al. (2026).
//! o = q + l*p, where q, p are quaternions and l^2 = +1.

use nalgebra::{OMatrix, U8};

/// Split-Octonion: 8-dimensional composition algebra with signature (4,4).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitOctonion {
    pub components: [f64; 8],
}

impl SplitOctonion {
    pub fn new(c: [f64; 8]) -> Self {
        Self { components: c }
    }

    pub fn unit() -> Self {
        let mut c = [0.0; 8];
        c[0] = 1.0;
        Self { components: c }
    }

    pub fn zero() -> Self {
        Self {
            components: [0.0; 8],
        }
    }

    /// Basis element e_i where i in {0..7}.
    pub fn basis(i: usize) -> Self {
        assert!(i < 8);
        let mut c = [0.0; 8];
        c[i] = 1.0;
        Self { components: c }
    }

    /// Split-octonion conjugate.
    pub fn conjugate(&self) -> Self {
        let mut conj = self.components;
        for c in conj.iter_mut().skip(1) {
            *c = -*c;
        }
        Self { components: conj }
    }

    /// Norm squared (signature 4,4).
    /// N(q + lp) = N(q) - N(p) for l^2 = +1.
    /// Standard basis: e0..e3 square to -1 (except e0), e4..e7 square to +1.
    pub fn norm_squared(&self) -> f64 {
        let c = &self.components;
        // e0^2=1, e1^2=e2^2=e3^2=-1, e4^2=e5^2=e6^2=e7^2=1
        c[0] * c[0] - c[1] * c[1] - c[2] * c[2] - c[3] * c[3]
            + c[4] * c[4]
            + c[5] * c[5]
            + c[6] * c[6]
            + c[7] * c[7]
    }

    /// Split-octonion multiplication using the Gazeau mnemonic.
    /// (q1, p1) * (q2, p2) = (q1*q2 + p2*conj(p1), p2*q1 + p1*conj(q2)) for l^2 = +1.
    pub fn multiply(&self, other: &Self) -> Self {
        let q1 = [
            self.components[0],
            self.components[1],
            self.components[2],
            self.components[3],
        ];
        let p1 = [
            self.components[4],
            self.components[5],
            self.components[6],
            self.components[7],
        ];
        let q2 = [
            other.components[0],
            other.components[1],
            other.components[2],
            other.components[3],
        ];
        let p2 = [
            other.components[4],
            other.components[5],
            other.components[6],
            other.components[7],
        ];

        let q1q2 = quat_mul(&q1, &q2);
        let p2cp1 = quat_mul(&quat_conj(&p2), &p1);
        let p2q1 = quat_mul(&p2, &q1);
        let p1q2c = quat_mul(&p1, &quat_conj(&q2));

        let mut res = [0.0; 8];
        for i in 0..4 {
            res[i] = q1q2[i] + p2cp1[i];
            res[i + 4] = p2q1[i] + p1q2c[i];
        }
        Self { components: res }
    }

    /// Returns the 8x8 real matrix representation of left multiplication L_a.
    /// L_a(x) = a * x.
    pub fn left_multiplication_matrix(&self) -> OMatrix<f64, U8, U8> {
        let mut m = OMatrix::<f64, U8, U8>::zeros();
        for j in 0..8 {
            let ej = Self::basis(j);
            let prod = self.multiply(&ej);
            for i in 0..8 {
                m[(i, j)] = prod.components[i];
            }
        }
        m
    }
}

fn quat_mul(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

fn quat_conj(a: &[f64; 4]) -> [f64; 4] {
    [a[0], -a[1], -a[2], -a[3]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_octonion_unit() {
        let u = SplitOctonion::unit();
        let prod = u.multiply(&u);
        assert_eq!(prod, u);
    }

    #[test]
    fn test_split_octonion_basis_squares() {
        // e1^2 = e2^2 = e3^2 = -1
        for i in 1..4 {
            let ei = SplitOctonion::basis(i);
            let prod = ei.multiply(&ei);
            assert_eq!(prod.components[0], -1.0);
        }
        // e4^2 = e5^2 = e6^2 = e7^2 = 1
        for i in 4..8 {
            let ei = SplitOctonion::basis(i);
            let prod = ei.multiply(&ei);
            assert_eq!(prod.components[0], 1.0);
        }
    }
}
