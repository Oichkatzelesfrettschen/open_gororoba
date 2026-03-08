//! Extended Cayley-Dickson algebra operations (Rocq-verified: C-893..C-910).
//!
//! Complex and quaternion arithmetic beyond what quaternion.rs provides.
//! Includes associator, inverse, and property-tower witnesses.
//! Kernel-checked in `proofs/theories/CayleyDicksonAlgebra.v`, CDAssociator.v, CDInverse.v.

use crate::{
    axioms,
    quaternion::{Quat, quat_conj, quat_mul, quat_norm_sq},
};

/// Complex number (a + bi).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

/// Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i.
pub fn complex_mul(z: &Complex, w: &Complex) -> Complex {
    Complex {
        re: z.re * w.re - z.im * w.im,
        im: z.re * w.im + z.im * w.re,
    }
}

/// Complex conjugate: (a+bi)* = (a-bi).
pub fn complex_conj(z: &Complex) -> Complex {
    Complex {
        re: z.re,
        im: -z.im,
    }
}

/// Complex norm squared: |z|^2 = a^2 + b^2.
pub fn complex_norm_sq(z: &Complex) -> f64 {
    z.re * z.re + z.im * z.im
}

/// Quaternion addition.
pub fn quat_add(p: &Quat, q: &Quat) -> Quat {
    Quat {
        qw: p.qw + q.qw,
        qx: p.qx + q.qx,
        qy: p.qy + q.qy,
        qz: p.qz + q.qz,
    }
}

/// Quaternion scalar multiplication.
pub fn quat_scale(r: f64, q: &Quat) -> Quat {
    Quat {
        qw: r * q.qw,
        qx: r * q.qx,
        qy: r * q.qy,
        qz: r * q.qz,
    }
}

/// Quaternion negation.
pub fn quat_neg(q: &Quat) -> Quat {
    quat_scale(-1.0, q)
}

/// Quaternion associator: [a,b,c] = (a*b)*c - a*(b*c).
/// Returns zero for quaternions (H is associative).
pub fn quat_assoc(a: &Quat, b: &Quat, c: &Quat) -> Quat {
    let lhs = quat_mul(&quat_mul(a, b), c);
    let rhs = quat_mul(a, &quat_mul(b, c));
    quat_add(&lhs, &quat_neg(&rhs))
}

/// Quaternion inverse: conj(q) / |q|^2.
///
/// # Panics
///
/// Panics if `quat_norm_sq(q) == 0`.
pub fn quat_inv(q: &Quat) -> Quat {
    let nsq = quat_norm_sq(q);
    assert!(nsq != 0.0, "cannot invert zero quaternion");
    quat_scale(axioms::div(1.0, nsq), &quat_conj(q))
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn complex_commutative() {
        let z = Complex { re: 2.0, im: 3.0 };
        let w = Complex { re: 5.0, im: 7.0 };
        let zw = complex_mul(&z, &w);
        let wz = complex_mul(&w, &z);
        assert!((zw.re - wz.re).abs() < TOL);
        assert!((zw.im - wz.im).abs() < TOL);
    }

    #[test]
    fn complex_norm_multiplicative() {
        let z = Complex { re: 3.0, im: 4.0 };
        let w = Complex { re: 1.0, im: 2.0 };
        let prod_norm = complex_norm_sq(&complex_mul(&z, &w));
        let norm_prod = complex_norm_sq(&z) * complex_norm_sq(&w);
        assert!(
            (prod_norm - norm_prod).abs() < TOL,
            "|z*w|^2 = {prod_norm} vs |z|^2*|w|^2 = {norm_prod}"
        );
    }

    #[test]
    fn quat_associator_zero() {
        let a = Quat {
            qw: 1.0,
            qx: 2.0,
            qy: 3.0,
            qz: 4.0,
        };
        let b = Quat {
            qw: 5.0,
            qx: 6.0,
            qy: 7.0,
            qz: 8.0,
        };
        let c = Quat {
            qw: 9.0,
            qx: 10.0,
            qy: 11.0,
            qz: 12.0,
        };
        let assoc = quat_assoc(&a, &b, &c);
        assert!(assoc.qw.abs() < TOL, "assoc.w = {}", assoc.qw);
        assert!(assoc.qx.abs() < TOL, "assoc.x = {}", assoc.qx);
        assert!(assoc.qy.abs() < TOL, "assoc.y = {}", assoc.qy);
        assert!(assoc.qz.abs() < TOL, "assoc.z = {}", assoc.qz);
    }

    #[test]
    fn quat_inverse_right() {
        let q = Quat {
            qw: 1.0,
            qx: 2.0,
            qy: 3.0,
            qz: 4.0,
        };
        let q_inv = quat_inv(&q);
        let prod = quat_mul(&q, &q_inv);
        assert!((prod.qw - 1.0).abs() < TOL, "w = {}", prod.qw);
        assert!(prod.qx.abs() < TOL, "x = {}", prod.qx);
        assert!(prod.qy.abs() < TOL, "y = {}", prod.qy);
        assert!(prod.qz.abs() < TOL, "z = {}", prod.qz);
    }

    #[test]
    fn quat_inverse_left() {
        let q = Quat {
            qw: 1.0,
            qx: 2.0,
            qy: 3.0,
            qz: 4.0,
        };
        let q_inv = quat_inv(&q);
        let prod = quat_mul(&q_inv, &q);
        assert!((prod.qw - 1.0).abs() < TOL, "w = {}", prod.qw);
        assert!(prod.qx.abs() < TOL, "x = {}", prod.qx);
        assert!(prod.qy.abs() < TOL, "y = {}", prod.qy);
        assert!(prod.qz.abs() < TOL, "z = {}", prod.qz);
    }

    #[test]
    fn quat_noncommutative() {
        let i = Quat {
            qw: 0.0,
            qx: 1.0,
            qy: 0.0,
            qz: 0.0,
        };
        let j = Quat {
            qw: 0.0,
            qx: 0.0,
            qy: 1.0,
            qz: 0.0,
        };
        let ij = quat_mul(&i, &j);
        let ji = quat_mul(&j, &i);
        // ij = k, ji = -k
        assert!((ij.qz - 1.0).abs() < TOL, "ij.z should be 1");
        assert!((ji.qz + 1.0).abs() < TOL, "ji.z should be -1");
    }
}
