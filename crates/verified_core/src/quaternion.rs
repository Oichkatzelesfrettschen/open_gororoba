//! Quaternion operations derived from Rocq extraction.
//!
//! This module is a MANUAL TRANSLATION of the OCaml output in
//! `proofs/extraction/golden/extracted_quat.ml.golden`. Every function
//! mirrors the extracted functor `QuatOps` instantiated at f64 via
//! `axioms.rs`.
//!
//! The Rocq proofs (FloatQuaternion.v, FloatRealBridge.v) guarantee:
//! - quat_rotate computes q * embed(v) * conj(q) correctly
//! - The Hamilton product follows the standard sign convention
//!
//! Mirrors: algebra_core/src/physics/quat_rotation.rs lines 40-61, 69-86

use crate::axioms::{self, T};

/// Quaternion as a 4-tuple: (w, x, y, z).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    pub qw: T,
    pub qx: T,
    pub qy: T,
    pub qz: T,
}

/// 3-vector as a 3-tuple: (x, y, z).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3 {
    pub vx: T,
    pub vy: T,
    pub vz: T,
}

/// Hamilton product of two quaternions.
///
/// Derived from extracted OCaml `QuatOps.quat_mul`.
pub fn quat_mul(p: &Quat, q: &Quat) -> Quat {
    Quat {
        qw: axioms::sub(
            axioms::sub(
                axioms::sub(axioms::mul(p.qw, q.qw), axioms::mul(p.qx, q.qx)),
                axioms::mul(p.qy, q.qy),
            ),
            axioms::mul(p.qz, q.qz),
        ),
        qx: axioms::add(
            axioms::add(
                axioms::add(axioms::mul(p.qw, q.qx), axioms::mul(p.qx, q.qw)),
                axioms::mul(p.qy, q.qz),
            ),
            axioms::opp(axioms::mul(p.qz, q.qy)),
        ),
        qy: axioms::add(
            axioms::add(
                axioms::sub(axioms::mul(p.qw, q.qy), axioms::mul(p.qx, q.qz)),
                axioms::mul(p.qy, q.qw),
            ),
            axioms::mul(p.qz, q.qx),
        ),
        qz: axioms::add(
            axioms::add(
                axioms::add(axioms::mul(p.qw, q.qz), axioms::mul(p.qx, q.qy)),
                axioms::opp(axioms::mul(p.qy, q.qx)),
            ),
            axioms::mul(p.qz, q.qw),
        ),
    }
}

/// Quaternion conjugate: (w, -x, -y, -z).
///
/// Derived from extracted OCaml `QuatOps.quat_conj`.
pub fn quat_conj(q: &Quat) -> Quat {
    Quat {
        qw: q.qw,
        qx: axioms::opp(q.qx),
        qy: axioms::opp(q.qy),
        qz: axioms::opp(q.qz),
    }
}

/// Embed a 3-vector as a pure quaternion (0, x, y, z).
///
/// Derived from extracted OCaml `QuatOps.embed_vec`.
pub fn embed_vec(v: &Vec3) -> Quat {
    Quat {
        qw: axioms::ZERO,
        qx: v.vx,
        qy: v.vy,
        qz: v.vz,
    }
}

/// Extract the vector part of a quaternion.
///
/// Derived from extracted OCaml `QuatOps.extract_vec`.
pub fn extract_vec(q: &Quat) -> Vec3 {
    Vec3 {
        vx: q.qx,
        vy: q.qy,
        vz: q.qz,
    }
}

//<*quatrotatefn>
/// Rotate a 3-vector by a unit quaternion: v' = Im(q * embed(v) * conj(q)).
///
/// Derived from extracted OCaml `QuatOps.quat_rotate`.
/// This is the core verified operation.
pub fn quat_rotate(q: &Quat, v: &Vec3) -> Vec3 {
    extract_vec(&quat_mul(&quat_mul(q, &embed_vec(v)), &quat_conj(q)))
}
//</quatrotatefn>

/// Squared norm of a quaternion: w^2 + x^2 + y^2 + z^2.
///
/// Derived from extracted OCaml `QuatOps.quat_norm_sq`.
pub fn quat_norm_sq(q: &Quat) -> T {
    axioms::add(
        axioms::add(
            axioms::add(axioms::mul(q.qw, q.qw), axioms::mul(q.qx, q.qx)),
            axioms::mul(q.qy, q.qy),
        ),
        axioms::mul(q.qz, q.qz),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-12;

    #[test]
    fn identity_rotation() {
        let q = Quat {
            qw: 1.0,
            qx: 0.0,
            qy: 0.0,
            qz: 0.0,
        };
        let v = Vec3 {
            vx: 1.0,
            vy: 2.0,
            vz: 3.0,
        };
        let r = quat_rotate(&q, &v);
        assert!((r.vx - v.vx).abs() < TOL);
        assert!((r.vy - v.vy).abs() < TOL);
        assert!((r.vz - v.vz).abs() < TOL);
    }

    #[test]
    fn conjugate_involution() {
        let q = Quat {
            qw: 0.5,
            qx: 0.5,
            qy: 0.5,
            qz: 0.5,
        };
        let qq = quat_conj(&quat_conj(&q));
        assert!((qq.qw - q.qw).abs() < TOL);
        assert!((qq.qx - q.qx).abs() < TOL);
        assert!((qq.qy - q.qy).abs() < TOL);
        assert!((qq.qz - q.qz).abs() < TOL);
    }

    #[test]
    fn unit_quaternion_norm() {
        let q = Quat {
            qw: 0.5,
            qx: 0.5,
            qy: 0.5,
            qz: 0.5,
        };
        assert!((quat_norm_sq(&q) - 1.0).abs() < TOL);
    }

    #[test]
    fn rotation_preserves_norm() {
        let half = std::f64::consts::FRAC_PI_4;
        let s = half.sin();
        let c = half.cos();
        let q = Quat {
            qw: c,
            qx: 0.0,
            qy: 0.0,
            qz: s,
        };
        let v = Vec3 {
            vx: 3.0,
            vy: 4.0,
            vz: 5.0,
        };
        let r = quat_rotate(&q, &v);
        let norm_v = v.vx * v.vx + v.vy * v.vy + v.vz * v.vz;
        let norm_r = r.vx * r.vx + r.vy * r.vy + r.vz * r.vz;
        assert!(
            (norm_v - norm_r).abs() < TOL,
            "norm changed: {norm_v} -> {norm_r}"
        );
    }

    #[test]
    fn ninety_degree_z_rotation() {
        let half = std::f64::consts::FRAC_PI_4;
        let s = half.sin();
        let c = half.cos();
        // 90-degree rotation about z-axis
        let q = Quat {
            qw: c,
            qx: 0.0,
            qy: 0.0,
            qz: s,
        };
        let v = Vec3 {
            vx: 1.0,
            vy: 0.0,
            vz: 0.0,
        };
        let r = quat_rotate(&q, &v);
        assert!(r.vx.abs() < TOL, "x should be ~0: {}", r.vx);
        assert!((r.vy - 1.0).abs() < TOL, "y should be ~1: {}", r.vy);
        assert!(r.vz.abs() < TOL, "z should be ~0: {}", r.vz);
    }
}
