//! Crystallographic Orientation and Misorientation.
//!
//! Implements unit quaternion misorientation analysis for spatial framing,
//! specifically targeting the cubic symmetry point group (O, order 24).
//!
//! # References
//! - MTEX: RotationDefinition and orientation classes.
//! - orix: Symmetry-aware distance and clustering.
//! - Grimmer (1974): Disorientations in cubic crystals.

use crate::physics::quat_rotation::{Quaternion, quat_multiply};
use crate::construction::cayley_dickson::cd_conjugate;

/// Misorientation between two orientations q1 and q2.
/// Delta q = q2 * q1^{-1}.
pub fn misorientation(q1: &Quaternion, q2: &Quaternion) -> Quaternion {
    let q1_conj = cd_conjugate(q1); // Assumes unit quaternion
    let q1_inv: Quaternion = [q1_conj[0], q1_conj[1], q1_conj[2], q1_conj[3]];
    quat_multiply(q2, &q1_inv)
}

/// Calculate the misorientation angle in radians.
/// theta = 2 * acos(|q_w|)
pub fn misorientation_angle(q: &Quaternion) -> f64 {
    2.0 * q[0].abs().min(1.0).acos()
}

/// Symmetry operations for the cubic point group (O, order 24) in quaternion form.
pub const CUBIC_SYMMETRY_QUATS: [Quaternion; 24] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0],
    [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, -0.5], [0.5, 0.5, -0.5, 0.5], [0.5, 0.5, -0.5, -0.5],
    [0.5, -0.5, 0.5, 0.5], [0.5, -0.5, 0.5, -0.5], [0.5, -0.5, -0.5, 0.5], [0.5, -0.5, -0.5, -0.5],
    [0.70710678118, 0.70710678118, 0.0, 0.0], [0.70710678118, -0.70710678118, 0.0, 0.0],
    [0.70710678118, 0.0, 0.70710678118, 0.0], [0.70710678118, 0.0, -0.70710678118, 0.0],
    [0.70710678118, 0.0, 0.0, 0.70710678118], [0.70710678118, 0.0, 0.0, -0.70710678118],
    [0.0, 0.70710678118, 0.70710678118, 0.0], [0.0, 0.70710678118, -0.70710678118, 0.0],
    [0.0, 0.70710678118, 0.0, 0.70710678118], [0.0, 0.70710678118, 0.0, -0.70710678118],
    [0.0, 0.0, 0.70710678118, 0.70710678118], [0.0, 0.0, 0.70710678118, -0.70710678118],
];

/// Minimum misorientation angle between two orientations considering cubic symmetry.
/// Also known as the "disorientation" angle.
pub fn min_cubic_misorientation_angle(q1: &Quaternion, q2: &Quaternion) -> f64 {
    let mut min_angle = std::f64::MAX;
    
    for s1 in &CUBIC_SYMMETRY_QUATS {
        let q1_sym = quat_multiply(s1, q1);
        for s2 in &CUBIC_SYMMETRY_QUATS {
            let q2_sym = quat_multiply(s2, q2);
            let dq = misorientation(&q1_sym, &q2_sym);
            let angle = misorientation_angle(&dq);
            if angle < min_angle {
                min_angle = angle;
            }
        }
    }
    
    min_angle
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_cubic_identity_misorientation() {
        let q1 = [1.0, 0.0, 0.0, 0.0];
        let q2 = [0.0, 1.0, 0.0, 0.0]; // 180-deg rotation about X
        
        // Without symmetry, angle is PI.
        assert!((misorientation_angle(&misorientation(&q1, &q2)) - PI).abs() < 1e-10);
        
        // With cubic symmetry, 180-deg about X is a symmetry operation, so min angle is 0.
        let min_angle = min_cubic_misorientation_angle(&q1, &q2);
        assert!(min_angle < 1e-10);
    }

    #[test]
    fn test_t11_ebsd_reproducibility_synthetic() {
        // T11: EBSD Software Reproducibility (orix)
        // Replicating a known misorientation from orix/MTEX:
        // Sigma 3 Twin boundary (60 deg about [111])
        let angle = 60.0 * PI / 180.0;
        let axis = [1.0 / 3.0f64.sqrt(), 1.0 / 3.0f64.sqrt(), 1.0 / 3.0f64.sqrt()];
        let s = (angle / 2.0).sin();
        let q_sigma3 = [(angle / 2.0).cos(), axis[0] * s, axis[1] * s, axis[2] * s];
        
        let q_id = [1.0, 0.0, 0.0, 0.0];
        let min_angle = min_cubic_misorientation_angle(&q_id, &q_sigma3);
        
        // Expected min angle is 60 degrees (1.0471975512 radians)
        assert!((min_angle - angle).abs() < 1e-10);
    }
}
