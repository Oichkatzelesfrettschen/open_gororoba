//! Quaternion SO(3) rotation operators for spatial frames.
//!
//! Provides quaternion-based rotation operations using the Cayley-Dickson
//! construction at dimension 4 (cd_multiply) internally. Quaternions are
//! the most efficient gimbal-lock-free representation of 3D rotations.
//!
//! Convention: q = [w, x, y, z] where w is the scalar part and (x,y,z)
//! is the vector (imaginary) part. A rotation by angle theta about unit
//! axis n is encoded as:
//!   q = cos(theta/2) + sin(theta/2) * (n_x*i + n_y*j + n_z*k)
//!
//! # References
//! - Hamilton (1844): On quaternions
//! - Shepperd (1978): J. Guidance Control 1, 223 (numerically stable extraction)
//! - Cayley-Dickson construction: cd_multiply at dim=4 gives quaternion product

use crate::construction::cayley_dickson::{cd_conjugate, cd_multiply, cd_norm_sq};

/// Quaternion type: [w, x, y, z].
pub type Quaternion = [f64; 4];

/// Identity quaternion (no rotation).
pub const QUAT_IDENTITY: Quaternion = [1.0, 0.0, 0.0, 0.0];

/// Construct a quaternion from an axis-angle rotation.
///
/// q = cos(theta/2) + sin(theta/2) * n
///
/// The axis does NOT need to be pre-normalized.
pub fn quat_from_axis_angle(axis: [f64; 3], angle: f64) -> Quaternion {
    let norm = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]).sqrt();
    if norm < 1e-30 {
        return QUAT_IDENTITY;
    }
    let half = angle / 2.0;
    let s = half.sin() / norm;
    [half.cos(), axis[0] * s, axis[1] * s, axis[2] * s]
}

/// Rotate a 3D vector by a unit quaternion.
///
/// v' = q * v * q^{-1}
///
/// where v is embedded as the pure quaternion (0, v_x, v_y, v_z).
/// Uses cd_multiply for the quaternion products.
pub fn quat_rotate_vector(q: &Quaternion, v: &[f64; 3]) -> [f64; 3] {
    // Embed v as pure quaternion
    let v_quat: [f64; 4] = [0.0, v[0], v[1], v[2]];

    // q * v
    let qv = cd_multiply(q, &v_quat);

    // q^{-1} = conjugate / |q|^2 (for unit quaternion, conjugate suffices)
    let q_conj = cd_conjugate(q);

    // (q * v) * q^{-1}
    let result = cd_multiply(&qv, &q_conj);

    // The scalar part should be ~0; return the vector part
    [result[1], result[2], result[3]]
}

/// Convert a unit quaternion to a 3x3 rotation matrix.
///
/// Direct formula (avoids double quaternion multiplication for batch use):
/// R = [[1-2(y^2+z^2), 2(xy-wz),     2(xz+wy)    ],
///      [2(xy+wz),     1-2(x^2+z^2),  2(yz-wx)    ],
///      [2(xz-wy),     2(yz+wx),      1-2(x^2+y^2)]]
pub fn quat_to_rotation_matrix(q: &Quaternion) -> [[f64; 3]; 3] {
    let (w, x, y, z) = (q[0], q[1], q[2], q[3]);
    let x2 = x * x;
    let y2 = y * y;
    let z2 = z * z;
    let xy = x * y;
    let xz = x * z;
    let yz = y * z;
    let wx = w * x;
    let wy = w * y;
    let wz = w * z;

    [
        [1.0 - 2.0 * (y2 + z2), 2.0 * (xy - wz), 2.0 * (xz + wy)],
        [2.0 * (xy + wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz - wx)],
        [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (x2 + y2)],
    ]
}

/// Convert a 3x3 rotation matrix to a unit quaternion (Shepperd's method).
///
/// Numerically stable for all rotations (avoids the singularity of the
/// naive trace-based extraction near 180-degree rotations).
pub fn quat_from_rotation_matrix(m: &[[f64; 3]; 3]) -> Quaternion {
    // Shepperd (1978): pick the largest diagonal element to avoid division by ~0
    let trace = m[0][0] + m[1][1] + m[2][2];

    if trace > 0.0 {
        let s = 0.5 / (trace + 1.0).sqrt();
        [
            0.25 / s,
            (m[2][1] - m[1][2]) * s,
            (m[0][2] - m[2][0]) * s,
            (m[1][0] - m[0][1]) * s,
        ]
    } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
        let s = 2.0 * (1.0 + m[0][0] - m[1][1] - m[2][2]).sqrt();
        [
            (m[2][1] - m[1][2]) / s,
            0.25 * s,
            (m[0][1] + m[1][0]) / s,
            (m[0][2] + m[2][0]) / s,
        ]
    } else if m[1][1] > m[2][2] {
        let s = 2.0 * (1.0 + m[1][1] - m[0][0] - m[2][2]).sqrt();
        [
            (m[0][2] - m[2][0]) / s,
            (m[0][1] + m[1][0]) / s,
            0.25 * s,
            (m[1][2] + m[2][1]) / s,
        ]
    } else {
        let s = 2.0 * (1.0 + m[2][2] - m[0][0] - m[1][1]).sqrt();
        [
            (m[1][0] - m[0][1]) / s,
            (m[0][2] + m[2][0]) / s,
            (m[1][2] + m[2][1]) / s,
            0.25 * s,
        ]
    }
}

/// Spherical linear interpolation (SLERP) between two quaternions.
///
/// Returns the quaternion at fraction t in [0, 1] along the shortest
/// great-circle arc from q1 to q2 on the 3-sphere.
pub fn quat_slerp(q1: &Quaternion, q2: &Quaternion, t: f64) -> Quaternion {
    // Dot product to find angle
    let mut dot = q1[0] * q2[0] + q1[1] * q2[1] + q1[2] * q2[2] + q1[3] * q2[3];

    // If dot < 0, negate one to take the short path
    let mut q2_adj = *q2;
    if dot < 0.0 {
        dot = -dot;
        for val in &mut q2_adj {
            *val = -*val;
        }
    }

    // If quaternions are very close, use linear interpolation
    if dot > 0.9995 {
        let mut result = [0.0; 4];
        for i in 0..4 {
            result[i] = q1[i] + t * (q2_adj[i] - q1[i]);
        }
        // Normalize
        let norm =
            (result[0].powi(2) + result[1].powi(2) + result[2].powi(2) + result[3].powi(2)).sqrt();
        for r in &mut result {
            *r /= norm;
        }
        return result;
    }

    let omega = dot.acos();
    let sin_omega = omega.sin();
    let s1 = ((1.0 - t) * omega).sin() / sin_omega;
    let s2 = (t * omega).sin() / sin_omega;

    let mut result = [0.0; 4];
    for i in 0..4 {
        result[i] = s1 * q1[i] + s2 * q2_adj[i];
    }
    result
}

/// Compute angular velocity from quaternion and its time derivative.
///
/// omega = 2 * q^{-1} * dq/dt (body frame)
///
/// Returns the angular velocity vector [omega_x, omega_y, omega_z].
pub fn quat_angular_velocity(q: &Quaternion, q_dot: &Quaternion) -> [f64; 3] {
    let q_conj = cd_conjugate(q);
    let norm_sq = cd_norm_sq(q);
    // q^{-1} = conjugate / |q|^2
    let q_inv: [f64; 4] = [
        q_conj[0] / norm_sq,
        q_conj[1] / norm_sq,
        q_conj[2] / norm_sq,
        q_conj[3] / norm_sq,
    ];

    // omega_quat = 2 * q^{-1} * dq/dt
    let product = cd_multiply(&q_inv, q_dot);
    [2.0 * product[1], 2.0 * product[2], 2.0 * product[3]]
}

/// Normalize a quaternion to unit length.
pub fn quat_normalize(q: &Quaternion) -> Quaternion {
    let norm = cd_norm_sq(q).sqrt();
    if norm < 1e-30 {
        return QUAT_IDENTITY;
    }
    [q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm]
}

/// Quaternion product using Cayley-Dickson multiplication.
pub fn quat_multiply(a: &Quaternion, b: &Quaternion) -> Quaternion {
    let result = cd_multiply(a, b);
    [result[0], result[1], result[2], result[3]]
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-10;

    #[test]
    fn test_identity_rotation() {
        let v = [1.0, 2.0, 3.0];
        let v_rot = quat_rotate_vector(&QUAT_IDENTITY, &v);
        for i in 0..3 {
            assert!(
                (v_rot[i] - v[i]).abs() < TOL,
                "identity rotation changed v[{i}]: {} -> {}",
                v[i],
                v_rot[i]
            );
        }
    }

    #[test]
    fn test_90_degree_rotation_about_z() {
        let q = quat_from_axis_angle([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2);
        let v = [1.0, 0.0, 0.0]; // x-axis
        let v_rot = quat_rotate_vector(&q, &v);
        // Should rotate to y-axis: [0, 1, 0]
        assert!((v_rot[0]).abs() < TOL, "x should be ~0: {}", v_rot[0]);
        assert!((v_rot[1] - 1.0).abs() < TOL, "y should be ~1: {}", v_rot[1]);
        assert!((v_rot[2]).abs() < TOL, "z should be ~0: {}", v_rot[2]);
    }

    #[test]
    fn test_180_degree_rotation() {
        let q = quat_from_axis_angle([0.0, 0.0, 1.0], std::f64::consts::PI);
        let v = [1.0, 0.0, 0.0];
        let v_rot = quat_rotate_vector(&q, &v);
        // Should rotate to [-1, 0, 0]
        assert!((v_rot[0] - (-1.0)).abs() < TOL, "180-deg: x = {}", v_rot[0]);
        assert!((v_rot[1]).abs() < TOL, "180-deg: y = {}", v_rot[1]);
    }

    #[test]
    fn test_rotation_preserves_norm() {
        let q = quat_from_axis_angle([1.0, 1.0, 1.0], 1.23);
        let v = [3.0, 4.0, 5.0];
        let v_rot = quat_rotate_vector(&q, &v);
        let norm_orig = (v[0].powi(2) + v[1].powi(2) + v[2].powi(2)).sqrt();
        let norm_rot = (v_rot[0].powi(2) + v_rot[1].powi(2) + v_rot[2].powi(2)).sqrt();
        assert!(
            (norm_orig - norm_rot).abs() < TOL,
            "norm changed: {norm_orig} -> {norm_rot}"
        );
    }

    #[test]
    fn test_matrix_round_trip() {
        let q = quat_from_axis_angle([0.5, 0.3, 0.8], 0.75);
        let m = quat_to_rotation_matrix(&q);
        let q_back = quat_from_rotation_matrix(&m);

        // Quaternion is unique up to sign
        let mut match_positive = true;
        let mut match_negative = true;
        for i in 0..4 {
            if (q[i] - q_back[i]).abs() > 1e-8 {
                match_positive = false;
            }
            if (q[i] + q_back[i]).abs() > 1e-8 {
                match_negative = false;
            }
        }
        assert!(
            match_positive || match_negative,
            "round-trip failed: {:?} vs {:?}",
            q,
            q_back
        );
    }

    #[test]
    fn test_rotation_composition_matches_matrix() {
        let q1 = quat_from_axis_angle([1.0, 0.0, 0.0], 0.5);
        let q2 = quat_from_axis_angle([0.0, 1.0, 0.0], 0.7);

        // Quaternion composition
        let q12 = quat_multiply(&q2, &q1); // q2 * q1 = rotate by q1 then q2
        let v = [1.0, 2.0, 3.0];
        let v_quat = quat_rotate_vector(&q12, &v);

        // Matrix composition
        let m1 = quat_to_rotation_matrix(&q1);
        let m2 = quat_to_rotation_matrix(&q2);

        // Apply m1 then m2
        let mut v_m1 = [0.0; 3];
        for i in 0..3 {
            for j in 0..3 {
                v_m1[i] += m1[i][j] * v[j];
            }
        }
        let mut v_m12 = [0.0; 3];
        for i in 0..3 {
            for j in 0..3 {
                v_m12[i] += m2[i][j] * v_m1[j];
            }
        }

        for i in 0..3 {
            assert!(
                (v_quat[i] - v_m12[i]).abs() < 1e-8,
                "composition mismatch at {i}: quat={} matrix={}",
                v_quat[i],
                v_m12[i]
            );
        }
    }

    #[test]
    fn test_slerp_endpoints() {
        let q1 = quat_from_axis_angle([1.0, 0.0, 0.0], 0.0);
        let q2 = quat_from_axis_angle([0.0, 0.0, 1.0], 1.0);

        let s0 = quat_slerp(&q1, &q2, 0.0);
        let s1 = quat_slerp(&q1, &q2, 1.0);

        for i in 0..4 {
            assert!((s0[i] - q1[i]).abs() < TOL, "slerp(0) should be q1: {i}");
            assert!((s1[i] - q2[i]).abs() < TOL, "slerp(1) should be q2: {i}");
        }
    }

    #[test]
    fn test_slerp_midpoint_is_unit() {
        let q1 = quat_from_axis_angle([1.0, 0.0, 0.0], 0.3);
        let q2 = quat_from_axis_angle([0.0, 1.0, 0.0], 0.8);
        let mid = quat_slerp(&q1, &q2, 0.5);
        let norm = cd_norm_sq(&mid).sqrt();
        assert!(
            (norm - 1.0).abs() < TOL,
            "slerp midpoint should be unit: |q| = {norm}"
        );
    }

    #[test]
    fn test_angular_velocity_stationary() {
        // Zero time derivative => zero angular velocity
        let q = quat_from_axis_angle([1.0, 0.0, 0.0], 0.5);
        let q_dot = [0.0, 0.0, 0.0, 0.0];
        let omega = quat_angular_velocity(&q, &q_dot);
        for (i, &w) in omega.iter().enumerate() {
            assert!(w.abs() < TOL, "stationary omega[{i}] = {}", w);
        }
    }
}
