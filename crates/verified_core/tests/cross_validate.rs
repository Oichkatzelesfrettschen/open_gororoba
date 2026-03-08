//! Cross-validation: verified_core vs algebra_core quaternion operations.
//!
//! Confirms that the Rocq-extracted quaternion rotation produces identical
//! results to the Cayley-Dickson-based implementation in algebra_core.

use algebra_core::physics::quat_rotation::{quat_from_axis_angle, quat_rotate_vector};
use verified_core::quaternion::{self, Quat, Vec3};

const TOL: f64 = 1e-12;

fn assert_vec3_eq(a: &Vec3, b: &[f64; 3], label: &str) {
    assert!(
        (a.vx - b[0]).abs() < TOL,
        "{label}: x mismatch: {} vs {}",
        a.vx,
        b[0]
    );
    assert!(
        (a.vy - b[1]).abs() < TOL,
        "{label}: y mismatch: {} vs {}",
        a.vy,
        b[1]
    );
    assert!(
        (a.vz - b[2]).abs() < TOL,
        "{label}: z mismatch: {} vs {}",
        a.vz,
        b[2]
    );
}

fn to_verified_quat(q: &[f64; 4]) -> Quat {
    Quat {
        qw: q[0],
        qx: q[1],
        qy: q[2],
        qz: q[3],
    }
}

#[test]
fn cross_validate_identity() {
    let q_ac = [1.0, 0.0, 0.0, 0.0];
    let v = [1.0, 2.0, 3.0];

    let result_ac = quat_rotate_vector(&q_ac, &v);
    let result_vc = quaternion::quat_rotate(
        &to_verified_quat(&q_ac),
        &Vec3 {
            vx: v[0],
            vy: v[1],
            vz: v[2],
        },
    );

    assert_vec3_eq(&result_vc, &result_ac, "identity");
}

#[test]
fn cross_validate_90_degree_z() {
    let q_ac = quat_from_axis_angle([0.0, 0.0, 1.0], std::f64::consts::FRAC_PI_2);
    let v = [1.0, 0.0, 0.0];

    let result_ac = quat_rotate_vector(&q_ac, &v);
    let result_vc = quaternion::quat_rotate(
        &to_verified_quat(&q_ac),
        &Vec3 {
            vx: v[0],
            vy: v[1],
            vz: v[2],
        },
    );

    assert_vec3_eq(&result_vc, &result_ac, "90-deg-z");
}

#[test]
fn cross_validate_arbitrary_rotation() {
    let q_ac = quat_from_axis_angle([1.0, 2.0, 3.0], 1.23);
    let v = [4.0, 5.0, 6.0];

    let result_ac = quat_rotate_vector(&q_ac, &v);
    let result_vc = quaternion::quat_rotate(
        &to_verified_quat(&q_ac),
        &Vec3 {
            vx: v[0],
            vy: v[1],
            vz: v[2],
        },
    );

    assert_vec3_eq(&result_vc, &result_ac, "arbitrary");
}

//<*crossval>
#[test]
fn cross_validate_multiple_axes() {
    let axes: &[[f64; 3]] = &[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ];
    let angles = [0.1, 0.5, 1.0, 2.0, 3.0, std::f64::consts::PI];
    let v = [7.0, 8.0, 9.0];

    for (i, axis) in axes.iter().enumerate() {
        for (j, &angle) in angles.iter().enumerate() {
            let q_ac = quat_from_axis_angle(*axis, angle);
            let result_ac = quat_rotate_vector(&q_ac, &v);
            let result_vc = quaternion::quat_rotate(
                &to_verified_quat(&q_ac),
                &Vec3 {
                    vx: v[0],
                    vy: v[1],
                    vz: v[2],
                },
            );
            assert_vec3_eq(&result_vc, &result_ac, &format!("axis[{i}]-angle[{j}]"));
        }
    }
}
//</crossval>
