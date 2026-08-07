use gororoba_algebra::construction::chingon::AlternativityViolationTensor;
use nalgebra::Vector3;

/// Computes the Bivector-Embedded Chingon drag acceleration.
///
/// This model implements 'Angular Momentum Bleed' with sign-sensitivity
/// to the orbital plane orientation.
///
/// Bivector B = r ^ v represents the orbital plane.
/// The drag force is modulated by the interaction of B with the AVT.
pub fn compute_bivector_chingon_drag(
    pos: Vector3<f64>,
    vel: Vector3<f64>,
    v_wind: Vector3<f64>,
    _rho: f64,
    alpha: f64,
    avt: &AlternativityViolationTensor,
) -> Vector3<f64> {
    let v_rel = vel - v_wind;
    let v_mag = v_rel.norm();
    if v_mag < 1e-9 {
        return Vector3::<f64>::zeros();
    }

    // 1. Compute the Orbital Bivector (Angular Momentum direction)
    let h = pos.cross(&vel); // L = r x p, direction of the bivector
    let h_unit = if h.norm() > 1e-15 {
        h.normalize()
    } else {
        Vector3::z()
    };

    // 2. Embed the 3D velocity AND the 3D bivector into 64D
    let mut v_64d = [0.0f64; 64];
    let mut h_64d = [0.0f64; 64];

    for i in 0..64 {
        let t = i as f64;
        let px = (t * 17.0).cos().abs();
        let py = (t * 31.0).sin().abs();
        let pz = (t * 43.0).cos().abs();
        v_64d[i] = v_rel.x * px + v_rel.y * py + v_rel.z * pz;
        h_64d[i] = h_unit.x * px + h_unit.y * py + h_unit.z * pz;
    }

    // 3. Compute the Alternativity Violation Torque
    // The force depends on the interaction between motion V and the orbital plane H.
    let mut torque_64d = [0.0f64; 64];
    for &(i, j, k, m, sign) in &avt.violations {
        // [V, H, Y] interaction
        // i: velocity component
        // j: bivector component
        // k: identity reference (0)
        if k == 0 {
            let contribution = alpha * v_64d[i] * h_64d[j] * (sign as f64);
            torque_64d[m] += contribution;
        }
    }

    // 4. Project the 64D non-conservative torque back to 3D
    let mut drag = Vector3::<f64>::zeros();
    for (i, &torque) in torque_64d.iter().enumerate() {
        let t = i as f64;
        let px = (t * 17.0).cos().abs();
        let py = (t * 31.0).sin().abs();
        let pz = (t * 43.0).cos().abs();
        drag.x += torque * px;
        drag.y += torque * py;
        drag.z += torque * pz;
    }

    // The result is a non-conservative 'geometric lift' force
    // that depends on the sign of the orbital angular momentum.
    drag * v_mag
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The `k == 0` guard in `compute_bivector_chingon_drag` selects an empty
    /// set. `e_0` is the multiplicative identity, so the associator
    /// `[e_i, e_j, e_0]` vanishes identically and the symmetrised sum with it;
    /// `exhaustive_capped` stores only non-zero sums, so no tuple with `k == 0`
    /// is ever pushed. The dimensions cover the sedenion level where
    /// alternativity first fails and the two levels above it.
    #[test]
    fn no_alternativity_violation_references_the_identity_axis() {
        for dim in [16usize, 32, 64] {
            let avt = AlternativityViolationTensor::new(dim);
            assert!(
                !avt.violations.is_empty(),
                "dim {dim} must violate alternativity"
            );
            assert!(
                avt.violations.iter().all(|&(_, _, k, _, _)| k != 0),
                "dim {dim} stored a violation on the identity axis"
            );
        }
    }

    /// Consequence of the theorem above, asserted independently of it: the
    /// accumulator body never executes, so the returned force is the zero
    /// vector for every input. The arguments below are deliberately generic --
    /// non-collinear position and velocity, a wind that leaves the relative
    /// speed well above the early-return threshold, and a coupling of order
    /// one -- so a non-zero return would signal a real contribution rather
    /// than a degenerate configuration.
    #[test]
    fn bivector_chingon_drag_is_identically_zero() {
        let avt = AlternativityViolationTensor::new(64);
        let cases = [
            (
                Vector3::new(1.0, 2.0, 3.0),
                Vector3::new(-4.0, 5.0, 6.0),
                Vector3::new(0.5, -0.25, 0.75),
            ),
            (
                Vector3::new(-7.0, 0.5, 2.5),
                Vector3::new(3.0, -1.5, 0.25),
                Vector3::new(-2.0, 2.0, -2.0),
            ),
        ];
        for (pos, vel, wind) in cases {
            for alpha in [1.0, 1.0e3, -5.0] {
                let drag = compute_bivector_chingon_drag(pos, vel, wind, 0.3, alpha, &avt);
                assert_eq!(
                    drag,
                    Vector3::zeros(),
                    "pos {pos:?} vel {vel:?} wind {wind:?} alpha {alpha} produced a force"
                );
            }
        }
    }
}
