use nalgebra::Vector3;
use algebra_core::construction::chingon::AlternativityViolationTensor;

/// Computes the non-alternative Chingon drag acceleration.
pub fn compute_chingon_drag(
    v_rel: Vector3<f64>,
    _rho: f64,
    alpha: f64,
    avt: &AlternativityViolationTensor,
) -> Vector3<f64> {
    let mut v_64d = [0.0f64; 64];

    // Robust embedding of 3D velocity into 64D space.
    for (i, slot) in v_64d.iter_mut().enumerate() {
        let t = i as f64;
        let px = (t * 17.0).cos().abs();
        let py = (t * 31.0).sin().abs();
        let pz = (t * 43.0).cos().abs();
        *slot = v_rel.x * px + v_rel.y * py + v_rel.z * pz;
    }

    let mut force_64d = [0.0f64; 64];
    for &(i, j, _k, m, sign) in &avt.violations {
        let contribution = alpha * v_64d[i] * v_64d[j] * (sign as f64);
        force_64d[m] += contribution;
    }

    // Project back to 3D
    let mut res = Vector3::zeros();
    for (i, &f) in force_64d.iter().enumerate() {
        let t = i as f64;
        let px = (t * 17.0).cos().abs();
        let py = (t * 31.0).sin().abs();
        let pz = (t * 43.0).cos().abs();
        res.x += f * px;
        res.y += f * py;
        res.z += f * pz;
    }

    res / 64.0
}
