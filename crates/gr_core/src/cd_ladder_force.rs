//! Cayley-Dickson ladder force: dimension-dependent drag from algebraic property loss.
//!
//! Maps CD algebra dimension to a non-associative drag force via the
//! Alternativity Violation Tensor (AVT). Below dim=16 the algebra is
//! alternative (octonions) so the drag vanishes identically. At dim>=16
//! the AVT is non-empty and produces a velocity-dependent perturbation.
//!
//! References:
//! - Schafer (1966): On the algebras formed by the Cayley-Dickson process
//! - Moreno (1998): The zero divisors of the Cayley-Dickson algebras over the real numbers

use algebra_core::construction::chingon::AlternativityViolationTensor;
use nalgebra::Vector3;

/// Configuration for a CD-ladder drag force at a specific algebra dimension.
pub struct CdLadderForce {
    /// Cayley-Dickson dimension (must be power of 2, >= 4).
    pub dim: usize,
    /// Coupling constant (dimensionless).
    pub alpha: f64,
    /// Precomputed AVT for this dimension.
    avt: AlternativityViolationTensor,
}

impl CdLadderForce {
    /// Build the force model for a given CD dimension.
    ///
    /// Panics if `dim` is not a power of two or is less than 4.
    pub fn new(dim: usize, alpha: f64) -> Self {
        assert!(
            dim >= 4 && dim.is_power_of_two(),
            "dim must be power-of-two >= 4"
        );
        let avt = AlternativityViolationTensor::new(dim);
        Self { dim, alpha, avt }
    }

    /// Number of AVT violations (zero for dim <= 8).
    pub fn violation_count(&self) -> usize {
        self.avt.violations.len()
    }

    /// Compute the CD-ladder drag acceleration on a body with velocity `v`.
    ///
    /// The 3D velocity is embedded into `dim`-D space via a deterministic
    /// trigonometric projection, the AVT is applied as a bilinear form on
    /// the embedded velocity, and the result is projected back to 3D.
    ///
    /// For dim <= 8 (quaternions, octonions) the AVT is empty and this
    /// returns zero exactly, matching the physical expectation that
    /// alternative algebras produce no anomalous drag.
    pub fn drag_acceleration(&self, v: &Vector3<f64>) -> Vector3<f64> {
        if self.avt.violations.is_empty() {
            return Vector3::zeros();
        }

        let dim = self.dim;

        // Embed 3D velocity into dim-D space.
        // Same deterministic projection as chingon_drag.rs for consistency.
        let mut v_nd = vec![0.0f64; dim];
        for (i, slot) in v_nd.iter_mut().enumerate() {
            let t = i as f64;
            let px = (t * 17.0).cos().abs();
            let py = (t * 31.0).sin().abs();
            let pz = (t * 43.0).cos().abs();
            *slot = v.x * px + v.y * py + v.z * pz;
        }

        // Apply AVT as bilinear form on embedded velocity.
        let mut force_nd = vec![0.0f64; dim];
        for &(i, j, _k, m, sign) in &self.avt.violations {
            force_nd[m] += self.alpha * v_nd[i] * v_nd[j] * (sign as f64);
        }

        // Project back to 3D.
        let mut res = Vector3::zeros();
        for (i, &f) in force_nd.iter().enumerate() {
            let t = i as f64;
            let px = (t * 17.0).cos().abs();
            let py = (t * 31.0).sin().abs();
            let pz = (t * 43.0).cos().abs();
            res.x += f * px;
            res.y += f * py;
            res.z += f * pz;
        }

        res / (dim as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn quaternion_no_drag() {
        let force = CdLadderForce::new(4, 1.0);
        assert_eq!(force.violation_count(), 0);
        let v = Vector3::new(1.0, 2.0, 3.0);
        let a = force.drag_acceleration(&v);
        assert_abs_diff_eq!(a.norm(), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn octonion_no_drag() {
        let force = CdLadderForce::new(8, 1.0);
        assert_eq!(force.violation_count(), 0);
        let v = Vector3::new(5.0, -3.0, 7.0);
        let a = force.drag_acceleration(&v);
        assert_abs_diff_eq!(a.norm(), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn sedenion_has_drag() {
        let force = CdLadderForce::new(16, 1e-3);
        assert!(force.violation_count() > 0);
        let v = Vector3::new(1.0, 0.0, 0.0);
        let a = force.drag_acceleration(&v);
        // Non-zero drag from sedenion AVT violations
        assert!(a.norm() > 0.0);
    }

    #[test]
    fn zero_velocity_zero_drag() {
        let force = CdLadderForce::new(32, 1.0);
        let a = force.drag_acceleration(&Vector3::zeros());
        assert_abs_diff_eq!(a.norm(), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn drag_scales_with_alpha() {
        let v = Vector3::new(1.0, 1.0, 1.0);
        let f1 = CdLadderForce::new(16, 1e-4);
        let f2 = CdLadderForce::new(16, 1e-3);
        let a1 = f1.drag_acceleration(&v);
        let a2 = f2.drag_acceleration(&v);
        // a2 should be 10x a1
        assert_abs_diff_eq!(a2.norm() / a1.norm(), 10.0, epsilon = 1e-10);
    }
}
