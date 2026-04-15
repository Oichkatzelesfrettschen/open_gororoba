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

use gororoba_algebra::construction::chingon::AlternativityViolationTensor;

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
    pub fn drag_acceleration(&self, v: &[f64; 3]) -> [f64; 3] {
        if self.avt.violations.is_empty() {
            return [0.0; 3];
        }

        let dim = self.dim;
        let (vx, vy, vz) = (v[0], v[1], v[2]);

        // Embed 3D velocity into dim-D space.
        // Same deterministic projection as chingon_drag.rs for consistency.
        let mut v_nd = vec![0.0f64; dim];
        for (i, slot) in v_nd.iter_mut().enumerate() {
            let t = i as f64;
            let px = (t * 17.0).cos().abs();
            let py = (t * 31.0).sin().abs();
            let pz = (t * 43.0).cos().abs();
            *slot = vx * px + vy * py + vz * pz;
        }

        // Apply AVT as bilinear form on embedded velocity.
        let mut force_nd = vec![0.0f64; dim];
        for &(i, j, _k, m, sign) in &self.avt.violations {
            force_nd[m] += self.alpha * v_nd[i] * v_nd[j] * (sign as f64);
        }

        // Project back to 3D.
        let mut res = [0.0f64; 3];
        for (i, &f) in force_nd.iter().enumerate() {
            let t = i as f64;
            res[0] += f * (t * 17.0).cos().abs();
            res[1] += f * (t * 31.0).sin().abs();
            res[2] += f * (t * 43.0).cos().abs();
        }

        let s = dim as f64;
        [res[0] / s, res[1] / s, res[2] / s]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn norm3(a: [f64; 3]) -> f64 {
        (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt()
    }

    #[test]
    fn quaternion_no_drag() {
        let force = CdLadderForce::new(4, 1.0);
        assert_eq!(force.violation_count(), 0);
        let a = force.drag_acceleration(&[1.0, 2.0, 3.0]);
        assert_abs_diff_eq!(norm3(a), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn octonion_no_drag() {
        let force = CdLadderForce::new(8, 1.0);
        assert_eq!(force.violation_count(), 0);
        let a = force.drag_acceleration(&[5.0, -3.0, 7.0]);
        assert_abs_diff_eq!(norm3(a), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn sedenion_has_drag() {
        let force = CdLadderForce::new(16, 1e-3);
        assert!(force.violation_count() > 0);
        let a = force.drag_acceleration(&[1.0, 0.0, 0.0]);
        // Non-zero drag from sedenion AVT violations
        assert!(norm3(a) > 0.0);
    }

    #[test]
    fn zero_velocity_zero_drag() {
        let force = CdLadderForce::new(32, 1.0);
        let a = force.drag_acceleration(&[0.0; 3]);
        assert_abs_diff_eq!(norm3(a), 0.0, epsilon = 1e-15);
    }

    #[test]
    fn drag_scales_with_alpha() {
        let v = [1.0, 1.0, 1.0];
        let f1 = CdLadderForce::new(16, 1e-4);
        let f2 = CdLadderForce::new(16, 1e-3);
        let a1 = f1.drag_acceleration(&v);
        let a2 = f2.drag_acceleration(&v);
        // a2 should be 10x a1
        assert_abs_diff_eq!(norm3(a2) / norm3(a1), 10.0, epsilon = 1e-10);
    }
}
