use crate::{algebraic_triad::AlgebraicTriad, spin_event::SpinEvent};
use nalgebra::{Matrix3, Vector3};

/// Accumulator for spin tomography moments.
#[derive(Debug, Clone, Default)]
pub struct TomographyMoments {
    /// Sum of weights
    pub weight_sum: f64,
    /// Sum of w * (3 n1 . e_i / alpha1)
    pub u_sum: Vector3<f64>,
    /// Sum of w * (3 n2 . e_j / alpha2)
    pub v_sum: Vector3<f64>,
    /// Sum of w * (3 n1 . e_i / alpha1) * (3 n2 . e_j / alpha2)^T
    pub uv_sum: Matrix3<f64>,
}

impl TomographyMoments {
    pub fn new() -> Self {
        Self::default()
    }

    /// Projects the event onto the triad and accumulates normalized moments.
    ///
    /// Normalized variables:
    /// u'_i = 3 * (n1 . e_i) / alpha1
    /// v'_j = 3 * (n2 . e_j) / alpha2
    ///
    /// Then E`[u']` = a, E`[v']` = b, E`[u'v']` = T + ab^T (if no single-spin correlations leak).
    pub fn add(&mut self, event: &SpinEvent, triad: &AlgebraicTriad) {
        let axes = triad.to_axes_vec();

        // Project analyzer directions onto triad axes
        let u_raw = Vector3::new(
            event.n1.dot(&axes[0]),
            event.n1.dot(&axes[1]),
            event.n1.dot(&axes[2]),
        );

        let v_raw = Vector3::new(
            event.n2.dot(&axes[0]),
            event.n2.dot(&axes[1]),
            event.n2.dot(&axes[2]),
        );

        // Normalize by analyzing power to get "spin vector" estimates directly
        // Factor of 3 comes from 1/3 trace identity for Pauli matrices
        let u_norm = u_raw * (3.0 / event.alpha1);
        let v_norm = v_raw * (3.0 / event.alpha2);

        let w = event.weight;

        self.weight_sum += w;
        self.u_sum += u_norm * w;
        self.v_sum += v_norm * w;
        // Outer product u * v^T
        self.uv_sum += (u_norm * v_norm.transpose()) * w;
    }

    /// Returns the estimated single-spin polarizations (a, b) and correlation tensor T.
    ///
    /// Uses the covariance estimator:
    /// T_ij = <u'_i v'_j> - <u'_i><v'_j>
    /// This removes the product of local polarizations from the correlation tensor.
    pub fn estimate(&self) -> (Vector3<f64>, Vector3<f64>, Matrix3<f64>) {
        if self.weight_sum == 0.0 {
            return (Vector3::zeros(), Vector3::zeros(), Matrix3::zeros());
        }

        let inv_w = 1.0 / self.weight_sum;

        let a_mean = self.u_sum * inv_w;
        let b_mean = self.v_sum * inv_w;

        let uv_mean = self.uv_sum * inv_w;

        // Covariance subtraction
        // T_est = <uv> - <a><b>^T
        let t_est = uv_mean - (a_mean * b_mean.transpose());

        (a_mean, b_mean, t_est)
    }

    /// Returns the STAR-compatible scalar P = Tr(T) / 3.
    pub fn p_scalar(&self) -> f64 {
        let (_, _, t) = self.estimate();
        t.trace() / 3.0
    }

    /// Returns the polarization anisotropy: T_zz - 0.5 * (T_xx + T_yy).
    /// For isotropic states, this should be near 0.
    pub fn anisotropy(&self) -> f64 {
        let (_, _, t) = self.estimate();
        t[(2, 2)] - 0.5 * (t[(0, 0)] + t[(1, 1)])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_moments_returns_zeros() {
        let tm = TomographyMoments::new();
        let (a, b, t) = tm.estimate();
        assert_eq!(a, Vector3::zeros());
        assert_eq!(b, Vector3::zeros());
        assert_eq!(t, Matrix3::zeros());
    }

    #[test]
    fn test_single_event_accumulation() {
        let mut tm = TomographyMoments::new();
        let triad = AlgebraicTriad::default();
        let event = SpinEvent::new(
            Vector3::new(0.0, 0.0, 1.0),
            Vector3::new(0.0, 0.0, 1.0),
            1.0,
            1.0,
        );
        tm.add(&event, &triad);
        assert_eq!(tm.weight_sum, 1.0);
        let (a, b, _t) = tm.estimate();
        // n1 = (0,0,1) projected on (x,y,z) axes = (0,0,1), scaled by 3/alpha1 = 3
        assert!((a[2] - 3.0).abs() < 1e-14);
        assert!((b[2] - 3.0).abs() < 1e-14);
    }

    #[test]
    fn test_p_scalar_zero_for_empty() {
        let tm = TomographyMoments::new();
        assert_eq!(tm.p_scalar(), 0.0);
    }

    #[test]
    fn test_anisotropy_zero_for_empty() {
        let tm = TomographyMoments::new();
        assert_eq!(tm.anisotropy(), 0.0);
    }
}
