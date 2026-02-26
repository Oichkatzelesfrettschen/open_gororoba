use nalgebra::Matrix4;
use num_complex::Complex64;
use spin_tomography_core::TwoQubitState;

/// Applies an isotropic depolarizing channel to the state.
/// rho -> (1-p) rho + p * I/4
/// p = 1 - exp(-gamma)
pub fn apply_depolarizing_channel(state: &TwoQubitState, gamma: f64) -> TwoQubitState {
    let p = 1.0 - (-gamma).exp();

    // Identity part
    let eye = Matrix4::<Complex64>::identity();
    let mixed = eye.scale(0.25 * p);

    // Original part
    let original = state.rho.scale(1.0 - p);

    TwoQubitState::new(original + mixed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Vector3};

    fn bell_state() -> TwoQubitState {
        // |Phi+> = (|00> + |11>) / sqrt(2) -> singlet-like entangled state
        let a = Vector3::zeros();
        let b = Vector3::zeros();
        let mut t = Matrix3::<f64>::zeros();
        t[(0, 0)] = 1.0;
        t[(1, 1)] = -1.0;
        t[(2, 2)] = 1.0;
        TwoQubitState::from_ab_t(&a, &b, &t)
    }

    #[test]
    fn test_gamma_zero_is_identity() {
        let state = bell_state();
        let out = apply_depolarizing_channel(&state, 0.0);
        // p = 1 - exp(0) = 0, so output should equal input
        for i in 0..4 {
            for j in 0..4 {
                let diff = (state.rho[(i, j)] - out.rho[(i, j)]).norm();
                assert!(diff < 1e-14, "gamma=0 should preserve state: diff={diff}");
            }
        }
    }

    #[test]
    fn test_large_gamma_approaches_maximally_mixed() {
        let state = bell_state();
        let out = apply_depolarizing_channel(&state, 100.0);
        // p -> 1, so output should be I/4
        let expected = Matrix4::<Complex64>::identity() * Complex64::from(0.25);
        for i in 0..4 {
            for j in 0..4 {
                let diff = (out.rho[(i, j)] - expected[(i, j)]).norm();
                assert!(diff < 1e-10, "Large gamma should give I/4: [{i},{j}] diff={diff}");
            }
        }
    }

    #[test]
    fn test_trace_preservation() {
        let state = bell_state();
        let out = apply_depolarizing_channel(&state, 0.5);
        let tr = out.rho.trace().re;
        assert!((tr - 1.0).abs() < 1e-12,
            "Depolarizing channel must preserve trace: got {tr}");
    }

    #[test]
    fn test_entanglement_decreases_with_gamma() {
        let state = bell_state();
        let neg_0 = state.negativity();
        let neg_half = apply_depolarizing_channel(&state, 0.5).negativity();
        let neg_large = apply_depolarizing_channel(&state, 5.0).negativity();
        assert!(neg_0 > neg_half, "Entanglement should decrease: {neg_0} > {neg_half}");
        assert!(neg_half > neg_large, "More decoherence = less entanglement: {neg_half} > {neg_large}");
    }
}
