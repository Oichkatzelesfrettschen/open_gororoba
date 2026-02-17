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
