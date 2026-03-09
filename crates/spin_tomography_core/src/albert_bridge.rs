use crate::state::TwoQubitState;
use gororoba_algebra::construction::albert::AlbertElement;
use nalgebra::{Matrix3, Vector3};

/// Maps a two-qubit state into the Albert algebra J_3(O) for spectral analysis.
///
/// Mapping:
/// - xi_1 = 1/3 (trace is 1, so divide evenly or use fixed weights)
/// - xi_2 = 1/3
/// - xi_3 = 1/3
/// - x_3 (1,2) octonion: real part = a[0], e1 = a[1], e2 = a[2], e3 = b[0], e4 = b[1], e5 = b[2], e6=0, e7=0
/// - x_2 (1,3) octonion: components from first 8 entries of T
/// - x_1 (2,3) octonion: last entry of T + padding
pub fn embed_to_albert(
    _state: &TwoQubitState,
    a: &Vector3<f64>,
    b: &Vector3<f64>,
    t: &Matrix3<f64>,
) -> AlbertElement {
    let diag = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];

    let mut off = [[0.0; 8]; 3];

    // x_3: (1,2) entry - single particle polarizations
    off[2][0] = a[0];
    off[2][1] = a[1];
    off[2][2] = a[2];
    off[2][3] = b[0];
    off[2][4] = b[1];
    off[2][5] = b[2];

    // x_2: (1,3) entry - correlation tensor T (first 8)
    for i in 0..3 {
        for j in 0..3 {
            let idx = i * 3 + j;
            if idx < 8 {
                off[1][idx] = t[(i, j)];
            } else {
                // Last one goes to x_1
                off[0][0] = t[(i, j)];
            }
        }
    }

    AlbertElement::new(diag, off)
}

/// Computes the delta^2 invariant for the state embedded in J_3(O).
/// Singh predicts 3/8 for certain universal spectra.
pub fn albert_delta_sq(
    state: &TwoQubitState,
    a: &Vector3<f64>,
    b: &Vector3<f64>,
    t: &Matrix3<f64>,
) -> f64 {
    let el = embed_to_albert(state, a, b, t);
    el.delta_squared()
}
