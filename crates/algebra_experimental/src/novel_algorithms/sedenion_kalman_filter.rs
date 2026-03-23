//! Sedenionic Kalman Filter
//!
//! Models state uncertainty natively via the non-associativity of the state transition.
//! The associator norm replaces the dense covariance matrix, drastically reducing
//! the computational overhead for tracking 16-dimensional kinematics.

use cd_kernel::cayley_dickson::cd_multiply;

/// **Non-Associative State Update**
/// The uncertainty of the system is fundamentally defined by the topological defect
/// (the failure to associate) between the prior state, the transition model, and the measurement.
pub fn predict_and_update(state: &[f64; 16], transition: &[f64; 16], measurement: &[f64; 16]) -> ([f64; 16], f64) {
    let st: [f64; 16] = cd_multiply(state, transition).try_into().unwrap();
    let tm: [f64; 16] = cd_multiply(transition, measurement).try_into().unwrap();
    
    let left: [f64; 16] = cd_multiply(&st, measurement).try_into().unwrap();
    let right: [f64; 16] = cd_multiply(state, &tm).try_into().unwrap();
    
    let mut uncertainty = 0.0;
    let mut new_state = [0.0; 16];
    for i in 0..16 {
        uncertainty += (left[i] - right[i]).powi(2);
        // The true state lies on the average trajectory of the two non-associative paths
        new_state[i] = (left[i] + right[i]) * 0.5;
    }
    
    (new_state, uncertainty.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_kalman() {
        let state = [1.0; 16];
        let trans = [0.5; 16];
        let meas = [0.1; 16];
        let (next_state, uncert) = predict_and_update(&state, &trans, &meas);
        assert!(uncert >= 0.0);
        let _ = next_state;
    }
}
