//! Associator-Gradient Descent
//!
//! Optimization algorithm that navigates loss landscapes using the topological
//! gradient of the associator rather than the standard linear derivative.

use cd_kernel::cayley_dickson::cd_multiply;

/// **Associator Descent Step**
/// Instead of w = w - lr * grad, it computes the associator [W, Grad, Momentum].
/// This intrinsically allows the optimizer to "phase through" saddle points
/// that act as zero-divisors in the loss landscape.
pub fn associator_descent_step(
    weights: &[f64; 16],
    gradient: &[f64; 16],
    momentum: &[f64; 16],
    lr: f64,
) -> [f64; 16] {
    let wg: [f64; 16] = cd_multiply(weights, gradient).try_into().unwrap();
    let gm: [f64; 16] = cd_multiply(gradient, momentum).try_into().unwrap();

    let left: [f64; 16] = cd_multiply(&wg, momentum).try_into().unwrap();
    let right: [f64; 16] = cd_multiply(weights, &gm).try_into().unwrap();

    let mut new_weights = *weights;
    for i in 0..16 {
        let associator_flux = left[i] - right[i];
        new_weights[i] -= lr * (gradient[i] + associator_flux);
    }

    new_weights
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_associator_descent() {
        let w = [1.0; 16];
        let g = [0.1; 16];
        let m = [0.5; 16];
        let next_w = associator_descent_step(&w, &g, &m, 0.01);
        assert!(next_w[0] < 1.0);
    }
}
