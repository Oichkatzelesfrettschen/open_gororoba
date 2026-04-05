//! Fractal Metric Pathfinding
//!
//! Uses Cayley-Dickson fractal metrics to find geodesics in procedurally
//! generated environments (e.g., multiverses or complex 3D gaming terrain).

use cd_kernel::cayley_dickson::cd_multiply;

/// **CD Fractal Pathing**
/// Evaluates the traversal cost through a recursive algebraic structure.
/// Paths that align with the local fractal geometry (multiplication by the local base)
/// traverse at lower cost.
pub fn fractal_heuristic(
    start: &[f64; 16],
    target: &[f64; 16],
    local_fractal_base: &[f64; 16],
) -> f64 {
    let transformation: [f64; 16] = cd_multiply(start, local_fractal_base).try_into().unwrap();
    let mut cost = 0.0;
    for i in 0..16 {
        cost += (transformation[i] - target[i]).powi(2);
    }
    cost.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_fractal_path() {
        let a = [1.0; 16];
        let b = [2.0; 16];
        let base = [0.5; 16];
        let cost = fractal_heuristic(&a, &b, &base);
        assert!(cost > 0.0);
    }
}
