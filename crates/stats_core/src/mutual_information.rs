//! Mutual Information and Entropy Estimators.
//!
//! Provides:
//! - Kraskov-Stoegbauer-Grassberger (KSG) Mutual Information estimator
//! - KSG Entropy estimator
//!
//! # Literature
//! - Kraskov et al. (2004): Estimating mutual information
//! - Kozachenko & Leonenko (1987): Sample estimate of the entropy of a random vector

use kiddo::{KdTree, float::distance::SquaredEuclidean};
use statrs::function::gamma::digamma;
use std::f64::consts::PI;

/// KSG mutual information estimator for 2D points (e.g., embedded phases).
///
/// `x` and `y` must have the same length. Points are assumed to be 2D.
/// `k` is the number of nearest neighbors.
pub fn ksg_mutual_information_2d(x: &[[f64; 2]], y: &[[f64; 2]], k: usize) -> f64 {
    let n = x.len();
    assert_eq!(n, y.len(), "x and y must have equal length");
    if n < k + 1 {
        return 0.0;
    }

    // Joint space: 4D points
    let mut xy = Vec::with_capacity(n);
    for (x_point, y_point) in x.iter().zip(y.iter()) {
        xy.push([x_point[0], x_point[1], y_point[0], y_point[1]]);
    }

    // Build k-d trees
    let mut tree_xy: KdTree<f64, 4> = KdTree::new();
    let mut tree_x: KdTree<f64, 2> = KdTree::new();
    let mut tree_y: KdTree<f64, 2> = KdTree::new();

    for (index, ((xy_point, x_point), y_point)) in xy.iter().zip(x.iter()).zip(y.iter()).enumerate()
    {
        tree_xy.add(xy_point, index as u64);
        tree_x.add(x_point, index as u64);
        tree_y.add(y_point, index as u64);
    }

    let mut sum_digamma_nx_ny = 0.0;

    for (index, x_point) in x.iter().enumerate() {
        // Find k-th nearest neighbor in joint space.
        // nearest_n returns `k+1` neighbors because the point itself is included (distance 0).
        let neighbors = tree_xy.nearest_n::<SquaredEuclidean>(&xy[index], k + 1);

        // The distance to the k-th neighbor (index k since 0 is self).
        // kiddo returns squared distance.
        let eps_sq = neighbors.last().unwrap().distance;
        let _eps = eps_sq.sqrt();

        // KSG Algorithm 1 uses strict inequality: distance < eps
        // However, for data on a grid, strict inequality can lead to nx=ny=0.
        // We use the full eps radius to include at least the k neighbors.
        let radius_sq = eps_sq;

        // Query marginal spaces
        let nx_elements = tree_x.within::<SquaredEuclidean>(x_point, radius_sq);
        let ny_elements = tree_y.within::<SquaredEuclidean>(&y[index], radius_sq);

        // Count elements with distance < eps.
        // Kiddo's within includes distance == radius_sq.
        // We approximate "strictly less" by checking distances if needed,
        // but typically just using the count is more robust for discrete data.
        let nx = nx_elements.len().saturating_sub(1).max(1) as f64;
        let ny = ny_elements.len().saturating_sub(1).max(1) as f64;

        sum_digamma_nx_ny += digamma(nx) + digamma(ny);
    }

    let mi = digamma(k as f64) + digamma(n as f64) - (sum_digamma_nx_ny / n as f64);
    mi.max(0.0)
}

/// KSG entropy estimator for 2D points.
pub fn entropy_ksg_2d(x: &[[f64; 2]], k: usize) -> f64 {
    let n = x.len();
    if n < k + 1 {
        return 0.0;
    }

    let mut tree: KdTree<f64, 2> = KdTree::new();
    for (index, point) in x.iter().enumerate() {
        tree.add(point, index as u64);
    }

    let d = 2.0; // 2D points
    let mut sum_log_eps = 0.0;

    for point in x.iter().take(n) {
        let neighbors = tree.nearest_n::<SquaredEuclidean>(point, k + 1);
        let eps_sq = neighbors.last().unwrap().distance;
        let eps = eps_sq.sqrt();
        sum_log_eps += (eps + 1e-30).ln();
    }

    let mean_log_eps = sum_log_eps / n as f64;

    // Volume of unit ball in 2D is pi.
    // h = d * mean_log_eps + ln(N) - digamma(k) + d * ln(2) + ln(pi) - ln(Gamma(d/2 + 1))
    // Since d=2, ln(pi) - ln(Gamma(2)) = ln(pi) - ln(1) = ln(pi)
    d * mean_log_eps + (n as f64).ln() - digamma(k as f64) + d * std::f64::consts::LN_2 + PI.ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ksg_mi_identical() {
        // For identical points (x == y), MI should be bounded but high.
        // Actually, identical continuous variables have infinite MI, but estimator gives finite.
        let n = 100;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);

        for i in 0..n {
            let val = i as f64 * 0.1;
            x.push([val.cos(), val.sin()]);
            y.push([val.cos(), val.sin()]);
        }

        let mi = ksg_mutual_information_2d(&x, &y, 5);
        assert!(mi > 0.5);
    }
}
