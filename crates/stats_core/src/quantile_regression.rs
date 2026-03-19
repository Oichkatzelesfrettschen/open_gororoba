//! Quantile Regression using Linear Programming.
//!
//! Provides:
//! - Quantile regression solver via Clarabel (Interior Point Method)
//! - Quantile periodogram computation

use clarabel::solver::*;
use clarabel::algebra::CscMatrix;
use std::f64::consts::PI;

/// Quantile regression result.
pub struct QuantileRegressionResult {
    pub beta: Vec<f64>,
}

/// Solve quantile regression: min sum rho_tau(y - X*beta)
///
/// Formulation as LP:
///   min tau * sum(u) + (1-tau) * sum(v)
///   s.t. X*beta + u - v = y
///        u, v >= 0
///
/// Decision variables z = [beta, u, v]  (p + n + n)
pub fn solve_quantile_regression(
    x: &Vec<Vec<f64>>, // [n][p]
    y: &[f64],         // [n]
    tau: f64,
) -> QuantileRegressionResult {
    let n = y.len();
    let p = x[0].len();

    // Objective: c'z
    // c = [0 (p), tau (n), 1-tau (n)]
    let mut c = vec![0.0; p + 2 * n];
    for i in 0..n {
        c[p + i] = tau;
        c[p + n + i] = 1.0 - tau;
    }

    // Matrix A in CSC format
    // A_eq = [X I -I] (n rows)
    // A_ineq = [0 -I_{2n}] (2n rows) to enforce u, v >= 0 via NonnegativeCone
    
    let mut a_total_data = Vec::new();
    let mut a_total_indices = Vec::new();
    let mut a_total_indptr = vec![0];

    // beta columns
    for j in 0..p {
        for i in 0..n {
            a_total_data.push(x[i][j]);
            a_total_indices.push(i);
        }
        // no entries in A_ineq for beta
        a_total_indptr.push(a_total_data.len());
    }

    // u columns
    for j in 0..n {
        // A_eq part: X*beta + 1*u ... = y
        a_total_data.push(1.0);
        a_total_indices.push(j);
        
        // A_ineq part: -1*u <= 0
        a_total_data.push(-1.0);
        a_total_indices.push(n + j);
        
        a_total_indptr.push(a_total_data.len());
    }

    // v columns
    for j in 0..n {
        // A_eq part: X*beta - 1*v = y
        a_total_data.push(-1.0);
        a_total_indices.push(j);
        
        // A_ineq part: -1*v <= 0
        a_total_data.push(-1.0);
        a_total_indices.push(n + n + j);
        
        a_total_indptr.push(a_total_data.len());
    }

    let a_total_csc = CscMatrix::new(
        3 * n,
        p + 2 * n,
        a_total_indptr,
        a_total_indices,
        a_total_data,
    );

    let mut b_total = y.to_vec();
    b_total.extend(vec![0.0; 2 * n]);

    // Cones must specify f64 explicitly
    let total_cones = [ZeroConeT::<f64>(n), NonnegativeConeT::<f64>(2 * n)];

    let p_mat = CscMatrix::new(
        p + 2 * n,
        p + 2 * n,
        vec![0; p + 2 * n + 1], // all zeros for LP
        vec![],
        vec![],
    );

    let settings = DefaultSettings::default();
    let mut solver = DefaultSolver::new(&p_mat, &c, &a_total_csc, &b_total, &total_cones, settings).unwrap();

    solver.solve();

    let beta = solver.variables.x[0..p].to_vec();

    QuantileRegressionResult { beta }
}

/// Compute quantile periodogram at a single frequency.
pub fn quantile_periodogram_at_freq(signal: &[f64], freq: f64, tau: f64) -> f64 {
    let n = signal.len();
    let mut x = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64;
        x.push(vec![
            1.0,
            (2.0 * PI * freq * t).cos(),
            (2.0 * PI * freq * t).sin(),
        ]);
    }

    let res = solve_quantile_regression(&x, signal, tau);
    // Q(f, tau) = beta_c^2 + beta_s^2
    res.beta[1] * res.beta[1] + res.beta[2] * res.beta[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantile_regression_simple() {
        // y = 2 + 0.5*x
        let x = vec![vec![1.0, 1.0], vec![1.0, 2.0], vec![1.0, 3.0], vec![1.0, 4.0], vec![1.0, 5.0]];
        let y = vec![2.5, 3.0, 3.5, 4.0, 4.5];
        
        let res = solve_quantile_regression(&x, &y, 0.5);
        assert!((res.beta[0] - 2.0).abs() < 1e-5);
        assert!((res.beta[1] - 0.5).abs() < 1e-5);
    }
}
