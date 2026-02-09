//! Negative-Dimension PDE Solver: Regularized Fractional Operators.
//!
//! Provides:
//! - Regularized fractional kinetic operator: T(k) = (|k| + epsilon)^alpha
//! - Eigenvalue computation via imaginary-time evolution
//! - Epsilon convergence sweep
//!
//! # Literature
//! - Caffarelli & Silvestre (2007), Comm. PDE 32, 1245
//! - Laskin (2000), Phys. Lett. A 268, 298

use num_complex::Complex64;
use rustfft::FftPlanner;
use std::f64::consts::PI;

/// Result from eigenvalue computation.
#[derive(Clone, Debug)]
pub struct EigenResult {
    /// Eigenvalues in ascending order
    pub eigenvalues: Vec<f64>,
    /// Corresponding eigenstates (normalized)
    pub eigenstates: Vec<Vec<f64>>,
}

/// Result from epsilon convergence sweep.
#[derive(Clone, Debug)]
pub struct ConvergenceResult {
    /// Epsilon value
    pub epsilon: f64,
    /// Eigenvalue index
    pub index: usize,
    /// Eigenvalue
    pub value: f64,
    /// Relative change from previous epsilon
    pub rel_change: f64,
}

/// Build the regularized fractional kinetic operator in k-space.
///
/// T(k) = (|k| + epsilon)^alpha
///
/// # Arguments
/// * `n` - Number of grid points
/// * `l` - Domain size [-L/2, L/2]
/// * `alpha` - Fractional exponent (alpha < 0 for negative-dimension)
/// * `epsilon` - Regularization parameter (epsilon > 0)
pub fn build_kinetic_operator(n: usize, l: f64, alpha: f64, epsilon: f64) -> (Vec<f64>, Vec<f64>) {
    let _dx = l / n as f64;
    let k: Vec<f64> = (0..n)
        .map(|i| {
            let freq = if i <= n / 2 {
                i as f64
            } else {
                i as f64 - n as f64
            };
            2.0 * PI * freq / l
        })
        .collect();

    let t_k: Vec<f64> = k
        .iter()
        .map(|&ki| (ki.abs() + epsilon).powf(alpha))
        .collect();

    (t_k, k)
}

/// Compute eigenvalues of H = T(k) + V(x) via imaginary-time evolution.
///
/// Uses Strang splitting with Gram-Schmidt orthogonalization for
/// successive eigenstate computation.
///
/// # Arguments
/// * `alpha` - Fractional exponent
/// * `epsilon` - Regularization parameter
/// * `n` - Grid points
/// * `l` - Domain size
/// * `n_eig` - Number of eigenvalues to compute
/// * `dt` - Imaginary time step
/// * `n_steps` - Number of time steps
pub fn eigenvalues_imaginary_time(
    alpha: f64,
    epsilon: f64,
    n: usize,
    l: f64,
    n_eig: usize,
    dt: f64,
    n_steps: usize,
) -> EigenResult {
    let dx = l / n as f64;
    let x: Vec<f64> = (0..n).map(|i| -l / 2.0 + i as f64 * dx).collect();
    let v: Vec<f64> = x.iter().map(|&xi| 0.5 * xi * xi).collect();

    let (t_k, _) = build_kinetic_operator(n, l, alpha, epsilon);

    // Precompute exponentials for Strang splitting
    let exp_v_half: Vec<f64> = v.iter().map(|&vi| (-vi * dt / 2.0).exp()).collect();
    let exp_t: Vec<f64> = t_k.iter().map(|&ti| (-ti * dt).exp()).collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    let mut eigenvalues = Vec::with_capacity(n_eig);
    let mut eigenstates: Vec<Vec<f64>> = Vec::with_capacity(n_eig);

    for m in 0..n_eig {
        // Initial trial: Hermite-Gaussian-like
        let mut psi: Vec<f64> = if m == 0 {
            x.iter().map(|&xi| (-xi * xi / 2.0).exp()).collect()
        } else {
            x.iter()
                .map(|&xi| xi.powi(m as i32) * (-xi * xi / 2.0).exp())
                .collect()
        };

        // Normalize
        let mut norm: f64 = psi.iter().map(|&p| p * p).sum::<f64>() * dx;
        norm = norm.sqrt();
        if norm > 1e-30 {
            for p in &mut psi {
                *p /= norm;
            }
        }

        // Imaginary-time evolution
        for _ in 0..n_steps {
            // V half-step
            for (p, &ev) in psi.iter_mut().zip(exp_v_half.iter()) {
                *p *= ev;
            }

            // T full step (in Fourier space)
            let mut buffer: Vec<Complex64> = psi.iter().map(|&p| Complex64::new(p, 0.0)).collect();
            fft.process(&mut buffer);
            for (b, &et) in buffer.iter_mut().zip(exp_t.iter()) {
                *b *= et;
            }
            ifft.process(&mut buffer);
            let scale = 1.0 / n as f64;
            psi = buffer.iter().map(|c| c.re * scale).collect();

            // V half-step
            for (p, &ev) in psi.iter_mut().zip(exp_v_half.iter()) {
                *p *= ev;
            }

            // Gram-Schmidt: orthogonalize against previous states
            for prev in &eigenstates {
                let overlap: f64 = psi
                    .iter()
                    .zip(prev.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f64>()
                    * dx;
                for (p, &pr) in psi.iter_mut().zip(prev.iter()) {
                    *p -= overlap * pr;
                }
            }

            // Renormalize
            norm = psi.iter().map(|&p| p * p).sum::<f64>() * dx;
            norm = norm.sqrt();
            if norm > 1e-30 {
                for p in &mut psi {
                    *p /= norm;
                }
            }
        }

        // Compute energy expectation: <H> = <T> + <V>
        let mut buffer: Vec<Complex64> = psi.iter().map(|&p| Complex64::new(p, 0.0)).collect();
        fft.process(&mut buffer);

        let t_exp: f64 = buffer
            .iter()
            .zip(t_k.iter())
            .map(|(c, &t)| t * c.norm_sqr())
            .sum::<f64>()
            * dx
            / n as f64;

        let v_exp: f64 = psi
            .iter()
            .zip(v.iter())
            .map(|(&p, &vi)| vi * p * p)
            .sum::<f64>()
            * dx;

        eigenvalues.push(t_exp + v_exp);
        eigenstates.push(psi);
    }

    EigenResult {
        eigenvalues,
        eigenstates,
    }
}

/// Sweep epsilon and track eigenvalue convergence.
pub fn epsilon_convergence_sweep(
    alpha: f64,
    epsilon_values: &[f64],
    n: usize,
    l: f64,
    n_eig: usize,
    dt: f64,
    n_steps: usize,
) -> Vec<ConvergenceResult> {
    let mut results = Vec::new();
    let mut prev_eigs: Option<Vec<f64>> = None;

    for &eps in epsilon_values {
        let eigen_result = eigenvalues_imaginary_time(alpha, eps, n, l, n_eig, dt, n_steps);

        for (i, &val) in eigen_result.eigenvalues.iter().enumerate() {
            let rel_change = if let Some(ref prev) = prev_eigs {
                if prev[i].abs() > 1e-30 {
                    (val - prev[i]).abs() / prev[i].abs()
                } else {
                    0.0
                }
            } else {
                0.0
            };

            results.push(ConvergenceResult {
                epsilon: eps,
                index: i,
                value: val,
                rel_change,
            });
        }

        prev_eigs = Some(eigen_result.eigenvalues);
    }

    results
}

/// Compute eigenvalues using standard fractional Laplacian |k|^{2s} in a
/// harmonic potential well, via imaginary-time propagation (split-step FFT).
///
/// Uses the Fourier-multiplier (Riesz) definition of (-Delta)^s. For s in
/// (0,1), this is equivalent to the Caffarelli-Silvestre extension operator
/// by their 2007 theorem, but we compute via the Fourier symbol rather than
/// solving the (d+1)-dimensional extension PDE.
///
/// # References
/// - Caffarelli & Silvestre (2007), Comm. PDE 32, 1245 (equivalence theorem)
/// - Kwasnicki (2017), Fract. Calc. Appl. Anal. 20, 7 (ten definitions)
pub fn caffarelli_silvestre_eigenvalues(s: f64, n: usize, l: f64, n_eig: usize) -> Vec<f64> {
    let dx = l / n as f64;
    let x: Vec<f64> = (0..n).map(|i| -l / 2.0 + i as f64 * dx).collect();
    let v: Vec<f64> = x.iter().map(|&xi| 0.5 * xi * xi).collect();

    let k: Vec<f64> = (0..n)
        .map(|i| {
            let freq = if i <= n / 2 {
                i as f64
            } else {
                i as f64 - n as f64
            };
            2.0 * PI * freq / l
        })
        .collect();

    // Standard fractional Laplacian: |k|^{2s}, with k=0 set to 0
    let t_k: Vec<f64> = k
        .iter()
        .map(|&ki| {
            if ki.abs() < 1e-30 {
                0.0
            } else {
                ki.abs().powf(2.0 * s)
            }
        })
        .collect();

    let dt = 0.005;
    let n_steps = 5000;
    let exp_v_half: Vec<f64> = v.iter().map(|&vi| (-vi * dt / 2.0).exp()).collect();
    let exp_t: Vec<f64> = t_k.iter().map(|&ti| (-ti * dt).exp()).collect();

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n);
    let ifft = planner.plan_fft_inverse(n);

    let mut eigenvalues = Vec::with_capacity(n_eig);
    let mut eigenstates: Vec<Vec<f64>> = Vec::with_capacity(n_eig);

    for m in 0..n_eig {
        let mut psi: Vec<f64> = if m == 0 {
            x.iter().map(|&xi| (-xi * xi / 2.0).exp()).collect()
        } else {
            x.iter()
                .map(|&xi| xi.powi(m as i32) * (-xi * xi / 2.0).exp())
                .collect()
        };

        let mut norm: f64 = psi.iter().map(|&p| p * p).sum::<f64>() * dx;
        norm = norm.sqrt();
        if norm > 1e-30 {
            for p in &mut psi {
                *p /= norm;
            }
        }

        for _ in 0..n_steps {
            for (p, &ev) in psi.iter_mut().zip(exp_v_half.iter()) {
                *p *= ev;
            }

            let mut buffer: Vec<Complex64> = psi.iter().map(|&p| Complex64::new(p, 0.0)).collect();
            fft.process(&mut buffer);
            for (b, &et) in buffer.iter_mut().zip(exp_t.iter()) {
                *b *= et;
            }
            ifft.process(&mut buffer);
            let scale = 1.0 / n as f64;
            psi = buffer.iter().map(|c| c.re * scale).collect();

            for (p, &ev) in psi.iter_mut().zip(exp_v_half.iter()) {
                *p *= ev;
            }

            for prev in &eigenstates {
                let overlap: f64 = psi
                    .iter()
                    .zip(prev.iter())
                    .map(|(&a, &b)| a * b)
                    .sum::<f64>()
                    * dx;
                for (p, &pr) in psi.iter_mut().zip(prev.iter()) {
                    *p -= overlap * pr;
                }
            }

            norm = psi.iter().map(|&p| p * p).sum::<f64>() * dx;
            norm = norm.sqrt();
            if norm > 1e-30 {
                for p in &mut psi {
                    *p /= norm;
                }
            }
        }

        let mut buffer: Vec<Complex64> = psi.iter().map(|&p| Complex64::new(p, 0.0)).collect();
        fft.process(&mut buffer);

        let t_exp: f64 = buffer
            .iter()
            .zip(t_k.iter())
            .map(|(c, &t)| t * c.norm_sqr())
            .sum::<f64>()
            * dx
            / n as f64;

        let v_exp: f64 = psi
            .iter()
            .zip(v.iter())
            .map(|(&p, &vi)| vi * p * p)
            .sum::<f64>()
            * dx;

        eigenvalues.push(t_exp + v_exp);
        eigenstates.push(psi);
    }

    eigenvalues
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_kinetic_operator_positive_alpha() {
        let (t_k, _) = build_kinetic_operator(64, 10.0, 2.0, 0.1);
        // All values should be positive
        for &t in &t_k {
            assert!(t > 0.0);
        }
    }

    #[test]
    fn test_kinetic_operator_negative_alpha() {
        let (t_k, _) = build_kinetic_operator(64, 10.0, -1.5, 0.1);
        // For alpha < 0, T(k) should decay for large |k|
        // Check that max occurs near k=0
        let max_idx = t_k
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert!(
            max_idx == 0 || max_idx == 1,
            "Max should be near k=0, got idx={}",
            max_idx
        );
    }

    #[test]
    fn test_ground_state_positive() {
        // Ground state of harmonic oscillator with fractional kinetic energy
        let result = eigenvalues_imaginary_time(-1.5, 0.1, 64, 10.0, 1, 0.005, 1000);
        assert_eq!(result.eigenvalues.len(), 1);
        assert!(
            result.eigenvalues[0] > 0.0,
            "E0 = {}",
            result.eigenvalues[0]
        );
    }

    #[test]
    fn test_eigenvalues_positive_alpha_ordered() {
        // For positive alpha, kinetic energy increases with |k|, so standard ordering holds
        let result = eigenvalues_imaginary_time(1.0, 0.1, 64, 10.0, 3, 0.005, 3000);
        assert!(
            result.eigenvalues[0] < result.eigenvalues[1],
            "E0={} should be < E1={}",
            result.eigenvalues[0],
            result.eigenvalues[1]
        );
        assert!(
            result.eigenvalues[1] < result.eigenvalues[2],
            "E1={} should be < E2={}",
            result.eigenvalues[1],
            result.eigenvalues[2]
        );
    }

    #[test]
    fn test_eigenvalues_negative_alpha_distinct() {
        // For negative alpha, kinetic energy DECREASES with |k| (inverted physics).
        // The algorithm finds orthogonal stationary states; ordering depends on
        // the interplay between inverted kinetic and harmonic potential.
        // Here we verify eigenvalues are distinct and positive.
        let result = eigenvalues_imaginary_time(-1.0, 0.1, 64, 10.0, 3, 0.005, 2000);
        assert!(result.eigenvalues[0] > 0.0);
        assert!(result.eigenvalues[1] > 0.0);
        assert!(result.eigenvalues[2] > 0.0);
        // Eigenvalues should be distinct
        let tol = 1e-6;
        assert!(
            (result.eigenvalues[0] - result.eigenvalues[1]).abs() > tol,
            "E0 and E1 should be distinct"
        );
        assert!(
            (result.eigenvalues[1] - result.eigenvalues[2]).abs() > tol,
            "E1 and E2 should be distinct"
        );
    }

    #[test]
    fn test_eigenstates_orthogonal() {
        let result = eigenvalues_imaginary_time(-1.0, 0.1, 64, 10.0, 3, 0.005, 2000);
        let dx = 10.0 / 64.0;

        // Check orthogonality of first two states
        let overlap: f64 = result.eigenstates[0]
            .iter()
            .zip(result.eigenstates[1].iter())
            .map(|(&a, &b)| a * b)
            .sum::<f64>()
            * dx;
        assert!(overlap.abs() < 0.1, "Overlap = {}", overlap);
    }

    #[test]
    fn test_eigenstates_normalized() {
        let result = eigenvalues_imaginary_time(-1.5, 0.1, 64, 10.0, 2, 0.005, 2000);
        let dx = 10.0 / 64.0;

        for state in &result.eigenstates {
            let norm: f64 = state.iter().map(|&p| p * p).sum::<f64>() * dx;
            assert_relative_eq!(norm, 1.0, epsilon = 0.05);
        }
    }

    #[test]
    fn test_epsilon_convergence() {
        let eps_values = vec![0.5, 0.1, 0.05];
        let results = epsilon_convergence_sweep(-1.5, &eps_values, 64, 10.0, 2, 0.005, 1000);

        assert_eq!(results.len(), 6); // 2 eigenvalues * 3 epsilon values

        // Later epsilon values should have smaller relative change
        let first_change = results[2].rel_change; // index 0, eps=0.1
        let second_change = results[4].rel_change; // index 0, eps=0.05
                                                   // Convergence should improve or stay similar
        assert!(second_change <= first_change + 0.1);
    }

    #[test]
    fn test_caffarelli_silvestre_ground_state() {
        let eigs = caffarelli_silvestre_eigenvalues(0.5, 64, 10.0, 1);
        assert_eq!(eigs.len(), 1);
        // Ground state should be positive
        assert!(eigs[0] > 0.0);
    }

    #[test]
    fn test_caffarelli_silvestre_ordered() {
        let eigs = caffarelli_silvestre_eigenvalues(0.5, 64, 10.0, 3);
        assert!(eigs[0] < eigs[1]);
        assert!(eigs[1] < eigs[2]);
    }
}
