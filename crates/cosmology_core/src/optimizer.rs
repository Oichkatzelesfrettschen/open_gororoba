//! Generic bounded Nelder-Mead optimizer with multi-start parallelism.
//!
//! Extracted from orthoplex_diffusion.rs to serve all cosmological fitting
//! routines (orthoplex free-beta, fixed-beta, profile likelihood, CDG-2).
//!
//! # Algorithm
//!
//! Standard Nelder-Mead simplex with box constraints (projection onto bounds).
//! Multi-start via rayon for escaping local minima.

use rayon::prelude::*;

/// Configuration for bounded Nelder-Mead optimization.
#[derive(Clone, Debug)]
pub struct NelderMeadConfig {
    /// Parameter bounds: (lower, upper) for each dimension.
    pub bounds: Vec<(f64, f64)>,
    /// Maximum iterations per start (default: 2000).
    pub max_iter: usize,
    /// Convergence tolerance on f-range across simplex (default: 1e-10).
    pub tol: f64,
    /// Number of multi-start initial guesses (default: 9).
    pub n_starts: usize,
    /// Reflection coefficient (default: 1.0).
    pub alpha: f64,
    /// Expansion coefficient (default: 2.0).
    pub gamma: f64,
    /// Contraction coefficient (default: 0.5).
    pub rho: f64,
    /// Shrink coefficient (default: 0.5).
    pub sigma: f64,
}

impl Default for NelderMeadConfig {
    fn default() -> Self {
        Self {
            bounds: Vec::new(),
            max_iter: 2000,
            tol: 1e-10,
            n_starts: 9,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }
    }
}

/// Run bounded Nelder-Mead from a single initial guess.
///
/// Returns (best_params, best_fval).
pub fn bounded_nelder_mead_single<F: Fn(&[f64]) -> f64>(
    f: &F,
    x0: &[f64],
    config: &NelderMeadConfig,
) -> (Vec<f64>, f64) {
    let n = x0.len();
    let bounds = &config.bounds;

    let project = |x: &[f64]| -> Vec<f64> {
        x.iter()
            .zip(bounds.iter())
            .map(|(&xi, &(lo, hi))| xi.clamp(lo, hi))
            .collect()
    };

    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(project(x0));
    for i in 0..n {
        let mut v = x0.to_vec();
        let range = bounds[i].1 - bounds[i].0;
        v[i] += range * 0.05;
        simplex.push(project(&v));
    }

    let mut fvals: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    for _ in 0..config.max_iter {
        let mut order: Vec<usize> = (0..=n).collect();
        order.sort_by(|&a, &b| fvals[a].partial_cmp(&fvals[b]).unwrap());
        let sorted_simplex: Vec<Vec<f64>> = order.iter().map(|&i| simplex[i].clone()).collect();
        let sorted_fvals: Vec<f64> = order.iter().map(|&i| fvals[i]).collect();
        simplex = sorted_simplex;
        fvals = sorted_fvals;

        let f_range = fvals[n] - fvals[0];
        if f_range < config.tol {
            break;
        }

        let centroid: Vec<f64> = (0..n)
            .map(|j| simplex[..n].iter().map(|v| v[j]).sum::<f64>() / n as f64)
            .collect();

        // Reflection
        let xr: Vec<f64> = (0..n)
            .map(|j| centroid[j] + config.alpha * (centroid[j] - simplex[n][j]))
            .collect();
        let xr = project(&xr);
        let fr = f(&xr);

        if fr < fvals[0] {
            // Expansion
            let xe: Vec<f64> = (0..n)
                .map(|j| centroid[j] + config.gamma * (xr[j] - centroid[j]))
                .collect();
            let xe = project(&xe);
            let fe = f(&xe);
            if fe < fr {
                simplex[n] = xe;
                fvals[n] = fe;
            } else {
                simplex[n] = xr;
                fvals[n] = fr;
            }
        } else if fr < fvals[n - 1] {
            simplex[n] = xr;
            fvals[n] = fr;
        } else {
            // Contraction
            let xc: Vec<f64> = (0..n)
                .map(|j| centroid[j] + config.rho * (simplex[n][j] - centroid[j]))
                .collect();
            let xc = project(&xc);
            let fc = f(&xc);
            if fc < fvals[n] {
                simplex[n] = xc;
                fvals[n] = fc;
            } else {
                // Shrink
                let best = simplex[0].clone();
                for i in 1..=n {
                    for (sij, &bj) in simplex[i].iter_mut().zip(best.iter()) {
                        *sij = bj + config.sigma * (*sij - bj);
                    }
                    simplex[i] = project(&simplex[i]);
                    fvals[i] = f(&simplex[i]);
                }
            }
        }
    }

    let best_idx = fvals
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap()
        .0;
    (simplex[best_idx].clone(), fvals[best_idx])
}

/// Run multi-start bounded Nelder-Mead optimization.
///
/// Evaluates `initial_guesses` in parallel via rayon, returning the global
/// best (params, fval) across all starts.
pub fn bounded_nelder_mead<F>(
    f: F,
    initial_guesses: &[Vec<f64>],
    config: &NelderMeadConfig,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64 + Sync,
{
    initial_guesses
        .par_iter()
        .map(|x0| bounded_nelder_mead_single(&f, x0, config))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nelder_mead_rosenbrock() {
        // Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        // Minimum at (1, 1) with f = 0.
        let rosenbrock = |p: &[f64]| {
            let x = p[0];
            let y = p[1];
            (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
        };

        let config = NelderMeadConfig {
            bounds: vec![(-5.0, 5.0), (-5.0, 5.0)],
            max_iter: 5000,
            tol: 1e-12,
            ..Default::default()
        };

        let guesses = vec![
            vec![0.0, 0.0],
            vec![-1.0, 2.0],
            vec![2.0, -1.0],
        ];

        let (best, fval) = bounded_nelder_mead(rosenbrock, &guesses, &config);
        assert!((best[0] - 1.0).abs() < 0.01, "x = {}, expected ~1.0", best[0]);
        assert!((best[1] - 1.0).abs() < 0.01, "y = {}, expected ~1.0", best[1]);
        assert!(fval < 1e-4, "fval = {fval}, expected ~0");
    }

    #[test]
    fn test_nelder_mead_bounded_constraint() {
        // Minimum of (x-3)^2 with bounds [0, 2]: solution at x=2.
        let f = |p: &[f64]| (p[0] - 3.0).powi(2);

        let config = NelderMeadConfig {
            bounds: vec![(0.0, 2.0)],
            max_iter: 1000,
            tol: 1e-12,
            ..Default::default()
        };

        let guesses = vec![vec![0.5]];
        let (best, _fval) = bounded_nelder_mead(f, &guesses, &config);
        assert!((best[0] - 2.0).abs() < 0.01, "x = {}, expected ~2.0", best[0]);
    }
}
