//! Optional `argmin` pilot integration for low-dimensional cosmology fits.
//!
//! This keeps the existing bounded Nelder-Mead implementation as the default
//! path while exposing a feature-gated adapter for side-by-side evaluation.

use std::sync::Arc;

use argmin::{
    core::{CostFunction, Error, Executor, State},
    solver::neldermead::NelderMead,
};
use rayon::prelude::*;

use crate::{
    halo_profile::{NfwFitResult, RadialBin, nfw_density},
    optimizer::NelderMeadConfig,
};

type ObjectiveFn = Arc<dyn Fn(&[f64]) -> f64 + Sync + Send>;

struct ProjectedObjective {
    bounds: Vec<(f64, f64)>,
    objective: ObjectiveFn,
}

impl CostFunction for ProjectedObjective {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let projected = project_to_bounds(param, &self.bounds);
        Ok((self.objective)(&projected))
    }
}

fn project_to_bounds(values: &[f64], bounds: &[(f64, f64)]) -> Vec<f64> {
    values
        .iter()
        .zip(bounds.iter())
        .map(|(&value, &(lo, hi))| value.clamp(lo, hi))
        .collect()
}

fn build_simplex(x0: &[f64], config: &NelderMeadConfig) -> Vec<Vec<f64>> {
    let n = x0.len();
    let mut simplex = Vec::with_capacity(n + 1);
    simplex.push(project_to_bounds(x0, &config.bounds));
    for i in 0..n {
        let mut candidate = x0.to_vec();
        let range = config.bounds[i].1 - config.bounds[i].0;
        candidate[i] += range * 0.05;
        simplex.push(project_to_bounds(&candidate, &config.bounds));
    }
    simplex
}

fn run_argmin_start(
    objective: ObjectiveFn,
    x0: &[f64],
    config: &NelderMeadConfig,
) -> (Vec<f64>, f64) {
    let simplex = build_simplex(x0, config);
    let solver = match NelderMead::new(simplex).with_sd_tolerance(config.tol) {
        Ok(solver) => solver,
        Err(_) => {
            let projected = project_to_bounds(x0, &config.bounds);
            let cost = objective(&projected);
            return (projected, cost);
        }
    };
    let problem = ProjectedObjective {
        bounds: config.bounds.clone(),
        objective: objective.clone(),
    };
    let run = Executor::new(problem, solver)
        .configure(|state| state.max_iters(config.max_iter as u64))
        .run();
    match run {
        Ok(result) => {
            let state = result.state();
            let best = state
                .get_best_param()
                .cloned()
                .unwrap_or_else(|| project_to_bounds(x0, &config.bounds));
            let best = project_to_bounds(&best, &config.bounds);
            (best.clone(), objective(&best))
        }
        Err(_) => {
            let projected = project_to_bounds(x0, &config.bounds);
            let cost = objective(&projected);
            (projected, cost)
        }
    }
}

/// Feature-gated `argmin` implementation of bounded Nelder-Mead.
///
/// The objective is projected back into the configured box constraints before
/// evaluation so the public contract matches the default optimizer module.
pub fn bounded_nelder_mead_argmin<F>(
    f: F,
    initial_guesses: &[Vec<f64>],
    config: &NelderMeadConfig,
) -> (Vec<f64>, f64)
where
    F: Fn(&[f64]) -> f64 + Sync + Send + 'static,
{
    let objective: ObjectiveFn = Arc::new(f);
    initial_guesses
        .par_iter()
        .map(|x0| run_argmin_start(objective.clone(), x0, config))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap()
}

/// Pilot NFW fit using `argmin` as the optimizer backend.
pub fn fit_nfw_profile_argmin(profile: &[RadialBin]) -> NfwFitResult {
    let valid_bins: Vec<&RadialBin> = profile
        .iter()
        .filter(|b| b.count > 0 && b.r > 0.0)
        .collect();

    if valid_bins.len() < 2 {
        return NfwFitResult {
            r_s: 1.0,
            rho_s: 1.0,
            chi2: f64::INFINITY,
            concentration: None,
        };
    }

    let rho_max = valid_bins.iter().map(|b| b.rho).fold(0.0f64, f64::max);
    let r_min_data = valid_bins[0].r;
    let r_max_data = valid_bins.last().map(|b| b.r).unwrap_or(10.0);

    let config = NelderMeadConfig {
        bounds: vec![
            (r_min_data * 0.1, r_max_data * 2.0),
            (rho_max * 0.001, rho_max * 1000.0),
        ],
        max_iter: 5000,
        tol: 1e-14,
        ..Default::default()
    };

    let first_bin = valid_bins[0];
    let guesses: Vec<Vec<f64>> = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
        .iter()
        .map(|&frac| {
            let r_s_guess = r_min_data + frac * (r_max_data - r_min_data);
            let x = first_bin.r / r_s_guess;
            let rho_s_guess = first_bin.rho * x * (1.0 + x).powi(2);
            vec![r_s_guess, rho_s_guess.max(rho_max * 0.001)]
        })
        .collect();

    let bins = valid_bins
        .iter()
        .map(|bin| RadialBin {
            r: bin.r,
            rho: bin.rho,
            count: bin.count,
        })
        .collect::<Vec<_>>();
    let objective = move |params: &[f64]| -> f64 {
        let r_s = params[0];
        let rho_s = params[1];
        bins.iter()
            .map(|bin| {
                let rho_model = nfw_density(bin.r, r_s, rho_s);
                let log_data = (bin.rho + 1e-30).ln();
                let log_model = (rho_model + 1e-30).ln();
                let diff = log_data - log_model;
                diff * diff
            })
            .sum()
    };

    let (best, chi2) = bounded_nelder_mead_argmin(objective, &guesses, &config);
    NfwFitResult {
        r_s: best[0],
        rho_s: best[1],
        chi2,
        concentration: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_argmin_rosenbrock() {
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
        let guesses = vec![vec![0.0, 0.0], vec![-1.0, 2.0], vec![2.0, -1.0]];

        let (best, fval) = bounded_nelder_mead_argmin(rosenbrock, &guesses, &config);
        assert!((best[0] - 1.0).abs() < 0.05, "x = {}", best[0]);
        assert!((best[1] - 1.0).abs() < 0.05, "y = {}", best[1]);
        assert!(fval < 1e-3, "fval = {fval}");
    }

    #[test]
    fn test_fit_nfw_profile_argmin_recovers_known_profile() {
        let r_s_true = 5.0;
        let rho_s_true = 100.0;
        let profile: Vec<RadialBin> = (1..50)
            .map(|i| RadialBin {
                r: i as f64,
                rho: nfw_density(i as f64, r_s_true, rho_s_true),
                count: 100,
            })
            .collect();

        let result = fit_nfw_profile_argmin(&profile);
        assert!((result.r_s - r_s_true).abs() / r_s_true < 0.15);
        assert!((result.rho_s - rho_s_true).abs() / rho_s_true < 0.15);
    }
}
