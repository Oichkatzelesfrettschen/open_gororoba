use qgp_scaling::{
    competing_models::{MeasuredRaaPoint, arleo_falmagne_raa, compute_bic},
    epsilon_fit::{RaaDataPoint, extract_epsilon},
};

fn data() -> Vec<RaaDataPoint> {
    include_str!("fixtures/alice_pbpb_5020_raa_central.csv")
        .lines()
        .filter_map(|line| {
            let fields: Vec<_> = line.split(',').map(str::parse::<f64>).collect();
            if fields.len() != 10 || fields.iter().any(Result::is_err) {
                return None;
            }
            let values: Vec<_> = fields.into_iter().map(Result::unwrap).collect();
            let momentum = (values[1] + values[2]) / 2.0;
            if momentum < 5.0 {
                return None;
            }
            let total_error = values[4].hypot(values[6]);
            Some(RaaDataPoint {
                pt: momentum,
                raa: values[3],
                stat_err: 0.7 * total_error,
                syst_err: 0.7 * total_error,
            })
        })
        .collect()
}

fn objective(data: &[RaaDataPoint], epsilon: f64, historical_plus_shift: bool) -> f64 {
    data.iter()
        .map(|point| {
            let prediction = if historical_plus_shift {
                (1.0 + epsilon / point.pt).powf(-6.1)
            } else {
                (1.0 - epsilon / point.pt).max(0.0).powf(5.1)
            };
            (point.raa - prediction).powi(2) / (point.stat_err.powi(2) + point.syst_err.powi(2))
        })
        .sum()
}

fn grid_then_golden(objective: impl Fn(f64) -> f64) -> (f64, f64) {
    let step = (20.0 - 0.1) / 20_000.0;
    let index = (0..=20_000)
        .min_by(|left, right| {
            objective(0.1 + f64::from(*left) * step)
                .total_cmp(&objective(0.1 + f64::from(*right) * step))
        })
        .unwrap();
    let mut lower = 0.1 + f64::from((index - 1).max(0)) * step;
    let mut upper = 0.1 + f64::from((index + 1).min(20_000)) * step;
    let fraction = (5.0_f64.sqrt() - 1.0) / 2.0;
    for _ in 0..100 {
        let left = upper - fraction * (upper - lower);
        let right = lower + fraction * (upper - lower);
        if objective(left) < objective(right) {
            upper = right;
        } else {
            lower = left;
        }
    }
    let best = (lower + upper) / 2.0;
    (best, objective(best))
}

#[test]
fn frozen_data_separates_minimizer_from_scored_ansatz() {
    let data = data();
    assert_eq!(data.len(), 6);
    let fit = extract_epsilon(&data, 6.1, 0.1, 20.0, 1e-6);
    let fitted_oracle = grid_then_golden(|epsilon| objective(&data, epsilon, false));
    let scored_oracle = grid_then_golden(|epsilon| objective(&data, epsilon, true));
    println!(
        "production_epsilon={:.17} production_chi2={:.17}",
        fit.epsilon_bar, fit.chi2_min
    );
    println!(
        "same_objective_oracle_epsilon={:.17} same_objective_chi2={:.17}",
        fitted_oracle.0, fitted_oracle.1
    );
    println!(
        "different_scored_objective_oracle_epsilon={:.17} different_scored_chi2={:.17} score_at_fitted_epsilon={:.17}",
        scored_oracle.0,
        scored_oracle.1 * 0.98,
        objective(&data, fit.epsilon_bar, true) * 0.98
    );
    assert!((fit.epsilon_bar - fitted_oracle.0).abs() < 1e-5);
    assert!((fit.chi2_min - fitted_oracle.1).abs() < 1e-7);
    assert!((fit.epsilon_bar - scored_oracle.0).abs() > 0.4);
}

#[test]
fn corrected_scoring_uses_the_fitted_ansatz_at_identical_weights() {
    let data = data();
    let fit = extract_epsilon(&data, 6.1, 0.1, 20.0, 1e-6);
    let momenta: Vec<_> = data.iter().map(|point| point.pt).collect();
    let curve = arleo_falmagne_raa(fit.epsilon_bar, 6.1, &momenta);
    let equal_weights: Vec<_> = data
        .iter()
        .map(|point| MeasuredRaaPoint {
            pt: point.pt,
            raa: point.raa,
            total_err: point.total_err(),
        })
        .collect();
    let corrected = compute_bic(&curve, &equal_weights);
    assert_eq!(corrected.n_points, 6);
    assert_eq!(corrected.n_excluded, 0);
    assert_eq!(corrected.n_params, 1);
    assert!((corrected.chi2 - fit.chi2_min).abs() < 1e-9);
    let original_scoring_weights: Vec<_> = equal_weights
        .iter()
        .map(|point| MeasuredRaaPoint {
            total_err: point.total_err / 0.98_f64.sqrt(),
            ..point.clone()
        })
        .collect();
    let rescaled = compute_bic(&curve, &original_scoring_weights);
    assert!((rescaled.chi2 - corrected.chi2 * 0.98).abs() < 1e-9);
    println!(
        "corrected_same_weight_chi2={:.17} corrected_original_scoring_weight_chi2={:.17} fitted_parameters={} physical_population_admission=separate",
        corrected.chi2, rescaled.chi2, corrected.n_params
    );
}
