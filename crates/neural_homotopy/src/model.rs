//! Lightweight neural-homotopy training surrogate.
//!
//! This keeps a deterministic, pure-Rust training path for fast falsification
//! loops while heavier Burn backends are optional.

use crate::stasheff::mean_pentagon_residual;
use crate::tensor_ops::alignment_score;
use crate::training_data::{
    build_sedenion_table, encode_pair, multiplication_samples, MultiplicationSample, SEDENION_DIM,
};
use cosmology_core::bounce::hubble_e_lcdm;

/// Deterministic transition model over encoded `(lhs, rhs)` tokens.
#[derive(Debug, Clone)]
pub struct PairTransitionModel {
    vocab: usize,
    probs: Vec<Vec<f64>>,
}

impl PairTransitionModel {
    /// Train with Laplace smoothing.
    pub fn train(samples: &[MultiplicationSample], smoothing: f64) -> Self {
        let vocab = SEDENION_DIM * SEDENION_DIM;
        let classes = SEDENION_DIM;
        let mut counts = vec![vec![smoothing; classes]; vocab];
        for s in samples {
            let token = encode_pair(s.lhs, s.rhs);
            counts[token][s.product_basis] += 1.0;
        }
        let probs = counts
            .into_iter()
            .map(|row| {
                let sum = row.iter().sum::<f64>().max(1e-12);
                row.into_iter().map(|v| v / sum).collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        Self { vocab, probs }
    }

    pub fn predict_distribution(&self, token: usize) -> &[f64] {
        &self.probs[token % self.vocab]
    }

    pub fn cross_entropy(&self, dataset: &[(usize, usize)]) -> f64 {
        if dataset.is_empty() {
            return 0.0;
        }
        let mut loss = 0.0;
        for &(token, target) in dataset {
            let p = self.predict_distribution(token)[target].max(1e-12);
            loss += -p.ln();
        }
        loss / (dataset.len() as f64)
    }
}

/// Training config for the surrogate optimizer.
#[derive(Debug, Clone, Copy)]
pub struct HomotopyTrainingConfig {
    pub epochs: usize,
    pub learning_rate: f64,
    pub plateau_tolerance: f64,
}

impl Default for HomotopyTrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 64,
            learning_rate: 0.05,
            plateau_tolerance: 1e-4,
        }
    }
}

/// Training trace used as falsifiable observable.
#[derive(Debug, Clone)]
pub struct TrainingTrace {
    pub pentagon_residual: f64,
    pub losses: Vec<f64>,
    pub plateau_epoch: Option<usize>,
    pub hubble_alignment: f64,
}

/// Deterministic benchmark word set for pentagon checks.
pub fn canonical_words(n_words: usize) -> Vec<[usize; 5]> {
    let mut out = Vec::with_capacity(n_words);
    for i in 0..n_words {
        out.push([
            (i + 1) % SEDENION_DIM,
            (3 * i + 2) % SEDENION_DIM,
            (5 * i + 4) % SEDENION_DIM,
            (7 * i + 8) % SEDENION_DIM,
            (11 * i + 3) % SEDENION_DIM,
        ]);
    }
    out
}

/// Reference Hubble curve `H(z)/H0` for comparison with loss trajectories.
pub fn reference_hubble_curve(n: usize) -> Vec<f64> {
    if n == 0 {
        return Vec::new();
    }
    (0..n)
        .map(|i| {
            let z = 2.0 * (i as f64) / ((n - 1).max(1) as f64);
            hubble_e_lcdm(z, 0.315)
        })
        .collect()
}

/// Train deterministic surrogate and return loss + alignment traces.
pub fn train_homotopy_surrogate(cfg: HomotopyTrainingConfig) -> TrainingTrace {
    let samples = multiplication_samples();
    let model = PairTransitionModel::train(&samples, 1.0);
    let dataset = samples
        .iter()
        .map(|s| (encode_pair(s.lhs, s.rhs), s.product_basis))
        .collect::<Vec<_>>();
    let base_loss = model.cross_entropy(&dataset);

    let table = build_sedenion_table();
    let words = canonical_words(64);
    let pentagon_residual = mean_pentagon_residual(&words, &table);

    let mut losses = Vec::with_capacity(cfg.epochs);
    let mut plateau_epoch = None;
    let mut current = base_loss * (1.0 + 0.5 * pentagon_residual);
    for epoch in 0..cfg.epochs {
        let step = cfg.learning_rate * (1.0 + pentagon_residual);
        current *= (1.0 - step).clamp(0.0, 1.0);
        losses.push(current);

        if epoch > 4 {
            let prev = losses[epoch - 1];
            let rel = (prev - current).abs() / prev.max(1e-12);
            if rel < cfg.plateau_tolerance && plateau_epoch.is_none() {
                plateau_epoch = Some(epoch);
            }
        }
    }

    let hubble = reference_hubble_curve(cfg.epochs);
    let hubble_alignment = alignment_score(&losses, &hubble);

    TrainingTrace {
        pentagon_residual,
        losses,
        plateau_epoch,
        hubble_alignment,
    }
}

/// Plateau detection result from curvature analysis.
#[derive(Debug, Clone)]
pub struct PlateauDetection {
    /// Epochs where plateaus begin (curvature drops below threshold)
    pub plateau_starts: Vec<usize>,
    /// Epochs where plateaus end (curvature rises above threshold)
    pub plateau_ends: Vec<usize>,
    /// Curvature values at each epoch (second derivative of loss)
    pub curvatures: Vec<f64>,
    /// Number of distinct plateaus detected
    pub n_plateaus: usize,
}

/// Detect plateaus in a loss curve using curvature analysis.
///
/// Computes the discrete second derivative (curvature) of the loss sequence
/// and identifies regions where the curvature magnitude is below a threshold,
/// indicating flat regions in the loss landscape.
///
/// # Arguments
/// * `losses` - Loss values at each epoch
/// * `curvature_threshold` - Maximum |curvature| to qualify as plateau (typical: 1e-4)
/// * `min_plateau_length` - Minimum consecutive epochs to count as plateau (typical: 3)
pub fn detect_plateaus(
    losses: &[f64],
    curvature_threshold: f64,
    min_plateau_length: usize,
) -> PlateauDetection {
    if losses.len() < 3 {
        return PlateauDetection {
            plateau_starts: Vec::new(),
            plateau_ends: Vec::new(),
            curvatures: Vec::new(),
            n_plateaus: 0,
        };
    }

    // Compute discrete second derivative: d2L/dt2 ~ L[i+1] - 2*L[i] + L[i-1]
    let n = losses.len();
    let mut curvatures = Vec::with_capacity(n);
    curvatures.push(0.0); // No curvature at first point
    for i in 1..n - 1 {
        let curv = losses[i + 1] - 2.0 * losses[i] + losses[i - 1];
        curvatures.push(curv);
    }
    curvatures.push(0.0); // No curvature at last point

    // Identify plateau regions: consecutive epochs with |curvature| < threshold
    let mut plateau_starts = Vec::new();
    let mut plateau_ends = Vec::new();
    let mut in_plateau = false;
    let mut start = 0;

    for (i, &curv) in curvatures.iter().enumerate() {
        let is_flat = curv.abs() < curvature_threshold;

        if is_flat && !in_plateau {
            start = i;
            in_plateau = true;
        } else if !is_flat && in_plateau {
            let length = i - start;
            if length >= min_plateau_length {
                plateau_starts.push(start);
                plateau_ends.push(i);
            }
            in_plateau = false;
        }
    }

    // Handle plateau that extends to end
    if in_plateau {
        let length = n - start;
        if length >= min_plateau_length {
            plateau_starts.push(start);
            plateau_ends.push(n);
        }
    }

    let n_plateaus = plateau_starts.len();

    PlateauDetection {
        plateau_starts,
        plateau_ends,
        curvatures,
        n_plateaus,
    }
}

/// Compute Wasserstein-1 distance between two 1D distributions.
///
/// Uses the sorted quantile representation: W_1 = mean |F^{-1}(u) - G^{-1}(u)|
/// over uniform samples. For comparing plateau locations against cosmological epochs.
pub fn wasserstein_1d(a: &[f64], b: &[f64]) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let mut a_sorted = a.to_vec();
    let mut b_sorted = b.to_vec();
    a_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    b_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));

    // Interpolate to common grid
    let n = a_sorted.len().max(b_sorted.len());
    let mut total = 0.0;
    for i in 0..n {
        let qa = a_sorted[(i * a_sorted.len() / n).min(a_sorted.len() - 1)];
        let qb = b_sorted[(i * b_sorted.len() / n).min(b_sorted.len() - 1)];
        total += (qa - qb).abs();
    }
    total / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pair_transition_model_probability_mass() {
        let samples = multiplication_samples();
        let model = PairTransitionModel::train(&samples, 1.0);
        let row_sum = model.predict_distribution(0).iter().sum::<f64>();
        assert!((row_sum - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_train_homotopy_surrogate_outputs_trace() {
        let trace = train_homotopy_surrogate(HomotopyTrainingConfig::default());
        assert!(!trace.losses.is_empty());
        assert!((0.0..=1.0).contains(&trace.hubble_alignment));
        assert!((0.0..=1.0).contains(&trace.pentagon_residual));
    }

    #[test]
    fn test_detect_plateaus_synthetic_step() {
        // Loss curve: steep descent, flat plateau, steep descent, flat plateau
        let mut losses = Vec::new();
        for i in 0..10 {
            losses.push(1.0 - 0.05 * i as f64); // Descent
        }
        for _ in 0..15 {
            losses.push(0.5); // Plateau 1
        }
        for i in 0..10 {
            losses.push(0.5 - 0.03 * i as f64); // Descent
        }
        for _ in 0..15 {
            losses.push(0.2); // Plateau 2
        }

        let result = detect_plateaus(&losses, 1e-4, 3);
        assert!(
            result.n_plateaus >= 2,
            "Should detect at least 2 plateaus: got {}",
            result.n_plateaus
        );
    }

    #[test]
    fn test_detect_plateaus_monotone_descent() {
        // Smoothly decaying loss with no plateaus
        let losses: Vec<f64> = (0..50).map(|i| (-0.05 * i as f64).exp()).collect();
        let result = detect_plateaus(&losses, 1e-6, 5);
        // Exponential decay has nonzero curvature everywhere, so few or no plateaus
        assert!(
            result.n_plateaus <= 1,
            "Smooth decay should have few plateaus: got {}",
            result.n_plateaus
        );
    }

    #[test]
    fn test_detect_plateaus_empty_input() {
        let result = detect_plateaus(&[], 1e-4, 3);
        assert_eq!(result.n_plateaus, 0);
        assert!(result.curvatures.is_empty());
    }

    #[test]
    fn test_detect_plateaus_curvature_values() {
        // Parabolic loss: L(t) = (t-25)^2 / 625 => constant positive curvature
        let losses: Vec<f64> = (0..50)
            .map(|i| (i as f64 - 25.0).powi(2) / 625.0)
            .collect();
        let result = detect_plateaus(&losses, 1e-10, 3);
        // Curvature should be approximately constant (2/625 = 0.0032)
        for &c in &result.curvatures[1..result.curvatures.len() - 1] {
            assert!(
                (c - 2.0 / 625.0).abs() < 1e-10,
                "Parabolic curvature should be constant: {}",
                c
            );
        }
    }

    #[test]
    fn test_wasserstein_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let w = wasserstein_1d(&a, &a);
        assert!(w.abs() < 1e-14, "Wasserstein of identical should be 0");
    }

    #[test]
    fn test_wasserstein_shifted() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let w = wasserstein_1d(&a, &b);
        assert!(
            (w - 1.0).abs() < 1e-14,
            "Shift by 1 should give W=1: {}",
            w
        );
    }
}
