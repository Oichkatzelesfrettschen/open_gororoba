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

/// Configuration for robust plateau detection.
#[derive(Debug, Clone, Copy)]
pub struct PlateauConfig {
    /// Maximum |curvature| to qualify as plateau.
    pub curvature_threshold: f64,
    /// Minimum consecutive epochs to count as plateau.
    pub min_plateau_length: usize,
    /// Gaussian smoothing window radius (0 = no smoothing).
    pub smoothing_radius: usize,
    /// Whether to use adaptive threshold (fraction of max |curvature|).
    pub adaptive: bool,
    /// Fraction of max curvature for adaptive threshold (typical: 0.05).
    pub adaptive_fraction: f64,
}

impl Default for PlateauConfig {
    fn default() -> Self {
        Self {
            curvature_threshold: 1e-4,
            min_plateau_length: 3,
            smoothing_radius: 2,
            adaptive: false,
            adaptive_fraction: 0.05,
        }
    }
}

/// Gaussian-smooth a signal with given radius.
pub fn gaussian_smooth(signal: &[f64], radius: usize) -> Vec<f64> {
    if radius == 0 || signal.len() < 3 {
        return signal.to_vec();
    }
    let sigma = radius as f64 / 2.0;
    let window_size = 2 * radius + 1;
    let kernel: Vec<f64> = (0..window_size)
        .map(|i| {
            let x = i as f64 - radius as f64;
            (-x * x / (2.0 * sigma * sigma)).exp()
        })
        .collect();
    let kernel_sum: f64 = kernel.iter().sum();
    let kernel: Vec<f64> = kernel.iter().map(|k| k / kernel_sum).collect();

    let n = signal.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mut val = 0.0;
        for (j, &k) in kernel.iter().enumerate() {
            let idx = i as isize + j as isize - radius as isize;
            let idx = idx.clamp(0, n as isize - 1) as usize;
            val += k * signal[idx];
        }
        out.push(val);
    }
    out
}

/// Robust plateau detection with optional smoothing and adaptive threshold.
///
/// Extends `detect_plateaus` with Gaussian pre-smoothing to filter noise
/// and an adaptive threshold mode that sets the cutoff as a fraction of
/// the maximum curvature magnitude.
pub fn detect_plateaus_robust(losses: &[f64], config: &PlateauConfig) -> PlateauDetection {
    if losses.len() < 3 {
        return PlateauDetection {
            plateau_starts: Vec::new(),
            plateau_ends: Vec::new(),
            curvatures: Vec::new(),
            n_plateaus: 0,
        };
    }

    // Optional Gaussian smoothing
    let smoothed = gaussian_smooth(losses, config.smoothing_radius);

    // Compute discrete second derivative
    let n = smoothed.len();
    let mut curvatures = Vec::with_capacity(n);
    curvatures.push(0.0);
    for i in 1..n - 1 {
        let curv = smoothed[i + 1] - 2.0 * smoothed[i] + smoothed[i - 1];
        curvatures.push(curv);
    }
    curvatures.push(0.0);

    // Determine threshold
    let threshold = if config.adaptive {
        let max_curv = curvatures
            .iter()
            .map(|c| c.abs())
            .fold(0.0_f64, f64::max);
        (config.adaptive_fraction * max_curv).max(1e-15)
    } else {
        config.curvature_threshold
    };

    // Identify plateau regions
    let mut plateau_starts = Vec::new();
    let mut plateau_ends = Vec::new();
    let mut in_plateau = false;
    let mut start = 0;

    for (i, &curv) in curvatures.iter().enumerate() {
        let is_flat = curv.abs() < threshold;

        if is_flat && !in_plateau {
            start = i;
            in_plateau = true;
        } else if !is_flat && in_plateau {
            let length = i - start;
            if length >= config.min_plateau_length {
                plateau_starts.push(start);
                plateau_ends.push(i);
            }
            in_plateau = false;
        }
    }

    if in_plateau {
        let length = n - start;
        if length >= config.min_plateau_length {
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
    fn test_gaussian_smooth_identity() {
        let signal = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = gaussian_smooth(&signal, 0);
        assert_eq!(result, signal, "radius=0 should return identity");
    }

    #[test]
    fn test_gaussian_smooth_constant() {
        let signal = vec![5.0; 20];
        let result = gaussian_smooth(&signal, 3);
        for v in &result {
            assert!(
                (v - 5.0).abs() < 1e-10,
                "Smoothing a constant should return constant"
            );
        }
    }

    #[test]
    fn test_gaussian_smooth_reduces_noise() {
        // Noisy step function
        let mut signal = Vec::new();
        for i in 0..40 {
            let base = if i < 20 { 1.0 } else { 0.0 };
            let noise = ((i * 7 + 3) % 11) as f64 / 100.0 - 0.05;
            signal.push(base + noise);
        }
        let smoothed = gaussian_smooth(&signal, 3);
        // Smoothed variance should be lower than original
        let mean = |s: &[f64]| s.iter().sum::<f64>() / s.len() as f64;
        let var = |s: &[f64]| {
            let m = mean(s);
            s.iter().map(|v| (v - m).powi(2)).sum::<f64>() / s.len() as f64
        };
        // Focus on the first half (constant region) -- variance should decrease
        let orig_var = var(&signal[2..18]);
        let smooth_var = var(&smoothed[2..18]);
        assert!(
            smooth_var <= orig_var,
            "Smoothing should reduce variance: orig={}, smooth={}",
            orig_var,
            smooth_var
        );
    }

    #[test]
    fn test_detect_plateaus_robust_with_smoothing() {
        // Noisy two-plateau curve
        let mut losses = Vec::new();
        let mut state = 17_u64;
        let mut noise = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64 / u64::MAX as f64) * 0.002 - 0.001
        };
        for i in 0..10 {
            losses.push(1.0 - 0.05 * i as f64 + noise());
        }
        for _ in 0..15 {
            losses.push(0.5 + noise());
        }
        for i in 0..10 {
            losses.push(0.5 - 0.03 * i as f64 + noise());
        }
        for _ in 0..15 {
            losses.push(0.2 + noise());
        }

        let config = PlateauConfig {
            smoothing_radius: 2,
            adaptive: true,
            adaptive_fraction: 0.05,
            min_plateau_length: 3,
            ..Default::default()
        };
        let result = detect_plateaus_robust(&losses, &config);
        assert!(
            result.n_plateaus >= 2,
            "Robust detection should find at least 2 plateaus: got {}",
            result.n_plateaus
        );
    }

    #[test]
    fn test_detect_plateaus_robust_adaptive_threshold() {
        // Loss with fast decay: curvature ~ 0.0025*exp(-0.05*i), drops below
        // 10% of max by i~47, giving ~53 points of near-flat tail.
        let losses: Vec<f64> = (0..100).map(|i| (-0.05 * i as f64).exp()).collect();
        let config = PlateauConfig {
            adaptive: true,
            adaptive_fraction: 0.1,
            min_plateau_length: 5,
            smoothing_radius: 0,
            ..Default::default()
        };
        let result = detect_plateaus_robust(&losses, &config);
        assert!(
            result.n_plateaus >= 1,
            "Exponential tail should be detected as plateau: got {}",
            result.n_plateaus
        );
    }

    #[test]
    fn test_detect_plateaus_robust_empty() {
        let config = PlateauConfig::default();
        let result = detect_plateaus_robust(&[], &config);
        assert_eq!(result.n_plateaus, 0);
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
