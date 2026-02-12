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
}
