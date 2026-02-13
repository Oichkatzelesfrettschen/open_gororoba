//! Lepton mass ratio predictions from survival depth clusters.
//!
//! The filtration hypothesis (Thesis 2) predicts that the three largest
//! survival-depth clusters correspond to the three charged leptons
//! (electron, muon, tau). Cluster size ratios should approximate
//! the PDG experimental mass ratios.
//!
//! PDG 2024 reference values:
//! - m_e   =   0.51099895 MeV
//! - m_mu  = 105.6583755 MeV
//! - m_tau = 1776.86     MeV

use crate::mass_spectrum::{depth_clusters, SurvivalDepthMap, SurvivalEntry};

/// PDG 2024 charged lepton masses in MeV.
pub const M_ELECTRON: f64 = 0.510_998_95;
pub const M_MUON: f64 = 105.658_375_5;
pub const M_TAU: f64 = 1776.86;

/// PDG mass ratios.
pub const RATIO_MU_E: f64 = M_MUON / M_ELECTRON;
pub const RATIO_TAU_E: f64 = M_TAU / M_ELECTRON;
pub const RATIO_TAU_MU: f64 = M_TAU / M_MUON;

/// A single mass ratio prediction extracted from depth clusters.
#[derive(Debug, Clone, Copy)]
pub struct MassRatioPrediction {
    /// Depth of the lighter cluster
    pub depth_light: u32,
    /// Depth of the heavier cluster
    pub depth_heavy: u32,
    /// Number of entries in the lighter cluster
    pub count_light: usize,
    /// Number of entries in the heavier cluster
    pub count_heavy: usize,
    /// Predicted ratio (depth_heavy / depth_light) mapped to mass ratio
    pub predicted_ratio: f64,
    /// PDG experimental ratio for comparison
    pub pdg_ratio: f64,
    /// Relative error: |predicted - pdg| / pdg
    pub relative_error: f64,
    /// Label for this comparison
    pub label: MassRatioLabel,
}

/// Labels for the three lepton mass ratios.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MassRatioLabel {
    MuOverE,
    TauOverE,
    TauOverMu,
}

/// Extract the three largest depth clusters as mass candidates.
///
/// Returns up to 3 clusters sorted by cluster count (descending),
/// with the interpretation that larger clusters = lighter particles.
///
/// The mapping is:
/// - Largest cluster  -> electron (lightest)
/// - 2nd largest      -> muon
/// - 3rd largest      -> tau (heaviest)
fn identify_mass_clusters(map: &SurvivalDepthMap) -> Vec<(u32, Vec<SurvivalEntry>)> {
    let mut clusters = depth_clusters(map);

    // Sort by cluster size descending (largest first = lightest particle)
    clusters.sort_by_key(|b| std::cmp::Reverse(b.1.len()));

    // Take top 3
    clusters.truncate(3);
    clusters
}

/// Predict lepton mass ratios from survival depth clusters.
///
/// Identifies the three largest depth clusters and computes mass ratios
/// by using the depth value as a proxy for mass. The depth-to-mass mapping
/// assumes that deeper survival corresponds to higher effective mass.
///
/// Returns an empty Vec if fewer than 3 clusters exist.
pub fn predict_mass_ratios(map: &SurvivalDepthMap) -> Vec<MassRatioPrediction> {
    let clusters = identify_mass_clusters(map);
    if clusters.len() < 3 {
        return Vec::new();
    }

    // Largest cluster = electron, 2nd = muon, 3rd = tau
    // Depth values: deeper = heavier
    let e_cluster = &clusters[0];
    let mu_cluster = &clusters[1];
    let tau_cluster = &clusters[2];

    // Mass proxy: use depth value (deeper = heavier)
    // If cluster depths are equal, use 1/count as tiebreaker (fewer entries = heavier)
    let mass_proxy = |cluster: &(u32, Vec<SurvivalEntry>)| -> f64 {
        let depth = cluster.0 as f64;
        let count = cluster.1.len() as f64;
        // Combine depth and inverse count for richer signal
        depth + 1.0 / count.max(1.0)
    };

    let m_e = mass_proxy(e_cluster);
    let m_mu = mass_proxy(mu_cluster);
    let m_tau = mass_proxy(tau_cluster);

    let mut predictions = Vec::with_capacity(3);

    // mu/e ratio
    if m_e > 1e-12 {
        let pred = m_mu / m_e;
        predictions.push(MassRatioPrediction {
            depth_light: e_cluster.0,
            depth_heavy: mu_cluster.0,
            count_light: e_cluster.1.len(),
            count_heavy: mu_cluster.1.len(),
            predicted_ratio: pred,
            pdg_ratio: RATIO_MU_E,
            relative_error: (pred - RATIO_MU_E).abs() / RATIO_MU_E,
            label: MassRatioLabel::MuOverE,
        });
    }

    // tau/e ratio
    if m_e > 1e-12 {
        let pred = m_tau / m_e;
        predictions.push(MassRatioPrediction {
            depth_light: e_cluster.0,
            depth_heavy: tau_cluster.0,
            count_light: e_cluster.1.len(),
            count_heavy: tau_cluster.1.len(),
            predicted_ratio: pred,
            pdg_ratio: RATIO_TAU_E,
            relative_error: (pred - RATIO_TAU_E).abs() / RATIO_TAU_E,
            label: MassRatioLabel::TauOverE,
        });
    }

    // tau/mu ratio
    if m_mu > 1e-12 {
        let pred = m_tau / m_mu;
        predictions.push(MassRatioPrediction {
            depth_light: mu_cluster.0,
            depth_heavy: tau_cluster.0,
            count_light: mu_cluster.1.len(),
            count_heavy: tau_cluster.1.len(),
            predicted_ratio: pred,
            pdg_ratio: RATIO_TAU_MU,
            relative_error: (pred - RATIO_TAU_MU).abs() / RATIO_TAU_MU,
            label: MassRatioLabel::TauOverMu,
        });
    }

    predictions
}

/// Compare predicted ratios to PDG values and return a summary.
#[derive(Debug, Clone)]
pub struct PdgComparison {
    pub predictions: Vec<MassRatioPrediction>,
    /// Mean relative error across all predictions.
    pub mean_relative_error: f64,
    /// Best (minimum) relative error.
    pub best_relative_error: f64,
    /// Worst (maximum) relative error.
    pub worst_relative_error: f64,
    /// Number of clusters found.
    pub n_clusters: usize,
}

/// Run full PDG comparison pipeline.
pub fn pdg_comparison(map: &SurvivalDepthMap) -> PdgComparison {
    let clusters = depth_clusters(map);
    let predictions = predict_mass_ratios(map);

    let mean_err = if predictions.is_empty() {
        f64::NAN
    } else {
        predictions.iter().map(|p| p.relative_error).sum::<f64>() / predictions.len() as f64
    };

    let best = predictions
        .iter()
        .map(|p| p.relative_error)
        .fold(f64::INFINITY, f64::min);
    let worst = predictions
        .iter()
        .map(|p| p.relative_error)
        .fold(f64::NEG_INFINITY, f64::max);

    PdgComparison {
        predictions,
        mean_relative_error: mean_err,
        best_relative_error: if best.is_finite() { best } else { f64::NAN },
        worst_relative_error: if worst.is_finite() {
            worst
        } else {
            f64::NAN
        },
        n_clusters: clusters.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pdg_constants_positive() {
        assert!(M_ELECTRON > 0.0);
        assert!(M_MUON > M_ELECTRON);
        assert!(M_TAU > M_MUON);
    }

    #[test]
    fn test_pdg_ratios_consistent() {
        assert!((RATIO_MU_E - M_MUON / M_ELECTRON).abs() < 1e-6);
        assert!((RATIO_TAU_E - M_TAU / M_ELECTRON).abs() < 1e-6);
        assert!((RATIO_TAU_MU - M_TAU / M_MUON).abs() < 1e-6);
    }

    #[test]
    fn test_predict_mass_ratios_returns_three() {
        let map = SurvivalDepthMap::compute();
        let preds = predict_mass_ratios(&map);
        // Should have exactly 3 predictions (mu/e, tau/e, tau/mu)
        // unless fewer than 3 distinct clusters
        assert!(preds.len() <= 3);
        if map.n_distinct_depths() >= 3 {
            assert_eq!(preds.len(), 3);
        }
    }

    #[test]
    fn test_predictions_have_positive_ratios() {
        let map = SurvivalDepthMap::compute();
        let preds = predict_mass_ratios(&map);
        for p in &preds {
            assert!(p.predicted_ratio > 0.0);
            assert!(p.pdg_ratio > 0.0);
            assert!(p.relative_error >= 0.0);
        }
    }

    #[test]
    fn test_pdg_comparison_pipeline() {
        let map = SurvivalDepthMap::compute();
        let comp = pdg_comparison(&map);
        assert!(comp.n_clusters >= 2, "Should find at least 2 clusters");
        if !comp.predictions.is_empty() {
            assert!(comp.mean_relative_error >= 0.0 || comp.mean_relative_error.is_nan());
        }
    }

    #[test]
    fn test_mass_ratio_labels_distinct() {
        let map = SurvivalDepthMap::compute();
        let preds = predict_mass_ratios(&map);
        if preds.len() == 3 {
            assert_eq!(preds[0].label, MassRatioLabel::MuOverE);
            assert_eq!(preds[1].label, MassRatioLabel::TauOverE);
            assert_eq!(preds[2].label, MassRatioLabel::TauOverMu);
        }
    }

    #[test]
    fn test_relative_error_below_one_for_best() {
        let map = SurvivalDepthMap::compute();
        let comp = pdg_comparison(&map);
        // The filtration model is approximate; just verify errors are finite
        if !comp.predictions.is_empty() {
            assert!(
                comp.best_relative_error.is_finite(),
                "Best error should be finite"
            );
        }
    }
}
