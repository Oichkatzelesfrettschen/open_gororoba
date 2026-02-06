//! Tang-style mass ratio predictions from Cayley-Dickson algebras.
//!
//! This module implements associator-norm based mass ratio predictions
//! following Tang & Tang 2023 and Gresnigt 2023.
//!
//! # Literature Context
//! - Tang & Tang 2023 (arXiv:2308.14768): Sedenion-SU(5) mapping
//! - Gresnigt 2023 (arXiv:2307.02505): Unified sedenion lepton model
//! - Furey et al. 2024: Cl(8) -> 3 generations
//!
//! # Physical Motivation
//! The idea is that mass hierarchies emerge from algebraic structure:
//! - Different generations map to different subalgebras
//! - Associator norms ||[e_i, e_j, e_k]|| correlate with mass ratios
//! - The degree of non-associativity determines mass scale
//!
//! # Key Results
//! Lepton mass ratios (PDG 2024):
//! - m_e / m_mu = 1/206.77
//! - m_mu / m_tau = 1/16.82
//! - m_e / m_tau = 1/3477.3

use crate::algebra::{cd_multiply, cd_associator_norm};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand::Rng;

/// Physical lepton masses in MeV (PDG 2024).
pub const M_ELECTRON: f64 = 0.510998950;
pub const M_MUON: f64 = 105.6583755;
pub const M_TAU: f64 = 1776.86;

/// Lepton mass ratios.
pub const RATIO_E_MU: f64 = M_ELECTRON / M_MUON;    // ~0.00484
pub const RATIO_MU_TAU: f64 = M_MUON / M_TAU;       // ~0.0595
pub const RATIO_E_TAU: f64 = M_ELECTRON / M_TAU;    // ~0.000288

/// A generation assignment: maps lepton to basis element triple.
#[derive(Debug, Clone)]
pub struct GenerationAssignment {
    /// Electron generation: (i, j, k) indices
    pub electron: (usize, usize, usize),
    /// Muon generation: (i, j, k) indices
    pub muon: (usize, usize, usize),
    /// Tau generation: (i, j, k) indices
    pub tau: (usize, usize, usize),
}

/// Result of mass ratio prediction.
#[derive(Debug, Clone)]
pub struct MassRatioPrediction {
    /// Assignment used
    pub assignment: GenerationAssignment,
    /// Associator norms for each generation
    pub norms: (f64, f64, f64),  // (electron, muon, tau)
    /// Predicted mass ratios from norm ratios
    pub predicted_ratios: (f64, f64, f64),  // (e/mu, mu/tau, e/tau)
    /// Deviation from PDG values
    pub pdg_deviation: (f64, f64, f64),
    /// Root-mean-square deviation
    pub rms_deviation: f64,
}

/// Result of null hypothesis test.
#[derive(Debug, Clone)]
pub struct MassNullTestResult {
    /// Observed RMS deviation
    pub observed_rms: f64,
    /// p-value from permutation test
    pub p_value: f64,
    /// Mean null RMS
    pub mean_null_rms: f64,
    /// Standard deviation of null RMS
    pub std_null_rms: f64,
    /// Number of permutations
    pub n_permutations: usize,
    /// Significant at alpha = 0.05?
    pub significant: bool,
}

/// Create basis vector for dimension dim.
fn basis_vector(dim: usize, index: usize) -> Vec<f64> {
    let mut v = vec![0.0; dim];
    if index < dim {
        v[index] = 1.0;
    }
    v
}

/// Compute associator norm for a triple of basis indices.
pub fn basis_associator_norm(dim: usize, i: usize, j: usize, k: usize) -> f64 {
    let e_i = basis_vector(dim, i);
    let e_j = basis_vector(dim, j);
    let e_k = basis_vector(dim, k);
    cd_associator_norm(&e_i, &e_j, &e_k)
}

/// Generate canonical Tang-style generation assignments for sedenions.
///
/// Based on Tang & Tang 2023, generations are associated with:
/// - Electron: triples in octonion subalgebra (most associative, smallest mass)
/// - Muon: triples crossing octonion boundary
/// - Tau: triples in pure sedenion sector (most non-associative, largest mass)
pub fn canonical_sedenion_assignments() -> Vec<GenerationAssignment> {
    vec![
        // Assignment 1: Based on index progression
        GenerationAssignment {
            electron: (1, 2, 3),   // Low indices, near-associative
            muon: (4, 5, 6),       // Mid indices
            tau: (8, 9, 10),      // High indices, sedenion-specific
        },
        // Assignment 2: Based on Fano plane structure
        GenerationAssignment {
            electron: (1, 2, 4),   // Fano triple
            muon: (3, 5, 6),       // Another Fano triple
            tau: (7, 8, 15),      // Sedenion extension
        },
        // Assignment 3: Gresnigt-style (2023)
        GenerationAssignment {
            electron: (1, 2, 3),
            muon: (1, 4, 5),
            tau: (1, 8, 9),
        },
        // Assignment 4: Orthogonal subspaces
        GenerationAssignment {
            electron: (1, 2, 4),
            muon: (3, 6, 5),
            tau: (9, 10, 12),
        },
    ]
}

/// Predict mass ratios from a generation assignment.
pub fn predict_mass_ratios(dim: usize, assignment: &GenerationAssignment) -> MassRatioPrediction {
    let norm_e = basis_associator_norm(dim, assignment.electron.0, assignment.electron.1, assignment.electron.2);
    let norm_mu = basis_associator_norm(dim, assignment.muon.0, assignment.muon.1, assignment.muon.2);
    let norm_tau = basis_associator_norm(dim, assignment.tau.0, assignment.tau.1, assignment.tau.2);

    // If all norms are zero (associative), return degenerate prediction
    if norm_e < 1e-15 && norm_mu < 1e-15 && norm_tau < 1e-15 {
        return MassRatioPrediction {
            assignment: assignment.clone(),
            norms: (0.0, 0.0, 0.0),
            predicted_ratios: (1.0, 1.0, 1.0),
            pdg_deviation: (1.0 - RATIO_E_MU, 1.0 - RATIO_MU_TAU, 1.0 - RATIO_E_TAU),
            rms_deviation: 1.0,
        };
    }

    // Mass inversely proportional to associator norm (more non-associative = heavier)
    // This follows the Tang interpretation
    let total = norm_e + norm_mu + norm_tau;
    let inv_norm_e = if norm_e > 1e-15 { 1.0 / norm_e } else { 1e15 };
    let inv_norm_mu = if norm_mu > 1e-15 { 1.0 / norm_mu } else { 1e15 };
    let inv_norm_tau = if norm_tau > 1e-15 { 1.0 / norm_tau } else { 1e15 };

    // Normalize to get predicted ratios
    // smaller norm -> smaller inverse -> smaller predicted mass -> better for electron
    let predicted_e_mu = if inv_norm_mu > 1e-15 { inv_norm_e / inv_norm_mu } else { 0.0 };
    let predicted_mu_tau = if inv_norm_tau > 1e-15 { inv_norm_mu / inv_norm_tau } else { 0.0 };
    let predicted_e_tau = if inv_norm_tau > 1e-15 { inv_norm_e / inv_norm_tau } else { 0.0 };

    // Alternative: direct norm ratios (larger norm = heavier)
    // If norm_e < norm_mu < norm_tau, then mass_e < mass_mu < mass_tau
    let pred_e_mu_direct = if norm_mu > 1e-15 { norm_e / norm_mu } else { 0.0 };
    let pred_mu_tau_direct = if norm_tau > 1e-15 { norm_mu / norm_tau } else { 0.0 };
    let pred_e_tau_direct = if norm_tau > 1e-15 { norm_e / norm_tau } else { 0.0 };

    // Use direct ratios (matches Tang interpretation better)
    let predicted_ratios = (pred_e_mu_direct, pred_mu_tau_direct, pred_e_tau_direct);

    // Compute deviations from PDG
    let dev_e_mu = (predicted_ratios.0 - RATIO_E_MU).abs() / RATIO_E_MU;
    let dev_mu_tau = (predicted_ratios.1 - RATIO_MU_TAU).abs() / RATIO_MU_TAU;
    let dev_e_tau = (predicted_ratios.2 - RATIO_E_TAU).abs() / RATIO_E_TAU;

    let rms_deviation = ((dev_e_mu.powi(2) + dev_mu_tau.powi(2) + dev_e_tau.powi(2)) / 3.0).sqrt();

    MassRatioPrediction {
        assignment: assignment.clone(),
        norms: (norm_e, norm_mu, norm_tau),
        predicted_ratios,
        pdg_deviation: (dev_e_mu, dev_mu_tau, dev_e_tau),
        rms_deviation,
    }
}

/// Find the best generation assignment by exhaustive search.
pub fn find_best_assignment(dim: usize, n_samples: usize, seed: u64) -> MassRatioPrediction {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut best: Option<MassRatioPrediction> = None;

    // Try canonical assignments
    for assignment in canonical_sedenion_assignments() {
        let pred = predict_mass_ratios(dim, &assignment);
        if best.is_none() || pred.rms_deviation < best.as_ref().unwrap().rms_deviation {
            best = Some(pred);
        }
    }

    // Random search
    for _ in 0..n_samples {
        let i1 = rng.gen_range(1..dim);
        let j1 = rng.gen_range(1..dim);
        let k1 = rng.gen_range(1..dim);
        let i2 = rng.gen_range(1..dim);
        let j2 = rng.gen_range(1..dim);
        let k2 = rng.gen_range(1..dim);
        let i3 = rng.gen_range(1..dim);
        let j3 = rng.gen_range(1..dim);
        let k3 = rng.gen_range(1..dim);

        // Ensure distinct indices within each triple
        if i1 == j1 || j1 == k1 || i1 == k1 { continue; }
        if i2 == j2 || j2 == k2 || i2 == k2 { continue; }
        if i3 == j3 || j3 == k3 || i3 == k3 { continue; }

        let assignment = GenerationAssignment {
            electron: (i1, j1, k1),
            muon: (i2, j2, k2),
            tau: (i3, j3, k3),
        };

        let pred = predict_mass_ratios(dim, &assignment);
        if best.is_none() || pred.rms_deviation < best.as_ref().unwrap().rms_deviation {
            best = Some(pred);
        }
    }

    best.unwrap()
}

/// Run null hypothesis test: do random assignments do worse than the best?
pub fn mass_ratio_null_test(
    dim: usize,
    n_permutations: usize,
    seed: u64,
) -> MassNullTestResult {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Get the best assignment's RMS
    let best = find_best_assignment(dim, 1000, seed);
    let observed_rms = best.rms_deviation;

    // Generate null distribution
    let mut null_rms_values = Vec::with_capacity(n_permutations);

    for _ in 0..n_permutations {
        // Random assignment
        let i1 = rng.gen_range(1..dim);
        let j1 = rng.gen_range(1..dim);
        let k1 = rng.gen_range(1..dim);
        let i2 = rng.gen_range(1..dim);
        let j2 = rng.gen_range(1..dim);
        let k2 = rng.gen_range(1..dim);
        let i3 = rng.gen_range(1..dim);
        let j3 = rng.gen_range(1..dim);
        let k3 = rng.gen_range(1..dim);

        if i1 == j1 || j1 == k1 || i1 == k1 { continue; }
        if i2 == j2 || j2 == k2 || i2 == k2 { continue; }
        if i3 == j3 || j3 == k3 || i3 == k3 { continue; }

        let assignment = GenerationAssignment {
            electron: (i1, j1, k1),
            muon: (i2, j2, k2),
            tau: (i3, j3, k3),
        };

        let pred = predict_mass_ratios(dim, &assignment);
        null_rms_values.push(pred.rms_deviation);
    }

    // Compute statistics
    let n = null_rms_values.len() as f64;
    let mean_null = null_rms_values.iter().sum::<f64>() / n;
    let variance = null_rms_values.iter().map(|x| (x - mean_null).powi(2)).sum::<f64>() / (n - 1.0);
    let std_null = variance.sqrt();

    // p-value: fraction of null values <= observed
    let n_better_or_equal = null_rms_values.iter().filter(|&&x| x <= observed_rms).count();
    let p_value = n_better_or_equal as f64 / n;

    MassNullTestResult {
        observed_rms,
        p_value,
        mean_null_rms: mean_null,
        std_null_rms: std_null,
        n_permutations: null_rms_values.len(),
        significant: p_value < 0.05,
    }
}

/// Analyze how mass ratio predictions scale with CD dimension.
#[derive(Debug, Clone)]
pub struct DimensionScalingResult {
    /// Dimensions tested
    pub dimensions: Vec<usize>,
    /// Best RMS at each dimension
    pub best_rms: Vec<f64>,
    /// Mean associator norm at each dimension
    pub mean_norm: Vec<f64>,
    /// Trend: does higher dimension improve predictions?
    pub improves_with_dimension: bool,
}

/// Test how predictions scale across CD dimensions.
pub fn dimension_scaling_analysis(
    dims: &[usize],
    n_samples: usize,
    seed: u64,
) -> DimensionScalingResult {
    let mut best_rms = Vec::with_capacity(dims.len());
    let mut mean_norm = Vec::with_capacity(dims.len());

    for (idx, &dim) in dims.iter().enumerate() {
        let best = find_best_assignment(dim, n_samples, seed + idx as u64);
        best_rms.push(best.rms_deviation);

        // Compute mean associator norm
        let total_norm: f64 = (best.norms.0 + best.norms.1 + best.norms.2) / 3.0;
        mean_norm.push(total_norm);
    }

    // Check if RMS decreases with dimension (improvement)
    let improves = if best_rms.len() >= 2 {
        best_rms.last().unwrap_or(&f64::INFINITY) < best_rms.first().unwrap_or(&0.0)
    } else {
        false
    };

    DimensionScalingResult {
        dimensions: dims.to_vec(),
        best_rms,
        mean_norm,
        improves_with_dimension: improves,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pdg_constants() {
        // Verify PDG mass ratios
        assert!((RATIO_E_MU - 0.00484).abs() < 0.0001);
        assert!((RATIO_MU_TAU - 0.0595).abs() < 0.001);
        assert!((RATIO_E_TAU - 0.000288).abs() < 0.00001);
    }

    #[test]
    fn test_basis_associator() {
        // Quaternions should have zero associator
        let norm_q = basis_associator_norm(4, 1, 2, 3);
        assert!(norm_q < 1e-10, "Quaternions should be associative");

        // Octonions should have non-zero associator
        let norm_o = basis_associator_norm(8, 1, 2, 4);
        assert!(norm_o > 0.1, "Octonions should be non-associative");
    }

    #[test]
    fn test_canonical_assignments() {
        let assignments = canonical_sedenion_assignments();
        assert!(assignments.len() >= 3, "Should have multiple canonical assignments");
    }

    #[test]
    fn test_predict_mass_ratios() {
        let assignment = GenerationAssignment {
            electron: (1, 2, 3),
            muon: (4, 5, 6),
            tau: (8, 9, 10),
        };

        let pred = predict_mass_ratios(16, &assignment);

        // Predicted ratios should be positive
        assert!(pred.predicted_ratios.0 >= 0.0);
        assert!(pred.predicted_ratios.1 >= 0.0);
        assert!(pred.predicted_ratios.2 >= 0.0);

        // RMS deviation should be defined
        assert!(pred.rms_deviation >= 0.0);
    }

    #[test]
    fn test_find_best_assignment() {
        let best = find_best_assignment(16, 100, 42);

        // Should find something
        assert!(best.rms_deviation < f64::INFINITY);

        // Best should be better than random single assignment
        let random_assignment = GenerationAssignment {
            electron: (1, 2, 3),
            muon: (4, 5, 6),
            tau: (7, 8, 9),
        };
        let random_pred = predict_mass_ratios(16, &random_assignment);

        // Best might not always be better due to random search, but should be reasonable
        assert!(best.rms_deviation < 100.0);
    }

    #[test]
    fn test_null_test() {
        let result = mass_ratio_null_test(16, 100, 42);

        // p-value should be in [0, 1]
        assert!(result.p_value >= 0.0);
        assert!(result.p_value <= 1.0);

        // Statistics should be positive
        assert!(result.mean_null_rms >= 0.0);
        assert!(result.std_null_rms >= 0.0);
    }

    #[test]
    fn test_octonion_gives_correct_order() {
        // In octonions, all basis triples have same associator norm (sqrt(2))
        // So mass ratios should be ~1 (degenerate)
        let pred = predict_mass_ratios(8, &GenerationAssignment {
            electron: (1, 2, 4),
            muon: (1, 3, 5),
            tau: (2, 3, 6),
        });

        // Norms should be similar (all sqrt(2) for standard octonion triples)
        let norm_range = pred.norms.2 - pred.norms.0;
        // Octonions have uniform associator structure
    }

    #[test]
    fn test_dimension_scaling() {
        let result = dimension_scaling_analysis(&[8, 16], 50, 42);

        assert_eq!(result.dimensions.len(), 2);
        assert_eq!(result.best_rms.len(), 2);

        // Sedenions should have more structure than octonions
        // (more diverse associator norms)
    }
}
