//! Bridge from LBM fluid lattice zero-divisor census to StabilizerLikeCode syndrome detection.
//!
//! The fluid lattice evolves macroscopic fields (density, velocity) on a 3D grid.
//! Each lattice site maps to a Cayley-Dickson basis index, and certain multi-site
//! configurations form zero-divisor (ZD) patterns. When the local fluid state
//! deviates from equilibrium, the corresponding algebraic axes become "active"
//! -- analogous to qubit errors in standard QEC.
//!
//! This module provides:
//! - `LatticeSiteState`: a snapshot of one lattice site's algebraic deviation
//! - `VoidCodeAdapter`: wraps a set of active ZD axes as a StabilizerLikeCode
//! - `LatticeQecSnapshot`: composite code over an entire lattice region
//!
//! See BIB-0307 (Moreno 1998) for CD zero-divisor structure and
//! BIB-0314 (Pastawski et al. 2015) for holographic code analogies.

use crate::stabilizer_like::{CompositeCode, StabilizerLikeCode};
use algebra_experimental::higher_cd::SparseApeironState;
use std::collections::HashSet;

/// A snapshot of one lattice site's algebraic state, expressed as deviation
/// from equilibrium in the CD basis.
///
/// When `deviation_norm` exceeds a threshold, the site is considered "active"
/// and its nonzero axes contribute to the syndrome error mask.
#[derive(Debug, Clone)]
pub struct LatticeSiteState {
    /// Lattice site index (flattened from 3D grid).
    pub site_index: usize,
    /// Algebraic state as sparse CD components.
    pub state: SparseApeironState,
    /// L2 deviation from equilibrium (rho_eq = 1.0 in LBM units).
    pub deviation_norm: f64,
}

impl LatticeSiteState {
    /// Create from a sparse algebraic state and its equilibrium deviation.
    pub fn new(site_index: usize, state: SparseApeironState, deviation_norm: f64) -> Self {
        Self {
            site_index,
            state,
            deviation_norm,
        }
    }

    /// Extract active axis indices where component magnitude exceeds threshold.
    ///
    /// These axes represent the "error pattern" visible to the stabilizer code.
    pub fn active_axes(&self, threshold: f64) -> Vec<usize> {
        self.state
            .components
            .iter()
            .filter(|(_, val)| val.abs() > threshold)
            .map(|(axis, _)| *axis)
            .collect()
    }
}

/// A stabilizer-like code element built from zero-divisor census data.
///
/// Wraps a set of axis indices (from ZD pair/triple enumeration) and detects
/// syndromes by checking overlap with active error axes from the lattice.
/// The syndrome strength is continuous: it measures the fraction of covered
/// ZD axes that are currently active (deviating from equilibrium).
#[derive(Debug, Clone)]
pub struct VoidCodeAdapter {
    /// Axis indices covered by this ZD stabilizer element.
    covered: Vec<usize>,
    /// Algebra dimension (16 for Sedenion, 1024 for DekaVoudon, etc.).
    dim: usize,
    /// Code distance (minimum weight of undetectable error).
    distance: usize,
}

impl VoidCodeAdapter {
    /// Create a new adapter from a set of ZD-related axis indices.
    ///
    /// `axes` are the CD basis indices participating in this zero-divisor
    /// configuration. `dim` is the parent algebra dimension. `distance`
    /// is the minimum number of axis errors needed to evade detection.
    pub fn new(axes: Vec<usize>, dim: usize, distance: usize) -> Self {
        Self {
            covered: axes,
            dim,
            distance,
        }
    }

    /// Build from an AVT violation tuple (i, j, k, m, sign).
    ///
    /// The four axes (i, j, k, m) involved in a single alternativity
    /// violation form a natural 4-element stabilizer check.
    pub fn from_violation(i: usize, j: usize, k: usize, m: usize, dim: usize) -> Self {
        let mut axes = vec![i, j, k, m];
        axes.sort_unstable();
        axes.dedup();
        // Distance is 2: flipping any 2 of 4 axes can mask the syndrome
        Self::new(axes, dim, 2)
    }
}

impl StabilizerLikeCode for VoidCodeAdapter {
    fn covered_indices(&self) -> &[usize] {
        &self.covered
    }

    fn syndrome_strength(&self, error_mask: &HashSet<usize>) -> f64 {
        if self.covered.is_empty() {
            return 0.0;
        }
        let hits = self
            .covered
            .iter()
            .filter(|idx| error_mask.contains(idx))
            .count();
        hits as f64 / self.covered.len() as f64
    }

    fn code_distance(&self) -> usize {
        self.distance
    }

    fn num_checks(&self) -> usize {
        // Each pair of covered axes is a parity check
        let n = self.covered.len();
        if n < 2 { 0 } else { n * (n - 1) / 2 }
    }

    fn algebra_dimension(&self) -> usize {
        self.dim
    }
}

/// Composite code over a lattice region's zero-divisor census.
///
/// Aggregates multiple `VoidCodeAdapter` elements (one per ZD violation)
/// into a single code that can report global stability and effective distance.
#[derive(Debug, Clone)]
pub struct LatticeQecSnapshot {
    /// Individual stabilizer elements from ZD census.
    pub stabilizers: Vec<VoidCodeAdapter>,
    /// Algebra dimension for all stabilizers.
    pub dim: usize,
}

impl LatticeQecSnapshot {
    /// Build from a list of AVT violations (i, j, k, m, sign).
    ///
    /// Each violation becomes one `VoidCodeAdapter` stabilizer element.
    pub fn from_violations(violations: &[(usize, usize, usize, usize, i32)], dim: usize) -> Self {
        let stabilizers = violations
            .iter()
            .map(|&(i, j, k, m, _sign)| VoidCodeAdapter::from_violation(i, j, k, m, dim))
            .collect();
        Self { stabilizers, dim }
    }

    /// Detect which stabilizers are triggered by the current lattice state.
    ///
    /// Returns (num_triggered, total_stabilizers, mean_syndrome_strength).
    pub fn detect_active_syndromes(&self, error_mask: &HashSet<usize>) -> (usize, usize, f64) {
        let total = self.stabilizers.len();
        if total == 0 {
            return (0, 0, 0.0);
        }
        let mut triggered = 0;
        let mut strength_sum = 0.0;
        for stab in &self.stabilizers {
            let s = stab.syndrome_strength(error_mask);
            if s > 0.0 {
                triggered += 1;
            }
            strength_sum += s;
        }
        (triggered, total, strength_sum / total as f64)
    }
}

impl CompositeCode for LatticeQecSnapshot {
    fn num_stabilizers(&self) -> usize {
        self.stabilizers.len()
    }

    fn global_stability(&self, error_mask: &HashSet<usize>) -> f64 {
        if self.stabilizers.is_empty() {
            return 1.0;
        }
        let triggered = self
            .stabilizers
            .iter()
            .filter(|s| s.syndrome_strength(error_mask) > 0.0)
            .count();
        1.0 - (triggered as f64 / self.stabilizers.len() as f64)
    }

    fn effective_distance(&self) -> usize {
        self.stabilizers
            .iter()
            .map(|s| s.code_distance())
            .min()
            .unwrap_or(0)
    }
}

/// Build an error mask from a collection of lattice site states.
///
/// Scans all sites, collects axes where deviation exceeds `threshold`,
/// and returns the union as a `HashSet` suitable for syndrome detection.
pub fn build_error_mask(sites: &[LatticeSiteState], threshold: f64) -> HashSet<usize> {
    let mut mask = HashSet::new();
    for site in sites {
        if site.deviation_norm > threshold {
            for axis in site.active_axes(threshold) {
                mask.insert(axis);
            }
        }
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sparse(dim: usize, pairs: &[(usize, f64)]) -> SparseApeironState {
        SparseApeironState {
            dim,
            components: pairs.to_vec(),
        }
    }

    #[test]
    fn test_void_code_no_errors() {
        let adapter = VoidCodeAdapter::new(vec![0, 3, 7, 12], 16, 2);
        let empty = HashSet::new();
        assert_eq!(adapter.syndrome_strength(&empty), 0.0);
        assert_eq!(adapter.code_distance(), 2);
        assert_eq!(adapter.algebra_dimension(), 16);
        assert_eq!(adapter.num_checks(), 6); // C(4,2) = 6
    }

    #[test]
    fn test_void_code_partial_syndrome() {
        let adapter = VoidCodeAdapter::new(vec![0, 3, 7, 12], 16, 2);
        let errors: HashSet<usize> = [3, 12].into_iter().collect();
        let strength = adapter.syndrome_strength(&errors);
        assert!(
            (strength - 0.5).abs() < 1e-10,
            "Expected 0.5, got {strength}"
        );
    }

    #[test]
    fn test_void_code_from_violation() {
        let adapter = VoidCodeAdapter::from_violation(2, 5, 5, 11, 16);
        // axes: [2, 5, 11] after dedup
        assert_eq!(adapter.covered_indices().len(), 3);
        assert_eq!(adapter.algebra_dimension(), 16);
    }

    #[test]
    fn test_lattice_site_active_axes() {
        let state = make_sparse(16, &[(0, 0.9), (3, 0.05), (7, 0.8), (15, 0.01)]);
        let site = LatticeSiteState::new(42, state, 1.5);
        let active = site.active_axes(0.1);
        assert_eq!(active, vec![0, 7]); // only axes with |val| > 0.1
    }

    #[test]
    fn test_build_error_mask() {
        let sites = vec![
            LatticeSiteState::new(0, make_sparse(16, &[(1, 0.5), (4, 0.3)]), 0.8),
            LatticeSiteState::new(
                1,
                make_sparse(16, &[(2, 0.9)]),
                0.05, // below threshold, should be skipped
            ),
            LatticeSiteState::new(2, make_sparse(16, &[(4, 0.7), (9, 0.6)]), 1.2),
        ];
        let mask = build_error_mask(&sites, 0.1);
        // Site 0: deviation 0.8 > 0.1, axes [1, 4] both > 0.1
        // Site 1: deviation 0.05 < 0.1, skipped
        // Site 2: deviation 1.2 > 0.1, axes [4, 9] both > 0.1
        assert!(mask.contains(&1));
        assert!(mask.contains(&4));
        assert!(mask.contains(&9));
        assert!(!mask.contains(&2)); // site 1 was skipped
        assert_eq!(mask.len(), 3); // {1, 4, 9}
    }

    #[test]
    fn test_snapshot_composite_code() {
        let violations = vec![(0, 3, 7, 12, 1), (1, 5, 9, 14, -1), (2, 6, 10, 13, 1)];
        let snapshot = LatticeQecSnapshot::from_violations(&violations, 16);
        assert_eq!(snapshot.num_stabilizers(), 3);
        assert_eq!(snapshot.effective_distance(), 2);

        // No errors: full stability
        let empty = HashSet::new();
        assert!((snapshot.global_stability(&empty) - 1.0).abs() < 1e-10);

        // Trigger one stabilizer (axis 3 hits first violation)
        let errors: HashSet<usize> = [3].into_iter().collect();
        let stability = snapshot.global_stability(&errors);
        // 1 of 3 triggered -> stability = 2/3
        assert!(
            (stability - 2.0 / 3.0).abs() < 1e-10,
            "Expected 2/3, got {stability}"
        );
    }

    #[test]
    fn test_detect_active_syndromes() {
        let violations = vec![(0, 3, 7, 12, 1), (1, 5, 9, 14, -1)];
        let snapshot = LatticeQecSnapshot::from_violations(&violations, 16);

        // Trigger axis 3 (first violation) and axis 5 (second violation)
        let errors: HashSet<usize> = [3, 5].into_iter().collect();
        let (triggered, total, mean_strength) = snapshot.detect_active_syndromes(&errors);
        assert_eq!(triggered, 2);
        assert_eq!(total, 2);
        // Each violation has 4 axes, 1 hit each -> strength = 0.25 each
        assert!(
            (mean_strength - 0.25).abs() < 1e-10,
            "Expected 0.25, got {mean_strength}"
        );
    }
}
