//! Transfinite/Infinitesimal Spectral Clustering and Asymptotic Search
//!
//! Replaces standard floating-point matching with surreal-number-inspired valuations.
//! Isolates features by Archimedean class (birthday valuation), so infinite or
//! infinitesimal differences dominate structural matching before finite differences.
//!
//! # Evidentiary classification: C (exploratory conjecture)
//!
//! Speculative ML application.  The `SurrealVal` type is a simplified valuation field,
//! not a full surreal number system.  The clustering logic is a prototype -- no claim
//! is made about convergence, optimality, or generalization to real datasets.
//!
//! # WHY Archimedean-class-aware clustering
//!
//! Standard k-means collapses across scale boundaries because Euclidean distance treats
//! all coordinates equally.  If the data has a known multi-scale structure (e.g., particle
//! masses spanning many orders of magnitude), forcing cluster centroids to respect
//! Archimedean class prevents spurious merging of lepton and quark sectors.  This module
//! tests whether that hypothesis holds on synthetic multi-scale data.
//!
//! # See also
//!
//! - `surreal_algebra::surreal_cd` -- the surreal number arithmetic this module mimics
//! - `crate::novel_algorithms::defect_sensing` -- related anomaly-detection approach

/// Represents a value in a simplified surreal-esque valuation field.
/// Consists of a real magnitude and an Archimedean class (omega power).
/// x = magnitude * omega^(valuation)
#[derive(Debug, Clone, Copy)]
pub struct SurrealVal {
    pub magnitude: f64,
    /// Valuation: 0 is finite (Real), >0 is infinite, <0 is infinitesimal.
    pub valuation: i32,
}

impl SurrealVal {
    pub fn new(magnitude: f64, valuation: i32) -> Self {
        Self {
            magnitude,
            valuation,
        }
    }

    /// True if this value is strictly infinitesimal compared to the other.
    pub fn is_infinitesimal_to(&self, other: &Self) -> bool {
        self.valuation < other.valuation
    }
}

impl PartialEq for SurrealVal {
    fn eq(&self, other: &Self) -> bool {
        self.valuation == other.valuation && (self.magnitude - other.magnitude).abs() < 1e-12
    }
}

/// **1. Transfinite/Infinitesimal Spectral Clustering**
/// Clusters data points by matching purely infinitesimal signatures, ignoring macroscopic
/// differences that would dominate standard L2 norms.
pub fn cluster_by_archimedean_class(data: &[Vec<SurrealVal>]) -> Vec<usize> {
    let mut labels = vec![0; data.len()];
    let mut current_label = 1;

    for i in 0..data.len() {
        if labels[i] != 0 {
            continue;
        }
        labels[i] = current_label;

        for j in (i + 1)..data.len() {
            if is_infinitesimal_match(&data[i], &data[j]) {
                labels[j] = current_label;
            }
        }
        current_label += 1;
    }
    labels
}

/// Two vectors match if their leading non-zero terms share the exact same valuation profile,
/// ignoring the actual finite magnitude.
fn is_infinitesimal_match(a: &[SurrealVal], b: &[SurrealVal]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for (va, vb) in a.iter().zip(b.iter()) {
        if va.valuation != vb.valuation {
            return false;
        }
    }
    true
}

/// **2. Asymptotic Box-Kite Amplitude Search**
/// Evaluates PDE boundaries or amplitude constraints at infinity (valuation > 0)
/// or the infinitesimal (valuation < 0).
pub fn asymptotic_solver(state: &[SurrealVal], target_valuation: i32) -> Vec<SurrealVal> {
    // Filters the state space to isolate only the strata matching the target asymptotic boundary.
    state
        .iter()
        .filter(|v| v.valuation == target_valuation)
        .copied()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_surreal_clustering() {
        let p1 = vec![SurrealVal::new(1.0, 0), SurrealVal::new(5.0, -1)]; // finite, inf
        let p2 = vec![SurrealVal::new(2.0, 0), SurrealVal::new(9.0, -1)]; // finite, inf -> MATCHES p1
        let p3 = vec![SurrealVal::new(1.0, 1), SurrealVal::new(5.0, -1)]; // infinite, inf -> DIFFERS

        let data = vec![p1, p2, p3];
        let labels = cluster_by_archimedean_class(&data);

        assert_eq!(labels[0], labels[1]); // p1 and p2 are in the same Archimedean class
        assert_ne!(labels[0], labels[2]); // p3 is in a different class
    }
}
