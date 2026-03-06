//! Box-kite alignment spectrum: project sedenion elements onto box-kite subspaces.
//!
//! Each sedenion box-kite defines a subspace spanned by its 6 assessor 2-planes.
//! For a given sedenion element v, the alignment with box-kite k is the fraction
//! of ||v||^2 that lives in the assessor subspace of box-kite k.
//!
//! This is a *structural/exploratory* tool, not a physical prediction.

use crate::boxkites::{cached_sedenion_boxkites, BoxKite};
use cd_kernel::cayley_dickson::cd_norm_sq;

/// Alignment spectrum: projection weights onto each of the 7 box-kites.
#[derive(Debug, Clone)]
pub struct AlignmentSpectrum {
    /// Projection weight for each box-kite (fraction of norm^2 in that subspace).
    pub weights: Vec<f64>,
    /// Index of the dominant box-kite (largest weight).
    pub dominant_bk: usize,
    /// Total weight captured by all box-kites (should be <= 1.0).
    pub total_captured: f64,
    /// Number of box-kites.
    pub n_boxkites: usize,
}

/// Compute the alignment spectrum of a sedenion element against all 7 box-kites.
///
/// The projection onto a box-kite subspace is computed by summing the squared
/// components at the basis indices spanned by each assessor (low, high).
///
/// Note: assessor subspaces may overlap between box-kites (shared indices),
/// so `total_captured` can exceed 1.0. The `weights` are raw projections,
/// not a partition of unity.
pub fn alignment_spectrum(v: &[f64], boxkites: &[BoxKite]) -> AlignmentSpectrum {
    assert_eq!(v.len(), 16, "Input must be a 16D sedenion");

    let norm_sq = cd_norm_sq(v);
    if norm_sq < 1e-30 {
        return AlignmentSpectrum {
            weights: vec![0.0; boxkites.len()],
            dominant_bk: 0,
            total_captured: 0.0,
            n_boxkites: boxkites.len(),
        };
    }

    let mut weights = Vec::with_capacity(boxkites.len());

    for bk in boxkites {
        // Collect unique basis indices spanned by this box-kite
        let mut indices = std::collections::BTreeSet::new();
        for a in &bk.assessors {
            indices.insert(a.low);
            indices.insert(a.high);
        }

        // Sum squared components at these indices
        let proj_sq: f64 = indices.iter().map(|&idx| v[idx] * v[idx]).sum();
        weights.push(proj_sq / norm_sq);
    }

    let total_captured: f64 = weights.iter().sum();
    let dominant_bk = weights
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0);

    AlignmentSpectrum {
        weights,
        dominant_bk,
        total_captured,
        n_boxkites: boxkites.len(),
    }
}

/// Convenience: compute alignment spectrum using cached sedenion box-kites.
pub fn compute_alignment(v: &[f64]) -> AlignmentSpectrum {
    alignment_spectrum(v, cached_sedenion_boxkites())
}

/// Scan result for a single time step in the entropy trap scan.
#[derive(Debug, Clone)]
pub struct AlignmentScanPoint {
    /// Julian Ephemeris Date.
    pub jed: f64,
    /// Alignment weights for each box-kite.
    pub weights: Vec<f64>,
    /// Dominant box-kite index.
    pub dominant_bk: usize,
    /// Associator norm of the lifted sedenion pair.
    pub assoc_norm: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alignment_of_zero_vector() {
        let v = vec![0.0; 16];
        let boxkites = cached_sedenion_boxkites();
        let spec = alignment_spectrum(&v, boxkites);
        assert_eq!(spec.n_boxkites, 7);
        assert!(spec.total_captured.abs() < 1e-12);
    }

    #[test]
    fn alignment_of_pure_real() {
        // Pure real sedenion: e0 = 1, all imaginary = 0
        // Should have zero alignment with all box-kites (they span imaginary indices)
        let mut v = vec![0.0; 16];
        v[0] = 1.0;
        let boxkites = cached_sedenion_boxkites();
        let spec = alignment_spectrum(&v, boxkites);
        assert!(
            spec.total_captured < 1e-12,
            "Pure real should have zero BK alignment, got {}",
            spec.total_captured
        );
    }

    #[test]
    fn alignment_of_imaginary_unit() {
        // e1 should be captured by box-kites that contain assessors with low=1
        let mut v = vec![0.0; 16];
        v[1] = 1.0;
        let boxkites = cached_sedenion_boxkites();
        let spec = alignment_spectrum(&v, boxkites);
        // At least one box-kite should capture this
        assert!(
            spec.total_captured > 0.5,
            "e1 should be captured by at least one BK, total = {}",
            spec.total_captured
        );
    }

    #[test]
    fn seven_boxkites_found() {
        let boxkites = cached_sedenion_boxkites();
        assert_eq!(boxkites.len(), 7, "Sedenions should have exactly 7 box-kites");
    }
}
