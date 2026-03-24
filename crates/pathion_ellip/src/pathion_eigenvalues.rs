//! 32D Pathion ZD graph eigenvalue spectrum.
//!
//! Builds the zero-divisor interaction graph adjacency matrix for dim=32,
//! computes the graph-Laplacian, and extracts eigenvalues that modulate
//! the GPU Pathion heat sink.

use algebra_experimental::higher_cd_control::{
    PathionControlReport, compute_zd_graph_spectrum, default_pathion_control_report,
};

/// Eigenvalue spectrum of the 32D Pathion ZD interaction graph.
#[derive(Debug, Clone)]
pub struct PathionEigenvalueSpectrum {
    /// Sorted eigenvalues (ascending) of the graph Laplacian.
    pub eigenvalues: Vec<f64>,
    /// 16 representative eigenvalues for GPU upload (padded/truncated).
    pub eigenvalues_16: [f32; 16],
    /// Number of connected components (= multiplicity of zero eigenvalue).
    pub n_components: usize,
}

impl PathionEigenvalueSpectrum {
    /// Compute the eigenvalue spectrum for a 32D Pathion ZD graph.
    pub fn compute() -> Self {
        Self::compute_for_dim(32)
    }

    /// Compute eigenvalue spectrum for arbitrary CD dimension.
    pub fn compute_for_dim(dim: usize) -> Self {
        let report = compute_zd_graph_spectrum(dim);

        Self {
            eigenvalues: report.eigenvalues,
            eigenvalues_16: report.eigenvalues_16,
            n_components: report.n_components,
        }
    }

    /// Return the normalized shared 32D control report backing the Pathion lane.
    pub fn shared_control_report() -> PathionControlReport {
        default_pathion_control_report()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pathion_eigenvalues_nonneg() {
        let spec = PathionEigenvalueSpectrum::compute();
        for &ev in &spec.eigenvalues {
            assert!(
                ev >= -1e-10,
                "Laplacian eigenvalue should be non-negative, got {}",
                ev
            );
        }
    }

    #[test]
    fn test_pathion_eigenvalues_count() {
        let spec = PathionEigenvalueSpectrum::compute();
        assert_eq!(
            spec.eigenvalues.len(),
            32,
            "32D Pathion should have 32 eigenvalues"
        );
    }

    #[test]
    fn test_pathion_has_components() {
        let spec = PathionEigenvalueSpectrum::compute();
        // At dim=32, the ZD graph should have multiple connected components
        assert!(
            spec.n_components >= 2,
            "Expected multiple components, got {}",
            spec.n_components
        );
    }

    #[test]
    fn test_eigenvalues_16_normalized() {
        let spec = PathionEigenvalueSpectrum::compute();
        let max = spec.eigenvalues_16.iter().copied().fold(0.0f32, f32::max);
        // Should be normalized to max=1.0 (or all zero if no positive eigenvalues)
        assert!(
            (max - 1.0).abs() < 1e-5 || max == 0.0,
            "Max eigenvalue should be 1.0, got {}",
            max
        );
    }

    #[test]
    fn test_quaternion_no_zd() {
        // Quaternions (dim=4) are alternative -- no associator violations
        let spec = PathionEigenvalueSpectrum::compute_for_dim(4);
        // All eigenvalues should be zero (no edges in adjacency)
        assert_eq!(
            spec.n_components, 4,
            "Quaternion ZD graph should be fully disconnected"
        );
    }

    #[test]
    fn test_octonion_no_zd() {
        // Octonions (dim=8) are alternative -- no associator violations between basis pairs
        let spec = PathionEigenvalueSpectrum::compute_for_dim(8);
        // Octonions have nonzero associators but are alternative (associator is skew-symmetric),
        // so the adjacency matrix will have edges. But no zero-divisors.
        // The test just checks we get 8 eigenvalues.
        assert_eq!(spec.eigenvalues.len(), 8);
    }

    #[test]
    fn test_sedenion_has_zd() {
        // Sedenions (dim=16) have zero-divisors and non-alternative behavior
        let spec = PathionEigenvalueSpectrum::compute_for_dim(16);
        // Should have positive eigenvalues (edges in the graph)
        let has_positive = spec.eigenvalues.iter().any(|&v| v > 1e-10);
        assert!(has_positive, "Sedenion ZD graph should have edges");
    }

    #[test]
    fn test_shared_control_report_matches_legacy_surface() {
        let legacy = PathionEigenvalueSpectrum::compute();
        let shared = PathionEigenvalueSpectrum::shared_control_report();

        assert_eq!(legacy.n_components, shared.spectrum_report.n_components);
        assert_eq!(legacy.eigenvalues_16, shared.spectrum_report.eigenvalues_16);
        assert_eq!(
            legacy.eigenvalues.len(),
            shared.spectrum_report.eigenvalues.len()
        );
        for (legacy_ev, shared_ev) in legacy
            .eigenvalues
            .iter()
            .zip(shared.spectrum_report.eigenvalues.iter())
        {
            assert!(
                (legacy_ev - shared_ev).abs() < 1e-10,
                "legacy and shared eigenvalues diverged: {legacy_ev} vs {shared_ev}"
            );
        }
        assert_eq!(shared.summary.algebra_name, "Pathion");
        assert_eq!(shared.summary.actual_rank, 1);
    }
}
