//! Pathion resonance band detection for N-body orbital dynamics.
//!
//! This client lane now consumes the normalized higher-CD resonance helpers from
//! `algebra_experimental::higher_cd_control` while preserving the familiar
//! Pathion-facing API.

use crate::pathion_eigenvalues::PathionEigenvalueSpectrum;
pub use algebra_experimental::higher_cd_control::{
    PathionResonanceReport, ZdResonanceBand as ResonanceBand, ZdResonanceConfig as ResonanceConfig,
};
use algebra_experimental::higher_cd_control::{
    compute_resonance_bands_from_eigenvalues, compute_resonance_report_from_control_report,
    default_pathion_resonance_report, resonance_modulated_perturbation_from_eigenvalues,
    total_resonance_coupling_from_bands,
};

/// Compute resonance bands between Pathion eigenvalues and an orbital frequency.
///
/// Returns a list of resonance bands sorted by coupling strength (strongest first).
pub fn compute_resonance_bands(
    spectrum: &PathionEigenvalueSpectrum,
    orbital_freq: f64,
    config: &ResonanceConfig,
) -> Vec<ResonanceBand> {
    compute_resonance_bands_from_eigenvalues(&spectrum.eigenvalues, orbital_freq, config)
}

/// Compute total resonance coupling for a given orbital frequency.
///
/// This is the sum of all individual band coupling strengths, which
/// modulates the overall Pathion perturbation amplitude.
pub fn total_resonance_coupling(
    spectrum: &PathionEigenvalueSpectrum,
    orbital_freq: f64,
    config: &ResonanceConfig,
) -> f64 {
    let bands = compute_resonance_bands(spectrum, orbital_freq, config);
    total_resonance_coupling_from_bands(&bands)
}

/// Compute the resonance-modulated perturbation matrix for N-body integration.
///
/// Returns a 3x3 diagonal matrix (as [f64; 3]) with the coupling-modulated
/// Pathion variance scaling factors for x, y, z directions.
/// The anisotropy comes from the ZD graph eigenvalue distribution.
pub fn resonance_modulated_perturbation(
    spectrum: &PathionEigenvalueSpectrum,
    orbital_freq: f64,
    alpha_pathion: f64,
    config: &ResonanceConfig,
) -> [f64; 3] {
    resonance_modulated_perturbation_from_eigenvalues(
        &spectrum.eigenvalues,
        orbital_freq,
        alpha_pathion,
        config,
    )
}

pub fn compute_resonance_report(
    spectrum: &PathionEigenvalueSpectrum,
    orbital_freq: f64,
    alpha_pathion: f64,
    config: &ResonanceConfig,
) -> PathionResonanceReport {
    let control = PathionEigenvalueSpectrum::shared_control_report();
    let uses_shared_pathion_basis = spectrum.eigenvalues == control.spectrum_report.eigenvalues;
    if uses_shared_pathion_basis {
        compute_resonance_report_from_control_report(&control, orbital_freq, alpha_pathion, config)
    } else {
        let bands = compute_resonance_bands(spectrum, orbital_freq, config);
        let total_coupling = total_resonance_coupling_from_bands(&bands);
        let perturbation =
            resonance_modulated_perturbation(spectrum, orbital_freq, alpha_pathion, config);
        PathionResonanceReport {
            algebra_name: "Pathion".to_string(),
            ambient_dim: spectrum.eigenvalues.len(),
            orbital_frequency: orbital_freq,
            alpha_scale: alpha_pathion,
            total_coupling,
            perturbation,
            bands,
        }
    }
}

pub fn default_resonance_report(
    orbital_freq: f64,
    alpha_pathion: f64,
    config: &ResonanceConfig,
) -> PathionResonanceReport {
    default_pathion_resonance_report(orbital_freq, alpha_pathion, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_spectrum() -> PathionEigenvalueSpectrum {
        PathionEigenvalueSpectrum::compute()
    }

    #[test]
    fn resonance_bands_nonempty() {
        let spec = test_spectrum();
        // Use a frequency near one of the eigenvalues
        let positive_evs: Vec<f64> = spec
            .eigenvalues
            .iter()
            .copied()
            .filter(|&v| v > 1e-10)
            .collect();

        if positive_evs.is_empty() {
            return; // Degenerate spectrum, skip
        }

        let target_freq = positive_evs[0] / (2.0 * std::f64::consts::PI);
        let config = ResonanceConfig::default();
        let bands = compute_resonance_bands(&spec, target_freq, &config);

        assert!(
            !bands.is_empty(),
            "Should find at least one resonance band near eigenvalue frequency"
        );
    }

    #[test]
    fn resonance_bands_sorted_by_coupling() {
        let spec = test_spectrum();
        let config = ResonanceConfig {
            max_harmonic: 3,
            width: 0.5,
            min_coupling: 0.0,
        };
        let bands = compute_resonance_bands(&spec, 1.0, &config);

        for w in bands.windows(2) {
            assert!(
                w[0].coupling_strength >= w[1].coupling_strength,
                "Bands should be sorted by coupling strength"
            );
        }
    }

    #[test]
    fn zero_frequency_returns_empty() {
        let spec = test_spectrum();
        let config = ResonanceConfig::default();
        let bands = compute_resonance_bands(&spec, 0.0, &config);
        assert!(bands.is_empty());
    }

    #[test]
    fn total_coupling_nonnegative() {
        let spec = test_spectrum();
        let config = ResonanceConfig::default();
        let total = total_resonance_coupling(&spec, 1.0, &config);
        assert!(total >= 0.0, "Total coupling should be non-negative");
    }

    #[test]
    fn resonance_modulated_perturbation_scales_with_alpha() {
        let spec = test_spectrum();
        let config = ResonanceConfig::default();

        let p1 = resonance_modulated_perturbation(&spec, 1.0, 1e-6, &config);
        let p2 = resonance_modulated_perturbation(&spec, 1.0, 2e-6, &config);

        // Doubling alpha should double the perturbation
        for i in 0..3 {
            if p1[i].abs() > 1e-20 {
                let ratio = p2[i] / p1[i];
                assert!(
                    (ratio - 2.0).abs() < 1e-10,
                    "Perturbation should scale linearly with alpha: ratio = {}",
                    ratio
                );
            }
        }
    }

    #[test]
    fn shared_resonance_report_matches_pathion_defaults() {
        let config = ResonanceConfig::default();
        let report = default_resonance_report(1.0, 1e-6, &config);
        assert_eq!(report.algebra_name, "Pathion");
        assert_eq!(report.ambient_dim, 32);
        assert!(report.total_coupling >= 0.0);
    }
}
