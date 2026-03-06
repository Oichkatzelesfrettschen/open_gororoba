//! Pathion resonance band detection for N-body orbital dynamics.
//!
//! Maps 32D Pathion ZD graph eigenvalues to orbital resonance bands.
//! When an orbital frequency matches a harmonic of a ZD eigenvalue,
//! the coupling between the Pathion perturbation and the orbit is enhanced.
//!
//! # Physics
//! The ZD graph Laplacian eigenvalues lambda_k define natural frequencies
//! of the Pathion algebraic structure. An orbit with frequency f_orb
//! experiences enhanced coupling when f_orb ~ lambda_k / (2*pi*n) for
//! integer n (harmonic resonance).
//!
//! # References
//! - Moreno (1998): Zero divisors of the Cayley-Dickson algebras
//! - Murray & Dermott (1999): Solar System Dynamics, Ch. 8

use crate::pathion_eigenvalues::PathionEigenvalueSpectrum;

/// A single resonance band identified between a ZD eigenvalue and an orbital frequency.
#[derive(Debug, Clone)]
pub struct ResonanceBand {
    /// Which harmonic (1 = fundamental, 2 = first overtone, etc.)
    pub harmonic: usize,
    /// Index of the ZD eigenvalue involved.
    pub zd_index: usize,
    /// ZD eigenvalue (from graph Laplacian).
    pub eigenvalue: f64,
    /// Detuning: |f_orb - eigenvalue / (2*pi*n)| / f_orb.
    pub detuning: f64,
    /// Coupling strength: 1 / (1 + detuning^2 / width^2) (Lorentzian profile).
    pub coupling_strength: f64,
}

/// Configuration for resonance detection.
#[derive(Debug, Clone)]
pub struct ResonanceConfig {
    /// Maximum harmonic order to check.
    pub max_harmonic: usize,
    /// Resonance width (fractional): bands within this fraction of f_orb are coupled.
    pub width: f64,
    /// Minimum coupling strength to report (filter weak resonances).
    pub min_coupling: f64,
}

impl Default for ResonanceConfig {
    fn default() -> Self {
        Self {
            max_harmonic: 5,
            width: 0.1,
            min_coupling: 0.01,
        }
    }
}

/// Compute resonance bands between Pathion eigenvalues and an orbital frequency.
///
/// Returns a list of resonance bands sorted by coupling strength (strongest first).
pub fn compute_resonance_bands(
    spectrum: &PathionEigenvalueSpectrum,
    orbital_freq: f64,
    config: &ResonanceConfig,
) -> Vec<ResonanceBand> {
    if orbital_freq <= 0.0 {
        return Vec::new();
    }

    let two_pi = 2.0 * std::f64::consts::PI;
    let mut bands = Vec::new();

    for (idx, &eigenvalue) in spectrum.eigenvalues.iter().enumerate() {
        if eigenvalue <= 1e-10 {
            continue; // Skip zero eigenvalues (connected component modes)
        }

        for n in 1..=config.max_harmonic {
            let resonant_freq = eigenvalue / (two_pi * n as f64);
            let detuning = (orbital_freq - resonant_freq).abs() / orbital_freq;

            // Lorentzian coupling profile
            let coupling = 1.0 / (1.0 + (detuning / config.width).powi(2));

            if coupling >= config.min_coupling {
                bands.push(ResonanceBand {
                    harmonic: n,
                    zd_index: idx,
                    eigenvalue,
                    detuning,
                    coupling_strength: coupling,
                });
            }
        }
    }

    bands.sort_by(|a, b| {
        b.coupling_strength
            .partial_cmp(&a.coupling_strength)
            .unwrap()
    });

    bands
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
    compute_resonance_bands(spectrum, orbital_freq, config)
        .iter()
        .map(|b| b.coupling_strength)
        .sum()
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
    let total = total_resonance_coupling(spectrum, orbital_freq, config);

    // Distribute coupling across 3 spatial dimensions using
    // the first 3 positive eigenvalues as weights (if available).
    let positive_evs: Vec<f64> = spectrum
        .eigenvalues
        .iter()
        .copied()
        .filter(|&v| v > 1e-10)
        .take(3)
        .collect();

    if positive_evs.is_empty() || total < 1e-15 {
        return [0.0; 3];
    }

    let sum_evs: f64 = positive_evs.iter().sum();
    let scale = alpha_pathion * total;

    let mut result = [scale / 3.0; 3]; // Default: isotropic
    if sum_evs > 0.0 {
        for (i, &ev) in positive_evs.iter().enumerate() {
            result[i] = scale * ev / sum_evs;
        }
    }

    result
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
}
