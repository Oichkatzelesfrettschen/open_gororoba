//! Spectral analysis and fractal cosmology.
//!
//! Implements spectral dimension flows, turbulence spectra, and
//! comparisons between different theoretical frameworks.
//!
//! Key results (from G2.1-G2.2):
//! - Calcagni 2010: d_S flows from 2 (UV) to 4 (IR)
//! - Kraichnan 1967: k^{-3} enstrophy cascade (EXACT match for project ansatz)
//! - Kolmogorov 1941: k^{-5/3} energy cascade (does NOT match)
//!
//! References:
//! - Calcagni, PRL 104 (2010) 251301
//! - Kraichnan, Phys. Fluids 10 (1967) 1417
//! - Kolmogorov, Dokl. Akad. Nauk SSSR 30 (1941) 299

/// Calcagni's running spectral dimension d_S(k).
///
/// In multi-fractional spacetime:
/// - d_S -> 2 as k -> infinity (UV, Planck scale)
/// - d_S -> 4 as k -> 0 (IR, macroscopic scale)
///
/// Formula: d_S(k) = 4 - 2 / (1 + (k/k_*)^{-alpha})
pub fn calcagni_spectral_dimension(k: f64, alpha: f64) -> f64 {
    let k_star = 1.0; // Transition scale in natural units
    let ratio = (k / k_star).powf(-alpha);
    4.0 - 2.0 / (1.0 + ratio)
}

/// CDT spectral dimension (Ambjorn et al. 2005).
///
/// Uses exponential interpolation: d_S(k) = 4 - 2 * exp(-k_pl / k)
pub fn cdt_spectral_dimension(k: f64, k_pl: f64) -> f64 {
    4.0 - 2.0 * (-k_pl / k).exp()
}

/// k^{-3} spectrum (project's vacuum dynamics ansatz).
///
/// This matches EXACTLY the Kraichnan 2D enstrophy cascade.
pub fn k_minus_3_spectrum(k: f64) -> f64 {
    k.powf(-3.0)
}

/// Kolmogorov 1941 energy cascade: E(k) ~ k^{-5/3}.
pub fn kolmogorov_spectrum(k: f64) -> f64 {
    k.powf(-5.0 / 3.0)
}

/// Kraichnan 1967 enstrophy cascade: E(k) ~ k^{-3}.
///
/// In 2D turbulence, there are two cascade regimes:
/// - Inverse energy cascade at large scales: k^{-5/3}
/// - Forward enstrophy cascade at small scales: k^{-3}
///
/// The k^{-3} spectrum is the EXACT match for the project's ansatz.
pub fn kraichnan_enstrophy_spectrum(k: f64) -> f64 {
    k_minus_3_spectrum(k)
}

/// Parisi-Sourlas dimensional reduction: D -> D-2.
///
/// In presence of quenched random disorder, effective dimension reduces.
/// For D=4, D_eff=2. This does NOT produce k^{-3}.
pub fn parisi_sourlas_effective_dimension(d: usize) -> usize {
    d.saturating_sub(2)
}

/// Spectral exponent from Parisi-Sourlas.
///
/// P(k) ~ k^{D_eff - 1}, so for D=4, exponent = +1 (not -3).
pub fn parisi_sourlas_spectrum_exponent(d: usize) -> f64 {
    let d_eff = parisi_sourlas_effective_dimension(d);
    d_eff as f64 - 1.0
}

/// Batch computation of Calcagni spectral dimension.
pub fn batch_calcagni_d_s(k_values: &[f64], alpha: f64) -> Vec<f64> {
    k_values
        .iter()
        .map(|&k| calcagni_spectral_dimension(k, alpha))
        .collect()
}

/// Batch computation of CDT spectral dimension.
pub fn batch_cdt_d_s(k_values: &[f64], k_pl: f64) -> Vec<f64> {
    k_values
        .iter()
        .map(|&k| cdt_spectral_dimension(k, k_pl))
        .collect()
}

/// Compute RMS deviation between two spectra in log space.
pub fn rms_deviation_log(spectrum1: &[f64], spectrum2: &[f64]) -> f64 {
    assert_eq!(spectrum1.len(), spectrum2.len());

    let n = spectrum1.len() as f64;
    let sum_sq: f64 = spectrum1
        .iter()
        .zip(spectrum2.iter())
        .map(|(&s1, &s2)| {
            let log1 = s1.max(1e-30).log10();
            let log2 = s2.max(1e-30).log10();
            (log1 - log2).powi(2)
        })
        .sum();

    (sum_sq / n).sqrt()
}

/// Analysis result for k^{-3} origin.
#[derive(Debug, Clone)]
pub struct SpectralAnalysisResult {
    pub kraichnan_matches: bool,
    pub kolmogorov_matches: bool,
    pub calcagni_matches: bool,
    pub parisi_sourlas_matches: bool,
    pub kraichnan_rms: f64,
    pub kolmogorov_rms: f64,
    pub calcagni_rms: f64,
}

/// Analyze the physical origin of k^{-3} spectrum.
///
/// Compares against known frameworks and returns which ones match.
pub fn analyze_k_minus_3_origin(k_min: f64, k_max: f64, n_points: usize) -> SpectralAnalysisResult {
    // Generate k values (log-spaced)
    let log_min = k_min.ln();
    let log_max = k_max.ln();
    let k_values: Vec<f64> = (0..n_points)
        .map(|i| {
            let t = i as f64 / (n_points - 1) as f64;
            (log_min + t * (log_max - log_min)).exp()
        })
        .collect();

    // Compute reference k^{-3} spectrum
    let k_minus_3: Vec<f64> = k_values.iter().map(|&k| k_minus_3_spectrum(k)).collect();

    // Compute comparison spectra
    let kraichnan: Vec<f64> = k_values
        .iter()
        .map(|&k| kraichnan_enstrophy_spectrum(k))
        .collect();
    let kolmogorov: Vec<f64> = k_values.iter().map(|&k| kolmogorov_spectrum(k)).collect();

    // Calcagni: use spectral density P(k) ~ k^{d_S(k) - 1}
    let alpha = 0.5;
    let calcagni: Vec<f64> = k_values
        .iter()
        .map(|&k| {
            let d_s = calcagni_spectral_dimension(k, alpha);
            k.powf(d_s - 1.0)
        })
        .collect();

    // Compute RMS deviations
    let kraichnan_rms = rms_deviation_log(&k_minus_3, &kraichnan);
    let kolmogorov_rms = rms_deviation_log(&k_minus_3, &kolmogorov);
    let calcagni_rms = rms_deviation_log(&k_minus_3, &calcagni);

    // Threshold for "match"
    let match_threshold = 0.01;

    // Parisi-Sourlas: D=4 gives exponent +1, not -3
    let ps_exp = parisi_sourlas_spectrum_exponent(4);
    let parisi_sourlas_matches = (ps_exp - (-3.0)).abs() < 0.1;

    SpectralAnalysisResult {
        kraichnan_matches: kraichnan_rms < match_threshold,
        kolmogorov_matches: kolmogorov_rms < match_threshold,
        calcagni_matches: calcagni_rms < match_threshold,
        parisi_sourlas_matches,
        kraichnan_rms,
        kolmogorov_rms,
        calcagni_rms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calcagni_uv_limit() {
        // At high k (UV), d_S should approach 2
        let d_s = calcagni_spectral_dimension(1000.0, 0.5);
        assert!(d_s < 2.5, "UV limit should approach 2, got {}", d_s);
    }

    #[test]
    fn test_calcagni_ir_limit() {
        // At low k (IR), d_S should approach 4
        let d_s = calcagni_spectral_dimension(0.001, 0.5);
        assert!(d_s > 3.5, "IR limit should approach 4, got {}", d_s);
    }

    #[test]
    fn test_kraichnan_equals_k_minus_3() {
        for k in [0.1, 1.0, 10.0, 100.0] {
            let kraichnan = kraichnan_enstrophy_spectrum(k);
            let k_m3 = k_minus_3_spectrum(k);
            assert!(
                (kraichnan - k_m3).abs() < 1e-10,
                "Kraichnan should equal k^{{-3}} at k={}",
                k
            );
        }
    }

    #[test]
    fn test_kolmogorov_differs_from_k_minus_3() {
        let k = 10.0;
        let kolm = kolmogorov_spectrum(k);
        let k_m3 = k_minus_3_spectrum(k);
        assert!(
            (kolm - k_m3).abs() > 0.01,
            "Kolmogorov should differ from k^{{-3}}"
        );
    }

    #[test]
    fn test_parisi_sourlas_d4() {
        assert_eq!(parisi_sourlas_effective_dimension(4), 2);
        assert!((parisi_sourlas_spectrum_exponent(4) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_k_minus_3_origin_analysis() {
        let result = analyze_k_minus_3_origin(0.01, 100.0, 100);

        // Kraichnan should match (exact)
        assert!(result.kraichnan_matches, "Kraichnan should match k^{{-3}}");
        assert!(result.kraichnan_rms < 1e-10, "Kraichnan RMS should be ~0");

        // Kolmogorov should NOT match
        assert!(!result.kolmogorov_matches, "Kolmogorov should NOT match");

        // Calcagni should NOT match
        assert!(!result.calcagni_matches, "Calcagni should NOT match");

        // Parisi-Sourlas should NOT match
        assert!(
            !result.parisi_sourlas_matches,
            "Parisi-Sourlas should NOT match"
        );
    }
}
