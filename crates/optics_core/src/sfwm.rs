//! Stimulated Four-Wave Mixing (SFWM) thin-layer phase matching.
//!
//! In a thin nonlinear layer, direct SFWM can dominate over cascaded
//! SHG+SPDC because the SFWM coherence length is much larger than
//! the crystal thickness, while SHG and SPDC coherence lengths are
//! much shorter and oscillate (destructive interference).
//!
//! # References
//! - Son, C. & Chekhova, M. (2026), arXiv:2601.23137

use std::f64::consts::PI;

/// Phase-matching function F(L) = sin(dk * L / 2) / (dk / 2).
///
/// For small dk*L, F(L) -> L (perfect phase matching).
/// Units: if dk is in [1/um] and L in [um], F is in [um].
pub fn phase_matching_function(delta_k: f64, thickness: f64) -> f64 {
    let arg = delta_k * thickness / 2.0;
    if arg.abs() < 1e-12 {
        thickness
    } else {
        arg.sin() / (delta_k / 2.0)
    }
}

/// Coherence length L_coh = pi / |dk| [same units as 1/dk].
pub fn coherence_length(delta_k: f64) -> f64 {
    PI / delta_k.abs()
}

/// Result of an SFWM dominance check at a given crystal thickness.
#[derive(Debug, Clone)]
pub struct SfwmDominanceResult {
    /// Crystal thickness [um].
    pub thickness_um: f64,
    /// SFWM coherence length [um].
    pub l_coh_sfwm_um: f64,
    /// SHG coherence length [um].
    pub l_coh_shg_um: f64,
    /// SPDC coherence length [um].
    pub l_coh_spdc_um: f64,
    /// Phase-matching function for SFWM.
    pub f_sfwm: f64,
    /// Phase-matching function for SHG.
    pub f_shg: f64,
    /// Phase-matching function for SPDC.
    pub f_spdc: f64,
    /// Direct SFWM efficiency ~ |F_sfwm|^2.
    pub eff_direct: f64,
    /// Cascaded efficiency ~ |F_shg|^2 * |F_spdc|^2.
    pub eff_cascaded: f64,
    /// Ratio eff_direct / eff_cascaded.
    pub dominance_ratio: f64,
}

/// Check that direct SFWM dominates over cascaded SHG+SPDC
/// for a thin layer of given thickness (in micrometers).
///
/// Uses experimental coherence lengths from Son & Chekhova (2026):
///   L_coh_SFWM = 33.3 um, L_coh_SHG = 3.1 um, L_coh_SPDC = 3.4 um.
pub fn sfwm_dominance_check(thickness_um: f64) -> SfwmDominanceResult {
    let dk_sfwm = PI / 33.3;
    let dk_shg = PI / 3.1;
    let dk_spdc = PI / 3.4;

    let f_sfwm = phase_matching_function(dk_sfwm, thickness_um);
    let f_shg = phase_matching_function(dk_shg, thickness_um);
    let f_spdc = phase_matching_function(dk_spdc, thickness_um);

    let eff_direct = f_sfwm * f_sfwm;
    let eff_cascaded = f_shg * f_shg * f_spdc * f_spdc;
    let dominance_ratio = eff_direct / eff_cascaded.max(1e-30);

    SfwmDominanceResult {
        thickness_um,
        l_coh_sfwm_um: coherence_length(dk_sfwm),
        l_coh_shg_um: coherence_length(dk_shg),
        l_coh_spdc_um: coherence_length(dk_spdc),
        f_sfwm,
        f_shg,
        f_spdc,
        eff_direct,
        eff_cascaded,
        dominance_ratio,
    }
}

/// Sweep crystal thickness and compute direct/cascaded ratio at each point.
pub fn thickness_sweep(thicknesses_um: &[f64]) -> Vec<SfwmDominanceResult> {
    thicknesses_um.iter().map(|&l| sfwm_dominance_check(l)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coherence_lengths_match_paper() {
        let dk_sfwm = PI / 33.3;
        let dk_shg = PI / 3.1;
        let dk_spdc = PI / 3.4;

        assert!((coherence_length(dk_sfwm) - 33.3).abs() < 0.1);
        assert!((coherence_length(dk_shg) - 3.1).abs() < 0.1);
        assert!((coherence_length(dk_spdc) - 3.4).abs() < 0.1);
    }

    #[test]
    fn test_phase_matching_small_dk() {
        let thickness = 5.0;
        let f = phase_matching_function(1e-15, thickness);
        assert!(
            (f - thickness).abs() < 1e-10,
            "F = {f}, expected L = {thickness}"
        );
    }

    #[test]
    fn test_sfwm_dominates_at_10um() {
        let result = sfwm_dominance_check(10.0);
        assert!(
            result.eff_direct > result.eff_cascaded,
            "Direct SFWM should exceed cascaded efficiency"
        );
        assert!(
            result.dominance_ratio > 5.0,
            "Dominance ratio {:.1} < 5x",
            result.dominance_ratio
        );
    }

    #[test]
    fn test_sfwm_efficiency_positive() {
        let result = sfwm_dominance_check(10.0);
        assert!(result.eff_direct > 0.0);
        assert!(result.eff_cascaded > 0.0);
        assert!(result.f_sfwm > 0.0);
    }

    #[test]
    fn test_thickness_sweep_monotonic_count() {
        let thicknesses: Vec<f64> = (1..=10).map(|i| i as f64 * 10.0).collect();
        let results = thickness_sweep(&thicknesses);
        assert_eq!(results.len(), 10);
    }
}
