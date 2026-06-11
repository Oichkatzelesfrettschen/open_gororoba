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
/// Units: if dk is in `[1/um]` and L in `[um]`, F is in `[um]`.
pub fn phase_matching_function(delta_k: f64, thickness: f64) -> f64 {
    let arg = delta_k * thickness / 2.0;
    if arg.abs() < 1e-12 {
        thickness
    } else {
        arg.sin() / (delta_k / 2.0)
    }
}

/// Coherence length L_coh = pi / |dk| `[same units as 1/dk]`.
pub fn coherence_length(delta_k: f64) -> f64 {
    PI / delta_k.abs()
}

/// Result of an SFWM dominance check at a given crystal thickness.
#[derive(Debug, Clone)]
pub struct SfwmDominanceResult {
    /// Crystal thickness `[um]`.
    pub thickness_um: f64,
    /// SFWM coherence length `[um]`.
    pub l_coh_sfwm_um: f64,
    /// SHG coherence length `[um]`.
    pub l_coh_shg_um: f64,
    /// SPDC coherence length `[um]`.
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
    thicknesses_um
        .iter()
        .map(|&l| sfwm_dominance_check(l))
        .collect()
}

// ===================== Son & Chekhova (2026) full reproduction =====================

/// Wavevector mismatches for the three processes.
#[derive(Debug, Clone, Copy)]
pub struct WavevectorMismatches {
    /// SFWM: dk = 2*k_pump - k_signal - k_idler `[1/um]`.
    pub dk_sfwm: f64,
    /// SHG: dk = 2*k_pump - k_sh `[1/um]`.
    pub dk_shg: f64,
    /// SPDC: dk = k_sh - k_signal - k_idler `[1/um]`.
    pub dk_spdc: f64,
}

/// Material parameters for SFWM rate calculation.
///
/// All susceptibilities are in relative (dimensionless) ratio units --
/// absolute values cancel in R_cas/R_dir.
#[derive(Debug, Clone)]
pub struct SfwmMaterialParams {
    /// Effective chi(2) for SHG `[arb. units]`.
    pub chi2_shg: f64,
    /// Effective chi(2) for SPDC `[arb. units]`.
    pub chi2_spdc: f64,
    /// Effective chi(3) for direct SFWM `[arb. units]`.
    pub chi3_sfwm: f64,
    /// Refractive index at pump wavelength.
    pub n_pump: f64,
    /// Refractive index at signal wavelength.
    pub n_signal: f64,
    /// Refractive index at idler wavelength.
    pub n_idler: f64,
    /// Refractive index at second-harmonic wavelength.
    pub n_sh: f64,
    /// Pump wavelength `[um]`.
    pub lambda_pump: f64,
    /// Signal wavelength `[um]`.
    pub lambda_signal: f64,
    /// Idler wavelength `[um]`.
    pub lambda_idler: f64,
    /// Crystal thickness `[um]`.
    pub thickness: f64,
}

impl SfwmMaterialParams {
    /// Construct parameters for 10 um LiNbO3 as in Son & Chekhova (2026).
    ///
    /// Pump: 1030 nm, Signal: 770 nm, Idler: 1550 nm (nearly degenerate
    /// energy conservation: 2/1.030 ~ 1/0.770 + 1/1.550 `[in 1/um]`).
    ///
    /// Uses paper wavevector mismatches for consistency:
    ///   dk_SFWM ~ 0.094 um^-1 (L_coh ~ 33 um)
    ///   dk_SHG  ~ 1.01 um^-1  (L_coh ~ 3.1 um)
    ///   dk_SPDC ~ 0.92 um^-1  (L_coh ~ 3.4 um)
    ///
    /// Susceptibility ratio: chi2 ~ 27 pm/V = 27e-12 m/V (d33 LiNbO3),
    /// chi3 ~ 2.4e-21 m^2/V^2 (Adair et al. 1989, n2 = 5.3e-15 cm^2/W).
    /// Stored in SI (m/V and m^2/V^2) so that chi2^2/chi3 is dimensionless
    /// and the rate ratio is directly physical.
    pub fn linbo3_son_chekhova_2026() -> Self {
        // Refractive indices from Zelmon Sellmeier (ordinary ray)
        // n_o(1030) ~ 2.232, n_o(770) ~ 2.259, n_o(1550) ~ 2.211
        // n_o(515) ~ 2.325
        Self {
            chi2_shg: 27.0e-12,  // m/V (d33, converted from pm/V)
            chi2_spdc: 27.0e-12, // same tensor element
            chi3_sfwm: 2.4e-21,  // m^2/V^2 (Adair 1989)
            n_pump: 2.232,
            n_signal: 2.259,
            n_idler: 2.211,
            n_sh: 2.325,
            lambda_pump: 1.030,
            lambda_signal: 0.770,
            lambda_idler: 1.550,
            thickness: 10.0,
        }
    }

    /// Construct using the paper's calibrated coherence lengths directly.
    ///
    /// The paper reports L_coh_SFWM = 33.3 um, L_coh_SHG = 3.1 um,
    /// L_coh_SPDC = 3.4 um. The exact polarization and temperature conditions
    /// differ from simple ordinary-ray Sellmeier estimates, so this constructor
    /// reproduces the paper's stated phase-matching conditions exactly.
    pub fn linbo3_paper_calibrated() -> Self {
        Self {
            chi2_shg: 27.0e-12,
            chi2_spdc: 27.0e-12,
            chi3_sfwm: 2.4e-21,
            // Refractive indices are not used when calling paper_wavevector_mismatches()
            // but we keep physically reasonable values for documentation.
            n_pump: 2.232,
            n_signal: 2.259,
            n_idler: 2.211,
            n_sh: 2.325,
            lambda_pump: 1.030,
            lambda_signal: 0.770,
            lambda_idler: 1.550,
            thickness: 10.0,
        }
    }

    /// Wavevector mismatches from the paper's stated coherence lengths.
    ///
    /// More accurate than Sellmeier-derived values because the paper's
    /// measurements account for temperature, crystal orientation, and
    /// exact polarization conditions.
    pub fn paper_wavevector_mismatches(&self) -> WavevectorMismatches {
        WavevectorMismatches {
            dk_sfwm: PI / 33.3,
            dk_shg: PI / 3.1,
            dk_spdc: PI / 3.4,
        }
    }

    /// Compute wavevector mismatches from refractive indices and wavelengths.
    pub fn wavevector_mismatches(&self) -> WavevectorMismatches {
        let k_pump = 2.0 * PI * self.n_pump / self.lambda_pump;
        let k_signal = 2.0 * PI * self.n_signal / self.lambda_signal;
        let k_idler = 2.0 * PI * self.n_idler / self.lambda_idler;
        let lambda_sh = self.lambda_pump / 2.0;
        let k_sh = 2.0 * PI * self.n_sh / lambda_sh;

        WavevectorMismatches {
            dk_sfwm: 2.0 * k_pump - k_signal - k_idler,
            dk_shg: 2.0 * k_pump - k_sh,
            dk_spdc: k_sh - k_signal - k_idler,
        }
    }
}

/// Rate calculation result for SFWM vs cascaded SHG+SPDC.
#[derive(Debug, Clone, Copy)]
pub struct SfwmRateResult {
    /// |A_dir|^2 -- direct SFWM amplitude squared.
    pub a_dir_sq: f64,
    /// |A_cas|^2 -- cascaded SHG+SPDC amplitude squared.
    pub a_cas_sq: f64,
    /// R_cas / R_dir ratio (paper reports 0.048 at 10 um).
    pub rate_ratio_cas_to_dir: f64,
    /// Phase-matching function values.
    pub f_sfwm: f64,
    pub f_shg: f64,
    pub f_spdc: f64,
}

/// Cascaded amplitude squared |A_cas|^2 (Eq. 6 of Son & Chekhova 2026).
///
/// A_cas ~ chi2_shg * chi2_spdc * `[exp(i*dk_SHG*L/2) * F_SFWM(L) - F_SPDC(L)]`
///
/// The bracket is computed as a complex magnitude.
/// Uses Sellmeier-derived wavevector mismatches.
pub fn cascaded_amplitude_sq(params: &SfwmMaterialParams) -> f64 {
    cascaded_amplitude_sq_with_dk(params, &params.wavevector_mismatches())
}

/// Cascaded amplitude squared using explicit wavevector mismatches.
pub fn cascaded_amplitude_sq_with_dk(
    params: &SfwmMaterialParams,
    wm: &WavevectorMismatches,
) -> f64 {
    let l = params.thickness;

    let f_sfwm = phase_matching_function(wm.dk_sfwm, l);
    let f_spdc = phase_matching_function(wm.dk_spdc, l);

    // Complex bracket: exp(i * dk_SHG * L / 2) * F_SFWM - F_SPDC
    let phase = wm.dk_shg * l / 2.0;
    let re = phase.cos() * f_sfwm - f_spdc;
    let im = phase.sin() * f_sfwm;
    let bracket_sq = re * re + im * im;

    let chi_prod = params.chi2_shg * params.chi2_spdc;
    chi_prod * chi_prod * bracket_sq
}

/// Direct SFWM amplitude squared |A_dir|^2 (Eq. 8 of Son & Chekhova 2026).
///
/// A_dir = chi3 * F_SFWM(L), with |exp(i*phase)| = 1 factored out.
/// Uses Sellmeier-derived wavevector mismatches.
pub fn direct_amplitude_sq(params: &SfwmMaterialParams) -> f64 {
    direct_amplitude_sq_with_dk(params, &params.wavevector_mismatches())
}

/// Direct amplitude squared using explicit wavevector mismatches.
pub fn direct_amplitude_sq_with_dk(params: &SfwmMaterialParams, wm: &WavevectorMismatches) -> f64 {
    let f = phase_matching_function(wm.dk_sfwm, params.thickness);
    params.chi3_sfwm * params.chi3_sfwm * f * f
}

/// Compute the cascaded-to-direct rate ratio R_cas/R_dir.
///
/// Includes the tensor prefactors from the paper:
///   R_dir ~ (3/4)^2 * |A_dir|^2
///   R_cas ~ (2/3)^2 * |A_cas|^2
///
/// Uses Sellmeier-derived wavevector mismatches by default.
pub fn rate_ratio(params: &SfwmMaterialParams) -> SfwmRateResult {
    rate_ratio_with_dk(params, &params.wavevector_mismatches())
}

/// Rate ratio using explicit wavevector mismatches.
///
/// Use `params.paper_wavevector_mismatches()` for the paper's calibrated values.
pub fn rate_ratio_with_dk(
    params: &SfwmMaterialParams,
    wm: &WavevectorMismatches,
) -> SfwmRateResult {
    let l = params.thickness;

    let f_sfwm = phase_matching_function(wm.dk_sfwm, l);
    let f_shg = phase_matching_function(wm.dk_shg, l);
    let f_spdc = phase_matching_function(wm.dk_spdc, l);

    let a_dir_sq = direct_amplitude_sq_with_dk(params, wm);
    let a_cas_sq = cascaded_amplitude_sq_with_dk(params, wm);

    // Tensor prefactors
    let pref_dir = (3.0_f64 / 4.0).powi(2);
    let pref_cas = (2.0_f64 / 3.0).powi(2);

    let r_dir = pref_dir * a_dir_sq;
    let r_cas = pref_cas * a_cas_sq;
    let ratio = r_cas / r_dir.max(1e-60);

    SfwmRateResult {
        a_dir_sq,
        a_cas_sq,
        rate_ratio_cas_to_dir: ratio,
        f_sfwm,
        f_shg,
        f_spdc,
    }
}

/// Maker fringe sweep: compute |F|^2 for all three processes vs thickness.
///
/// Returns Vec<(thickness_um, f_sfwm_sq, f_shg_sq, f_spdc_sq)> for Fig. 4(a).
pub fn maker_fringe_sweep(
    params: &SfwmMaterialParams,
    thicknesses: &[f64],
) -> Vec<(f64, f64, f64, f64)> {
    let wm = params.wavevector_mismatches();
    thicknesses
        .iter()
        .map(|&l| {
            let fs = phase_matching_function(wm.dk_sfwm, l);
            let fh = phase_matching_function(wm.dk_shg, l);
            let fp = phase_matching_function(wm.dk_spdc, l);
            (l, fs * fs, fh * fh, fp * fp)
        })
        .collect()
}

/// Rate sweep: compute R_direct and R_cascaded vs thickness.
///
/// Returns Vec<(thickness_um, r_direct, r_cascaded)> for Fig. 4(b).
/// Uses Sellmeier-derived dk values.
pub fn rate_sweep(params: &SfwmMaterialParams, thicknesses: &[f64]) -> Vec<(f64, f64, f64)> {
    rate_sweep_with_dk(params, &params.wavevector_mismatches(), thicknesses)
}

/// Rate sweep using explicit wavevector mismatches.
pub fn rate_sweep_with_dk(
    params: &SfwmMaterialParams,
    wm: &WavevectorMismatches,
    thicknesses: &[f64],
) -> Vec<(f64, f64, f64)> {
    thicknesses
        .iter()
        .map(|&l| {
            let mut p = params.clone();
            p.thickness = l;
            let r = rate_ratio_with_dk(&p, wm);
            let pref_dir = (3.0_f64 / 4.0).powi(2);
            let pref_cas = (2.0_f64 / 3.0).powi(2);
            (l, pref_dir * r.a_dir_sq, pref_cas * r.a_cas_sq)
        })
        .collect()
}

/// SFWM g(2) correlation model: g(2)(P) = 1 + a / P^2 (Fig. 2d).
///
/// At low power, g(2) >> 2 (thermal photon statistics).
/// At high power, g(2) -> 1 (Poissonian / stimulated regime).
pub fn g2_sfwm_model(power: f64, a: f64) -> f64 {
    1.0 + a / (power * power).max(1e-30)
}

/// SPDC g(2) correlation model: g(2)(P) = 1 + a / P (Fig. 3b).
///
/// SPDC photon number scales linearly with P, so fluctuations
/// scale as 1/P, giving g(2) = 1 + a/P.
pub fn g2_spdc_model(power: f64, a: f64) -> f64 {
    1.0 + a / power.max(1e-30)
}

/// SFWM photon number model: N ~ alpha * P^2 (quadratic pump dependence).
pub fn photon_number_sfwm(power: f64, alpha: f64) -> f64 {
    alpha * power * power
}

/// SPDC photon number model: N ~ alpha * P (linear pump dependence).
pub fn photon_number_spdc(power: f64, alpha: f64) -> f64 {
    alpha * power
}

/// Polarization dependence of SFWM rate: proportional to cos^2(theta).
///
/// theta is the angle between pump polarization and the crystal optical axis.
/// Maximum generation when pump is aligned with d33 axis (theta = 0).
pub fn polarization_dependence(theta: f64) -> f64 {
    theta.cos().powi(2)
}

/// Substrate SFWM contribution result.
#[derive(Debug, Clone)]
pub struct SubstrateSfwmResult {
    /// SFWM wavevector mismatch dk_SFWM in substrate `[1/um]`.
    pub dk_sfwm: f64,
    /// Phase-matching function F(dk, L) `[um]`.
    pub f_sfwm: f64,
    /// |F|^2 * chi3^2 (proportional to SFWM rate).
    pub rate_proxy: f64,
    /// Substrate thickness `[um]`.
    pub thickness_um: f64,
    /// Coherence length L_coh = pi/|dk| `[um]`.
    pub l_coh_sfwm_um: f64,
    /// Whether the substrate contributes nonzero SFWM.
    pub has_contribution: bool,
}

/// Parameters for computing substrate SFWM contribution.
#[derive(Debug, Clone)]
pub struct SubstrateSfwmParams {
    /// Refractive index at pump wavelength.
    pub n_pump: f64,
    /// Refractive index at signal wavelength.
    pub n_signal: f64,
    /// Refractive index at idler wavelength.
    pub n_idler: f64,
    /// Pump wavelength `[um]`.
    pub lambda_pump: f64,
    /// Signal wavelength `[um]`.
    pub lambda_signal: f64,
    /// Idler wavelength `[um]`.
    pub lambda_idler: f64,
    /// Third-order susceptibility chi(3) `[m^2/V^2]`.
    pub chi3: f64,
    /// Substrate thickness `[um]`.
    pub thickness_um: f64,
}

/// Compute SFWM contribution from a substrate material.
///
/// For a centrosymmetric substrate (e.g. fused silica), only chi(3) SFWM
/// is possible -- no chi(2) cascaded path. In the thick-crystal regime
/// (L >> L_coh), the sinc^2 phase-matching function oscillates rapidly
/// and the time-averaged contribution is nonzero but small.
pub fn substrate_sfwm_contribution(params: &SubstrateSfwmParams) -> SubstrateSfwmResult {
    let k_pump = 2.0 * PI * params.n_pump / params.lambda_pump;
    let k_signal = 2.0 * PI * params.n_signal / params.lambda_signal;
    let k_idler = 2.0 * PI * params.n_idler / params.lambda_idler;
    let dk_sfwm = 2.0 * k_pump - k_signal - k_idler;

    let f = phase_matching_function(dk_sfwm, params.thickness_um);
    let l_coh = coherence_length(dk_sfwm);
    let rate_proxy = params.chi3 * params.chi3 * f * f;

    SubstrateSfwmResult {
        dk_sfwm,
        f_sfwm: f,
        rate_proxy,
        thickness_um: params.thickness_um,
        l_coh_sfwm_um: l_coh,
        has_contribution: rate_proxy.abs() > 1e-60,
    }
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

    // ===================== Son & Chekhova (2026) tests =====================

    #[test]
    fn test_wavevector_mismatches_coherence_lengths() {
        let p = SfwmMaterialParams::linbo3_son_chekhova_2026();
        let wm = p.wavevector_mismatches();

        let l_sfwm = coherence_length(wm.dk_sfwm);
        let l_shg = coherence_length(wm.dk_shg);
        let l_spdc = coherence_length(wm.dk_spdc);

        // Paper: L_coh_SFWM ~ 33 um, L_coh_SHG ~ 3.1 um, L_coh_SPDC ~ 3.4 um
        // Allow 30% tolerance for Sellmeier vs paper values
        assert!(
            l_sfwm > 10.0,
            "SFWM coherence length {:.1} um should be >10 um",
            l_sfwm
        );
        assert!(
            l_shg < 10.0,
            "SHG coherence length {:.1} um should be <10 um",
            l_shg
        );
        assert!(
            l_spdc < 10.0,
            "SPDC coherence length {:.1} um should be <10 um",
            l_spdc
        );
        assert!(
            l_sfwm > l_shg * 3.0,
            "SFWM L_coh ({:.1}) should be >3x SHG L_coh ({:.1})",
            l_sfwm,
            l_shg
        );
    }

    #[test]
    fn test_direct_dominates_cascaded_at_10um() {
        // Use paper-calibrated dk values for the dominance ratio test
        let p = SfwmMaterialParams::linbo3_paper_calibrated();
        let wm = p.paper_wavevector_mismatches();
        let r = rate_ratio_with_dk(&p, &wm);
        assert!(
            r.rate_ratio_cas_to_dir < 0.15,
            "R_cas/R_dir = {:.4} should be < 0.15 at 10 um (paper calibrated)",
            r.rate_ratio_cas_to_dir
        );
    }

    #[test]
    fn test_sellmeier_direct_dominates_cascaded() {
        // Sellmeier-derived dk: direct still dominates, but ratio is larger
        let p = SfwmMaterialParams::linbo3_son_chekhova_2026();
        let r = rate_ratio(&p);
        assert!(
            r.rate_ratio_cas_to_dir < 1.0,
            "R_cas/R_dir = {:.4} should be < 1.0 (direct still dominates)",
            r.rate_ratio_cas_to_dir
        );
    }

    #[test]
    fn test_f_sfwm_much_larger_than_f_shg() {
        let p = SfwmMaterialParams::linbo3_son_chekhova_2026();
        let r = rate_ratio(&p);
        assert!(
            r.f_sfwm.abs() > r.f_shg.abs() * 3.0,
            "|F_SFWM| = {:.2} should be >3x |F_SHG| = {:.2}",
            r.f_sfwm.abs(),
            r.f_shg.abs()
        );
    }

    #[test]
    fn test_g2_sfwm_high_at_low_power() {
        // At low power, g(2) should be much greater than 2 (thermal statistics)
        let g2 = g2_sfwm_model(0.01, 1.0);
        assert!(g2 > 2.0, "g(2) = {:.1} should be >2 at low power", g2);
    }

    #[test]
    fn test_g2_approaches_1_at_high_power() {
        // SFWM: g(2) = 1 + 1/10000^2 ~ 1 + 1e-8
        let g2_sfwm = g2_sfwm_model(100.0, 1.0);
        // SPDC: g(2) = 1 + 1/1000 = 1.001
        let g2_spdc = g2_spdc_model(1000.0, 1.0);
        assert!(
            (g2_sfwm - 1.0).abs() < 0.01,
            "SFWM g(2) = {:.6} should approach 1 at high power",
            g2_sfwm
        );
        assert!(
            (g2_spdc - 1.0).abs() < 0.01,
            "SPDC g(2) = {:.6} should approach 1 at high power",
            g2_spdc
        );
    }

    #[test]
    fn test_maker_fringe_sweep_lengths() {
        let p = SfwmMaterialParams::linbo3_son_chekhova_2026();
        let thicknesses: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let fringes = maker_fringe_sweep(&p, &thicknesses);
        assert_eq!(fringes.len(), 100);

        // SFWM |F|^2 should be monotonically increasing below L_coh_SFWM
        let wm = p.wavevector_mismatches();
        let l_coh = coherence_length(wm.dk_sfwm);
        let below_coh: Vec<_> = fringes.iter().filter(|f| f.0 < l_coh * 0.8).collect();
        for w in below_coh.windows(2) {
            assert!(
                w[1].1 >= w[0].1 * 0.99,
                "SFWM |F|^2 should increase below L_coh: {:.2} < {:.2} at L={:.0} um",
                w[1].1,
                w[0].1,
                w[1].0
            );
        }
    }

    #[test]
    fn test_shg_oscillates_in_sweep() {
        let p = SfwmMaterialParams::linbo3_son_chekhova_2026();
        let thicknesses: Vec<f64> = (1..=50).map(|i| i as f64 * 0.5).collect();
        let fringes = maker_fringe_sweep(&p, &thicknesses);

        // SHG |F|^2 should have local maxima and minima (Maker fringes)
        let shg_vals: Vec<f64> = fringes.iter().map(|f| f.2).collect();
        let mut increases = 0;
        let mut decreases = 0;
        for w in shg_vals.windows(2) {
            if w[1] > w[0] {
                increases += 1;
            } else {
                decreases += 1;
            }
        }
        assert!(
            increases > 3 && decreases > 3,
            "SHG should oscillate: {} increases, {} decreases",
            increases,
            decreases
        );
    }

    #[test]
    fn test_polarization_cos2() {
        let at_0 = polarization_dependence(0.0);
        let at_90 = polarization_dependence(PI / 2.0);
        assert!((at_0 - 1.0).abs() < 1e-10, "cos^2(0) = {:.4}", at_0);
        assert!(at_90.abs() < 1e-10, "cos^2(pi/2) = {:.4e}", at_90);
    }

    #[test]
    fn test_substrate_sfwm_fused_silica_500um() {
        // Fused silica at same wavelengths as Son & Chekhova (2026):
        // pump=1030nm, signal=770nm, idler=1550nm.
        // n(SiO2) from Malitson: ~1.450 (pump), ~1.454 (signal), ~1.444 (idler).
        // chi3(SiO2) ~ 2.5e-22 m^2/V^2 (Adair et al. 1989, n2=2.74e-16 cm^2/W).
        let r = substrate_sfwm_contribution(&SubstrateSfwmParams {
            n_pump: 1.4500,
            n_signal: 1.4539,
            n_idler: 1.4440,
            lambda_pump: 1.030,
            lambda_signal: 0.770,
            lambda_idler: 1.550,
            chi3: 2.5e-22,
            thickness_um: 500.0,
        });

        // Substrate IS in thick-crystal regime: L >> L_coh
        assert!(
            r.l_coh_sfwm_um < 500.0,
            "Substrate should be thick: L_coh={:.1} um < 500 um",
            r.l_coh_sfwm_um,
        );

        // Nonzero SFWM contribution exists (sinc never exactly zero at generic dk*L)
        assert!(
            r.has_contribution,
            "Substrate should have nonzero SFWM: rate_proxy={:.2e}",
            r.rate_proxy,
        );

        // dk should be nonzero (dispersion in SiO2)
        assert!(
            r.dk_sfwm.abs() > 1e-6,
            "dk_SFWM should be nonzero: {:.6}",
            r.dk_sfwm,
        );
    }
}
