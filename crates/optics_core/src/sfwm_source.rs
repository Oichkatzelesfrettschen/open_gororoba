//! Source-faithful complex SFWM amplitudes for Son and Chekhova (2026).
//!
//! This module owns the signed phase-matching convention and the complex
//! amplitudes in Eqs. 6 and 8. The older `sfwm` module remains available for
//! legacy characterization and is not used by this source-owned path.

use num_complex::Complex64;
use std::f64::consts::PI;
use thiserror::Error;

/// Errors raised by the source-owned SFWM calculation.
#[derive(Debug, Error, PartialEq)]
pub enum SfwmSourceError {
    /// A named input is not finite.
    #[error("{field} must be finite, got {value}")]
    NonFinite { field: &'static str, value: f64 },
    /// A named input must be strictly positive.
    #[error("{field} must be positive, got {value}")]
    NonPositive { field: &'static str, value: f64 },
    /// A named input must be nonnegative.
    #[error("{field} must be nonnegative, got {value}")]
    Negative { field: &'static str, value: f64 },
    /// Eq. 6 is undefined when the SHG mismatch is exactly zero.
    #[error("Eq. 6 has an undefined SHG prefactor for zero Delta_k_SHG")]
    ZeroShgMismatch,
    /// A rate ratio has no finite denominator.
    #[error("R_cas/R_dir is undefined because the direct rate is zero")]
    UndefinedRateRatio,
}

fn require_finite(field: &'static str, value: f64) -> Result<(), SfwmSourceError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(SfwmSourceError::NonFinite { field, value })
    }
}

fn require_positive(field: &'static str, value: f64) -> Result<(), SfwmSourceError> {
    require_finite(field, value)?;
    if value > 0.0 {
        Ok(())
    } else {
        Err(SfwmSourceError::NonPositive { field, value })
    }
}

fn require_nonnegative(field: &'static str, value: f64) -> Result<(), SfwmSourceError> {
    require_finite(field, value)?;
    if value >= 0.0 {
        Ok(())
    } else {
        Err(SfwmSourceError::Negative { field, value })
    }
}

/// A signed wavevector mismatch in inverse micrometres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WavevectorMismatch {
    /// Signed mismatch in inverse micrometres.
    pub value_per_um: f64,
}

impl WavevectorMismatch {
    /// Construct a finite signed mismatch.
    pub fn new(value_per_um: f64) -> Result<Self, SfwmSourceError> {
        require_finite("wavevector mismatch", value_per_um)?;
        Ok(Self { value_per_um })
    }

    /// Return the nonlinear coherence length, or `None` for exact phase match.
    pub fn coherence_length_um(self) -> Option<f64> {
        if self.value_per_um == 0.0 {
            None
        } else {
            Some(PI / self.value_per_um.abs())
        }
    }
}

/// Signed mismatches for the three source processes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceWavevectorMismatches {
    /// Direct SFWM mismatch, `2*k_p - k_s - k_i`.
    pub sfwm: WavevectorMismatch,
    /// SHG mismatch, `2*k_p - k_sh`.
    pub shg: WavevectorMismatch,
    /// SPDC mismatch, `k_sh - k_s - k_i`.
    pub spdc: WavevectorMismatch,
}

impl SourceWavevectorMismatches {
    /// Return the algebraic closure defect of the source identity.
    pub fn identity_defect_per_um(self) -> f64 {
        self.sfwm.value_per_um - self.shg.value_per_um - self.spdc.value_per_um
    }

    /// Return the signed source identity residual as a dimensionless relative
    /// value against the largest participating mismatch.
    pub fn normalized_identity_defect(self) -> f64 {
        let scale = self
            .sfwm
            .value_per_um
            .abs()
            .max(self.shg.value_per_um.abs())
            .max(self.spdc.value_per_um.abs());
        if scale == 0.0 {
            0.0
        } else {
            self.identity_defect_per_um().abs() / scale
        }
    }
}

/// Printed coherence-length anchors from Son and Chekhova, in micrometres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceCoherenceAnchors {
    /// Printed SFWM coherence length.
    pub sfwm_um: f64,
    /// Printed SHG coherence length.
    pub shg_um: f64,
    /// Printed SPDC coherence length.
    pub spdc_um: f64,
}

impl SourceCoherenceAnchors {
    /// Return the anchors printed for the main source comparison.
    pub const fn son_chekhova() -> Self {
        Self {
            sfwm_um: 33.3,
            shg_um: 3.1,
            spdc_um: 3.4,
        }
    }

    fn validate(self) -> Result<(), SfwmSourceError> {
        require_positive("SFWM coherence length", self.sfwm_um)?;
        require_positive("SHG coherence length", self.shg_um)?;
        require_positive("SPDC coherence length", self.spdc_um)
    }
}

/// Audit of the signed mismatch resolution from rounded source anchors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceMismatchAudit {
    /// Printed source magnitudes.
    pub anchors: SourceCoherenceAnchors,
    /// Signed mismatches used by the coherent calculation.
    pub mismatches: SourceWavevectorMismatches,
    /// Coherence lengths implied by the signed calculation.
    pub derived_coherence_lengths_um: [f64; 3],
    /// Absolute difference between derived and printed coherence lengths.
    pub coherence_length_defects_um: [f64; 3],
}

impl SourceMismatchAudit {
    /// Resolve the source signs using `Delta_k_SFWM` and `Delta_k_SHG` as
    /// independent rounded anchors, then derive `Delta_k_SPDC` from closure.
    ///
    /// The resulting SHG mismatch is positive and the SPDC mismatch is
    /// negative. The source's printed SPDC length remains an audit anchor,
    /// rather than an independent value that can override the identity.
    pub fn from_source_anchors(anchors: SourceCoherenceAnchors) -> Result<Self, SfwmSourceError> {
        anchors.validate()?;
        let sfwm = WavevectorMismatch::new(PI / anchors.sfwm_um)?;
        let shg = WavevectorMismatch::new(PI / anchors.shg_um)?;
        let spdc = WavevectorMismatch::new(sfwm.value_per_um - shg.value_per_um)?;
        let mismatches = SourceWavevectorMismatches { sfwm, shg, spdc };
        let derived = [
            PI / sfwm.value_per_um.abs(),
            PI / shg.value_per_um.abs(),
            PI / spdc.value_per_um.abs(),
        ];
        let coherence_length_defects_um = [
            (derived[0] - anchors.sfwm_um).abs(),
            (derived[1] - anchors.shg_um).abs(),
            (derived[2] - anchors.spdc_um).abs(),
        ];
        Ok(Self {
            anchors,
            mismatches,
            derived_coherence_lengths_um: derived,
            coherence_length_defects_um,
        })
    }
}

/// Wavelength and material inputs used by Eqs. 6 and 8.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfwmSourceParameters {
    /// Effective chi^(2) in m/V.
    pub chi2_m_per_v: f64,
    /// Effective chi^(3) in m^2/V^2.
    pub chi3_m2_per_v2: f64,
    /// Pump field amplitude squared in the source amplitude normalization.
    pub pump_field_squared: f64,
    /// Pump refractive index.
    pub n_pump: f64,
    /// Signal refractive index.
    pub n_signal: f64,
    /// Idler refractive index.
    pub n_idler: f64,
    /// Second-harmonic refractive index.
    pub n_sh: f64,
    /// Pump wavelength in micrometres.
    pub lambda_pump_um: f64,
    /// Signal wavelength in micrometres.
    pub lambda_signal_um: f64,
    /// Idler wavelength in micrometres.
    pub lambda_idler_um: f64,
}

impl SfwmSourceParameters {
    /// Construct and validate a source parameter set.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        chi2_m_per_v: f64,
        chi3_m2_per_v2: f64,
        pump_field_squared: f64,
        n_pump: f64,
        n_signal: f64,
        n_idler: f64,
        n_sh: f64,
        lambda_pump_um: f64,
        lambda_signal_um: f64,
        lambda_idler_um: f64,
    ) -> Result<Self, SfwmSourceError> {
        require_finite("chi^(2)", chi2_m_per_v)?;
        require_finite("chi^(3)", chi3_m2_per_v2)?;
        require_nonnegative("pump field squared", pump_field_squared)?;
        require_positive("pump index", n_pump)?;
        require_positive("signal index", n_signal)?;
        require_positive("idler index", n_idler)?;
        require_positive("SH index", n_sh)?;
        require_positive("pump wavelength", lambda_pump_um)?;
        require_positive("signal wavelength", lambda_signal_um)?;
        require_positive("idler wavelength", lambda_idler_um)?;
        Ok(Self {
            chi2_m_per_v,
            chi3_m2_per_v2,
            pump_field_squared,
            n_pump,
            n_signal,
            n_idler,
            n_sh,
            lambda_pump_um,
            lambda_signal_um,
            lambda_idler_um,
        })
    }

    /// Validate a parameter set that was assembled without `new`.
    pub fn validate(&self) -> Result<(), SfwmSourceError> {
        require_finite("chi^(2)", self.chi2_m_per_v)?;
        require_finite("chi^(3)", self.chi3_m2_per_v2)?;
        require_nonnegative("pump field squared", self.pump_field_squared)?;
        require_positive("pump index", self.n_pump)?;
        require_positive("signal index", self.n_signal)?;
        require_positive("idler index", self.n_idler)?;
        require_positive("SH index", self.n_sh)?;
        require_positive("pump wavelength", self.lambda_pump_um)?;
        require_positive("signal wavelength", self.lambda_signal_um)?;
        require_positive("idler wavelength", self.lambda_idler_um)
    }

    /// Return the source wavelengths used in the main comparison.
    pub const fn source_wavelengths_um() -> (f64, f64, f64) {
        (1.030, 0.770, 1.550)
    }

    /// Compute signed mismatches from one declared index branch.
    pub fn wavevector_mismatches(&self) -> Result<SourceWavevectorMismatches, SfwmSourceError> {
        let k_pump = 2.0 * PI * self.n_pump / self.lambda_pump_um;
        let k_signal = 2.0 * PI * self.n_signal / self.lambda_signal_um;
        let k_idler = 2.0 * PI * self.n_idler / self.lambda_idler_um;
        let lambda_sh_um = self.lambda_pump_um / 2.0;
        let k_sh = 2.0 * PI * self.n_sh / lambda_sh_um;
        Ok(SourceWavevectorMismatches {
            sfwm: WavevectorMismatch::new(2.0 * k_pump - k_signal - k_idler)?,
            shg: WavevectorMismatch::new(2.0 * k_pump - k_sh)?,
            spdc: WavevectorMismatch::new(k_sh - k_signal - k_idler)?,
        })
    }
}

/// Complex source amplitudes from Eq. 6 and Eq. 8.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfwmSourceAmplitudes {
    /// F(L) for the direct SFWM mismatch.
    pub f_sfwm: Complex64,
    /// F(L) for the SHG mismatch.
    pub f_shg: Complex64,
    /// F(L) for the SPDC mismatch.
    pub f_spdc: Complex64,
    /// Cascaded amplitude from Eq. 6.
    pub a_cas: Complex64,
    /// Direct amplitude from Eq. 8.
    pub a_dir: Complex64,
}

/// Rate values derived from the two complex amplitudes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SfwmSourceRates {
    /// Cascaded rate factor `(4/9)*abs(A_cas)^2`.
    pub r_cas: f64,
    /// Direct rate factor `(9/16)*abs(A_dir)^2`.
    pub r_dir: f64,
    /// Ratio when the direct rate is nonzero.
    pub ratio_cas_to_dir: Option<f64>,
}

fn phase_matching_amplitude(
    mismatch: WavevectorMismatch,
    thickness_um: f64,
) -> Result<Complex64, SfwmSourceError> {
    require_finite("thickness", thickness_um)?;
    if thickness_um < 0.0 {
        return Err(SfwmSourceError::Negative {
            field: "thickness",
            value: thickness_um,
        });
    }
    let argument = mismatch.value_per_um * thickness_um / 2.0;
    let value = if argument == 0.0 {
        thickness_um
    } else {
        argument.sin() / (mismatch.value_per_um / 2.0)
    };
    Ok(Complex64::new(value, 0.0))
}

/// Compute the source amplitudes without collapsing complex phases.
pub fn source_amplitudes(
    params: &SfwmSourceParameters,
    mismatches: SourceWavevectorMismatches,
    thickness_um: f64,
) -> Result<SfwmSourceAmplitudes, SfwmSourceError> {
    params.validate()?;
    require_finite("thickness", thickness_um)?;
    if thickness_um < 0.0 {
        return Err(SfwmSourceError::Negative {
            field: "thickness",
            value: thickness_um,
        });
    }
    if mismatches.shg.value_per_um == 0.0 {
        return Err(SfwmSourceError::ZeroShgMismatch);
    }

    let f_sfwm = phase_matching_amplitude(mismatches.sfwm, thickness_um)?;
    let f_shg = phase_matching_amplitude(mismatches.shg, thickness_um)?;
    let f_spdc = phase_matching_amplitude(mismatches.spdc, thickness_um)?;
    let phase_spdc = Complex64::from_polar(1.0, mismatches.spdc.value_per_um * thickness_um / 2.0);
    let phase_shg = Complex64::from_polar(1.0, mismatches.shg.value_per_um * thickness_um / 2.0);
    let lambda_sh_um = params.lambda_pump_um / 2.0;
    let eq6_prefactor = 2.0 * PI * params.chi2_m_per_v.powi(2) * params.pump_field_squared
        / (params.n_sh * lambda_sh_um * mismatches.shg.value_per_um);
    let a_cas = eq6_prefactor * phase_spdc * (phase_shg * f_sfwm - f_spdc);
    let a_dir = params.chi3_m2_per_v2
        * params.pump_field_squared
        * Complex64::from_polar(1.0, mismatches.sfwm.value_per_um * thickness_um / 2.0)
        * f_sfwm;
    Ok(SfwmSourceAmplitudes {
        f_sfwm,
        f_shg,
        f_spdc,
        a_cas,
        a_dir,
    })
}

/// Compute the separate source rate factors and an honest ratio result.
pub fn source_rates(amplitudes: SfwmSourceAmplitudes) -> Result<SfwmSourceRates, SfwmSourceError> {
    let r_cas = (2.0_f64 / 3.0).powi(2) * amplitudes.a_cas.norm_sqr();
    let r_dir = (3.0_f64 / 4.0).powi(2) * amplitudes.a_dir.norm_sqr();
    require_finite("cascaded rate", r_cas)?;
    require_finite("direct rate", r_dir)?;
    let ratio_cas_to_dir = if r_dir == 0.0 {
        None
    } else {
        Some(r_cas / r_dir)
    };
    Ok(SfwmSourceRates {
        r_cas,
        r_dir,
        ratio_cas_to_dir,
    })
}

/// Compute source amplitudes and rates for one thickness.
pub fn evaluate_source_case(
    params: &SfwmSourceParameters,
    mismatches: SourceWavevectorMismatches,
    thickness_um: f64,
) -> Result<(SfwmSourceAmplitudes, SfwmSourceRates), SfwmSourceError> {
    let amplitudes = source_amplitudes(params, mismatches, thickness_um)?;
    let rates = source_rates(amplitudes)?;
    Ok((amplitudes, rates))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_parameters() -> SfwmSourceParameters {
        SfwmSourceParameters::new(
            2.5e-11, 1.5e-20, 1.0, 2.156, 2.19, 2.14, 2.20, 1.030, 0.770, 1.550,
        )
        .expect("valid source parameters")
    }

    #[test]
    fn signed_source_anchors_close_by_derivation() {
        let audit =
            SourceMismatchAudit::from_source_anchors(SourceCoherenceAnchors::son_chekhova())
                .expect("valid source anchors");
        assert!(audit.mismatches.shg.value_per_um > 0.0);
        assert!(audit.mismatches.spdc.value_per_um < 0.0);
        assert_eq!(audit.mismatches.identity_defect_per_um(), 0.0);
        assert!((audit.derived_coherence_lengths_um[2] - 3.4).abs() < 0.02);
    }

    #[test]
    fn sellmeier_mismatch_identity_closes() {
        let mismatches = test_parameters()
            .wavevector_mismatches()
            .expect("valid indices");
        assert!(mismatches.normalized_identity_defect() < 1e-12);
    }

    #[test]
    fn source_amplitudes_retain_eq6_prefactor_and_phase() {
        let params = test_parameters();
        let mismatches =
            SourceMismatchAudit::from_source_anchors(SourceCoherenceAnchors::son_chekhova())
                .expect("valid source anchors")
                .mismatches;
        let amplitudes = source_amplitudes(&params, mismatches, 10.0).expect("valid amplitudes");
        assert!(amplitudes.a_cas.im.abs() > 0.0);
        assert!(amplitudes.a_dir.im.abs() > 0.0);
        assert!(amplitudes.a_cas.norm_sqr() > 0.0);
        assert!(amplitudes.a_dir.norm_sqr() > 0.0);
    }

    #[test]
    fn zero_direct_rate_is_not_replaced_by_a_floor() {
        let mut params = test_parameters();
        params.chi3_m2_per_v2 = 0.0;
        let mismatches =
            SourceMismatchAudit::from_source_anchors(SourceCoherenceAnchors::son_chekhova())
                .expect("valid source anchors")
                .mismatches;
        let amplitudes = source_amplitudes(&params, mismatches, 10.0).expect("valid amplitudes");
        let rates = source_rates(amplitudes).expect("valid rates");
        assert_eq!(rates.ratio_cas_to_dir, None);
    }

    #[test]
    fn zero_shg_mismatch_is_an_explicit_error() {
        let params = test_parameters();
        let mismatch = WavevectorMismatch::new(0.0).expect("finite mismatch");
        let mismatches = SourceWavevectorMismatches {
            sfwm: mismatch,
            shg: mismatch,
            spdc: mismatch,
        };
        assert_eq!(
            source_amplitudes(&params, mismatches, 10.0),
            Err(SfwmSourceError::ZeroShgMismatch)
        );
    }

    #[test]
    fn negative_pump_field_squared_is_rejected() {
        let result = SfwmSourceParameters::new(
            2.5e-11, 1.5e-20, -1.0, 2.156, 2.19, 2.14, 2.20, 1.030, 0.770, 1.550,
        );
        assert!(matches!(result, Err(SfwmSourceError::Negative { .. })));
    }

    #[test]
    fn negative_thickness_is_rejected() {
        let params = test_parameters();
        let mismatches = params
            .wavevector_mismatches()
            .expect("valid source mismatches");
        assert!(matches!(
            source_amplitudes(&params, mismatches, -0.1),
            Err(SfwmSourceError::Negative { .. })
        ));
    }
}
