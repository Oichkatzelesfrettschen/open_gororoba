//! Source-faithful temporal coupled-mode channels for Ruan and Fan.
//!
//! The validated path keeps the complex reflection and scattering amplitudes
//! until all channel observables are evaluated. Scattering, absorption, and
//! extinction use independent definitions. The balance relation is a check,
//! not the constructor for extinction.
//!
//! The source uses exp(-i*omega*t), incoming H^(2), outgoing H^(1),
//! R_l = h_l^- / h_l^+, and S_l = (R_l - 1)/2. See Ruan and Fan,
//! "Temporal coupled-mode theory for Fano resonance in light scattering by a
//! single obstacle", arXiv:0909.3323v2, equations 1-23.

use num_complex::Complex64;
use thiserror::Error;

/// Parameters for one dimensional TCMT channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FanoChannel {
    /// Resonance angular frequency omega_0.
    pub omega_0: f64,
    /// Radiative decay rate gamma.
    pub gamma: f64,
    /// Intrinsic decay rate gamma_0.
    pub gamma_0: f64,
    /// Real background phase phi.
    pub phi: f64,
    /// Angular momentum channel index l.
    pub l: i32,
}

/// Source-local name for the validated channel parameter record.
pub type FanoChannelParameters = FanoChannel;

/// Dimensionless line-shape parameters for source Fig. 1 and Fig. 2.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DimensionlessFanoChannel {
    /// Intrinsic to radiative damping ratio gamma_0/gamma.
    pub gamma_0_over_gamma: f64,
    /// Real background phase phi.
    pub phi: f64,
}

/// Drude permittivity model parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FanoDrudeParams {
    /// Plasma frequency omega_p.
    pub omega_p: f64,
    /// Damping rate gamma_d.
    pub gamma_d: f64,
}

/// Error returned by the validated analytical channel path.
#[derive(Debug, Error, Clone, PartialEq)]
pub enum FanoChannelError {
    #[error("{field} must be finite")]
    NonFinite { field: &'static str },
    #[error("omega_0 must be positive")]
    NonPositiveResonance,
    #[error("gamma must be positive")]
    NonPositiveRadiativeRate,
    #[error("gamma_0 must be nonnegative")]
    NegativeIntrinsicRate,
    #[error("gamma_0/gamma must be nonnegative")]
    NegativeIntrinsicRatio,
    #[error("the channel denominator is singular")]
    SingularDenominator,
    #[error("the Drude denominator is singular")]
    SingularDrudeDenominator,
}

/// Complex channel amplitudes retained before scalar observables.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelAmplitudes {
    /// Reflection amplitude R_l.
    pub reflection: Complex64,
    /// Scattering amplitude S_l.
    pub scattering: Complex64,
}

/// Normalized observables for one channel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelCrossSections {
    /// Normalized scattering observable.
    pub scattering: f64,
    /// Normalized absorption observable.
    pub absorption: f64,
    /// Normalized extinction observable.
    pub extinction: f64,
}

/// Compatibility view retaining the optics_core field names.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CrossSections {
    /// Normalized scattering observable.
    pub c_sct: f64,
    /// Normalized absorption observable.
    pub c_abs: f64,
    /// Normalized extinction observable.
    pub c_ext: f64,
}

impl From<ChannelCrossSections> for CrossSections {
    fn from(value: ChannelCrossSections) -> Self {
        Self {
            c_sct: value.scattering,
            c_abs: value.absorption,
            c_ext: value.extinction,
        }
    }
}

/// Component values and defects for the independent channel observables.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelObservableResiduals {
    /// Absorption computed from S_l.
    pub absorption_from_s: f64,
    /// Absorption computed from the R_l flux defect.
    pub absorption_from_r: f64,
    /// Absorption from the source closed form.
    pub absorption_closed_form: f64,
    /// Extinction computed from -Re(S_l).
    pub extinction_from_s: f64,
    /// Defect of extinction - scattering - absorption_from_s.
    pub balance_defect: f64,
    /// Maximum pairwise absorption representation defect.
    pub absorption_representation_defect: f64,
    /// Defect between the R_l flux and closed-form absorption.
    pub flux_representation_defect: f64,
}

/// Evaluated complex amplitudes, observables, and component-level defects.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelEvaluation {
    /// Complex amplitudes.
    pub amplitudes: ChannelAmplitudes,
    /// Independently defined observables.
    pub cross_sections: ChannelCrossSections,
    /// Component values and defects retained before norms.
    pub residuals: ChannelObservableResiduals,
}

/// Unreduced source coupling values from the temporal coupled-mode equations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SourceCouplingParameters {
    /// Background reflection coefficient B.
    pub background: Complex64,
    /// Input coupling coefficient kappa.
    pub kappa: Complex64,
    /// Output coupling coefficient eta.
    pub eta: Complex64,
    /// Radiative decay rate gamma.
    pub gamma: f64,
    /// Resonance frequency used for the validity ratio.
    pub omega_0: f64,
    /// Intrinsic loss rate used for the validity ratio.
    pub gamma_0: f64,
}

/// Separate residuals for source constraints and amplitude constraints.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChannelConstraintResiduals {
    /// Difference between |R|^2 and one in the lossless case.
    pub lossless_flux_defect: f64,
    /// Positive excess of |R|^2 above one in a passive case.
    pub passive_contractivity_excess: f64,
    /// Defect in |eta|^2 = 2*gamma.
    pub eta_norm_defect: f64,
    /// Defect in kappa*conjugate(eta) = 2*gamma.
    pub time_reversal_drive_defect: Complex64,
    /// Defect in B*conjugate(eta) + eta = 0.
    pub time_reversal_background_defect: Complex64,
    /// Defect in kappa = eta.
    pub reciprocal_coupling_defect: Complex64,
    /// TCMT validity ratio (gamma + gamma_0)/omega_0.
    pub validity_ratio: f64,
}

impl FanoChannel {
    /// Validate a dimensional channel and the frequency at which it is used.
    pub fn validate(&self, omega: f64) -> Result<(), FanoChannelError> {
        for (field, value) in [
            ("omega_0", self.omega_0),
            ("gamma", self.gamma),
            ("gamma_0", self.gamma_0),
            ("phi", self.phi),
            ("omega", omega),
        ] {
            if !value.is_finite() {
                return Err(FanoChannelError::NonFinite { field });
            }
        }
        if self.omega_0 <= 0.0 {
            return Err(FanoChannelError::NonPositiveResonance);
        }
        if self.gamma <= 0.0 {
            return Err(FanoChannelError::NonPositiveRadiativeRate);
        }
        if self.gamma_0 < 0.0 {
            return Err(FanoChannelError::NegativeIntrinsicRate);
        }
        let denominator = Complex64::new(self.gamma_0 + self.gamma, self.omega_0 - omega);
        if denominator.norm_sqr() == 0.0 {
            return Err(FanoChannelError::SingularDenominator);
        }
        Ok(())
    }
}

impl DimensionlessFanoChannel {
    /// Validate dimensionless line-shape inputs.
    pub fn validate(&self, x: f64) -> Result<(), FanoChannelError> {
        if !self.gamma_0_over_gamma.is_finite() {
            return Err(FanoChannelError::NonFinite {
                field: "gamma_0_over_gamma",
            });
        }
        if !self.phi.is_finite() {
            return Err(FanoChannelError::NonFinite { field: "phi" });
        }
        if !x.is_finite() {
            return Err(FanoChannelError::NonFinite { field: "x" });
        }
        if self.gamma_0_over_gamma < 0.0 {
            return Err(FanoChannelError::NegativeIntrinsicRatio);
        }
        Ok(())
    }
}

impl SourceCouplingParameters {
    fn validate(&self) -> Result<(), FanoChannelError> {
        for (field, value) in [
            ("gamma", self.gamma),
            ("omega_0", self.omega_0),
            ("gamma_0", self.gamma_0),
        ] {
            if !value.is_finite() {
                return Err(FanoChannelError::NonFinite { field });
            }
        }
        if !self.background.re.is_finite() || !self.background.im.is_finite() {
            return Err(FanoChannelError::NonFinite {
                field: "background",
            });
        }
        if !self.kappa.re.is_finite() || !self.kappa.im.is_finite() {
            return Err(FanoChannelError::NonFinite { field: "kappa" });
        }
        if !self.eta.re.is_finite() || !self.eta.im.is_finite() {
            return Err(FanoChannelError::NonFinite { field: "eta" });
        }
        if self.gamma <= 0.0 {
            return Err(FanoChannelError::NonPositiveRadiativeRate);
        }
        if self.omega_0 <= 0.0 {
            return Err(FanoChannelError::NonPositiveResonance);
        }
        if self.gamma_0 < 0.0 {
            return Err(FanoChannelError::NegativeIntrinsicRate);
        }
        Ok(())
    }
}

impl FanoDrudeParams {
    /// Validate the dimensional Drude parameters.
    pub fn validate(&self) -> Result<(), FanoChannelError> {
        if !self.omega_p.is_finite() || !self.gamma_d.is_finite() {
            return Err(FanoChannelError::NonFinite {
                field: "Drude parameter",
            });
        }
        if self.omega_p <= 0.0 {
            return Err(FanoChannelError::NonPositiveResonance);
        }
        if self.gamma_d < 0.0 {
            return Err(FanoChannelError::NegativeIntrinsicRate);
        }
        Ok(())
    }
}

/// Validated complex reflection amplitude R_l from source Eq. 21.
pub fn try_fano_reflection(
    channel: &FanoChannel,
    omega: f64,
) -> Result<Complex64, FanoChannelError> {
    channel.validate(omega)?;
    let delta = channel.omega_0 - omega;
    let numerator = Complex64::new(channel.gamma_0 - channel.gamma, delta);
    let denominator = Complex64::new(channel.gamma_0 + channel.gamma, delta);
    let phase = Complex64::new(0.0, channel.phi).exp();
    Ok(phase * numerator / denominator)
}

/// Compatibility wrapper for source Eq. 21.
pub fn fano_reflection(channel: &FanoChannel, omega: f64) -> Complex64 {
    try_fano_reflection(channel, omega)
        .unwrap_or_else(|error| panic!("invalid Fano channel input: {error}"))
}

/// Validated complex scattering amplitude S_l from source Eq. 8.
pub fn try_scattering_coefficient(
    channel: &FanoChannel,
    omega: f64,
) -> Result<Complex64, FanoChannelError> {
    Ok((try_fano_reflection(channel, omega)? - 1.0) / 2.0)
}

/// Compatibility wrapper for source Eq. 8.
pub fn scattering_coefficient(channel: &FanoChannel, omega: f64) -> Complex64 {
    try_scattering_coefficient(channel, omega)
        .unwrap_or_else(|error| panic!("invalid Fano channel input: {error}"))
}

/// Evaluate the independent source observables for one channel.
pub fn evaluate_channel(
    channel: &FanoChannel,
    omega: f64,
) -> Result<ChannelEvaluation, FanoChannelError> {
    let reflection = try_fano_reflection(channel, omega)?;
    let scattering = (reflection - 1.0) / 2.0;
    let scattering_observable = scattering.norm_sqr();
    let absorption_from_s = -(scattering.re + scattering_observable);
    let absorption_from_r = (1.0 - reflection.norm_sqr()) / 4.0;
    let detuning = omega - channel.omega_0;
    let denominator = detuning * detuning + (channel.gamma_0 + channel.gamma).powi(2);
    if denominator == 0.0 {
        return Err(FanoChannelError::SingularDenominator);
    }
    let absorption_closed_form = channel.gamma_0 * channel.gamma / denominator;
    let extinction_from_s = -scattering.re;
    let balance_defect = extinction_from_s - scattering_observable - absorption_from_s;
    let absorption_representation_defect = (absorption_from_s - absorption_from_r)
        .abs()
        .max((absorption_from_s - absorption_closed_form).abs());
    let flux_representation_defect = (absorption_from_r - absorption_closed_form).abs();

    let cross_sections = ChannelCrossSections {
        scattering: scattering_observable,
        absorption: absorption_from_s,
        extinction: extinction_from_s,
    };
    Ok(ChannelEvaluation {
        amplitudes: ChannelAmplitudes {
            reflection,
            scattering,
        },
        cross_sections,
        residuals: ChannelObservableResiduals {
            absorption_from_s,
            absorption_from_r,
            absorption_closed_form,
            extinction_from_s,
            balance_defect,
            absorption_representation_defect,
            flux_representation_defect,
        },
    })
}

/// Evaluate a dimensionless source line shape without a zero resonance.
pub fn evaluate_dimensionless_channel(
    channel: &DimensionlessFanoChannel,
    x: f64,
) -> Result<ChannelEvaluation, FanoChannelError> {
    channel.validate(x)?;
    let dimensional = FanoChannel {
        omega_0: 1.0,
        gamma: 1.0,
        gamma_0: channel.gamma_0_over_gamma,
        phi: channel.phi,
        l: 0,
    };
    evaluate_channel(&dimensional, 1.0 + x)
}

/// Evaluate the source coupling constraints independently.
pub fn evaluate_source_constraints(
    channel: &FanoChannel,
    coupling: &SourceCouplingParameters,
    omega: f64,
) -> Result<ChannelConstraintResiduals, FanoChannelError> {
    coupling.validate()?;
    let reflection = try_fano_reflection(channel, omega)?;
    let lossless_flux_defect = reflection.norm_sqr() - 1.0;
    let passive_contractivity_excess = (reflection.norm_sqr() - 1.0).max(0.0);
    Ok(ChannelConstraintResiduals {
        lossless_flux_defect,
        passive_contractivity_excess,
        eta_norm_defect: coupling.eta.norm_sqr() - 2.0 * coupling.gamma,
        time_reversal_drive_defect: coupling.kappa * coupling.eta.conj()
            - Complex64::new(2.0 * coupling.gamma, 0.0),
        time_reversal_background_defect: coupling.background * coupling.eta.conj() + coupling.eta,
        reciprocal_coupling_defect: coupling.kappa - coupling.eta,
        validity_ratio: (coupling.gamma + coupling.gamma_0) / coupling.omega_0,
    })
}

/// Evaluate each channel and return the independently summed observables.
pub fn try_multi_channel_evaluations(
    channels: &[FanoChannel],
    omega: f64,
) -> Result<(Vec<ChannelEvaluation>, CrossSections), FanoChannelError> {
    let evaluations: Vec<ChannelEvaluation> = channels
        .iter()
        .map(|channel| evaluate_channel(channel, omega))
        .collect::<Result<_, _>>()?;
    let mut totals = ChannelCrossSections {
        scattering: 0.0,
        absorption: 0.0,
        extinction: 0.0,
    };
    for evaluation in &evaluations {
        totals.scattering += evaluation.cross_sections.scattering;
        totals.absorption += evaluation.cross_sections.absorption;
        totals.extinction += evaluation.cross_sections.extinction;
    }
    Ok((evaluations, totals.into()))
}

/// Corrected single-channel observables in the compatibility field layout.
pub fn fano_cross_sections_normalized(channel: &FanoChannel, omega: f64) -> CrossSections {
    evaluate_channel(channel, omega)
        .unwrap_or_else(|error| panic!("invalid Fano channel input: {error}"))
        .cross_sections
        .into()
}

/// Corrected multi-channel observables in the compatibility field layout.
pub fn multi_channel_cross_sections(channels: &[FanoChannel], omega: f64) -> CrossSections {
    try_multi_channel_evaluations(channels, omega)
        .unwrap_or_else(|error| panic!("invalid Fano channel input: {error}"))
        .1
}

/// Normalized scattering cross-section for the dimensionless source line shape.
pub fn normalized_fano_c_sct(x: f64, phi: f64, gamma_0_over_gamma: f64) -> f64 {
    evaluate_dimensionless_channel(
        &DimensionlessFanoChannel {
            gamma_0_over_gamma,
            phi,
        },
        x,
    )
    .unwrap_or_else(|error| panic!("invalid dimensionless Fano input: {error}"))
    .cross_sections
    .scattering
}

/// Normalized absorption cross-section for the dimensionless source line shape.
pub fn normalized_fano_c_abs(x: f64, phi: f64, gamma_0_over_gamma: f64) -> f64 {
    evaluate_dimensionless_channel(
        &DimensionlessFanoChannel {
            gamma_0_over_gamma,
            phi,
        },
        x,
    )
    .unwrap_or_else(|error| panic!("invalid dimensionless Fano input: {error}"))
    .cross_sections
    .absorption
}

/// Normalized extinction cross-section for the dimensionless source line shape.
pub fn normalized_fano_c_ext(x: f64, phi: f64, gamma_0_over_gamma: f64) -> f64 {
    evaluate_dimensionless_channel(
        &DimensionlessFanoChannel {
            gamma_0_over_gamma,
            phi,
        },
        x,
    )
    .unwrap_or_else(|error| panic!("invalid dimensionless Fano input: {error}"))
    .cross_sections
    .extinction
}

/// Validated Drude permittivity.
pub fn try_drude_epsilon(
    params: &FanoDrudeParams,
    omega: f64,
) -> Result<Complex64, FanoChannelError> {
    params.validate()?;
    if !omega.is_finite() {
        return Err(FanoChannelError::NonFinite { field: "omega" });
    }
    let omega_c = Complex64::new(omega, 0.0);
    let denominator = omega_c * omega_c + Complex64::new(0.0, params.gamma_d * omega);
    if denominator.norm_sqr() == 0.0 {
        return Err(FanoChannelError::SingularDrudeDenominator);
    }
    Ok(Complex64::new(1.0, 0.0) - params.omega_p * params.omega_p / denominator)
}

/// Compatibility Drude permittivity wrapper.
pub fn drude_epsilon(params: &FanoDrudeParams, omega: f64) -> Complex64 {
    try_drude_epsilon(params, omega).unwrap_or_else(|error| panic!("invalid Drude input: {error}"))
}

/// Fano asymmetry parameter q = -cot(phi/2).
pub fn fano_q(phi: f64) -> f64 {
    -(phi / 2.0).cos() / (phi / 2.0).sin()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn source_channel(phi: f64, gamma_0: f64) -> FanoChannel {
        FanoChannel {
            omega_0: 1.0,
            gamma: 1.0,
            gamma_0,
            phi,
            l: 0,
        }
    }

    #[test]
    fn lossless_phi_zero_lorentzian() {
        let c_at_resonance = normalized_fano_c_sct(0.0, 0.0, 0.0);
        assert!((c_at_resonance - 1.0).abs() < 1e-12);
        let c_far = normalized_fano_c_sct(100.0, 0.0, 0.0);
        assert!(c_far < 1e-4);
    }

    #[test]
    fn lossless_phi_pi_antiresonance() {
        let c_at_resonance = normalized_fano_c_sct(0.0, PI, 0.0);
        assert!(c_at_resonance < 1e-12);
    }

    #[test]
    fn absorption_is_phase_independent_and_even() {
        for phi in [0.0, PI / 2.0, PI, -PI / 2.0] {
            for x in [-10.0, -3.0, 0.0, 3.0, 10.0] {
                let positive = evaluate_dimensionless_channel(
                    &DimensionlessFanoChannel {
                        gamma_0_over_gamma: 1.0,
                        phi,
                    },
                    x,
                )
                .expect("valid dimensionless channel");
                let zero_phase = evaluate_dimensionless_channel(
                    &DimensionlessFanoChannel {
                        gamma_0_over_gamma: 1.0,
                        phi: 0.0,
                    },
                    x,
                )
                .expect("valid dimensionless channel");
                assert!(
                    (positive.cross_sections.absorption - zero_phase.cross_sections.absorption)
                        .abs()
                        < 1e-12
                );
            }
            let positive = normalized_fano_c_abs(3.0, phi, 1.0);
            let negative = normalized_fano_c_abs(-3.0, phi, 1.0);
            assert!((positive - negative).abs() < 1e-12);
        }
    }

    #[test]
    fn equal_damping_peak_absorption_is_one_quarter() {
        let evaluation = evaluate_dimensionless_channel(
            &DimensionlessFanoChannel {
                gamma_0_over_gamma: 1.0,
                phi: PI / 2.0,
            },
            0.0,
        )
        .expect("valid dimensionless channel");
        assert!((evaluation.cross_sections.absorption - 0.25).abs() < 1e-12);
        assert!((evaluation.residuals.absorption_from_r - 0.25).abs() < 1e-12);
        assert!((evaluation.residuals.absorption_closed_form - 0.25).abs() < 1e-12);
    }

    #[test]
    fn independent_observables_satisfy_balance() {
        for x in [-5.0, -1.0, 0.0, 0.5, 3.0] {
            for phi in [0.0, PI / 4.0, PI / 2.0, PI] {
                let evaluation =
                    evaluate_channel(&source_channel(phi, 0.5), 1.0 + x).expect("valid channel");
                assert!(evaluation.residuals.balance_defect.abs() < 1e-14);
                assert!(evaluation.residuals.absorption_representation_defect < 1e-14);
                assert!(evaluation.residuals.flux_representation_defect < 1e-14);
            }
        }
    }

    #[test]
    fn source_constraints_are_separate_predicates() {
        let phi = PI / 3.0;
        let gamma: f64 = 0.25;
        let eta = (2.0 * gamma).sqrt() * Complex64::new(0.0, phi / 2.0 + PI / 2.0).exp();
        let coupling = SourceCouplingParameters {
            background: Complex64::new(0.0, phi).exp(),
            kappa: eta,
            eta,
            gamma,
            omega_0: 2.0,
            gamma_0: 0.0,
        };
        let residuals = evaluate_source_constraints(&source_channel(phi, 0.0), &coupling, 2.0)
            .expect("valid source coupling");
        assert!(residuals.lossless_flux_defect.abs() < 1e-12);
        assert!(residuals.passive_contractivity_excess.abs() < 1e-12);
        assert!(residuals.eta_norm_defect.abs() < 1e-12);
        assert!(residuals.time_reversal_drive_defect.norm() < 1e-12);
        assert!(residuals.time_reversal_background_defect.norm() < 1e-12);
        assert!(residuals.reciprocal_coupling_defect.norm() < 1e-12);
    }

    #[test]
    fn old_absorption_oracle_is_detected() {
        let evaluation = evaluate_dimensionless_channel(
            &DimensionlessFanoChannel {
                gamma_0_over_gamma: 1.0,
                phi: 0.0,
            },
            0.0,
        )
        .expect("valid dimensionless channel");
        let old_absorption = -evaluation.amplitudes.scattering.re;
        assert!((old_absorption - evaluation.cross_sections.absorption).abs() > 0.2);
    }

    #[test]
    fn amplitude_mutations_are_detected() {
        let channel = source_channel(PI / 2.0, 0.1);
        let evaluation = evaluate_channel(&channel, 1.5).expect("valid channel");
        let plus_scattering = (evaluation.amplitudes.reflection + 1.0) / 2.0;
        assert!((plus_scattering - evaluation.amplitudes.scattering).norm() > 0.5);
        let reversed_detuning = evaluate_channel(&channel, 0.5).expect("valid channel");
        assert!(
            (reversed_detuning.amplitudes.scattering - evaluation.amplitudes.scattering).norm()
                > 1e-3
        );
        let conjugated_phase = Complex64::new(
            evaluation.amplitudes.scattering.re,
            -evaluation.amplitudes.scattering.im,
        );
        assert!((conjugated_phase - evaluation.amplitudes.scattering).norm() > 1e-3);
        let corrupted_extinction =
            evaluation.cross_sections.scattering + evaluation.cross_sections.absorption + 0.25;
        assert!((corrupted_extinction - evaluation.residuals.extinction_from_s).abs() > 0.2);
    }

    #[test]
    fn source_constraint_mutations_are_detected() {
        let phi = PI / 3.0;
        let gamma: f64 = 0.25;
        let eta = (2.0 * gamma).sqrt() * Complex64::new(0.0, phi / 2.0 + PI / 2.0).exp();
        let coupling = SourceCouplingParameters {
            background: Complex64::new(0.0, phi).exp(),
            kappa: eta,
            eta,
            gamma,
            omega_0: 2.0,
            gamma_0: 0.0,
        };
        let channel = FanoChannel {
            omega_0: 2.0,
            gamma,
            gamma_0: 0.0,
            phi,
            l: 0,
        };

        let mut changed_kappa = coupling;
        changed_kappa.kappa += Complex64::new(0.2, 0.1);
        let residuals = evaluate_source_constraints(&channel, &changed_kappa, 2.0)
            .expect("valid mutated coupling");
        assert!(residuals.time_reversal_drive_defect.norm() > 1e-3);
        assert!(residuals.reciprocal_coupling_defect.norm() > 1e-3);

        let mut changed_background = coupling;
        changed_background.background *= Complex64::new(0.0, 0.2).exp();
        assert!((changed_background.background.norm() - coupling.background.norm()).abs() < 1e-12);
        let residuals = evaluate_source_constraints(&channel, &changed_background, 2.0)
            .expect("valid phase-mutated coupling");
        assert!(residuals.time_reversal_background_defect.norm() > 1e-3);

        let mut changed_eta = coupling;
        changed_eta.eta *= 1.1;
        let residuals = evaluate_source_constraints(&channel, &changed_eta, 2.0)
            .expect("valid eta-mutated coupling");
        assert!(residuals.eta_norm_defect.abs() > 1e-3);
    }

    #[test]
    fn intrinsic_loss_removes_exact_lossless_extrema() {
        let lossless_peak = evaluate_channel(&source_channel(0.0, 0.0), 1.0)
            .expect("valid lossless channel")
            .cross_sections
            .scattering;
        let lossy_peak = evaluate_channel(&source_channel(0.0, 0.1), 1.0)
            .expect("valid lossy channel")
            .cross_sections
            .scattering;
        let lossless_zero = evaluate_channel(&source_channel(PI, 0.0), 1.0)
            .expect("valid lossless antiresonance")
            .cross_sections
            .scattering;
        let lossy_zero = evaluate_channel(&source_channel(PI, 0.1), 1.0)
            .expect("valid lossy antiresonance")
            .cross_sections
            .scattering;
        assert!((lossless_peak - 1.0).abs() < 1e-12);
        assert!(lossy_peak < lossless_peak);
        assert!(lossless_zero < 1e-12);
        assert!(lossy_zero > lossless_zero);
    }

    #[test]
    fn invalid_channel_inputs_are_rejected() {
        let mut channel = source_channel(0.0, 0.0);
        channel.gamma_0 = -1.0;
        assert_eq!(
            evaluate_channel(&channel, 1.0),
            Err(FanoChannelError::NegativeIntrinsicRate)
        );
        channel.gamma_0 = 0.0;
        channel.gamma = 0.0;
        assert_eq!(
            evaluate_channel(&channel, 1.0),
            Err(FanoChannelError::NonPositiveRadiativeRate)
        );
    }

    #[test]
    fn background_limit() {
        let channel = FanoChannel {
            omega_0: 10.0,
            gamma: 0.01,
            gamma_0: 0.005,
            phi: PI / 3.0,
            l: 0,
        };
        let reflection = fano_reflection(&channel, 10.0 + 10000.0 * channel.gamma);
        let expected = Complex64::new(0.0, channel.phi).exp();
        assert!((reflection - expected).norm() < 1e-3);
    }

    #[test]
    fn fano_parameter_values() {
        assert!((fano_q(PI / 2.0) + 1.0).abs() < 1e-12);
        assert!(fano_q(PI).abs() < 1e-12);
    }
}
