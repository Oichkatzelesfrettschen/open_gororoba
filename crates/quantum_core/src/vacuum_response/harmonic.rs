// SPDX-License-Identifier: MIT
//! Exact temporal filters for an undamped classical thermal harmonic mode.

use super::{
    GaussianCoherence, MatrixCorrection, ResponseError, checked_product, finite, positive,
    preserve_nonzero,
};

const BOLTZMANN_J_PER_K: f64 = 1.380_649e-23;

/// Ideal switching applied to angular-frequency fluctuations during evolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarmonicPulseSequence {
    /// Constant positive switching throughout the evolution.
    Ramsey,
    /// Positive switching followed by negative switching at half the duration.
    HahnEcho,
}

fn sinc(argument: f64) -> f64 {
    if argument.abs() < 1e-4 {
        let squared = argument * argument;
        1.0 - squared / 6.0 + squared * squared / 120.0
    } else {
        argument.sin() / argument
    }
}

/// Integrate the full classical harmonic trajectory through an ideal pulse filter.
///
/// The undamped mode has omega=sqrt(stiffness/mass). Thermal initial displacement
/// q0 and scaled velocity v0/omega are independent zero-mean Gaussian variables
/// with common variance sigma^2=k_B*T/stiffness. The trajectory is
/// q(t)=q0*cos(omega*t)+(v0/omega)*sin(omega*t). The weak linear probe has
/// angular-frequency shift G*q(t), with negligible mechanical backaction.
/// Inputs use kg, N/m, K, rad/s/m, and seconds respectively.
///
/// Exact integration, rather than sampled trajectory reconstruction, gives
/// V_R=sigma^2*(G*t)^2*sinc^2(omega*t/2) and
/// V_E=sigma^2*(G*t)^2*sinc^2(omega*t/4)*sin^2(omega*t/4).
/// These formulas integrate both initial Gaussian coordinates. Their analytic
/// covariance is positive semidefinite, so the returned covariance corrections
/// are zero. Mechanical frequency and pulse-filter zeros can change the ordering
/// across stiffness values; softening supplies no universal coherence ordering.
///
/// Coherence is exp(-V/2-t/(2*T1)); omitted T1 declares relaxation absent.
/// T1 and G are caller-supplied model values unless independent records establish
/// their calibration. Zero temperature gives zero *classical* phase variance;
/// ordered quantum zero-point noise requires its separate contract. Damping,
/// anharmonicity, non-Gaussian preparation and pulse imperfections require
/// additional models. Nonzero products lost to underflow, overflow and unresolved
/// harmonic phase return Arithmetic rather than a silent zero contribution.
pub fn thermal_harmonic_coherence(
    mass_kg: f64,
    stiffness_n_per_m: f64,
    temperature_k: f64,
    coupling_rad_per_s_per_m: f64,
    duration_s: f64,
    t1_s: Option<f64>,
    sequence: HarmonicPulseSequence,
) -> Result<GaussianCoherence, ResponseError> {
    positive(mass_kg)?;
    positive(stiffness_n_per_m)?;
    if !temperature_k.is_finite()
        || temperature_k < 0.0
        || !coupling_rad_per_s_per_m.is_finite()
        || !duration_s.is_finite()
        || duration_s < 0.0
    {
        return Err(ResponseError::Domain);
    }
    let relaxation = if let Some(t1) = t1_s {
        positive(t1)?;
        let half_duration = checked_product(duration_s, 0.5)?;
        preserve_nonzero(half_duration / t1, half_duration)?
    } else {
        0.0
    };
    let variance = if temperature_k == 0.0 || coupling_rad_per_s_per_m == 0.0 || duration_s == 0.0 {
        0.0
    } else {
        let omega = finite(stiffness_n_per_m.sqrt() / mass_kg.sqrt())?;
        positive(omega)?;
        let thermal_energy = checked_product(BOLTZMANN_J_PER_K, temperature_k)?;
        let displacement_variance =
            preserve_nonzero(thermal_energy / stiffness_n_per_m, thermal_energy)?;
        let phase = checked_product(omega, duration_s)?;
        let argument = checked_product(
            phase,
            match sequence {
                HarmonicPulseSequence::Ramsey => 0.5,
                HarmonicPulseSequence::HahnEcho => 0.25,
            },
        )?;
        let filter = sinc(argument);
        let coupled_time = checked_product(coupling_rad_per_s_per_m, duration_s)?;
        let mut variance = displacement_variance;
        for factor in [coupled_time, coupled_time, filter, filter] {
            variance = checked_product(variance, factor)?;
        }
        if sequence == HarmonicPulseSequence::HahnEcho {
            for _ in 0..2 {
                variance = checked_product(variance, argument.sin())?;
            }
        }
        variance
    };
    let exponent = finite(relaxation + checked_product(variance, 0.5)?)?;
    Ok(GaussianCoherence {
        phase_variance: variance,
        coherence: (-exponent).exp(),
        covariance_correction: MatrixCorrection {
            symmetry_relative: 0.0,
            psd_projection_relative: 0.0,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;
    const K_B: f64 = 1.380_649e-23;

    fn quadrature_basis(duration: f64, omega: f64, sequence: HarmonicPulseSequence) -> (f64, f64) {
        let intervals = match sequence {
            HarmonicPulseSequence::Ramsey => vec![(0.0, duration, 1.0)],
            HarmonicPulseSequence::HahnEcho => {
                vec![(0.0, duration / 2.0, 1.0), (duration / 2.0, duration, -1.0)]
            }
        };
        let mut cosine = 0.0;
        let mut sine = 0.0;
        for (start, end, sign) in intervals {
            let step = (end - start) / 4096.0;
            for index in 0..=4096 {
                let coefficient = if index == 0 || index == 4096 {
                    1.0
                } else if index % 2 == 0 {
                    2.0
                } else {
                    4.0
                };
                let time = start + index as f64 * step;
                cosine += sign * coefficient * step * (omega * time).cos() / 3.0;
                sine += sign * coefficient * step * (omega * time).sin() / 3.0;
            }
        }
        (cosine, sine)
    }

    #[test]
    fn harmonic_filters_match_independent_fine_basis_quadrature() {
        for sequence in [
            HarmonicPulseSequence::Ramsey,
            HarmonicPulseSequence::HahnEcho,
        ] {
            let result =
                thermal_harmonic_coherence(2.0, 18.0, 300.0, 1e11, 1.3, None, sequence).unwrap();
            let (cosine, sine) = quadrature_basis(1.3, 3.0, sequence);
            let expected = K_B * 300.0 / 18.0 * 1e22 * (cosine * cosine + sine * sine);
            assert!((result.phase_variance / expected - 1.0).abs() < 1e-11);
        }
    }

    #[test]
    fn quasistatic_limit_recovers_ramsey_and_suppresses_echo() {
        let duration = 1e-5;
        let ramsey = thermal_harmonic_coherence(
            1.0,
            1.0,
            1.0 / K_B,
            1.0,
            duration,
            None,
            HarmonicPulseSequence::Ramsey,
        )
        .unwrap();
        let echo = thermal_harmonic_coherence(
            1.0,
            1.0,
            1.0 / K_B,
            1.0,
            duration,
            None,
            HarmonicPulseSequence::HahnEcho,
        )
        .unwrap();
        assert!((ramsey.phase_variance / duration.powi(2) - 1.0).abs() < 1e-10);
        assert!((echo.phase_variance / (duration.powi(4) / 16.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn zero_classical_temperature_coupling_or_duration_leaves_relaxation_only() {
        for (temperature, coupling, duration) in
            [(0.0, 1.0, 1.0), (300.0, 0.0, 1.0), (300.0, 1.0, 0.0)]
        {
            for sequence in [
                HarmonicPulseSequence::Ramsey,
                HarmonicPulseSequence::HahnEcho,
            ] {
                let result = thermal_harmonic_coherence(
                    1.0,
                    1.0,
                    temperature,
                    coupling,
                    duration,
                    Some(2.0),
                    sequence,
                )
                .unwrap();
                assert_eq!(result.phase_variance, 0.0);
                assert_eq!(result.coherence, (-duration / 4.0).exp());
                assert_eq!(result.covariance_correction.symmetry_relative, 0.0);
                assert_eq!(result.covariance_correction.psd_projection_relative, 0.0);
            }
        }
    }

    #[test]
    fn period_nulls_and_echo_resonance_preclude_universal_softening_order() {
        let evaluate = |duration, sequence| {
            thermal_harmonic_coherence(1.0, 1.0, 1.0 / K_B, 1.0, duration, None, sequence).unwrap()
        };
        let ramsey_null = evaluate(2.0 * PI, HarmonicPulseSequence::Ramsey);
        let echo_null = evaluate(4.0 * PI, HarmonicPulseSequence::HahnEcho);
        let echo_maximum = evaluate(2.0 * PI, HarmonicPulseSequence::HahnEcho);
        assert!(ramsey_null.phase_variance < 1e-28);
        assert!(echo_null.phase_variance < 1e-28);
        assert!((echo_maximum.phase_variance - 16.0).abs() < 1e-12);
    }

    #[test]
    fn harmonic_domains_and_unrepresentable_products_reject() {
        let evaluate = |mass, stiffness, temperature, coupling, duration, relaxation| {
            thermal_harmonic_coherence(
                mass,
                stiffness,
                temperature,
                coupling,
                duration,
                relaxation,
                HarmonicPulseSequence::Ramsey,
            )
        };
        for result in [
            evaluate(0.0, 1.0, 300.0, 1.0, 1.0, None),
            evaluate(1.0, -1.0, 300.0, 1.0, 1.0, None),
            evaluate(1.0, 1.0, -1.0, 1.0, 1.0, None),
            evaluate(1.0, 1.0, 300.0, f64::NAN, 1.0, None),
            evaluate(1.0, 1.0, 300.0, 1.0, -1.0, None),
            evaluate(1.0, 1.0, 300.0, 1.0, 1.0, Some(0.0)),
        ] {
            assert!(result.is_err());
        }
        assert!(evaluate(1.0, 1.0, 1.0 / K_B, f64::MAX, 2.0, None).is_err());
    }
}
