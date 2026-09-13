//! Complete-positivity checks for reduced quantum-channel models.
//!
//! Positive scalar memory-kernel weights constrain a classical convolution.
//! They do not by themselves prove that the induced quantum map is completely
//! positive and trace preserving.

/// Smallest eigenvalue of the normalized Choi matrix for a qubit depolarizing
/// channel with Bloch-vector eigenvalue lambda.
pub fn qubit_depolarizing_minimum_choi_eigenvalue(lambda: f64) -> f64 {
    ((1.0 + 3.0 * lambda) / 4.0).min((1.0 - lambda) / 4.0)
}

/// Test complete positivity for a trace-preserving qubit depolarizing channel.
pub fn qubit_depolarizing_is_cptp(lambda: f64, tolerance: f64) -> bool {
    lambda.is_finite()
        && tolerance.is_finite()
        && tolerance >= 0.0
        && qubit_depolarizing_minimum_choi_eigenvalue(lambda) >= -tolerance
}

/// Depolarizing eigenvalue induced by `K(t) = weight*exp(-decay*t)`.
///
/// The convolution equation reduces to
/// `lambda'' + decay*lambda' + weight*lambda = 0` with `lambda(0)=1` and
/// `lambda'(0)=0`. `time` and `decay` use reciprocal units, while `weight`
/// uses inverse-time-squared units. Rescaling the time unit by a factor `s`
/// therefore maps `(weight, decay, time)` to
/// `(weight*s^2, decay*s, time/s)`. Positive weight and decay do not imply a
/// CPTP channel.
pub fn exponential_memory_depolarizing_eigenvalue(
    weight: f64,
    decay: f64,
    time: f64,
) -> Option<f64> {
    if !weight.is_finite()
        || !decay.is_finite()
        || !time.is_finite()
        || weight < 0.0
        || decay < 0.0
        || time < 0.0
    {
        return None;
    }
    if time == 0.0 || weight == 0.0 {
        return Some(1.0);
    }

    fn positive_product_ratio(
        numerator_left: f64,
        numerator_right: f64,
        denominator_left: f64,
        denominator_right: f64,
    ) -> f64 {
        let first =
            (numerator_left / denominator_left) * (numerator_right / denominator_right);
        if first.is_finite() && first != 0.0 {
            return first;
        }
        let second =
            (numerator_left / denominator_right) * (numerator_right / denominator_left);
        if second.is_finite() && second != 0.0 {
            return second;
        }
        (numerator_left.ln() + numerator_right.ln()
            - denominator_left.ln()
            - denominator_right.ln())
        .exp()
    }

    let half_decay = decay / 2.0;
    let root_weight = weight.sqrt();
    let decay_time = half_decay * time;
    let value = if root_weight > half_decay {
        let damping_ratio = half_decay / root_weight;
        if decay_time.is_infinite() {
            return Some(0.0);
        }
        let phase = root_weight * time * (1.0 - damping_ratio * damping_ratio).sqrt();
        if !phase.is_finite() {
            return None;
        }
        let sinc = if phase.abs() < 1e-4 {
            let phase_squared = phase * phase;
            1.0 - phase_squared / 6.0 + phase_squared * phase_squared / 120.0
        } else {
            phase.sin() / phase
        };
        (-decay_time).exp() * (phase.cos() + decay_time * sinc)
    } else if root_weight == half_decay {
        if decay_time.is_infinite() {
            0.0
        } else {
            (decay_time.ln_1p() - decay_time).exp()
        }
    } else {
        let rate_ratio = ((half_decay - root_weight) / half_decay
            * ((half_decay + root_weight) / half_decay))
            .sqrt();
        if rate_ratio == 0.0 {
            return if decay_time.is_infinite() {
                Some(0.0)
            } else {
                Some((decay_time.ln_1p() - decay_time).exp())
            };
        }
        let scaled_rate = decay_time * rate_ratio;
        if decay_time.is_finite() && scaled_rate < 0.5 {
            let sinhc = if scaled_rate.abs() < 1e-4 {
                let rate_squared = scaled_rate * scaled_rate;
                1.0 + rate_squared / 6.0 + rate_squared * rate_squared / 120.0
            } else {
                scaled_rate.sinh() / scaled_rate
            };
            (-decay_time).exp() * (scaled_rate.cosh() + decay_time * sinhc)
        } else {
            let slow_exponent = -positive_product_ratio(
                weight,
                time,
                half_decay,
                1.0 + rate_ratio,
            );
            let fast_exponent = -(half_decay * time) * (1.0 + rate_ratio);
            let slow_log_coefficient =
                (1.0 + rate_ratio).ln() - 2.0_f64.ln() - rate_ratio.ln();
            let fast_log_coefficient =
                (1.0 - rate_ratio).ln() - 2.0_f64.ln() - rate_ratio.ln();
            (slow_log_coefficient + slow_exponent).exp()
                - (fast_log_coefficient + fast_exponent).exp()
        }
    };
    value.is_finite().then_some(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn depolarizing_cptp_interval_has_both_boundaries() {
        assert!(qubit_depolarizing_is_cptp(-1.0 / 3.0, 1e-14));
        assert!(qubit_depolarizing_is_cptp(1.0, 1e-14));
        assert!(!qubit_depolarizing_is_cptp(-0.34, 1e-14));
        assert!(!qubit_depolarizing_is_cptp(1.01, 1e-14));
    }

    #[test]
    fn positive_exponential_kernel_can_violate_complete_positivity() {
        let weight = 4.0;
        let decay = 0.2;
        let frequency = (weight - decay * decay / 4.0_f64).sqrt();
        let time = std::f64::consts::PI / frequency;
        let lambda = exponential_memory_depolarizing_eigenvalue(weight, decay, time).unwrap();
        let minimum_choi = qubit_depolarizing_minimum_choi_eigenvalue(lambda);
        assert_relative_eq!(time, 1.572_763_511_444_000_8, epsilon = 1e-14);
        assert_relative_eq!(lambda, -0.854_467_893_006_756_5, epsilon = 1e-14);
        assert_relative_eq!(minimum_choi, -0.390_850_919_755_067_4, epsilon = 1e-14);
        assert!(!qubit_depolarizing_is_cptp(lambda, 1e-14));
    }

    #[test]
    fn zero_weight_remains_constant_at_large_decay_time() {
        let lambda = exponential_memory_depolarizing_eigenvalue(0.0, 2.0, 1_000.0).unwrap();
        assert_eq!(lambda, 1.0);
    }

    #[test]
    fn zero_weight_remains_constant_when_decay_time_overflows() {
        let lambda =
            exponential_memory_depolarizing_eigenvalue(0.0, 1.0e308, 4.0).unwrap();
        assert_eq!(lambda, 1.0);
    }

    #[test]
    fn weak_overdamped_kernel_retains_slow_root_at_long_time() {
        let weight = 1e-12;
        let decay = 2.0;
        let time = 1e6;
        let lambda = exponential_memory_depolarizing_eigenvalue(weight, decay, time).unwrap();
        let slow_root_limit = (-(weight / decay) * time).exp();
        assert_relative_eq!(lambda, slow_root_limit, epsilon = 1e-12);
    }

    #[test]
    fn extreme_overdamping_retains_finite_slow_root() {
        let lambda =
            exponential_memory_depolarizing_eigenvalue(1.0e-140, 2.0e100, 1.0e240)
                .unwrap();
        assert_relative_eq!(lambda, (-0.5_f64).exp(), epsilon = 1e-13);
    }

    #[test]
    fn critical_large_time_underflows_to_zero() {
        let lambda =
            exponential_memory_depolarizing_eigenvalue(1.0, 2.0, 1.0e308).unwrap();
        assert_eq!(lambda, 0.0);
    }

    #[test]
    fn near_critical_solution_is_continuous() {
        let decay = 2.0;
        let time = 0.75;
        let critical_weight = decay * decay / 4.0;
        let critical =
            exponential_memory_depolarizing_eigenvalue(critical_weight, decay, time).unwrap();
        let underdamped = exponential_memory_depolarizing_eigenvalue(
            critical_weight * (1.0 + 1e-12),
            decay,
            time,
        )
        .unwrap();
        let overdamped = exponential_memory_depolarizing_eigenvalue(
            critical_weight * (1.0 - 1e-12),
            decay,
            time,
        )
        .unwrap();
        assert_relative_eq!(underdamped, critical, epsilon = 1e-12);
        assert_relative_eq!(overdamped, critical, epsilon = 1e-12);
    }

    #[test]
    fn time_unit_rescaling_preserves_the_solution() {
        let weight = 4.0;
        let decay = 0.2;
        let time = 1.25;
        let scale = 1e6;
        let baseline = exponential_memory_depolarizing_eigenvalue(weight, decay, time).unwrap();
        let rescaled = exponential_memory_depolarizing_eigenvalue(
            weight * scale * scale,
            decay * scale,
            time / scale,
        )
        .unwrap();
        assert_relative_eq!(rescaled, baseline, epsilon = 1e-14);
    }

    #[test]
    fn overdamped_unit_rescaling_does_not_overflow_the_discriminant() {
        let baseline =
            exponential_memory_depolarizing_eigenvalue(1.0e-300, 1.0, 1.0).unwrap();
        let rescaled =
            exponential_memory_depolarizing_eigenvalue(1.0e10, 1.0e155, 1.0e-155).unwrap();

        assert_relative_eq!(rescaled, baseline, epsilon = 1e-15);
        assert_relative_eq!(rescaled, 1.0, epsilon = 1e-15);
    }
}
