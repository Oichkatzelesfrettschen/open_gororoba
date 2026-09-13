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
    let half_decay = decay / 2.0;
    let root_weight = weight.sqrt();
    let decay_time = half_decay * time;
    let value = if root_weight > half_decay {
        let damping_ratio = half_decay / root_weight;
        let phase = root_weight * time * (1.0 - damping_ratio * damping_ratio).sqrt();
        let sinc = if phase.abs() < 1e-4 {
            let phase_squared = phase * phase;
            1.0 - phase_squared / 6.0 + phase_squared * phase_squared / 120.0
        } else {
            phase.sin() / phase
        };
        (-decay_time).exp() * (phase.cos() + decay_time * sinc)
    } else if root_weight == half_decay {
        (-decay_time).exp() * (1.0 + decay_time)
    } else {
        let frequency_ratio = if half_decay == 0.0 {
            0.0
        } else {
            root_weight / half_decay
        };
        let rate_ratio = (1.0 - frequency_ratio * frequency_ratio).sqrt();
        let scaled_rate = decay_time * rate_ratio;
        if scaled_rate < 0.5 {
            let sinhc = if scaled_rate.abs() < 1e-4 {
                let rate_squared = scaled_rate * scaled_rate;
                1.0 + rate_squared / 6.0 + rate_squared * rate_squared / 120.0
            } else {
                scaled_rate.sinh() / scaled_rate
            };
            (-decay_time).exp() * (scaled_rate.cosh() + decay_time * sinhc)
        } else {
            let slow_exponent =
                -decay_time * frequency_ratio * frequency_ratio / (1.0 + rate_ratio);
            let slow_exponential = slow_exponent.exp();
            let fast_exponential = (-scaled_rate - decay_time).exp();
            0.5 * (1.0 + 1.0 / rate_ratio) * slow_exponential
                + 0.5 * (1.0 - 1.0 / rate_ratio) * fast_exponential
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
    fn weak_overdamped_kernel_retains_slow_root_at_long_time() {
        let weight = 1e-12;
        let decay = 2.0;
        let time = 1e6;
        let lambda = exponential_memory_depolarizing_eigenvalue(weight, decay, time).unwrap();
        let slow_root_limit = (-(weight / decay) * time).exp();
        assert_relative_eq!(lambda, slow_root_limit, epsilon = 1e-12);
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
