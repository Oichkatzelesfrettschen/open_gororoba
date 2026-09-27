use nalgebra::Vector3;

/// Seven-stage Dormand-Prince 5(4) integrator for second-order motion.
///
/// The derivative callback supplies acceleration for `y' = v` and
/// `v' = f(t, y, v)`. The embedded position and velocity errors share an
/// absolute RMS tolerance over their six scalar components.
pub struct DormandPrinceRK45 {
    pub tolerance: f64,
    pub min_step: f64,
    pub max_step: f64,
}

impl DormandPrinceRK45 {
    pub fn new(tolerance: f64) -> Self {
        Self {
            tolerance,
            min_step: 1e-6,
            max_step: 3600.0,
        }
    }

    /// Advance one accepted step and store the next step proposal in `dt`.
    pub fn step<F>(
        &self,
        t: f64,
        y: Vector3<f64>,
        v: Vector3<f64>,
        dt: &mut f64,
        f: F,
    ) -> (f64, Vector3<f64>, Vector3<f64>)
    where
        F: Fn(f64, Vector3<f64>, Vector3<f64>) -> Vector3<f64>,
    {
        let minimum_step = self.min_step.abs().max(f64::MIN_POSITIVE);
        let maximum_step = self.max_step.abs().max(minimum_step);
        let direction = if dt.is_sign_negative() { -1.0 } else { 1.0 };
        let mut h_used = dt.abs().clamp(minimum_step, maximum_step) * direction;

        loop {
            let (y_new, v_new, error) = self.single_step(t, y, v, h_used, &f);

            if error <= self.tolerance || h_used.abs() <= minimum_step {
                let next_step = self
                    .calc_next_dt(h_used.abs(), error)
                    .clamp(minimum_step, maximum_step);
                *dt = direction * next_step;
                return (t + h_used, y_new, v_new);
            }

            h_used = direction * (h_used.abs() * 0.5).max(minimum_step);
        }
    }

    fn single_step<F>(
        &self,
        t: f64,
        y: Vector3<f64>,
        v: Vector3<f64>,
        h: f64,
        f: &F,
    ) -> (Vector3<f64>, Vector3<f64>, f64)
    where
        F: Fn(f64, Vector3<f64>, Vector3<f64>) -> Vector3<f64>,
    {
        let k1_y = v;
        let k1_v = f(t, y, v);

        let y2 = linear_combination(y, h, &[(1.0 / 5.0, k1_y)]);
        let v2 = linear_combination(v, h, &[(1.0 / 5.0, k1_v)]);
        let k2_y = v2;
        let k2_v = f(t + h / 5.0, y2, v2);

        let y3 = linear_combination(y, h, &[(3.0 / 40.0, k1_y), (9.0 / 40.0, k2_y)]);
        let v3 = linear_combination(v, h, &[(3.0 / 40.0, k1_v), (9.0 / 40.0, k2_v)]);
        let k3_y = v3;
        let k3_v = f(t + 3.0 * h / 10.0, y3, v3);

        let y4_stage = linear_combination(
            y,
            h,
            &[
                (44.0 / 45.0, k1_y),
                (-56.0 / 15.0, k2_y),
                (32.0 / 9.0, k3_y),
            ],
        );
        let v4_stage = linear_combination(
            v,
            h,
            &[
                (44.0 / 45.0, k1_v),
                (-56.0 / 15.0, k2_v),
                (32.0 / 9.0, k3_v),
            ],
        );
        let k4_y = v4_stage;
        let k4_v = f(t + 4.0 * h / 5.0, y4_stage, v4_stage);

        let y5_stage = linear_combination(
            y,
            h,
            &[
                (19372.0 / 6561.0, k1_y),
                (-25360.0 / 2187.0, k2_y),
                (64448.0 / 6561.0, k3_y),
                (-212.0 / 729.0, k4_y),
            ],
        );
        let v5_stage = linear_combination(
            v,
            h,
            &[
                (19372.0 / 6561.0, k1_v),
                (-25360.0 / 2187.0, k2_v),
                (64448.0 / 6561.0, k3_v),
                (-212.0 / 729.0, k4_v),
            ],
        );
        let k5_y = v5_stage;
        let k5_v = f(t + 8.0 * h / 9.0, y5_stage, v5_stage);

        let y6_stage = linear_combination(
            y,
            h,
            &[
                (9017.0 / 3168.0, k1_y),
                (-355.0 / 33.0, k2_y),
                (46732.0 / 5247.0, k3_y),
                (49.0 / 176.0, k4_y),
                (-5103.0 / 18656.0, k5_y),
            ],
        );
        let v6_stage = linear_combination(
            v,
            h,
            &[
                (9017.0 / 3168.0, k1_v),
                (-355.0 / 33.0, k2_v),
                (46732.0 / 5247.0, k3_v),
                (49.0 / 176.0, k4_v),
                (-5103.0 / 18656.0, k5_v),
            ],
        );
        let k6_y = v6_stage;
        let k6_v = f(t + h, y6_stage, v6_stage);

        let y5 = linear_combination(
            y,
            h,
            &[
                (35.0 / 384.0, k1_y),
                (500.0 / 1113.0, k3_y),
                (125.0 / 192.0, k4_y),
                (-2187.0 / 6784.0, k5_y),
                (11.0 / 84.0, k6_y),
            ],
        );
        let v5 = linear_combination(
            v,
            h,
            &[
                (35.0 / 384.0, k1_v),
                (500.0 / 1113.0, k3_v),
                (125.0 / 192.0, k4_v),
                (-2187.0 / 6784.0, k5_v),
                (11.0 / 84.0, k6_v),
            ],
        );
        let k7_y = v5;
        let k7_v = f(t + h, y5, v5);

        let y4 = linear_combination(
            y,
            h,
            &[
                (5179.0 / 57600.0, k1_y),
                (7571.0 / 16695.0, k3_y),
                (393.0 / 640.0, k4_y),
                (-92097.0 / 339200.0, k5_y),
                (187.0 / 2100.0, k6_y),
                (1.0 / 40.0, k7_y),
            ],
        );
        let v4 = linear_combination(
            v,
            h,
            &[
                (5179.0 / 57600.0, k1_v),
                (7571.0 / 16695.0, k3_v),
                (393.0 / 640.0, k4_v),
                (-92097.0 / 339200.0, k5_v),
                (187.0 / 2100.0, k6_v),
                (1.0 / 40.0, k7_v),
            ],
        );
        let position_error = y5 - y4;
        let velocity_error = v5 - v4;
        let error = ((position_error.norm_squared() + velocity_error.norm_squared()) / 6.0).sqrt();

        (y5, v5, error)
    }

    fn calc_next_dt(&self, dt: f64, error: f64) -> f64 {
        if !error.is_finite() {
            return self.min_step.abs();
        }
        if error < 1e-30 {
            return dt * 2.0;
        }
        let factor = (self.tolerance / error).powf(0.2) * 0.9;
        dt * factor.clamp(0.1, 2.0)
    }
}

fn linear_combination(
    base: Vector3<f64>,
    step: f64,
    terms: &[(f64, Vector3<f64>)],
) -> Vector3<f64> {
    terms.iter().fold(base, |state, (weight, derivative)| {
        state + *derivative * (step * weight)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn integrate_with_fixed_step(step_size: f64) -> f64 {
        let tolerance = 1e-2 * step_size.powi(5);
        let integrator = DormandPrinceRK45 {
            tolerance,
            min_step: 1e-12,
            max_step: step_size,
        };
        let mut time = 0.0;
        let mut position = Vector3::repeat(1.0);
        let mut velocity = Vector3::repeat(1.0);
        let steps = (1.0 / step_size).round() as usize;

        for _ in 0..steps {
            let mut proposed_step = step_size;
            let (next_time, next_position, next_velocity) = integrator.step(
                time,
                position,
                velocity,
                &mut proposed_step,
                |_, _, velocity| velocity,
            );
            assert!((next_time - time - step_size).abs() < 1e-13);
            time = next_time;
            position = next_position;
            velocity = next_velocity;
        }

        assert!((time - 1.0).abs() < 1e-13);
        (position[0] - std::f64::consts::E).abs()
    }

    #[test]
    fn test_exponential_solution_converges_at_fifth_order_as_tolerance_tightens() {
        let errors = [0.25, 0.125, 0.0625].map(integrate_with_fixed_step);
        for pair in errors.windows(2) {
            let observed_order = (pair[0] / pair[1]).log2();
            assert!(
                (4.5..=5.5).contains(&observed_order),
                "observed convergence order was {observed_order}: {pair:?}"
            );
        }
    }

    #[test]
    fn test_returned_time_uses_accepted_step_before_next_step_update() {
        let integrator = DormandPrinceRK45 {
            tolerance: 1e-9,
            min_step: 1e-10,
            max_step: 1.0,
        };
        let initial_time: f64 = 2.0;
        let initial_value = initial_time.exp();
        let initial_position = Vector3::repeat(initial_value);
        let initial_velocity = Vector3::repeat(initial_value);
        let mut proposed_step = 1.0;
        let (returned_time, position, velocity) = integrator.step(
            initial_time,
            initial_position,
            initial_velocity,
            &mut proposed_step,
            |_, _, velocity| velocity,
        );

        let accepted_step = (position[0] / initial_value).ln();
        assert!(accepted_step < 1.0);
        assert!((proposed_step - accepted_step).abs() > 1e-6);
        assert!((returned_time - initial_time - accepted_step).abs() < 1e-12);
        assert!((position[0] - returned_time.exp()).abs() < 5e-8);
        assert!((velocity[0] - returned_time.exp()).abs() < 5e-8);
    }
}
