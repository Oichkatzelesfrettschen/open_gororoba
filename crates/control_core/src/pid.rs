//! PID controller implementation.
//!
//! The PID (Proportional-Integral-Derivative) controller is the workhorse
//! of industrial control. This module provides a full-featured implementation
//! with anti-windup, derivative filtering, and setpoint weighting.
//!
//! # Literature
//! - Astrom & Hagglund (2006): Advanced PID Control
//! - Visioli (2006): Practical PID Control

use crate::feedback::{ControlOutput, Controller};

/// PID gains.
#[derive(Debug, Clone, Copy)]
pub struct PidGains {
    /// Proportional gain.
    pub kp: f64,
    /// Integral gain (often expressed as kp/ti).
    pub ki: f64,
    /// Derivative gain (often expressed as kp*td).
    pub kd: f64,
}

impl PidGains {
    /// Create gains from Kp, Ki, Kd directly.
    pub fn new(kp: f64, ki: f64, kd: f64) -> Self {
        Self { kp, ki, kd }
    }

    /// Create gains from Kp, Ti, Td (parallel form).
    ///
    /// Ki = Kp / Ti, Kd = Kp * Td.
    pub fn from_time_constants(kp: f64, ti: f64, td: f64) -> Self {
        let ki = if ti > 0.0 { kp / ti } else { 0.0 };
        let kd = kp * td;
        Self { kp, ki, kd }
    }

    /// P-only controller.
    pub fn p_only(kp: f64) -> Self {
        Self {
            kp,
            ki: 0.0,
            kd: 0.0,
        }
    }

    /// PI controller.
    pub fn pi(kp: f64, ki: f64) -> Self {
        Self { kp, ki, kd: 0.0 }
    }

    /// PD controller.
    pub fn pd(kp: f64, kd: f64) -> Self {
        Self { kp, ki: 0.0, kd }
    }
}

impl Default for PidGains {
    fn default() -> Self {
        Self {
            kp: 1.0,
            ki: 0.0,
            kd: 0.0,
        }
    }
}

/// Internal state of a PID controller.
#[derive(Debug, Clone, Copy, Default)]
pub struct PidState {
    /// Integral accumulator.
    pub integral: f64,
    /// Previous error (for derivative).
    pub prev_error: f64,
    /// Previous measurement (for derivative on measurement).
    pub prev_measurement: f64,
    /// Filtered derivative.
    pub derivative_filtered: f64,
    /// Previous unfiltered derivative (for filter).
    pub prev_derivative: f64,
}

/// PID controller with anti-windup and derivative filtering.
#[derive(Debug, Clone)]
pub struct PidController {
    /// Controller gains.
    pub gains: PidGains,
    /// Internal state.
    pub state: PidState,
    /// Output limits (min, max).
    pub limits: Option<(f64, f64)>,
    /// Derivative filter coefficient (0 = no filter, 1 = full filter).
    /// Typical value: 0.1 to 0.3.
    pub derivative_filter_coeff: f64,
    /// Use derivative on measurement instead of error.
    /// Avoids derivative kick on setpoint changes.
    pub derivative_on_measurement: bool,
    /// Setpoint weight for proportional term (0 to 1).
    /// b=0: error = -y, b=1: error = r - y.
    pub setpoint_weight_p: f64,
    /// Setpoint weight for derivative term (0 to 1).
    pub setpoint_weight_d: f64,
    /// Anti-windup method.
    pub anti_windup: AntiWindupMethod,
}

/// Anti-windup methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AntiWindupMethod {
    /// No anti-windup (integral can wind up indefinitely).
    None,
    /// Clamp integral when output saturates.
    #[default]
    Clamp,
    /// Back-calculation with tracking time constant.
    BackCalculation { tt_ratio: u32 }, // tt = td / tt_ratio
}

impl PidController {
    /// Create a new PID controller with given gains.
    pub fn new(gains: PidGains) -> Self {
        Self {
            gains,
            state: PidState::default(),
            limits: None,
            derivative_filter_coeff: 0.1,
            derivative_on_measurement: true,
            setpoint_weight_p: 1.0,
            setpoint_weight_d: 0.0,
            anti_windup: AntiWindupMethod::Clamp,
        }
    }

    /// Create a simple P controller.
    pub fn proportional(kp: f64) -> Self {
        Self::new(PidGains::p_only(kp))
    }

    /// Create a PI controller.
    pub fn pi(kp: f64, ki: f64) -> Self {
        Self::new(PidGains::pi(kp, ki))
    }

    /// Create a PD controller.
    pub fn pd(kp: f64, kd: f64) -> Self {
        Self::new(PidGains::pd(kp, kd))
    }

    /// Create a full PID controller.
    pub fn pid(kp: f64, ki: f64, kd: f64) -> Self {
        Self::new(PidGains::new(kp, ki, kd))
    }

    /// Set output limits.
    pub fn with_limits(mut self, min: f64, max: f64) -> Self {
        self.limits = Some((min, max));
        self
    }

    /// Set derivative filter coefficient.
    pub fn with_derivative_filter(mut self, coeff: f64) -> Self {
        self.derivative_filter_coeff = coeff.clamp(0.0, 1.0);
        self
    }

    /// Use derivative on measurement (avoids kick).
    pub fn with_derivative_on_measurement(mut self, enabled: bool) -> Self {
        self.derivative_on_measurement = enabled;
        self
    }

    /// Set anti-windup method.
    pub fn with_anti_windup(mut self, method: AntiWindupMethod) -> Self {
        self.anti_windup = method;
        self
    }

    /// Compute control signal with full metadata.
    pub fn compute(&mut self, reference: f64, measurement: f64, dt: f64) -> ControlOutput {
        if dt <= 0.0 {
            return ControlOutput::simple(0.0, reference - measurement);
        }

        // Compute error
        let error = reference - measurement;

        // Proportional term (with setpoint weighting)
        let p_error = self.setpoint_weight_p * reference - measurement;
        let p_term = self.gains.kp * p_error;

        // Integral term
        let i_term = self.gains.ki * self.state.integral;

        // Derivative term
        let raw_derivative = if self.derivative_on_measurement {
            // Derivative on measurement (avoids kick)
            let d_error = self.setpoint_weight_d * reference - measurement;
            let prev_d_error = self.setpoint_weight_d * reference - self.state.prev_measurement;
            -(d_error - prev_d_error) / dt
        } else {
            // Derivative on error
            (error - self.state.prev_error) / dt
        };

        // Apply derivative filter
        let derivative = derivative_filter(
            raw_derivative,
            self.state.derivative_filtered,
            self.derivative_filter_coeff,
        );
        self.state.derivative_filtered = derivative;

        let d_term = self.gains.kd * derivative;

        // Compute total output
        let u_unsat = p_term + i_term + d_term;

        // Apply saturation
        let (u, saturated) = if let Some((min, max)) = self.limits {
            let clamped = u_unsat.clamp(min, max);
            (clamped, (u_unsat - clamped).abs() > 1e-12)
        } else {
            (u_unsat, false)
        };

        // Update integral with anti-windup
        let new_integral = match self.anti_windup {
            AntiWindupMethod::None => self.state.integral + error * dt,
            AntiWindupMethod::Clamp => {
                if saturated {
                    // Don't accumulate if saturated in same direction
                    if (u_unsat > u && error > 0.0) || (u_unsat < u && error < 0.0) {
                        self.state.integral
                    } else {
                        self.state.integral + error * dt
                    }
                } else {
                    self.state.integral + error * dt
                }
            }
            AntiWindupMethod::BackCalculation { tt_ratio } => {
                let tt = if self.gains.kd > 0.0 && tt_ratio > 0 {
                    self.gains.kd / self.gains.kp / tt_ratio as f64
                } else {
                    dt
                };
                let correction = (u - u_unsat) / tt;
                self.state.integral + error * dt + correction * dt
            }
        };

        // Clamp integral directly if using clamp method
        self.state.integral =
            if let (Some((min, max)), AntiWindupMethod::Clamp) = (self.limits, self.anti_windup) {
                anti_windup_clamp(new_integral, self.gains.ki, min, max)
            } else {
                new_integral
            };

        // Update state
        self.state.prev_error = error;
        self.state.prev_measurement = measurement;

        ControlOutput {
            u,
            saturated,
            error,
            integral: Some(self.state.integral),
            derivative: Some(derivative),
        }
    }
}

impl Controller for PidController {
    fn control(&mut self, reference: f64, measurement: f64, dt: f64) -> f64 {
        self.compute(reference, measurement, dt).u
    }

    fn control_full(&mut self, reference: f64, measurement: f64, dt: f64) -> ControlOutput {
        self.compute(reference, measurement, dt)
    }

    fn reset(&mut self) {
        self.state = PidState::default();
    }

    fn set_limits(&mut self, min: f64, max: f64) {
        self.limits = Some((min, max));
    }
}

/// Apply derivative filter (first-order low-pass).
///
/// d_filtered = alpha * d_raw + (1 - alpha) * d_prev
pub fn derivative_filter(raw: f64, prev_filtered: f64, alpha: f64) -> f64 {
    alpha * raw + (1.0 - alpha) * prev_filtered
}

/// Clamp integral term to prevent windup.
///
/// Limits the integral such that ki * integral stays within output limits.
pub fn anti_windup_clamp(integral: f64, ki: f64, u_min: f64, u_max: f64) -> f64 {
    if ki.abs() < 1e-12 {
        return integral;
    }

    let i_min = u_min / ki;
    let i_max = u_max / ki;

    if i_min < i_max {
        integral.clamp(i_min, i_max)
    } else {
        integral.clamp(i_max, i_min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_pid_gains_creation() {
        let g = PidGains::new(1.0, 0.5, 0.1);
        assert_eq!(g.kp, 1.0);
        assert_eq!(g.ki, 0.5);
        assert_eq!(g.kd, 0.1);
    }

    #[test]
    fn test_pid_gains_from_time_constants() {
        // Kp=2, Ti=4 -> Ki = 2/4 = 0.5
        // Kp=2, Td=0.5 -> Kd = 2*0.5 = 1.0
        let g = PidGains::from_time_constants(2.0, 4.0, 0.5);
        assert_relative_eq!(g.kp, 2.0, epsilon = 1e-10);
        assert_relative_eq!(g.ki, 0.5, epsilon = 1e-10);
        assert_relative_eq!(g.kd, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_p_controller() {
        let mut ctrl = PidController::proportional(2.0);
        let u = ctrl.control(10.0, 3.0, 0.01);
        // error = 10 - 3 = 7, u = 2 * 7 = 14
        assert_relative_eq!(u, 14.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pi_controller_integral_accumulation() {
        let mut ctrl = PidController::pi(1.0, 1.0);

        // First step: error = 1.0
        let u1 = ctrl.control(1.0, 0.0, 0.1);
        // P = 1.0, I = 0 (not yet accumulated in output)
        assert!(u1 > 0.0);

        // Second step: integral should have accumulated
        let u2 = ctrl.control(1.0, 0.0, 0.1);
        assert!(u2 > u1); // More integral action
    }

    #[test]
    fn test_pid_saturation() {
        let mut ctrl = PidController::proportional(10.0).with_limits(-5.0, 5.0);

        let output = ctrl.compute(10.0, 0.0, 0.01);
        // error = 10, u_raw = 100, should saturate to 5
        assert_relative_eq!(output.u, 5.0, epsilon = 1e-10);
        assert!(output.saturated);
    }

    #[test]
    fn test_anti_windup_clamp() {
        let integral = 100.0;
        let ki = 1.0;
        let result = anti_windup_clamp(integral, ki, -10.0, 10.0);
        assert_relative_eq!(result, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_anti_windup_clamp_negative() {
        let integral = -100.0;
        let ki = 1.0;
        let result = anti_windup_clamp(integral, ki, -10.0, 10.0);
        assert_relative_eq!(result, -10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_derivative_filter() {
        let raw = 10.0;
        let prev = 0.0;
        let alpha = 0.1;

        let filtered = derivative_filter(raw, prev, alpha);
        // 0.1 * 10 + 0.9 * 0 = 1.0
        assert_relative_eq!(filtered, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pid_reset() {
        let mut ctrl = PidController::pid(1.0, 1.0, 0.1);

        // Accumulate some state
        ctrl.control(1.0, 0.0, 0.1);
        ctrl.control(1.0, 0.0, 0.1);
        assert!(ctrl.state.integral > 0.0);

        // Reset
        ctrl.reset();
        assert_relative_eq!(ctrl.state.integral, 0.0, epsilon = 1e-10);
        assert_relative_eq!(ctrl.state.prev_error, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_pid_convergence() {
        // Simple first-order plant simulation
        let mut ctrl = PidController::pi(2.0, 1.0).with_limits(-10.0, 10.0);
        let mut x = 0.0;
        let target = 1.0;
        let dt = 0.01;

        for _ in 0..1000 {
            let u = ctrl.control(target, x, dt);
            // Simple plant: dx/dt = -x + u
            x += dt * (-x + u);
        }

        // Should converge near target
        assert_relative_eq!(x, target, epsilon = 0.15);
    }

    #[test]
    fn test_derivative_on_measurement() {
        let mut ctrl = PidController::pd(1.0, 0.5).with_derivative_on_measurement(true);

        // First measurement
        ctrl.control(1.0, 0.0, 0.01);

        // Sudden setpoint change - derivative on measurement should not kick
        let output = ctrl.compute(10.0, 0.0, 0.01);

        // With derivative on measurement, the derivative term should be small
        // (only based on measurement change, not setpoint change)
        assert!(output.derivative.unwrap().abs() < 1.0);
    }

    #[test]
    fn test_control_output_metadata() {
        let mut ctrl = PidController::pid(1.0, 0.5, 0.1);

        let output = ctrl.compute(1.0, 0.0, 0.01);

        assert!(output.integral.is_some());
        assert!(output.derivative.is_some());
        assert_relative_eq!(output.error, 1.0, epsilon = 1e-10);
    }
}
