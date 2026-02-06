//! Feedback control abstractions.
//!
//! Provides the Controller trait and FeedbackLoop for closed-loop control.

use thiserror::Error;

/// Error types for control operations.
#[derive(Debug, Clone, Error)]
pub enum ControlError {
    /// Control input saturated.
    #[error("Control saturated: requested {requested:.4}, limited to [{min:.4}, {max:.4}]")]
    Saturation {
        requested: f64,
        min: f64,
        max: f64,
    },

    /// Reference signal out of range.
    #[error("Reference out of range: {value:.4} not in [{min:.4}, {max:.4}]")]
    ReferenceOutOfRange {
        value: f64,
        min: f64,
        max: f64,
    },

    /// Unstable oscillation detected.
    #[error("Instability detected: oscillation amplitude {amplitude:.4} exceeds threshold")]
    Instability { amplitude: f64 },

    /// Convergence failure.
    #[error("Convergence failure after {iterations} iterations")]
    ConvergenceFailure { iterations: usize },
}

/// Reference signal for tracking control.
#[derive(Debug, Clone)]
pub enum ReferenceSignal {
    /// Constant setpoint.
    Constant(f64),

    /// Step from one value to another at a given time.
    Step { from: f64, to: f64, time: f64 },

    /// Ramp between values.
    Ramp { from: f64, to: f64, duration: f64 },

    /// Sinusoidal reference.
    Sinusoid { amplitude: f64, frequency: f64, phase: f64, offset: f64 },

    /// Custom trajectory (time -> value).
    Trajectory(Vec<(f64, f64)>),
}

impl ReferenceSignal {
    /// Evaluate the reference at a given time.
    pub fn at(&self, t: f64) -> f64 {
        match self {
            ReferenceSignal::Constant(v) => *v,

            ReferenceSignal::Step { from, to, time } => {
                if t < *time { *from } else { *to }
            }

            ReferenceSignal::Ramp { from, to, duration } => {
                if t <= 0.0 {
                    *from
                } else if t >= *duration {
                    *to
                } else {
                    from + (to - from) * t / duration
                }
            }

            ReferenceSignal::Sinusoid { amplitude, frequency, phase, offset } => {
                offset + amplitude * (2.0 * std::f64::consts::PI * frequency * t + phase).sin()
            }

            ReferenceSignal::Trajectory(points) => {
                // Linear interpolation
                if points.is_empty() {
                    return 0.0;
                }
                if t <= points[0].0 {
                    return points[0].1;
                }
                if t >= points.last().unwrap().0 {
                    return points.last().unwrap().1;
                }

                for i in 0..points.len() - 1 {
                    if t >= points[i].0 && t < points[i + 1].0 {
                        let t0 = points[i].0;
                        let t1 = points[i + 1].0;
                        let v0 = points[i].1;
                        let v1 = points[i + 1].1;
                        let alpha = (t - t0) / (t1 - t0);
                        return v0 + alpha * (v1 - v0);
                    }
                }
                points.last().unwrap().1
            }
        }
    }
}

/// Control output with metadata.
#[derive(Debug, Clone)]
pub struct ControlOutput {
    /// The control signal.
    pub u: f64,
    /// Whether saturation was applied.
    pub saturated: bool,
    /// The error (reference - measurement).
    pub error: f64,
    /// Integral of error (if tracked).
    pub integral: Option<f64>,
    /// Derivative of error (if tracked).
    pub derivative: Option<f64>,
}

impl ControlOutput {
    /// Simple output with just the control signal.
    pub fn simple(u: f64, error: f64) -> Self {
        Self {
            u,
            saturated: false,
            error,
            integral: None,
            derivative: None,
        }
    }
}

/// A feedback controller computes control input from error.
pub trait Controller {
    /// Compute control signal given reference and measurement.
    ///
    /// Returns the control input u.
    fn control(&mut self, reference: f64, measurement: f64, dt: f64) -> f64;

    /// Compute control with full output metadata.
    fn control_full(&mut self, reference: f64, measurement: f64, dt: f64) -> ControlOutput {
        let error = reference - measurement;
        let u = self.control(reference, measurement, dt);
        ControlOutput::simple(u, error)
    }

    /// Reset controller state.
    fn reset(&mut self);

    /// Set output saturation limits.
    fn set_limits(&mut self, min: f64, max: f64);
}

/// A feedback loop connecting a controller to a plant.
#[derive(Debug)]
pub struct FeedbackLoop<P, C> {
    /// The plant being controlled.
    pub plant: P,
    /// The controller.
    pub controller: C,
    /// Current time.
    pub time: f64,
    /// Control history.
    pub control_history: Vec<(f64, f64)>,
    /// Output history.
    pub output_history: Vec<(f64, f64)>,
    /// Error history.
    pub error_history: Vec<(f64, f64)>,
    /// Maximum history length (0 = unlimited).
    pub max_history: usize,
}

impl<P, C> FeedbackLoop<P, C>
where
    P: crate::plant::Plant<Input = f64, Output = f64>,
    C: Controller,
{
    /// Create a new feedback loop.
    pub fn new(plant: P, controller: C) -> Self {
        Self {
            plant,
            controller,
            time: 0.0,
            control_history: Vec::new(),
            output_history: Vec::new(),
            error_history: Vec::new(),
            max_history: 10000,
        }
    }

    /// Run one control step.
    pub fn step(&mut self, reference: f64) -> f64 {
        let dt = self.plant.dt();
        let measurement = self.plant.output();

        // Compute control
        let u = self.controller.control(reference, measurement, dt);

        // Apply to plant
        let y = self.plant.step(&u, dt);

        // Update time
        self.time += dt;

        // Record history
        if self.max_history == 0 || self.control_history.len() < self.max_history {
            self.control_history.push((self.time, u));
            self.output_history.push((self.time, y));
            self.error_history.push((self.time, reference - y));
        }

        y
    }

    /// Run simulation for given duration.
    pub fn simulate(&mut self, reference: &ReferenceSignal, duration: f64) {
        let dt = self.plant.dt();
        let n_steps = (duration / dt).ceil() as usize;

        for _ in 0..n_steps {
            let r = reference.at(self.time);
            self.step(r);
        }
    }

    /// Run until error is below threshold or max steps reached.
    pub fn settle(
        &mut self,
        reference: f64,
        tolerance: f64,
        max_steps: usize,
    ) -> Result<usize, ControlError> {
        for i in 0..max_steps {
            self.step(reference);
            let error = (reference - self.plant.output()).abs();
            if error < tolerance {
                return Ok(i + 1);
            }
        }
        Err(ControlError::ConvergenceFailure { iterations: max_steps })
    }

    /// Reset the feedback loop.
    pub fn reset(&mut self) {
        self.plant.reset();
        self.controller.reset();
        self.time = 0.0;
        self.control_history.clear();
        self.output_history.clear();
        self.error_history.clear();
    }

    /// Compute settling time (time to reach within percentage of final value).
    pub fn settling_time(&self, percentage: f64) -> Option<f64> {
        if self.output_history.is_empty() {
            return None;
        }

        let final_value = self.output_history.last()?.1;
        let threshold = final_value * (1.0 - percentage / 100.0);

        // Find last time output crosses threshold
        for &(t, y) in self.output_history.iter().rev() {
            if (y - final_value).abs() > (final_value - threshold).abs() {
                return Some(t);
            }
        }

        Some(0.0)
    }

    /// Compute overshoot percentage.
    pub fn overshoot(&self, reference: f64) -> f64 {
        if self.output_history.is_empty() || reference.abs() < 1e-12 {
            return 0.0;
        }

        let max_output = self.output_history
            .iter()
            .map(|(_, y)| *y)
            .fold(f64::NEG_INFINITY, f64::max);

        if max_output > reference {
            100.0 * (max_output - reference) / reference
        } else {
            0.0
        }
    }
}

/// A simple proportional controller: u = Kp * e.
#[derive(Debug, Clone)]
pub struct ProportionalController {
    /// Proportional gain.
    pub kp: f64,
    /// Output limits.
    pub limits: Option<(f64, f64)>,
}

impl ProportionalController {
    /// Create a new P controller.
    pub fn new(kp: f64) -> Self {
        Self { kp, limits: None }
    }

    /// Create with output limits.
    pub fn with_limits(kp: f64, min: f64, max: f64) -> Self {
        Self { kp, limits: Some((min, max)) }
    }
}

impl Controller for ProportionalController {
    fn control(&mut self, reference: f64, measurement: f64, _dt: f64) -> f64 {
        let error = reference - measurement;
        let u = self.kp * error;

        if let Some((min, max)) = self.limits {
            u.clamp(min, max)
        } else {
            u
        }
    }

    fn reset(&mut self) {
        // No state to reset
    }

    fn set_limits(&mut self, min: f64, max: f64) {
        self.limits = Some((min, max));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plant::Plant;
    use approx::assert_relative_eq;

    #[test]
    fn test_reference_constant() {
        let r = ReferenceSignal::Constant(5.0);
        assert_eq!(r.at(0.0), 5.0);
        assert_eq!(r.at(100.0), 5.0);
    }

    #[test]
    fn test_reference_step() {
        let r = ReferenceSignal::Step { from: 0.0, to: 1.0, time: 1.0 };
        assert_eq!(r.at(0.5), 0.0);
        assert_eq!(r.at(1.5), 1.0);
    }

    #[test]
    fn test_reference_ramp() {
        let r = ReferenceSignal::Ramp { from: 0.0, to: 10.0, duration: 2.0 };
        assert_eq!(r.at(0.0), 0.0);
        assert_eq!(r.at(1.0), 5.0);
        assert_eq!(r.at(2.0), 10.0);
        assert_eq!(r.at(3.0), 10.0);
    }

    #[test]
    fn test_reference_sinusoid() {
        let r = ReferenceSignal::Sinusoid {
            amplitude: 1.0,
            frequency: 1.0,
            phase: 0.0,
            offset: 0.0,
        };
        assert_relative_eq!(r.at(0.0), 0.0, epsilon = 1e-10);
        assert_relative_eq!(r.at(0.25), 1.0, epsilon = 1e-10);
        assert_relative_eq!(r.at(0.5), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_proportional_controller() {
        let mut ctrl = ProportionalController::new(2.0);
        let u = ctrl.control(10.0, 3.0, 0.01);
        assert_relative_eq!(u, 14.0, epsilon = 1e-10); // 2 * (10 - 3)
    }

    #[test]
    fn test_proportional_controller_with_limits() {
        let mut ctrl = ProportionalController::with_limits(2.0, -5.0, 5.0);
        let u = ctrl.control(10.0, 0.0, 0.01);
        assert_relative_eq!(u, 5.0, epsilon = 1e-10); // Saturated
    }

    #[test]
    fn test_feedback_loop_step() {
        use crate::plant::FirstOrderPlant;

        let plant = FirstOrderPlant::new(1.0, 1.0, 0.01);
        let controller = ProportionalController::new(10.0);
        let mut loop_ = FeedbackLoop::new(plant, controller);

        // Run a few steps
        for _ in 0..100 {
            loop_.step(1.0);
        }

        // Should approach setpoint
        assert!(loop_.plant.output() > 0.5);
    }

    #[test]
    fn test_control_output_simple() {
        let output = ControlOutput::simple(5.0, 0.1);
        assert_eq!(output.u, 5.0);
        assert_eq!(output.error, 0.1);
        assert!(!output.saturated);
    }
}
