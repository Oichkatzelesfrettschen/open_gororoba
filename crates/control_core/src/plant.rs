//! Plant trait for physical systems.
//!
//! A "plant" is a physical system that can be controlled. It has:
//! - State: internal variables (position, velocity, temperature, etc.)
//! - Input: control signals (force, voltage, power, etc.)
//! - Output: observable measurements (position, current, etc.)
//!
//! The Plant trait abstracts this for use with feedback controllers.

use nalgebra::{DVector, DMatrix};

/// A physical system that can be controlled.
///
/// Type parameters:
/// - `S`: State type (e.g., `DVector<f64>` or custom struct)
/// - `U`: Input type (control signal)
/// - `Y`: Output type (measurement)
pub trait Plant {
    /// State type.
    type State;
    /// Input (control) type.
    type Input;
    /// Output (measurement) type.
    type Output;

    /// Advance the system by one time step given control input.
    ///
    /// Returns the output after the step.
    fn step(&mut self, input: &Self::Input, dt: f64) -> Self::Output;

    /// Get the current state.
    fn state(&self) -> &Self::State;

    /// Get the current output without advancing.
    fn output(&self) -> Self::Output;

    /// Reset to initial conditions.
    fn reset(&mut self);

    /// Time step used for discretization (if applicable).
    fn dt(&self) -> f64;
}

/// Dynamics model for a plant (for simulation and prediction).
pub trait PlantDynamics {
    /// State type.
    type State;
    /// Input type.
    type Input;

    /// Compute state derivative: dx/dt = f(x, u).
    fn derivative(&self, state: &Self::State, input: &Self::Input) -> Self::State;

    /// Compute output from state: y = h(x).
    fn output(&self, state: &Self::State) -> Self::State;
}

/// A linear time-invariant (LTI) plant: dx/dt = Ax + Bu, y = Cx + Du.
pub trait LinearPlant: Plant {
    /// State matrix A (n x n).
    fn state_matrix(&self) -> &DMatrix<f64>;

    /// Input matrix B (n x m).
    fn input_matrix(&self) -> &DMatrix<f64>;

    /// Output matrix C (p x n).
    fn output_matrix(&self) -> &DMatrix<f64>;

    /// Feedthrough matrix D (p x m), often zero.
    fn feedthrough_matrix(&self) -> &DMatrix<f64>;

    /// State dimension n.
    fn state_dim(&self) -> usize {
        self.state_matrix().nrows()
    }

    /// Input dimension m.
    fn input_dim(&self) -> usize {
        self.input_matrix().ncols()
    }

    /// Output dimension p.
    fn output_dim(&self) -> usize {
        self.output_matrix().nrows()
    }
}

/// Marker trait for continuous-time plants.
pub trait ContinuousPlant: Plant {}

/// Marker trait for discrete-time plants.
pub trait DiscretePlant: Plant {}

/// A simple scalar first-order plant: dx/dt = -a*x + b*u, y = x.
///
/// Useful for testing and as a building block.
#[derive(Debug, Clone)]
pub struct FirstOrderPlant {
    /// Pole location (negative for stable).
    pub a: f64,
    /// Input gain.
    pub b: f64,
    /// Current state.
    pub x: f64,
    /// Time step for discretization.
    pub dt: f64,
}

impl FirstOrderPlant {
    /// Create a new first-order plant.
    pub fn new(a: f64, b: f64, dt: f64) -> Self {
        Self { a, b, x: 0.0, dt }
    }

    /// Create with initial state.
    pub fn with_initial(a: f64, b: f64, x0: f64, dt: f64) -> Self {
        Self { a, b, x: x0, dt }
    }

    /// Time constant tau = 1/a (for a > 0).
    pub fn time_constant(&self) -> f64 {
        if self.a.abs() > 1e-12 {
            1.0 / self.a.abs()
        } else {
            f64::INFINITY
        }
    }

    /// DC gain = b/a (for a > 0).
    pub fn dc_gain(&self) -> f64 {
        if self.a.abs() > 1e-12 {
            self.b / self.a
        } else {
            f64::INFINITY * self.b.signum()
        }
    }
}

impl Plant for FirstOrderPlant {
    type State = f64;
    type Input = f64;
    type Output = f64;

    fn step(&mut self, input: &f64, dt: f64) -> f64 {
        // Euler integration: x_{k+1} = x_k + dt * (-a*x_k + b*u_k)
        let dx = -self.a * self.x + self.b * input;
        self.x += dt * dx;
        self.x
    }

    fn state(&self) -> &f64 {
        &self.x
    }

    fn output(&self) -> f64 {
        self.x
    }

    fn reset(&mut self) {
        self.x = 0.0;
    }

    fn dt(&self) -> f64 {
        self.dt
    }
}

impl ContinuousPlant for FirstOrderPlant {}

/// A second-order oscillator: m*x'' + c*x' + k*x = u.
///
/// State: [position, velocity].
#[derive(Debug, Clone)]
pub struct SecondOrderPlant {
    /// Mass.
    pub m: f64,
    /// Damping coefficient.
    pub c: f64,
    /// Spring constant.
    pub k: f64,
    /// State: (position, velocity).
    pub state: (f64, f64),
    /// Time step.
    pub dt: f64,
}

impl SecondOrderPlant {
    /// Create a new second-order plant.
    pub fn new(m: f64, c: f64, k: f64, dt: f64) -> Self {
        Self { m, c, k, state: (0.0, 0.0), dt }
    }

    /// Create with initial conditions.
    pub fn with_initial(m: f64, c: f64, k: f64, x0: f64, v0: f64, dt: f64) -> Self {
        Self { m, c, k, state: (x0, v0), dt }
    }

    /// Get position.
    pub fn x(&self) -> f64 {
        self.state.0
    }

    /// Get velocity.
    pub fn v(&self) -> f64 {
        self.state.1
    }

    /// Natural frequency omega_n = sqrt(k/m).
    pub fn natural_frequency(&self) -> f64 {
        (self.k / self.m).sqrt()
    }

    /// Damping ratio zeta = c / (2*sqrt(k*m)).
    pub fn damping_ratio(&self) -> f64 {
        self.c / (2.0 * (self.k * self.m).sqrt())
    }

    /// Is the system underdamped (zeta < 1)?
    pub fn is_underdamped(&self) -> bool {
        self.damping_ratio() < 1.0
    }

    /// Is the system critically damped (zeta = 1)?
    pub fn is_critically_damped(&self, tolerance: f64) -> bool {
        (self.damping_ratio() - 1.0).abs() < tolerance
    }

    /// Is the system overdamped (zeta > 1)?
    pub fn is_overdamped(&self) -> bool {
        self.damping_ratio() > 1.0
    }
}

impl Plant for SecondOrderPlant {
    type State = (f64, f64);
    type Input = f64;
    type Output = f64;

    fn step(&mut self, input: &f64, dt: f64) -> f64 {
        // State: [x, v], dynamics: x' = v, v' = (u - c*v - k*x) / m
        let (x, v) = self.state;
        let a = (input - self.c * v - self.k * x) / self.m;

        // Semi-implicit Euler (Stormer-Verlet style)
        self.state.1 += dt * a;
        self.state.0 += dt * self.state.1;

        self.state.0
    }

    fn state(&self) -> &(f64, f64) {
        &self.state
    }

    fn output(&self) -> f64 {
        self.state.0
    }

    fn reset(&mut self) {
        self.state = (0.0, 0.0);
    }

    fn dt(&self) -> f64 {
        self.dt
    }
}

impl ContinuousPlant for SecondOrderPlant {}

/// Multi-input multi-output (MIMO) state-space plant.
///
/// dx/dt = A*x + B*u
/// y = C*x + D*u
#[derive(Debug, Clone)]
pub struct StateSpacePlant {
    /// State matrix A (n x n).
    pub a: DMatrix<f64>,
    /// Input matrix B (n x m).
    pub b: DMatrix<f64>,
    /// Output matrix C (p x n).
    pub c: DMatrix<f64>,
    /// Feedthrough matrix D (p x m).
    pub d: DMatrix<f64>,
    /// Current state x (n x 1).
    pub x: DVector<f64>,
    /// Time step.
    pub dt: f64,
}

impl StateSpacePlant {
    /// Create a new state-space plant.
    pub fn new(
        a: DMatrix<f64>,
        b: DMatrix<f64>,
        c: DMatrix<f64>,
        d: DMatrix<f64>,
        dt: f64,
    ) -> Self {
        let n = a.nrows();
        Self {
            a,
            b,
            c,
            d,
            x: DVector::zeros(n),
            dt,
        }
    }

    /// Create with initial state.
    pub fn with_initial(
        a: DMatrix<f64>,
        b: DMatrix<f64>,
        c: DMatrix<f64>,
        d: DMatrix<f64>,
        x0: DVector<f64>,
        dt: f64,
    ) -> Self {
        Self { a, b, c, d, x: x0, dt }
    }
}

impl Plant for StateSpacePlant {
    type State = DVector<f64>;
    type Input = DVector<f64>;
    type Output = DVector<f64>;

    fn step(&mut self, input: &DVector<f64>, dt: f64) -> DVector<f64> {
        // Euler: x_{k+1} = x_k + dt * (A*x_k + B*u_k)
        let dx = &self.a * &self.x + &self.b * input;
        self.x += dt * dx;

        // Output: y = C*x + D*u
        &self.c * &self.x + &self.d * input
    }

    fn state(&self) -> &DVector<f64> {
        &self.x
    }

    fn output(&self) -> DVector<f64> {
        &self.c * &self.x
    }

    fn reset(&mut self) {
        self.x.fill(0.0);
    }

    fn dt(&self) -> f64 {
        self.dt
    }
}

impl LinearPlant for StateSpacePlant {
    fn state_matrix(&self) -> &DMatrix<f64> {
        &self.a
    }

    fn input_matrix(&self) -> &DMatrix<f64> {
        &self.b
    }

    fn output_matrix(&self) -> &DMatrix<f64> {
        &self.c
    }

    fn feedthrough_matrix(&self) -> &DMatrix<f64> {
        &self.d
    }
}

impl ContinuousPlant for StateSpacePlant {}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_first_order_plant_step_response() {
        // dx/dt = -x + u, with u = 1 (step input)
        // Steady state: x_ss = 1
        let mut plant = FirstOrderPlant::new(1.0, 1.0, 0.01);

        // Simulate for 5 time constants
        let n_steps = 500;
        for _ in 0..n_steps {
            plant.step(&1.0, plant.dt);
        }

        // Should be close to steady state
        assert_relative_eq!(plant.x, 1.0, epsilon = 0.01);
    }

    #[test]
    fn test_first_order_time_constant() {
        let plant = FirstOrderPlant::new(2.0, 1.0, 0.01);
        assert_relative_eq!(plant.time_constant(), 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_first_order_dc_gain() {
        let plant = FirstOrderPlant::new(2.0, 4.0, 0.01);
        assert_relative_eq!(plant.dc_gain(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_second_order_plant_parameters() {
        // Standard mass-spring-damper: m=1, c=0.5, k=1
        let plant = SecondOrderPlant::new(1.0, 0.5, 1.0, 0.01);

        assert_relative_eq!(plant.natural_frequency(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(plant.damping_ratio(), 0.25, epsilon = 1e-10);
        assert!(plant.is_underdamped());
    }

    #[test]
    fn test_second_order_critically_damped() {
        // zeta = 1 requires c = 2*sqrt(k*m)
        let m = 1.0_f64;
        let k = 4.0_f64;
        let c = 2.0 * (k * m).sqrt(); // c = 4.0
        let plant = SecondOrderPlant::new(m, c, k, 0.01);

        assert!(plant.is_critically_damped(0.01));
    }

    #[test]
    fn test_second_order_step_response() {
        // Critically damped oscillator with step input for faster settling
        let m = 1.0_f64;
        let k = 1.0_f64;
        let c = 2.0 * (k * m).sqrt(); // Critical damping
        let mut plant = SecondOrderPlant::new(m, c, k, 0.001);

        // Apply step force
        for _ in 0..20000 {
            plant.step(&1.0, plant.dt);
        }

        // Steady state: F = k*x -> x_ss = F/k = 1.0
        assert_relative_eq!(plant.x(), 1.0, epsilon = 0.05);
    }

    #[test]
    fn test_state_space_plant_dimensions() {
        let a = DMatrix::from_row_slice(2, 2, &[0.0, 1.0, -1.0, -0.5]);
        let b = DMatrix::from_row_slice(2, 1, &[0.0, 1.0]);
        let c = DMatrix::from_row_slice(1, 2, &[1.0, 0.0]);
        let d = DMatrix::from_row_slice(1, 1, &[0.0]);

        let plant = StateSpacePlant::new(a, b, c, d, 0.01);

        assert_eq!(plant.state_dim(), 2);
        assert_eq!(plant.input_dim(), 1);
        assert_eq!(plant.output_dim(), 1);
    }

    #[test]
    fn test_plant_reset() {
        let mut plant = FirstOrderPlant::with_initial(1.0, 1.0, 5.0, 0.01);
        assert_eq!(*plant.state(), 5.0);

        plant.reset();
        assert_eq!(*plant.state(), 0.0);
    }
}
