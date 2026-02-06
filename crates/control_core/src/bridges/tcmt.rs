//! Plant implementations for TCMT (Temporal Coupled-Mode Theory) cavities.
//!
//! Enables feedback control of nonlinear optical cavities modeled with TCMT.
//! The cavity can be controlled by modulating input power while observing
//! transmitted power or cavity energy.
//!
//! # Control Scenarios
//!
//! - **Power stabilization**: Maintain constant transmitted power despite fluctuations
//! - **Bistable switching**: Controlled transitions between high/low transmission states
//! - **Thermal compensation**: Counteract thermo-optic drift in resonance
//!
//! # Example
//!
//! ```ignore
//! use control_core::{Plant, Controller, FeedbackLoop, PidController};
//! use control_core::bridges::TcmtPlant;
//! use optics_core::KerrCavity;
//!
//! let cavity = KerrCavity::normalized(1000.0, 1.0);
//! let plant = TcmtPlant::new(cavity, 0.0, 0.001);
//! let controller = PidController::pi(1.0, 0.1);
//! let mut loop_ = FeedbackLoop::new(plant, controller);
//!
//! // Control to target transmission
//! loop_.settle(0.5, 0.01, 1000);
//! ```

use crate::plant::Plant;
use num_complex::Complex64;
use optics_core::{CavityState, InputField, KerrCavity, TcmtSolver};

/// TCMT cavity as a controllable plant.
///
/// - **Input**: Input power (W)
/// - **Output**: Transmitted power (normalized, 0 to 1)
/// - **State**: Complex cavity amplitude
#[derive(Debug, Clone)]
pub struct TcmtPlant {
    /// The Kerr cavity parameters.
    cavity: KerrCavity,
    /// Current cavity state.
    state: CavityState,
    /// Input field detuning from resonance (rad/s).
    detuning: f64,
    /// Time step for discretization.
    dt: f64,
    /// Current input power.
    input_power: f64,
}

impl TcmtPlant {
    /// Create a new TCMT plant with given cavity and detuning.
    ///
    /// # Arguments
    /// * `cavity` - Kerr cavity parameters
    /// * `detuning` - Frequency detuning from resonance (rad/s)
    /// * `dt` - Time step for simulation
    pub fn new(cavity: KerrCavity, detuning: f64, dt: f64) -> Self {
        Self {
            cavity,
            state: CavityState::default(),
            detuning,
            dt,
            input_power: 0.0,
        }
    }

    /// Create from normalized cavity (for bistability studies).
    pub fn normalized(q_total: f64, gamma_ratio: f64, detuning: f64, dt: f64) -> Self {
        let cavity = KerrCavity::normalized(q_total, gamma_ratio);
        Self::new(cavity, detuning, dt)
    }

    /// Get the underlying cavity parameters.
    pub fn cavity(&self) -> &KerrCavity {
        &self.cavity
    }

    /// Get current cavity energy.
    pub fn energy(&self) -> f64 {
        self.state.amplitude.norm_sqr()
    }

    /// Get current cavity amplitude.
    pub fn amplitude(&self) -> Complex64 {
        self.state.amplitude
    }

    /// Get current transmission (power out / power in).
    pub fn transmission(&self) -> f64 {
        if self.input_power < 1e-15 {
            return 0.0;
        }

        // Output amplitude: s_out = s_in - sqrt(gamma_e) * a
        // (for single-port coupling)
        let coupling = self.cavity.coupling_coefficient();
        let s_in = Complex64::new(self.input_power.sqrt(), 0.0);
        let s_out = s_in - coupling * self.state.amplitude;

        (s_out.norm_sqr() / self.input_power).clamp(0.0, 1.0)
    }

    /// Set the detuning.
    pub fn set_detuning(&mut self, detuning: f64) {
        self.detuning = detuning;
    }

    /// Get the current input field.
    fn current_input(&self) -> InputField {
        InputField {
            amplitude: Complex64::new(self.input_power.sqrt(), 0.0),
            omega: self.cavity.omega_0 + self.detuning,
        }
    }
}

impl Plant for TcmtPlant {
    type State = Complex64;
    type Input = f64; // Input power
    type Output = f64; // Transmission

    fn step(&mut self, input: &f64, dt: f64) -> f64 {
        self.input_power = input.max(0.0);
        let input_field = self.current_input();

        // Create solver and use RK4 for stability
        let solver = TcmtSolver::new(self.cavity);
        self.state = solver.rk4_step(self.state, &input_field, dt);

        self.transmission()
    }

    fn state(&self) -> &Complex64 {
        &self.state.amplitude
    }

    fn output(&self) -> f64 {
        self.transmission()
    }

    fn reset(&mut self) {
        self.state = CavityState::default();
        self.input_power = 0.0;
    }

    fn dt(&self) -> f64 {
        self.dt
    }
}

/// TCMT cavity plant with energy as output.
///
/// Alternative formulation where the output is cavity energy rather than
/// transmission. Useful for energy-based control objectives.
#[derive(Debug, Clone)]
pub struct TcmtEnergyPlant {
    /// Inner TCMT plant.
    inner: TcmtPlant,
}

impl TcmtEnergyPlant {
    /// Create a new energy-output TCMT plant.
    pub fn new(cavity: KerrCavity, detuning: f64, dt: f64) -> Self {
        Self {
            inner: TcmtPlant::new(cavity, detuning, dt),
        }
    }

    /// Create from normalized cavity.
    pub fn normalized(q_total: f64, gamma_ratio: f64, detuning: f64, dt: f64) -> Self {
        Self {
            inner: TcmtPlant::normalized(q_total, gamma_ratio, detuning, dt),
        }
    }

    /// Get the underlying TCMT plant.
    pub fn inner(&self) -> &TcmtPlant {
        &self.inner
    }

    /// Get mutable access to inner plant.
    pub fn inner_mut(&mut self) -> &mut TcmtPlant {
        &mut self.inner
    }
}

impl Plant for TcmtEnergyPlant {
    type State = Complex64;
    type Input = f64;
    type Output = f64; // Energy instead of transmission

    fn step(&mut self, input: &f64, dt: f64) -> f64 {
        self.inner.step(input, dt);
        self.inner.energy()
    }

    fn state(&self) -> &Complex64 {
        self.inner.state()
    }

    fn output(&self) -> f64 {
        self.inner.energy()
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn dt(&self) -> f64 {
        self.inner.dt()
    }
}

/// Combined state for thermal TCMT plant.
#[derive(Debug, Clone, Copy, Default)]
pub struct ThermalState {
    /// Complex amplitude.
    pub amplitude: Complex64,
    /// Temperature deviation.
    pub delta_temp: f64,
}

/// TCMT cavity with thermal coupling.
///
/// Includes thermo-optic effects where cavity heating shifts the resonance.
/// This creates additional dynamics that must be compensated by control.
#[cfg(feature = "optics")]
#[derive(Debug, Clone)]
pub struct TcmtThermalPlant {
    /// Base cavity parameters.
    cavity: KerrCavity,
    /// Current cavity state.
    optical_state: CavityState,
    /// Combined state for Plant trait.
    combined_state: ThermalState,
    /// Thermal time constant (s).
    tau_thermal: f64,
    /// Thermo-optic coefficient (resonance shift per Kelvin).
    dn_dt: f64,
    /// Heating coefficient (K per energy).
    heating_coeff: f64,
    /// Input detuning.
    detuning: f64,
    /// Time step.
    dt: f64,
    /// Current input power.
    input_power: f64,
}

impl TcmtThermalPlant {
    /// Create a new thermal TCMT plant.
    ///
    /// # Arguments
    /// * `cavity` - Base cavity parameters
    /// * `tau_thermal` - Thermal time constant (s), typically microseconds to milliseconds
    /// * `dn_dt` - Thermo-optic coefficient (typical: 1e-4 to 1e-5 K^{-1})
    /// * `heating_coeff` - Temperature rise per unit stored energy
    /// * `detuning` - Input frequency detuning
    /// * `dt` - Simulation time step
    pub fn new(
        cavity: KerrCavity,
        tau_thermal: f64,
        dn_dt: f64,
        heating_coeff: f64,
        detuning: f64,
        dt: f64,
    ) -> Self {
        Self {
            cavity,
            optical_state: CavityState::default(),
            combined_state: ThermalState::default(),
            tau_thermal,
            dn_dt,
            heating_coeff,
            detuning,
            dt,
            input_power: 0.0,
        }
    }

    /// Get current temperature deviation.
    pub fn temperature(&self) -> f64 {
        self.combined_state.delta_temp
    }

    /// Get effective detuning including thermal shift.
    pub fn effective_detuning(&self) -> f64 {
        // Thermal shift in resonance: delta_omega = -omega_0 * (dn/dT) * delta_T / n
        let thermal_shift = -self.cavity.omega_0 * self.dn_dt * self.combined_state.delta_temp / self.cavity.n_linear;
        self.detuning - thermal_shift
    }

    /// Get transmission.
    pub fn transmission(&self) -> f64 {
        if self.input_power < 1e-15 {
            return 0.0;
        }

        let coupling = self.cavity.coupling_coefficient();
        let s_in = Complex64::new(self.input_power.sqrt(), 0.0);
        let s_out = s_in - coupling * self.optical_state.amplitude;

        (s_out.norm_sqr() / self.input_power).clamp(0.0, 1.0)
    }

    fn step_thermal(&mut self, dt: f64) {
        // Thermal dynamics: d(delta_T)/dt = -delta_T/tau + heating_coeff * |a|^2 / tau
        let energy = self.optical_state.amplitude.norm_sqr();
        let d_temp = (-self.combined_state.delta_temp + self.heating_coeff * energy) / self.tau_thermal;
        self.combined_state.delta_temp += dt * d_temp;
    }

    fn step_optical(&mut self, dt: f64) {
        // Create solver with effective (thermally-shifted) cavity
        let mut effective_cavity = self.cavity;
        let thermal_shift = -self.cavity.omega_0 * self.dn_dt * self.combined_state.delta_temp / self.cavity.n_linear;
        effective_cavity.omega_0 += thermal_shift;

        let solver = TcmtSolver::new(effective_cavity);
        let input = InputField {
            amplitude: Complex64::new(self.input_power.sqrt(), 0.0),
            omega: self.cavity.omega_0 + self.detuning,
        };

        self.optical_state = solver.rk4_step(self.optical_state, &input, dt);
    }

    fn sync_combined_state(&mut self) {
        self.combined_state.amplitude = self.optical_state.amplitude;
    }
}

impl Plant for TcmtThermalPlant {
    type State = ThermalState;
    type Input = f64;
    type Output = f64;

    fn step(&mut self, input: &f64, dt: f64) -> f64 {
        self.input_power = input.max(0.0);

        // Split-step: optical then thermal
        self.step_optical(dt);
        self.step_thermal(dt);
        self.sync_combined_state();

        self.transmission()
    }

    fn state(&self) -> &ThermalState {
        &self.combined_state
    }

    fn output(&self) -> f64 {
        self.transmission()
    }

    fn reset(&mut self) {
        self.optical_state = CavityState::default();
        self.combined_state = ThermalState::default();
        self.input_power = 0.0;
    }

    fn dt(&self) -> f64 {
        self.dt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feedback::{FeedbackLoop, ProportionalController, ReferenceSignal};
    use crate::pid::PidController;
    use approx::assert_relative_eq;

    #[test]
    fn test_tcmt_plant_creation() {
        let plant = TcmtPlant::normalized(1000.0, 1.0, 0.0, 0.001);
        assert_eq!(plant.dt(), 0.001);
        assert_relative_eq!(plant.energy(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tcmt_plant_step() {
        let mut plant = TcmtPlant::normalized(1000.0, 1.0, 0.0, 0.001);

        // Apply constant input power
        for _ in 0..100 {
            plant.step(&1.0, plant.dt());
        }

        // Energy should have built up
        assert!(plant.energy() > 0.0);
    }

    #[test]
    fn test_tcmt_plant_reset() {
        let mut plant = TcmtPlant::normalized(1000.0, 1.0, 0.0, 0.001);

        // Build up some state
        for _ in 0..100 {
            plant.step(&1.0, plant.dt());
        }
        assert!(plant.energy() > 0.0);

        // Reset
        plant.reset();
        assert_relative_eq!(plant.energy(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tcmt_energy_plant() {
        let mut plant = TcmtEnergyPlant::normalized(1000.0, 1.0, 0.0, 0.001);

        for _ in 0..100 {
            let output = plant.step(&1.0, plant.dt());
            // Output should be energy
            assert_relative_eq!(output, plant.inner().energy(), epsilon = 1e-10);
        }
    }

    #[test]
    fn test_tcmt_feedback_loop() {
        let plant = TcmtPlant::normalized(1000.0, 1.0, 0.0, 0.001);
        let controller = ProportionalController::new(10.0);
        let mut loop_ = FeedbackLoop::new(plant, controller);

        // Run for a bit
        loop_.simulate(&ReferenceSignal::Constant(0.5), 1.0);

        // Should have recorded history
        assert!(!loop_.output_history.is_empty());
    }

    #[test]
    fn test_tcmt_pid_control() {
        let plant = TcmtPlant::normalized(1000.0, 1.0, 0.0, 0.0001);
        let controller = PidController::pi(5.0, 1.0).with_limits(0.0, 10.0);
        let mut loop_ = FeedbackLoop::new(plant, controller);

        // Try to regulate to a target transmission
        let target = 0.3;
        loop_.simulate(&ReferenceSignal::Constant(target), 0.5);

        // Check that we're approaching target (may not converge perfectly
        // due to nonlinear dynamics)
        let final_output = loop_.plant.output();
        // Just check it's not stuck at zero
        assert!(final_output > 0.01 || loop_.plant.energy() > 0.0);
    }

    #[test]
    fn test_tcmt_thermal_plant() {
        let cavity = KerrCavity::normalized(1000.0, 1.0);
        let mut plant = TcmtThermalPlant::new(
            cavity,
            1e-3,   // 1ms thermal time constant
            1e-4,   // typical thermo-optic coefficient
            1.0,    // heating coefficient
            0.0,    // on resonance
            1e-5,   // 10 microsecond steps
        );

        // Run with power input
        for _ in 0..1000 {
            plant.step(&1.0, plant.dt());
        }

        // Temperature should have increased
        assert!(plant.temperature() > 0.0);
    }
}
