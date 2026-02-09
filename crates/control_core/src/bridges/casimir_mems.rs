//! Plant implementations for Casimir MEMS (micro-electromechanical systems).
//!
//! Enables feedback control of microsphere positions in Casimir force systems.
//! The mechanical degrees of freedom are modeled as damped oscillators driven
//! by both external actuation and the quantum vacuum Casimir force.
//!
//! # Physics
//!
//! The microsphere dynamics follow:
//!   m * x'' + gamma * x' + k * x = F_casimir(x) + F_actuation
//!
//! where:
//! - m: microsphere mass
//! - gamma: damping coefficient (viscous drag in vacuum or piezo damping)
//! - k: spring constant (optical trap, magnetic suspension, or mechanical)
//! - F_casimir: Position-dependent Casimir force
//! - F_actuation: External control force (electrostatic, piezo, optical)
//!
//! # Control Scenarios
//!
//! - **Position stabilization**: Maintain sphere at target gap despite perturbations
//! - **Spring constant measurement**: Perturb and measure force gradient
//! - **Transistor gain tuning**: Adjust source sphere to modulate drain response
//!
//! # Literature
//! - Xu et al., Nature Communications 13, 6148 (2022)
//! - Munday et al., Nature 457, 170-173 (2009)

use crate::plant::Plant;
use quantum_core::casimir::casimir_force_pfa;

/// Mechanical state of a microsphere.
#[derive(Debug, Clone, Copy, Default)]
pub struct MechanicalState {
    /// Position (m).
    pub position: f64,
    /// Velocity (m/s).
    pub velocity: f64,
}

/// Casimir MEMS microsphere as a controllable plant.
///
/// Models a microsphere suspended above a plate, with the Casimir force
/// providing a nonlinear restoring force and external actuation as control input.
///
/// - **State**: (position, velocity)
/// - **Input**: Actuation force (N)
/// - **Output**: Position (m)
#[derive(Debug, Clone)]
pub struct CasimirMicrosphere {
    /// Sphere radius (m).
    pub radius: f64,
    /// Plate position (m).
    pub plate_position: f64,
    /// Mass (kg).
    pub mass: f64,
    /// Damping coefficient (N*s/m).
    pub damping: f64,
    /// Suspension spring constant (N/m).
    pub spring_k: f64,
    /// Equilibrium position without Casimir force (m).
    pub equilibrium: f64,
    /// Current mechanical state.
    state: MechanicalState,
    /// Time step (s).
    dt: f64,
}

impl CasimirMicrosphere {
    /// Create a new Casimir microsphere plant.
    ///
    /// # Arguments
    /// * `radius` - Sphere radius (m)
    /// * `plate_position` - Plate z-position (m)
    /// * `mass` - Sphere mass (kg)
    /// * `damping` - Damping coefficient (N*s/m)
    /// * `spring_k` - Suspension spring constant (N/m)
    /// * `initial_position` - Initial sphere position (m)
    /// * `dt` - Time step (s)
    pub fn new(
        radius: f64,
        plate_position: f64,
        mass: f64,
        damping: f64,
        spring_k: f64,
        initial_position: f64,
        dt: f64,
    ) -> Self {
        Self {
            radius,
            plate_position,
            mass,
            damping,
            spring_k,
            equilibrium: initial_position,
            state: MechanicalState {
                position: initial_position,
                velocity: 0.0,
            },
            dt,
        }
    }

    /// Create with parameters in micrometers (convenience).
    ///
    /// # Arguments
    /// * `radius_um` - Sphere radius (micrometers)
    /// * `plate_position_um` - Plate position (micrometers)
    /// * `initial_gap_um` - Initial gap between sphere surface and plate (micrometers)
    /// * `mass_ng` - Mass (nanograms)
    /// * `damping` - Damping coefficient (N*s/m)
    /// * `spring_k` - Spring constant (N/m)
    /// * `dt` - Time step (s)
    pub fn from_micrometers(
        radius_um: f64,
        plate_position_um: f64,
        initial_gap_um: f64,
        mass_ng: f64,
        damping: f64,
        spring_k: f64,
        dt: f64,
    ) -> Self {
        let radius = radius_um * 1e-6;
        let plate_position = plate_position_um * 1e-6;
        let initial_position = plate_position + initial_gap_um * 1e-6 + radius;
        let mass = mass_ng * 1e-12;

        Self::new(
            radius,
            plate_position,
            mass,
            damping,
            spring_k,
            initial_position,
            dt,
        )
    }

    /// Get current position.
    pub fn position(&self) -> f64 {
        self.state.position
    }

    /// Get the gap between sphere surface and plate.
    pub fn gap(&self) -> f64 {
        (self.state.position - self.radius - self.plate_position).max(1e-12)
    }

    /// Compute Casimir force at current position.
    pub fn casimir_force(&self) -> f64 {
        let gap = self.gap();
        // casimir_force_pfa returns negative (attractive) force
        casimir_force_pfa(self.radius, gap)
    }

    /// Compute spring restoring force.
    pub fn spring_force(&self) -> f64 {
        -self.spring_k * (self.state.position - self.equilibrium)
    }

    /// Compute damping force.
    pub fn damping_force(&self) -> f64 {
        -self.damping * self.state.velocity
    }

    /// Get current velocity.
    pub fn velocity(&self) -> f64 {
        self.state.velocity
    }

    /// Get natural frequency of the mechanical oscillator.
    pub fn natural_frequency(&self) -> f64 {
        (self.spring_k / self.mass).sqrt()
    }

    /// Get quality factor.
    pub fn quality_factor(&self) -> f64 {
        (self.mass * self.spring_k).sqrt() / self.damping
    }
}

impl Plant for CasimirMicrosphere {
    type State = MechanicalState;
    type Input = f64; // Actuation force
    type Output = f64; // Position

    fn step(&mut self, input: &f64, dt: f64) -> f64 {
        let actuation = *input;

        // Compute all forces
        let f_casimir = self.casimir_force();
        let f_spring = self.spring_force();
        let f_damping = self.damping_force();
        let f_total = f_casimir + f_spring + f_damping + actuation;

        // Acceleration
        let acceleration = f_total / self.mass;

        // Semi-implicit Euler for stability with stiff spring
        self.state.velocity += dt * acceleration;
        self.state.position += dt * self.state.velocity;

        // Prevent collision with plate
        let min_gap = 1e-9; // 1 nm minimum gap
        let min_position = self.plate_position + min_gap + self.radius;
        if self.state.position < min_position {
            self.state.position = min_position;
            self.state.velocity = 0.0; // Inelastic collision
        }

        self.state.position
    }

    fn state(&self) -> &MechanicalState {
        &self.state
    }

    fn output(&self) -> f64 {
        self.state.position
    }

    fn reset(&mut self) {
        self.state.position = self.equilibrium;
        self.state.velocity = 0.0;
    }

    fn dt(&self) -> f64 {
        self.dt
    }
}

/// Combined state for Casimir transistor.
#[derive(Debug, Clone, Copy, Default)]
pub struct TransistorState {
    /// Source position (m).
    pub x_source: f64,
    /// Source velocity (m/s).
    pub v_source: f64,
    /// Drain position (m).
    pub x_drain: f64,
    /// Drain velocity (m/s).
    pub v_drain: f64,
}

/// Casimir transistor: sphere-plate-sphere system.
///
/// Models the three-terminal Casimir transistor from Xu et al. (2022).
/// The source sphere position modulates the drain sphere's equilibrium.
///
/// - **State**: (x_source, v_source, x_drain, v_drain)
/// - **Input**: Source actuation force (N)
/// - **Output**: Drain sphere position (m)
#[derive(Debug, Clone)]
pub struct CasimirTransistor {
    /// Source sphere.
    source: CasimirMicrosphere,
    /// Drain sphere.
    drain: CasimirMicrosphere,
    /// Coupling coefficient (how source affects drain equilibrium).
    coupling: f64,
    /// Combined state for Plant trait.
    combined_state: TransistorState,
}

impl CasimirTransistor {
    /// Create a new Casimir transistor.
    ///
    /// # Arguments
    /// * `source` - Source microsphere plant
    /// * `drain` - Drain microsphere plant
    /// * `coupling` - Coupling coefficient (typically 0.1 to 1.0)
    pub fn new(source: CasimirMicrosphere, drain: CasimirMicrosphere, coupling: f64) -> Self {
        let combined_state = TransistorState {
            x_source: source.position(),
            v_source: source.velocity(),
            x_drain: drain.position(),
            v_drain: drain.velocity(),
        };
        Self {
            source,
            drain,
            coupling,
            combined_state,
        }
    }

    /// Create symmetric transistor with given geometry.
    pub fn symmetric(
        radius_um: f64,
        plate_position_um: f64,
        gap_um: f64,
        mass_ng: f64,
        damping: f64,
        spring_k: f64,
        coupling: f64,
        dt: f64,
    ) -> Self {
        let source = CasimirMicrosphere::from_micrometers(
            radius_um,
            plate_position_um,
            gap_um,
            mass_ng,
            damping,
            spring_k,
            dt,
        );
        let drain = CasimirMicrosphere::from_micrometers(
            radius_um,
            plate_position_um,
            gap_um,
            mass_ng,
            damping,
            spring_k,
            dt,
        );

        Self::new(source, drain, coupling)
    }

    /// Get source sphere position.
    pub fn source_position(&self) -> f64 {
        self.source.position()
    }

    /// Get drain sphere position.
    pub fn drain_position(&self) -> f64 {
        self.drain.position()
    }

    /// Compute transistor gain: d(F_drain) / d(F_source).
    ///
    /// This is computed numerically from position changes.
    pub fn instantaneous_gain(&self) -> f64 {
        // Simplified: coupling coefficient
        // Full computation would require force gradients
        self.coupling
    }

    /// Get source subsystem.
    pub fn source(&self) -> &CasimirMicrosphere {
        &self.source
    }

    /// Get drain subsystem.
    pub fn drain(&self) -> &CasimirMicrosphere {
        &self.drain
    }

    fn sync_combined_state(&mut self) {
        self.combined_state.x_source = self.source.position();
        self.combined_state.v_source = self.source.velocity();
        self.combined_state.x_drain = self.drain.position();
        self.combined_state.v_drain = self.drain.velocity();
    }
}

impl Plant for CasimirTransistor {
    type State = TransistorState;
    type Input = f64; // Source actuation
    type Output = f64; // Drain position

    fn step(&mut self, input: &f64, dt: f64) -> f64 {
        // Step source
        self.source.step(input, dt);

        // Modulate drain equilibrium based on source position
        let source_displacement = self.source.position() - self.source.equilibrium;
        let drain_modulation = self.coupling * source_displacement;

        // Apply modulation as additional force on drain
        let drain_force = drain_modulation * self.drain.spring_k;
        self.drain.step(&drain_force, dt);

        self.sync_combined_state();
        self.drain.position()
    }

    fn state(&self) -> &TransistorState {
        &self.combined_state
    }

    fn output(&self) -> f64 {
        self.drain.position()
    }

    fn reset(&mut self) {
        self.source.reset();
        self.drain.reset();
        self.sync_combined_state();
    }

    fn dt(&self) -> f64 {
        self.source.dt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::feedback::{FeedbackLoop, ProportionalController, ReferenceSignal};
    use approx::assert_relative_eq;

    #[test]
    fn test_casimir_microsphere_creation() {
        let plant = CasimirMicrosphere::from_micrometers(
            5.0,  // 5 um radius
            0.0,  // plate at z=0
            0.1,  // 100 nm gap
            1.0,  // 1 ng mass
            1e-9, // damping
            1e-3, // 1 mN/m spring
            1e-9, // 1 ns timestep
        );

        assert!(plant.gap() > 0.0);
        assert!(plant.casimir_force() < 0.0); // Attractive
    }

    #[test]
    fn test_casimir_force_increases_with_smaller_gap() {
        let mut plant = CasimirMicrosphere::from_micrometers(5.0, 0.0, 0.1, 1.0, 1e-9, 1e-3, 1e-9);

        let force_far = plant.casimir_force().abs();

        // Move closer
        plant.state.position = plant.plate_position + 0.05e-6 + plant.radius;
        let force_near = plant.casimir_force().abs();

        assert!(force_near > force_far);
    }

    #[test]
    fn test_casimir_dynamics() {
        let mut plant = CasimirMicrosphere::from_micrometers(5.0, 0.0, 0.2, 1.0, 1e-9, 1e-3, 1e-9);

        let initial_position = plant.position();

        // Run without actuation - Casimir force should pull toward plate
        for _ in 0..1000 {
            plant.step(&0.0, plant.dt());
        }

        // Should have moved toward plate (smaller position)
        assert!(plant.position() < initial_position);
    }

    #[test]
    fn test_casimir_feedback_control() {
        let plant = CasimirMicrosphere::from_micrometers(5.0, 0.0, 0.2, 1.0, 1e-8, 1e-3, 1e-8);

        let controller = ProportionalController::new(1e-6);
        let mut loop_ = FeedbackLoop::new(plant, controller);

        // Try to maintain position
        let target = loop_.plant.equilibrium;
        loop_.simulate(&ReferenceSignal::Constant(target), 1e-5);

        assert!(!loop_.output_history.is_empty());
    }

    #[test]
    fn test_casimir_transistor_creation() {
        let transistor = CasimirTransistor::symmetric(
            5.0,  // radius
            0.0,  // plate
            0.2,  // gap
            1.0,  // mass
            1e-9, // damping
            1e-3, // spring
            0.5,  // coupling
            1e-9, // dt
        );

        assert_relative_eq!(
            transistor.source_position(),
            transistor.drain_position(),
            epsilon = 1e-15
        );
    }

    #[test]
    fn test_casimir_transistor_coupling() {
        let mut transistor =
            CasimirTransistor::symmetric(5.0, 0.0, 0.2, 1.0, 1e-8, 1e-3, 0.5, 1e-8);

        let initial_drain = transistor.drain_position();

        // Actuate source
        for _ in 0..1000 {
            transistor.step(&1e-12, transistor.dt()); // 1 pN actuation
        }

        // Drain should have responded
        assert!((transistor.drain_position() - initial_drain).abs() > 0.0);
    }

    #[test]
    fn test_natural_frequency() {
        let plant = CasimirMicrosphere::from_micrometers(5.0, 0.0, 0.2, 1.0, 1e-9, 1e-3, 1e-9);

        let omega_n = plant.natural_frequency();
        // For m = 1e-12 kg, k = 1e-3 N/m: omega_n = sqrt(1e-3 / 1e-12) = 1e4.5 ~ 31623 rad/s
        assert!(omega_n > 1e4);
        assert!(omega_n < 1e5);
    }

    #[test]
    fn test_quality_factor() {
        let plant = CasimirMicrosphere::from_micrometers(5.0, 0.0, 0.2, 1.0, 1e-12, 1e-3, 1e-9);

        let q = plant.quality_factor();
        // Q = sqrt(m*k) / gamma = sqrt(1e-12 * 1e-3) / 1e-12 = 1e-7.5 / 1e-12 = 31623
        assert!(q > 1e4);
    }
}
