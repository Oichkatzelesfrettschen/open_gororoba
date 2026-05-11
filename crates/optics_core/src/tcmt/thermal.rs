// ============================================================================
// Thermal Dynamics Extension
// ============================================================================
//
// Thermal nonlinearity arises from optical absorption generating heat, heat
// changing the refractive index (thermo-optic effect), and the resulting
// temperature-dependent resonance shift. See Johnson et al. (2006) and
// Carmon et al. (2004) for the foundational treatment of thermal
// nonlinearity in microcavities.

use num_complex::Complex64;

use super::{InputField, KerrCavity};

/// Thermal properties for a cavity with thermo-optic coupling.
///
/// The thermal nonlinearity arises from:
/// 1. Optical absorption generating heat
/// 2. Heat changing the refractive index (thermo-optic effect)
/// 3. Temperature-dependent resonance shift
///
/// # Coupled Equations
/// ```text
/// da/dt = (j*delta_eff - gamma/2)*a + sqrt(gamma_e)*s_in
/// dT/dt = P_abs / C_th - (T - T_amb) / tau_th
///
/// where:
///   delta_eff = delta - gamma_kerr*|a|^2 - (dn/dT)*(omega_0/n_0)*(T - T_0)
///   P_abs = gamma_0 * |a|^2 * hbar * omega_0
/// ```
///
/// # Literature
/// - Johnson et al., Opt. Express 14, 817 (2006) - Thermal nonlinearity in microcavities
/// - Carmon et al., Opt. Express 12, 4742 (2004) - Thermal oscillations
#[derive(Debug, Clone, Copy)]
pub struct ThermalCavity {
    /// Base Kerr cavity parameters.
    pub kerr: KerrCavity,

    /// Thermo-optic coefficient dn/dT (K^-1).
    /// Typical values: Si ~ 1.8e-4, SiO2 ~ 1.0e-5, InP ~ 2.0e-4.
    pub dn_dt: f64,

    /// Thermal time constant tau_th (s).
    /// Heat dissipation: dT/dt ~ -(T - T_amb) / tau_th
    /// Typical values: 1-100 microseconds for microcavities.
    pub tau_thermal: f64,

    /// Thermal capacitance C_th (J/K).
    /// Determines heating rate: dT/dt = P_abs / C_th
    pub heat_capacity: f64,

    /// Ambient temperature (K).
    pub t_ambient: f64,

    /// Reference temperature for resonance (K).
    pub t_reference: f64,
}

impl ThermalCavity {
    /// Creates a new thermal cavity.
    pub fn new(
        kerr: KerrCavity,
        dn_dt: f64,
        tau_thermal: f64,
        heat_capacity: f64,
        t_ambient: f64,
    ) -> Self {
        Self {
            kerr,
            dn_dt,
            tau_thermal,
            heat_capacity,
            t_ambient,
            t_reference: t_ambient,
        }
    }

    /// Creates a silicon microcavity with typical thermal parameters.
    ///
    /// # Arguments
    /// * `kerr` - Base Kerr cavity
    /// * `tau_thermal` - Thermal time constant (s), typically 1-100 us
    pub fn silicon(kerr: KerrCavity, tau_thermal: f64) -> Self {
        Self {
            kerr,
            dn_dt: 1.8e-4, // K^-1 for Si
            tau_thermal,
            heat_capacity: 1e-12, // J/K (order of magnitude for microcavity)
            t_ambient: 300.0,
            t_reference: 300.0,
        }
    }

    /// Thermal frequency shift coefficient (rad/s per Kelvin).
    ///
    /// delta_omega_thermal = -gamma_thermal * (T - T_ref)
    pub fn gamma_thermal(&self) -> f64 {
        self.dn_dt * self.kerr.omega_0 / self.kerr.n_linear
    }

    /// Absorbed power for given cavity energy (W).
    ///
    /// P_abs = gamma_intrinsic * |a|^2 * hbar * omega_0
    pub fn absorbed_power(&self, a_norm_sq: f64) -> f64 {
        // hbar * omega_0 converts normalized energy to Joules
        // For normalized units where hbar*omega = 1, this simplifies
        self.kerr.gamma_intrinsic() * a_norm_sq
    }

    /// Thermal relaxation rate (1/s).
    pub fn gamma_th(&self) -> f64 {
        1.0 / self.tau_thermal
    }
}

/// State of cavity with thermal dynamics.
#[derive(Debug, Clone, Copy)]
pub struct ThermalCavityState {
    /// Complex amplitude a.
    pub amplitude: Complex64,
    /// Temperature deviation from reference (K).
    pub temperature: f64,
    /// Current time (s).
    pub time: f64,
}

impl Default for ThermalCavityState {
    fn default() -> Self {
        Self {
            amplitude: Complex64::new(0.0, 0.0),
            temperature: 0.0, // At reference temperature
            time: 0.0,
        }
    }
}

impl ThermalCavityState {
    /// Stored optical energy |a|^2.
    pub fn energy(&self) -> f64 {
        self.amplitude.norm_sqr()
    }

    /// Absolute temperature (K).
    pub fn absolute_temperature(&self, t_ref: f64) -> f64 {
        t_ref + self.temperature
    }
}

/// Result of thermal steady-state analysis.
#[derive(Debug, Clone)]
pub struct ThermalSteadyStateResult {
    /// Cavity amplitude solutions.
    pub amplitudes: Vec<Complex64>,
    /// Temperature for each solution (K above reference).
    pub temperatures: Vec<f64>,
    /// Stored energies.
    pub energies: Vec<f64>,
    /// Total effective detuning (Kerr + thermal) for each solution.
    pub effective_detunings: Vec<f64>,
    /// Stability of each solution.
    pub stability: Vec<bool>,
    /// Number of solutions.
    pub num_solutions: usize,
}

/// Solver for thermal TCMT dynamics.
pub struct ThermalTcmtSolver {
    pub cavity: ThermalCavity,
}

impl ThermalTcmtSolver {
    /// Creates a new thermal TCMT solver.
    pub fn new(cavity: ThermalCavity) -> Self {
        Self { cavity }
    }

    /// Computes (da/dt, dT/dt) for coupled thermo-optical dynamics.
    pub fn derivative(&self, state: &ThermalCavityState, input: &InputField) -> (Complex64, f64) {
        let a = state.amplitude;
        let delta_t = state.temperature;
        let a_norm_sq = a.norm_sqr();

        // Effective detuning including thermal shift
        let delta_base = input.omega - self.cavity.kerr.omega_0;
        let delta_kerr = -self.cavity.kerr.gamma_kerr() * a_norm_sq;
        let delta_thermal = -self.cavity.gamma_thermal() * delta_t;
        let delta_eff = delta_base + delta_kerr + delta_thermal;

        // Optical amplitude dynamics
        let gamma_tot = self.cavity.kerr.gamma_total();
        let linear = Complex64::new(-gamma_tot / 2.0, delta_eff);
        let coupling = self.cavity.kerr.coupling_coefficient() * input.amplitude;
        let da_dt = linear * a + coupling;

        // Thermal dynamics: dT/dt = P_abs/C - (T - T_amb)/tau
        // Here T is measured from T_ambient, so dT/dt = P_abs/C - T/tau
        let p_abs = self.cavity.absorbed_power(a_norm_sq);
        let heating_rate = p_abs / self.cavity.heat_capacity;
        let cooling_rate = delta_t / self.cavity.tau_thermal;
        let dt_dt = heating_rate - cooling_rate;

        (da_dt, dt_dt)
    }

    /// RK4 step for coupled thermo-optical system.
    pub fn rk4_step(
        &self,
        state: &ThermalCavityState,
        input: &InputField,
        dt: f64,
    ) -> ThermalCavityState {
        let (k1a, k1t) = self.derivative(state, input);

        let state2 = ThermalCavityState {
            amplitude: state.amplitude + 0.5 * dt * k1a,
            temperature: state.temperature + 0.5 * dt * k1t,
            time: state.time + 0.5 * dt,
        };
        let (k2a, k2t) = self.derivative(&state2, input);

        let state3 = ThermalCavityState {
            amplitude: state.amplitude + 0.5 * dt * k2a,
            temperature: state.temperature + 0.5 * dt * k2t,
            time: state.time + 0.5 * dt,
        };
        let (k3a, k3t) = self.derivative(&state3, input);

        let state4 = ThermalCavityState {
            amplitude: state.amplitude + dt * k3a,
            temperature: state.temperature + dt * k3t,
            time: state.time + dt,
        };
        let (k4a, k4t) = self.derivative(&state4, input);

        ThermalCavityState {
            amplitude: state.amplitude + dt / 6.0 * (k1a + 2.0 * k2a + 2.0 * k3a + k4a),
            temperature: state.temperature + dt / 6.0 * (k1t + 2.0 * k2t + 2.0 * k3t + k4t),
            time: state.time + dt,
        }
    }

    /// Evolve the thermal cavity for a given number of steps.
    pub fn evolve(
        &self,
        initial: ThermalCavityState,
        input: &InputField,
        dt: f64,
        n_steps: usize,
    ) -> Vec<ThermalCavityState> {
        let mut states = Vec::with_capacity(n_steps + 1);
        states.push(initial);

        let mut state = initial;
        for _ in 0..n_steps {
            state = self.rk4_step(&state, input, dt);
            states.push(state);
        }

        states
    }

    /// Find thermal steady state by iterating until convergence.
    ///
    /// Returns None if oscillations or instability prevent convergence.
    pub fn find_steady_state(
        &self,
        input: &InputField,
        tolerance: f64,
        max_iterations: usize,
    ) -> Option<ThermalCavityState> {
        // Start from cold cavity
        let mut state = ThermalCavityState::default();

        // Time step: use smaller of optical and thermal time scales
        let tau_opt = 1.0 / self.cavity.kerr.gamma_total();
        let dt = tau_opt.min(self.cavity.tau_thermal / 10.0);

        // Iterate until energy and temperature stabilize
        for _ in 0..max_iterations {
            let prev_energy = state.energy();
            let prev_temp = state.temperature;

            // Evolve for one thermal time constant
            let n_steps = (self.cavity.tau_thermal / dt).ceil() as usize;
            let states = self.evolve(state, input, dt, n_steps);
            state = *states.last().unwrap();

            // Check convergence
            let energy_change = (state.energy() - prev_energy).abs() / (prev_energy + 1e-20);
            let temp_change = (state.temperature - prev_temp).abs() / (prev_temp.abs() + 1e-10);

            if energy_change < tolerance && temp_change < tolerance {
                return Some(state);
            }
        }

        None
    }

    /// Thermal bistability threshold in terms of input power.
    ///
    /// Thermal nonlinearity adds to Kerr, so total effective gamma is:
    /// gamma_eff = gamma_kerr + gamma_thermal * (tau_thermal * gamma_intrinsic / C_th)
    ///
    /// Bistability requires total nonlinearity to exceed sqrt(3) * linewidth/2.
    pub fn effective_gamma(&self) -> f64 {
        // Thermal contribution: temperature rise per unit stored energy
        // At steady state: T = tau_th * P_abs / C_th = tau_th * gamma_0 * |a|^2 / C_th
        // Thermal shift: delta_omega_th = gamma_th * T = gamma_th * tau_th * gamma_0 * |a|^2 / C_th
        let thermal_factor = self.cavity.gamma_thermal()
            * self.cavity.tau_thermal
            * self.cavity.kerr.gamma_intrinsic()
            / self.cavity.heat_capacity;

        self.cavity.kerr.gamma_kerr() + thermal_factor
    }

    /// Check if thermal bistability is possible.
    pub fn thermal_bistability_possible(&self) -> bool {
        // Bistability requires |delta| > sqrt(3) * g for some detuning
        // With thermal nonlinearity, the threshold is lower
        let g = self.cavity.kerr.gamma_total() / 2.0;
        let gamma_eff = self.effective_gamma();

        // Any nonlinearity can produce bistability at sufficient power
        gamma_eff > 0.0 && g > 0.0
    }
}

/// Timescale comparison for thermal dynamics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ThermalRegime {
    /// Thermal time >> optical time: adiabatic thermal following
    Adiabatic,
    /// Comparable timescales: coupled dynamics
    Intermediate,
    /// Thermal time << optical time: thermal equilibrium
    FastThermal,
}

/// Classify the thermal regime for a given cavity.
pub fn thermal_regime(cavity: &ThermalCavity) -> ThermalRegime {
    let tau_opt = 1.0 / cavity.kerr.gamma_total();
    let ratio = cavity.tau_thermal / tau_opt;

    if ratio > 100.0 {
        ThermalRegime::Adiabatic
    } else if ratio < 0.01 {
        ThermalRegime::FastThermal
    } else {
        ThermalRegime::Intermediate
    }
}
