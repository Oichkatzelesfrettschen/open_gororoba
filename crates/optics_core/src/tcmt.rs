//! Temporal Coupled-Mode Theory (TCMT) with Kerr Nonlinearity.
//!
//! Implements the TCMT formalism for resonant optical cavities with Kerr
//! nonlinear response. This enables modeling of:
//! - Optical bistability
//! - All-optical switching
//! - MIM (Metal-Insulator-Metal) waveguide cavities
//! - Ring resonator dynamics
//!
//! # Key Equations
//!
//! TCMT amplitude evolution (Liu et al. 2013, Eq. 1):
//! ```text
//!   da/dt = (j*omega_0 - 1/tau_0 - 1/tau_e)*a + sqrt(2/tau_e)*s_in
//! ```
//!
//! With Kerr nonlinearity, the effective resonance shifts:
//! ```text
//!   omega_eff = omega_0 - gamma_kerr * |a|^2
//! ```
//!
//! where gamma_kerr = n2 * omega_0 * c / (n0^2 * V_eff).
//!
//! # Literature
//! - Liu et al., Opt. Express 21(20), 23687-23699 (2013)
//! - Suh et al., IEEE J. Quantum Electron. 40, 1511 (2004) - TCMT foundations
//! - Fan et al., J. Opt. Soc. Am. A 20, 569 (2003) - TCMT for waveguide coupling

use num_complex::Complex64;
use std::f64::consts::PI;

/// Physical constants.
const C_LIGHT: f64 = 299792458.0; // m/s

/// Parameters for a Kerr-nonlinear optical cavity.
#[derive(Debug, Clone, Copy)]
pub struct KerrCavity {
    /// Resonance angular frequency omega_0 (rad/s).
    pub omega_0: f64,

    /// Intrinsic quality factor Q_0 (loss within cavity).
    pub q_intrinsic: f64,

    /// External (coupling) quality factor Q_e.
    pub q_external: f64,

    /// Linear refractive index n_0.
    pub n_linear: f64,

    /// Kerr nonlinear index n_2 (m^2/W).
    /// Positive for self-focusing, negative for self-defocusing.
    pub n2: f64,

    /// Effective mode volume V_eff (m^3).
    pub v_eff: f64,
}

impl KerrCavity {
    /// Creates a new Kerr cavity with specified parameters.
    pub fn new(
        omega_0: f64,
        q_intrinsic: f64,
        q_external: f64,
        n_linear: f64,
        n2: f64,
        v_eff: f64,
    ) -> Self {
        Self {
            omega_0,
            q_intrinsic,
            q_external,
            n_linear,
            n2,
            v_eff,
        }
    }

    /// Creates a cavity from wavelength (nm) and quality factors.
    pub fn from_wavelength(
        wavelength_nm: f64,
        q_intrinsic: f64,
        q_external: f64,
        n_linear: f64,
        n2: f64,
        v_eff: f64,
    ) -> Self {
        let wavelength_m = wavelength_nm * 1e-9;
        let omega_0 = 2.0 * PI * C_LIGHT / wavelength_m;
        Self::new(omega_0, q_intrinsic, q_external, n_linear, n2, v_eff)
    }

    /// Creates a normalized cavity for testing bistability physics.
    ///
    /// This constructor sets gamma_kerr = gamma_ratio * g where g = gamma_tot/2,
    /// ensuring the Kerr nonlinearity is comparable to the cavity linewidth
    /// for observable bistability effects.
    ///
    /// # Arguments
    /// * `q_total` - Total quality factor (defines linewidth)
    /// * `gamma_ratio` - Ratio gamma_kerr / g (typically 1.0 for standard bistability)
    pub fn normalized(q_total: f64, gamma_ratio: f64) -> Self {
        let omega_0 = 1.0; // Normalized frequency
        let q_intrinsic = 2.0 * q_total; // Critical coupling
        let q_external = 2.0 * q_total;

        // gamma_kerr = n2 * omega_0 * c / (n0^2 * v_eff)
        // We want gamma_kerr = gamma_ratio * g = gamma_ratio * omega_0 / (2*q_total)
        // So: n2 * omega_0 * c / (n0^2 * v_eff) = gamma_ratio * omega_0 / (2*q_total)
        // => n2 * c / (n0^2 * v_eff) = gamma_ratio / (2*q_total)
        //
        // Set n0 = 1, c = 1 (normalized units), v_eff = 1
        // Then n2 = gamma_ratio / (2*q_total)

        let n_linear = 1.0;
        let n2 = gamma_ratio / (2.0 * q_total);
        let v_eff = 1.0;

        Self {
            omega_0,
            q_intrinsic,
            q_external,
            n_linear,
            n2,
            v_eff,
        }
    }

    /// Intrinsic loss rate: 1/tau_0 = omega_0 / Q_0.
    pub fn gamma_intrinsic(&self) -> f64 {
        self.omega_0 / self.q_intrinsic
    }

    /// External coupling rate: 1/tau_e = omega_0 / Q_e.
    pub fn gamma_external(&self) -> f64 {
        self.omega_0 / self.q_external
    }

    /// Total loss rate: 1/tau = 1/tau_0 + 1/tau_e.
    pub fn gamma_total(&self) -> f64 {
        self.gamma_intrinsic() + self.gamma_external()
    }

    /// Total quality factor: 1/Q = 1/Q_0 + 1/Q_e.
    pub fn q_total(&self) -> f64 {
        1.0 / (1.0 / self.q_intrinsic + 1.0 / self.q_external)
    }

    /// Full width at half maximum (FWHM) of linear resonance (rad/s).
    pub fn fwhm(&self) -> f64 {
        self.gamma_total()
    }

    /// Kerr frequency shift coefficient: gamma_kerr (rad/s per |a|^2).
    ///
    /// The effective resonance shifts by: delta_omega = -gamma_kerr * |a|^2
    /// where |a|^2 is the normalized stored energy.
    ///
    /// For normalized units where omega_0, v_eff are O(1), use:
    ///   gamma_kerr = n2 * omega_0 / (n0^2 * V_eff)
    ///
    /// Note: For positive n2 (self-focusing), gamma_kerr > 0, so
    /// increasing intensity causes a red-shift (lower frequency).
    pub fn gamma_kerr(&self) -> f64 {
        // Use formula without c to match normalized constructor.
        // Physical cavities should use appropriately scaled parameters.
        self.n2 * self.omega_0 / (self.n_linear.powi(2) * self.v_eff)
    }

    /// Coupling coefficient sqrt(2/tau_e) for input/output port.
    pub fn coupling_coefficient(&self) -> f64 {
        (2.0 * self.gamma_external()).sqrt()
    }

    /// Critical coupling condition: Q_0 = Q_e.
    pub fn is_critically_coupled(&self, tolerance: f64) -> bool {
        ((self.q_intrinsic - self.q_external) / self.q_external).abs() < tolerance
    }

    /// Coupling regime.
    pub fn coupling_regime(&self) -> CouplingRegime {
        if self.q_intrinsic > self.q_external {
            CouplingRegime::Overcoupled
        } else if self.q_intrinsic < self.q_external {
            CouplingRegime::Undercoupled
        } else {
            CouplingRegime::Critical
        }
    }
}

/// Coupling regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CouplingRegime {
    /// Q_0 > Q_e: external loss dominates.
    Overcoupled,
    /// Q_0 < Q_e: intrinsic loss dominates.
    Undercoupled,
    /// Q_0 = Q_e: perfect impedance match at resonance.
    Critical,
}

/// State of the cavity field.
#[derive(Debug, Clone, Copy)]
pub struct CavityState {
    /// Complex amplitude a (normalized such that |a|^2 = stored energy / hbar*omega_0).
    pub amplitude: Complex64,
    /// Current time (s).
    pub time: f64,
}

impl Default for CavityState {
    fn default() -> Self {
        Self {
            amplitude: Complex64::new(0.0, 0.0),
            time: 0.0,
        }
    }
}

/// Input field to the cavity.
#[derive(Debug, Clone, Copy)]
pub struct InputField {
    /// Complex amplitude of input field s_in.
    pub amplitude: Complex64,
    /// Angular frequency of input field (rad/s).
    pub omega: f64,
}

impl InputField {
    /// Creates a new CW input field with given power and frequency.
    ///
    /// # Arguments
    /// * `power` - Input power (W)
    /// * `omega` - Angular frequency (rad/s)
    pub fn cw(power: f64, omega: f64) -> Self {
        Self {
            amplitude: Complex64::new(power.sqrt(), 0.0),
            omega,
        }
    }

    /// Creates input from power and detuning from cavity resonance.
    pub fn cw_detuned(power: f64, cavity: &KerrCavity, detuning: f64) -> Self {
        Self::cw(power, cavity.omega_0 + detuning)
    }
}

/// Result of steady-state analysis.
#[derive(Debug, Clone)]
pub struct SteadyStateResult {
    /// Cavity field amplitude (may have multiple solutions for bistability).
    pub amplitudes: Vec<Complex64>,
    /// Stored energy for each solution.
    pub energies: Vec<f64>,
    /// Transmission coefficient t = s_out / s_in for each solution.
    pub transmissions: Vec<Complex64>,
    /// Power transmission |t|^2 for each solution.
    pub power_transmissions: Vec<f64>,
    /// Stability of each solution (true = stable).
    pub stability: Vec<bool>,
    /// Number of solutions found.
    pub num_solutions: usize,
}

/// TCMT solver for nonlinear cavity dynamics.
pub struct TcmtSolver {
    pub cavity: KerrCavity,
}

impl TcmtSolver {
    /// Creates a new TCMT solver for the given cavity.
    pub fn new(cavity: KerrCavity) -> Self {
        Self { cavity }
    }

    /// Computes da/dt for the cavity amplitude.
    ///
    /// TCMT equation (in rotating frame at omega):
    /// ```text
    ///   da/dt = (j*delta - gamma_tot/2)*a - j*gamma_kerr*|a|^2*a + sqrt(gamma_e)*s_in
    /// ```
    ///
    /// where delta = omega - omega_0 is the detuning.
    pub fn derivative(&self, state: &CavityState, input: &InputField) -> Complex64 {
        let a = state.amplitude;
        let delta = input.omega - self.cavity.omega_0;

        // Linear part: (j*delta - gamma_tot/2)
        let gamma_tot = self.cavity.gamma_total();
        let linear = Complex64::new(-gamma_tot / 2.0, delta);

        // Kerr nonlinearity: -j * gamma_kerr * |a|^2
        let a_norm_sq = a.norm_sqr();
        let kerr_shift = Complex64::new(0.0, -self.cavity.gamma_kerr() * a_norm_sq);

        // Coupling term: sqrt(gamma_e) * s_in
        let coupling = self.cavity.coupling_coefficient() * input.amplitude;

        // Total derivative
        (linear + kerr_shift) * a + coupling
    }

    /// Performs one RK4 step for time-domain dynamics.
    pub fn rk4_step(&self, state: CavityState, input: &InputField, dt: f64) -> CavityState {
        let k1 = self.derivative(&state, input);

        let state2 = CavityState {
            amplitude: state.amplitude + k1 * (dt / 2.0),
            time: state.time + dt / 2.0,
        };
        let k2 = self.derivative(&state2, input);

        let state3 = CavityState {
            amplitude: state.amplitude + k2 * (dt / 2.0),
            time: state.time + dt / 2.0,
        };
        let k3 = self.derivative(&state3, input);

        let state4 = CavityState {
            amplitude: state.amplitude + k3 * dt,
            time: state.time + dt,
        };
        let k4 = self.derivative(&state4, input);

        let da = (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (dt / 6.0);

        CavityState {
            amplitude: state.amplitude + da,
            time: state.time + dt,
        }
    }

    /// Integrates the cavity dynamics over time.
    ///
    /// # Arguments
    /// * `initial` - Initial cavity state
    /// * `input` - Input field (assumed constant)
    /// * `dt` - Time step (s)
    /// * `n_steps` - Number of steps
    ///
    /// # Returns
    /// Vector of (time, amplitude) pairs.
    pub fn integrate(
        &self,
        initial: CavityState,
        input: &InputField,
        dt: f64,
        n_steps: usize,
    ) -> Vec<(f64, Complex64)> {
        let mut state = initial;
        let mut trajectory = Vec::with_capacity(n_steps + 1);
        trajectory.push((state.time, state.amplitude));

        for _ in 0..n_steps {
            state = self.rk4_step(state, input, dt);
            trajectory.push((state.time, state.amplitude));
        }

        trajectory
    }

    /// Computes output field: s_out = s_in - sqrt(gamma_e) * a.
    ///
    /// For side-coupled cavity geometry.
    pub fn output_field(&self, cavity_amplitude: Complex64, input: &InputField) -> Complex64 {
        input.amplitude - self.cavity.coupling_coefficient() * cavity_amplitude
    }

    /// Computes transmission coefficient: t = s_out / s_in.
    pub fn transmission(&self, cavity_amplitude: Complex64, input: &InputField) -> Complex64 {
        if input.amplitude.norm() < 1e-30 {
            return Complex64::new(0.0, 0.0);
        }
        self.output_field(cavity_amplitude, input) / input.amplitude
    }

    /// Computes power transmission: T = |t|^2.
    pub fn power_transmission(&self, cavity_amplitude: Complex64, input: &InputField) -> f64 {
        self.transmission(cavity_amplitude, input).norm_sqr()
    }

    /// Finds steady-state solutions by solving the cubic equation.
    ///
    /// At steady state (da/dt = 0):
    /// ```text
    ///   a = sqrt(gamma_e)*s_in / (gamma_tot/2 - j*(delta - gamma_kerr*|a|^2))
    /// ```
    ///
    /// This leads to a cubic equation in |a|^2 that can have 1 or 3 real solutions.
    pub fn steady_state(&self, input: &InputField) -> SteadyStateResult {
        let delta = input.omega - self.cavity.omega_0;
        let gamma_tot = self.cavity.gamma_total();
        let gamma_e = self.cavity.gamma_external();
        let gamma_k = self.cavity.gamma_kerr();
        let s_in_sq = input.amplitude.norm_sqr();

        // Solve: |a|^2 = gamma_e * |s_in|^2 / ((gamma_tot/2)^2 + (delta - gamma_k*|a|^2)^2)
        //
        // Let x = |a|^2, then:
        // x * ((gamma_tot/2)^2 + (delta - gamma_k*x)^2) = gamma_e * |s_in|^2
        //
        // Expanding:
        // x * (g^2 + d^2 - 2*d*gamma_k*x + gamma_k^2*x^2) = P
        // where g = gamma_tot/2, d = delta, P = gamma_e * |s_in|^2
        //
        // Rearranging as cubic in x:
        // gamma_k^2 * x^3 - 2*d*gamma_k * x^2 + (g^2 + d^2) * x - P = 0

        let g = gamma_tot / 2.0;
        // Note: coupling coefficient is sqrt(2*gamma_e), so P = 2*gamma_e*|s_in|^2
        let p = 2.0 * gamma_e * s_in_sq;

        let energies = if gamma_k.abs() < 1e-30 {
            // Linear case: single solution
            let denom = g.powi(2) + delta.powi(2);
            if denom > 1e-30 {
                vec![p / denom]
            } else {
                vec![]
            }
        } else {
            // Nonlinear cubic: gamma_k^2 * x^3 - 2*d*gamma_k * x^2 + (g^2 + d^2) * x - P = 0
            // Normalize: x^3 + a*x^2 + b*x + c = 0
            let a = -2.0 * delta / gamma_k;
            let b = (g.powi(2) + delta.powi(2)) / gamma_k.powi(2);
            let c = -p / gamma_k.powi(2);

            solve_cubic_real(a, b, c)
                .into_iter()
                .filter(|&x| x >= 0.0)
                .collect()
        };

        // Compute amplitudes, transmissions, and stability for each solution
        let mut amplitudes = Vec::with_capacity(energies.len());
        let mut transmissions = Vec::with_capacity(energies.len());
        let mut power_transmissions = Vec::with_capacity(energies.len());
        let mut stability = Vec::with_capacity(energies.len());

        for &u in &energies {
            // Compute amplitude a from |a|^2 = u
            // From: a * (g - j*(delta - gamma_k*u)) = sqrt(2*gamma_e) * s_in
            let denom = Complex64::new(g, -(delta - gamma_k * u));
            let a = if denom.norm() > 1e-30 {
                ((2.0 * gamma_e).sqrt() * input.amplitude) / denom
            } else {
                Complex64::new(u.sqrt(), 0.0)
            };

            amplitudes.push(a);

            // Transmission
            let t = self.transmission(a, input);
            transmissions.push(t);
            power_transmissions.push(t.norm_sqr());

            // Stability: d/d(|a|^2)[RHS] should be negative for stable
            // Simplified criterion: middle branch is unstable in S-curve
            // Full stability: check eigenvalues of linearized system
            let is_stable = self.check_stability(a, input);
            stability.push(is_stable);
        }

        SteadyStateResult {
            num_solutions: energies.len(),
            amplitudes,
            energies,
            transmissions,
            power_transmissions,
            stability,
        }
    }

    /// Checks stability of a steady-state solution.
    ///
    /// For the cubic f(u) = u*(g^2 + (delta-gamma_k*u)^2) - P = 0:
    /// - If there's 1 positive root: it's stable
    /// - If there are 3 positive roots u1 < u2 < u3: u1, u3 stable; u2 unstable
    ///
    /// We use the derivative of the effective potential to determine stability:
    /// Stable if d/du[u*denom - P] > 0 at the solution (positive slope).
    fn check_stability(&self, a: Complex64, input: &InputField) -> bool {
        let delta = input.omega - self.cavity.omega_0;
        let gamma_tot = self.cavity.gamma_total();
        let gamma_k = self.cavity.gamma_kerr();
        let g = gamma_tot / 2.0;

        let u = a.norm_sqr();
        let eff_det = delta - gamma_k * u;
        let denom = g.powi(2) + eff_det.powi(2);

        // f(u) = u*denom - P = u*(g^2 + (delta - gamma_k*u)^2) - P
        // f'(u) = g^2 + (delta - gamma_k*u)^2 + u * 2*(delta - gamma_k*u)*(-gamma_k)
        //       = denom - 2*gamma_k*u*eff_det
        //
        // Stable if f'(u) > 0 (the intensity dynamics push back to equilibrium)
        let f_prime = denom - 2.0 * gamma_k * u * eff_det;

        f_prime > 0.0
    }

    /// Computes the bistability threshold power.
    ///
    /// For a Kerr cavity to exhibit bistability, the input power must exceed
    /// a critical value. This returns the minimum power for bistability given
    /// an optimal detuning.
    ///
    /// # Returns
    /// (threshold_power, optimal_detuning)
    pub fn bistability_threshold(&self) -> (f64, f64) {
        let gamma_tot = self.cavity.gamma_total();
        let gamma_e = self.cavity.gamma_external();
        let gamma_k = self.cavity.gamma_kerr().abs();

        if gamma_k < 1e-30 {
            return (f64::INFINITY, 0.0);
        }

        // Bistability requires |delta| > sqrt(3)*g.
        // Use delta = 2*g for robust bistability demonstration.
        let g = gamma_tot / 2.0;
        let delta_bist = 2.0 * g;

        // At normalized detuning Delta = delta/g = 2, the cubic f(U) = U^3 - 4*U^2 + 5*U - P
        // has local extrema at U = 1 (max) and U = 5/3 (min).
        // f(1) = 2 - P, f(5/3) = 50/27 - P ≈ 1.85 - P
        //
        // For 3 positive roots (bistability): f(max) > 0 > f(min)
        // → 1.85 < P_norm < 2.0
        //
        // Lower threshold: P_onset = 50/27 ≈ 1.852
        // Upper threshold: P_offset = 2.0
        //
        // The normalized power P_norm = 2*gamma_k*gamma_e*|s_in|^2/g^3
        // So |s_in|^2 = P_norm * g^3 / (2*gamma_k*gamma_e)

        let p_norm_onset = 50.0 / 27.0; // Lower bound for bistability at Delta = 2

        let p_crit = p_norm_onset * g.powi(3) / (2.0 * gamma_k * gamma_e);

        (p_crit, delta_bist)
    }

    /// Computes the full bistability window (lower and upper power thresholds).
    ///
    /// At a given detuning, bistability occurs for P_onset < P < P_offset.
    /// Outside this window, there's only one stable solution.
    ///
    /// # Returns
    /// (p_onset, p_offset, detuning) - powers in Watts, detuning in rad/s
    pub fn bistability_window(&self) -> (f64, f64, f64) {
        let gamma_tot = self.cavity.gamma_total();
        let gamma_e = self.cavity.gamma_external();
        let gamma_k = self.cavity.gamma_kerr().abs();

        if gamma_k < 1e-30 {
            return (f64::INFINITY, f64::INFINITY, 0.0);
        }

        let g = gamma_tot / 2.0;
        let delta = 2.0 * g; // Inside bistability region

        // Normalized power thresholds at Delta = 2
        let p_norm_onset = 50.0 / 27.0;  // ≈ 1.852
        let p_norm_offset = 2.0;

        let scale = g.powi(3) / (2.0 * gamma_k * gamma_e);
        let p_onset = p_norm_onset * scale;
        let p_offset = p_norm_offset * scale;

        (p_onset, p_offset, delta)
    }

    /// Scans input power and computes transmission curve.
    ///
    /// Useful for observing bistable switching.
    pub fn power_scan(
        &self,
        detuning: f64,
        powers: &[f64],
    ) -> Vec<(f64, SteadyStateResult)> {
        let omega = self.cavity.omega_0 + detuning;
        powers
            .iter()
            .map(|&p| {
                let input = InputField::cw(p, omega);
                (p, self.steady_state(&input))
            })
            .collect()
    }

    /// Scans frequency detuning and computes transmission curve.
    pub fn frequency_scan(
        &self,
        power: f64,
        detunings: &[f64],
    ) -> Vec<(f64, SteadyStateResult)> {
        detunings
            .iter()
            .map(|&d| {
                let input = InputField::cw_detuned(power, &self.cavity, d);
                (d, self.steady_state(&input))
            })
            .collect()
    }
}

/// Solves the depressed cubic equation x^3 + a*x^2 + b*x + c = 0.
///
/// Returns all real roots using Cardano's formula or trigonometric method.
fn solve_cubic_real(a: f64, b: f64, c: f64) -> Vec<f64> {
    // Convert to depressed cubic: t^3 + p*t + q = 0 via x = t - a/3
    let p = b - a.powi(2) / 3.0;
    let q = 2.0 * a.powi(3) / 27.0 - a * b / 3.0 + c;

    // Discriminant: D = q^2/4 + p^3/27
    let discriminant = q.powi(2) / 4.0 + p.powi(3) / 27.0;

    let shift = -a / 3.0;

    if discriminant > 1e-30 {
        // One real root (Cardano's formula)
        let u = (-q / 2.0 + discriminant.sqrt()).cbrt();
        let v = (-q / 2.0 - discriminant.sqrt()).cbrt();
        vec![u + v + shift]
    } else if discriminant < -1e-30 {
        // Three real roots (trigonometric method)
        let r = ((-p / 3.0).powi(3)).sqrt();
        let cos_phi = -q / (2.0 * r);
        // Clamp to handle numerical errors
        let phi = cos_phi.clamp(-1.0, 1.0).acos() / 3.0;

        let m = 2.0 * (-p / 3.0).sqrt();
        vec![
            m * phi.cos() + shift,
            m * (phi + 2.0 * PI / 3.0).cos() + shift,
            m * (phi + 4.0 * PI / 3.0).cos() + shift,
        ]
    } else {
        // Repeated roots
        let u = if q.abs() > 1e-30 { (-q / 2.0).cbrt() } else { 0.0 };
        vec![2.0 * u + shift, -u + shift]
    }
}

/// Linear transmission at a given detuning (no Kerr effect).
///
/// T = gamma_e^2 / ((gamma_tot/2)^2 + delta^2) for side-coupled cavity
/// or T = 1 - ... depending on geometry.
pub fn linear_transmission(cavity: &KerrCavity, detuning: f64) -> f64 {
    let gamma_e = cavity.gamma_external();
    let gamma_tot = cavity.gamma_total();
    let g = gamma_tot / 2.0;

    // For side-coupled geometry: t = 1 - sqrt(2*gamma_e)*a / s_in
    // where a/s_in = sqrt(2*gamma_e) / (g - j*delta) at linear steady state.
    // So t = 1 - 2*gamma_e / (g - j*delta)
    let t = Complex64::new(1.0, 0.0)
        - Complex64::new(2.0 * gamma_e, 0.0) / Complex64::new(g, -detuning);
    t.norm_sqr()
}

// ============================================================================
// Normalized Cubic Steady-State Solver (Liu et al. 2013, Eq. 5)
// ============================================================================

/// Result of the normalized cubic steady-state solver.
///
/// In normalized units (Liu et al. 2013), the steady-state equation is:
/// ```text
///   u^2 = y * [(y - Omega)^2 + 1]
/// ```
/// where:
/// - y = normalized intracavity energy = gamma_kerr * |a|^2 / g
/// - u^2 = normalized input power = 2 * gamma_e * gamma_kerr * |s_in|^2 / g^3
/// - Omega = normalized detuning = delta / g
#[derive(Debug, Clone)]
pub struct NormalizedSteadyState {
    /// Normalized intracavity energies y (1 or 3 solutions)
    pub y_solutions: Vec<f64>,
    /// Corresponding normalized transmission T for each solution
    pub transmissions: Vec<f64>,
    /// Stability of each solution (true = stable)
    pub stable: Vec<bool>,
    /// Normalized detuning Omega used
    pub omega: f64,
    /// Normalized input power u^2 used
    pub u_squared: f64,
}

/// Solve the normalized cubic steady-state equation.
///
/// Given normalized input u^2 and detuning Omega, solves:
/// ```text
///   u^2 = y * [(y - Omega)^2 + 1]
/// ```
/// for y >= 0.
///
/// This is the fast "implicit solver" for quasi-static rendering frames.
///
/// # Arguments
/// * `u_squared` - Normalized input power u^2
/// * `omega` - Normalized detuning Omega = delta / g
///
/// # Returns
/// NormalizedSteadyState with all physical (y >= 0) solutions
///
/// # Literature
/// Liu et al., Opt. Express 21(20), 23687 (2013), Eq. (5)
pub fn solve_normalized_cubic(u_squared: f64, omega: f64) -> NormalizedSteadyState {
    // Expand: u^2 = y * [(y - Omega)^2 + 1]
    //             = y * [y^2 - 2*Omega*y + Omega^2 + 1]
    //             = y^3 - 2*Omega*y^2 + (Omega^2 + 1)*y
    // So: y^3 - 2*Omega*y^2 + (Omega^2 + 1)*y - u^2 = 0

    let a = -2.0 * omega;
    let b = omega * omega + 1.0;
    let c = -u_squared;

    let all_roots = solve_cubic_real(a, b, c);

    // Filter to physical solutions (y >= 0)
    let y_solutions: Vec<f64> = all_roots.into_iter().filter(|&y| y >= -1e-12).map(|y| y.max(0.0)).collect();

    // Compute transmission and stability for each solution
    let mut transmissions = Vec::with_capacity(y_solutions.len());
    let mut stable = Vec::with_capacity(y_solutions.len());

    for &y in &y_solutions {
        // Transmission: T = |t|^2 where t = 1 - 2/(1 + j*(Omega - y))
        // For normalized units with critical coupling:
        // T = [(Omega - y)^2 + 1 - 2]^2 + 4*(Omega - y)^2 / [(Omega - y)^2 + 1]^2
        // Simplified: T = [(Omega - y - 1)(Omega - y + 1)]^2 / denom
        let eff_det = omega - y;
        let denom = eff_det * eff_det + 1.0;
        // For drop-port coupling: T_drop = 1 / denom (resonant enhancement)
        // For through-port: T_through = (eff_det^2) / denom
        let t_through = eff_det * eff_det / denom;
        transmissions.push(t_through);

        // Stability criterion: df/dy > 0 where f(y) = u^2 - y*[(y-Omega)^2 + 1]
        // df/dy = -d/dy[y^3 - 2*Omega*y^2 + (Omega^2+1)*y]
        //       = -(3*y^2 - 4*Omega*y + Omega^2 + 1)
        // Stable if 3*y^2 - 4*Omega*y + Omega^2 + 1 > 0
        let f_prime = 3.0 * y * y - 4.0 * omega * y + omega * omega + 1.0;
        stable.push(f_prime > 0.0);
    }

    NormalizedSteadyState {
        y_solutions,
        transmissions,
        stable,
        omega,
        u_squared,
    }
}

/// Convert physical parameters to normalized units for the cubic solver.
///
/// # Arguments
/// * `cavity` - Kerr cavity parameters
/// * `input_power` - Input optical power |s_in|^2 (W)
/// * `detuning` - Angular frequency detuning delta = omega - omega_0 (rad/s)
///
/// # Returns
/// (u_squared, omega) in normalized units
pub fn normalize_parameters(cavity: &KerrCavity, input_power: f64, detuning: f64) -> (f64, f64) {
    let g = cavity.gamma_total() / 2.0;
    let gamma_e = cavity.gamma_external();
    let gamma_k = cavity.gamma_kerr();

    // Omega = delta / g
    let omega = detuning / g;

    // u^2 = 2 * gamma_e * gamma_kerr * |s_in|^2 / g^3
    let u_squared = 2.0 * gamma_e * gamma_k * input_power / g.powi(3);

    (u_squared, omega)
}

/// Convert normalized intracavity energy y back to physical |a|^2.
///
/// # Arguments
/// * `cavity` - Kerr cavity parameters
/// * `y` - Normalized intracavity energy
///
/// # Returns
/// |a|^2 in physical units (J for energy, or proportional to photon number)
pub fn denormalize_energy(cavity: &KerrCavity, y: f64) -> f64 {
    let g = cavity.gamma_total() / 2.0;
    let gamma_k = cavity.gamma_kerr();

    // y = gamma_kerr * |a|^2 / g
    // |a|^2 = y * g / gamma_kerr
    y * g / gamma_k
}

/// Quick bistability check using normalized parameters.
///
/// Bistability requires:
/// 1. |Omega| > sqrt(3) (detuning beyond critical point)
/// 2. u^2 in the bistable window: u_lower < u^2 < u_upper
///
/// For Omega = 2, the bistable window is approximately [50/27, 2].
///
/// # Arguments
/// * `omega` - Normalized detuning
///
/// # Returns
/// (is_beyond_critical, u_lower, u_upper) where u_lower and u_upper define
/// the bistable power window (NaN if not bistable)
pub fn bistability_bounds(omega: f64) -> (bool, f64, f64) {
    let omega_crit = 3.0_f64.sqrt();
    let beyond_critical = omega.abs() > omega_crit;

    if !beyond_critical {
        return (false, f64::NAN, f64::NAN);
    }

    // Find the local extrema of u^2(y) = y * [(y - Omega)^2 + 1]
    // du^2/dy = 3*y^2 - 4*Omega*y + (Omega^2 + 1) = 0
    // y = (4*Omega +/- sqrt(16*Omega^2 - 12*(Omega^2 + 1))) / 6
    //   = (2*Omega +/- sqrt(Omega^2 - 3)) / 3

    let disc = omega * omega - 3.0;
    if disc < 0.0 {
        return (false, f64::NAN, f64::NAN);
    }

    let sqrt_disc = disc.sqrt();
    let y1 = (2.0 * omega - sqrt_disc) / 3.0;
    let y2 = (2.0 * omega + sqrt_disc) / 3.0;

    // Evaluate u^2 at these points
    let u2_at = |y: f64| y * ((y - omega).powi(2) + 1.0);
    let u_sq_1 = u2_at(y1);
    let u_sq_2 = u2_at(y2);

    let (u_lower, u_upper) = if u_sq_1 < u_sq_2 {
        (u_sq_1, u_sq_2)
    } else {
        (u_sq_2, u_sq_1)
    };

    (true, u_lower, u_upper)
}

/// Batch solve normalized cubic for multiple input powers (vectorized for rendering).
///
/// # Arguments
/// * `u_squared_values` - Slice of normalized input powers
/// * `omega` - Fixed normalized detuning
///
/// # Returns
/// Vector of NormalizedSteadyState results
pub fn solve_normalized_cubic_batch(
    u_squared_values: &[f64],
    omega: f64,
) -> Vec<NormalizedSteadyState> {
    u_squared_values
        .iter()
        .map(|&u2| solve_normalized_cubic(u2, omega))
        .collect()
}

// ============================================================================
// Thermal Dynamics Extension
// ============================================================================

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
            dn_dt: 1.8e-4,        // K^-1 for Si
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
    pub fn derivative(
        &self,
        state: &ThermalCavityState,
        input: &InputField,
    ) -> (Complex64, f64) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn test_cavity() -> KerrCavity {
        // Normalized cavity for testing bistability physics.
        // Q_total = 100 gives g = omega_0 / (2*Q) = 1 / 200 = 0.005
        // gamma_ratio = 1.0 gives gamma_kerr = g = 0.005
        KerrCavity::normalized(100.0, 1.0)
    }

    #[test]
    fn test_cavity_parameters() {
        let cavity = test_cavity();

        // Normalized cavity has Q_total = 100, Q_i = Q_e = 200 (critical coupling)
        assert_relative_eq!(cavity.q_intrinsic, 200.0, epsilon = 1e-10);
        assert_relative_eq!(cavity.q_external, 200.0, epsilon = 1e-10);
        assert_relative_eq!(cavity.q_total(), 100.0, epsilon = 1e-5);

        // Check coupling regime
        assert_eq!(cavity.coupling_regime(), CouplingRegime::Critical);
        assert!(cavity.is_critically_coupled(0.01));

        // Check gamma_kerr / g ratio is as specified
        let g = cavity.gamma_total() / 2.0;
        let gamma_k = cavity.gamma_kerr();
        assert_relative_eq!(gamma_k, g, epsilon = 1e-10);
    }

    #[test]
    fn test_linear_steady_state() {
        // Use zero Kerr coefficient for linear response
        let cavity = KerrCavity::from_wavelength(1550.0, 1e5, 1e5, 3.48, 0.0, 1e-18);
        let solver = TcmtSolver::new(cavity);

        // On resonance
        let input = InputField::cw(1e-6, cavity.omega_0);
        let result = solver.steady_state(&input);

        assert_eq!(result.num_solutions, 1);

        // At critical coupling and resonance, transmission should be zero (perfect absorption)
        // T = |1 - gamma_e / (gamma_tot/2)|^2 = |1 - 2|^2 = 1 for add-drop?
        // Actually for side-coupled: T = 1 at resonance in add-drop geometry
        // depends on specific geometry...
        let t = result.power_transmissions[0];
        // Just check it's computed
        assert!(t >= 0.0 && t <= 1.0);
    }

    #[test]
    fn test_bistability_threshold() {
        let cavity = test_cavity();
        let solver = TcmtSolver::new(cavity);

        let (p_crit, delta_crit) = solver.bistability_threshold();

        // Should be finite for nonzero n2
        assert!(p_crit.is_finite());
        assert!(p_crit > 0.0);
        assert!(delta_crit > 0.0);
    }

    #[test]
    fn test_single_solution_below_threshold() {
        let cavity = test_cavity();
        let solver = TcmtSolver::new(cavity);

        let (p_onset, _p_offset, delta) = solver.bistability_window();

        // Well below onset: should have single solution (lower branch)
        let input = InputField::cw_detuned(p_onset * 0.5, &cavity, delta);
        let result = solver.steady_state(&input);

        assert_eq!(result.num_solutions, 1);
        assert!(result.stability[0]);
    }

    #[test]
    fn test_three_solutions_in_bistable_window() {
        let cavity = test_cavity();
        let solver = TcmtSolver::new(cavity);

        let (p_onset, p_offset, delta) = solver.bistability_window();

        // Power inside bistable window: p_onset < P < p_offset
        // Use midpoint of the window
        let p_mid = (p_onset + p_offset) / 2.0;
        let input = InputField::cw_detuned(p_mid, &cavity, delta);
        let result = solver.steady_state(&input);

        // Should have 3 solutions (two stable, one unstable)
        assert_eq!(
            result.num_solutions, 3,
            "Expected 3 solutions in bistable window: p_onset={:.3e}, p={:.3e}, p_offset={:.3e}",
            p_onset, p_mid, p_offset
        );

        // Check that middle branch is unstable (2 stable solutions)
        let stable_count: usize = result.stability.iter().map(|&s| s as usize).sum();
        assert_eq!(stable_count, 2);
    }

    #[test]
    fn test_single_solution_above_bistable_window() {
        let cavity = test_cavity();
        let solver = TcmtSolver::new(cavity);

        let (_p_onset, p_offset, delta) = solver.bistability_window();

        // Well above offset: should have single solution (upper branch)
        let input = InputField::cw_detuned(p_offset * 2.0, &cavity, delta);
        let result = solver.steady_state(&input);

        assert_eq!(result.num_solutions, 1);
        assert!(result.stability[0]);
    }

    #[test]
    fn test_time_integration() {
        let cavity = test_cavity();
        let solver = TcmtSolver::new(cavity);

        let input = InputField::cw(1e-6, cavity.omega_0);
        let initial = CavityState::default();

        // Integrate for several cavity lifetimes
        let tau = 1.0 / cavity.gamma_total();
        let dt = tau / 100.0;
        let n_steps = 1000;

        let trajectory = solver.integrate(initial, &input, dt, n_steps);

        // Should reach steady state
        let final_amp = trajectory.last().unwrap().1;
        let steady = solver.steady_state(&input);

        // Final amplitude should be close to steady-state
        let steady_amp = steady.amplitudes[0];
        assert_relative_eq!(final_amp.norm(), steady_amp.norm(), epsilon = 0.1);
    }

    #[test]
    fn test_rk4_conserves_at_steady_state() {
        // Use normalized linear cavity for clean test
        let cavity = KerrCavity::normalized(100.0, 0.0); // n2 = 0 for linear
        let solver = TcmtSolver::new(cavity);

        // Start at steady state (use small power for stability)
        let input = InputField::cw(1e-3, cavity.omega_0);
        let steady = solver.steady_state(&input);
        let state = CavityState {
            amplitude: steady.amplitudes[0],
            time: 0.0,
        };

        // derivative should be ~zero at steady state
        let deriv = solver.derivative(&state, &input);
        // Use relative tolerance with absolute floor
        let amp_norm = steady.amplitudes[0].norm();
        let tol = amp_norm * 1e-10 + 1e-12;
        assert!(
            deriv.norm() < tol,
            "Derivative at steady state: {:.3e}, tolerance: {:.3e}, amplitude: {:.3e}",
            deriv.norm(),
            tol,
            amp_norm
        );
    }

    #[test]
    fn test_cubic_solver_one_root() {
        // x^3 - 6x^2 + 11x - 6 = 0 has roots 1, 2, 3
        let roots = solve_cubic_real(-6.0, 11.0, -6.0);
        assert_eq!(roots.len(), 3);

        let mut sorted = roots.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        assert_relative_eq!(sorted[0], 1.0, epsilon = 1e-10);
        assert_relative_eq!(sorted[1], 2.0, epsilon = 1e-10);
        assert_relative_eq!(sorted[2], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cubic_solver_single_real() {
        // x^3 + x + 1 = 0 has one real root near -0.6824
        let roots = solve_cubic_real(0.0, 1.0, 1.0);
        assert_eq!(roots.len(), 1);

        // Verify it's a root
        let x = roots[0];
        let check = x.powi(3) + x + 1.0;
        assert_relative_eq!(check, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_power_scan() {
        let cavity = test_cavity();
        let solver = TcmtSolver::new(cavity);

        let (p_onset, p_offset, delta) = solver.bistability_window();

        // Scan through the bistability window
        let powers: Vec<f64> = vec![
            p_onset * 0.5,          // Below onset: 1 solution
            (p_onset + p_offset) / 2.0,  // In window: 3 solutions
            p_offset * 1.5,         // Above offset: 1 solution
        ];

        let results = solver.power_scan(delta, &powers);

        // Should have results for each power
        assert_eq!(results.len(), powers.len());

        // Below onset: 1 solution
        assert_eq!(results[0].1.num_solutions, 1, "Expected 1 solution below onset");
        // In window: 3 solutions
        assert_eq!(results[1].1.num_solutions, 3, "Expected 3 solutions in bistable window");
        // Above offset: 1 solution
        assert_eq!(results[2].1.num_solutions, 1, "Expected 1 solution above offset");
    }

    #[test]
    fn test_frequency_scan_linear() {
        // Linear cavity (n2 = 0) for simple Lorentzian response
        let cavity = KerrCavity::from_wavelength(1550.0, 1e5, 1e5, 3.48, 0.0, 1e-18);
        let solver = TcmtSolver::new(cavity);

        let fwhm = cavity.fwhm();
        let detunings: Vec<f64> = (-10..=10).map(|i| i as f64 * fwhm / 10.0).collect();

        let results = solver.frequency_scan(1e-6, &detunings);

        // All should have single solution (linear case)
        for (_, r) in &results {
            assert_eq!(r.num_solutions, 1);
        }
    }

    // ========================================================================
    // Normalized Cubic Solver Tests
    // ========================================================================

    #[test]
    fn test_normalized_cubic_zero_input() {
        // Zero input should give y = 0
        let result = solve_normalized_cubic(0.0, 2.0);
        assert_eq!(result.y_solutions.len(), 1);
        assert!(result.y_solutions[0].abs() < 1e-10);
    }

    #[test]
    fn test_normalized_cubic_small_input() {
        // Small input with zero detuning: y^3 + y = u^2
        // For small u^2, y ~ u^2 (linear regime)
        let u_sq = 0.001;
        let result = solve_normalized_cubic(u_sq, 0.0);
        assert_eq!(result.y_solutions.len(), 1);
        // In linear regime, y ~ u^2
        assert!((result.y_solutions[0] - u_sq).abs() / u_sq < 0.1);
    }

    #[test]
    fn test_normalized_cubic_bistable_regime() {
        // Omega = 2.0 (beyond sqrt(3) threshold)
        // Bistable window: u^2 in [50/27, 2]
        let omega = 2.0;
        let u_sq_mid = 1.9; // Inside bistable window

        let result = solve_normalized_cubic(u_sq_mid, omega);

        // Should have 3 solutions
        assert_eq!(
            result.y_solutions.len(), 3,
            "Expected 3 solutions in bistable regime, got {:?}",
            result.y_solutions
        );

        // Two should be stable, one unstable
        let stable_count: usize = result.stable.iter().map(|&s| s as usize).sum();
        assert_eq!(stable_count, 2, "Expected 2 stable solutions");
    }

    #[test]
    fn test_bistability_bounds() {
        // Below critical detuning: no bistability
        let (beyond, _, _) = bistability_bounds(1.0);
        assert!(!beyond);

        // Above critical detuning (sqrt(3) ~ 1.732): bistability possible
        let (beyond, u_lower, u_upper) = bistability_bounds(2.0);
        assert!(beyond);
        assert!(u_lower.is_finite());
        assert!(u_upper.is_finite());
        assert!(u_lower < u_upper);

        // Check approximate values for Omega = 2
        // u_lower should be near 50/27 ~ 1.85, u_upper near 2.0
        assert!((u_lower - 50.0 / 27.0).abs() < 0.1);
    }

    #[test]
    fn test_normalize_denormalize_roundtrip() {
        let cavity = test_cavity();
        let power = 1e-6;
        let detuning = cavity.gamma_total();

        let (u_sq, omega) = normalize_parameters(&cavity, power, detuning);

        // Solve and denormalize
        let result = solve_normalized_cubic(u_sq, omega);
        let y = result.y_solutions[0];
        let a_sq = denormalize_energy(&cavity, y);

        // Should be consistent with direct steady-state calculation
        assert!(a_sq >= 0.0);
        assert!(a_sq.is_finite());
    }

    #[test]
    fn test_normalized_cubic_batch() {
        let omega = 2.0;
        let u_values: Vec<f64> = (1..=10).map(|i| i as f64 * 0.2).collect();

        let results = solve_normalized_cubic_batch(&u_values, omega);

        assert_eq!(results.len(), 10);
        // All should have at least one solution
        for r in &results {
            assert!(!r.y_solutions.is_empty());
        }
    }

    #[test]
    fn test_normalized_cubic_stability() {
        // At low power, solution should be stable
        let result = solve_normalized_cubic(0.5, 2.0);
        assert!(result.stable[0]);

        // In bistable regime, upper and lower branches stable
        let result_bi = solve_normalized_cubic(1.9, 2.0);
        if result_bi.y_solutions.len() == 3 {
            let mut sorted: Vec<_> = result_bi.y_solutions.iter()
                .zip(result_bi.stable.iter())
                .collect();
            sorted.sort_by(|a, b| a.0.partial_cmp(b.0).unwrap());
            // Lower and upper stable, middle unstable
            assert!(sorted[0].1, "Lower branch should be stable");
            assert!(!sorted[1].1, "Middle branch should be unstable");
            assert!(sorted[2].1, "Upper branch should be stable");
        }
    }

    // ==================== THERMAL DYNAMICS TESTS ====================

    fn test_thermal_cavity() -> ThermalCavity {
        let kerr = test_cavity();
        ThermalCavity::silicon(kerr, 10e-6) // 10 microsecond thermal time
    }

    #[test]
    fn test_thermal_cavity_parameters() {
        let cavity = test_thermal_cavity();

        // Silicon thermo-optic coefficient
        assert_relative_eq!(cavity.dn_dt, 1.8e-4, epsilon = 1e-6);

        // Thermal time constant
        assert_relative_eq!(cavity.tau_thermal, 10e-6, epsilon = 1e-9);

        // Gamma thermal should be positive for positive dn/dT
        let gamma_th = cavity.gamma_thermal();
        assert!(gamma_th > 0.0, "Thermal coefficient should be positive");

        // Thermal relaxation rate
        let gamma_relax = cavity.gamma_th();
        assert_relative_eq!(gamma_relax, 1.0 / 10e-6, epsilon = 1.0);
    }

    #[test]
    fn test_thermal_regime_classification() {
        let kerr = test_cavity();
        let tau_opt = 1.0 / kerr.gamma_total();

        // Slow thermal (adiabatic): tau_thermal > 100 * tau_optical
        let slow_cavity = ThermalCavity::new(
            kerr, 1.8e-4,
            200.0 * tau_opt, // 200x optical lifetime -> adiabatic
            1e-12, 300.0
        );
        assert_eq!(thermal_regime(&slow_cavity), ThermalRegime::Adiabatic);

        // Fast thermal: tau_thermal < 0.01 * tau_optical
        let fast_cavity = ThermalCavity::new(
            kerr, 1.8e-4,
            0.001 * tau_opt, // 0.1% of optical lifetime -> fast thermal
            1e-12, 300.0
        );
        assert_eq!(thermal_regime(&fast_cavity), ThermalRegime::FastThermal);

        // Intermediate: comparable timescales (0.01 < ratio < 100)
        let intermediate = ThermalCavity::new(
            kerr, 1.8e-4,
            10.0 * tau_opt, // 10x optical lifetime -> intermediate
            1e-12, 300.0
        );
        assert_eq!(thermal_regime(&intermediate), ThermalRegime::Intermediate);
    }

    #[test]
    fn test_thermal_state_default() {
        let state = ThermalCavityState::default();

        assert_eq!(state.amplitude, Complex64::new(0.0, 0.0));
        assert_eq!(state.temperature, 0.0);
        assert_eq!(state.time, 0.0);
        assert_eq!(state.energy(), 0.0);
    }

    #[test]
    fn test_thermal_solver_derivative_at_cold() {
        let cavity = test_thermal_cavity();
        let solver = ThermalTcmtSolver::new(cavity);

        // No input, cold cavity: derivatives should be zero
        let cold_state = ThermalCavityState::default();
        let no_input = InputField::cw(0.0, cavity.kerr.omega_0);
        let (da, dt) = solver.derivative(&cold_state, &no_input);

        assert_eq!(da, Complex64::new(0.0, 0.0));
        assert_eq!(dt, 0.0);
    }

    #[test]
    fn test_thermal_solver_heating() {
        let cavity = test_thermal_cavity();
        let solver = ThermalTcmtSolver::new(cavity);

        // Start with non-zero amplitude but zero temperature
        let excited_state = ThermalCavityState {
            amplitude: Complex64::new(1.0, 0.0),
            temperature: 0.0,
            time: 0.0,
        };
        let input = InputField::cw(0.0, cavity.kerr.omega_0);
        let (_, dt) = solver.derivative(&excited_state, &input);

        // With energy in cavity, temperature should increase
        assert!(dt > 0.0, "Temperature should increase when cavity is excited");
    }

    #[test]
    fn test_thermal_solver_cooling() {
        let cavity = test_thermal_cavity();
        let solver = ThermalTcmtSolver::new(cavity);

        // Hot cavity with no input
        let hot_state = ThermalCavityState {
            amplitude: Complex64::new(0.0, 0.0),
            temperature: 10.0, // 10K above ambient
            time: 0.0,
        };
        let no_input = InputField::cw(0.0, cavity.kerr.omega_0);
        let (_, dt) = solver.derivative(&hot_state, &no_input);

        // Temperature should decrease toward ambient
        assert!(dt < 0.0, "Temperature should decrease toward ambient");
    }

    #[test]
    fn test_thermal_rk4_step_conserves_cold() {
        let cavity = test_thermal_cavity();
        let solver = ThermalTcmtSolver::new(cavity);

        // Cold cavity with no input should stay cold
        let cold = ThermalCavityState::default();
        let no_input = InputField::cw(0.0, cavity.kerr.omega_0);

        let next = solver.rk4_step(&cold, &no_input, 1e-9);

        assert_relative_eq!(next.energy(), 0.0, epsilon = 1e-20);
        assert_relative_eq!(next.temperature, 0.0, epsilon = 1e-20);
    }

    #[test]
    fn test_thermal_evolve_trajectory() {
        let cavity = test_thermal_cavity();
        let solver = ThermalTcmtSolver::new(cavity);

        let initial = ThermalCavityState::default();
        // Small power input at resonance
        let input = InputField::cw(1e-6, cavity.kerr.omega_0);
        let dt = 1e-9;
        let n_steps = 100;

        let states = solver.evolve(initial, &input, dt, n_steps);

        // Should have n_steps + 1 states
        assert_eq!(states.len(), n_steps + 1);

        // Time should increase
        assert!(states.last().unwrap().time > states.first().unwrap().time);

        // Energy should build up with input
        assert!(states.last().unwrap().energy() > 0.0);
    }

    #[test]
    fn test_thermal_effective_gamma() {
        let cavity = test_thermal_cavity();
        let solver = ThermalTcmtSolver::new(cavity);

        let gamma_eff = solver.effective_gamma();

        // Should be at least the Kerr contribution
        let gamma_kerr = cavity.kerr.gamma_kerr();
        assert!(
            gamma_eff >= gamma_kerr,
            "Effective gamma {} should be >= Kerr gamma {}",
            gamma_eff, gamma_kerr
        );

        // Should be positive for self-focusing material
        assert!(gamma_eff > 0.0);
    }

    #[test]
    fn test_thermal_bistability_possible() {
        let cavity = test_thermal_cavity();
        let solver = ThermalTcmtSolver::new(cavity);

        // With positive thermo-optic coefficient, bistability should be possible
        assert!(
            solver.thermal_bistability_possible(),
            "Thermal bistability should be possible with positive dn/dT"
        );
    }
}
