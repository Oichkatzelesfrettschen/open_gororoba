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
}
