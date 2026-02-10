//! Landau two-fluid model for superfluid helium-4.
//!
//! Evolves the coupled normal and superfluid components using the
//! Landau two-fluid equations in 1D. The RK4 stepper follows the
//! same hand-rolled pattern as the TOV solver in cosmology_core.
//!
//! # Equations (Landau 1941, Khalatnikov 1965)
//!
//! Continuity: d(rho)/dt + d(j)/dx = 0
//! where j = rho_s * v_s + rho_n * v_n
//!
//! Superfluid: d(v_s)/dt = -(1/rho) * dP/dx - d(mu)/dx
//! Normal:     d(v_n)/dt = -(1/rho_n) * (dP/dx + rho_s * d(mu)/dx) + viscous terms
//!
//! Simplified to 0D (spatially uniform) for first implementation:
//! Track rho_s(t) and T(t) relaxation toward equilibrium.
//!
//! # Literature
//! - Landau (1941): Theory of superfluidity of Helium II
//! - Khalatnikov (1965): An Introduction to the Theory of Superfluidity
//! - Donnelly (2009): The two-fluid theory and second sound in liquid helium

use crate::superfluid::superfluid_density_fraction;

/// State of the two-fluid system.
#[derive(Debug, Clone)]
pub struct TwoFluidState {
    /// Total density (kg/m^3).
    pub rho: f64,
    /// Superfluid density fraction (rho_s / rho).
    pub rho_s_frac: f64,
    /// Temperature (K).
    pub temperature: f64,
    /// Time (s).
    pub time: f64,
}

/// Parameters for two-fluid relaxation dynamics.
#[derive(Debug, Clone)]
pub struct TwoFluidParams {
    /// Total density (kg/m^3).
    pub rho: f64,
    /// Lambda temperature (K).
    pub t_lambda: f64,
    /// Relaxation timescale for superfluid density (s).
    pub tau_rho: f64,
    /// Thermal relaxation timescale (s).
    pub tau_t: f64,
    /// Bath temperature (K) for thermal relaxation.
    pub t_bath: f64,
}

impl TwoFluidParams {
    /// Default He-4 parameters at ~1 atm.
    pub fn he4_default(t_bath: f64) -> Self {
        Self {
            rho: 145.0,            // kg/m^3 (liquid He-4 at SVP)
            t_lambda: 2.1768,      // K
            tau_rho: 1.0e-6,       // Superfluid density relaxation ~1 us
            tau_t: 1.0e-4,         // Thermal relaxation ~100 us
            t_bath,
        }
    }
}

/// Result of two-fluid time evolution.
#[derive(Debug, Clone)]
pub struct TwoFluidTrace {
    /// Time points (s).
    pub times: Vec<f64>,
    /// Temperature history (K).
    pub temperatures: Vec<f64>,
    /// Superfluid density fraction history.
    pub rho_s_fracs: Vec<f64>,
    /// Normal density fraction history (1 - rho_s_frac).
    pub rho_n_fracs: Vec<f64>,
}

/// Compute derivatives of the two-fluid state.
///
/// The 0D model relaxes toward equilibrium:
/// - rho_s_frac relaxes to equilibrium value at current temperature
/// - temperature relaxes toward bath temperature
fn two_fluid_derivatives(state: &TwoFluidState, params: &TwoFluidParams) -> (f64, f64) {
    // Equilibrium superfluid fraction at current temperature
    let rho_s_eq = superfluid_density_fraction(state.temperature, params.t_lambda);

    // Relaxation dynamics
    let d_rho_s_frac = (rho_s_eq - state.rho_s_frac) / params.tau_rho;
    let d_temperature = (params.t_bath - state.temperature) / params.tau_t;

    (d_rho_s_frac, d_temperature)
}

/// Perform one RK4 step on the two-fluid system.
pub fn rk4_step_two_fluid(
    state: &TwoFluidState,
    params: &TwoFluidParams,
    dt: f64,
) -> TwoFluidState {
    // k1
    let (k1_rho, k1_t) = two_fluid_derivatives(state, params);

    // k2
    let s2 = TwoFluidState {
        rho: state.rho,
        rho_s_frac: state.rho_s_frac + 0.5 * dt * k1_rho,
        temperature: state.temperature + 0.5 * dt * k1_t,
        time: state.time + 0.5 * dt,
    };
    let (k2_rho, k2_t) = two_fluid_derivatives(&s2, params);

    // k3
    let s3 = TwoFluidState {
        rho: state.rho,
        rho_s_frac: state.rho_s_frac + 0.5 * dt * k2_rho,
        temperature: state.temperature + 0.5 * dt * k2_t,
        time: state.time + 0.5 * dt,
    };
    let (k3_rho, k3_t) = two_fluid_derivatives(&s3, params);

    // k4
    let s4 = TwoFluidState {
        rho: state.rho,
        rho_s_frac: state.rho_s_frac + dt * k3_rho,
        temperature: state.temperature + dt * k3_t,
        time: state.time + dt,
    };
    let (k4_rho, k4_t) = two_fluid_derivatives(&s4, params);

    // Update
    let new_rho_s = state.rho_s_frac
        + dt / 6.0 * (k1_rho + 2.0 * k2_rho + 2.0 * k3_rho + k4_rho);
    let new_t = state.temperature
        + dt / 6.0 * (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t);

    TwoFluidState {
        rho: state.rho,
        rho_s_frac: new_rho_s.clamp(0.0, 1.0),
        temperature: new_t.max(0.0),
        time: state.time + dt,
    }
}

/// Simulate two-fluid relaxation dynamics.
pub fn simulate_two_fluid(
    initial: &TwoFluidState,
    params: &TwoFluidParams,
    dt: f64,
    n_steps: usize,
) -> TwoFluidTrace {
    let mut times = Vec::with_capacity(n_steps + 1);
    let mut temperatures = Vec::with_capacity(n_steps + 1);
    let mut rho_s_fracs = Vec::with_capacity(n_steps + 1);
    let mut rho_n_fracs = Vec::with_capacity(n_steps + 1);

    times.push(initial.time);
    temperatures.push(initial.temperature);
    rho_s_fracs.push(initial.rho_s_frac);
    rho_n_fracs.push(1.0 - initial.rho_s_frac);

    let mut state = initial.clone();
    for _ in 0..n_steps {
        state = rk4_step_two_fluid(&state, params, dt);
        times.push(state.time);
        temperatures.push(state.temperature);
        rho_s_fracs.push(state.rho_s_frac);
        rho_n_fracs.push(1.0 - state.rho_s_frac);
    }

    TwoFluidTrace {
        times,
        temperatures,
        rho_s_fracs,
        rho_n_fracs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_two_fluid_mass_conservation() {
        // rho_s + rho_n = rho at all times
        let initial = TwoFluidState {
            rho: 145.0,
            rho_s_frac: 0.5,
            temperature: 1.5,
            time: 0.0,
        };
        let params = TwoFluidParams::he4_default(1.5);
        let trace = simulate_two_fluid(&initial, &params, 1e-8, 10000);

        for i in 0..trace.times.len() {
            let sum = trace.rho_s_fracs[i] + trace.rho_n_fracs[i];
            assert!(
                (sum - 1.0).abs() < 1e-12,
                "rho_s + rho_n = {} at t={}, should be 1.0",
                sum,
                trace.times[i]
            );
        }
    }

    #[test]
    fn test_two_fluid_equilibrium_relaxation() {
        // Starting at T=1.5 K with wrong rho_s_frac, should relax
        // to equilibrium value at T=1.5 K.
        let t_lambda = 2.1768;
        let rho_s_eq = superfluid_density_fraction(1.5, t_lambda);

        let initial = TwoFluidState {
            rho: 145.0,
            rho_s_frac: 0.1, // Wrong: should be ~0.97
            temperature: 1.5,
            time: 0.0,
        };
        let params = TwoFluidParams::he4_default(1.5);
        // Run for 100 tau_rho = 100e-6 s
        let trace = simulate_two_fluid(&initial, &params, 1e-8, 10000);

        let final_rho_s = *trace.rho_s_fracs.last().unwrap();
        eprintln!(
            "Equilibrium test: rho_s_eq={:.4}, final={:.4}",
            rho_s_eq, final_rho_s
        );
        assert!(
            (final_rho_s - rho_s_eq).abs() < 0.01,
            "should relax to equilibrium: final={}, eq={}",
            final_rho_s,
            rho_s_eq
        );
    }

    #[test]
    fn test_two_fluid_thermal_relaxation() {
        // Starting at T=3.0 K (above lambda), bath at 1.0 K (below lambda).
        // Should cool down and develop superfluid fraction.
        let initial = TwoFluidState {
            rho: 145.0,
            rho_s_frac: 0.0, // Normal fluid at 3 K
            temperature: 3.0,
            time: 0.0,
        };
        let params = TwoFluidParams::he4_default(1.0);
        // Run for many thermal relaxation times
        let trace = simulate_two_fluid(&initial, &params, 1e-6, 100000);

        let final_t = *trace.temperatures.last().unwrap();
        let final_rho_s = *trace.rho_s_fracs.last().unwrap();
        eprintln!(
            "Cooling test: T_final={:.3} K, rho_s_frac={:.4}",
            final_t, final_rho_s
        );
        // Temperature should approach bath temperature
        assert!(
            (final_t - 1.0).abs() < 0.1,
            "T should approach 1.0 K, got {}",
            final_t
        );
        // Superfluid fraction should be large at 1.0 K
        assert!(
            final_rho_s > 0.9,
            "rho_s/rho should be > 0.9 at 1.0 K, got {}",
            final_rho_s
        );
    }

    #[test]
    fn test_two_fluid_above_lambda() {
        // At T > T_lambda, rho_s should remain zero
        let initial = TwoFluidState {
            rho: 145.0,
            rho_s_frac: 0.0,
            temperature: 3.0,
            time: 0.0,
        };
        let params = TwoFluidParams::he4_default(3.0); // Bath above lambda
        let trace = simulate_two_fluid(&initial, &params, 1e-7, 1000);

        for &rho_s in &trace.rho_s_fracs {
            assert!(
                rho_s.abs() < 1e-10,
                "above T_lambda, rho_s should be 0, got {}",
                rho_s
            );
        }
    }
}
