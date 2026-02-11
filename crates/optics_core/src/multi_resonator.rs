//! Multi-resonator TCMT system from box-kite clique structure.
//!
//! Extends the single-cavity TCMT solver to a system of N independent
//! resonator channels, where each channel corresponds to one box-kite
//! component from the sedenion zero-divisor graph.
//!
//! At dim=16, there are exactly 7 box-kite components (K6 cliques),
//! yielding 7 independent resonator channels. The channels are
//! uncoupled because the box-kite components are disconnected --
//! this is the key prediction of T3 (Holographic Entropy Trap).
//!
//! # Literature
//! - de Marrais (2000): Box-kite structure
//! - Suh et al., IEEE JQE 40, 1511 (2004): Multi-port TCMT

use crate::tcmt::{CavityState, InputField, KerrCavity, TcmtSolver};
use num_complex::Complex64;

/// A single resonator channel derived from a box-kite component.
#[derive(Debug, Clone)]
pub struct ResonatorChannel {
    /// Index of the box-kite component (0..6 for sedenions).
    pub component_index: usize,
    /// The Kerr cavity for this channel.
    pub cavity: KerrCavity,
    /// Number of assessors in this box-kite (6 for sedenions).
    pub n_assessors: usize,
}

/// Multi-resonator system with N independent channels.
#[derive(Debug, Clone)]
pub struct MultiResonatorSystem {
    /// The resonator channels (one per box-kite component).
    pub channels: Vec<ResonatorChannel>,
    /// Coupling matrix: coupling[i][j] = coupling strength between channels.
    /// For disconnected box-kites, this is block-diagonal (zero off-diagonal).
    pub coupling: Vec<Vec<f64>>,
}

/// State of the multi-resonator system at a given time.
#[derive(Debug, Clone)]
pub struct MultiResonatorState {
    /// Per-channel cavity states.
    pub channel_states: Vec<CavityState>,
    /// Current time.
    pub time: f64,
}

/// Result of multi-resonator time integration.
#[derive(Debug, Clone)]
pub struct MultiResonatorTrace {
    /// Time points.
    pub times: Vec<f64>,
    /// Per-channel amplitude trajectories: amplitudes[channel][time_step].
    pub amplitudes: Vec<Vec<Complex64>>,
    /// Per-channel energy trajectories: energies[channel][time_step].
    pub energies: Vec<Vec<f64>>,
}

impl MultiResonatorSystem {
    /// Build a 7-channel system from sedenion box-kite enumeration.
    ///
    /// Each box-kite component gets a KerrCavity with resonance frequency
    /// offset by the component index, simulating distinct spectral channels.
    ///
    /// # Arguments
    /// * `base_cavity` - Template cavity (all channels share Q, n2, etc.)
    /// * `frequency_spacing` - Frequency offset between channels (rad/s).
    pub fn from_boxkites(base_cavity: &KerrCavity, frequency_spacing: f64) -> Self {
        // Sedenion box-kites: 7 components, each with 6 assessors
        let n_components = 7;
        let n_assessors = 6;

        let mut channels = Vec::with_capacity(n_components);
        for i in 0..n_components {
            let omega = base_cavity.omega_0 + (i as f64) * frequency_spacing;
            let cavity = KerrCavity::new(
                omega,
                base_cavity.q_intrinsic,
                base_cavity.q_external,
                base_cavity.n_linear,
                base_cavity.n2,
                base_cavity.v_eff,
            );
            channels.push(ResonatorChannel {
                component_index: i,
                cavity,
                n_assessors,
            });
        }

        // Disconnected components => zero coupling (block-diagonal identity)
        let coupling = vec![vec![0.0; n_components]; n_components];

        Self { channels, coupling }
    }

    /// Build a system with custom channel parameters and coupling.
    pub fn new(channels: Vec<ResonatorChannel>, coupling: Vec<Vec<f64>>) -> Self {
        assert_eq!(coupling.len(), channels.len());
        for row in &coupling {
            assert_eq!(row.len(), channels.len());
        }
        Self { channels, coupling }
    }

    /// Number of channels.
    pub fn n_channels(&self) -> usize {
        self.channels.len()
    }

    /// Perform one RK4 step on the multi-resonator system.
    ///
    /// Each channel evolves independently (zero coupling) or with
    /// inter-channel coupling terms added to the derivative.
    pub fn rk4_step(
        &self,
        state: &MultiResonatorState,
        inputs: &[InputField],
        dt: f64,
    ) -> MultiResonatorState {
        let n = self.n_channels();
        assert_eq!(state.channel_states.len(), n);
        assert_eq!(inputs.len(), n);

        let solvers: Vec<TcmtSolver> = self
            .channels
            .iter()
            .map(|ch| TcmtSolver::new(ch.cavity))
            .collect();

        // k1
        let k1: Vec<Complex64> = (0..n)
            .map(|i| {
                let base = solvers[i].derivative(&state.channel_states[i], &inputs[i]);
                let coupling_term: Complex64 = (0..n)
                    .map(|j| {
                        if i != j {
                            Complex64::new(self.coupling[i][j], 0.0)
                                * state.channel_states[j].amplitude
                        } else {
                            Complex64::new(0.0, 0.0)
                        }
                    })
                    .sum();
                base + coupling_term
            })
            .collect();

        // k2
        let s2: Vec<CavityState> = (0..n)
            .map(|i| CavityState {
                amplitude: state.channel_states[i].amplitude + 0.5 * dt * k1[i],
                time: state.time + 0.5 * dt,
            })
            .collect();
        let k2: Vec<Complex64> = (0..n)
            .map(|i| {
                let base = solvers[i].derivative(&s2[i], &inputs[i]);
                let coupling_term: Complex64 = (0..n)
                    .map(|j| {
                        if i != j {
                            Complex64::new(self.coupling[i][j], 0.0) * s2[j].amplitude
                        } else {
                            Complex64::new(0.0, 0.0)
                        }
                    })
                    .sum();
                base + coupling_term
            })
            .collect();

        // k3
        let s3: Vec<CavityState> = (0..n)
            .map(|i| CavityState {
                amplitude: state.channel_states[i].amplitude + 0.5 * dt * k2[i],
                time: state.time + 0.5 * dt,
            })
            .collect();
        let k3: Vec<Complex64> = (0..n)
            .map(|i| {
                let base = solvers[i].derivative(&s3[i], &inputs[i]);
                let coupling_term: Complex64 = (0..n)
                    .map(|j| {
                        if i != j {
                            Complex64::new(self.coupling[i][j], 0.0) * s3[j].amplitude
                        } else {
                            Complex64::new(0.0, 0.0)
                        }
                    })
                    .sum();
                base + coupling_term
            })
            .collect();

        // k4
        let s4: Vec<CavityState> = (0..n)
            .map(|i| CavityState {
                amplitude: state.channel_states[i].amplitude + dt * k3[i],
                time: state.time + dt,
            })
            .collect();
        let k4: Vec<Complex64> = (0..n)
            .map(|i| {
                let base = solvers[i].derivative(&s4[i], &inputs[i]);
                let coupling_term: Complex64 = (0..n)
                    .map(|j| {
                        if i != j {
                            Complex64::new(self.coupling[i][j], 0.0) * s4[j].amplitude
                        } else {
                            Complex64::new(0.0, 0.0)
                        }
                    })
                    .sum();
                base + coupling_term
            })
            .collect();

        // Update
        let new_states: Vec<CavityState> = (0..n)
            .map(|i| {
                let da = (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) * (dt / 6.0);
                CavityState {
                    amplitude: state.channel_states[i].amplitude + da,
                    time: state.time + dt,
                }
            })
            .collect();

        MultiResonatorState {
            channel_states: new_states,
            time: state.time + dt,
        }
    }

    /// Integrate the multi-resonator system for n_steps.
    pub fn integrate(
        &self,
        initial: &MultiResonatorState,
        inputs: &[InputField],
        dt: f64,
        n_steps: usize,
    ) -> MultiResonatorTrace {
        let n = self.n_channels();
        let mut times = Vec::with_capacity(n_steps + 1);
        let mut amplitudes: Vec<Vec<Complex64>> =
            (0..n).map(|_| Vec::with_capacity(n_steps + 1)).collect();
        let mut energies: Vec<Vec<f64>> = (0..n).map(|_| Vec::with_capacity(n_steps + 1)).collect();

        // Record initial state
        times.push(initial.time);
        for i in 0..n {
            amplitudes[i].push(initial.channel_states[i].amplitude);
            energies[i].push(initial.channel_states[i].amplitude.norm_sqr());
        }

        let mut state = initial.clone();
        for _ in 0..n_steps {
            state = self.rk4_step(&state, inputs, dt);
            times.push(state.time);
            for i in 0..n {
                amplitudes[i].push(state.channel_states[i].amplitude);
                energies[i].push(state.channel_states[i].amplitude.norm_sqr());
            }
        }

        MultiResonatorTrace {
            times,
            amplitudes,
            energies,
        }
    }

    /// Create zero initial state (all channels at rest).
    pub fn zero_state(&self) -> MultiResonatorState {
        let n = self.n_channels();
        MultiResonatorState {
            channel_states: (0..n)
                .map(|_| CavityState {
                    amplitude: Complex64::new(0.0, 0.0),
                    time: 0.0,
                })
                .collect(),
            time: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_base_cavity() -> KerrCavity {
        KerrCavity::new(
            1.0e15,  // omega_0 ~ 1 PHz (optical)
            1.0e6,   // Q_intrinsic
            5.0e5,   // Q_external
            1.5,     // n_linear (glass-like)
            2.0e-18, // n2 (silica-like, m^2/W)
            1.0e-18, // v_eff (1 um^3 mode volume)
        )
    }

    #[test]
    fn test_from_boxkites_7_channels() {
        let base = test_base_cavity();
        let system = MultiResonatorSystem::from_boxkites(&base, 1.0e12);
        assert_eq!(
            system.n_channels(),
            7,
            "sedenion box-kites produce 7 channels"
        );
        for (i, ch) in system.channels.iter().enumerate() {
            assert_eq!(ch.component_index, i);
            assert_eq!(ch.n_assessors, 6);
        }
    }

    #[test]
    fn test_uncoupled_channels_zero_coupling() {
        let base = test_base_cavity();
        let system = MultiResonatorSystem::from_boxkites(&base, 1.0e12);
        for i in 0..7 {
            for j in 0..7 {
                assert_eq!(
                    system.coupling[i][j], 0.0,
                    "disconnected box-kites => zero coupling"
                );
            }
        }
    }

    #[test]
    fn test_single_channel_matches_tcmt() {
        // One channel should exactly match single-cavity TCMT
        let cavity = test_base_cavity();
        let channel = ResonatorChannel {
            component_index: 0,
            cavity: cavity.clone(),
            n_assessors: 6,
        };
        let system = MultiResonatorSystem::new(vec![channel], vec![vec![0.0]]);

        let input = InputField {
            amplitude: Complex64::new(1.0e-3, 0.0),
            omega: cavity.omega_0,
        };
        let dt = 1.0e-12;
        let n_steps = 1000;

        // Multi-resonator integration
        let initial = system.zero_state();
        let trace = system.integrate(&initial, &[input.clone()], dt, n_steps);

        // Single-cavity integration
        let solver = TcmtSolver::new(cavity);
        let single_state = CavityState {
            amplitude: Complex64::new(0.0, 0.0),
            time: 0.0,
        };
        let single_trace = solver.integrate(single_state, &input, dt, n_steps);

        // Compare final amplitudes
        let multi_final = trace.amplitudes[0].last().unwrap();
        let single_final = single_trace.last().unwrap().1;
        let diff = (multi_final - single_final).norm();
        assert!(
            diff < 1e-15,
            "single-channel multi-resonator should match TCMT exactly, diff={}",
            diff
        );
    }

    #[test]
    fn test_rk4_energy_bounded() {
        let base = test_base_cavity();
        let system = MultiResonatorSystem::from_boxkites(&base, 1.0e12);

        let inputs: Vec<InputField> = (0..7)
            .map(|i| InputField {
                amplitude: Complex64::new(1.0e-3, 0.0),
                omega: base.omega_0 + (i as f64) * 1.0e12,
            })
            .collect();

        let initial = system.zero_state();
        let trace = system.integrate(&initial, &inputs, 1.0e-12, 500);

        // Energy should be bounded (not blow up)
        for ch in 0..7 {
            let max_energy = trace.energies[ch].iter().cloned().fold(0.0f64, f64::max);
            assert!(
                max_energy < 1e10,
                "channel {} energy should be bounded, got {}",
                ch,
                max_energy
            );
        }
    }

    #[test]
    fn test_7clique_no_crosstalk() {
        // With zero coupling, each channel should be independent
        let base = test_base_cavity();
        let system = MultiResonatorSystem::from_boxkites(&base, 1.0e12);

        // Drive only channel 0, leave others with zero input
        let mut inputs: Vec<InputField> = (0..7)
            .map(|i| InputField {
                amplitude: Complex64::new(0.0, 0.0),
                omega: base.omega_0 + (i as f64) * 1.0e12,
            })
            .collect();
        inputs[0].amplitude = Complex64::new(1.0e-3, 0.0);

        let initial = system.zero_state();
        let trace = system.integrate(&initial, &inputs, 1.0e-12, 1000);

        // Channel 0 should have nonzero energy
        let ch0_final_energy = *trace.energies[0].last().unwrap();
        assert!(ch0_final_energy > 0.0, "driven channel should have energy");

        // All other channels should remain at zero
        for ch in 1..7 {
            let energy = *trace.energies[ch].last().unwrap();
            assert!(
                energy < 1e-30,
                "undriven uncoupled channel {} should have zero energy, got {}",
                ch,
                energy
            );
        }
    }
}
