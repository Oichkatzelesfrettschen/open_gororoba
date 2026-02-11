use serde::{Deserialize, Serialize};

/// Minimal alpha-chain state used by calibration workflows.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct AlphaChainState {
    pub he4: f64,
    pub c12: f64,
    pub o16: f64,
    pub si28: f64,
    pub ni56: f64,
}

impl Default for AlphaChainState {
    fn default() -> Self {
        Self {
            he4: 0.0,
            c12: 0.5,
            o16: 0.5,
            si28: 0.0,
            ni56: 0.0,
        }
    }
}

impl AlphaChainState {
    #[must_use]
    pub fn normalized(self) -> Self {
        let sum = self.he4 + self.c12 + self.o16 + self.si28 + self.ni56;
        if sum <= f64::EPSILON {
            return Self::default();
        }
        Self {
            he4: self.he4.max(0.0) / sum,
            c12: self.c12.max(0.0) / sum,
            o16: self.o16.max(0.0) / sum,
            si28: self.si28.max(0.0) / sum,
            ni56: self.ni56.max(0.0) / sum,
        }
    }
}

/// Reduced alpha-chain network kinetics.
#[derive(Debug, Clone, Copy)]
pub struct AlphaChainNetwork {
    pub rate_scale: f64,
}

impl Default for AlphaChainNetwork {
    fn default() -> Self {
        Self { rate_scale: 1.0 }
    }
}

impl AlphaChainNetwork {
    #[must_use]
    pub fn advance(self, state: AlphaChainState, dt: f64, temperature: f64) -> AlphaChainState {
        let temp_factor = (temperature / 1.0e9).max(0.0);
        let r_c_to_o = self.rate_scale * 0.8 * temp_factor * dt;
        let r_o_to_si = self.rate_scale * 0.5 * temp_factor * dt;
        let r_si_to_ni = self.rate_scale * 0.3 * temp_factor * dt;

        let c_to_o = (state.c12 * r_c_to_o).clamp(0.0, state.c12);
        let o_available = state.o16 + c_to_o;
        let o_to_si = (o_available * r_o_to_si).clamp(0.0, o_available);
        let si_available = state.si28 + o_to_si;
        let si_to_ni = (si_available * r_si_to_ni).clamp(0.0, si_available);

        AlphaChainState {
            he4: state.he4,
            c12: (state.c12 - c_to_o).max(0.0),
            o16: (o_available - o_to_si).max(0.0),
            si28: (si_available - si_to_ni).max(0.0),
            ni56: state.ni56 + si_to_ni,
        }
        .normalized()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_chain_conserves_mass_fraction() {
        let network = AlphaChainNetwork::default();
        let state = AlphaChainState::default();
        let next = network.advance(state, 1.0e-3, 2.0e9);
        let sum = next.he4 + next.c12 + next.o16 + next.si28 + next.ni56;
        assert!((sum - 1.0).abs() < 1.0e-12);
    }
}
