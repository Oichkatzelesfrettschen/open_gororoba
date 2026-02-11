use serde::{Deserialize, Serialize};

/// Thermodynamic state for white-dwarf matter.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ThermoState {
    pub density: f64,
    pub temperature: f64,
    pub pressure: f64,
    pub specific_internal_energy: f64,
    pub electron_fraction: f64,
}

/// 1D Euler primitive state.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HydroState1D {
    pub density: f64,
    pub velocity: f64,
    pub pressure: f64,
    pub specific_internal_energy: f64,
}

impl HydroState1D {
    #[must_use]
    pub fn total_specific_energy(self) -> f64 {
        self.specific_internal_energy + 0.5 * self.velocity * self.velocity
    }
}

/// Burning composition state for a reduced C/O -> Ni56 model.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct BurnState {
    pub carbon_mass_fraction: f64,
    pub oxygen_mass_fraction: f64,
    pub nickel56_mass_fraction: f64,
    pub specific_nuclear_energy: f64,
}

impl Default for BurnState {
    fn default() -> Self {
        Self {
            carbon_mass_fraction: 0.5,
            oxygen_mass_fraction: 0.5,
            nickel56_mass_fraction: 0.0,
            specific_nuclear_energy: 0.0,
        }
    }
}

impl BurnState {
    #[must_use]
    pub fn normalized(self) -> Self {
        let carbon = self.carbon_mass_fraction.max(0.0);
        let oxygen = self.oxygen_mass_fraction.max(0.0);
        let nickel = self.nickel56_mass_fraction.max(0.0);
        let sum = carbon + oxygen + nickel;
        if sum <= f64::EPSILON {
            return Self::default();
        }
        Self {
            carbon_mass_fraction: carbon / sum,
            oxygen_mass_fraction: oxygen / sum,
            nickel56_mass_fraction: nickel / sum,
            specific_nuclear_energy: self.specific_nuclear_energy,
        }
    }
}
