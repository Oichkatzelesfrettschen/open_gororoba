use crate::error::SniaError;
use crate::types::{BurnState, ThermoState};

/// Reduced C12 burning model with screening and subcycling.
#[derive(Debug, Clone, Copy)]
pub struct CarbonBurnModel {
    /// Arrhenius-like prefactor in 1/s for reduced model fallback.
    pub prefactor: f64,
    /// Effective activation temperature in K for reduced model fallback.
    pub activation_temperature: f64,
    /// Specific nuclear energy release in erg/g for complete fuel burn.
    pub q_value: f64,
    /// Minimal ash level left to prevent hard zero-locking.
    pub ash_floor: f64,
    /// Multiplicative screening strength.
    pub screening_strength: f64,
    /// Maximum allowed substep for burn integration.
    pub max_substep_s: f64,
}

impl Default for CarbonBurnModel {
    fn default() -> Self {
        Self {
            prefactor: 2.5e6,
            activation_temperature: 1.8e9,
            q_value: 6.0e17,
            ash_floor: 1.0e-8,
            screening_strength: 0.15,
            max_substep_s: 2.5e-5,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BurnStepResult {
    pub updated_state: BurnState,
    pub specific_energy_added: f64,
    pub burn_fraction: f64,
    pub substeps: usize,
}

impl CarbonBurnModel {
    /// Approximate electrostatic screening enhancement.
    #[must_use]
    pub fn screening_factor(self, thermo: ThermoState) -> f64 {
        let rho7 = (thermo.density / 1.0e7).max(1.0e-12);
        let t9 = (thermo.temperature / 1.0e9).max(1.0e-12);
        (self.screening_strength * rho7.powf(1.0 / 3.0) / t9).exp()
    }

    /// Reduced C12 + C12 rate law.
    pub fn c12_c12_rate(self, thermo: ThermoState) -> Result<f64, SniaError> {
        if thermo.density <= 0.0 {
            return Err(SniaError::NonPositiveDensity(thermo.density));
        }
        if thermo.temperature <= 0.0 {
            return Err(SniaError::NonPositiveTemperature(thermo.temperature));
        }

        let t9 = (thermo.temperature / 1.0e9).max(1.0e-8);
        let rate_base = 4.27e26 * t9.powf(-2.0 / 3.0) * (-84.165 / t9.powf(1.0 / 3.0)).exp();
        Ok(rate_base * self.screening_factor(thermo))
    }

    /// Full reduced reaction rate including composition and fallback Arrhenius lane.
    pub fn reaction_rate_c12(
        self,
        thermo: ThermoState,
        burn_state: BurnState,
    ) -> Result<f64, SniaError> {
        let x_c = burn_state.carbon_mass_fraction.clamp(0.0, 1.0);
        let rho = thermo.density.max(1.0e-30);
        let c12 = self.c12_c12_rate(thermo)?;
        let rate_nuclear = rho * x_c * x_c * c12;

        // Keep reduced fallback lane to avoid under-burning in low-temperature regimes.
        let arrhenius = self.prefactor
            * (rho / 1.0e7).powf(1.3)
            * (-self.activation_temperature / thermo.temperature).exp()
            * x_c;
        Ok(rate_nuclear + arrhenius)
    }

    /// Single burn substep update.
    pub fn burn_substep(
        self,
        burn_state: BurnState,
        dt: f64,
        thermo: ThermoState,
    ) -> Result<BurnStepResult, SniaError> {
        if dt <= 0.0 {
            return Err(SniaError::InvalidTimeStep(dt));
        }
        let rate = self.reaction_rate_c12(thermo, burn_state)?;
        let burn_fraction = (rate * dt).clamp(0.0, 1.0);

        let burned_carbon = burn_state.carbon_mass_fraction * burn_fraction;
        let new_carbon = (burn_state.carbon_mass_fraction - burned_carbon).max(0.0);
        let new_nickel =
            (burn_state.nickel56_mass_fraction + burned_carbon).min(1.0 - self.ash_floor);
        let new_oxygen = burn_state.oxygen_mass_fraction;

        let specific_energy_added = burned_carbon * self.q_value;
        let updated_state = BurnState {
            carbon_mass_fraction: new_carbon,
            oxygen_mass_fraction: new_oxygen,
            nickel56_mass_fraction: new_nickel,
            specific_nuclear_energy: burn_state.specific_nuclear_energy + specific_energy_added,
        }
        .normalized();

        Ok(BurnStepResult {
            updated_state,
            specific_energy_added,
            burn_fraction,
            substeps: 1,
        })
    }

    /// Subcycled burn update over a macro step `dt`.
    pub fn burn_step_subcycled(
        self,
        burn_state: BurnState,
        dt: f64,
        thermo: ThermoState,
    ) -> Result<BurnStepResult, SniaError> {
        if dt <= 0.0 {
            return Err(SniaError::InvalidTimeStep(dt));
        }
        let n_sub = ((dt / self.max_substep_s).ceil() as usize).max(1);
        let dt_sub = dt / n_sub as f64;

        let mut state = burn_state;
        let mut e_total = 0.0;
        let mut burn_total = 0.0;
        for _ in 0..n_sub {
            let out = self.burn_substep(state, dt_sub, thermo)?;
            state = out.updated_state;
            e_total += out.specific_energy_added;
            burn_total += out.burn_fraction;
        }

        Ok(BurnStepResult {
            updated_state: state,
            specific_energy_added: e_total,
            burn_fraction: burn_total.clamp(0.0, 1.0),
            substeps: n_sub,
        })
    }

    /// Chapman-Jouguet detonation speed estimate (cm/s).
    pub fn chapman_jouguet_velocity(
        self,
        thermo: ThermoState,
        burn_state: BurnState,
    ) -> Result<f64, SniaError> {
        if thermo.density <= 0.0 {
            return Err(SniaError::NonPositiveDensity(thermo.density));
        }
        let fuel_fraction =
            (burn_state.carbon_mass_fraction + burn_state.oxygen_mass_fraction).clamp(0.0, 1.0);
        let gamma = 1.4;
        let q = self.q_value * fuel_fraction;
        let cj = (2.0 * (gamma * gamma - 1.0) * q).max(0.0).sqrt();
        Ok(cj)
    }

    pub fn advance(
        self,
        burn_state: BurnState,
        dt: f64,
        thermo: ThermoState,
    ) -> Result<BurnStepResult, SniaError> {
        self.burn_step_subcycled(burn_state, dt, thermo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn burning_reduces_carbon_and_adds_energy() {
        let model = CarbonBurnModel::default();
        let thermo = ThermoState {
            density: 2.5e7,
            temperature: 2.2e9,
            pressure: 1.0,
            specific_internal_energy: 1.0,
            electron_fraction: 0.5,
        };
        let burn = BurnState::default();
        let out = model.advance(burn, 1.0e-4, thermo).expect("advance");
        assert!(out.updated_state.carbon_mass_fraction <= burn.carbon_mass_fraction);
        assert!(out.specific_energy_added >= 0.0);
        assert!(out.substeps >= 1);
    }

    #[test]
    fn screening_factor_is_positive() {
        let model = CarbonBurnModel::default();
        let thermo = ThermoState {
            density: 1.0e8,
            temperature: 8.0e8,
            pressure: 1.0,
            specific_internal_energy: 1.0,
            electron_fraction: 0.5,
        };
        assert!(model.screening_factor(thermo) > 0.0);
    }

    #[test]
    fn cj_velocity_is_finite() {
        let model = CarbonBurnModel::default();
        let thermo = ThermoState {
            density: 2.0e7,
            temperature: 2.0e9,
            pressure: 1.0,
            specific_internal_energy: 1.0,
            electron_fraction: 0.5,
        };
        let burn = BurnState::default();
        let cj = model.chapman_jouguet_velocity(thermo, burn).expect("cj");
        assert!(cj.is_finite());
        assert!(cj > 0.0);
    }
}
