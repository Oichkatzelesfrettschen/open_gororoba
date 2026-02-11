use crate::error::SniaError;
use crate::types::ThermoState;

/// Composition hooks for EOS terms.
#[derive(Debug, Clone, Copy)]
pub struct EosComposition {
    pub electron_fraction: f64,
    pub mean_ion_weight: f64,
}

impl Default for EosComposition {
    fn default() -> Self {
        Self {
            electron_fraction: 0.5,
            mean_ion_weight: 14.0,
        }
    }
}

/// Extended EOS state with pressure and energy decomposition.
#[derive(Debug, Clone, Copy)]
pub struct EosState {
    pub thermo: ThermoState,
    pub pressure_degenerate: f64,
    pub pressure_ion: f64,
    pub pressure_radiation: f64,
    pub specific_energy_degenerate: f64,
    pub specific_energy_ion: f64,
    pub specific_energy_radiation: f64,
    pub sound_speed: f64,
    pub effective_gamma: f64,
}

#[derive(Debug, Clone, Copy)]
struct EosCacheEntry {
    density: f64,
    temperature: f64,
    composition: EosComposition,
    state: EosState,
}

/// Mutable cache wrapper for repeated EOS queries at the same state point.
#[derive(Debug, Clone, Copy)]
pub struct CachedEos {
    eos: WhiteDwarfEos,
    cache: Option<EosCacheEntry>,
}

impl CachedEos {
    #[must_use]
    pub fn new(eos: WhiteDwarfEos) -> Self {
        Self { eos, cache: None }
    }

    pub fn eos_from_rho_t(
        &mut self,
        density: f64,
        temperature: f64,
        composition: EosComposition,
    ) -> Result<EosState, SniaError> {
        if let Some(cached) = self.cache {
            let same_density = cached.density == density;
            let same_temperature = cached.temperature == temperature;
            let same_ye = cached.composition.electron_fraction == composition.electron_fraction;
            let same_mu = cached.composition.mean_ion_weight == composition.mean_ion_weight;
            if same_density && same_temperature && same_ye && same_mu {
                return Ok(cached.state);
            }
        }
        let state = self.eos.eos_from_rho_t(density, temperature, composition)?;
        self.cache = Some(EosCacheEntry {
            density,
            temperature,
            composition,
            state,
        });
        Ok(state)
    }
}

/// Reduced white-dwarf EOS with degenerate, ion, and radiation terms.
#[derive(Debug, Clone, Copy)]
pub struct WhiteDwarfEos {
    pub k_degenerate: f64,
    pub ion_gas_constant: f64,
    pub radiation_constant: f64,
    pub temperature_floor: f64,
    pub newton_tol: f64,
    pub newton_max_iters: usize,
}

impl Default for WhiteDwarfEos {
    fn default() -> Self {
        Self {
            k_degenerate: 1.00e13,
            ion_gas_constant: 8.314_462_618e7,
            radiation_constant: 7.5657e-15,
            temperature_floor: 1.0e6,
            newton_tol: 1.0e-10,
            newton_max_iters: 64,
        }
    }
}

impl WhiteDwarfEos {
    fn validate_inputs(
        self,
        density: f64,
        temperature: f64,
        composition: EosComposition,
    ) -> Result<(), SniaError> {
        if density <= 0.0 {
            return Err(SniaError::NonPositiveDensity(density));
        }
        if temperature <= 0.0 {
            return Err(SniaError::NonPositiveTemperature(temperature));
        }
        if composition.electron_fraction <= 0.0 {
            return Err(SniaError::NonPositiveTemperature(
                composition.electron_fraction,
            ));
        }
        if composition.mean_ion_weight <= 0.0 {
            return Err(SniaError::NonPositiveTemperature(
                composition.mean_ion_weight,
            ));
        }
        Ok(())
    }

    #[must_use]
    pub fn pressure_degenerate(self, density: f64, composition: EosComposition) -> f64 {
        let rho_e = (density * composition.electron_fraction).max(1.0e-30);
        self.k_degenerate * rho_e.powf(5.0 / 3.0)
    }

    #[must_use]
    pub fn pressure_ion(self, density: f64, temperature: f64, composition: EosComposition) -> f64 {
        density * self.ion_gas_constant * temperature / composition.mean_ion_weight
    }

    #[must_use]
    pub fn pressure_radiation(self, temperature: f64) -> f64 {
        (self.radiation_constant / 3.0) * temperature.powi(4)
    }

    #[must_use]
    pub fn specific_energy_degenerate(self, density: f64, composition: EosComposition) -> f64 {
        1.5 * self.pressure_degenerate(density, composition) / density
    }

    #[must_use]
    pub fn specific_energy_ion(self, temperature: f64, composition: EosComposition) -> f64 {
        1.5 * self.ion_gas_constant * temperature / composition.mean_ion_weight
    }

    #[must_use]
    pub fn specific_energy_radiation(self, density: f64, temperature: f64) -> f64 {
        self.radiation_constant * temperature.powi(4) / density
    }

    pub fn eos_from_rho_t(
        self,
        density: f64,
        temperature: f64,
        composition: EosComposition,
    ) -> Result<EosState, SniaError> {
        self.validate_inputs(density, temperature, composition)?;

        let p_deg = self.pressure_degenerate(density, composition);
        let p_ion = self.pressure_ion(density, temperature, composition);
        let p_rad = self.pressure_radiation(temperature);
        let pressure = p_deg + p_ion + p_rad;
        if pressure <= 0.0 {
            return Err(SniaError::NegativePressure(pressure));
        }

        let e_deg = self.specific_energy_degenerate(density, composition);
        let e_ion = self.specific_energy_ion(temperature, composition);
        let e_rad = self.specific_energy_radiation(density, temperature);
        let specific_internal_energy = e_deg + e_ion + e_rad;

        let effective_gamma = ((5.0 / 3.0) * p_deg + (5.0 / 3.0) * p_ion + (4.0 / 3.0) * p_rad)
            / pressure.max(1.0e-30);
        let sound_speed = (effective_gamma * pressure / density).max(0.0).sqrt();

        let thermo = ThermoState {
            density,
            temperature,
            pressure,
            specific_internal_energy,
            electron_fraction: composition.electron_fraction,
        };
        Ok(EosState {
            thermo,
            pressure_degenerate: p_deg,
            pressure_ion: p_ion,
            pressure_radiation: p_rad,
            specific_energy_degenerate: e_deg,
            specific_energy_ion: e_ion,
            specific_energy_radiation: e_rad,
            sound_speed,
            effective_gamma,
        })
    }

    pub fn temperature_from_rho_e(
        self,
        density: f64,
        specific_internal_energy: f64,
        composition: EosComposition,
    ) -> Result<f64, SniaError> {
        if density <= 0.0 {
            return Err(SniaError::NonPositiveDensity(density));
        }
        if specific_internal_energy <= 0.0 {
            return Err(SniaError::NonPositiveTemperature(specific_internal_energy));
        }
        if composition.mean_ion_weight <= 0.0 || composition.electron_fraction <= 0.0 {
            return Err(SniaError::NonPositiveTemperature(
                composition
                    .mean_ion_weight
                    .min(composition.electron_fraction),
            ));
        }

        let e_deg = self.specific_energy_degenerate(density, composition);
        let target = specific_internal_energy;
        if target <= e_deg {
            return Ok(self.temperature_floor);
        }

        let a = 1.5 * self.ion_gas_constant / composition.mean_ion_weight;
        let b = self.radiation_constant / density;

        let mut t = ((target - e_deg) / a).max(self.temperature_floor);
        for _ in 0..self.newton_max_iters {
            let f = e_deg + a * t + b * t.powi(4) - target;
            let df = a + 4.0 * b * t.powi(3);
            let dt = f / df.max(1.0e-30);
            let next_t = (t - dt).max(self.temperature_floor);
            let rel = ((next_t - t) / t.max(1.0)).abs();
            t = next_t;
            if rel < self.newton_tol {
                break;
            }
        }
        Ok(t)
    }

    pub fn eos_from_rho_e(
        self,
        density: f64,
        specific_internal_energy: f64,
        composition: EosComposition,
    ) -> Result<EosState, SniaError> {
        let temperature =
            self.temperature_from_rho_e(density, specific_internal_energy, composition)?;
        self.eos_from_rho_t(density, temperature, composition)
    }

    pub fn pressure(self, density: f64, temperature: f64) -> Result<f64, SniaError> {
        Ok(self
            .eos_from_rho_t(density, temperature, EosComposition::default())?
            .thermo
            .pressure)
    }

    pub fn specific_internal_energy(self, temperature: f64) -> Result<f64, SniaError> {
        Ok(self
            .eos_from_rho_t(1.0e7, temperature, EosComposition::default())?
            .thermo
            .specific_internal_energy)
    }

    pub fn temperature_from_internal_energy(
        self,
        specific_internal_energy: f64,
    ) -> Result<f64, SniaError> {
        self.temperature_from_rho_e(1.0e7, specific_internal_energy, EosComposition::default())
    }

    pub fn sound_speed(self, density: f64, temperature: f64) -> Result<f64, SniaError> {
        Ok(self
            .eos_from_rho_t(density, temperature, EosComposition::default())?
            .sound_speed)
    }

    pub fn effective_gamma(self, density: f64, temperature: f64) -> Result<f64, SniaError> {
        Ok(self
            .eos_from_rho_t(density, temperature, EosComposition::default())?
            .effective_gamma)
    }

    pub fn make_state(
        self,
        density: f64,
        temperature: f64,
        electron_fraction: f64,
    ) -> Result<ThermoState, SniaError> {
        let state = self.eos_from_rho_t(
            density,
            temperature,
            EosComposition {
                electron_fraction,
                ..EosComposition::default()
            },
        )?;
        Ok(state.thermo)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn eos_gives_positive_pressure() {
        let eos = WhiteDwarfEos::default();
        let p = eos.pressure(2.5e7, 8.0e8).expect("pressure");
        assert!(p > 0.0);
    }

    #[test]
    fn rho_e_round_trip_is_stable() {
        let eos = WhiteDwarfEos::default();
        let comp = EosComposition::default();
        let state = eos.eos_from_rho_t(2.5e7, 1.1e9, comp).expect("state rho_t");
        let recovered = eos
            .eos_from_rho_e(2.5e7, state.thermo.specific_internal_energy, comp)
            .expect("state rho_e");
        let rel = ((recovered.thermo.temperature - state.thermo.temperature)
            / state.thermo.temperature)
            .abs();
        assert!(rel < 5.0e-8);
    }

    proptest! {
        #[test]
        fn pressure_is_monotone_in_density(
            rho in 1.0e6f64..1.0e8f64,
            factor in 1.0001f64..3.0f64,
            temp in 1.0e8f64..3.0e9f64
        ) {
            let eos = WhiteDwarfEos::default();
            let p1 = eos.pressure(rho, temp).expect("p1");
            let p2 = eos.pressure(rho * factor, temp).expect("p2");
            prop_assert!(p2 > p1);
        }

        #[test]
        fn pressure_is_monotone_in_temperature(
            rho in 1.0e6f64..1.0e8f64,
            t1 in 1.0e8f64..2.0e9f64,
            dt in 1.0f64..1.0e9f64
        ) {
            let eos = WhiteDwarfEos::default();
            let p1 = eos.pressure(rho, t1).expect("p1");
            let p2 = eos.pressure(rho, t1 + dt).expect("p2");
            prop_assert!(p2 > p1);
        }

        #[test]
        fn gamma_and_sound_speed_stay_physical(
            rho in 1.0e6f64..1.0e8f64,
            temp in 1.0e8f64..3.0e9f64
        ) {
            let eos = WhiteDwarfEos::default();
            let gamma = eos.effective_gamma(rho, temp).expect("gamma");
            let cs = eos.sound_speed(rho, temp).expect("cs");
            prop_assert!(gamma >= 1.0);
            prop_assert!(gamma <= 5.0/3.0 + 1.0e-9);
            prop_assert!(cs.is_finite());
            prop_assert!(cs > 0.0);
        }
    }
}
