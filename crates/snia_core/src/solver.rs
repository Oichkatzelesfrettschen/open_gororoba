use crate::eos::{EosComposition, WhiteDwarfEos};
use crate::error::SniaError;
use crate::hydro::{BoundaryCondition, HllcFlux1D, LimiterKind};
use crate::reaction::CarbonBurnModel;
use crate::types::{BurnState, HydroState1D};
use crate::yield_model::NickelYieldModel;
use serde::{Deserialize, Serialize};

const SOLAR_MASS_G: f64 = 1.988_47e33;

/// Coupled solver configuration for reduced Type Ia DDT dynamics.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SolverConfig {
    pub n_cells: usize,
    pub dx_cm: f64,
    pub cfl: f64,
    pub final_time_s: f64,
    pub max_steps: usize,
    pub cell_volume_cm3: f64,
    pub detonation_burn_fraction_threshold: f64,
    pub detonation_cj_fraction_threshold: f64,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            n_cells: 128,
            dx_cm: 1.0e5,
            cfl: 0.35,
            final_time_s: 0.2,
            max_steps: 2_000,
            cell_volume_cm3: 1.0e23,
            detonation_burn_fraction_threshold: 0.05,
            detonation_cj_fraction_threshold: 0.8,
        }
    }
}

/// Per-cell coupled hydro + composition state.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct CellState {
    pub hydro: HydroState1D,
    pub burn: BurnState,
    pub temperature: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetonationEvent {
    pub time_s: f64,
    pub cell_index: usize,
    pub burn_fraction: f64,
    pub cj_velocity_cm_s: f64,
    pub flow_velocity_cm_s: f64,
    pub density: f64,
    pub temperature: f64,
}

/// Final solver summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationResult {
    pub final_time_s: f64,
    pub n_steps: usize,
    pub burned_mass_msun: f64,
    pub nickel56_mass_msun: f64,
    pub peak_density: f64,
    pub cells: Vec<CellState>,
    pub detonation_events: Vec<DetonationEvent>,
}

/// Main DDT core solver scaffold.
pub struct SniaCoreSolver {
    pub config: SolverConfig,
    pub eos: WhiteDwarfEos,
    pub hydro_solver: HllcFlux1D,
    pub burner: CarbonBurnModel,
    pub yield_model: NickelYieldModel,
    pub cells: Vec<CellState>,
    pub time_s: f64,
    pub n_steps: usize,
    pub detonation_events: Vec<DetonationEvent>,
}

impl SniaCoreSolver {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: SolverConfig,
        eos: WhiteDwarfEos,
        hydro_solver: HllcFlux1D,
        burner: CarbonBurnModel,
        yield_model: NickelYieldModel,
        initial_hydro: HydroState1D,
        initial_temperature: f64,
        initial_burn: BurnState,
    ) -> Result<Self, SniaError> {
        if config.n_cells == 0 {
            return Err(SniaError::EmptyGrid);
        }
        if config.dx_cm <= 0.0
            || config.final_time_s <= 0.0
            || config.cfl <= 0.0
            || config.cell_volume_cm3 <= 0.0
        {
            return Err(SniaError::InvalidTimeStep(config.final_time_s));
        }
        if initial_hydro.density <= 0.0 {
            return Err(SniaError::NonPositiveDensity(initial_hydro.density));
        }
        if initial_temperature <= 0.0 {
            return Err(SniaError::NonPositiveTemperature(initial_temperature));
        }

        let cells = vec![
            CellState {
                hydro: initial_hydro,
                burn: initial_burn.normalized(),
                temperature: initial_temperature,
            };
            config.n_cells
        ];

        Ok(Self {
            config,
            eos,
            hydro_solver,
            burner,
            yield_model,
            cells,
            time_s: 0.0,
            n_steps: 0,
            detonation_events: Vec::new(),
        })
    }

    pub fn suggest_dt(&self) -> Result<f64, SniaError> {
        if self.cells.is_empty() {
            return Err(SniaError::EmptyGrid);
        }
        let mut max_signal = 0.0_f64;
        for cell in &self.cells {
            let cs = self.eos.sound_speed(cell.hydro.density, cell.temperature)?;
            max_signal = max_signal.max(cell.hydro.velocity.abs() + cs);
        }
        let dt = self.config.cfl * self.config.dx_cm / max_signal.max(1.0e-12);
        if !dt.is_finite() || dt <= 0.0 {
            return Err(SniaError::InvalidTimeStep(dt));
        }
        Ok(dt.min(self.config.final_time_s - self.time_s))
    }

    fn update_hydro(&mut self, dt: f64) -> Result<(), SniaError> {
        let hydro_cells: Vec<HydroState1D> = self.cells.iter().map(|c| c.hydro).collect();
        let updated_hydro = self.hydro_solver.rk2_update(
            &hydro_cells,
            self.config.dx_cm,
            dt,
            LimiterKind::MonotonizedCentral,
            BoundaryCondition::Outflow,
        )?;

        for (cell, hydro) in self.cells.iter_mut().zip(updated_hydro.iter()) {
            cell.hydro = *hydro;
            cell.temperature = self.eos.temperature_from_rho_e(
                cell.hydro.density,
                cell.hydro.specific_internal_energy,
                EosComposition::default(),
            )?;
            cell.hydro.pressure = self.eos.pressure(cell.hydro.density, cell.temperature)?;
        }
        Ok(())
    }

    fn couple_burning(&mut self, dt: f64) -> Result<(), SniaError> {
        let mut new_events: Vec<DetonationEvent> = Vec::new();
        for (idx, cell) in self.cells.iter_mut().enumerate() {
            let initial_burn = cell.burn;
            let thermo = self
                .eos
                .make_state(cell.hydro.density, cell.temperature, 0.5)?;
            let burn_step = self.burner.advance(cell.burn, dt, thermo)?;
            let cj_velocity = self.burner.chapman_jouguet_velocity(thermo, initial_burn)?;
            cell.burn = burn_step.updated_state;

            cell.hydro.specific_internal_energy += burn_step.specific_energy_added;
            cell.temperature = self.eos.temperature_from_rho_e(
                cell.hydro.density,
                cell.hydro.specific_internal_energy,
                EosComposition {
                    electron_fraction: thermo.electron_fraction,
                    ..EosComposition::default()
                },
            )?;
            cell.hydro.pressure = self.eos.pressure(cell.hydro.density, cell.temperature)?;

            let flow_speed = cell.hydro.velocity.abs();
            if burn_step.burn_fraction >= self.config.detonation_burn_fraction_threshold
                && flow_speed >= self.config.detonation_cj_fraction_threshold * cj_velocity
            {
                new_events.push(DetonationEvent {
                    time_s: self.time_s,
                    cell_index: idx,
                    burn_fraction: burn_step.burn_fraction,
                    cj_velocity_cm_s: cj_velocity,
                    flow_velocity_cm_s: flow_speed,
                    density: cell.hydro.density,
                    temperature: cell.temperature,
                });
            }
        }
        self.detonation_events.extend(new_events);
        Ok(())
    }

    pub fn step(&mut self) -> Result<f64, SniaError> {
        if self.time_s >= self.config.final_time_s {
            return Ok(0.0);
        }

        let dt = self.suggest_dt()?;
        if dt <= 0.0 {
            return Ok(0.0);
        }

        self.couple_burning(0.5 * dt)?;
        self.update_hydro(dt)?;
        self.couple_burning(0.5 * dt)?;

        self.time_s += dt;
        self.n_steps += 1;
        Ok(dt)
    }

    #[must_use]
    pub fn integrated_burned_mass_msun(&self) -> f64 {
        let burned_mass_g: f64 = self
            .cells
            .iter()
            .map(|c| {
                let burned_fraction = c.burn.nickel56_mass_fraction.clamp(0.0, 1.0);
                c.hydro.density * self.config.cell_volume_cm3 * burned_fraction
            })
            .sum();
        burned_mass_g / SOLAR_MASS_G
    }

    pub fn run(&mut self) -> Result<SimulationResult, SniaError> {
        while self.time_s < self.config.final_time_s && self.n_steps < self.config.max_steps {
            let dt = self.step()?;
            if dt <= 0.0 {
                break;
            }
        }

        let burned_mass_msun = self.integrated_burned_mass_msun();
        let peak_density = self
            .cells
            .iter()
            .map(|c| c.hydro.density)
            .fold(0.0_f64, f64::max);
        let detonation_strength = self
            .cells
            .iter()
            .map(|c| c.hydro.velocity.abs())
            .sum::<f64>()
            / (self.cells.len() as f64);
        let nickel56_mass_msun = self.yield_model.estimate_nickel56_mass(
            burned_mass_msun,
            peak_density,
            detonation_strength.max(0.1),
        );

        Ok(SimulationResult {
            final_time_s: self.time_s,
            n_steps: self.n_steps,
            burned_mass_msun,
            nickel56_mass_msun,
            peak_density,
            cells: self.cells.clone(),
            detonation_events: self.detonation_events.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solver_runs_short_window() {
        let config = SolverConfig {
            n_cells: 16,
            final_time_s: 1.0e-3,
            max_steps: 64,
            ..SolverConfig::default()
        };
        let initial_hydro = HydroState1D {
            density: 2.0e7,
            velocity: 1.0e6,
            pressure: 3.0e23,
            specific_internal_energy: 1.0e17,
        };
        let mut solver = SniaCoreSolver::new(
            config,
            WhiteDwarfEos::default(),
            HllcFlux1D::default(),
            CarbonBurnModel::default(),
            NickelYieldModel::default(),
            initial_hydro,
            1.5e9,
            BurnState::default(),
        )
        .expect("solver");
        let result = solver.run().expect("run");
        assert!(result.final_time_s > 0.0);
        assert!(result.n_steps > 0);
        assert!(result.nickel56_mass_msun >= 0.0);
    }
}
