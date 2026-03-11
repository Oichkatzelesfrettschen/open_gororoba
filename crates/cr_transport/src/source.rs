//! Dark matter source term injection into the Parker Transport Equation.
//!
//! The source term is applied as a tabulated spectrum over kinetic energy:
//!   Q_DM(r, K) = (rho_DM(r)^2 / (2 m_chi^2)) * <sigma v> * dN/dlog10(K)
//!
//! The solver evolves on a rigidity grid, so this module integrates the
//! tabulated kinetic-energy spectrum over each rigidity bin before injection.

use crate::grid::RigidityGrid;

/// Proton rest mass in GeV.
pub const M_PROTON_GEV: f64 = 0.938272;

/// Secondary-particle species for a DM yield table.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DmSpecies {
    Proton,
    Antiproton,
    Positron,
}

/// Annihilation channel determining the secondary-particle spectrum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DmChannel {
    BbBar,
    WW,
    Tau,
    PositronAntiproton,
}

/// Tabulated per-annihilation yield as a function of kinetic energy.
///
/// The ordinate is dN/dlog10(K), where K is kinetic energy in GeV.
#[derive(Clone, Debug)]
pub struct DmYieldSpectrum {
    pub species: DmSpecies,
    pub kinetic_energy_gev: Vec<f64>,
    pub dndlog10k: Vec<f64>,
}

impl DmYieldSpectrum {
    /// Construct a validated tabulated spectrum.
    pub fn new(
        species: DmSpecies,
        kinetic_energy_gev: Vec<f64>,
        dndlog10k: Vec<f64>,
    ) -> Result<Self, String> {
        if kinetic_energy_gev.len() != dndlog10k.len() {
            return Err("kinetic_energy_gev and dndlog10k length mismatch".to_string());
        }
        if kinetic_energy_gev.len() < 2 {
            return Err("need at least two spectrum points".to_string());
        }
        for window in kinetic_energy_gev.windows(2) {
            if !(window[0].is_finite() && window[1].is_finite()) || window[0] <= 0.0 {
                return Err("spectrum energies must be positive finite values".to_string());
            }
            if window[1] <= window[0] {
                return Err("spectrum energies must be strictly increasing".to_string());
            }
        }
        if !kinetic_energy_gev
            .last()
            .is_some_and(|value| value.is_finite() && *value > 0.0)
        {
            return Err("spectrum energies must end at a positive finite value".to_string());
        }
        if dndlog10k
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err("yield values must be finite and non-negative".to_string());
        }
        Ok(Self {
            species,
            kinetic_energy_gev,
            dndlog10k,
        })
    }

    fn logk_values(&self) -> Vec<f64> {
        self.kinetic_energy_gev
            .iter()
            .map(|value| value.log10())
            .collect()
    }

    fn value_at_logk(&self, logk: f64) -> f64 {
        let logk_values = self.logk_values();
        if logk < logk_values[0] || logk > logk_values[logk_values.len() - 1] {
            return 0.0;
        }
        let idx = logk_values
            .partition_point(|value| *value <= logk)
            .saturating_sub(1)
            .min(logk_values.len() - 2);
        let left = logk_values[idx];
        let right = logk_values[idx + 1];
        let span = (right - left).max(1e-12);
        let t = (logk - left) / span;
        let y0 = self.dndlog10k[idx];
        let y1 = self.dndlog10k[idx + 1];
        y0 * (1.0 - t) + y1 * t
    }

    fn integrate_logk_range(&self, logk_min: f64, logk_max: f64) -> f64 {
        if !(logk_min.is_finite() && logk_max.is_finite()) || logk_max <= logk_min {
            return 0.0;
        }
        let logk_values = self.logk_values();
        let support_min = logk_values[0];
        let support_max = logk_values[logk_values.len() - 1];
        let start = logk_min.max(support_min);
        let end = logk_max.min(support_max);
        if end <= start {
            return 0.0;
        }

        let mut nodes = Vec::with_capacity(logk_values.len() + 2);
        nodes.push(start);
        for &value in &logk_values {
            if value > start && value < end {
                nodes.push(value);
            }
        }
        nodes.push(end);
        nodes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        nodes.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

        let mut integral = 0.0;
        for window in nodes.windows(2) {
            let a = window[0];
            let b = window[1];
            let fa = self.value_at_logk(a);
            let fb = self.value_at_logk(b);
            integral += 0.5 * (fa + fb) * (b - a);
        }
        integral
    }

    /// Integrate the table over every rigidity bin of the target grid.
    pub fn integrated_bin_yields(&self, grid: &RigidityGrid, m_rest_gev: f64) -> Vec<f64> {
        let mut yields = Vec::with_capacity(grid.n_p);
        for p_idx in 0..grid.n_p {
            let ln_r_center = grid.ln_r[p_idx];
            let ln_r_min = if p_idx == 0 {
                ln_r_center - 0.5 * grid.d_ln_r
            } else {
                0.5 * (grid.ln_r[p_idx - 1] + ln_r_center)
            };
            let ln_r_max = if p_idx + 1 == grid.n_p {
                ln_r_center + 0.5 * grid.d_ln_r
            } else {
                0.5 * (ln_r_center + grid.ln_r[p_idx + 1])
            };
            let r_min = ln_r_min.exp();
            let r_max = ln_r_max.exp();
            let k_min = (r_min * r_min + m_rest_gev * m_rest_gev).sqrt() - m_rest_gev;
            let k_max = (r_max * r_max + m_rest_gev * m_rest_gev).sqrt() - m_rest_gev;
            if !(k_min.is_finite() && k_max.is_finite()) || k_min <= 0.0 || k_max <= k_min {
                yields.push(0.0);
                continue;
            }
            yields.push(self.integrate_logk_range(k_min.log10(), k_max.log10()));
        }
        yields
    }
}

/// Dark matter source term configuration.
#[derive(Clone, Debug)]
pub struct DmSource {
    /// DM particle mass (GeV/c^2).
    pub mass_gev: f64,
    /// Annihilation channel.
    pub channel: DmChannel,
    /// Velocity-averaged annihilation cross section (cm^3/s).
    pub sigma_v_cm3_per_s: f64,
    /// Per-cell spatial weight.
    pub spatial_profile: Vec<f64>,
    /// Per-annihilation yield spectrum.
    pub yield_spectrum: DmYieldSpectrum,
}

impl DmSource {
    fn species_rest_mass_gev(&self) -> f64 {
        match self.yield_spectrum.species {
            DmSpecies::Proton | DmSpecies::Antiproton => M_PROTON_GEV,
            DmSpecies::Positron => 0.000_510_998_95,
        }
    }

    /// Inject Q_DM into the phase-space distribution f.
    ///
    /// The tabulated spectrum is first integrated over the target rigidity bins,
    /// then applied at every grid cell with the local spatial weight.
    pub fn inject(&self, f: &mut [f64], grid: &RigidityGrid, dm_density: &[f64], dt: f64) {
        let n_cells = self.spatial_profile.len();
        let m2 = self.mass_gev * self.mass_gev;
        if m2 <= 0.0 {
            return;
        }

        let sigma_v_au = self.sigma_v_cm3_per_s / (1.496e13_f64 * 1.496e13_f64 * 1.496e13_f64);
        let bin_yields = self
            .yield_spectrum
            .integrated_bin_yields(grid, self.species_rest_mass_gev());

        for idx in 0..n_cells {
            if idx >= dm_density.len() {
                break;
            }
            let rho2 = dm_density[idx] * dm_density[idx];
            let cell_prefactor = 0.5 * rho2 / m2 * sigma_v_au * self.spatial_profile[idx] * dt;
            if !cell_prefactor.is_finite() || cell_prefactor == 0.0 {
                continue;
            }
            for (p_idx, yield_bin) in bin_yields.iter().enumerate() {
                if *yield_bin <= 0.0 {
                    continue;
                }
                let f_idx = idx * grid.n_p + p_idx;
                if f_idx < f.len() {
                    f[f_idx] += cell_prefactor * yield_bin;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_spectrum() -> DmYieldSpectrum {
        DmYieldSpectrum::new(
            DmSpecies::Proton,
            vec![0.1, 0.5, 1.0, 5.0, 10.0],
            vec![0.2, 1.0, 1.5, 0.7, 0.1],
        )
        .expect("valid test spectrum")
    }

    #[test]
    fn test_invalid_spectrum_rejected() {
        let err = DmYieldSpectrum::new(DmSpecies::Proton, vec![1.0], vec![1.0])
            .expect_err("single-point spectrum must be rejected");
        assert!(err.contains("at least two"));
    }

    #[test]
    fn test_integrated_bin_yields_span_multiple_bins() {
        let grid = RigidityGrid::new(20, 0.1, 20.0);
        let yields = sample_spectrum().integrated_bin_yields(&grid, M_PROTON_GEV);
        let non_zero = yields.iter().filter(|value| **value > 0.0).count();
        assert!(
            non_zero >= 3,
            "expected tabulated yield to populate multiple bins, got {non_zero}"
        );
    }

    #[test]
    fn test_inject_populates_multiple_bins() {
        let grid = RigidityGrid::new(20, 0.1, 20.0);
        let source = DmSource {
            mass_gev: 100.0,
            channel: DmChannel::BbBar,
            sigma_v_cm3_per_s: 3e-26,
            spatial_profile: vec![1.0; 2],
            yield_spectrum: sample_spectrum(),
        };
        let dm_density = vec![1.0_f64; 2];
        let mut f = vec![0.0_f64; 2 * grid.n_p];

        source.inject(&mut f, &grid, &dm_density, 1.0);

        for cell in 0..2 {
            let start = cell * grid.n_p;
            let non_zero = f[start..start + grid.n_p]
                .iter()
                .filter(|value| **value > 0.0)
                .count();
            assert!(
                non_zero >= 3,
                "expected spectrum injection to hit multiple bins, got {non_zero}"
            );
        }
    }
}
