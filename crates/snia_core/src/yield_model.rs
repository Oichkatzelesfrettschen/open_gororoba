/// Reduced Ni56 yield estimator for DDT studies.
#[derive(Debug, Clone, Copy)]
pub struct NickelYieldModel {
    /// Density normalization in g/cm^3.
    pub reference_density: f64,
    /// Detonation normalization (dimensionless).
    pub reference_detonation_strength: f64,
}

impl Default for NickelYieldModel {
    fn default() -> Self {
        Self {
            reference_density: 2.0e7,
            reference_detonation_strength: 1.0,
        }
    }
}

impl NickelYieldModel {
    /// Estimates Ni56 mass in solar masses from effective burned mass.
    #[must_use]
    pub fn estimate_nickel56_mass(
        self,
        burned_mass_msun: f64,
        peak_density: f64,
        detonation_strength: f64,
    ) -> f64 {
        if burned_mass_msun <= 0.0 {
            return 0.0;
        }
        let density_factor = (peak_density / self.reference_density)
            .powf(0.35)
            .clamp(0.1, 2.5);
        let detonation_factor = (detonation_strength / self.reference_detonation_strength)
            .powf(0.5)
            .clamp(0.2, 2.0);
        let nickel_fraction = (0.45 * density_factor * detonation_factor).clamp(0.05, 0.95);
        burned_mass_msun * nickel_fraction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nickel_yield_is_clamped() {
        let model = NickelYieldModel::default();
        let yield_mass = model.estimate_nickel56_mass(1.0, 3.0e7, 1.4);
        assert!(yield_mass > 0.0);
        assert!(yield_mass <= 0.95);
    }
}
