use serde::{Deserialize, Serialize};

/// One sample on a synthetic bolometric light curve.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LightCurveSample {
    pub time_days: f64,
    pub luminosity_erg_s: f64,
}

/// Reduced Arnett-like luminosity model.
#[derive(Debug, Clone, Copy)]
pub struct LightCurveModel {
    pub ni56_decay_time_days: f64,
    pub co56_decay_time_days: f64,
    pub gamma_escape_time_days: f64,
}

impl Default for LightCurveModel {
    fn default() -> Self {
        Self {
            ni56_decay_time_days: 8.8,
            co56_decay_time_days: 111.3,
            gamma_escape_time_days: 42.0,
        }
    }
}

impl LightCurveModel {
    /// Synthesize a bolometric curve from Ni56 mass and a list of times.
    #[must_use]
    pub fn synthesize(self, nickel56_mass_msun: f64, times_days: &[f64]) -> Vec<LightCurveSample> {
        let m_ni = nickel56_mass_msun.max(0.0);
        times_days
            .iter()
            .copied()
            .map(|time_days| {
                let t = time_days.max(0.0);
                let ni_term = (-t / self.ni56_decay_time_days).exp();
                let co_term = (-t / self.co56_decay_time_days).exp() - ni_term;
                let deposition = (-(t / self.gamma_escape_time_days).powi(2)).exp();
                let heating_per_msun = 6.45e43 * ni_term + 1.45e43 * co_term.max(0.0);
                LightCurveSample {
                    time_days: t,
                    luminosity_erg_s: m_ni * heating_per_msun * deposition,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lightcurve_is_non_negative() {
        let model = LightCurveModel::default();
        let samples = model.synthesize(0.55, &[0.0, 5.0, 15.0, 30.0, 60.0]);
        assert_eq!(samples.len(), 5);
        assert!(samples.iter().all(|s| s.luminosity_erg_s >= 0.0));
    }
}
