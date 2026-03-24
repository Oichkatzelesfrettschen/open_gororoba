//! Sedenion Mass Ladder and Observational Cross-matching.
//!
//! Maps predicted mass modes from the Sedenion Negative Dimension hypothesis
//! against real observational datasets (LIGO BHs, ATNF Pulsars).
//!
//! Migrated from src/map_cosmic_objects.py and src/fetch_ligo_gwpy.py.

use data_core::catalogs::{atnf::Pulsar, gwtc::GwEvent};
use stats_core::helpers::histogram;

/// A predicted mass mode from the Sedenion ladder.
#[derive(Debug, Clone)]
pub struct SedenionMassMode {
    pub n: usize,
    pub mass_solar: f64,
}

/// Results of cross-matching Sedenion predictions with observations.
#[derive(Debug, Clone)]
pub struct MassLadderMatch {
    pub modes: Vec<SedenionMassMode>,
    pub ligo_masses: Vec<f64>,
    pub pulsar_masses: Vec<f64>,
}

impl MassLadderMatch {
    /// Initialize with predicted modes.
    ///
    /// Predicted mass M(n) ~ n^(-alpha) * M_0.
    /// For alpha = -1.5, M ~ n^1.5.
    pub fn new(alpha: f64, m0: f64, max_n: usize) -> Self {
        let mut modes = Vec::with_capacity(max_n);
        for n in 1..=max_n {
            modes.push(SedenionMassMode {
                n,
                mass_solar: m0 * (n as f64).powf(-alpha),
            });
        }
        Self {
            modes,
            ligo_masses: Vec::new(),
            pulsar_masses: Vec::new(),
        }
    }

    /// Load LIGO masses from a list of events.
    pub fn load_ligo(&mut self, events: &[GwEvent]) {
        self.ligo_masses = events
            .iter()
            .map(|e| e.mass_1_source)
            .filter(|&m| m > 0.0)
            .collect();
    }

    /// Load Pulsar masses (simulated or parsed).
    ///
    /// Since most ATNF pulsars lack mass data, we use the fallback distribution
    /// from the original script: N(1.35, 0.15) if no direct measurements provided.
    pub fn load_pulsars(&mut self, _pulsars: &[Pulsar]) {
        // Placeholder for real mass parsing if available in future ATNF versions.
        // For now, generate the statistical distribution used in the Python script.
        use rand::prelude::*;
        use rand_distr::Normal;
        let mut rng = thread_rng();
        let dist = Normal::new(1.35, 0.15).unwrap();
        self.pulsar_masses = (0..500).map(|_| dist.sample(&mut rng)).collect();
    }

    /// Compute histograms for the aggregated masses.
    pub fn compute_histograms(
        &self,
        n_bins: usize,
        max_mass: f64,
    ) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
        let (centers, ligo_counts) = histogram(&self.ligo_masses, n_bins, 0.0, max_mass);
        let (_, pulsar_counts) = histogram(&self.pulsar_masses, n_bins, 0.0, max_mass);
        (centers, ligo_counts, pulsar_counts)
    }

    /// Find the nearest Sedenion mode for a given observed mass.
    pub fn find_nearest_mode(&self, mass: f64) -> Option<&SedenionMassMode> {
        self.modes.iter().min_by(|a, b| {
            (a.mass_solar - mass)
                .abs()
                .partial_cmp(&(b.mass_solar - mass).abs())
                .unwrap()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mass_ladder_generation() {
        // M ~ n^1.5, M0=1.0 -> n=1: 1.0, n=2: 2.8, n=3: 5.2
        let ladder = MassLadderMatch::new(-1.5, 1.0, 10);
        assert_eq!(ladder.modes.len(), 10);
        assert!((ladder.modes[0].mass_solar - 1.0).abs() < 1e-10);
        assert!(ladder.modes[1].mass_solar > 2.8);
    }

    #[test]
    fn test_nearest_mode() {
        let ladder = MassLadderMatch::new(-1.5, 1.0, 10);
        let mode = ladder.find_nearest_mode(3.0).unwrap();
        assert_eq!(mode.n, 2);
    }
}
