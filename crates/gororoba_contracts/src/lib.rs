use serde::{Deserialize, Serialize};

/// Canonical Contract for Warp Ring Experiments.
///
/// This schema defines the input parameters and output metrics for a
/// standard Warp Ring simulation, ensuring reproducibility and
/// consistent analysis across different backends (Rust, Python, etc).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarpRingExperiment {
    /// Unique Experiment ID (e.g., "EXP-2026-02-14-WR-128")
    pub experiment_id: String,

    /// Simulation Configuration
    pub config: WarpRingConfig,

    /// Simulation Results (Summary)
    pub results: WarpRingResults,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarpRingConfig {
    pub resolution: usize, // e.g., 128
    pub steps: usize,
    pub tau: f64,
    pub forcing_type: String, // "E7_Gaussian"
    pub coupling_lambda: f64,
    pub initial_condition: String, // "Random", "VortexRing"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarpRingResults {
    pub final_enstrophy: f64,
    pub mean_density: f64,
    pub betti_1_persistence: Option<f64>,
    pub execution_time_s: f64,
    pub steps_per_second: f64,
    pub mlups: f64, // Mega Lattice Updates Per Second
    pub artifact_path: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_config() -> WarpRingConfig {
        WarpRingConfig {
            resolution: 128,
            steps: 1000,
            tau: 0.505,
            forcing_type: "E7_Gaussian".into(),
            coupling_lambda: 0.01,
            initial_condition: "Random".into(),
        }
    }

    fn sample_results() -> WarpRingResults {
        WarpRingResults {
            final_enstrophy: 1.23e-4,
            mean_density: 1.0,
            betti_1_persistence: Some(0.42),
            execution_time_s: 12.5,
            steps_per_second: 80.0,
            mlups: 167.77,
            artifact_path: "data/h5/test.h5".into(),
        }
    }

    #[test]
    fn serde_round_trip_experiment() {
        let exp = WarpRingExperiment {
            experiment_id: "EXP-TEST-001".into(),
            config: sample_config(),
            results: sample_results(),
        };
        let json = serde_json::to_string(&exp).unwrap();
        let recovered: WarpRingExperiment = serde_json::from_str(&json).unwrap();
        assert_eq!(recovered.experiment_id, "EXP-TEST-001");
        assert_eq!(recovered.config.resolution, 128);
        assert!((recovered.results.final_enstrophy - 1.23e-4).abs() < 1e-10);
    }

    #[test]
    fn serde_round_trip_config() {
        let cfg = sample_config();
        let toml = toml::to_string(&cfg).unwrap();
        let recovered: WarpRingConfig = toml::from_str(&toml).unwrap();
        assert_eq!(recovered.steps, 1000);
        assert_eq!(recovered.forcing_type, "E7_Gaussian");
    }

    #[test]
    fn serde_round_trip_results_none_betti() {
        let mut res = sample_results();
        res.betti_1_persistence = None;
        let json = serde_json::to_string(&res).unwrap();
        let recovered: WarpRingResults = serde_json::from_str(&json).unwrap();
        assert!(recovered.betti_1_persistence.is_none());
    }
}
