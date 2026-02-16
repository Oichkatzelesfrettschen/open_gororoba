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
