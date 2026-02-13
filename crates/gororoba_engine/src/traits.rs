//! Core traits for the six-layer engine pipeline.

/// Shared state propagated through pipeline layers.
#[derive(Debug, Clone, Default)]
pub struct PipelineState {
    pub words: Vec<u64>,
    pub signs: Vec<i32>,
    pub frustration: Vec<f64>,
    pub viscosity: Vec<f64>,
    pub correction_gain: f64,
}

/// Final verification payload.
#[derive(Debug, Clone)]
pub struct VerificationReport {
    pub pass: bool,
    pub messages: Vec<String>,
}

pub trait BitSourceLayer {
    fn sample_words(&self, n: usize) -> Vec<u64>;
}

pub trait ParityLayer {
    fn compute_signs(&self, words: &[u64]) -> Vec<i32>;
}

pub trait TopologyLayer {
    fn frustration_density(&self, signs: &[i32]) -> Vec<f64>;
}

pub trait DynamicsLayer {
    fn viscosity_field(&self, frustration: &[f64]) -> Vec<f64>;
}

pub trait CorrectionLayer {
    fn correction_gain(&self, frustration: &[f64]) -> f64;
}

pub trait VerificationLayer {
    fn verify(&self, state: &PipelineState) -> VerificationReport;
}

// ---------------------------------------------------------------------------
// Thesis-specific pipeline orchestration
// ---------------------------------------------------------------------------

/// Evidence produced by a thesis pipeline run.
#[derive(Debug, Clone)]
pub struct ThesisEvidence {
    /// Which thesis (1-4)
    pub thesis_id: usize,
    /// Short label for the evidence
    pub label: String,
    /// Key numeric result (e.g., correlation, R^2, slope ratio)
    pub metric_value: f64,
    /// Threshold for pass/fail
    pub threshold: f64,
    /// Whether the evidence passes the falsification gate
    pub passes_gate: bool,
    /// Human-readable messages
    pub messages: Vec<String>,
}

/// Trait for thesis-specific pipeline execution.
///
/// Each thesis implements this to define its falsification pipeline:
/// 1. Setup: initialize simulation parameters
/// 2. Execute: run the simulation/analysis
/// 3. Gate: check falsification criterion
/// 4. Report: produce structured evidence
pub trait ThesisPipeline {
    /// Short name of the thesis (e.g., "T1: Viscous Vacuum")
    fn name(&self) -> &str;

    /// Execute the full pipeline and produce evidence.
    fn execute(&self) -> ThesisEvidence;

    /// Check if the evidence passes the falsification gate.
    fn passes_gate(&self, evidence: &ThesisEvidence) -> bool {
        evidence.passes_gate
    }
}
