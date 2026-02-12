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
