//! Comparison baseline quantizers for TurboQuant evaluation.
//!
//! These implement the core mechanisms of competing KV cache quantization
//! methods so we can benchmark TurboQuant head-to-head on the same data.

pub mod kivi;
pub mod nsnquant;
