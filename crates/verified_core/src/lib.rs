//! Rocq-verified operations extracted to Rust.
//!
//! This crate contains pure arithmetic operations whose correctness is
//! guaranteed by machine-checked Rocq proofs. The extraction pipeline:
//!
//! 1. Define operations abstractly in Rocq (FloatAxioms.v, FloatQuaternion.v)
//! 2. Prove theorems about those operations (FloatRealBridge.v, C876)
//! 3. Extract to OCaml via `Extraction` (extracted_quat.ml.golden)
//! 4. Translate OCaml functor to Rust (this crate)
//!
//! The trust boundary is `axioms.rs`: it maps the abstract FLOAT_OPS field
//! operations to concrete f64 arithmetic. Everything else is derived.
//!
//! # Zero Dependencies
//!
//! This crate has no external dependencies. It contains only pure f64
//! arithmetic derived from formal proofs.

pub mod axioms;
pub mod binary_entropy;
pub mod brans_dicke;
pub mod cayley_dickson;
pub mod fano_resonance;
pub mod gf2;
pub mod neg_dim;
pub mod parity_clique;
pub mod quaternion;
pub mod pathion_entropy;
pub mod s3_mixing;
pub mod spectral_dim;
