//! Thesis 3: A-Infinity Correction Protocol
//!
//! The A-infinity correction tensor m_4 resolving the Sedenion Lagrangian
//! obstruction can be synthesized via neural search constrained by Stasheff
//! polytope geometry.

pub mod burn_backend;
pub mod model;
pub mod stasheff;
pub mod tensor_ops;
pub mod training_data;

pub use burn_backend::{selected_backend, BackendKind};
pub use model::{
    canonical_words, reference_hubble_curve, train_homotopy_surrogate, HomotopyTrainingConfig,
    PairTransitionModel, TrainingTrace,
};
pub use stasheff::{mean_pentagon_residual, pentagon_residual, PentagonResidual, SignedBasis};
pub use tensor_ops::{alignment_score, cosine_similarity, min_max_normalize};
pub use training_data::{
    build_sedenion_table, encode_pair, multiplication_samples, MultiplicationSample, SEDENION_DIM,
};
