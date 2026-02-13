//! Thesis 3: A-Infinity Correction Protocol
//!
//! The A-infinity correction tensor m_4 resolving the Sedenion Lagrangian
//! obstruction can be synthesized via neural search constrained by Stasheff
//! polytope geometry.

// Force cargo to link OpenBLAS (needed by burn-ndarray for matmul).
// Without this, cargo omits -lopenblas from the linker command because
// no Rust code directly references the blas_src crate.
extern crate blas_src;

pub mod burn_backend;
pub mod burn_model;
pub mod m4_tensor;
pub mod model;
pub mod optimizer;
pub mod perturbation;
pub mod stasheff;
pub mod tensor_ops;
pub mod training_data;

pub use burn_backend::{selected_backend, BackendKind};
pub use burn_model::{
    assemble_neural_correction, CorrectionTensorModel, CorrectionTensorModelConfig,
};
pub use m4_tensor::CorrectionTensor;
pub use model::{
    canonical_words, detect_plateaus, reference_hubble_curve, train_homotopy_surrogate,
    wasserstein_1d, HomotopyTrainingConfig, PairTransitionModel, PlateauDetection, TrainingTrace,
};
pub use optimizer::{
    compare_ansatz_vs_optimized, optimize_batch_coordinate_descent, optimize_correction_tensor,
    optimize_with_restarts, AnsatzComparison, PentagonOptimizationConfig,
    PentagonOptimizationResult,
};
pub use perturbation::{perturbed_sedenion_table, PerturbationDataset};
pub use stasheff::{mean_pentagon_residual, pentagon_residual, PentagonResidual, SignedBasis};
pub use tensor_ops::{
    alignment_score, chi_squared_fit, cosine_similarity, min_max_normalize,
    weighted_alignment_score,
};
pub use training_data::{
    build_sedenion_table, encode_pair, multiplication_samples, MultiplicationSample, SEDENION_DIM,
};
