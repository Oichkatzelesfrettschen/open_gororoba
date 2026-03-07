//! Thesis 3: A-Infinity Correction Protocol
//!
//! The A-infinity correction tensor m_4 resolving the Sedenion Lagrangian
//! obstruction can be synthesized via neural search constrained by Stasheff
//! polytope geometry.

// Native BLAS backends are selected at compile time via this crate's Cargo
// features (see `[features]` in `Cargo.toml`). Keep them opt-in so the default
// path stays offline-friendly and CI-portable.

pub mod burn_backend;
pub mod burn_model;
pub mod m4_tensor;
pub mod model;
pub mod optimizer;
pub mod perturbation;
pub mod stasheff;
pub mod tensor_ops;
pub mod training_data;

pub use burn_backend::{BackendKind, selected_backend};
pub use burn_model::{
    BurnTrainingResult, CorrectionTensorModel, CorrectionTensorModelConfig,
    assemble_neural_correction, train_burn_correction,
};
pub use m4_tensor::{CorrectionTensor, M4CorrectionTensor};
pub use model::{
    HomotopyTrainingConfig, PairTransitionModel, PlateauConfig, PlateauDetection, TrainingTrace,
    canonical_words, detect_plateaus, detect_plateaus_robust, gaussian_smooth,
    reference_hubble_curve, train_homotopy_surrogate, wasserstein_1d,
};
pub use optimizer::{
    AnsatzComparison, PentagonOptimizationConfig, PentagonOptimizationResult,
    compare_ansatz_vs_optimized, optimize_batch_coordinate_descent, optimize_correction_tensor,
    optimize_with_restarts,
};
pub use perturbation::{PerturbationDataset, perturbed_sedenion_table};
pub use stasheff::{PentagonResidual, SignedBasis, mean_pentagon_residual, pentagon_residual};
pub use tensor_ops::{
    alignment_score, chi_squared_fit, cosine_similarity, min_max_normalize,
    weighted_alignment_score,
};
pub use training_data::{
    MultiplicationSample, SEDENION_DIM, build_sedenion_table, encode_pair, multiplication_samples,
};
