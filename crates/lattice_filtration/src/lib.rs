//! Thesis 2: Knotted Filtration of Particle Mass
//!
//! Elementary particle masses emerge from survival depth in the Cayley-Dickson
//! filtration cascade (Lambda_2048 -> Lambda_256).

pub mod basis_index;
pub mod filtration;
pub mod lbm_coupling;
pub mod lepton_ratio;
pub mod mass_spectrum;
pub mod patricia_trie;
pub mod survival_spectrum;
pub mod zero_divisor_census;

pub use basis_index::{BasisIndexCodec, project_to_lattice};
pub use filtration::{
    CollisionObservation, CollisionStormStats, FrustrationStormConfig, ShellReturnBin,
    ShellReturnStats, simulate_fibonacci_collision_storm, simulate_frustration_modulated_storm,
    simulate_sedenion_collision_storm, simulate_shell_return_storm,
};
pub use lbm_coupling::{FiltrationFromVelocity, filtration_from_velocity_field};
pub use lepton_ratio::{MassRatioPrediction, PdgComparison, pdg_comparison, predict_mass_ratios};
pub use mass_spectrum::{SurvivalDepthMap, SurvivalEntry, depth_clusters, depth_histogram};
pub use patricia_trie::PatriciaIndex;
pub use survival_spectrum::{
    GammaCI, LatencyLaw, LatencyLawDetail, SpectrumBin, classify_latency_law,
    classify_latency_law_detailed, exponential_r2, inverse_square_r2, power_law_gamma_ci,
    power_law_r2, radial_bins,
};
pub use zero_divisor_census::{
    CollapseManifold, ContinuousVacuum, ZeroDivisorWalker, simulate_collapse_manifold,
};
