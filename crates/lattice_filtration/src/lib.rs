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

pub use basis_index::{project_to_lattice, BasisIndexCodec};
pub use filtration::{
    simulate_fibonacci_collision_storm, simulate_frustration_modulated_storm,
    simulate_sedenion_collision_storm, simulate_shell_return_storm,
    CollisionObservation, CollisionStormStats, FrustrationStormConfig,
    ShellReturnBin, ShellReturnStats,
};
pub use lbm_coupling::{filtration_from_velocity_field, FiltrationFromVelocity};
pub use lepton_ratio::{pdg_comparison, predict_mass_ratios, MassRatioPrediction, PdgComparison};
pub use mass_spectrum::{depth_clusters, depth_histogram, SurvivalDepthMap, SurvivalEntry};
pub use patricia_trie::PatriciaIndex;
pub use survival_spectrum::{
    classify_latency_law, classify_latency_law_detailed, exponential_r2, inverse_square_r2,
    power_law_r2, radial_bins, LatencyLaw, LatencyLawDetail, SpectrumBin,
};
