//! Thesis 1: Viscous Vacuum of Signed-Graph Frustration
//!
//! Fluid viscosity emerges from algebraic frustration in Cayley-Dickson graphs.
//! The 3/8 frustration attractor defines the vacuum state.
//!
//! Key abstractions:
//! - SignedGraph: Nodes = basis elements, edges = psi signs from CD multiplication
//! - FrustrationResult: Harary-Zaslavsky balance index computation
//! - SedenionField: 3D lattice of 16D Sedenion algebra elements
//! - FrustrationViscosityBridge: Maps frustration density to kinematic viscosity nu(x,y,z)

pub mod apt_sedenion;
pub mod balance;
pub mod bridge;
pub mod frustration;
pub mod frustration_energy;
pub mod immirzi_bridge;
pub mod kubo_transport;
pub mod percolation;
pub mod sedenion_foliation;
pub mod signed_graph;
pub mod spatial_correlation;
pub mod vietoris_rips;

// GPU-accelerated modules (optional feature)
#[cfg(feature = "gpu")]
pub mod besag_clifford_cuda;
#[cfg(feature = "gpu")]
pub mod kubo_transport_gpu;

// Re-export key types for ergonomics
pub use apt_sedenion::{AptSedenionField, FrustrationStats};
pub use balance::{FrustrationResult, SolverMethod, compute_frustration_index};
pub use bridge::{
    FrustrationViscosityBridge, SedenionField, SedenionField4D, VACUUM_ATTRACTOR,
    ViscosityCouplingModel,
};
pub use frustration::{
    CASSINI_OMEGA_BD_LOWER_BOUND, FrustrationStarConfig, FrustrationStarResult,
    ScalarFrustrationMap, evaluate_frustration_star, frustration_density_from_edges,
    omega_eff_from_phi, violates_cassini,
};
pub use frustration_energy::{
    FrustrationEnergy, compute_frustration_energy, compute_lambda, estimate_e0_from_associators,
    predict_lambda_sedenion,
};
pub use immirzi_bridge::{
    BRIDGE_MAX, GAMMA_BG as IMMIRZI_GAMMA_BG, GAMMA_NZJ as IMMIRZI_GAMMA_NZJ, ImmirziMappingResult,
    VACUUM_PHI, best_bg_match, best_nzj_match, entropy_bridge, evaluate_all_mappings,
    frustration_entropy_bridge, invert_entropy_bridge, invert_entropy_bridge_bg,
    invert_entropy_bridge_nzj, linear_bridge_bg, log_bridge_bg, power_bridge_bg,
};
pub use percolation::{
    CorrelationResult, PercolationChannel, PercolationDetector, auto_velocity_threshold,
    correlate_with_frustration,
};
pub use signed_graph::SignedGraph;
pub use spatial_correlation::{
    SpatialCorrelationResult, coefficient_of_variation, dynamic_range_ratio, grid_partition_3d,
    nonlinearity_index, pearson_correlation, point_cloud_overlap, regional_means,
    spatial_correlation, spearman_correlation, velocity_magnitude_field,
};
pub use vietoris_rips::PersistenceDiagram;
