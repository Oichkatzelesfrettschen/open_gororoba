//! Thesis 1: Viscous Vacuum of Signed-Graph Imbalance
//!
//! Fluid viscosity emerges from algebraic imbalance in Cayley-Dickson graphs.
//! The 3/8 imbalance attractor defines the vacuum state.
//!
//! Key abstractions:
//! - SignedGraph: Nodes = basis elements, edges = psi signs from CD multiplication
//! - ImbalanceResult: Harary-Zaslavsky balance index computation
//! - SedenionField: 3D lattice of 16D Sedenion algebra elements
//! - ImbalanceViscosityBridge: Maps imbalance density to kinematic viscosity nu(x,y,z)

pub mod apt_sedenion;
pub mod balance;
pub mod bridge;
pub mod imbalance;
pub mod imbalance_energy;
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
pub use apt_sedenion::{AptSedenionField, ImbalanceStats};
pub use balance::{ImbalanceResult, SolverMethod, compute_imbalance_index};
pub use bridge::{
    ImbalanceViscosityBridge, SedenionField, SedenionField4D, IMBALANCE_ATTRACTOR,
    ViscosityCouplingModel,
};
pub use imbalance::{
    CASSINI_OMEGA_BD_LOWER_BOUND, ImbalanceStarConfig, ImbalanceStarResult,
    ScalarImbalanceMap, evaluate_imbalance_star, imbalance_density_from_edges,
    omega_eff_from_phi, violates_cassini,
};
pub use imbalance_energy::{
    ImbalanceEnergy, compute_imbalance_energy, compute_lambda, estimate_e0_from_associators,
    predict_lambda_sedenion,
};
pub use immirzi_bridge::{
    BRIDGE_MAX, GAMMA_BG as IMMIRZI_GAMMA_BG, GAMMA_MAX, GAMMA_NZJ as IMMIRZI_GAMMA_NZJ,
    ImmirziMappingResult, SelectivityResult, VACUUM_PHI, best_bg_match, best_nzj_match,
    binary_entropy, binary_entropy_deriv, cd_imbalance_density, entropy_bridge,
    evaluate_all_mappings, imbalance_density_census, imbalance_entropy_bridge,
    invert_entropy_bridge, invert_entropy_bridge_bg, invert_entropy_bridge_newton,
    invert_entropy_bridge_nzj, invert_entropy_bridge_nzj_newton, linear_bridge_bg,
    log_bridge_bg, phi_in_achievable_range, power_bridge_bg, selectivity_analysis,
};
pub use percolation::{
    CorrelationResult, PercolationChannel, PercolationDetector, auto_velocity_threshold,
    correlate_with_imbalance,
};
pub use signed_graph::SignedGraph;
pub use spatial_correlation::{
    SpatialCorrelationResult, coefficient_of_variation, dynamic_range_ratio, grid_partition_3d,
    nonlinearity_index, pearson_correlation, point_cloud_overlap, regional_means,
    spatial_correlation, spearman_correlation, velocity_magnitude_field,
};
pub use vietoris_rips::PersistenceDiagram;
