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
pub mod kubo_transport;
pub mod percolation;
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
pub use balance::{compute_frustration_index, FrustrationResult, SolverMethod};
pub use bridge::{
    FrustrationViscosityBridge, SedenionField, SedenionField4D, ViscosityCouplingModel,
    VACUUM_ATTRACTOR,
};
pub use vietoris_rips::PersistenceDiagram;
pub use frustration::{
    evaluate_frustration_star, frustration_density_from_edges, omega_eff_from_phi,
    violates_cassini, FrustrationStarConfig, FrustrationStarResult, ScalarFrustrationMap,
    CASSINI_OMEGA_BD_LOWER_BOUND,
};
pub use frustration_energy::{
    compute_frustration_energy, compute_lambda, estimate_e0_from_associators,
    predict_lambda_sedenion, FrustrationEnergy,
};
pub use percolation::{
    auto_velocity_threshold, correlate_with_frustration, CorrelationResult, PercolationChannel,
    PercolationDetector,
};
pub use signed_graph::SignedGraph;
pub use spatial_correlation::{
    coefficient_of_variation, dynamic_range_ratio, grid_partition_3d, nonlinearity_index,
    pearson_correlation, point_cloud_overlap, regional_means, spatial_correlation,
    spearman_correlation, velocity_magnitude_field, SpatialCorrelationResult,
};
