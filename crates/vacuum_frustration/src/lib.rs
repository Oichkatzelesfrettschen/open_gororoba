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

pub mod signed_graph;
pub mod balance;
pub mod frustration;
pub mod bridge;
pub mod percolation;
pub mod apt_sedenion;

// Re-export key types for ergonomics
pub use bridge::{SedenionField, FrustrationViscosityBridge};
pub use signed_graph::SignedGraph;
pub use balance::{FrustrationResult, SolverMethod, compute_frustration_index};
pub use percolation::{PercolationChannel, PercolationDetector, CorrelationResult, auto_velocity_threshold, correlate_with_frustration};
pub use apt_sedenion::{AptSedenionField, FrustrationStats};
